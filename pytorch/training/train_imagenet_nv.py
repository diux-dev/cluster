import argparse, os, shutil, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import math

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

# import models
from fp16util import *
import gc

import resnet
# import resnet_sd as resnet

from dataloader import *

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
    parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=45, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=192, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resize-sched', default='0.4,0.92', type=str,
                        help='Scheduler to resize from 128 -> 224 -> 288')
    parser.add_argument('--lr-sched', default='0.1,0.47,0.78,0.95', type=str,
                        help='Learning rate scheduler warmup -> lr -> lr/10 -> lr/100 -> lr/1000')
    parser.add_argument('--init-bn0', action='store_true', help='Intialize running batch norm mean to 0')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
    parser.add_argument('--loss-scale', type=float, default=1,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--prof', dest='prof', action='store_true', help='Only run a few iters for profiling.')
    parser.add_argument('--val-ar', action='store_true', help='Do final validation by nearest aspect ratio')
    parser.add_argument('--distributed', action='store_true', help='Run distributed training')
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='Number of gpus per machine. Param only needed for single machine training when using (faster) file sync')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    return parser

cudnn.benchmark = True
args = get_parser().parse_args()
if args.local_rank > 0: sys.stdout = open(f'{args.save_dir}/GPU_{args.local_rank}.log', 'w')


class DataManager():
    def __init__(self, resize_sched=[0.4, 0.92]):
        self.resize_sched = resize_sched
        self.load_data('-sz/160', args.batch_size, 128)
        
    def set_epoch(self, epoch):
        if epoch==int(args.epochs*self.resize_sched[0]+0.5):
            # self.load_data('-sz/320', args.batch_size, 224) # lower validation accuracy when enabled for some reason
            print('DataManager changing image size to 244')
            self.load_data('', args.batch_size, 224)
        if epoch==int(args.epochs*self.resize_sched[1]+0.5):
            self.load_data('', 128, 288, min_scale=0.5, use_ar=args.val_ar)

        if hasattr(self.trn_smp, 'set_epoch'): self.trn_smp.set_epoch(epoch)
        if hasattr(self.val_smp, 'set_epoch'): self.val_smp.set_epoch(epoch)
    
    # For val_ar faster scheduler - [0.35,0.88]

    def get_trn_iter(self):
        self.trn_iter = iter(self.trn_dl)
        return self.trn_iter

    def get_val_iter(self):
        self.val_iter = iter(self.val_dl)
        return self.val_iter
        
    def load_data(self, dir_prefix, batch_size, image_size, **kwargs):
        datadir = args.data+dir_prefix
        print(f'Dataset changed. \nImage size: {image_size} \nBatch size: {batch_size} \nDirectory: {datadir}')
        loaders = get_loaders(datadir, bs=batch_size, sz=image_size, workers=args.workers, distributed=args.distributed, **kwargs)
        self.trn_dl,self.val_dl,self.trn_smp,self.val_smp = loaders
        self.trn_dl = DataPrefetcher(self.trn_dl)
        self.val_dl = DataPrefetcher(self.val_dl, prefetch=False)
        self.trn_len = len(self.trn_dl)
        self.val_len = len(self.val_dl)
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()

class Scheduler():
    def __init__(self, optimizer, lr_sched=[0.1, 0.47, 0.78, 0.95]):
        self.optimizer = optimizer
        self.current_lr = None
        self.current_epoch = 0
        self.lr_sched = lr_sched

    def bn0_lr_warmup(self, epoch, epoch_tot, batch_num, batch_tot):
        world_size = args.world_size
        lr_step = args.lr / (epoch_tot * batch_tot)
        lr = args.lr + (epoch * batch_tot + batch_num) * lr_step
        if world_size >= 64: lr *= .75
        return lr

    def linear_lr_warmup(self, epoch, epoch_tot, batch_num, batch_tot):
        starting_lr = args.lr/epoch_tot
        ending_lr = args.lr
        step_size = (ending_lr - starting_lr)/epoch_tot
        batch_step_size = step_size/batch_tot
        lr = step_size*(epoch+1) + batch_step_size*batch_num

        if (args.world_size >= 32) and (epoch < epoch_tot):
            starting_lr = starting_lr/(4 - epoch)
        return lr

    def get_lr(self, epoch, batch_num, batch_tot):
        """Sets the learning rate to the initial LR decayed by 10 every few epochs"""
        # faster lr schedule [0.14, 0.43, 0.73, 0.94]
        # original lr schedule [0.1, 0.47, 0.78, 0.95]
        if epoch<int(args.epochs*self.lr_sched[0]+0.5):
            epoch_tot = args.epochs*self.lr_sched[0]+0.5
            if args.init_bn0: lr = self.bn0_lr_warmup(epoch, epoch_tot, batch_num, batch_tot)
            else: lr = self.linear_lr_warmup(epoch, epoch_tot, batch_num, batch_tot)
        elif epoch<int(args.epochs*self.lr_sched[1]+0.5): return args.lr/1
        elif epoch<int(args.epochs*self.lr_sched[2]+0.5): return args.lr/10
        elif epoch<int(args.epochs*self.lr_sched[3]+0.5): return args.lr/100
        else         : lr = args.lr/1000
        return lr

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot)
        if (self.current_lr != lr) and ((batch_num == 0) or (batch_num+1 == batch_tot)): 
            print(f'Changing LR from {self.current_lr} to {lr}')

        self.current_lr = lr
        self.current_epoch = epoch
        self.current_batch = batch_num

        for param_group in self.optimizer.param_groups:
            lr_old = param_group['lr'] or lr
            param_group['lr'] = lr

            # Trick 4: apply momentum correction when lr is updated
            # https://github.com/pytorch/examples/pull/262
            if lr > lr_old: param_group['momentum'] = lr / lr_old * args.momentum
            else: param_group['momentum'] = args.momentum


def init_dist_weights(model):
    # https://arxiv.org/pdf/1706.02677.pdf
    # https://github.com/pytorch/examples/pull/262
    if args.arch.startswith('resnet'):
        for m in model.modules():
            if isinstance(m, resnet.BasicBlock):
                m.bn2.weight = Parameter(torch.zeros_like(m.bn2.weight))
            if isinstance(m, resnet.Bottleneck):
                m.bn3.weight = Parameter(torch.zeros_like(m.bn3.weight))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

def main():
    print(args)
    print("~~epoch\thours\ttop1Accuracy\n")

    # need to index validation directory before we start counting the time
    if args.val_ar: sort_ar(args.data+'/validation')

    start_time = datetime.now()

    if args.distributed:
        print('Distributed: initializing process group')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)
        assert(args.world_size == dist.get_world_size())

    if args.fp16: assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    model = resnet.resnet50(pretrained=args.pretrained)
    print("Loaded model")

    model = model.cuda()
    n_dev = torch.cuda.device_count()
    if args.fp16: model = network_to_half(model)
    if args.distributed:
        if args.init_bn0: init_dist_weights(model) # (AS) Performs pretty poorly for first 10 epochs when enabled
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    global model_params, master_params
    if args.fp16:  model_params, master_params = prep_param_lists(model)
    else: master_params = list(model.parameters())

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(master_params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = Scheduler(optimizer, str_to_num_array(args.lr_sched))

    print("Defined loss and optimizer")

    best_prec5 = 93 # only save models over 92%. Otherwise it stops to save every time
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.local_rank))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else: print("=> no checkpoint found at '{}'".format(args.resume))

    dm = DataManager(str_to_num_array(args.resize_sched))
    print("Created data loaders")

    if args.evaluate: return validate(dm.get_val_iter(), len(dm.val_dl), model, criterion, 0, start_time)

    print("Begin training")
    estart = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        estart = time.time()
        dm.set_epoch(epoch)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            train(dm.get_trn_iter(), len(dm.trn_dl), model, criterion, optimizer, scheduler, epoch)

        if args.prof: break
        prec5 = validate(dm.get_val_iter(), len(dm.val_dl), model, criterion, epoch, start_time)

        is_best = prec5 > best_prec5
        if args.local_rank == 0 and is_best:
            best_prec5 = max(prec5, best_prec5)
            save_checkpoint({
                'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                'best_prec5': best_prec5, 'optimizer' : optimizer.state_dict(),
            }, is_best)

def str_to_num_array(argstr):
    return [float(s) for s in argstr.split(',')]

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def train(trn_iter, trn_len, model, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    st = time.time()
    # print('Begin training loop:', st)
    for i,(input,target) in enumerate(trn_iter):
        # if i == 0: print('Received input:', time.time()-st)
        if args.prof and (i > 200): break

        # measure data loading time
        data_time.update(time.time() - end)
        scheduler.update_lr(epoch, i, trn_len)


        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        loss = loss*args.loss_scale
        # compute gradient and do SGD step
        # if i == 0: print('Evaluate and loss:', time.time()-st)

        if args.fp16:
            model.zero_grad()
            loss.backward()
            model_grads_to_master_grads(model_params, master_params)
            for param in master_params:
                param.grad.data = param.grad.data/args.loss_scale
            optimizer.step()
            master_params_to_model_params(model_params, master_params)
            torch.cuda.synchronize()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if i == 0: print('Backward step:', time.time()-st)
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        should_print = ((i+1) % args.print_freq == 0) or (i+1 == trn_len)
        if args.local_rank == 0 and should_print:
            output = ('Epoch: [{0}][{1}/{2}]\t' \
                    + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    + 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                    + 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    + 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                    + 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                    epoch, i+1, trn_len, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5)
            print(output)
            with open(f'{args.save_dir}/full.log', 'a') as f:
                f.write(output + '\n')
    
def validate(val_iter, val_len, model, criterion, epoch, start_time):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()

    for i,(input,target) in enumerate(val_iter):
        if args.distributed:
            prec1, prec5, loss, tot_batch = distributed_predict(input, target, model, criterion)
        else:
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target).data
            tot_batch = input.size(0)
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            
        losses.update(to_python_float(loss), tot_batch)
        top1.update(to_python_float(prec1), tot_batch)
        top5.update(to_python_float(prec5), tot_batch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        should_print = ((i+1) % args.print_freq == 0) or (i+1 == val_len)
        if args.local_rank == 0 and should_print:
            output = ('Test: [{0}/{1}]\t' \
                    + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    + 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    + 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                    + 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                    i+1, val_len, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5)
            print(output)
            with open(f'{args.save_dir}/full.log', 'a') as f:
                f.write(output + '\n')

    time_diff = datetime.now()-start_time
    print(f'~~{epoch}\t{float(time_diff.total_seconds() / 3600.0)}\t{top5.avg:.3f}\n')
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top5.avg

def distributed_predict(input, target, model, criterion):
    batch_size = input.size(0)
    output = loss = corr1 = corr5 = valid_batches = torch.tensor([0]).cuda()
    
    if batch_size:
        # compute output
        with torch.no_grad():
            # using module instead of model because DistributedDataParallel forward function has a sync point.
            # with distributed validation sampler, we don't always have data for each gpu
            assert(isinstance(model, nn.parallel.DistributedDataParallel))
            output = model.module(input)
            loss = criterion(output, target)
        # measure accuracy and record loss
        valid_batches = torch.tensor([1]).cuda()
        corr1, corr5 = correct(output.data, target, topk=(1, 5))
    batch_tensor = torch.tensor([batch_size]).cuda()
    tot_batch = sum_tensor(batch_tensor).item()
    valid_batches = sum_tensor(valid_batches).item()
    reduced_loss = sum_tensor(loss.data)/valid_batches

    corr1 = sum_tensor(corr1).float()
    corr5 = sum_tensor(corr5).float()
    prec1 = corr1*(100.0/tot_batch)
    prec5 = corr5*(100.0/tot_batch)
    return prec1, prec5, reduced_loss, tot_batch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{args.save_dir}/model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    corrrect_ks = correct(output, target, topk)
    batch_size = target.size(0)
    return [correct_k.float().mul_(100.0 / batch_size) for correct_k in corrrect_ks]

def correct(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum(0, keepdim=True)
        res.append(correct_k)
    return res


def sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__': main()

