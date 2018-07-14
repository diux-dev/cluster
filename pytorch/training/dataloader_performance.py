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
# from daliloader import HybridPipe

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
    parser.add_argument('-b', '--batch-size', default=256, type=int,
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
        self.load_data('-sz/160', 256, 128)
        
    def set_epoch(self, epoch):
        if epoch==2:
            self.load_data('-sz/320', 256, 224) # lower validation accuracy when enabled for some reason
        if epoch==4:
            self.load_data('', 256, 224)
        if epoch==6:
            self.load_data('-sz/320', 256, 256) 
        if epoch==8:
            self.load_data('', 256, 288, min_scale=0.5, use_ar=args.val_ar)

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
        print(f'Dataset changing. \nImage size: {image_size}. \nBatch size: {batch_size} \nDirectory: {dir_prefix}\n')
        estart = time.time()
        loaders = get_loaders(args.data+dir_prefix, bs=batch_size, sz=image_size, workers=args.workers, distributed=args.distributed, **kwargs)
        self.trn_dl,self.val_dl,self.trn_smp,self.val_smp = loaders
        self.trn_dl = DataPrefetcher(self.trn_dl)
        self.val_dl = DataPrefetcher(self.val_dl, prefetch=False)
        self.trn_len = len(self.trn_dl)
        self.val_len = len(self.val_dl)
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()
        endtime = time.time() - estart
        print(f'Time took to load data: {endtime}')


    # def dali_load_data(self, dir_prefix, batch_size, image_size, **kwargs):
    #     print(f'Dali Dataset changing. \nImage size: {image_size}. \nBatch size: {batch_size} \nDirectory: {dir_prefix}\n')
    #     traindir = args.data+dir_prefix+'/train'
    #     valdir = args.data+dir_prefix+'/validation'

    #     pipe = HybridPipe(batch_size=batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir)
    #     pipe.build()
    #     test_run = pipe.run()
    #     from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    #     train_loader = DALIClassificationIterator(pipe, size=int(1281167 / args.world_size) )


    #     pipe = HybridPipe(batch_size=batch_size, num_threads=args.workers, device_id = args.local_rank, data_dir=valdir)
    #     pipe.build()
    #     test_run = pipe.run()
    #     from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    #     val_loader = DALIClassificationIterator(pipe, size=int(50000 / args.world_size) )

    #     self.trn_dl = train_loader
    #     self.val_dl = val_loader
    #     self.trn_len = len(self.trn_dl)
    #     self.val_len = len(self.val_dl)
    #     # clear memory
    #     gc.collect()
    #     torch.cuda.empty_cache()

def main():
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
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    global model_params, master_params
    if args.fp16:  model_params, master_params = prep_param_lists(model)
    else: master_params = list(model.parameters())

    criterion = nn.CrossEntropyLoss().cuda()

    print("Creating data loaders")
    dm = DataManager(str_to_num_array(args.resize_sched))

    if args.evaluate: return validate(dm.get_val_iter(), len(dm.val_dl), model, criterion, 0, start_time)

    print("Begin training")
    for epoch in range(10):
        estart = time.time()
        dm.set_epoch(epoch)

        train_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            train(dm.get_trn_iter(), len(dm.trn_dl), model, criterion, epoch)
        train_end = (time.time() - train_start)
        num_images = 1281167
        img_per_sec = num_images / train_end
        print(f'Training processed {img_per_sec} images per second.\n')


        val_start = time.time()
        validate(dm.get_val_iter(), len(dm.val_dl), model, criterion, epoch, start_time)
        val_end = (time.time() - val_start)
        num_images = 50000
        img_per_sec = num_images / train_end
        print(f'Validation processed {img_per_sec} images per second.\n')

        num_images = 1281167+50000
        end_time = (time.time() - estart)
        img_per_sec = num_images / end_time
        print(f'Total images processed per second: {img_per_sec}\n')

def str_to_num_array(argstr):
    return [float(s) for s in argstr.split(',')]

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def train(trn_iter, trn_len, model, criterion, epoch):
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
    for i,data in enumerate(trn_iter):
        # if i == 0: print('Received input:', time.time()-st)

        # measure data loading time
        data_time.update(time.time() - end)


        # compute output
        with torch.no_grad():	
            fake_input = torch.zeros([1,3,64,64]).cuda()	
            if args.fp16: fake_input = fake_input.half()	
            _ = model(fake_input)

        # compute gradient and do SGD step
        # if i == 0: print('Evaluate and loss:', time.time()-st)

        torch.cuda.synchronize()

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

    for i,data in enumerate(val_iter):
        
        # compute output
        with torch.no_grad():	
            fake_input = torch.zeros([1,3,64,64]).cuda()	
            if args.fp16: fake_input = fake_input.half()	
            _ = model(fake_input)

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
    print(f'~~{epoch}\t{float(time_diff.total_seconds() / 3600.0)}\t')

    return top5.avg

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

