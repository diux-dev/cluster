
import torch
from pathlib import Path
import os
import numpy as np
import torch.nn as nn
from datetime import datetime

import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse, os, shutil, time, warnings
import torch.distributed as dist
import torch.utils.data.distributed

from fp16util import *
from resnet import *
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--phases', default='[(0,2e-1,16),(2e-1,1e-2,16),(1e-2,0,5)]', type=str,
                    help='Should be a string formatted like this: [(start_lr,end_lr,num_epochs),(phase2...)]')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
#     parser.add_argument('--init-bn0', action='store_true', help='Intialize running batch norm mean to 0')
    parser.add_argument('--print-freq', '-p', default=200, type=int,
                        metavar='N', help='print every this many steps (default: 5)')
#     parser.add_argument('--no-bn-wd', action='store_true', help='Remove batch norm from weight decay')
    parser.add_argument('--full-precision', action='store_true', help='Run model full precision mode. Default fp16')
    parser.add_argument('--loss-scale', type=float, default=512,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--distributed', action='store_true', help='Run distributed training')
    parser.add_argument('--scale-lr', type=float, default=1, help='You should learning rate propotionally to world size')
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='total number of processes (machines*gpus)')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    return parser


global args
args = get_parser().parse_args()
torch.backends.cudnn.benchmark = True

class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass
import sys
if args.local_rank > 0: sys.stdout = DummyFile()


# Model
class PreActBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class ResNet18(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__()
        
        self.in_channels = 64
        
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_blocks[0], stride=1),
            self._make_layer(64, 128, num_blocks[1], stride=2),
            self._make_layer(128, 256, num_blocks[2], stride=2),
            self._make_layer(256, 256, num_blocks[3], stride=2),
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.prep(x)
        
        x = self.layers(x)
        
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)
        
        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)
        
        x = torch.cat([x_avg, x_max], dim=-1)
        
        x = self.classifier(x)
        
        return x


# ### Torch loader
def pad(img, p=4, padding_mode='reflect'):
    return Image.fromarray(np.pad(np.asarray(img), ((p, p), (p, p), (0, 0)), padding_mode))

def torch_loader(data_path, size, bs, val_bs=None):

    val_bs = val_bs or bs
    # Data loading code
    tfms = [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703,0.24349,0.26159))]

    train_tfms = transforms.Compose([
        pad, # TODO: use `padding` rather than assuming 4
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
    ] + tfms)
    val_tfms = transforms.Compose(tfms)

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=(args.local_rank==0), transform=train_tfms)
    val_dataset  = datasets.CIFAR10(root=data_path, train=False, download=(args.local_rank==0), transform=val_tfms)

    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None)
    # val_sampler = (torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None)
    val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler)
    
    train_loader = DataPrefetcher(train_loader)
    val_loader = DataPrefetcher(val_loader)
    
    return train_loader, val_loader

# Seems to speed up training by ~2%
class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break


# ### Learning rate scheduler
class Scheduler():
    def __init__(self, optimizer, phases=[(0,2e-1,15),(2e-1,1e-2,15),(1e-2,0,5)]):
        self.optimizer = optimizer
        self.current_lr = None
        self.phases = phases
        self.tot_epochs = sum([p[2] for p in phases])

    def linear_lr(self, start_lr, end_lr, epoch_curr, batch_curr, epoch_tot, batch_tot):
        if args.scale_lr != 1:
            start_lr *= args.scale_lr
            end_lr *= args.scale_lr
        step_tot = epoch_tot * batch_tot
        step_curr = epoch_curr * batch_tot + epoch_curr
        step_size = (end_lr - start_lr)/step_tot
        return start_lr + step_curr * step_size
    
    def get_current_phase(self, epoch):
        epoch_accum = 0
        for phase in self.phases:
            start_lr,end_lr,num_epochs = phase
            if epoch <= epoch_accum+num_epochs: return start_lr, end_lr, num_epochs, epoch - epoch_accum
            epoch_accum += num_epochs
        raise Exception('Epoch out of range')
            
    def get_lr(self, epoch, batch_curr, batch_tot):
        start_lr, end_lr, num_epochs, relative_epoch = self.get_current_phase(epoch)
        return self.linear_lr(start_lr, end_lr, relative_epoch, batch_curr, num_epochs, batch_tot)

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot)
        if args.verbose and (self.current_lr != lr) and ((batch_num == 1) or (batch_num == batch_tot)): 
            print(f'Changing LR from {self.current_lr} to {lr}')

        self.current_lr = lr

        for param_group in self.optimizer.param_groups:
            lr_old = param_group['lr'] or lr
            param_group['lr'] = lr


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if isinstance(t, float): return t
    if isinstance(t, int): return t
    if hasattr(t, 'item'): return t.item()
    else: return t[0]

def train(trn_loader, model, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    st = time.time()
    trn_len = len(trn_loader)

    # print('Begin training loop:', st)
    for i,(input,target) in enumerate(trn_loader):
        batch_size = input.size(0)
        batch_num = i+1
        # if i == 0: print('Received input:', time.time()-st)

        # measure data loading time
        scheduler.update_lr(epoch, i+1, trn_len)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        if args.distributed:
            # Must keep track of global batch size, since not all machines are guaranteed equal batches at the end of an epoch
            corr1 = correct(output.data, target)[0]
            metrics = torch.tensor([batch_size, loss, corr1]).float().cuda()
            batch_total, reduced_loss, corr1 = sum_tensor(metrics)
            reduced_loss = reduced_loss/dist.get_world_size()
            prec1 = corr1*(100.0/batch_total)
        else:
            reduced_loss = loss.data
            batch_total = input.size(0)
            prec1 = accuracy(output.data, target)[0] # measure accuracy and record loss
        losses.update(to_python_float(reduced_loss), to_python_float(batch_total))
        top1.update(to_python_float(prec1), to_python_float(batch_total))

        loss = loss*args.loss_scale
        
        # compute gradient and do SGD step
        if args.full_precision:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            model.zero_grad()
            loss.backward()
            model_grads_to_master_grads(model_params, master_params)
            for param in master_params:
                param.grad.data = param.grad.data/args.loss_scale
            optimizer.step()
            master_params_to_model_params(model_params, master_params)
            torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()

        should_print = (batch_num%args.print_freq == 0) or (batch_num==trn_len)
        if should_print: log_batch(epoch, batch_num, trn_len, batch_time, losses, top1)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, epoch, start_time):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    val_len = len(val_loader)

    for i,(input,target) in enumerate(val_loader):
        batch_num = i+1
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target).data
        batch_total = input.size(0)
        prec1 = accuracy(output.data, target)[0]
            
        losses.update(to_python_float(loss), batch_total)
        top1.update(to_python_float(prec1), batch_total)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        should_print = (batch_num%args.print_freq == 0) or (batch_num==val_len)
        if should_print: log_batch(epoch, batch_num, val_len, batch_time, losses, top1)
            
    return top1.avg, losses.avg

def log_batch(epoch, batch_num, batch_len, batch_time, loss, top1):
    if args.local_rank==0 and args.verbose:
        output = ('Epoch: [{0}][{1}/{2}]\t'                 
                + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'                 
                + 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'                 
                + 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})').format(
                epoch, batch_num, batch_len, batch_time=batch_time, loss=loss, top1=top1)
        print(output)
        with open(f'{args.save_dir}/full.log', 'a') as f:
            f.write(output + '\n')
            
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
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


def main():
    if args.distributed:
        print('Distributed: initializing process group')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)
        assert(args.world_size == dist.get_world_size())
        print("Distributed: success (%d/%d)"%(args.local_rank, args.world_size))

    model = ResNet18()
    model = model.cuda()

    # AS: todo: don't copy over weights as it seems to help performance

    if not args.full_precision: model = network_to_half(model)
    elif args.distributed: model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


    global model_params, master_params
    if args.full_precision: master_params = list(model.parameters())
    else: model_params, master_params = prep_param_lists(model)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(master_params, lr=0, nesterov=True, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = Scheduler(optimizer, phases=eval(args.phases))


    sz = 32
    trn_loader, val_loader = torch_loader(args.data, sz, args.batch_size, args.batch_size*2)

    print(args)
    print('\n\n')
    print("epoch\t\tnum_batch\ttime (min)\ttrn_loss\tval_loss\taccuracy")
    start_time = datetime.now() # Loading start to after everything is loaded
    for epoch in range(scheduler.tot_epochs):
        trn_top1, trn_loss = train(trn_loader, model, criterion, optimizer, scheduler, epoch)
        val_top1, val_loss = validate(val_loader, model, criterion, epoch, start_time)

        time_diff = datetime.now()-start_time
        minutes = float(time_diff.total_seconds() / 60.0)
        # epoch   time   trn_loss   val_loss   accuracy     
        metrics = [str(round(i, 4)) for i in [epoch, len(trn_loader), minutes, trn_loss, val_loss, val_top1]]
        print('\t\t'.join(metrics))


if __name__ == '__main__': 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()
