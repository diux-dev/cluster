import argparse, os, shutil, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import os
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

from tensorboardX import SummaryWriter

# import models
from fp16util import *
import gc

import resnet

from dataloader import *

from experimental_utils import *

################################################################################
# Generic utility methods, eventually refactor into separate file
################################################################################
def network_bytes():
  """Returns received bytes, transmitted bytes."""
  
  import subprocess
  proc = subprocess.Popen(['cat', '/proc/net/dev'], stdout=subprocess.PIPE)
  stdout,stderr = proc.communicate()
  stdout=stdout.decode('ascii')

  recv_bytes = 0
  transmit_bytes = 0
  lines=stdout.strip().split('\n')
  lines = lines[2:]  # strip header
  for line in lines:
    line = line.strip()
    # ignore loopback interface
    if line.startswith('lo'):
      continue
    toks = line.split()

    recv_bytes += int(toks[1])
    transmit_bytes += int(toks[9])
  return recv_bytes, transmit_bytes

# no_op method/object that accept every signature
def no_op(*args, **kwargs): pass
class NoOp:
  def __getattr__(self, *args):
    return no_op
################################################################################

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
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-b', '--batch-sched', default='192,192,128', type=str,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--resize-sched', default='0,18,41', type=str,
                        help='Scheduler to resize from 128 -> 224 -> 288')
    parser.add_argument('--use-352-folder', action='store_true', help='Train images from 352 resized folder - faster at cost of accuracy')
    parser.add_argument('--lr-sched', default='5,21,35,43', type=str,
                        help='Learning rate scheduler warmup -> lr -> lr/10 -> lr/100 -> lr/1000')
    parser.add_argument('--lr-linear-scale', action='store_true',
                        help='Linear scale the learning rate if we change the batch size later on')
    parser.add_argument('--init-bn0', action='store_true', help='Intialize running batch norm mean to 0')
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='print every this many steps (default: 5)')
    parser.add_argument('--no-bn-wd', action='store_true', help='Remove batch norm from weight decay')
    parser.add_argument('--factorized-resnet', action='store_true', help='Speed up convolutions by factorizing - https://arxiv.org/pdf/1608.04337.pdf')
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
    parser.add_argument('--c10d', action='store_true', help='Run distributed training with c10d')
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='total number of processes (machines*gpus)')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--logdir', default='', type=str,
                        help='where logs go')
    return parser

cudnn.benchmark = True
args = get_parser().parse_args()
if args.local_rank > 0: sys.stdout = open(f'{args.save_dir}/GPU_{args.local_rank}.log', 'w')
if args.c10d:
    assert(args.distributed)
    from torch.distributed import c10d
    from torch.nn.parallel import distributed_c10d

class DataManager():
    def __init__(self, resize_sched, batch_sched):
        self.resize_sched = resize_sched
        self.batch_sched = batch_sched
        if len(batch_sched) == 1: self.batch_sched = self.batch_sched * 3

        self.data0 = self.load_data('-sz/160', '-sz/160', self.batch_sched[0], 128)
        if args.use_352_folder: self.data1 = self.load_data('-sz/352', '', self.batch_sched[1], 224, min_scale=0.086) # faster loading at the cost of accuracy
        else: self.data1 = self.load_data('', '', self.batch_sched[1], 224)
        self.data2 = self.load_data('', '', self.batch_sched[2], 288, min_scale=0.5, use_ar=args.val_ar)
        
    def set_epoch(self, epoch):
        if epoch==self.resize_sched[0]: self.set_data(self.data0)
        if epoch==self.resize_sched[1]:
            if args.lr_linear_scale: args.lr = args.lr * self.batch_sched[1]/self.batch_sched[0]
            self.set_data(self.data1)
        if epoch==self.resize_sched[2]:
            if args.lr_linear_scale: args.lr = args.lr * self.batch_sched[2]/self.batch_sched[1]
            self.set_data(self.data2)

        if hasattr(self.trn_smp, 'set_epoch'): self.trn_smp.set_epoch(epoch)
        if hasattr(self.val_smp, 'set_epoch'): self.val_smp.set_epoch(epoch)
    
    def get_trn_loader(self): return DataPrefetcher(self.trn_dl)
    def get_val_loader(self): return DataPrefetcher(self.val_dl)

    def set_data(self, data):
        """Initializes data loader."""
        global last_batch_size, last_image_size
        loaders, data_info = data
        data_info_string = f'Dataset changed. \nImage size: {data_info["image_size"]} \nBatch size: {data_info["batch_size"]} \nTrain Directory: {data_info["traindir"]}\nValidation Directory: {data_info["valdir"]}'

        print(data_info_string)
        log_tb('sizes/image', data_info['image_size'])
        log_tb('sizes/batch', data_info['batch_size'])
        log_tb('sizes/world', args.world_size)
        last_batch_size = data_info['batch_size']
        last_image_size = data_info['image_size']
        self.trn_dl,self.val_dl,self.trn_smp,self.val_smp = loaders
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()
        

    def load_data(self, dir_prefix, valdir_prefix, batch_size, image_size, **kwargs):
        """Pre-initializes data-loaders. Use set_data to start using it."""
        traindir = args.data+dir_prefix+'/train'
        valdir = args.data+valdir_prefix+'/validation'

        data_info = {}
        data_info['image_size'] = image_size
        data_info['batch_size'] = batch_size
        data_info['traindir'] = traindir
        data_info['valdir'] = valdir
        
        return get_loaders(traindir, valdir, bs=batch_size, sz=image_size, workers=args.workers, distributed=args.distributed, **kwargs), data_info

class Scheduler():
    def __init__(self, optimizer, lr_sched):
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
            starting_lr = starting_lr/(epoch_tot - epoch)
        return lr

    def get_lr(self, epoch, batch_num, batch_tot):
        """Sets the learning rate to the initial LR decayed by 10 every few epochs"""
        # faster lr schedule [0.14, 0.43, 0.73, 0.94]
        # original lr schedule [0.1, 0.47, 0.78, 0.95]
        if epoch<=self.lr_sched[0]:
            if args.init_bn0: return self.bn0_lr_warmup(epoch, self.lr_sched[0], batch_num, batch_tot)
            else:             return self.linear_lr_warmup(epoch, self.lr_sched[0], batch_num, batch_tot)
        elif epoch<=self.lr_sched[1]: return args.lr/1
        elif epoch<=self.lr_sched[2]: return args.lr/10
        elif epoch<=self.lr_sched[3]: return args.lr/100
        else                        : return args.lr/1000

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot)
        if (self.current_lr != lr) and ((batch_num == 1) or (batch_num == batch_tot)): 
            print(f'Changing LR from {self.current_lr} to {lr}')
            log_tb('lr', lr)

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
    if not args.arch.startswith('resnet'): return
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock): m.bn2.weight = Parameter(torch.zeros_like(m.bn2.weight))
        if isinstance(m, resnet.Bottleneck): m.bn3.weight = Parameter(torch.zeros_like(m.bn3.weight))
        if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

def log_tb(tag, val):
  """Log value to tensorboard (relies on global_example_count being set properly)"""
  global global_example_count, event_writer
  event_writer.add_scalar(tag, val, global_example_count)
  
def main():
    # is_chief indicates this machine will do shared tasks for the cluster
    # such as logging and checkpointing
    # is_chief must be true only for at most 1 process in training cluster
    # $RANK is set by pytorch.distributed.launch
    # https://github.com/pytorch/pytorch/blob/db6e4576dab097abf01d032c3326e4b285eb8499/torch/distributed/launch.py#L193
    global is_chief, event_writer, global_example_count, last_recv_bytes, last_transmit_bytes, last_log_time

    is_chief = (not args.distributed) or (int(os.environ['RANK'])==0)

    global_example_count = 0
    if is_chief:
      print(f"Logging to {args.logdir}")
      event_writer = SummaryWriter(args.logdir)
      log_tb("first", time.time())
    else:
      event_writer = NoOp()

    # baseline number for network bytes
    last_recv_bytes, last_transmit_bytes = network_bytes()
    last_log_time = time.time()
    
    print(args)
    print("~~epoch\thours\ttop1Accuracy\n")

    # need to index validation directory before we start counting the time
    if args.val_ar: sort_ar(args.data+'/validation')
    
    global reduce_function
    if args.c10d:
        print('Distributed: loading c10d process group')
        # https://github.com/pytorch/pytorch/blob/master/torch/lib/c10d/TCPStore.hpp
        torch.cuda.set_device(args.local_rank)
        rank = int(os.environ['RANK'])
        store = c10d.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']), rank==0) # (masterAddr, masterPort, isServer) 
        process_group = c10d.ProcessGroupNCCL(store, rank, args.world_size) # (store, rank, size)
        reduce_function = lambda t: process_group.allreduce(t, c10d.AllreduceOptions().reduceOp)
    elif args.distributed:
        print('Distributed: initializing process group')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)
        assert(args.world_size == dist.get_world_size())
        reduce_function = lambda t: dist.all_reduce(t, op=dist.reduce_op.SUM)
        print("Distributed: success (%d/%d)"%(args.local_rank, args.world_size))

    if args.fp16: assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    print("Loading model")
    if args.factorized_resnet: model = resnet.resnet50factorized(pretrained=args.pretrained)
    else: model = resnet.resnet50(pretrained=args.pretrained)

    model = model.cuda()
    if args.init_bn0: init_dist_weights(model) # Sets batchnorm std to 0
    if args.fp16: model = network_to_half(model)
    best_prec5 = 93 # only save models over 92%. Otherwise it stops to save every time

    # Load model from checkpoint. This must happen distributed as model is saved without it
    if args.resume:
        checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.local_rank))
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_prec5 = checkpoint['best_prec5']

    if args.c10d:
        model = distributed_c10d._DistributedDataParallelC10d(model, process_group, device_ids=[args.local_rank], output_device=args.local_rank)
        c10d_sanity_check()
    elif args.distributed: model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    global model_params, master_params
    if args.fp16: model_params, master_params = prep_param_lists(model)
    else: master_params = list(model.parameters())

    optim_params = bnwd_optim_params(model, model_params, master_params) if args.no_bn_wd else master_params

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(optim_params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = Scheduler(optimizer, str_to_num_array(args.lr_sched))

    if args.resume: # we must load optimizer params separately
        checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.local_rank))
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("Creating data loaders (this could take 6-12 minutes)")
    dm = DataManager(resize_sched=str_to_num_array(args.resize_sched), batch_sched=str_to_num_array(args.batch_sched))

    start_time = datetime.now() # Loading start to after everything is loaded
    if args.evaluate: return validate(dm.get_val_loader(), model, criterion, 0, start_time)
    print("Begin training")
    estart = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        estart = time.time()
        dm.set_epoch(epoch)
        train(dm.get_trn_loader(), model, criterion, optimizer, scheduler, epoch)
        if args.prof: break
        prec5 = validate(dm.get_val_loader(), model, criterion, epoch, start_time)

        is_best = prec5 > best_prec5
        best_prec5 = max(prec5, best_prec5)
        if args.local_rank == 0:
            if is_best: save_checkpoint(epoch, model, best_prec5, optimizer, is_best=True, filename='model_best.pth.tar')
            if (epoch+1)==int(args.epochs*dm.resize_sched[0]+0.5):
                save_checkpoint(epoch, model, best_prec5, optimizer, filename='sz128_checkpoint.path.tar')
            elif (epoch+1)==int(args.epochs*dm.resize_sched[1]+0.5):
                save_checkpoint(epoch, model, best_prec5, optimizer, filename='sz244_checkpoint.path.tar')

    event_writer.export_scalars_to_json(args.logdir+'/scalars.json')
    event_writer.close()

    
def str_to_num_array(argstr, num_type=int):
    return [num_type(s) for s in argstr.split(',')]

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if isinstance(t, float): return t
    if hasattr(t, 'item'): return t.item()
    else: return t[0]

def c10d_sanity_check():
    print('Sanity check to make sure tensor creation works')
    tt = torch.tensor([1]).float().cuda()
    print('Currently deadlock here', tt)
    print('Woot able to reduce tensor:', sum_tensor(tt))

def train(trn_loader, model, criterion, optimizer, scheduler, epoch):
    global is_chief, event_writer, global_example_count, last_recv_bytes, last_transmit_bytes, last_log_time

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    recv_meter = AverageMeter()
    transmit_meter = AverageMeter()
    
    # switch to train mode
    model.train()
    end = time.time()

    st = time.time()
    trn_len = len(trn_loader)

    # print('Begin training loop:', st)
    for i,(input,target) in enumerate(iter(trn_loader)):
        batch_size = input.size(0)
        batch_num = i+1
        # if i == 0: print('Received input:', time.time()-st)
        if args.prof and (i > 200): break

        # measure data loading time
        data_time.update(time.time() - end)
        scheduler.update_lr(epoch, i+1, trn_len)


        # compute output
        output = model(input)
        loss = criterion(output, target)

        if args.distributed:
            # Must keep track of global batch size, since not all machines are guaranteed equal batches at the end of an epoch
            corr1, corr5 = correct(output.data, target, topk=(1, 5))
            metrics = torch.tensor([batch_size, loss, corr1, corr5]).float().cuda()
            batch_total, reduced_loss, corr1, corr5 = sum_tensor(metrics)
            reduced_loss = reduced_loss/args.world_size
            prec1 = corr1*(100.0/batch_total)
            prec5 = corr5*(100.0/batch_total)
        else:
            reduced_loss = loss.data
            batch_total = input.size(0)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5)) # measure accuracy and record loss

        losses.update(to_python_float(reduced_loss), to_python_float(batch_total))
        top1.update(to_python_float(prec1), to_python_float(batch_total))
        top5.update(to_python_float(prec5), to_python_float(batch_total))

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

        should_print = (batch_num%args.print_freq == 0) or (batch_num==trn_len)
        if args.local_rank == 0 and should_print:
            log_tb("times/step", 1000*batch_time.val)
            log_tb("times/data", 1000*data_time.val)
            log_tb("losses/xent", losses.val)
            log_tb("losses/train_1", top1.val)   # precision@1
            log_tb("losses/train_5", top5.val)   # precision@5
            images_per_sec = batch_size/batch_time.val
            log_tb("times/1gpu_images_per_sec", images_per_sec)
            log_tb("times/8gpu_images_per_sec", 8*images_per_sec)

            time_delta = time.time()-last_log_time
            recv_bytes, transmit_bytes = network_bytes()
            
            recv_delta = recv_bytes - last_recv_bytes
            transmit_delta = transmit_bytes - last_transmit_bytes

            # turn into Gbps
            recv_gbit = 8*recv_delta/time_delta/1e9
            transmit_gbit = 8*recv_delta/time_delta/1e9
            
            last_log_time = time.time()
            last_recv_bytes = recv_bytes
            last_transmit_bytes = transmit_bytes

            recv_meter.update(recv_gbit)
            transmit_meter.update(transmit_gbit)
            log_tb('net/recv_gbit', recv_gbit)
            log_tb('net/transmit_gbit', transmit_gbit)
            
            output = ('Epoch: [{0}][{1}/{2}]\t' \
                    + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    + 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                    + 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    + 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                    + 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t' \
                      + 'bw {recv_meter.val:.3f} {transmit_meter.val:.3f}').format(
                    epoch, batch_num, trn_len, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5,
                      recv_meter=recv_meter, transmit_meter=transmit_meter)
            print(output)
            with open(f'{args.save_dir}/full.log', 'a') as f:
                f.write(output + '\n')

        global_example_count+=batch_total

             
            
    # save script so we can reproduce from logs
    shutil.copy2(os.path.realpath(__file__), f'{args.save_dir}')
    shutil.copy2(os.path.realpath(__file__), f'{args.logdir}')

    
def validate(val_loader, model, criterion, epoch, start_time):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    eval_start_time = time.time()
    
    model.eval()
    end = time.time()
    val_len = len(val_loader)

    for i,(input,target) in enumerate(iter(val_loader)):
        batch_num = i+1
        if args.distributed:
            prec1, prec5, loss, batch_total = distributed_predict(input, target, model, criterion)
        else:
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target).data
            batch_total = input.size(0)
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            
        losses.update(to_python_float(loss), to_python_float(batch_total))
        top1.update(to_python_float(prec1), to_python_float(batch_total))
        top5.update(to_python_float(prec5), to_python_float(batch_total))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        should_print = (batch_num%args.print_freq == 0) or (batch_num==val_len)
        if args.local_rank == 0 and should_print:
            output = ('Test: [{0}/{1}]\t' \
                    + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    + 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    + 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                    + 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                    batch_num, val_len, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5)
            print(output)
            with open(f'{args.save_dir}/full.log', 'a') as f:
                f.write(output + '\n')

    time_diff = datetime.now()-start_time
    print(f'~~{epoch}\t{float(time_diff.total_seconds() / 3600.0)}\t{top5.avg:.3f}\n')
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    
    log_tb('losses/test_1', top1.avg)
    log_tb('losses/test_5', top5.avg)
    log_tb('times/eval_sec', time.time()-eval_start_time)
    log_tb('epoch', epoch)


    return top5.avg

def distributed_predict(input, target, model, criterion):
    batch_size = input.size(0)
    output = loss = corr1 = corr5 = valid_batches = 0
    
    if batch_size:
        # compute output
        with torch.no_grad():
            # using module instead of model because DistributedDataParallel forward function has a sync point.
            # with distributed validation sampler, we don't always have data for each gpu
            assert(is_distributed_model(model))
            output = model.module(input)
            loss = criterion(output, target).data
        # measure accuracy and record loss
        valid_batches = 1
        corr1, corr5 = correct(output.data, target, topk=(1, 5))

    metrics = torch.tensor([batch_size, valid_batches, loss, corr1, corr5]).float().cuda()
    batch_total, valid_batches, reduced_loss, corr1, corr5 = sum_tensor(metrics)
    reduced_loss = reduced_loss/valid_batches

    prec1 = corr1*(100.0/batch_total)
    prec5 = corr5*(100.0/batch_total)
    return prec1, prec5, reduced_loss, batch_total


def save_checkpoint(epoch, model, best_prec5, optimizer, is_best=False, filename='checkpoint.pth.tar'):
    if is_distributed_model(model): model = model.module # do not save distributed module. Makes loading from checkpoint more flexible
    state = {
        'epoch': epoch+1, 'state_dict': model.state_dict(),
        'best_prec5': best_prec5, 'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, filename)
    if is_best: shutil.copyfile(filename, f'{args.save_dir}/{filename}')

def is_distributed_model(model):
    return isinstance(model, nn.parallel.DistributedDataParallel) or (args.c10d and isinstance(model, distributed_c10d._DistributedDataParallelC10d))

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


def reduce_tensor(tensor): return sum_tensor(tensor)/args.world_size
def sum_tensor(tensor):
    rt = tensor.clone()
    reduce_function(rt)
    return rt

if __name__ == '__main__': 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()

