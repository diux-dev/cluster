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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import models
from fp16util import *
import gc

import resnet
from torch.utils.data.sampler import Sampler
import torchvision
import pickle
from tqdm import tqdm
# import resnet_sd as resnet

# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))
#print(model_names)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
    #                     choices=model_names,
    #                     help='model architecture: ' +
    #                     ' | '.join(model_names) +
    #                     ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=0, type=int, metavar='N',
                        help='number of additional epochs to warmup')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--small', action='store_true', help='start with smaller images')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
    parser.add_argument('--sz',       default=224, type=int, help='Size of transformed image.')
    parser.add_argument('--decay-int', default=30, type=int, help='Decay LR by 10 every decay-int epochs')
    parser.add_argument('--loss-scale', type=float, default=1,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--prof', dest='prof', action='store_true', help='Only run a few iters for profiling.')
    parser.add_argument('--val-ar', action='store_true', help='Do final validation by nearest aspect ratio')

    parser.add_argument('--distributed', action='store_true', help='Run distributed training')
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='Number of gpus per machine. Param only needed for single machine training when using (faster) file sync')
    parser.add_argument('--dist-url', default='file://sync.file', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    return parser

cudnn.benchmark = True
args = get_parser().parse_args()
if args.local_rank > 0: sys.stdout = open(f'{args.save_dir}/GPU_{args.local_rank}.log', 'w')


def fast_collate(batch):
    if not batch: return torch.tensor([]), torch.tensor([])
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets


import os.path
def sort_ar(valdir):
    idx2ar_file = args.data+'/sorted_idxar.p'
    if os.path.isfile(idx2ar_file): return pickle.load(open(idx2ar_file, 'rb'))
    print('Creating AR indexes. Please be patient this may take a couple minutes...')
    val_dataset = datasets.ImageFolder(valdir)
    sizes = [img[0].size for img in tqdm(val_dataset, total=len(val_dataset))]
    idx_ar = [(i, round(s[0]/s[1], 5)) for i,s in enumerate(sizes)]
    sorted_idxar = sorted(idx_ar, key=lambda x: x[1])
    pickle.dump(sorted_idxar, open(idx2ar_file, 'wb'))
    print('Done')
    return sorted_idxar

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def map_idx2ar(idx_ar_sorted, batch_size):
    # idx2ar_map_file = args.data+f'/idxar_map_{batch_size}.p'
    # if os.path.isfile(idx2ar_map_file): return pickle.load(open(idx2ar_map_file, 'rb'))
    ar_chunks = list(chunks(idx_ar_sorted, batch_size))
    idx2ar = {}
    for chunk in ar_chunks:
        idxs, ars = list(zip(*chunk))
        mean = round(np.mean(ars), 5)
        for idx in idxs:
            idx2ar[idx] = mean
    # pickle.dump(idx2ar, open(idx2ar_map_file, 'wb'))
    return idx2ar

class ValDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            for tfm in self.transform:
                if isinstance(tfm, CropArTfm): sample = tfm(sample, index)
                else: sample = tfm(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

class ValDistSampler(Sampler):
    # min_batch_size - validation by nearest aspect ratio expects the batch size to be constant
    # Otherwise you'll mix different images with different aspect ratio's and tensor will not be constant size
    def __init__(self, indices, batch_size, distributed_batch=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed_batch and args.distributed: 
            self.world_size = get_world_size() 
            self.rank = dist.get_rank()
        else: 
            self.rank = 0
            self.world_size = 1
            
        # expected number of batches per sample. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_batches = math.ceil(len(self.indices) / self.world_size / self.batch_size)
        
        # num_samples = total images / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_batches * self.batch_size
        
    def __iter__(self):
        offset = self.num_samples * self.rank
        sampled_indices = self.indices[offset:offset+self.num_samples]
        for i in range(self.expected_num_batches):
            offset = i*self.batch_size
            yield sampled_indices[offset:offset+self.batch_size]
    def __len__(self): return self.expected_num_batches
    def set_epoch(self, epoch): return
    

class CropArTfm(object):
    def __init__(self, idx2ar, target_size):
        self.idx2ar, self.target_size = idx2ar, target_size
    def __call__(self, img, idx):
        target_ar = self.idx2ar[idx]
        if target_ar < 1: 
            w = int(self.target_size/target_ar)
            size = (w//8*8, self.target_size)
        else: 
            h = int(self.target_size*target_ar)
            size = (self.target_size, h//8*8)
        return torchvision.transforms.functional.center_crop(img, size)

def create_validation_set(valdir, batch_size, target_size, use_ar):
    idx_ar_sorted = sort_ar(valdir)
    idx_sorted, _ = zip(*idx_ar_sorted)
    idx2ar = map_idx2ar(idx_ar_sorted, batch_size)
    
    if use_ar:
        ar_tfms = [transforms.Resize(int(target_size*1.14)), CropArTfm(idx2ar, target_size)]
        val_dataset = ValDataset(valdir, transform=ar_tfms)
        val_sampler = ValDistSampler(idx_sorted, batch_size=batch_size)
        return val_dataset, val_sampler
    
    val_tfms = [transforms.Resize(int(args.sz*1.14)), transforms.CenterCrop(args.sz)]
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_tfms))
    val_sampler = ValDistSampler(list(range(len(val_dataset))), batch_size=batch_size)
    return val_dataset, val_sampler

def get_loaders(traindir, valdir, sz, bs, val_bs=None, use_ar=False, min_scale=0.08):
    val_bs = val_bs or bs
    train_dataset = datasets.ImageFolder(
        traindir, transforms.Compose([
            transforms.RandomResizedCrop(sz, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]))
    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, collate_fn=fast_collate, 
        sampler=train_sampler)

    val_dataset, val_sampler = create_validation_set(valdir, val_bs, sz, use_ar=use_ar)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=args.workers, pin_memory=True, collate_fn=fast_collate, 
        batch_sampler=val_sampler)
    return train_loader,val_loader,train_sampler,val_sampler


# Seems to speed up training by ~2%
class DataPrefetcher():
    def __init__(self, loader, prefetch=True):
        self.loader = loader
        self.prefetch = prefetch
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        if args.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.next_input = None
            self.next_target = None

    def __len__(self): return len(self.loader)

    def preload(self):
        self.next_input, self.next_target = next(self.loaditer)
        with torch.cuda.stream(self.stream):
            self.next_input = self.process_input(self.next_input)
            self.next_target = self.next_target.cuda(async=True)
    
    def process_input(self, input, async=True):
        input = input.cuda(async=async)
        if args.fp16: input = input.half()
        else: input = input.float()
        if len(input.shape) < 3: return input
        return input.sub_(self.mean).div_(self.std)
            
    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        if not self.prefetch:
            for input, target in self.loaditer:
                yield self.process_input(input), target.cuda()
            return
        self.preload()
        while True:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            yield input, target

class DataManager():
    def __init__(self):
        if args.small: self.load_data('-sz/160', args.batch_size, 128)
        # else: self.load_data('-sz/320', args.batch_size, 224)
        else: self.load_data('', args.batch_size, 224)
        
    def set_epoch(self, epoch):
        if epoch==int(args.epochs*0.4+0.5)+args.warmup:
            # self.load_data('-sz/320', args.batch_size, 224) # lower validation accuracy when enabled for some reason
            print('DataManager changing image size to 244')
            self.load_data('', args.batch_size, 224)
        if epoch==int(args.epochs*0.92+0.5)+args.warmup:
            print('DataManager changing image size to 288')
            # self.load_data('', 128, 288, val_bs=64, min_scale=0.5, use_ar=args.val_ar)
            self.load_data('', 128, 288, min_scale=0.5, use_ar=args.val_ar)
        if args.distributed:
            if self.trn_smp: self.trn_smp.set_epoch(epoch)
            if self.val_smp: self.val_smp.set_epoch(epoch)

    # def set_epoch(self, epoch):
    #     if epoch==int(args.epochs*0.35+0.5)+args.warmup:
    #         # self.load_data('-sz/320', args.batch_size, 224) # lower validation accuracy when enabled for some reason
    #         print('DataManager changing image size to 244')
    #         self.load_data('', args.batch_size, 224)
    #     if epoch==int(args.epochs*0.88+0.5)+args.warmup:
    #         print('DataManager changing image size to 288')
    #         # self.load_data('', 128, 288, val_bs=64, min_scale=0.5, use_ar=args.val_ar)
    #         self.load_data('', 128, 288, min_scale=0.5, use_ar=args.val_ar)
    #     if args.distributed:
    #         if self.trn_smp: self.trn_smp.set_epoch(epoch)
    #         if self.val_smp: self.val_smp.set_epoch(epoch)

    def get_trn_iter(self):
        # trn_iter = self.trn_iter
        self.trn_iter = iter(self.trn_dl)
        return self.trn_iter

    def get_val_iter(self):
        # val_iter = self.val_iter
        self.val_iter = iter(self.val_dl)
        return self.val_iter
        
    def load_data(self, dir_prefix, batch_size, image_size, **kwargs):
        traindir = args.data+dir_prefix+'/train'
        valdir = args.data+dir_prefix+'/validation'
        self.trn_dl,self.val_dl,self.trn_smp,self.val_smp = get_loaders(traindir, valdir, bs=batch_size, sz=image_size, **kwargs)
        self.trn_dl = DataPrefetcher(self.trn_dl)
        self.val_dl = DataPrefetcher(self.val_dl, prefetch=False)

        self.trn_len = len(self.trn_dl)
        self.val_len = len(self.val_dl)
        # self.trn_iter = iter(self.trn_dl)
        # self.val_iter = iter(self.val_dl)

        # clear memory
        gc.collect()
        torch.cuda.empty_cache()

class Scheduler():
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.current_lr = None
        self.current_epoch = 0

    def get_lr(self, epoch, batch_num, batch_tot):
        """Sets the learning rate to the initial LR decayed by 10 every few epochs"""
        # if epoch<int(args.epochs*0.14)+args.warmup:
        #     epoch_tot = int(args.epochs*0.14)+args.warmup
        #     world_size = get_world_size()
        #     # lr_step = (world_size/4 - 1) * args.lr / (epoch_tot * batch_tot)
        #     lr_step = args.lr / (epoch_tot * batch_tot)
        #     lr = args.lr + (epoch * batch_tot + batch_num) * lr_step

        #     if world_size == 32: lr /= 1.5
        #     if world_size == 64: lr /= 2

        # the following works best for 8 machines I think
        # Works better when
        # if epoch<int(args.epochs*0.14)+args.warmup:
        #     epoch_tot = int(args.epochs*0.14)+args.warmup
        #     starting_lr = args.lr/epoch_tot
        #     world_size = get_world_size()
        #     if (world_size > 20) and (epoch < 4):
        #         # starting_lr = starting_lr/(world_size/2)
        #         starting_lr = starting_lr/(4 - epoch)
        #     ending_lr = args.lr
        #     step_size = (ending_lr - starting_lr)/epoch_tot
        #     batch_step_size = step_size/batch_tot
        #     lr = step_size*epoch + batch_step_size*batch_num

            # lr = args.lr/(int(args.epochs*0.1)+args.warmup-epoch)
        # elif epoch<int(args.epochs*0.43+0.5)+args.warmup: lr = args.lr/1
        # elif epoch<int(args.epochs*0.73+0.5)+args.warmup: lr = args.lr/10
        # elif epoch<int(args.epochs*0.94+0.5)+args.warmup: lr = args.lr/100
        # else         : lr = args.lr/1000
        # return lr

        if epoch<int(args.epochs*0.1)+args.warmup:
            epoch_tot = int(args.epochs*0.1)+args.warmup
            starting_lr = args.lr/epoch_tot
            ending_lr = args.lr
            step_size = (ending_lr - starting_lr)/epoch_tot
            batch_step_size = step_size/batch_tot
            lr = step_size*epoch + batch_step_size*batch_num

        elif epoch<int(args.epochs*0.47+0.5)+args.warmup: lr = args.lr/1
        elif epoch<int(args.epochs*0.78+0.5)+args.warmup: lr = args.lr/10
        elif epoch<int(args.epochs*0.95+0.5)+args.warmup: lr = args.lr/100
        else         : lr = args.lr/1000
        return lr

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot)
        if (self.current_lr != lr) and ((batch_num == 0) or (batch_num+1 == batch_tot)): 
            print(f'Changing LR from {self.current_lr} to {lr}')

        self.current_lr = lr
        self.current_epoch = epoch
        self.current_batch = batch_num

        for param_group in self.optimizer.param_groups: param_group['lr'] = lr

        if not args.distributed: return
        for param_group in self.optimizer.param_groups:
            lr_old = param_group['lr']
            param_group['lr'] = lr
            # Trick 4: apply momentum correction when lr is updated
            if lr > lr_old:
                param_group['momentum'] = lr / lr_old * args.momentum
            else:
                param_group['momentum'] = args.momentum

def init_dist_weights(model):
    # Distributed training uses 4 tricks to maintain the accuracy
    # with much larger batchsize, see
    # https://arxiv.org/pdf/1706.02677.pdf
    # for more details

    if args.arch.startswith('resnet'):
        for m in model.modules():
            # Trick 1: the last BatchNorm layer in each block need to
            # be initialized as zero gamma
            if isinstance(m, resnet.BasicBlock):
                m.bn2.weight = Parameter(torch.zeros_like(m.bn2.weight))
            if isinstance(m, resnet.Bottleneck):
                m.bn3.weight = Parameter(torch.zeros_like(m.bn3.weight))
            # Trick 2: linear layers are initialized by
            # drawing weights from a zero-mean Gaussian with
            # standard deviation of 0.01. In the paper it was only
            # fc layer, but in practice we found this better for
            # accuracy.
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

def main():
    print("~~epoch\thours\ttop1Accuracy\n")

    # need to index validation directory before we start counting the time
    if args.val_ar: sort_ar(args.data+'/validation')

    start_time = datetime.now()

    if args.distributed:
        print('Distributed: initializing process group')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)

    if args.fp16: assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    # create model
    # if args.pretrained: model = models.__dict__[args.arch](pretrained=True)
    # else: model = models.__dict__[args.arch]()
    # AS: force use resnet50 for now, until we figure out whether to upload model directory
    
    model = resnet.resnet50()
    print("Loaded model")

    model = model.cuda()
    n_dev = torch.cuda.device_count()
    if args.fp16: model = network_to_half(model)
    if args.distributed:
        # init_dist_weights(model) # (AS) Performs pretty poorly for first 10 epochs when enabled
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    global model_params, master_params
    if args.fp16:  model_params, master_params = prep_param_lists(model)
    else: master_params = list(model.parameters())

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(master_params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = Scheduler(optimizer)

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

    dm = DataManager()
    print("Created data loaders")

    if args.evaluate: return validate(dm.get_val_iter(), len(dm.val_dl), model, criterion, 0, start_time)

    print("Begin training")
    estart = time.time()
    for epoch in range(args.start_epoch, args.epochs+args.warmup):
        estart = time.time()
        dm.set_epoch(epoch)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            train(dm.get_trn_iter(), len(dm.trn_dl), model, criterion, optimizer, scheduler, epoch, base_model_pointer)

        if args.prof: break
        prec5 = validate(dm.get_val_iter(), len(dm.val_dl), model, criterion, epoch, start_time)

        is_best = prec5 > best_prec5
        if args.local_rank == 0 and is_best:
            best_prec5 = max(prec5, best_prec5)
            save_checkpoint({
                'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                'best_prec5': best_prec5, 'optimizer' : optimizer.state_dict(),
            }, is_best)


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
    return [correct_k.mul_(100.0 / batch_size) for correct_k in corrrect_ks]

def correct(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

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

def get_world_size():
    if args.distributed: return dist.get_world_size()
    return 1

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    size = get_world_size()
    # rt /= args.world_size
    rt /= size
    return rt

if __name__ == '__main__': main()

