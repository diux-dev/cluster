import argparse, os, shutil, time, warnings
from pathlib import Path
import numpy as np
import sys
import math

import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data.sampler import Sampler
import torchvision
import pickle
from tqdm import tqdm

# from autoaugment import ImageNetPolicy

def get_loaders(datadir, sz, bs, val_bs=None, workers=8, use_ar=False, min_scale=0.08, distributed=False):
    traindir = datadir+'/train'
    valdir = datadir+'/validation'
    val_bs = val_bs or bs
    train_dataset = datasets.ImageFolder(
        traindir, transforms.Compose([
            transforms.RandomResizedCrop(sz, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(), 
            # ImageNetPolicy()
        ]))
    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, collate_fn=fast_collate, 
        sampler=train_sampler)

    val_dataset, val_sampler = create_validation_set(valdir, val_bs, sz, use_ar=use_ar, distributed=distributed)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=workers, pin_memory=True, collate_fn=fast_collate, 
        batch_sampler=val_sampler)
    return train_loader,val_loader,train_sampler,val_sampler


def create_validation_set(valdir, batch_size, target_size, use_ar, distributed):
    if use_ar:
        idx_ar_sorted = sort_ar(valdir)
        idx_sorted, _ = zip(*idx_ar_sorted)
        idx2ar = map_idx2ar(idx_ar_sorted, batch_size)

        ar_tfms = [transforms.Resize(int(target_size*1.14)), CropArTfm(idx2ar, target_size)]
        val_dataset = ValDataset(valdir, transform=ar_tfms)
        val_sampler = DistValSampler(idx_sorted, batch_size=batch_size, distributed=distributed)
        return val_dataset, val_sampler
    
    val_tfms = [transforms.Resize(int(target_size*1.14)), transforms.CenterCrop(target_size)]
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_tfms))
    val_sampler = DistValSampler(list(range(len(val_dataset))), batch_size=batch_size, distributed=distributed)
    return val_dataset, val_sampler

# Seems to speed up training by ~2%
class DataPrefetcher():
    def __init__(self, loader, prefetch=True, fp16=False):
        self.loader = loader
        self.prefetch = prefetch
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.fp16 = fp16
        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if self.prefetch:
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
        if self.fp16: input = input.half()
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
    idx2ar_file = valdir+'/../sorted_idxar.p'
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
    ar_chunks = list(chunks(idx_ar_sorted, batch_size))
    idx2ar = {}
    for chunk in ar_chunks:
        idxs, ars = list(zip(*chunk))
        mean = round(np.mean(ars), 5)
        for idx in idxs:
            idx2ar[idx] = mean
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

class DistValSampler(Sampler):
    # DistValSampler distrbutes batches equally (based on batch size) to every gpu (even if there aren't enough images). 
    # Some baches will contain an empty array to signify there aren't enough images
    # Distributed=False - same validation happens on every single gpu
    def __init__(self, indices, batch_size, distributed=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed:
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else: 
            self.global_rank = 0
            self.world_size = 1
            
        # expected number of batches per sample. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_batches = math.ceil(len(self.indices) / self.world_size / self.batch_size)
        
        # num_samples = total images / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_batches * self.batch_size
        
    def __iter__(self):
        offset = self.num_samples * self.global_rank
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

