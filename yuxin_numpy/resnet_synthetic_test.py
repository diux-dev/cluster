#!/usr/bin/env python
# File: resnet.b512.baseline.py
# Run for a few iterations and make sure accuracies are correct


import sys
import argparse
import numpy as np
import os
from itertools import count
import time

import tensorflow as tf

from tensorpack import *
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import TrainConfig, SyncMultiGPUTrainerParameterServer
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import (
  ImageNetModel,
  get_imagenet_dataflow,
  fbresnet_augmentor_fast,
)

from resnet_model import (
  resnet_group, resnet_basicblock, resnet_bottleneck)


DATASET_SIZE=1281  # 0.1% of original dataset size
PER_GPU_BATCH_SIZE = 64
BASE_LR = 0.1 * (512 // 256)

class Model(ImageNetModel):
  def get_logits(self, image):
    group_func = resnet_group
    block_func = resnet_bottleneck
    num_blocks = [3, 4, 6, 3]
    with argscope(
        [Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm],
        data_format='NCHW'), \
        argscope(Conv2D, nl=tf.identity, use_bias=False,
                 W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
      logits = (LinearWrap(image)
                .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2)
                .GlobalAvgPooling('gap')
                .FullyConnected('linear', 1000, nl=tf.identity)())
      return logits

def get_data(name, batch):
  isTrain = name == 'train'
  global args
  augmentors = fbresnet_augmentor_fast(isTrain)

  if isTrain:
    print("Training batch:", batch)
    return get_imagenet_dataflow(args.data, name, batch, augmentors)
  else:
    imagenet1k = get_imagenet_dataflow(args.data, name, batch, augmentors)
    return imagenet1k

class StepTimeCallback(Callback):
  def _before_run(self, _):
    self._start = time.time()

  def _after_run(self, _, __):
    self.trainer.monitors.put_scalar('step-time', time.time() - self._start)

class EpochTimeCallback(Callback):
  def _before_epoch(self):
    self._start = time.time()

  def _after_epoch(self):
    self.trainer.monitors.put_scalar('epoch-time', time.time() - self._start)

def get_config(model):
  nr_tower = max(get_nr_gpu(), 1)
  batch = PER_GPU_BATCH_SIZE

  logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
  dataset_train = get_data('train', batch)
  dataset_val = get_data('val', batch)

  infs = [ClassificationError('wrong-top1', 'val-error-top1'),
          ClassificationError('wrong-top5', 'val-error-top5')]
  callbacks = [
    StepTimeCallback(),
    EpochTimeCallback(),
    GPUUtilizationTracker(),
  ]


  if args.fake:
    dataset_train = FakeData(
      [[batch, 224, 224, 3], [batch]], 1000,
      random=False, dtype=['uint8', 'int32'])


  input = QueueInput(dataset_train)
  input = StagingInput(input, nr_stage=1)

  num_gpus = get_nr_gpu()
  
  return TrainConfig(
    model=model,
    data=input,
    callbacks=callbacks,
    extra_callbacks=train.DEFAULT_CALLBACKS()+[
      MergeAllSummaries(period=1),
    ],
    steps_per_epoch=DATASET_SIZE // (PER_GPU_BATCH_SIZE*get_nr_gpu()),
    max_epoch=2,
  )


if __name__ == '__main__':
  from os.path import expanduser
  home = expanduser("~")

  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', default='0',
                      help='comma separated list of GPU(s) to use.')
  parser.add_argument('--data', default='/tmpfs/data/imagenet',
                      help='ILSVRC dataset dir')
  parser.add_argument('--load', help='load model')
  parser.add_argument('--fake', default=1, help='use fake data')
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--group', default='resnet_synthetic')
  parser.add_argument('--name', default='fresh00')
  args = parser.parse_args()

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

  logdir = home+'/logs/resnet_synthetic_test'
  print("Writing to logging directory ", logdir)
  model = Model()
  logger.set_logger_dir(logdir, 'd')

  config = get_config(model)
  logger.set_logger_dir(logdir, 'd')

  nr_tower = max(get_nr_gpu(), 1)

  tf.set_random_seed(1)
  np.random.seed(1)
  launch_train_with_config(config, SyncMultiGPUTrainerReplicated(nr_tower))

  # get events from the events file
  step_times = []
  epoch_times = []
  losses = []
  from tensorflow.python.summary import summary_iterator
  import glob

  for fname in glob.glob(logdir+'/events*'):
    print('opening ', fname)
    events = summary_iterator.summary_iterator(fname)

    events = [e for e in events if e.step]

    for event in events:
      step = event.step
      wall_time = event.wall_time
      vals = {val.tag: val.simple_value for val in event.summary.value}
      for tag in vals:
        if 'xentropy-loss' in tag:
          losses.append(vals[tag])
          #          print(step, tag, vals[tag])
        if 'step-time' in tag:
          step_times.append(vals[tag])
        if 'epoch-time' in tag:
          epoch_times.append(vals[tag])
          
  losses = np.array(losses)
  step_times = np.array(step_times)
  epoch_times = np.array(epoch_times)
  print('Final loss: %10.5f' %(losses[-1]))
  print('Median step time: %10.1f ms'%( 1000*np.median(step_times)))
  print('Final epoch time: %10.3f sec' %(epoch_times[-1]))
  im_per_sec = DATASET_SIZE/epoch_times[-1]
  print('Images/second: %10.2f sec' %(im_per_sec))

  # Example two runs:
  #
  # Final loss:  0.046783510595560074
  # Median step time:  0.20586061477661133
  # Final epoch time:  4.064533233642578
  #
  # Final loss:  0.046783510595560074
  # Median step time:  0.20688438415527344
  # Final epoch time:  4.07200813293457

  assert(abs(losses[-1]-0.046783510595560074)<0.001)
  assert(np.median(step_times)<0.21)
  assert(epoch_times[-1]<4.1)
  assert(epoch_times[-1]>4)
  print("Test passed")
