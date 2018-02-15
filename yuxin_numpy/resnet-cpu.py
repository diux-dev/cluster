#!/usr/bin/env python
# File: resnet.b512.baseline.py
# from yuxin

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
    eval_on_ILSVRC12,
    fbresnet_augmentor,
  fbresnet_augmentor_fast,
)
from resnet_model import (
    resnet_group, resnet_basicblock, resnet_bottleneck)


#TOTAL_BATCH_SIZE = 512
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
        self.trainer.monitors.put_scalar('step_time', time.time() - self._start)

def get_config(model):
    nr_tower = max(get_nr_gpu(), 1)
    #    batch = TOTAL_BATCH_SIZE // nr_tower
    batch = PER_GPU_BATCH_SIZE

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch)
    dataset_val = get_data('val', batch)

    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    callbacks = [
        StepTimeCallback(),
        GPUUtilizationTracker(),
    ]




    if args.fake:
        dataset_train = FakeData(
            [[batch, 224, 224, 3], [batch]], 1000,
            random=False, dtype=['uint8', 'int32'])


    input = QueueInput(dataset_train)
    #    import pdb; pdb.set_trace()
    input = StagingInput(input, nr_stage=1)

    num_gpus = get_nr_gpu()
    return TrainConfig(
        model=model,
        data=input,
        callbacks=callbacks,
        steps_per_epoch=1281 // (PER_GPU_BATCH_SIZE*get_nr_gpu()),
        max_epoch=100,
    )


if __name__ == '__main__':
    from os.path import expanduser
    home = expanduser("~")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', default=home+'/data/imagenet',
                        help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', action='store_true', help='use fake data')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--logdir', default='/efs/runs/yuxin_numpy_imagenet/scratch10')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model()
    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(
            model,
            get_model_loader(args.load), ds,
            ['input', 'label0'],
            ['wrong-top1', 'wrong-top5'])
    else:
        os.system('mv %s /efs/trash/%d'%(args.logdir,time.time()))
        logger.set_logger_dir(args.logdir, 'd')

        config = get_config(model)
        if args.load:
            config.session_init = get_model_loader(args.load)
        nr_tower = max(get_nr_gpu(), 1)
        launch_train_with_config(config,
            SyncMultiGPUTrainerReplicated(nr_tower))
