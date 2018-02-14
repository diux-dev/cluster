#!/usr/bin/env python
# File: resnet.b512.baseline.py
# from yuxin

import sys
import argparse
import numpy as np
import os
from itertools import count

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
    fbresnet_augmentor)
from resnet_model import (
    resnet_group, resnet_basicblock, resnet_bottleneck)


TOTAL_BATCH_SIZE = 512
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
    augmentors = fbresnet_augmentor(isTrain)
    if isTrain:
        print("Training batch:", batch)
        return get_imagenet_dataflow(args.data, name, batch, augmentors)
    else:
        imagenet1k = get_imagenet_dataflow(args.data, name, batch, augmentors)
        return imagenet1k


def get_config(model):
    nr_tower = max(get_nr_gpu(), 1)
    batch = TOTAL_BATCH_SIZE // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch)
    dataset_val = get_data('val', batch)
    import pdb; pdb.set_trace()

    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    callbacks = [
        ModelSaver(),
        GPUUtilizationTracker(),
        ScheduledHyperParamSetter('learning_rate',
            [(0, 0.1), (3, BASE_LR)], interp='linear'),
        ScheduledHyperParamSetter('learning_rate',
            [(30, BASE_LR * 1e-1), (60, BASE_LR * 1e-2), (80, BASE_LR * 1e-3)]),
        PeriodicTrigger(
            DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))),
            every_k_epochs=1),
    ]

    input = QueueInput(dataset_train)
    input = StagingInput(input, nr_stage=1)
    return TrainConfig(
        model=model,
        data=input,
        callbacks=callbacks,
        steps_per_epoch=1281167 // TOTAL_BATCH_SIZE,
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
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--logdir', default='train_log/tmp')
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
        logger.set_logger_dir(args.logdir, 'd')

        config = get_config(model)
        if args.load:
            config.session_init = get_model_loader(args.load)
        nr_tower = max(get_nr_gpu(), 1)
        launch_train_with_config(config, QueueInputTrainer())
