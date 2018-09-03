#!/usr/bin/env python
#
# TODO: don't use env-var, too easy to forget to unset it
# export NCLUSTER_AWS_FAST_ROOTDISK=1
# python train_imagenet.py --machines=1  # 1 machine
# python train_imagenet.py --machines=4  # 4 machines
# python train_imagenet.py --machines=8  # 8 machines
# python train_imagenet.py --machines=16 # 16 machines


from collections import OrderedDict
import argparse
import os
import sys
import time
import collections
import boto3
import datetime
import threading
import ncluster
import launch_utils as launch_utils_lib

# TODO: bake source activate into install script
INSTALL_SCRIPT_FN = 'setup_env_nv.sh'
IMAGE_NAME = 'pytorch.imagenet.source.v7'
ENV_NAME = 'pytorch_source'
INSTANCE_TYPE = 'p3.16xlarge'
NUM_GPUS = 8

parser = argparse.ArgumentParser()
parser.add_argument('--preemptible', action='store_true', help='launch using preemptible/spot instances')
parser.add_argument('--name', type=str, default='imagenet', help="name of the current run")
parser.add_argument('--machines', type=int, default=1, help="how many machines to use")
args = parser.parse_args()

lr = 0.47
one_machine = [
  {'ep': 0, 'sz': 128, 'bs': 512, 'trndir': '-sz/160'},
  {'ep': (0, 5), 'lr': (lr, lr * 2)},  # lr warmup is better with --init-bn0
  {'ep': 5, 'lr': lr},
  {'ep': 14, 'sz': 224, 'bs': 192},
  {'ep': 16, 'lr': lr / 10},
  {'ep': 27, 'lr': lr / 100},
  {'ep': 32, 'sz': 288, 'bs': 128, 'min_scale': 0.5, 'rect_val': True},
  {'ep': (33, 35), 'lr': lr / 1000}
]

four_machines = []

lr = 0.235 * 8 # 8 = num tasks
eight_machines = [
  {'ep':0,  'sz':128, 'bs':128, 'trndir':'-sz/160'},
  {'ep':(0,6),  'lr':(lr,lr*2)},
  {'ep':6,            'bs':256, 'keep_dl':True},
  {'ep':6,      'lr':lr*2},
  {'ep':16, 'sz':224,'bs':128},
  {'ep':16,      'lr':lr},
  {'ep':19,          'bs':192, 'keep_dl':True},
  {'ep':19,     'lr':lr/(10/1.5)},
  {'ep':31,     'lr':lr/(100/1.5)},
  {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True},
  {'ep':37,     'lr':lr/100},
  {'ep':(38,40),'lr':lr/1000}
]

sixteen_machines = []

schedules = {1: one_machine,
             4: four_machines,
             8: eight_machines,
             16: sixteen_machines}


def main():

  # since we are using configurable name of conda env, modify install script
  # to run in that conda env
  install_script = open(INSTALL_SCRIPT_FN).read()
  install_script = f'source activate {ENV_NAME}\n' + install_script

  os.environ['NCLUSTER_AWS_FAST_ROOTDISK'] = '1'
  job = ncluster.make_job(name=args.name,
                          run_name=args.name,
                          num_tasks=args.machines,
                          image_name=IMAGE_NAME,
                          instance_type=INSTANCE_TYPE,
                          install_script=install_script,
                          preemptible=args.preemptible)
  job.upload('training')
  job.run(f'source activate {ENV_NAME}')

  world_size = NUM_GPUS * args.machines
  nccl_args = launch_utils_lib.get_nccl_args(args.machines, NUM_GPUS)

  # Training script args
  default_params = [
    '~/data/imagenet',
    '--fp16',
    '--logdir', job.logdir,
    '--distributed',
    '--skip-auto-shutdown',
    '--init-bn0',
    '--no-bn-wd',
  ]

  params = ['--phases', schedules[args.machines]]
  
  training_args = default_params + params
  training_args = ' '.join(map(launch_utils_lib.format_args, training_args))

  # TODO: simplify args processing
  # Run tasks
  task_cmds = []
  for i, task in enumerate(job.tasks):
    dist_args = f'--nproc_per_node=8 --nnodes={args.machines} --node_rank={i} --master_addr={job.tasks[0].ip} --master_port={6006}'
    cmd = f'{nccl_args} python -m torch.distributed.launch {dist_args} training/train_imagenet_nv.py {training_args}'
    task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
    task.run(cmd, async=True)

  print(f"Logging to {job.logdir}")


if __name__ == '__main__':
  main()
  
