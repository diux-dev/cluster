#!/usr/bin/env python
# numpy01 image, see environment-numpy.org for construction
# (DL AMI v 3.0 based)
#
# us-east-1 AMIs
# numpy00: ami-f9d6dc83
# numpy01: ami-5b524f21

from collections import defaultdict
import argparse
import boto3
import os
import sys
import time

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util
import aws_backend
import backend

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami-name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 12.0',
                    help="name of AMI to use")
parser.add_argument('--placement-group', type=str, default='pytorch_cluster',
                     help="name of the current run")
parser.add_argument('--name', type=str, default='pytorch',
                     help="name of the current run")
parser.add_argument('--instance-type', type=str, default='p3.2xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-west-2a',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
parser.add_argument('--num-tasks', type=int, default=1,
                    help='number of instances to create')
parser.add_argument('--skip-efs-mount', action='store_true',
                    help='skip mounting EFS for speed')
parser.add_argument('--logdir-prefix', default='/efs/runs.cifar',
                    help='where to put logs')
args = parser.parse_args()


gpu_count = defaultdict(lambda:0, { 'p3.2xlarge': 1, 'p3.8xlarge': 4, 'p3.16xlarge': 8, 'p2.xlarge': 1, 'p2.8xlarge': 4, 'p2.16xlarge': 8 })
def create_job(run, job_name, num_tasks):
  job = run.make_job(job_name, num_tasks=num_tasks, instance_type=args.instance_type, placement_group=args.placement_group)
  job.wait_until_ready()

  print(job.connect_instructions)

  backend.set_global_logdir_prefix(args.logdir_prefix)
  run.setup_logdir()
#   run pytorch
  job.run('killall python || echo failed')  # kill previous run

  # upload files
  job.upload('resnet.py')
  job.upload('train_cifar10.py')
  job.upload('fp16util.py')

  # setup env
  job.run('source activate pytorch_p36')


  # single machine
  num_gpus = gpu_count[args.instance_type]
  if (num_tasks == 1) and (num_gpus == 1):
    job.run_async('python train_cifar10.py') # single instance
    return

  # multi job
  world_0_ip = job.tasks[0].instance.private_ip_address
  port = '6006' # 6006, 6007, 6008, 8890, 6379
  job.run('ulimit -n 9000') # to prevent tcp too many files open error
  world_size = num_gpus * num_tasks

  for i,t in enumerate(job.tasks):
    # Pytorch distributed
    training_args = f'--dist-url env:// --dist-backend gloo --distributed --world-size {world_size} --scale-lr 2' # must tweak --scale-lr
    dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
    t.run_async(f'python -m torch.distributed.launch {dist_args} train_cifar10.py {training_args}')



def main():
  run = aws_backend.make_run(args.name, ami_name=args.ami_name,
                             availability_zone=args.zone,
                             linux_type=args.linux_type,
                             skip_efs_mount=args.skip_efs_mount)
  create_job(run, 'worker', args.num_tasks)

if __name__=='__main__':
  main()
