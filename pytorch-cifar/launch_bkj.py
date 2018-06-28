#!/usr/bin/env python
# numpy01 image, see environment-numpy.org for construction
# (DL AMI v 3.0 based)
#
# us-east-1 AMIs
# numpy00: ami-f9d6dc83
# numpy01: ami-5b524f21

import collections
import argparse
import os
import sys
import time

import boto3

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util
util.install_pdb_handler()

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='ami-e580c79d',
                     help="name of AMI to use ")
parser.add_argument('--placement-group', type=str, default='pytorch_cluster',
                     help="name of the current run")
parser.add_argument('--name', type=str, default='pytorch',
                     help="name of the current run")
parser.add_argument('--job-name', type=str, default='distributed',
                     help="name of the jobs to run")
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
parser.add_argument('--install-script', type=str, default='setup_env.sh',
                    help='location of script to install')
args = parser.parse_args()


gpu_count = collections.defaultdict(lambda:0, { 'p3.2xlarge': 1, 'p3.8xlarge': 4, 'p3.16xlarge': 8, 'p2.xlarge': 1, 'p2.8xlarge': 4, 'p2.16xlarge': 8 })
def create_job(run, job_name, num_tasks):
  install_script = ''
  with open(args.install_script, 'r') as script:
    install_script = script.read()
  job = run.make_job(job_name, num_tasks=num_tasks, instance_type=args.instance_type, install_script=install_script, placement_group=args.placement_group)
  job.wait_until_ready()
  print(job.connect_instructions)

#   run pytorch
  job.run('killall python || echo failed')  # kill previous run

  # upload files
  job.upload('resnet.py')
  job.upload('train_cifar10_bkj.py')

  # setup env
  job.run('source activate fastai')


  # single machine
  if num_tasks == 1:
    # job.run_async('python train_cifar10.py ~/data --loss-scale 512 --fp16 --lr 1.3') # multi-gpu
    job.run_async('python train_cifar10_bkj.py') # single instance
    return

  # multi job
  world_0_ip = job.tasks[0].instance.private_ip_address
  port = '6006' # 6006, 6007, 6008, 8890, 6379
  job.run('ulimit -n 9000') # to prevent tcp too many files open error

  for i,t in enumerate(job.tasks):
    # tcp only supports CPU - https://pytorch.org/docs/master/distributed.html
    # dist_params = f'--world-size {num_tasks} --rank {i} --dist-url tcp://{world_0_ip}:{port} --dist-backend tcp' # tcp
    
    # Pytorch distributed
    num_gpus = gpu_count[args.instance_type]
    training_args = '~/data --loss-scale 512 --fp16 -b 128 --lr 1.3 -j 7 --dist-url env:// --dist-backend gloo --distributed'
    dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
    t.run_async(f'python -m torch.distributed.launch {dist_args} train_cifar10.py {training_args}')



def main():
  import aws_backend

  run = aws_backend.make_run(args.name, ami=args.ami,
                             availability_zone=args.zone,
                             linux_type=args.linux_type)
  create_job(run, args.job_name, args.num_tasks)

if __name__=='__main__':
  main()
