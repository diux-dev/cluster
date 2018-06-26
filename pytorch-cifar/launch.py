#!/usr/bin/env python
# numpy01 image, see environment-numpy.org for construction
# (DL AMI v 3.0 based)
#
# us-east-1 AMIs
# numpy00: ami-f9d6dc83
# numpy01: ami-5b524f21

from collections import OrderedDict
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
parser.add_argument('--group', type=str, default='dawn_runs',
                     help="name of the current run")
parser.add_argument('--name', type=str, default='as_cifar10',
                     help="name of the current run")
# parser.add_argument('--instance-type', type=str, default='p3.2xlarge',
parser.add_argument('--instance-type', type=str, default='t2.large',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-west-2a',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
args = parser.parse_args()

def create_job(run, job_name, num_tasks=2):
  install_script = ''
  with open('setup_env.sh', 'r') as script:
    install_script = script.read()
  job = run.make_job(job_name, num_tasks=num_tasks, instance_type=args.instance_type, install_script=install_script)
  job.wait_until_ready()
  print(job.connect_instructions)

#   run pytorch
  job.run('killall python || echo failed')  # kill previous run

  # upload files
  job.upload('resnet.py')
  job.upload('distributed.py')
  job.upload('multiproc.py')
  job.upload('train_cifar10.py')
  job.upload('train_cifar10_cpu.py')
  job.upload('setup_env.sh')

  # setup env
  job.run('chmod +x setup_env.sh')
  job.run('./setup_env.sh')
  job.run('source activate fastai')


  # single machine
  if num_tasks == 1:
    # job.run_async('python train_cifar10.py ~/data --loss-scale 512 --fp16 --lr 1.3') # multi-gpu
    job.run_async('python train_cifar10.py ~/data --loss-scale 512 --fp16') # single instance
    return

  # multi job
  world_0_ip = job.tasks[0].instance.private_ip_address
  port = '6006' # 6006, 6007, 6008, 8890, 6379
  use_tcp = False
        
  for i,t in enumerate(job.tasks):
    # tcp only supports CPU - https://pytorch.org/docs/master/distributed.html
    if use_tcp: dist_params = f'--world-size {num_tasks} --rank {i} --dist-url tcp://{world_0_ip}:{port} --dist-backend tcp'
    else: dist_params = f'--world-size {num_tasks} --rank {i} --dist-addr {world_0_ip} --dist-port {port} --dist-url env:// --dist-backend gloo'
    
    # t.run_async(f'python train_cifar10.py ~/data --loss-scale 512 --fp16 --lr 1.4 -b 128 --lr 1.3 {dist_params}') # multi-gpu
    t.run_async(f'python train_cifar10_cpu.py ~/data --lr 1.4 -b 128 --lr 1.3 {dist_params} --cpu') # multi-cpu



def main():
  import aws_backend

  run = aws_backend.make_run(args.name, ami=args.ami,
                             availability_zone=args.zone,
                             linux_type=args.linux_type)
  create_job(run, 'cifar10_multi')

if __name__=='__main__':
  main()
