#!/usr/bin/env python
# numpy00 image, see environmnent-numpy.org for construction (DL AMI v 3.0 based)
# ami-f9d6dc83

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
parser.add_argument('--ami', type=str, default='ami-f9d6dc83',
                     help="name of AMI to use ")
parser.add_argument('--name', type=str, default='resnet50',
                     help="name of the current run")
parser.add_argument('--instance-type', type=str, default='p3.16xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-east-1c',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
args = parser.parse_args()

def main():
  import aws_backend

  # TODO: add API to create jobs with default run
  run = aws_backend.make_run(args.name, ami=args.ami,
                             availability_zone=args.zone,
                             linux_type=args.linux_type)
  job = run.make_job('main', instance_type=args.instance_type)
  job.wait_until_ready()
  print(job.connect_instructions)
  
  # if tensorboard is running, kill it, it will prevent efs logdir from being
  # deleted
  job.run("tmux kill-session -t tb || echo ok")
  job.run('rm -Rf /efs/runs/yuxin_numpy/mnist-convnet || echo failed') # delete prev logs
  
  # Launch tensorboard visualizer in separate tmux session
  job.run("tmux new-session -s tb -n 0 -d")
  job.run("tmux send-keys -t tb:0 'source activate mxnet_p36' Enter")
  job.run("tmux send-keys -t tb:0 'tensorboard --logdir /efs/runs/yuxin_numpy' Enter")

  job.run('source activate mxnet_p36')
  job.upload(module_path+'/mnist-convnet.py')
  job.run('killall python || echo failed')  # kill previous run
  job.run_async('python mnist-convnet.py')


if __name__=='__main__':
  main()
