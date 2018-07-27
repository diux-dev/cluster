#!/usr/bin/env python

import argparse
import os
import sys

# import cluster tools, one level up
module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import aws_backend
import util as u

parser = argparse.ArgumentParser()
parser.add_argument('--ami-name', type=str,
                    default="Deep Learning AMI (Ubuntu) Version 12.0",
                    help="name of AMI to use")
parser.add_argument('--name', type=str, default='tensorboard-example',
                    help='name of run, logs go to /efs/runs/{args.name}')
parser.add_argument('--instance-type', type=str, default='c5.large',
                     help='instance type to use for training')
parser.add_argument('--tb-instance-type', type=str, default='r5.large',
                     help='instance type to use for tensorboard')
parser.add_argument('--zone', type=str, default='us-west-2a',
                    help='which availability zone to use')
args = parser.parse_args()

def main():
  run = aws_backend.make_run(args.name,
                             ami_name=args.ami_name,
                             availability_zone=args.zone)

  logdir = u.LOGDIR_ROOT+'/'+args.name
  job1 = run.make_job('worker', instance_type=args.instance_type)
  job2 = run.make_job('tb', instance_type=args.tb_instance_type)

  job1.wait_until_ready()
  job1.upload('pytorch-mnist-example.py')
  job1.run('source activate pytorch_p36')
  job1.run('pip install tensorboardX')
  job1.run_async(f'python pytorch-mnist-example.py --logdir={logdir}')

  job2.wait_until_ready()
  job2.run('source activate tensorflow_p36')
  job2.run_async(f'tensorboard --logdir={u.LOGDIR_ROOT}')
  print(f'Tensorboard will be at http://{job2.tasks[0].public_ip}:6006')
  

if __name__=='__main__':
  main()
