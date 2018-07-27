#!/usr/bin/env python

# Prepare env with PyTorch
# conda create --name pytorch_p36 --clone <myenv>
#
# Prepare env with TensorBoard
# conda create --name tensorflow_p36 --clone <myenv>
#
# Run locally:
# ./launch.py
#
# Run on AWS:
# ./launch.py --backend=aws

import argparse
import os
import sys

# import cluster tools, one level up
module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import backend as backend_lib
import aws_backend
import tmux_backend
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
parser.add_argument('--spot', action='store_true', 
                    help='launch using spot requests')
parser.add_argument('--backend', type=str, default='tmux',
                    help='cluster backend, tmux (local) or aws')
args = parser.parse_args()

def main():

  if args.backend == 'tmux':
    backend = tmux_backend
  elif args.backend == 'aws':
    backend = aws_backend
  else:
    assert False, "unknown backend"
    
  run = backend.make_run(args.name,
                         ami_name=args.ami_name,
                         availability_zone=args.zone)
  job1 = run.make_job('worker', instance_type=args.instance_type)
  job2 = run.make_job('tb', instance_type=args.tb_instance_type)

  job1.wait_until_ready()
  run.setup_logdir()
  
  job1.upload('pytorch-mnist-example.py')
  job1.run('source activate pytorch_p36')
  job1.run('pip install tensorboardX')
  job1.run_async(f'python pytorch-mnist-example.py --logdir={run.logdir}')

  job2.wait_until_ready()
  job2.run('source activate tensorflow_p36')
  job2.run_async(f'tensorboard --logdir={run.logdir} --port=6006')
  print(f'Tensorboard will be at http://{job2.tasks[0].public_ip}:6006')
  

if __name__=='__main__':
  main()
