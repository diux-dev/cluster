#!/usr/bin/env python
# launch TensorBoard/monitoring server for runs

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='monitoring')
# t3.micros get OOM killed, used r5
parser.add_argument('--instance-type', type=str, default='r5.large')
parser.add_argument('--image', type=str,
                    default="Deep Learning AMI (Ubuntu) Version 13.0")
args = parser.parse_args()
import ncluster


def main():
  task = ncluster.make_task(args.name, instance_type=args.instance_type,
                            image_name=args.image)
  task.run('source activate tensorflow_p36')
  zone = ncluster.get_zone()
  window_title = 'Tensorboard'
  if zone.startswith('us-west-2'):
    window_title = 'Oregon'
  elif zone.startswith('us-east-1'):
    window_title = 'Virginia'
  elif zone.startswith('us-east-2'):
    window_title = 'Ohio'

  task.run(f'cd {ncluster.get_logdir_root()}')
  task.run(f'tensorboard --logdir=. --port=6006 --window_title={window_title}',
           async=True)
  print(f'Tensorboard will be at http://{task.public_ip}:6006')

if __name__=='__main__':
  main()
