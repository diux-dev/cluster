#!/usr/bin/env python
#

from collections import OrderedDict
import argparse
import os
import sys
import time

import boto3

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='ami-5b524f21', # numpy01 ami
                     help="name of AMI to use ")
parser.add_argument('--name', type=str, default='prewarming_test',
                     help="name of the current run")
parser.add_argument('--instance', type=str, default='c5.18xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-east-1f',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
args = parser.parse_args()

def main():
  if args.role == "launcher":
    launcher()
  elif args.role == "worker":
    worker()
  else:
    assert False, "Unknown role "+FLAGS.role


def launcher():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import tmux_backend
  import aws_backend
  import create_resources as create_resources_lib
  import util as u

  create_resources_lib.create_resources()
  region = u.get_region()
  assert args.zone.startswith(region), "Availability zone %s must be in default region %s. Default region is taken from environment variable AWS_DEFAULT_REGION" %(args.zone, region)

  install_script = ''

  ami = args.ami

  # TODO: add API to create jobs with default run
  run = aws_backend.make_run(args.name, install_script=install_script,
                             ami=ami, availability_zone=args.zone,
                             linux_type=args.linux_type)
  job = run.make_job('gpubox', instance_type=args.instance)
  
  job.wait_until_ready()

  job.run('source activate mxnet_p36')
  job.run('sudo apt install -y fio')
  job.run('volume=/dev/xvda1')
  job.run('time sudo fio --filename=$volume --rw=read --bs=128k --iodepth=32 --ioengine=libaio --direct=1 --name=volume-initialize')

if __name__=='__main__':
  main()
