#!/usr/bin/env python
# script to launch tensorboard job for a single run
import argparse
import json
import os
import sys
import time

import aws_backend
import util as u

parser = argparse.ArgumentParser(description='Launch CIFAR training')
# TODO: rename to gradient instance type
parser.add_argument('--instance-type', type=str, default='t2.micro',
                    help='instance type to use')
parser.add_argument('--zone', type=str, default='us-east-1c',
                    help='which availability zone to use')
parser.add_argument('--name', type=str, default='cifar00',
                     help="name of the current run")

args = parser.parse_args()

# Amazon Ubuntu Deep Learning AMI
generic_ami_dict = {
    "us-west-2": "ami-3b6bce43",
    "us-east-1": "ami-9ba7c4e1",
}


def main():
  backend = aws_backend
  region = os.environ.get("AWS_DEFAULT_REGION")
  ami = generic_ami_dict[region]

  run = backend.make_run(args.name, ami=ami, availability_zone=args.zone)
  job = run.make_job('tb', 1, instance_type=args.instance_type)

  job.wait_until_ready()
  job.run("source activate tensorflow_p36  # env with cuda 8")

  # Launch tensorboard visualizer.
  tb_cmd = "tensorboard --logdir={logdir} --port=6006".format(logdir=run.logdir)
  job.run(tb_cmd, sync=False)
  print("See tensorboard at http://%s:%s"%(job.public_ip, 6006))


if __name__=='__main__':
  main()
  
