#!/usr/bin/env python
# Simple MPI example with 2 machines

from collections import OrderedDict
import argparse
import os
import sys
import time

import boto3


# Deep learning AMI v10
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue
ami_dict = {
  "us-east-1": "ami-6d720012",
  "us-east-2": "ami-23c4fb46",
  "us-west-2": "ami-e580c79d",
}

install_script="pip install mpi4py"

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='',
                     help="name of AMI to use ")
parser.add_argument('--name', type=str, default='mpi',
                     help="name of the current run")
parser.add_argument('--instance', type=str, default='t2.large',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-west-2a',
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

  assert region in ami_dict, "Define proper AMI mapping for this region."
  ami = ami_dict[region]

  # TODO: add API to create jobs with default run
  run = aws_backend.make_run(args.name, install_script=install_script,
                             ami=ami, availability_zone=args.zone,
                             linux_type=args.linux_type)
  job = run.make_job('mpi', instance_type=args.instance, num_tasks=2)
  
  job.wait_until_ready()

  print("Job ready for connection, run the following:")
  print("../connect "+args.name)
  print("Alternatively run")
  print(job.connect_instructions)
  print()
  print()
  print()
  print()

  print("Task internal IPs")
  for task in job.tasks:
    print(task.ip)
  
  job.upload(__file__)
  job.run('killall python || echo failed')  # kill previous run
  job.run_async('python launch_mpi.py --role=worker')

def worker():
  """Worker script that runs on AWS machine. Sends numpy arrays forever,
  prints MB/s."""

  from mpi4py import MPI
  import numpy, time

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  n = 100000
  data_mb = n*4./1000000
  print("Sending %.1f MBs"%(data_mb,))
  def communicate():
    if rank == 0:
      data = numpy.arange(n, dtype=numpy.float32)
      comm.Send(data, dest=1, tag=13)
    elif rank == 1:
      data = numpy.empty(100, dtype=numpy.float32)
      comm.Recv(data, source=0, tag=13)

  while True:
    start_time = time.perf_counter()
    communicate()
    elapsed_time = time.perf_counter() - start_time
    rate = data_mb/elapsed_time
    print('%.2f MB/s'%(rate,))    

if __name__=='__main__':
  main()
