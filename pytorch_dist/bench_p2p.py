#!/usr/bin/env python
#
# Launching simple MPI job on AWS cluster or locally


import os
import sys
import time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

import argparse

parser = argparse.ArgumentParser(description='launch')

# launcher flags
parser.add_argument('--zone', type=str, default='', help=("which AWS zone to use leave blank to run locally"))
parser.add_argument('--placement', action='store_true',
                    help=("whether to launch instances inside a placement "
                          "group"))
parser.add_argument('--backend', type=str, default='tcp',
                    help='which PyTorch communication backend to use')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
parser.add_argument('--name', type=str, default='p2p',
                     help="name of the current run")
parser.add_argument('--instance-type', type=str, default='t2.large',
                     help="type of instance to use")
parser.add_argument('--data-mb', type=int, default=100,
                    help='size of data to send')
parser.add_argument('--ami', type=str, default='',
                     help="name of AMI to use ")


# mpi flags
parser.add_argument('--rank', type=int, default=0,
                    help='mpi rank')
parser.add_argument('--size', type=int, default=0,
                    help='size of mpi world')
parser.add_argument('--master-addr', type=str, default='127.0.0.1',
                    help='address of master node')
parser.add_argument('--master-port', type=int, default=6006,
                    help='port of master node')


args = parser.parse_args()

ami_dict_ubuntu = {
  "us-east-1": "ami-6d720012",
  "us-east-2": "ami-23c4fb46",
  "us-west-2": "ami-e580c79d",
}
ami_dict = ami_dict_ubuntu

def worker():
  """ Initialize the distributed environment. """
  print("Initializing distributed pytorch")
  os.environ['MASTER_ADDR'] = str(args.master_addr)
  os.environ['MASTER_PORT'] = str(args.master_port)
  dist.init_process_group(args.backend, rank=args.rank, world_size=args.size)

  for i in range(100):
    tensor = torch.ones(args.data_mb*250*1000)*(args.rank+1)
    # print('before: rank ', args.rank, ' has data ', tensor[0])

    start_time = time.perf_counter()
    if args.rank == 0:
      dist.send(tensor=tensor, dst=1)
    else:
      dist.recv(tensor=tensor, src=0)
      
    elapsed_time = time.perf_counter() - start_time
    # print('after: rank ', args.rank, ' has data ', tensor[0])
    rate = args.data_mb/elapsed_time

    print("Process %d transferred %d MB in %.1f ms (%.1f MB/sec)" % (args.rank, args.data_mb, elapsed_time*1000, rate))


def launcher():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import tmux_backend
  import aws_backend
  import create_resources as create_resources_lib
  import util as u

  if args.placement:
    placement_group = args.name
  else:
    placement_group = ''


  if not args.zone:
    backend = tmux_backend
    run = backend.make_run(args.name)
  else:
    region = u.get_region()
    print("Using region", region)
    assert args.zone.startswith(region), "Availability zone %s must be in default region %s. Default region is taken from environment variable AWS_DEFAULT_REGION" %(args.zone, region)

    if args.ami:
      print("Warning, using provided AMI, make sure that --linux-type argument "
            "is set correctly")
      ami = args.ami
    else:
      assert region in ami_dict, "Define proper AMI mapping for this region."
      ami = ami_dict[region]

    create_resources_lib.create_resources()
    region = u.get_region()
    backend = aws_backend  
    run = backend.make_run(args.name, ami=ami, availability_zone=args.zone)
    


  job = run.make_job('worker', instance_type=args.instance_type, num_tasks=2,
                     placement_group=placement_group)
  
  job.wait_until_ready()

  print("Job ready for connection, to connect to most recent task, run the following:")
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
  if args.zone:
    job.run('killall python || echo failed')  # kill previous run
    job.run('source activate pytorch_p36')

  script_name = os.path.basename(__file__)
  job.tasks[0].run('python '+script_name+' --role=worker --rank=0 --size=2 --master-addr='+job.tasks[0].ip, sync=False)
  job.tasks[1].run('python '+script_name+' --role=worker --rank=1 --size=2 --master-addr='+job.tasks[0].ip, sync=False)

def main():
  if args.role == "launcher":
    launcher()
  elif args.role == "worker":
    worker()
  else:
    assert False, "Unknown role "+FLAGS.role

  
if __name__ == "__main__":
  main()
