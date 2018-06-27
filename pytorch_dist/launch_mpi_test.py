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

# mpi flags
parser.add_argument('--rank', type=int, default=0,
                    help='mpi rank')
parser.add_argument('--size', type=int, default=0,
                    help='size of mpi world')
parser.add_argument('--master-addr', type=str, default='127.0.0.1',
                    help='address of master node')
parser.add_argument('--master-port', type=int, default=6006,
                    help='port of master node')

# launcher flags

parser.add_argument('--zone', type=str, default='us-west-2b')
parser.add_argument('--placement', type=int, default=0,
                    help=("whether to launch instances inside a placement "
                          "group"))
parser.add_argument('--backend', type=str, default='tcp',
                    help='which PyTorch communication backend to use')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
parser.add_argument('--name', type=str, default='mpi_test',
                     help="name of the current run")
parser.add_argument('--instance', type=str, default='t2.large',
                     help="type of instance to use")
parser.add_argument('--run-local', action='store_true',
                    help='run the launcher locally')
parser.add_argument('--data-mb', type=int, default=100,
                    help='size of data to send')
args = parser.parse_args()

def worker():
  """ Initialize the distributed environment. """
  print("Initializing distributed pytorch")
  os.environ['MASTER_ADDR'] = str(args.master_addr)
  os.environ['MASTER_PORT'] = str(args.master_port)
  dist.init_process_group(args.backend, rank=args.rank, world_size=args.size)

  tensor = torch.zeros(args.data_mb*250*1000)
  for i in range(100):
    start_time = time.perf_counter()
    if args.rank == 0:
      dist.send(tensor=tensor, dst=1)
    else:
      dist.recv(tensor=tensor, src=0)
      
    elapsed_time = time.perf_counter() - start_time
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
    placement_group = args.run
  else:
    placement_group = ''

  if args.run_local:
    backend = tmux_backend
    run = backend.make_run(args.name)
  else:
    create_resources_lib.create_resources()
    region = u.get_region()
    ami = 'ami-e580c79d'
    backend = aws_backend  
    run = backend.make_run(args.name, ami=ami, availability_zone=args.zone)
  job = run.make_job('mpi', instance_type=args.instance, num_tasks=2,
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
  if not args.run_local:
    job.run('killall python || echo failed')  # kill previous run
    job.run('source activate pytorch_p36')

  job.tasks[0].run('python launch_mpi_test.py --role=worker --rank=0 --size=2 --master-addr='+job.tasks[0].ip, sync=False)
  job.tasks[1].run('python launch_mpi_test.py --role=worker --rank=1 --size=2 --master-addr='+job.tasks[0].ip, sync=False)


def main():
  if args.role == "launcher":
    launcher()
  elif args.role == "worker":
    worker()
  else:
    assert False, "Unknown role "+FLAGS.role

  
if __name__ == "__main__":
  main()
