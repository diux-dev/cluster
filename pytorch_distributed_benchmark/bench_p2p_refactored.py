#!/usr/bin/env python
#
# Launching simple MPI job on AWS cluster or locally


import os
import sys
import time

import argparse

PYTORCH_DIST_BACKEND = 'tcp'
parser = argparse.ArgumentParser(description='launch')

# launcher flags
parser.add_argument('--name', type=str, default='p2p',
                     help="name of the current run")
parser.add_argument('--instance-type', type=str, default='t2.large',
                     help="type of instance to use")
parser.add_argument('--data-size-mb', type=int, default=100,
                    help='size of data to send')
parser.add_argument('--ami-name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 12.0',
                    help="name of AMI to use ")
parser.add_argument('--run-locally', action='store_true')
parser.add_argument('--placement', default=0)
parser.add_argument('--internal-role', type=str, default='launcher',
                    help='internal flag, launcher or worker')


# mpi flags
parser.add_argument('--rank', type=int, default=0,
                    help='mpi rank')
parser.add_argument('--size', type=int, default=0,
                    help='size of mpi world')
parser.add_argument('--master-addr', type=str, default='127.0.0.1',
                    help='address of master node')
parser.add_argument('--master-port', type=int, default=6006,
                    help='port of master node')

import ncluster

args = parser.parse_args()

def worker():
  """ Initialize the distributed environment. """

  import torch
  import torch.distributed as dist
  from torch.multiprocessing import Process

  print("Initializing distributed pytorch")
  os.environ['MASTER_ADDR'] = str(args.master_addr)
  os.environ['MASTER_PORT'] = str(args.master_port)
  dist.init_process_group(PYTORCH_DIST_BACKEND, rank=args.rank,
                          world_size=args.size)

  for i in range(100):
    tensor = torch.ones(args.data_size_mb*250*1000)*(args.rank+1)
    # print('before: rank ', args.rank, ' has data ', tensor[0])

    start_time = time.perf_counter()
    if args.rank == 0:
      dist.send(tensor=tensor, dst=1)
    else:
      dist.recv(tensor=tensor, src=0)
      
    elapsed_time = time.perf_counter() - start_time
    # print('after: rank ', args.rank, ' has data ', tensor[0])
    rate = args.data_size_mb/elapsed_time

    print("Process %d transferred %d MB in %.1f ms (%.1f MB/sec)" % (args.rank, args.data_size_mb, elapsed_time*1000, rate))


def launcher():
    
  job = ncluster.make_job('worker',
                          instance_type=args.instance_type, num_tasks=2)
  job.join()

  print("Job ready for connection, to connect to most recent task:")
  print("../connect "+args.name)
  print("Alternatively run")
  print(job.connect_instructions)
  print()

  job.upload(__file__)
  
  job.run('source activate pytorch_p36')
  script_name = os.path.basename(__file__)
  job.tasks[0].run_async(f'python {script_name} --internal-role=worker --rank=0 --size=2 --master-addr={job.tasks[0].ip}')
  job.tasks[1].run_async(f'python {script_name} --internal-role=worker --rank=1 --size=2 --master-addr={job.tasks[0].ip}')

def main():
  if args.internal_role == "launcher":
    launcher()
  elif args.internal_role == "worker":
    worker()
  else:
    assert False, "Unknown role "+FLAGS.role

  
if __name__ == "__main__":
  main()
