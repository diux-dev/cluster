#!/usr/bin/env python
#
# Example of two process Ray program, worker sends values to parameter
# server on a different machine
#
# Run locally:
# ./ray_localsgd.py
#
# Run on AWS:
# ./ray_localsgd.py --aws

import argparse
import os
import threading
import time
import numpy as np
import ray

parser = argparse.ArgumentParser()
parser.add_argument("--role", default='launcher', type=str,
                    help="launcher/driver")
parser.add_argument('--image',
                    default='Deep Learning AMI (Ubuntu) Version 14.0')
parser.add_argument("--size-mb", default=10, type=int,
                    help='how much data per worker')
parser.add_argument("--num-workers", default=2, type=int,
                    help='how many workers to run in parallel')

parser.add_argument("--iters", default=1, type=int)
parser.add_argument("--aws", action="store_true", help="enable to run on AWS")
parser.add_argument("--xray", default=1, type=int,
                    help="whether to use XRay backend")
parser.add_argument('--nightly', default=1, type=int,
                    help='whether to use nightly version')
parser.add_argument('--macos', default=0, type=int,
                    help='whether we are on Mac')
parser.add_argument('--name', default='ray', type=str,
                    help='name of the run')
parser.add_argument('--instance', default='c5.large', type=str,
                    help='instance type to use')

parser.add_argument("--ip", default='', type=str,
                    help="internal flag, used to point worker to head node")
args = parser.parse_args()

#  dim = args.size_mb * 250 * 1000
dim = 1


@ray.remote(resources={"worker": 1})
class Worker(object):
  def __init__(self, index, total_workers):
    self.params = np.zeros(dim * total_workers, dtype=np.float32)
    self.gradients = np.ones(dim, dtype=np.float32)
    self.index = index

    # each worker only updates a part of parameters between low and high
    self.low, self.high = index * dim, (index + 1) * dim

  def train(self, iters):
    for i in range(iters):
      self.params[self.low:self.high] += self.gradients
      self.log(f"worker {self.index}, params is {self.params}")
      time.sleep(0.25)

  def get_params(self):
    return self.params

  def set_params(self, params):
    self.params = params

  def log(self, msg):
    print(msg)
    with open(f'/tmp/w{self.index}', 'a') as f:
      f.write(msg + '\n')


def run_launcher():
  import ncluster

  if args.aws:
    ncluster.set_backend('aws')

  script = os.path.basename(__file__)
  if args.nightly:
    if args.macos:
      install_script = 'pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.5.2-cp36-cp36m-macosx_10_6_intel.whl'
    else:
      install_script = 'pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.5.2-cp36-cp36m-manylinux1_x86_64.whl'
  else:
    install_script = 'pip install ray'

  job = ncluster.make_job(name=args.name,
                          install_script=install_script,
                          image_name=args.image,
                          instance_type=args.instance,
                          num_tasks=args.num_workers + 1)
  job.upload(script)
  if args.xray:
    job.run('export RAY_USE_XRAY=1')
  job.run('ray stop')

  # https://ray.readthedocs.io/en/latest/resources.html?highlight=resources
  driver = job.tasks[0]
  driver.run(f"ray start --head --redis-port=6379")
  for worker_task in job.tasks[1:]:
    worker_resource = """--resources='{"worker": 1}'"""
    worker_task.run(f"ray start --redis-address={driver.ip}:6379 "
                    f"{worker_resource}")
  driver.run(f'./{script} --role=driver --ip={driver.ip}:6379')


def run_driver():
  ray.init(redis_address=args.ip)

  # start workers training asynchronously
  workers = [Worker.remote(i, args.num_workers) for i in
             range(args.num_workers)]

  workers[0].train.remote(100)
  print(ray.get(workers[0].get_params.remote()))

  def start_worker(w):
    w.train.remote(args.iters)
  print("First part done")

  threads = []
  for worker in workers:
    threads.append(threading.Thread(target=start_worker, args=(worker,)))
    threads[-1].start()

  while True:
    params0 = workers[0].get_params.remote()
    print(ray.get(params0))
    time.sleep(0.25)


def main():
  if args.role == 'launcher':
    run_launcher()
  elif args.role == 'driver':
    run_driver()
  else:
    assert False, f"Unknown role {args.role}, must be laucher/driver"


if __name__ == '__main__':
  main()
