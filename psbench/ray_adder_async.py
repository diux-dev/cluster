# copied from
# https://github.com/robertnishihara/ray/blob/e9ef96e3b6346580412b4f47133448724ca27f6a/examples/parameter_server/sharded_sync_parameter_server_benchmark.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
import threading

from collections import OrderedDict
from collections import defaultdict


import ray

parser = argparse.ArgumentParser(description="Run a synchronous parameter "
                                             "server performance benchmark.")
parser.add_argument("--workers", default=0, type=int,
                    help="The number of workers to use.")
parser.add_argument("--ps", default=0, type=int,
                    help="number of parameter servers to use.")
parser.add_argument("--size-mb", default=100, type=int,
                    help="size of data in MBs")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")
parser.add_argument("--enforce-different-ips", default=0, type=int,
                    help="Check that all workers are on different ips,"
                    "crash otherwise")
parser.add_argument("--iters", default=100, type=int,
                    help="how many iterations to go for")
parser.add_argument("--memcpy-threads", default=0, type=int,
                    help="how many threads to use for memcpy (0 for unchanged)")

args = parser.parse_args()
args_dim = args.size_mb * 250*1000
local_redis = False if args.redis_address else True
num_gpus_per_worker = 0 if local_redis else 1

import torch

class FileLogger:
  """Helper class to log to file (possibly mirroring to stderr)
     logger = FileLogger('somefile.txt')
     logger = FileLogger('somefile.txt', mirror=True)
     logger('somemessage')
     logger('somemessage: %s %.2f', 'value', 2.5)
  """
  
  def __init__(self, fn, mirror=False):
    self.fn = fn
    self.f = open(fn, 'w')
    self.mirror = mirror
    
  def __call__(self, s='', *args):
    """Either ('asdf %f', 5) or (val1, val2, val3, ...)"""
    if (isinstance(s, str) or isinstance(s, bytes)) and '%' in s :
      formatted_s = s % args
    else:
      toks = [s]+list(args)
      formatted_s = ', '.join(str(s) for s in toks)
      
    self.f.write(formatted_s+'\n')
    self.f.flush()
    if self.mirror:
      print(formatted_s)

  def __del__(self):
    self.f.close()

global_timeit_dict = OrderedDict()
class timeit:
  """Decorator to measure length of time spent in the block in millis and log
  it to TensorBoard."""
  
  def __init__(self, tag=""):
    self.tag = tag
    
  def __enter__(self):
    self.start = time.perf_counter()
    return self
  
  def __exit__(self, *args):
    self.end = time.perf_counter()
    interval_ms = 1000*(self.end - self.start)
    with open('/tmp/log.txt', 'a') as f:
      f.write("%s %.2f ms\n"%(self.tag, interval_ms))
    global_timeit_dict.setdefault(self.tag, []).append(interval_ms)


@ray.remote
class ParameterServer(object):
  def __init__(self, num_params):
    params0 = np.zeros(num_params, dtype=np.float32)
    self.params = torch.from_numpy(params0).clone()
    if args.memcpy_threads:
      ray.worker.global_worker.memcopy_threads = args.memcpy_threads

  def push(self, grad):
    """Adds all gradients to current value of parameters, returns result."""
    # todo: use .sum instead of loop
    grad.flags.writeable = True
    torch_params = torch.from_numpy(grad)
    self.params += torch_params

  def pull(self):
    return self.params.numpy()
  
  def get_weights(self):
    return self.params

  def ip(self):
    return ray.services.get_node_ip_address()


@ray.remote
class Worker(object):
  def __init__(self, dim, *pss):
    self.gradient = np.ones(dim, dtype=np.float32)
    if args.memcpy_threads:
      ray.worker.global_worker.memcopy_threads = args.memcpy_threads

    self.pss = pss
    self.grads = np.split(self.gradient, args.ps)
    self.val = 0


  @ray.method(num_return_vals=args.ps)
  def compute_gradients(self, *weights):
    
    # TODO(rkn): Potentially use array_split to avoid requiring an
    # exact multiple.
    if args.ps == 1:
      return self.gradients
    return np.split(self.gradients, args.ps)

  def train(self):
    for iteration in range(args.iters):
      with timeit('iteration'):
        push_ids = []
        for ps_shard, grad_shard in zip(self.pss, self.grads):
          push_ids.append(ps_shard.push.remote(grad_shard))
        result_shards = [ps_shard.pull.remote() for ps_shard in self.pss]
        ray.wait(result_shards+push_ids, num_returns=2*len(result_shards))
        self.val = ray.get(result_shards[0])[0]

  def iteration_time(self):
    return global_timeit_dict.get('iteration', [0])[-1]

  def value(self):
    return self.val
  
  def ip(self):
    return ray.services.get_node_ip_address()


def main():
  global logger
  
  if args_dim % args.ps != 0:
    raise Exception("The dimension argument must be divisible by the "
                    "number of parameter servers.")

  if args.redis_address is None:
    try:
      ray.init(object_store_memory=(5 * 10 ** 9))
    except:
      ray.init()
  else:
    ray.init(redis_address=args.redis_address)

  if args.memcpy_threads:
    ray.worker.global_worker.memcopy_threads = args.memcpy_threads

  logger = FileLogger('log.txt', mirror=True)

  # Create the parameter servers.
  pss = [ParameterServer.remote(args_dim // args.ps)
                       for _ in range(args.ps)]

  # Create workers.
  workers = [Worker.remote(args_dim, *pss) for worker_index in range(args.workers)]

  for worker in workers:
    worker.train.remote()
  print("Done")

  for i in range(100):
    print(ray.get(workers[0].iteration_time.remote()))  # this blocks
    print("query1")
    print(ray.get(workers[0].value.remote()))
    print("query2")
    time.sleep(1)

if __name__ == "__main__":
  main()
