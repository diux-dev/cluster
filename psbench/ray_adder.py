# copied from
# https://github.com/robertnishihara/ray/blob/e9ef96e3b6346580412b4f47133448724ca27f6a/examples/parameter_server/sharded_sync_parameter_server_benchmark.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time

from collections import OrderedDict
from collections import defaultdict


import ray

parser = argparse.ArgumentParser(description="Run a synchronous parameter "
                                             "server performance benchmark.")
parser.add_argument("--workers", default=1, type=int,
                    help="The number of workers to use.")
parser.add_argument("--ps", default=1, type=int,
                    help="number of parameter servers to use.")
parser.add_argument("--size-mb", default=100, type=int,
                    help="size of data in MBs")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")
parser.add_argument("--enforce-different-ips", default=0, type=int,
                    help="Check that all workers are on different ips,"
                    "crash otherwise")
args = parser.parse_args()
args_dim = args.size_mb * 250*1000

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


@ray.remote(num_gpus=1)
class ParameterServer(object):
  def __init__(self, num_params):
    params0 = np.zeros(num_params, dtype=np.float32)
    self.params = torch.from_numpy(params0).clone()

  def assign_add(self, *gradients):
    """Adds all gradients to current value of parameters, returns result."""
    # todo: use .sum instead of loop
    for grad in gradients:
      grad.flags.writeable = True
      torch_params = torch.from_numpy(grad)
      self.params += torch_params
    return self.params.numpy()

  def get_weights(self):
    return self.params

  def ip(self):
    return ray.services.get_node_ip_address()


@ray.remote(num_gpus=1)
class Worker(object):
  def __init__(self, dim):
    self.gradients = np.ones(dim, dtype=np.float32)

  @ray.method(num_return_vals=args.ps)
  def compute_gradients(self, *weights):
    
    # TODO(rkn): Potentially use array_split to avoid requiring an
    # exact multiple.
    if args.ps == 1:
      return self.gradients
    return np.split(self.gradients, args.ps)

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

  logger = FileLogger('log.txt', mirror=True)

  # Create the parameter servers.
  pss = [ParameterServer.remote(args_dim // args.ps)
                       for _ in range(args.ps)]

  # Create workers.
  workers = [Worker.remote(args_dim) for worker_index in range(args.workers)]
  current_weights = [ps.get_weights.remote() for ps in pss]
  iteration = 0
  reporting_interval = 10

  start_time = time.time()
  previous_time = time.time()

  # As a sanity check, make sure all workers and parameter servers are on
  # different machines.
  if args.enforce_different_ips:
    if args.redis_address is not None:
      all_ips = ray.get([ps.ip.remote() for ps in pss] +
                        [w.ip.remote() for w in workers])
      #        assert len(all_ips) == len(set(all_ips))
      print("ps ips:")
      for (i, ps) in enumerate(pss):
        print(i, ps.ip.remote(), ray.get([ps.ip.remote()]))
      print("worker ips:")
      for (i, worker) in enumerate(workers):
        print(i, worker.ip.remote(), ray.get([worker.ip.remote()]))
      if len(all_ips) != len(set(all_ips)):
        assert False, "Some IPs are reused"

  num_iterations = 100
  for iteration in range(num_iterations):
    # Compute gradients.
    gradient_ids = np.empty((args.workers, args.ps), dtype=object)
    for i in range(args.workers):
      gradient_partition_ids = workers[i].compute_gradients.remote(
        *current_weights)
      if args.ps == 1:
        gradient_partition_ids = [gradient_partition_ids]
      assert len(gradient_partition_ids) == args.ps
      for j in range(args.ps):
        gradient_ids[i, j] = gradient_partition_ids[j]

    # Get the updated weights.
    current_weights = []
    for j in range(args.ps):
      current_weights.append(
          pss[j].assign_add.remote(*gradient_ids[:, j]))

    iteration += 1
    if iteration % reporting_interval == 0:
      # Wait for this iteration to finish.
      ray.wait(current_weights, num_returns=len(current_weights))
      current_time = time.time()
      logger("Iteration %d: average time per iteration %.1f" % (iteration,
                    1e3*(current_time - previous_time) / reporting_interval))
      previous_time = current_time

  end_time = time.time()

  compute_gradient_times = []
  put_gradient_times = []
  update_params_times = []
  put_params_times = []

  task_info = ray.global_state.task_profiles(num_tasks=1000)

  for _, info in task_info.items():
    if info["function_name"] == "compute_gradients":
      compute_gradient_times.append(
          info["execute_end"] - info["execute_start"])
      put_gradient_times.append(
          info["store_outputs_end"] - info["store_outputs_start"])
    elif info["function_name"] == "assign_add":
      update_params_times.append(
          info["execute_end"] - info["execute_start"])
      put_params_times.append(
          info["store_outputs_end"] - info["store_outputs_start"])

  logger("Average time per iteration (total): %.2f"%(
    1000*(end_time - start_time) / num_iterations))
  logger()
  logger("Average time computing gradients:%.2f"%(
        np.mean(compute_gradient_times)*1e3))
  logger("Average time putting gradients:  %.2f"%( 1e3*np.mean(put_gradient_times)))
  logger("Average time updating params:    %.2f"%( 1e3*np.mean(update_params_times)))
  logger("Average time putting params:     %.2f"%( 1e3*np.mean(put_params_times)))
  logger()

  total_accounted_time = (np.mean(compute_gradient_times) +
                          np.mean(put_gradient_times) +
                          np.mean(update_params_times) +
                          np.mean(put_params_times))

  logger("Total accounted for:            %.2f" %(total_accounted_time*1e3,))
  logger("Total unaccounted for:          %.2f"%(
        ((end_time - start_time) / num_iterations - total_accounted_time)*1e3))

if __name__ == "__main__":
  main()
