# copied from
# https://github.com/robertnishihara/ray/blob/e9ef96e3b6346580412b4f47133448724ca27f6a/examples/parameter_server/sharded_sync_parameter_server_benchmark.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time

import ray

parser = argparse.ArgumentParser(description="Run a synchronous parameter "
                                             "server performance benchmark.")
parser.add_argument("--workers", default=1, type=int,
                    help="The number of workers to use.")
parser.add_argument("--ps", default=1, type=int,
                    help="number of parameter servers to use.")
parser.add_argument("--dim", default=(25 * 10 ** 6), type=int,
                    help="The dimension of the parameter vector (the vector "
                         "consists of np.float32, so the default is 100MB).")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")
args = parser.parse_args()
import torch

@ray.remote
class ParameterServer(object):
  def __init__(self, num_params):
    self.params = np.zeros(num_params, dtype=np.float32)
    self.torch_params = torch.from_numpy(self.params).clone()

  def assign_add(self, *gradients):
    """Adds all gradients to current value of parameters, returns result."""
    # todo: use .sum instead of loop
    for grad in gradients:
      grad.flags.writeable = True
      self.params += torch.from_numpy(grad)
    self.params = self.torch_params.numpy()
    print('value: ', self.torch_params[0])
    return self.params

  def get_weights(self):
    return self.params

@ray.remote
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


def main():
  if args.dim % args.ps != 0:
    raise Exception("The dimension argument must be divisible by the "
                    "number of parameter servers.")

  if args.redis_address is None:
    try:
      ray.init(object_store_memory=(5 * 10 ** 9))
    except:
      ray.init()
  else:
    ray.init(redis_address=args.redis_address)

  # Create the parameter servers.
  parameter_servers = [ParameterServer.remote(args.dim // args.ps)
                       for _ in range(args.ps)]

  # Create workers.
  workers = [Worker.remote(args.dim) for worker_index in range(args.workers)]
  current_weights = [ps.get_weights.remote() for ps in parameter_servers]
  iteration = 0
  reporting_interval = 10

  start_time = time.time()
  previous_time = time.time()

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
          parameter_servers[j].assign_add.remote(*gradient_ids[:, j]))

    iteration += 1
    if iteration % reporting_interval == 0:
      # Wait for this iteration to finish.
      ray.wait(current_weights, num_returns=len(current_weights))
      current_time = time.time()
      print("Iteration %d: average time per iteration %.1f" % (iteration,
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

  print("Average time per iteration (total): %.2f"%(
    1000*(end_time - start_time) / num_iterations))
  print()
  print("Average time computing gradients:%.2f"%(
        np.mean(compute_gradient_times)*1e3))
  print("Average time putting gradients:  %.2f"%( 1e3*np.mean(put_gradient_times)))
  print("Average time updating params:    %.2f"%( 1e3*np.mean(update_params_times)))
  print("Average time putting params:     %.2f"%( 1e3*np.mean(put_params_times)))
  print()

  total_accounted_time = (np.mean(compute_gradient_times) +
                          np.mean(put_gradient_times) +
                          np.mean(update_params_times) +
                          np.mean(put_params_times))

  print("Total accounted for:            %.2f" %(total_accounted_time*1e3,))
  print("Total unaccounted for:          %.2f"%(
        ((end_time - start_time) / num_iterations - total_accounted_time)*1e3))

if __name__ == "__main__":
  main()
