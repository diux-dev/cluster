# To run the example, use a command like the following.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time

import ray

parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", default=1, type=int,
                    help="The number of workers to use.")
parser.add_argument("--num-parameter-servers", default=1, type=int,
                    help="The number of parameter servers to use.")
parser.add_argument("--dim", default=25*1000*1000, type=int,
                    help="The number of parameters.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")

args = parser.parse_args()


# TODO(rkn): This is a placeholder.
class CNN(object):
    def __init__(self, dim):
        self.dim = dim

    def get_gradients(self):
        return np.ones(self.dim, dtype=np.float32)

    def set_weights(self, weights):
        pass


@ray.remote
class ParameterServer(object):
    def __init__(self, dim):
        self.params = np.zeros(dim)

    def update_and_get_new_weights(self, gradients):
        self.params += gradients

    def ip(self):
        return ray.services.get_node_ip_address()

@ray.remote
class Worker(object):
    def __init__(self, num_ps, dim):
        self.net = CNN(dim)
        self.num_ps = num_ps

    @ray.method(num_return_vals=args.num_parameter_servers)
    def compute_gradient(self, weights):
        all_weights = weights
        self.net.set_weights(all_weights)
        gradient = self.net.get_gradients()
        return gradient

    def ip(self):
        return ray.services.get_node_ip_address()


if __name__ == "__main__":
  if args.redis_address is None:
    ray.init(num_gpus=args.num_parameter_servers + args.num_workers)
  else:
    ray.init(redis_address=args.redis_address)


  # initial parameter value
  split_weights = np.zeros(args.dim, dtype=np.float32)

  # worker and parameter server
  ps = ParameterServer.remote(split_weights.size)
  worker = Worker.remote(args.num_parameter_servers, args.dim)

  for i in range(100):
    t1 = time.time()

    gradients = worker.compute_gradient.remote(split_weights)
    ray.wait([gradients])

    t2 = time.time()

    split_weights = ps.update_and_get_new_weights.remote(gradients)
    ray.wait([split_weights])

    t3 = time.time()
    t1ms = 1000*t1
    t2ms = 1000*t2
    t3ms = 1000*t3
    print("elapsed times: total %4.2f worker update %4.2f ps update %4.2f" %( t3ms - t1ms, t2ms - t1ms, t3ms - t2ms))
        
