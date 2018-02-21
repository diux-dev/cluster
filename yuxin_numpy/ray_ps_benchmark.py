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
parser.add_argument("--skip-ray", default=0, type=int,
                    help="skip ray and add directly in numpy")

args = parser.parse_args()

class CNN(object):
    def __init__(self, dim):
        self.dim = dim
        self.grads = np.ones(self.dim, dtype=np.float32)
        
    def get_gradients(self, params):
        return self.grads


@ray.remote
class ParameterServer(object):
    def __init__(self, dim):
        self.params = np.zeros(dim)

    def update_and_get_new_params(self, gradients):
        self.params += gradients
        return self.params

@ray.remote
class Worker(object):
    def __init__(self, dim):
        self.net = CNN(dim)

    def compute_gradient(self, params):
        return self.net.get_gradients(params)


def main():
  # initial parameter value
  params = np.zeros(args.dim, dtype=np.float32)

  if not args.skip_ray:
    ray.init()

    ps = ParameterServer.remote(args.dim)
    worker = Worker.remote(args.dim)

  RUN_RAY = True
  net = CNN(args.dim)

  for i in range(100):
    t1 = time.time()

    if args.skip_ray:
      params += net.get_gradients(params)
      t2 = time.time()
    else:
      grads = worker.compute_gradient.remote(params)
      ray.wait([grads])

      t2 = time.time()

      params = ps.update_and_get_new_params.remote(grads)
      ray.wait([params])

    t3 = time.time()
    t1ms = 1000*t1
    t2ms = 1000*t2
    t3ms = 1000*t3
    print("elapsed times: total %4.2f worker update %4.2f ps update %4.2f" %( t3ms - t1ms, t2ms - t1ms, t3ms - t2ms))
        

if __name__ == "__main__":
  main()
