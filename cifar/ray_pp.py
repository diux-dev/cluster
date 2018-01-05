#
# Ray partial pull implementation from
# https://gist.github.com/robertnishihara/073f0b329f1494fe75555322ba3ef2cc

# To run the example, use a command like the following.
#
#     python partial_pull_ps.py \
#         --num-parameter-servers=2 \
#         --num-workers=4 \
#         --time-to-wait-for-ps-ms=100 \
#         --dim=25000000

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import time

import ray

parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", default=3, type=int,
                    help="The number of workers to use.")
parser.add_argument("--num-parameter-servers", default=5, type=int,
                    help="The number of parameter servers to use.")
parser.add_argument("--dim", default=1000, type=int,
                    help="The number of parameters.")
parser.add_argument("--time-to-wait-for-ps-ms", default=100, type=int,
                    help="The number of parameters.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")

args = parser.parse_args()


# TODO(rkn): This is a placeholder.
class CNN(object):
    def __init__(self, dim):
        self.dim = dim

    def get_gradients(self):
        time.sleep(0.16)
        return np.ones(self.dim, dtype=np.float32)

    def set_weights(self, weights):
        pass


# TODO(rkn): Once we have better custom resource support for actors, we should
# not use GPUs here.
@ray.remote(num_gpus=1)
class ParameterServer(object):
    def __init__(self, dim):
        self.params = np.zeros(dim, dtype=np.float32)

    def update_and_get_new_weights(self, *gradients):
        for grad in gradients:
            self.params += grad
        return self.params

    def ip(self):
        return ray.services.get_node_ip_address()


@ray.remote(num_gpus=2)
class Worker(object):
    def __init__(self, num_ps, dim, time_to_wait_for_ps_ms):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.net = CNN(dim)
        self.num_ps = num_ps
        self.time_to_wait_for_ps_ms = time_to_wait_for_ps_ms
        self.previous_params = None

    @ray.method(num_return_vals=args.num_parameter_servers)
    def compute_gradient(self, weights):
        if self.previous_params is None:
            # This should only happen on the first call to compute_gradient.
            self.previous_params = ray.get(weights)

        ready_ids, remaining_ids = ray.wait(
            weights, num_returns=len(weights),
            timeout=self.time_to_wait_for_ps_ms)

        print("Ignoring {} parameter servers.".format(len(remaining_ids)))
        ready_weights = ray.get(ready_ids)

        current_weights = []
        for i in range(len(weights)):
            object_id = weights[i]
            if object_id in ready_ids:
                current_weights.append(
                    ready_weights[ready_ids.index(object_id)])
            else:
                current_weights.append(self.previous_params[i])

        all_weights = np.concatenate(current_weights)
        self.net.set_weights(all_weights)
        gradient = self.net.get_gradients()

        self.previous_params = current_weights

        if self.num_ps == 1:
            return gradient
        else:
            return np.split(gradient, self.num_ps)

    def ip(self):
        return ray.services.get_node_ip_address()


if __name__ == "__main__":
    if args.redis_address is None:
        # Run everything locally.
        ray.init(num_gpus=args.num_parameter_servers + 2 * args.num_workers)
    else:
        # Connect to a cluster.
        ray.init(redis_address=args.redis_address)

    split_weights = np.split(np.zeros(args.dim, dtype=np.float32),
                             args.num_parameter_servers)
    sizes = [weights.size for weights in split_weights]
    split_weights = [ray.put(weights) for weights in split_weights]

    # Create the workers.
    workers = [Worker.remote(args.num_parameter_servers, args.dim,
                             args.time_to_wait_for_ps_ms)
               for _ in range(args.num_workers)]

    # Create the parameter servers.
    pss = [ParameterServer.remote(sizes[i])
           for i in range(args.num_parameter_servers)]

    # As a sanity check, make sure all workers and parameter servers are on
    # different machines.
    if args.redis_address is not None:
        all_ips = ray.get([ps.ip.remote() for ps in pss] +
                          [w.ip.remote() for w in workers])
        assert len(all_ips) == len(set(all_ips))

    while True:
        t1 = time.time()

        # Compute and apply gradients.
        assert len(split_weights) == args.num_parameter_servers
        grad_id_lists = [[] for _ in range(len(pss))]
        for worker in workers:
            gradients = worker.compute_gradient.remote(split_weights)
            if len(pss) == 1:
                gradients = [gradients]

            assert len(gradients) == len(pss)
            for i in range(len(gradients)):
                grad_id_lists[i].append(gradients[i])

        # TODO(rkn): This weight should not be removed. Does it affect
        # performance?
        all_grad_ids = [grad_id for grad_id_list in grad_id_lists
                        for grad_id in grad_id_list]
        ray.wait(all_grad_ids, num_returns=len(all_grad_ids))

        t2 = time.time()

        split_weights = []
        for i in range(len(pss)):
            assert len(grad_id_lists[i]) == args.num_workers
            new_weights_id = pss[i].update_and_get_new_weights.remote(
                *(grad_id_lists[i]))
            split_weights.append(new_weights_id)

        t3 = time.time()
        print("elapsed times: ", t3 - t1, t2 - t1, t3 - t2)
