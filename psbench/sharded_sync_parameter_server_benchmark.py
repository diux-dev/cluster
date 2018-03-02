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
parser.add_argument("--num-workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--num-parameter-servers", default=2, type=int,
                    help="The number of parameter servers to use.")
parser.add_argument("--include-ps-update", action="store_true",
                    help="Include the parameter server update in the timing.")
parser.add_argument("--include-gradient-computation", action="store_true",
                    help="Include the gradient computation in the timing.")
parser.add_argument("--dim", default=(25 * 10 ** 6), type=int,
                    help="The dimension of the parameter vector (the vector "
                         "consists of np.float32, so the default is 100MB).")
parser.add_argument("--summation-with-torch", action="store_true",
                    help="Do the gradient summation with pytorch instead of "
                         "numpy.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.summation_with_torch:
        import torch

    if args.dim % args.num_parameter_servers != 0:
        raise Exception("The dimension argument must be divisible by the "
                        "number of parameter servers.")

    @ray.remote
    class ParameterServer(object):
        def __init__(self, num_params):
            self.params = np.zeros(num_params, dtype=np.float32)
            if args.summation_with_torch:
                self.torch_params = torch.from_numpy(self.params)

        def apply_gradients_and_get_params(self, *gradients):
            if args.include_ps_update:
                if args.summation_with_torch:
                    # Torch seems to be faster at summing arrays than numpy.
                    for grad in gradients:
                        grad.flags.writeable = True
                        self.torch_params += (torch.from_numpy(grad) /
                                              len(gradients))
                    self.params = self.torch_params.numpy()
                else:
                    self.params += np.mean(gradients, axis=0)

            return self.params

        def get_weights(self):
            return self.params

    @ray.remote
    class Worker(object):
        def __init__(self, dim):
            self.gradients = np.zeros(dim, dtype=np.float32)

        @ray.method(num_return_vals=args.num_parameter_servers)
        def compute_gradients(self, *weights):
            if args.include_gradient_computation:
                # Simulate a 160ms gradient computation.
                time.sleep(0.16)

            # TODO(rkn): Potentially use array_split to avoid requiring an
            # exact multiple.
            if args.num_parameter_servers == 1:
                return self.gradients
            return np.split(self.gradients, args.num_parameter_servers)

    if args.redis_address is None:
        try:
            ray.init(object_store_memory=(5 * 10 ** 9))
        except:
            ray.init()
    else:
        ray.init(redis_address=args.redis_address)

    # Create the parameter servers.
    parameter_servers = [ParameterServer.remote(args.dim //
                                                args.num_parameter_servers)
                         for _ in range(args.num_parameter_servers)]

    # Create workers.
    workers = [Worker.remote(args.dim)
               for worker_index in range(args.num_workers)]

    current_weights = [ps.get_weights.remote() for ps in parameter_servers]
    iteration = 0
    reporting_interval = 10

    start_time = time.time()
    previous_time = time.time()

    num_iterations = 100
    for iteration in range(num_iterations):
        # Compute gradients.
        gradient_ids = np.empty((args.num_workers, args.num_parameter_servers),
                                dtype=object)
        for i in range(args.num_workers):
            gradient_partition_ids = workers[i].compute_gradients.remote(
                *current_weights)
            if args.num_parameter_servers == 1:
                gradient_partition_ids = [gradient_partition_ids]
            assert len(gradient_partition_ids) == args.num_parameter_servers
            for j in range(args.num_parameter_servers):
                gradient_ids[i, j] = gradient_partition_ids[j]

        # Get the updated weights.
        current_weights = []
        for j in range(args.num_parameter_servers):
            current_weights.append(
                parameter_servers[j].apply_gradients_and_get_params.remote(
                    *gradient_ids[:, j]))

        iteration += 1
        if iteration % reporting_interval == 0:
            # Wait for this iteration to finish.
            ray.wait(current_weights, num_returns=len(current_weights))
            current_time = time.time()
            print("Iteration {}: average time per iteration {}"
                  .format(iteration,
                          (current_time - previous_time) / reporting_interval))
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
        elif info["function_name"] == "apply_gradients_and_get_params":
            update_params_times.append(
                info["execute_end"] - info["execute_start"])
            put_params_times.append(
                info["store_outputs_end"] - info["store_outputs_start"])

    print("Average time per iteration (total): ",
          (end_time - start_time) / num_iterations)
    print("")
    print("Average time computing gradients:   ",
          np.mean(compute_gradient_times))
    print("Average time putting gradients:     ", np.mean(put_gradient_times))
    print("Average time updating params:       ", np.mean(update_params_times))
    print("Average time putting params:        ", np.mean(put_params_times))
    print("")

    total_accounted_time = (np.mean(compute_gradient_times) +
                            np.mean(put_gradient_times) +
                            np.mean(update_params_times) +
                            np.mean(put_params_times))

    print("Total accounted for:                ", total_accounted_time)
    print("Total unaccounted for:              ",
          (end_time - start_time) / num_iterations - total_accounted_time)
