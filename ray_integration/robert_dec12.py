# copied on Dec 12 from
# https://gist.githubusercontent.com/robertnishihara/24979fb01b4b70b89e5cf9fbbf9d7d65/raw/b2d3bb66e881034039fbd244d7f72c5f6b425235/async_sgd_benchmark_multinode.py

# Start head node with
#
#     ray start --head --redis-port=6379 --num-gpus=0 --num-cpus=<1000 * num_workers> --num-workers=<num_workers>
#
# Start the other node with
#
#     ray start --redis-address=... --num-gpus=<num_parameter_servers> --num-cpus=<num_parameter_servers> --num-workers=0
#
# To benchmark READ throughput, run the following.
#
#     python async_sgd_benchmark_multinode.py --redis-address=... --num-workers=10 --num-parameter-servers=10 --data-size=100000000 --read
#
# 10 workers, 10 parameter servers, 100MB objects, 64 cores:
#     read throughput: reaches 0.7MB/s
#
# 1 worker, 1 parameter server, 100MB objects, 64 cores:
#     read throughput: 0.5GB/s
# ------------------------------------------------------------------------------
#
# To benchmark WRITE throughput, run the following.
#
#     python async_sgd_benchmark_multinode.py --redis-address=... --num-workers=10 --num-parameter-servers=10 --data-size=100000000
#
# 10 workers, 10 parameter servers, 100MB objects, 64 cores:
#     write throughput: 0.5GB/s
#
# 1 worker, 1 parameter server, 100MB objects, 64 cores:
#     write throughput: 0.4GB/s

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time

import ray

parser = argparse.ArgumentParser(description='Benchmark asynchronous '
                                             'training.')
parser.add_argument('--redis-address', default=None, type=str,
                    help='The Redis address of the cluster.')
parser.add_argument('--num-workers', default=4, type=int,
                    help='The number of workers to use.')
parser.add_argument('--num-parameter-servers', default=4, type=int,
                    help='The number of parameter servers to use.')
parser.add_argument('--data-size', default=1000000, type=int,
                    help='The size of the data to use.')
parser.add_argument('--read', action='store_true',
                    help='measure read throughput, the default is to measure '
                         'write throughput')


args = parser.parse_args()

@ray.remote(num_gpus=1)
class ParameterServer(object):
    def __init__(self, data_size, read):
        self.data_size = data_size
        self.read = read
        self.value = np.zeros(data_size, dtype=np.uint8)
        self.times = []

    def push(self, value):
        if not self.read:
            self.update_times()
        self.value += value

    def pull(self):
        if self.read:
            self.update_times()
        return self.value

    def update_times(self):
        self.times.append(time.time())
        if len(self.times) > 100:
            self.times = self.times[-100:]

    def get_throughput(self):
        return (self.data_size * (len(self.times) - 1) /
                (self.times[-1] - self.times[0]))


@ray.remote(num_cpus=1000)
def worker_task(data_size, read, *parameter_servers):
    while True:
        if read:
            # Get the current value from the parameter server.
            values = ray.get([ps.pull.remote() for ps in parameter_servers])
        else:
            # Push an update to the parameter server.
            ray.get([ps.push.remote(np.zeros(data_size, dtype=np.uint8))
                     for ps in parameter_servers])


ray.init(redis_address=args.redis_address)

parameter_servers = [ParameterServer.remote(args.data_size, args.read)
                     for _ in range(args.num_parameter_servers)]

# Let the parameter servers start up.
time.sleep(3)

# Start some training tasks.
worker_tasks = [worker_task.remote(args.data_size, args.read,
                                   *parameter_servers)
                for _ in range(args.num_workers)]

while True:
    time.sleep(2)
    throughput = (ray.get(parameter_servers[0].get_throughput.remote()) *
                  args.num_parameter_servers)
    if args.read:
        print('Read throughput is {}MB/s.'.format(throughput / 1e6))
    else:
        print('Write throughput is {}MB/s.'.format(throughput / 1e6))

