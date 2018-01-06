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
import numpy as np
import os
import sys
import time

from collections import OrderedDict
from collections import defaultdict

import cifar10
import cifar10_model
import cifar10_utils

# move some methods to util later, for now "u" points to this file
util = sys.modules[__name__]   
u = util


import ray

parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", default=2, type=int,
                    help="The number of workers to use.")
parser.add_argument("--num-parameter-servers", default=2, type=int,
                    help="The number of parameter servers to use.")
parser.add_argument("--dim", default=75360, type=int,
                    help="The number of parameters, defaults to size of "
                    "TF default CIFAR10 model")
parser.add_argument("--time-to-wait-for-ps-ms", default=100, type=int,
                    help="The number of parameters.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")
parser.add_argument("--add-pause", default=0, type=int,
                    help="Add pause to avoid melting my laptop.")
parser.add_argument('--logdir', type=str, default='asdfasdfasdf',
                     help="location of logs")
parser.add_argument('--real-model', action='store_true',
                    default=False,
                    help="use real CIFAR model for gradients?")
parser.add_argument('--insert-pauses', action='store_true',
                    default=False,
                    help="Make ps0 freeze periodically")

args = parser.parse_args()

########################################
# Tensorboard logging, move to util.py
########################################
def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]

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
    global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
    logger = u.get_last_logger(skip_existence_check=True)
    if logger:
      newtag = 'time/'+self.tag
      logger(newtag, interval_ms)

# TODO: have global experiment_base that I can use to move logging to
# non-current directory
GLOBAL_RUNS_DIRECTORY='runs'
global_last_logger = None

def get_last_logger(skip_existence_check=False):
  """Returns last logger, if skip_existence_check is set, doesn't
  throw error if logger doesn't exist."""
  global global_last_logger
  if not skip_existence_check:
    assert global_last_logger
  return global_last_logger

class TensorboardLogger:
  """Helper class to log to single tensorboard writer from multiple places.
   logger = u.TensorboardLogger("mnist7")
   logger = u.get_last_logger()  # gets last logger created
   logger('svd_time', 5)  # records "svd_time" stat at 5
   logger.next_step()     # advances step counter
   logger.set_step(5)     # sets step counter to 5
  """
  
  def __init__(self, logdir, step=0):
    # TODO: do nothing for default run
    
    global global_last_logger
    assert global_last_logger is None
    self.logdir = logdir,
    self.summary_writer = tf.summary.FileWriter(logdir,
                                                flush_secs=5,
                                                graph=tf.get_default_graph())
    self.step = step
    self.summary = tf.Summary()
    global_last_logger = self
    self.last_timestamp = time.perf_counter()

  def __call__(self, *args):
    assert len(args)%2 == 0
    for (tag, value) in chunks(args, 2):
      self.summary.value.add(tag=tag, simple_value=float(value))

  def next_step(self):
    new_timestamp = time.perf_counter()
    interval_ms = 1000*(new_timestamp - self.last_timestamp)
    self.summary.value.add(tag='time/step',
                           simple_value=interval_ms)
    self.last_timestamp = new_timestamp
    self.summary_writer.add_summary(self.summary, self.step)
    self.step+=1
    self.summary = tf.Summary()

################################################################################
## Main stuff
################################################################################

# TODO(rkn): This is a placeholder.
class CNN(object):
    def __init__(self, dim):
        self.dim = dim
        # param values from cifar10_main.py
        if not tf.test.is_gpu_available():
            data_format = 'channels_last'
        else:
            data_format = 'channels_first'
        

        is_training = True
        weight_decay = 2e-4,
        num_layers = 8
        batch_size = 32
        batch_norm_decay=0.997
        batch_norm_epsilon=1e-5
        image_batch = tf.random_uniform((batch_size, 32, 32, 3))
        label_batch = tf.ones((batch_size,), dtype=tf.int32)

        self.model = cifar10_model.ResNetCifar10(
            num_layers,
            batch_norm_decay=batch_norm_decay,
            batch_norm_epsilon=batch_norm_epsilon,
            is_training=is_training,
            data_format=data_format)
        self.logits = self.model.forward_pass(image_batch,
                                              input_data_format='channels_last')

        # make size of parameters multiple of 8 (75360)
        dummy_var = tf.Variable(tf.ones((5,)))
        self.pred = {
            'classes': tf.argmax(input=self.logits, axis=1),
            'probabilities': tf.nn.softmax(self.logits)
        }

        self.loss = tf.losses.sparse_softmax_cross_entropy(logits=self.logits,
                                                      labels=label_batch)
        self.model_params = tf.trainable_variables()
        self.loss += weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in self.model_params])

        grads = tf.gradients(self.loss, self.model_params)
        self.grad = tf.concat([tf.reshape(g,[-1]) for g in grads], axis=0)
        self.weights = np.zeros(self.grad.shape, dtype=np.float32)

        # TODO: make this into an op that accepts actual values
        self.set_weights_op = tf.global_variables_initializer()
        
        # todo(y): pad things so that it's divisible by num_ps?

        self.sess = tf.Session()

    def get_gradients(self):
        if args.real_model:
            return self.sess.run(self.grad)
        else:
            return np.ones(self.dim, dtype=np.float32)

    def set_weights(self, weights):
        self.weights = weights
        # TODO, pass weights into set_weights_op
        if args.real_model:
            self.sess.run(self.set_weights_op)


# TODO(rkn): Once we have better custom resource support for actors, we should
# not use GPUs here.
@ray.remote(num_gpus=1)
class ParameterServer(object):
    def __init__(self, dim, idx):
        self.idx = idx
        self.params = np.zeros(dim, dtype=np.float32)

    def update_and_get_new_weights(self, *gradients):
        if args.insert_pauses:
            if self.idx == 0:
                print("0th ps shard")
        for grad in gradients:
            self.params += grad
        return self.params

    def ip(self):
        return ray.services.get_node_ip_address()


@ray.remote(num_gpus=1)
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

          #        logger('ps_drop', len(remaining_ids))
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
    import tensorflow as tf
    tf.constant(1)  # dummy default graph to appease tensorboard
    
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

    # create tensorboard logger
    logger = u.TensorboardLogger(args.logdir)

    # Create the workers.
    workers = [Worker.remote(args.num_parameter_servers, args.dim,
                             args.time_to_wait_for_ps_ms)
               for _ in range(args.num_workers)]

    # Create the parameter servers.
    pss = [ParameterServer.remote(sizes[i], i)
           for i in range(args.num_parameter_servers)]

    # As a sanity check, make sure all workers and parameter servers are on
    # different machines.
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
            print("Warning, some IPs are reused")

    LOG_FREQUENCY = 10
    step = 0
    last_step = 0
    last_time = time.time()

    while True:
        step+=1
        logger.next_step()
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
        with u.timeit('wait_compute_grads'):
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
