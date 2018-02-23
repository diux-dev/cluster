import tensorflow as tf


import argparse
import numpy as np
import time
import ray


from collections import OrderedDict

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

def summarize_time(tag, time_list_ms):
  # delete first large interval if exists
  if time_list_ms and time_list_ms[0]>3600*10:
    del time_list_ms[0]
    
  if len(time_list_ms)>0:
    min = np.min(time_list_ms)
    median = np.median(time_list_ms)
    formatted = ["%.2f"%(d,) for d in time_list_ms[:10]]
    print("%-20s: min: %.2f, median: %.2f, mean: %.2f"%(tag, min, median,
                                                        np.mean(time_list_ms)))
  else:
    print("Times: <empty>")
    


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
    print(self.tag, interval_ms)

def main():
  params0 = np.ones((args.dim,), dtype=np.float32)/(np.sqrt(args.dim))
  params = tf.Variable(params0)
  loss = tf.reduce_sum(tf.square(params))

  #  optimizer = tf.train.GradientDescentOptimizer(0.01)
  grad = tf.gradients(loss, params)[0]
  slow_grad = tf.identity(grad)
  train_op = params.assign_sub(0.01*slow_grad)
  
  sess = tf.InteractiveSession()
  sess.run(params.initializer)

  for i in range(100):
    loss0 = loss.eval()
    print(loss0)
    with timeit('train_op'):
      sess.run(train_op)

  for key, times in global_timeit_dict.items():
    summarize_time(key, times)

if __name__ == '__main__':
  main()
