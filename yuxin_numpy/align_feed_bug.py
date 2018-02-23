import tensorflow as tf


import argparse
import numpy as np
import time

import gc

import os
import subprocess
import sys
import tensorflow as tf
import threading
import time
import pickle


from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--placement", default='cpu', type=str,
                    help="The number of parameter servers to use.")
parser.add_argument("--align", default=0, type=int,
                    help="Whether to manually align array")
parser.add_argument("--dim", default=25*1000*1000, type=int,
                    help="The number of parameters.")
args = parser.parse_args()


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
    print("%20s %10.2f"%(self.tag, interval_ms))

# https://github.com/numpy/numpy/issues/5312
import numpy as np

def empty_aligned(n, align=128):
    """
    Get n bytes of memory wih alignment align.
    """

    a = np.empty(n + (align - 1), dtype=np.float32)
    data_align = a.ctypes.data % align
    offset = 0 if data_align == 0 else (align - data_align)
    return a[offset : offset + n]

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
    

def main():
  gc.disable()
  if args.align:
    params0 = empty_aligned(args.dim)
  else:
    params0 = np.ones((args.dim,), dtype=np.float32)
  
  
  with tf.device('/gpu:0'):
    gpu_params = tf.Variable(initial_value=params0)
    gpu_tensor = tf.ones((args.dim,), dtype=np.float32)
  with tf.device('/cpu:0'):
    cpu_params = tf.Variable(initial_value=params0)
    cpu_tensor = tf.ones((args.dim,), dtype=np.float32)
    
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  for i in range(20):
    with timeit('feed-cpu-variable'):
      sess.run(cpu_params.initializer,
               feed_dict={cpu_params.initial_value:params0})


  print("%.0f MB variable"%(args.dim*4./1e6,))
  for key, times in global_timeit_dict.items():
    summarize_time(key, times)
  
if __name__ == '__main__':
  main()
