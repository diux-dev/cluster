import tensorflow as tf


import argparse
import numpy as np
import time
import ray

import gc

import os
import portpicker
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
    


timeline_counter = 0
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
def sessrun(*args, **kwargs):
  """Runs fetches, dumps timeline files in current directory."""

  global timeline_counter
  run_metadata = tf.RunMetadata()

  log_fn = "%s"%(timeline_counter,)
  sess = tf.get_default_session()
  
  root = os.getcwd()+"/data"
  os.system('mkdir -p '+root)
  
  from tensorflow.python.client import timeline

  kwargs['options'] = run_options
  kwargs['run_metadata'] = run_metadata
  results = sess.run(*args, **kwargs)
  
  tl = timeline.Timeline(step_stats=run_metadata.step_stats)
  ctf = tl.generate_chrome_trace_format(show_memory=True,
                                          show_dataflow=False)
  open(root+"/timeline_%s.json"%(log_fn,), "w").write(ctf)
  open(root+"/stepstats_%s.pbtxt"%(log_fn,), "w").write(str(
    run_metadata.step_stats))
  timeline_counter+=1
  return results

def main():
  gc.disable()
  params0 = np.ones((args.dim,), dtype=np.float32)

  with tf.device('/gpu:0'):
    gpu_params = tf.Variable(initial_value=params0)
    gpu_tensor = tf.ones((args.dim,), dtype=np.float32)
  with tf.device('/cpu:0'):
    cpu_params = tf.Variable(initial_value=params0)
    cpu_tensor = tf.ones((args.dim,), dtype=np.float32)
    
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  for i in range(100):
    with timeit('fetch-cpu-variable'):
      val0 = sess.run(cpu_params)
    with timeit('fetch-cpu-tensor'):
      val1  = sess.run(cpu_tensor)
    with timeit('fetch-gpu-variable'):
      val2 = sess.run(gpu_params)
    with timeit('fetch-gpu-tensor'):
      val3  = sess.run(gpu_tensor)
      
    with timeit('feed-cpu-variable'):
      sess.run(cpu_params.initializer,
               feed_dict={cpu_params.initial_value:params0})
    with timeit('feed-gpu-variable'):
      sess.run(gpu_params.initializer,
               feed_dict={gpu_params.initial_value:params0})


  print("%.0f MB variable"%(args.dim*4./1e6,))
  for key, times in global_timeit_dict.items():
    summarize_time(key, times)
  
if __name__ == '__main__':
  main()
