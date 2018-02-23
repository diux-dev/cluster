import tensorflow as tf


# generating profiles
# sudo apt-get install -y google-perftools

# export LD_PRELOAD="/usr/lib/libtcmalloc_and_profiler.so.4"
# export CPUPROFILE="/home/ubuntu/git0/cluster/yuxin_numpy/cpuprofile/prof0"
# python allocation_bug_report.py --placement=cpu
# unset LD_PRELOAD
# export p=/home/ubuntu/git0/cluster/yuxin_numpy/cpuprofile/prof0
# google-pprof `which python` $p --svg > cpuprofile/prof0.svg

# export LD_PRELOAD="/usr/lib/libtcmalloc_and_profiler.so.4"
# export CPUPROFILE="/home/ubuntu/git0/cluster/yuxin_numpy/cpuprofile/prof1"
# python allocation_bug_report.py --placement=gpu
# unset LD_PRELOAD
# export p=/home/ubuntu/git0/cluster/yuxin_numpy/cpuprofile/prof1
# google-pprof `which python` $p --svg > cpuprofile/prof1.svg


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
parser.add_argument("--dim", default=25*1000*10000, type=int,
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

  if args.placement == 'gpu':
    dev = '/gpu:0'
  elif args.placement == 'cpu':
    dev = '/cpu:0'
    
  with tf.device(dev):
    tensor = tf.ones((args.dim,), dtype=np.float32)
    
  sess = tf.InteractiveSession()

  for i in range(10):
    with timeit('fetch'):
      sess.run(tensor)

  print("%.0f MB variable on %s"%(args.dim*4./1e6,args.placement))
  for key, times in global_timeit_dict.items():
    summarize_time(key, times)
  
if __name__ == '__main__':
  main()
