# Run D2H and H2D benchmark with synthetic workload with feed-fetch step

import tensorflow as tf

import argparse
import numpy as np
import time
import ray

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
parser.add_argument("--dim", default=25*1000*1000, type=int,
                    help="The number of parameters.")
parser.add_argument("--align", default='none', type=str,
                    help="none/cpu/gpu/ray")
parser.add_argument("--target", default='cpu', type=str,
                    help="where target tensor lives (cpu or gpu)")
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
    #    print("%20s %10.2f"%(self.tag, interval_ms))


def summarize_time(tag, time_list_ms):
  # delete first large interval if exists
  #  if time_list_ms and time_list_ms[0]>3600*10:
  del time_list_ms[0]

  if len(time_list_ms)>0:
    min = np.min(time_list_ms)
    mean = np.mean(time_list_ms)
    median = np.median(time_list_ms)
    data_size_gb = args.dim*4/1e9
    time_sec = min/1000
    bw = data_size_gb/time_sec
    formatted = ["%.2f"%(d,) for d in time_list_ms[:10]]
    print("%-20s: %.1f GB/sec, min: %.2f, median: %.2f, mean: %.2f"%(tag, bw, min, median, mean))
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

def fetch_cpu_variable():
  data = np.ones((args.dim,), dtype=np.float32)
  with tf.device('/cpu:0'):
    params = tf.Variable(initial_value=data)

  sess.run(tf.global_variables_initializer())
  for i in range(20):
    with timeit('fetch_cpu_variable'):
      sess.run(params)

def fetch_cpu_variable_add():
  data = np.ones((args.dim,), dtype=np.float32)
  with tf.device('/cpu:0'):
    params = tf.Variable(initial_value=data)
    params = params+0.1
    
  sess.run(tf.global_variables_initializer())
  for i in range(20):
    with timeit('fetch_cpu_variable_add'):
      sess.run(params)


def fetch_cpu_variable_concat():
  data = np.ones((args.dim,), dtype=np.float32)
  with tf.device('/cpu:0'):
    params = tf.Variable(initial_value=data)
    params = tf.concat([params, tf.fill([1],1.0)], axis=0)
  sess.run(tf.global_variables_initializer())
  for i in range(20):
    with timeit('fetch_cpu_variable_concat'):
      sess.run(params)


def main():
  global grad_cached_const
  import gc
  gc.disable()
  
  params0 = np.ones((args.dim,), dtype=np.float32)/(np.sqrt(args.dim))

  if args.align == 'none':
    pass
  elif args.align == 'cpu':
    params0 = align_numpy_cpu(params0)
  elif args.align == 'gpu':
    params0 = align_numpy_gpu(params0)
    
  loss, params, grad_cached, grad_assign_op = create_net('net1', params0)
  
  sess.run(tf.global_variables_initializer())

  lr = 0.01
  for i in range(10):
    loss0 = loss.eval()
    print(loss0)

    with timeit('step'):
      pass
      #      sess.run(grad_assign_op)
      
    with timeit('fetch'):
      #      grad0 = sess.run(grad_cached)
      grad0 = sess.run(grad_cached_const)

    # takes 75ms, 33ms is on allocation, 16ms on multiplication
    with timeit('add'):
      params0-=grad0*lr

    with timeit('feed'):
      #      params.load(params0)
      sess.run(params.initializer, feed_dict={params.initial_value:params0})


  for key, times in global_timeit_dict.items():
    summarize_time(key, times)

  assert abs(loss0-0.69513524)<0.01
  print('test passed')
  
if __name__ == '__main__':
  import gc
  gc.disable()
  sess = tf.InteractiveSession()
  fetch_cpu_variable()
  fetch_cpu_variable_add()
  fetch_cpu_variable_concat()
  for key, times in global_timeit_dict.items():
    summarize_time(key, times)
