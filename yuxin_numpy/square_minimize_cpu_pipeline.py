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
parser.add_argument("--allocator", default='numpy', type=str,
                    help="Which allocator to use for numpy array memory: "
                    "numpy/tf/tfgpu/ray")
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
    


def create_net(name, params0):
  """Creates network that runs minimization.
  Returns parameter tensor and gradient tensor (on GPU)."""

  with tf.device('/gpu:0'):
    params = tf.Variable(initial_value=params0, name=name+'-params')
  loss = tf.reduce_sum(tf.square(params))
  grad_cached = tf.Variable(initial_value=params0, name=name+'-grads')
  grad = tf.gradients(loss, params)[0]
  
  n = 11200  # this takes about 200ms on V100
  a = tf.random_uniform((n, n))/n
  b = tf.random_uniform((n, n))/n
  slow_op = a@b
  slow_grad = grad+1e-10*tf.reduce_sum(slow_op)
  grad_assign_op = grad_cached.assign(slow_grad)

  return loss, params, grad_cached, grad_assign_op

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

def align_numpy_tf(unaligned):
  sess = tf.get_default_session()
  with tf.device('/cpu:0'):
    tensor = tf.ones(unaligned.shape, dtype=unaligned.dtype)
  aligned = sess.run(tensor)
  np.copyto(aligned, unaligned)
  return aligned


def align_numpy_tfgpu(unaligned):
  sess = tf.get_default_session()
  with tf.device('/gpu:0'):
    tensor = tf.zeros(unaligned.shape, dtype=unaligned.dtype)
  aligned = sess.run(tensor)
  np.copyto(aligned, unaligned)
  return aligned


def align_numpy_ray(unaligned):
  if 'ray' not in sys.modules:  # avoid calling ray.init twice which crashes
    import ray
    ray.init(object_store_memory=(10 ** 9), num_workers=0)

  import ray
  @ray.remote
  def f():
    return unaligned

  result = ray.get(f.remote())
  return result

def align_numpy_pytorch(unaligned):
  import torch
  return torch.from_numpy(unaligned).clone().numpy()


def create_array():
  """Creates numpy array, using size and allocator specified in args."""
  
  params0 = np.ones((args.dim,), dtype=np.float32)
  #  params0 = np.random.randn(args_dim).astype(dtype=np.float32)

  if args.allocator == 'numpy':
    pass
  elif args.allocator == 'tf':
    params0 = align_numpy_tf(params0)
  elif args.allocator == 'tfgpu':
    params0 = align_numpy_tfgpu(params0)
  elif args.allocator == 'ray':
    params0 = align_numpy_ray(params0)
  elif args.allocator == 'pytorch':
    params0 = align_numpy_pytorch(params0)
  else:
    assert False, "Unknown allocator type "+str(args.allocator)
  return params0


def main():
  sess = tf.InteractiveSession()
  
  params0 = np.ones((args.dim,), dtype=np.float32)/(np.sqrt(args.dim))
  params0 = create_array()/np.sqrt(args.dim)

  loss, params, grad_cached, grad_assign_op = create_net('net1', params0)
  
  sess.run(tf.global_variables_initializer())

  lr = 0.01
  for i in range(10):
    loss0 = loss.eval()
    print(loss0)

    with timeit('step'):
      sess.run(grad_assign_op)
      
    with timeit('fetch'):
      grad0 = sess.run(grad_cached)

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
  main()
  params0 = create_array()/np.sqrt(args.dim)
  
