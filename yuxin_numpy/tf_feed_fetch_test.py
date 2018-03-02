# Benchmark various ways of getting numpy arrays in and out of TensorFlow
# --allocator types are:
# numpy: default numpy array
# tf: 64-byte aligned numpy array
# tfgpu: 64-byte aligned numpy array in pinned memory
# ray: numpy array returned from ray


"""
test
"""

import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import threading
import time

from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--size-mb", default=100, type=int,
                    help="size of data in MBs")
parser.add_argument("--allocator", default='tf', type=str,
                    help="Which allocator to use for numpy array memory: "
                    "numpy/tf/tfgpu/ray/ray_hacked")
parser.add_argument("--num-iters", default=51, type=int,
                    help="number of iterations")
parser.add_argument("--profile", default=0, type=int,
                    help="dump stepstats/timelines into 'data' directory")
parser.add_argument('--benchmark', default='all', type=str)
args = parser.parse_args()
args_dim = args.size_mb * 250*1000


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


def summarize_time(tag, time_list_ms):
  """Print summary of times/bandwidth."""

  del time_list_ms[0]  # first entry is noisy

  if len(time_list_ms)>0:
    min = np.min(time_list_ms)
    mean = np.mean(time_list_ms)
    median = np.median(time_list_ms)
    data_size_gb = args_dim*4/1e9
    time_sec = min/1000
    bw = data_size_gb/time_sec
    formatted = ["%.2f"%(d,) for d in time_list_ms[:10]]
    print("%-30s: %5.1f GB/sec, min: %5.2f, median: %5.2f, mean: %5.2f"%(tag, bw, min, median, mean))
  else:
    print("Times: <empty>")
    

def sessrun(*args, **kwargs):
  if args.profile:
    traced_run(*args, **kwargs)
  else:
    regular_run(*args, **kwargs)
    

def regular_run(*args, **kwargs):
  sess = tf.get_default_session()
  sess.run(*args, **kwargs)


timeline_counter = 0
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
def traced_run(*args, **kwargs):
  """Runs fetches, dumps timeline files in current directory."""

  global timeline_counter, run_options
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
    try:
      ray.init(object_store_memory=(10 ** 9), num_workers=0)
    except:  # older version doesn't have object_store_memory
      print("Falling back on older Ray init")
      ray.init(num_workers=0)
      

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
  
  params0 = np.ones((args_dim,), dtype=np.float32)
  #  params0 = np.random.randn(args_dim).astype(dtype=np.float32)

  if args.allocator == 'numpy':
    pass
  elif args.allocator == 'numpy_readonly':
    params0.flags['WRITEABLE'] = False
  elif args.allocator == 'tf':
    params0 = align_numpy_tf(params0)
  elif args.allocator == 'tf_readonly':
    params0 = align_numpy_tf(params0)
    params0.flags['WRITEABLE'] = False
  elif args.allocator == 'tfgpu':
    params0 = align_numpy_tfgpu(params0)
  elif args.allocator == 'ray':
    params0 = align_numpy_ray(params0)
  elif args.allocator == 'ray_hacked':
    params0 = align_numpy_ray(params0)
    params0.flags['WRITEABLE'] = True
  elif args.allocator == 'pytorch':
    params0 = align_numpy_pytorch(params0)
  elif args.allocator == 'pytorch_readonly':
    params0 = align_numpy_pytorch(params0)
    params0.flags['WRITEABLE'] = False
  else:
    assert False, "Unknown allocator type "+str(args.allocator)
  return params0


def fetch_cpu_tensor():
  with tf.device('/cpu:0'):
    params = tf.fill((args_dim,), 2.0)
    
  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('fetch_cpu_tensor'):
      sess.run(params)

def fetch_gpu_tensor():
  data = np.ones((args_dim,), dtype=np.float32)
  with tf.device('/cpu:0'):
    params = tf.fill((args_dim,), 2.0)    

  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('fetch_gpu_tensor'):
      result = sess.run(params)

def feed_cpu_tensor():
  params0 = create_array()
  with tf.device('/cpu:0'):
    one = tf.fill([1],1.0)
    params = tf.placeholder(tf.float32)
    result = tf.concat([params, one], axis=0)
  for i in range(args.num_iters):
    with timeit('feed_cpu_tensor'):
      sess.run(result.op, feed_dict = {params: params0})

def feed_gpu_tensor():
  params0 = create_array()
  with tf.device('/gpu:0'):
    params = tf.placeholder(tf.float32)
    one = tf.fill([1],1.0)
    result = tf.concat([params, one], axis=0)
  for i in range(args.num_iters):
    with timeit('feed_gpu_tensor'):
      sess.run(result.op, feed_dict = {params: params0})

def cpu_vector_add():
  with tf.device('/gpu:0'):
    data = np.ones((args_dim,), dtype=np.float32)
    params = tf.get_variable("cpu_vector_add", initializer=data,
                             use_resource=True)
    result_op = params.assign_add(data)
  sess = tf.get_default_session()
  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('cpu_vector_add'):
      sess.run(result_op)
      
def gpu_vector_add():
  with tf.device('/gpu:0'):
    data = np.ones((args_dim,), dtype=np.float32)
    params = tf.get_variable("gpu_vector_add", initializer=data,
                             use_resource=True)
    result_op = params.assign_add(data)
  sess = tf.get_default_session()
  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('gpu_vector_add'):
      sess.run(result_op)

      
def print_socket_info():
  import re
  socket_re = re.compile(".*?processor.*?(?P<cpu>\d+).*?physical id.*?(?P<socket>\d+).*?power", flags=re.S)
  from collections import defaultdict
  socket_dict = defaultdict(list)
  for cpu, socket in socket_re.findall(open('/proc/cpuinfo').read()):
    socket_dict[socket].append(cpu)

  print("number of sockets:", len(socket_dict))

if __name__ == '__main__':

  # remove garbage colleciton, automatic optimizations and tuning
  import gc
  gc.disable()

  os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
  os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
  os.environ['CUDA_LAUNCH_BLOCKING']='1'
  import tensorflow as tf
  from tensorflow.core.protobuf import rewriter_config_pb2
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
  config.graph_options.place_pruned_graph = True

  print_socket_info()
  print("TensorFlow version", tf.__version__)
  
  sess = tf.InteractiveSession(config=config)
  data = create_array()

  feed_cpu_tensor()
  fetch_cpu_tensor()
  #  cpu_vector_add()
  #  gpu_vector_add()

  if tf.test.is_gpu_available():
    feed_gpu_tensor()
    fetch_gpu_tensor()
  
  else:
    cmd = args.benchmark+'()'
    exec(cmd)
    
  for key, times in global_timeit_dict.items():
    summarize_time(key, times)
  #main()
