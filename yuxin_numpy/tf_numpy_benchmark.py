# Benchmark various ways of getting numpy arrays in and out of TensorFlow
# --allocator types are:
# numpy: default numpy array
# tf: 64-byte aligned numpy array
# tfgpu: 64-byte aligned numpy array in pinned memory
# ray: numpy array returned from ray


"""
# p3.16xlarge, Deep Learning AMI v3.0
# Numpy: MKL 2018.0.1 Product Build 20171007
# TensorFlow 1.7.0-dev20180221 (d100729)
# Ray 0.3.1 (1c35f06)

python tf_numpy_benchmark.py
numpy_create                  :   4.4 GB/sec, min: 22.86, median: 23.32, mean: 23.44
numpy_copy                    :   7.8 GB/sec, min: 12.79, median: 12.97, mean: 13.10
fetch_cpu_variable            :   2.5 GB/sec, min: 40.09, median: 40.51, mean: 40.70
fetch_cpu_variable_add        :  13.6 GB/sec, min:  7.37, median:  8.64, mean:  8.64
fetch_cpu_variable_concat     :  14.6 GB/sec, min:  6.85, median:  7.53, mean:  7.74
fetch_cpu_tensor              :  19.2 GB/sec, min:  5.22, median:  6.30, mean:  6.39
fetch_gpu_variable            :  11.9 GB/sec, min:  8.42, median:  8.50, mean:  8.54
fetch_gpu_variable_add        :  11.6 GB/sec, min:  8.62, median:  8.71, mean:  8.77
fetch_gpu_tensor              :  17.2 GB/sec, min:  5.82, median:  6.97, mean:  6.90
feed_cpu_variable             :   3.5 GB/sec, min: 28.91, median: 31.20, mean: 31.57
feed_gpu_variable             :   3.5 GB/sec, min: 28.69, median: 32.28, mean: 32.66
feed_cpu_tensor               :   4.8 GB/sec, min: 20.93, median: 22.91, mean: 24.44
feed_gpu_tensor               :   3.1 GB/sec, min: 32.16, median: 34.05, mean: 34.01


python tf_numpy_benchmark.py --allocator=tf
numpy_create                  :   4.7 GB/sec, min: 21.48, median: 21.61, mean: 21.68
numpy_copy                    :   7.6 GB/sec, min: 13.23, median: 13.65, mean: 13.66
fetch_cpu_variable            :   2.5 GB/sec, min: 39.53, median: 39.97, mean: 40.03
fetch_cpu_variable_add        :  13.8 GB/sec, min:  7.23, median:  8.37, mean:  8.40
fetch_cpu_variable_concat     :  12.7 GB/sec, min:  7.89, median:  8.71, mean:  8.88
fetch_cpu_tensor              :  19.4 GB/sec, min:  5.15, median:  6.15, mean:  6.27
fetch_gpu_variable            :  11.6 GB/sec, min:  8.61, median:  8.70, mean:  8.74
fetch_gpu_variable_add        :  11.6 GB/sec, min:  8.64, median:  8.76, mean:  8.76
fetch_gpu_tensor              :  18.5 GB/sec, min:  5.41, median:  6.56, mean:  6.75
feed_cpu_variable             :   5.1 GB/sec, min: 19.64, median: 20.48, mean: 20.38
feed_gpu_variable             :   5.0 GB/sec, min: 20.05, median: 21.37, mean: 21.29
feed_cpu_tensor               :  13.7 GB/sec, min:  7.28, median:  8.28, mean:  8.47
feed_gpu_tensor               :   6.5 GB/sec, min: 15.45, median: 19.64, mean: 18.96

python tf_numpy_benchmark.py --allocator=tfgpu
numpy_create                  :   4.6 GB/sec, min: 21.64, median: 21.80, mean: 21.94
numpy_copy                    :   6.4 GB/sec, min: 15.71, median: 15.86, mean: 16.35
fetch_cpu_variable            :   2.3 GB/sec, min: 42.78, median: 43.45, mean: 44.01
fetch_cpu_variable_add        :  13.4 GB/sec, min:  7.46, median:  8.25, mean:  8.29
fetch_cpu_variable_concat     :  13.5 GB/sec, min:  7.38, median:  8.83, mean:  9.01
fetch_cpu_tensor              :  18.4 GB/sec, min:  5.42, median:  6.00, mean:  6.29
fetch_gpu_variable            :  11.7 GB/sec, min:  8.54, median:  8.60, mean:  8.64
fetch_gpu_variable_add        :  11.6 GB/sec, min:  8.63, median:  8.75, mean:  8.77
fetch_gpu_variable_concat     :  11.4 GB/sec, min:  8.79, median:  9.00, mean:  9.00
fetch_gpu_tensor              :  18.4 GB/sec, min:  5.43, median:  6.10, mean:  6.31
feed_cpu_variable             :   6.2 GB/sec, min: 16.06, median: 18.04, mean: 17.88
feed_gpu_variable             :  10.1 GB/sec, min:  9.94, median: 10.15, mean: 10.15
feed_cpu_tensor               :  12.8 GB/sec, min:  7.84, median:  9.36, mean:  9.23
feed_gpu_tensor               :  10.0 GB/sec, min:  9.96, median: 10.27, mean: 10.25

python tf_numpy_benchmark.py --allocator=ray
numpy_create                  :   4.4 GB/sec, min: 22.94, median: 23.67, mean: 24.08
fetch_cpu_variable            :   2.5 GB/sec, min: 40.75, median: 41.03, mean: 41.34
fetch_cpu_variable_add        :  13.0 GB/sec, min:  7.69, median:  8.71, mean:  8.71
fetch_cpu_tensor              :  17.9 GB/sec, min:  5.58, median:  6.38, mean:  6.46
fetch_gpu_variable            :  11.7 GB/sec, min:  8.57, median:  8.73, mean:  8.72
fetch_gpu_variable_add        :  11.6 GB/sec, min:  8.65, median:  8.71, mean:  8.73
fetch_gpu_tensor              :  18.2 GB/sec, min:  5.50, median:  6.62, mean:  6.61
feed_cpu_variable             :   1.5 GB/sec, min: 67.34, median: 72.91, mean: 71.25
feed_gpu_variable             :   1.4 GB/sec, min: 73.41, median: 75.11, mean: 75.08
feed_cpu_tensor               :   1.6 GB/sec, min: 62.32, median: 66.16, mean: 68.01
feed_gpu_tensor               :   1.3 GB/sec, min: 76.19, median: 83.05, mean: 83.73


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
parser.add_argument("--allocator", default='numpy', type=str,
                    help="Which allocator to use for numpy array memory: "
                    "numpy/tf/tfgpu/ray")
parser.add_argument("--num-iters", default=21, type=int,
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


def numpy_create():
  for i in range(args.num_iters):
    with timeit('numpy_create'):
      np.ones((args_dim,), dtype=np.float32)

def numpy_copy():
  """Copy data into existing numpy array"""
  if args.allocator == 'ray':
    return # ray creates read-only arrays
  
  data = create_array()
  target = create_array()
  for i in range(args.num_iters):
    with timeit('numpy_copy'):
      np.copyto(target, data)


def numpy_array():
  params0 = create_array()
  for i in range(args.num_iters):
    with timeit('numpy_array'):
      np.array(params0)
  
def numpy_access():
  for i in range(args.num_iters):
    params0 = create_array()
    with timeit('numpy_access'):
      params0[:100]
  
def numpy_add():
  """Copy data into existing numpy array"""

  params0 = create_array()
  params1 = create_array()
  for i in range(args.num_iters):
    with timeit('numpy_add'):
      params0+=params1

def pytorch_add():
  """Copy data into existing numpy array"""

  import torch
  params0 = torch.from_numpy(create_array())
  params1 = torch.from_numpy(create_array())
  for i in range(args.num_iters):
    with timeit('pytorch_add'):
      params0+=params1


def pytorch_add0():
  import torch
  params0 = torch.from_numpy(create_array())
  for i in range(args.num_iters):
    with timeit('pytorch_add0'):
      params1 = params0+0


def pytorch_add1():
  import torch
  params0 = torch.from_numpy(create_array())
  for i in range(args.num_iters):
    with timeit('pytorch_add1'):
      params1 = params0+0

def pytorch_clone():
  import torch
  params0 = torch.from_numpy(create_array())
  for i in range(args.num_iters):
    with timeit('pytorch_clone'):
      params1 = params0.clone()
  
def pytorch_add_newobject():
  """add vectors, put result into new memory"""

  import torch
  params0 = torch.from_numpy(create_array())
  params1 = torch.from_numpy(create_array())
  for i in range(args.num_iters):
    with timeit('pytorch_add_newobject'):
      params0+params1

def pytorch_add_fast():
  """Add custom allocated object to pytorch allocated object"""

  import torch
  params0 = torch.from_numpy(create_array()).clone()
  params1 = torch.from_numpy(create_array())
  for i in range(args.num_iters):
    with timeit('pytorch_add_fast'):
      params0+=params1


def fetch_cpu_variable():
  with tf.device('/cpu:0'):
    params = tf.Variable(initial_value=data)

  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('fetch_cpu_variable'):
      sess.run(params)

def fetch_cpu_resource_variable():
  with tf.device('/cpu:0'):
    params = tf.get_variable("var", initializer=data, use_resource=True)

  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('fetch_cpu_resource_variable'):
      sess.run(params)

def fetch_cpu_variable_add():
  with tf.device('/cpu:0'):
    params_var = tf.Variable(initial_value=data)
    params = params_var+0.1
    
  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    #    params_var.load(np.random.randn(args_dim).astype(dtype=np.float32))
    with timeit('fetch_cpu_variable_add'):
      result = sess.run(params)
    #    print(result[0])


def fetch_cpu_tensor():
  with tf.device('/cpu:0'):
    params = tf.fill((args_dim,), 2.0)
    
  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('fetch_cpu_tensor'):
      sess.run(params)


def fetch_gpu_variable():
  with tf.device('/gpu:0'):
    params = tf.Variable(initial_value=data)

  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('fetch_gpu_variable'):
      sess.run(params)


def fetch_gpu_variable_add():
  with tf.device('/gpu:0'):
    params = tf.Variable(initial_value=data)
    params = params+0.1
    
  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('fetch_gpu_variable_add'):
      sess.run(params)


def fetch_gpu_tensor():
  data = np.ones((args_dim,), dtype=np.float32)
  with tf.device('/cpu:0'):
    params = tf.fill((args_dim,), 2.0)    

  sess.run(tf.global_variables_initializer())
  for i in range(args.num_iters):
    with timeit('fetch_gpu_tensor'):
      result = sess.run(params)

def feed_cpu_variable():
  params0 = create_array()

  with tf.device('/cpu:0'):
    params = tf.Variable(initial_value=params0)

  for i in range(args.num_iters):
    with timeit('feed_cpu_variable'):
      params.load(params0)

def feed_gpu_variable():
  params0 = create_array()

  with tf.device('/gpu:0'):
    params = tf.Variable(initial_value=params0)

  for i in range(args.num_iters):
    with timeit('feed_gpu_variable'):
      params.load(params0)
  
    
def feed_cpu_tensor():
  params0 = create_array()
  with tf.device('/cpu:0'):
    params = tf.placeholder(tf.float32)
    result = tf.concat([params, tf.fill([1],1.0)], axis=0)
  for i in range(args.num_iters):
    with timeit('feed_cpu_tensor'):
      sess.run(result.op, feed_dict = {params: params0})

def feed_cpu_tensor_with_copy():
  params0 = create_array()
  with tf.device('/cpu:0'):
    params = tf.placeholder(tf.float32)
    result = tf.concat([params, tf.fill([1],1.0)], axis=0)

  params0 = np.ones((args_dim,), dtype=np.float32)
  params1 = 2*np.ones((args_dim,), dtype=np.float32)
  
  params0 = align_numpy_tf(params0)
  params1 = align_numpy_ray(params1)


  for i in range(args.num_iters):
    with timeit('feed_cpu_tensor'):
      np.copyto(params0, params1)
      sess.run(result.op, feed_dict = {params: params0})

def feed_gpu_tensor():
  params0 = create_array()
  with tf.device('/gpu:0'):
    params = tf.placeholder(tf.float32)
    result = tf.concat([params, tf.fill([1],1.0)], axis=0)
  for i in range(args.num_iters):
    with timeit('feed_gpu_tensor'):
      sess.run(result.op, feed_dict = {params: params0})

def feed_gpu_tensor_fast():
  import torch

  # allocate page-locked memory and turn it into PyTorch tensor
  params0 = np.zeros((args_dim,), dtype=np.float32)
  params0 = align_numpy_tfgpu(params0)
  params0 = torch.from_numpy(params0)
  
  # copy Ray array into page-locked memory using PyTorch OMP add
  # for top performance, restrict GOMP_CPU_AFFINITY to single physical socket
  # https://github.com/pytorch/pytorch/issues/5412

  # new array with custom allocator
  params1 = create_array()
  
  with tf.device('/gpu:0'):
    params = tf.placeholder(tf.float32)
    result = params + 1
  
  for i in range(args.num_iters):
    with timeit('feed_gpu_tensor_fast1'):
      params0+=torch.from_numpy(params1)  # use PyTorch to copy data into page-locked memory
    with timeit('feed_gpu_tensor_fast2'):
      sess.run(result.op, feed_dict = {params: params0})

def ray_copy():

  params0 = np.ones((args_dim,), dtype=np.float32)
  params1 = 2*np.ones((args_dim,), dtype=np.float32)
  
  params0 = align_numpy_tf(params0)
  params1 = create_array()

  for i in range(args.num_iters):
    with timeit('ray_copy'):
      np.copyto(params0, params1)

def pytorch_from_numpy():
  """Convert numpy array to pytorch, using 'clone' to move it to proper
  memory."""
  
  import torch
  
  params0 = create_array()
  for i in range(args.num_iters):
    with timeit('pytorch_from_numpy'):
      align_numpy_pytorch(params0)
  
def pytorch_from_numpy_noclone():
  """Convert numpy array to pytorch, <1ms because no copy takes place."""
  import torch
  
  params0 = create_array()
  for i in range(args.num_iters):
    with timeit('pytorch_from_numpy'):
      torch.from_numpy(params0)
  
def pytorchadd_from_numpy():
  """Convert numpy array to pytorch, add a number to resulting tensor."""
  import torch
  
  params0 = create_array()
  for i in range(args.num_iters):
    with timeit('pytorch_from_numpy'):
      result = torch.from_numpy(params0)
      result+=1
  
def pytorchgpu_from_numpy():
  import torch
  
  params0 = create_array()
  for i in range(args.num_iters):
    with timeit('pytorchgpu_from_numpy'):
      result = torch.from_numpy(params0)
      result = result.cuda()
  
def pytorch_sharememory():
  import torch
  
  params0 = create_array()
  for i in range(args.num_iters):
    with timeit('pytorch_sharememory'):
      result = torch.share_memory(params0)
  

def allocator_alignment():
  for allocator in ['numpy', 'ray', 'pytorch', 'tf']:
    args.allocator = allocator
    result = create_array()
    print("%10s: %d"%(allocator, result.ctypes.data%64))

def allreduce():
  from mpi4py import MPI

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  print("hello world from process ", rank)


  senddata = create_array()
  recvdata = create_array()
  print('on task',rank,'before allreduce:    data = ',recvdata[0])

  for i in range(args.num_iters):
    with timeit('mpi_reduce_sum'):
      comm.Allreduce(senddata,recvdata)
      
  print('on task',rank,'after Allreduce:    data = ',recvdata[0])
      

if __name__ == '__main__':

  # remove garbage colleciton, automatic optimizations and tuning
  import gc
  gc.disable()

  os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
  os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
  os.environ['CUDA_LAUNCH_BLOCKING']='1' 
  import tensorflow as tf
  from tensorflow.core.protobuf import rewriter_config_pb2
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
  config.graph_options.place_pruned_graph = True
  config.gpu_options.allow_growth=True  # leave some space for PyTorch
  sess = tf.InteractiveSession(config=config)

  data = create_array()

  print("Using %d MB of data, times are in ms"%(args.size_mb))
  if args.benchmark == 'all':
    numpy_create()
    numpy_copy()
    fetch_cpu_variable()
    fetch_cpu_variable_add()
    fetch_cpu_tensor()

    fetch_gpu_variable()
    fetch_gpu_variable_add()
    fetch_gpu_tensor()

    feed_cpu_variable()
    feed_gpu_variable()
    feed_cpu_tensor()
    feed_gpu_tensor()

  else:
    cmd = args.benchmark+'()'
    exec(cmd)
    
  for key, times in global_timeit_dict.items():
    summarize_time(key, times)
  #main()
