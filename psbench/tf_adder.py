#!/usr/bin/env python
# Asynchronous parameter server benchmark in TensorFlow


import argparse
import base64
import os
import pickle
import subprocess
import sys
import threading
import time

from collections import OrderedDict
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import device as pydev

import util as u

parser = argparse.ArgumentParser()
parser.add_argument("--size-mb", default=100, type=int,
                    help="size of data in MBs")
parser.add_argument("--iters", default=10000, type=int,
                    help="number of iterations for worker")
parser.add_argument("--profile", default=1, type=int,
                    help="dump stepstats/timelines into 'data' directory")
parser.add_argument("--logdir", default='', type=str,
                    help="TensorBoard events go here")
parser.add_argument("--label", default='', type=str,
                    help="location of logging directory")
parser.add_argument("--ps", default=1, type=int,
                    help="number of parameter server shards")
args = parser.parse_args()
params_size = args.size_mb * 250*1000
sharded_params_size = params_size//args.ps

dtype = np.float32
RETRY_DELAY_SEC = 5   # distributed session retries

# TODO: when ps server restarts, it doesn't reinitialize the variables
# TODO: document TF_CONFIG

timeline_counter = 0
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
def traced_run(*args, **kwargs):
  """Runs fetches, dumps timeline files into {cwd}/data."""

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


def sessrun(*args1, **kwargs):
  if args.profile:
    return traced_run(*args1, **kwargs)
  else:
    return regular_run(*args1, **kwargs)


def regular_run(*args, **kwargs):
  sess = tf.get_default_session()
  return sess.run(*args, **kwargs)


def get_ps_device(task=0, op_device_str=''):
  device_str = '/job:ps'
  device = pydev.DeviceSpec.from_string(device_str)
  device.task = task
  op_device = pydev.DeviceSpec.from_string(op_device_str)
  device.merge_from(op_device)
  return device.to_string()

# todo: private methods
def get_worker_device(task, op_device_str=''):
  device_str = '/job:worker'
  device = pydev.DeviceSpec.from_string(device_str)
  device.task = task
  op_device = pydev.DeviceSpec.from_string(op_device_str)
  device.merge_from(op_device)
  return device.to_string()


def session_config():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(
    graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  
  config.operation_timeout_in_ms = 10*1000  # abort after 10 seconds
  return config


def make_params():
  ps_device = get_ps_device(0)
  with tf.device(ps_device):
    params = tf.get_variable("params", [params_size], dtype,
                             initializer=tf.ones_initializer())
  return params


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
    
    newtag = 'time/'+self.tag
    tb_logger = get_tb_logger()
    if tb_logger:
      tb_logger(newtag, interval_ms)

  

global_last_logger = None
def get_tb_logger():
  global_last_logger
  return global_last_logger

class FileLogger:
  """Helper class to log to file (possibly mirroring to stderr)
     logger = FileLogger('somefile.txt')
     logger = FileLogger('somefile.txt', mirror=True)
     logger('somemessage')
     logger('somemessage: %s %.2f', 'value', 2.5)
  """
  
  def __init__(self, fn, mirror=False):
    self.fn = fn
    self.f = open(fn, 'w')
    self.mirror = mirror
    
  def __call__(self, s, *args):
    """Either ('asdf %f', 5) or (val1, val2, val3, ...)"""
    if (isinstance(s, str) or isinstance(s, bytes)) and '%' in s :
      formatted_s = s % args
    else:
      toks = [s]+list(args)
      formatted_s = ', '.join(str(s) for s in toks)
      
    self.f.write(formatted_s+'\n')
    self.f.flush()
    if self.mirror:
      print(formatted_s)

  def __del__(self):
    self.f.close()

    
class TensorboardLogger:
  """Helper class to log to single tensorboard writer from multiple places.
   logger = u.TensorboardLogger('/efs/runs/somedirectory')
   logger = u.get_last_logger()  # gets last logger created
   logger('svd_time', 5)  # records "svd_time" stat at 5
   logger.next_step()     # advances step counter, flushes stats
   logger.set_step(5)     # sets step counter to 5
  """
  
  def __init__(self, logdir, step=0, flush_secs=1):
    # TODO: do nothing for default run
    
    global global_last_logger
    assert global_last_logger is None
    self.logdir = logdir
    print("Creating filewriter")
    self.summary_writer = tf.summary.FileWriter(logdir,
                                                graph=tf.get_default_graph(),
                                                flush_secs=flush_secs)
    self.step = step
    self.summary = tf.Summary()
    self.last_timestamp = time.perf_counter()
    global_last_logger = self

  def __call__(self, *args):
    assert len(args)%2 == 0
    for (tag, value) in u.chunks(args, 2):
      self.summary.value.add(tag=tag, simple_value=float(value))

  def set_step(self, step):
    self.next_step()
    self.step = step
    
  def next_step(self):
    new_timestamp = time.perf_counter()
    interval_ms = 1000*(new_timestamp - self.last_timestamp)
    self.summary.value.add(tag='time/step',
                           simple_value=interval_ms)
    self.last_timestamp = new_timestamp
    self.summary_writer.add_summary(self.summary, self.step)
    self.step+=1
    self.summary_writer.flush()
    
    self.summary = tf.Summary()

  def __del__(self):
    self.summary_writer.flush()


def run_worker():
  """Main worker loop."""

  config = load_config()
  cluster_spec = config.cluster_spec
  assert config.task_type == 'worker'
  ps_tasks = len(cluster_spec['ps'])
  assert ps_tasks >= 0

  # can only log to TensorBoard from single worker (TensorBoard logger)
  # see https://github.com/tensorflow/tensorboard/issues/1011
  # https://github.com/tensorflow/tensorboard/blob/1.6.0/README.md#tensorboard-is-showing-only-some-of-my-data-or-isnt-properly-updating
  #
  tb_logger = get_tb_logger() if config.task_id == 0 else None
  
  # log file/stderr from each worker (FileLogger)
  file_logger = FileLogger('log.txt', mirror=True)

  if config.task_id == 1:
    time.sleep(10)  # slow-down second worker for async testing

  # create worker graph
  worker_id = config.task_id
  global_params = []
  global_init_ops = []
  for i in range(args.ps):
    with tf.device("/job:ps/task:"+str(i)):
      param = tf.get_variable(name="params"+str(i),
                              shape=[sharded_params_size],
                              dtype=dtype,
                              initializer=tf.ones_initializer)
      global_params.append(param)
      global_init_ops.append(param.initializer)

  fetch_ops = []
  push_ops = []
  local_init_ops = []
  with tf.device("/job:worker/task:"+str(worker_id)):
    for i in range(args.ps):
      local_params = tf.get_variable(name="local"+str(i),
                                     shape=[sharded_params_size],
                                     dtype=dtype,
                                     initializer=tf.ones_initializer)
      fetch_ops.append(local_params.assign(global_params[i]).op)
      one =  tf.ones((), dtype=dtype)
      grads = tf.fill([sharded_params_size], one)
      push_ops.append(global_params[i].assign_add(grads).op)
      local_init_ops.append(local_params.initializer)
      local_val_op = local_params[0]  # entry of last ps shard

  def are_variables_initialized(vars):
    """op which is True iff all vars are initialized."""
    result = True
    for var in vars:
      result = tf.logical_and(tf.is_variable_initialized(var), result)
    return result

  initialized_op = are_variables_initialized(global_params)
  
  # TODO: add retries for errors during server creation?
  # it can fail if assigned port is unavailable
  # check how estimator does it
  server = tf.train.Server(cluster_spec, config=session_config(),
                           job_name=config.task_type,
                           task_index=config.task_id)

  # follow logic in prepare_session
  # https://github.com/tensorflow/tensorflow/blob/22586bdf900640217deac6dc826054bc6e785518/tensorflow/python/training/session_manager.py#L71
  def create_session():
    is_initialized = False
    while not is_initialized:
      try:
        sess = tf.InteractiveSession(server.target, config=session_config())
        sessrun(local_init_ops)
        if config.task_id == 0:
          sessrun(global_init_ops)
          
        is_initialized = sessrun(initialized_op)
      except Exception as e:
        print("Initialization/init check failed with %s, retrying" %(e,))
        
      print(("Model not initialized, "
             "retrying in %.1f seconds" %(RETRY_DELAY_SEC,)))
      time.sleep(RETRY_DELAY_SEC)
    return sess
    
  # TODO: check for failures in creating session?
  sess = create_session()
  
  last_local_val = 0
  last_local_val_ts = 0
  global_rate = 0      # measures how fast global param value changes
  
  for step in range(args.iters):
    start_time = time.time()
    sess_run_succeeded = False
    while not sess_run_succeeded:
      try:
        with timeit('worker_fetch'):
          sessrun(fetch_ops)    # ps -> worker tf
        with timeit('worker_push'):
          sessrun(push_ops)
        local_val = sessrun(local_val_op)
        sess_run_succeeded = True
        
      # Exception when ps restarts, need to recreate session
      except Exception as e:  
        print(("sess run failed with %s, "
               "retrying in %.1f seconds" %(e, RETRY_DELAY_SEC,)))
        time.sleep(RETRY_DELAY_SEC)
        sess = create_session()

    elapsed_time = time.time() - start_time  # todo: replace with perf_counter?
    local_rate = args.size_mb/elapsed_time
    
    elapsed_time_global = time.time() - last_local_val_ts
    last_local_val_ts = time.time()
    
    if last_local_val:
      global_rate = args.size_mb*(local_val - last_local_val)/elapsed_time_global
    last_local_val = local_val
    
    file_logger('step time: %8.2f ms, local rate: %8.2f MB/s, global rate: %8.2f MB/s', 1000*elapsed_time, local_rate, global_rate)

    if tb_logger:
      tb_logger('local_rate', local_rate)
      tb_logger('global_rate', global_rate)
      tb_logger('worker-'+str(config.task_id), local_val)
      tb_logger.next_step()


# Replacement of estimators.run_config.ClusterConfig that works with sparse
# cluster config. Default cluster config doesn't supported it
# https://github.com/tensorflow/tensorflow/issues/14502

class MyClusterConfig:
  def __init__(self):
    self.task_id = -1
    self.task_type = "asdf"
    self.cluster_spec = {"asdf":"asdf"}

  def __str__(self):
    return self.__dict__.__str__()

def load_config():
  """Loads ClusterConfig object from envrionment variable containing pickled
  sparse cluster config dict in dictionary format as below
  # {"task": {"index": 0, "type": "worker"},
     "cluster": {"worker": {0: "localhost:24724"}, "ps": ["localhost:15960"]}}
  """
  config = MyClusterConfig()
  config_dict = pickle.loads(base64.b16decode(os.environ["TF_PICKLE_BASE16"]))
  config.task_type = config_dict["task"]["type"]
  config.task_id = config_dict["task"]["index"]
  config.cluster_spec = config_dict["cluster"]
  return config

def run_ps():
  config = load_config()
  
  assert config.task_type == 'ps'
  params = make_params()
  
  print("Starting server with target %s"%(config.cluster_spec[config.task_type][config.task_id]))
  server = tf.train.Server(config.cluster_spec, config=session_config(),
                           job_name=config.task_type,
                           task_index=config.task_id)

  # doing init run from ps master fails with
  # sess run failed with No worker known as /job:worker/replica:0/task:1
  #      [[Node: Fill_S3 = _Recv[client_terminated=false, recv_device="/job:ps/replica:0/task:0/device:CPU:0", send_device="/job:worker/replica:0/task:1/device:CPU:0", send_device_incarnation=7403937842608207616, tensor_name="edge_3_Fill", tensor_type=DT_INT32, _device="/job:ps/replica:0/task:0/device:CPU:0"]()]], retrying in 5.0 seconds

  # todo: replace with dequeue for graceful shutdown (https://stackoverflow.com/questions/39810356/shut-down-server-in-tensorflow/40186129#40186129)
  # todo: done_queue from sharded_ps_benchmark
  # done_queue = create_done_queue(0)
  time.sleep(365*24*3600)

def _get_master():
  """Returns the appropriate string for local grpc TensorFlow master.
  For compat with server.target, return bytes instead of string.

  The address is derived from server spec, so it may not match the value
  returned by server.target stared locally (server.target can be localhost:129,
  server spec for matching task would be 127.0.0.1:129)
  """

  def _get_master_str():
    config = load_config()
    task_type = config.task_type
    task_id = config.task_id
    cluster_spec = config.cluster_spec

    if not cluster_spec:
      return ''

    # If there is only one node in the cluster, do things locally.
    jobs = cluster_spec.jobs
    if len(jobs) == 1 and len(cluster_spec.job_tasks(jobs[0])) == 1:
      return ''

    # Lookup the master in cluster_spec using task_type and task_id,
    # if possible.
    if task_type:
      if task_type not in jobs:
        raise ValueError(
            '%s is not a valid task_type in the cluster_spec:\n'
            '%s\n\n'
            'Note that these values may be coming from the TF_CONFIG environment '
            'variable.' % (task_type, cluster_spec))
      addresses = cluster_spec.job_tasks(task_type)
      if task_id >= len(addresses) or task_id < 0:
        raise ValueError(
            '%d is not a valid task_id for task_type %s in the '
            'cluster_spec:\n'
            '%s\n\n'
            'Note that these value may be coming from the TF_CONFIG environment '
            'variable.' % (task_id, task_type, cluster_spec))
      return 'grpc://' + addresses[task_id]

    # For backwards compatibility, we return empty string if task_type was
    # not set (task_type did not previously exist).
    return ''

  return _get_master_str().encode('ascii')

def main():
  config = load_config()

  logdir = args.logdir
  print("Logging to "+logdir)
  os.system('mkdir -p '+logdir)
    
  if  config.task_type == 'worker':
    if config.task_id == 0:
      TensorboardLogger(logdir)
    run_worker()
  elif config.task_type == 'ps':
    # don't start logger, tensorboard gets confused by multiple event files
    # https://github.com/tensorflow/tensorboard/issues/1011
    run_ps()
  else:
    assert False, "Unknown task type "+str(config.task_type)
    


if __name__=='__main__':
  main()
