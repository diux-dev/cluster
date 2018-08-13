#!/usr/bin/env python
#
# Launch a single GPU instance with Amazon Deep Learning AMI
# export AWS_DEFAULT_REGION=us-west-2
# export ZONE=us-west-2c
# ./launch.py

from collections import OrderedDict
import argparse
import os
import sys
import time

import boto3

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='gpubox',
                     help="name of the current run")
parser.add_argument('--ami-name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 12.0',
                    help="name of AMI to use ")
parser.add_argument('--instance-type', type=str, default='g3.4xlarge',
                     help="type of instance")
parser.add_argument('--benchmark', default=0,
                    help='launch simple addition benchmark')
parser.add_argument('--jupyter', default=0,
                    help='launch jupyter notebook')
parser.add_argument('--jupyter-password',
                    default='DefaultNexusPasswordPleaseChange',
                    help='password to use for jupyter notebook')
parser.add_argument('--role', type=str, default='launcher',
                    help='internal flag, launcher or worker')
parser.add_argument('--spot', type=int, default=0,
                    help='use spot instances')
args = parser.parse_args()


def launcher():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import tmux_backend
  import aws_backend
  import create_resources as create_resources_lib
  import util as u

  create_resources_lib.create_resources()
  run = aws_backend.make_run(args.name, ami_name=args.ami_name)
  job = run.make_job('worker', instance_type=args.instance_type,
                     use_spot=args.spot)
  job.wait_until_ready()

  print("Job ready for connection, run the following:")
  print("../connect "+args.name)
  print("Alternatively run")
  print(job.connect_instructions)
  print()
  print()
  print()
  print()

  if args.jupyter:
    job.upload('jupyter_notebook_config.py') # 2 step upload since don't know ~
    run_tmux_async(sess, 'cp jupyter_notebook_config.py ~/.jupyter')
    run_tmux_async(sess, 'mkdir -p /efs/notebooks')
    
  if args.benchmark:
    job.run('source activate tensorflow_p36')
    job.upload(__file__)
    job.run('killall python || echo pass')  # kill previous run
    job.run_async('python launch.py --role=worker')

def worker():
  """Worker script that runs on AWS machine. Adds vectors of ones forever,
  prints MB/s."""

  import tensorflow as tf
  
  def session_config():
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
    config = tf.ConfigProto(
      graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
    config.operation_timeout_in_ms = 10*1000  # abort after 10 seconds
    return config

  iters_per_step = 10
  data_mb = 128
  params_size = 250*1000*data_mb # 1MB is 250k floats
  dtype=tf.float32
  val = tf.ones((), dtype=dtype)
  vals = tf.fill([params_size], val)
  params = tf.Variable(vals)
  update = params.assign_add(vals)
  
  sess = tf.Session(config=session_config())
  sess.run(params.initializer)
  
  while True:
    start_time = time.perf_counter()
    for i in range(iters_per_step):
      sess.run(update.op)

    elapsed_time = time.perf_counter() - start_time
    rate = float(iters_per_step)*data_mb/elapsed_time
    print('%.2f MB/s'%(rate,))    

def main():
  if args.role == "launcher":
    launcher()
  elif args.role == "worker":
    worker()
  else:
    assert False, "Unknown role "+FLAGS.role

if __name__=='__main__':
  main()
