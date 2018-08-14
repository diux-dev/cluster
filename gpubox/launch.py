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
parser.add_argument('--tf-benchmark', default=0,
                    help='launch simple TensorFlow addition benchmark')
parser.add_argument('--mode', default='jupyter',
                    help='either jupyter or tf-benchmark')
parser.add_argument('--password',
                    default='DefaultNotebookPasswordPleaseChange',
                    help='password to use for jupyter notebook')
parser.add_argument('--create-resources', type=int, default=1,
                    help='first-time run, create EFS/VPC/security groups/etc')
parser.add_argument('--spot', type=int, default=0,
                    help='use spot instances')
parser.add_argument('--internal-role', type=str, default='launcher',
                    help='internal flag for tf benchmark, launcher or worker')
args = parser.parse_args()


def main():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import tmux_backend
  import aws_backend
  import util as u

  u.maybe_create_resources(args)
  
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

  if args.mode == 'jupyter':
    # upload notebook config with provided password
    from notebook.auth import passwd
    sha = passwd(args.password)
    local_config_fn = f'{module_path}/jupyter_notebook_config.py'
    temp_config_fn = '/tmp/'+os.path.basename(local_config_fn)
    remote_config_fn = f'/home/ubuntu/.jupyter/{os.path.basename(local_config_fn)}'
    os.system(f'cp {local_config_fn} {temp_config_fn}')
    _replace_lines(temp_config_fn, 'c.NotebookApp.password',
                   f"c.NotebookApp.password = '{sha}'")
    job.upload(temp_config_fn, remote_config_fn)

    # upload sample notebook and start server
    job.run('mkdir -p /efs/notebooks')
    job.upload(f'{module_path}/sample.ipynb', '/efs/notebooks/sample.ipynb',
               dont_overwrite=True)
    job.run('cd /efs/notebooks')
    job.run_async('jupyter notebook')
    print(f'Jupyter notebook will be at http://{job.public_ip}:8888')
  elif args.mode == 'tf-benchmark':
    job.run('source activate tensorflow_p36')
    job.upload(__file__)
    job.run('killall python || echo pass')  # kill previous run
    job.run_async('python launch.py --internal-role=worker')
  else:
    assert False, "Unknown --mode, must be jupyter or tf-benchmark."

def _replace_lines(fn, startswith, new_line):
  """Replace lines starting with starts_with in fn with new_line."""
  new_lines = []
  for line in open(fn):
    if line.startswith(startswith):
      new_lines.append(new_line)
    else:
      new_lines.append(line)
  with open(fn, 'w') as f:
    f.write('\n'.join(new_lines))
  

def worker():
  """tf worker script that runs on AWS machine. Adds vectors of ones forever,
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

def main_root():
  if args.internal_role == "launcher":
    main()
  elif args.internal_role == "worker":
    worker()
  else:
    assert False, "Unknown role "+FLAGS.role

if __name__=='__main__':
  main_root()
