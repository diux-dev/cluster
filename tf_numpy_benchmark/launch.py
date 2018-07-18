#!/usr/bin/env python
#
# Launch a single GPU instance with Amazon Deep Learning AMI
# ./launch.py --instance-type=g3.4xlarge --zone=us-east-1f
#
#   Default AMI used:
#
# https://aws.amazon.com/blogs/ai/new-aws-deep-learning-amis-for-machine-learning-practitioners/
# 

from collections import OrderedDict
import argparse
import os
import sys
import time

import boto3

# Deep learning AMI v11
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue

ami_dict_ubuntu = {
  "us-east-1": "ami-6356761c",
  "us-east-2": "ami-a2ecd4c7",
  "us-west-2": "ami-c47c28bc",
}

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='',
                     help="name of AMI to use ")
parser.add_argument('--name', type=str, default='tf_numpy_benchmark',
                     help="name of the current run")
parser.add_argument('--instance-type', type=str, default='c5.18xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-east-1c',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
args = parser.parse_args()

# TODO: if linux type is specified, ignore ami?

# TODO: get rid of this?
INSTALL_SCRIPT_UBUNTU="""
python --version
sudo mkdir -p /efs
sudo chmod 777 /efs
"""

INSTALL_SCRIPT_AMAZON="""
python --version
sudo mkdir -p /efs
sudo chmod 777 /efs
"""

def main():
  if args.role == "launcher":
    launcher()
  elif args.role == "worker":
    worker()
  else:
    assert False, "Unknown role "+FLAGS.role


def launcher():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import tmux_backend
  import aws_backend
  import create_resources as create_resources_lib
  import util as u

  create_resources_lib.create_resources()
  region = u.get_region()
  assert args.zone.startswith(region), "Availability zone %s must be in default region %s. Default region is taken from environment variable AWS_DEFAULT_REGION" %(args.zone, region)

  if args.linux_type == 'ubuntu':
    install_script = INSTALL_SCRIPT_UBUNTU
    ami_dict = ami_dict_ubuntu
  elif args.linux_type == 'amazon':
    install_script = INSTALL_SCRIPT_AMAZON
    ami_dict = ami_dict_amazon
  else:
    assert False, "Unknown linux type "+args.linux_type

  if args.ami:
    print("Warning, using provided AMI, make sure that --linux-type argument "
          "is set correctly")
    ami = args.ami
  else:
    assert region in ami_dict, "Define proper AMI mapping for this region."
    ami = ami_dict[region]

  # TODO: add API to create jobs with default run
  run = aws_backend.make_run(args.name, install_script=install_script,
                             ami=ami, availability_zone=args.zone,
                             linux_type=args.linux_type)
  job = run.make_job('worker',instance_type=args.instance_type)
  
  job.wait_until_ready()

  print("Job ready for connection, run the following:")
  print("../connect "+args.name)
  print("Alternatively run")
  print(job.connect_instructions)
  print()
  print()
  print()
  print()
  
  job.run('source activate tensorflow_p36')
  job.run('pip install cython')
  job.run('pip install ray')
  # below can fail on
  # E: Could not get lock /var/lib/dpkg/lock - open (11: Resource temporarily unavailable)
  job.run('sudo apt install htop')

  job.run('yes | sudo apt-get install google-perftools')
  job.run('export LD_PRELOAD="/usr/lib/libtcmalloc.so.4"')

  job.upload(__file__)
  job.upload('tf_numpy_benchmark.py')
  job.run('killall python || echo failed')  # kill previous run
  job.run('python tf_numpy_benchmark.py')

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
  
  for i in range(10):
    start_time = time.perf_counter()
    for i in range(iters_per_step):
      sess.run(update.op)

    elapsed_time = time.perf_counter() - start_time
    rate = float(iters_per_step)*data_mb/elapsed_time
    print('%.2f MB/s'%(rate,))    

if __name__=='__main__':
  main()
# testing things
