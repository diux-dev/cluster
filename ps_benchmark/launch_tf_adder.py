#!/usr/bin/env python
#
# Launch a single GPU instance with Amazon Deep Learning AMI
# ./launch.py --instance-type=g3.4xlarge --zone=us-east-1f
#
# Default AMI used:
#
# https://aws.amazon.com/blogs/ai/new-aws-deep-learning-amis-for-machine-learning-practitioners/
# 
# Ubuntu Conda based Amazon Deep Learning AMI
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue
# US East (N. Virginia)	ami-9ba7c4e1
# US East (Ohio)        ami-69f6df0c
# US West (Oregon)	ami-3b6bce43
# EU (Ireland)	        ami-5bf34b22
# Asia Pacific (Sydney)	ami-4fa3542d
# Asia Pacific (Seoul)	ami-7d872113
# Asia Pacific (Tokyo)	ami-90b432f6

# Amazon Linux version:
# https://aws.amazon.com/marketplace/fulfillment?productId=f3afce45-648d-47d7-9f6c-1ec273f7df70&ref_=dtl_psb_continue
# US East (N. Virginia)        ami-3a533040
# US East (Ohio)               ami-72f4dd17
# US West (Oregon)             ami-5c60c524
# EU (Frankfurt)               ami-88aa23e7
# EU (Ireland)                 ami-70fe4609
# Asia Pacific (Singapore)     ami-0798fc7b
# Asia Pacific (Sydney)	       ami-38a5525a
# Asia Pacific (Seoul)	       ami-a4b91fca
# Asia Pacific (Tokyo)	       ami-98ad2bfe
# Asia Pacific (Mumbai)	       ami-6ce8a103

from collections import OrderedDict
from collections import defaultdict
import argparse
import json
import os
import portpicker
import pickle
import base64
import sys
import time



import boto3


# map availability zones that contain given instance type
# TODO: this mapping is randomized between username on AWS side
# availability_mapping_us_east_1 = {'g3': ['us-east-1a', 'us-east-1b',
#                                          'us-east-1e', 'us-east-1c'],
#                                   'p2': ['us-east-1f'],
#                                   'p3': [us-east-1d, us-east-1c, us-east-1f]}
# availability_mapping_us_west_2 = {'g3': ['us-west-2a'],
#                                   'p2': ['us-west-2a', 'us-west-2b'],
#                                   'p3': ['us-west-2b', 'us-west-2c']}
# availability_mapping = {'us-east-1': availability_mapping_us_east_1,
#                         'us-west-2': availability_mapping_us_west_2}

# Deep Learning AMI v3
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue

ami_dict_ubuntu = {
    "us-west-2": "ami-ee48f796",
    "us-east-1": "ami-0a9fac70",
}
#ami_dict_amazon = {
#    "us-west-2": "ami-5c60c524",
#    "us-east-1": "ami-3a533040"
#}

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='',
                     help="name of AMI to use ")
parser.add_argument('--name', type=str, default='ps_benchmark',
                     help="name of the current run")
parser.add_argument('--instance', type=str, default='p2.xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-west-2a',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
parser.add_argument('--cluster', type=str, default='local',
                    help='local or aws')
parser.add_argument('--ps', type=int, default=1,
                    help='number of parameter server tasks')
parser.add_argument('--workers', type=int, default=1,
                    help='number of parameter server tasks')

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

def launch_local(backend, install_script):
  run = backend.make_run(args.name, install_script=install_script)
  worker_job = run.make_job('worker', args.workers) # 
  ps_job = run.make_job('ps', args.ps)
  tb_job = run.make_job('tb')

  run.upload('tf_adder.py')
  run.upload('../util.py')

  
  def tf_env_setup(task, dense_cluster_spec, task_spec):
    """Helper method to initialize clusterspec for a task."""

    task_type = task_spec['type']
    task_id = task_spec['index']

    # full cluster spec (needed for estimator)
    dense_cluster_config = {'cluster': dense_cluster_spec, 'task': task_spec}
    TF_CONFIG = json.dumps(dense_cluster_spec)
    task.run("export TF_CONFIG='%s'"%(TF_CONFIG,))

    # construct sparse cluster spec
    # every worker needs its own location
    sparse_cluster_spec = defaultdict(dict)
    host = dense_cluster_spec[task_type][task_id]
    sparse_cluster_spec[task_type][task_id] = host

    # gradient workers know about all ps workers
    if task_type == 'worker':
      sparse_cluster_spec['ps'] = dense_cluster_spec['ps']

    # ps workers know about all gradient workers
    if task_type == 'ps':
      sparse_cluster_spec['worker'] = dense_cluster_spec['worker']

    sparse_cluster_config = {'cluster': sparse_cluster_spec,
                             'task': task_spec}

    # sparse cluster spec
    pickle_string = pickle.dumps(sparse_cluster_config)
    pickle_string_encoded = base64.b16encode(pickle_string)
    pickle_string_encoded = pickle_string_encoded.decode('ascii')
    task.run("export TF_PICKLE_BASE16=%s"%(pickle_string_encoded,))

    task.run("source activate cifar")

  worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
  ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
  cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}
  
  # Launch tensorflow tasks.
  tf_cmd = "python tf_adder.py --logdir={logdir}".format(logdir=run.logdir)
  
  # ps tasks go first because tensorboard doesn't support multiple processes
  # creating events in same directory locally (only shows latest created
  # event file)
  for task in ps_job.tasks:
    task_spec = {'type': 'ps', 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run(tf_cmd+' --label='+task.job.name+':'+str(task.id), sync=False)

  for task in worker_job.tasks:
    task_spec = {'type': 'worker', 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run(tf_cmd+' --label='+task.job.name+':'+str(task.id), sync=False)

  tb_job.run("tensorboard --logdir={logdir} --port={port}".format(
    logdir=run.logdir, port=tb_job.port), sync=False)
  print("See tensorboard at http://%s:%s"%(tb_job.ip, tb_job.port))

    
def main():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import tmux_backend
  import aws_backend

  # todo: create resources
  launch_local(tmux_backend, '')

if __name__=='__main__':
  main()
