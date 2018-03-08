#!/usr/bin/env python

"""Usage

# launch locally (in tmux sessions)
python launch_tf_adder.py --workers=2 --cluster=local

# launch on AWS using given instance type
python launch_tf_adder.py --workers=2 --cluster=aws --name=psbench1 --instance=c5.18xlarge


"""
from collections import OrderedDict
from collections import defaultdict
import argparse
import base64
import json
import os
import pickle
import sys
import time

import boto3

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util as u

# Deep learning AMI v5
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue
ami_dict_ubuntu = {
  "us-east-1": "ami-7336d50e",
  "us-east-2": "ami-eb596e8e",
  "us-west-2": "ami-c27af5ba",
}

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='',
                     help="name of AMI to use ")
parser.add_argument('--name', type=str, default='tfbench',
                     help="name of the current run")
parser.add_argument('--instance', type=str, default='c5.4xlarge', # c5.18xlarge
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-east-1a',
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
parser.add_argument('--profile', type=int, default=0,
                    help='dump timelines')
parser.add_argument('--placement', type=int, default=1,
                    help='whether or not to use placement group')

args = parser.parse_args()



def launch(backend, install_script='', init_cmd=''):
  if args.placement:
    placement_group = args.name
  else:
    placement_group = ''
    
  if backend.__name__ == 'aws_backend':
    ami = ami_dict_ubuntu[u.get_region()]
    run = backend.make_run(args.name, user_data=install_script,
                           ami=ami, availability_zone=args.zone)
    worker_job = run.make_job('worker', num_tasks=args.workers,
                              instance_type=args.instance,
                              placement_group=placement_group) # 
    ps_job = run.make_job('ps', num_tasks=args.ps,
                          instance_type=args.instance,
                          placement_group=placement_group)
    tb_job = run.make_job('tb', instance_type='t2.large')
  else:  # local mode
    run = backend.make_run(args.name, install_script=install_script)
    worker_job = run.make_job('worker', args.workers) # 
    ps_job = run.make_job('ps', args.ps)
    tb_job = run.make_job('tb')

  for job in run.jobs:
    job.wait_until_ready()

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

  worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
  ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
  cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}
  
  # Launch tensorflow tasks.
  run.run(init_cmd)
  tf_cmd = "python tf_adder.py --logdir={logdir} --profile={profile}".format(logdir=run.logdir, profile=args.profile)
  
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

  # todo: for local runs need to do task.port because multiple tb's
  # 6006 is hardwired because it's open through the security group
  tb_port = tb_job.public_port #6006
  tb_job.run("tensorboard --logdir={logdir} --port={port}".format(
    logdir=run.logdir, port=tb_port), sync=False)
  print("*"*80)
  print("See tensorboard at http://%s:%s"%(tb_job.public_ip, tb_port))
  print("*"*80)
  print(" "*80)

  print("Streaming log.txt of worker[0]")
  worker_job.tasks[0].stream_file('log.txt')

    
def main():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import tmux_backend
  import aws_backend

  install_script="""#!/bin/bash
source /home/ubuntu/anaconda3/bin/activate tensorflow_p36
pip install ray
echo 'INSTALLED ray' > /home/ubuntu/ray_installed.txt
"""

  if args.cluster == 'local':
    launch(tmux_backend, init_cmd='source activate cifar')
  elif args.cluster == 'aws':
    launch(aws_backend, init_cmd='source activate tensorflow_p36',
           install_script=install_script)
    
  else:
    print("Unknown cluster", args.cluster)
    # todo: create resources

if __name__=='__main__':
  main()
