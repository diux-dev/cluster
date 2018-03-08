#!/usr/bin/env python

"""Usage

# launch locally (in tmux sessions)
python launch_ray_adder.py --workers=2 --cluster=local

# launch on AWS using given instance type
python launch_ray_adder.py --workers=2 --cluster=aws --instance=c5.18xlarge
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

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import tmux_backend
import aws_backend
import create_resources as create_resources_lib
import util as u

# Deep learning AMI v5
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue
ami_dict_ubuntu = {
  "us-east-1": "ami-7336d50e",
  "us-east-2": "ami-eb596e8e",
  "us-west-2": "ami-c27af5ba",
}

#LOCAL_CONDA_ENV='cifar'        # use this conda env when running locally
#REMOTE_CONDA_ENV='pytorch_p36' # use this conda env when running remotely

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--local-conda-env', default='cifar',
                    help='name of conda env to use when running locally')
parser.add_argument('--remote-conda-env', default='pytorch_p36',
                    help='name of conda env to use when running remotely')
parser.add_argument('--ami', type=str, default='',
                     help="name of AMI to use ")
parser.add_argument('--name', type=str, default='raybench',
                     help="name of the current run")
parser.add_argument('--instance', type=str, default='c5.large', # c5.18xlarge
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-east-1a',
                    help='which availability zone to use')
parser.add_argument('--cluster', type=str, default='local',
                    help='local or aws')
parser.add_argument('--ps', type=int, default=1,
                    help='number of parameter server tasks')
parser.add_argument('--workers', type=int, default=1,
                    help='number of parameter server tasks')
parser.add_argument('--placement', type=int, default=1,
                    help='whether or not to use placement group')
parser.add_argument("--size-mb", default=100, type=int,
                    help="size of data in MBs")


args = parser.parse_args()
DEFAULT_PORT = 6379  # default redis port


def launch(backend, install_script='', init_cmd=''):
  if args.placement:
    placement_group = args.name
  else:
    placement_group = ''

  num_tasks = args.workers+args.ps
  run_local = False if backend.__name__ == 'aws_backend' else True

  if run_local:
    run = backend.make_run(args.name, install_script=install_script)
    job = run.make_job('worker', num_tasks) 
  else:
    region = u.get_region()
    assert args.zone.startswith(region), "Your specified zone is %s but your region (from AWS_DEFAULT_REGION) is %s, please specify zone correctly, such as --zone=%sa" %(args.zone, region, region)
    create_resources_lib.create_resources()
    ami = ami_dict_ubuntu[u.get_region()]
    run = backend.make_run(args.name, user_data=install_script,
                           ami=ami, availability_zone=args.zone)
    job = run.make_job('worker', num_tasks=num_tasks,
                       instance_type=args.instance,
                       placement_group=placement_group) 

  for job in run.jobs:
    job.wait_until_ready()

  head_task = job.tasks[0]  # worker 0 is also the head node
  head_task.upload('ray_adder.py')
  head_task.upload('../util.py')  # just in case?
  
  # todo: use task.port instead of DEFAULT_PORT
  run.run(init_cmd)
  run.run('ray stop || echo "ignoring error"')

  # Ray start for head node. When running locally, specify more gpus since
  # all workers go on same machine
  ray_cmd = "ray start --head --redis-port=%d --num-workers=0"%(DEFAULT_PORT,)
  if run_local:
    ray_cmd+=' --num-gpus=10'
  else:
    ray_cmd+=' --num-gpus=1'
    
  head_task.run(ray_cmd)

  # Ray start command for leaf nodes
  if not run_local:
    ray_cmd = "ray start --redis-address %s:%d --num-gpus=1 --num-workers=0"%(head_task.ip, DEFAULT_PORT)
    for task in job.tasks[1:]:
      task.run(ray_cmd)

  # Client script
  client_cmd = 'python ray_adder.py --redis-address %s:%d --size-mb %d'%(head_task.ip, DEFAULT_PORT, args.size_mb)
  if not run_local:
    client_cmd+=' --enforce-different-ips=1'
  head_task.run('rm log.txt || echo nevermind')
  head_task.run(client_cmd, sync=False)

  log("Streaming log.txt of task[0]")
  job.tasks[0].stream_file('log.txt')


def log(message, *args):
  """Log to client console."""
  ts = u.current_timestamp()
  if args:
    message = message % args
  print(message)

    
def main():
  # todo: add "source deactivate" to fix https://github.com/conda/conda/issues/7007
  # timeout for slow pre-warming: https://github.com/ray-project/ray/issues/1682
  install_script="""#!/bin/bash
source /home/ubuntu/anaconda3/bin/deactivate
source /home/ubuntu/anaconda3/bin/activate pytorch_p36
pip install --default-timeout=100 -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.3.1-cp36-cp36m-manylinux1_x86_64.whl
# pre-warm caches
ray start --head
ray stop
python -c "import ray, torch"
"""

  # todo: document local conda environment better
  if args.cluster == 'local':
    launch(tmux_backend, init_cmd='source activate ' + args.local_conda_env)
  elif args.cluster == 'aws':
    launch(aws_backend, init_cmd='source activate ' + args.remote_conda_env,
           install_script=install_script)
    
  else:
    print("Unknown cluster", args.cluster)
    # todo: create resources

if __name__=='__main__':
  main()
