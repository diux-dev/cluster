#!/usr/bin/env python
# numpy01 image, see environment-numpy.org for construction
# (DL AMI v 3.0 based)
#
# us-east-1 AMIs
# numpy00: ami-f9d6dc83
# numpy01: ami-5b524f21


# master command:
# python launch_fastai.py --instance-type p3.16xlarge --num-tasks 4 --job-name cluster_4_region_c --zone us-west-2c --ami ami-53c8822b --placement-group pytorch_cluster_c --spot --attach-volume imagenet_high_perf

from collections import OrderedDict
import argparse
import os
import sys
import time
import collections
import boto3
import datetime
import threading


module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util
import aws_backend
from launch_utils import *
util.install_pdb_handler()

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='',
                     help="id of AMI to use")
parser.add_argument('--ami_name', type=str,
                    default='pytorch.imagenet.source.v2',
                    help="name of AMI to use")
parser.add_argument('--placement-group', type=str, default='',
                     help="name of the current run")
parser.add_argument('--name', type=str, default='pytorch',
                     help="name of the current run")
parser.add_argument('--job-name', type=str, default='distributed',
                     help="name of the jobs to run")
parser.add_argument('--instance-type', type=str, default='p3.2xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-west-2a',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
parser.add_argument('--num-tasks', type=int, default=1,
                    help='number of instances to create')
parser.add_argument('--install-script', type=str, default='',
                    help='location of script to install')
parser.add_argument('--attach-volume', type=str, default=None,
                    help='tag name of ebs volume to attach')
parser.add_argument('--volume-offset', type=int, default=0,
                    help='Offset number for vollume attachment. If running multiple jobs')
parser.add_argument('--spot', action='store_true', 
                    help='launch using spot requests')
parser.add_argument('--mount-efs', action='store_true',
                    help='Mount efs. For loading imagenet')
parser.add_argument('--params', type=str, default="x_args",
                    help='args to use, see "params = " line')
args = parser.parse_args()


# Current best settings

# Current benchmark for 1x p3
x_args = [
  # '--lr-sched', '0.9,0.47,0.78,0.95',
  '--epochs', 45,
  '--lr', 0.4,
  '--dist-url', 'file:///home/ubuntu/data/file.sync', # single instances are faster with file sync
  '--batch-size', 192
]

def main():
  run = aws_backend.make_run(args.name, ami=args.ami,
                             ami_name=args.ami_name,
                             availability_zone=args.zone,
                             linux_type=args.linux_type,
                             skip_efs_mount=(not args.mount_efs))
  job = create_job(run, args.job_name, args.num_tasks)

  # Define custom params for training or use a preset above
  params = eval(args.params)
  start_training(job, params, save_tag='testing_refactor',)


def create_job(run, job_name, num_tasks):
  install_script = ''
  if args.install_script:
    with open(args.install_script, 'r') as f:
      install_script = f.read()
  
  ebs = get_ebs_settings(use_iops=(args.attach_volume is None))
  job = run.make_job(job_name, num_tasks=num_tasks, ebs=ebs, instance_type=args.instance_type, install_script=install_script, placement_group=args.placement_group, use_spot=args.spot)
  job.wait_until_ready()
  print(job.connect_instructions)

  if args.attach_volume: mount_volume_data(job, tag=args.attach_volume, offset=args.volume_offset)

  job.run_async_join('killall python || echo failed')  # kill previous run
  job.run_async_join('ulimit -n 9000') # to prevent tcp too many files open error

  # upload files
  job.upload_async('training/resnet.py')
  job.upload_async('training/train_imagenet_fastai.py')

  # setup machines
  # TODO: file_exists check below complains...need to make sure ssh sessions
  # are alive.
  #  paramiko.ssh_exception.SSHException: SSH session not active

  # setup machines
  setup_complete = [t.file_exists('/tmp/fastai_setup_complete') for t in job.tasks]
  if not all(setup_complete):
    job.run_async_join('conda create -n fastai -y', check_interval=5) # (AS) WARNING remember to revert back 
    job.run('source activate fastai') # (AS) WARNING remember to revert back 
    job.upload('setup/setup_env_fastai.sh')
    job.run('chmod +x setup_env_fastai.sh')
    job.run_async_join('bash setup_env_fastai.sh', max_wait_sec=60*60, check_interval=5)

  job.run('source activate fastai') # (AS) WARNING remember to revert back 

  return job

def start_training(job, params, save_tag):
  num_tasks = len(job.tasks)
  instance_0 = job.tasks[0].instance
  world_0_ip = instance_0.private_ip_address
  num_gpus = get_gpu_count(instance_0)
  port = '6006' # 6006, 6007, 6008, 8890, 6379
  world_size = num_gpus * num_tasks

  # Use NCCL rings for faster network throughput
  nccl_args = get_nccl_args(num_tasks, num_gpus)

  # Create save directory
  base_save_dir = '~/data/training/fastai'
  datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
  # base_save_dir = f'/efs/training/fastai' # TODO: save to efs instead
  save_dir = f'{base_save_dir}/{datestr}-{save_tag}-w{world_size}'
  job.run_async_join(f'mkdir {save_dir} -p')

  # Training script args
  default_params = [
    '~/data/imagenet',
    '--save-dir', save_dir,
    '--fp16',
    '--loss-scale', 512,
    '--world-size', world_size,
    '--distributed'
  ]
  training_args = default_params + params
  training_args = ' '.join(map(str, training_args))

  # Run tasks
  task_cmds = []
  for i,t in enumerate(job.tasks):
    dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
    cmd = f'{nccl_args} python -m torch.distributed.launch {dist_args} train_imagenet_fastai.py {training_args}'
    t.run(f'echo {cmd} > {save_dir}/script.log')
    task_cmds.append(cmd)

  for t,cmd in zip(job.tasks, task_cmds):
    t.run_async(cmd)

if __name__=='__main__':
  main()
