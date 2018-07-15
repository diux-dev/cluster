#!/usr/bin/env python
# master command:
# python launch_nv.py --name 4gpu_distributed --instance-type p3.16xlarge --num-tasks 4 --zone us-west-2c --use-placement-group 1 --spot --attach-volume imagenet_high_perf --params x4_args --ami-name="Deep Learning AMI (Ubuntu) Version 11.0"

# 8 gpu training
# python launch_nv.py --name 8gpu_zoneb --instance-type p3.16xlarge --num-tasks 8 --zone us-west-2b --placement-group pytorch_cluster_b --ami-name pytorch.imagenet.source.v2
# spot command (attaching ebs volume and using default amazon ami):
# python launch_nv.py --name 8gpu_distributed --instance-type p3.16xlarge --num-tasks 8 --zone us-west-2c --use-placement-group 1 --spot --attach-volume imagenet_high_perf --params x8_args --ami-name="Deep Learning AMI (Ubuntu) Version 11.0"

# virginia 8 machine run
# export AWS_DEFAULT_REGION=us-east-1
# python launch_nv.py --name yaro-friday-8 --num-tasks 8 --zone us-east-1c --params x8_args

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
parser.add_argument('--ami-name', type=str,
                    default='pytorch.imagenet.source.v2',
                    help="name of AMI to use")
parser.add_argument('--placement-group', type=str, default='',
                     help=("name of placement group to use (depecated, name "
                           "is automatically picked"))
parser.add_argument('--use-placement-group', type=int, default=1,
                     help="whether to use placement group")
parser.add_argument('--spot', action='store_true', 
                    help='launch using spot requests')
parser.add_argument('--name', type=str, default='pytorch',
                     help=("name of the current run, this determines placement "
                           "group name, instance names and EFS logging "
                           "directory."))
parser.add_argument('--job-name', type=str, default='distributed',
                     help="name of the worker job (deprecated, use --name)")
parser.add_argument('--instance-type', type=str, default='p3.16xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-west-2a',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
# todo: rename to "num-machines", num-tasks is confusing since multiple pytorch
# per machine
parser.add_argument('--num-tasks', type=int, default=1,
                    help='number of instances to create')
parser.add_argument('--install-script', type=str, default='',
                    help='location of script to install')
parser.add_argument('--attach-volume', type=str, default=None,
                    help='tag name of ebs volume to attach')
parser.add_argument('--volume-offset', type=int, default=0,
                    help='Offset number for vollume attachment. If running multiple jobs')
parser.add_argument('--mount-efs', action='store_true',
                    help='Mount efs. For loading imagenet')
parser.add_argument('--params', type=str, default="x4_args",
                    help='args to use, see "params = " line')
args = parser.parse_args()


# Current best settings

# Current benchmark for 1x p3
x_args = [
  '--lr-sched', '0.14,0.47,0.78,0.95',
  '--epochs', 45,
  '--lr', 0.4,
  '--dist-url', 'file:///home/ubuntu/data/file.sync', # single instances are faster with file sync
  '--init-bn0',
  '--batch-size', 192
]
x_args_128 = [
  '--lr-sched', '0.14,0.47,0.78,0.95',
  '--epochs', 45,
  '--lr', 0.4,
  '--dist-url', 'file:///home/ubuntu/data/file.sync', # single instances are faster with file sync
  '--init-bn0',
  '--batch-size', 128
]

# Current benchmark for 4x p3's - without Aspect Ratio Validatoin
x4_args = [
  '--lr-sched', '0.14,0.47,0.78,0.95',
  '--epochs', 50,
  '--lr', 0.4 * 4,
  '--init-bn0',
  '--batch-size', 192
]
# Current benchmark for 4x p3's - with Aspect Ratio Validatoin
x4ar_args = [
  '--lr-sched', '0.14,0.43,0.73,0.94',
  '--resize-sched', '0.35,0.88',
  '--epochs', 40,
  '--lr', 0.35 * 4,
  '--init-bn0',
  '--batch-size', 192,
  '--val-ar'
]
# Current benchmark for 8x p3's - without Aspect Ratio Validatoin
x8_args = [
  '--lr-sched', '0.14,0.47,0.78,0.95',
  '--epochs', 55,
  '--lr', 0.3 * 8,
  '--init-bn0',
  '--batch-size', 128
]
# Current benchmark for 8x p3's - with Aspect Ratio Validatoin
x8ar_args = [
  '--lr-sched', '0.14,0.43,0.75,0.94',
  '--resize-sched', '0.35,0.88',
  '--epochs', 40,
  '--lr', 0.25 * 8,
  '--init-bn0',
  '--batch-size', 128,
  '--val-ar'
]

# Current benchmark for 8x p3's - with Aspect Ratio Validatoin and more epochs
x8ar_args_forever = [
  '--lr-sched', '0.14,0.43,0.73,0.94',
  '--resize-sched', '0.35,0.88',
  '--epochs', 400,
  '--lr', 0.25 * 8,
  '--init-bn0',
  '--batch-size', 128,
  '--val-ar'
]

# Current benchmark for 16x p3's - with Aspect Ratio Validatoin
# python launch_nv.py --name yaro-friday-16 --num-tasks 16 --zone us-east-1c --params x16ar_args

x16ar_args = [
  '--lr-sched', '0.14,0.47,0.78,0.95',
  '--resize-sched', '0.35,0.88',
  '--epochs', 100,
  '--lr', 0.25 * 8,
  '--init-bn0',
  '--batch-size', 64,
  '--val-ar'
]

# changing set for testing
yaro = [
  '--lr-sched', '0.14,0.47,0.78,0.95',
  '--epochs', 45,
  '--lr', 0.4,
  '--dist-url', 'file:///home/ubuntu/data/file.sync', # single instances are faster with file sync
  '--init-bn0',
  '--batch-size', 64
]

def main():
  run = aws_backend.make_run(args.name, ami=args.ami,
                             ami_name=args.ami_name,
                             availability_zone=args.zone,
                             linux_type=args.linux_type,
                             skip_efs_mount=(not args.mount_efs))
  job = create_job(run, 'worker', args.num_tasks)

  # Define custom params for training or use a preset above
  # TODO: move "save_tag" into command-line parameter
  params = eval(args.params)
  start_training(job, params, save_tag=args.name)


def create_job(run, job_name, num_tasks):
  install_script = ''
  if args.install_script:
    with open(args.install_script, 'r') as f:
      install_script = f.read()
  
  ebs = get_ebs_settings(use_iops=(args.attach_volume is None))
  if args.placement_group:
    print("Warning, placement_group is deprecated, use --use-placement-group 1 for automatically picked placement group (same as run name).")
    placement_group_name = args.placement_group
  if args.use_placement_group:
    placement_group_name = args.name
  else:
    placement_group_name = ''
    
  job = run.make_job(job_name, num_tasks=num_tasks, ebs=ebs, instance_type=args.instance_type, install_script=install_script, placement_group=placement_group_name, use_spot=args.spot)
  job.wait_until_ready()
  print(job.connect_instructions)

  if args.attach_volume: mount_volume_data(job, tag=args.attach_volume, offset=args.volume_offset)

  job.run_async_join('killall python || echo failed')  # kill previous run
  job.run_async_join('source activate pytorch_p36')
  # job.run_async_join('source activate pytorch_source', ignore_errors=True) # currently a bug in latest pytorch
  job.run_async_join('ulimit -n 9000') # to prevent tcp too many files open error

  # upload files
  job.upload_async('training/resnet.py')
  job.upload_async('training/fp16util.py')
  job.upload_async('training/autoaugment.py')
  job.upload_async('training/dataloader.py')
  job.upload_async('training/dataloader_performance.py')
  job.upload_async('training/train_imagenet_nv.py')

  # Sometimes get SSH session not active or "connection reset by peer"
  # bad internet?

  setup_complete = [t.file_exists('/tmp/nv_setup_complete') for t in job.tasks]
  if not all(setup_complete):
    job.upload_async('setup/setup_env_nv.sh')
    job.run_async_join('chmod +x setup_env_nv.sh')
    job.run_async_join('bash setup_env_nv.sh', max_wait_sec=60*60, check_interval=5)

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
  base_save_dir = '~/data/training/nv'
  datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
  # base_save_dir = f'/efs/training/nv' # TODO: save to efs instead
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
    cmd = f'{nccl_args} python -m torch.distributed.launch {dist_args} train_imagenet_nv.py {training_args}'
    t.run(f'echo {cmd} > {save_dir}/script.log')
    task_cmds.append(cmd)

  for t,cmd in zip(job.tasks, task_cmds):
    t.run_async(cmd)

if __name__=='__main__':
  main()
