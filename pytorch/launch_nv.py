#!/usr/bin/env python
# numpy01 image, see environment-numpy.org for construction
# (DL AMI v 3.0 based)
#
# us-east-1 AMIs
# numpy00: ami-f9d6dc83
# numpy01: ami-5b524f21



# master command:
# python launch_nv.py --instance-type p3.16xlarge --num-tasks 4 --job-name cluster_4_region_c --zone us-west-2c --ami ami-6583d71d --placement-group pytorch_cluster_c

# 8 gpu training
# python launch_nv.py --instance-type p3.16xlarge --num-tasks 8 --job-name cluster_8_region_b --zone us-west-2b --placement-group pytorch_cluster_b --ami ami-6583d71d

# spot command:
# python launch_nv.py --instance-type p3.16xlarge --num-tasks 4 --job-name cluster_4_region_c_spot --zone us-west-2c --ami ami-6583d71d --placement-group pytorch_cluster_c --spot --attach-volume imagenet_high_perf
# you can use default aws provided ami

# python launch_nv.py --instance-type p3.16xlarge --num-tasks 8 --job-name cluster_8_region_c_spot --zone us-west-2c --placement-group pytorch_cluster_c --spot --attach-volume imagenet_high_perf

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
util.install_pdb_handler()

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='ami-e580c79d',
                     help="name of AMI to use ")
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
parser.add_argument('--attach-volume', type=str, default='',
                    help='tag name of ebs volume to attach')
parser.add_argument('--volume-offset', type=int, default=0,
                    help='Offset number for vollume attachment. If running multiple jobs')
parser.add_argument('--spot', action='store_true', 
                    help='launch using spot requests')
parser.add_argument('--mount-efs', action='store_true',
                    help='Mount efs. For loading imagenet')
args = parser.parse_args()


## Setup instance:
# ebs = None
ebs = [{
  'DeviceName': '/dev/sda1',
  'Ebs': {
    'VolumeSize': 500, 
    'DeleteOnTermination': True,
    'VolumeType': 'io1',
    'Iops': 14000
  }
}]

# For using external ebs (so we don't hit iops limit)
# ebs = [{
#   'DeviceName': '/dev/sda1',
#   'Ebs': {
#     'VolumeSize': 500, 
#     'DeleteOnTermination': True,
#     'VolumeType': 'gp2',
#     # 'Iops': 18000
#   }
# }]

def attach_instance_ebs(aws_instance, tag):
  ec2 = util.create_ec2_resource()
  v = list(ec2.volumes.filter(Filters=[{'Name':'tag:Name', 'Values':[tag]}]).all())
  assert(v)
  v = v[0]
  already_attached = v.attachments and v.attachments[0]['InstanceId'] == aws_instance.id
  if already_attached: return
  if v.state != 'available': 
    print('Detaching from current instance')
    v.detach_from_instance()
    time.sleep(7)
  try:
    v.attach_to_instance(InstanceId=aws_instance.id, Device='/dev/xvdf')
  except Exception as e:
    print('Error attaching volume. Continuing...', e)
  time.sleep(3)

def mount_volume_data(job, tag):
  for i,t in enumerate(job.tasks):
    attach_instance_ebs(t.instance, f'{tag}_{i+args.volume_offset}')
  job.run_async_join('sudo mkdir data -p')
  job.run_async_join('sudo mount /dev/xvdf data', ignore_errors=True)
  job.run_async_join('sudo chown `whoami` data')
  

gpu_count = collections.defaultdict(lambda:0, { 'p3.2xlarge': 1, 'p3.8xlarge': 4, 'p3.16xlarge': 8, 'p2.xlarge': 1, 'p2.8xlarge': 4, 'p2.16xlarge': 8 })
def create_job(run, job_name, num_tasks):
  install_script = ''
  if args.install_script:
    with open(args.install_script, 'r') as f:
      install_script = f.read()
  
  job = run.make_job(job_name, num_tasks=num_tasks, ebs=ebs, instance_type=args.instance_type, install_script=install_script, placement_group=args.placement_group, use_spot=args.spot)
  job.wait_until_ready()
  print(job.connect_instructions)

  if args.attach_volume: mount_volume_data(job, tag=args.attach_volume)
#   run pytorch
  job.run_async_join('killall python || echo failed')  # kill previous run
  job.run_async_join('source activate pytorch_p36')
  # job.run_async_join('source activate pytorch_source', ignore_errors=True)


  # upload files
  job.upload_async('training/resnet.py')
  job.upload_async('training/resnet_sd.py')
  job.upload_async('training/fp16util.py')
  job.upload_async('training/train_imagenet_nv.py')

  # setup machines
  setup_complete = [t.file_exists('/tmp/nv_setup_complete') for t in job.tasks]
  if not all(setup_complete):
    job.upload_async('setup/setup_env_nv.sh')
    job.run_async_join('chmod +x setup_env_nv.sh')
    job.run_async_join('bash setup_env_nv.sh', max_wait_sec=60*60, check_interval=5)

  
  # multi job
  instance_0 = job.tasks[0].instance
  world_0_ip = instance_0.private_ip_address
  num_gpus = gpu_count[instance_0.instance_type]
  port = '6006' # 6006, 6007, 6008, 8890, 6379
  datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
  job.run_async_join('ulimit -n 9000') # to prevent tcp too many files open error

  # Training on 8 gpus
  # task_cmds = []
  # for i,t in enumerate(job.tasks):
  #   # Pytorch distributed
  #   # save_dir = f'/efs/training/{datestr}-{job_name}-{i}'
  #   epochs = 40
  #   warmup = 0
  #   batch_size = 128
  #   lr = 0.25 * num_tasks
  #   tag = 'test_ar'
  #   save_dir = f'~/data/training/nv/{datestr}-{job_name}-lr{lr*10}e{epochs}bs{batch_size}w{warmup}-{tag}'
  #   t.run(f'mkdir {save_dir} -p')
  #   training_args = f'~/data/imagenet --save-dir {save_dir} --loss-scale 512 --fp16 -b {batch_size} --sz 224 -j 8 --lr {lr} --warmup {warmup} --epochs {epochs} --small --dist-url env:// --dist-backend nccl --distributed --val-ar'
  #   dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
  #   nccl_rings = get_nccl_rings(num_tasks, num_gpus)
  #   nccl_args = f'NCCL_RINGS="{nccl_rings}" NCCL_DEBUG=VERSION'
  #   cmd = f'{nccl_args} python -m torch.distributed.launch {dist_args} train_imagenet_nv.py {training_args}'
  #   t.run(f'echo {cmd} > {save_dir}/script.log')
  #   task_cmds.append(cmd)

  # Training on 4 machines
  # task_cmds = []
  # for i,t in enumerate(job.tasks):
  #   # Pytorch distributed
  #   # save_dir = f'/efs/training/{datestr}-{job_name}-{i}'
  #   epochs = 38
  #   warmup = 0
  #   batch_size = 192
  #   lr = 0.35 * num_tasks
  #   tag = 'test_ar'
  #   save_dir = f'~/data/training/nv/{datestr}-{job_name}-lr{lr*10}e{epochs}bs{batch_size}w{warmup}-{tag}'
  #   t.run(f'mkdir {save_dir} -p')
  #   training_args = f'~/data/imagenet --save-dir {save_dir} --loss-scale 512 --fp16 -b {batch_size} --sz 224 -j 8 --lr {lr} --warmup {warmup} --epochs {epochs} --small --dist-url env:// --dist-backend nccl --distributed --val-ar'
  #   dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
  #   nccl_rings = get_nccl_rings(num_tasks, num_gpus)
  #   nccl_args = f'NCCL_RINGS="{nccl_rings}" NCCL_DEBUG=VERSION'
  #   cmd = f'{nccl_args} python -m torch.distributed.launch {dist_args} train_imagenet_nv.py {training_args}'
  #   t.run(f'echo {cmd} > {save_dir}/script.log')
  #   task_cmds.append(cmd)
  

  task_cmds = []
  for i,t in enumerate(job.tasks):
    # Pytorch distributed
    # save_dir = f'/efs/training/{datestr}-{job_name}-{i}'
    epochs = 45
    warmup = 0
    batch_size = 192
    lr = 0.40 * num_tasks
    world_size = num_gpus * num_tasks
    tag = 'dawn'
    save_dir = f'~/data/training/nv/{datestr}-{job_name}-lr{lr*10}e{epochs}bs{batch_size}w{warmup}-{tag}'
    t.run(f'mkdir {save_dir} -p')
    training_args = f'~/data/imagenet --save-dir {save_dir} --loss-scale 512 --fp16 -b {batch_size} --sz 224 -j 8 --lr {lr} --warmup {warmup} --epochs {epochs} --small --dist-url file:///home/ubuntu/data/file.sync --dist-backend nccl --distributed --world-size {world_size}'
    dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
    cmd = f'python -m torch.distributed.launch {dist_args} train_imagenet_nv.py {training_args}'
    t.run(f'echo {cmd} > {save_dir}/script.log')
    task_cmds.append(cmd)

  # async calls need to be run last for multiple tasks. Otherwise they don't all run
  for t,cmd in zip(job.tasks, task_cmds):
    t.run_async(cmd)


def build_ring_order(machine_order, gpu_order):
  gpu_order = list(gpu_order)
  machine_order = list(machine_order)
  ngpus = len(gpu_order)
  r_order = [(x*ngpus) + y for x in machine_order for y in gpu_order]
  return ' '.join(map(str, r_order))

def get_nccl_rings(num_tasks, num_gpus):
    ring = build_ring_order(range(num_tasks), range(num_gpus))
    ring_rev = build_ring_order(reversed(range(num_tasks)), reversed(range(num_gpus)))
    if num_tasks == 8:
      ring_skip = build_ring_order([1,4,7,2,5,0,3,6], [3,2,1,0,7,6,5,4])
      ring_skip_rev = build_ring_order(reversed([1,4,7,2,5,0,3,6]), [3,2,1,0,7,6,5,4])
      rings_arr = [ring, ring_rev, ring_skip, ring_skip_rev]
    elif num_tasks == 4:
      ring_skip = build_ring_order([0,2,1,3], [3,2,1,0,7,6,5,4])
      rings_arr = [ring, ring_rev, ring_skip]
    else:
      rings_arr = [ring, ring_rev]
    return ' | '.join(rings_arr)


def main():
  import aws_backend

  run = aws_backend.make_run(args.name, ami=args.ami,
                             availability_zone=args.zone,
                             linux_type=args.linux_type, skip_efs_mount=(not args.mount_efs))
  create_job(run, args.job_name, args.num_tasks)


if __name__=='__main__':
  main()
