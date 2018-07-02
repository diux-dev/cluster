#!/usr/bin/env python
# numpy01 image, see environment-numpy.org for construction
# (DL AMI v 3.0 based)
#
# us-east-1 AMIs
# numpy00: ami-f9d6dc83
# numpy01: ami-5b524f21


# master command:
# python launch_fastai.py --instance-type p3.16xlarge --num-tasks 4 --job-name cluster_4_region_c --zone us-west-2c --ami ami-53c8822b --placement-group pytorch_cluster_c

from collections import OrderedDict
import argparse
import os
import sys
import time
import collections
import boto3
import datetime

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util
util.install_pdb_handler()

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='ami-e580c79d',
                     help="name of AMI to use ")
# parser.add_argument('--placement-group', type=str, default='',
parser.add_argument('--placement-group', type=str, default='pytorch_cluster',
                     help="name of the current run")
parser.add_argument('--name', type=str, default='pytorch',
                     help="name of the current run")
parser.add_argument('--job-name', type=str, default='distributed',
                     help="name of the jobs to run")
# parser.add_argument('--instance-type', type=str, default='p3.2xlarge',
parser.add_argument('--instance-type', type=str, default='t2.large',
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
parser.add_argument('--spot', action='store_true', 
                    help='launch using spot requests')
args = parser.parse_args()


## Setup instance:
ebs = None
ebs = [{
  'DeviceName': '/dev/sda1',
  'Ebs': {
    'VolumeSize': 1000, 
    'DeleteOnTermination': True,
    'VolumeType': 'io1',
    'Iops': 14000
  }
}]

# For using external ebs (so we don't hit iops limit)
# ebs = [{
#   'DeviceName': '/dev/sda1',
#   'Ebs': {
#     'VolumeSize': 300, 
#     'DeleteOnTermination': True,
#     'VolumeType': 'io1',
#     'Iops': 5000
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
    time.sleep(5)
  try:
    v.attach_to_instance(InstanceId=aws_instance.id, Device='/dev/xvdf')
  except Exception as e:
    print('Error attaching volume. Continuing...', e)
  time.sleep(3)

def mount_volume_data(job, tag):
  for i,t in enumerate(job.tasks):
    attach_instance_ebs(t.instance, f'{tag}_{i+1}')
  job.run('sudo mkdir data -p')
  job.run('sudo mount /dev/xvdf data', ignore_errors=True)
  job.run('sudo chown `whoami` data')

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
  job.run('killall python || echo failed')  # kill previous run

  # upload files
  job.upload('resnet.py')
  job.upload('train_imagenet_fastai.py')
  job.upload('train_imagenet_fastai_v2.py')
  
  # setup machines
  setup_complete = [t.file_exists('/tmp/fastai_setup_complete') for t in job.tasks]
  if not all(setup_complete):
    job.run_async_join('conda create -n fastai -y', check_interval=5) # (AS) WARNING remember to revert back 
    job.run('source activate fastai') # (AS) WARNING remember to revert back 
    job.upload('setup_env_fastai.sh')
    job.run('chmod +x setup_env_fastai.sh')
    job.run_async_join('bash setup_env_fastai.sh', max_wait_sec=60*60, check_interval=60)

  job.run('source activate fastai') # (AS) WARNING remember to revert back 
  # job.run('source activate pytorch_p36')


  # multi job
  world_0_ip = job.tasks[0].instance.private_ip_address
  port = '6006' # 6006, 6007, 6008, 8890, 6379
  datestr = datetime.datetime.now().replace(microsecond=0).isoformat()
  job.run('ulimit -n 9000') # to prevent tcp too many files open error
  num_gpus = gpu_count[args.instance_type]

  if num_gpus <= 1:
    save_dir = f'~/training/fastai/{datestr}-{job_name}'
    job.run(f'mkdir {save_dir} -p')
    training_args = f'~/data/imagenet --save-dir {save_dir} --loss-scale 512 --fp16 -b 192 --sz 224 -j 8 --lr 0.40 --epochs 45' # old file sync
    job.run_async(f'python train_imagenet_fastai.py {training_args}')
    return

  task_cmds = []
  for i,t in enumerate(job.tasks):
    # Pytorch distributed
    # save_dir = f'/efs/training/{datestr}-{job_name}-{i}'
    save_dir = f'~/data/training/fastai/{datestr}-{job_name}-{i}-lr3d84-e55'
    t.run(f'mkdir {save_dir} -p')
    lr = 0.40 * num_tasks
    training_args = f'~/data/imagenet --save-dir {save_dir} --loss-scale 512 --fp16 -b 192 --sz 224 -j 8 --lr {lr} --epochs 55 --dist-url env:// --dist-backend nccl --distributed' # old file sync
    dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
    cmd = f'python -m torch.distributed.launch {dist_args} train_imagenet_fastai_v2.py {training_args}'
    t.run(f'echo "{cmd}" > {save_dir}/script.log')
    task_cmds.append(cmd)

  # async calls need to be run last for multiple tasks. Otherwise they don't all run
  for t,cmd in zip(job.tasks, task_cmds):
    t.run_async(cmd)

def main():
  import aws_backend

  run = aws_backend.make_run(args.name, ami=args.ami,
                             availability_zone=args.zone,
                             linux_type=args.linux_type)
  create_job(run, args.job_name, args.num_tasks)


if __name__=='__main__':
  main()
