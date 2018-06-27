#!/usr/bin/env python
# numpy01 image, see environment-numpy.org for construction
# (DL AMI v 3.0 based)
#
# us-east-1 AMIs
# numpy00: ami-f9d6dc83
# numpy01: ami-5b524f21

from collections import OrderedDict
import argparse
import os
import sys
import time

import boto3

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util
util.install_pdb_handler()

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami', type=str, default='ami-e580c79d',
                     help="name of AMI to use ")
parser.add_argument('--placement-group', type=str, default='pytorch_cluster',
                     help="name of the current run")
parser.add_argument('--name', type=str, default='pytorch_gpus',
                     help="name of the current run")
# parser.add_argument('--instance-type', type=str, default='p3.2xlarge',
parser.add_argument('--instance-type', type=str, default='t2.large',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-west-2a',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
args = parser.parse_args()


def attach_imagnet_ebs(aws_instance, job, tag='imagenet_high_perf'):
  ec2 = util.create_ec2_resource()
  v = list(ec2.volumes.filter(Filters=[{'Name':'tag:Name', 'Values':['imagenet_high_perf']}]).all())
  if not v: return
  v = v[0]
  if v.attachments and v.attachments[0]['InstanceId'] == aws_instance.id:
        return
  if v.state != 'available': 
      print('Detaching from current instance')
      v.detach_from_instance()
      time.sleep(3)
  print('Attaching to instance:' , aws_instance.id)
  v.attach_to_instance(InstanceId=aws_instance.id, Device='/dev/xvdf')
  time.sleep(3)
  job.run('sudo mkdir mount_point -p')
  job.run('sudo mount /dev/xvdf mount_point')
  

gpu_count = collections.defaultdict(lambda:0, { 'p3.2xlarge': 1, 'p3.8xlarge': 4, 'p3.16xlarge': 8, 'p2.xlarge': 1, 'p2.8xlarge': 4, 'p2.16xlarge': 8 })


def create_job(run, job_name, num_tasks=2):
      install_script = ''
  with open('setup_env.sh', 'r') as script:
    install_script = script.read()


  
  ebs = [{
    'DeviceName': '/dev/sda1',
    'Ebs': {
        'VolumeSize': 1000, 
        'DeleteOnTermination': True,
        'VolumeType': 'io1',
        'Iops': 32000
    }
  }]

  job = run.make_job(job_name, num_tasks=num_tasks, ebs=ebs, instance_type=args.instance_type, install_script=install_script, placement_group=args.placement_group)
  job.wait_until_ready()
  print(job.connect_instructions)

  attach_imagnet_ebs(job.tasks[0].instance, job)

#   run pytorch
  job.run('killall python || echo failed')  # kill previous run
  job.run('source activate pytorch_p36')

  # upload files
  job.upload('resnet.py')
  job.upload('fp16util.py')
  # job.upload('distributed.py')
  # job.upload('multiproc.py')
  job.upload('train_imagenet_nv.py')
  job.upload('resize_images.py')

  job.upload('setup_env_nv.sh')
  job.run('chmod +x setup_env_nv.sh')
  job.run('./setup_env_nv.sh')
  

  # start training
  # job.run_async('python -m multiproc train_imagenet_fastai.py ~/mount_point/imagenet  --sz 224 -b 192 -j 8 --fp16 -a resnet50 --lr 0.40 --epochs 45 --small')

  # single machine, multi-gpu
  if num_tasks == 1:
    job.run_async('python -m multiproc train_imagenet_nv.py ~/mount_point/imagenet -b 128 -a resnet50 -j 8 --fp16 -a resnet50 --lr 0.40 --epochs 45 --small')
    return

  # multi job
  world_0_ip = job.tasks[0].instance.private_ip_address
  port = '6006' # 6006, 6007, 6008, 8890, 6379

  for i,t in enumerate(job.tasks):
    # tcp only supports CPU - https://pytorch.org/docs/master/distributed.html
    # dist_params = f'--world-size {num_tasks} --rank {i} --dist-url tcp://{world_0_ip}:{port} --dist-backend tcp' # tcp
    
    # Pytorch distributed
    num_gpus = gpu_count[args.instance_type]
    training_args = '~/data --loss-scale 512 --fp16 --lr 0.4 -b 128 -j 7 --dist-url env:// --dist-backend gloo --distributed' # half precision
    dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
    t.run_async(f'python -m torch.distributed.launch {dist_args} train_imagenet_nv.py {training_args}')



def main():
  import aws_backend

  run = aws_backend.make_run(args.name, ami=args.ami,
                             availability_zone=args.zone,
                             linux_type=args.linux_type)
  create_job(run, 'distributed_imagenet')


if __name__=='__main__':
  main()
