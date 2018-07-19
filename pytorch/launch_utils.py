
from collections import OrderedDict
import os
import sys
import time
import collections

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util

def get_gpu_count(instance):
  gpu_count = { 
    'p3.2xlarge': 1, 
    'p3.8xlarge': 4, 
    'p3.16xlarge': 8, 
    'p2.xlarge': 1, 
    'p2.8xlarge': 4, 
    'p2.16xlarge': 8 
  }
  gpu_count = collections.defaultdict(lambda:0, gpu_count)
  return gpu_count[instance.instance_type]

# EBS Utils
def mount_volume_data(job, tag, offset):
  for i,t in enumerate(job.tasks):
    attach_instance_ebs(t.instance, f'{tag}_{i+offset}')
  job.run_async_join('sudo mkdir data -p')
  job.run_async_join('sudo mount /dev/xvdf data', ignore_errors=True)
  job.run_async_join('sudo chown `whoami` data')
  
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

def get_ebs_settings(use_iops):
  ebs = {
    'VolumeSize': 500, 
    'DeleteOnTermination': True,
    'VolumeType': 'gp2'
  }
  # Use higher io ebs if we are using default instance storage
  if use_iops: 
    ebs['VolumeType'] = 'io1'
    ebs['Iops'] = 11500  # lowered from 14k to allow 17 instances

  return [{
    'DeviceName': '/dev/sda1',
    'Ebs': ebs
  }]


# NCCL Rings
def get_nccl_args(num_tasks, num_gpus):
  if num_tasks <= 1: return 'NCCL_DEBUG=VERSION'
  nccl_rings = get_nccl_rings(num_tasks, num_gpus)
  return f'NCCL_RINGS="{nccl_rings}" NCCL_DEBUG=VERSION'

def get_nccl_rings(num_tasks, num_gpus):
  ring = build_ring_order(range(num_tasks), range(num_gpus))
  ring_rev = build_ring_order(reversed(range(num_tasks)), reversed(range(num_gpus)))
  rotated_gpu_order = [3,2,1,0,7,6,5,4]
  if (num_tasks >= 8) and (num_tasks % 8 != 0):
    skip_step = 5 if num_tasks == 16 else 3
    # step size of 3 yields - [0,3,6,1,4,7,2,5]
    skip_machine_order = [(i*skip_step)%num_tasks for i in range(num_tasks)]
    ring_skip = build_ring_order(skip_machine_order, rotated_gpu_order)
    ring_skip_rev = build_ring_order(reversed(skip_machine_order), rotated_gpu_order)
    rings_arr = [ring, ring_rev, ring_skip, ring_skip_rev]
  elif num_tasks == 4:
    ring_skip = build_ring_order([0,2,1,3], rotated_gpu_order)
    rings_arr = [ring, ring_rev, ring_skip]
  else:
    rings_arr = [ring, ring_rev]
  return ' | '.join(rings_arr)

def build_ring_order(machine_order, gpu_order):
  gpu_order = list(gpu_order)
  machine_order = list(machine_order)
  ngpus = len(gpu_order)
  r_order = [(x*ngpus) + y for x in machine_order for y in gpu_order]
  return ' '.join(map(str, r_order))
