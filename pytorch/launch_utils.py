
from collections import OrderedDict
import os
import sys
import time
import collections

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util as u

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

def format_args(arg):
  if isinstance(arg, list) or isinstance(arg, dict): return '\"'+str(arg)+'\"'
  else: return str(arg)

def extract_param(params, name, default=None):
  args = [i,k for i,k in enumerate(params) if k==name]
  if (not args) and (default is not None): return default
  assert len(args) == 1, f"Must specify exactly 1 '{name}'"

  idx,key = args[0]
  del params[idx]
  val = params.pop(idx+1)
  return val

# EBS Utils
ATTACH_WAIT_INTERVAL_SEC = 5
def mount_volume_data(job, tag, offset, unix_device=u.DEFAULT_UNIX_DEVICE):
  for i,t in enumerate(job.tasks):
      attach_instance_ebs(t.instance, '%s_%02d'%(tag, i+offset))
  job.run_async_join('sudo mkdir data -p')
  while True:
    try:
      # Need retry logic because attachment is async and can be slow
      # run_async doesn't propagate exceptions raised on workers, use regular
      #
      # Possibilities:
      # mount fails because volume attachment is not fully ready, need retry
      # mount fails because volume is already attached
      # umount fails because some processes are slow to die, need retry
      # umount fails because volume is not attached
      # Use heuristic on whether need to mount, first task already has it
      # mounted
      job.tasks[0].run('df > /tmp/mount_status')
      status = job.tasks[0].file_read('/tmp/mount_status')
      if unix_device in status:
        print('Volume already mounted, ignoring')
      else:
        job.run(f'sudo mount {unix_device} data')
    except Exception as e:
      print(f'(un)mount failed with: ({e})')
      print(f'Retrying in {ATTACH_WAIT_INTERVAL_SEC}')
      time.sleep(ATTACH_WAIT_INTERVAL_SEC)
      continue
    else:
      print(f'Mount successful')
      break
    
  job.run_async_join('sudo chown `whoami` data')

def attach_instance_ebs(aws_instance, tag, unix_device=u.DEFAULT_UNIX_DEVICE):
  """Attaches volume to instance. Will try to detach volume if it's already mounted somewhere else. Will retry indefinitely on error."""
  
  ec2 = u.create_ec2_resource()
  v = list(ec2.volumes.filter(Filters=[{'Name':'tag:Name', 'Values':[tag]}, {"Name":"availability-zone", 'Values':[os.environ['ZONE']]}]).all())
  assert(v), f"Volume {tag} not found."
  v = v[0]
  volume_name = u.get_name(v)
  already_attached = v.attachments and v.attachments[0]['InstanceId'] == aws_instance.id
  instance_name = u.get_name(aws_instance)
  # TODO: still have edge case when it doesn't report as already attached
  # and keeps trying to attach to an instance that has data mounted already
  if already_attached:
    print(f'volume {volume_name} ({v.id}) already attached to {instance_name}')
    return
  while v.state != 'available':
    response = v.detach_from_instance()
    instance_id = v.attachments[0]['InstanceId']
    instance_name = u.get_name(instance_id)
    print(f'Volume {tag} is attached to {instance_name}, detaching, response={response.get("State", "none")}')
    time.sleep(ATTACH_WAIT_INTERVAL_SEC)
    v.reload()
  while True:
    try:
      response = v.attach_to_instance(InstanceId=aws_instance.id,
                                      Device=unix_device)
      print(f'Attaching {volume_name} to {instance_name}: response={response.get("State", "none")}')

    # sometimes have unrecoverable failure on brand new instance with
    # possibly because of https://forums.aws.amazon.com/thread.jspa?threadID=66192
    #    Error attaching volume: (An error occurred (InvalidParameterValue) when calling the AttachVolume operation: Invalid value '/dev/xvdf' for unixDevice. Attachment point /dev/xvdf is already in use). Retrying in 5 An error occurred (InvalidParameterValue) when calling the AttachVolume operation: Invalid value '/dev/xvdf' for unixDevice. Attachment point /dev/xvdf is already in use

    except Exception as e:
      print(f"Failed attaching ({v.id}) to ({aws_instance.id})")
      print(f'Error attaching volume: ({e}). Retrying in {ATTACH_WAIT_INTERVAL_SEC}', e)
      time.sleep(ATTACH_WAIT_INTERVAL_SEC)
      continue
    else:
      print('Attachment successful')
      break

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
  return f'NCCL_RINGS="{nccl_rings}" NCCL_SINGLE_RING_THRESHOLD=10 NCCL_DEBUG=VERSION'
  # return 'NCCL_MIN_NRINGS=2 NCCL_SINGLE_RING_THRESHOLD=10 NCCL_DEBUG=VERSION'

def get_skip_order(size):
    if size == 4: return [0,2,1,3]
    skip_step = 5 if size == 16 else 3
    # step size of 3 yields - [0,3,6,1,4,7,2,5]
    return [(i*skip_step)%size for i in range(size)]
  
def get_nccl_rings(num_tasks, num_gpus):
  ring = build_ring_order(range(num_tasks), range(num_gpus))
  ring_rev = build_ring_order(reversed(range(num_tasks)), reversed(range(num_gpus)))
  rotated_gpu_order = [3,2,1,0,7,6,5,4]
  skip_gpu_order = get_skip_order(num_gpus)
  if (num_tasks >= 4) and (num_gpus == 8):
    assert((num_tasks % 4) == 0)
    skip_machine_order = get_skip_order(num_tasks)
    ring_skip = build_ring_order(skip_machine_order, rotated_gpu_order)
    ring_skip_rev = build_ring_order(reversed(skip_machine_order), skip_gpu_order)
    rings_arr = [ring, ring_rev, ring_skip, ring_skip_rev]
    # rings_arr = [ring, ring_rev, ring_skip]
  else:
    rings_arr = [ring, ring_rev]
  return ' | '.join(rings_arr)

def build_ring_order(machine_order, gpu_order):
  gpu_order = list(gpu_order)
  machine_order = list(machine_order)
  ngpus = len(gpu_order)
  r_order = [(x*ngpus) + y for x in machine_order for y in gpu_order]
  return ' '.join(map(str, r_order))
