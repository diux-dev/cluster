#!/usr/bin/env python
#
# 8 machine training
# python launch_nv.py --name release-eight --params x8ar_args_benchmark --upgrade-root-volume
#
# Old settings. TODO: update
# export ami="Deep Learning AMI (Ubuntu) Version 12.0"
# 1 machine training
# python launch_nv.py --name test --spot
#
# 4 machine training
# python launch_nv.py --name 4gpu_distributed --spot --params x4_args

# 16 machine training
# export AWS_DEFAULT_REGION=us-east-1
# ./launch_nv.py --name yaro16 --params x16ar_args

# one machine training with slow pytorch
# python launch_nv.py --name pytorch-one-machines-ar --params=xar_args_pytorch --attach-volume imagenet_high_perf


DATA_ROOT='/home/ubuntu/data' # location where attached EBS "data" volume is mounted

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
import util as u
import aws_backend
import launch_utils as launch_utils_lib
u.install_pdb_handler()

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--spot', action='store_true', 
                    help='launch using spot requests')
parser.add_argument('--name', type=str, default='imagenet',
                     help=("name of the current run, this determines placement "
                           "group name, instance names and EFS logging "
                           "directory."))
parser.add_argument('--instance-type', type=str, default='p3.16xlarge',
                     help="type of instance")
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
parser.add_argument('--attach-volume', type=str, default='',
                    help='tag name of ebs volume to attach')
parser.add_argument('--use-local-conda', action='store_true',
                    help='use local conda installation (for initial setup, see recipes.md)')
parser.add_argument('--volume-offset', type=int, default=0,
                    help='Offset number for volume attachment. If running multiple jobs')
parser.add_argument('--skip-efs-mount', action='store_true',
                    help='skip mounting EFS for speed')
parser.add_argument('--params', type=str, default="xar_args",
                    help='args to use, see "params = " line')
args = parser.parse_args()

DEFAULT_PYTORCH_SOURCE = 'pytorch.imagenet.source.v7'
DEFAULT_DLAMI = 'Deep Learning AMI (Ubuntu) Version 12.0'

# OOM after 1-10 seconds
lr = 1.0
quick_oom = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':512, 'trndir':'-sz/160'},
    {'ep':(0, 8),  'lr':lr},
    {'ep':2,  'sz':224, 'bs':224, 'trndir': '-sz/352', 'min_scale': 0.087},
    {'ep':4,  'sz':288, 'bs':160},
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 1,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_source',
  # '--env-name', 'pytorch_p36',
  '--short-epoch',
  '--skip-auto-shutdown'
]

# throughput/batch-size testing
lr = 1.0
xar_throughput = [
  '--phases', [
    {'ep':0,  'sz':224, 'bs':192},
    {'ep':(0,5),  'lr':(lr,lr*2)}, # lr warmup is better with --init-bn0
    {'ep':(5,100), 'lr': lr}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 1,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_source',
  '--skip-auto-shutdown'
]

# throughput/batch-size testing
lr = 1.0
xar_throughput_dlami = [
  '--phases', [
    {'ep':0,  'sz':224, 'bs':256},
    {'ep':(0,5),  'lr':(lr,lr*2)}, # lr warmup is better with --init-bn0
    {'ep':(5,100), 'lr': lr}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 1,
  '--ami-name', DEFAULT_DLAMI,
  '--env-name', 'pytorch_p36',
]


# Current best settings
# Current benchmark for 1x p3
lr = 1.0
xar_args_pytorch = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':512, 'trndir':'-sz/160'},
    {'ep':(0,5),  'lr':(lr,lr*2)}, # lr warmup is better with --init-bn0
    # {'ep':5,      'lr':lr},
    {'ep':(5,10), 'lr':(lr*2,lr)}, # trying one cycle
    {'ep':14, 'sz':224, 'bs':224,
                  'lr':lr/(512/224)},
    {'ep':16,     'lr':lr/10/(512/224)},
    {'ep':27,     'lr':lr/100/(512/224)},
    {'ep':32, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True,
                  'lr':lr/100/(512/128)},
    {'ep':(33,35),'lr':lr/1000/(512/128)}
  ],
  '--init-bn0',
  '--no-bn-wd',
  # '--autoscale-lr2batch',
  '--num-tasks', 1,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_source',
  '--dist-url', 'file:///home/ubuntu/data/file.sync', # single instances are faster with file sync
]


# Current best settings 4x p3 - 34.5 minutes
lr = 0.50 * 4 # 4 = num tasks
scale_224 = 224/256
scale_288 = 128/256
x4ar_args = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':256, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)}, 
    {'ep':6,  'sz':128, 'bs':512, 'keep_dl':True,
                  'lr':lr*2},
    {'ep':16, 'sz':224, 'bs':224, 'trndir': '-sz/352', 'min_scale': 0.087,
                  'lr':lr*scale_224},
    {'ep':19,     'lr':lr/10*scale_224},
    {'ep':30,     'lr':lr/100*scale_224},
    {'ep':35, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True,
                  'lr':lr/100*scale_288},
    {'ep':(37,39),'lr':lr/1000*scale_288}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 4,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_c10d'
]



# Current best settings 4x p3 - 34.5 minutes
lr = 0.50 * 4 # 4 = num tasks
scale_224 = 224/256
scale_288 = 128/256
c10d = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':256, 'trndir':'-sz/160',
                  'lr':lr*2}
  ],
  '--num-tasks', 4,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_c10d',
  '--c10d',
  # '--dist-url', 'file:///home/ubuntu/data/file.sync', # single instances are faster with file sync
  # '--dist-url', 'tcp://localhost:6006', # single instances are faster with file sync
  # '--dist-url', 'env://',
]

# Current benchmark for 8x p3's - with Aspect Ratio Validation - Works right now for under 30 min (25:45, memory-eight.06, 25:03 sun-eight, 24:31 release-eight.02)
lr = 0.235 * 8 # 8 = num tasks
x8ar_args_benchmark = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':128, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,            'bs':256, 'keep_dl':True},
    {'ep':6,      'lr':lr*2},
    {'ep':16, 'sz':224,'bs':128},
    {'ep':16,      'lr':lr},
    {'ep':19,          'bs':192, 'keep_dl':True},
    {'ep':19,     'lr':lr/(10/1.5)},
    {'ep':31,     'lr':lr/(100/1.5)},
    {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True},
    {'ep':37,     'lr':lr/100},
    {'ep':(38,40),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 8,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_source',
]

# Also ~27 minutes. Faster per epoch, but takes one extra
lr = 0.25 * 8 # 8 = num tasks
scale_224 = 224/128
x8ar_args_352_folder = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':128, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,            'bs':256, 'keep_dl':True,
                  'lr':lr*2},
    {'ep':16, 'sz':224, 'bs':128, 'trndir':'-sz/352', 'min_scale':0.087,
                  'lr':lr},
    {'ep':19,           'bs':224, 'keep_dl':True,
                  'lr':lr/10*scale_224},
    {'ep':30,     'lr':lr/100*scale_224},
    {'ep':35, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True,
                  'lr':lr/100},
    {'ep':(38,40),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 8,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_source',
]

# Current benchmark for 16x p3's - with Aspect Ratio Validation

# Ohio-sixteen base
# 18:17 mins to 93.03, ohio-sixteen, 19:33 sun-sixteen.01
# after refactor, 16:36 to 92.97 in release-sixteen.02, 16:51 to 93.11 in release-sixteen.04 
lr = 0.235 * 8 # 
bs = 64
x16ar_args_benchmark = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':64, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,            'bs':128, 'keep_dl':True},
    {'ep':6,      'lr':lr*2},
    {'ep':16, 'sz':224,'bs':64}, # todo: increase this bs
    {'ep':16,      'lr':lr},
    {'ep':19,           'bs':192, 'keep_dl':True},
    {'ep':19,     'lr':2*lr/(10/1.5)},
    {'ep':31,     'lr':2*lr/(100/1.5)},
    {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True},
    {'ep':37,     'lr':2*lr/100},
    {'ep':(38,50),'lr':2*lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 16,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_source',
]


# 24-machine run, forked from 16 benchmark
lr = 0.235
x24ar_args_test = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':32, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,            'bs':128, 'keep_dl':True},
    {'ep':6,      'lr':lr*2},
    {'ep':16, 'sz':224,'bs':64},
    {'ep':16,      'lr':lr},
    {'ep':19,           'bs':192, 'keep_dl':True},
    {'ep':19,     'lr':3*lr/(10/1.5)},
    {'ep':31,     'lr':3*lr/(100/1.5)},
    {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True},
    {'ep':37,     'lr':3*lr/100},
    {'ep':(38,50),'lr':3*lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--scale-lr', 8, # 8 = num tasks
  '--num-tasks', 24,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_source',
]

# 32 machine throughput test, forked from xar_throughput
lr = 0.235
x32ar_throughput = [
  '--phases', [
    {'ep':0,  'sz':224, 'bs':192},
    {'ep':(0,5),  'lr':(lr,lr*2)},
    {'ep':(5,100), 'lr': lr}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 32,
  '--ami-name', DEFAULT_PYTORCH_SOURCE,
  '--env-name', 'pytorch_source',
]


def main():
  params = eval(args.params)
  num_tasks = launch_utils_lib.extract_param(params, '--num-tasks')
  ami_name = launch_utils_lib.extract_param(params, '--ami-name', default='Deep Learning AMI (Ubuntu) Version 13.0')
  env_name = launch_utils_lib.extract_param(params, '--env-name', default='pytorch_p36')
  
  run = aws_backend.make_run(args.name,
                             ami_name=ami_name,
                             skip_efs_mount=args.skip_efs_mount)
  
  job = create_job(run, 'worker', num_tasks, env_name)
  run.setup_logdir()  # must happen after first job is created and ready
  print(f"Logging to {job.logdir}")

  # Define custom params for training or use a preset above
  start_training(job, params)

def create_job(run, job_name, num_tasks, env_name):
  """Creates job, blocks until job is ready."""
  ebs = launch_utils_lib.get_ebs_settings(use_iops=not bool(args.attach_volume)) # higher iops if no ebs attached
    
  job = run.make_job(job_name, num_tasks=num_tasks, ebs=ebs, instance_type=args.instance_type, use_spot=args.spot, use_placement_group=True)
  job.wait_until_ready()
  print(job.connect_instructions)

  job.run_async_join('killall python || echo ignoring')  # kill previous run
  job.run_async_join(f'shutdown -c') # cancel old shutdown command
  job.run_async_join('ulimit -n 9000') # to prevent tcp too many files open error

  # mount_volume hardcoded to use data now
  # TODO: this should be global setting/constant instead
  assert DATA_ROOT.endswith('/data')
  if args.attach_volume:
    launch_utils_lib.mount_volume_data(job, tag=args.attach_volume, offset=args.volume_offset)

  if not args.use_local_conda:
    job.run_async_join(f'source activate {env_name}')
  else:
    # enable conda command
    job.run_async_join('. /home/ubuntu/anaconda3/etc/profile.d/conda.sh')
    job.run_async_join(f'conda activate {DATA_ROOT}/anaconda3/envs/{env_name}')

  # upload files
  job.upload_async('training', remote_fn='training')

  setup_complete = [t.file_exists('/tmp/nv_setup_complete') for t in job.tasks]
  if not all(setup_complete):
    job.upload_async('setup_env_nv.sh')
    job.run_async_join('chmod +x setup_env_nv.sh')
    job.run_async_join('bash setup_env_nv.sh', max_wait_sec=60*60, check_interval=5)

  return job

def start_training(job, params):
  num_tasks = len(job.tasks)  
  instance_0 = job.tasks[0].instance
  world_0_ip = instance_0.private_ip_address
  num_gpus = launch_utils_lib.get_gpu_count(instance_0)
  port = '6006' # 6006, 6007, 6008, 8890, 6379
  world_size = num_gpus * num_tasks

  # Use NCCL rings for faster network throughput
  nccl_args = launch_utils_lib.get_nccl_args(num_tasks, num_gpus)

  # Training script args
  default_params = [
    '~/data/imagenet',
    '--fp16',
    '--logdir', job.logdir,
    '--dist-url', f'tcp://{world_0_ip}:6006', # single instances are faster with file sync
    # '--dist-url', 'file:///home/ubuntu/data/file.sync', # single instances are faster with file sync
    # '--dist-url', 'tcp://localhost:6006', # single instances are faster with file sync
    # '--dist-url', 'env://',
  ]
  if world_size > 1: default_params.append('--distributed')
  training_args = default_params + params
  training_args = ' '.join(map(launch_utils_lib.format_args, training_args))

  # Run tasks
  task_cmds = []
  for i,t in enumerate(job.tasks):
    dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
    cmd = f'{nccl_args} python -m torch.distributed.launch {dist_args} training/train_imagenet_nv.py {training_args}'
    if i == 0: t.run(f'echo {cmd} > {job.logdir}/script.log', ignore_errors=True)
    t.run(cmd, sync=False)

if __name__=='__main__':
  main()
