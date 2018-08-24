#!/usr/bin/env python
#
# export ami="Deep Learning AMI (Ubuntu) Version 12.0"
# 1 machine training
# python launch_nv.py --name test --spot
#
# 4 machine training
# python launch_nv.py --name 4gpu_distributed --spot --attach-volume imagenet_high_perf --params x4_args

# 8 machine training
# python launch_nv.py --name yaro8 --spot --attach-volume imagenet_high_perf  --params x8ar_args

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
parser.add_argument('--attach-volume', type=str, default='imagenet',
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

lr = 1.0
ashaw = [
  '--phases', [
    # {'ep':0,  'sz':128, 'bs':512, 'trndir':'-sz/160'},
    {'ep':(0, 100),  'lr':lr},
    {'ep':0,  'sz':224, 'bs':256},
    {'ep':8,  'sz':288, 'bs':128},
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 1,
  # '--ami-name', 'Deep Learning AMI (Ubuntu) Version 12.0',
  # '--env-name', 'pytorch_p36',
  '--ami-name', 'pytorch.imagenet.source.v6',
  '--env-name', 'pytorch_source',
  '--short-epoch',
  '--skip-auto-shutdown'
]


# OOM after 1-10 seconds
lr = 1.0
quick_oom = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':512, 'trndir':'-sz/160'},
    {'ep':(0, 100),  'lr':lr},
    {'ep':2,  'sz':224, 'bs':224},
    {'ep':4,  'sz':288, 'bs':160},
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 1,
  '--ami-name', 'pytorch.imagenet.source.v6',
  '--env-name', 'pytorch_source',
  # '--env-name', 'pytorch_p36',
  '--short-epoch',
  '--skip-auto-shutdown'
]

# fast run to check shutdown behavior
quick_run = [
  '--phases', [
    {'ep':0,  'sz':224, 'bs':128},
    {'ep':(0,5),  'lr':(lr,lr*2)}, # lr warmup is better with --init-bn0
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 1,
  '--ami-name', 'pytorch.imagenet.source.v6',
  '--env-name', 'pytorch_source',
  '--short-epoch'
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
  '--ami-name', 'pytorch.imagenet.source.v6',
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
  '--ami-name', 'Deep Learning AMI (Ubuntu) Version 12.0',
  '--env-name', 'pytorch_p36',
]


# Current best settings
# Current benchmark for 1x p3
lr = 1.0
xar_args_pytorch = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':512, 'trndir':'-sz/160'},
    {'ep':(0,5),  'lr':(lr,lr*2)}, # lr warmup is better with --init-bn0
    {'ep':5,      'lr':lr},
    {'ep':14, 'sz':224, 'bs':192},
    {'ep':16,     'lr':lr/10},
    {'ep':27,     'lr':lr/100},
    {'ep':32, 'sz':288, 'bs':128, 'min_scale':0.5, 'use_ar':True},
    {'ep':(33,35),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--autoscale-lr2batch',
  '--num-tasks', 1,
  '--ami-name', 'pytorch.imagenet.source.v6',
  '--env-name', 'pytorch_source',
  '--dist-url', 'file:///home/ubuntu/data/file.sync', # single instances are faster with file sync
  # '--c10d'
]


# Current benchmark for 4x p3
lr = 0.47 * 4 # 4 = num tasks
x4ar_args = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':256, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,      'lr':lr},
    {'ep':16, 'sz':224, 'bs':192},
    {'ep':19,     'lr':lr/10},
    {'ep':31,     'lr':lr/100},
    {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'use_ar':True},
    {'ep':(38,40),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--autoscale-lr2batch',
  '--num-tasks', 4,
  '--ami-name', 'Deep Learning AMI (Ubuntu) Version 12.0',
  # '--resume', 'sz128_checkpoint.path.tar'
]


# Faster benchmark for 4x p3 - 43 minutes
lr = 0.47 * 4 # 4 = num tasks
x4ar_args_bench = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':256, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,      'lr':lr},
    {'ep':15, 'sz':224, 'bs':192, 'trndir':'-sz/352', 'min_scale':0.086},
    {'ep':18,     'lr':lr/10},
    {'ep':29,     'lr':lr/100},
    {'ep':34, 'sz':288, 'bs':128, 'min_scale':0.5, 'use_ar':True},
    {'ep':(35,38),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--autoscale-lr2batch',
  '--num-tasks', 4,
  '--ami-name', 'pytorch.imagenet.source.v6',
  '--env-name', 'pytorch_source',
  '--factorized-resnet',
]

# Current testing params 4x p3
lr = 0.47 * 4 # 4 = num tasks
x4ar_args_test_bench_2 = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':256, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,  'sz':128, 'bs':512, 'keep_dl':True},
    {'ep':6,      'lr':lr*2},
    {'ep':15, 'sz':224, 'bs':194, 'trndir':'-sz/352', 'min_scale':0.086},
    {'ep':15,      'lr':lr/1.5},
    {'ep':18,     'lr':lr/10/1.5},
    {'ep':29,     'lr':lr/100/1.5},
    {'ep':34, 'sz':288, 'bs':128, 'min_scale':0.5, 'use_ar':True},
    {'ep':34,     'lr':lr/100},
    {'ep':(35,38),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 4,
  '--ami-name', 'pytorch.imagenet.source.v6',
  # '--resume', 'sz128_checkpoint.path.tar'
  '--env-name', 'pytorch_source',
  # '--factorized-resnet',
  # '--c10d'
]

# Current benchmark for 8x p3's - with Aspect Ratio Validation - Works right now for under 30 min (25:45, memory-eight.06, 25:03 sun-eight)
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
    {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'use_ar':True},
    {'ep':37,     'lr':lr/100},
    {'ep':(38,40),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 8,
  '--ami-name', 'pytorch.imagenet.source.v6',
  '--env-name', 'pytorch_source',
]

# Also ~27 minutes. Faster per epoch, but takes one extra
lr = 0.235 * 8 # 8 = num tasks
x8ar_args_352_folder = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':128, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,            'bs':256, 'keep_dl':True},
    {'ep':6,      'lr':lr*2},
    {'ep':16, 'sz':224, 'bs':128, 'trndir':'-sz/352', 'min_scale':0.086},
    {'ep':16,      'lr':lr},
    {'ep':19,           'bs':192, 'keep_dl':True},
    {'ep':19,     'lr':lr/(10/1.5)},
    {'ep':31,     'lr':lr/(100/1.5)},
    {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'use_ar':True},
    {'ep':37,     'lr':lr/100},
    {'ep':(38,40),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 8,
  '--ami-name', 'pytorch.imagenet.source.v6',
  '--env-name', 'pytorch_source',
]

# Trying faster training schedule with original size (original gets 93.1%) - also increasing batch size but doesn't work
lr = 0.235 * 8 # 8 = num tasks
x8ar_args_test_2 = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':128, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,            'bs':256, 'keep_dl':True}, # (AS) definitely diverges here - remove batch size
    {'ep':6,      'lr':lr*4},
    {'ep':16, 'sz':224, 'bs':128},
    {'ep':16,     'lr':lr*1.5},
    {'ep':19,           'bs':192, 'keep_dl':True},
    {'ep':19,     'lr':lr/(10/1.5)},
    {'ep':31,     'lr':lr/(100/1.5)},
    {'ep':36, 'sz':288, 'bs':128, 'min_scale':0.5, 'use_ar':True},
    {'ep':36,     'lr':lr/100},
    {'ep':(37,40),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--num-tasks', 8,
  '--ami-name', 'pytorch.imagenet.source.v6',
  '--env-name', 'pytorch_source',
]

# Current benchmark for 16x p3's - with Aspect Ratio Validatoin
# python launch_nv.py --name yaro-friday-16 --num-tasks 16 --params x16ar_args

# Current benchmark for 16x p3's - with Aspect Ratio Validatoin
lr = 0.235
x16ar_args = [
  '--phases', [
    {'ep':0,  'sz':128, 'bs':64, 'trndir':'-sz/160'},
    {'ep':(0,6),  'lr':(lr,lr*2)},
    {'ep':6,      'lr':lr},
    {'ep':16, 'sz':224, 'bs':64},
    {'ep':19,     'lr':lr/10},
    {'ep':31,     'lr':lr/100},
    {'ep':37, 'sz':288, 'bs':64, 'min_scale':0.5, 'use_ar':True},
    {'ep':(38,40),'lr':lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--autoscale-lr2batch',
  '--scale-lr', 8, # 8 = num tasks / 2 (because 64 batch size)
  '--num-tasks', 16,
  '--ami-name', 'Deep Learning AMI (Ubuntu) Version 12.0',
]


# Ohio-sixteen base
# 18:17 mins to 93.03, ohio-sixteen
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
    {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'use_ar':True},
    {'ep':37,     'lr':2*lr/100},
    {'ep':(38,40),'lr':2*lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--scale-lr', 8, # 8 = num tasks
  '--num-tasks', 16,
  '--ami-name', 'pytorch.imagenet.source.v6',
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
    {'ep':37, 'sz':288, 'bs':128, 'min_scale':0.5, 'use_ar':True},
    {'ep':37,     'lr':3*lr/100},
    {'ep':(38,50),'lr':3*lr/1000}
  ],
  '--init-bn0',
  '--no-bn-wd',
  '--scale-lr', 8, # 8 = num tasks
  '--num-tasks', 24,
  '--ami-name', 'pytorch.imagenet.source.v6',
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
  '--ami-name', 'pytorch.imagenet.source.v6',
  '--env-name', 'pytorch_source',
]


# hacks to allow launcher level flags in worker params list

def _extract_num_tasks(params): return _extract_param(params, '--num-tasks')
def _extract_ami_name(params): return _extract_param(params, '--ami-name')
def _extract_env_name(params):
  name = _extract_param(params, '--env-name', strict=False)
  if not name:
    return DEFAULT_ENV_NAME
  else:
    return name


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
  ebs = launch_utils_lib.get_ebs_settings(use_iops=(args.attach_volume is None))
    
  job = run.make_job(job_name, num_tasks=num_tasks, ebs=ebs, instance_type=args.instance_type, install_script=install_script, use_spot=args.spot, use_placement_group=True)
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
  training_files = ['resnet.py', 'fp16util.py', 'dataloader.py', 'meter.py', 'logger.py', 'dist_utils.py', 'train_imagenet_nv.py', 'experimental_utils.py']
  for f in training_files: job.upload_async(f)

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
    '--logdir', job.logdir
  ]
  if world_size > 1: default_params.append('--distributed')
  training_args = default_params + params
  training_args = ' '.join(map(launch_utils_lib.format_args, training_args))

  # Run tasks
  task_cmds = []
  for i,t in enumerate(job.tasks):
    dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={world_0_ip} --master_port={port}'
    cmd = f'{nccl_args} python -m torch.distributed.launch {dist_args} train_imagenet_nv.py {training_args}'
    t.run(f'echo {cmd} > {job.logdir}/script.log')
    t.run(cmd, sync=False)

if __name__=='__main__':
  main()
