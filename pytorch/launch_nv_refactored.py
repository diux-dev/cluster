#!/usr/bin/env python
#
#
# python train_imagenet.py --config=1  # 1 machine
# python train_imagenet.py --config=4  # 4 machines
# python train_imagenet.py --config=8  # 8 machines
# python train_imagenet.py --config=16 # 16 machines


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
parser.add_argument('--use-local-conda', action='store_true',
                    help='use local conda installation (for initial setup, see recipes.md)')
parser.add_argument('--config', type=str, default='1', "which config to use, "
                    "1, 4, 8, 16")
parser.add_argument('--ami', default='pytorch.imagenet.source.v6')
parser.add_argument('--env-name', default='pytorch.imagenet.source.v6')
args = parser.parse_args()

# https://goodcode.io/articles/python-dict-object/
lr = 0.47
one_machine = Config(
  phases=[
    {'ep':0,  'sz':128, 'bs':512, 'trndir':'-sz/160'},
    {'ep':(0,5),  'lr':(lr,lr*2)}, # lr warmup is better with --init-bn0
    {'ep':5,      'lr':lr},
    {'ep':14, 'sz':224, 'bs':192},
    {'ep':16,     'lr':lr/10},
    {'ep':27,     'lr':lr/100},
    {'ep':32, 'sz':288, 'bs':128, 'min_scale':0.5, 'rect_val':True},
    {'ep':(33,35),'lr':lr/1000}
  ],
  num_tasks=1,
)

configs = {"1": one_machine,
           "4", four_machines,
           "8", eight_machines,
           "16", sixteen_machines}

def main():
  params = eval(args.params)
  
  job = ncluster.create_job('worker', config.num_tasks, run_name='final',
                            spot=args.spot, ami=args.ami)
  for i in range(num_tasks):
    job.tasks[i].attach_volume('imagenet_%02d'%(i), '/data')

  if not args.use_local_conda:
    job.run(f'source activate {config.conda_env}')
  else:
    # enable conda command
    job.run('. /home/ubuntu/anaconda3/etc/profile.d/conda.sh')
    job.run(f'conda activate /data/anaconda3/envs/{config.env_name}')
    
  job.run('ulimit -n 9000')
  job.upload('*.py')
  

  # runs setup script if necessary. Installs packages in current conda env
  # checks for "complete" file in the end, hence no-op if it exists
  job._run_setup_script('setup_env_nv.sh')

  world_size = 8*num_tasks
  nccl_args = launch_utils_lib.get_nccl_args(num_tasks, 8)

  base_save_dir = '~/data/training/nv'
  datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
  save_dir = f'{base_save_dir}/{datestr}-{save_tag}-w{world_size}'
  job.run(f'mkdir {save_dir} -p')

  # todo: refactor params/config
    # Training script args
  default_params = [
    '~/data/imagenet',
    '--save-dir', save_dir,
    '--fp16',
    '--loss-scale', 1024,
    '--world-size', world_size,
    '--distributed'
  ]
  training_args = default_params + params
  training_args = training_args + ["--logdir", job.logdir]
  training_args = ' '.join(map(launch_utils_lib.format_args, training_args))

  # Run tasks
  task_cmds = []
  port = 6006
  for i, task in enumerate(job.tasks):
    dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={job.tasks[0].ip} --master_port={port}'
    cmd = f'{nccl_args} python -m torch.distributed.launch {dist_args} train_imagenet_nv.py {training_args}'
    task.run(f'echo {cmd} > {save_dir}/script-{i}.log')
    task.run(cmd, async=True)
  
  print(f"Logging to {job.logdir}")


if __name__=='__main__':
  main()
