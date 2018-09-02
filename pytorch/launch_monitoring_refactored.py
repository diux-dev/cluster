#!/usr/bin/env python
# launch TensorBoard/monitoring server for runs
# Run
# ./launch.py
#
# Run on AWS:
# ./launch.py --backend=aws

import argparse
import os
import sys

# import cluster tools, one level up
module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import backend as backend_lib
import aws_backend
import tmux_backend
import util as u

parser = argparse.ArgumentParser()
parser.add_argument('--ami-name', type=str,
                    default="Deep Learning AMI (Ubuntu) Version 12.0",
                    help="name of AMI to use")
parser.add_argument('--name', type=str, default='monitoring', help='run name')
parser.add_argument('--instance-type', type=str, default='r5.large',
                     help='instance type to use for tensorboard job')
parser.add_argument('--backend', type=str, default='aws',
                    help='cluster backend, tmux (local) or aws')
parser.add_argument('--skip-install', action='store_true',
                    help='skip conda install (because slow)')
args = parser.parse_args()

def main():
  if args.backend == 'tmux':
    backend = tmux_backend
  elif args.backend == 'aws':
    backend = aws_backend
  else:
    assert False, "unknown backend"

  run = backend.make_run(args.name,
                         ami_name=args.ami_name)
  job = run.make_job('tb', instance_type=args.instance_type)
  job.wait_until_ready()

  job.run('source activate tensorflow_p36')
  if not args.skip_install:
   job.run('conda install -c conda-forge jupyter_nbextensions_configurator -y')
   job.run('conda install -c conda-forge jupyter_contrib_nbextensions -y')
   job.run('conda install ipyparallel -y') # to get rid of error https://github.com/jupyter/jupyter/issues/201
   job.run('jupyter nbextension enable toc2/main')

  job.upload('jupyter_notebook_config.py') # 2 step upload since don't know ~
  job.run('mkdir -p /efs/notebooks')
  job.run('cp jupyter_notebook_config.py ~/.jupyter')

  job.run('cd /efs/notebooks')

  zone = u.get_zone()
  if zone.startswith('us-west-2'):
    window_title = 'Oregon'
  elif zone.startswith('us-east-1'):
    window_title = 'Virginia'
  elif zone.startswith('us-east-2'):
    window_title = 'Ohio'
  else:
    window_title = 'Tensor-Board'
  job.run_async(f'tensorboard --logdir={backend_lib.LOGDIR_PREFIX} --port=6006 --window_title={window_title}')
  
  print(f'Tensorboard will be at http://{job.public_ip}:6006')

  # run second Tensorboard in new tmux session for "selected runs"
  # to select runs,
  # on instance, do "ln -s /efs/runs/<run_name> /efs/runs.selected/<run_name>
  # (must use abspath for ln -s left hand side for linking to work)

  
  # TODO: maybe replace "run_tmux_async" with task.run_tmux_async(name, cmd)

  def run_tmux_async(session, cmd):   # run command in "selected" tmux session
    job._run_raw(f'tmux send-keys -t {session}:0 "{cmd}" Enter')

  selected_logdir = backend_lib.LOGDIR_PREFIX+'.selected'
  job._run_raw("tmux kill-session -t selected")
  job._run_raw("tmux new-session -s selected -n 0 -d")
  run_tmux_async('selected', 'source activate tensorflow_p36')
  run_tmux_async('selected',
                 f"tensorboard --logdir {selected_logdir} --port=6007")
  print(f'Tensorboard selected will be at http://{job.public_ip}:6007')

  cifar_logdir = backend_lib.LOGDIR_PREFIX+'.cifar'
  job._run_raw("tmux kill-session -t cifar")
  job._run_raw("tmux new-session -s cifar -n 0 -d")
  run_tmux_async('cifar', 'source activate tensorflow_p36')
  run_tmux_async('cifar',
                 f"tensorboard --logdir {cifar_logdir} --port=6008")
  print(f'Tensorboard cifar will be at http://{job.public_ip}:6008')

  # launch jupyter notebook server
  job._run_raw('tmux kill-session -t jupyter')
  job._run_raw('tmux new-session -s jupyter -n 0 -d')
  run_tmux_async('jupyter', 'cd /efs/notebooks')
  run_tmux_async('jupyter', 'source activate tensorflow_p36')
  run_tmux_async('jupyter', 'jupyter notebook')
  print(f'Jupyter notebook will be at http://{job.public_ip}:8888')

  


if __name__=='__main__':
  main()
