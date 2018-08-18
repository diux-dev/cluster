#!/usr/bin/env python
# numpy01 image, see environment-numpy.org for construction
# (DL AMI v 3.0 based)
#
# us-east-1 AMIs
# numpy00: ami-f9d6dc83
# numpy01: ami-5b524f21

from collections import defaultdict
import argparse
import boto3
import os
import sys
import time

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ami-name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 12.0',
                    help="name of AMI to use")
parser.add_argument('--placement-group', type=str, default='pytorch_cluster',
                     help="name of the current run")
parser.add_argument('--name', type=str, default='pytorch',
                     help="name of the current run")
parser.add_argument('--instance-type', type=str, default='p3.2xlarge',
                     help="type of instance")
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')
parser.add_argument('--role', type=str, default='launcher',
                    help='launcher or worker')
parser.add_argument('--num-tasks', type=int, default=1,
                    help='number of instances to create')
parser.add_argument('--skip-efs-mount', action='store_true',
                    help='skip mounting EFS for speed')
parser.add_argument('--logdir-prefix', default='/efs/runs.cifar',
                    help='where to put logs')
parser.add_argument('--spot', action="store_true",
                    help='launch using spot requests')


args = parser.parse_args()


gpu_count = defaultdict(lambda:0, { 'p3.2xlarge': 1, 'p3.8xlarge': 4, 'p3.16xlarge': 8, 'p2.xlarge': 1, 'p2.8xlarge': 4, 'p2.16xlarge': 8 })
def create_job(run, job_name, num_tasks):
  



def launch_jupyter(job, sess='jupyter'):

  def run_tmux_async(session, cmd):   # run command in "selected" tmux session
    job._run_raw(f'tmux send-keys -t {session}:0 "{cmd}" Enter')

  job._run_raw(f'tmux kill-session -t {sess}')
  job._run_raw(f'tmux new-session -s {sess} -n 0 -d')

  run_tmux_async(sess, 'source activate tensorflow_p36') # for TensorBoard/events
  run_tmux_async(sess, 'conda install pytorch torchvision -c pytorch -y')

  # Commands below add TOC extension, but take few minutes to install
  # conda solving environment is slow, disable
  #  run_tmux_async(sess, 'conda install -c conda-forge jupyter_nbextensions_configurator -y')
  #  run_tmux_async(sess, 'conda install ipyparallel -y') # to get rid of error https://github.com/jupyter/jupyter/issues/201
  job.upload('../jupyter_notebook_config.py') # 2 step upload since don't know ~
  run_tmux_async(sess, 'cp jupyter_notebook_config.py ~/.jupyter')
  run_tmux_async(sess, 'mkdir -p /efs/notebooks')

  run_tmux_async(sess, 'cd /efs/notebooks')
  run_tmux_async(sess, 'jupyter notebook')
  print(f'Jupyter notebook will be at http://{job.public_ip}:8888')
  

def main():
  # log into different logging root
  ncluster._set_global_logdir_prefix(args.logdir_prefix)
  
  job = ncluster.create_job('cifar', num_tasks=args.num_tasks,
                            run_name=args.name, spot=args.spot)
  job.join()  # wait for job to come up

  # upload files
  module_path=os.path.dirname(os.path.abspath(__file__))
  job.upload(f'{module_path}/*.py')

  # setup env
  job.run('source activate pytorch_p36')
  job.run('pip install tensorboardX')

  # single process
  num_gpus = gpu_count[args.instance_type]
  if (num_tasks == 1) and (num_gpus == 1):
    job.run(f'python train_cifar10.py --logdir={job.logdir}',
             async=True) # single instance

  # multi process
  else:
    task0 = job.tasks[0]
    port = '12345'
    #    job.run('ulimit -n 9000') # to prevent tcp too many files open error
    world_size = num_gpus * num_tasks

    for task in job.tasks:
      training_args = f'--dist-url env:// --dist-backend gloo --distributed --world-size {world_size} --scale-lr 2  --logdir={job.logdir}'
      dist_args = f'--nproc_per_node={num_gpus} --nnodes={num_tasks} --node_rank={i} --master_addr={task0.ip} --master_port={port}'
      task.run(f'python -m torch.distributed.launch {dist_args} train_cifar10.py {training_args}', async=True)


  # also run jupyter notebook on task 0
  task0.switch_tmux('jupyter')
  task0.run('source activate tensorflow_p36') # for TensorBoard/events
  task0.run('conda install pytorch torchvision -c pytorch -y')
  task0.upload('../jupyter_notebook_config.py') # 2 step upload since don't know ~
  task0.run('cp jupyter_notebook_config.py ~/.jupyter')
  task0.run('mkdir -p /efs/notebooks')

  task0.run('cd /efs/notebooks')
  task0.run('jupyter notebook')
  print(f'Jupyter notebook will be at http://{job.public_ip}:8888')
  print(f'Jupyter notebook will be at http://{task0.name}:8888')
  
      
if __name__=='__main__':
  main()
