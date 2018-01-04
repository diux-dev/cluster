# script to launch Cifar-10 training with a small set
import argparse
import json
import os
import portpicker
import sys
import time

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import tmux
import tmux_backend
import aws
import aws_backend
import util as u

u.install_pdb_handler()


parser = argparse.ArgumentParser(description='Launch CIFAR training')
parser.add_argument('--placement', type=int, default=0,
                     help="launch all jobs inside placement group")
# TODO: rename to gradient instance type
parser.add_argument('--gpu-instance-type', type=str, default='g3.4xlarge',
                    help='instance to use for gradient workers')
parser.add_argument('--cpu-instance-type', type=str, default='c5.xlarge',
                    help='default instance type')
parser.add_argument("--num-workers", default=2, type=int,
                    help="The number of gradient workers to use.")
parser.add_argument("--num-gpus", default=1, type=int,
                    help="Number of GPUs to use per worker.")
parser.add_argument("--num-ps", default=1, type=int,
                    help="The number of parameter servers workers to use.")
parser.add_argument('--name', type=str, default='cafar',
                     help="name of the current run")
parser.add_argument('--zone', type=str, default='us-east-1c',
                    help='which availability zone to use')
parser.add_argument('--cluster', type=str, default='tmux',
                    help='tmux or aws')

args = parser.parse_args()
assert args.num_workers>0  # need non-empty for both worker and master jobs

# Amazon Ubuntu Deep Learning AMI
generic_ami_dict = {
    "us-west-2": "ami-3b6bce43",
    "us-east-1": "ami-9ba7c4e1",
}

# Script used to make custom AMI.
#
# source activate mxnet_p36
# export wheel=https://pypi.python.org/packages/86/f9/7b773ba67997b5c001ca57c4681d618f62d3743b7ee4b32ce310c1100cd7/tf_nightly_gpu-1.5.0.dev20171221-cp36-cp36m-manylinux1_x86_64.whl#md5=d126089a6fbbd81b104d303baeb649ff
# pip install $wheel --upgrade-strategy=only-if-needed

# sudo apt install -y emacs24
# # disable auto-update
# emacs /etc/apt/apt.conf.d/20auto-upgrades

# # Change the below setting to zero:
# # APT::Periodic::Unattended-Upgrade "0";
# sudo apt install -y pssh
# source activate mxnet_p36
# pip install jupyter ipywidgets bokeh
# pip install https://s3-us-west-2.amazonaws.com/ray-wheels/772527caa484bc04169c54959b6a0c5001650bf6/ray-0.3.0-cp36-cp36m-manylinux1_x86_64.whl

# wget https://raw.githubusercontent.com/tensorflow/models/master/official/resnet/resnet_model.py
# wget https://raw.githubusercontent.com/tensorflow/models/master/official/resnet/cifar10_main.py
# wget https://raw.githubusercontent.com/tensorflow/models/master/official/resnet/cifar10_download_and_extract.py
# wget https://raw.githubusercontent.com/tensorflow/models/da62bb0b52e0f5f6919f053a98e8cf9a032fa60a/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py

# source activate mxnet_p27
# pip install tf-nightly-gpu
# python cifar10_download_and_extract.py
# python generate_cifar10_tfrecords.py

# git clone https://github.com/tensorflow/models.git
# git clone https://github.com/ray-project/ray.git
# cp models/tutorials/image/cifar10_estimator/cifar10.py .
# cp models/tutorials/image/cifar10_estimator/cifar10_main.py .
# cp models/tutorials/image/cifar10_estimator/cifar10_model.py .
# cp models/tutorials/image/cifar10_estimator/cifar10_utils.py .
# cp models/tutorials/image/cifar10_estimator/model_base.py .

# # Emacs ~/.tmux.conf and add "set-option -g history-limit 250000" to top
# source activate mxnet_p36

custom_ami_dict = {
   "us-east-1": "ami-67c4871d",
   "us-west-2": "ami-47e04e3f",
}


# todo: make upload work on AWS backend
# TODO: transition to using TMUX per-line execution (this way source activate works)
AWS_INSTALL_SCRIPT="""
source activate mxnet_p36  # env with cuda 9
export wheel=https://pypi.python.org/packages/86/f9/7b773ba67997b5c001ca57c4681d618f62d3743b7ee4b32ce310c1100cd7/tf_nightly_gpu-1.5.0.dev20171221-cp36-cp36m-manylinux1_x86_64.whl#md5=d126089a6fbbd81b104d303baeb649ff

# --upgrade-strategy=only-if-needed prevents wiping of Intel MKL numpy
pip install $wheel --upgrade-strategy=only-if-needed

#%upload *.py
#%upload *.tfrecords
"""

TMUX_INSTALL_SCRIPT="""
source activate cifar
%upload *.py
%upload *.tfrecords
"""

def launch_tmux(backend, install_script):
  run = backend.make_run(args.name, install_script=install_script)
  master_job = run.make_job('master', 1)
  worker_job = run.make_job('worker', args.num_workers-1)
  ps_job = run.make_job('ps', args.num_ps)
  tb_job = run.make_job('tb', 1)

  # Orchestration: every worker needs to know:
  # 1. their own role (task_spec), ie {type: worker, index: 0}
  # 2. role->ip mapping of all machines (cluster_spec), ie
  #    {"worker": ["localhost:24724"], "ps": ["localhost:15960"]}}
  # 
  # There are 3 types of workers, "master" ("chief worker"), "worker", ps
  def tf_env_setup(task, cluster_spec, task_spec):
    """Helper method to initialize clusterspec for a task."""
    TF_CONFIG = json.dumps(
      {'cluster': cluster_spec, 'task': task_spec, 'model_dir': run.logdir,
      'environment': 'cloud'})
    # TODO: replace with single command
    #    task.file_write('TF_CONFIG.sh', "export TF_CONFIG='%s'\n"%(TF_CONFIG,))
    #    task.run('source TF_CONFIG.sh')
    task.run("export TF_CONFIG='%s'"%(TF_CONFIG,))

  master_hosts = ["%s:%d"%(task.ip, task.port) for task in master_job.tasks]
  worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
  ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
  cluster_spec = {'master': master_hosts, 'worker': worker_hosts,
                  'ps': ps_hosts}
  if not worker_hosts:  # special case of 1 master, 0 workers
    del cluster_spec['worker']
  
  # Launch tensorflow tasks.
  tf_cmd = "python cifar10_main.py --data-dir=. --job-dir={logdir} --num-gpus={num_gpus} --log-device-placement".format(logdir=run.logdir, num_gpus=args.num_gpus)
  
  task_type = 'master' 
  for task in master_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run(tf_cmd+' --label='+task.job.name+':'+str(task.id), sync=False)

  task_type = 'worker' 
  for task in worker_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run(tf_cmd+' --label='+task.job.name+':'+str(task.id), sync=False)

  task_type = 'ps'
  for task in ps_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run(tf_cmd+' --label='+task.job.name+':'+str(task.id), sync=False)

  tb_cmd = "tensorboard --logdir={logdir} --port={port}".format(
    logdir=run.logdir, port=tb_job.port)
  tb_job.run(tb_cmd, sync=False)
  print("See tensorboard at http://%s:%s"%(tb_job.ip, tb_job.port))

def launch_aws(backend, install_script):
  region = os.environ.get("AWS_DEFAULT_REGION")
  ami = custom_ami_dict[region]

  # todo: use tf.slim-like API with args decorators
  run = backend.make_run(args.name, install_script=install_script,
                         ami=ami, availability_zone=args.zone)
  master_job = run.make_job('master', 1, instance_type=args.gpu_instance_type)
  worker_job = run.make_job('worker', args.num_workers-1,
                            instance_type=args.gpu_instance_type)
  ps_job = run.make_job('ps', args.num_ps, instance_type=args.cpu_instance_type)
  tb_job = run.make_job('tb', 1, instance_type=args.cpu_instance_type)
  all_jobs = [master_job, worker_job, ps_job, tb_job]
  all_tasks = [task for job in all_jobs for task in job.tasks]

  for job in all_jobs:
    job.wait_until_ready()


  for task in all_tasks:
    task.upload('generate_cifar10_tfrecords.py', skip_existing=True)
    task.upload('cifar10_download_and_extract.py', skip_existing=True)
    task.upload('cifar10.py', skip_existing=True)
    task.upload('cifar10_main.py', skip_existing=True)
    task.upload('cifar10_model.py', skip_existing=True)
    task.upload('cifar10_utils.py', skip_existing=True)
    task.upload('model_base.py', skip_existing=True)
    #    task.upload('eval.tfrecords', skip_existing=True)
    #    task.upload('train.tfrecords', skip_existing=True)
    #    task.upload('validation.tfrecords', skip_existing=True)
    task.run('source activate mxnet_p36')
    

  def tf_env_setup(task, cluster_spec, task_spec):
    """Helper method to initialize clusterspec for a task."""
    TF_CONFIG = json.dumps(
      {'cluster': cluster_spec, 'task': task_spec, 'model_dir': run.logdir,
      'environment': 'cloud'})
    task.run("export TF_CONFIG='%s'\n"%(TF_CONFIG,))

  master_hosts = ["%s:%d"%(task.ip, task.port) for task in master_job.tasks]
  worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
  ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
  cluster_spec = {'master': master_hosts, 'worker': worker_hosts,
                  'ps': ps_hosts}
  if not worker_hosts:  # special case of 1 master, 0 workers
    del cluster_spec['worker']

  # Launch tensorflow tasks.
  tf_cmd = "python cifar10_main.py --data-dir=/efs/cifar-10-data --job-dir={logdir}".format(logdir=run.logdir)
  
  task_type = 'master' 
  for task in master_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run(tf_cmd+' --num-gpus=1', sync=False)

  task_type = 'worker' 
  for task in worker_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run(tf_cmd+' --num-gpus=1', sync=False)

  # launch parameter server tasks
  task_type = 'ps'
  for task in ps_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run(tf_cmd, sync=False)


  # Launch tensorboard visualizer.
  tb_cmd = "tensorboard --logdir={logdir} --port=6006".format(logdir=run.logdir)
  tb_job.run(tb_cmd, sync=False)
  print("See tensorboard at http://%s:%s"%(tb_job.public_ip, 6006))


def main():
  if args.cluster == 'tmux':
    if args.num_gpus != 0:
      args.num_gpus = 0
      print("Overriding num_gpus to 0 since running locall")
    launch_tmux(tmux_backend, TMUX_INSTALL_SCRIPT)
  elif args.cluster == 'aws':
    launch_aws(aws_backend, AWS_INSTALL_SCRIPT)
  else:
    assert False, "Unknown cluster: "+args.cluster


if __name__=='__main__':
  main()
  
