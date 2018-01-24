#!/usr/bin/env python
# script to launch cifar-10 training on a single machine
import argparse
import json
import os
import portpicker
import sys
import time

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import tmux_backend
import aws_backend
import util as u

u.install_pdb_handler()  # drops into PDF on CTRL+c

parser = argparse.ArgumentParser(description='Launch CIFAR training')
# TODO: rename to gradient instance type
parser.add_argument('--instance-type', type=str, default='g3.4xlarge',
                    help='instance to use for gradient workers')
parser.add_argument("--num-gpus", default=1, type=int,
                    help="Number of GPUs to use per worker.")
parser.add_argument('--name', type=str, default='cifar00',
                     help="name of the current run")
parser.add_argument('--zone', type=str, default='us-east-1c',
                    help='which availability zone to use')
parser.add_argument('--backend', type=str, default='tmux',
                    help='tmux or aws')

args = parser.parse_args()

# Amazon Ubuntu Deep Learning AMI
generic_ami_dict = {
    "us-west-2": "ami-3b6bce43",
    "us-east-1": "ami-9ba7c4e1",
}

# Script used to make custom AMI. (TODO: it tensorflow CUDA 8 version, need 9)
# TODO: need dtach (sudo apt install -y dtach)
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
# TODO: transition tousing TMUX per-line execution (this way source activate works)
# TODO: make upload work with directories
AWS_INSTALL_SCRIPT="""
"""

TMUX_INSTALL_SCRIPT="""
source activate oct12
%upload cifar10_estimator
"""

def launch_tmux(backend, install_script):
  run = backend.make_run(args.name, install_script=install_script)
  master_job = run.make_job('master', 1)
  
  # Launch tensorflow tasks.
  # TODO: rename tmp to efs
  master_job.run('cd cifar10_estimator')
  tf_cmd = """python cifar10_main.py --data-dir=/tmp/cifar-10-data \
                     --job-dir={logdir} \
                     --num-gpus=1 \
                     --train-steps=1000""".format(logdir=run.logdir)

  master_job.run(tf_cmd)
  
#  tb_cmd = "tensorboard --logdir={logdir} --port={port}".format(
#    logdir=run.logdir, port=tb_job.port)
#  tb_job.run(tb_cmd, sync=False)
#  print("See tensorboard at http://%s:%s"%(tb_job.ip, tb_job.port))

def launch_aws(backend, install_script):
  region = os.environ.get("AWS_DEFAULT_REGION")
  ami = generic_ami_dict[region]

  run = backend.make_run(args.name, install_script=install_script,
                         ami=ami, availability_zone=args.zone)
  master_job = run.make_job('master', 1, instance_type=args.instance_type)
  master_job.wait_until_ready()
  # TODO: rename to initialize or call automatically

  
  master_job.run("source activate tensorflow_p36  # env with cuda 8")
  master_job.run("mkdir cifar10_estimator")
  master_job.run("%upload cifar10_estimator/model_base.py cifar10_estimator/model_base.py")
  master_job.run("%upload cifar10_estimator/cifar10_model.py cifar10_estimator/cifar10_model.py")
  master_job.run("%upload cifar10_estimator/cifar10.py cifar10_estimator/cifar10.py")
  master_job.run("%upload cifar10_estimator/cifar10_utils.py cifar10_estimator/cifar10_utils.py")
  
  # Launch tensorflow tasks.
  master_job.run('cd cifar10_estimator')
  tf_cmd = """python cifar10_main.py --data-dir=/efs/cifar-10-data \
                     --job-dir={logdir} \
                     --num-gpus=1 \
                     --train-steps=1000""".format(logdir=run.logdir)

  master_job.run(tf_cmd)

  # Launch tensorboard visualizer.
  #  tb_cmd = "tensorboard --logdir={logdir} --port=6006".format(logdir=run.logdir)
  #  tb_job.run(tb_cmd, sync=False)
  #  print("See tensorboard at http://%s:%s"%(tb_job.public_ip, 6006))


def main():
  if args.backend == 'tmux':
    launch_tmux(tmux_backend, TMUX_INSTALL_SCRIPT)
  elif args.backend == 'aws':
    launch_aws(aws_backend, AWS_INSTALL_SCRIPT)
  else:
    assert False, "Unknown backend: "+args.backend


if __name__=='__main__':
  main()
  
