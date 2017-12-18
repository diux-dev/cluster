#!/usr/bin/env python
# Launch resnet cifar training on AWS
#
# For pre-requisites, see
# https://github.com/diux-dev/cluster/blob/master/ray_integration/README.md
#
# In particular, need to set following variables (example)
# export KEY_NAME=yaroslav
# export SSH_KEY_PATH=~/d/yaroslav.pem
# export SECURITY_GROUP=open
# export AWS_DEFAULT_REGION=us-west-2
#
# Assumes EFS is created. Currently following are hardwired for existing EFS
# export EFS_ID=fs-ab2b8102
# export EFS_REGION=us-west-2

# TODO: this needs CUDA setup, currently it runs CPU-only tensorflow
# 
# This are setup commands to run on each AWS instance
INSTALL_SCRIPT="""
tmux set-option -g history-limit 250000

# python3 and pip
sudo apt update -y
sudo apt install -y python3-pip
sudo ln -s /usr/bin/python3 /usr/bin/python
sudo ln -s /usr/bin/pip3 /usr/bin/pip

pip install tensorflow==1.4.1
wget https://raw.githubusercontent.com/tensorflow/models/master/official/resnet/resnet_model.py
wget https://raw.githubusercontent.com/tensorflow/models/master/official/resnet/cifar10_main.py
wget https://raw.githubusercontent.com/tensorflow/models/master/official/resnet/cifar10_download_and_extract.py
python cifar10_download_and_extract.py

sudo apt install -y nfs-common  # this overwrites python with python2 
sudo mv /usr/bin/python /usr/bin/python2
sudo ln -s /usr/bin/python3 /usr/bin/python

export EFS_ID=fs-ab2b8102
export EFS_REGION=us-west-2
sudo mkdir -p /efs
sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 $EFS_ID.efs.$EFS_REGION.amazonaws.com:/ /efs
"""

# Ubuntu Server 16.04 LTS (HVM), SSD Volume Type - ami-0def3275

LINUX_TYPE = "ubuntu"  # linux type determines username to use to login
ami_dict = {
    "us-east-1": "ami-aa2ea6d0",
    "us-west-2": "ami-0def3275"
}



import os
import sys
import argparse
import time
parser = argparse.ArgumentParser(description='Ray parameter server experiment')
parser.add_argument('--name', type=str, default='cifar00',
                     help="name of the current run")
parser.add_argument('--worker_instance', type=str, default='g3.4xlarge',
                     help="instance type to use for gradient worker")
parser.add_argument('--tb_instance', type=str, default='m4.xlarge',
                     help="instance type to use for tensorboard")
args = parser.parse_args()


# g3.4xlarge for training
# r4.2xlarge for tensorboard (61 GB of ram)
# r4.4xlarge for tensorboard (122 GB of RAM)
# m4.xlarge  16 GB

def main():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import aws

  assert 'AWS_DEFAULT_REGION' in os.environ
  assert os.environ.get("AWS_DEFAULT_REGION") in ami_dict
  assert os.environ.get("AWS_DEFAULT_REGION") == 'us-west-2', "Currently EFS is hardwired to us-west-2 region"

  AMI = 'ami-0def3275' # Ubuntu 16.04 in us-west-2
  worker_job = aws.simple_job(args.name+'-worker', num_tasks=1,
                              instance_type=args.worker_instance,
                              install_script=INSTALL_SCRIPT,
                              ami=AMI)
  tb_job = aws.simple_job(args.name+'-tb', num_tasks=1,
                          instance_type=args.tb_instance,
                          install_script=INSTALL_SCRIPT,
                          ami=AMI)

  # block until things launch to run commands
  worker_job.wait_until_ready()
  tb_job.wait_until_ready()
  
  worker = worker_job.tasks[0]
  tb = tb_job.tasks[0]

  logdir = '/efs/runs/'+args.name
  worker.run('python cifar10_main.py --model_dir='+logdir)
  tb.run('tensorboard --logdir='+logdir)
  
  print ("Connect to worker:")
  print(worker.connect_instructions)
  print("See tensorboard at http://%s:%d"%(tb.public_ip, 6006))
  
  
if __name__=='__main__':
  main()
