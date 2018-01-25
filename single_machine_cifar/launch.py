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

u.install_pdb_handler()  # drops into pdb on CTRL+\

parser = argparse.ArgumentParser(description='Launch CIFAR training')
# TODO: rename to gradient instance type
parser.add_argument('--instance-type', type=str, default='g3.4xlarge',
                    help='instance to use for gradient workers')
parser.add_argument("--num-gpus", default=1, type=int,
                    help="Number of GPUs to use per worker.")
parser.add_argument('--name', type=str, default='cifar00',
                     help="name of the current run")
parser.add_argument('--steps', type=int, default=1000,
                     help="number of steps to run for")
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

# todo: make upload work on AWS backend
# TODO: transition tousing TMUX per-line execution (this way source activate works)
AWS_INSTALL_SCRIPT="""
"""

TMUX_INSTALL_SCRIPT="""
source activate oct12
%upload cifar10_estimator
"""

def launch_tmux(backend, install_script):
  run = backend.make_run(args.name, install_script=install_script)
  job = run.make_job('master', 1)
  
  # Launch tensorflow tasks.
  job.run('source activate oct12')  # must replace with local env that has TF
  job.upload('cifar10_estimator')
  job.run('cd cifar10_estimator')
  tf_cmd = """python cifar10_main.py --data-dir=/efs/cifar-10-data \
                     --job-dir={logdir} \
                     --num-gpus=1 \
                     --train-steps=1000""".format(logdir=run.logdir)

  job.run(tf_cmd)
  
  tb_cmd = "tensorboard --logdir={logdir} --port={port}".format(
    logdir=run.logdir, port=job.port)
  job.run(tb_cmd, sync=False)
  print("See tensorboard at http://%s:%s"%(job.ip, job.port))

def launch_aws(backend, install_script):
  region = os.environ.get("AWS_DEFAULT_REGION")
  ami = generic_ami_dict[region]

  run = backend.make_run(args.name, install_script=install_script,
                         ami=ami, availability_zone=args.zone)
  job = run.make_job('master', 1, instance_type=args.instance_type)

  job.wait_until_ready()
  job.run("source activate tensorflow_p36  # env with cuda 8")
  job.upload('cifar10_estimator')

  # Launch tensorboard visualizer in separate tmux session
  job.run("tmux kill-session -t tb || echo ok")
  job.run("tmux new-session -s tb -n 0 -d")
  job.run("tmux send-keys -t tb:0 'source activate tensorflow_p36' Enter")
  job.run("tmux send-keys -t tb:0 'tensorboard --logdir {logdir}' Enter".format(logdir=run.logdir))
  print("Tensorboard available at %s:6006"%(job.public_ip,))

  # Launch tensorflow tasks.
  job.run('cd cifar10_estimator')
  tf_cmd = """python cifar10_main.py --data-dir=/efs/cifar-10-data \
                     --job-dir={logdir} \
                     --num-gpus={gpus} \
                     --train-steps={steps}""".format(logdir=run.logdir,
                                                     steps=args.steps,
                                                     gpus=args.num_gpus)

  job.run(tf_cmd, sync=False)


def main():
  if args.backend == 'tmux':
    launch_tmux(tmux_backend, TMUX_INSTALL_SCRIPT)
  elif args.backend == 'aws':
    launch_aws(aws_backend, AWS_INSTALL_SCRIPT)
  else:
    assert False, "Unknown backend: "+args.backend


if __name__=='__main__':
  main()
  
