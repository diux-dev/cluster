#!/usr/bin/env python
# script to launch Ray training with push pull parameter server

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
                    help='instance for GPU workers')
parser.add_argument('--cpu-instance-type', type=str, default='c5.large',
                    help='instance for CPU workers')
parser.add_argument('--tb-instance-type', type=str, default='m3.xlarge',
                    help='instance for TensorBoard jobs')
parser.add_argument("--num-workers", default=2, type=int,
                    help="The number of gradient workers to use.")
parser.add_argument("--num-gpus", default=1, type=int,
                    help="Number of GPUs to use per worker.")
parser.add_argument("--dim", default=75360, type=int,
                    help="The number of parameters, defaults to size of "
                    "TF default CIFAR10 model")
parser.add_argument("--num-ps", default=1, type=int,
                    help="The number of parameter servers workers to use.")
parser.add_argument('--zone', type=str, default='us-east-1c',
                    help='which availability zone to use')
parser.add_argument('--cluster', type=str, default='tmux',
                    help='tmux or aws')
parser.add_argument('--sync', type=int, default=0,
                    help='whether to enable sync mode')
parser.add_argument('--name', type=str, default='ray00',
                     help="name of the current run")
parser.add_argument('--name', type=str, default='ray00',
                     help="name of the current run")
parser.add_argument('--insert-pauses', action='store_true',
                    default=False,
                    help="Make ps0 freeze periodically")

args = parser.parse_args()
assert args.num_workers>0  # need non-empty for both worker and master jobs

# Official Amazon Ubuntu Deep Learning AMI
generic_ami_dict = {
    "us-west-2": "ami-3b6bce43",
    "us-east-1": "ami-9ba7c4e1",
}

# see launch.py for script used to make this AMI
custom_ami_dict = {
   "us-east-1": "ami-67c4871d",
   "us-west-2": "ami-47e04e3f",
}

AWS_INSTALL_SCRIPT="""
source activate mxnet_p36  # env with cuda 9
"""

TMUX_INSTALL_SCRIPT="""
source activate cifar
%upload *.py
%upload *.tfrecords
"""

REDIS_PORT = 6379  # default redis port
SCRIPT_NAME='ray_pp.py'

def launch_tmux(backend, install_script):
  num_tasks = 1 + args.num_workers + args.num_ps
  run = backend.make_run(args.name, install_script=install_script)
  ray_job = run.make_job('worker', num_tasks)
  tb_job = run.make_job('tb', 1)

  # task 0 is ray head node, also it is client node where main script runs
  head_task = ray_job.tasks[0]
  head_task.run('ray stop   || echo "ray not started, ignoring"')
  head_task.run("ray start --head --redis-port=%d --num-gpus=0 \
                           --num-cpus=10000 --num-workers=10"%(REDIS_PORT,))


  
  # for task in ray_job.tasks[1:]:
  #   task.run('ray stop || echo "ray not started, ignoring"')
  #   task.run("ray start --redis-address %s:%d --num-gpus=1 --num-cpus=1 --num-workers=0" % (head_task.ip, REDIS_PORT))
  ray_job.tasks[1].run("ray start --redis-address %s:%d --num-gpus=%d --num-cpus=%d --num-workers=%d" % (head_task.ip, REDIS_PORT, num_tasks, num_tasks, num_tasks))
    
  head_task.upload(SCRIPT_NAME)
  #  head_task.upload('../util.py')
  head_task.run_async("python {script} \
                    --redis-address={redis_ip}:{redis_port} \
                    --num-workers={num_workers} \
                    --num-parameter-servers={num_ps} \
  --logdir={logdir} \
                    --dim={dim} --insert-pauses".format(script=SCRIPT_NAME,
                                        redis_ip=head_task.ip,
                                        redis_port=REDIS_PORT,
                                        num_workers=args.num_workers,
                                        num_ps=args.num_ps,
                                        logdir=run.logdir,
                                        dim=args.dim))
  print("Connect to head node:")
  print(head_task.connect_instructions)

  print("Other nodes:")
  for (i, task) in enumerate(ray_job.tasks[1:]):
    print(i, task.connect_instructions)
    
  tb_cmd = "tensorboard --logdir={logdir} --port=6006".format(logdir=run.logdir)
  tb_job.run(tb_cmd, sync=False)
  print("See tensorboard at http://%s:%s"%(tb_job.ip, 6006))

def launch_aws(backend, install_script):
  region = u.get_region()
  ami = custom_ami_dict[region]
  
  num_tasks = 1 + args.num_workers + args.num_ps
  run = backend.make_run(args.name, install_script=install_script,
                         ami=ami, availability_zone=args.zone)
  ray_job = run.make_job('worker', num_tasks,
                         instance_type=args.gpu_instance_type)
  tb_job = run.make_job('tb', 1, instance_type=args.tb_instance_type)
  ray_job.wait_until_ready()
  tb_job.wait_until_ready()

  ray_job.run('source activate mxnet_p36')
  tb_job.run('source activate mxnet_p36')
  
  # task 0 is ray head node, also it is client node where main script runs
  head_task = ray_job.tasks[0]
  head_task.run('ray stop || echo "ray not started, ignoring"')
  head_task.run("ray start --head --redis-port=%d --num-gpus=0 \
                           --num-cpus=10000 --num-workers=10"%(REDIS_PORT,))
  
  for task in ray_job.tasks[1:]:
    task.run('ray stop || echo "ray not started, ignoring"')
    task.run("ray start --redis-address %s:%d --num-gpus=1 --num-cpus=1 --num-workers=0" % (head_task.ip, REDIS_PORT))
    
  head_task.upload(SCRIPT_NAME)
  #  head_task.upload('../util.py')
  head_task.run_async("python {script} \
                    --redis-address={redis_ip}:{redis_port} \
                    --num-workers={num_workers} \
                    --num-parameter-servers={num_ps} \
                    --dim={dim} \
                    --real-model \
                    --logdir={logdir}".format(script=SCRIPT_NAME,
                                        redis_ip=head_task.ip,
                                        redis_port=REDIS_PORT,
                                        num_workers=args.num_workers,
                                                    logdir=run.logdir,
                                        num_ps=args.num_ps,
                                                    dim=args.dim))
  print("Connect to head node:")
  print(head_task.connect_instructions)

  print("Other nodes:")
  for (i, task) in enumerate(ray_job.tasks[1:]):
    print(i, task.connect_instructions)
    

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
  
