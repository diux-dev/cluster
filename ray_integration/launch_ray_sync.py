#!/usr/bin/env python

# Launch Ray distributed benchmark from ray_sync.py
# https://gist.github.com/robertnishihara/87aa7a9a68ef8fa0f3184129346cffc3
SCRIPT_NAME='ray_sync.py'

# This are setup commands to run on each AWS instance
INSTALL_SCRIPT="""
tmux set-option -g history-limit 250000

rm -f /tmp/install_finished
sudo apt update -y

# install ray and dependencies
sudo apt install -y pssh

# make Python3 the default
sudo apt install -y wget
sudo apt install -y python3
sudo apt install -y python3-pip

sudo mv /usr/bin/pip /usr/bin/pip.old || echo      # || echo to ignore errors
sudo ln -s /usr/bin/pip3 /usr/bin/pip
sudo mv /usr/bin/python /usr/bin/python.old || echo
sudo ln -s /usr/bin/python3 /usr/bin/python

pip install https://s3-us-west-2.amazonaws.com/ray-wheels/20d6b74aa6c034fdf35422e6805e2283c672e03f/ray-0.3.0-cp35-cp35m-manylinux1_x86_64.whl
pip install numpy
pip install jupyter ipywidgets bokeh

ray stop   || echo "ray not started, ignoring"
"""

REDIS_PORT = 6379  # default redis port

import os
import sys
import argparse
import time
parser = argparse.ArgumentParser(description='Ray parameter server experiment')

# turn off placement for less maintainance
# 1. placement groups need to periodically get deleted
# 2. launching into placement group fails when an existing instance in it is "terminating"
parser.add_argument('--placement', type=int, default=0,
                     help=("whether to launch workers inside a placement "
                           "group"))
parser.add_argument('--instance_type', type=str, default='c5.large',
                    help='default instance type')
parser.add_argument("--num-workers", default=2, type=int,
                    help="The number of gradient workers to use.")
parser.add_argument("--num-ps", default=2, type=int,
                    help="The number of parameter servers workers to use.")
parser.add_argument("--dim", default=25000, type=int,
                    help="Number of parameters.")
parser.add_argument('--name', type=str, default='sync',
                     help="name of the current run")
args = parser.parse_args()


def main():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import aws
  
  # job launches are asynchronous, can spin up multiple jobs in parallel
  if args.placement:
    placement_name = args.name
  else:
    placement_name = ''
  print("Launching job")
  num_tasks = 1 + args.num_workers + args.num_ps
  job = aws.simple_job(args.name, num_tasks=num_tasks,
                       instance_type=args.instance_type,
                       install_script=INSTALL_SCRIPT,
                       placement_group=placement_name)

  # block until things launch to run commands
  job.wait_until_ready()
  
  # task 0 is ray head node, also it is client node where main script runs
  head_task = job.tasks[0]
  head_task.run('ray stop   || echo "ray not started, ignoring"')
  head_task.run("ray start --head --redis-port=%d --num-gpus=0 \
                           --num-cpus=10000 --num-workers=10"%(REDIS_PORT,))
  # wait for this command to complete before running others.
  # TODO: add locking to remove need for sleep
  # https://github.com/tmux/tmux/issues/1185
  time.sleep(2)
  
  # start workers
  for task in job.tasks[1:]:
    task.run('ray stop || echo "ray not started, ignoring"')
    task.run("ray start --redis-address %s:%d --num-gpus=4 --num-cpus=4 --num-workers=0" % (head_task.ip, REDIS_PORT))

  # download benchmark script and exeucte it on head node
  head_task.run("rm -f "+SCRIPT_NAME) # todo: remove?
  head_task.upload(SCRIPT_NAME)
  head_task.run("python {script} \
                    --redis-address={redis_ip}:{redis_port} \
                    --num-workers={num_workers} \
                    --num-parameter-servers={num_ps} \
                    --dim=25000".format(script=SCRIPT_NAME,
                                        redis_ip=head_task.ip,
                                        redis_port=REDIS_PORT,
                                        num_workers=args.num_workers,
                                        num_ps=args.num_ps,
                                        dim=args.dim))

  print ("Connect to head node:")
  print(head_task.connect_instructions)

  print("Other nodes:")
  for (i, task) in enumerate(job.tasks[1:]):
    print(i, task.connect_instructions)
  
if __name__=='__main__':
  main()
