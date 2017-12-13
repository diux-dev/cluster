#!/usr/bin/env python

# Launch Ray distributed benchmark from robert_dec12.py
SCRIPT_NAME='robert_dec12_comm.py'

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
pip install jupyter
ray stop   || echo "ray not started, ignoring"
"""

DEFAULT_PORT = 6379  # default redis port

import os
import sys
import argparse
parser = argparse.ArgumentParser(description='Ray parameter server experiment')
parser.add_argument('--placement', type=int, default=1,
                     help=("whether to launch workers inside a placement "
                           "group"))
parser.add_argument('--run', type=str, default='beefy',
                     help="name of the current run")

args = parser.parse_args()



def main():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import aws
  
  # job launches are asynchronous, can spin up multiple jobs in parallel
  if args.placement:
    placement_name = args.run
  else:
    placement_name = ''
  print("Launching job")
  job = aws.simple_job(args.run, num_tasks=3,
                       instance_type='c5.18xlarge',
                       install_script=INSTALL_SCRIPT,
                       placement_group=placement_name)

  # block until things launch to run commands
  job.wait_until_ready()
  

  # start ray on head node
  head_task = job.tasks[0]
  head_task.run('ray stop   || echo "ray not started, ignoring"')
  head_task.run("ray start --head --redis-port=%d --num-gpus=0 \
                           --num-cpus=10000 --num-workers=10"%(DEFAULT_PORT,))

  # start ray on slave node1
  slave_task1 = job.tasks[1]
  slave_task1.run('ray stop   || echo "ray not started, ignoring"')
  slave_task1.run("ray start --redis-address %s:%d --num-gpus=4 --num-cpus=4 \
                            --num-workers=0" % (head_task.ip, DEFAULT_PORT))

  # start ray on slave node2
  slave_task2 = job.tasks[2]
  slave_task2.run('ray stop   || echo "ray not started, ignoring"')
  slave_task2.run("ray start --redis-address %s:%d --num-gpus=4 --num-cpus=4 \
                            --num-workers=0" % (head_task.ip, DEFAULT_PORT))

  # download benchmark script and exeucte it on head node
  head_task.run("rm -f "+SCRIPT_NAME)
  head_task.upload(SCRIPT_NAME)
  head_task.run("python %s \
                  --num-workers=1 \
                  --num-parameter-servers=1 \
                  --dim=25000 \
                  --redis-address=%s:%d" % (SCRIPT_NAME, head_task.ip,
                                            DEFAULT_PORT))


  print ("To see results:")
  print(head_task.connect_instructions)
  
if __name__=='__main__':
  main()
