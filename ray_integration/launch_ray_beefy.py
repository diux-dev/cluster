#!/usr/bin/env python

# Launch Ray distributed benchmark from the following URL
BENCHMARK_URL="https://gist.githubusercontent.com/robertnishihara/24979fb01b4b70b89e5cf9fbbf9d7d65/raw/b2d3bb66e881034039fbd244d7f72c5f6b425235/async_sgd_benchmark_multinode.py"

# This are setup commands to run on each AWS instance
INSTALL_SCRIPT="""
sudo apt update

# install ray and dependencies
sudo apt install -y pssh

# make Python3 the default
sudo apt install -y wget
sudo apt install -y python3
sudo apt install -y python3-pip
sudo ln -s /usr/bin/pip3 /usr/bin/pip
sudo mv /usr/bin/python /usr/bin/python.old
sudo ln -s /usr/bin/python3 /usr/bin/python

pip install ray
pip install numpy
pip install jupyter
"""

DEFAULT_PORT = 6379  # default redis port

import argparse
parser = argparse.ArgumentParser(description='ImageNet experiment')
parser.add_argument('--placement', type=int, default=1,
                     help=("whether to launch workers inside a placement "
                           "group"))
parser.add_argument('--run', type=str, default='beefy',
                     help="name of the current run")

args = parser.parse_args()


def main():
  import aws
  import os
  
  # job launches are asynchronous, can spin up multiple jobs in parallel
  if args.placement:
    placement_name = args.run
  else:
    placement_name = ''
  job = aws.simple_job(args.run, num_tasks=2,
                       instance_type='c5.18xlarge',
                       install_script=INSTALL_SCRIPT,
                       placement_group=placement_name)

  # block until things launch to run commands
  job.wait_until_ready()

  head_task = job.tasks[0]
  head_task.run("ray start --head --redis-port=%d --num-gpus=0 --num-cpus=10000 --num-workers=10"%(DEFAULT_PORT,))

  slave_task = job.tasks[1]
  script_name = os.path.basename(BENCHMARK_URL)
  slave_task.run("rm -f "+script_name)
  slave_task.run("wget "+BENCHMARK_URL)
  
  slave_task.run("ray start --redis-address %s:%d --num-gpus=4 --num-cpus=4 --num-workers=0"%(head_task.ip, DEFAULT_PORT))
  slave_task.run("python async_sgd_benchmark_multinode.py --redis-address=%s:%d --num-workers=10 --num-parameter-servers=4 --data-size=100000000"%(head_task.ip, DEFAULT_PORT))

  print ("To see results:")
  print(slave_task.connect_instructions)
  
if __name__=='__main__':
  main()
