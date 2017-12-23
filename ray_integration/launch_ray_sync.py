#!/usr/bin/env python

# https://aws.amazon.com/blogs/ai/new-aws-deep-learning-amis-for-machine-learning-practitioners/

# Ubuntu Conda based Amazon Deep Learning AMI
# US East (N. Virginia)	ami-9ba7c4e1
# US East (Ohio)        ami-69f6df0c
# US West (Oregon)	ami-3b6bce43
# EU (Ireland)	        ami-5bf34b22
# Asia Pacific (Sydney)	ami-4fa3542d
# Asia Pacific (Seoul)	ami-7d872113
# Asia Pacific (Tokyo)	ami-90b432f6

ami_dict = {
    "us-west-2": "ami-3b6bce43",
    "us-east-1": "ami-9ba7c4e1",
}
LINUX_TYPE = "ubuntu"  # linux type determines username to use to login
AMI_USERNAME = 'ubuntu'  # ami-specific username needed to login
          # ubuntu for Ubuntu images, ec2-user for Amazon Linux images


# Launch Ray distributed benchmark from ray_sync.py
# https://gist.github.com/robertnishihara/87aa7a9a68ef8fa0f3184129346cffc3
SCRIPT_NAME='ray_sync.py'

# This are setup commands to run on each AWS instance
INSTALL_SCRIPT="""
tmux set-option -g history-limit 250000

# work-around for lock file being taken on initial login
# https://console.aws.amazon.com/support/v1?region=us-east-1#/case/?displayId=4755926091&language=en

ps aux | grep apt-get
pstree -a
ls -ltr /var/lib/dpkg/lock || echo "ignoring"
# sudo killall apt-get || echo "nothing running"
# sudo rm -f /var/lib/dpkg/lock


# delete confirmation file to always reinstall things
# rm -f /tmp/install_finished

# comment-out, causes "dpg lock" errors for next command
# sudo apt update -y

# more work-around for install failures
sudo apt install -y pssh || echo "failed 1"
sleep 1
sudo apt install -y pssh || echo "failed 2"
sleep 1
sudo apt install -y pssh || echo "failed 3"
sleep 1
sudo apt install -y pssh || echo "failed 4"
sleep 1
sudo apt install -y pssh || echo "failed 5"
sleep 1
sudo apt install -y pssh || echo "failed 6"
sleep 1
sudo apt install -y pssh || echo "failed 7"
sleep 1
sudo apt install -y pssh || echo "failed 8"
sleep 1
sudo apt install -y pssh || echo "failed 9"
sleep 1
sudo apt install -y pssh

# make Python3 the default
sudo apt install -y wget
sudo apt install -y python3
sudo apt install -y python3-pip

sudo mv /usr/bin/pip /usr/bin/pip.old || echo      # || echo to ignore errors
sudo ln -s /usr/bin/pip3 /usr/bin/pip
sudo mv /usr/bin/python /usr/bin/python.old || echo
sudo ln -s /usr/bin/python3 /usr/bin/python

#pip install https://s3-us-west-2.amazonaws.com/ray-wheels/20d6b74aa6c034fdf35422e6805e2283c672e03f/ray-0.3.0-cp35-cp35m-manylinux1_x86_64.whl

#pip install https://s3-us-west-2.amazonaws.com/ray-wheels/20d6b74aa6c034fdf35422e6805e2283c672e03f/ray-0.3.0-cp36-cp36m-manylinux1_x86_64.whl

# commit right before pyarrow ugprade
pip install https://s3-us-west-2.amazonaws.com/ray-wheels/772527caa484bc04169c54959b6a0c5001650bf6/ray-0.3.0-cp36-cp36m-manylinux1_x86_64.whl

#export local_services=/home/ubuntu/anaconda3/lib/python3.6/site-packages/ray/services.py
#export remote_services=https://raw.githubusercontent.com/ray-project/ray/3a301c3d56743956cfb4d0a4d30e3bd3157946e9/python/ray/services.py
# rm $local_services
# wget $remote_services -O $local_services

pip install numpy
pip install jupyter ipywidgets bokeh

ray stop   || echo "ray not started, ignoring"
"""

REDIS_PORT = 6379  # default redis port

import os
import sys
import argparse
import time

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util as u

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
parser.add_argument("--dim", default=25000000, type=int,
                    help="Number of parameters.")
parser.add_argument('--name', type=str, default='sync',
                     help="name of the current run")
parser.add_argument('--zone', type=str, default='us-east-1c',
                    help='which availability zone to use')
parser.add_argument('--local', type=int, default=0,
                    help='set to 1 to run locally')
args = parser.parse_args()


def main():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import aws
  import tmux
  
  # job launches are asynchronous, can spin up multiple jobs in parallel
  if args.placement:
    placement_name = args.name
  else:
    placement_name = ''
  print("Launching job")
  num_tasks = 1 + args.num_workers + args.num_ps

  if args.local:
    job = tmux.server_job(args.name, num_tasks=num_tasks)
  else:
    region = os.environ.get("AWS_DEFAULT_REGION")
    ami = ami_dict[region]
    job = aws.server_job(args.name, ami=ami, num_tasks=num_tasks,
                         instance_type=args.instance_type,
                         install_script=INSTALL_SCRIPT,
                         availability_zone=args.zone,
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

  if args.local:
    job.tasks[1].run("ray start --redis-address %s:%d --num-gpus=%d --num-cpus=%d --num-workers=%d" % (head_task.ip, REDIS_PORT, num_tasks, num_tasks, num_tasks))

  else:
    # start workers
    for task in job.tasks[1:]:
      task.run('ray stop || echo "ray not started, ignoring"')
      task.run("ray start --redis-address %s:%d --num-gpus=1 --num-cpus=1 --num-workers=0" % (head_task.ip, REDIS_PORT))

  # download benchmark script and execute it on head node
  head_task.run("rm -f "+SCRIPT_NAME) # todo: remove?
  head_task.upload(SCRIPT_NAME)
  # todo: make sure "dim" arg is actually getting used
  head_task.run("python {script} \
                    --redis-address={redis_ip}:{redis_port} \
                    --num-workers={num_workers} \
                    --num-parameter-servers={num_ps} \
                    --dim={dim}".format(script=SCRIPT_NAME,
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
