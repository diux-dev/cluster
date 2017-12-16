#!/usr/bin/env python
# Launches two workers and sets up DNS on them using Yong's recipe
# This launches two workers and sets up DNS, so that they can locate each other
# on the following addresses
#
# worker0.tf.local
# worker1.tf.local
# worker2.tf.local

import argparse
import os
import sys
import time

AMI='ami-bf4193c7'  # us-west-2 Amazon linux AMI
assert os.getenv("AWS_DEFAULT_REGION") == "us-west-2"

# Use CoreDNS setup script from this location
# https://gist.github.com/yaroslavvb/2ecf88af8b909178aa6c45148ce3057b

# This are setup commands to run on each AWS instance
INSTALL_SCRIPT="""
sudo yum install -y docker
sudo chkconfig docker on
sudo service docker start

sudo ls -l /etc/dhcp/dhclient.conf
echo "prepend domain-name-servers 127.0.0.1;" | sudo tee --append /etc/dhcp/dhclient.conf
sudo dhclient -r
sudo dhclient

wget https://gist.githubusercontent.com/yaroslavvb/2ecf88af8b909178aa6c45148ce3057b/raw/3b8636cea81289d80217f0b44941f9d7201a11f3/deploy_coredns.sh
chmod 755 deploy_coredns.sh
sudo ./deploy_coredns.sh

"""

parser = argparse.ArgumentParser(description='CoreDNS experiment')
parser.add_argument('--name', type=str, default='coredns',
                     help="run name")
parser.add_argument('--instance_type', type=str, default='c5.large',
                    help='default instance type')
parser.add_argument('--num_tasks', type=int, default=3,
                    help='default instance type')

args = parser.parse_args()


def main():
  module_path=os.path.dirname(os.path.abspath(__file__))
  sys.path.append(module_path+'/..')
  import aws

  job = aws.simple_job(args.name, num_tasks=args.num_tasks,
                       instance_type=args.instance_type,
                       install_script=INSTALL_SCRIPT,
                       ami=AMI,
                       linux_type='debian')

  # block until things launch to run commands
  job.wait_until_ready()

  for task_num,task in enumerate(job.tasks):
    for i in range(args.num_tasks):
      task.run("ping worker%d.tf.local -c 1" %(i,))
      time.sleep(1)
    print ("To connect to task %d: "%(i,))
    print(task.connect_instructions)
  
if __name__=='__main__':
  main()
