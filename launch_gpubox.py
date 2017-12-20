#!/usr/bin/env python
#
# Launch a single GPU instance
#
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


# instance name restrictions
# http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_Tags.html#tag-restrictions
from collections import OrderedDict
import argparse
import boto3
import os
import time
import util as u

import aws

parser = argparse.ArgumentParser(description='launch simple')
parser.add_argument('--name', type=str, default='gpubox00',
                     help="name of the current run")
parser.add_argument('--instance_type', type=str, default='p2.8xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='us-east-1c',
                    help='which availability zone to use')
args = parser.parse_args()

# region is taken from environment variable AWS_DEFAULT_REGION
assert 'AWS_DEFAULT_REGION' in os.environ
assert os.environ.get("AWS_DEFAULT_REGION") in ami_dict

INSTALL_SCRIPT="""
"""
  
def main():
  ami = ami_dict[os.environ.get("AWS_DEFAULT_REGION")]
  vpc = u.get_vpc_dict()['nexus']
  
  subnets = list(vpc.subnets.all())
  if not subnets:
    print("<no subnets>, failing")
    sys.exit()
  else:
    subnets = list(vpc.subnets.all())
    subnet_dict = {}
    for subnet in subnets:
      zone = subnet.availability_zone
      assert zone not in subnet_dict, "More than one subnet in %s, why?" %(zone,)
      subnet_dict[zone] = subnet
      #      print("%-16s %-16s"%(subnet.id, subnet.availability_zone))
      
    subnet = subnet_dict[args.zone]
    print("Chose %-16s %-16s"%(subnet.id, subnet.availability_zone))

  zone = subnet.availability_zone
  security_group = u.get_security_group_dict()['nexus']
  keypair = u.get_keypair_dict()['nexus']
    
  job = aws.server_job(args.name, ami=ami, num_tasks=1,
                       instance_type=args.instance_type,
                       install_script=INSTALL_SCRIPT,
                       availability_zone=zone)

  job.wait_until_ready()
  task = job.tasks[0]
  task.run('python --version')

  # connect instructions
  print("To connect:")
  print(task.connect_instructions)


if __name__=='__main__':
  main()
