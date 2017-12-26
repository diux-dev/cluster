#!/usr/bin/env python
#
# Launch a single GPU instance with Amazon Deep Learning AMI
# ./launch_gpubox.py --instance-type=g3.4xlarge --zone=us-east-1f
#
#
# https://aws.amazon.com/blogs/ai/new-aws-deep-learning-amis-for-machine-learning-practitioners/
#
# Ubuntu version:
# Ubuntu Conda based Amazon Deep Learning AMI
# US East (N. Virginia)	ami-9ba7c4e1
# US East (Ohio)        ami-69f6df0c
# US West (Oregon)	ami-3b6bce43
# EU (Ireland)	        ami-5bf34b22
# Asia Pacific (Sydney)	ami-4fa3542d
# Asia Pacific (Seoul)	ami-7d872113
# Asia Pacific (Tokyo)	ami-90b432f6

# Amazon Linux version:
# https://aws.amazon.com/marketplace/fulfillment?productId=f3afce45-648d-47d7-9f6c-1ec273f7df70&ref_=dtl_psb_continue
# US East (N. Virginia)        ami-3a533040
# US East (Ohio)               ami-72f4dd17
# US West (Oregon)             ami-5c60c524
# EU (Frankfurt)               ami-88aa23e7
# EU (Ireland)                 ami-70fe4609
# Asia Pacific (Singapore)     ami-0798fc7b
# Asia Pacific (Sydney)	       ami-38a5525a
# Asia Pacific (Seoul)	       ami-a4b91fca
# Asia Pacific (Tokyo)	       ami-98ad2bfe
# Asia Pacific (Mumbai)	       ami-6ce8a103

# TODO: determine availability zones automatically
availability_mapping_us_east_1 = {'g3': ['us-east-1a', 'us-east-1b', 'us-east-1e', 'us-east-1c'],
                                  'p2': ['us-east-1f'],
                                  'p3': ['us-east-1f']}
availability_mapping_us_west_2 = {'g3': ['us-west-2a'],
                                  'p2': ['us-west-2a', 'us-west-2b'],
                                  'p3': ['us-west-2b', 'us-west-2c']}
availability_mapping = {'us-east-1': availability_mapping_us_east_1,
                        'us-west-2': availability_mapping_us_west_2}

ami_dict_ubuntu = {
    "us-west-2": "ami-3b6bce43",
    "us-east-1": "ami-9ba7c4e1",
}
ami_dict_amazon = {
    "us-west-2": "ami-3b6bce43",
    "us-east-1": "ami-9ba7c4e1",
}

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
parser.add_argument('--instance-type', type=str, default='p2.xlarge',
                     help="type of instance")
parser.add_argument('--zone', type=str, default='',
                    help='which availability zone to use')
parser.add_argument('--linux-type', type=str, default='ubuntu',
                    help='which linux to use: ubuntu or amazon')

args = parser.parse_args()

# region is taken from environment variable AWS_DEFAULT_REGION
assert 'AWS_DEFAULT_REGION' in os.environ

# adding new regions require adding entries to ami_dict and
# availability_zone_mapping
assert os.environ.get("AWS_DEFAULT_REGION") in ['us-east-1', 'us-west-2']

# TODO(y): add locking, install fails while main script is installing things
# P2 instances: sudo nvidia-smi -ac 2505,875
# P3 instances: sudo nvidia-smi -ac 877,1530
# G3 instances: sudo nvidia-smi -ac 2505,1177
# sudo nvidia-smi -ac 877,1530

INSTALL_SCRIPT_UBUNTU="""
python --version
sudo mkdir -p /efs
sudo chmod 777 /efs
"""

INSTALL_SCRIPT_AMAZON="""
python --version
sudo mkdir -p /efs
sudo chmod 777 /efs
"""

def main():
  if args.linux_type == 'ubuntu':
    install_script = INSTALL_SCRIPT_UBUNTU
    ami_dict = ami_dict_ubuntu
  elif args.linux_type == 'amazon':
    install_script = INSTALL_SCRIPT_AMAZON
    ami_dict = ami_dict_amazon
  else:
    assert False, "Unknown linux type "+args.linux_type

  region = os.environ.get("AWS_DEFAULT_REGION")
  ami = ami_dict[region]
  
  # #  vpc = u.get_vpc_dict()[u.RESOURCE_NAME]

  # # pick AZ to use for instance based on available subnets
  # subnets = list(vpc.subnets.all())
  # if not subnets:
  #   print("<no subnets>, failing")
  #   sys.exit()
  # subnets = list(vpc.subnets.all())
  # subnet_dict = {}
  # for subnet in subnets:
  #   zone = subnet.availability_zone
  #   assert zone not in subnet_dict, "More than one subnet in %s, why?" %(zone,)
  #   subnet_dict[zone] = subnet

  if not args.zone:
    machine_class = args.instance_type[:2]
    zone = availability_mapping[region][machine_class][0]
    print("Chose %s based on availability mapping for %s"%(zone,
                                                           machine_class))
  else:
    zone = args.zone

    #  subnet = subnet_dict[zone]
    #  print("Available zones: %s" %(', '.join(sorted(subnet_dict.keys()))))
    #  print("Using %-16s %-16s"%(subnet.id, subnet.availability_zone))

  print("Launching %s in %s" %(args.name, zone))
  security_group = u.get_security_group_dict()[u.RESOURCE_NAME]
  keypair = u.get_keypair_dict()[u.RESOURCE_NAME]
    
  job = aws.server_job(args.name, ami=ami, num_tasks=1,
                       instance_type=args.instance_type,
                       install_script=install_script,
                       availability_zone=zone)

  job.wait_until_ready()
  task = job.tasks[0]

  # this needs DNS to be enabled on VPC
  # alternative way is to provide direct IP from efs_tool.py
  efs_id = u.get_efs_dict()[u.RESOURCE_NAME]
  dns = "{efs_id}.efs.{region}.amazonaws.com".format(**locals())

  # try mounting EFS several times
  for i in range(3):
    try:
      task.run("sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 %s:/ /efs && sudo chmod 777 /efs"%(dns,))
      print("EFS Mount succeeded")
      break
    except Exception as e:
      print("Got error %s, retrying in 10 seconds"%(str(e)))
      time.sleep(10)
      
  # connect instructions
  print("To connect:")
  print(task.connect_instructions)


if __name__=='__main__':
  main()
