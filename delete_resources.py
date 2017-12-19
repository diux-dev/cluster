#!/usr/bin/env python
#
# Deletes resources

import os
import sys
import boto3

ami_dict = {
    "us-west-1": "ami-a089b2c0", #"ami-45ead225",
    "us-east-1": "ami-08c35d72",
    "eu-west-2": "ami-4a4b9232",
}
LINUX_TYPE = "ubuntu"  # linux type determines username to use to login

import os
import argparse
import boto3
import time
from collections import OrderedDict

parser = argparse.ArgumentParser(description='launch simple')
parser.add_argument('--name', type=str, default='nexus',
                     help="name to use for resources")
parser.add_argument('--instance_type', type=str, default='t2.micro',
                     help="type of instance")
args = parser.parse_args()

brand=args.name
VPC_NAME=brand
SECURITY_GROUP_NAME=brand
ROUTE_TABLE_NAME=brand
KEYPAIR_LOCATION=os.environ["HOME"]+'/.'+brand+'.pem'
KEYPAIR_NAME=brand
EFS_NAME=brand

import common_resources as c

def main():
  existing_vpcs = c.get_vpc_dict()
  client = c.create_ec2_client()
  ec2 = c.create_ec2_resource()
  
  if VPC_NAME not in existing_vpcs:
    print("VPC %s doesn't exist" %(VPC_NAME,))
    return
  
  vpc = ec2.Vpc(existing_vpcs[VPC_NAME].id)
  print("Deleting VPC %s (%s) subresources:"%(VPC_NAME, vpc.id))

  def response_type(response):
    return 'ok' if c.is_good_response(response) else 'failed'

  for subnet in vpc.subnets.all():
    try:
      sys.stdout.write("Deleting subnet %s ... " % (subnet.id))
      sys.stdout.write(response_type(subnet.delete())+'\n')
    except:
      sys.stdout.write('failed\n')

  for gateway in vpc.internet_gateways.all():
    sys.stdout.write("Deleting gateway %s ... " % (gateway.id))
    sys.stdout.write(' detached ... ' if c.is_good_response(gateway.detach_from_vpc(VpcId=vpc.id)) else ' detach_failed ')
    sys.stdout.write(' deleted ' if c.is_good_response(gateway.delete()) else ' delete_failed ')
    sys.stdout.write('\n')

  for route_table in vpc.route_tables.all():
    sys.stdout.write("Deleting route table %s ... " % (route_table.id))
    try:
      sys.stdout.write(response_type(route_table.delete())+'\n')
    except Exception as e:
      sys.stdout.write('failed\n')

  for security_group in vpc.security_groups.all():
    sys.stdout.write('Deleting security group %s ... ' %(security_group.id,))
    try:
      sys.stdout.write(response_type(security_group.delete())+'\n')
    except Exception as e:
      sys.stdout.write('failed\n')
      
  sys.stdout.write("Deleting VPC %s ... " % (vpc.id))
  sys.stdout.write(response_type(vpc.delete())+'\n')

  # todo: also delete security key pair


if __name__=='__main__':
  main()
