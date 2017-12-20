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

DEFAULT_NAME=args.name
VPC_NAME=DEFAULT_NAME
SECURITY_GROUP_NAME=DEFAULT_NAME
ROUTE_TABLE_NAME=DEFAULT_NAME
KEYPAIR_NAME=DEFAULT_NAME
EFS_NAME=DEFAULT_NAME

import util as u


def main():
  # TODO: also bring down all the instances and wait for them to come down
  region = os.environ['AWS_DEFAULT_REGION']
  print("Deleting %s resources in region %s"%(args.name, region,))
  existing_vpcs = u.get_vpc_dict()
  client = u.create_ec2_client()
  ec2 = u.create_ec2_resource()
  
  def response_type(response):
    return 'ok' if u.is_good_response(response) else 'failed'

  # delete EFS
  efss = u.get_efs_dict()
  efs_id = efss.get(DEFAULT_NAME, '')
  efs_client = u.create_efs_client()
  if efs_id:
    try:
      # delete mount targets first
      print("About to delete %s (%s)" % (efs_id, DEFAULT_NAME))
      response = efs_client.describe_mount_targets(FileSystemId=efs_id)
      assert u.is_good_response(response)
      for mount_response in response['MountTargets']:
        subnet = ec2.Subnet(mount_response['SubnetId'])
        zone = subnet.availability_zone
        state = mount_response['LifeCycleState']
        id = mount_response['MountTargetId']
        ip = mount_response['IpAddress']
        sys.stdout.write('Deleting mount target %s ... ' %(id,))
        sys.stdout.flush()
        response = efs_client.delete_mount_target(MountTargetId=id)
        print(response_type(response))


      sys.stdout.write('Deleting EFS %s (%s)... ' %(efs_id, DEFAULT_NAME))
      sys.stdout.flush()
      u.delete_efs_id(efs_id)

    except Exception as e:
      sys.stdout.write('failed\n')
      u.loge(str(e)+'\n')

  if VPC_NAME in existing_vpcs:
    vpc = ec2.Vpc(existing_vpcs[VPC_NAME].id)
    print("Deleting VPC %s (%s) subresources:"%(VPC_NAME, vpc.id))

    for subnet in vpc.subnets.all():
      try:
        sys.stdout.write("Deleting subnet %s ... " % (subnet.id))
        sys.stdout.write(response_type(subnet.delete())+'\n')
      except Exception as e:
        sys.stdout.write('failed\n')
        u.loge(str(e)+'\n')

    for gateway in vpc.internet_gateways.all():
      sys.stdout.write("Deleting gateway %s ... " % (gateway.id))
      sys.stdout.write('detached ... ' if u.is_good_response(gateway.detach_from_vpc(VpcId=vpc.id)) else ' detach_failed ')
      sys.stdout.write('deleted ' if u.is_good_response(gateway.delete()) else ' delete_failed ')
      sys.stdout.write('\n')

    def desc(route_table):
      return "%s (%s)"%(route_table.id, u.get_name(route_table.tags))
    for route_table in vpc.route_tables.all():
      sys.stdout.write("Deleting route table %s ... " % (desc(route_table)))
      try:
        sys.stdout.write(response_type(route_table.delete())+'\n')
      except Exception as e:
        sys.stdout.write('failed\n')
        u.loge(str(e)+'\n')

    def desc(security_group):
      return "%s (%s, %s)"%(security_group.id, u.get_name(security_group.tags),
                            security_group.group_name)
    for security_group in vpc.security_groups.all():
      sys.stdout.write('Deleting security group %s ... ' %(desc(security_group)))
      try:
        sys.stdout.write(response_type(security_group.delete())+'\n')
      except Exception as e:
        sys.stdout.write('failed\n')
        u.loge(str(e)+'\n')

    sys.stdout.write("Deleting VPC %s ... " % (vpc.id))
    sys.stdout.write(response_type(vpc.delete())+'\n')
  
  # delete keypair
  keypairs = u.get_keypair_dict()
  keypair = keypairs.get(DEFAULT_NAME, '')
  if keypair:
    try:
      sys.stdout.write("Deleting keypair %s (%s) ... " % (keypair.key_name,
                                                     DEFAULT_NAME))
      sys.stdout.write(response_type(keypair.delete())+'\n')
    except Exception as e:
      sys.stdout.write('failed\n')
      u.loge(str(e)+'\n')

  keypair_fn = u.get_keypair_fn(KEYPAIR_NAME)
  if os.path.exists(keypair_fn):
    print("Deleting local keypair file %s" % (keypair_fn,))
    os.system('rm -f '+keypair_fn)
    


if __name__=='__main__':
  main()
