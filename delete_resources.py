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

import util as u

parser = argparse.ArgumentParser()
parser.add_argument('--kind', type=str, default='all',
                    help=("which resources to delete, all/network/keypair/efs"))
parser.add_argument('--force-delete-efs', action='store_true',
                    help="force deleting main EFS")

args = parser.parse_args()

EFS_NAME=u.get_resource_name()
VPC_NAME=u.get_resource_name()
SECURITY_GROUP_NAME=u.get_resource_name()
ROUTE_TABLE_NAME=u.get_resource_name()
KEYPAIR_NAME=u.get_keypair_name()
EFS_NAME=u.get_resource_name()

client = u.create_ec2_client()
ec2 = u.create_ec2_resource()

def response_type(response):
  return 'ok' if u.is_good_response(response) else 'failed'

def delete_efs():
  efss = u.get_efs_dict()
  efs_id = efss.get(EFS_NAME, '')
  efs_client = u.create_efs_client()
  if efs_id:
    try:
      # delete mount targets first
      print("About to delete %s (%s)" % (efs_id, EFS_NAME))
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


      sys.stdout.write('Deleting EFS %s (%s)... ' %(efs_id, EFS_NAME))
      sys.stdout.flush()
      u.delete_efs_id(efs_id)

    except Exception as e:
      sys.stdout.write('failed\n')
      u.loge(str(e)+'\n')

def delete_network():
  existing_vpcs = u.get_vpc_dict()
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
      # todo: if instances are using VPC, this fails with
      # botocore.exceptions.ClientError: An error occurred (DependencyViolation) when calling the DetachInternetGateway operation: Network vpc-ca4abab3 has some mapped public address(es). Please unmap those public address(es) before detaching the gateway.

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
    # TODO: this tries to remove default security group, maybe not remove it?
    for security_group in vpc.security_groups.all():
      sys.stdout.write('Deleting security group %s ... ' %(desc(security_group)))
      try:
        sys.stdout.write(response_type(security_group.delete())+'\n')
      except Exception as e:
        sys.stdout.write('failed\n')
        u.loge(str(e)+'\n')

    sys.stdout.write("Deleting VPC %s ... " % (vpc.id))
    try:
      sys.stdout.write(response_type(vpc.delete())+'\n')
    except Exception as e:
      sys.stdout.write('failed\n')
      u.loge(str(e)+'\n')
  
def delete_keypair():
  keypairs = u.get_keypair_dict()
  keypair = keypairs.get(KEYPAIR_NAME, '')
  if keypair:
    try:
      sys.stdout.write("Deleting keypair %s (%s) ... " % (keypair.key_name,
                                                          KEYPAIR_NAME))
      sys.stdout.write(response_type(keypair.delete())+'\n')
    except Exception as e:
      sys.stdout.write('failed\n')
      u.loge(str(e)+'\n')

  keypair_fn = u.get_keypair_fn()
  if os.path.exists(keypair_fn):
    print("Deleting local keypair file %s" % (keypair_fn,))
    os.system('rm -f '+keypair_fn)

def delete_resources():
  # TODO: also bring down all the instances and wait for them to come down
  region = os.environ['AWS_DEFAULT_REGION']

  resource = u.get_resource_name()
  print("Deleting %s resources in region %s"%(resource, region,))
  print("Make sure all connected instances are terminated or this will fail.")
  
  if 'efs' in args.kind or 'all' in args.kind:
    if EFS_NAME == u.DEFAULT_RESOURCE_NAME and not args.force_delete_efs:
      # this is default EFS, likely has stuff, require extra flag to delete it
      print("Nexus EFS has useful stuff in it, not deleting it")
    else:
      delete_efs()
  if 'network' in args.kind or 'all' in args.kind:
    delete_network()
  if 'keypair' in args.kind or 'all' in args.kind:
    delete_keypair()


if __name__=='__main__':
  delete_resources()
