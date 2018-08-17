#!/usr/bin/env python
#
# Creates resources
# This script creates VPC/security group/keypair if not already present

import os
import argparse
import boto3
import sys
import time
from collections import OrderedDict

import util as u

DRYRUN=False
DEBUG=True

# Names of Amazon resources that are created. These settings are fixed across
# all runs, and correspond to resources created once per user per region.

# todo: move these out of default namespace?
DEFAULT_NAME=u.get_resource_name()
VPC_NAME=u.get_resource_name()
SECURITY_GROUP_NAME=u.get_resource_name()
ROUTE_TABLE_NAME=u.get_resource_name()
KEYPAIR_NAME=u.get_keypair_name()
EFS_NAME=u.get_resource_name()

PUBLIC_TCP_RANGES = [
  22, # ssh 
  # ipython notebook ports
  (8888, 8899),
  # redis port
  6379,              
   # tensorboard ports
  (6006, 6016)
]

PUBLIC_UDP_RANGES = [(60000,60100)] # mosh ports

# region is taken from environment variable AWS_DEFAULT_REGION
# assert 'AWS_DEFAULT_REGION' in os.environ
#assert os.environ['AWS_DEFAULT_REGION'] in {'us-east-2','us-east-1','us-west-1','us-west-2','ap-south-1','ap-northeast-2','ap-southeast-1','ap-southeast-2','ap-northeast-1','ca-central-1','eu-west-1','eu-west-2','sa-east-1'}


import util as u

def network_setup():
  """Creates VPC if it doesn't already exists, configures it for public
  internet access, returns vpc, subnet, security_group"""

  # from https://gist.github.com/nguyendv/8cfd92fc8ed32ebb78e366f44c2daea6

  ec2 = u.create_ec2_resource()
  existing_vpcs = u.get_vpc_dict()
  zones = u.get_available_zones()
  if VPC_NAME in existing_vpcs:
    print("Reusing VPC "+VPC_NAME)
    vpc = existing_vpcs[VPC_NAME]
    subnets = list(vpc.subnets.all())
    assert len(subnets) == len(zones), "Has %s subnets, but %s zones, something went wrong during resource creation, try delete_resources.py/create_resources.py"%(len(subnets), len(zones))
    
  else:
    print("Creating VPC "+VPC_NAME)
    vpc = ec2.create_vpc(CidrBlock='192.168.0.0/16')

    # enable DNS on the VPC
    response = vpc.modify_attribute(EnableDnsHostnames={"Value":True})
    assert u.is_good_response(response)
    response = vpc.modify_attribute(EnableDnsSupport={"Value":True})
    assert u.is_good_response(response)
    
    vpc.create_tags(Tags=u.make_name(VPC_NAME))
    vpc.wait_until_available()

  gateways = u.get_gateway_dict(vpc)
  if DEFAULT_NAME in gateways:
    print("Reusing gateways "+DEFAULT_NAME)
  else:
    print("Creating gateway "+DEFAULT_NAME)
    ig = ec2.create_internet_gateway()
    ig.attach_to_vpc(VpcId=vpc.id)
    ig.create_tags(Tags=u.make_name(DEFAULT_NAME))

    # check that attachment succeeded
    # TODO: sometimes get
    # AssertionError: vpc vpc-33d0804b is in state None

    attach_state = u.get1(ig.attachments, State=-1, VpcId=vpc.id)
    assert attach_state == 'available', "vpc %s is in state %s"%(vpc.id,
                                                                 attach_state)
    
    
    route_table = vpc.create_route_table()
    route_table.create_tags(Tags=u.make_name(ROUTE_TABLE_NAME))

    dest_cidr = '0.0.0.0/0'
    route = route_table.create_route(
      DestinationCidrBlock=dest_cidr,
      GatewayId=ig.id
    )
    # check success
    for route in route_table.routes:
      # result looks like this
      # ec2.Route(route_table_id='rtb-a8b438cf',
      #    destination_cidr_block='0.0.0.0/0')
      if route.destination_cidr_block == dest_cidr:
        break
    else:
      # sometimes get
      #      AssertionError: Route for 0.0.0.0/0 not found in [ec2.Route(route_table_id='rtb-cd9153b0', destination_cidr_block='192.168.0.0/16')]
      # TODO: add a wait/retry?
      assert False, "Route for %s not found in %s"%(dest_cidr,
                                                    route_table.routes)

    assert len(zones)<=16  # for cidr/20 to fit into cidr/16
    ip = 0
    for zone in zones:
      cidr_block = '192.168.%d.0/20'%(ip,)
      ip+=16
      print("Creating subnet %s in zone %s"%(cidr_block, zone))
      subnet = vpc.create_subnet(CidrBlock=cidr_block,
                                 AvailabilityZone=zone)
      subnet.create_tags(Tags=[{'Key':'Name','Value':f'{VPC_NAME}-subnet'}, {'Key':'Region','Value':zone}])
      u.wait_until_available(subnet)
      route_table.associate_with_subnet(SubnetId=subnet.id)

  # Creates security group if necessary
  existing_security_groups = u.get_security_group_dict()
  if SECURITY_GROUP_NAME in existing_security_groups:
    print("Reusing security group "+SECURITY_GROUP_NAME)
    security_group = existing_security_groups[SECURITY_GROUP_NAME]
  else:
    print("Creating security group "+SECURITY_GROUP_NAME)
    security_group = ec2.create_security_group(
      GroupName=SECURITY_GROUP_NAME, Description=SECURITY_GROUP_NAME,
      VpcId=vpc.id)

    security_group.create_tags(Tags=[{"Key": "Name",
                                      "Value": SECURITY_GROUP_NAME}])

    # allow ICMP access for public ping
    security_group.authorize_ingress(
      CidrIp='0.0.0.0/0',
      IpProtocol='icmp',
      FromPort=-1,
      ToPort=-1
    )

    # open public ports
    # always include SSH port which is required for basic functionality
    assert 22 in PUBLIC_TCP_RANGES, "Must enable SSH access"
    for port in PUBLIC_TCP_RANGES:
      if u.is_list_or_tuple(port):
        assert len(port) == 2
        from_port, to_port = port
      else:
        from_port, to_port = port, port
        
      response = security_group.authorize_ingress(IpProtocol="tcp",
                                                  CidrIp="0.0.0.0/0",
                                                  FromPort=from_port,
                                                  ToPort=to_port)
      assert u.is_good_response(response)

    for port in PUBLIC_UDP_RANGES:
      if u.is_list_or_tuple(port):
        assert len(port) == 2
        from_port, to_port = port
      else:
        from_port, to_port = port, port
        
      response = security_group.authorize_ingress(IpProtocol="udp",
                                                  CidrIp="0.0.0.0/0",
                                                  FromPort=from_port,
                                                  ToPort=to_port)
      assert u.is_good_response(response)


    # allow ingress within security group
    # Authorizing ingress doesn't work with names in a non-default VPC,
    # so must use more complicated syntax
    # https://github.com/boto/boto3/issues/158

    for protocol in ['icmp']:
      try:
        rule ={'FromPort': -1,
               'IpProtocol': protocol,
               'IpRanges': [],
               'PrefixListIds': [],
               'ToPort': -1,
               'UserIdGroupPairs': [{'GroupId': security_group.id}]}
        security_group.authorize_ingress(IpPermissions=[rule])
      except Exception as e:
        if e.response['Error']['Code']=='InvalidPermission.Duplicate':
          print("Warning, got "+str(e))
        else:
          assert False, "Failed while authorizing ingress with "+str(e)

    for protocol in ['tcp', 'udp']:
      try:
        rule ={'FromPort': 0,
               'IpProtocol': protocol,
               'IpRanges': [],
               'PrefixListIds': [],
               'ToPort': 65535,
               'UserIdGroupPairs': [{'GroupId': security_group.id}]}
        security_group.authorize_ingress(IpPermissions=[rule])
      except Exception as e:
        if e.response['Error']['Code']=='InvalidPermission.Duplicate':
          print("Warning, got "+str(e))
        else:
          assert False, "Failed while authorizing ingress with "+str(e)
      
  return vpc, security_group


def keypair_setup():
  """Creates keypair if necessary, saves private key locally, returns contents
  of private key file."""

  os.system('mkdir -p '+u.PRIVATE_KEY_LOCATION)
  
  keypair = u.get_keypair_dict().get(KEYPAIR_NAME, None)
  keypair_fn = u.get_keypair_fn()
  if keypair:
    print("Reusing keypair "+KEYPAIR_NAME)
    # check that local pem file exists and is readable
    assert os.path.exists(keypair_fn), "Keypair %s exists, but corresponding .pem file %s is not found, delete keypair %s through console and run again to recreate keypair/.pem together"%(KEYPAIR_NAME, keypair_fn, KEYPAIR_NAME)
    keypair_contents = open(keypair_fn).read()
    assert len(keypair_contents)>0
    # todo: check that fingerprint matches keypair.key_fingerprint
  else:
    print("Creating keypair "+KEYPAIR_NAME)
    ec2 = u.create_ec2_resource()
    assert not os.path.exists(keypair_fn), "previous keypair exists, delete it with 'sudo rm %s' and also delete corresponding keypair through console"%(keypair_fn)
    keypair = ec2.create_key_pair(KeyName=KEYPAIR_NAME)

    open(keypair_fn, 'w').write(keypair.key_material)
    os.system('chmod 400 '+keypair_fn)
    
  return keypair


def placement_group_setup(group_name):
  """Creates placement group if necessary. Returns True if new placement
  group was created, False otherwise."""
  
  existing_placement_groups = u.get_placement_group_dict()

  group = existing_placement_groups.get(group_name, None)
  if group:
    assert group.state == 'available'
    assert group.strategy == 'cluster'
    print("Reusing group ", group.name)
    return group

  print("Creating group "+group_name)
  ec2 = u.create_ec2_resource()
  group = ec2.create_placement_group(GroupName=group_name, Strategy='cluster')
  return group

  
def create_resources():

  region = u.get_region()
  print("Creating %s resources in region %s"%(DEFAULT_NAME, region,))

  vpc, security_group = network_setup()
  keypair = keypair_setup()  # saves private key locally to keypair_fn

  # create EFS
  efss = u.get_efs_dict()
  efs_id = efss.get(DEFAULT_NAME, '')
  if not efs_id:
    print("Creating EFS "+DEFAULT_NAME)
    efs_id = u.create_efs(DEFAULT_NAME)
  else:
    print("Reusing EFS "+DEFAULT_NAME)
    
  efs_client = u.create_efs_client()

  # create mount target for each subnet in the VPC

  # added retries because efs is not immediately available
  MAX_FAILURES = 10
  RETRY_INTERVAL_SEC = 1
  for subnet in vpc.subnets.all():
    for retry_attempt in range(MAX_FAILURES):
      try:
        sys.stdout.write("Creating efs mount target for %s ... "%(subnet.availability_zone,))
        sys.stdout.flush()
        response = efs_client.create_mount_target(FileSystemId=efs_id,
                                                  SubnetId=subnet.id,
                                                  SecurityGroups=[security_group.id])
        if u.is_good_response(response):
          print("success")
          break
      except Exception as e:
        if 'already exists' in str(e): # ignore "already exists" errors
          print('already exists')
          break

        # Takes couple of seconds for EFS to come online, with
        # errors like this:
        # Creating efs mount target for us-east-1f ... Failed with An error occurred (IncorrectFileSystemLifeCycleState) when calling the CreateMountTarget operation: None, retrying in 1 sec

        print("Got %s, retrying in %s sec"%(str(e), RETRY_INTERVAL_SEC))
        time.sleep(RETRY_INTERVAL_SEC)
    else:
      print("Giving up.")

if __name__=='__main__':
  create_resources()

