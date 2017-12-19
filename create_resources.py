#!/usr/bin/env python
#
# Creates resources
# This script creates VPC/security group/keypair if not already present

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

DRYRUN=False
DEBUG=True

# Names of Amazon resources that are created. These settings are fixed across
# all runs, and correspond to resources created once per user per region.
brand=args.name
VPC_NAME=brand
SECURITY_GROUP_NAME=brand
ROUTE_TABLE_NAME=brand
KEYPAIR_LOCATION=os.environ["HOME"]+'/.'+brand+'.pem'
KEYPAIR_NAME=brand
EFS_NAME=brand

WAIT_PERIOD_SEC=1  # how long to use for wait period
WAIT_TIMEOUT_SEC=10 # timeout after this many seconds

PUBLIC_TCP_PORTS = [8888, 8889, 8890,  # ipython notebook ports
                    6379,              # redis port
                    6006, 6007, 6008,  # tensorboard ports
]

# region is taken from environment variable AWS_DEFAULT_REGION
assert 'AWS_DEFAULT_REGION' in os.environ
assert os.environ['AWS_DEFAULT_REGION'] in {'us-east-2','us-east-1','us-west-1','us-west-2','ap-south-1','ap-northeast-2','ap-southeast-1','ap-southeast-2','ap-northeast-1','ca-central-1','eu-west-1','eu-west-2','sa-east-1'}
assert os.environ.get("AWS_DEFAULT_REGION") in ami_dict


import common_resources as c

def network_setup():
  """Creates VPC if it doesn't already exists, configures it for public
  internet access, returns vpc, subnet, security_group"""

  # from https://gist.github.com/nguyendv/8cfd92fc8ed32ebb78e366f44c2daea6
  
  existing_vpcs = c.get_vpc_dict()
  zones = c.get_available_zones()
  if VPC_NAME in existing_vpcs:
    print("Reusing VPC "+VPC_NAME)
    vpc = existing_vpcs[VPC_NAME]
    subnets = list(vpc.subnets.all())
    assert len(subnets) == len(zones), "Has %s subnets, but %s zones"%(len(subnets), len(zones))
    
  else:
    print("Creating VPC "+VPC_NAME)
    ec2 = c.create_ec2_resource()
    vpc = ec2.create_vpc(CidrBlock='192.168.0.0/16')
    vpc.create_tags(Tags=[{"Key": "Name", "Value": VPC_NAME}])
    vpc.wait_until_available()

    ig = ec2.create_internet_gateway()
    ig.attach_to_vpc(VpcId=vpc.id)

    # check that attachment succeeded
    attach_state = c.get1(ig.attachments, State=-1, VpcId=vpc.id)
    assert attach_state == 'available', "vpc %s is in state %s"%(vpc.id,
                                                                 attach_state)
    
    
    route_table = vpc.create_route_table()
    route_table.create_tags(Tags=c.make_name(ROUTE_TABLE_NAME))

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
      c.wait_until_available(subnet)
      route_table.associate_with_subnet(SubnetId=subnet.id)

  # Creates security group if necessary
  existing_security_groups = c.get_security_group_dict()
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
    for port in PUBLIC_TCP_PORTS+[22]:
      response = security_group.authorize_ingress(IpProtocol="tcp",
                                                  CidrIp="0.0.0.0/0",
                                                  FromPort=port,ToPort=port)
      assert c.is_good_response(response)

  return vpc, subnet, security_group


def keypair_setup():
  """Creates keypair if necessary, saves private key locally, returns contents
  of private key file."""
  
  
  existing_keypairs = c.get_keypair_dict()
  keypair = existing_keypairs.get(KEYPAIR_NAME, None)
  if keypair:
    print("Reusing keypair "+KEYPAIR_NAME)
    # check that local pem file exists and is readable
    assert os.path.exists(KEYPAIR_LOCATION)
    keypair_contents = open(KEYPAIR_LOCATION).read()
    assert len(keypair_contents)>0
    # todo: check that fingerprint matches keypair.key_fingerprint
    return keypair

  
  print("Creating keypair "+KEYPAIR_NAME)
  ec2 = c.create_ec2_resource()
  keypair = ec2.create_key_pair(KeyName=KEYPAIR_NAME)
  assert not os.path.exists(KEYPAIR_LOCATION), "previous, keypair exists, delete it with 'sudo rm %s'"%(KEYPAIR_LOCATION,)
  
  open(KEYPAIR_LOCATION, 'w').write(keypair.key_material)
  os.system('chmod 400 '+KEYPAIR_LOCATION)
  return keypair


def placement_group_setup(group_name):
  """Creates placement group if necessary. Returns True if new placement
  group was created, False otherwise."""
  
  existing_placement_groups = c.get_placement_group_dict()

  group = existing_placement_groups.get(group_name, None)
  if group:
    assert group.state == 'available'
    assert group.strategy == 'cluster'
    return group

  ec2 = c.create_ec2_resource()
  group = ec2.create_placement_group(GroupName=group_name, Strategy='cluster')
  return group

  
def main():
  ami = ami_dict[os.environ.get("AWS_DEFAULT_REGION")]

  DISABLE_PLACEMENT_GROUP = True   # t2.micro doesn't allow them
  placement_group_name = args.name

  vpc, subnet, security_group = network_setup()
  keypair = keypair_setup()  # saves private key locally to KEYPAIR_LOCATION
  placement_group = placement_group_setup(placement_group_name)

  ec2 = c.create_ec2_resource()
  if DISABLE_PLACEMENT_GROUP:
    placement_arg = {}
  else:
    placement_arg = {'GroupName': placement_group.name}
    

if __name__=='__main__':
  main()
