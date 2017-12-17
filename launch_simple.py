#!/usr/bin/env python
#
# Stand-alone launcher. Creates all necessary AWS resources and starts
# run.
#
# TODO: provide a tool to delete all the resources created
# TODO: authorize access roles so that instance kills itself after some time
#
# Ubuntu 16.04 AMI
# https://aws.amazon.com/marketplace/fulfillment?productId=d83d0782-cb94-46d7-8993-f4ce15d1a484&ref_=dtl_psb_continue&region=eu-west-2
# EU (London)	ami-ce736daa	Launch with EC2 Console
# EU (Ireland)	ami-f7e8558e	Launch with EC2 Console
# Asia Pacific (Seoul)	ami-d21fb8bc	Launch with EC2 Console
# Asia Pacific (Tokyo)	ami-09cb706f	Launch with EC2 Console
# South America (Sao Paulo)	ami-81a1e5ed	Launch with EC2 Console
# Canada (Central)	ami-5541fa31	Launch with EC2 Console
# Asia Pacific (Singapore)	ami-54b3e137	Launch with EC2 Console
# Asia Pacific (Sydney)	ami-c10ffaa3	Launch with EC2 Console
# EU (Frankfurt)	ami-f4de519b	Launch with EC2 Console
# US East (N. Virginia)	ami-08c35d72	Launch with EC2 Console
# US East (Ohio)	ami-b84b62dd	Launch with EC2 Console
# US West (N. California)	ami-a089b2c0	Launch with EC2 Console
# US West (Oregon)	ami-4a4b9232	Launch with EC2 Console


# This script creates VPC/security group/keypair if not already present

ami_dict = {
    "us-west-1": "ami-a089b2c0", #"ami-45ead225",
    "us-east-1": "ami-08c35d72",
    "eu-west-2": "ami-4a4b9232",
}
LINUX_TYPE = "ubuntu"  # linux type determines username to use to login
AMI_USERNAME = 'ubuntu'  # ami-specific username needed to login
          # ubuntu for Ubuntu images, ec2-user for Amazon Linux images

import os
import argparse
import boto3
import time
from collections import OrderedDict

parser = argparse.ArgumentParser(description='launch simple')
parser.add_argument('--run', type=str, default='simple',
                     help="name of the current run")
parser.add_argument('--instance_type', type=str, default='t2.micro',
                     help="type of instance")
args = parser.parse_args()

# Names of Amazon resources that are created. These settings are fixed across
# all runs, and correspond to resources created once per user per region.
brand='nexus2'
VPC_NAME=brand
SECURITY_GROUP_NAME=brand
KEYPAIR_LOCATION=os.environ["HOME"]+'/.'+brand+'.pem'
KEYPAIR_NAME=brand
PUBLIC_TCP_PORTS = [8888, 8889, 8890,  # ipython notebook ports
                    6379,              # redis port
                    6006, 6007, 6008,  # tensorboard ports
]

# region is taken from environment variable AWS_DEFAULT_REGION
assert 'AWS_DEFAULT_REGION' in os.environ
assert os.environ['AWS_DEFAULT_REGION'] in {'us-east-2','us-east-1','us-west-1','us-west-2','ap-south-1','ap-northeast-2','ap-southeast-1','ap-southeast-2','ap-northeast-1','ca-central-1','eu-west-1','eu-west-2','sa-east-1'}
assert os.environ.get("AWS_DEFAULT_REGION") in ami_dict


def _get_name(tags):
  """Helper utility to extract name out of tags dictionary.
      [{'Key': 'Name', 'Value': 'nexus'}] -> 'nexus'
 
     Assert fails if there's more than one name.
     Returns '' if there's less than one name.
  """
  
  names = [entry['Value'] for entry in tags if entry['Key']=='Name']
  if not names:
    names = ['']
  assert len(names)==1, "have more than one name: "+str(names)
  return names[0]


def _create_ec2_client():
  REGION = os.environ['AWS_DEFAULT_REGION']
  return boto3.client('ec2', region_name=REGION)


def _create_ec2_resource():
  REGION = os.environ['AWS_DEFAULT_REGION']
  return boto3.resource('ec2',region_name=REGION)


def _is_good_response(response):
  """Helper method to check if boto3 call was a success."""
  
  return response["ResponseMetadata"]['HTTPStatusCode'] == 200


def get_vpc_dict():
  """Returns dictionary of named VPCs {name: vpc}

  Assert fails if there's more than one VPC with same name."""

  client = _create_ec2_client()
  response = client.describe_vpcs()
  assert _is_good_response(response)

  result = OrderedDict()
  ec2 = _create_ec2_resource()
  for vpc_response in response['Vpcs']:
    key = _get_name(vpc_response.get('Tags', []))
    if not key:  # skip VPC's that don't have a name assigned
      continue
    
    assert key not in result, ("Duplicate VPC group " + key)
    result[key] = ec2.Vpc(vpc_response['VpcId'])

  print(result)
  print(client.describe_key_pairs())
  return result


def get_security_group_dict():
  """Returns dictionary of named security groups {name: securitygroup}."""

  client = _create_ec2_client()
  response = client.describe_security_groups()
  assert _is_good_response(response)

  result = OrderedDict()
  ec2 = _create_ec2_resource()
  for security_group_response in response['SecurityGroups']:
    key = _get_name(security_group_response.get('Tags', []))
    if not key:
      continue  # ignore unnamed security groups
    #    key = security_group_response['GroupName']
    assert key not in result, ("Duplicate security group " + key)
    result[key] = ec2.SecurityGroup(security_group_response['GroupId'])

  return result


def get_placement_group_dict():
  """Returns dictionary of {placement_group_name: (state, strategy)}"""

  client = _create_ec2_client()
  response = client.describe_placement_groups()
  assert _is_good_response(response)

  result = OrderedDict()
  ec2 = _create_ec2_resource()
  for placement_group_response in response['PlacementGroups']:
    key = placement_group_response['GroupName']
    assert key not in result, ("Duplicate placement group " +
                               key)
    #    result[key] = (placement_group_response['State'],
    #                   placement_group_response['Strategy'])
    result[key] = ec2.PlacementGroup(key)
  return result


def get_keypair_dict():
  """Returns dictionry of {keypairname: keypair}"""
  
  client = _create_ec2_client()
  response = client.describe_key_pairs()
  assert _is_good_response(response)
  
  result = {}
  ec2 = _create_ec2_resource()
  for keypair in response['KeyPairs']:
    keypair_name = keypair.get('KeyName', '')
    assert keypair_name not in result, "Duplicate key "+keypair_name
    result[keypair_name] = ec2.KeyPair(keypair_name)
  return result


def network_setup():
  """Creates VPC if it doesn't already exists, configures it for public
  internet access, returns vpc, subnet, security_group"""

  # from https://gist.github.com/nguyendv/8cfd92fc8ed32ebb78e366f44c2daea6
  
  existing_vpcs = get_vpc_dict()
  print(existing_vpcs)
  if VPC_NAME in existing_vpcs:
    print("Reusing VPC "+VPC_NAME)
    vpc = existing_vpcs[VPC_NAME]
    subnets = list(vpc.subnets.all())
    assert len(subnets) == 1
    subnet = subnets[0]
    
  else:
    print("Creating VPC "+VPC_NAME)
    ec2 = _create_ec2_resource()
    vpc = ec2.create_vpc(CidrBlock='192.168.0.0/16')
    vpc.create_tags(Tags=[{"Key": "Name", "Value": VPC_NAME}])
    vpc.wait_until_available()

    ig = ec2.create_internet_gateway()
    ig.attach_to_vpc(VpcId=vpc.id)
    
    route_table = vpc.create_route_table()
    route = route_table.create_route(
      DestinationCidrBlock='0.0.0.0/0',
      GatewayId=ig.id
    )

    subnet = vpc.create_subnet(CidrBlock='192.168.1.0/24')
    route_table.associate_with_subnet(SubnetId=subnet.id)

  # Creates security group if necessary
  existing_security_groups = get_security_group_dict()
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
      assert _is_good_response(response)

  return vpc, subnet, security_group


def keypair_setup():
  """Creates keypair if necessary, saves private key locally, returns contents
  of private key file."""
  
  
  existing_keypairs = get_keypair_dict()
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
  ec2 = _create_ec2_resource()
  keypair = ec2.create_key_pair(KeyName=KEYPAIR_NAME)
  assert not os.path.exists(KEYPAIR_LOCATION), "previous, keypair exists, delete it with 'sudo rm %s'"%(KEYPAIR_LOCATION,)
  
  open(KEYPAIR_LOCATION, 'w').write(keypair.key_material)
  os.system('chmod 400 '+KEYPAIR_LOCATION)
  return keypair


def placement_group_setup(group_name):
  """Creates placement group if necessary. Returns True if new placement
  group was created, False otherwise."""
  
  existing_placement_groups = get_placement_group_dict()

  group = existing_placement_groups.get(group_name, None)
  if group:
    assert group.state == 'available'
    assert group.strategy == 'cluster'
    return group

  ec2 = _create_ec2_resource()
  group = ec2.create_placement_group(GroupName=group_name, Strategy='cluster')
  return group

  
def main():
  ami = ami_dict[os.environ.get("AWS_DEFAULT_REGION")]

  DISABLE_PLACEMENT_GROUP = True   # t2.micro doesn't allow them
  placement_group_name = args.run
  
  vpc, subnet, security_group = network_setup()
  keypair = keypair_setup()  # saves private key locally to KEYPAIR_LOCATION
  placement_group = placement_group_setup(placement_group_name)

  ec2 = _create_ec2_resource()
  if DISABLE_PLACEMENT_GROUP:
    placement_arg = {}
  else:
    placement_arg = {'GroupName': placement_group.name}
    
  instances = ec2.create_instances(
    ImageId=ami,
    InstanceType=args.instance_type,
    MaxCount=1, MinCount=1,
    KeyName=KEYPAIR_NAME,
    NetworkInterfaces=[{'SubnetId': subnet.id,
                        'DeviceIndex': 0,
                        'AssociatePublicIpAddress': True,
                        'Groups': [security_group.id]}],
    Placement=placement_arg,
  )
  
  for instance in instances:
    tag = ec2.create_tags(
      Resources=[instance.id], Tags=[{
        'Key': 'Name',
        'Value': args.run
      }])

  print("Waiting for instances to come up")
  for instance in instances:
    instance.wait_until_running()

  # connect instructions
  head_instance = instances[0]
  head_instance.load()
  ip = head_instance.public_ip_address
  print("ssh -i %s -o StrictHostKeyChecking=no %s@%s"%(KEYPAIR_LOCATION,
                                                       AMI_USERNAME,
                                                       ip))

  time.sleep(36000)

  # clean-up
  for instance in instances:
    instance.terminate()

  # TODO: clean-up placement group
  if not DISABLE_PLACEMENT_GROUP:
    placement_group.delete()

if __name__=='__main__':
  main()
