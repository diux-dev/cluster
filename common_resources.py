# methods common to create_resources and delete_resources
import os
import argparse
import boto3
import time
from collections import OrderedDict

def get_name(tags):
  """Helper utility to extract name out of tags dictionary.
      [{'Key': 'Name', 'Value': 'nexus'}] -> 'nexus'
 
     Assert fails if there's more than one name.
     Returns '' if there's less than one name.
  """
  
  names = [entry['Value'] for entry in tags if entry['Key']=='Name']
  if not names:
    return ''
  if len(names)>1:
    assert False, "have more than one name: "+str(names)
  return names[0]

def make_name(name):
  return [{'Key': 'Name', 'Value': name}]

def create_ec2_client():
  REGION = os.environ['AWS_DEFAULT_REGION']
  return boto3.client('ec2', region_name=REGION)


def create_ec2_resource():
  REGION = os.environ['AWS_DEFAULT_REGION']
  return boto3.resource('ec2',region_name=REGION)


def is_good_response(response):
  """Helper method to check if boto3 call was a success."""

  code = response["ResponseMetadata"]['HTTPStatusCode']
  # get response code 201 on EFS creation
  return  code >= 200 and code<300

def wait_until_available(resource):
  start_time = time.time()
  while True:
    resource.load()
    if resource.state == 'available':
      break
    if time.time() - start_time - WAIT_INTERVAL_SEC > WAIT_TIMEOUT_SEC:
      assert False, "Timeout exceeded waiting for %s"%(resource,)
    time.sleep(WAIT_TIMEOUT_SEC)

def get_vpc_dict():
  """Returns dictionary of named VPCs {name: vpc}

  Assert fails if there's more than one VPC with same name."""

  client = create_ec2_client()
  response = client.describe_vpcs()
  assert is_good_response(response)

  result = OrderedDict()
  ec2 = create_ec2_resource()
  for vpc_response in response['Vpcs']:
    key = get_name(vpc_response.get('Tags', []))
    if not key:  # skip VPC's that don't have a name assigned
      continue
    
    assert key not in result, ("Duplicate VPC group %s in %s" %(key,
                                                                response))
    result[key] = ec2.Vpc(vpc_response['VpcId'])

  return result


def get_security_group_dict():
  """Returns dictionary of named security groups {name: securitygroup}."""

  client = create_ec2_client()
  response = client.describe_security_groups()
  assert is_good_response(response)

  result = OrderedDict()
  ec2 = create_ec2_resource()
  for security_group_response in response['SecurityGroups']:
    key = get_name(security_group_response.get('Tags', []))
    if not key:
      continue  # ignore unnamed security groups
    #    key = security_group_response['GroupName']
    assert key not in result, ("Duplicate security group " + key)
    result[key] = ec2.SecurityGroup(security_group_response['GroupId'])

  return result


def get_placement_group_dict():
  """Returns dictionary of {placement_group_name: (state, strategy)}"""

  client = create_ec2_client()
  response = client.describe_placement_groups()
  assert is_good_response(response)

  result = OrderedDict()
  ec2 = create_ec2_resource()
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
  
  client = create_ec2_client()
  response = client.describe_key_pairs()
  assert is_good_response(response)
  
  result = {}
  ec2 = create_ec2_resource()
  for keypair in response['KeyPairs']:
    keypair_name = keypair.get('KeyName', '')
    assert keypair_name not in result, "Duplicate key "+keypair_name
    result[keypair_name] = ec2.KeyPair(keypair_name)
  return result

def get_available_zones():
  client = create_ec2_client()
  response = client.describe_availability_zones()
  assert is_good_response(response)
  zones = []
  for avail_response in response['AvailabilityZones']:
    messages = avail_response['Messages']
    zone = avail_response['ZoneName']
    state = avail_response['State']
    assert not messages, "zone %s is broken? Has messages %s"%(zone, message)
    assert state == 'available', "zone %s is broken? Has state %s"%(zone, state)
    zones.append(zone)
  return zones


def get1(items, **kwargs):
  """Helper method to extract values, ie
  response = [{'State': 'available', 'VpcId': 'vpc-2bb1584c'}]
  get1(response, State=-1, VpcId='vpc-2bb1584c') #=> 'available'"""

  # find the value of attribute to return
  query_arg = None
  for arg, value in kwargs.items():
    if value == -1:
      assert query_arg is None, "Only single query arg (-1 valued) is allowed"
      query_arg = arg
  result = []
  
  filterset = set(kwargs.keys())
  for item in items:
    match = True
    assert filterset.issubset(item.keys()), "Filter set contained %s which was not in record %s" %(filterset.difference(item.keys()),
                                                                                                  item)
    for arg in item:
      if arg == query_arg:
        continue
      if arg in kwargs:
        if item[arg] != kwargs[arg]:
          match = False
          break
    if match:
      result.append(item[query_arg])
  assert len(result) <= 1, "%d values matched %s, only allow 1" % (len(result), kwargs)
  if result:
    return result[0]
  return None
