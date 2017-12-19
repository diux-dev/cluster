# methods common to create_resources and delete_resources
import os
import argparse
import boto3
import sys
import time
from collections import OrderedDict

# shortcuts to refer to util module, this lets move external code into
# this module unmodified
util = sys.modules[__name__]   
u = util

WAIT_INTERVAL_SEC=1  # how long to use for wait period
WAIT_TIMEOUT_SEC=10 # timeout after this many seconds



def get_name(tags):
  """Helper utility to extract name out of tags dictionary.
      [{'Key': 'Name', 'Value': 'nexus'}] -> 'nexus'
 
     Assert fails if there's more than one name.
     Returns '' if there's less than one name.
  """

  if not tags:
    return ''
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


def create_efs_client():
  REGION = os.environ['AWS_DEFAULT_REGION']
  return boto3.client('efs', region_name=REGION)


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
  """Returns dictionary of {keypairname: keypair}"""
  
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


def get_efs_dict():
  """Returns dictionary of {efs_name: efs_id}"""
  # there's no EC2 resource for EFS objects, so return EFS_ID instead
  # https://stackoverflow.com/questions/47870342/no-ec2-resource-for-efs-objects

  efs_client = u.create_efs_client()
  response = efs_client.describe_file_systems()
  assert u.is_good_response(response)
  result = OrderedDict()
  for efs_response in response['FileSystems']:
    fs_id = efs_response['FileSystemId']
    tag_response = efs_client.describe_tags(FileSystemId=fs_id)
    assert u.is_good_response(tag_response)
    key = u.get_name(tag_response['Tags'])
    if not key:   # skip EFS's without a name
      continue
    assert key not in result
    result[key] = fs_id

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


def create_efs(name):
  efs_client = u.create_efs_client()
  token = str(int(time.time()*1e6)) # epoch usec
  
  response = efs_client.create_file_system(CreationToken=token,
                                           PerformanceMode='generalPurpose')
  assert is_good_response(response)
  start_time = time.time()
  while True:
    try:
      response = efs_client.create_file_system(CreationToken=token,
                                               PerformanceMode='generalPurpose')
      assert is_good_response(response)
      time.sleep(WAIT_INTERVAL_SEC)
    except Exception as e:
      if e.response['Error']['Code']=='FileSystemAlreadyExists':
        break
      else:
        u.loge(e)
      break

    if time.time() - start_time - WAIT_INTERVAL_SEC > WAIT_TIMEOUT_SEC:
      assert False, "Timeout exceeded creating EFS %s (%s)"%(token, name)
      
    time.sleep(WAIT_TIMEOUT_SEC)

  # find efs id from given token
  response = efs_client.describe_file_systems()
  assert is_good_response(response)
  fs_id = get1(response['FileSystems'], FileSystemId=-1, CreationToken=token)
  response = efs_client.create_tags(FileSystemId=fs_id,
                                    Tags=u.make_name(name))
  assert is_good_response(response)

  # make sure EFS is now visible
  efs_dict = get_efs_dict()
  assert name in efs_dict

def delete_efs_id(efs_id):
  """Deletion sometimes fails, try several times."""
  start_time = time.time()
  efs_client = u.create_efs_client()
  print("Deleting "+efs_id)
  while True:
    try:
      response = efs_client.delete_file_system(FileSystemId=efs_id)
      if is_good_response(response):
        print("Succeeded")
        break
      time.sleep(WAIT_INTERVAL_SEC)
    except Exception as e:
      print("Failed with %s"%(e,))
      if time.time() - start_time - WAIT_INTERVAL_SEC < WAIT_TIMEOUT_SEC:
        print("Retrying in %s sec"%(WAIT_INTERVAL_SEC,))
        time.sleep(WAIT_INTERVAL_SEC)
      else:
        print("Giving up")
        break


  
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

def _current_timestamp():
  # timestamp logic from https://github.com/tensorflow/tensorflow/blob/155b45698a40a12d4fef4701275ecce07c3bb01a/tensorflow/core/platform/default/logging.cc#L80
  current_seconds=time.time();
  remainder_micros=int(1e6*(current_seconds-int(current_seconds)))
  time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_seconds))
  full_time_str = "%s.%06d"%(time_str, remainder_micros)
  return full_time_str


def loge(message, args=None):
  """Log error."""
  ts = _current_timestamp()
  if args:
    message = message % args
  open("/tmp/nexus_errors", "a").write("%s %s\n"%(ts, message))

class timeit:
  """Decorator to measure length of time spent in the block in millis and log
  it to TensorBoard."""
  
  def __init__(self, tag=""):
    self.tag = tag
    
  def __enter__(self):
    self.start = time.perf_counter()
    return self
  
  def __exit__(self, *args):
    self.end = time.perf_counter()
    interval_sec = (self.end - self.start)
    print("%s took %.2f seconds"%(self.tag, interval_sec))

def get_instance_ip_map():
  """Return instance_id->private_ip map for all running instances."""
  
  ec2 = boto3.resource('ec2')

  # Get information for all running instances
  running_instances = ec2.instances.filter(Filters=[{
    'Name': 'instance-state-name',
    'Values': ['running']}])

  ec2info = OrderedDict()
  for instance in running_instances:
    name = ''
    for tag in instance.tags or []:
      if 'Name' in tag['Key']:
        name = tag['Value']
    ec2info[instance.id] = instance.private_ip_address
    
  return ec2info
