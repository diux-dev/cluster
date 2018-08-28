
import argparse
import boto3
import os
import os
import random
import re
import shlex
import string
import sys
import threading
import time

from collections import Iterable
from collections import OrderedDict
from collections import defaultdict
from operator import itemgetter


# shortcuts to refer to util module, this lets move external code referencing
# util or u into this module unmodified
util = sys.modules[__name__]   
u = util

PRIVATE_KEY_LOCATION = os.environ['HOME']+'/.nexus' # location for pem files,
                                                    # this should be permanent
                                                    
DEFAULT_RESOURCE_NAME = 'nexus'  # name used for all persistent resources
                                 # (name of EFS, VPC, keypair prefixes)
                                 # can be changed through $RESOURCE_NAME for
                                 # debugging purposes

WAIT_INTERVAL_SEC=1  # how long to use for wait period
WAIT_TIMEOUT_SEC=20 # timeout after this many seconds

# name to use for mounting external drive
DEFAULT_UNIX_DEVICE='/dev/xvdq' # used to be /dev/xvdf

EMPTY_NAME="noname"   # name to use when name attribute is missing

def now_micros():
  """Return current micros since epoch as integer."""
  return int(time.time()*1e6)

aws_regexp=re.compile('^[a-zA-Z0-9+-=._:/@.]*$')
def validate_aws_name(name):
  """Validate resource name using AWS name restrictions from # http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_Tags.html#tag-restrictions"""
  assert len(name)<=127
  # disallow unicode characters to avoid pain
  assert name == name.encode('ascii').decode('ascii')
  assert aws_regexp.match(name)

resource_regexp=re.compile('^[a-z0-9]+$')
def validate_resource_name(name):
  """Check that name is valid as substitute for DEFAULT_RESOURCE_NAME. Since it's used in unix filenames, be more conservative than AWS requirements, just allow 30 chars, lowercase only."""
  assert len(name)<=30
  assert resource_regexp.match(name)
  validate_aws_name(name)


def validate_run_name(name):
  """Name used for run. Used as part of instance name, tmux session name."""
  validate_resource_name(name)
  
def deprecated_validate_name(name):
  """Checks that name matches AWS requirements."""
  
  # run name is used in tmux session name and instance name, so must restrict
  # run name to also be valid part of tmux/instance names
  # -Maximum number of tags per resource—50
  # -Maximum key length—127 Unicode characters in UTF-8
  # -Maximum value length—255 Unicode characters in UTF-8
  # letters, spaces, and numbers representable in UTF-8, plus the following special characters: + - = . _ : / @.
  import re
  assert len(name)<30
  assert re.match("[-+=._:\/@a-zA-Z0-9]+", name)


def get_resource_name(default=DEFAULT_RESOURCE_NAME):
  """Global default name for singleton AWS resources, see DEFAULT_RESOURCE_NAME."""
  name =os.environ.get('RESOURCE_NAME', default)
  if name != default:
    validate_resource_name(name)
  return name

def get_name(tags_or_instance_or_id):
  """Helper utility to extract name out of tags dictionary or intancce.
      [{'Key': 'Name', 'Value': 'nexus'}] -> 'nexus'
 
     Assert fails if there's more than one name.
     Returns '' if there's less than one name.
  """

  ec2 = u.create_ec2_resource()
  if hasattr(tags_or_instance_or_id, 'tags'):
    tags = tags_or_instance_or_id.tags
  elif isinstance(tags_or_instance_or_id, str):
    tags = ec2.Instance(tags_or_instance_or_id).tags
  elif tags_or_instance_or_id is None:
    return EMPTY_NAME
  else:
    assert isinstance(tags_or_instance_or_id, Iterable), "expected iterable of tags"
    tags = tags_or_instance_or_id

  if not tags:
    return EMPTY_NAME
  names = [entry['Value'] for entry in tags if entry['Key']=='Name']
  if not names:
    return ''
  if len(names)>1:
    assert False, "have more than one name: "+str(names)
  return names[0]

def get_state(instance):
  """Return state name like 'terminated' or 'running'"""
  return instance.state['Name']

def parse_job_name(name):
  """Parses job name of the form "taskid.job.run" and returns components
  job.run and taskid.
  If name does not have given form, returns Nones."""
  toks = name.split('.')
  if len(toks)!=3:
    return None, None
  task_id, role, run = toks
  try:
    task_id = int(task_id)
  except:
    task_id = -1
  return task_id, role+'.'+run

def get_session():
  # in future can add profiles with Session(profile_name=...)
  return boto3.Session()

def get_parsed_job_name(tags_or_instance):
  """Return jobname,task_id for given aws instance tags. IE, for
  0.worker.somerun you get '0' and 'worker.somerun'"""
  if hasattr(tags_or_instance, 'tags'):
    tags = retrieve_tags_with_retries(tags_or_instance)
  else:
    tags = tags_or_instance
  return parse_job_name(get_name(tags))

def retrieve_tags_with_retries(instance):
  WAIT_INTERVAL_SEC=1  # how long to use for wait period
  WAIT_TIMEOUT_SEC=20 # timeout after this many seconds
  while True:
    try:
      tags = instance.tags
      break
    except Exception as e:
      print("instance.tags failed with %s, retrying in %d seconds"%(str(e),
                                                                    WAIT_INTERVAL_SEC))
      time.sleep(WAIT_INTERVAL_SEC)
  return tags

def format_job_name(role, run_name):
  return "{}.{}".format(role, run_name)

def format_task_name(task_id, job_name):
  return "{}.{}".format(task_id, job_name)


def make_name(name):
  return [{'Key': 'Name', 'Value': name}]

def get_region():
  return get_session().region_name

def get_zone():
  assert 'ZONE' in os.environ
  return os.environ['ZONE']

def get_zone():
  assert 'ZONE' in os.environ, "Must specify ZONE environment variable"
  zone = os.environ['ZONE']
  region = get_region()
  assert zone.startswith(region), "Availability zone %s must be in default region %s. Default region is taken from environment variable AWS_DEFAULT_REGION, default zone is taken from environment variable ZONE" %(zone, region)
  return zone

def get_account_number():
  return str(boto3.client('sts').get_caller_identity()['Account'])


# keypairs:
# keypair name: nexus-yaroslav
# keypair filename: ~/.nexus/nexus-yaroslav-12395924-us-east-1.pem
# keypair name: nexus2-yaroslav
# keypair filename: ~/.nexus/nexus2-yaroslav-12395924-us-east-1.pem
# https://docs.google.com/document/d/14-zpee6HMRYtEfQ_H_UN9V92bBQOt0pGuRKcEJsxLEA/edit#

def get_keypair_name():
  """Returns keypair name to use for current resource, region and user."""
  
  assert 'USER' in os.environ, "why isn't USER defined?"
  username = os.environ['USER']
  assert '-' not in username, "username must not contain -, change $USER"
  validate_aws_name(username) # if this fails, override USER with something nice
  assert len(username)<30     # to avoid exceeding AWS 127 char limit
  return u.get_resource_name() +'-'+username


def get_keypair_fn():
  """Location of .pem file for current keypair"""

  keypair_name = get_keypair_name()
  account = u.get_account_number()
  region = u.get_region()
  fn = f'{PRIVATE_KEY_LOCATION}/{keypair_name}-{account}-{region}.pem'
  return fn

  
def create_ec2_client():
  return get_session().client('ec2')


def create_efs_client():
  # TODO: below sometimes fails with
  # botocore.exceptions.DataNotFoundError: Unable to load data for: endpoints
  # need to add retry
  return get_session().client('efs')


def create_ec2_resource():
  return get_session().resource('ec2')


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

def get_volume_dict():
  """Returns dictionary of named volumes"""
  ec2 = u.create_ec2_resource()
  volumes = list(ec2.volumes.all())
  d = {}
  for v in volumes:
    name = u.get_name(v)
    if name != u.EMPTY_NAME:
      d[name] = v
  return d

def get_snapshot_dict():
  """Returns dictionary of named volumes"""
  ec2 = u.create_ec2_resource()
  volumes = list(ec2.snapshots.filter(Filters=[], OwnerIds=['self']))
  d = {}
  for v in volumes:
    name = u.get_name(v)
    if name != u.EMPTY_NAME:
      d[name] = v
  return d

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
    if not key or key==EMPTY_NAME:  # skip VPC's that don't have a name assigned
      continue
    
    assert key not in result, ("Duplicate VPC group %s in %s" %(key,
                                                                response))
    result[key] = ec2.Vpc(vpc_response['VpcId'])

  return result

def get_gateway_dict(vpc):
  """Returns dictionary of named gateways for given VPC {name: gateway}"""
  return {u.get_name(gateway): gateway for
          gateway in vpc.internet_gateways.all()}

def get_security_group_dict():
  """Returns dictionary of named security groups {name: securitygroup}."""

  client = create_ec2_client()
  response = client.describe_security_groups()
  assert is_good_response(response)

  result = OrderedDict()
  ec2 = create_ec2_resource()
  for security_group_response in response['SecurityGroups']:
    key = get_name(security_group_response.get('Tags', []))
    if not key or key==EMPTY_NAME:
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
    if not key or key==EMPTY_NAME:   # skip EFS's without a name
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
  return efs_dict[name]

def delete_efs_id(efs_id):
  """Deletion sometimes fails, try several times."""
  start_time = time.time()
  efs_client = u.create_efs_client()
  sys.stdout.write("deleting %s ... " %(efs_id,))
  while True:
    try:
      response = efs_client.delete_file_system(FileSystemId=efs_id)
      if is_good_response(response):
        print("succeeded")
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

def current_timestamp():
  # timestamp logic from https://github.com/tensorflow/tensorflow/blob/155b45698a40a12d4fef4701275ecce07c3bb01a/tensorflow/core/platform/default/logging.cc#L80
  current_seconds=time.time();
  remainder_micros=int(1e6*(current_seconds-int(current_seconds)))
  time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_seconds))
  full_time_str = "%s.%06d"%(time_str, remainder_micros)
  return full_time_str


PRINTED_ERROR_INFO = False
ERROR_LOG_LOCATION = "/tmp/nexus_errors"
def loge(message, args=None):
  """Log error."""
  global PRINTED_ERROR_INFO
  if not PRINTED_ERROR_INFO:
    print("Errors encounted, logging to ", ERROR_LOG_LOCATION)
    PRINTED_ERROR_INFO = True
    
  ts = current_timestamp()
  if args:
    message = message % args
  open(ERROR_LOG_LOCATION, "a").write("%s %s\n"%(ts, message))

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
    print("%s took %.2f ms"%(self.tag, 1000*interval_sec))


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

def get_instance_dict():
  """Returns dictionary of {name: [instance, instance, ..]}"""
  
  ec2 = boto3.resource('ec2')

  result = OrderedDict()
  for i in ec2.instances.all():
    name = u.get_name(i.tags)
    if i.state['Name'] != 'running':
      continue
    instance_list = result.setdefault(name, [])
    instance_list.append(i)

  return result

def get_mount_targets_list(efs_id):
  """Returns list of all mount targets for given EFS id."""
  efs_client = u.create_efs_client()
  ec2 = u.create_ec2_resource()
  
  response = efs_client.describe_mount_targets(FileSystemId=efs_id)
  assert u.is_good_response(response)

  result = []
  for mount_response in response['MountTargets']:
    subnet = ec2.Subnet(mount_response['SubnetId'])
    zone = subnet.availability_zone
    state = mount_response['LifeCycleState']
    id = mount_response['MountTargetId']
    ip = mount_response['IpAddress']
    result.append(id)
    
  return result

def get_subnet_dict(vpc):
  """Returns dictionary of "availability zone" -> subnet for given VPC."""
  subnet_dict = {}
  for subnet in vpc.subnets.all():
    zone = subnet.availability_zone
    assert zone not in subnet_dict, "More than one subnet in %s, why?" %(zone,)
    subnet_dict[zone] = subnet
  return subnet_dict


def get_mount_targets_dict(efs_id):
  """Returns dict of {zone: mount_target_id} for given EFS id."""
  efs_client = u.create_efs_client()
  ec2 = u.create_ec2_resource()
  
  response = efs_client.describe_mount_targets(FileSystemId=efs_id)
  assert u.is_good_response(response)

  result = OrderedDict()
  for mount_response in response['MountTargets']:
    subnet = ec2.Subnet(mount_response['SubnetId'])
    zone = subnet.availability_zone
    state = mount_response['LifeCycleState']
    id = mount_response['MountTargetId']
    ip = mount_response['IpAddress']
    assert zone not in result
    result[zone] = id
    
  return result



def wait_on_fulfillment(ec2c, reqs):
    def get_instance_id(req):
      while req['State'] != 'active':
          print('Waiting on spot fullfillment...')
          time.sleep(5)
          reqs = ec2c.describe_spot_instance_requests(Filters=[{'Name': 'spot-instance-request-id', 'Values': [req['SpotInstanceRequestId']]}])
          if not reqs['SpotInstanceRequests']:
            print(f"SpotInstanceRequest for {req['SpotInstanceRequestId']} not found")
            continue
          req = reqs['SpotInstanceRequests'][0]
          req_status = req['Status']
          if req_status['Code'] not in ['pending-evaluation', 'pending-fulfillment', 'fulfilled']:
              print('Spot instance request failed:', req_status['Message'])
              print('Cancelling request. Please try again or use on demand.')
              ec2c.cancel_spot_instance_requests(SpotInstanceRequestIds=[req['SpotInstanceRequestId']])
              print(req)
              return None
      instance_id = req['InstanceId']
      print('Fulfillment completed. InstanceId:', instance_id)
      return instance_id
    return [get_instance_id(req) for req in reqs]

def create_spot_instances(launch_specs, spot_price=25, expiration_mins=15):
    """Args
      expiration_mins: this request only valid for this many mins from now
    """

    spot_price = str(spot_price)

    ec2c = create_ec2_client()
    num_tasks = launch_specs['MinCount']
    del launch_specs['MinCount']
    del launch_specs['MaxCount']


    import pytz      # datetime is not timezone aware, use pytz to fix
    import datetime as dt
    now = dt.datetime.utcnow().replace(tzinfo=pytz.utc)

    spot_args = {}
    spot_args['LaunchSpecification'] = launch_specs
    spot_args['SpotPrice'] = spot_price
    spot_args['InstanceCount'] = num_tasks
    spot_args['ValidUntil'] = now + dt.timedelta(minutes=expiration_mins)
    # "stop" doesn't seem to be supported for spot instances
    # spot_args['InstanceInterruptionBehavior']='stop'
    
    print(launch_specs)
    
    try:
      spot_requests = ec2c.request_spot_instances(**spot_args)
    except Exception as e:
      assert False, f"Spot instance request failed (out of capacity?), error was {e}"
      
    spot_requests = spot_requests['SpotInstanceRequests']
    instance_ids = wait_on_fulfillment(ec2c, spot_requests)
    for i in instance_ids: 
      if i == None: 
        print('Failed to create spot instances')
        return

    print('Success...')
    ec2 = create_ec2_resource()
    instances = list(ec2.instances.filter(Filters=[{'Name': 'instance-id', 'Values': instance_ids}]))
    # instance.reboot()
    # instance.wait_until_running()
    # instance.create_tags(Tags=[{'Key':'Name','Value':f'{name}'}])
    # volume = list(instance.volumes.all())[0]
    # volume.create_tags(Tags=[{'Key':'Name','Value':f'{name}'}])
    # print(f'Completed. SSH: ', get_ssh_command(instance))
    return instances


def make_ssh_command(instance):
  keypair_fn = u.get_keypair_fn()
  username = u.get_username(instance)
  ip = instance.public_ip_address
  cmd = "ssh -i %s -o StrictHostKeyChecking=no %s@%s" % (keypair_fn, username,
                                                         ip)
  

class SshClient:
  def __init__(self,
               hostname,
               ssh_key_fn=None,
               username=None,
               retry=1):
    """Create ssh connection to host

    Creates and returns and ssh connection to the host passed in.  

    Args:
      hostname: host name or ip address of the system to connect to.
      retry: number of time to retry.
      ssh_key_fn: full path to the ssk hey to use to connect.
      username: username to connect with.

    returns SSH client connected to host.
    """
    import paramiko

    print("ssh_to_host %s@%s"%(username, hostname))
    k = paramiko.RSAKey.from_private_key_file(ssh_key_fn)

    self.ssh_client = paramiko.SSHClient()
    self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    counter = retry
    while counter > 0:
      try:
        self.ssh_client.connect(hostname=hostname, username=username, pkey=k)
        break
      except Exception as e:
        counter = counter - 1
        print('Exception connecting to host via ssh (could be a timeout):'.format(e))
        if counter == 0:
          return None

  def run(self, cmd):
    """Runs given cmd in the task, returns stdout/stderr as strings.
    Because it blocks until cmd is done, use it for short cmds.

    Also, because this can run before task is initialized, use it
    for running initialization commands in sequence.

    This is a barebones method to be used during initialization that have
    minimal dependencies (no tmux)
    """
    print("run_sync: %s"%(cmd,))
    stdin, stdout, stderr = self.ssh_client.exec_command(cmd, get_pty=True)
    stdout_str = stdout.read().decode('ascii')
    stderr_str = stderr.read().decode('ascii')
    print("run_sync returned: " + stdout_str)
    return stdout_str, stderr_str


  def run_and_stream(self, cmd):
    """Runs command and streams output to local stderr."""
    
    stdin, stdout, stderr = self.ssh_client.exec_command(cmd, get_pty=True)
    if stdout:
      t1 = _StreamOutputToStdout(stdout, stdout_file)
    if stderr:
      t2 = _StreamOutputToStdout(stderr, stderr_file)
      
    if stdout:
      t1.join()
    if stderr:
      t2.join()    
    

def _add_echo(script):
  """Goes over each line script, adds "echo cmd" in front of each cmd.

  ls a

  becomes

  echo * ls a
  ls a
  """
  new_script = ""
  for cmd in script.split('\n'):
    cmd = cmd.strip()
    if not cmd:
      continue
    new_script+="echo \\* " + shlex.quote(cmd) + "\n"
    new_script+=cmd+"\n"
  return new_script

def _StreamOutputToStdout(fd):  # todo: pep convention
  """Stream output to stdout"""

  def func(fd):
    for line in iter(lambda: fd.readline(2048), ''):
      print(line.strip())
      sys.stdout.flush()

  t = threading.Thread(target=func, args=(fd,))
  t.start()
  
  return t

def _parse_key_name(keyname):
  """keyname => resource, username"""
  # Relies on resource name not containing -, validated in
  # validate_resource_name
  toks = keyname.split('-')
  if len(toks)!=2:
    return None, None       # some other keyname not launched by nexus
  else:
    return toks
  
def lookup_aws_instances(job_name, states=['running', 'stopped'],
                         instance_type=None):
  """Returns all AWS instances for given AWS job name, like
   simple.worker"""

  #  print("looking up", job_name)

  # todo: assert fail when there are multiple instances with same name?
  ec2 = u.create_ec2_resource()

  # TODO: add waiting so that instances in state "initializing" are supported
  instances = ec2.instances.filter(
    Filters=[{'Name': 'instance-state-name', 'Values': states}])

  result = []
  resource = u.get_resource_name()
  username = os.environ['USER']
  

  # look for an existing instance matching job, ignore instances launched
  # by different user or under different resource name
  for i in instances.all():
    task_id, current_job_name = u.get_parsed_job_name(i.tags)
    if current_job_name != job_name: continue
    
    target_resource, target_username = _parse_key_name(i.key_name)
    if resource != target_resource:
      print(f"Found {current_job_name} launched by {resource}, ignoring")
      continue
    if username != target_username:
      print(f"Found {current_job_name} launched by {username}, ignoring")
      continue

    if instance_type:
      assert i.instance_type == instance_type, f"Found existing instance for job {job_name} but different instance type ({i.instance_type}) than requested ({instance_type}), terminate {job_name} first or use new job name."
    result.append(i)


  return result

def lookup_volume(name):
  """Looks up volume matching given name or id."""
  ec2 = u.create_ec2_resource()
  vols = []
  for v in ec2.volumes.all():
    if u.get_name(v) == name or v.id == name:
      vols.append(v)
  assert len(vols)>0, f"volume {name} not found"
  assert len(vols)<2, f"multiple volumes with name={name}"
  return vols[0]


def maybe_create_placement_group(name='', max_retries=10):
  """Creates placement group or reuses existing one. crash if unable to create
  placement group. If name is empty, ignores request."""
  
  if not name or name==EMPTY_NAME:
    return
  
  client = u.create_ec2_client()
  try:
    # TODO: check that the error is actually about placement group
    client.describe_placement_groups(GroupNames=[name])
  except boto3.exceptions.botocore.exceptions.ClientError as e:
    print("Creating placement group: "+name)
    res = client.create_placement_group(GroupName=name, Strategy='cluster')

  counter = 0
  while True:
    try:
      res = client.describe_placement_groups(GroupNames=[name])
      res_entry = res['PlacementGroups'][0]
      if res_entry['State'] == 'available':
        print("Found placement group: "+name)
        assert res_entry['Strategy'] == 'cluster'
        break
    except Exception as e:
      print("Got exception: %s"%(e,))
    counter = counter + 1
    if counter >= max_retries:
      assert False, 'Failed to create placement group ' + name
    time.sleep(WAIT_INTERVAL_SEC)


def merge_kwargs(kwargs1, kwargs2):
  """Merges two dictionaries, assert fails if there's overlap in keys."""
  assert kwargs1.keys().isdisjoint(kwargs2.keys())
  kwargs3 = {}
  kwargs3.update(kwargs1)
  kwargs3.update(kwargs2)
  return kwargs3



def install_pdb_handler():
  """Make CTRL+\ break into gdb."""
  
  import signal
  import pdb

  def handler(signum, frame):
    pdb.set_trace()
  signal.signal(signal.SIGQUIT, handler)


# TODO: merge with u.SshClient
def ssh_to_host(hostname,
                ssh_key_fn=None,
                username=None,
                retry=1):

  """Create ssh connection to host

  Creates and returns and ssh connection to the host passed in.  

  Args:
    hostname: host name or ip address of the system to connect to.
    retry: number of time to retry.
    ssh_key_fn: full path to the ssh key to use to connect.
    username: username to connect with.

  returns Paramiko SSH client connected to host.

  """
  import paramiko

  print(f"ssh -i {ssh_key_fn} {username}@{hostname}")
  k = paramiko.RSAKey.from_private_key_file(ssh_key_fn)
  
  ssh_client = paramiko.SSHClient()
  ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

  counter = retry
  while counter > 0:
    try:
      ssh_client.connect(hostname=hostname, username=username, pkey=k)
      break
    except Exception as e:
      counter = counter - 1
      print('Exception connecting to host via ssh (could be a timeout):'.format(e))
      if counter == 0:
        return None

  return ssh_client

# TODO: inversion procedure is incorrect
# TODO: probably want to be seconds in local time zone instead
def seconds_from_datetime(dt):
  """Convert datetime into seconds since epochs in UTC timezone. IE, to use for
  instance.launch_time:
     toseconds(instance.launch_time).

  to invert (assuming UTC timestamps):
     import pytz
     utc = pytz.UTC
     utc.localize(datetime.fromtimestamp(seconds))
  """
  return time.mktime(dt.utctimetuple())

def datetime_from_seconds(seconds, timezone="US/Pacific"):
  """
  timezone: pytz timezone name to use for conversion, ie, UTC or US/Pacific
  """
  return dt.datetime.fromtimestamp(seconds, pytz.timezone(timezone))


# augmented SFTP client that can transfer directories, from
# https://stackoverflow.com/a/19974994/419116
def put_dir(sftp, source, target):
  ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are 
            created under target.
        '''
  def _safe_mkdir(sftp, path, mode=511, ignore_existing=True):
    ''' Augments mkdir by adding an option to not fail if the folder exists  '''
    try:
      sftp.mkdir(path, mode)
    except IOError:
      if ignore_existing:
        pass
      else:
        raise

  assert os.path.isdir(source)
  _safe_mkdir(sftp, target)
              
  for item in os.listdir(source):
    if os.path.isfile(os.path.join(source, item)):
      sftp.put(os.path.join(source, item), os.path.join(target, item))
    else:
      _safe_mkdir(sftp, '%s/%s' % (target, item))
      put_dir(sftp, os.path.join(source, item), '%s/%s' % (target, item))


def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]

def lookup_ami_id(wildcard):
  """Returns ami matching given wildcard
  # lookup_ami('pytorch*').name => ami-29fa"""
  ec2 = u.create_ec2_resource()
  filter = {'Name': 'name', 'Values' : [wildcard]}
  #  images = list(ec2.images.filter(Filters = [filter], Owners=['self', 'amazon']))
  images = list(ec2.images.filter(Filters = [filter]))
  assert len(images)<=1, "Multiple images match "+str(wildcard)
  assert len(images)>0, "No images match "+str(wildcard)
  return images[0]


def toseconds(dt):
  """Converts datetime object to seconds."""
  return time.mktime(dt.utctimetuple())

def get_instances(fragment, verbose=True, filter_by_key=True):
  """Returns ec2.Instance object whose name contains fragment, in reverse order of launching (ie, most recent intance first). Optionally filters by key, only including instances launched with key_name matching current username.

  args:
    verbose: print information about all matching instances found

    filter_by_key  if True, ignore instances that are not launched with current
        user's default key
  """

  from tzlocal import get_localzone # $ pip install tzlocal


  def vprint(*args):
    if verbose: print(*args)
    
  region = u.get_region()
  client = u.create_ec2_client()
  ec2 =u.create_ec2_resource()
  response = client.describe_instances()
    
  instance_list = []
  for instance in ec2.instances.all():
    if instance.state['Name'] != 'running':
      continue
    
    name = u.get_name(instance.tags)
    if (fragment in name or fragment in str(instance.public_ip_address) or
        fragment in str(instance.id) or fragment in str(instance.private_ip_address)):
      instance_list.append((toseconds(instance.launch_time), instance))

  sorted_instance_list = reversed(sorted(instance_list, key=itemgetter(0)))
  cmd = ''
  filtered_instance_list = []  # filter by key
  vprint("Using region ", region)
  for (ts, instance) in sorted_instance_list:
    if filter_by_key and instance.key_name != u.get_keypair_name():
      vprint(f"Got key {instance.key_name}, expected {u.get_keypair_name()}")
      continue
    filtered_instance_list.append(instance)
  return filtered_instance_list

def get_instance(fragment, verbose=False, filter_by_key=True):
  """Gets a single instance containing given fragment."""
  instances = get_instances(fragment, verbose=verbose,
                            filter_by_key=filter_by_key)
  assert len(instances)>0, f"instances with ({fragment}) not found"
  assert len(instances)<2, f"multiple instances matching ({fragment})"
  return instances[0]

def get_username(instance):
  """Gets username needed to connect to given instance."""

  # TODO: extract from tags?
  # for now use ubuntu by default, with env override if needed
  username = os.environ.get("USERNAME", "ubuntu")
  return username

def random_id(N=5):
  """Random id to use for AWS identifiers."""
  #  https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
  return ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))


# no_op method/object to accept every signature
def no_op(*args, **kwargs): pass
class NoOp:
  def __getattr__(self, *args):
    return no_op


def create_blank_volume(name, zone, size, iops, voltype='io1'):
  """Creates blank volume with given specs."""
  
  tag_specs = [{
    'ResourceType': 'volume',
    'Tags': [{
        'Key': 'Name',
        'Value': name
    }]
  }]
  volume = ec2.create_volume(Size=size, VolumeType=voltype,
                             TagSpecifications=tag_specs,
                             AvailabilityZone=zone,
                             Iops=iops)
  return volume


ATTACH_WAIT_INTERVAL_SEC=5
def attach_volume(vol, instance, device=DEFAULT_UNIX_DEVICE):
  """Attaches volume to given instance."""

  existing_attachments = [d['InstanceId'] for d in vol.attachments]
  if instance.id in existing_attachments:
    print("Volume is already attached, skipping")
    return
  
  while True:
    try:
      response = vol.attach_to_instance(InstanceId=instance.id, Device=device)
      print(f'Attaching {u.get_name(vol)} to {u.get_name(instance)}: response={response.get("State", "none")}')
    except Exception as e:
      print(f'Error attaching volume: ({e}). Retrying in {ATTACH_WAIT_INTERVAL_SEC}', e)
      time.sleep(ATTACH_WAIT_INTERVAL_SEC)
      continue
    else:
      print('Attachment successful')
      break

def mount_volume(volume, task, mount_directory, device=DEFAULT_UNIX_DEVICE):
  while True:
    try:
      # Need retry logic because attachment is async and can be slow
      # run_async doesn't propagate exceptions raised on workers, use regular
      df_output = task.run_and_capture_output('df')
      if device in df_output:
        print('Volume already mounted, skipping')
        return
      task.run(f'sudo mkdir {mount_directory}', ignore_errors=True)
      task.run(f'sudo mount {device} {mount_directory}')
      task.run(f'sudo chown `whoami` {mount_directory}')
    except Exception as e:
      print(f'mount failed with: ({e})')
      print(f'Retrying in {ATTACH_WAIT_INTERVAL_SEC}')
      time.sleep(ATTACH_WAIT_INTERVAL_SEC)
      continue
    else:
      print(f'Mount successful')
      break

def maybe_create_resources():
  """Use heuristics to decide to possibly create resources"""

  def do_create_resources():
    """Check if gateway, keypair, vpc exist."""
    resource = u.get_resource_name()
    if u.get_keypair_name() not in u.get_keypair_dict():
      return True
    vpcs = u.get_vpc_dict()
    if resource not in vpcs:
      return True
    vpc = vpcs[resource]
    gateways = u.get_gateway_dict(vpc)
    if resource not in gateways:
      return True
    return False
  
  if do_create_resources():
    import create_resources as create_resources_lib
    create_resources_lib.create_resources()

def is_list_or_tuple(value):
  return isinstance(value, list) or isinstance(value, tuple)
