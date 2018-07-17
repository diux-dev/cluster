# methods common to create_resources and delete_resources
import os
import argparse
import boto3
import shlex
import re
import sys
import threading
import time
import os

from collections import OrderedDict
from collections import defaultdict


import tensorflow as tf

# shortcuts to refer to util module, this lets move external code into
# this module unmodified
util = sys.modules[__name__]   
u = util

WAIT_INTERVAL_SEC=1  # how long to use for wait period
WAIT_TIMEOUT_SEC=20 # timeout after this many seconds


def now_micros():
  """Return current micros since epoch as integer."""
  return int(time.time()*1e6)

# http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_Tags.html#tag-restrictions
aws_regexp=re.compile('^[a-zA-Z0-9+-=._:/@.]*$')
def validate_aws_name(name):
  assert len(name)<=127
  # disallow unicode names to avoid pain
  assert name == name.encode('ascii').decode('ascii')
  assert aws_regexp.match(name)


resource_regexp=re.compile('^[a-z0-9]*$')
def validate_resource_name(name):
  """Check that resource name is valid. To be conservative allow 30 chars, lowercase only."""
  assert len(name)<=30
  # disallow unicode names to avoid pain
  assert resource_regexp.match(name)

def get_resource_name(default='nexus'):
  """Gives global default name for singleton AWS resources (VPC name, keypair name, etc)."""
  name =os.environ.get('RESOURCE_NAME', default)
  if name != default:
    validate_resource_name(name)
  return name
                           
def get_name(tags_or_instance):
  """Helper utility to extract name out of tags dictionary or intance.
      [{'Key': 'Name', 'Value': 'nexus'}] -> 'nexus'
 
     Assert fails if there's more than one name.
     Returns '' if there's less than one name.
  """

  if hasattr(tags_or_instance, 'tags'):
    tags = tags_or_instance.tags
  else:
    tags = tags_or_instance
    
  if not tags:
    return ''
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


def get_parsed_job_name(tags):
  """Return jobname,task_id for given aws instance tags."""
  return parse_job_name(get_name(tags))

def format_job_name(role, run):
  return "{}.{}".format(role, run)

def format_task_name(task_id, role, run):
  assert int(task_id) == task_id
  return "{}.{}.{}".format(task_id, role, run)


def make_name(name):
  return [{'Key': 'Name', 'Value': name}]

def get_session(profile_name='diux'):
  return boto3.Session(profile_name=profile_name)

def get_region():
  # assert 'AWS_DEFAULT_REGION' in os.environ, "Must specify AWS_DEFAULT_REGION environment variable, ie 'export AWS_DEFAULT_REGION=us-west-2'"
  # return os.environ['AWS_DEFAULT_REGION']
  return get_session().region_name

def get_keypair_name():
  """Returns keypair name to use for current region and user."""
  assert 'USER' in os.environ, "why isn't USER defined?"
  username = os.environ['USER']
  validate_aws_name(username) # if this fails, override USER with something nice
  assert len(username)<30     # to avoid exceeding AWS 127 char limit
  return u.get_resource_name() +'-'+username

def create_ec2_client():
  return get_session().client('ec2')


def create_efs_client():
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


def loge(message, args=None):
  """Log error."""
  ts = current_timestamp()
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

def create_spot_instances(launch_specs, spot_price=None):
    # (AS) forcing spot price to be higher for now
    spot_price = '25'

    ec2c = create_ec2_client()
    num_tasks = launch_specs['MinCount']
    del launch_specs['MinCount']
    del launch_specs['MaxCount']

    print(launch_specs)

    if spot_price is None:
      spot_requests = ec2c.request_spot_instances(LaunchSpecification=launch_specs, InstanceCount=num_tasks)    
    else:
      spot_requests = ec2c.request_spot_instances(SpotPrice=spot_price, LaunchSpecification=launch_specs, InstanceCount=num_tasks)
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


def get_keypair_fn(keypair_name):
  """Generate canonical location for .pem file for given keypair and
  default region."""
  return "%s/%s-%s.pem" % (os.environ["HOME"], keypair_name,
                           get_region(),)

class SshClient:
  def __init__(self,
               hostname,
               ssh_key=None,
               username=None,
               retry=1):
    """Create ssh connection to host

    Creates and returns and ssh connection to the host passed in.  

    Args:
      hostname: host name or ip address of the system to connect to.
      retry: number of time to retry.
      ssh_key: full path to the ssk hey to use to connect.
      username: username to connect with.

    returns SSH client connected to host.
    """
    import paramiko

    print("ssh_to_host %s@%s"%(username, hostname))
    k = paramiko.RSAKey.from_private_key_file(ssh_key)

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

def lookup_aws_instances(job_name, states=['running', 'stopped']):
  """Returns all AWS instances for given AWS job name, like
   simple.worker"""

  #  print("looking up", job_name)

  # todo: assert fail when there are multiple instances with same name?
  ec2 = u.create_ec2_resource()

  # TODO: add waiting so that instances in state "initializing" are supported
  instances = ec2.instances.filter(
    Filters=[{'Name': 'instance-state-name', 'Values': states}])

  result = []
  for i in instances.all():
    task_id, current_job_name = u.get_parsed_job_name(i.tags)
    #    print("Obtained job name", current_job_name, "task", task_id)

    if current_job_name == job_name:
      result.append(i)


  return result

def maybe_create_placement_group(name='', max_retries=10):
  """Creates placement group or reuses existing one. crash if unable to create
  placement group. If name is empty, ignores request."""
  
  if not name:
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
                ssh_key=None,
                username=None,
                retry=1):

  """Create ssh connection to host

  Creates and returns and ssh connection to the host passed in.  

  Args:
    hostname: host name or ip address of the system to connect to.
    retry: number of time to retry.
    ssh_key: full path to the ssk hey to use to connect.
    username: username to connect with.

  returns Paramiko SSH client connected to host.

  """
  import paramiko

  print("ssh_to_host %s@%s" % (username, hostname))
  k = paramiko.RSAKey.from_private_key_file(ssh_key)
  
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

  to invert:
     import pytz
     utc = pytz.UTC
     utc.localize(datetime.fromtimestamp(seconds))
  """
  return time.mktime(dt.utctimetuple())


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
  images = list(ec2.images.filter(Filters = [filter], Owners=['self', 'amazon']))
  assert len(images)<=1, "Multiple images match "+str(wildcard)
  assert len(images)>=0, "No images match "+str(wildcard)
  return images[0]
