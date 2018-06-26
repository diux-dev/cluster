"""Utilities to launch jobs on AWS.

Example usage:
job = aws.tf_job('myjob', 1)
task = job.tasks[0]
task.upload(__file__)   # copies current script onto machine
task.run("python %s --role=worker" % (__file__,)) # runs script and streams output locally to file in /tmp

"""

# list current jobs

# TODO: look up how sagemaker allocates machines
# TODO: simplify connecting to existing jobs (remove need to call wait_until..)
# TODO: add ability to check for whether SSH has been initialized
# todo: use task id as part of instance name

# TODO: document all uses of local files (tasklogdir?)
# TODO: document initialization markers
# TODO: redirect output to files to make available locally
# TODO: make instance initialization happen in parallel

import argparse
import base64
import os
import struct
import sys
import threading
import shlex
import time

import boto3
import paramiko
import yaml

import util as u


# TODO: add | tee to make things show up both in emacs, and in console
# TODO: improve errors when root script hangs, ie tmux install can send
# Another app is currently holding the yum lock; waiting for it to exit...
#  The other application is: yum
#    Memory :  46 M RSS (394 MB VSZ)
#    Started: Wed Dec 13 19:54:40 2017 - 03:15 ago
# To reproduce this, run "sudo yum install tmux" on new Amazon Linux image
# then launch install script which does "sudo yum install -y tmux"
# this will hang because of lock issues, but error is not printed anyhwere
# until the command finishes. Need to make printing asynchronous


# TODO: for robustness, redirect install command output somewhere
# tmux seems to drop "send-keys" commands sometimes

# todo: add timestamps to log messages so I can get delay before connecte
# TODO: if the instance exists, but is stopped, add ability to start it

from collections import OrderedDict
from pprint import pprint as pp

# global settings that we don't expect to change
DEFAULT_PORT = 3000
LOCAL_TASKLOGDIR_PREFIX='/tmp/tasklogs'
TIMEOUT_SEC=5
DEFAULT_LINUX_TYPE='ubuntu'

#TASKDIR_PREFIX='/temp/tasklogs'
LOGDIR_PREFIX='/efs/runs'


# TODO: document KEY_NAME restriction a bit better
# TODO: move installation script to run under tmux for easier debugging of
# installation failures

# global AWS vars from environment

# TODO: remove need for this global var
KEY_NAME = os.environ.get('KEY_NAME', '')
SSH_KEY_PATH = os.environ.get('SSH_KEY_PATH', '')
SECURITY_GROUP = os.environ.get('SECURITY_GROUP', '')

# TODO: add support for running install scripts on all instances in parallel
INSTALL_IN_PARALLEL=False

ENABLE_MIRRORING=False # copy every command invocation locally

# work-around for https://github.com/tmux/tmux/issues/1185
BULK_INSTALL=True

# Things that are automatically installed on all instances, all job types
# These install necessary dependencies like tmux, commands run synchronously
# through regular shell
ROOT_INSTALL_SCRIPT_UBUNTU="""
"""
ROOT_INSTALL_SCRIPT_AMAZON="""
sudo yum install -y tmux
"""

USERNAME_UBUNTU="ubuntu"
USERNAME_AMAZON="ec2-user"
USERNAME="username_is_not_defined"


def _current_timestamp():
  # timestamp logic from https://github.com/tensorflow/tensorflow/blob/155b45698a40a12d4fef4701275ecce07c3bb01a/tensorflow/core/platform/default/logging.cc#L80
  current_seconds=time.time();
  remainder_micros=int(1e6*(current_seconds-int(current_seconds)))
  time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_seconds))
  full_time_str = "%s.%06d"%(time_str, remainder_micros)
  return full_time_str

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


# TODO: logging with timestamps
def setup_logdir(name):
  logdir = LOGDIR_PREFIX + '/' + name
  if os.path.exists(logdir):
    print("Warning, %s exists"%(logdir,))
  os.system('mkdir -p '+logdir)
  return logdir


def _ExecuteCommandInThread(ssh_client,
                           cmd,
                           stdout_file=None,
                           stderr_file=None,
                           line_extractor=None,
                           print_error=False):
  """Returns a thread that executes the given cmd.  Non-Blocking call.
  

  Args:
    ssh_client: ssh client setup to connect to the server to run the tests on
    cmd: cmd to run in the ssh_client
    stdout_file: local file to write standard output of the cmd to
    stderr_file: local file to write standard error of the cmd to
    line_extractor: method to call on each line to determine if the line
    should be printed to the local console.
    print_error: True to print output if there is an error, e.g. non-'0' exit code.

  returns a thread that executes the given cmd
  """
  t = threading.Thread(
      target=_ExecuteCommandAndStreamOutput,
      args=(ssh_client, cmd, stdout_file, stderr_file, line_extractor,
            print_error))
  print(t.daemon)
  t.start()
  return t


def _StreamOutputToFile(fd, file, line_extractor, cmd=None):
  """Stream output to local file print select content to console

  Streams output to a local file and if a line_extractor is passed
  uses it to determine which data is printed to the local console.

  """
  def func(fd, file, line_extractor):
    with open(file, 'ab+') as f:
      if cmd:
        line = cmd + '\n'
        f.write(line.encode('utf-8'))
      try:
        for line in iter(lambda: fd.readline(2048), ''):
          f.write(line.encode('utf-8', errors='ignore'))
          f.flush()
          if line_extractor:
            line_extractor(line)
      except UnicodeDecodeError as err:
        print('UnicodeDecodeError parsing stdout/stderr, bug in paramiko:{}'
              .format(err))
  t = threading.Thread(target=func, args=(fd, file, line_extractor))
  t.start()
  return t


def _ExecuteCommandAndStreamOutput(ssh_client,
                                  cmd,
                                  stdout_file=None,
                                  stderr_file=None,
                                  line_extractor=None,
                                  print_error=False,
                                  ok_exit_status=[0]):
  """Executes cmd in ssh_client.  Blocking call.


  Args:
    ssh_client: ssh client setup to connect to the server to run the tests on
    cmd: cmd to run in the ssh_client
    stdout_file: local file to write standard output of the cmd to
    stderr_file: local file to write standard error of the cmd to
    line_extractor: method to call on each line to determine if the line
    should be printed to the local console.
    print_error: True to print output if there is an error
    ok_exit_status: List of status codes that are not errors, defaults to '0'

  """
  _, stdout, stderr = ssh_client.exec_command(cmd, get_pty=True)
  if stdout_file:
    t1 = _StreamOutputToFile(stdout, stdout_file, line_extractor, cmd=cmd)
  if stderr_file:
    t2 = _StreamOutputToFile(stderr, stderr_file, line_extractor)
  if stdout_file:
    t1.join()
  if stderr_file:
    t2.join()
  exit_status = stdout.channel.recv_exit_status()
  if exit_status in ok_exit_status:
    return True
  else:
    if print_error:
      print('Command execution failed! Check log. Exit Status({}):{}'.format(exit_status, cmd))
    return False


def lookup_aws_instances(name):
  """Returns all AWS instances for given job."""
  
  ec2 = boto3.resource('ec2')

  # TODO: add waiting so that instances that "initializing" are supported
  instances = ec2.instances.filter(
    Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])

  result = []
  for i in instances.all():
    inst_name = u.get_name(i.tags)
    key_name = i.key_name

    if inst_name == name:
      # if key_name != KEY_NAME:
      #   print("name matches, but key name %s doesn't match %s, skipping"%(key_name, KEY_NAME))
      #   continue
      result.append(i)

    ec2 = boto3.resource('ec2')
  instances = ec2.instances.filter(
    Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])

  result = []
  for i in instances:
    names = []
    if i.tags:
      names = [tag['Value'] for tag in i.tags if tag['Key'] == 'Name']
    key_name = i.key_name

    assert len(names) <= 1
    if names:
      inst_name = names[0]
    else:
      inst_name = ''
    if inst_name == name:
      # if key_name != KEY_NAME:
      #   print("name matches, but key name %s doesn't match %s, skipping"%(key_name, KEY_NAME))
      #   continue
      result.append(i)
  return result

def _maybe_create_placement_group(name):
  client = boto3.client('ec2')
  try:
    client.describe_placement_groups(GroupNames=[name])
  except boto3.exceptions.botocore.exceptions.ClientError as e:
    print("Creating placement group: "+name)
    res = client.create_placement_group(GroupName=name, Strategy='cluster')

  counter = 0
  while True:
    try:
      res = client.describe_placement_groups(GroupNames=[name])
      if res['PlacementGroups'][0]['State'] == 'available':
        print("Found placement group: "+name)
        break
    except Exception as e:
      print(e)
    counter = counter + 1
    if counter >= 10:
      print('Failed to create placement group %s' % name)
    time.sleep(TIMEOUT_SEC)


def _create_ec2_client():
  REGION = u.get_region()
  return boto3.client('ec2', region_name=REGION)


def _create_ec2_resource():
  REGION = u.get_region()
  return boto3.resource('ec2',region_name=REGION)


def _is_good_response(response):
  """Helper method to check if boto3 call was a success."""
  
  return response["ResponseMetadata"]['HTTPStatusCode'] == 200


def _check_security_group_exists(group_name):
  """To catch boto3 errors of the form
  Value () for parameter groupId is invalid. The value cannot be empty
  """

  client = _create_ec2_client()
  response = client.describe_security_groups()
  assert _is_good_response(response)

  result = OrderedDict()
  ec2 = _create_ec2_resource()
  for security_group_response in response['SecurityGroups']:
    observed_group_name = security_group_response['GroupName']
    if group_name == observed_group_name:
      return True
  return False


# TODO: deprecate this or remove reliance on env vars
def simple_job(name, num_tasks=1, instance_type=None, install_script='',
               placement_group='', ami='', linux_type='ubuntu'):
  """Creates simple job on AWS cluster. If job with same name already
  exist on AWS cluster, then reuse those instances instead of creating new.

  Reuse requires that that job launched previous under same name has identical
  settings (number of tasks/instace type/placement group)
  """

  global ROOT_INSTALL_SCRIPT
  if linux_type == 'ubuntu':
    ROOT_INSTALL_SCRIPT = ROOT_INSTALL_SCRIPT_UBUNTU
  elif linux_type == 'amazon':
    ROOT_INSTALL_SCRIPT = ROOT_INSTALL_SCRIPT_AMAZON
  else:
    assert False, "Unknown linux type '%s', expected 'ubuntu' or 'amazon'."

  if instance_type is None:
    instance_type = 'c5.large'
  instances = lookup_aws_instances(name)
  if instances:
    assert len(instances) == num_tasks, ("Found job with same name, but number"
       " of tasks %d doesn't match requested %d, kill job manually."%(len(instances), num_tasks))
    print("Found existing job "+name)
  else:
    print("Launching new job "+name)

    ec2 = boto3.resource('ec2')
    if placement_group:
      _maybe_create_placement_group(placement_group)
      
    print("Requesting %d %s" %(num_tasks, instance_type))

    if not ami:
      ami = os.environ.get('AMI', '')

    assert ami, "No AMI specified, need AMI envvar or explicit parameter"

    args = {'ImageId':ami,
            'InstanceType':instance_type,
            'MinCount':num_tasks,
            'MaxCount':num_tasks,
            'SecurityGroups':[SECURITY_GROUP],
            'KeyName':KEY_NAME}
    if placement_group:
      args['Placement']={'GroupName': placement_group}

    assert _check_security_group_exists(SECURITY_GROUP), "Security group '%s' does not exist in region '%s'" %(SECURITY_GROUP, u.get_region)
    
    instances = ec2.create_instances(**args)
    
    for instance in instances:
      tag = ec2.create_tags(
        Resources=[instance.id], Tags=[{
            'Key': 'Name',
            'Value': name
        }])

    assert len(instances) == num_tasks
    print('{} Instances created'.format(len(instances)))

  job = Job(name, instances=instances, install_script=install_script,
            linux_type=linux_type)
  return job

# TODO: remove requirement to specify availability zones
# TODO: add checks on instance_type when launching into existing job
def server_job(name, num_tasks=1, instance_type=None, install_script='',
               placement_group='', ami='', availability_zone='',
               linux_type=DEFAULT_LINUX_TYPE):
  """Creates a job on AWS cluster with publicly facing ports.

  Reuse requires that that job launched previous under same name has identical
  settings (number of tasks/instace type/placement group)
  """

  global SSH_KEY_PATH

  DEFAULT_NAME = u.RESOURCE_NAME
  security_group = u.get_security_group_dict()[DEFAULT_NAME]
  keypair = u.get_keypair_dict()[DEFAULT_NAME]
  # get availability zone -> subnet dictionary
  vpc = u.get_vpc_dict()[DEFAULT_NAME]
  subnet_dict = {}
  for subnet in vpc.subnets.all():
    zone = subnet.availability_zone
    assert zone not in subnet_dict, "More than one subnet in %s, why?" %(zone,)
    subnet_dict[zone] = subnet
  subnet = subnet_dict[availability_zone]
  
  global ROOT_INSTALL_SCRIPT
  if linux_type == 'ubuntu':
    ROOT_INSTALL_SCRIPT = ROOT_INSTALL_SCRIPT_UBUNTU
  elif linux_type == 'amazon':
    ROOT_INSTALL_SCRIPT = ROOT_INSTALL_SCRIPT_AMAZON
  else:
    assert False, "Unknown linux type '%s', expected 'ubuntu' or 'amazon'."

  if instance_type is None:
    instance_type = 'c5.large'
  instances = lookup_aws_instances(name)

  # todo: get rid of this global variable?
  SSH_KEY_PATH = "%s/%s-%s.pem" % (os.environ["HOME"], DEFAULT_NAME,
                                          u.get_region(),)

  if instances:
    assert len(instances) == num_tasks, ("Found job with same name, but number"
       " of tasks %d doesn't match requested %d, kill job manually."%(len(instances), num_tasks))
    print("Found existing job "+name)
  else:
    print("Launching new job %s into VPC %s" %(name, DEFAULT_NAME))

    ec2 = boto3.resource('ec2')
    if placement_group:
      _maybe_create_placement_group(placement_group)
      
    print("Requesting %d %s" %(num_tasks, instance_type))

    if not ami:
      ami = os.environ.get('AMI', '')

    assert ami, "No AMI specified, need AMI env-var or explicit parameter"

    args = {'ImageId': ami,
            'InstanceType': instance_type,
            'MinCount': num_tasks,
            'MaxCount': num_tasks,
            'KeyName': keypair.name}

    # network setup
    args['NetworkInterfaces'] = [{'SubnetId': subnet.id,
                                 'DeviceIndex': 0,
                                 'AssociatePublicIpAddress': True,
                                 'Groups': [security_group.id]}]
    
    placement_arg = {'AvailabilityZone': availability_zone}
    if placement_group: placement_arg['GroupName'] = placement_group
    args['Placement'] = placement_arg
      
    instances = ec2.create_instances(**args)

    # todo: use task index in name
    for instance in instances:
      while True:
        try:
          # sometimes get "An error occurred (InvalidInstanceID.NotFound)"
          tag = ec2.create_tags(
            Resources=[instance.id], Tags=[{
                'Key': 'Name',
                'Value': name
            }])
          break
        except Exception as e:
          self.log("create_tags failed with %s, retrying in %d seconds"%(
            str(e), TIMEOUT_SEC))
          time.sleep(TIMEOUT_SEC)


    assert len(instances) == num_tasks
    print('{} Instances created'.format(len(instances)))

  job = Job(name, instances=instances, install_script=install_script,
            linux_type=linux_type)
  return job


def _ssh_to_host(hostname,
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

  print("ssh_to_host %s@%s"%(username, hostname))
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


class Job:
  def __init__(self, name, instances, install_script="",
               linux_type="also-crash-linux-type"):
    self.name = name
    self.tasks = []
    # todo: make task_ids asignment deterministic
    for task_id, instance in enumerate(instances):
      self.tasks.append(Task(instance, self, task_id,
                             install_script=install_script,
                             linux_type=linux_type))

  def initialize(self):
    for task in self.tasks:
      task.initialize()  # todo: make initialization run in parallel

  # todo: rename to initialize
  def wait_until_ready(self):
    """Waits until all tasks in the job are available and initialized."""
    for task in self.tasks:
      task.wait_until_ready()
      # todo: initialization should start async in constructor instead of here

def _encode_float(value):
  ba = bytearray(struct.pack('d', value))  
  return base64.b16encode(ba).decode('ascii')

def _decode_float(b16):
  return struct.unpack('d', base64.b16decode(b16))[0]

def _add_echo(script):
  """Goes over install script, adds "echo cmd" in front of each cmd.

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


# add access to instance id from task
class Task:
  def __init__(self, instance, job, task_id, install_script="",
               linux_type="crash-here-linux-type"):
    self.initialized_called = False
    self.instance = instance
    self.job = job
    self.id = task_id
    self.install_script = install_script
    
    self.initialized = False
    self.cmd_idx = 0
    self.local_tasklogdir = '%s/%s/%s' %(LOCAL_TASKLOGDIR_PREFIX, self.job.name,
                                         self.id)
    self.last_stdout = None  # path of last stdout file location
    self.last_stderr = None  # path of last stderr file location
    self.connect_instructions = "waiting for initialize()"

    # username to use to ssh into instances
    # ec2-user or ubuntu
    if linux_type == 'ubuntu':
      self.username = USERNAME_UBUNTU
    elif linux_type == 'amazon':
      self.username = USERNAME_AMAZON
    else:
      assert False, "Unknown linux type '%s', expected 'ubuntu' or 'amazon'."


  def log(self, message, args=None):
    """Log to client console."""
    ts = _current_timestamp()

    if args:
      message = message % args
    print("%s %s:%d: (%s) %s"%(ts, self.job.name, self.id, self.ip, message))
    
  # todo: rename wait_until_ready to wait_until_initialized
  def wait_until_ready(self):
    if not self.initialized_called:
      self.initialize()
    else:
      assert False, "Don't call initialize, just call wait_until_ready()"
    while not self.initialized:
      if self.initialized:
        break
      self.log("wait_until_ready: Not initialized, retrying in %d seconds"%(TIMEOUT_SEC))
      time.sleep(TIMEOUT_SEC)
      

  def wait_until_file_ok(self, fn):
    while not self._is_custom_file_present(fn):
      self.log("wait_until_file_ok: not present, waiting %d seconds"%(TIMEOUT_SEC))
      time.sleep(TIMEOUT_SEC)

  def initialize(self):
    """Tries to initialize the task. This can fail for various reasons, so
    the user must retry until self.initialized is True."""

    self.log("Running initialize")
    self.initialize_called = True
    while True:
      try:
        public_ip = self.public_ip
        break
      except:
        self.log("no public IP, retrying in %d seconds"%(TIMEOUT_SEC))
      time.sleep(TIMEOUT_SEC)

    # todo: this sometimes fails because public_ip is not ready
    # add query/wait?
    while True:
      self.ssh_client = _ssh_to_host(self.public_ip, SSH_KEY_PATH,
                                     self.username)
      if self.ssh_client is None:
        self.log("SSH into %s:%s failed, retrying in %d seconds" %(self.job.name, self.id,TIMEOUT_SEC))
        time.sleep(TIMEOUT_SEC)
      else:
        break
    
    # run initialization commands here
    if self._is_initialized_file_present():
      self.log("reusing previous initialized state")
      self._setup_tmux()
    else:
      self.log("running install script")
      for cmd in ROOT_INSTALL_SCRIPT.split('\n'):
        cmd = cmd.strip()
        if not cmd:
          continue
        # todo: add checking of return codes to report when some command failed
        self.run_sync(cmd)
      self._setup_tmux()
      # TODO: make upload/send commands work

      if BULK_INSTALL:
        self.install_script+='\necho ok > /tmp/is_initialized\n'
        open('/tmp/install.sh','w').write(self.install_script)
        open('/tmp/echo_install.sh','w').write(_add_echo(self.install_script))
        self.upload('/tmp/install.sh')
        self.upload('/tmp/echo_install.sh')
        self.run('bash -e echo_install.sh') # fail on errors
      else:
        for cmd in self.install_script.split('\n'):
          cmd = cmd.strip()
          if not cmd:
            continue

          self.run(cmd)
          time.sleep(5)  # avoid overwhelming tmux

        self.run("echo 'ok' > /tmp/is_initialized")
        

    self.connect_instructions = """
ssh -i %s -o StrictHostKeyChecking=no %s@%s
tmux a
""".strip() % (SSH_KEY_PATH, self.username, self.public_ip)

    # wait for things to install
    while True:
      self.initialized = self._is_initialized_file_present()
      if self.initialized:
        break
      self.log("initialize: no is_initialized file, waiting %d seconds"%(TIMEOUT_SEC))
      time.sleep(TIMEOUT_SEC)


  def _is_custom_file_present(self, remote_fn):
    self.log("Checking for custom file "+remote_fn)
    try:
      # construct unique local name
      local_fn = "%d-%d.is_initialized"%(self.id, int(time.time()*1e6))
      local_fn = LOCAL_TASKLOGDIR_PREFIX+'/'+local_fn
      # todo: replace with file_read()
      self.download(remote_fn, local_fn)
      return 'ok' in open(local_fn).read()
    except Exception as e:
      self.log("Got exception %s"%(e,))
      return False

  def _is_initialized_file_present(self):
    self.log("Checking for initialized file")
    # TODO: replace this with file_read()
    try:
      # construct unique local name
      fn = "%d-%d.is_initialized"%(self.id, int(time.time()*1e6))
      fn = LOCAL_TASKLOGDIR_PREFIX+'/'+fn
      self.download('/tmp/is_initialized', fn)
      return 'ok' in open(fn).read()
    except:
      return False


  # TODO: rename to run_ssh (vs run_tmux, run->run_tmux)
  def run_sync(self, cmd):
    """Runs given cmd in the task, returns stdout/stderr as strings.
    Because it blocks until cmd is done, use it for short cmds.
   
    Also, because this can run before task is initialized, use it
    for running initialization commands in sequence.

    This is a barebones method to be used during initialization that have
    minimal dependencies (no tmux)
    """
    self.log("run_sync: %s"%(cmd,))
    stdin, stdout, stderr = self.ssh_client.exec_command(cmd, get_pty=True)
    stdout_str = stdout.read().decode('ascii')
    stderr_str = stderr.read().decode('ascii')
    self.log("run_sync returned: " + stdout_str)
    return stdout_str, stderr_str

  def _setup_tasklogdir(self):
    if not os.path.exists(self.local_tasklogdir):
      os.system('mkdir -p '+self.local_tasklogdir)


  def _setup_tmux(self):
    # shell command will fail if session exists, but it's ok
    self.run_sync('mkdir -p /tmp/tmux')
    self.run_sync('tmux kill-session -t tmux')
    self.run_sync('tmux new-session -s tmux -n 0 -d')

  def run(self, cmd, max_wait_sec=600, check_interval=1, ignore_errors=False,
          wait_to_finish=True):
    """Runs command in tmux session. No need for multiple tmux sessions per
    task, so assume tmux session/window is always called tmux:0"""

    self.cmd_idx+=1
    self.log("tmux> %s", cmd)

    if cmd.startswith("upload "):
      _, fname = cmd.split()
      fname = fname.replace("~", os.environ["HOME"])
      self.upload(fname)

    # locking to wait for command to finish
    ts = str(u.now_micros())
    cmd_fn_out = '/tmp/tmux/'+str(self.cmd_idx)+'.'+ts+'.out'

    modified_cmd = '%s; echo $? > %s'%(cmd, cmd_fn_out)
    tmux_window = 'tmux:0'
    tmux_cmd = "tmux send-keys -t {} '{}' Enter".format(tmux_window,
                                                        modified_cmd)
    self.run_sync(tmux_cmd)
    if not wait_to_finish:
      return
    
    start_time = time.time()

    while True:
      if time.time() - start_time > max_wait_sec:
        assert False, "Timeout %s exceeded for %s" %(max_wait_sec, cmd)
      if not self.file_exists(cmd_fn_out):
        self.log("waiting %s for %s"%(check_interval, cmd))
        time.sleep(check_interval)
        continue
    
      contents = self.file_read(cmd_fn_out)
      # if empty wait a bit to allow for race condition
      if len(contents) == 0:
        time.sleep(check_interval)
        contents = task.file_read(cmd_fn_out)

      contents = contents.strip()
      if contents != '0':
        if not ignore_errors:
          assert False, "Command %s returned status %s"%(cmd, contents)
        else:
          self.log("Warning: command %s returned status %s"%(cmd, contents))
      break    


  def upload(self, local_fn, remote_fn=None, skip_existing=False):
    """Uploads file to remote instance. If location not specified, dumps it
    in default directory."""
    # TODO: self.ssh_client is sometimes None
    sftp = self.ssh_client.open_sftp()
    if remote_fn is None:
      remote_fn = os.path.basename(local_fn)
    if skip_existing and self.file_exists(remote_fn):
      self.log("Remote file %s exists, skipping"%(remote_fn,))
    else:
      sftp.put(local_fn, remote_fn)

  def download(self, remote_fn, local_fn=None):
    # TODO: self.ssh_client is sometimes None
    sftp = self.ssh_client.open_sftp()
    if local_fn is None:
      local_fn = os.path.basename(local_fn)
    self.log("downloading %s to %s"%(remote_fn, local_fn))
    sftp.get(remote_fn, local_fn)

  def file_exists(self, remote_fn):
    stdin, stdout, stderr = self.ssh_client.exec_command('stat '+remote_fn,
                                                    get_pty=True)
    stdout_bytes = stdout.read()
    stdout_str = stdout_bytes.decode()
    stderr_bytes = stderr.read()
    stderr_str = stderr_bytes.decode()
    if 'No such file' in stdout_str:
      return False
    else:
      return True
  
  def file_write(self, remote_fn, contents):
    # TODO: create tasklogdir for everything for given job
    tmp_fn = '/tmp/tmux/'+str(u.now_micros())
    open(tmp_fn, 'w').write(contents)
    self.upload(tmp_fn, remote_fn)
  

  def file_read(self, remote_fn):
    # TODO: create tasklogdir for everything for given job
    tmp_fn = '/tmp/tmux/'+str(u.now_micros())
    self.download(remote_fn, tmp_fn)
    return open(tmp_fn).read()

  # TODO: make private
  def mount_efs(self):
    region = u.get_region()
    efs_id = u.get_efs_dict()[u.RESOURCE_NAME]
    dns = "{efs_id}.efs.{region}.amazonaws.com".format(**locals())
    self.run('sudo mkdir -p /efs')
    self.run('sudo chmod 777 /efs')
    self.run("sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 %s:/ /efs"%(dns,), ignore_errors=True) # error on remount

  
  @property
  def public_ip(self):
    self.instance.load()
    return self.instance.public_ip_address

  @property
  def port(self):
    return DEFAULT_PORT

  @property
  def ip(self):  # private ip
    self.instance.load()
    return self.instance.private_ip_address
