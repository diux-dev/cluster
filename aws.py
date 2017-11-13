import threading
import base64
import struct
from collections import OrderedDict
from pprint import pprint as pp
import argparse
import boto3
import os
import sys
import time
#import util as myutil
import yaml

from myutil import timeit

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_path+'/tf-tools/benchmark/runner')

# User specific config. These settings are reused for all jobs.
AMI = 'ami-9ddb0fe5'
KEY_NAME = 'yaroslav'  # AWS key-name to use, only modify instances with this
# key
SSH_KEY_PATH = os.environ['HOME']+'/d/yaroslav.pem' # location of .pem file locally
SECURITY_GROUP = 'open' # security group for all instances


LOCAL_TASKLOGDIR_PREFIX='/temp/tasklogs'
LOCAL_LOGDIR_PREFIX='/temp/logs'

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
      if key_name != KEY_NAME:
        print("name matches, but key name %s doesn't match %s, skipping"%(key_name, KEY_NAME))
        continue
      result.append(i)
  return result

def tf_job(name, num_tasks):
  """Creates TensorFlow job on AWS cluster. If job with same name already
  exist on AWS cluster, then reuse those instances instead of creating new."""

  # assume if given job exists, it's been configured properly
  # this is a performance optimization to avoid AWS startup delay
  instances = lookup_aws_instances(name)
  if instances:
    assert len(instances) == num_tasks, ("Found job with same name, but number"
       " of tasks %d doesn't match requested %d, kill job manually."%(len(instances), num_tasks))
    print("Found existing job "+name)
  else:
    print("Launching new job "+name)

    ec2 = boto3.resource('ec2')
    instances = ec2.create_instances(
      ImageId=AMI,
      InstanceType='c5.large',
      MinCount=num_tasks,
      MaxCount=num_tasks,
      SecurityGroups=[SECURITY_GROUP],
      KeyName=KEY_NAME)
    
    for instance in instances:
      tag = ec2.create_tags(
        Resources=[instance.id], Tags=[{
            'Key': 'Name',
            'Value': name
        }])

    assert len(instances) == num_tasks
    print('{} Instances created'.format(len(instances)))
    
  job = Job(name, instances=instances)
  
  # todo: setup EFS logdir
  # todo: setup remote tasklogdir?

  return job

def terminate_job(name):
  instances = lookup_aws_instances(name)
  for i in instances:
    print("Killing '%s' '%s' '%s'" %(name, i.id, i.instance_type))
    i.terminate()

  for i in instances:
    i.load()
    while True:
      if i.state['Name'] ==  'terminated':
        break
      print("Waiting for %s to die, instance state is %s"%(i.id, i.state))
      time.sleep(5)
      i.load()

import paramiko
def _ssh_to_host(hostname,
              ssh_key=None,
              username='ubuntu',
              retry=10):

  """Create ssh connection to host

  Creates and returns and ssh connection to the host passed in.  

  Args:
    hostname: host name or ip address of the system to connect to.
    retry: number of time to retry.
    ssh_key: full path to the ssk hey to use to connect.
    username: username to connect with.

  returns SSH client connected to host.

  """

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
        print('Got impatient with retrying ssh to host. Time to give up.')
        return None

  return ssh_client


class Job:
  def __init__(self, name, instances):
    self.name = name
    self.tasks = []
    for task_id, instance in enumerate(instances):
      self.tasks.append(Task(instance, self, task_id))

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

class Task:
  def __init__(self, instance, job, task_id):
    self.instance = instance
    self.job = job
    self.task_id = task_id
    self.initialized = False
    self.local_tasklogdir = '%s/%s/%s' %(LOCAL_TASKLOGDIR_PREFIX, self.job.name,
                                         self.task_id)
    self.last_stdout = None  # path of last stdout file location
    self.last_stderr = None  # path of last stderr file location


  def wait_until_ready(self):
    while not self.initialized:
      self.initialize()
      if self.initialized:
        break
      time.sleep(5)
    self.connect_instructions = '<todo: add instructions>'
      

  def initialize(self):
    # todo: do we need to wait until public_ip is available?
    assert self.public_ip
    self.ssh_client = _ssh_to_host(self.public_ip, SSH_KEY_PATH)
    # this blocks until instance is up
    self.initialized = True
    
  def run_sync(self, cmd):
    """Runs given cmd in the task, returns stdout/stderr as strings.
    Because it blocks until cmd is done, use it for short cmds."""
    # TODO: run doesn't preserve tty
    # find paramiko recipe to use tty and use that
    stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
    stdout_str = stdout.read().decode('ascii')
    stderr_str = stderr.read().decode('ascii')
    return stdout_str, stderr_str

  def _setup_tasklogdir(self):
    if not os.path.exists(self.local_tasklogdir):
      os.system('mkdir -p '+self.local_tasklogdir)
      
  def run(self, cmd, mirror_output=False):
    """Runs given command in the task, streams stdout/stderr to local files."""

    assert self.initialized, ("Trying to run command on task that's not "
                              "initialized")
    
    self._setup_tasklogdir()
    timestamp = _encode_float(time.time())
    stdout_fn = "%s/%s.stdout"%(self.local_tasklogdir, timestamp)
    stderr_fn = "%s/%s.stderr"%(self.local_tasklogdir, timestamp)
    self.last_stdout = stdout_fn
    self.last_stderr = stderr_fn

    if mirror_output:
      def line_extractor(line):
        print(line)
    else:
      line_extractor = None
      
    _ExecuteCommandInThread(ssh_client=self.ssh_client,
                            cmd=cmd,
                            stdout_file=stdout_fn,
                            stderr_file=stderr_fn,
                            line_extractor=line_extractor)

  def upload(self, local_file, remote_file=None):
    """Uploads file to remote instance. If location not specified, dumps it
    in default directory with same name."""
    self.wait_until_ready()
      
    sftp = self.ssh_client.open_sftp()
    if remote_file is None:
      remote_file = os.path.basename(local_file)
    sftp.put(local_file, remote_file)
    
  @property
  def public_ip(self):
    self.instance.load()
    return self.instance.public_ip_address

  @property
  def ip(self):  # private ip
    self.instance.load()
    return self.instance.private_ip_address
