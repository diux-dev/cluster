# AWS implementation of backend.py

# todo: move EFS mounting into userdata for things to happen in parallel
# TODO: fix remote_fn must be absolute for uploading with check_with_existing
import glob
import os
import shlex
import sys
import time

import backend
import util as u

TASKDIR_PREFIX='/tmp/tasklogs'
LOGDIR_PREFIX='/efs/runs'
TIMEOUT_SEC=5
MAX_RETRIES = 10
DEFAULT_PORT=3000  # port used for task internal communication
TENSORBOARD_PORT=6006  # port used for external HTTP communication

def make_run(name, **kwargs):
  return Run(name, **kwargs)


def _strip_comment(cmd):
  """ hi # testing => hi"""
  if '#' in cmd:
    return cmd.split('#', 1)[0]
  else:
    return cmd


class Run(backend.Run):
  """In charge of creating resources and allocating instances. AWS instances
  are then wrapped in Job and Task objects."""
  
  def __init__(self, name, **kwargs):
    self.name = name

    # run name is used in tmux session name and instance name, so must restrict
    # run name to also be valid part of tmux/instance names
    # -Maximum number of tags per resource—50
    # -Maximum key length—127 Unicode characters in UTF-8
    # -Maximum value length—255 Unicode characters in UTF-8
    # letters, spaces, and numbers representable in UTF-8, plus the following special characters: + - = . _ : / @.
    import re
    assert len(name)<30
    assert re.match("[-+=._:\/@a-zA-Z0-9]+", name)
    
    self.kwargs = kwargs
    self.jobs = []

  # TODO: get rid of linux type (only login username)
  def make_job(self, role_name, num_tasks=1, **kwargs):
    assert num_tasks>=0

    # TODO: document launch parameters
    job_name = u.format_job_name(role_name, self.name)
    instances = u.lookup_aws_instances(job_name)
    kwargs = u.merge_kwargs(kwargs, self.kwargs)
    ami = kwargs.get('ami', '')
    ami_name = kwargs.get('ami_name', '')
    instance_type = kwargs['instance_type']
    availability_zone = kwargs['availability_zone']
    placement_group = kwargs.get('placement_group', '')
    install_script = kwargs.get('install_script','')
    skip_efs_mount = kwargs.get('skip_efs_mount', False)
    linux_type = kwargs.get('linux_type', 'ubuntu')
    user_data = kwargs.get('user_data', '')
    ebs = kwargs.get('ebs', '')
    use_spot = kwargs.get('use_spot', False)

    if user_data:
      user_data+='\necho userdata_ok >> /tmp/is_initialized\n'

    #    print("Using user_data", user_data)

    # TODO: also make sure instance type is the same
    if instances:
      assert len(instances) == num_tasks, ("Found job with same name, but number of tasks %d doesn't match requested %d, kill job manually." % (len(instances), num_tasks))
      print("Found existing job "+job_name)
      for i in instances:
            if i.state['Name'] == 'stopped': i.start()
      print(instances)
    else:
      print("Launching new job %s into VPC %s" %(job_name, u.get_resource_name()))

      assert not (ami and ami_name), "Must have only one of ami and ami_name, got "+ami+", "+ami_name
      if ami_name:
        ami = u.lookup_ami_id(ami_name).id
      security_group = u.get_security_group_dict()[u.get_resource_name()]
      keypair = u.get_keypair_dict()[u.get_keypair_name()]
      vpc = u.get_vpc_dict()[u.get_resource_name()]
      subnet_dict = u.get_subnet_dict(vpc)
      region = u.get_region()
      assert availability_zone in subnet_dict, "Availability zone %s is not in subnet dict for current AWS default region %s, available subnets are %s. (hint, set AWS_DEFAULT_REGION)"%(availability_zone, region, ', '.join(subnet_dict.keys()))
      subnet = subnet_dict[availability_zone]
      ec2 = u.create_ec2_resource()
      u.maybe_create_placement_group(placement_group)

      self.log("Requesting %d %s" %(num_tasks, instance_type))

      args = {'ImageId': ami,
              'InstanceType': instance_type,
              'MinCount': num_tasks,
              'MaxCount': num_tasks,
              'KeyName': keypair.name}
              
      # storage setup
      if ebs: args['BlockDeviceMappings'] = ebs
      # network setup
      args['NetworkInterfaces'] = [{'SubnetId': subnet.id,
                                    'DeviceIndex': 0,
                                    'AssociatePublicIpAddress': True,
                                    'Groups': [security_group.id]}]
      

      placement_arg = {'AvailabilityZone': availability_zone}
      
      if placement_group: placement_arg['GroupName'] = placement_group
      args['Placement'] = placement_arg
      args['UserData'] = user_data

      if use_spot: instances = u.create_spot_instances(args)
      else: instances = ec2.create_instances(**args)
      assert len(instances) == num_tasks

      # assign proper names to tasks
      for instance in instances:
        while True:
          try:
            # sometimes get "An error occurred (InvalidInstanceID.NotFound)"
            task_name = u.format_task_name(instance.ami_launch_index, role_name,
                                           self.name)
            instance.create_tags(Tags=u.make_name(task_name))
            break
          except Exception as e:
            self.log("create_tags failed with %s, retrying in %d seconds"%(
              str(e), TIMEOUT_SEC))
            time.sleep(TIMEOUT_SEC)
    job = Job(self, job_name, instances=instances,
              install_script=install_script,
              linux_type=linux_type,
              user_data=user_data,
              skip_efs_mount=skip_efs_mount)
    self.jobs.append(job)
    return job

  @property
  def logdir(self):
    return LOGDIR_PREFIX+'/'+self.name


class Job(backend.Job):
  # TODO: get rid of linux_type
  def __init__(self, run, name, instances, install_script=None,
               linux_type=None, user_data='', skip_efs_mount=False):
    self._run = run
    self.name = name


    self._run_command_available = False, "Have you done wait_until_ready?"
    
    # initialize list of tasks, in order of AMI launch index
    self.tasks = [None]*len(instances)
    for instance in instances:
      task_id, current_job_name = u.get_parsed_job_name(instance.tags) # use job name in case ami's were not launched at the same time
      task_id = task_id or instance.ami_launch_index
      task = Task(instance, self, task_id, install_script=install_script,
                  linux_type=linux_type, user_data=user_data,
                  skip_efs_mount=skip_efs_mount)
      self.tasks[task_id] = task


  # def run_async_join(self, cmd, *args, **kwargs):
  #   import threading
  #   """Runs command on every task in the job async. Then waits for all to finish"""
  #   def t_run_cmd(t): t.run(cmd, *args, **kwargs)
  #   t_threads = [threading.Thread(name=f't_{i}', target=t_run_cmd, args=[t]) for i,t in enumerate(self.tasks)]
  #   for thread in t_threads: thread.start()
  #   for thread in t_threads: thread.join()

  def _initialize(self):
    for task in self.tasks:
      task._initialize()


class Task(backend.Task):
  # TODO: replace linux_type with username
  def __init__(self, instance, job, task_id, install_script=None,
               linux_type=None, user_data='', skip_efs_mount=False):
    self.initialize_called = False
    self.instance = instance
    self.job = job
    self.id = task_id
    self.install_script = install_script
    self.user_data = user_data
    self._run_counter = 0
    self.cached_ip = None
    self.cached_public_ip = None
    self.skip_efs_mount = skip_efs_mount
    
    self.initialized = False

    # scratch is client-local space for temporary files
    self.scratch = "{}/{}.{}.{}.{}/scratch".format(TASKDIR_PREFIX,
                                                   job._run.name,
                                                   job.name, self.id,
                                                   0) # u.now_micros())
    self.remote_scratch = '/tmp/tmux'
    #    self.log("Creating local scratch dir %s", self.scratch)
    self._ossystem('rm -Rf '+self.scratch)  # TODO: don't delete this?
    self._ossystem('mkdir -p '+self.scratch)
    #    os.chdir(self.scratch)

    # todo: create taskdir
    self.connect_instructions = "waiting for initialize()"
    self.keypair_fn = u.get_keypair_fn(u.get_keypair_name())

    # username to use to ssh into instances
    # ec2-user or ubuntu
    if linux_type == 'ubuntu':
      self.username = 'ubuntu'
    elif linux_type == 'amazon':
      self.username = 'ec2-user'
    else:
      assert False, "Unknown linux type '%s', expected 'ubuntu' or 'amazon'."

    self.taskdir = '/home/'+self.username

  # todo: replace with file_read
  def _is_initialized_file_present(self):
    self.log("Checking for /tmp/is_initialized file")
    try:
      # TODO: either get rid of /tmp or taskdir
      # this location is hardwired in uninitialize.py
      return 'ok' in self.file_read('/tmp/is_initialized')
    except:
      return False

  def _setup_tmux(self):
    self._tmux_session_name = self.job._run.name
    self._run_ssh('tmux kill-session -t '+self._tmux_session_name)
    self._run_ssh('tmux new-session -s %s -n 0 -d'%(self._tmux_session_name,))
    self._run_command_available = True

  def _mount_efs(self):
    self.log("Mounting EFS")
    region = u.get_region()
    efs_id = u.get_efs_dict()[u.get_resource_name()]
    dns = "{efs_id}.efs.{region}.amazonaws.com".format(**locals())
    self.run('sudo mkdir -p /efs')
    self.run('sudo chmod 777 /efs')
    # ignore error on remount
    self.run("sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 %s:/ /efs"%(dns,), ignore_errors=True) 

  def _initialize(self):
    """Tries to initialize the task."""

    self.log("Running initialize")
    self.initialize_called = True
    public_ip = self.public_ip # todo: add retry logic to public_ip property

    while True:
      self.ssh_client = u.ssh_to_host(self.public_ip, self.keypair_fn,
                                      self.username)
      if self.ssh_client is None:
        self.log("SSH into %s:%s failed, retrying in %d seconds" %(self.job.name, self.id,TIMEOUT_SEC))
        time.sleep(TIMEOUT_SEC)
      else:
        break

    # todo: install tmux
    self._setup_tmux()
    self.run('mkdir -p '+self.remote_scratch)
    if not self.skip_efs_mount:
      self._mount_efs()

    # run initialization commands here
    if self._is_initialized_file_present():
      self.log("reusing previous initialized state")
    elif self.install_script:
      self.log("running install script")

      self.install_script+='\necho ok > /tmp/is_initialized\n'
      self.file_write('install.sh', u._add_echo(self.install_script))
      self.run('bash -e install.sh', max_wait_sec=2400) # fail on errors
      # TODO(y): propagate error messages printed on console to the user
      # right now had to log into tmux to see it
      assert self._is_initialized_file_present()
    else:
      self.log('No install script. Skipping to end')
      # installation happens through user-data instead of install script
      # if neither one is passed, manually create is_initialized
      self.run('echo ok > /tmp/is_initialized')


    self.connect_instructions = """
ssh -i %s -o StrictHostKeyChecking=no %s@%s
tmux a
""".strip() % (self.keypair_fn, self.username, self.public_ip)
    self.log("Initialize complete")
    self.log(self.connect_instructions)


  # todo: rename wait_until_ready to wait_until_initialized
  def wait_until_ready(self):
    if not self.initialize_called:
      self._initialize()
    while not self._is_initialized_file_present():
      self.log("wait_until_ready: Not initialized, retrying in %d seconds"%(TIMEOUT_SEC))
      time.sleep(TIMEOUT_SEC)


  # TODO: dedup with tmux_backend.py?
  def _upload_handler(self, line):
    """Handle following types of commands.

    Individual files, ie
    %upload file.txt

    Individual files with target location
    %upload file.txt remote_dir/file.txt

    #    Glob expressions, ie
    #    %upload *.py
    """

    toks = line.split()
    
    # # glob expression, things copied into 
    # if '*' in line:
    #   assert len(toks) == 2
    #   assert toks[0] == '%upload'

    #   for fn in glob.glob(toks[1]):
    #     fn = fn.replace("~", os.environ["HOME"])
    #     self.upload(fn)

    assert toks[0] == '%upload'
    source = toks[1]
    assert len(toks) < 4
    if len(toks)>2:
      target = toks[2]
    else:
      target = None
    self.upload(source, target)
    

  def upload(self, local_fn, remote_fn=None, skip_existing=False):
    """Uploads file to remote instance. If location not specified, dumps it
    in default directory."""
    # TODO: self.ssh_client is sometimes None
    self.log('uploading '+local_fn)
    sftp = self.ssh_client.open_sftp()
    
    if remote_fn is None:
      remote_fn = os.path.basename(local_fn)
    if skip_existing and self.file_exists(remote_fn):
      self.log("Remote file %s exists, skipping"%(remote_fn,))
      return

    if os.path.isdir(local_fn):
      u.put_dir(sftp, local_fn, remote_fn)
    else:
      assert os.path.isfile(local_fn), "%s is not a file"%(local_fn,)
      sftp.put(local_fn, remote_fn)


  def download(self, remote_fn, local_fn=None):
    # TODO: self.ssh_client is sometimes None
    #    self.log("downloading %s"%(remote_fn))
    sftp = self.ssh_client.open_sftp()
    if local_fn is None:
      local_fn = os.path.basename(local_fn)
      self.log("downloading %s to %s"%(remote_fn, local_fn))
    sftp.get(remote_fn, local_fn)


  def file_exists(self, remote_fn):
    # since ssh commands don't have state, can't rely on current directory
    if not remote_fn.startswith('/'):
      remote_fn = self.taskdir + '/'+remote_fn
    assert remote_fn.startswith('/'), "Remote fn must be absolute"
    stdin, stdout, stderr = self.ssh_client.exec_command('stat '+remote_fn,
                                                    get_pty=True)
    stdout_bytes = stdout.read()
    stdout_str = stdout_bytes.decode()  # AWS linux uses unicode quotes
    stderr_bytes = stderr.read()
    stderr_str = stderr_bytes.decode()
    if 'No such file' in stdout_str:
      return False
    else:
      return True
  
  def file_write(self, remote_fn, contents):
    tmp_fn = self.scratch+'/'+str(u.now_micros())
    open(tmp_fn, 'w').write(contents)
    self.upload(tmp_fn, remote_fn)
  

  def file_read(self, remote_fn):
    #    self.log("file_read")
    tmp_fn = self.scratch+'/'+str(u.now_micros())
    self.download(remote_fn, tmp_fn)
    return open(tmp_fn).read()

  def _run_ssh(self, cmd):
    """Runs given cmd in the task using current SSH session, returns
    stdout/stderr as strings. Because it blocks until cmd is done, use it for
    short cmds.
   
    Also, because this can run before task is initialized, use it
    for running initialization commands in sequence.

    This is a barebones method to be used during initialization that have
    minimal dependencies (no tmux)
    """
    #    self.log("run_ssh: %s"%(cmd,))
    stdin, stdout, stderr = self.ssh_client.exec_command(cmd, get_pty=True)
    stdout_str = stdout.read().decode()
    stderr_str = stderr.read().decode()
    # todo, line below always prints nothing
    #    self.log("run_ssh returned: " + stdout_str)
        
    return stdout_str, stderr_str


  # todo: transition to higher-level SshClient instead of paramiko.SSHClient
  def run(self, cmd, sync=True, ignore_errors=False,
          max_wait_sec=600, check_interval=0.5):
    """Runs command in tmux session. No need for multiple tmux sessions per
    task, so assume tmux session/window is always called tmux:0"""

    assert self._run_command_available, "Have you done wait_until_ready?"
    cmd = cmd.strip()
    
    self._run_counter+=1
    self.log("tmux> %s", cmd)

    # todo: match logic in tmux_session (upload magic handling done
    # in init instead of run)
    if cmd.startswith('%upload'):
      self._upload_handler(cmd)
      return

    # locking to wait for command to finish
    ts = str(u.now_micros())
    cmd_fn_out = self.remote_scratch+'/'+str(self._run_counter)+'.'+ts+'.out'

    cmd = _strip_comment(cmd)
    assert not '&' in cmd, "cmd '%s' contains &, that breaks things"%(cmd,)
    modified_cmd = '%s; echo $? > %s'%(cmd, cmd_fn_out)
    tmux_window = self._tmux_session_name+':0'
    tmux_cmd = "tmux send-keys -t {} {} Enter".format(tmux_window,
                                                        shlex.quote(modified_cmd))
    self._run_ssh(tmux_cmd)
    if not sync:
      return
    
    start_time = time.time()

    while True:
      if time.time() - start_time > max_wait_sec:
        assert False, "Timeout %s exceeded for %s" %(max_wait_sec, cmd)
      if not self.file_exists(cmd_fn_out):
        self.log("waiting for %s"%(cmd,))
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


  def run_and_stream_output(self, cmd, sync):
    """Runs command on task and streams output locally on stderr."""


    stdin, stdout, stderr = self.ssh_client.exec_command(cmd, get_pty=True)
    # todo: add error handling, currently see this in stdout.readline()
    # 'bash: asdfasdf: command not found\r\n'
    if stdout:
      t1 = u._StreamOutputToStdout(stdout)
    if stderr:
      t2 = u._StreamOutputToStdout(stderr)
      
    if stdout and sync:
      t1.join()
    if stderr and sync:
      t2.join()    


  def stream_file(self, fn, sync=True):
    """Does tail -f on task-local file and streams output locally.
    Can use on files that haven't been created yet, it will create the file."""
    if not fn.startswith('/'): fn = self.taskdir+'/'+fn
    print("stream_file", fn)

    # todo: only create this file if it doesn't exist already
    self._run_ssh('mkdir -p '+os.path.dirname(fn))
    self._run_ssh('touch '+fn)
    self.run_and_stream_output('tail -f '+fn, sync)

    
  # todo: follow same logic as in def ip(self)
  @property
  def public_ip(self):
    if not self.cached_public_ip:
      self.instance.load()
      self.cached_public_ip = self.instance.public_ip_address
    return self.cached_public_ip

  @property
  def port(self):
    return DEFAULT_PORT

  @property
  def public_port(self):
    return TENSORBOARD_PORT

  @property
  def ip(self):  # private ip
    # this can fail with following
    #    botocore.exceptions.ClientError: An error occurred (InvalidInstanceID.NotFound) when calling the DescribeInstances operation: The instance ID 'i-0de492df6b20c35fe' does not exist
    if not self.cached_ip:
      retry_sec = 1
      for i in range(10):
        try:
          self.instance.load()
        except Exception as e:
          print("instance.load failed with %s, retrying in %d seconds"%(str(e),
                                                                        retry_sec))
          time.sleep(retry_sec)
      self.cached_ip = self.instance.private_ip_address

    return self.cached_ip
