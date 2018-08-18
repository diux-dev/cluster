# AWS implementation of backend.py

# todo: move EFS mounting into userdata for things to happen in parallel
# TODO: fix remote_fn must be absolute for uploading with check_with_existing
import glob
import os
import shlex
import sys
import time
import datetime

import backend
import util as u

TASKDIR_PREFIX='/tmp/tasklogs'
TIMEOUT_SEC=5  # todo: rename to RETRY_INTERVAL_SEC
MAX_RETRIES = 10
DEFAULT_PORT=3000  # port used for task internal communication
TENSORBOARD_PORT=6006  # port used for external HTTP communication

# TODO: a way to capture output of task.run. This could help with checking if umount is needed ('/dev/xdvf' in df)

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
    u.validate_run_name(name)
    
    self.logdir_ = None   # set during setup_logdir()
    self.kwargs = kwargs
    self.jobs = []
    self.placement_group_name = self.name+'-'+u.random_id()

  @property
  def logdir(self):
    assert self.logdir_ is not None, "logdir not yet initialized"
    return self.logdir_
  
  # TODO: get rid of linux type (only login username)
  # move everything into kwargs
  def make_job(self, role_name, num_tasks=1, skip_existing_job_validation=False, **kwargs):
    """skip_existing_job_validation: if True, doesn't check that existing job on server has same number of tasks as requested."""

    u.maybe_create_resources()

    assert num_tasks>=0

    # TODO: document launch parameters
    job_name = u.format_job_name(role_name, self.name)
    instance_type = kwargs['instance_type']
    instances = u.lookup_aws_instances(job_name, instance_type=instance_type)
    kwargs = u.merge_kwargs(kwargs, self.kwargs)
    ami = kwargs.get('ami', '')
    ami_name = kwargs.get('ami_name', '')
    availability_zone = kwargs.get('availability_zone', '')
    if not availability_zone:
      availability_zone = os.environ['ZONE']
    placement_group = kwargs.get('placement_group', '')

    # automatically generated placement_group_name
    use_placement_group = kwargs.get('use_placement_group', False)
    assert use_placement_group == False or placement_group == ''
    if use_placement_group:
      placement_group = self.placement_group_name

    
    install_script = kwargs.get('install_script','')
    skip_efs_mount = kwargs.get('skip_efs_mount', False)
    linux_type = kwargs.get('linux_type', 'ubuntu')
    # TODO: use heuristics to tell linux type from AMI name
    user_data = kwargs.get('user_data', '')

    if user_data:
      assert user_data.startswith('#!/bin/bash')
      
    ebs = kwargs.get('ebs', '')
    use_spot = kwargs.get('use_spot', False)
    monitoring = kwargs.get('monitoring', True)

    # always install tmux on Amazon linux types
    # TODO: has no effect for some reason
    # https://console.aws.amazon.com/support/v1?region=us-west-2#/case/?displayId=5256445351&language=en
    if linux_type == 'amazon':
      user_data+='sudo yum install tmux -y'
    
    if user_data:
      user_data+='\necho userdata_ok >> /tmp/is_initialized\n'

    #    print("Using user_data", user_data)

    # TODO: also make sure instance type is the same
    if instances:
      if not skip_existing_job_validation:
        assert len(instances) == num_tasks, ("Found job with same name %s(%s), but number of tasks %d doesn't match requested %d, kill job manually." % (job_name, instances[0].state, len(instances), num_tasks))

      print("Found existing job "+job_name)
      starting_instances = False
      for i in instances:
        if i.state['Name'] == 'stopped':
          i.start()
          starting_instances = True

      # TODO: replace with proper wait loop
      if starting_instances:
        while True:
          print("Waiting forever for instances to start")
          time.sleep(10)
        
      print(instances)
    else:
      print("Launching new job %s into VPC %s" %(job_name, u.get_resource_name()))

      assert not (ami and ami_name), "Must have only one of ami and ami_name, got "+ami+", "+ami_name
      assert ami or ami_name, "Must specify at least one of ami and ami_name"
      if ami_name:
        ami = u.lookup_ami_id(ami_name).id
      security_group = u.get_security_group_dict()[u.get_resource_name()]
      
      keypair = u.get_keypair_dict()[u.get_keypair_name()]
      vpc = u.get_vpc_dict()[u.get_resource_name()]
      subnet_dict = u.get_subnet_dict(vpc)
      region = u.get_region()
      assert availability_zone in subnet_dict, "Availability zone %s is not in subnet dict for current AWS default region %s, available subnets are %s. (hint, set AWS_DEFAULT_REGION=%s)"%(availability_zone, region, ', '.join(subnet_dict.keys()), availability_zone[:-1])
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
      # TODO: get rid of zone? Zone seems to be required for constructor
      # that allows to enable AssociatePublicIpAddress field
      args['NetworkInterfaces'] = [{'SubnetId': subnet.id,
                                    'DeviceIndex': 0,
                                    'AssociatePublicIpAddress': True,
                                    'Groups': [security_group.id]}]
      

      placement_arg = {'AvailabilityZone': availability_zone}
      if placement_group: placement_arg['GroupName'] = placement_group
      args['Placement'] = placement_arg
      
      if monitoring: args['Monitoring'] = {'Enabled': True}
      args['UserData'] = user_data

      if use_spot: instances = u.create_spot_instances(args)
      else:
        try:
          instances = ec2.create_instances(**args)
        except Exception as e:
          print(f"Instance creation failed with ({e})")
          print("Account number: ", u.get_account_number())
          print("Region: ", u.get_region())
          sys.exit()
          
      assert instances
      assert len(instances) == num_tasks

      # TODO: make instances match their launch indices. This way
      # tasks can figure out which # they are
      for (task_num, instance) in enumerate(instances):
        while True:
          try:
            # sometimes get "An error occurred (InvalidInstanceID.NotFound)"
            # task_name = u.format_task_name(instance.ami_launch_index, role_name,
            #                                self.name)
            task_name = u.format_task_name(task_num, job_name)
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


  def set_global_logdir_prefix(self, logdir_prefix):
    backend.set_global_logdir_prefix(logdir_prefix)
    
  def setup_logdir(self):
    """Create logdir (using first task of first job).

    This is necessary to be called, and must run after first job/task is ready.
    """
    print("creating logdir")
    assert self.jobs
    head_job = self.jobs[0]
    assert head_job.tasks
    head_task = head_job.tasks[0]
    assert head_task.initialized, "Head task not initialized, must wait_until_ready"

    # get list of all logdirs
    find_command = f'find {backend.LOGDIR_PREFIX} -type d -maxdepth 1'
    # TODO: get rid of find warning
    #    find: warning: you have specified the -maxdepth option after a non-option argument -type, but options are not positional (-maxdepth affects tests specified befo
    # re it as well as those specified after it).  Please specify options before other
    # arguments.

    logdir_ls = head_task.run_and_capture_output(find_command)
    new_logdir = f"{backend.LOGDIR_PREFIX}/{self.name}"
    # TODO: change logic to count backwards from 99 instead (with error
    # checking). Otherwise run clean-up will cause insertion of new runs into
    # unobvious position
    
    counter = 0
    while new_logdir in logdir_ls:
      counter+=1
      lll = '%s.%02d'%(f"{backend.LOGDIR_PREFIX}/{self.name}", counter)
      self.log(f'Warning, logdir {new_logdir} exists, deduping to {lll}')
      new_logdir = lll
    self.logdir_ = new_logdir
    head_task.run(f'sudo mkdir -p {self.logdir}')
    head_task.run(f'sudo chown `whoami` {self.logdir}')
    

# TODO: refactor common fields like "linux_type", "user_data" to be
# stored in job instead of task
class Job(backend.Job):
  def __init__(self, run, name, instances, install_script=None,
               linux_type=None, user_data='', skip_efs_mount=False):
    self._run = run
    self.name = name


    self._run_command_available = False, "Have you done wait_until_ready?"
    
    # initialize list of tasks, in order of AMI launch index
    self.tasks = [None]*len(instances)
    for instance in instances:
      task_id, current_job_name = u.get_parsed_job_name(instance) # use job name in case ami's were not launched at the same time
      task_id = task_id or instance.ami_launch_index
      task = Task(instance, self, task_id, install_script=install_script,
                  linux_type=linux_type, user_data=user_data,
                  skip_efs_mount=skip_efs_mount)
      self.tasks[task_id] = task


  def _initialize(self):
    for task in self.tasks:
      task._initialize()


class Task(backend.Task):
  # TODO: replace linux_type with username
  def __init__(self, instance, job, task_id, install_script=None,
               user_data='', linux_type=None, skip_efs_mount=False):
    self.initialize_called = False
    self.instance = instance
    self.job = job
    self.id = task_id

    if user_data:
      assert user_data.startswith('#!/bin/bash')
    self.install_script = install_script
    self.user_data = user_data
    self.linux_type = linux_type
    self._run_counter = 0
    self.cached_ip = None
    self.cached_public_ip = None
    self.skip_efs_mount = skip_efs_mount

    self.name = u.format_task_name(task_id, job.name)
    # TODO, make below actually mean stuff (also, run_command_available)
    self.initialized = False 

    # scratch is client-local space for temporary files
    # TODO: this job.name already contains name of run, the directory
    # below uses run name twice
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
    self.keypair_fn = u.get_keypair_fn()

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
    # hack to get around Amazon linux not having tmux
    if self.linux_type == 'amazon':
      self._run_ssh('sudo yum install tmux -y')
    self._run_ssh('tmux kill-session -t '+self._tmux_session_name)
    self._run_ssh('tmux set-option -g history-limit 50000 \; set-option -g mouse on \; new-session -s %s -n 0 -d'%(self._tmux_session_name,))
    self._run_command_available = True
    
  def _mount_efs(self):
    self.log("Mounting EFS")
    region = u.get_region()
    efs_id = u.get_efs_dict()[u.get_resource_name()]
    dns = "{efs_id}.efs.{region}.amazonaws.com".format(**locals())
    self.run('sudo mkdir -p /efs')
    
    # ignore error on remount (efs already mounted)
    self.run("sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 %s:/ /efs"%(dns,), ignore_errors=True) 

    # make sure chmod is successful, hack to fix occasional permission errors
    self.run('sudo chmod 777 /efs')
    while 'drwxrwxrwx' not in self.run_and_capture_output('ls -ld /efs'):
      print(f"chmod 777 /efs didn't take, retrying in {TIMEOUT_SEC}")
      time.sleep(TIMEOUT_SEC)
      self.run('sudo chmod 777 /efs')

  def _initialize(self):
    """Tries to initialize the task."""

    self.log("Running initialize")
    self.initialize_called = True
    public_ip = self.public_ip # todo: add retry logic to public_ip property
    assert public_ip, f"Trying to initialize, but task {self.name} doesn't have public_ip"

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
    self.initialized = True
    self.log("Initialize complete")
    self.log(self.connect_instructions)



  # TODO: add "wait_for_file", that will help user-defined synchronization
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
    

  def upload(self, local_fn, remote_fn=None, dont_overwrite=False):
    """Uploads file to remote instance. If location not specified, dumps it
    in default directory."""
    # TODO: self.ssh_client is sometimes None
    self.log('uploading '+local_fn)
    sftp = self.ssh_client.open_sftp()
    
    if remote_fn is None:
      remote_fn = os.path.basename(local_fn)
    if dont_overwrite and self.file_exists(remote_fn):
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
    if 'command not found' in stdout_str or 'command not found' in stderr_str:
      self.log(f"command ({cmd}) failed with ({stdout_str}), ({stderr_str})")
      assert False, "run_ssh command failed"
    return stdout_str, stderr_str

  def run_and_capture_output(self, cmd, sync=True, ignore_errors=False):
    assert '|' not in cmd, "don't support piping (since we append piping here)"
    
    ts = str(u.now_micros())
    cmd_stdout_fn = self.remote_scratch+'/'+str(self._run_counter)+'.'+ts+'.out'
    cmd = f'{cmd} | tee {cmd_stdout_fn}'
    self.run(cmd, sync, ignore_errors)
    return self.file_read(cmd_stdout_fn)

  # TODO: make run_tmux a proper first-class citizen
  def _run_raw(self, cmd):
    print(f"_run_raw: {cmd}") 
    return self._run_ssh(cmd)

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

  # deprecate
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
