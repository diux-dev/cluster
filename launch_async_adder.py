#!/usr/bin/env python


# Works either with remote or local instances.
# Local instances are tmux sessions. Each session has separate window
# corresponding to the task.

# Remote instances are on AWS. If given job name exists, it will assume
# it has correct number of instances (tasks), and reuse those instances.

# Job naming:
# test-ps (2 instances)
# test-worker (3 instances)
# test-tb (tensorboard process)

# Locally
# tmux session test-ps
# tmux session test-worker

# todo: utility to scp local file to amazon machine
import base64
import json
import os
import portpicker
import shlex
import subprocess
import sys
import tensorflow as tf
import threading
import time
from collections import defaultdict
import pickle

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_path+'/tf-tools/benchmark/runner')
import cluster_aws



################################################################################
# User-specific AWS config
AMI = 'ami-9ddb0fe5'
KEY_NAME = 'yaroslav'  # AWS key-name to use
KEY_PATH = os.environ['HOME']+'/d/yaroslav.pem' # location of .pem file on
                                                # local filesystem
SECURITY_GROUP = 'open' # security group for all instances
################################################################################


LOCAL_LOGDIR_PREFIX='/temp/logs'
EFS_LOGDIR_PREFIX='/efs/logs'

# TODO: replace tmux_name with just name
flags = tf.flags
flags.DEFINE_string('run', 'default',
                    'tag used to keep track of machines in this experiment')
flags.DEFINE_integer("num_workers", 2, "number of gradient workers")
flags.DEFINE_integer("num_ps", 2, "number of ps workers")
#flags.DEFINE_boolean("verbose", False, "whether to have verbose logging")
flags.DEFINE_string("tmux_name", "async_adder", "name to use for tmux session")
#flags.DEFINE_string('localdir_prefix', '/temp/stdout',
#                     'where to mirror worker logs locally')
#flags.DEFINE_string('logdir_prefix', '/efs/logs',
#                     'where to dump EFS logs')
flags.DEFINE_string('instance_type', 'c5.large',
                     'instance type to use')
flags.DEFINE_integer('remote_worker_port', 3333, 'port to use for '
                     'remote connections')
flags.DEFINE_integer('remote_tb_port', 6006, 'port to use for '
                     'tensorboard service')
FLAGS = flags.FLAGS


WORKER_CMD='python ./async_adder.py'  # todo: path robustness?
PS_CMD='python ./async_adder.py'


def ossystem(cmd):
  print(cmd)
  os.system(cmd)

# todo: factor out into tmux_lib


def setup_local_logdir(run):
  logdir = LOCAL_LOGDIR_PREFIX + '/' + run
  os.system('rm -Rf '+logdir)
  os.system('mkdir -p '+logdir)
  return logdir

initialized_windows = set()
def run_in_window(window, cmd_list):
  """Runs command in tmux window, initializing tmux session and window if
    necessary. cmd_list is list of args"""

  if isinstance(cmd_list, str):
    cmd_list = [cmd_list]

  assert isinstance(cmd_list, list)
  
  global initialized_windows
  def run(cmd):
    ossystem("tmux send-keys -t {} '{}' Enter".format(window, cmd))

  # if nothing initialized, restart session
  if not initialized_windows:
    ossystem('tmux kill-session -t ' + FLAGS.tmux_name)
    # -d starts new session in detached mode
    # since can't start windowless tmux, start with dummy window and rename
    ossystem('tmux new-session -s %s -n %s -d '% (FLAGS.tmux_name, "blargh"))
    
  if not window in initialized_windows:
    if not initialized_windows:
      ossystem('tmux rename-window -t blargh '+window)
    else:
      ossystem("tmux new-window -t {} -n {}".format(FLAGS.tmux_name, window))

    initialized_windows.add(window)
    
  for cmd in cmd_list:
    run(cmd)

# all instances->tasks
def launch_job_tmux(role, num_tasks):
  job_name = FLAGS.run + '-'+role
  DEFAULT_NAME = 'blargh'
  ossystem('tmux kill-session -t ' + job_name)

  # TODO: don't need default name
  ossystem('tmux new-session -s %s -n %s -d '% (job_name, DEFAULT_NAME))
  ossystem('tmux rename-window -t %s %s '%(DEFAULT_NAME, '0'))
  for task_id in range(1, num_tasks):
    ossystem("tmux new-window -t {} -n {}".format(job_name, task_id))

  job = LocalJob(job_name, num_tasks)
  # setup environment
  for task in job.tasks:
    task.run('source activate sep22')
    
  return job


class LocalJob:
  def __init__(self, name, num_tasks):
    self.name = name
    self.num_tasks = num_tasks
    self.tasks = []
    for task_id in range(num_tasks):
      self.tasks.append(LocalTask(self, task_id))


class LocalTask: # same as Instance
  """Local tasks interacts with tmux session where session name is derived
  from job name, and windows are task ids."""

  def __init__(self, job, task_id):
    self.job = job
    self.ip = '127.0.0.1' # hostname/ip address
    self.id = task_id
    self.port = portpicker.pick_unused_port()

  def run(self, cmd):
    window = self.job.name+":"+str(self.id)
    ossystem("tmux send-keys -t {} '{}' Enter".format(window, cmd))


  def tf_env_setup(self, full_cluster_spec, task_spec):
    # full cluster config
    # todo: not needed
    #    cluster_config = {'cluster': cluster_spec, 'task': task_spec}

    task_type = task_spec['type']
    task_id = task_spec['index']
    print("Task id is %r"%(task_id,))
    host = full_cluster_spec[task_type][task_id]

    # every worker needs its own location
    sparse_cluster_spec = defaultdict(dict)
    sparse_cluster_spec[task_type][task_id] = host
    
    # worker workers know about all ps workers
    if task_type == 'worker':
      sparse_cluster_spec['ps'] = full_cluster_spec['ps']
      
    # ps workers know about all worker workers
    if task_type == 'ps':
      pass
      sparse_cluster_spec['worker'] = full_cluster_spec['worker']
      #sparse_cluster_spec['worker'] = {0: full_cluster_spec['worker'][0]}

    sparse_cluster_config = {'cluster': sparse_cluster_spec,
                             'task': task_spec}
    print("Cluster config for %s %s is %s"%(task_type, task_id,
                                            sparse_cluster_spec))
    json_string = json.dumps(sparse_cluster_config)
    json_string_encoded = base64.b16encode(json_string.encode('ascii'))
    json_string_encoded = json_string_encoded.decode('ascii')
    export_command = "export TF_CONFIG_BASE16=%s"%(json_string_encoded,)
    self.run(export_command)

    # json has problem with sparse clusterspec (0 can't be key, only "0")
    # therefore also dump clusterspec as pickle object
    pickle_string = pickle.dumps(sparse_cluster_config)
    pickle_string_encoded = base64.b16encode(pickle_string)
    pickle_string_encoded = pickle_string_encoded.decode('ascii')
    export_command = "export TF_PICKLE_BASE16=%s"%(pickle_string_encoded,)
    self.run(export_command)
    
    logdir = LOCAL_LOGDIR_PREFIX + '/' + FLAGS.run
    self.run("export LOGDIR="+logdir)

def select_window(window):
  """select the window to be in the foreground"""
  ossystem('tmux select-window -t %s:%s'% (FLAGS.tmux_name, window))

def launch_local():
  # TODO: tee tmux outputs into local logs
  # TODO: travis compatible logs location
  
  # launch parameter servers
  # setup tf_config for each ps server, run
  # launch workers

  # the launcher takes care of setting up logdir
  logdir = '/temp/logs/'+FLAGS.name
  os.system('rm -Rf '+logdir)
  os.system('mkdir -p '+logdir)

  # todo: remove opencl stuff
  common_setup_cmds = ["export OPENCV_OPENCL_RUNTIME=",
                       "source activate sep22",
                       "export LOGDIR="+logdir]

  # allocate ports
  hostname = 'localhost'
  ps_hosts = []
  for task_id in range(FLAGS.num_workers):
    ps_hosts.append("%s:%d"%(hostname, portpicker.pick_unused_port()))
  worker_hosts = []
  for task_id in range(FLAGS.num_ps):
    worker_hosts.append("%s:%d"%(hostname, portpicker.pick_unused_port()))
  cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}


  def tf_config_setup():
    task_spec = {'type': task_type, 'index': task_id}
    cluster_config = {'cluster': cluster_spec, 'task': task_spec}
    json_string = json.dumps(cluster_config)
    json_string_encoded = base64.b16encode(json_string.encode('ascii'))
    json_string_encoded = json_string_encoded.decode('ascii')
    export_command = "export TF_CONFIG_BASE16=%s"%(json_string_encoded,)
    print('export command')
    print(repr(export_command))
    return export_command
    
  task_type = 'ps'
  for task_id in range(FLAGS.num_ps):
    cmds = list(common_setup_cmds)
    cmds.append(tf_config_setup())
    cmds.append(PS_CMD)
    window_name = '%s-%04d'%(task_type[0], task_id)
    run_in_window(window_name, cmds)

  task_type = 'worker'
  for task_id in range(FLAGS.num_workers):
    cmds = list(common_setup_cmds)
    cmds.append(tf_config_setup())
    cmds.append(WORKER_CMD)
    window_name = '%s-%04d'%(task_type[0], task_id)
    run_in_window(window_name, cmds)
  
  # launch tensorboard process with logdir
  port = portpicker.pick_unused_port()
  run_in_window('tb', 'echo "Running on port %d"'%(port,))
  run_in_window('tb', 'tensorboard --port=%d --logdir=%s'%(port, logdir))

  select_window('w-0000')
  
    
def launch_job_aws(name, replicas):

  # todo: rename instance_tag to name
  instances = cluster_aws.CreateAwsInstances(num_instances=num_instances,
                                             image_id=AMI,
                                             key_name=KEY_NAME,
                                             ssh_key=KEY_PATH,
                                             security_group=SECURITY_GROUP,
                                             instance_tag=name,
                                             placement_group='',
                                             instance_type=INSTANCE_TYPE)
  

class Instance:
  # todo: move inside instance
  def tf_env_setup(self, cluster_spec, task_spec):
    cluster_config = {'cluster': cluster_spec, 'task': task_spec}
    json_string = json.dumps(cluster_config)
    json_string_encoded = base64.b16encode(json_string.encode('ascii'))
    json_string_encoded = json_string_encoded.decode('ascii')
    export_command = "export TF_CONFIG_BASE16=%s"%(json_string_encoded,)
    self.run(export_command)

# TODO: rename .ip to get_ip()

def launch_local2():
  ps_job = launch_job_tmux('ps', FLAGS.num_ps)
  worker_job = launch_job_tmux('worker', FLAGS.num_workers)
  tb_job = launch_job_tmux('tb', 1)

  logdir = setup_local_logdir(FLAGS.run)

  # Orchestration: every worker needs to know:
  # 1. their own role (task_spec), ie {type: worker, index: 0}
  # 2. role->ip mapping of all machines (cluster_spec), ie
  #    {"worker": ["localhost:24724"], "ps": ["localhost:15960"]}}
   
  ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
  worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
  cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}

  # launch parameter server tasks
  task_type = 'ps'  
  for task in ps_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    task.tf_env_setup(cluster_spec, task_spec)
    task.run(PS_CMD)

  # launch worker tasks
  task_type = 'worker' # task type can also be "chief", overlapping with worker
  for task in worker_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    task.tf_env_setup(cluster_spec, task_spec)
    task.run(WORKER_CMD)

  # launch tensorboard visualizer
  task = tb_job.tasks[0]
  task.run('tensorboard --port=%d --logdir=%s'%(task.port, logdir))


def launch_remote():
  ps_instances = launch_instances_aws('ps', 1)
  worker_instances = launch_instances_aws('worker', 1)
  tb_instance = launch_instances_aws('tb', 1)[0]

  # Orchestration, every worker needs to know their own role (task_spec)
  # and role->ip mapping of all machines (cluster_spec)
  # This information is saved as base16 encoded dictionary in env var
  # and loaded in the task script
  PORT = 3000
  ps_hosts = ["%s:%d"%(i.ip, PORT) for i in ps_instances]
  worker_hosts = ["%s:%d"%(i.ip, PORT) for i in worker_instances]
  cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}

  def tf_config_setup():
    task_spec = {'type': task_type, 'index': task_id}
    cluster_config = {'cluster': cluster_spec, 'task': task_spec}
    json_string = json.dumps(cluster_config)
    json_string_encoded = base64.b16encode(json_string.encode('ascii'))
    json_string_encoded = json_string_encoded.decode('ascii')
    export_command = "export TF_CONFIG_BASE16=%s"%(json_string_encoded,)
    print('export command')
    print(repr(export_command))
    return export_command

  task_type = 'ps'
  for task_id, instance in enumerate(ps_instances):
    instance.run(tf_config_setup())
    instance.run(PS_CMD)

  task_type = 'worker'
  for task_id, instance in enumerate(worker_instances):
    instance.run(tf_config_setup())
    instance.run(WORKER_CMD)

  tb_instance.run('tensorboard --port=%d --logdir=%s'%(port, logdir))


def main():
  os.system('rm -Rf data') # todo: remove
  launch_local2()
  

if __name__=='__main__':
  main()
