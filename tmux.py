from pprint import pprint as pp
import argparse
import base64
import json
import os
import os
import pickle
import portpicker
import shlex
import struct
import subprocess
import sys
import threading
import time
import yaml

import tensorflow as tf
import boto3

import util as u

# TODO: factor out/document abstract interface for job/task

# location of output files on target machine
TASKDIR_PREFIX='/tmp/tasklogs'
SCRATCH_PREFIX='/tmp'

LOGDIR_PREFIX='/efs_local/runs'

def setup_logdir(name):
  """Creates appropriate logdir for given run."""
  logdir = LOGDIR_PREFIX+'/'+name
  os.system('rm -Rf '+logdir)
  os.system('mkdir -p '+logdir)
  return logdir


def _ossystem(cmd):
  print(cmd)
  os.system(cmd)


# TODO: rename to "launch_job"?
def server_job(name, num_tasks, install_script=''):
  assert num_tasks>=0
  
  _ossystem('tmux kill-session -t ' + name)
  
  tmux_windows = []
  print("Creating %s with %s"%(name, num_tasks))
  if num_tasks>0:
    _ossystem('tmux new-session -s %s -n %d -d' % (name, 0))
    tmux_windows.append(name+":"+str(0))
  for task_id in range(1, num_tasks):
    _ossystem("tmux new-window -t {} -n {}".format(name, task_id))
    tmux_windows.append(name+":"+str(task_id))

  # todo: remove num_tasks
  job = Job(name, tmux_windows, install_script=install_script)
  return job
  

class Job:
  def __init__(self, name, tmux_windows, install_script=""):
    self.name = name
    self.tasks = []
    for task_id, tmux_window in enumerate(tmux_windows):
      self.tasks.append(Task(tmux_window, self, task_id,
                             install_script=install_script))

  def wait_until_ready(self):
    for task in self.tasks:
      task.wait_until_ready()


# need: task.file_exists
# need: task.file_read

# TODO: dedup "cmd_idx:aws.py" with "tmux_counter:tmux.py"

tmux_counter = 0
def tmux_run_sync(tmux_window, cmd, check_interval=0.2, max_wait_sec=600):
  """Uses tmux send-keys command, adds file locking to block until command
  finishes executing."""
  global tmux_counter
  if not os.path.exists('/tmp/tmux'):
    _ossystem('mkdir -p /tmp/tmux')
  ts = str(u.now_micros())
  cmd_fn_in  = '/tmp/tmux/'+str(tmux_counter)+'.'+ts+'.in'
  cmd_fn_out = '/tmp/tmux/'+str(tmux_counter)+'.'+ts+'.out'
  open(cmd_fn_in, 'w').write(cmd+'\n')
  modified_cmd = '%s && echo $? > %s'%(cmd, cmd_fn_out)
  start_time = time.time()
  
  _ossystem("tmux send-keys -t {} '{}' Enter".format(tmux_window, modified_cmd))

  while True:
    if time.time() - start_time > max_wait_sec:
      assert False, "Timeout %s exceeded for %s" %(max_wait_sec, cmd)
    if not os.path.exists(cmd_fn_out):
      time.sleep(check_interval)
      continue
    
    contents = open(cmd_fn_out).read()
    # if empty wait a bit to allow for race condition
    if len(contents) == 0:
      time.sleep(check_interval)
      contents = open(cmd_fn_out).read()

    contents = contents.strip()
    assert contents == '0', "Command %s returned status %s"%(cmd, contents)
    break
    
  
class Task:
  """Local tasks interacts with tmux session where session name is derived
  from job name, and window names are task ids."""

  def __init__(self, tmux_window, job, task_id):
    self.tmux_window = tmux_window
    self.job = job
    self.ip = '127.0.0.1'  # hostname/ip address
    self.id = task_id
    self.port = portpicker.pick_unused_port()
    print("Assigning %s:%s to port %s"%(self.job.name, self.id, self.port))
    self.connect_instructions = 'tmux a -t '+self.tmux_window

    self.last_stdout = '<unavailable>'  # compatiblity with aws.py:Task
    self.last_stderr = '<unavailable>'

    self.scratch = SCRATCH_PREFIX
    self.taskdir = "{}/{}.{}/{}".format(
      TASKDIR_PREFIX, job.name, u.now_micros(), self.id)
    self.run('mkdir -p '+self.taskdir)
    self.run('cd '+self.taskdir)

  def run_sync(self, cmd):
    tmux_run_sync(self.tmux_window, cmd)

  def run(self, cmd):
    _ossystem("tmux send-keys -t {} '{}' Enter".format(self.tmux_window, cmd))

  def upload(self, source_fn, target_fn='.'):
    print("%s/%s uploading %s to %s"%(self.job.name, self.id, source_fn,
                                      target_fn))
    source_fn_full = os.path.abspath(source_fn)
    self.run("cp %s %s" %(source_fn_full, target_fn))

  def file_write(self, contents, fn):
    local_fn = '/tmp/'+str(u.now_micros())
    with open(local_fn, 'w') as f:
      f.write(contents)
    self.upload(local_fn, os.path.basename(fn))


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

  def wait_until_ready(self):
    return
