from pprint import pprint as pp
import argparse
import base64
import boto3
import json
import os
import os
import pickle
import portpicker
import shlex
import struct
import subprocess
import sys
import tensorflow as tf
import threading
import time
import yaml

import util as u

TASKDIR_PREFIX='/temp/tasklogs'
LOGDIR_PREFIX='/temp/runs' # use same name for remote/local?

# TODO: add locking to tmux commands

def setup_logdir(name):
  """Creates appropriate logdir for given run."""
  logdir = LOGDIR_PREFIX+'/'+name
  os.system('rm -Rf '+logdir)
  os.system('mkdir -p '+logdir)
  return logdir


def _ossystem(cmd):
  print(cmd)
  os.system(cmd)


def server_job(name, num_tasks):
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
  job = Job(name, tmux_windows)

  return job
  

class Job:
  def __init__(self, name, tmux_windows):
    self.name = name
    self.tasks = []
    for task_id, tmux_window in enumerate(tmux_windows):
      self.tasks.append(Task(tmux_window, self, task_id))

  def wait_until_ready(self):
    for task in self.tasks:
      task.wait_until_ready()


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
    
    self.taskdir = "{}/{}.{}/{}".format(
      TASKDIR_PREFIX, job.name, u.now_micros(), self.id)
    self.run('mkdir -p '+self.taskdir)
    self.run('cd '+self.taskdir)

  def run(self, cmd):
    tmux_run_sync(self.tmux_window, cmd)

  def run_async(self, cmd):
    _ossystem("tmux send-keys -t {} '{}' Enter".format(self.tmux_window, cmd))

  def upload(self, source_fn, target_fn='.'):
    print("%s/%s uploading %s to %s"%(self.job.name, self.id, source_fn,
                                      target_fn))
    source_fn_full = os.path.abspath(source_fn)
    self.run("cp %s %s" %(source_fn_full, target_fn))

  def write_to_file(self, contents, fn):
    local_fn = '/tmp/'+str(u.now_micros())
    with open(local_fn, 'w') as f:
      f.write(contents)
    self.upload(local_fn, os.path.basename(fn))

  def wait_until_ready(self):
    return
