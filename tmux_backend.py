# Local implementation of backend.py using separate tmux sessions for jobs

import datetime
import glob
import os
import subprocess
import sys
import shlex
import sys
import socket
import time

import portpicker

import backend
import util as u

TASKDIR_PREFIX='/tmp/tasklogs'

# TODO: use separate session for each task, for parity with AWS job launcher

# todo: tmux session names are backwards from AWS job names (runname-jobname)
# TODO: add kwargs so that tmux backend can be drop-in replacement
def make_run(name, install_script='', **kwargs):
  if kwargs:
    print("Warning, unused kwargs", kwargs)
  return Run(name, install_script)


class Run(backend.Run):
  def __init__(self, name, install_script=''):
    self.name = name
    self.install_script = install_script
    self.jobs = []
    self.logdir = f'{backend.LOGDIR_PREFIX}/{self.name}'

  # TODO: rename job_name to role_name
  def make_job(self, job_name, num_tasks=1, install_script='', **kwargs):
    assert num_tasks>=0

    if kwargs:
      print("Warning, unused kwargs", kwargs)

    # TODO, remove mandatory delete and make separate method for killing?
    tmux_name = self.name+'-'+job_name # tmux can't use . in name
    os.system('tmux kill-session -t ' + tmux_name)
    tmux_windows = []
    self.log("Creating %s with %s"%(tmux_name, num_tasks))
    if num_tasks>0:
      os.system('tmux new-session -s %s -n %d -d' % (tmux_name, 0))
      tmux_windows.append(tmux_name+":"+str(0))
    for task_id in range(1, num_tasks):
      os.system("tmux new-window -t {} -n {}".format(tmux_name, task_id))
      tmux_windows.append(tmux_name+":"+str(task_id))

    if not install_script:
      install_script = self.install_script
    job = Job(self, job_name, tmux_windows, install_script=install_script)
    self.jobs.append(job)
    return job

  def setup_logdir(self):
    if os.path.exists(self.logdir):
      datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      new_logdir = f'{backend.LOGDIR_PREFIX}/{self.name}.{datestr}'
      self.log(f'Warning, logdir {self.logdir} exists, deduping to {new_logdir}')
      self.logdir = new_logdir
    os.makedirs(self.logdir)
        
class Job(backend.Job):
  def __init__(self, run, name, tmux_windows, install_script=''):
    self._run = run
    self.name = name
    self.tasks = []
    for task_id, tmux_window in enumerate(tmux_windows):
      self.tasks.append(Task(tmux_window, self, task_id,
                             install_script=install_script))


class Task(backend.Task):
  """Local tasks interact with tmux session where session name is derived
  from job name, and window names are task ids."""

  def __init__(self, tmux_window, job, task_id, install_script):
    self.tmux_window = tmux_window  # TODO: rename tmux_window to window?
    self.job = job
    self.id = task_id
    self.cached_ip = None
    self._port = portpicker.pick_unused_port()
    print("Assigning %s:%s to port %s"%(self.job.name, self.id, self.port))
    self.connect_instructions = 'tmux a -t '+self.tmux_window

    self.taskdir = "{}/{}.{}.{}.{}".format(TASKDIR_PREFIX, job._run.name,
                                           job.name, self.id,
                                           0) # u.now_micros())
    self.log("Creating taskdir %s", self.taskdir)
    self.scratch = self.taskdir+'/scratch'
    self._ossystem('mkdir -p '+self.taskdir)
    
    self._ossystem('rm -Rf '+self.scratch)
    self._ossystem('mkdir -p '+self.scratch)
    self._run_counter = 0

    # At this point, ".run" command is available so can use that
    # install things
    self.run('cd '+self.taskdir)
    self.install_script = install_script
    for line in install_script.split('\n'):
      if line.startswith('%upload'):
        self._upload_handler(line)
      else:
        self.run(line)


  def _wait_for_file(self, fn, max_wait_sec=600, check_interval=0.02):
    start_time = time.time()
    while True:
      if time.time() - start_time > max_wait_sec:
        assert False, "Timeout %s exceeded for %s" %(max_wait_sec, cmd)
      if not self.file_exists(fn):
        # print("check for %s returned %s, sleeping for %s"%(fn,
        #                                                    self.file_exists(fn),
        #                                                    check_interval))
        time.sleep(check_interval)
        continue
      else:
        break


  def run(self, cmd, sync=True, ignore_errors=False):
    self._run_counter+=1
    self.log(cmd)
    cmd = cmd.strip()
    if not cmd:  # ignore empty command lines
      return

    if cmd.startswith('#'):  # ignore commented out lines
      return
    
    cmd_in_fn  = '%s/%d.in'%(self.scratch, self._run_counter)
    cmd_out_fn  = '%s/%d.out'%(self.scratch, self._run_counter)

    assert not os.path.exists(cmd_out_fn)
    
    open(cmd_in_fn, 'w').write(cmd+'\n')
    modified_cmd = '%s ; echo $? > %s'%(cmd, cmd_out_fn)
    start_time = time.time()
    #    time.sleep(0.01) # tmux gets confused when too many messages get sent at
    # once
    self.log("%s> %s"%(self.tmux_window, cmd))
    tmux_cmd = 'tmux send-keys -t {} {} Enter'.format(self.tmux_window, shlex.quote(modified_cmd))
    self._ossystem(tmux_cmd)
    if not sync:
      return
    
    self._wait_for_file(cmd_out_fn)
    contents = open(cmd_out_fn).read()
    
    # if empty wait a bit to allow for race condition
    if len(contents) == 0: time.sleep(0.1)
    contents = open(cmd_out_fn).read().strip()

    if contents != '0':
      if not ignore_errors:
        assert False, "Command %s returned status %s"%(cmd, contents)
      else:
        self.log("Warning: command %s returned status %s"%(cmd, contents))

 
  def upload(self, source_fn, target_fn='.'):
    #    self.log("uploading %s to %s"%(source_fn, target_fn))
    source_fn_full = os.path.abspath(source_fn)
    self.run("cp -R %s %s" %(source_fn_full, target_fn))
    #os.system("cp %s %s" %(source_fn_full, target_fn))


  def download(self, source_fn, target_fn='.'):
    raise NotImplementedError()
    # self.log("downloading %s to %s"%(source_fn, target_fn))
    # source_fn_full = os.path.abspath(source_fn)
    # os.system("cp %s %s" %(source_fn_full, target_fn))

  
  def file_exists(self, remote_fn):
    #    print("Checking "+remote_fn+" , "+str(os.path.exists(remote_fn)))
    return os.path.exists(remote_fn)


  def _make_temp_fn(self):
    """Returns temporary filename for this task."""
    return self.scratch+'/file_write.'+str(u.now_micros())

  def file_write(self, remote_fn, contents):
    tmp_fn = self._make_temp_fn()
    open(tmp_fn, 'w').write(contents)
    self.upload(tmp_fn, remote_fn)
  

  def file_read(self, remote_fn):
    tmp_fn = self._make_temp_fn()
    self.download(remote_fn, tmp_fn)
    return open(tmp_fn).read()

  def wait_until_ready(self):
    return
  
  def stream_file(self, fn):
    if not fn.startswith('/'):
      fn = self.taskdir+'/'+fn

    if not os.path.exists(fn):
      os.system('mkdir -p '+os.path.dirname(fn))
      os.system('touch '+fn)

    from time import strftime, gmtime, localtime

    def ts():
      return strftime("%a, %d %b %Y %H:%M:%S", localtime())

    start_time = time.time()

    p = subprocess.Popen(['tail', '-f', fn], stdout=subprocess.PIPE)
    
    for line in iter(p.stdout.readline, ''):
      elapsed = time.time()-start_time
      #      print(ts()+": %.2f read line"%(elapsed,))
      sys.stdout.write(line.decode('ascii', errors='ignore'))
      start_time = time.time()
      
      
  @property
  def ip(self):
    if not self.cached_ip:
      self.cached_ip = socket.gethostbyname(socket.gethostname())
    return self.cached_ip


  @property
  def public_ip(self):
    return self.ip

  @property
  def public_port(self):
    return self.port
