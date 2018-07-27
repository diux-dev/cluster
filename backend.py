"""Interface for job launching backend."""
# Job launcher Python API: https://docs.google.com/document/d/1yTkb4IPJXOUaEWksQPCH7q0sjqHgBf3f70cWzfoFboc/edit
# AWS job launcher (concepts): https://docs.google.com/document/d/1IbVn8_ckfVO3Z9gIiE0b9K3UrBRRiO9HYZvXSkPXGuw/edit

import os
import glob
import threading

import util as u

# aws_backend.py
# tmux_backend.py

LOGDIR_PREFIX='/efs/runs'


"""
backend = aws_backend # alternatively, backend=tmux_backend to launch jobs locally in separate tmux sessions
run = backend.make_run("helloworld")  # sets up /efs/runs/helloworld
worker_job = run.make_job("worker", instance_type="g3.4xlarge", num_tasks=4, ami=ami, setup_script=setup_script)
ps_job = run.make_job("ps", instance_type="c5.xlarge", num_tasks=4, ami=ami, setup_script=setup_script)
setup_tf_config(worker_job, ps_job)
ps_job.run("python cifar10_main.py --num_gpus=0")  # runs command on each task
worker_job.run("python cifar10_main.py --num_gpus=4")

tb_job = run.make_job("tb", instance_type="m4.xlarge", num_tasks=1, public_port=6006)
tb_job.run("tensorboard --logdir=%s --port=%d" %(run.logdir, 6006))
# when job has one task, job.task[0].ip can be accessed as job.ip
print("See TensorBoard progress on %s:%d" %(tb_job.ip, 6006))
print("To interact with workers: %s" %(worker_job.connect_instructions))


To reconnect to existing job:

"""

# todo: rename to "start_run" instead of setup_run?
def make_run(name):
  """Sets up "run" with given name, such as "training run"."""
  raise NotImplementedError()

# def make_job(run_name, job_name, **kwargs):
#   """Initializes Job object. It will reuse existing cluster resources if the job with given parameters has already been launched."""
#   raise NotImplementedError()


class Run:
  """Run is a collection of jobs that share statistics. IE, training run will contain gradient worker job, parameter server job, and TensorBoard visualizer job. These jobs will use the same shared directory to store checkpoints and event files."""

  def __init__(self, name, install_script=None):
    """Creates a run. If install_script is specified, it's used as default
    install_script for all jobs (can be overridden by Job constructor)"""
    raise NotImplementedError()
  
  def make_job(self, name, num_tasks=1, install_script=None, **kwargs):
    """Creates job in the given run. If install_script is None, uses
    install_script associated with the Run."""
    raise NotImplementedError()
  

  def run(self, *args, **kwargs):
    """Runs command on every job in the run."""
    
    for job in self.jobs:
      job.run(*args, **kwargs)

      
  def upload(self, *args, **kwargs):
    
    """Runs command on every job in the run."""
    
    for job in self.jobs:
      job.upload(*args, **kwargs)

  def log(self, message, *args):
    """Log to client console."""
    ts = u.current_timestamp()
    if args:
      message = message % args

    print("%s %s: %s"%(ts, self.name, message))


class Job:
  def __init__(self):
    self.tasks = []

  def run_async(self, cmd, *args, **kwargs):
    self.run(cmd, sync=False, *args, **kwargs)
    
  def run(self, cmd, *args, **kwargs):
    """Runs command on every task in the job."""

    for task in self.tasks:
      task.run(cmd, *args, **kwargs)

  def run_async_join(self, cmd, *args, **kwargs):
    """Runs command on every task in the job async. Then waits for all to finish"""
    def t_run_cmd(t): t.run(cmd, *args, **kwargs)
    self.async_join(t_run_cmd)
  
  def upload(self, *args, **kwargs):
    """Runs command on every task in the job."""
    
    for task in self.tasks:
      task.upload(*args, **kwargs)

  def upload_async(self, *args, **kwargs):
    def t_upload(t): t.upload(*args, **kwargs)
    self.async_join(t_upload)

  def async_join(self, task_fn):
    t_threads = [threading.Thread(name=f't_{i}', target=task_fn, args=[t]) for i,t in enumerate(self.tasks)]
    for thread in t_threads: thread.start()
    for thread in t_threads: thread.join()
      
  # todo: rename to initialize
  def wait_until_ready(self):
    """Waits until all tasks in the job are available and initialized."""
    # import threading
    # t_threads = [threading.Thread(name=f't_{i}', target=lambda t: t.wait_until_ready(), args=[t]) for i,t in enumerate(self.tasks)]
    # for thread in t_threads: thread.start()
    # for thread in t_threads: thread.join()
    for task in self.tasks:
      task.wait_until_ready()
      # todo: initialization should start async in constructor instead of here
  
  # these methods redirect to the first task
  @property
  def ip(self):
    return self.tasks[0].ip

  @property
  def public_ip(self):
    return self.tasks[0].public_ip
  
  @property
  def port(self):
    return self.tasks[0].port

  @property
  def public_port(self):
    return self.tasks[0].public_port

  @property
  def connect_instructions(self):
    return self.tasks[0].connect_instructions

  @property
  def logdir(self):
    return self._run.logdir

    
class Task:
  def run(self, cmd, sync, ignore_errors):
    """Runs command on given task."""
    raise NotImplementedError()    

  def run_async(self, cmd, *args, **kwargs):
    self.run(cmd, sync=False, *args, **kwargs)
    
  def _upload_handler(self, line):
    """Handle following types of commands.

    Individual files, ie
    %upload file.txt

    Glob expressions, ie
    %upload *.py"""


    toks = line.split()
    assert len(toks) == 2
    assert toks[0] == '%upload'
    fname = toks[1]
    fname = fname.replace("~", os.environ["HOME"])

    for fn in glob.glob(fname):
      self.upload(fn)


  def upload(self, local_fn, remote_fn=None, skip_existing=False):
    """Uploads given file to the task. If remote_fn is not specified, dumps it
    into task current directory with the same name."""
    raise NotImplementedError()    


  def download(self, remote_fn, local_fn=None):
    """Downloads remote file to current directory."""
    raise NotImplementedError()

  @property
  def ip(self):
    raise NotImplementedError()

  @property
  def public_ip(self):
    """Helper method to provide a publicly facing ip for given task when
    tasks run on a different network than user (ie, AWS internal vs. user's
    laptop)"""
    raise NotImplementedError()

  @property
  def logdir(self):
    return self.job.logdir

    
  
  @property
  def port(self):
    """This is (the main) internal port that this task will use for
    communicating with other tasks. When using TensorFlow, this would be the 
    port on which TensorFlow server is listening."""
    return self._port

  @property
  def public_port(self):
    """This is a port that's used to access task from public internet.
    On AWS it tends to be fixed because it's set by underlying infrastructure
    (security group), defer implementation to backend."""
    raise NotImplementedError()


  def log(self, message, *args):
    """Log to client console."""
    ts = u.current_timestamp()
    if args:
      message = message % args

    print("%s %d.%s: %s"%(ts, self.id, self.job.name, message))

  def file_write(self, fn, contents):
    """Write string contents to file fn in task."""
    raise NotImplementedError()

  def file_read(self, fn):
    """Read contents of file and return it as string."""
    raise NotImplementedError()

  def file_exists(self, fn):
    """Return true if file exists in task current directory."""
    raise NotImplementedError()

  def stream_file(self, fn):
    """Streams task-local file to console (path relative to taskdir)."""
    raise NotImplementedError()
  

  def _ossystem(self, cmd):
    #    self.log(cmd)
    os.system(cmd)
