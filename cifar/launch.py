import argparse
import json
import os
import portpicker
import sys
import time


module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import tmux
import aws
import util as u


parser = argparse.ArgumentParser(description='Launch CIFAR training')
parser.add_argument('--placement', type=int, default=0,
                     help=("whether to launch workers inside a placement "
                           "group"))
parser.add_argument('--gpu-instance-type', type=str, default='g3.4xlarge',
                    help='default instance type')
parser.add_argument('--cpu-instance-type', type=str, default='c5.xlarge',
                    help='default instance type')
parser.add_argument("--num-workers", default=2, type=int,
                    help="The number of gradient workers to use.")
parser.add_argument("--num-ps", default=1, type=int,
                    help="The number of parameter servers workers to use.")
parser.add_argument('--name', type=str, default='cifar',
                     help="name of the current run")
parser.add_argument('--zone', type=str, default='us-east-1c',
                    help='which availability zone to use')
parser.add_argument('--cluster', type=str, default='tmux',
                    help='tmux or aws')

args = parser.parse_args()
assert args.num_workers>0  # need non-empty for both worker and master jobs

# Deep Learning AMI
ami_dict = {
    "us-west-2": "ami-3b6bce43",
    "us-east-1": "ami-9ba7c4e1",
}
LINUX_TYPE = "ubuntu"  # linux type determines username to use to login
AMI_USERNAME = 'ubuntu'  # ami-specific username needed to login

AWS_INSTALL_SCRIPT="""
tmux set-option -g history-limit 250000

source activate mxnet_p36
export wheel=https://pypi.python.org/packages/86/f9/7b773ba67997b5c001ca57c4681d618f62d3743b7ee4b32ce310c1100cd7/tf_nightly_gpu-1.5.0.dev20171221-cp36-cp36m-manylinux1_x86_64.whl#md5=d126089a6fbbd81b104d303baeb649ff
pip install $wheel --upgrade-strategy=only-if-needed
"""


def launch_tmux():
  master_job = tmux.server_job('master', 1)
  worker_job = tmux.server_job('worker', args.num_workers-1)
  ps_job = tmux.server_job('ps', args.num_ps)
  tb_job = tmux.server_job('tb', 1)
  all_jobs = [master_job, worker_job, ps_job, tb_job]
  all_tasks = [task for job in all_jobs for task in job.tasks]

  logdir = tmux.setup_logdir(args.name)
    
  for task in all_tasks:
    task.upload('cifar10.py')
    task.upload('cifar10_main.py')
    task.upload('cifar10_model.py')
    task.upload('cifar10_utils.py')
    task.upload('model_base.py')
    task.upload('eval.tfrecords')
    task.upload('train.tfrecords')
    task.upload('validation.tfrecords')

  # Orchestration: every worker needs to know:
  # 1. their own role (task_spec), ie {type: worker, index: 0}
  # 2. role->ip mapping of all machines (cluster_spec), ie
  #    {"worker": ["localhost:24724"], "ps": ["localhost:15960"]}}
  # 
  # There are 3 types of workers, "master" ("chief worker"), "worker", ps
  def tf_env_setup(task, cluster_spec, task_spec):
    """Helper method to initialize clusterspec for a task."""
    json_string = json.dumps(cluster_spec)
    TF_CONFIG = json.dumps(
      {'cluster': cluster_spec, 'task': task_spec, 'model_dir': logdir,
      'environment': 'cloud'})
    task.write_to_file("export TF_CONFIG='%s'\n"%(TF_CONFIG,), 'TF_CONFIG.sh')
    task.run('source TF_CONFIG.sh')
  master_hosts = ["%s:%d"%(task.ip, task.port) for task in master_job.tasks]
  worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
  ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
  cluster_spec = {
    'master': master_hosts,
    'worker': worker_hosts,
    'ps': ps_hosts,
  }
  if not worker_hosts:  # special case of 1 master, 0 workers
    del cluster_spec['worker']
  
  # Launch tensorflow tasks.
  tf_cmd = "python cifar10_main.py --data-dir=. --job-dir={logdir} --log-device-placement".format(logdir=logdir)
  
  task_type = 'master' 
  for task in master_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run_async(tf_cmd+' --label='+task.job.name+':'+str(task.id))

  task_type = 'worker' 
  for task in worker_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run_async(tf_cmd+' --label='+task.job.name+':'+str(task.id))

  # launch parameter server tasks
  task_type = 'ps'
  for task in ps_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run_async(tf_cmd+' --label='+task.job.name+':'+str(task.id))

  # Launch tensorboard visualizer.
  tb_task = tb_job.tasks[0]
  tb_cmd = "tensorboard --logdir={logdir} --port={port}".format(
    logdir=logdir, port=tb_task.port)
  tb_task.run_async(tb_cmd)

  print("See tensorboard at http://%s:%s"%(tb_task.ip, tb_task.port))

def launch_aws():
  region = os.environ.get("AWS_DEFAULT_REGION")
  ami = ami_dict[region]

  master_job = aws.server_job(args.name+'-master', num_tasks=1,
                              instance_type=args.gpu_instance_type,
                              availability_zone=args.zone,
                              ami=ami)
  worker_job = aws.server_job(args.name+'-worker', num_tasks=args.num_workers-1,
                              instance_type=args.gpu_instance_type,
                              availability_zone=args.zone,
                              ami=ami)
  ps_job = aws.server_job(args.name+'-ps', num_tasks=args.num_ps,
                          instance_type=args.cpu_instance_type,
                          availability_zone=args.zone,
                          ami=ami)
  tb_job = aws.server_job(args.name+'-tb', num_tasks=1, instance_type=
                          args.cpu_instance_type,
                          availability_zone=args.zone,
                          ami=ami)
  all_jobs = [master_job, worker_job, ps_job, tb_job]
  all_tasks = [task for job in all_jobs for task in job.tasks]

  for job in all_jobs:
    job.wait_until_ready()

  logdir = aws.setup_logdir(args.name)

  efs_id = u.get_efs_dict()[u.RESOURCE_NAME]
  dns = "{efs_id}.efs.{region}.amazonaws.com".format(**locals())

  for task in all_tasks:
    task.run('source activate mxnet_p36')
    task.upload('cifar10.py')
    task.upload('cifar10_main.py')
    task.upload('cifar10_model.py')
    task.upload('cifar10_utils.py')
    task.upload('model_base.py')

    # mount EFS
    task.run('sudo mkdir -p /efs')
    task.run('sudo chmod 777 /efs')
    task.run("sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 %s:/ /efs"%(dns,), ignore_errors=True) # error on remount

    # don't need to upload data to tb/ps jobs
    if task.job.name.endswith('-tb') or task.job.name.endswith('-ps'):
      continue
    task.upload('eval.tfrecords')
    task.upload('train.tfrecords')
    task.upload('validation.tfrecords')
    

  # Orchestration: every worker needs to know:
  # 1. their own role (task_spec), ie {type: worker, index: 0}
  # 2. role->ip mapping of all machines (cluster_spec), ie
  #    {"worker": ["localhost:24724"], "ps": ["localhost:15960"]}}
  # 
  # There are 3 types of workers, "master" ("chief worker"), "worker", ps
  def tf_env_setup(task, cluster_spec, task_spec):
    """Helper method to initialize clusterspec for a task."""
    json_string = json.dumps(cluster_spec)
    TF_CONFIG = json.dumps(
      {'cluster': cluster_spec, 'task': task_spec, 'model_dir': logdir,
      'environment': 'cloud'})
    task.file_write('TF_CONFIG.sh', "export TF_CONFIG='%s'\n"%(TF_CONFIG,))
    task.run('source TF_CONFIG.sh')
  master_hosts = ["%s:%d"%(task.ip, task.port) for task in master_job.tasks]
  worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
  ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
  cluster_spec = {
    'master': master_hosts,
    'worker': worker_hosts,
    'ps': ps_hosts,
  }
  if not worker_hosts:  # special case of 1 master, 0 workers
    del cluster_spec['worker']
  
  # Launch tensorflow tasks.
  tf_cmd = "python cifar10_main.py --data-dir=. --job-dir={logdir} --log-device-placement".format(logdir=logdir)
  
  task_type = 'master' 
  for task in master_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run_async(tf_cmd+' --label='+task.job.name+':'+str(task.id))

  task_type = 'worker' 
  for task in worker_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run_async(tf_cmd+' --label='+task.job.name+':'+str(task.id))

  # launch parameter server tasks
  task_type = 'ps'
  for task in ps_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    tf_env_setup(task, cluster_spec, task_spec)
    task.run_async(tf_cmd+' --label='+task.job.name+':'+str(task.id))

  # Launch tensorboard visualizer.
  tb_task = tb_job.tasks[0]
  tb_cmd = "tensorboard --logdir={logdir} --port={port}".format(
    logdir=logdir, port=tb_task.port)
  tb_task.run_async(tb_cmd)

  print("See tensorboard at http://%s:%s"%(tb_task.ip, tb_task.port))


def main():
  if args.cluster == 'tmux':
    launch_tmux()
  elif args.cluster == 'aws':
    launch_aws()
  else:
    assert False, "Unknown cluster: "+args.cluster


if __name__=='__main__':
  main()
  
