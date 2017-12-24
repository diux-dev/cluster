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
parser.add_argument('--instance-type', type=str, default='g3.4xlarge',
                    help='default instance type')
parser.add_argument("--num-workers", default=1, type=int,
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


def launch_tmux():
  master_job = tmux.server_job('master', 1)
  worker_job = tmux.server_job('worker', args.num_workers-1)
  ps_job = tmux.server_job('ps', args.num_ps)
  tb_job = tmux.server_job('tb', 1)
  all_jobs = [master_job, worker_job, ps_job, tb_job]
  all_tasks = [task for job in all_jobs for task in job.tasks]

  logdir = tmux.setup_logdir(args.name)
    
  for task in all_tasks:
    task.run('source activate cifar')
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


def main():
  if args.cluster == 'tmux':
    launch_tmux()
  elif args.cluster == 'aws':
    launch_aws()
  else:
    assert False, "Unknown cluster: "+args.cluster


if __name__=='__main__':
  main()
  
