# launch TensorBoard/monitoring server for runs
# Run
# ./launch.py
#
# Run on AWS:
# ./launch.py --backend=aws

import argparse
import os
import sys

# import cluster tools, one level up
module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import backend as backend_lib
import aws_backend
import tmux_backend
import util as u

parser = argparse.ArgumentParser()
parser.add_argument('--ami-name', type=str,
                    default="Deep Learning AMI (Ubuntu) Version 12.0",
                    help="name of AMI to use")
parser.add_argument('--name', type=str, default='monitoring', help='run name')
parser.add_argument('--instance-type', type=str, default='r5.4xlarge',
                     help='instance type to use for tensorboard job')
parser.add_argument('--zone', type=str, default='us-west-2c',
                    help='which availability zone to use')
parser.add_argument('--backend', type=str, default='aws',
                    help='cluster backend, tmux (local) or aws')
args = parser.parse_args()

def main():
  if args.backend == 'tmux':
    backend = tmux_backend
  elif args.backend == 'aws':
    backend = aws_backend
  else:
    assert False, "unknown backend"
    
  run = backend.make_run(args.name,
                         ami_name=args.ami_name,
                         availability_zone=args.zone)
  tb_job = run.make_job('tb', instance_type=args.instance_type)
  tb_job.wait_until_ready()
  tb_job.run('source activate tensorflow_p36')
  tb_job.run_async(f'tensorboard --logdir={backend_lib.LOGDIR_PREFIX} --port=6006')
  print(f'Tensorboard will be at http://{tb_job.public_ip}:6006')

  # run second Tensorboard in new tmux session for "selected runs"
  # to select runs,
  # on instance, do "ln -s /efs/runs/<run_name> /efs/runs.selected/<run_name>
  # (must use abspath for ln -s left hand side for linking to work)
  selected_logdir = backend_lib.LOGDIR_PREFIX+'.selected'
  tb_job._run_raw("tmux kill-session -t selected")
  tb_job._run_raw("tmux new-session -s selected -n 0 -d")
  tb_job._run_raw("tmux send-keys -t selected:0 'source activate tensorflow_p36' Enter")
  tb_job._run_raw(f"tmux send-keys -t selected:0 'tensorboard --logdir {selected_logdir} --port=6007' Enter")

  print(f'Tensorboard selected will be at http://{tb_job.public_ip}:6007')


if __name__=='__main__':
  main()
