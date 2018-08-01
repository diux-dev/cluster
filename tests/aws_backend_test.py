import os
import sys
import time


module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import aws_backend
import argparse

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--zone', type=str, default='us-west-2c')
parser.add_argument('--ami-name', type=str, default='')
parser.add_argument('--ami', type=str, default='ami-ba602bc2') # ubuntu default
parser.add_argument('--instance-type', type=str, default='r5.4xlarge')

args = parser.parse_args()


def main():
  backend = aws_backend
  run = backend.make_run('testrun')
  job = run.make_job('testjob', instance_type=args.instance_type,
                     ami=args.ami,
                     availability_zone=args.zone)
  job.wait_until_ready()
  result = job.run_and_capture_output('ls /tmp')
  print("Captured output was")
  print(result)

if __name__ == "__main__":
  main()
