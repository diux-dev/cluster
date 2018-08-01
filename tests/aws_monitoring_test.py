import os
import sys
import time

# images = list(ec2.images.all())
# [im for im in images if im.id == 'ami-a9d09ed1'][0].name
# => 'amzn2-ami-hvm-2.0.20180622.1-x86_64-gp2'

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import aws_backend
import argparse

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--zone', type=str, default='us-east-1c')
parser.add_argument('--ami-name', type=str, default='amzn2-ami-hvm-2.0.20180622.1-x86_64-gp2')
parser.add_argument('--linux-type', type=str, default='amazon')
#parser.add_argument('--ami', type=str, default='') # ubuntu default
parser.add_argument('--instance-type', type=str, default='r5.large')

args = parser.parse_args()


def main():
  backend = aws_backend
  run = backend.make_run('test')
  job = run.make_job('monitoring', instance_type=args.instance_type,
                     ami_name=args.ami_name,
                     availability_zone=args.zone,
                     linux_type=args.linux_type)
  job.wait_until_ready()
  result = job.run_and_capture_output('ls /tmp')
  print("Captured output was")
  print(result)

if __name__ == "__main__":
  main()
