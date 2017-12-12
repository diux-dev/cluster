#!/usr/bin/env python
# example of launching instance and running Python loop on it

import argparse

parser = argparse.ArgumentParser(description='ImageNet experiment')
parser.add_argument('--role', type=str, default='launcher',
                     help=('script role (launcher or worker)'))
args = parser.parse_args()

INSTALL_SCRIPT="""
sudo apt update
sudo apt install python3
sudo apt install -y python3-pip

# make Python 3 the default script
sudo ln -s /usr/bin/pip3 /usr/bin/pip
sudo mv /usr/bin/python /usr/bin/python2
sudo ln -s /usr/bin/python3 /usr/bin/python
"""


def main():
  if args.role == 'launcher':
    import aws
    job = aws.simple_job('simple', num_tasks=1, install_script=INSTALL_SCRIPT)
    task = job.tasks[0]
    job.wait_until_ready()
    task.upload(__file__)   # copies current script onto machine
    
    task.run("python %s --role=worker" % (__file__,)) # runs script and streams output locally to file in /temp
    print("To connect:")
    print(task.connect_instructions)
  elif args.role == 'worker':
    import sys, time
    print('hello world')
    print('Python version is '+str(sys.version))
    for i in range(10000):
      time.sleep(1)
      print("step %d"%(i,))
    
  else:
    print('Unknown role')


if __name__=='__main__':
  main()
