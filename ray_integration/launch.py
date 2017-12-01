#!/usr/bin/env python

# Launch Ray distributed benchmark from the following URL
BENCHMARK_URL="https://gist.githubusercontent.com/robertnishihara/4e246d6942cd692a0838414ff01975c1/raw/486a6a3f8c5cfc3d198e73cea50507a829e0cc9d/async_sgd_benchmark.py"

# This are setup commands to run on each AWS instance
INSTALL_SCRIPT="""
sudo apt update
sudo apt install -y wget
sudo apt install -y python3
sudo apt install -y python3-pip

# make Python3 work by default
sudo ln -s /usr/bin/pip3 /usr/bin/pip
sudo mv /usr/bin/python /usr/bin/python.old
sudo ln -s /usr/bin/python3 /usr/bin/python

# install ray and dependencies
sudo apt install -y pssh
pip install ray
pip install numpy
pip install jupyter
"""

DEFAULT_PORT = 6379  # default redis port

def main():
  import aws
  import os
  
  # job launches are asynchronous, can spin up multiple jobs in parallel
  job = aws.simple_job('ray', num_tasks=2, install_script=INSTALL_SCRIPT)

  # block until things launch to run commands
  job.wait_until_ready()

  head_task = job.tasks[0]
  head_task.run("ray start --head --redis-port=%d"%(DEFAULT_PORT,))

  slave_task = job.tasks[1]
  slave_task.run("ray start --redis-address %s:%d"%(head_task.ip,
                                                      DEFAULT_PORT))
  script_name = os.path.basename(BENCHMARK_URL)
  slave_task.run("rm -f "+script_name)
  slave_task.run("wget "+BENCHMARK_URL)
  slave_task.run("python "+script_name)

  print ("To see results:")
  print("ssh -i %s -o StrictHostKeyChecking=no ubuntu@%s"%(os.environ['SSH_KEY_PATH'], slave_task.public_ip))
  print("tmux a -t tmux")
  
if __name__=='__main__':
  main()
