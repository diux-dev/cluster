#!/usr/bin/env python

# Launch Ray distributed benchmark from the following URL
BENCHMARK_URL="https://gist.githubusercontent.com/robertnishihara/4e246d6942cd692a0838414ff01975c1/raw/486a6a3f8c5cfc3d198e73cea50507a829e0cc9d/async_sgd_benchmark.py"

# This are setup commands to run on each AWS instance
INSTALL_SCRIPT="""
sudo apt update

# install ray and dependencies
sudo apt install -y pssh

# make Python3 the default
sudo apt install -y wget
sudo apt install -y python3
sudo apt install -y python3-pip
sudo ln -s /usr/bin/pip3 /usr/bin/pip
sudo mv /usr/bin/python /usr/bin/python.old
sudo ln -s /usr/bin/python3 /usr/bin/python

pip install ray
pip install numpy
pip install jupyter
"""

DEFAULT_PORT = 6379  # default redis port

def main():
  import aws
  import os
  
  # job launches are asynchronous, can spin up multiple jobs in parallel
  run_name = 'beefy'
  job = aws.simple_job(run_name, num_tasks=2,
                       instance_type='c5.18xlarge',
                       install_script=INSTALL_SCRIPT,
                       placement_group=run_name)

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
  print(slave_task.connect_instructions)
  
if __name__=='__main__':
  main()
