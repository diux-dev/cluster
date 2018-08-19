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
sudo mv /usr/bin/python /usr/bin/python2
sudo ln -s /usr/bin/python3 /usr/bin/python

# install ray and dependencies
sudo apt install -y pssh
pip install ray
pip install numpy
pip install jupyter
"""

DEFAULT_PORT = 6379  # default redis port

import ncluster

def main():
  # job launches are asynchronous, can spin up multiple jobs in parallel
  job = ncluster.make_job('ray', num_tasks=2, install_script=INSTALL_SCRIPT)
  job.join()

  head_task = job.tasks[0]
  head_task.run(f"ray start --head --redis-port={DEFAULT_PORT}")

  slave_task = job.tasks[1]
  slave_task.run("ray start --redis-address {head_task.ip}:{DEFAULT_PORT}")
  script_name = os.path.basename(BENCHMARK_URL)
  slave_task.run("rm -f "+script_name)
  slave_task.run("wget "+BENCHMARK_URL)
  slave_task.run("python "+script_name)

  print ("To see results:")
  print(slave_task.connect_instructions)
  
if __name__=='__main__':
  main()
