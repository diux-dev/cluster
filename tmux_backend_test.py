import os
import sys
import time


module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import tmux_backend

def main():
  backend = tmux_backend
  run = backend.make_run('test')
  job = run.make_job('test')
  result = job.run_and_capture_output('echo hi')
  print("Captured output was")
  print(result)

if __name__ == "__main__":
  main()
