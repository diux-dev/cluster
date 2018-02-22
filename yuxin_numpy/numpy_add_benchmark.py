# compare adding numbers using in-place addition, and with copies


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--inplace', type=int, default=1,
                    help='do additions in place')
args = parser.parse_args()

def main():
  dim = 25*1000*1000
  weights = np.zeros(dim, dtype=np.float32)
  times = []
  print(args.inplace)
  
  for i in range(100):
    t1 = time.time()
    grads = np.ones(dim, dtype=np.float32)
    if args.inplace:
      weights+=grads
    else:
      weights = np.copy(weights)
      weights += grads
    t2 = time.time()
    t1ms = 1000*t1
    t2ms = 1000*t2

    times.append(t2ms-t1ms)
    print("elapsed times: total %4.2f"%(t2ms-t1ms))

  times = np.array(times)
  print("Min %.2f, median %.2f mean: %.2f"%(np.min(times), np.median(times),
                                            np.mean(times)))

if __name__ == "__main__":
  main()
