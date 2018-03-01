import argparse
import numpy as np
import ray
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--size-mb", default=1000, type=int,
                    help="size of data in MBs")
args = parser.parse_args()

import ray
ray.init(num_workers=0)

@ray.remote
def f():
  return np.ones((args.size_mb * 250*1000,), np.float32)


arr = ray.get(f.remote())
assert str(arr.__class__) == "<class 'numpy.ndarray'>"
assert arr[0] == 1.0

# uncomment to get fail with "ValueError: output array is read-only"
# arr+=1

arr = torch.from_numpy(arr)
arr+=1
assert arr[0] == 2.0

