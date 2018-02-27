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
@ray.remote
def f():
  return np.ones((args.size_mb * 250*1000,), np.float32)

ray.init(num_workers=0)

arr = ray.get(f.remote())
start_time = time.time()
torch.from_numpy(arr)
print("%.1f ms "%((time.time()-start_time)*1000,))
