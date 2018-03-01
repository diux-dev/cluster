import numpy as np
import time
import torch

start_time = time.time()
arr = np.zeros((1000*250*1000,))
arr.flags['WRITEABLE'] = False
start_time = time.time()
arr2 = torch.from_numpy(arr)
print("%.1f ms "%((time.time()-start_time)*1000,))
