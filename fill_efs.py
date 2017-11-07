#!/usr/bin/env python

import numpy as np
import argparse
parser = argparse.ArgumentParser(description='script to fill EFS with data')

parser.add_argument('--gb', type=int, default=100, metavar='N',
                    help='how many GBs to dump')
parser.add_argument('--fn', type=str, default="dummy.bin", metavar='N',
                    help='filename')
args = parser.parse_args()

def main():
  chunk_size = 100e6
  current_size = 0
  out = open(args.fn, 'wb')
  while current_size < args.gb*1e9:
    print("Wrote %5.1f GBs"%(current_size/1e9))
    out.write(np.random.bytes(chunk_size))
    current_size+=chunk_size
  out.close()

if __name__=='__main__':
  main()
