#!/bin/env python
# Getting graph of event files
# To get event files
# scp -i /Users/yaroslav/nexus-us-east-1.pem ubuntu@52.86.36.67:/efs/runs/tfpp00/events.out.tfevents.1515205593.ip-192-168-46-3 .
# scp -i /Users/yaroslav/nexus-us-east-1.pem ubuntu@52.86.36.67:/efs/runs/tfpp00/events.out.tfevents.1515206122.ip-192-168-46-3 .
# mv events.out.tfevents.1515205593.ip-192-168-46-3 events1
# mv events.out.tfevents.1515206122.ip-192-168-46-3 events2


from collections import OrderedDict
import argparse
import os
import sys
import time


parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--tag', type=str, default='gpuutil',
                    help='launcher or worker')
parser.add_argument('--dir', type=str, default='/efs/runs/yuxin_numpy/0',
                    help='launcher or worker')
args = parser.parse_args()


from tensorflow.python.summary import summary_iterator
import glob

def main():

  for fname in glob.glob(args.dir+'/events*'):
    print('opening ', fname)
    events = summary_iterator.summary_iterator(fname)

    summaries = [e.summary for e in events if e.summary.value]
    values = []
    for summary in summaries:
      tag = summary.value[0].tag
      if args.tag.lower() not in tag.lower():
        continue
      print(summary.value[0].simple_value)

if __name__ == '__main__':
  main()
