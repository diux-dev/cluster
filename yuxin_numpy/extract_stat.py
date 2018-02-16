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
parser.add_argument('--tag', type=str, default='train-error-top1',
                    help='launcher or worker')
parser.add_argument('--group', default='resnet_synthetic')
parser.add_argument('--name', default='fresh00')
parser.add_argument('--dir', type=str, default='/efs/runs/resnet_synthetic/fresh01',
                    help='launcher or worker')
args = parser.parse_args()


from tensorflow.python.summary import summary_iterator
import glob

def main():

  logdir = '/efs/runs/'+args.group + '/' + args.name

  for fname in glob.glob(logdir+'/events*'):
    print('opening ', fname)
    events = summary_iterator.summary_iterator(fname)

    events = [e for e in events if e.step]

    for event in events:
      step = event.step
      wall_time = event.wall_time
      vals = {val.tag: val.simple_value for val in event.summary.value}
      # step_time: value
      for tag in vals:
        if args.tag in tag:
          print(step, tag, vals[tag])

if __name__ == '__main__':
  main()
