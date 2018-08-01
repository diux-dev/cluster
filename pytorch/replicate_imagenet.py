#!/usr/bin/env python
# Script to initialize a set of high-performance volumes with ImageNet data
# 
# python replicate_imagenet.py --replicas 8 --zone=us-east-1c
# Creates volumes: imagenet_00, imagenet_01, imagenet_02, etc
#
# This relies on snapshot with ImageNet data following structure as in
# https://github.com/diux-dev/cluster/tree/master/pytorch#data-preparation
# (paths replace ~/data with /)
#
# steps to create snapshot:
# create blank volume (ec2.create_volume())
# attach it to an existing instance with ImageNet under data 
# sudo mkfs -t ext4 /dev/xvdf
# mkdir data
# sudo mount /dev/xvdf data
# sudo chown data `whoami`
# cp -R data0 data
# snapshot = ec2.create_snapshot(Description=f'{u.get_name(vol)} snapshot',
# VolumeId=vol.id,)

import argparse
parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--zone', type=str, default='us-east-1c')
parser.add_argument('--replicas', type=int, default=8)
parser.add_argument('--snapshot', type=str, default='snap-0d37b903e01bb794a')
args = parser.parse_args()

import os
import sys

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util as u

def create_tags(name):
  return [{
    'ResourceType': 'volume',
    'Tags': [{
        'Key': 'Name',
        'Value': name
    }]
}]

if __name__=='__main__':
  ec2 = u.create_ec2_resource()
  print(f"Making {args.replicas} replicas in {args.zone}")
  for i in range(args.replicas):
    vol = ec2.create_volume(Size=300, VolumeType='io1',
                      TagSpecifications=create_tags('imagenet_%02d'%(i)),
                      AvailabilityZone=args.zone,
                      SnapshotId=args.snapshot,
                      Iops=10000)
    print(f"Creating {vol.id}")
