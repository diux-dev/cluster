#!/usr/bin/env python
# Script to initialize a set of high-performance volumes with ImageNet data
# 
# replicate_imagenet.py --replicas 8 --zone=us-east-1c
# replicate_imagenet.py --replicas 8 --zone=us-east-1c --volume-offset=8
#
# or
#
# replicate_imagenet.py --replicas 16 --zone=us-east-1c
# Creates volumes: imagenet_00, imagenet_01, imagenet_02, ..., imagenet_15
#
# ImageNet data should follow structure as in
# https://github.com/diux-dev/cluster/tree/master/pytorch#data-preparation
# (paths replace ~/data with /)
#
# steps to create snapshot:
# create blank volume (ec2.create_volume())
# attach it to an existing instance with ImageNet under data, then
# sudo mkfs -t ext4 /dev/xvdf
# mkdir data
# sudo mount /dev/xvdf data
# sudo chown data `whoami`
# cp -R data0 data
# snapshot = ec2.create_snapshot(Description=f'{u.get_name(vol)} snapshot',
# VolumeId=vol.id,)

import argparse
parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--zone', type=str, default='')
parser.add_argument('--replicas', type=int, default=8)
parser.add_argument('--snapshot', type=str, default='') # snap-0d37b903e01bb794a
parser.add_argument('--snapshot-desc', type=str, default='imagenet_blank',
                    help='look for snapshot containing given string')
parser.add_argument('--volume-offset', type=int, default=0, help='start numbering with this value')
parser.add_argument('--iops', type=int, default=10000, help="iops requirement")
parser.add_argument('--size_gb', type=int, default=0, help="size in GBs")

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

def main():
  ec2 = u.create_ec2_resource()
  assert not args.snapshot, "Switched to snapshot_desc"
  if not args.zone:
    assert 'zone' in os.environ, 'must specify --zone or $zone'
    args.zone = os.environ['zone']
  
  snapshots = []
  for snap in ec2.snapshots.filter(OwnerIds=['self']):
    if args.snapshot_desc in snap.description:
      snapshots.append(snap)

  assert len(snapshots)>0, f"no snapshot matching {args.snapshot_desc}"
  assert len(snapshots)<2, f"multiple snapshots matching {args.snapshot_desc}"
  snap = snapshots[0]
  if not args.size_gb:
    args.size_gb = snap.volume_size
    
  print(f"Making {args.replicas} {args.size_gb} GB replicas in {args.zone}")
  
  for i in range(args.volume_offset, args.replicas+args.volume_offset):
    vol_name = 'imagenet_%02d'%(i)

    vol = ec2.create_volume(Size=args.size_gb, VolumeType='io1',
                      TagSpecifications=create_tags(vol_name),
                      AvailabilityZone=args.zone,
                      SnapshotId=snap.id,
                            Iops=args.iops)
    print(f"Creating {vol_name} {vol.id}")

if __name__=='__main__':
  main()
