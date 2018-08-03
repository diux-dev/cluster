#!/usr/bin/env python
# Script to delete a set of high-performance volumes with ImageNet data
# 
# unreplicate_imagenet.py --replicas 8 --zone=us-east-1c
# unreplicate_imagenet.py --replicas 8 --zone=us-east-1c --volume-offset=8
#
# or
#
# unreplicate_imagenet.py --replicas 16 --zone=us-east-1c
# Deletes volumes: imagenet_00, imagenet_01, imagenet_02, ..., imagenet_15

import argparse
parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--zone', type=str, default='')
parser.add_argument('--replicas', type=int, default=8)
parser.add_argument('--volume-offset', type=int, default=0, help='start numbering with this value')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--iops', type=int, default=10000, help="unused")
parser.add_argument('--size_gb', type=int, default=0, help="unused")
args = parser.parse_args()

import os
import sys

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')
import util as u

if __name__=='__main__':
  ec2 = u.create_ec2_resource()
  assert args.zone
  
  vols = {}
  for vol in ec2.volumes.all():
    if vol.availability_zone == args.zone:
      vols[u.get_name(vol)] = vol

  print(f"Deleting {args.replicas} replicas in {args.zone}")
  for i in range(args.volume_offset, args.replicas+args.volume_offset):
    vol_name = 'imagenet_%02d'%(i)
    print(f"Deleting {vol_name}")
    if not vol_name in vols:
      print("    Not found")
      continue
    vol = vols[vol_name]
    assert vol.volume_type == 'io1', "Safety check to prevent killing XView volumes (they are gp2)"
    if not args.dryrun:
      vol.delete()
      print(f"   {vol.id} deleted")
