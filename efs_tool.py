#!/usr/bin/env python
# Tools for EFS related tasks
import boto3
import sys
import os

import util as u


def list_efss():
  for region in ['us-west-2', 'us-east-1']:
    print()
    print('='*80)
    print(region)
    print('='*80)
    efs_client = boto3.client('efs', region_name=region)
    response = efs_client.describe_file_systems()
    assert u.is_good_response(response)

    for efs_response in response['FileSystems']:
      #  {'CreationTime': datetime.datetime(2017, 12, 19, 10, 3, 44, tzinfo=tzlocal()),
      # 'CreationToken': '1513706624330134',
      # 'Encrypted': False,
      # 'FileSystemId': 'fs-0f95ab46',
      # 'LifeCycleState': 'available',
      # 'Name': 'nexus01',
      # 'NumberOfMountTargets': 0,
      # 'OwnerId': '316880547378',
      # 'PerformanceMode': 'generalPurpose',
      # 'SizeInBytes': {'Value': 6144}},
      efs_id = efs_response['FileSystemId']
      tags_response = efs_client.describe_tags(FileSystemId=efs_id)
      assert u.is_good_response(tags_response)
      key = u.get_name(tags_response.get('Tags', ''))
      print("%10s %10s" %(efs_id, key))
      
      # list mount points
      response = efs_client.describe_mount_targets(FileSystemId=efs_id)
      ec2 = boto3.resource('ec2', region_name=region)
      if not response['MountTargets']:
        print("<no mount targets>")
      else:
       for mount_response in response['MountTargets']:
         subnet = ec2.Subnet(mount_response['SubnetId'])
         zone = subnet.availability_zone
         state = mount_response['LifeCycleState']
         id = mount_response['MountTargetId']
         ip = mount_response['IpAddress']
         print('%-14s %-14s %-14s %-14s' %(zone, ip, id, state, ))
        
      print()
      

def _create_ec2_client():
  REGION = os.environ['AWS_DEFAULT_REGION']
  return boto3.client('ec2', region_name=REGION)


def main():
  if len(sys.argv) < 2:
    mode = 'list'
  else:
    mode = sys.argv[1]

  if mode == 'list':
    list_efss()
  elif mode == 'delete':
    name_or_id = sys.argv[2]
    efs_dict = u.get_efs_dict()
    if name_or_id in efs_dict:
      efs_id = efs_dict[name_or_id]
      sys.stdout.write('Deleting EFS %s (%s)... ' %(efs_id, name_or_id))
      sys.stdout.flush()
      u.delete_efs_id(efs_id)
    else:
      efs_id = name_or_id
      sys.stdout.write('Deleting EFS %s ()... ' %(efs_id,))
      sys.stdout.flush()
      u.delete_efs_id(name_or_id)
    print("success")
      
if __name__=='__main__':
  main()
