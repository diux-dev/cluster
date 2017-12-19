#!/usr/bin/env python
# Tools for manipulating VPCs
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
    assert len(sys.argv) == 3
    
    assert 'AWS_DEFAULT_REGION' in os.environ
    client = c.create_ec2_client()
    ec2 = c.create_ec2_resource()
    response = client.describe_vpcs()
    for vpc_response in response['Vpcs']:
      vpc_name = _get_name(vpc_response.get('Tags', []))
      vpc = ec2.Vpc(vpc_response['VpcId'])
      if vpc_name == sys.argv[2]:
        print("Deleting VPC name=%s, id=%s"%(vpc_name, vpc.id))
        
        for subnet in vpc.subnets.all():
          print("Deleting subnet %s" % (subnet.id))
          assert c.is_good_response(subnet.delete())

        for gateway in vpc.internet_gateways.all():
          print("Deleting gateway %s" % (gateway.id))
          assert c.is_good_response(gateway.detach_from_vpc(VpcId=vpc.id))
          assert c.is_good_response(gateway.delete())

        for security_group in vpc.security_groups.all():
          try:
            assert c.is_good_response(security_group.delete())
          except Exception as e:
            print("Failed with "+str(e))
            
        for route_table in vpc.route_tables.all():
          print("Deleting route table %s" % (route_table.id))
          try:
            assert c.is_good_response(route_table.delete())
          except Exception as e:
            print("Failed with "+str(e))
          
        if c.is_good_response(client.delete_vpc(VpcId=vpc.id)):
          print("Succeeded deleting VPC ", vpc.id)

if __name__=='__main__':
  main()
