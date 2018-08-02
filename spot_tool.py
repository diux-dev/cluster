#!/usr/bin/env python
# Tools for spot instance related tasks


  
# By default, lists all volumes, along with their availability zones and attacment points

# TODO: clone tool to create clones of an EBS disk

import boto3
import sys
import os
import time

from operator import itemgetter

import util as u

def list_spot_requests():
  ec2 = u.create_ec2_resource()
  client = u.create_ec2_client()
  for request in client.describe_spot_instance_requests()['SpotInstanceRequests']:
    launch_spec = request['LaunchSpecification']
    print(request['SpotInstanceRequestId'], launch_spec['InstanceType'], launch_spec['KeyName'], request['State'])

def cancel_spot_requests():
  ec2 = u.create_ec2_resource()
  client = u.create_ec2_client()
  for request in client.describe_spot_instance_requests()['SpotInstanceRequests']:
    state = request['State']
    if state == 'cancelled' or state == 'closed':
      continue
    
    launch_spec = request['LaunchSpecification']
    
    print('cancelling', request['SpotInstanceRequestId'], launch_spec['InstanceType'], launch_spec['KeyName'], request['State'])
    
    client.cancel_spot_instance_requests(SpotInstanceRequestIds=[request['SpotInstanceRequestId']])


def main():
  if len(sys.argv) < 2:
    mode = 'list'
  else:
    mode = sys.argv[1]

  if mode == 'list' or mode == 'ls':
    list_spot_requests()
  elif mode == 'cancel':
    cancel_spot_requests()
    
  else:
    assert False, "Unknown mode "+mode
      
if __name__=='__main__':
  main()
