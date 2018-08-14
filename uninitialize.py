#!/usr/bin/env python
"""

Script to unitialize a list of instances.
Example usage

# unitialize all instances with name containing "baseline"
uninitialize baseline

"""

# todo: allow to do ls, show tags
# todo: handle KeyError: 'PublicIpAddress'

import boto3
import time
import sys
import os
from datetime import datetime
from operator import itemgetter

import util as u

def main():
  fragment = ''
  if len(sys.argv)>1:
    fragment = sys.argv[1]

  def get_name(instance_response):
    names = [entry['Value'] for entry in instance_response.get('Tags',[]) if
             entry['Key']=='Name']
    if not names:
      names = ['']
    assert len(names)==1
    return names[0]

  region = u.get_region()
  client = boto3.client('ec2', region_name=region)
  ec2 = boto3.resource('ec2', region_name=region)
  response = client.describe_instances()

  username = os.environ.get("EC2_USER", "ubuntu")
  print("Using username '%s'"%(username,))
    
  instance_list = []
  for instance in ec2.instances.all():
    if instance.state['Name'] != 'running':
      continue
    
    name = u.get_name(instance.tags)
    if (fragment in name or fragment in instance.public_ip_address or
        fragment in instance.id or fragment in instance.private_ip_address):

      print("Uninitializing %s %s %s"%(name, instance.public_ip_address, instance.private_ip_address))

      key_file = u.get_keypair_fn()
      ssh_client = u.SshClient(hostname=instance.public_ip_address,
                               ssh_key=key_file,
                               username=username)
      ssh_client.run('rm /tmp/is_initialized || echo "failed 1"')
      ssh_client.run('rm /tmp/nv_setup_complete || echo "failed 2"')
      ssh_client.run('rm *.sh')  # remove install scripts



if __name__=='__main__':
  main()
