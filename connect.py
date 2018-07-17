#!/usr/bin/env python
"""

Script to connect to most recent instance whose id or name has substring

Usage:
To connect to most recently launched instance:
  connect

To connect to most recently launched instance containing 5i3 either in instance id or in instance name:
  connect 5i3

To connect to most recent instance with name simple
  connect simple


"""

# TODO: automatically determine RESOURCE_NAME from instance?
# TODO: automatically determine LINUX type from instance tags?
# todo: allow to do ls, show tags
# todo: handle KeyError: 'PublicIpAddress'

import boto3
import time
import sys
import os
from datetime import datetime
from operator import itemgetter

import util as u

import argparse
parser = argparse.ArgumentParser(description='Launch CIFAR training')
parser.add_argument('--skip-tmux', type=int, default=0,
                    help='whether to skip TMUX launch')
parser.add_argument('--fragment', type=str, default='',
                    help='fragment to filter by')
args = parser.parse_args()


def make_cmd(keypair_fn, username, public_ip_address):
  if args.skip_tmux:
    cmd = "ssh -i %s -o StrictHostKeyChecking=no %s@%s" % (keypair_fn, username,public_ip_address)
  else:
    cmd = 'connect_helper.sh %s %s %s'%(keypair_fn, username, public_ip_address)
  return cmd


def main():
  fragment = args.fragment

  # TODO: prevent CTRL+c/CTRL+d from killing session
  if not args.skip_tmux:
    print("Launching into TMUX session, use CTRL+b d to exit")

  region = u.get_region()
  client = u.create_ec2_client()
  ec2 =u.create_ec2_resource()
  response = client.describe_instances()

  username = os.environ.get("USERNAME", "ubuntu")
  print("Using username '%s'"%(username,))
    
  instance_list = []
  for instance in ec2.instances.all():
    if instance.state['Name'] != 'running':
      continue
    
    name = u.get_name(instance.tags)
    if (fragment in name or fragment in instance.public_ip_address or
        fragment in instance.id or fragment in instance.private_ip_address):
      instance_list.append((u.toseconds(instance.launch_time), instance))
      
  from tzlocal import get_localzone # $ pip install tzlocal

  filtered_instance_list = u.get_instances(fragment)
  if not filtered_instance_list:
    print("no instance id contains fragment '%s'"%(fragment,))
    return

  instance = filtered_instance_list[0]
  print("Found instance ", u.get_name(instance),
        " launched ", instance.launch_time.astimezone(get_localzone()))
  cmd = ''
  keypair_fn = u.get_keypair_fn(instance.key_name)
  cmd = make_cmd(keypair_fn, username, instance.public_ip_address)

  print(cmd)
  result = os.system(cmd)
  if username == 'ubuntu':
    username = 'ec2-user'
  elif username == 'ec2-user':
    username = 'ubuntu'
    
  if result != 0:
    print("ssh failed with code %d, trying username %s"%(result, username))
  cmd = make_cmd(keypair_fn, username, instance.public_ip_address)
  os.system(cmd)

if __name__=='__main__':
  main()
