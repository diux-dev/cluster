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

# todo: allow to do ls, show tags
# todo: handle KeyError: 'PublicIpAddress'

import boto3
import time
import sys
import os
from datetime import datetime
from operator import itemgetter

import util as u

RESOURCE_NAME='nexus'


def toseconds(dt):
  # to invert:
  # import pytz
  # utc = pytz.UTC
  # utc.localize(datetime.fromtimestamp(seconds))
  return time.mktime(dt.utctimetuple())

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

  ec2 = boto3.client('ec2')
  response = ec2.describe_instances()

  username = os.environ.get("EC2_USER", "ubuntu")
  print("Using username '%s'"%(username,))
    
  
  instance_list = []
  for reservation in response['Reservations']:
    for instance in reservation['Instances']:
      if instance["State"]["Name"] != "running":
        continue
      instance_list.append((toseconds(instance['LaunchTime']), instance))
      
  import pytz
  from tzlocal import get_localzone # $ pip install tzlocal

  sorted_instance_list = sorted(instance_list, key=itemgetter(0))
  cmd = ''
  region = os.environ['AWS_DEFAULT_REGION']
  for (ts, instance) in reversed(sorted_instance_list):
    if fragment in instance['InstanceId'] or fragment in get_name(instance):
      
      localtime = instance['LaunchTime'].astimezone(get_localzone())
      keyname = instance.get('KeyName','none')
      assert keyname == RESOURCE_NAME
      keypair_fn = u.get_keypair_fn(keyname)

      print("Connecting to %s in %s launched at %s with key %s" % (instance['InstanceId'], region, localtime, keyname))
      cmd = "ssh -i %s -o StrictHostKeyChecking=no %s@%s" % (keypair_fn, username, instance['PublicIpAddress'])
      break
  if not cmd:
    print("no instance id contains fragment '%s'"%(fragment,))
  else:
    print(cmd)
    os.system(cmd)



if __name__=='__main__':
  main()
