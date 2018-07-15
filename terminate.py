#!/usr/bin/env python
"""
Script to kill all instances whose name matches given fragment

Usage:

./terminate.py gpu   # terminates all instances matching "gpu*"

# TODO: also delete corresponding placement groups
# TODO: filter by IAM instead of key name
"""

import boto3
import time
import getpass
import sys
import os

import util as u


# By default only touch instances launched with LIMIT_TO_KEY
# this is to prevent accidentally wiping all jobs on the account
# set to '' to remove this restriction
#LIMIT_TO_KEY = os.environ.get("LIMIT_TO_KEY", "dontkillanything")
LIMIT_TO_KEY=getpass.getuser()

# todo: make this into an overridable flag for when you need to kill TB
SKIP_TENSORBOARD=True  # true to avoid killing tensorboard jobs
SKIP_STOPPED=True  # don't terminate stopped jobs

def main():
  global LIMIT_TO_KEY
  if len(sys.argv)>1:
    if sys.argv[1] == 'CLEANSLATE': # kill everything
      fragment = ''
      LIMIT_TO_KEY = ''
    else:
      fragment = sys.argv[1]
  else:
    fragment = ''
    
  ec2 = boto3.client('ec2')
  response = ec2.describe_instances()

  def get_name(instance_response):
    names = [entry['Value'] for entry in instance_response.get('Tags',[]) if
             entry['Key']=='Name']
    if not names:
      names = ['']
    assert len(names)==1
    return names[0]

  instance_list = []
  for reservation in response['Reservations']:
    for instance_response in reservation['Instances']:
      instance_list.append((get_name(instance_response),
                            instance_response))

  region = u.get_region()

  instances_to_kill = []
  for (name, instance_response) in instance_list:
    if not fragment in name:
      continue
    if SKIP_TENSORBOARD and '.tb.' in name:
      #      print("Not killing tensorboard job", name)
      continue
    if SKIP_STOPPED and instance_response['State']['Name'] == 'stopped':
      #      print("Not killing stopped job", name)
      continue
    
    key = instance_response.get('KeyName', '')
    if LIMIT_TO_KEY and not (LIMIT_TO_KEY in key):
      #      print("instance %s matches but key %s doesn't match desired key %s, "
      #            "skipping" %(name, key, LIMIT_TO_KEY))
      continue
    state = instance_response['State']['Name']
    if state == 'terminated':
      continue
    instances_to_kill.append((instance_response['InstanceId'],
                              name,
                              instance_response['AmiLaunchIndex'],
                              state))

  for (instance_id, name, task_id, state) in instances_to_kill:
    print("%s:%s   %s"%(name, task_id, state))


  if not instances_to_kill:
    valid_names = sorted(list(set("%s,%s"%(name,instance_response['State']['Name']) for (name, instance_response) in instance_list)))
    from pprint import pprint as pp
    print("Current instances:")
    pp(valid_names)
    print("No running instances found for: Fragment '%s', key '%s'"%
          (fragment, LIMIT_TO_KEY))
    return
  
  answer = input("%d instances found, terminate in %s? (Y/n) " % (len(instances_to_kill),region))
  if not answer:
    answer = "y"
  if answer.lower() == "y":
    instance_ids = [record[0] for record in instances_to_kill]
    response = ec2.terminate_instances(InstanceIds=instance_ids)
    print("Terminating, got response: %s", response)
  else:
    print("Didn't get y, doing nothing")
  

if __name__=='__main__':
  main()
