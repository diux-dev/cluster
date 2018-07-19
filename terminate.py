#!/usr/bin/env python
"""
Script to kill all instances whose name matches given name

Usage:

./terminate.py gpu   # terminates all instances matching "gpu*"

# TODO: shortcut for --name
# TODO: also delete corresponding placement groups (see "delete_placement_groups.py")
"""

import boto3
import time
import getpass
import sys
import os

import util as u

import argparse
parser = argparse.ArgumentParser(description='terminate')
parser.add_argument('--limit-to-key', type=int, default=1,
                    help=("ignores any jobs not launched by current user "
                          "(determined by examining instance private_key name"))
parser.add_argument('--skip-stopped', type=int, default=1,
                     help="don't terminate any instances that are stopped")
parser.add_argument('--skip-tensorboard', type=int, default=1,
                     help="don't terminate tensorboard jobs")
parser.add_argument('--soft', type=int, default=0,
                     help="use 'soft terminate', ie stop")
# todo: add -n version
parser.add_argument('--name', type=str, default="",
                     help="name of tasks to kill, can be fragment of name")
args = parser.parse_args()

USER_KEY_NAME=getpass.getuser()
if not args.limit_to_key:
  print("*"*80)
  print("Warning: killing jobs not launched by this user!")
  print("*"*80)

# TODO: not list stopped instances when doing stopped termination (its no-op in
# that case)
def main():
  ec2 = u.create_ec2_resource()         # ec2 resource
  ec2_client = u.create_ec2_client()    # ec2 client
  instances = list(ec2.instances.all()) # todo: use filter?
  region = u.get_region()

  instances_to_kill = []
  for i in instances:
    name = u.get_name(i.tags)
    state = i.state['Name']
    if not args.name in name:
      continue
    if args.skip_tensorboard and '.tb.' in name:
      continue
    if args.skip_stopped and state == 'stopped':
      continue
    if args.limit_to_key and not (USER_KEY_NAME in i.key_name):
      continue
    if state == 'terminated':
      continue
    instances_to_kill.append(i)
    print(u.get_name(i))


  # print extra info if couldn't find anything to kill
  if not instances_to_kill:
    valid_names = sorted(list(set("%s,%s"%(u.get_name(i),
                                           u.get_state(i)) for i in instances)))
    from pprint import pprint as pp
    print("Current instances:")
    pp(valid_names)
    print("No running instances found for: Name '%s', key '%s'"%
          (args.name, USER_KEY_NAME))
    return

  action = 'soft terminate' if args.soft else 'terminate'
  answer = input("%d instances found, %s in %s? (Y/n) " % (len(instances_to_kill), action, region))
  if not answer:
    answer = "y"
  if answer.lower() == "y":
    instance_ids = [i.id for i in instances_to_kill]
    if args.soft:
      response = ec2_client.stop_instances(InstanceIds=instance_ids)
      print("soft terminating, got response: %s", response)
    else:
      response = ec2_client.terminate_instances(InstanceIds=instance_ids)
      print("terminating, got response: %s", response)
  else:
    print("Didn't get y, doing nothing")
  

if __name__=='__main__':
  main()
