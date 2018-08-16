#!/usr/bin/env python
"""
Script to kill all instances whose name matches given name

Usage:

./terminate.py gpu   # terminates all instances matching "gpu*"

If instance doesn't have name, it's assigned name "noname", so to kill those
./terminate.py noname

# TODO: shortcut for --name
# TODO: also delete corresponding placement groups if they are no longer used. (for now, workaround is to call "delete_placement_groups.py" once you hit placement group limit)
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
parser.add_argument('--skip-stopped', type=int, default=0,
                     help="don't terminate any instances that are stopped")
parser.add_argument('--soft', type=int, default=0,
                     help="use 'soft terminate', ie stop")
parser.add_argument('-d', '--delay', type=int, default=0,
                     help="delay termination for this many seconds")
# todo: add -n version
parser.add_argument('-n', '--name', type=str, default="",
                     help="name of tasks to kill, can be fragment of name")
parser.add_argument('-a', '--all', action='store_true', default="",
                     help="kill all instances, even tensorboard")
parser.add_argument('-y', '--yes', action='store_true', default="",
                     help="skip confirmation")
parser.add_argument('name2', nargs='*')
args = parser.parse_args()

# optionally to use "terminate name" command
if not args.name:
  assert len(args.name2) <= 1
  if not args.name2:
    fragment = ''
  else:
    fragment = args.name2[0]
else:
  assert len(args.name2) == 0
  fragment = args.name

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
    if not fragment in name:
      continue
    if not args.all and '.tb.' in name:
      continue
    if args.skip_stopped and state == 'stopped':
      continue
    if args.limit_to_key and not (USER_KEY_NAME in i.key_name):
      continue
    if state == 'terminated':
      continue
    instances_to_kill.append(i)
    print(u.get_name(i), i.instance_type, i.key_name,
          state if state=='stopped' else '')


  # print extra info if couldn't find anything to kill
  if not instances_to_kill:
    valid_names = sorted(list(set("%s,%s"%(u.get_name(i),
                                           u.get_state(i)) for i in instances)))
    from pprint import pprint as pp
    print("Current instances:")
    pp(valid_names)
    print("No running instances found for: Name '%s', key '%s'"%
          (fragment, USER_KEY_NAME))
    if not args.all:
      print("skipping tensorboard")
    return

  action = 'soft terminate' if args.soft else 'terminate'
  if args.yes:
    answer = 'y'
  else:
    answer = input("%d instances found, %s in %s? (y/N) " % (len(instances_to_kill), action, region))
  if not answer:
    answer = "n"
  if answer.lower() == "y" or args.yes:
    instance_ids = [i.id for i in instances_to_kill]
    if args.delay:
      print(f"Sleeping for {args.delay} seconds")
      time.sleep(args.delay)
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
