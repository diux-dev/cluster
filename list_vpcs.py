#!/usr/bin/env python
import boto3

def _get_name(tags):
  """Helper utility to extract name out of tags dictionary.
      [{'Key': 'Name', 'Value': 'nexus'}] -> 'nexus'
 
     Assert fails if there's more than one name.
     Returns '' if there's less than one name.
  """
  
  names = [entry['Value'] for entry in tags if entry['Key']=='Name']
  if not names:
    return ''
  if len(names)>1:
    assert False, "have more than one name: "+str(names)
  return names[0]


for region in ['us-west-1', 'us-west-2', 'us-east-1']:
  print('='*80)
  print(region)
  print('='*80)
  client = boto3.client('ec2', region_name=region)
  response = client.describe_vpcs()

  for vpc_response in response['Vpcs']:
    key = _get_name(vpc_response.get('Tags', []))
    print("%30s %10s" %(key, vpc_response['VpcId']))
