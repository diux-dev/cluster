#!/usr/bin/env python
#
# Creates resources
# This script creates VPC/security group/keypair if not already present

import os
import argparse
import boto3
import sys
import time
from collections import OrderedDict

parser = argparse.ArgumentParser(description='launch simple')
parser.add_argument('--instance_type', type=str, default='t2.micro',
                     help="type of instance")
args = parser.parse_args()

import create_resources as create_resources_lib

def main():
  create_resources_lib.create_resources()
  
if __name__=='__main__':
  main()
