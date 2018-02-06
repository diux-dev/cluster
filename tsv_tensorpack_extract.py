import sys, os, re
from dateutil import parser

epoch_num_re = re.compile('.*Start Epoch (\d+) .*')
top1_re = re.compile('.*val-error-top1: ([.0123456789]+)')
top5_re = re.compile('.*val-error-top5: ([.0123456789]+)')
time_re = re.compile('.* ([:0123456789]+) @.*')

epoch_number = 0
first_time = None
for line in open(sys.argv[1]):
  if not first_time:
    first_time = parser.parse(time_re.findall(line)[0])
  if epoch_num_re.match(line):
    epoch_number = int(epoch_num_re.findall(line)[0])
  if top1_re.match(line):
    top1 = top1_re.findall(line)[0]
  if top5_re.match(line):
    top5 = top5_re.findall(line)[0]
    current_time = parser.parse(time_re.findall(line)[0])

    delta = current_time - first_time
    hours = delta.seconds/3600
    print(epoch_number, hours, top1, top5)

