#!/bin/env python
# Getting graph of event files
# scp -i /Users/yaroslav/nexus-us-east-1.pem ubuntu@52.86.36.67:/efs/runs/raypp02/events.out.tfevents.1515209718.ip-192-168-42-131 events3

from tensorflow.python.summary import summary_iterator


events = summary_iterator.summary_iterator('events3')
summaries = [e.summary for e in events]
values = []
counter = 0
with open('events.csv', 'w') as outf:
  outf.write('[')
  for summary in summaries:
    # try:
    #     tag = summary.value[0].tag
    # except:
    #     continue
    # if tag != 'steps_per_sec':
    #     continue
      #    print(tag)
    print(summary)
    counter+=1
    if counter>4000:
      break        # convert time/step to steps/sec
  #    outf.write('%s,'%(summary.value[0].simple_value,))
  outf.write(']')
