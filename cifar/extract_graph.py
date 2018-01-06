#!/bin/env python
# Getting graph of event files
# To get event files
# scp -i /Users/yaroslav/nexus-us-east-1.pem ubuntu@52.86.36.67:/efs/runs/tfpp00/events.out.tfevents.1515205593.ip-192-168-46-3 .
# scp -i /Users/yaroslav/nexus-us-east-1.pem ubuntu@52.86.36.67:/efs/runs/tfpp00/events.out.tfevents.1515206122.ip-192-168-46-3 .
# mv events.out.tfevents.1515205593.ip-192-168-46-3 events1
# mv events.out.tfevents.1515206122.ip-192-168-46-3 events2



from tensorflow.python.summary import summary_iterator


events = summary_iterator.summary_iterator('events2')
summaries = [e.summary for e in events if e.summary.value]
values = []
for summary in summaries:

  tag = summary.value[0].tag
  if tag != 'global_step/sec':
    continue
  import pdb; pdb.set_trace()
  print(summary.value[0].simple_value)
