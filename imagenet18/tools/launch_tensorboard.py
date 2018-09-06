#!/usr/bin/env python
import ncluster
ncluster.use_aws()

task = ncluster.make_task('tensorboard',
                          instance_type='r5.large',
                          image_name='Deep Learning AMI (Ubuntu) Version 13.0')
task.run('source activate tensorflow_p36')
task.run(f'tensorboard --logdir={task.logdir}/..', async=True)
print(f"Tensorboard at http://{task.public_ip}:6006")
