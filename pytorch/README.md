# ImageNet training in PyTorch

Distributed training of imagenet implemented in pytorch.
Attempt to reproduce DawnBench results here: https://github.com/fastai/imagenet-fast, but scaled to multiple machines (not just GPU's).

# Setting up your environment

### Step 1: Install conda
```
git clone https://github.com/diux-dev/cluster.git
cd cluster
conda env update
source activate gpubox
```

(todo: rename gpubox to something else here and in environment.yml)

### Step 2: Setup AWS credentials
```
https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html
```
```
$ aws configure --profile diux
AWS Access Key ID [None]: AKIAI44QH8DHBEXAMPLE
AWS Secret Access Key [None]: je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: text
```

# Data preparation
Script assumes you already have imagenet downloaded and saved to ~/data directory inside an aws instance.  
Data directory structure should look like this:
- ~/data
  - imagenet
    - train
    - validation
  - imagenet-sz
    - 160
      - train
      - validation
    - 320
      - train
      - validation


If the data is not present, the script (setup/setup_env_nv.sh) will try to pull them from EFS located at `/efs/data/imagenet-data-sorted.tar` and `/efs/data/imagenet-sz.tar`. You should place those items on EFS, or use existing EBS volumes (see below)

You can find the files here.
- https://s3.amazonaws.com/yaroslavvb/imagenet-data-sorted.tar  
- https://s3.amazonaws.com/yaroslavvb/imagenet-sz.tar  

Data can be loaded onto an instance in 3 ways.

- Create/reuse AMI image based on an instance with imagenet already loaded in the data directory. Example of existing AMI which has it is ami-78dfe807 in us-east-1
- Create several EBS volumes with imagenet loaded on. Then attach volume to instance during runtime (see below)
- Save data to efs. Script will copy over the data and create the required directory structure


# Training pytorch script

Training on 1 p3.16xlarge instance

```
export REPOROOT=<git_repo_root/cluster>
cd ${REPOROOT}/pytorch
python launch_nv.py --name test --num-tasks 1 --spot
```

To view progress and interact

```
export PATH=${PATH}:${REPOROOT}
connect test
# press "CTRL+b d" to disconnect
# press "CTRL+b c" to create new window in same session
```

Training on 4 p3.16xlarge instances
```
cd ${REPOROOT}/pytorch
python launch_nv.py --name 4gpu_distributed --num-tasks 4 --spot --attach-volume imagenet_high_perf --params x4_args --ami-name=$ami
```
This command will launch 4 p3.16xlarge instances. Setup the environment. 
Then on each machine it will run the train_imagenet_nv.py script.

# Tricks

## Speeding up reruns by re-using EBS volumes

Launching through AMI is slow because AMI is stored on S3, so each time you launch, it will pull the data from S3. Instead you could create a set of EBS volumes, and attach your instances to them. EBS is relatively cheap compared to p3 instance cost, so for frequent runs you will save money in addition to time. IE, 500GB EBS volume will cost $50/month

Once you have a single instance with the data setup, you can snapshot the value, and create several volumes from that snapshot with names like vol_0, vol_1. Then you can use parameter `attach-volume` in the launcher script. IE, `launch_nv.py --num-tasks 2 --attach-volume vol`, it will automatically attach tasks 0 and 1 to volumes `vol_0` and `vol_1` respectively.

Example of creating single volume from snapshot
```
import util as u
ec2 = u.create_ec2_resource()
volumes = list(ec2.volumes.all())
v = volumes[0]

response = ec2.create_snapshot(
    Description='Imagenet data snapshot',
    VolumeId=v.id,
    TagSpecifications=[
        {
            'ResourceType': 'snapshot',
            'Tags': [
                {
                    'Key': 'Name',
                    'Value': 'snapshot'
                },
            ]
        },
    ]
)
...
tag_specs = [{
    'ResourceType': 'volume',
    'Tags': [{
        'Key': 'Name',
        'Value': 'imagenet_high_perf_1'
    }]
}]
volume = ec2.create_volume(Size=600, VolumeType='io1', TagSpecifications=tag_specs, AvailabilityZone='us-west-2c', SnapshotId='snap-0b1c0cd991ccf85df', Iops=18000)
```
