# ImageNet training in PyTorch

Distributed training of imagenet implemented in pytorch.
Attempt to reproduce DawnBench results here: https://github.com/fastai/imagenet-fast, but scaled to multiple machines (not just GPU's).

# Setting up your environment

### Step 1: Install conda
```
git clone https://github.com/diux-dev/cluster.git
cd cluster
conda env update
```

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


If not, you can find the files here:  
https://s3.amazonaws.com/yaroslavvb/imagenet-data-sorted.tar  
https://s3.amazonaws.com/yaroslavvb/imagenet-sz.tar  

Download these files and save them to EFS. 

Data can be loaded onto an instance in 3 ways.  
Option 1: Create an AMI image based on an instance with imagenet already loaded in the data directory  
Option 2: Create several EBS volumes with imagenet loaded on. Then attach volume to instance during runtime.  
Option 3: Save data to efs. Script will copy over the data and create the required directory structure. This will take the longest  
Option 4: Create a few instances and just reuse those for training.

# Training pytorch script

Training on 4 p3.16xlarge instances
```
cd pytorch
python launch_nv.py --instance-type p3.16xlarge --num-tasks 4 --job-name cluster_4_region_c --zone us-west-2c --ami ami-53c8822b --placement-group pytorch_cluster_c
```
This command will launch 4 p3.16xlarge instances. Setup the environment. 
Then on each machine it will run the train_imagenet_nv.py script.

# Training using fastai library
