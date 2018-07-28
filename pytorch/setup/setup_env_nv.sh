#!/bin/bash

# ImageNet training setup script for DLAMI + p3 instance
# (tested on "Deep Learning AMI (Ubuntu) Version 12.0")
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue
  
echo 'Starting script'

# echo '> update'
# Turn off apt-get update for now, since using recent DLAMI 12
# sudo apt-get update
# sudo apt-get upgrade -y

# echo '> gdb'
# sudo apt install -y gdb
# echo '> nload'
# sudo apt install -y nload
# echo '> htop'
# sudo apt install -y htop

# Change nccl to 9-1
sed -i -e 's/cuda-9.0/cuda-9.2/g' ~/.bashrc
source ~/.bashrc

# echo '>pytorch'
# conda install pytorch torchvision cuda91 -c pytorch -y
# echo '>tqdm'
# conda install tqdm -y

# index file used to speed up evaluation
echo '>indices'
pushd ~/data/imagenet 
wget --no-clobber https://s3.amazonaws.com/yaroslavvb/sorted_idxar.p
popd

# Installing through pip is faster than following this https://gist.github.com/soumith/01da3874bf014d8a8c53406c2b95d56b
# echo '>pillow'
# pip uninstall pillow -y
# CC="cc -mavx2" pip install -U --force-reinstall pillow-simd


# following GPU settings from below (not clear if helps) 
# http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/optimize_gpu.html
sudo nvidia-smi -ac 877,1530

# setting network settings -
# https://github.com/aws-samples/deep-learning-models/blob/5f00600ebd126410ee5a85ddc30ff2c4119681e4/hpc-cluster/prep_client.sh
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
sudo sysctl -w net.ipv4.tcp_rmem='4096 87380 16777216'
sudo sysctl -w net.ipv4.tcp_wmem='4096 65536 16777216'
sudo sysctl -w net.core.netdev_max_backlog=30000
sudo sysctl -w net.core.rmem_default=16777216
sudo sysctl -w net.core.wmem_default=16777216
sudo sysctl -w net.ipv4.tcp_mem='16777216 16777216 16777216'
sudo sysctl -w net.ipv4.route.flush=1

cd ~/

DATA_DIR=~/data
if [ ! -d "$DATA_DIR" ]; then
    mkdir data
fi

if [ ! -d "$DATA_DIR/imagenet" ]; then
    echo '>imagenet'
    cd $DATA_DIR
    # cat get those files from
    # wget https://s3.amazonaws.com/yaroslavvb/imagenet-data-sorted.tar
    rsync --progress /efs/data/imagenet-data-sorted.tar $DATA_DIR
    tar -xvf $DATA_DIR/imagenet-data-sorted.tar
    rm $DATA_DIR/imagenet-data-sorted.tar
    mv $DATA_DIR/raw-data $DATA_DIR/imagenet

    # can get those files from
    # wget https://s3.amazonaws.com/yaroslavvb/imagenet-sz.tar
    rsync --progress /efs/data/imagenet-sz.tar $DATA_DIR
    tar -xvf $DATA_DIR/imagenet-sz.tar
    rm $DATA_DIR/imagenet-sz.tar
    
    cd ~/
fi

echo ok > /tmp/nv_setup_complete
