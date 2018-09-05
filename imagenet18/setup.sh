#!/bin/bash

# ImageNet training setup script for DLAMI + p3 instance
# must be run after "source activate" to put things into current env

# Change nccl to 9-1
# sed -i -e 's/cuda-9.0/cuda-9.2/g' ~/.bashrc
source ~/.bashrc

conda install tqdm -y
pip install tensorboardX

# index file used to speed up evaluation
echo '>indices'
pushd ~/data/imagenet 
wget --no-clobber https://s3.amazonaws.com/yaroslavvb/sorted_idxar.p
popd

# Installing through pip is faster than following this https://gist.github.com/soumith/01da3874bf014d8a8c53406c2b95d56b
# echo '>pillow'
pip uninstall pillow -y
CC="cc -mavx2" pip install -U pillow-simd
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

echo ok > /tmp/nv_setup_complete
