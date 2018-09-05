#!/bin/bash
#
# ImageNet training setup script for pytorch.imagenet.source.v7 AMI
# That image is a fork of Ubuntu DLAMI v12 with August build of pytorch from source made available in
# pytorch_source conda env
source activate pytorch_source

conda install tqdm -y
pip install tensorboardX

# index file used to speed up evaluation
pushd ~/data/imagenet
wget --no-clobber https://s3.amazonaws.com/yaroslavvb/sorted_idxar.p
popd

pip uninstall pillow -y
CC="cc -mavx2" pip install -U pillow-simd

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
