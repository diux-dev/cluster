#!/bin/bash

# This assumes base DLAMI - "Deep Learning AMI (Ubuntu) Version 12.0"

conda update -n base conda -y
pip install --upgrade pip
conda install python -y

sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt-get purge nvidia* -y
sudo apt-get autoremove -y
sudo apt-get autoclean -y
sudo rm -rf /usr/local/cuda*

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list

sudo apt-get update  -y
sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-9-2 cuda-drivers -y

# MAY NEED TO REBOOT COMPUTER HERE
#sudo reboot

sudo ldconfig
nvidia-smi
# Driver version should be: 396.37

# This is for cudnn 7.1.4 - however there's a bug that prevents us from using fp16 - https://github.com/pytorch/pytorch/issues/9465
# wget https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/cudnn-9.2-linux-x64-v7.1.tgz
# tar -xf cudnn-9.2-linux-x64-v7.1.tgz
# sudo cp -R cuda/include/* /usr/local/cuda-9.2/include
# sudo cp -R cuda/lib64/* /usr/local/cuda-9.2/lib64

# Install cudnn 7.1.3
wget https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/cudnn-9.1-linux-x64-v7.1.tgz
tar -xf cudnn-9.1-linux-x64-v7.1.tgz
sudo cp -R ~/cuda/include/* /usr/local/cuda-9.2/include
sudo cp -R ~/cuda/lib64/* /usr/local/cuda-9.2/lib64

# Install nccl 2.2.13 - might not need this
wget https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/nccl_2.2.13-1%2Bcuda9.2_x86_64.txz
tar -xf nccl_2.2.13-1+cuda9.2_x86_64.txz
sudo cp -R ~/nccl_2.2.13-1+cuda9.2_x86_64/* /usr/local/cuda-9.2/targets/x86_64-linux/
sudo cp -R ~/nccl_2.2.13-1+cuda9.2_x86_64/* /lib/nccl/cuda-9.2
sudo ldconfig


# Update .bashrc file - /lib/nccl/cuda-9.2
sed -i -e 's/cuda-9.1/cuda-9.2/g' ~/.bashrc
source ~/.bashrc
conda activate pytorch_p36


sudo apt-get install libcupti-dev

# Create pytorch_source conda env and install pytorch
git clone --recursive https://github.com/pytorch/pytorch.git
cd ~/pytorch

conda create -n pytorch_source -y
source activate pytorch_source
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing -y
conda install -c mingfeima mkldnn -y
conda install -c pytorch magma-cuda90 -y


# THIS VERSION IS SLOWER THAN JUST INSTALLING FROM PIP BELOW...
# Install libjpeg-turbo with pillow-simd
# sudo apt install yasm
# https://gist.github.com/soumith/01da3874bf014d8a8c53406c2b95d56b
# seems like it's faster without the top versioncond
pip uninstall pillow -y
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd


# Temporarily move old cuda libraries. For some reason building from source wants to point to these
pushd /usr/local/
sudo mv cuda-9.1/ cuda-9.1.bak
sudo mv cuda-9.0/ cuda-9.0.bak
sudo mv cuda-8.0/ cuda-8.0.bak
popd

# Must remember to change this line in DistributedSampler, otherwise performance is degraded
# https://github.com/pytorch/pytorch/pull/8958#issuecomment-410125932

pushd ~/pytorch
USE_C10D=1 USE_DISTRIBUTED=1 CUDA_HOME=/usr/local/cuda NCCL_ROOT_DIR=/lib/nccl/cuda-9.2 NCCL_LIB_DIR=/lib/nccl/cuda-9.2/lib NCCL_INCLUDE_DIR=/lib/nccl/cuda-9.2/include python setup.py install
popd

# Move back older cuda version libraries
pushd /usr/local/
sudo mv cuda-9.1.bak cuda-9.1/
sudo mv cuda-9.0.bak/ cuda-9.0/
sudo mv cuda-8.0.bak/ cuda-8.0
popd


# Install rest of dependencies
pip install torchvision torchtext
# pip uninstall pillow --yes
# CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
conda install jupyter bcolz scipy tqdm -y
pip install sklearn-pandas
conda install tqdm -y
pip install tensorboardX

# Create fastai environment
conda create -n fastai_source --clone pytorch_source
source activate fastai_source
pip install opencv-python isoweek pandas-summary seaborn graphviz
# conda install seaborn python-graphviz -y
git clone https://github.com/fastai/fastai.git
ln -s ~/fastai/fastai ~/anaconda3/envs/fastai/lib/python3.6/site-packages

