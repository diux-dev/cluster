#!/bin/bash

# to be used with vanilla DLAMI v10
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue
# "us-east-1": "ami-6d720012", "us-east-2": "ami-23c4fb46", "us-west-2": "ami-e580c79d",
  
echo 'Starting script'

# Change nccl to 9-1
sed -i -e 's/cuda-9.0/cuda-9.1/g' ~/.bashrc
source ~/.bashrc

conda install pytorch torchvision cuda91 -c pytorch -y
conda install tqdm -y

if ! conda list Pillow-SIMD | grep -q Pillow-SIMD; then
    pip uninstall pillow --yes
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
fi



# install nvidia DALI
# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali
# wget https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/cuda-linux64-nvjpeg-9.0.tar.gz
# tar xzvf cuda-linux64-nvjpeg-9.0.tar.gz
# pip install protobuf -y
# pip install opencv-contrib-python


# Installing libjpeg-turbo
# sudo apt-get remove libjpeg8
# sudo apt-get install libjpeg-turbo8

conda uninstall --force pillow -y 
# install libjpeg-turbo to $HOME/turbojpeg 
git clone https://github.com/libjpeg-turbo/libjpeg-turbo 
pushd libjpeg-turbo 
mkdir build 
cd build 
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME/turbojpeg 
make 
make install 
popd 
# install pillow-simd with jpeg-turbo support 
git clone https://github.com/uploadcare/pillow-simd 
pushd pillow-simd 
CPATH=$HOME/turbojpeg/include LIBRARY_PATH=$HOME/turbojpeg/lib CC="cc -mavx2" python setup.py install 
# add turbojpeg to LD_LIBRARY_PATH 
export LD_LIBRARY_PATH="$HOME/turbojpeg/lib:$LD_LIBRARY_PATH" 
popd


# setting max speed. Doesn't seem to boost performace though
sudo nvidia-smi -ac 877,1530

cd ~/

DATA_DIR=~/data
if [ ! -d "$DATA_DIR" ]; then
    mkdir data
fi

if [ ! -d "$DATA_DIR/imagenet" ]; then
    cd $DATA_DIR
    rsync --progress /efs/data/imagenet-data-sorted.tar $DATA_DIR
    tar -xvf $DATA_DIR/imagenet-data-sorted.tar
    rm $DATA_DIR/imagenet-data-sorted.tar
    mv $DATA_DIR/raw-data $DATA_DIR/imagenet

    rsync --progress /efs/data/imagenet-sz.tar $DATA_DIR
    tar -xvf $DATA_DIR/imagenet-sz.tar
    rm $DATA_DIR/imagenet-sz.tar
    
    cd ~/
fi

echo ok > /tmp/nv_setup_complete
