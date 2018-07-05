#!/bin/bash

# to be used with vanilla DLAMI v10
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue
# "us-east-1": "ami-6d720012", "us-east-2": "ami-23c4fb46", "us-west-2": "ami-e580c79d",
  
echo 'Starting script'

if ! conda list Pillow-SIMD | grep -q Pillow-SIMD; then
    pip uninstall pillow --yes
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
fi

conda install pytorch torchvision cuda90 -c pytorch -y

cd ~/

DATA_DIR=~/data
if [ ! -d "$DATA_DIR" ]; then
    mkdir data
fi

if [ ! -d "$DATA_DIR/imagenet" ]; then
    cd $DATA_DIR
    rsync --progress /efs/imagenet-data-sorted.tar $DATA_DIR
    tar -xvf $DATA_DIR/imagenet-data-sorted.tar
    rm $DATA_DIR/imagenet-data-sorted.tar
    mv $DATA_DIR/raw-data $DATA_DIR/imagenet

    rsync --progress /efs/imagenet-sz.tar $DATA_DIR
    tar -xvf $DATA_DIR/imagenet-sz.tar
    rm $DATA_DIR/imagenet-sz.tar
    
    cd ~/
fi

echo ok > /tmp/nv_setup_complete
