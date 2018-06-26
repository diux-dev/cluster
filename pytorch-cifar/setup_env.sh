#!/bin/bash
echo 'Starting script'

FASTAI_DIR=~/fastai
if [ ! -d "$FASTAI_DIR" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    git clone https://github.com/fastai/fastai.git
    cd ~/fastai
    conda env update fastai

    # Distributed requires putorch 0.4 - consider using cuda91
    conda install pytorch torchvision cuda90 -c pytorch -y
    ln -s ~/fastai/fastai ~/anaconda3/envs/fastai/lib/python3.6/site-packages
fi

cd ~/fastai
git pull
# conda env update fastai
# SHELL=/bin/bash
# source activate fastai

cd ~/

DATA_DIR=~/data
if [ ! -d "$DATA_DIR" ]; then
    mkdir data
fi

