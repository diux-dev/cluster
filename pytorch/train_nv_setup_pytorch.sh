#!/bin/bash

set -e
set -o xtrace
DEBIAN_FRONTEND=noninteractive

sudo rm /etc/apt/apt.conf.d/*.*
sudo apt update
# sudo apt install unzip -y
# sudo apt -y upgrade --force-yes -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"
sudo apt -y autoremove
# sudo ufw allow 8888:8898/tcp
# sudo apt -y install --force-yes -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" qtdeclarative5-dev qml-module-qtquick-controls
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh -b

# run pytorch and dependencies. ideally from an environment.yml
conda install pytorch torchvision cuda91 -c pytorch -y

# we should customize this to pytorch only
echo ". $HOME/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
# echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
# export PATH=~/anaconda3/bin:$PATH
source ~/.bashrc

# fastai?
git clone https://github.com/fastai/fastai.git
cd fastai/
conda env update
echo 'source activate fastai' >> ~/.bashrc
conda activate fastai
source ~/.bashrc