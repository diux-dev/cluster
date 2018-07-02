#!/bin/bash
conda upgrade conda -y
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

sudo reboot

sudo ldconfig
nvidia-smi

wget https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/cudnn-9.2-linux-x64-v7.1.tgz
tar -xf cudnn-9.2-linux-x64-v7.1.tgz
sudo cp -R cuda/include/* /usr/local/cuda-9.2/include
sudo cp -R cuda/lib64/* /usr/local/cuda-9.2/lib64



wget https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/nccl_2.2.13-1%2Bcuda9.2_x86_64.txz
tar -xf nccl_2.2.13-1+cuda9.2_x86_64.txz
cd nccl_2.2.13-1+cuda9.2_x86_64
sudo cp -R * /usr/local/cuda-9.2/targets/x86_64-linux/
sudo ldconfig

cd ~/
sudo cp -R ~/nccl_2.2.13-1+cuda9.2_x86_64 /lib/nccl/cuda-9.2

# MUST REMEMBER TO UDPATE .bashrc file - /lib/nccl/cuda-9.2

sudo apt-get install libcupti-dev



git clone --recursive https://github.com/pytorch/pytorch.git
cd ~/pytorch

conda create -n pytorch_source -y
source activate pytorch_source
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing -y
conda install -c mingfeima mkldnn -y
conda install -c pytorch magma-cuda90 -y

python setup.py install

cd ~/
pip install torchvision torchtext
pip uninstall pillow --yes
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
conda install jupyter bcolz scipy tqdm -y
pip install sklearn-pandas

conda create -n fastai --clone pytorch_source
source activate fastai
pip install opencv-python isoweek pandas-summary seaborn graphviz
# conda install seaborn python-graphviz -y
git clone https://github.com/fastai/fastai.git
ln -s ~/fastai/fastai ~/anaconda3/envs/fastai/lib/python3.6/site-packages
cd ~/

