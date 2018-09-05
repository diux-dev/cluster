Code to reproduce ImageNet in 18 minutes


Pre-requisites: Python 3.6 or higher

```
pip install -r requirements.txt
aws configure  (or set your AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_DEFAULT_REGION)
python train.py  # pre-warming
python train.py 
```

Additionally there are 8 machine and 1 machine configurations

```
python train.py --machines=1
python train.py --machines=4
python train.py --machines=8
```

You can see the progress by launching tensorboard, or by connecting to one of the instances, connecting to tmux session and looking at printed progress logs.
