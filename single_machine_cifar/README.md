Self-contained launcher for single-machine CIFAR experiment.

Assumes that training/validation/eval data is in /efs/cifar-10-data

Need following files under `/efs/cifar-10-data`

```
-rw-rw-r-- 1 ubuntu ubuntu  31260000 Jan 23 23:54 eval.tfrecords
-rw-rw-r-- 1 ubuntu ubuntu 125040000 Jan 23 23:54 train.tfrecords
-rw-rw-r-- 1 ubuntu ubuntu  31260000 Jan 23 23:54 validation.tfrecords
```

Those datafiles were generated following instructions from
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator

To generate data, must do it in Python 2 environment
```
source activate mxnet_p27
python generate_cifar10_tfrecords.py --data-dir=/tmp/cifar-10-data
cp -R /tmp/cifar-10-data /tmp
```

For local sanity check

```
cd cifar10_estimator
source activate py2
python generate_cifar10_tfrecords.py --data-dir=/tmp/cifar-10-data

python cifar10_main.py --data-dir=/tmp/cifar-10-data \
                       --job-dir=/tmp/cifar10 \
                       --num-gpus=1 \
                       --train-steps=1000
```

On GTX1080 machine, this should finish in 1 minute, with final loss around 2.8549

<img src="local-tb.png">

To launch locally using tmux backend

```
pip install awscli boto3 paramiko pyyaml tzlocal portpicker
python launch.py
```

To launch on AWS

```
export AWS_DEFAULT_REGION=us-east-1
python launch.py --backend=aws
```

