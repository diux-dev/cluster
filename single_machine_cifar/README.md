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

```
python generate_cifar10_tfrecords.py --data-dir=${PWD}/cifar-10-data
```