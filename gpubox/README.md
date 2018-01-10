# GPU box launcher

First, setup your local cluster environment.
Follow instructions for installing Anaconda, then create local Python 3.6 environment:

```
conda create --name gpubox python=3.6
source activate gpubox
```



```
git clone https://github.com/diux-dev/cluster.git
cd cluster/gpubox
```
Now obtain your secret and public keys and set them appropriately.

```
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=KAa...
```

Now set your AWS region where you have resources

```
export AWS_DEFAULT_REGION=us-west-2
```

Now launch your job

```
./launch.py
```

This 