# GPU box launcher

First, setup your local cluster environment.
Follow official instructions for installing Anaconda on your machine, then create local Python 3.6 environment:

```
conda create --name gpubox -y python=3.6
source activate gpubox
```

Now download install launcher package

```
git clone https://github.com/diux-dev/cluster.git
cd cluster/gpubox
pip install -r ../requirements.txt
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

This sets up necessary resources and starts the run. It should take 5-10 minutes the first time you start the job because of the time to spin up AWS instance. You can modify the script and run again, and it'll take <30 seconds because instance is reused.

You can connect to machine using ssh command printed in the console after `Job ready for connection, run the following:`, or use helper utility which connects to the last instance that was launched.

```
export PATH=../:$PATH
connect
```

After connecting you should see something like this being printed:

```
8731.05 MB/s
8702.87 MB/s
8878.40 MB/s
8807.04 MB/s
8787.52 MB/s
8855.95 MB/s
8759.33 MB/s
8920.10 MB/s
8673.51 MB/s
8716.72 MB/s
8588.37 MB/s
8772.86 MB/s
8792.54 MB/s
8885.57 MB/s
8755.56 MB/s
8874.12 MB/s
```
