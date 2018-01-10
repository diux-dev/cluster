# Cluster launcher

```
git clone https://github.com/diux-dev/cluster.git
export AWS_DEFAULT_REGION=us-west-2
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=KAa...

cd cluster
pip install -r requirements.txt
./create_resources_main.py
./launch_gpubox.py --instance-type=p2.xlarge --zone=us-west-2a
```

This creates necessary resources and launches p2.xlarge machine in zone us-west-2a with several ports open to outside traffic (8888, 8889, 8890, 6379, 6006, 6007, 6008) and EFS mounted at /efs. The script will print `ssh` instructions to use to connect, but also you can use `connect` utility in this folder to get in.

```
./connect gpubox
```

To bring down the instance, issue following command
```
./terminate gpubox
```


AWS resources will be reused for future invocations. If you want to delete the resources for some reason (ie, for debugging)
first terminate all the instances, and then issue

```
./delete_resources.py
```

This this will delete EFS and all data contained within it, networking resources, keypair and local keypair file.
