# Cluster launcher

```
git clone https://github.com/diux-dev/cluster.git
export AWS_DEFAULT_REGION=us-west-2
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=KAa...

cd cluster
pip install -r requirements.txt
./create_resources.py
./launch_gpubox.py --instance_type=p2.xlarge --zone=us-west-2a
```

This creates necessary resources and launches p2.xlarge machine in zone us-west-2a with several ports open to outside traffic (8888, 8889, 8890, 6379, 6006, 6007, 6008) and EFS mounted at /efs.

To clean-up resources, first terminate all the instances using following command
```
terminate gpubox
```

Then issue
```
./delete_resources.py
```

Note that this will delete EFS and all data contained within it, VPC, keypairs.
