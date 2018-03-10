Parameter server benchmarks for TensorFlow and Ray

# Running Ray benchmark on AWS
```
git clone https://github.com/diux-dev/cluster.git
cd cluster/psbench
export AWS_ACCESS_KEY_ID=AKIAJZY.....5Y
export AWS_SECRET_ACCESS_KEY=HnKmxcnOvVBIUBiK....28Uu
export AWS_DEFAULT_REGION=us-west-2
# run one of the following
python launch_ray_adder.py --cluster=aws --zone=us-west-2a --instance=c5.large
python launch_ray_adder.py --cluster=aws --zone=us-west-2a --instance=c5.large --ps=4 --workers=1 --name=run2
python launch_ray_adder.py --cluster=aws --zone=us-west-2a --instance=c5.large --ps=1 --workers=4 --name=run3
```

# Running Ray benchmark locally
```
conda create --name cifar --clone my_favourite_env
source activate cifar
pip install portpicker
pip install tensorflow
pip install ray

git clone https://github.com/diux-dev/cluster.git
cd cluster/psbench
# run one of the following
python launch_ray_adder.py
python launch_ray_adder.py --workers=1 --ps=2
```


# Running TensorFlow benchmarks

```
python launch_tf_adder.py # runs locally with 1 parameter server, 1 worker
python launch_tf_adder.py --cluster=aws --zone=us-west-2a --instance=c5.18xlarge --ps=1 --workers=4 # runs on AWS with 1 parameter server, 4 workers
```

Running AWS command above will provide a link to TensorBoard and you would see something like this

<img src="4worker.png">

# Experiments

TensorFlow async transfer summaries: https://docs.google.com/document/d/1K0-39NW3ywSx9SDMMF0dUFqSKr6RxtUUIV5XKIp5r18/edit

Ray sync summary transfer summaries: https://docs.google.com/document/d/1StrxUjDxOmhiiPD2_sZZElEE0mbq16tBEOqLaGSY6jc/edit


# Troubleshooting

1. Something doesn't work:

Connect to the instance and look at the errors. Everything is launched in tmux session, so you can ssh to instance, and do "tmux a" to attach to default tmux session and see error messages. Also you can use "up" arrow to scroll through commands that have been issues and possibly rerun them.

A "connect" script makes it easier to connect to instances, ie to ssh into instance with name 0.worker.pbench15, do this
```
connect 0.worker.psbench15
tmux a
```

The "pem" file needed for ssh connection is under ~, with your username and region in its name. IE, to connect to Ubuntu instance use this

```
ssh -i ~/nexus-yaroslav-us-east-1.pem ubuntu@123.123.123.32
tmux a
```

To connect to Amazon Linux instance, the username is 'ec2-user'

2. Install failed

Installation is handled through "UserData". Install script passes a string during instance creation, and this string is run through bash at startup. If something failed, double-check the userdata, and look at the output:

```
curl http://169.254.169.254/latest/user-data/
cat /var/log/cloud-init-output.log
```
