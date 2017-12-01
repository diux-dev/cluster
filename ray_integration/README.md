# Setup

You need to choose AMI, create a security group that allows SSH access, and get keyname/.pem pair, and install Python 3 locally

AMI's are region specific. For Oregon, I do this to use standard Ubuntu 16.04 AMI from Amazon

```
export AMI=ami-0def3275
export KEY_NAME=yaroslav
export SSH_KEY_PATH=yaroslav.pem
export SECURITY_GROUP=open
```


# Single machine example

```
pip install -r requirements.txt
./launch_simple.py
```

This will spin up an instance, install deps, and start a Python loop. You'll see something like this printed on the console

```
To connect:
ssh -i yaroslav.pem -o StrictHostKeyChecking=no ubuntu@34.215.161.205
tmux a
```
Once you execute this, you should see something like this

```
step 34
step 35
step 36
step 37
step 38
step 39
step 40
```

There's also a helper script in the directory above to simply connecting to instances. Instead of `ssh` command you can run this to connect to most recently launched instance:

```
connect
```

Instances are launched with a specific job name (`simple` in the case above), you can terminate all instances with this name using `terminate` script in the directory above

```
terminate simple
```


## Two machine Ray example

```
pip install -r requirements.txt
./launch_ray.py
```

This will spin up 2 instances, install necessary dependencies, download Robert's gist and start the benchmark running. You will see something like this printed on the console:

```
To see results:
ssh -i yaroslav.pem -o StrictHostKeyChecking=no ubuntu@34.215.161.205
tmux a
```

Once you execute these two commands, you'll attach to the tmux session that was used to launch the experiment and see the numbers

```
Write throughput is 521.7789154048995MB/s.
Write throughput is 526.2796561862427MB/s.
Write throughput is 531.4685290340004MB/s.
Write throughput is 533.2124070257983MB/s.
Write throughput is 522.066088047618MB/s.
```

You can clean-up your instanes using `terminate` script in directory above

```
terminate ray
```
