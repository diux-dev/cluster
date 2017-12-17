# Setup

You need to choose AMI, create a security group that allows SSH access, and get keyname/.pem pair, and install Python 3 locally


AMI's are region specific. For Oregon, I do this to use standard Ubuntu 16.04 AMI from Amazon

```
pip install -r requirements.txt
aws configure
export AWS_DEFAULT_REGION=us-west-2
export AMI=ami-0def3275
export KEY_NAME=yaroslav
export SSH_KEY_PATH=yaroslav.pem
export SECURITY_GROUP=open
```


# Single machine example

```
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

## Two machine Ray example on Beefy machines
Uses script downloaded from Robert's async_sgd_benchmark_multinode.py [gist](https://gist.githubusercontent.com/robertnishihara/24979fb01b4b70b89e5cf9fbbf9d7d65/raw/b2d3bb66e881034039fbd244d7f72c5f6b425235/async_sgd_benchmark_multinode.py) on Dec 12th.

```
./launch_ray_beefy.py
```
This will spin up 2 instances, install necessary dependencies, upload gist file. You will see something like this printed on the console:

```
To see results:
ssh -i yaroslav.pem -o StrictHostKeyChecking=no ubuntu@34.215.161.205
tmux a
```

Once you execute these two commands, you'll attach to the tmux session that was used to launch the experiment and see the numbers.

The first 5 numbers are due to measurement bug, ignore them
```
Write throughput is 3912.902203452332MB/s.
Write throughput is 3912.902203452332MB/s.
Write throughput is 1299.0097899535428MB/s.
Write throughput is 1299.0097899535428MB/s.
Write throughput is 942.7658299178238MB/s.
Write throughput is 1052.2112765757333MB/s.
Write throughput is 948.8590160709089MB/s.
Write throughput is 1036.3927498042233MB/s.
Write throughput is 957.4880657957102MB/s.
```

To disconnect from tmux session, but stay in shell, CTRL+b d

[## Four machine Ray example with synchronous parameter server](#sync-example)

```
./launch_ray_sync.py
```

This will reserve 5 c5.large machines and start running ray synchronous parameter server. The first machine is used as the head node and as the client node to launch commands. Remaining 4 machines is used as 2 ps workers and 2 gradient workers.

Setup for 5 machines is currently sequential and takes about 6 minutes end-to-end.

After 6 minutes you should see the instructions to connect all nodes

```
Connect to head node:
ssh -i /Users/yaroslav/d/yaroslav.pem -o StrictHostKeyChecking=no ubuntu@34.215.140.112
tmux a
Other nodes:
0 ssh -i /Users/yaroslav/d/yaroslav.pem -o StrictHostKeyChecking=no ubuntu@35.161.116.239
tmux a
1 ssh -i /Users/yaroslav/d/yaroslav.pem -o StrictHostKeyChecking=no ubuntu@54.68.60.179
tmux a
2 ssh -i /Users/yaroslav/d/yaroslav.pem -o StrictHostKeyChecking=no ubuntu@34.211.21.138
tmux a
3 ssh -i /Users/yaroslav/d/yaroslav.pem -o StrictHostKeyChecking=no ubuntu@34.215.154.208
tmux a
```