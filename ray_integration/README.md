You need to choose AMI, create a security group that allows SSH access, and get keyname/.pem pair

AMI's are region specific. For Oregon, I do this (Ubuntu 16.04 AMI)

```
export AMI=ami-0def3275
export KEY_NAME=yaroslav
export SSH_KEY_PATH=yaroslav.pem
export SECURITY_GROUP=open
```

To install and run in Python 3

```
pip install -r requirements.txt
./launch.py
```

This will spin up 2 instances, install necessary dependencies, download Robert's gist and start the benchmark running. You will see something like this printed on the console:

```
To see results:
ssh -i yaroslav.pem -o StrictHostKeyChecking=no ubuntu@34.215.161.205
tmux a -t tmux
```

Once you execute these two commands, you'll attach to the tmux session that was used to launch the experiment and see the numbers

```
Write throughput is 521.7789154048995MB/s.
Write throughput is 526.2796561862427MB/s.
Write throughput is 531.4685290340004MB/s.
Write throughput is 533.2124070257983MB/s.
Write throughput is 522.066088047618MB/s.
```