# GPU box launcher

Steps: 1) install Anaconda environment 2) setup AWS credentials 3) launch


## Step 1: Anaconda

Follow official instructions for installing Anaconda on your machine, then create local Python 3.6 environment:

```
git clone https://github.com/diux-dev/cluster.git
cd cluster/gpubox
conda env remove --name nexus
conda env create -f ../nexus.yml 
source activate nexus
```

## Step 2: AWS credentials

Environment variables contain your account credentials (official [docs](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)).
Get your public/private access key from AWS console (https://console.aws.amazon.com/iam/home#/security_credential) and set them as below:

```
export AWS_ACCESS_KEY_ID=AKIAIIsomevalues
export AWS_SECRET_ACCESS_KEY=g9wO2esomevalues
```

Now set your AWS region where you have resources. IE, for Oregon

```
export AWS_DEFAULT_REGION=us-west-2
```

Finally set the availability zone to use.

```
export AWS_DEFAULT_REGION=us-west-2d
```



## Step 3: launch remote Jupyter notebook

```
./launch.py --password somepassword
```
After a couple of minutes you should see something like this

```
Jupyter notebook will be at http://52.222.3.28:8888
```
Log into this jupyter server using password you provided.

## Step 4 (optional): run a benchmark remotely

You can modify example below to create scripts that will self-execute on the machine. Try:

```
./launch.py --mode tf-benchmark
```

This will run a simple tensorflow benchmark that adds vectors of 1 non-stop.
You can connect to machine using ssh command printed in the console after `Job ready for connection, run the following:`, or use helper utility which connects to the last instance that was launched.

```
export PATH=../:$PATH
connect
```

This will SSH into the instance and attach you to tmux session, and you should see something like this being printed:

```
36924.75 MB/s
36913.56 MB/s
36921.97 MB/s
36932.63 MB/s
36915.70 MB/s
36918.94 MB/s
36917.45 MB/s
36925.79 MB/s
36911.69 MB/s
36898.70 MB/s
36922.09 MB/s
36895.21 MB/s
36924.89 MB/s
36918.37 MB/s
36924.48 MB/s
```

# Troubleshooting

## Error "not supported in your requested Availability Zone"
`botocore.exceptions.ClientError: An error occurred (Unsupported) when calling the RunInstances operation: Your requested instance type (g3.4xlarge) is not supported in your requested Availability Zone (us-east-1f). Please retry your request by not specifying an Availability Zone or choosing us-east-1b, us-east-1a, us-east-1c, us-east-1e.`

Zones are randomized between users so one person's us-west-2c may not have any GPUs for another person.

Solution is to specify a different zone:

```
export ZONE=us-west-2a
./launch.py
```

## Error of the form "Availability zone us-east-1a must be in default region"

`AssertionError: Availability zone us-east-1a must be in default region us-west-1.`

Region is taken out of AWS_DEFAULT_REGION environment variable, fix it by setting that variable to match ZONE

```
export AWS_DEFAULT_REGION=us-east-1
```

## Other errors
The script calls "create_resources.py" to create various resources like VPC, subnets and EFS. If something went wrong with this process, you can change default name of resource to use and try again

```
export RESOURCE_NAME=debug
./launch.py
```

To clean-up resources created by this script, use `delete_resources.py` script from directory above. IE

```
export RESOURCE_NAME=debug
../delete_resources.py
```

or to delete default resources ("nexus" by default)

```
unset RESOURCE_NAME
../delete_resources.py
```
