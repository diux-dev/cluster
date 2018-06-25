# GPU box launcher

Steps: 1) setup Python 3.6 env 2) setup gpubox 3) setup AWS credentials 4) launch


## Step 1: Python 3.6

Follow official instructions for installing Anaconda on your machine, then create local Python 3.6 environment:

```
conda create --name gpubox -y python=3.6
source activate gpubox
```

You can also install things without Anaconda, in which case you need to run pip with `--user` flag, ie `pip install -r ../requirements --user`

## Step 2: gpubox package

```
git clone https://github.com/diux-dev/cluster.git
cd cluster/gpubox
pip install -r ../requirements.txt
```

## Step 3: AWS credentials

Get your public/private access key from AWS console and fill them in

```
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=KAa...
```

Now set your AWS region where you have resources. IE, for Oregon

```
export AWS_DEFAULT_REGION=us-west-2
```

## Step 4: 

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

You can configure to use different region, zone, instance type by setting proper environemnt variables, parameters, ie

```
export AWS_DEFAULT_REGION=us-east-1
launch.py --zone=us-east-1a --instance=p3.16xlarge
```

You can also provide custom AMI. When doing this, make sure that "linux-type" argument is set properly because that determines the username that is used to SSH into the instance to finish setup. Linux type `amazon` uses username `ec2-user` and linux type `ubuntu` uses username `ubuntu`.

```
export AWS_DEFAULT_REGION=us-east-1
launch.py --zone=us-east-1a --ami=ami-3a533040 --linux-type=amazon
```

# Troubleshooting

## Error "not supported in your requested Availability Zone"
`botocore.exceptions.ClientError: An error occurred (Unsupported) when calling the RunInstances operation: Your requested instance type (g3.4xlarge) is not supported in your requested Availability Zone (us-east-1f). Please retry your request by not specifying an Availability Zone or choosing us-east-1b, us-east-1a, us-east-1c, us-east-1e.`

Default zone used by script didn't contain instances of requested type. You need to specify zone manually, ie

```
./launch.py --zone=us-east-1a --name=memory
```

## Error of the form "Availability zone us-east-1a must be in default region"

`AssertionError: Availability zone us-east-1a must be in default region us-west-1.`

Region is taken out of AWS_DEFAULT_REGION environment variable, fix it by setting that variable to match your zone argument

```
export AWS_DEFAULT_REGION=us-east-1
```

## Other errors
The script calls "create_resources.py" to create various resources like VPC, subnets and EFS. If something went wrong with this process, you can change default name of resource to use and try again

```
export RESOURCE_NAME=gpubox
launch.py
```

To clean-up resources created by this script, use `delete_resources.py` script from directory above. IE

```
export RESOURCE_NAME=gpubox
../delete_resources.py
```

or to delete default resources (called "nexus")

```
unset RESOURCE_NAME
../delete_resources.py
```
