
# Benchmarking PyTorch MPI speed on AWS

To setup your environment (PATH, AWS keys), follow instructions for gpubox
https://github.com/diux-dev/cluster/tree/master/gpubox

Use bench_p2p.py to test MPI data exchange speed.

## Benchmark p2p speed locally

```
python bench_p2p.py
```
This creates tmux session with default name mpi_test (--name argument) and starts the exchange. To attach and see the progress

```
tmux a -t mpi_test
...
Process 1 transferred 100 MB in 23.1 ms (4323.8 MB/sec)
Process 1 transferred 100 MB in 18.6 ms (5387.0 MB/sec)
Process 1 transferred 100 MB in 21.5 ms (4648.1 MB/sec)
Process 1 transferred 100 MB in 23.2 ms (4312.5 MB/sec)
```

## Benchmark p2p speed on default instance

```
export AWS_DEFAULT_REGION=us-west-2
python bench_p2p.py --zone=us-west-2c
```
This allocates two default instance types (--instance-type argument), with names `0.mpi.mpi_test`, `1.mpi.mpi_test` and starts the benchmark on them. You can connect to one of the instances using `ssh` instructions printed. Look for something like this in console

```
Job ready for connection, to connect to most recent task, run the following:
../connect mpi_test
Alternatively run
ssh -i /Users/yaroslav/nexus-yaroslav-us-west-2.pem -o StrictHostKeyChecking=no ubuntu@34.219.81.100
tmux a
```

With t2.large default instances you may see the following

```
Process 0 transferred 100 MB in 854.4 ms (117.0 MB/sec)
Process 0 transferred 100 MB in 860.0 ms (116.3 MB/sec)
Process 0 transferred 100 MB in 873.2 ms (114.5 MB/sec)
Process 0 transferred 100 MB in 859.9 ms (116.3 MB/sec)
```

## Benchmark p2p speed on fast network/placement group

Using c5 instances you get access to faster bandwidth, using placement group guarantees 10 Gbps per stream, 25 Gbps total.

```
python bench_p2p.py --name=c5 --placement --instance-type=c5.18xlarge --zone=us-west-2c
connect c5
Process 1 transferred 100 MB in 83.3 ms (1201.0 MB/sec)
Process 1 transferred 100 MB in 83.3 ms (1200.2 MB/sec)
Process 1 transferred 100 MB in 83.3 ms (1200.5 MB/sec)
Process 1 transferred 100 MB in 83.3 ms (1200.6 MB/sec)
Process 1 transferred 100 MB in 83.3 ms (1200.4 MB/sec)
```

## Benchmark all-reduce speed

Similar steps can be used to benchmark all-reduce speed, using bench_allreduce.py script

```
python bench_allreduce.py --instance-type=p3.16xlarge --zone=us-east-1c --placement

...

Process 0 transferred 100 MB in 367.9 ms (271.8 MB/sec)
Rank  0  has data  tensor(1990.)
Process 0 transferred 100 MB in 168.3 ms (594.3 MB/sec)
Rank  0  has data  tensor(1992.)
Process 0 transferred 100 MB in 190.9 ms (524.0 MB/sec)
Rank  0  has data  tensor(1994.)
Process 0 transferred 100 MB in 172.4 ms (580.0 MB/sec)
Rank  0  has data  tensor(1996.)
Process 0 transferred 100 MB in 161.8 ms (618.1 MB/sec)
Rank  0  has data  tensor(1998.)
Process 0 transferred 100 MB in 348.9 ms (286.6 MB/sec)
```

## Benchmarks

```
# c5 2-worker allreduce:
python bench_allreduce.py --instance-type=c5.18xlarge --zone=us-east-1c --placement --backend=gloo --name=allreduce

Process 0 transferred 100 MB in 108.6 ms (921.0 MB/sec)
Process 0 transferred 100 MB in 111.0 ms (900.8 MB/sec)
Process 0 transferred 100 MB in 107.4 ms (930.9 MB/sec)
Process 0 transferred 100 MB in 108.5 ms (921.6 MB/sec)
Process 0 transferred 100 MB in 109.0 ms (917.5 MB/sec)
Process 0 transferred 100 MB in 111.2 ms (899.7 MB/sec)

# c5 8-worker allreduce:
python bench_allreduce.py --instance-type=c5.18xlarge --zone=us-east-1c --placement --backend=gloo --name=eight --num-machines=8

Process 0 transferred 100 MB in 346.4 ms (288.7 MB/sec)
Process 0 transferred 100 MB in 344.9 ms (290.0 MB/sec)
Process 0 transferred 100 MB in 459.5 ms (217.6 MB/sec)
Process 0 transferred 100 MB in 568.6 ms (175.9 MB/sec)
Process 0 transferred 100 MB in 565.7 ms (176.8 MB/sec)
Process 0 transferred 100 MB in 387.5 ms (258.1 MB/sec)
Process 0 transferred 100 MB in 350.9 ms (285.0 MB/sec)

# c5 16-worker allreduce:
python bench_allreduce.py --instance-type=c5.18xlarge --zone=us-east-1c --placement --backend=gloo --name=sixteen --num-machines=16

Process 0 transferred 100 MB in 627.7 ms (159.3 MB/sec)
Process 0 transferred 100 MB in 672.6 ms (148.7 MB/sec)
Process 0 transferred 100 MB in 685.2 ms (146.0 MB/sec)
Process 0 transferred 100 MB in 464.2 ms (215.4 MB/sec)
Process 0 transferred 100 MB in 673.3 ms (148.5 MB/sec)
Process 0 transferred 100 MB in 589.9 ms (169.5 MB/sec)
Process 0 transferred 100 MB in 609.9 ms (164.0 MB/sec)
Process 0 transferred 100 MB in 775.1 ms (129.0 MB/sec)
Process 0 transferred 100 MB in 482.1 ms (207.4 MB/sec)
Process 0 transferred 100 MB in 688.9 ms (145.2 MB/sec)
Process 0 transferred 100 MB in 673.2 ms (148.5 MB/sec)
Process 0 transferred 100 MB in 551.7 ms (181.3 MB/sec)
Process 0 transferred 100 MB in 470.2 ms (212.7 MB/sec)
Process 0 transferred 100 MB in 471.4 ms (212.1 MB/sec)

# c5 with Amazon Linux and 2 machines
python bench_allreduce.py --instance-type=c5.18xlarge --zone=us-east-1c --placement --backend=gloo --linux-type=amazon --name=c5amazon00

Process 1 transferred 100 MB in 118.6 ms (842.9 MB/sec)
Process 1 transferred 100 MB in 118.8 ms (841.5 MB/sec)
Process 1 transferred 100 MB in 120.4 ms (830.5 MB/sec)
Process 1 transferred 100 MB in 115.9 ms (862.5 MB/sec)
Process 1 transferred 100 MB in 119.4 ms (837.5 MB/sec)
Process 1 transferred 100 MB in 119.4 ms (837.4 MB/sec)
Process 1 transferred 100 MB in 119.3 ms (838.3 MB/sec)
Process 1 transferred 100 MB in 119.5 ms (837.0 MB/sec)
Process 1 transferred 100 MB in 119.0 ms (840.5 MB/sec)
Process 1 transferred 100 MB in 119.5 ms (836.8 MB/sec)
Process 1 transferred 100 MB in 119.2 ms (839.1 MB/sec)
Process 1 transferred 100 MB in 119.0 ms (840.2 MB/sec)
Process 1 transferred 100 MB in 118.8 ms (841.7 MB/sec)
Process 1 transferred 100 MB in 118.5 ms (843.6 MB/sec)
Process 1 transferred 100 MB in 120.6 ms (829.2 MB/sec)
Process 1 transferred 100 MB in 117.4 ms (852.0 MB/sec)

# c5 run with Amazon Linux and  16 machines
python bench_allreduce.py --instance-type=c5.18xlarge --zone=us-east-1c --placement --backend=gloo --linux-type=amazon --name=c5amazon01  --num-machines=16

Process 12 transferred 100 MB in 440.8 ms (226.8 MB/sec)
Process 12 transferred 100 MB in 606.5 ms (164.9 MB/sec)
Process 12 transferred 100 MB in 445.6 ms (224.4 MB/sec)
Process 12 transferred 100 MB in 453.8 ms (220.4 MB/sec)
Process 12 transferred 100 MB in 600.0 ms (166.7 MB/sec)
Process 12 transferred 100 MB in 453.5 ms (220.5 MB/sec)
Process 12 transferred 100 MB in 629.6 ms (158.8 MB/sec)
Process 12 transferred 100 MB in 443.5 ms (225.5 MB/sec)
Process 12 transferred 100 MB in 437.2 ms (228.7 MB/sec)
Process 12 transferred 100 MB in 448.3 ms (223.1 MB/sec)
Process 12 transferred 100 MB in 426.6 ms (234.4 MB/sec)
Process 12 transferred 100 MB in 437.8 ms (228.4 MB/sec)
Process 12 transferred 100 MB in 440.4 ms (227.1 MB/sec)
Process 12 transferred 100 MB in 613.9 ms (162.9 MB/sec)
Process 12 transferred 100 MB in 430.7 ms (232.2 MB/sec)
Process 12 transferred 100 MB in 429.9 ms (232.6 MB/sec)
Process 12 transferred 100 MB in 455.1 ms (219.7 MB/sec)
Process 12 transferred 100 MB in 456.0 ms (219.3 MB/sec)
Process 12 transferred 100 MB in 425.0 ms (235.3 MB/sec)
Process 12 transferred 100 MB in 617.9 ms (161.8 MB/sec)
Process 12 transferred 100 MB in 444.1 ms (225.2 MB/sec)
Process 12 transferred 100 MB in 434.5 ms (230.2 MB/sec)
Process 12 transferred 100 MB in 436.4 ms (229.1 MB/sec)
Process 12 transferred 100 MB in 440.2 ms (227.2 MB/sec)
Process 12 transferred 100 MB in 428.6 ms (233.3 MB/sec)
Process 12 transferred 100 MB in 434.1 ms (230.4 MB/sec)
Process 12 transferred 100 MB in 823.8 ms (121.4 MB/sec)
Process 12 transferred 100 MB in 461.0 ms (216.9 MB/sec)


# p3 2-worker allreduce:
python bench_allreduce.py --instance-type=p3.16xlarge --zone=us-east-1c --placement --backend=gloo --name=gpu

Process 0 transferred 100 MB in 356.6 ms (280.4 MB/sec)
Process 0 transferred 100 MB in 183.7 ms (544.5 MB/sec)
Process 0 transferred 100 MB in 594.4 ms (168.2 MB/sec)
Process 0 transferred 100 MB in 182.0 ms (549.3 MB/sec)
Process 0 transferred 100 MB in 369.0 ms (271.0 MB/sec)
Process 0 transferred 100 MB in 155.3 ms (644.0 MB/sec)
Process 0 transferred 100 MB in 146.1 ms (684.4 MB/sec)
Process 0 transferred 100 MB in 339.2 ms (294.8 MB/sec)
Process 0 transferred 100 MB in 185.9 ms (538.0 MB/sec)

# p3 2-worker p2p:
python bench_p2p.py --instance-type=p3.16xlarge --zone=us-east-1c --placement --backend=gloo --name=p2pgpu

Process 1 transferred 100 MB in 90.5 ms (1104.5 MB/sec)
Process 1 transferred 100 MB in 89.6 ms (1116.4 MB/sec)
Process 1 transferred 100 MB in 90.3 ms (1107.5 MB/sec)
Process 1 transferred 100 MB in 87.9 ms (1137.3 MB/sec)
Process 1 transferred 100 MB in 87.5 ms (1143.0 MB/sec)
Process 1 transferred 100 MB in 88.5 ms (1130.1 MB/sec)
Process 1 transferred 100 MB in 89.0 ms (1123.7 MB/sec)
Process 1 transferred 100 MB in 89.2 ms (1120.5 MB/sec)


```