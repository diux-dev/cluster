
# Benchmarking PyTorch MPI speed on AWS

To setup your environment (PATH, AWS keys), follow instructions for gpubox
https://github.com/diux-dev/cluster/tree/master/gpubox

Use launch_mpi_test.py to test MPI data exchange speed.

## Benchmark speed locally

```
python launch_mpi_test.py --run-local
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

## Benchmark speed on default instance

```
export AWS_DEFAULT_REGION=us-west-2
python launch_mpi_test.py
```
This allocates two default instance types (--instance argument), with names `0.mpi.mpi_test`, `1.mpi.mpi_test` and starts the benchmark on them. You can connect to one of the instances using `ssh` instructions printed. Look for something like this in console

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

## Benchmark speed on fast network/placement group

Using c5 instances you get access to faster bandwidth, using placement group guarantees 10 Gbps per stream, 25 Gbps total.

```
python launch_mpi_test.py --name=c5 --placement=1 --instance=c5.18xlarge
connect c5
Process 1 transferred 100 MB in 83.3 ms (1201.0 MB/sec)
Process 1 transferred 100 MB in 83.3 ms (1200.2 MB/sec)
Process 1 transferred 100 MB in 83.3 ms (1200.5 MB/sec)
Process 1 transferred 100 MB in 83.3 ms (1200.6 MB/sec)
Process 1 transferred 100 MB in 83.3 ms (1200.4 MB/sec)
```
