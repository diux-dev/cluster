# Recipes for rapid iteration
Assumes {GIT_ROOT}/cluster is in your $PATH



## Moving regions/zones

Following 9 zones have p3's at spot pricing:

- us-east-1a  $24.4800
- us-east-1c  $7.3440
- us-east-1d  $7.3440
- us-east-1f  $7.3440

- us-east-2a  $7.3440
- us-east-2b  $7.751

- us-west-2a  $15.4726
- us-west-2b  $8.0036
- us-west-2c  $8.0479

To create ImageNet high performance volumes use `replicate_imagenet.py` script. It creates volumes with names like imagenet_00, imagenet_01, ...
```
source go-west
python replicate_imagenet.py --zone=us-west-2b --replicas=8
```

Then to launch
```
python launch_nv.py --zone=us-west-2b --spot  --attach-volume imagenet
```

There's a region-wide limit on IOPS, so occasionally may need to undo this operation


```
python unreplicate_imagenet.py --zone=us-west-2b --replicas=8
```
Also use `ebs_tool.py` to see existing volumes

## Monitoring
```
cd ~/git0/cluster/pytorch
conda activate gpubox

Oregon monitoring 
source go-west
./launch_monitoring --zone=us-west-2c
Tensorboard will be at http://35.161.232.69:6006
Tensorboard selected will be at http://35.161.232.69:6007
Jupyter notebook will be at http://35.161.232.69:8888

Virginia monitoring:
source go-east
./launch_monitoring --zone=us-east-2c
Tensorboard will be at http://52.202.163.26:6006
Tensorboard selected will be at http://52.202.163.26:6007
Jupyter notebook will be at http://52.202.163.26:8888

Ohio Monitoring
source go-ohio
./launch_monitoring --zone=us-east-2a
Tensorboard will be at http://18.188.60.21:6006
Tensorboard selected will be at http://18.188.60.21:6007
Jupyter notebook will be at http://18.188.60.21:8888
```

## Testing changes setup scripts like setup_nv.sh

Use `uninitialize` script to remove initialization markers like `/tmp/is_initialized` and `/tmp/nv_setup_complete` that will force various first-time setup scripts to run

# Helper scripts

- pytorch/launch_monitoring.py to launch TensorBoard/Jupyter in region
- aws_tool.py to list your running jobs
- `connect <jobsubstring>` and `connect2 <jobsubstring>`
   The first one (connect) looks up instance IP, ssh's and does "tmux a"
   The second one (connect2) skips "tmux a" step, useful for connecting to tasks that have multiple tmux sessions running, like monitoring
- ebs_tool.py to list all volumes and their attachments
  - `ebs_tool.py grow myvol` grows volume myvol to 500 GB
- efs_tool.py list all EFS systems

- spot_tool.py list spot requests. Run `spot_tool.py cancel` to cancel outstanding requests
- terminate:
  - `terminate myjob` kills all instances
  - `terminate myjob -y` skips confirmation
  - `terminate myjob -d 3600 -y` kill myjob after 3600 second delay

## Using local conda on ~/data volume instead of root
Implemented using `use-local-conda` flag in `launch_nv.py`, but only if you attach to imagenet_high_perf_0 volume right now.

- https://conda.io/docs/user-guide/tasks/manage-environments.html

To use this, you must provide conda env on the volume that's mounted under data. Using fresh volume, and DLAMI image, you can do this:
```
source activate <some conda env> # enable conda command, 4.4 or later
conda create -p ~/data/anaconda3/envs/pytorch_p36 --clone pytorch_p36
conda activate /home/ubuntu/data/anaconda3/envs/pytorch_p36
```

# Notes

### ImageNet data snapshots
- us-east-1 snap-0d37b903e01bb794a imagenet_blank snapshot
- us-east-2 snap-0aa785c2cf2a16887 imagenet_blank snapshot [Copied snap-0d37b903e01bb794a
- us-west-2 snap-02a02dbb22d521968 imagenet_blank snapshot [Copied snap-0d37b903e01bb794a
