# Recipes for rapid iteration


## Using local conda on ~/data volume instead of root

To use this, you must provide conda env on the volume that's mounted under data. Using fresh volume, and DLAMI image, you can do this:
```
source activate <some conda env> # enable conda command, 4.4 or later
conda create -p ~/data/anaconda3/envs/pytorch_p36 --clone pytorch_p36
conda activate /home/ubuntu/data/anaconda3/envs/pytorch_p36
```

### Notes:
- https://conda.io/docs/user-guide/tasks/manage-environments.html
- Simply "cp -R" on env dir doesn't work: https://github.com/ContinuumIO/anaconda-issues/issues/9833



# Testing setup changes

Use `uninitialize` script to remove initialization markers like `/tmp/is_initialized` and `/tmp/nv_setup_complete` that will force various first-time setup scripts to run
