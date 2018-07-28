# Recipes for rapid iteration


## Using conda on ~/data volume instead of root

```
# Using 4.4 or later, you can do the following
conda create -p ~/data/anaconda3/envs/pytorch_p36 --clone pytorch_p36
conda activate /home/ubuntu/data/anaconda3/envs/pytorch_p36
```

### Notes:
- https://conda.io/docs/user-guide/tasks/manage-environments.html
- Simply "cp -R" on env dir doesn't work: https://github.com/ContinuumIO/anaconda-issues/issues/9833



# Testing setup changes

Use `uninitialize` script to remove initialization markers like `/tmp/is_initialized` and `/tmp/nv_setup_complete` that will force various first-time setup scripts to run
