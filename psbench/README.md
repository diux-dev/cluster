Parameter server benchmarks for TensorFlow and Ray

```
python launch_tf.py --workers=1 --ps=2
python launch_ray.py --workers=1 --ps=2

# launch on AWS using c5.large instances
python launch_ray.py --workers=1 --ps=2 --backend=aws
python launch_tf.py --workers=1 --ps=2 --backend=aws
```