# SCorP MIUA2024
[SCorP: Statistics-Informed Dense Correspondence Prediction Directly from Unsegmented Medical Images](https://arxiv.org/abs/2404.17967)


To run training and inference 
```
python run_train.py --config configs/liver.json
```


To run with point cloud data, replace `idx` to `None` in the `run_train.py`


Data organization: 
```
train
	- meshes 
	- images 
val
	- meshes
	- images 
test 
	- meshes
	- images 
mean.particles  
```

