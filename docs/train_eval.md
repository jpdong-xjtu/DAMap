# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train DAMap with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/damap/damap_tiny_r50_24e.py 8
```

Eval MapTR with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/damap/damap_tiny_r50_24e.py ./path/to/ckpts.pth 8
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/DAMap/tools/maptrv2`