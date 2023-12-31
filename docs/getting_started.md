# Getting Started

This page provides basic usage based MMdetection. For installation instructions, please see [install.md](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/install.md)

# Inferenece with pretrained models
We provide the testing scripts to evaluate the trained models.

Examples:

Assume that you have already downloaded the checkpoints to `work_dirs/orientedreppoint_r50_demo/`.

1. Test OrientedRepPoints with single GPU.

```shell
python tools/test.py configs/dota/orientedrepoints_r50_demo.py \
    work_dirs/orientedreppoints_r50_demo/epoch_40.pth \ 
    --out work_dirs/orientedreppoints_r50_demo/results.pkl

```
2. Test OrientedRepPoints with 4 GPUs.
```shell
./tools/dist_test.sh configs/dota/orientedrepoints_r50_demo.py \
    work_dirs/orientedreppoints_r50_demo/epoch_40.pth 4 \ 
    --out work_dirs/orientedreppoints_r50_demo/results.pkl
```



*If you want to evaluate the result on DOTA test-dev, please read the results.pkl, and run mergs the txt results. and zip the files  and submit it to the  [evaluation server](https://captain-whu.github.io/DOTA/index.html).
*The code for parsing the .pkl file and merging the results for DOTA, please refer to [this](https://github.com/LiWentomng/OrientedRepPoints/blob/main/tools/parse_pkl/parse_pkl_mege_results_for_dota_evaluation.py).

To evaluate on the val set with ground-truth annotations, please refer to [DOTA_devkit/dota_evaluation_task1.py](https://github.com/LiWentomng/OrientedRepPoints/blob/main/DOTA_devkit/dota_evaluation_task1.py) for mAP,  and [DOTA_devkit/mAOE_evaluation.py](https://github.com/LiWentomng/OrientedRepPoints/blob/main/DOTA_devkit/mAOE_evaluation.py) for mAOE.

# Train a model

MMDetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

1. Train  with a single GPU 

```shell
python tools/train.py ${CONFIG_FILE} 

example:
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/dota/orientedrepoints_r50_demo.py 
```

2. Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]

example:
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/dota/orientedrepoints_r50_demo.py 4
```


