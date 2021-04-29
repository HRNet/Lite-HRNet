# Lite-HRNet: A Lightweight High-Resolution Network

## Introduction
This is an official pytorch implementation of [Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403). In this work, we present an efficient high-resolution network, Lite-HRNet, for human pose estimation. We start by simply applying the efficient shuffle block in ShuffleNet to HRNet (high-resolution network), yielding stronger performance over popular lightweight networks, such as MobileNet, ShuffleNet, and Small HRNet. We find that the heavily-used pointwise (1x1) convolutions in shuffle blocks become the computational bottleneck. We introduce a lightweight unit, conditional channel weighting, to replace costly pointwise (1x1) convolutions in shuffle blocks. The complexity of channel weighting is linear w.r.t the number of channels and lower than the quadratic time complexity for pointwise convolutions. Our solution learns the weights from all the channels and over multiple resolutions that are readily available in the parallel branches in HRNet. It uses the weights as the bridge to exchange information across channels and resolutions, compensating the role played by the pointwise (1x1) convolution. Lite-HRNet demonstrates superior results on human pose estimation over popular lightweight networks. Moreover, Lite-HRNet can be easily applied to semantic segmentation task in the same lightweight manner.

<img width="512" height="512" src="/resources/litehrnet_block.png"/>

## Results and models

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | #Params | FLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt |
| :----------------- | :-----------: | :------: | :-----------: | :------: |:------: | :------: | :------: | :------: | :------: |
| [Naive Lite-HRNet-18](/configs/top_down/naive_litehrnet/coco/naive_litehrnet_18_coco_256x192.py)  | 256x192 | 0.7M | 194.8M | 0.628 | 0.855 | 0.699 | 0.691 | 0.901 | [GoogleDrive](https://drive.google.com/file/d/1rW1gqKtCUxpsos-eLItEHZThEF48eRjQ/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHnfki3iJZiE-v0dqE?e=42PlOT)|
| [Wider Naive Lite-HRNet-18](/configs/top_down/naive_litehrnet/coco/wider_naive_litehrnet_18_coco_256x192.py)  | 256x192 | 1.3M | 311.1M | 0.660 | 0.871 | 0.737 | 0.721 | 0.913 | [GoogleDrive](https://drive.google.com/file/d/1Amb0yE677zV18KaH6gruUQnHUiwBx_-H/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHngQLsLGlp3r16ALfZ?e=XgMYMd) |
| [Lite-HRNet-18](/configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py)  | 256x192 | 1.1M | 205.2M |0.648 | 0.867 | 0.730 | 0.712 | 0.911 | [GoogleDrive](https://drive.google.com/file/d/1ZewlvpncTvahbqcCFb-95C3NHet30mk5/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHngQE0r-EVnMNPObk7?e=ojJosi) |
| [Lite-HRNet-18](/configs/top_down/lite_hrnet/coco/litehrnet_18_coco_384x288.py)  | 384x288 | 1.1M | 461.6M | 0.676 | 0.878 | 0.750 | 0.737 | 0.921 | [GoogleDrive](https://drive.google.com/file/d/1E3S18YbUfBm7YtxYOV7I9FmrntnlFKCp/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHnfZE0w9s_h9oK98c?e=xPiAxS) |
| [Lite-HRNet-30](/configs/top_down/lite_hrnet/coco/litehrnet_30_coco_256x192.py)  | 256x192 | 1.8M | 319.2M | 0.672 | 0.880 | 0.750 | 0.733 | 0.922 | [GoogleDrive](https://drive.google.com/file/d/1KLjNInzFfmZWSbEQwx-zbyaBiLB7SnEj/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHnexxp5RCK15meEWw?e=g4ObHb) |
| [Lite-HRNet-30](/configs/top_down/lite_hrnet/coco/litehrnet_30_coco_384x288.py)  | 384x288 | 1.8M | 717.8M | 0.704 | 0.887 | 0.777 | 0.762 | 0.928 | [GoogleDrive](https://drive.google.com/file/d/1BcHnLka4FWiXRmPnJgJKmsSuXXqN4dgn/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHnfOng41YajWZg478?e=wVKeIS) |

### Results on MPII val set

| Arch  | Input Size | #Params | FLOPs | Mean | Mean@0.1   | ckpt |
| :--- | :--------: | :------: | :--------: | :------: | :------: | :------: |
| [Naive Lite-HRNet-18](/configs/top_down/naive_litehrnet/mpii/naive_litehrnet_18_mpii_256x256.py) | 256x256 | 0.7M | 259.6M | 0.853 | 0.305 | [GoogleDrive](https://drive.google.com/file/d/1tUdrJd_SI5HGBuYT9FxYX5XFhfcb3cD_/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHngQAjPfwag38olIcH?e=01kH1t) |
| [Wider Naive Lite-HRNet-18](/configs/top_down/naive_litehrnet/mpii/wider_naive_litehrnet_18_mpii_256x256.py) | 256x256 | 1.3M | 418.7M | 0.868 | 0.311 | [GoogleDrive](https://drive.google.com/file/d/12cPeB8MZs1o6_qOS7HavGwkk6r7hUMAS/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHngQMqhUOcrYtjTDZ1?e=iuD9Jz) |
| [Lite-HRNet-18](/configs/top_down/lite_hrnet/mpii/litehrnet_18_mpii_256x256.py) | 256x256 | 1.1M | 273.4M | 0.854 | 0.295 | [GoogleDrive](https://drive.google.com/file/d/1bcnn5Ic2-FiSNqYOqLd1mOfQchAz_oCf/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHnev2he_nA_VOLSqg?e=f9zACb) |
| [Lite-HRNet-30](/configs/top_down/lite_hrnet/mpii/litehrnet_30_mpii_256x256.py) | 256x256 | 1.8M | 425.3M | 0.870 | 0.313 | [GoogleDrive](https://drive.google.com/file/d/1JB9LOwkuz5OUtry0IQqXammFuCrGvlEd/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AvreNzlRJaHnf0LR6jpyGoTJZIA?e=653jEF) |


## Enviroment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 8 NVIDIA V100 GPU cards. Other platforms or GPU cards are not fully tested.
## Quick Start

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) (Please install the latest version of mmcv-full)
- Numpy
- cv2
- json_tricks
- [xtcocotools](https://github.com/jin-s13/xtcocoapi)


### Installation
<!-- The code is based on [MMPose](https://github.com/open-mmlab/mmpose).
You need clone the mmpose project and integrate the codes into mmpose first. -->

a. Install mmcv, we recommend you to install the pre-build mmcv as below.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired one. For example, to install the latest ``mmcv-full`` with ``CUDA 11`` and ``PyTorch 1.7.0``, use the following command:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

If it compiles during installation, then please check that the cuda version and pytorch version **exactly"" matches the version in the mmcv-full installation command. For example, pytorch 1.7.0 and 1.7.1 are treated differently.
See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

Optionally you can choose to compile mmcv from source by the following command

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

Or directly run

```shell
pip install mmcv-full
# alternative: pip install mmcv
```

**Important:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

b. Install build requirements

```shell
pip install -r requirements.txt
```

### Prepare datasets

It is recommended to symlink the dataset root to `$LITE_HRNET/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-)
Download and extract them under `$LITE_HRNET/data`, and make them look like this:

```
lite_hrnet
├── configs
├── models
├── tools
`── data
    │── coco
        │-- annotations
        │   │-- person_keypoints_train2017.json
        │   |-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        │-- train2017
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        `-- val2017
            │-- 000000000139.jpg
            │-- 000000000285.jpg
            │-- 000000000632.jpg
            │-- ...

```

**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).
We have converted the original annotation files into json format, please download them from [mpii_annotations](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/datasets/mpii_annotations.tar).
Extract them under `$LITE_HRNET/data`, and make them look like this:

```
lite_hrnet
├── configs
├── models
├── tools
`── data
    │── mpii
        |── annotations
        |   |── mpii_gt_val.mat
        |   |── mpii_test.json
        |   |── mpii_train.json
        |   |── mpii_trainval.json
        |   `── mpii_val.json
        `── images
            |── 000001163.jpg
            |── 000003072.jpg

```

## Training and Testing
All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by modifying the interval argument in the training config

```python
evaluation = dict(interval=5)  # This evaluate the model per 5 epoch.
```

According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or videos per GPU, e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.

### Training

```shell
# train with a signle GPU
python tools/train.py ${CONFIG_FILE} [optional arguments]

# train with multiple GPUs
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 5 epochs during the training.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--gpus ${GPU_NUM}`: Number of gpus to use, which is only applicable to non-distributed training.
- `--seed ${SEED}`: Seed id for random state in python, numpy and pytorch to generate random numbers.
- `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.
- `--autoscale-lr`: If specified, it will automatically scale lr with the number of gpus by [Linear Scaling Rule](https://arxiv.org/abs/1706.02677).

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

Examples:

#### Training on COCO train2017 dataset
```shell
./tools/dist_train.sh configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py 8
```

#### Training on MPII dataset

```shell
./tools/dist_train.sh configs/top_down/lite_hrnet/mpii/litehrnet_18_mpii_256x256.py 8
```

### Testing
You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRIC}] \
    [--proc_per_gpu ${NUM_PROC_PER_GPU}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--average_clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]

# multiple-gpu testing
./tools/dist_test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRIC}] \
    [--proc_per_gpu ${NUM_PROC_PER_GPU}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--average_clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]
```

Optional arguments:

- `RESULT_FILE`: Filename of the output results. If not specified, the results will not be saved to a file.
- `EVAL_METRIC`: Items to be evaluated on the results. Allowed values depend on the dataset.
- `NUM_PROC_PER_GPU`: Number of processes per GPU. If not specified, only one process will be assigned for a single gpu.
- `--gpu_collect`: If specified, recognition results will be collected using gpu communication. Otherwise, it will save the results on different gpus to `TMPDIR` and collect them by the rank 0 worker.
- `TMPDIR`: Temporary directory used for collecting results from multiple workers, available when `--gpu_collect` is not specified.
- `AVG_TYPE`: Items to average the test clips. If set to `prob`, it will apply softmax before averaging the clip scores. Otherwise, it will directly average the clip scores.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.

Examples:
#### Test LiteHRNet-18 on COCO with 8 GPUS, and evaluate the mAP.

```shell
./tools/dist_test.sh configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py \
    checkpoints/SOME_CHECKPOINT.pth 8 \
    --eval mAP
```

### Get the compulationaly complexity
You can use the following commands to compute the complexity of one model.
```shell
python tools/summary_network.py ${CONFIG_FILE} --shape ${SHAPE}
```

Arguments:

- `SHAPE`: Input size.

Examples:

#### Test the complexity of LiteHRNet-18 with 256x256 resolution input.

```shell
python tools/summary_network.py configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py \
    --shape 256 256 \
```

## Acknowledgement

Thanks to:

- [MMPose](https://github.com/open-mmlab/mmpose)
- [HRNet](https://github.com/HRNet/deep-high-resolution-net.pytorch)

## Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI}
  year={2019}
}

```
