# Lite-HRNet: A Lightweight High-Resolution Network

## Introduction
```
@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}
```

## Results and models
### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | #Params | FLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> |
| :----------------- | :-----------: | :------: | :-----------: | :------: |:------: | :------: | :------: | :------: |
| [Lite-HRNet-18](/configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py)  | 256x192 | 1.1M | 205.2M |0.648 | 0.867 | 0.730 | 0.712 | 0.911 |
| [Lite-HRNet-18](/configs/top_down/lite_hrnet/coco/litehrnet_18_coco_384x288.py)  | 384x288 | 1.1M | 461.6M | 0.676 | 0.878 | 0.750 | 0.737 | 0.921 |
| [Lite-HRNet-30](/configs/top_down/lite_hrnet/coco/litehrnet_30_coco_256x192.py)  | 256x192 | 1.8M | 319.2M | 0.672 | 0.880 | 0.750 | 0.733 | 0.922 |
| [Lite-HRNet-30](/configs/top_down/lite_hrnet/coco/litehrnet_30_coco_384x288.py)  | 384x288 | 1.8M | 717.8M | 0.704 | 0.887 | 0.777 | 0.762 | 0.928 |


### Results on MPII val set.

| Arch  | Input Size | #Params | FLOPs | Mean | Mean@0.1   |
| :--- | :--------: | :------: | :--------: | :------: | :------: |
| [Lite-HRNet-18](/configs/top_down/lite_hrnet/mpii/litehrnet_18_mpii_256x256.py) | 256x256 | 1.1M | 273.4M | 0.854 | 0.295 |
| [Lite-HRNet-30](/configs/top_down/lite_hrnet/mpii/litehrnet_30_mpii_256x256.py) | 256x256 | 1.8M | 425.3M | 0.870 | 0.313 |
