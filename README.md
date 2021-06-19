# PyTorch EfficientDet Solution for Global Wheat Detection Challenge

(Documentation is still WIP)

## Requirements

1. torch>=1.7.0
1. pytorch-lightning>=1.3.6
1. pytorch-lightning-spells==0.0.3
1. efficientdet-pytorch==0.2.4

Note: You'll need to use [my fork of efficientdet-pytorch](https://github.com/ceshine/efficientdet-pytorch) to use the O2 level of Apex AMP.

## Instructions

Resizing images:

```bash
python scripts/resize_images.py 512 --root data/
```

Training (pass `--help` for more information):

```bash
python -m wheat.train data/512 --epochs 10 --grad-accu 4 --batch-size 8 --arch tf_efficientdet_d3 --fold 0 --mixup 24 --mosaic-p 0.5
```

Evaluation (pass `--help` for more information):

```bash
python -m wheat.eval data/512 export/tf_efficientdet_d3-mosaic-mixup-fold0.pth --batch-size 8 --arch tf_efficientdet_d3 --fold 0
```
