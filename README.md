# PyTorch EfficientDet Solution for Global Wheat Detection Challenge

1. [Training notebook (on Kaggle)](https://www.kaggle.com/ceshine/wheat-detection-training-efficientdet-public?scriptVersionId=67789208&select=wheatdet.pth)
2. Inference notebook: [Single model](https://www.kaggle.com/ceshine/effdet-wheat-head-detection-inference-public?scriptVersionId=67809685); [Ensemble](https://www.kaggle.com/ceshine/effdet-wheat-head-detection-inference-public/output?scriptVersionId=67812782)

See [wheat/config.py](wheat/config.py) for hyper-parameters and system configurations.

The best mAP score I'm able to get is 0.6167 (Private) / 0.7084 (Public) with a D4 model trained on 768x768 resolution (using a single P100 GPU).

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
