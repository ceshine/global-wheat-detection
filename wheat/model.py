import math
from pathlib import Path
from typing import Tuple, Union, Dict, List

import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from effdet import get_efficientdet_config, EfficientDet
from effdet.bench import DetBenchTrain, DetBenchPredict
from effdet.helpers import load_pretrained
from effdet.data.loader import DetectionFastCollate

import pytorch_lightning_spells as pls
from pytorch_lightning_spells import BaseModule
from torch.utils.data import DataLoader

from .dataset import WheatDataset, get_train_transforms, Mode
from .metrics import mAP

TARGET_TYPE = Dict[str, torch.Tensor]


class DetectionModule(BaseModule):
    def validation_step_end(self, outputs):
        """This method logs the validation loss and metrics for you.

        The output from `.validation_step()` method must contains these three entries:

            1. loss: the validation loss.
            2. pred: the predicted labels or values.
            3. target: the ground truth lables or values.

        Args:
            outputs (Dict): the output from `.validation_step()` method.
        """
        self.log("val_loss", outputs["loss"].mean())
        for name, metric in self.metrics:
            metric(outputs["detections"], outputs["targets"])
            self.log("val_" + name, metric)


def get_train_efficientdet(
    model_name: str = "tf_efficientdet_d5",
    image_size: Tuple[int, int] = (384, 384),
    mode: str = "train",
    pretrained: bool = True,
):
    config = get_efficientdet_config(model_name)
    config.image_size = image_size
    net = EfficientDet(config, pretrained_backbone=False)
    # load pretrained
    if pretrained:
        load_pretrained(net, config.url)
    net.reset_head(num_classes=1)
    if mode == "train":
        return DetBenchTrain(net, create_labeler=True)
    else:
        return DetBenchPredict(net)


def collate_fn(batch):
    return tuple(zip(*batch))


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class PrefetchLoader:
    def __init__(
        self,
        loader,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        half: bool = False,
    ):
        self.half = half
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_input = next_input.float().sub_(self.mean).div_(self.std)
                next_target = {
                    k: v.cuda(non_blocking=True) for k, v in next_target.items()
                }
                if self.half:
                    next_input = next_input.half()

            if not first:
                yield [input, target]
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield [input, target]

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class WheatModel(DetectionModule):
    def __init__(
        self,
        config: DictConfig,
        df: pd.DataFrame,
        fold: int,
        half: bool = False,
    ):
        super().__init__()
        self.df = df
        self.config = config
        self.train_df = self.df.loc[lambda df: df["fold"] != fold]
        self.valid_df = self.df.loc[lambda df: df["fold"] == fold]
        self.image_dir = str(Path(config.base_dir) / "train")
        self.model = get_train_efficientdet(
            config.arch, image_size=(config.image_size, config.image_size)
        )
        self.min_box_edge = 10 / (1024 / config.image_size)
        self.num_workers = config.num_workers
        self.batch_size = config.batch_size
        self.metrics = [
            (
                "MAP",
                mAP(
                    thresholds=np.arange(0.5, 0.76, 0.05),
                    form="pascal_voc",
                    confidence_threshold=0.4,
                ),
            )
            # ("acc", RetrievalMAP(compute_on_step=False)),
        ]
        train_transforms = get_train_transforms(config.image_size, cutout=config.cutout)
        self.train_dataset = WheatDataset(
            df=self.train_df,
            image_dir=self.image_dir,
            transforms=train_transforms,
            min_box_edge=self.min_box_edge,
            mode=Mode.train,
            mosaic_p=self.config.mosaic_p,
        )
        # valid_transforms = get_valid_transforms()
        self.valid_dataset = WheatDataset(
            df=self.valid_df,
            image_dir=self.image_dir,
            transforms=None,
            mode=Mode.validation,
        )
        self.grad_accu = config.grad_accu
        self.epochs = config.epochs
        self.half = half
        print("# of train images:", len(self.train_dataset))
        print("# of valid images:", len(self.valid_dataset))

    def forward(self, image, target):
        return self.model(image, target)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        losses_dict = self.forward(images, targets)

        return {
            "loss": losses_dict["loss"],
            "log": batch_idx % self.trainer.accumulate_grad_batches == 0,
        }

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        losses_dict = self.model(images, targets)
        loss_val = losses_dict["loss"]
        detections = losses_dict["detections"]
        # Back to xyxy form
        bbox = targets["bbox"][:, :, [1, 0, 3, 2]]
        return {"loss": loss_val, "detections": detections, "targets": bbox}

    def configure_optimizers(self):
        steps_per_epochs = math.floor(
            len(self.train_dataset)
            / self.batch_size
            / self.grad_accu  # / self.num_gpus # dpp mode
        )
        print("Steps per epochs:", steps_per_epochs)
        n_steps = steps_per_epochs * self.epochs
        lr_durations = [int(n_steps * 0.05), int(np.ceil(n_steps * 0.95)) + 1]
        break_points = [0] + list(np.cumsum(lr_durations))[:-1]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.base_lr)
        scheduler = {
            "scheduler": pls.lr_schedulers.MultiStageScheduler(
                [
                    pls.lr_schedulers.LinearLR(optimizer, 0.01, lr_durations[0]),
                    pls.lr_schedulers.CosineAnnealingLR(optimizer, lr_durations[1]),
                ],
                start_at_epochs=break_points,
            ),
            "interval": "step",
            "frequency": 1,
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=DetectionFastCollate(anchor_labeler=None),
            num_workers=self.num_workers,
        )
        return PrefetchLoader(loader, half=self.half)

    def val_dataloader(self):
        valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=False,
            collate_fn=DetectionFastCollate(anchor_labeler=None),
            num_workers=self.num_workers,
        )

        # iou_types = ["bbox"]

        return PrefetchLoader(valid_dataloader, half=self.half)
