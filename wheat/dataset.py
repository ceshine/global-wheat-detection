import enum
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A


class Mode(enum.Enum):
    train = 0
    validation = 1
    test = 2


class WheatDataset(Dataset):
    """
    [reference](https://www.kaggle.com/dangnam739/faster-rcnn-global-wheat-detection)
    """

    def __init__(
        self,
        image_dir: str,
        df=None,
        mode: Mode = Mode.train,
        transforms=None,
        min_box_edge=-1,
        mosaic_p: float = -1,
    ):
        super().__init__()
        if df is not None:
            self.df = df.copy()
            self.image_ids = df["image_id"].unique()
            assert self.df["width"].nunique() == 1
            assert self.df["height"].nunique() == 1
            assert self.df["width"].values[0] == self.df["height"].values[0]
            self.image_size = self.df["width"].values[0]
            self.min_box_edge = min_box_edge
        else:
            # test case
            self.df = None
            self.image_ids = [p.stem for p in Path(image_dir).glob("*.jpg")]
            # TODO: set image size automatically
            self.image_size = 384
        self.mosaic_p = mosaic_p
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode

    def _load_image(self, image_id):
        image = Image.open(f"{self.image_dir}/{image_id}.jpg").convert("RGB")
        return np.array(image)

    def _load_bbox(self, image_id):
        records = self.df[self.df["image_id"] == image_id]
        if self.min_box_edge > 0:
            records = records[
                (records.w >= self.min_box_edge) & (records.h >= self.min_box_edge)
            ]
        boxes = records[["x", "y", "x2", "y2"]].values
        return boxes

    def __getitem__(self, index: int):
        if self.mode in (Mode.test, Mode.validation) or (
            self.mode is Mode.train and (random.random() > self.mosaic_p)
        ):
            image_id = self.image_ids[index]
            image = self._load_image(image_id)
            if self.mode in (Mode.train, Mode.validation):
                boxes = self._load_bbox(image_id)
            else:
                boxes = (np.asarray([[0, 0, 0, 0]], dtype=np.float32),)
        else:
            image, boxes = self._load_mosaic(index)

        target = {}
        target["bbox"] = boxes
        target["cls"] = np.ones((len(boxes),), dtype=np.int64)
        # These are needed as well by the efficientdet model.
        target["img_size"] = (self.image_size, self.image_size)
        target["img_scale"] = 1.0

        if self.transforms:
            sample = {"image": image, "bboxes": target["bbox"], "labels": target["cls"]}
            if len(sample["bboxes"]) > 0:
                # apply augmentation on the fly
                boxes = sample["bboxes"]
                target["cls"] = sample["labels"]
                target["bbox"] = sample["bboxes"]
                image = sample["image"].transpose(2, 0, 1)
        else:
            image = image.transpose(2, 0, 1)
        # convert to yxyx format
        target["bbox"] = target["bbox"][:, [1, 0, 3, 2]]
        return image, target

    def _load_mosaic(self, index):
        """
        Adapted from:
        1. https://github.com/ultralytics/yolov5/blob/831773f5a23926658ee76459ce37550643432123/utils/datasets.py#L529
        2. https://www.kaggle.com/shonenkov/training-efficientdet
        """
        w, h = self.image_size, self.image_size
        border_size = self.image_size // 2

        xc, yc = [
            int(random.uniform(border_size // 2, self.image_size - border_size // 2))
            for _ in range(2)
        ]  # center x, y
        indexes = [index] + np.random.choice(
            range(len(self)), 3, replace=False
        ).tolist()

        result_image = np.full((self.image_size, self.image_size, 3), 0, dtype=np.uint8)
        result_boxes = []

        for i, index in enumerate(indexes):
            image = self._load_image(self.image_ids[index])
            boxes = self._load_bbox(self.image_ids[index])
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w), min(h, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh
            # Filter non-boxes
            np.clip(boxes, 0, self.image_size, out=boxes)
            boxes = boxes[
                np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0)
            ]
            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        return result_image, result_boxes

    def __len__(self) -> int:
        return len(self.image_ids)


def get_train_transforms(image_size: int, cutout: bool = False):
    transforms = [
        A.RandomSizedCrop(
            (int(image_size * 0.8), int(image_size * 0.8)),
            image_size,
            image_size,
            p=0.5,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(
                    hue_shift_limit=0.2,
                    sat_shift_limit=0.2,
                    val_shift_limit=0.2,
                    p=0.9,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.9
                ),
            ],
            p=0.9,
        ),
        # A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    if cutout:
        size = int(image_size * 0.1)
        transforms.append(
            A.Cutout(
                num_holes=8, max_h_size=size, max_w_size=size, fill_value=0, p=0.5
            ),
        )
    print(transforms)
    return A.Compose(
        transforms,
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=0,
            min_visibility=0,
            label_fields=["labels"],
        ),
    )
