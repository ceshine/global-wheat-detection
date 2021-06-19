import os
import copy
from datetime import datetime
from pathlib import Path

import torch
import typer
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning_spells.loggers import ScreenLogger
from pytorch_lightning_spells.callbacks import (
    Callback,
    RandomAugmentationChoiceCallback,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from .config import WheatConfig
from .model import WheatModel

from sklearn.model_selection import StratifiedKFold


def eval(model_path: str, config: DictConfig, fold: int = 0):
    seed_everything(int(config.seed))
    base_path = Path(config.base_dir)
    df_train = pd.read_csv(str(base_path / "train.csv"))
    bboxes = np.stack(df_train["bbox"].apply(lambda x: np.fromstring(x[1:-1], sep=",")))

    for i, col in enumerate(["x", "y", "w", "h"]):
        df_train[col] = bboxes[:, i]
    df_train["area"] = df_train["w"] * df_train["h"]
    df_train["x2"] = df_train["x"] + df_train["w"]
    df_train["y2"] = df_train["y"] + df_train["h"]

    skf = StratifiedKFold(n_splits=5, random_state=88, shuffle=True)
    # one row for one image
    df_trunc = df_train[["image_id", "source"]].drop_duplicates().reset_index(drop=True)
    for fold_idx, (_, valid_index) in enumerate(
        skf.split(df_trunc, y=df_trunc["source"])
    ):
        df_trunc.loc[valid_index, "fold"] = fold_idx
    # Add fold column back
    df_train = df_train.merge(df_trunc, on=["image_id", "source"])

    # create model for one fold
    model = WheatModel(
        config,
        df_train,
        fold=fold,
        half=(
            config.precision == 16
            and config.amp_backend == "apex"
            and config.amp_level == "O2"
        ),
    )
    model.model.load_state_dict(torch.load(model_path)["states"])
    trainer = Trainer(
        amp_backend=config.amp_backend,
        amp_level=config.amp_level,
        precision=config.precision,
        gpus=1,
    )
    trainer.validate(model)


def main(
    base_dir: str,
    model_path: str,
    grad_accu: int = 1,
    arch: str = "tf_efficientdet_d3",
    batch_size: int = 8,
    fold: int = 0,
    mixup: float = -1,
    cutout: bool = False,
    mosaic_p: float = -1,
):
    config = WheatConfig(
        base_dir=base_dir,
        epochs=0,
        image_size=int(Path(base_dir).name),
        arch=arch,
        grad_accu=grad_accu,
        batch_size=batch_size,
        precision=16,
        amp_backend="apex",
        amp_level="O2",
        cutout=cutout,
        mixup_alpha=mixup,
        mosaic_p=mosaic_p,
    )
    assert not (cutout is True and mixup > 0), "Can only enable one of MixUp and CutOut"
    omega_conf = OmegaConf.structured(config)
    eval(model_path, omega_conf, fold=fold)


if __name__ == "__main__":
    typer.run(main)
