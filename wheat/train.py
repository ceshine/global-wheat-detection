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


class MixUpDetectionCallback(Callback):
    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        old_batch = batch
        batch, targets = batch
        lambd = np.clip(np.random.beta(self.alpha, self.alpha, 1), 0.35, 0.65)
        lambd_ = torch.tensor(max(lambd, 1 - lambd), device=batch.device).float()
        # Combine input batch
        new_batch = batch * lambd_ + batch.flip(0) * (1 - lambd_)
        # Combine targets
        assert isinstance(targets, dict)
        for col in ("bbox", "cls"):
            targets[col] = torch.cat([targets[col], targets[col].flip(0)], dim=1)

        old_batch[0] = new_batch
        old_batch[1] = targets


def train(config: DictConfig, fold: int = 0):
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
    checkpoints = ModelCheckpoint(
        dirpath="cache/checkpoints/",
        monitor="val_MAP",
        mode="max",
        filename="{step:06d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=False,
    )
    callbacks = [LearningRateMonitor(logging_interval="step"), checkpoints]
    if config.mixup_alpha > 0:
        callbacks.append(
            RandomAugmentationChoiceCallback(
                [MixUpDetectionCallback(config.mixup_alpha)],
                p=[1.0],
                no_op_warmup=config.no_op_warmup_steps,
                no_op_prob=config.no_op_ratio,
            )
        )
    trainer = Trainer(
        amp_backend=config.amp_backend,
        amp_level=config.amp_level,
        precision=config.precision,
        gpus=1,
        callbacks=callbacks,
        # val_check_interval=0.5,
        gradient_clip_val=config.gradient_clip_val,
        logger=[
            TensorBoardLogger(
                "cache/tb_logs",
                name="wheat",
                version=f"fold-{fold}-{datetime.now():%Y%m%dT%H%M}",
            ),
            ScreenLogger(),
        ],
        accumulate_grad_batches=config.grad_accu,
        # fast_dev_run=True,
        max_epochs=config.epochs,
    )

    trainer.fit(model)

    print(checkpoints.best_model_path, checkpoints.best_model_score)
    pl_module = WheatModel.load_from_checkpoint(
        checkpoints.best_model_path,
        config=copy.deepcopy(config),
        df=df_train,
        fold=fold,
        half=False,
    )
    torch.save(
        {"states": pl_module.model.state_dict(), "arch": config.arch}, "wheatdet.pth"
    )


def main(
    base_dir: str,
    epochs: int = 2,
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
        epochs=epochs,
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
    if os.environ.get("SEED"):
        config.seed = int(os.environ["SEED"])
    omega_conf = OmegaConf.structured(config)
    with open("train_config.yaml", "w") as fout:
        OmegaConf.save(config=omega_conf, f=fout)
    train(omega_conf, fold=fold)


if __name__ == "__main__":
    typer.run(main)
