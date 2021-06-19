from dataclasses import dataclass


@dataclass
class WheatConfig:
    base_dir: str
    epochs: int
    arch: str
    image_size: int
    batch_size: int
    base_lr: float = 2e-4
    num_workers: int = 4
    seed: int = 999
    amp_backend: str = "native"
    amp_level: str = "O2"  # only effective if amp_backend == apex
    precision: int = 32
    gradient_clip_val: float = 10
    grad_accu: int = 1
    cutout: bool = False
    mixup_alpha: float = -1
    no_op_ratio: float = 0.2
    no_op_warmup_steps: int = 100
    mosaic_p: float = -1
