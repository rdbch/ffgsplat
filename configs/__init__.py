from dataclasses import dataclass, field

from data.augment import AugmentConfig
from core.trainer import DataConfig, HeadConfig, ModelConfig, TrainerConfig, WandbConfig


@dataclass
class OptimizerConfig:
    """AdamW + one-cycle LR schedule: linear warmup to `lr`, then cosine decay."""

    # Peak learning rate (reached at the end of warmup).
    lr: float = 1e-3
    # Decoupled weight decay; kept minimal (per-scene fitting, nothing to
    # regularize towards). 0 disables it.
    weight_decay: float = 1e-6
    # Fraction of `trainer.num_steps` spent warming up.
    warmup_pct: float = 0.05
    # LR at the start of warmup and at the last step, as fractions of `lr`.
    warmup_start_ratio: float = 0.1
    final_lr_ratio: float = 0.01
    # Max global grad norm of the model's weights; 0 disables clipping.
    grad_clip: float = 1.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # Train-time scene augmentation; disabled by default (single-scene overfit).
    augment: AugmentConfig = field(default_factory=AugmentConfig)
