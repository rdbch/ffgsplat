from dataclasses import dataclass, field
from typing import Optional

from data.augment import AugmentConfig
from data.edgs_points import InputConfig
from core.trainer import DataConfig, HeadConfig, ModelConfig, TrainerConfig, WandbConfig


@dataclass
class OptimizerConfig:
    """AdamW or Muon + one-cycle LR schedule: linear warmup to `lr`, then cosine decay."""

    # "adamw": AdamW on every parameter. "muon": Muon (`others/muon_optimizer.py`)
    # on the backbone's / feature MLP's hidden `nn.Linear` weights, AdamW on the
    # rest (biases, norms, sparse-conv kernels, first feature-MLP layer, GS head);
    # see `Trainer.build_optimizer`.
    name: str = "adamw"
    # Peak learning rate (reached at the end of warmup).
    lr: float = 1e-3
    # Muon only. Peak LR of the Muon groups; None = `lr`. With
    # `muon_update_scale=moonlight` Muon's update RMS is ~0.2 x lr, close to
    # AdamW's, so the two can share `lr`.
    muon_lr: Optional[float] = None
    # Muon only: "moonlight" | "keller" | "none" (see `Muon`).
    muon_update_scale: str = "moonlight"
    muon_momentum: float = 0.95
    muon_polar_steps: int = 6
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
    # Input points / per-point features; defaults = MCMC reference means + rgb.
    input: InputConfig = field(default_factory=InputConfig)
