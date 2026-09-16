from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    in_dim: int = 16
    hidden_dim: int = 64
    out_dim: int = 1


@dataclass
class DataConfig:
    train_size: int = 1000
    eval_size: int = 200
    batch_size: int = 32


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    lr_step: int = 10


@dataclass
class TrainerConfig:
    device: str = "cuda"
    epochs: int = 100
    eval_every: int = 1
    save_every: int = 1
    resume: Optional[str] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
