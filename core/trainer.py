import os
import sys
import math
import contextlib
from datetime import timedelta
import random
import torch
import numpy as np
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from fused_ssim import fused_ssim
from gsplat.rendering import rasterization
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from data.augment import augment, max_expansion
from data.datasets.colmap import Dataset, DatasetConfig, Parser, ParserConfig
from models.point_model import LitePtGSModel

_LITEPT_ROOT = Path(__file__).resolve().parent.parent / "submodules" / "LitePT"
if str(_LITEPT_ROOT) not in sys.path:
    sys.path.append(str(_LITEPT_ROOT))

from litept.model import Point  # noqa: E402

from typing import Any, Dict



def _describe(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor  dtype={value.dtype}  shape={tuple(value.shape)}"
    if isinstance(value, np.ndarray):
        return f"ndarray dtype={value.dtype}  shape={value.shape}"
    if isinstance(value, (list, tuple)):
        kind = type(value).__name__
        if len(value) == 0:
            return f"{kind}[0]"
        return f"{kind}[{len(value)}] of {_describe(value[0])}"
    return f"{type(value).__name__} = {value!r}"


def print_batch(batch: Dict[str, Any], name: str = "batch") -> None:
    """Pretty-print a batch dict: one line per key with its type/dtype/shape."""
    print(f"{name}:")
    key_width = max((len(k) for k in batch), default=0)
    for key, value in batch.items():
        print(f"  {key.ljust(key_width)} : {_describe(value)}")


def _collate_camera_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for `data.datasets.colmap.Dataset` items. `K`/`camtoworld`/`image`
    are fixed-shape per view and get stacked; `mask`/`points`/`depths` vary in
    size per view (and aren't present on every item), so they're left as lists.
    """
    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in ("K", "camtoworld", "image"):
            out[key] = torch.stack(values, dim=0)
        elif key in ("image_id", "scene"):
            out[key] = torch.tensor(values)
        else:
            out[key] = values
    return out


@contextlib.contextmanager
def bn_batch_stats(model: nn.Module, enabled: bool = True):
    """Within the block, BatchNorm layers normalize with the current batch's
    statistics (training behavior) without updating their running stats;
    every other module keeps its mode (e.g. DropPath stays off in eval)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)] if enabled else []
    saved = [(m.training, m.momentum, m.num_batches_tracked.clone()) for m in bns]
    for m in bns:
        # momentum 0: running = 1 * running + 0 * batch, i.e. unchanged.
        m.train()
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, (training, momentum, tracked) in zip(bns, saved):
            m.train(training)
            m.momentum = momentum
            m.num_batches_tracked.copy_(tracked)


class _MultiSceneDataset(torch.utils.data.Dataset):
    """Views of several scenes, indexed by `(scene, view)` pairs (from
    `_RandomSceneBatchSampler`); each item is tagged with its `scene` index."""

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, key):
        scene, index = key
        item = self.datasets[scene][index]
        item["scene"] = scene
        return item


class _RandomSceneBatchSampler(torch.utils.data.Sampler):
    """Endless batches of `batch_size` views of one scene, picked uniformly at
    random each batch (so each rank renders a random scene each step). Views
    come from a per-scene shuffled order, reshuffled when used up. Draws from
    torch's default RNG, which is seeded per rank."""

    def __init__(self, sizes, batch_size):
        self.sizes = list(sizes)
        self.batch_size = batch_size

    def __iter__(self):
        orders = [torch.randperm(n).tolist() for n in self.sizes]
        pos = [0] * len(self.sizes)
        while True:
            scene = int(torch.randint(len(self.sizes), ()))
            batch = []
            while len(batch) < self.batch_size:
                if pos[scene] == len(orders[scene]):
                    orders[scene] = torch.randperm(self.sizes[scene]).tolist()
                    pos[scene] = 0
                take = orders[scene][pos[scene] : pos[scene] + self.batch_size - len(batch)]
                pos[scene] += len(take)
                batch.extend((scene, i) for i in take)
            yield batch


@dataclass
class Scene:
    """One scene's data: parser, datasets, eval loader and the reference
    Gaussians' `means` / base RGB `colors` (the model's fixed input, pinned on
    CPU and attached to every batch from the main process, so the full cloud
    isn't shipped through the loader workers every item)."""

    name: str
    parser: Parser
    train_dataset: Dataset
    eval_dataset: Dataset
    eval_loader: DataLoader
    means: torch.Tensor
    colors: torch.Tensor


@dataclass
class DataConfig:
    """Config for `Trainer`'s data pipeline: the scene parser plus train/eval
    dataset and dataloader settings, as one flat dataclass with `train_`/`eval_`
    prefixed fields rather than separate per-split sub-configs."""

    parser: ParserConfig = field(default_factory=lambda: ParserConfig(data_dir=MISSING))

    # Multi-scene training: scene names under `scene_root`, each with its
    # reference checkpoint at `gaussian_ckpt_template.format(scene=name)`; all
    # other parser settings come from `parser` (whose `data_dir` and
    # `gaussian_ckpt_path` are then unused). Each training batch is
    # `train_batch_size` views of one random scene. Every scene is evaluated
    # on its own val split. Empty = single scene from `parser`.
    scenes: List[str] = field(default_factory=list)
    scene_root: Optional[str] = None
    gaussian_ckpt_template: Optional[str] = None

    train_patch_size: Optional[int] = None
    train_load_depths: bool = False
    train_batch_size: int = 4
    train_num_workers: int = 4
    train_shuffle: bool = True
    train_pin_memory: bool = False
    train_drop_last: bool = False

    eval_patch_size: Optional[int] = None
    eval_load_depths: bool = False
    eval_batch_size: int = 4
    eval_num_workers: int = 4
    eval_shuffle: bool = False
    eval_pin_memory: bool = False
    eval_drop_last: bool = False


@dataclass
class ModelConfig:
    """Explicit config for the `LitePT` backbone
    (`submodules/LitePT/litept/model.py`); field names match its constructor
    kwargs 1:1. Defaults below reproduce LitePT's own defaults, except
    `dec_conv`/`dec_attn`, which default to the full-conv decoder variant
    (conv at every decoder stage, no decoder attention)."""

    # Number of input feature channels per point. Default (3) matches feeding
    # the Gaussians' base RGB color (from `sh0`, in [0, 1]) as `feat`, as done by
    # `Trainer.train_step`.
    in_channels: int = 3
    # Space-filling curves used to serialize the point cloud into 1D sequences for
    # windowed attention; each entry gives one ordering, consumed round-robin
    # across blocks (`order_index = i % len(order)`).
    order: Tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")
    # Per-stage downsampling factor applied by `GridPooling` between encoder
    # stages (`len(stride) == len(enc_depths) - 1`).
    stride: Tuple[int, ...] = (2, 2, 2, 2)
    # Number of `Block`s in each encoder stage.
    enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2)
    # Feature width (channels) at each encoder stage.
    enc_channels: Tuple[int, ...] = (36, 72, 144, 252, 504)
    # Number of attention heads at each encoder stage (only used where
    # `enc_attn` is True for that stage).
    enc_num_head: Tuple[int, ...] = (2, 4, 8, 14, 28)
    # Windowed-attention patch size (max points per attention window) at each
    # encoder stage.
    enc_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024)
    # Whether each encoder stage's blocks include the local sparse-conv branch.
    enc_conv: Tuple[bool, ...] = (True, True, True, False, False)
    # Whether each encoder stage's blocks include the windowed self-attention
    # branch (mutually complementary with `enc_conv` in the default config:
    # conv for early/dense stages, attention for later/sparse ones).
    enc_attn: Tuple[bool, ...] = (False, False, False, True, True)
    # RoPE (rotary position embedding) frequency used by attention at each
    # encoder stage; ignored where `enc_attn` is False.
    enc_rope_freq: Tuple[float, ...] = (100.0, 100.0, 100.0, 100.0, 100.0)
    # Number of `Block`s in each decoder stage (0 means the stage is just an
    # unpooling + skip connection with no extra blocks). Ignored if `enc_mode`.
    dec_depths: Tuple[int, ...] = (0, 0, 0, 0)
    # Feature width (channels) at each decoder stage (coarsest-to-finest is
    # `dec_channels + [enc_channels[-1]]` internally). Ignored if `enc_mode`.
    dec_channels: Tuple[int, ...] = (72, 72, 144, 252)
    # Number of attention heads at each decoder stage (only used where
    # `dec_attn` is True for that stage). Ignored if `enc_mode`.
    dec_num_head: Tuple[int, ...] = (4, 4, 8, 14)
    # Windowed-attention patch size at each decoder stage. Ignored if `enc_mode`.
    dec_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024)
    # Whether each decoder stage's blocks include the local sparse-conv branch.
    # All True here: the "full conv decoder" variant (conv at every decoder
    # stage instead of the library default's attention-only decoder).
    dec_conv: Tuple[bool, ...] = (True, True, True, True)
    # Whether each decoder stage's blocks include the windowed self-attention
    # branch. All False to pair with the full-conv decoder above.
    dec_attn: Tuple[bool, ...] = (False, False, False, False)
    # RoPE frequency used by attention at each decoder stage; ignored where
    # `dec_attn` is False (i.e. unused under the full-conv decoder default).
    dec_rope_freq: Tuple[float, ...] = (100.0, 100.0, 100.0, 100.0)
    # Hidden-layer expansion ratio of the MLP inside each attention block
    # (hidden width = channels * mlp_ratio).
    mlp_ratio: int = 4
    # Whether the attention QKV projection includes a bias term.
    qkv_bias: bool = True
    # Manual override for the attention softmax scale; `None` uses the default
    # `head_dim ** -0.5`.
    qk_scale: Optional[float] = None
    # Dropout probability applied to attention weights.
    attn_drop: float = 0.0
    # Dropout probability applied after the attention/MLP output projections.
    proj_drop: float = 0.0
    # Max stochastic-depth (DropPath) rate, linearly scheduled across all
    # encoder/decoder blocks.
    drop_path: float = 0.3
    # If True, apply LayerNorm before each attention/MLP sub-block (pre-norm);
    # if False, apply it after (post-norm).
    pre_norm: bool = True
    # If True, randomly shuffle which serialization `order` each stage's
    # pooling re-serializes with, for augmentation/regularization.
    shuffle_orders: bool = True
    # If True, run encoder-only (no decoder) and return coarse per-stage
    # features instead of per-point ones -- not used for point-conditioned GS
    # prediction, which needs one output per input point.
    enc_mode: bool = False
    # Norm used in the stem and the grid pooling/unpooling layers (attention
    # blocks always use LayerNorm). "batch" is LitePT's BatchNorm; "layer" is
    # per-point LayerNorm, which has no running statistics, so eval matches
    # training even when every step sees a differently posed cloud
    # (augmentation), where BatchNorm's running averages fit no single pose.
    norm: str = "batch"


@dataclass
class HeadConfig:
    """Config for `LitePtGSModel`'s per-point GS-parameter head."""

    # Max SH degree predicted (`shN` has `(sh_degree + 1) ** 2 - 1` coeffs).
    sh_degree: int = 3
    # Initial (post-activation) scale and opacity every Gaussian starts from,
    # carried by the head's bias; same role as `simple_trainer.py`'s
    # `init_scale`/`init_opa`, but a constant (no per-point kNN distance).
    init_scale: float = 0.01
    init_opacity: float = 0.1
    # Upper bound on every Gaussian's (post-activation) scale. Baseline MCMC
    # checkpoints have 99.9% of max-axis scales below ~0.5, a few far-background
    # ones larger.
    max_scale: float = 1.0


@dataclass
class WandbConfig:
    """Config for Weights & Biases logging (`wandb.init` kwargs)."""

    project: str = "ffgsplat"
    entity: Optional[str] = None
    name: Optional[str] = None
    # "online", "offline" or "disabled".
    mode: str = "online"


@dataclass
class TrainerConfig:
    """Config for `Trainer`, structured so it can be embedded in an OmegaConf tree."""

    device: str = "cuda"
    # Cap on this process's share of GPU memory (`torch.cuda.
    # set_per_process_memory_fraction`); `None` = no cap. With a cap, PyTorch's
    # caching allocator frees its cache and retries before the card is full,
    # instead of letting the WSL2/WDDM driver silently spill allocations into
    # system RAM (no OOM, just a ~100x slowdown).
    cuda_memory_fraction: Optional[float] = None
    num_steps: int = 100
    # Evaluate on the full eval split every this many steps (and at the last step).
    eval_every: int = 1000
    # LPIPS backbone, as `simple_trainer.py`'s `lpips_net` ("alex" or "vgg").
    lpips_net: str = "alex"
    # Log the first this many eval views (GT | render) to wandb at each eval;
    # -1 logs all of them. Same views every eval, to compare progress.
    log_eval_images: int = 8
    # Save every this many steps (and at the last step). Writes the full
    # training state to `<output_dir>/ckpts/` and the model weights alone to
    # `<output_dir>/weights/`.
    save_every: int = 1000
    output_dir: str = "results/run"
    # Path to a `<output_dir>/ckpts/step_*.pt` full checkpoint to resume from.
    resume: Optional[str] = None
    # Log train metrics (stdout + wandb) every this many steps.
    log_train_steps: int = 5
    # Log one random view of the batch (GT | render, side by side) to wandb
    # every this many steps.
    log_train_images_steps: int = 50
    # Voxel size (scene units) used to derive LitePT's `grid_coord` from the
    # raw point coordinates (grid pooling/sparse conv need a voxel grid, not
    # continuous coords); see scripts/001_points_per_voxel.py for picking this
    # per scene from its points-per-voxel histogram.
    point_grid_size: float = 0.01
    # Loss and rendering settings, same meaning/defaults as `simple_trainer.py`.
    ssim_lambda: float = 0.2
    # MCMC's regularizers (`simple_trainer.py`'s `opacity_reg`/`scale_reg`,
    # 0.01 each in its MCMC preset): weight on the mean activated opacity and
    # the mean activated scale over all Gaussians, pushing them to be as
    # transparent / small as the render loss allows. 0 disables.
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    # Composite each training render over a random solid color (per view)
    # instead of black, so Gaussians can't lean on a black background for
    # dark/empty regions. Eval always renders over black.
    random_bkgd: bool = True
    # Normalize with the evaluated cloud's own statistics in BatchNorm layers
    # at eval (as in training, where each batch is one whole cloud) instead of
    # the running averages. Needed with augmentation + `model.norm="batch"`:
    # the running averages mix all random poses and fit none, which blows up
    # predicted scales at eval. No-op for `model.norm="layer"`.
    eval_bn_batch_stats: bool = False
    sh_degree_interval: int = 1000
    # Process-group backend for multi-GPU data parallel (launched with
    # `torchrun`; ignored otherwise). NCCL works between MIG slices on
    # compute1 (no P2P, ~2.5 ms per 64 MB all-reduce); "gloo" is the fallback.
    dist_backend: str = "nccl"
    near_plane: float = 0.01
    far_plane: float = 1e10


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # Data parallel when launched with `torchrun` (which sets WORLD_SIZE /
        # RANK): every rank renders its own `train_batch_size` views of its own
        # augmented cloud and gradients are averaged, so the effective batch is
        # `world_size * train_batch_size` views. Each rank must see exactly one
        # GPU (CUDA enumerates only one MIG slice per process), hence `cuda`.
        # Without torchrun this is a single process and every `dist` call is
        # skipped.
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.is_main = self.rank == 0

        self.device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and cfg.trainer.cuda_memory_fraction is not None:
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(cfg.trainer.cuda_memory_fraction, index)
        if self.world_size > 1:
            if self.device.type == "cuda":
                # `cuda` has no index; each rank sees one GPU, so this is cuda:0.
                self.device = torch.device("cuda", torch.cuda.current_device())
                torch.cuda.set_device(self.device)
            # Long timeout: the other ranks wait in a barrier while rank 0
            # evaluates and saves.
            dist.init_process_group(
                backend=cfg.trainer.dist_backend,
                timeout=timedelta(hours=1),
                device_id=self.device if self.device.type == "cuda" and cfg.trainer.dist_backend == "nccl" else None,
            )
            # torch's default seed is the same in every process, so without
            # this all ranks would draw identical augmentations / backgrounds.
            # Rank 0 keeps the default, matching single-process runs.
            if self.rank > 0:
                torch.manual_seed(torch.initial_seed() + self.rank)
            print(f"Rank {self.rank}/{self.world_size} on {self.device} ({cfg.trainer.dist_backend})")

        self.model = None
        self.train_sampler = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.logger = None

        # One `Scene` per scene (a single one unless `data.scenes` is set).
        self.scenes: List[Scene] = []
        self.multi_scene = bool(cfg.data.scenes)
        self.train_loader = None

        self.step = 0
        self.start_step = 0
        self.best_metric = None
        # Set by `load_checkpoint` so `build_logger` continues the same wandb run.
        self.wandb_run_id = None

    def build_model(self):
        self.model = LitePtGSModel(
            backbone_cfg=OmegaConf.to_container(self.cfg.model, resolve=True),
            sh_degree=self.cfg.head.sh_degree,
            init_scale=self.cfg.head.init_scale,
            init_opacity=self.cfg.head.init_opacity,
            max_scale=self.cfg.head.max_scale,
        ).to(self.device)
        # Start every rank from rank 0's weights.
        if self.world_size > 1:
            for t in itertools.chain(self.model.parameters(), self.model.buffers()):
                self._broadcast(t)

    def build_optimizer(self):
        # The model's weights are the only trainable parameters; `means` are a
        # fixed input.
        opt_cfg = self.cfg.optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay
        )
        # Linear warmup from `warmup_start_ratio * lr` to `lr` over the first
        # `warmup_pct` of training, then cosine decay to `final_lr_ratio * lr`.
        # Stepped once per training step. Adam's betas are left fixed.
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=opt_cfg.lr,
            total_steps=self.cfg.trainer.num_steps,
            pct_start=opt_cfg.warmup_pct,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=1.0 / opt_cfg.warmup_start_ratio,
            final_div_factor=opt_cfg.warmup_start_ratio / opt_cfg.final_lr_ratio,
        )

    def build_dataloaders(self):
        data_cfg = self.cfg.data
        if self.multi_scene:
            if not data_cfg.scene_root or not data_cfg.gaussian_ckpt_template:
                raise ValueError("data.scenes needs data.scene_root and data.gaussian_ckpt_template")
            self.scenes = [
                self._build_scene(name, OmegaConf.merge(data_cfg.parser, {
                    "data_dir": str(Path(data_cfg.scene_root) / name),
                    "gaussian_ckpt_path": data_cfg.gaussian_ckpt_template.format(scene=name),
                }))
                for name in data_cfg.scenes
            ]
            # A random scene per batch (per rank); endless, so no epochs.
            self.train_loader = DataLoader(
                _MultiSceneDataset([scene.train_dataset for scene in self.scenes]),
                collate_fn=_collate_camera_batch,
                batch_sampler=_RandomSceneBatchSampler(
                    [len(scene.train_dataset) for scene in self.scenes], data_cfg.train_batch_size
                ),
                num_workers=data_cfg.train_num_workers,
                pin_memory=data_cfg.train_pin_memory,
            )
            return

        scene = self._build_scene(Path(data_cfg.parser.data_dir).name, data_cfg.parser)
        self.scenes = [scene]
        # Each rank draws a disjoint shard of the views every epoch.
        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                scene.train_dataset, shuffle=data_cfg.train_shuffle, drop_last=data_cfg.train_drop_last
            )
        self.train_loader = DataLoader(
            scene.train_dataset,
            collate_fn=_collate_camera_batch,
            batch_size=data_cfg.train_batch_size,
            num_workers=data_cfg.train_num_workers,
            shuffle=data_cfg.train_shuffle if self.train_sampler is None else False,
            sampler=self.train_sampler,
            pin_memory=data_cfg.train_pin_memory,
            drop_last=data_cfg.train_drop_last,
        )

    def _build_scene(self, name, parser_cfg):
        data_cfg = self.cfg.data
        parser = Parser(parser_cfg)
        train_dataset = Dataset(parser, DatasetConfig(
            split="train",
            patch_size=data_cfg.train_patch_size,
            load_depths=data_cfg.train_load_depths,
        ))
        eval_dataset = Dataset(parser, DatasetConfig(
            split="val",
            patch_size=data_cfg.eval_patch_size,
            load_depths=data_cfg.eval_load_depths,
        ))
        eval_loader = DataLoader(
            eval_dataset,
            collate_fn=_collate_camera_batch,
            batch_size=data_cfg.eval_batch_size,
            num_workers=data_cfg.eval_num_workers,
            shuffle=data_cfg.eval_shuffle,
            pin_memory=data_cfg.eval_pin_memory,
            drop_last=data_cfg.eval_drop_last,
        )

        if parser.gaussian_means is None:
            raise ValueError(f"{name}: data.parser.gaussian_ckpt_path must be set")

        # Drop far-away outlier Gaussians (MCMC puts a few up to ~9000 units out on bonsai)
        # so the voxelized cloud fits LitePT's serialization limit of 2^16 voxels
        # per axis (`Point.serialization`'s `depth <= 16` assert). All `gaussian_*`
        # arrays are filtered with the same mask to stay aligned.
        means = parser.gaussian_means
        half_extent = (2**16 - 1) * self.cfg.trainer.point_grid_size / 2
        # Training augmentation can rotate/scale the cloud (and shift the grid
        # by up to a voxel), so leave room for its worst-case extent.
        if self.cfg.augment.enabled:
            half_extent = (half_extent - self.cfg.trainer.point_grid_size) / max_expansion(self.cfg.augment)
        keep = (np.abs(means - np.median(means, axis=0)) < half_extent).all(axis=1)
        for attr_name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            attr = f"gaussian_{attr_name}"
            setattr(parser, attr, getattr(parser, attr)[keep])
        print(f"[{name}] Removed {(~keep).sum()} / {len(keep)} outlier Gaussians")

        means = torch.from_numpy(parser.gaussian_means).float()
        # Degree-0 SH to RGB: `sh0 * C0 + 0.5`, with C0 the l=0 SH basis constant.
        sh0 = torch.from_numpy(parser.gaussian_sh0).float()[:, 0, :]
        colors = (sh0 * 0.28209479177387814 + 0.5).clamp(0.0, 1.0)
        if self.device.type == "cuda":
            means = means.pin_memory()
            colors = colors.pin_memory()
        return Scene(name, parser, train_dataset, eval_dataset, eval_loader, means, colors)

    def train(self):
        train_iter = self._train_batches()
        self.model.train()

        # Eval metrics, set up as in `simple_trainer.py` so numbers are comparable
        # with the MCMC baseline.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False)
        self.lpips = self.lpips.to(self.device)

        num_steps = self.cfg.trainer.num_steps
        for self.step in range(self.start_step, num_steps):
            batch = next(train_iter)
            scene = self.scenes[int(batch["scene"][0])] if "scene" in batch else self.scenes[0]
            batch["means"] = scene.means
            batch["colors"] = scene.colors
            metrics = self.train_step(batch)
            self.log(metrics, self.step)

            # Eval split by scene over the ranks (a single scene: rank 0 only);
            # checkpoints on rank 0 only. The others wait.
            if (self.step + 1) % self.cfg.trainer.eval_every == 0 or self.step == num_steps - 1:
                self.model.eval()
                self.eval()
                self.model.train()
                self._barrier()

            if (self.step + 1) % self.cfg.trainer.save_every == 0 or self.step == num_steps - 1:
                if self.is_main:
                    self.save_checkpoint()
                self._barrier()

    def _train_batches(self):
        # Re-iterate the loader each epoch (reshuffles). `itertools.cycle` would
        # cache every batch of the first epoch in RAM and replay them unshuffled.
        for epoch in itertools.count():
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            yield from self.train_loader

    def train_step(self, batch):
        torch.cuda.reset_peak_memory_stats(self.device)
        coord = batch["means"].to(self.device, non_blocking=True)
        feat = batch["colors"].to(self.device, non_blocking=True)
        camtoworlds = batch["camtoworld"].to(self.device, non_blocking=True)  # [B, 4, 4]
        # Random similarity transform of points + cameras (identity if
        # `augment.enabled` is False); renders are unchanged, so the targets are.
        coord, camtoworlds, grid_coord = augment(
            self.cfg.augment, coord, camtoworlds, self.cfg.trainer.point_grid_size
        )
        offset = torch.tensor([coord.shape[0]], device=self.device)
        point = Point(
            coord=coord,
            feat=feat,
            grid_size=self.cfg.trainer.point_grid_size,
            offset=offset,
        )
        if grid_coord is not None:
            point["grid_coord"] = grid_coord

        # Predict raw GS params for every Gaussian once, with activations as in
        # `simple_trainer.py`'s `rasterize_splats`.
        splats = self.model(point)
        # Render/backprop one view at a time into a detached copy of the
        # predictions, then push the accumulated gradient through the model in a
        # single backward: same gradients as rendering the whole batch at once,
        # but rasterization memory peaks at one view instead of `B`.
        leaves = {k: v.detach().requires_grad_() for k, v in splats.items()}
        colors = torch.cat([leaves["sh0"], leaves["shN"]], dim=1)
        scales = torch.exp(leaves["scales"])
        opacities = torch.sigmoid(leaves["opacities"])

        Ks = batch["K"].to(self.device, non_blocking=True)                    # [B, 3, 3]
        pixels = batch["image"].to(self.device, non_blocking=True) / 255.0    # [B, H, W, 3]
        num_views, height, width = pixels.shape[:3]
        sh_degree = min(self.step // self.cfg.trainer.sh_degree_interval, self.cfg.head.sh_degree)
        ssim_lambda = self.cfg.trainer.ssim_lambda
        # One random view of the batch to log as an image this step (-1: none).
        image_view = (
            random.randrange(num_views)
            if self.step % self.cfg.trainer.log_train_images_steps == 0
            else -1
        )
        image = None

        # Regularizers, once per step (not per view), as in `simple_trainer.py`.
        # Backpropagated into the leaves first, keeping the activation graph for
        # the per-view render losses below.
        reg = torch.zeros((), device=self.device)
        if self.cfg.trainer.opacity_reg > 0.0:
            reg = reg + self.cfg.trainer.opacity_reg * opacities.mean()
        if self.cfg.trainer.scale_reg > 0.0:
            reg = reg + self.cfg.trainer.scale_reg * scales.mean()
        if reg.requires_grad:
            reg.backward(retain_graph=True)

        loss_sum = reg.item()
        l1_sum = ssim_sum = mse_sum = 0.0
        for i in range(num_views):
            render, alpha, _ = rasterization(
                means=coord,
                quats=leaves["quats"],  # normalized inside `rasterization`
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds[i : i + 1]),
                Ks=Ks[i : i + 1],
                width=width,
                height=height,
                sh_degree=sh_degree,
                near_plane=self.cfg.trainer.near_plane,
                far_plane=self.cfg.trainer.far_plane,
            )
            # Rendered over black; composite over a random color, as
            # `simple_trainer.py` does for `random_bkgd`.
            if self.cfg.trainer.random_bkgd:
                render = render + torch.rand(1, 3, device=self.device) * (1.0 - alpha)
            gt = pixels[i : i + 1]
            l1loss = F.l1_loss(render, gt)
            ssimloss = 1.0 - fused_ssim(
                render.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), padding="valid"
            )
            loss = (l1loss * (1.0 - ssim_lambda) + ssimloss * ssim_lambda) / num_views
            # Keep the activation graph (exp/sigmoid/cat) alive across views.
            loss.backward(retain_graph=i < num_views - 1)

            loss_sum += loss.item()
            l1_sum += l1loss.item() / num_views
            ssim_sum += (1.0 - ssimloss.item()) / num_views
            mse_sum += F.mse_loss(render.detach(), gt).item() / num_views
            if i == image_view:
                image = torch.cat([gt[0], render[0].detach().clamp(0.0, 1.0)], dim=1)  # [H, 2W, 3]
                image = (image * 255.0).byte().cpu().numpy()

        self.optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(
            list(splats.values()), [leaves[k].grad for k in splats]
        )
        if self.world_size > 1:
            # Average over ranks before clipping, so the clip and the update
            # see the full batch's gradient.
            self._all_reduce_grads()
            # Batch means over all ranks, for logging.
            l1_sum, ssim_sum, mse_sum, loss_sum = self._all_reduce_mean([l1_sum, ssim_sum, mse_sum, loss_sum])
        if self.cfg.optimizer.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.optimizer.grad_clip
            ).item()
        else:
            grad_norm = float("nan")
        lr = self.scheduler.get_last_lr()[0]
        self.optimizer.step()
        self.scheduler.step()

        metrics = {
            "loss": loss_sum,
            "l1": l1_sum,
            "ssim": ssim_sum,
            "psnr": -10.0 * math.log10(mse_sum),
            "lr": lr,
            "grad_norm": grad_norm,
            "sh_degree": sh_degree,
            "reg": reg.item(),
            # Mean / max activated scale and mean opacity, to watch for
            # oversized Gaussians.
            "scale_mean": scales.detach().mean().item(),
            "scale_max": scales.detach().max().item(),
            "opacity_mean": opacities.detach().mean().item(),
            "mem_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
            # Peak held by the caching allocator (allocated + cached blocks); the
            # number to compare against the card's capacity.
            "mem_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30,
        }
        if image is not None:
            metrics["image"] = image
            metrics["image_scene"] = self.scenes[int(batch["scene"][0])].name if self.multi_scene else None
        return metrics

    @torch.no_grad()
    def eval(self):
        """Render every eval view of every scene and log mean PSNR/SSIM/LPIPS vs.
        ground truth. Scene `i` is evaluated on rank `i % world_size`; results
        are gathered on rank 0. Call on every rank."""
        results = {
            scene.name: self.eval_scene(scene)
            for i, scene in enumerate(self.scenes)
            if i % self.world_size == self.rank
        }
        if self.world_size > 1:
            gathered = [None] * self.world_size if self.is_main else None
            dist.gather_object(results, gathered, dst=0)
            if not self.is_main:
                return
            results = {k: v for r in gathered for k, v in r.items()}

        log = {"step": self.step}
        for scene in self.scenes:
            metrics = results[scene.name]
            images = metrics.pop("images")
            stats = {k: sum(v) / len(v) for k, v in metrics.items()}
            print(
                f"[eval step {self.step}]{f' {scene.name}' if self.multi_scene else ''} "
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, "
                f"LPIPS: {stats['lpips']:.3f} ({len(metrics['psnr'])} images)"
            )
            # Single scene: `eval/<metric>` as before; multi-scene:
            # `eval/<scene>/<metric>`, plus their mean over scenes below.
            prefix = f"eval/{scene.name}/" if self.multi_scene else "eval/"
            log.update({f"{prefix}{k}": v for k, v in stats.items()})
            log[f"{prefix}images"] = [wandb.Image(image, caption=caption) for image, caption in images]
        if self.multi_scene:
            for k in ("psnr", "ssim", "lpips"):
                log[f"eval/{k}"] = sum(log[f"eval/{scene.name}/{k}"] for scene in self.scenes) / len(self.scenes)
            print(
                f"[eval step {self.step}] mean over {len(self.scenes)} scenes: PSNR: {log['eval/psnr']:.3f}, "
                f"SSIM: {log['eval/ssim']:.4f}, LPIPS: {log['eval/lpips']:.3f}"
            )
        self.logger.log(log)

    @torch.no_grad()
    def eval_scene(self, scene):
        """Per-image PSNR/SSIM/LPIPS lists (and logged images) over `scene`'s
        eval split."""
        # The model's weights are fixed during eval, so predict the Gaussians once
        # for the whole eval split.
        coord = scene.means.to(self.device, non_blocking=True)
        point = Point(
            coord=coord,
            feat=scene.colors.to(self.device, non_blocking=True),
            grid_size=self.cfg.trainer.point_grid_size,
            offset=torch.tensor([coord.shape[0]], device=self.device),
        )
        with bn_batch_stats(self.model, self.cfg.trainer.eval_bn_batch_stats):
            splats = self.model(point)
        splats = {
            "means": coord,
            "quats": splats["quats"],
            "scales": torch.exp(splats["scales"]),
            "opacities": torch.sigmoid(splats["opacities"]),
            "colors": torch.cat([splats["sh0"], splats["shN"]], dim=1),
        }

        metrics = {"psnr": [], "ssim": [], "lpips": [], "images": []}
        for batch in scene.eval_loader:
            for k, v in self.eval_step(batch, splats).items():
                metrics[k].extend(v)
        return metrics

    @torch.no_grad()
    def eval_step(self, batch, splats):
        """Render each view of `batch` with the (activated) `splats` and return
        per-image PSNR/SSIM/LPIPS lists."""
        camtoworlds = batch["camtoworld"].to(self.device, non_blocking=True)  # [B, 4, 4]
        Ks = batch["K"].to(self.device, non_blocking=True)                    # [B, 3, 3]
        pixels = batch["image"].to(self.device, non_blocking=True) / 255.0    # [B, H, W, 3]
        num_views, height, width = pixels.shape[:3]
        # Same SH degree the model is currently trained with.
        sh_degree = min(self.step // self.cfg.trainer.sh_degree_interval, self.cfg.head.sh_degree)

        metrics = {"psnr": [], "ssim": [], "lpips": [], "images": []}
        num_images = self.cfg.trainer.log_eval_images
        # One view at a time, like `train_step`, so memory doesn't scale with
        # the eval batch size.
        for i in range(num_views):
            render, _, _ = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=splats["scales"],
                opacities=splats["opacities"],
                colors=splats["colors"],
                viewmats=torch.linalg.inv(camtoworlds[i : i + 1]),
                Ks=Ks[i : i + 1],
                width=width,
                height=height,
                sh_degree=sh_degree,
                near_plane=self.cfg.trainer.near_plane,
                far_plane=self.cfg.trainer.far_plane,
            )
            render = render.clamp(0.0, 1.0).permute(0, 3, 1, 2)  # [1, 3, H, W]
            gt = pixels[i : i + 1].permute(0, 3, 1, 2)
            metrics["psnr"].append(self.psnr(render, gt).item())
            metrics["ssim"].append(self.ssim(render, gt).item())
            metrics["lpips"].append(self.lpips(render, gt).item())

            # `image_id` is the view's index within the eval split.
            image_id = int(batch["image_id"][i])
            if num_images < 0 or image_id < num_images:
                image = torch.cat([gt[0], render[0]], dim=2).permute(1, 2, 0)  # [H, 2W, 3]
                caption = f"view {image_id} | GT | render | PSNR {metrics['psnr'][-1]:.2f}"
                metrics["images"].append(((image * 255.0).byte().cpu().numpy(), caption))
        return metrics

    def build_logger(self):
        # Only rank 0 logs.
        if not self.is_main:
            return
        wandb_cfg = self.cfg.wandb
        self.logger = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.name,
            mode=wandb_cfg.mode,
            config={**OmegaConf.to_container(self.cfg, resolve=True), "world_size": self.world_size},
            id=self.wandb_run_id,
            resume="allow" if self.wandb_run_id else None,
        )
        # Plot against our own `step` instead of wandb's internal one, which must
        # strictly increase: resuming from a checkpoint older than the last
        # logged step would otherwise have those steps silently dropped.
        # (wandb's "rewind" would truncate instead, but is private preview.)
        self.logger.define_metric("step", hidden=True)
        self.logger.define_metric("train/*", step_metric="step")
        self.logger.define_metric("eval/*", step_metric="step")

    def log(self, metrics, step):
        if not self.is_main:
            return
        # `train/` prefix groups these into one "train" panel section in wandb.
        if step % self.cfg.trainer.log_train_steps == 0:
            print(f"[step {step}] { {k: v for k, v in metrics.items() if k not in ('image', 'image_scene')} }")
            self.logger.log(
                {"step": step, **{f"train/{k}": metrics[k] for k in ("loss", "l1", "ssim", "psnr", "lr", "reg", "scale_mean", "scale_max", "opacity_mean", "mem_gib", "mem_reserved_gib")}}
            )
        if "image" in metrics:
            caption = f"{metrics['image_scene']} | GT | render" if metrics.get("image_scene") else "GT | render"
            self.logger.log({"step": step, "train/image": wandb.Image(metrics["image"], caption=caption)})

    def save_checkpoint(self):
        """Save the full training state (to resume) and, separately, the model
        weights alone. `step` is the last *completed* step."""
        output_dir = Path(self.cfg.trainer.output_dir)
        name = f"step_{self.step:06d}.pt"
        (output_dir / "ckpts").mkdir(parents=True, exist_ok=True)
        (output_dir / "weights").mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), output_dir / "weights" / name)
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "wandb_run_id": self.logger.id if self.logger is not None else None,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            },
            output_dir / "ckpts" / name,
        )
        print(f"Saved checkpoint {output_dir / 'ckpts' / name}")

    def load_checkpoint(self, path):
        """Restore a full checkpoint from `save_checkpoint`. Call after
        `build_model`/`build_optimizer` and before `build_logger`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # The Gaussian set (outlier filter) and the model input depend on these;
        # resuming with different values would silently train on something else.
        # Fill fields added since the checkpoint was saved with their defaults,
        # so older checkpoints still resume.
        defaults = OmegaConf.structured(OmegaConf.get_type(self.cfg))
        saved = OmegaConf.to_container(OmegaConf.merge(defaults, ckpt["config"]), resolve=True)
        current = OmegaConf.to_container(self.cfg, resolve=True)
        for section, key in (
            ("trainer", "point_grid_size"),
            ("data", "parser"),
            ("data", "scenes"),
            ("data", "scene_root"),
            ("data", "gaussian_ckpt_template"),
            ("model", None),
            ("head", None),
            # The LR schedule is laid out over `num_steps` with these settings.
            ("trainer", "num_steps"),
            *(("optimizer", k) for k in current["optimizer"] if k in saved["optimizer"]),
        ):
            old = saved[section] if key is None else saved[section][key]
            new = current[section] if key is None else current[section][key]
            if old != new:
                raise ValueError(f"Config mismatch on resume for {section}{'.' + key if key else ''}: {old} != {new}")

        self.model.load_state_dict(ckpt["model"])
        # Keys `OneCycleLR` keeps in each param group; missing from optimizer
        # states saved before the scheduler existed, so restore them from the
        # fresh groups after loading.
        sched_keys = [{k: g[k] for k in ("initial_lr", "max_lr", "min_lr")} for g in self.optimizer.param_groups]
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_step = ckpt["step"] + 1
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Checkpoint from before the scheduler existed: fast-forward it.
            for group, keys in zip(self.optimizer.param_groups, sched_keys):
                group.update(keys)
                group["weight_decay"] = self.cfg.optimizer.weight_decay
            for _ in range(self.start_step):
                self.scheduler.step()
        self.wandb_run_id = ckpt["wandb_run_id"]

        # The saved RNG state is rank 0's; the other ranks keep their own
        # (per-rank seeded) streams so their augmentations stay distinct.
        if self.is_main:
            rng = ckpt["rng"]
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        print(f"Resumed from {path} at step {self.start_step}")

    # ---- data parallel helpers (no-ops / unused with a single process) --------

    def _comm_tensor(self, t):
        # Gloo is run on CPU copies; NCCL works on the GPU tensors directly.
        return t if self.cfg.trainer.dist_backend == "nccl" else t.cpu()

    def _broadcast(self, t):
        buf = self._comm_tensor(t.data)
        dist.broadcast(buf, src=0)
        if buf is not t.data:
            t.data.copy_(buf)

    def _all_reduce_grads(self):
        # One flat all-reduce for all gradients. Missing grads become zeros so
        # every rank sends a buffer of the same layout.
        params = [p for p in self.model.parameters() if p.requires_grad]
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        flat = self._comm_tensor(torch.cat([p.grad.reshape(-1) for p in params]))
        dist.all_reduce(flat)
        flat = flat.to(self.device) / self.world_size
        offset = 0
        for p in params:
            n = p.grad.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
            offset += n

    def _all_reduce_mean(self, values):
        buf = self._comm_tensor(torch.tensor(values, dtype=torch.float64, device=self.device))
        dist.all_reduce(buf)
        return (buf / self.world_size).tolist()

    def _barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def cleanup(self):
        if self.logger is not None:
            self.logger.finish()
        if self.world_size > 1:
            dist.destroy_process_group()

    def _to_device(self, batch):
        return [b.to(self.device) for b in batch]
