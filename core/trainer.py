import sys
import math
import random
import torch
import numpy as np
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from fused_ssim import fused_ssim
from gsplat.rendering import rasterization
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

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
        elif key == "image_id":
            out[key] = torch.tensor(values)
        else:
            out[key] = values
    return out


@dataclass
class DataConfig:
    """Config for `Trainer`'s data pipeline: the scene parser plus train/eval
    dataset and dataloader settings, as one flat dataclass with `train_`/`eval_`
    prefixed fields rather than separate per-split sub-configs."""

    parser: ParserConfig = field(default_factory=lambda: ParserConfig(data_dir=MISSING))

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
    # Composite each training render over a random solid color (per view)
    # instead of black, so Gaussians can't lean on a black background for
    # dark/empty regions. Eval always renders over black.
    random_bkgd: bool = True
    sh_degree_interval: int = 1000
    near_plane: float = 0.01
    far_plane: float = 1e10


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.logger = None

        self.parser = None
        self.train_dataset = None
        self.eval_dataset = None
        self.train_loader = None
        self.eval_loader = None
        # Pinned CPU copies of the reference Gaussians' `means` (the model's fixed
        # input coordinates) and their base RGB colors (the input features),
        # attached to every batch in `train()`.
        self.means = None
        self.colors = None

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
        self.parser = Parser(data_cfg.parser)

        self.train_dataset = Dataset(self.parser, DatasetConfig(
            split="train",
            patch_size=data_cfg.train_patch_size,
            load_depths=data_cfg.train_load_depths,
        ))
        self.eval_dataset = Dataset(self.parser, DatasetConfig(
            split="val",
            patch_size=data_cfg.eval_patch_size,
            load_depths=data_cfg.eval_load_depths,
        ))

        self.train_loader = DataLoader(
            self.train_dataset,
            collate_fn=_collate_camera_batch,
            batch_size=data_cfg.train_batch_size,
            num_workers=data_cfg.train_num_workers,
            shuffle=data_cfg.train_shuffle,
            pin_memory=data_cfg.train_pin_memory,
            drop_last=data_cfg.train_drop_last,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            collate_fn=_collate_camera_batch,
            batch_size=data_cfg.eval_batch_size,
            num_workers=data_cfg.eval_num_workers,
            shuffle=data_cfg.eval_shuffle,
            pin_memory=data_cfg.eval_pin_memory,
            drop_last=data_cfg.eval_drop_last,
        )

        # Attached per batch from the main process (not in `Dataset.__getitem__`)
        # so the full cloud isn't shipped through the loader workers every item.
        if self.parser.gaussian_means is None:
            raise ValueError("data.parser.gaussian_ckpt_path must be set")

        # Drop far-away outlier Gaussians (MCMC puts a few up to ~9000 units out on bonsai)
        # so the voxelized cloud fits LitePT's serialization limit of 2^16 voxels
        # per axis (`Point.serialization`'s `depth <= 16` assert). All `gaussian_*`
        # arrays are filtered with the same mask to stay aligned.
        means = self.parser.gaussian_means
        half_extent = (2**16 - 1) * self.cfg.trainer.point_grid_size / 2
        keep = (np.abs(means - np.median(means, axis=0)) < half_extent).all(axis=1)
        for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            attr = f"gaussian_{name}"
            setattr(self.parser, attr, getattr(self.parser, attr)[keep])
        print(f"Removed {(~keep).sum()} / {len(keep)} outlier Gaussians")

        self.means = torch.from_numpy(self.parser.gaussian_means).float()
        # Degree-0 SH to RGB: `sh0 * C0 + 0.5`, with C0 the l=0 SH basis constant.
        sh0 = torch.from_numpy(self.parser.gaussian_sh0).float()[:, 0, :]
        self.colors = (sh0 * 0.28209479177387814 + 0.5).clamp(0.0, 1.0)
        if self.device.type == "cuda":
            self.means = self.means.pin_memory()
            self.colors = self.colors.pin_memory()

    def train(self):
        # Re-iterate the loader each epoch (reshuffles). `itertools.cycle` would
        # cache every batch of the first epoch in RAM and replay them unshuffled.
        train_iter = (batch for _ in itertools.count() for batch in self.train_loader)
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
            batch["means"] = self.means
            batch["colors"] = self.colors
            metrics = self.train_step(batch)
            self.log(metrics, self.step)

            if (self.step + 1) % self.cfg.trainer.eval_every == 0 or self.step == num_steps - 1:
                self.model.eval()
                self.eval()
                self.model.train()

            if (self.step + 1) % self.cfg.trainer.save_every == 0 or self.step == num_steps - 1:
                self.save_checkpoint()

    def train_step(self, batch):
        torch.cuda.reset_peak_memory_stats(self.device)
        coord = batch["means"].to(self.device, non_blocking=True)
        feat = batch["colors"].to(self.device, non_blocking=True)
        offset = torch.tensor([coord.shape[0]], device=self.device)
        point = Point(
            coord=coord,
            feat=feat,
            grid_size=self.cfg.trainer.point_grid_size,
            offset=offset,
        )

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

        camtoworlds = batch["camtoworld"].to(self.device, non_blocking=True)  # [B, 4, 4]
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

        loss_sum = l1_sum = ssim_sum = mse_sum = 0.0
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
            "mem_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
        }
        if image is not None:
            metrics["image"] = image
        return metrics

    @torch.no_grad()
    def eval(self):
        """Render every eval view and log mean PSNR/SSIM/LPIPS vs. ground truth."""
        # The model's weights are fixed during eval, so predict the Gaussians once
        # for the whole eval split.
        coord = self.means.to(self.device, non_blocking=True)
        point = Point(
            coord=coord,
            feat=self.colors.to(self.device, non_blocking=True),
            grid_size=self.cfg.trainer.point_grid_size,
            offset=torch.tensor([coord.shape[0]], device=self.device),
        )
        splats = self.model(point)
        splats = {
            "means": coord,
            "quats": splats["quats"],
            "scales": torch.exp(splats["scales"]),
            "opacities": torch.sigmoid(splats["opacities"]),
            "colors": torch.cat([splats["sh0"], splats["shN"]], dim=1),
        }

        metrics = {"psnr": [], "ssim": [], "lpips": [], "images": []}
        for batch in self.eval_loader:
            for k, v in self.eval_step(batch, splats).items():
                metrics[k].extend(v)
        images = metrics.pop("images")
        stats = {k: sum(v) / len(v) for k, v in metrics.items()}

        print(
            f"[eval step {self.step}] PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, "
            f"LPIPS: {stats['lpips']:.3f} ({len(metrics['psnr'])} images)"
        )
        self.logger.log({
            "step": self.step,
            **{f"eval/{k}": v for k, v in stats.items()},
            "eval/images": [wandb.Image(image, caption=caption) for image, caption in images],
        })

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
        wandb_cfg = self.cfg.wandb
        self.logger = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.name,
            mode=wandb_cfg.mode,
            config=OmegaConf.to_container(self.cfg, resolve=True),
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
        # `train/` prefix groups these into one "train" panel section in wandb.
        if step % self.cfg.trainer.log_train_steps == 0:
            print(f"[step {step}] { {k: v for k, v in metrics.items() if k != 'image'} }")
            self.logger.log(
                {"step": step, **{f"train/{k}": metrics[k] for k in ("loss", "l1", "ssim", "psnr", "lr")}}
            )
        if "image" in metrics:
            self.logger.log(
                {"step": step, "train/image": wandb.Image(metrics["image"], caption="GT | render")}
            )

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
        saved = ckpt["config"]
        current = OmegaConf.to_container(self.cfg, resolve=True)
        for section, key in (
            ("trainer", "point_grid_size"),
            ("data", "parser"),
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

        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        print(f"Resumed from {path} at step {self.start_step}")

    def _to_device(self, batch):
        return [b.to(self.device) for b in batch]
