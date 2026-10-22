import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_LITEPT_ROOT = Path(__file__).resolve().parent.parent / "submodules" / "LitePT"
if str(_LITEPT_ROOT) not in sys.path:
    sys.path.append(str(_LITEPT_ROOT))

from litept.model import LitePT, Point  # noqa: E402

from models.pos_encoding import fourier_pos, pos_channels  # noqa: E402


class LitePtGSModel(nn.Module):
    """LitePT backbone + a per-point linear head predicting Gaussian Splatting
    parameters for every input point.

    Outputs are *raw* (pre-activation) params in `simple_trainer.py`'s checkpoint
    convention -- log `scales`, unnormalized `quats`, logit `opacities`, `sh0`,
    `shN` -- with `exp`/normalize/`sigmoid` applied at render time. With
    `max_offset > 0` it also predicts `offsets`, bounded world-space
    displacements added to the input points' positions.

    With `pos_encoding="fourier"`, Fourier features of the points' positions
    (`models/pos_encoding.py`) are added: concatenated to the input after the
    feature MLP (`pos_inject="input"`, widens LitePT's `in_channels`), added
    to the stem's output through a zero-initialized linear (`"embed"`), or
    both.

    `skip` adds a path from the backbone's input `x0` ([rgb | projected
    features | PE], before LitePT's stem) straight to the head: `"add"` adds
    a zero-initialized `Linear(x0)` to the backbone output, `"concat"`
    concatenates `GELU(Linear(x0))` to it. `hidden_dim > 0` puts a hidden
    layer (+ GELU) before the output linears. With `color_residual`, `sh0` is
    predicted as a residual on the input rgb (every Gaussian starts at its
    input colour instead of grey).
    """

    def __init__(
        self,
        backbone_cfg: Dict[str, Any],
        sh_degree: int = 3,
        init_scale: float = 0.01,
        init_opacity: float = 0.1,
        max_scale: float = 1.0,
        direct_channels: int = 3,
        feature_channels: int = 0,
        feature_mlp_dim: int = 64,
        feature_mlp_layers: int = 2,
        max_offset: float = 0.0,
        pos_encoding: str = "none",
        pos_freqs: int = 4,
        pos_percentile: float = 95.0,
        pos_include_raw: bool = True,
        pos_inject: str = "input",
        pos_norm: str = "none",
        pos_clamp_percentile: float = 0.0,
        skip: str = "none",
        skip_dim: int = 64,
        hidden_dim: int = 0,
        color_residual: bool = False,
    ):
        super().__init__()
        # Input `feat` = [direct_channels (rgb) | feature_channels (image features)].
        # Image features go through an MLP to `feature_mlp_dim` and are concatenated
        # back with the direct channels, so the backbone's `in_channels` must be
        # `direct_channels + feature_mlp_dim`. Without image features there is no
        # MLP (no extra weights), so rgb-only models and checkpoints are unchanged.
        self.direct_channels = direct_channels
        self.feature_mlp = None
        if feature_channels > 0:
            layers, width = [], feature_channels
            for i in range(feature_mlp_layers):
                layers.append(nn.Linear(width, feature_mlp_dim))
                if i < feature_mlp_layers - 1:
                    layers.append(nn.GELU())
                width = feature_mlp_dim
            self.feature_mlp = nn.Sequential(*layers)
        self.backbone = LitePT(**backbone_cfg)

        # Positional encoding (see the class docstring). With `"none"` nothing
        # below exists, so models and checkpoints are unchanged.
        self.pos_encoding = pos_encoding
        self.pos_freqs = pos_freqs
        self.pos_percentile = pos_percentile
        self.pos_include_raw = pos_include_raw
        self.pos_inject = pos_inject
        self.pos_clamp_percentile = pos_clamp_percentile
        self.pos_to_input = pos_encoding != "none" and pos_inject in ("input", "both")
        # Optional per-point LayerNorm over the PE channels (before concat / pe_proj):
        # bounds the raw-p tail (|p| up to ~46 on bonsai) like the other inputs.
        self.pe_norm = None
        if pos_encoding != "none" and pos_norm == "layer":
            self.pe_norm = nn.LayerNorm(pos_channels(pos_freqs, pos_include_raw))
        self.pe_proj = None
        if pos_encoding != "none" and pos_inject in ("embed", "both"):
            # Zero init: at step 0 the stem output is exactly that of the model
            # without this path.
            self.pe_proj = nn.Linear(pos_channels(pos_freqs, pos_include_raw), backbone_cfg["enc_channels"][0])
            nn.init.zeros_(self.pe_proj.weight)
            nn.init.zeros_(self.pe_proj.bias)
        # Last forward's percentile radius `r` (for logging); None without PE.
        self.pos_radius = None
        self.num_sh_rest = (sh_degree + 1) ** 2 - 1
        self.log_max_scale = math.log(max_scale)

        # LitePT's decoder ends at `dec_channels[0]` (one row per input point).
        out_channels = backbone_cfg["dec_channels"][0]

        # Input -> head skip (see the class docstring). With `"none"`,
        # `hidden_dim=0` and no `color_residual` nothing below exists, so
        # models and checkpoints are unchanged.
        if skip not in ("none", "add", "concat"):
            raise ValueError(f"skip must be none, add or concat, got {skip!r}")
        if color_residual and direct_channels != 3:
            raise ValueError("color_residual needs the rgb input channels (input.features with rgb)")
        self.skip = skip
        self.color_residual = color_residual
        self.skip_proj = None
        if skip == "add":
            # Zero init: at step 0 the head sees exactly the backbone output.
            self.skip_proj = nn.Linear(backbone_cfg["in_channels"], out_channels)
            nn.init.zeros_(self.skip_proj.weight)
            nn.init.zeros_(self.skip_proj.bias)
        elif skip == "concat":
            self.skip_proj = nn.Sequential(nn.Linear(backbone_cfg["in_channels"], skip_dim), nn.GELU())
            out_channels += skip_dim
        self.head_mlp = None
        if hidden_dim > 0:
            self.head_mlp = nn.Sequential(nn.Linear(out_channels, hidden_dim), nn.GELU())
            out_channels = hidden_dim

        self.split = [3, 4, 1, 3, self.num_sh_rest * 3]
        self.head = nn.Linear(out_channels, sum(self.split))

        # Start from the same constant init `create_splats_with_optimizers` uses
        # (identity rotation, `init_scale`, `init_opacity`, black SH): near-zero
        # weights so the first renders are well-behaved, biases carry the init.
        nn.init.normal_(self.head.weight, std=1e-3)
        bias = torch.zeros(sum(self.split))
        # Inverse of the bounded scale activation in `forward`.
        bias[0:3] = math.log(init_scale / (max_scale - init_scale))
        bias[3] = 1.0
        bias[7] = math.log(init_opacity / (1.0 - init_opacity))
        with torch.no_grad():
            self.head.bias.copy_(bias)
            # `shN` exactly zero until trained: rendered below its degree it
            # gets no gradient and stays zero, so raising the SH degree mid-run
            # starts from the current render instead of from random-weight
            # outputs on the trained features (which made the loss jump).
            self.head.weight[sum(self.split[:4]):].zero_()

        # Position offsets: a separate zero-initialized linear (not extra rows
        # of `head`), so with `max_offset=0` the model and its checkpoints are
        # unchanged, and with it every Gaussian starts exactly at its input
        # point. `max_offset * tanh` keeps each one inside a box around it.
        self.max_offset = max_offset
        self.offset_head = None
        if max_offset > 0:
            self.offset_head = nn.Linear(out_channels, 3)
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)

    def forward(self, point: Point, pos_radius=None) -> Dict[str, torch.Tensor]:
        """`pos_radius` (optional): the PE divider `r` to use instead of the
        percentile radius of `point.coord` (see `fourier_pos`)."""
        pe = None
        if self.pos_encoding != "none":
            # From the (augmented) coordinates the backbone sees.
            pe, self.pos_radius = fourier_pos(
                point.coord, self.pos_freqs, self.pos_percentile, self.pos_include_raw,
                self.pos_clamp_percentile, radius=pos_radius)
            if self.pe_norm is not None:
                pe = self.pe_norm(pe)
        if self.feature_mlp is not None:
            direct, image = point.feat.split(
                [self.direct_channels, point.feat.shape[1] - self.direct_channels], dim=1)
            point.feat = torch.cat([direct, self.feature_mlp(image)], dim=1)
        if self.pos_to_input:
            # [rgb | projected features | PE]
            point.feat = torch.cat([point.feat, pe.to(point.feat.dtype)], dim=1)
        # The backbone's input, kept for the skip / colour residual (LitePT
        # returns one row per input point, in the input order).
        x0 = point.feat
        if self.pe_proj is None:
            out = self.backbone(point).feat
        else:
            out = self._backbone_with_embed_pe(point, pe).feat
        if self.skip == "add":
            out = out + self.skip_proj(x0)
        elif self.skip == "concat":
            out = torch.cat([out, self.skip_proj(x0)], dim=1)
        if self.head_mlp is not None:
            out = self.head_mlp(out)
        feat = self.head(out)
        scales, quats, opacities, sh0, shN = feat.split(self.split, dim=-1)
        if self.color_residual:
            # Degree-0 SH of the input rgb: `rgb = 0.5 + C0 * sh0`.
            sh0 = sh0 + (x0[:, :3] - 0.5) / 0.28209479177387814
        # Bounded log-scale: `exp(scales) <= max_scale`. Unbounded, a few steps
        # of network updates can blow Gaussians up to cover whole images, and
        # rasterization's tile-intersection buffers grow with that (OOM).
        scales = self.log_max_scale + F.logsigmoid(scales)
        splats = {
            "scales": scales,                             # [N, 3]
            "quats": quats,                               # [N, 4]
            "opacities": opacities.squeeze(-1),           # [N]
            "sh0": sh0.view(-1, 1, 3),                    # [N, 1, 3]
            "shN": shN.view(-1, self.num_sh_rest, 3),     # [N, K-1, 3]
        }
        if self.offset_head is not None:
            # World units (the input `coord`'s frame); means = coord + offsets.
            splats["offsets"] = self.max_offset * torch.tanh(self.offset_head(out))  # [N, 3]
        return splats

    def _backbone_with_embed_pe(self, point: Point, pe: torch.Tensor) -> Point:
        """`LitePT.forward`, with `pe_proj(pe)` added to the stem's output."""
        bb = self.backbone
        if not isinstance(point, Point):
            point = Point(point)
        if bb.enc_attn[0]:
            point.serialization(order=bb.order, shuffle_orders=bb.shuffle_orders)
        point.sparsify()
        # The stem is a submanifold conv: one output row per input row, in input
        # order, so `pe` (computed from `point.coord`) stays aligned.
        point = bb.embedding(point)
        point.feat = point.feat + self.pe_proj(pe).to(point.feat.dtype)
        # Sparse convs read `sparse_conv_feat`, not `feat`.
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point = bb.enc(point)
        if not bb.enc_mode:
            point = bb.dec(point)
        return point
