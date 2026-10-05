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


class LitePtGSModel(nn.Module):
    """LitePT backbone + a per-point linear head predicting Gaussian Splatting
    parameters for every input point.

    Outputs are *raw* (pre-activation) params in `simple_trainer.py`'s checkpoint
    convention -- log `scales`, unnormalized `quats`, logit `opacities`, `sh0`,
    `shN` -- with `exp`/normalize/`sigmoid` applied at render time.
    """

    def __init__(
        self,
        backbone_cfg: Dict[str, Any],
        sh_degree: int = 3,
        init_scale: float = 0.01,
        init_opacity: float = 0.1,
        max_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone = LitePT(**backbone_cfg)
        self.num_sh_rest = (sh_degree + 1) ** 2 - 1
        self.log_max_scale = math.log(max_scale)

        # LitePT's decoder ends at `dec_channels[0]` (one row per input point).
        out_channels = backbone_cfg["dec_channels"][0]
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

    def forward(self, point: Point) -> Dict[str, torch.Tensor]:
        feat = self.head(self.backbone(point).feat)
        scales, quats, opacities, sh0, shN = feat.split(self.split, dim=-1)
        # Bounded log-scale: `exp(scales) <= max_scale`. Unbounded, a few steps
        # of network updates can blow Gaussians up to cover whole images, and
        # rasterization's tile-intersection buffers grow with that (OOM).
        scales = self.log_max_scale + F.logsigmoid(scales)
        return {
            "scales": scales,                             # [N, 3]
            "quats": quats,                               # [N, 4]
            "opacities": opacities.squeeze(-1),           # [N]
            "sh0": sh0.view(-1, 1, 3),                    # [N, 1, 3]
            "shN": shN.view(-1, self.num_sh_rest, 3),     # [N, K-1, 3]
        }
