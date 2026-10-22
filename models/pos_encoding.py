import math
from typing import Optional, Tuple

import torch


def pos_channels(num_freqs: int, include_raw: bool) -> int:
    """Width of `fourier_pos`'s output: sin + cos per axis and frequency (+ p)."""
    return 6 * num_freqs + (3 if include_raw else 0)


def _percentile(values: torch.Tensor, percentile: float) -> torch.Tensor:
    k = max(1, math.ceil(percentile / 100.0 * values.numel()))
    return values.kthvalue(k).values


@torch.no_grad()
def fourier_pos(
    coord: torch.Tensor, num_freqs: int, percentile: float, include_raw: bool,
    clamp_percentile: float = 0.0, radius: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fourier features of the input points' positions.

    Centering / normalization:
      - center: the origin. With `data.parser.normalize` the scene frame is
        already centered (the median of the COLMAP points sits at 0), and
        augmentation rotates / scales about that same origin, so no
        per-batch center is estimated.
      - scale: one isotropic divider for all three axes, `r` = the
        `percentile`-th percentile of the points' distances to the origin,
        computed on `coord` as given (i.e. *after* augmentation). Rotation
        preserves distances and a scale `s` multiplies all of them, so
        `r_aug = s * r` and `p = coord / r` only rotates with the cloud: the
        encoding is invariant to the augmentation's scale.
      - tail: with `clamp_percentile > 0`, points farther than that
        percentile's distance r_c are projected radially onto the sphere of
        radius r_c (direction kept, distance beyond r_c dropped), before both p
        and its sin / cos: everything is bounded by r_c / r. Distances only, so
        still consistent under augmentation. 0 = no clamp.
      - `radius` (optional) replaces the percentile `r`, e.g. the radius of the
        whole cloud when `coord` is only its frustum-culled part.

    Returns (enc [N, pos_channels(num_freqs, include_raw)], r scalar tensor).
    With p in [-1, 1] the encoding is unique: frequencies `2^k * pi`,
    k = 0..num_freqs-1, all have periods dividing 2. Without the clamp, points
    beyond `r` (100 - percentile % of them) are encoded as-is and wrap.
    """
    coord = coord.float()
    dist = coord.norm(dim=1)
    r = (_percentile(dist, percentile) if radius is None else torch.as_tensor(radius, device=coord.device)).clamp_min(1e-6)
    p = coord / r                                                    # [N, 3]
    if clamp_percentile > 0:
        r_c = _percentile(dist, clamp_percentile)
        p = p * (r_c / dist.clamp_min(1e-12)).clamp(max=1.0)[:, None]
    freqs = (2.0 ** torch.arange(num_freqs, device=coord.device, dtype=coord.dtype)) * math.pi
    ang = p[:, :, None] * freqs                                      # [N, 3, F]
    enc = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)       # [N, 6F]
    if include_raw:
        enc = torch.cat([p, enc], dim=1)
    return enc, r
