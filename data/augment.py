import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class AugmentConfig:
    """Per-step random similarity transform of the scene (points + cameras),
    applied jointly so the rendered images are unchanged; the network has to
    predict the correspondingly transformed quats / scales / SH. Train only --
    eval always sees the canonical (normalized) scene.

    No translation option: LitePT voxelizes `coord - coord.min(0)`, so a global
    shift is an exact no-op for the network. `voxel_jitter` is the meaningful
    part of a translation (the sub-voxel phase of the grid).
    """

    enabled: bool = False
    # Rotation about the up axis (z after `data.normalize`'s PCA alignment),
    # uniform in [-z_rot_deg, z_rot_deg]. 180 = any heading.
    z_rot_deg: float = 180.0
    # Small rotations about x and y, each uniform in [-tilt_deg, tilt_deg].
    tilt_deg: float = 5.0
    # Uniform scale, log-uniform in [scale_min, scale_max]. Changes points per
    # voxel at a fixed `point_grid_size`; predicted scales must follow by `s`.
    scale_min: float = 0.8
    scale_max: float = 1.25
    # Random sub-voxel offset of LitePT's voxel grid, uniform in [0, 1) voxel
    # per axis.
    voxel_jitter: bool = True
    # Probability of mirroring x (x -> -x) before the rotation. With the full z
    # rotation this covers every mirror about a vertical plane. Points and
    # cameras are mirrored together, so images stay unchanged (gsplat renders
    # bit-identically with the det = -1 camera rotation); the network has to
    # learn the mirrored covariances and the sign flip of x-odd SH coefficients.
    mirror_prob: float = 0.0


def max_expansion(cfg: AugmentConfig) -> float:
    """Upper bound on how much the augmentation can grow the cloud's
    axis-aligned extent (rotation of a cube, then scale). Takes the dataclass
    or its OmegaConf node."""
    if not cfg.enabled:
        return 1.0
    if cfg.tilt_deg > 0:
        rot = math.sqrt(3.0)
    elif cfg.z_rot_deg > 0:
        rot = math.sqrt(2.0)
    else:
        rot = 1.0
    return rot * max(cfg.scale_max, 1.0)


def _axis_rotation(axis: int, angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    i, j = [a for a in range(3) if a != axis]
    R = torch.eye(3, device=angle.device, dtype=angle.dtype)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def sample_similarity(cfg: AugmentConfig, device: torch.device) -> Tuple[torch.Tensor, float]:
    """Random orthogonal `R` [3, 3] (a rotation, times an x-mirror with
    probability `cfg.mirror_prob`) and scale `s` from `cfg`: x -> s * R @ x."""

    def uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.empty((), device=device).uniform_(lo, hi)

    z = math.radians(cfg.z_rot_deg)
    tilt = math.radians(cfg.tilt_deg)
    R = (
        _axis_rotation(2, uniform(-z, z))
        @ _axis_rotation(1, uniform(-tilt, tilt))
        @ _axis_rotation(0, uniform(-tilt, tilt))
    )
    if cfg.mirror_prob > 0 and torch.rand(()).item() < cfg.mirror_prob:
        R[:, 0] = -R[:, 0]  # R @ diag(-1, 1, 1)
    s = math.exp(uniform(math.log(cfg.scale_min), math.log(cfg.scale_max)).item())
    return R, s


def apply_similarity(
    R: torch.Tensor, s: float, means: torch.Tensor, camtoworlds: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map points [N, 3] and OpenCV cam-to-world [B, 4, 4] through x -> s R x.

    `R` may be a reflection (det = -1). Camera rotations become `R @ R_c` and
    centers `s R t_c`, so every point
    keeps its pixel location (depths scale by `s`, which perspective projection
    divides out); intrinsics and images are unchanged.
    """
    means = s * means @ R.T
    camtoworlds = camtoworlds.clone()
    camtoworlds[:, :3, :3] = R @ camtoworlds[:, :3, :3]
    camtoworlds[:, :3, 3] = s * camtoworlds[:, :3, 3] @ R.T
    return means, camtoworlds


def jittered_grid_coord(coord: torch.Tensor, grid_size: float) -> torch.Tensor:
    """LitePT's `grid_coord` (`Point.serialization`'s default) with the grid
    origin shifted by a random sub-voxel offset."""
    offset = torch.rand(3, device=coord.device, dtype=coord.dtype) * grid_size
    return torch.div(coord - coord.min(0)[0] + offset, grid_size, rounding_mode="trunc").int()


def augment(
    cfg: AugmentConfig, means: torch.Tensor, camtoworlds: torch.Tensor, grid_size: float
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return (means, camtoworlds, grid_coord-or-None); identity if disabled."""
    if not cfg.enabled:
        return means, camtoworlds, None
    R, s = sample_similarity(cfg, means.device)
    means, camtoworlds = apply_similarity(R, s, means, camtoworlds)
    grid_coord = jittered_grid_coord(means, grid_size) if cfg.voxel_jitter else None
    return means, camtoworlds, grid_coord
