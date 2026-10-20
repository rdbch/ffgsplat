"""EDGS point clouds (+ per-point RoMa features) as the model's input points.

Reads an `edgs_extract_ref.py` output folder (submodules/EDGS):

    <dir>/point_cloud/iteration_0/point_cloud.ply   EDGS init, raw COLMAP coordinates
    <dir>/features.npz       feat (N, D) f16, ref_cam (N,), ref_uv (N, 2), valid (N,)
    <dir>/features_meta.json block column spans, camera names, ...

Row i of `features.npz` is row i of the PLY.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData

from .normalize import transform_points

SH_C0 = 0.28209479177387814
FEATURE_BLOCKS = ("roma1", "roma2", "roma4", "roma16")


@dataclass
class InputConfig:
    """What the model gets as input points and per-point features. The defaults
    reproduce the original setup (MCMC reference Gaussians' means + base RGB)."""

    # "mcmc": means/colours of the reference checkpoint (`data.parser.gaussian_ckpt_path`
    # / `data.gaussian_ckpt_template`). "edgs": EDGS points from `edgs_dir`.
    source: str = "mcmc"
    # EDGS output folder; `{scene}` is replaced by the scene name.
    edgs_dir: str = "/projects/hpc/radub/experiments/edgs_feat/{scene}"
    # Per-point input features: "rgb" (base colour, 3 ch) and, with source="edgs",
    # any of the RoMa feature blocks roma1 / roma2 / roma4 / roma16 (DINOv2).
    features: List[str] = field(default_factory=lambda: ["rgb"])
    # Fixed random subset of this many points (after dropping invalid ones and
    # before the outlier filter); None = all.
    max_points: Optional[int] = None
    # Per-scene `max_points` (multi-scene runs), e.g. {bicycle: 1500000}.
    scene_max_points: Dict[str, int] = field(default_factory=dict)
    # Drop EDGS points that failed the reprojection test (zero opacity in the PLY).
    drop_invalid: bool = True
    # Seed of the `max_points` subset (independent of the training RNG).
    seed: int = 0
    # Feature blocks go through an MLP before LitePT: standardize -> Linear(D, dim)
    # -> GELU -> ... -> Linear(dim, dim), concatenated with rgb. Number of Linear layers.
    feature_mlp_dim: int = 64
    feature_mlp_layers: int = 2
    # Positional encoding of the input points (`models/pos_encoding.py`):
    # "none" | "fourier". p = coord / r, centered at the scene origin (needs
    # `data.parser.normalize`), with one divider r for all axes: the
    # `pos_percentile`-th percentile of the points' distances to the origin,
    # computed after augmentation. Encoded as sin / cos(2^k pi p), k < pos_freqs,
    # plus p itself if `pos_include_raw`.
    pos_encoding: str = "none"
    pos_freqs: int = 4
    pos_percentile: float = 95.0
    pos_include_raw: bool = True
    # Where it enters the model: "input" (concatenated after the feature MLP,
    # widens LitePT's in_channels), "embed" (added after LitePT's stem through a
    # zero-initialized linear) or "both".
    pos_inject: str = "input"
    # "layer": per-point LayerNorm over the PE channels before they enter the model
    # (bounds the raw-p tail); "none": as computed.
    pos_norm: str = "none"
    # Radial clamp of the tail: points farther than this percentile of the
    # distances (after augmentation) are projected onto that sphere before
    # encoding (p and sin / cos bounded by r_c / r). 0 = off.
    pos_clamp_percentile: float = 0.0
    # Alignment check: fail if the median distance (pixels) between the projected
    # EDGS points and the pixels they were triangulated from exceeds this.
    max_align_error_px: float = 3.0


def validate_input_config(cfg: InputConfig) -> None:
    if cfg.source not in ("mcmc", "edgs"):
        raise ValueError(f"input.source must be 'mcmc' or 'edgs', got {cfg.source!r}")
    unknown = set(cfg.features) - {"rgb", *FEATURE_BLOCKS}
    if unknown:
        raise ValueError(f"input.features: unknown {sorted(unknown)}; valid: rgb, {', '.join(FEATURE_BLOCKS)}")
    if not cfg.features:
        raise ValueError("input.features is empty")
    if cfg.source == "mcmc" and list(cfg.features) != ["rgb"]:
        raise ValueError("input.source=mcmc only provides features=[rgb]")
    if cfg.pos_encoding not in ("none", "fourier"):
        raise ValueError(f"input.pos_encoding must be 'none' or 'fourier', got {cfg.pos_encoding!r}")
    if cfg.pos_clamp_percentile and not cfg.pos_percentile <= cfg.pos_clamp_percentile <= 100:
        raise ValueError(f"input.pos_clamp_percentile must be 0 (off) or in [pos_percentile, 100], "
                         f"got {cfg.pos_clamp_percentile}")
    if cfg.pos_norm not in ("none", "layer"):
        raise ValueError(f"input.pos_norm must be 'none' or 'layer', got {cfg.pos_norm!r}")
    if cfg.pos_inject not in ("input", "embed", "both"):
        raise ValueError(f"input.pos_inject must be 'input', 'embed' or 'both', got {cfg.pos_inject!r}")
    if cfg.pos_encoding != "none" and not (cfg.pos_freqs >= 1 and 0 < cfg.pos_percentile <= 100):
        raise ValueError(f"input.pos_freqs must be >= 1 and pos_percentile in (0, 100], "
                         f"got {cfg.pos_freqs}, {cfg.pos_percentile}")


def feature_layout(cfg: InputConfig) -> Tuple[int, List[str]]:
    """(number of direct channels = 3 if rgb else 0, feature blocks in input order)."""
    return (3 if "rgb" in cfg.features else 0), [b for b in FEATURE_BLOCKS if b in cfg.features]


def _project(points, camtoworld, K):
    """World points [N, 3] -> pixel coords [N, 2] and depth [N] (OpenCV camera)."""
    w2c = np.linalg.inv(camtoworld)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    pix = cam @ K.T
    return pix[:, :2] / pix[:, 2:3], cam[:, 2]


def load_edgs_points(cfg: InputConfig, scene_dir: str, parser, train_indices) -> Dict:
    """Load one scene's EDGS points in the parser's (normalized) frame.

    Returns {"means": [N, 3] f32, "rgb": [N, 3] f32 in [0, 1], "features": [N, D] f32
    standardized or None, "feature_dims": {block: d}, "info": {...}}.

    Checks (raise on failure):
      - the EDGS reference cameras are exactly the parser's training views, so held-out
        views never contributed points or features;
      - projecting the points with the parser's cameras lands on the pixels EDGS
        triangulated them from (same world frame, same normalization).
    """
    with open(os.path.join(scene_dir, "features_meta.json")) as f:
        meta = json.load(f)
    npz = np.load(os.path.join(scene_dir, "features.npz"))
    valid, ref_cam, ref_uv = npz["valid"], npz["ref_cam"], npz["ref_uv"]

    # --- split check: EDGS camera i <-> parser image -----------------------------
    # Compare without extensions: EDGS keeps COLMAP's names ("DSCF5566.JPG"), the
    # parser may point at converted images.
    stem = lambda name: os.path.splitext(os.path.basename(name))[0]
    train_names = [stem(parser.image_names[i]) for i in train_indices]
    edgs_names = [stem(name) for name in meta["camera_names"]]
    if edgs_names != train_names:
        missing = sorted(set(edgs_names) ^ set(train_names))[:5]
        raise ValueError(
            f"{scene_dir}: EDGS cameras ({len(edgs_names)}) != parser training views "
            f"({len(train_names)}); e.g. {missing}. Re-extract with --holdout_test and "
            f"the same test_every.")
    edgs_to_parser = np.asarray(train_indices)          # EDGS camera index -> parser image index

    # --- point subset ----------------------------------------------------------------
    rows = np.flatnonzero(valid) if cfg.drop_invalid else np.arange(len(valid))
    if cfg.max_points is not None and len(rows) > cfg.max_points:
        rng = np.random.default_rng(cfg.seed)
        rows = np.sort(rng.choice(rows, size=cfg.max_points, replace=False))

    vertex = PlyData.read(meta["ply_path"])["vertex"].data
    if len(vertex) != len(valid):
        raise ValueError(f"{scene_dir}: PLY has {len(vertex)} rows, features.npz {len(valid)}")
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], 1)[rows].astype(np.float64)
    means = transform_points(parser.transform, xyz).astype(np.float32)
    dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], 1)[rows]
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0).astype(np.float32)

    # --- alignment check: project into the reference view ------------------------------
    sample = np.random.default_rng(0).choice(len(rows), size=min(20_000, len(rows)), replace=False)
    errors = []
    for cam in np.unique(ref_cam[rows[sample]]):
        sel = sample[ref_cam[rows[sample]] == cam]
        index = edgs_to_parser[cam]
        camera_id = parser.camera_ids[index]
        width, height = parser.imsize_dict[camera_id]
        pix, depth = _project(means[sel], parser.camtoworlds[index], parser.Ks_dict[camera_id])
        # EDGS stores [-1, 1] coordinates with -1 / +1 at the image's outer edges.
        target = (ref_uv[rows[sel]] + 1.0) * 0.5 * np.array([width, height])
        errors.append(np.linalg.norm(pix - target, axis=1)[depth > 0])
    errors = np.concatenate(errors)
    align_px = float(np.median(errors))
    if not align_px <= cfg.max_align_error_px:
        raise ValueError(
            f"{scene_dir}: EDGS points don't line up with the parser's cameras: median "
            f"reprojection error {align_px:.2f} px > {cfg.max_align_error_px} px "
            f"(different normalization / image factor / COLMAP model?)")

    # --- features -------------------------------------------------------------------
    _, blocks = feature_layout(cfg)
    features, feature_dims = None, {}
    if blocks:
        spans = meta["blocks"]
        missing = [b for b in blocks if b not in spans]
        if missing:
            raise ValueError(f"{scene_dir}: no feature block(s) {missing}; has {sorted(spans)}")
        feat = npz["feat"][rows]
        features = np.concatenate([feat[:, spans[b][0]:spans[b][1]] for b in blocks], 1).astype(np.float32)
        feature_dims = {b: spans[b][1] - spans[b][0] for b in blocks}
        # Per-dimension standardization (the PCA components have very different
        # variances), with this scene's own statistics.
        features = (features - features.mean(0)) / (features.std(0) + 1e-6)

    info = {
        "num_total": int(len(valid)),
        "num_valid": int(valid.sum()),
        "num_used": int(len(rows)),
        "align_median_px": align_px,
        "align_p90_px": float(np.percentile(errors, 90)),
        "pca_basis_sha1": meta.get("pca_basis_sha1"),
    }
    return {"means": means, "rgb": rgb, "features": features, "feature_dims": feature_dims, "info": info}
