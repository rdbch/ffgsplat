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
    # Sky points (rows with `is_sky` in features.npz, from `edgs_extract_ref.py
    # --sky`). False: dropped, so scenes with and without them load the same points.
    # True: all of them are kept on top of the scene points (`max_points` and the
    # feature standardization use the scene points only, so the scene rows are the
    # same as with False), and the model gets them as sky (see `core.trainer`).
    sky: bool = False
    # Seed of the `max_points` subset (independent of the training RNG).
    seed: int = 0
    # How the `max_points` subset of the scene points is drawn: "random" (uniform) or
    # "complexity" (features.npz's `complexity`, from `edgs_extract_ref.py
    # --complexity_dir`): `subset_random_frac` of it uniform, the rest without
    # replacement with weights exp(complexity_beta * rho), rho = the complexity
    # clipped to its [complexity_clip, 100 - complexity_clip] percentiles over the
    # scene's points, then min-max normalized to [0, 1] (KoalaGS's weighting + the
    # clip: delentropy saturates at 5.0 with 32 x 32 windows and a few ~0 outliers
    # would squeeze everything else into rho ~ 0.9-1). Sky points are never subsampled.
    # "quantile": the complexity becomes its rank q in [0, 1] over the scene points;
    # first a floor: in each of `complexity_bins` equal-width bins over [min, max]
    # complexity, the min(complexity_floor, available) points with the highest keys
    # beta * q + Gumbel; then of the rest of the budget, `subset_random_frac` uniform and
    # the others the largest keys among the remaining points (= drawing without replacement
    # with probability ~ exp(complexity_beta * q)). If the floors alone exceed the budget,
    # the floors win (more points than `max_points`).
    subset: str = "random"
    subset_random_frac: float = 0.2
    complexity_beta: float = 1.0
    complexity_clip: float = 1.0
    complexity_bins: int = 50
    complexity_floor: int = 100
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
    # The PE divider r: "percentile" (`pos_percentile` of the distances, above) or
    # "sigma": |mu| + pos_sigma_k * sigma of one Gaussian fitted to the scene points
    # (mean mu, sigma = sqrt(tr(Sigma) / 3), sky points excluded), computed once per
    # scene in the training frame: the farthest distance from the origin of the
    # sphere at pos_sigma_k sigma around mu. With pos_sigma_k = the sky's sigma_k,
    # every point incl. the sky has |p| <= 1; the same r with or without `sky`.
    pos_radius: str = "percentile"
    pos_sigma_k: float = 4.0
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


def complexity_subset(complexity: np.ndarray, n: int, random_frac: float, beta: float,
                      rng: np.random.Generator, clip: float = 0.0) -> np.ndarray:
    """Sorted indices of `n` of the rows of `complexity`: n - round(random_frac * n) drawn
    without replacement with weights exp(beta * rho) (rho = complexity clipped to its
    [clip, 100 - clip] percentiles, then min-max normalized; Gumbel top-k), then the rest
    uniformly from the rows not drawn yet."""
    c = complexity.astype(np.float64)
    if clip > 0:
        c = np.clip(c, *np.percentile(c, [clip, 100.0 - clip]))
    rho = (c - c.min()) / (c.max() - c.min() + 1e-12)
    n_weighted = n - int(round(random_frac * n))
    keys = beta * rho + rng.gumbel(size=len(c))             # log weight + Gumbel noise
    weighted = np.argpartition(-keys, n_weighted - 1)[:n_weighted] if n_weighted > 0 else np.empty(0, int)
    rest = np.setdiff1d(np.arange(len(c)), weighted, assume_unique=True)
    uniform = rng.choice(rest, size=n - n_weighted, replace=False)
    return np.sort(np.concatenate([weighted, uniform]))


def quantile_subset(complexity: np.ndarray, n: int, random_frac: float, beta: float, bins: int,
                    floor: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices of the rows of `complexity` chosen by the quantile rule (see
    InputConfig.subset): rank q, keys beta * q + Gumbel, a floor of the top-key points per
    equal-width complexity bin, then (1 - random_frac) of the remaining budget by largest key
    and random_frac uniformly among the rest."""
    c = complexity.astype(np.float64)
    m = len(c)
    q = np.empty(m)
    q[np.argsort(c, kind="stable")] = np.arange(m) / max(1, m - 1)
    keys = beta * q + rng.gumbel(size=m)
    lo, hi = c.min(), c.max()
    b = np.minimum(((c - lo) / (max(hi - lo, 1e-12) / bins)).astype(np.int64), bins - 1)
    # Floor: rank within the bin by descending key, keep the first `floor`.
    order = np.lexsort((-keys, b))
    first = np.searchsorted(b[order], np.arange(bins))
    within = np.arange(m) - first[b[order]]
    floor_rows = order[within < floor]
    free = np.ones(m, dtype=bool)
    free[floor_rows] = False
    rem = max(0, n - len(floor_rows))
    n_weighted = rem - int(round(random_frac * rem))
    free_rows = np.flatnonzero(free)
    weighted = (free_rows[np.argpartition(-keys[free_rows], n_weighted - 1)[:n_weighted]]
                if n_weighted > 0 else np.empty(0, np.int64))
    free[weighted] = False
    rest = np.flatnonzero(free)
    uniform = rng.choice(rest, size=min(rem - n_weighted, len(rest)), replace=False)
    return np.sort(np.concatenate([floor_rows, weighted, uniform]))


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
    standardized or None, "feature_dims": {block: d}, "is_sky": [N] bool (sky rows
    last; all False unless `cfg.sky`), "info": {...}}.

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
    is_sky_all = npz["is_sky"] if "is_sky" in npz.files else np.zeros(len(valid), dtype=bool)
    if cfg.sky and not is_sky_all.any():
        raise ValueError(f"{scene_dir}: input.sky=true but features.npz has no sky points")
    sky_rows = rows[is_sky_all[rows]]
    rows = rows[~is_sky_all[rows]]
    if cfg.subset not in ("random", "complexity", "quantile"):
        raise ValueError(f"input.subset must be random, complexity or quantile, got {cfg.subset!r}")
    if cfg.subset != "random" and "complexity" not in npz.files:
        raise ValueError(f"{scene_dir}: input.subset={cfg.subset} but features.npz has no complexity "
                         f"(edgs_extract_ref.py --complexity_dir)")
    if cfg.max_points is not None and len(rows) > cfg.max_points:
        rng = np.random.default_rng(cfg.seed)
        if cfg.subset == "random":
            rows = np.sort(rng.choice(rows, size=cfg.max_points, replace=False))
        elif cfg.subset == "quantile":
            rows = rows[quantile_subset(npz["complexity"][rows], cfg.max_points, cfg.subset_random_frac,
                                        cfg.complexity_beta, cfg.complexity_bins, cfg.complexity_floor, rng)]
        else:
            rows = rows[complexity_subset(npz["complexity"][rows], cfg.max_points,
                                          cfg.subset_random_frac, cfg.complexity_beta, rng,
                                          clip=cfg.complexity_clip)]
    num_scene = len(rows)
    if cfg.sky:
        rows = np.concatenate([rows, sky_rows])          # sky rows last
    is_sky = is_sky_all[rows]

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
        # variances), with this scene's own statistics (scene points only).
        ref = features[:num_scene]
        features = (features - ref.mean(0)) / (ref.std(0) + 1e-6)

    if "complexity" in npz.files:
        # Median complexity of all valid scene points vs the ones used (the subset's bias).
        cplx = npz["complexity"]
        scene_valid = np.flatnonzero(valid & ~is_sky_all) if cfg.drop_invalid else np.flatnonzero(~is_sky_all)
        cplx_info = {"complexity_median_all": float(np.median(cplx[scene_valid])),
                     "complexity_median_used": float(np.median(cplx[rows[:num_scene]]))}
    else:
        cplx_info = {}
    info = {
        **cplx_info,
        "num_total": int(len(valid)),
        "num_valid": int(valid.sum()),
        "num_used": int(len(rows)),
        "num_sky": int(is_sky.sum()),
        "subset": cfg.subset,
        "align_median_px": align_px,
        "align_p90_px": float(np.percentile(errors, 90)),
        "pca_basis_sha1": meta.get("pca_basis_sha1"),
    }
    return {"means": means, "rgb": rgb, "features": features, "feature_dims": feature_dims,
            "is_sky": is_sky, "info": info}
