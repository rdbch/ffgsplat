"""`Trainer` on the DL3DV benchmark scenes (`data.dl3dv`), with two evals.

Differences from `core.trainer.Trainer`'s multi-scene mode:

- Scenes come from a fixed split file (`dl3dv.split_file`, `python -m data.dl3dv`):
  `train` scenes are trained on; `eval_seen` (a subset of them) and
  `eval_unseen` (never trained on) are evaluated on their held-out views
  (every `test_every`-th), logged as `eval_seen/...` and `eval_unseen/...`
  (per scene, mean, and mean per environment).
- Scene folders are `<dl3dv.root>/<hash>/<dl3dv.scene_subdir>` (only `images_4`,
  no full-resolution `images`; the parser maps them by name).
- Input points: all valid EDGS points of every training scene stay in RAM (fp16
  features, ~1 GB per scene), and every training micro-batch draws a fresh random
  subset of `dl3dv.points_bounded` / `points_unbounded` points (by the scene's
  environment). Eval always uses one fixed subset per scene (seed `input.seed`),
  the same at every eval and in every run. `input.max_points` /
  `scene_max_points` are not used.
- Scale normalization: on top of the parser's normalization, every scene is
  scaled (points and cameras together, images unchanged) so that the 95th
  percentile of its COLMAP points' distances to the origin is
  `dl3dv.target_radius` (3.56 = bonsai's EDGS r95, the scale the 017 recipe's
  0.005 voxels were tuned on). Without it DL3DV scenes are ~2x larger, LitePT
  keeps far more voxels per stage, and a single forward ran out of memory on a
  full H200 (job 765). The scale comes from the COLMAP points, so it is known
  before any EDGS points are loaded; the EDGS radius after scaling is printed
  per scene.
- `dl3dv.lazy` (debugging): trains on the first `dl3dv.lazy_train_scenes`
  training scenes only, and builds parsers only for those and the eval scenes;
  a scene's points are read from its EDGS folder the first time a batch or an
  eval needs them (~2 GB, a few seconds). Same points, same scales and same eval
  subsets as the default (everything loaded up front).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import Config
from core.trainer import (
    Scene,
    Trainer,
    _collate_camera_batch,
    _MultiSceneDataset,
    _RandomSceneBatchSampler,
    frustum_mask,
    percentile_radius,
)
from data.augment import max_expansion
from data.datasets.colmap import Dataset, DatasetConfig, Parser
from data.edgs_points import load_edgs_points


@dataclass
class DL3DVConfig:
    root: str = "/projects/hpc/radub/datasets/dl3dv"
    # Each scene's COLMAP folder inside `<root>/<hash>`.
    scene_subdir: str = "gaussian_splat"
    # Fixed split (train / eval_seen / eval_unseen / environment), from `python -m data.dl3dv`.
    split_file: str = "data/splits/dl3dv_133.json"
    # Input points per forward, by the scene's environment in the DL3DV meta.
    points_bounded: int = 1_000_000
    points_unbounded: int = 1_250_000
    # True: a fresh random subset per training micro-batch. False: the fixed
    # eval subset in training too.
    resample_points: bool = True
    # Scenes loaded in parallel at startup.
    load_threads: int = 8
    # Every scene is scaled so its COLMAP points' 95th-percentile radius is this.
    # null = no extra scaling (the parser's normalization only).
    target_radius: Optional[float] = 3.56
    # Train on a subset of the split's training scenes: only this environment
    # (bounded | unbounded) and/or only the first `train_limit` (sorted by hash).
    # With either set, eval_seen = exactly these training scenes.
    train_environment: Optional[str] = None
    train_limit: Optional[int] = None
    # Which of the split's lists to train on: "train" (120 scenes) or
    # "eval_seen" (its 10 scenes, 5 bounded / 5 unbounded; eval_seen = them).
    # `train_environment` / `train_limit` filter that list.
    train_on: str = "train"
    # Evaluate only the first `eval_unseen_limit` of the split's eval_unseen scenes
    # (null = all 13).
    eval_unseen_limit: Optional[int] = None
    # Debugging: train on the first `lazy_train_scenes` training scenes only and
    # load each scene's points on first use instead of all at startup.
    lazy: bool = False
    lazy_train_scenes: int = 4
    # Draw / frustum-cull / gather each micro-batch's input points in the train
    # loader's workers (CPU, in parallel, overlapped with the GPU step) instead
    # of the main process: the main process only copies the (culled) points to
    # the GPU. The workers are forked after the scenes are loaded and share
    # their points. Ignored with `lazy` (each worker would load its own copy).
    points_in_workers: bool = False


@dataclass
class DL3DVTrainConfig(Config):
    dl3dv: DL3DVConfig = field(default_factory=DL3DVConfig)


@dataclass
class DL3DVScene(Scene):
    """A `Scene` whose `means` / `feat` (fp16) hold all of its valid points
    (None until loaded, see `_load_points`); `num_points` of them are used per
    forward."""

    environment: str = ""
    num_points: int = 0
    # Extra scale applied on top of the parser's normalization (`target_radius`).
    scale: float = 1.0
    # Fixed eval subset (fp32 means / feat), set on first use (`_eval_view`).
    eval_view: Optional["DL3DVScene"] = None
    # PE radius of all of `means` (frustum_cull="full"), set on first use.
    pos_radius: Optional[float] = None
    # Sky rows (`input.sky`): the last `num_sky` rows of `means` / `feat`, all of
    # them in every forward on top of the `num_points` scene points.
    num_sky: int = 0
    # Sky rows dropped by the outlier filter (beyond the voxel grid's extent).
    num_sky_removed: int = 0


def _sky_rows(scene):
    """Row indices of `scene`'s sky points (the last `num_sky` rows)."""
    n = scene.means.shape[0]
    return torch.arange(n - scene.num_sky, n)


def _eval_rows(scene, seed):
    """The fixed subset of `scene`'s points used for eval (same every eval / run),
    + all sky rows."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(scene.means.shape[0] - scene.num_sky, size=scene.num_points, replace=False))
    return torch.cat([torch.from_numpy(rows), _sky_rows(scene)])


def _sample_points(scene, batch, args):
    """A random subset of `scene.num_points` of the scene's points (the fixed
    eval subset without `args["resample"]`), frustum-culled to `batch`'s views
    with `args["cull"]`: "subset" culls the subset, "full" culls all points and
    draws the subset from the visible ones (resampled only). On the CPU; only
    the kept rows are gathered. Sky rows (`scene.num_sky`) all go in on top of
    the subset (culled the same way). Returns (means fp32, feat fp16, PE radius
    of the cloud before culling or None, kept fraction); the PE radius is
    `scene.fixed_pos_radius` when set (`input.pos_radius=sigma`)."""
    cull, pe = args["cull"], args["pe_percentile"]
    if scene.fixed_pos_radius is not None:
        pe = None
    num_scene = scene.means.shape[0] - scene.num_sky
    pos_radius, keep_frac = None, 1.0
    if cull != "none":
        height, width = batch["image"].shape[1:3]
        mask = lambda means: frustum_mask(means, batch["camtoworld"], batch["K"], height, width,
                                          args["margin"], args["near_plane"])
    if cull == "full" and args["resample"]:
        visible = mask(scene.means).nonzero().squeeze(1)
        # A uniform subset of the whole cloud has (in expectation) the whole
        # cloud's percentile radius.
        if pe is not None and scene.pos_radius is None:
            scene.pos_radius = percentile_radius(scene.means, pe)
        pos_radius, keep_frac = scene.pos_radius, len(visible) / len(scene.means)
        vis_sky = visible[visible >= num_scene]
        visible = visible[visible < num_scene]
        rows = torch.cat([visible[torch.randperm(len(visible))[: scene.num_points]], vis_sky])
    else:
        if args["resample"]:
            rows = torch.cat([torch.randperm(num_scene)[: scene.num_points], _sky_rows(scene)])
        else:
            rows = _eval_rows(scene, args["seed"])
        if cull != "none":
            sub_means = scene.means[rows]
            keep = mask(sub_means)
            pos_radius = percentile_radius(sub_means, pe) if pe is not None else None
            keep_frac = keep.float().mean().item()
            rows = rows[keep]
    if scene.fixed_pos_radius is not None:
        pos_radius = scene.fixed_pos_radius
    return scene.means[rows], scene.feat[rows], pos_radius, keep_frac


class _PointsCollate:
    """Train loader collate (`dl3dv.points_in_workers`): the camera batch plus
    its input points (`_sample_points`), so the draw / cull / gather run in
    the loader workers. The workers are forked after the scenes are loaded,
    so `scenes`' points are shared, not copied."""

    def __init__(self, scenes, args):
        self.scenes = scenes
        self.args = args

    def __call__(self, items):
        batch = _collate_camera_batch(items)
        scene = self.scenes[int(batch["scene"][0])]
        batch["means"], batch["feat"], batch["pos_radius"], batch["cull_keep_frac"] = _sample_points(scene, batch, self.args)
        return batch


class TrainerDL3DV(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.multi_scene = True
        with open(cfg.dl3dv.split_file) as f:
            self.split = json.load(f)
        if set(self.split["eval_unseen"]) & set(self.split["train"]):
            raise ValueError(f"{cfg.dl3dv.split_file}: eval_unseen scenes are in train")
        if not set(self.split["eval_seen"]) <= set(self.split["train"]):
            raise ValueError(f"{cfg.dl3dv.split_file}: eval_seen scenes must be in train")
        self.eval_seen = []
        self.eval_unseen = []

    def build_dataloaders(self):
        data_cfg, dl3dv = self.cfg.data, self.cfg.dl3dv
        if dl3dv.train_on not in ("train", "eval_seen"):
            raise ValueError(f"dl3dv.train_on must be train or eval_seen, got {dl3dv.train_on!r}")
        train_names = self.split[dl3dv.train_on]
        seen_names = self.split["eval_seen"]
        unseen_names = self.split["eval_unseen"][: dl3dv.eval_unseen_limit]
        if dl3dv.train_environment is not None or dl3dv.train_limit is not None:
            if dl3dv.train_environment is not None:
                train_names = [n for n in train_names if self.split["environment"][n] == dl3dv.train_environment]
            if dl3dv.train_limit is not None:
                train_names = train_names[: dl3dv.train_limit]
            seen_names = train_names
        if dl3dv.lazy:
            train_names = train_names[: dl3dv.lazy_train_scenes]
        names = train_names + [n for n in seen_names if n not in train_names] + unseen_names
        with ThreadPoolExecutor(dl3dv.load_threads) as pool:
            scenes = list(tqdm(pool.map(self._build_dl3dv_scene, names), total=len(names), desc="DL3DV parsers"))
        by_name = dict(zip(names, scenes))

        # One feature width and one PCA basis across all scenes (from their metadata).
        widths = {width for _, width, _ in scenes}
        bases = {sha for _, _, sha in scenes}
        if len(widths) != 1 or len(bases) != 1:
            raise ValueError(f"scenes differ in feature width {widths} / PCA basis {bases}")
        # + the sky flag (`input.sky`), the last feature channel as in `Trainer`.
        self.feature_channels = widths.pop() + int(self.cfg.input.sky)

        self.scenes = [by_name[name][0] for name in train_names]
        self.eval_seen = [by_name[name][0] for name in seen_names]
        self.eval_unseen = [by_name[name][0] for name in unseen_names]
        if not dl3dv.lazy:
            with ThreadPoolExecutor(dl3dv.load_threads) as pool:
                to_load = self.scenes + self.eval_unseen
                list(tqdm(pool.map(self._load_points, to_load), total=len(to_load), desc="DL3DV points"))
            for scene in self.eval_seen + self.eval_unseen:
                self._eval_view(scene)
            # The unseen scenes only need their eval subset.
            for scene in self.eval_unseen:
                scene.means = scene.feat = None
            total = sum(s.means.numel() * 4 + s.feat.numel() * 2 for s in self.scenes) / 2**30
            print(f"DL3DV: {len(self.scenes)} training scenes ({total:.1f} GiB of points in RAM), "
                  f"{len(self.eval_seen)} eval_seen, {len(self.eval_unseen)} eval_unseen")
        else:
            print(f"DL3DV (lazy): {len(self.scenes)} training scenes (the first {dl3dv.lazy_train_scenes}), "
                  f"{len(self.eval_seen)} eval_seen, {len(self.eval_unseen)} eval_unseen; points loaded on first use")

        in_workers = dl3dv.points_in_workers and not dl3dv.lazy and data_cfg.train_num_workers > 0
        if in_workers:
            print(f"DL3DV: input points drawn / culled ({self.cfg.trainer.frustum_cull}) in the "
                  f"{data_cfg.train_num_workers} train loader workers")
        self.train_loader = DataLoader(
            _MultiSceneDataset([scene.train_dataset for scene in self.scenes]),
            collate_fn=_PointsCollate(self.scenes, self._points_args()) if in_workers else _collate_camera_batch,
            batch_sampler=_RandomSceneBatchSampler(
                [len(scene.train_dataset) for scene in self.scenes], data_cfg.train_batch_size,
                positions=self._view_positions(),
            ),
            num_workers=data_cfg.train_num_workers,
            pin_memory=data_cfg.train_pin_memory,
        )

    def _build_dl3dv_scene(self, name):
        """Parser, datasets and eval loader of one scene, without its points.
        Returns (DL3DVScene, feature width, PCA basis sha1); the last two from
        the EDGS metadata."""
        data_cfg, dl3dv = self.cfg.data, self.cfg.dl3dv
        parser = Parser(OmegaConf.merge(data_cfg.parser, {
            "data_dir": str(Path(dl3dv.root) / name / dl3dv.scene_subdir),
        }))
        scale = 1.0
        if dl3dv.target_radius is not None:
            scale = dl3dv.target_radius / float(np.percentile(np.linalg.norm(parser.points, axis=1), 95))
            self._rescale_parser(parser, scale)
        train_dataset = Dataset(parser, DatasetConfig(
            split="train", patch_size=data_cfg.train_patch_size, load_depths=data_cfg.train_load_depths,
        ))
        eval_dataset = Dataset(parser, DatasetConfig(
            split="val", patch_size=data_cfg.eval_patch_size, load_depths=data_cfg.eval_load_depths,
        ))
        eval_loader = DataLoader(
            eval_dataset,
            collate_fn=_collate_camera_batch,
            batch_size=data_cfg.eval_batch_size,
            num_workers=data_cfg.eval_num_workers,
            shuffle=False,
            pin_memory=data_cfg.eval_pin_memory,
        )
        with open(Path(self.cfg.input.edgs_dir.format(scene=name)) / "features_meta.json") as f:
            meta = json.load(f)
        width = sum(meta["blocks"][b][1] - meta["blocks"][b][0] for b in self.feature_blocks)
        scene = DL3DVScene(name, parser, train_dataset, eval_dataset, eval_loader, None, None,
                           environment=self.split["environment"][name], scale=scale)
        return scene, width, meta.get("pca_basis_sha1")

    @staticmethod
    def _rescale_parser(parser, scale):
        """Scale the normalized scene by `scale` about the origin: camera centers
        and COLMAP points move, orientations and intrinsics stay (renders are
        unchanged); `parser.transform` maps raw COLMAP coordinates (e.g. the EDGS
        points) into the scaled frame."""
        parser.camtoworlds = parser.camtoworlds.copy()
        parser.camtoworlds[:, :3, 3] *= scale
        parser.points = parser.points * scale
        parser.transform = np.diag([scale, scale, scale, 1.0]) @ parser.transform
        parser.scene_scale *= scale

    def _load_points(self, scene):
        """All valid EDGS points of `scene` (outliers removed) into `scene.means`
        / `scene.feat` (rgb + standardized features, fp16); sets `num_points`."""
        edgs_dir = self.cfg.input.edgs_dir.format(scene=scene.name)
        edgs = load_edgs_points(OmegaConf.merge(self.cfg.input, {"max_points": None}), edgs_dir,
                                scene.parser, scene.train_dataset.indices)
        means, info = edgs["means"], edgs["info"]

        # Outlier filter as in `Trainer._build_scene` (LitePT's 2^16 voxels per axis).
        half_extent = (2**16 - 1) * self.cfg.trainer.point_grid_size / 2
        if self.cfg.augment.enabled:
            half_extent = (half_extent - self.cfg.trainer.point_grid_size) / max_expansion(self.cfg.augment)
        keep = (np.abs(means - np.median(means, axis=0)) < half_extent).all(axis=1)

        parts = [edgs["rgb"][keep]] if self.direct_channels else []
        if edgs["features"] is not None:
            parts.append(edgs["features"][keep])
        if self.cfg.input.sky:
            # Sky flag = last feature channel (unstandardized), as in `Trainer._build_scene`.
            parts.append(edgs["is_sky"][keep, None].astype(np.float32))
        scene.feat = torch.from_numpy(np.concatenate(parts, axis=1).astype(np.float16))
        scene.means = torch.from_numpy(means[keep])
        is_sky = edgs["is_sky"][keep]
        scene.num_sky = int(is_sky.sum())
        scene.num_sky_removed = int((edgs["is_sky"] & ~keep).sum())
        if scene.num_sky and is_sky[: len(is_sky) - scene.num_sky].any():
            raise ValueError(f"[{scene.name[:8]}] sky rows are not the last rows")

        if self.cfg.input.pos_radius == "sigma":
            # |mu| + k sigma of all valid scene points (as `Trainer._build_scene`).
            mu, sigma = edgs["scene_mu"], edgs["scene_sigma"]
            scene.fixed_pos_radius = float(np.linalg.norm(mu) + self.cfg.input.pos_sigma_k * sigma)
        elif self.cfg.input.pos_radius != "percentile":
            raise ValueError(f"input.pos_radius must be percentile or sigma, got {self.cfg.input.pos_radius!r}")

        target = self.cfg.dl3dv.points_bounded if scene.environment == "bounded" else self.cfg.dl3dv.points_unbounded
        scene.num_points = min(target, len(scene.means) - scene.num_sky)
        print(
            f"[{scene.name[:8]}] {scene.environment}: {len(scene.means)} points ({info['num_valid']} valid, "
            f"{(~keep).sum()} outliers removed), {scene.num_points} per forward; {len(scene.train_dataset)} train / "
            f"{len(scene.eval_dataset)} eval views; reprojection median {info['align_median_px']:.2f} px; "
            f"scale {scene.scale:.3f}, EDGS r95 {np.percentile(np.linalg.norm(means[keep], axis=1), 95):.2f}"
            + (f"; {scene.num_sky} sky points (all per forward, {scene.num_sky_removed} removed as outliers)"
               if scene.num_sky or scene.num_sky_removed else "")
            + (f"; PE radius {scene.fixed_pos_radius:.3f}" if scene.fixed_pos_radius is not None else "")
        )

    def _eval_rows(self, scene):
        """The fixed subset of `scene`'s points used for eval (same every eval / run)."""
        return _eval_rows(scene, self.cfg.input.seed)

    def _eval_view(self, scene):
        """`scene.eval_view`: a `DL3DVScene` sharing `scene`'s parser and eval
        loader, with its fixed eval subset of points in fp32 (what `eval_scene`
        feeds the model). Made on first use; in lazy mode the scene's points are
        loaded for it and freed again unless it is a loaded training scene."""
        if scene.eval_view is None:
            loaded_here = scene.means is None
            if loaded_here:
                self._load_points(scene)
            rows = self._eval_rows(scene)
            scene.eval_view = DL3DVScene(
                scene.name, scene.parser, scene.train_dataset, scene.eval_dataset, scene.eval_loader,
                scene.means[rows].contiguous(), scene.feat[rows].float(),
                environment=scene.environment, num_points=scene.num_points,
                fixed_pos_radius=scene.fixed_pos_radius,
            )
            if loaded_here:
                scene.means = scene.feat = None
        return scene.eval_view

    def _train_points(self, scene, batch):
        """Input points of one training micro-batch, in the main process (see
        `_sample_points`; `_PointsCollate` does the same in the loader workers).
        Upcasts the features on the GPU."""
        if scene.means is None:
            self._load_points(scene)
        means, feat, pos_radius, keep_frac = _sample_points(scene, batch, self._points_args())
        return (means.to(self.device, non_blocking=True), feat.to(self.device, non_blocking=True).float(),
                pos_radius, keep_frac)

    def _points_args(self):
        trainer = self.cfg.trainer
        return dict(
            cull=trainer.frustum_cull, margin=trainer.cull_margin, near_plane=trainer.near_plane,
            resample=self.cfg.dl3dv.resample_points, seed=self.cfg.input.seed,
            pe_percentile=None if self.cfg.input.pos_encoding == "none" else self.cfg.input.pos_percentile,
        )

    @torch.no_grad()
    def eval(self):
        """Both eval groups; scenes split over the ranks, results on rank 0.
        Logs `<group>/<hash[:8]>/<metric>`, `<group>/<metric>` (mean over the
        group's scenes) and `<group>/<environment>/<metric>`."""
        groups = {"eval_seen": self.eval_seen, "eval_unseen": self.eval_unseen}
        results = {}
        for group, scenes in groups.items():
            results[group] = {
                scene.name: self.eval_scene(self._eval_view(scene))
                for i, scene in enumerate(tqdm(scenes, desc=group))
                if i % self.world_size == self.rank
            }
        if self.world_size > 1:
            gathered = [None] * self.world_size if self.is_main else None
            dist.gather_object(results, gathered, dst=0)
            if not self.is_main:
                return
            results = {g: {k: v for r in gathered for k, v in r[g].items()} for g in groups}

        log = {"step": self.step}
        for group, scenes in groups.items():
            per_env = {}
            for scene in scenes:
                metrics = results[group][scene.name]
                images = metrics.pop("images")
                stats = {k: sum(v) / len(v) for k, v in metrics.items()}
                print(
                    f"[eval step {self.step}] {group} {scene.name[:8]} ({scene.environment}) "
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, "
                    f"LPIPS: {stats['lpips']:.3f} ({len(metrics['psnr'])} images)"
                )
                prefix = f"{group}/{scene.name[:8]}/"
                log.update({f"{prefix}{k}": v for k, v in stats.items()})
                log[f"{prefix}images"] = [wandb.Image(image, caption=caption) for image, caption in images]
                per_env.setdefault(scene.environment, []).append(stats)
            all_stats = [s for env_stats in per_env.values() for s in env_stats]
            for k in ("psnr", "ssim", "lpips"):
                log[f"{group}/{k}"] = sum(s[k] for s in all_stats) / len(all_stats)
                for env, env_stats in per_env.items():
                    log[f"{group}/{env}/{k}"] = sum(s[k] for s in env_stats) / len(env_stats)
            print(
                f"[eval step {self.step}] {group} mean over {len(all_stats)} scenes: "
                f"PSNR: {log[f'{group}/psnr']:.3f}, SSIM: {log[f'{group}/ssim']:.4f}, "
                f"LPIPS: {log[f'{group}/lpips']:.3f}; "
                + ", ".join(f"{env} PSNR {log[f'{group}/{env}/psnr']:.3f}" for env in sorted(per_env))
            )
        self.logger.log(log)

    def build_logger(self):
        super().build_logger()
        if self.is_main:
            for group in ("eval_seen", "eval_unseen"):
                self.logger.define_metric(f"{group}/*", step_metric="step")

    def load_checkpoint(self, path):
        # The scenes and points per forward must match the run being resumed.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        saved = ckpt["config"].get("dl3dv")
        current = OmegaConf.to_container(self.cfg.dl3dv, resolve=True)
        for key in ("root", "scene_subdir", "split_file", "points_bounded", "points_unbounded", "target_radius",
                    "train_environment", "train_limit"):
            if saved is None or saved[key] != current[key]:
                raise ValueError(f"Config mismatch on resume for dl3dv.{key}: "
                                 f"{None if saved is None else saved[key]} != {current[key]}")
        del ckpt
        super().load_checkpoint(path)
