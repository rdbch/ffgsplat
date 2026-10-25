#!/usr/bin/env python3
"""Pre-flight check of the 022 inputs (DL3DV + sky sphere), one scene at a time.

    python scripts/022_check_sky.py <the same key=value overrides as the training run>

For every training / eval_seen / eval_unseen scene (`dl3dv.eval_unseen_limit` applies), loaded exactly as `TrainerDL3DV`
does it (`_build_dl3dv_scene` + `_load_points`: parser normalization, target_radius
scaling, outlier filter, sky flag channel), prints:

- scene / sky points, sky points the outlier filter removed (beyond the voxel grid's
  extent: (2^16 - 1) x grid / 2 / augmentation expansion per axis around the median);
- the sky's per-axis extent from the median as a fraction of that limit;
- the PE radius r: the fixed |mu| + k sigma (`input.pos_radius=sigma`) and, for
  comparison, 019's r95 of a random subset of `num_points` scene points + the sky;
- max |p| = |x| / r of the sky (and of the scene points) for both: > 1 wraps the
  Fourier PE (period 2 in p);
- the sky flag channel: the last feature channel is 1 exactly on the last num_sky rows.

Exit code 1 if any sky point is removed, the flag is wrong, or the sky's max |p| with
the radius the run uses exceeds 1 + 1e-3.
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from core.trainer import percentile_radius
from core.trainer_dl3dv import DL3DVTrainConfig, TrainerDL3DV
from data.augment import max_expansion


def main():
    cfg = OmegaConf.merge(OmegaConf.structured(DL3DVTrainConfig), OmegaConf.from_cli())
    trainer = TrainerDL3DV(cfg)
    dl3dv = cfg.dl3dv
    split = trainer.split
    train = split[dl3dv.train_on]
    if dl3dv.train_environment is not None:
        train = [n for n in train if split["environment"][n] == dl3dv.train_environment]
    if dl3dv.train_limit is not None:
        train = train[: dl3dv.train_limit]
    names = train + [n for n in split["eval_seen"] if n not in train] + split["eval_unseen"][: dl3dv.eval_unseen_limit]

    grid = cfg.trainer.point_grid_size
    half_extent = (2**16 - 1) * grid / 2
    if cfg.augment.enabled:
        half_extent = (half_extent - grid) / max_expansion(cfg.augment)
    print(f"{len(names)} scenes ({len(train)} training); outlier limit {half_extent:.1f} per axis from the "
          f"median; input.sky={cfg.input.sky}, input.pos_radius={cfg.input.pos_radius}")
    head = (f"{'scene':8} {'env':3} {'group':6} {'scene pts':>9} {'sky':>7} {'sky rm':>6} {'ext/lim':>7} "
            f"{'r sigma':>7} {'r95':>6} {'sky|p|s':>7} {'sky|p|95':>8} {'pts|p|s':>7} {'flag':>4}")
    print(head)
    failures = []
    rng = torch.Generator().manual_seed(0)
    for name in names:
        group = "train" if name in train else ("seen" if name in split["eval_seen"] else "unseen")
        scene, _, _ = trainer._build_dl3dv_scene(name)
        trainer._load_points(scene)
        means = scene.means.double()
        n_sky = scene.num_sky
        n_scene = len(means) - n_sky
        sky, pts = means[n_scene:], means[:n_scene]
        median = means.median(0).values

        flag_ok = True
        if cfg.input.sky:
            flag = scene.feat[:, -1].float()
            flag_ok = bool((flag[n_scene:] == 1).all() and (flag[:n_scene] == 0).all())
        ext = float((sky - median).abs().max() / half_extent) if n_sky else float("nan")

        sub = torch.randperm(n_scene, generator=rng)[: scene.num_points]
        r95 = percentile_radius(torch.cat([pts[sub], sky]).float(), cfg.input.pos_percentile)
        r_sigma = scene.fixed_pos_radius
        sky_norm = sky.norm(dim=1).max().item() if n_sky else float("nan")
        pts_norm = pts.norm(dim=1).max().item()
        p_sigma = sky_norm / r_sigma if r_sigma else float("nan")
        p_95 = sky_norm / r95
        print(f"{name[:8]} {scene.environment[:3]} {group:6} {n_scene:9d} {n_sky:7d} {scene.num_sky_removed:6d} "
              f"{ext:7.3f} {r_sigma or float('nan'):7.2f} {r95:6.2f} {p_sigma:7.3f} {p_95:8.3f} "
              f"{pts_norm / (r_sigma or r95):7.2f} {'ok' if flag_ok else 'BAD':>4}", flush=True)

        used_p = p_sigma if r_sigma else p_95
        if scene.num_sky_removed:
            failures.append(f"{name[:8]}: {scene.num_sky_removed} sky points removed as outliers")
        if not flag_ok:
            failures.append(f"{name[:8]}: sky flag channel wrong")
        if cfg.input.sky and n_sky and used_p > 1 + 1e-3:
            failures.append(f"{name[:8]}: sky max |p| {used_p:.3f} > 1")
        if cfg.input.sky and not n_sky:
            failures.append(f"{name[:8]}: no sky points")
        scene.means = scene.feat = None

    print("\npts|p|s: max |x| / r of the scene points (far outliers beyond the sphere wrap regardless).")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
