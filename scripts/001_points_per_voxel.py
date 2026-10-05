"""Compute the number of Gaussian centers (means) per voxel from a gsplat
simple_trainer.py checkpoint, for one or more voxel sizes.

Usage:
    python scripts/001_points_per_voxel.py --ckpt path/to/ckpt_29999_rank0.pt \
        --voxel-sizes 0.01 0.05 0.1
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro


@dataclass
class Config:
    ckpt: List[Path]
    """One or more checkpoint files (e.g. multiple ranks) to load and concatenate."""

    voxel_sizes: List[float]
    """Voxel edge lengths to evaluate, in the same world units as the checkpoint's means."""

    out_plot: Path = Path("points_per_voxel.png")
    """Path to save the stacked points-per-voxel histograms to."""

    min_alpha: float = 0.005
    """Discard points whose opacity (after sigmoid) is <= this value."""


def main(cfg: Config) -> None:
    ckpts = [torch.load(f, map_location="cpu", weights_only=True) for f in cfg.ckpt]
    means = torch.cat([ckpt["splats"]["means"] for ckpt in ckpts]).numpy()
    opacities = torch.sigmoid(
        torch.cat([ckpt["splats"]["opacities"] for ckpt in ckpts])
    ).numpy()
    print(f"Loaded {means.shape[0]} points from {len(ckpts)} checkpoint(s)")

    keep = opacities > cfg.min_alpha
    means = means[keep]
    print(f"Kept {means.shape[0]} points with alpha > {cfg.min_alpha} "
          f"(dropped {(~keep).sum()})")

    fig, axes = plt.subplots(
        len(cfg.voxel_sizes), 1, figsize=(8, 4 * len(cfg.voxel_sizes)), squeeze=False
    )

    for ax, voxel_size in zip(axes[:, 0], cfg.voxel_sizes):
        voxel_idx = np.floor(means / voxel_size).astype(np.int64)
        _, counts = np.unique(voxel_idx, axis=0, return_counts=True)

        print(f"\nVoxel size: {voxel_size}")
        print(f"Occupied voxels: {counts.shape[0]}")
        print(f"Points per voxel: mean={counts.mean():.2f} median={np.median(counts):.1f} "
              f"std={counts.std():.2f} min={counts.min()} max={counts.max()}")
        for p in (50, 90, 95, 99):
            print(f"  p{p}: {np.percentile(counts, p):.1f}")

        # Occupied voxels always have count >= 1, so bins start at 1. One bin
        # per integer count value, centered on that value.
        bins = np.arange(1, counts.max() + 2) - 0.5
        ax.hist(counts, bins=bins, color="steelblue")
        ax.set_xlabel("Points per voxel")
        ax.set_ylabel("Number of voxels")
        ax.set_yscale("log")
        ax.set_title(f"voxel_size={voxel_size} | occupied_voxels={counts.shape[0]}")

        ax.set_axisbelow(True)
        ax.grid(True, linestyle=":", linewidth=0.5, color="0.85")
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)

        mean_count = counts.mean()
        ax.axhline(mean_count, color="red", linewidth=1, label=f"mean = {mean_count:.2f}")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    cfg.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out_plot, dpi=150)
    print(f"\nWrote histogram to {cfg.out_plot}")


if __name__ == "__main__":
    main(tyro.cli(Config))
