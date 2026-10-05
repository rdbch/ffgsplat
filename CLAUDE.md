# ffgsplat ("Very Fast GS")

Research project: replace `gsplat`'s per-Gaussian free-parameter optimization with a
**point-conditioned network** (backbone: LitePT) that predicts Gaussian Splatting
parameters from point coordinates, trained end-to-end through the differentiable
rasterizer. The full research & design plan lives in [`n.md`](./n.md) — read it before
proposing architecture or training-loop changes. Summary of its key decisions below;
n.md is the source of truth if anything here goes stale.

## Core idea

`f(mean_i) -> (scale_i, quat_i, opacity_i, sh0_i, shN_i)` for every Gaussian, where `f`
is LitePT + a thin per-point head. `means` are a fixed input coordinate, never a leaf
`nn.Parameter` — this *replaces* `simple_trainer.py`'s per-Gaussian Adam optimization,
it doesn't run alongside it. The only trainable weights are the point model's.
Supervision is purely the rendering loss (L1 + SSIM, optionally LPIPS) via gsplat's
`rasterization()`, backpropagated into the network.

Open question (not yet decided): is this a *per-scene reparameterization* (network
trained fresh per scene) or a *cross-scene generalizing prior* (trained once, applied
zero-shot to new point clouds)? See n.md's Goal/Risks sections.

## Layout

- `submodules/gsplat/examples/simple_trainer.py` — the baseline pipeline this builds
  on: COLMAP data loading (`datasets/colmap.py`), `create_splats_with_optimizers`
  (splat param init), the MCMC/ADC training loop, checkpoint format
  (`{"step", "splats": <ParameterDict state_dict>}`).
- `submodules/LitePT/litept/model.py` — vendored LitePT point transformer (Pointcept-
  style U-Net: sparse-conv + windowed attention, `GridPooling`/`GridUnpooling`). See
  `submodules/LitePT/demo_use.py` for its actual input/output contract
  (`coord`/`feat`/`grid_coord`/`grid_size`/`offset`).
- `models/point_model.py` — `LitePtGSModel`, currently a stub. This is where the
  LitePT-backbone + GS-param-head wiring goes.
- `scripts/001_points_per_voxel.py` — histograms points-per-voxel for a gsplat
  checkpoint across a list of voxel sizes; directly informs picking LitePT's
  `grid_size` input knob (same concept).
- `scripts/002_points_per_voxel_mipnerf.sh` — runs the above across all MipNeRF-360
  scenes.
- `configs/`, `engine/` — generic `Config`/`Trainer` scaffold, not yet specialized for
  this project.
- `submodules/gsplat/exp_dir/baseline/` — MCMC baseline training runs/logs, used as
  the comparison point for eval.

## Conventions

- Checkpoint format for anything reading/writing splat params: match
  `simple_trainer.py`'s `{"step": int, "splats": {means, scales, quats, opacities,
  sh0, shN}}`, with activations applied at render time (`exp` for scales, `sigmoid`
  for opacities, normalize for quats) — not stored pre-activated.
- CLI scripts in `scripts/` use `tyro.cli` (see `001_points_per_voxel.py`), matching
  `simple_trainer.py`'s own CLI convention.
