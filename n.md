# Point-to-Gaussian Trainer: Research & Design Plan

2026-09-18 · @Someone

## Goal

Replace the per-Gaussian free parameters that `simple_trainer.py` optimizes directly with a **point-conditioned network**: a model `f(mean_i) -> (scale_i, quat_i, opacity_i, sh0_i, shN_i)` for every Gaussian in a scene, trained end-to-end through gsplat's differentiable rasterizer instead of through per-parameter Adam.

Why this over direct per-Gaussian optimization:

- **Amortized/faster init.** The network can be warm-started or pretrained, potentially reaching a good render loss in far fewer steps than MCMC/ADC from scratch.
- **Generalization.** If trained across multiple scenes, the same weights could predict reasonable Gaussian parameters for an unseen scene's point cloud without any per-scene optimization (a learned prior over "what a Gaussian at this location, in this local geometry, should look like").
- **Regularization.** A shared network implicitly ties together nearby Gaussians (through shared weights / receptive field), which free per-point parameters can't do.

This is exploratory — the primary open question (see Risks) is whether we're building a *per-scene reparameterization* (network replaces one scene's parameters, still trained per-scene) or a *cross-scene generalizing prior* (network trained once, applied zero-shot). The two imply different architectures and data requirements.

**Concretely:** the only trainable weights are the point model's (see Architecture) — `means` are a fixed input coordinate, never a leaf `nn.Parameter`. This fully replaces `simple_trainer.py`'s per-Gaussian free-parameter Adam optimization; it doesn't run alongside it.

## Background: the existing pipeline

`submodules/gsplat/examples/simple_trainer.py` is the pipeline this builds on:

- **Data loading** (`datasets/colmap.py`, used at `simple_trainer.py:~300`): a `Parser` reads a COLMAP reconstruction — per-image camera-to-world poses, per-camera intrinsics (`Ks`), image paths, and the sparse `points3D` point cloud. A `Dataset` wraps it for iteration, loading and downscaling images by `cfg.data_factor` (2x for indoor MipNeRF-360 scenes, 4x for outdoor, per `run_mipnerf.sh`).
- **Model init** (`create_splats_with_optimizers`, `simple_trainer.py:211`): builds an `nn.ParameterDict` — `means`, `scales`, `quats`, `opacities`, `sh0`, `shN` — initialized from the COLMAP sparse points, each with its own per-parameter Adam optimizer and learning rate.
- **Training loop** (`Runner.train`, `simple_trainer.py:~556`): each step samples a camera, runs `gsplat.rendering.rasterization(means, quats, scales, opacities, colors, viewmats, Ks, ...)`, computes L1 + SSIM (+ optional LPIPS) against the ground-truth image, and backprops **directly into the free per-Gaussian parameters**. A `DefaultStrategy`/`MCMCStrategy` periodically relocates, splits, or prunes Gaussians (density control).
- **Checkpoints** are `{"step": int, "splats": <ParameterDict state_dict>}` saved to `<result_dir>/ckpts/ckpt_<step>_rank<r>.pt` — this is the exact format `scripts/001_points_per_voxel.py` already reads.

The key structural change we're proposing: `means` stays as-is (comes from the point cloud), but `scales`/`quats`/`opacities`/`sh0`/`shN` become the **output of a network** conditioned on `means`, rather than free leaf parameters.

## Data pipeline additions

Keep the existing `Parser`/`Dataset` untouched for cameras, poses and images (data-factor downscaling already handled there). Add one new loader:

1. **Reference point cloud.** A converged `simple_trainer.py` checkpoint (`ckpt_<step>_rank0.pt`), loaded the same way the eval-only path already does it (`simple_trainer.py:1169-1177`: `torch.load(..., weights_only=True)`, then `ckpt["splats"][k]`). This gives `means` (model input) and, if we want a distillation/auxiliary loss, the reference `scales`/`quats`/`opacities`/`sh0`/`shN` too.
2. **Point ordering/chunking.** For large scenes (up to \~1.5M points at `cap_max` per `run_mipnerf.sh`), decide whether the model sees all points every step or a sampled/chunked subset — affects both the loader (needs a stable point index) and the training loop (see below).

Open question: do we source `means` from the *raw COLMAP sparse cloud* (sparser, matches the true init used by `create_splats_with_optimizers`) or from a *converged GS checkpoint* (denser, already through MCMC densification)? The former is the more faithful "replace the optimizer" setup; the latter gives a fixed, denser point count to condition on but couples us to having already run the baseline per scene.

## Model architecture: LitePT backbone

Decided, not a survey: **LitePT** (`submodules/LitePT/litept/model.py`), a vendored Pointcept-style point transformer. This repo already has a stub entry point for it at `models/point_model.py::LitePtGSModel`.

**Input contract** (a `Point`/dict, per `submodules/LitePT/demo_use.py`):

| Key | Shape | Meaning |
| --- | --- | --- |
| `coord` | `[N,3]` | point coordinates — our Gaussian `means` |
| `feat` | `[N,C_in]` | per-point input features (at minimum `coord` itself; optionally reference scale/opacity/color if bootstrapping from a converged checkpoint) |
| `grid_coord` + `grid_size` | — | voxelizes the input before the network's own encoder/decoder — **this `grid_size` is the same knob `scripts/001_points_per_voxel.py --voxel-sizes` sweeps**, so that analysis directly informs picking it |
| `offset` / `batch` | `[B]` / `[N]` | batch separator for batched point clouds |

**Internal structure:** a U-Net over the point cloud — sparse-conv + windowed-attention `Block`s (`enc_conv`/`enc_attn` toggled per stage) with serialization-order (`z`, `hilbert`, ...) windowed attention, `GridPooling`/`GridUnpooling` for multi-scale down/up-sampling. With the default (non-`enc_mode`) config it decodes back to the *same resolution it was given* — one feature vector per input point (after any external `GridSample` downsampling), not a further-reduced set.

**Output → GS params:** LitePT's forward returns a `Point` with `.feat` `[N,C_out]`, one row per input point. `LitePtGSModel` is where a thin per-point head goes on top: `feat -> (scale[3], quat[4], opacity[1], sh0[3], shN[(K-1)*3])`, with the same activations `simple_trainer.py` applies at render time (`exp`, normalize, `sigmoid`).

**What's optimized:** only LitePT's + that head's parameters (see Goal). If the input point cloud was itself downsampled by an external `GridSample` before the model sees it, predicted params live on those voxel-center points, not the raw COLMAP/reference points — `GridSample`'s `return_inverse` gives the scatter-back index if dense per-raw-point params are ever needed.

## Supervision: rendering loss

The core idea: run the *same* `gsplat.rendering.rasterization()` call `simple_trainer.py` uses, but with model-predicted parameters, and backprop the photometric loss through rasterization into the **model's weights** instead of into free `nn.Parameter`s.

```
means (fixed, from point cloud)
   |
   v
point model  --predicts-->  scales, quats, opacities, sh0, shN
   |                              |
   |                              v
   |                    gsplat.rendering.rasterization(means, quats, scales,
   |                        opacities, colors, viewmats, Ks, ...)
   |                              |
   |                              v
   |                       rendered image  --L1+SSIM vs GT-->  loss
   |                              |
   +<-----------------------backprop--------------------------+
```

**Visibility / contribution masking.** Not every Gaussian contributes to every view. Two options:

- Let the loss naturally zero out gradients for non-contributing Gaussians (rasterization's own backward pass should already do this — a Gaussian that's culled or contributes \~0 alpha to any pixel gets \~0 gradient). This needs verifying against gsplat's actual autograd behavior, not assumed.
- Explicitly restrict which points the model predicts for, per sampled view, using frustum culling + whatever per-Gaussian visibility info `rasterization()` exposes (need to check its return values/`info` dict for per-Gaussian contribution, e.g. `radii` or similar). This would make the per-step model forward pass cheaper by only running it over visible points.

Either way, this needs a research spike against gsplat's actual `rasterization()` API before committing to an approach (see Risks).

## Training loop mechanics

Per step (mirroring `Runner.train` in `simple_trainer.py`):

1. Sample a batch of cameras from the `Dataset` (same as today).
2. Forward the point model over `means` to predict `(scales, quats, opacities, sh0, shN)` — either for the whole point cloud, or only points visible in the sampled cameras (see Supervision).
3. Call `rasterization()` with those predicted params for each sampled view.
4. Compute L1 + SSIM (+ optional LPIPS) against ground-truth images.
5. Backprop through rasterization into the point model's weights (no per-Gaussian optimizer — one optimizer over the network).

**Open design question — forward-pass frequency.** Re-running the point model over \~1M+ points every step is a real cost on top of rasterization itself. Options: (a) forward the full point cloud every step (simplest, likely too slow at scale); (b) forward only points visible in the sampled views' frustums; (c) cache predictions and refresh every K steps (risks staleness vs. the model's own weight updates). Needs a throughput measurement before picking.

No density control (split/prune/relocate) in the first version — point count is fixed by the input point cloud. Revisit once the base loop works.

## Losses and regularizers

- **Photometric:** L1 + SSIM, same weighting as `simple_trainer.py`'s `l1loss`/`ssimloss`; optionally add LPIPS.
- **Opacity/scale regularization:** reuse `cfg.opacity_reg` / `cfg.scale_reg` (mean opacity / mean exp-scale penalties already in `simple_trainer.py`) to discourage degenerate predictions.
- **Optional distillation term:** L2 between predicted and reference-checkpoint `scales`/`opacities`/`sh0`/`shN`, only if we loaded a converged reference checkpoint (see Data pipeline). Useful as a warm-start / stabilizer early in training, weighted down (or dropped) over time so the render loss dominates — the point is to eventually not need a reference checkpoint at inference time.

## Open questions and risks

| Risk | Why it matters | How to de-risk |
| --- | --- | --- |
| Per-scene vs. cross-scene generalization undecided | Changes architecture, data (one scene vs. many), and what "success" means | Pick one explicitly before building the backbone (see Goal) |
| Per-Gaussian visibility info from `rasterization()` unconfirmed | Determines whether masking is needed or automatic via autograd | Research spike against the actual gsplat API before finalizing Supervision |
| Full-point-cloud forward pass cost (1M+ points/step) | Could make training much slower than the MCMC baseline it's meant to improve on | Measure throughput early; fall back to chunking/caching if too slow |
| Gradient flow through a deep network into render loss | Free-parameter optimization is very well-behaved; a network adds depth and shared-weight coupling that could be less stable | Start with the simplest backbone (per-point MLP) to validate before adding capacity |
| Evaluation protocol undefined | Need an apples-to-apples comparison to know if this is working | PSNR/SSIM/LPIPS vs. the existing MCMC baseline (`submodules/gsplat/exp_dir/baseline/`) at matched Gaussian count |
| Interaction with densification | MCMC relocates/splits/prunes points during training; this design fixes N from the input cloud | Decide if v1 explicitly skips density control (see Training loop) or if it's a later addition |

## Implementation plan

1. **Data loading.** Add the reference-checkpoint loader alongside the existing COLMAP `Parser`/`Dataset`; confirm it lines up point-for-point with what `create_splats_with_optimizers` would have initialized.
2. **Minimal point MLP.** No spatial context, predicts all five output params for a small scene; verify the predicted-param checkpoint format is drop-in compatible with existing eval/render/compression code paths (reuse the `cfg.ckpt` loading logic at `simple_trainer.py:1169-1177`).
3. **Single-view overfit sanity check.** One camera, a few thousand points — confirm render-loss gradients actually flow back into and update the model's weights, and that loss goes down.
4. **Visibility research spike.** Inspect `gsplat.rendering.rasterization()`'s actual outputs/autograd behavior for per-Gaussian contribution, resolving the Supervision open question.
5. **Full multi-view, single-scene run.** Compare PSNR/SSIM/LPIPS and wall-clock training time against the existing MCMC baseline at matched Gaussian count.
6. **(Stretch) Multi-scene training** for generalization, and backbone upgrade (point-transformer / sparse conv) if the per-point MLP underperforms.
