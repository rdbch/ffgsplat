# Research log

One entry per change that was trained. Newest last. Metrics are the eval split
(37 views), PSNR / SSIM / LPIPS, unless noted. All runs: bonsai, EDGS points (1M)
+ RoMa features, LayerNorm LitePT, augmentation (z-rot 180°, tilt 5°, scale
0.8–1.25, no mirror), Muon (moonlight, lr 2e-3), 20k steps, SH degree 1 from 10k,
3g.71gb MIG slice, ~1.6 s/step.

Run folders: `results/<run name>`; logs: `$WORK/logs/<job name>_<job id>.out`.

---

## 2026-10-14 · 015 — predicted position offsets (1.5 voxels)

**Change.** 012 + a zero-initialized offset head: each Gaussian's mean is
`input point + 0.0075 · tanh(·)` (bound = 1.5 LitePT voxels of 0.005).
`scripts/015_train_edgs_muon_offsets.sh`, job 498, wandb `8yb3t4cm`.

| step | 012 (no offsets) | 015 (offsets 1.5 vox) |
|---|---|---|
| 9k  | 21.37 / 0.757 / 0.403 | 21.85 / 0.771 / 0.389 |
| 15k | 21.98 / 0.767 / 0.395 | 22.23 / 0.780 / 0.381 |
| 20k | 22.07 / 0.770 / 0.393 | **22.36 / 0.784 / 0.378** |

**Findings.**
- Offsets help: +0.3 dB PSNR, +0.014 SSIM, −0.015 LPIPS at 20k; the gap is
  largest in SSIM / LPIPS (fine structure).
- Offsets stay small (mean |offset| ≈ 0.0016 vs mean scale ≈ 0.01); 2–3 % of
  the components sit at the bound, slowly rising during training.

---

## 2026-10-14 · 015 with offsets 2.5 voxels

**Change.** Same as 015 with `MAX_OFFSET_VOXELS=2.5` (bound 0.0125).
Job 506, wandb `w9jgtj6y`.

| step | 015 (1.5 vox) | 2.5 vox |
|---|---|---|
| 1k | 19.08 | 19.60 |
| 4k | 20.64 | 21.10 |
| 9k | 21.85 | 21.86 |
| 15k | 22.23 / 0.780 / 0.381 | 22.36 / 0.782 / 0.378 |
| 20k | 22.36 / 0.783 / 0.378 | 22.38 / 0.783 / 0.377 |

**Findings.**
- No gain at 20k (+0.02 dB, within noise); the early lead at 4k was noise.
- The bound was not limiting: with 2.5 vox the offsets grow only slightly
  (mean |offset| 0.0020 vs 0.0016) and the saturated fraction stays ~2 %
  (1.9 % vs 2.2 %). Keep 1.5 vox.

---

## 2026-10-14 · 016 — Fourier positional encoding of the input points

**Change.** 015 (offsets 1.5 vox) + positional encoding of the input points
(`models/pos_encoding.py`, `scripts/016_train_edgs_muon_offsets_pe.sh`):

- center = scene origin (with `data.parser.normalize`, the median of the COLMAP
  points; augmentation rotates/scales about it) — no per-batch center;
- one isotropic divider `r` = 95th percentile of `|coord|`, computed **after**
  augmentation → `p = R x / r` (rotates with the cloud, invariant to the
  augmentation's scale). Bonsai: r95 = 3.56 normalized units, max `|p|` ≈ 46,
  0.22 % of points beyond 1.5 r;
- encoding `[p | sin/cos(2^k π p)], k < 4` = 27 channels;
- two injection points:
  - `input`: concatenated after the RoMa MLP, `[rgb | mlp(RoMa) | PE]`,
    LitePT in_channels 67 → 94 — job 504, wandb `pdbb1nad`;
  - `embed`: added after LitePT's stem through a zero-initialized
    Linear(27 → 36) — job 505, wandb `3gglxh35`.

| step | 015 | 016 input | 016 embed |
|---|---|---|---|
| 1k | 19.08 | **20.05** | **20.02** |
| 2k | 20.26 | 13.18 | 19.36 |
| 3k | 19.95 | 14.88 | 20.98 |
| 4k | 20.64 | 13.30 | 15.93 |

Both cancelled at ~4k (checkpoints up to step 3999 kept).

**Findings.**
- PE speeds up early training: +1 dB over 015 at 1k, both variants alike.
- Then **both collapse on eval** (input from 2k, embed at 4k) while the
  *training* PSNR stays healthy (~21.5, ahead of 015) and training-time scales
  are normal (max ≈ 0.1, same as 015).
- The collapse is a **train / eval mode mismatch**: the same step-1999
  checkpoint of `input` gives 12.86 dB rendered in eval mode but 20.65 dB in
  train mode (same weights, pose and inputs); `embed` 19.36 vs 20.35. The only
  mode-dependent layer is DropPath (`model.drop_path=0.3`, active only in the
  attention stages 3–4; no BatchNorm with `model.norm=layer`, dropouts 0).
- Hypothesis (not verified): with absolute position available, the deep
  branches learn position-specific corrections tuned to the stochastic
  (dropped, ×1/keep) training forward; with all branches on in eval they
  overshoot. Without PE (015) this doesn't happen. `embed` collapses later
  because its PE path starts at zero.
- Suspected amplifier: the unbounded raw-p tail (|p| up to ~46 entering the
  stem conv next to O(1) inputs) → tested in the next two entries.

---

## 2026-10-15 · 016 without raw p

**Change.** 016 `input` with `POS_INCLUDE_RAW=false`: only the 24 bounded
sin/cos channels (in_channels 91); the tail still wraps (period 2).
Job 509, wandb `as0lhhc3`. *Running.*

| step | 015 | 016 input (raw p) | 016 input, no raw p |
|---|---|---|---|
| 1k | 19.08 | 20.05 | 19.95 |
| 2k | 20.26 | 13.18 | **13.02** |

**Findings.**
- Collapses on eval at 2k exactly like 504, with a fully bounded encoding
  (all channels in [−1, 1]); training PSNR healthy (21.1 mean over 1.5k–2.2k),
  scales normal (max ≈ 0.1).
- → The unbounded raw-p tail is **not** the cause (nor a visible amplifier):
  any absolute positional input in `input` mode triggers the collapse. The
  drop-path train/eval mismatch remains the main suspect.

---

## 2026-10-15 · 016 with raw p, radial clamp at r98

**Change.** 016 `input` with raw p and `POS_CLAMP=98`: still normalized by r95,
but before encoding every point farther than the 98th-percentile distance is
projected radially onto that sphere (direction kept). 98 % of points are
unchanged, the farthest 2 % are clamped; the whole encoding is bounded by
r98 / r95 = 1.16 on bonsai (was ~46).
Job 511, wandb `9087lgtc`. *Running.*

| step | 015 | 016 input (raw p) | 016 no raw p | 016 clamp r98 |
|---|---|---|---|---|
| 1k | 19.08 | 20.05 | 19.95 | 20.07 |
| 2k | 20.26 | 13.18 | 13.02 | **20.68** |
| 3k | 19.95 | 14.88 | — | **14.30** |

**Findings.**
- Best of all runs at 2k (20.68), then collapses on eval at 3k; training PSNR
  healthy (21.2 mean over 1.5k–2.3k).
- The clamp **delays** the collapse by ~1k steps but doesn't prevent it: every
  bounded variant collapses. The tail is at most an accelerator, not the cause.

---

## 2026-10-15 · 016 with raw p, clamp r98, no drop-path

**Change.** As the previous entry (raw p, radial clamp at r98, `input`) with
`model.drop_path=0.0` (LitePT default 0.3), so the training and eval forward
are the same network. Tests the drop-path hypothesis directly against 511.
Job 512 (overrides `model.drop_path=0.0`, own `trainer.output_dir` /
`wandb.name` with `_dp0`), wandb `bgwgg2ah`. 509 (no raw p) cancelled to
free its slice.

| step | 015 | 511 (clamp, drop-path 0.3) | 512 (clamp, drop-path 0) |
|---|---|---|---|
| 1k | 19.08 | 20.07 | 20.13 |
| 2k | 20.26 | 20.68 | 20.79 |
| 3k | 19.95 | 14.30 | **21.10** |
| 7k | 21.57 | — | 21.70 |
| 11k | 21.73 | — | 20.97 |
| 15k | 22.23 / 0.780 / 0.381 | — | 22.20 / 0.779 / 0.384 |
| 20k | 22.36 / 0.783 / 0.378 | — | 22.34 / 0.783 / 0.380 |

**Findings.**
- No collapse at 3k, where 511 (identical except drop-path) collapsed:
  **drop-path is the cause** of the eval collapse. The eval data path
  (canonical pose, no grid jitter) is unchanged, so it is not the cause.
- +1.15 dB over 015 at 3k. 511 cancelled after its collapse.
- The lead is gone by ~7k; **20k equals 015** (22.34 vs 22.36). One dip at
  11k (20.97, recovered by 12k). PE with raw p + clamp only speeds up the
  start.

---

## 2026-10-15 · 016 without raw p, no drop-path

**Change.** 509's setup (sin/cos only, `POS_INCLUDE_RAW=false`, `input`) with
`model.drop_path=0.0`. Pairs with 509 the way 512 pairs with 511.
Job 513 (overrides, own `_dp0` output dir / wandb name), wandb `4tlh1xuf`.

| step | 015 | 512 (raw p + clamp, dp 0) | 513 (no raw p, dp 0) |
|---|---|---|---|
| 1k | 19.08 | 20.13 | 19.99 |
| 2k | 20.26 | 20.79 | 20.97 |
| 3k | 19.95 | 21.10 | 21.18 |
| 9k | 21.85 | 21.95 | 21.84 |
| 15k | 22.23 / 0.780 / 0.381 | 22.20 / 0.779 / 0.384 | 22.49 / 0.786 / 0.375 |
| 20k | 22.36 / 0.783 / 0.378 | 22.34 / 0.783 / 0.380 | **22.59 / 0.789 / 0.373** |

**Findings.**
- No collapse at 2k, where 509 collapsed — same conclusion as 512, for the
  encoding without raw p. +0.7 dB over 015 at 2k.
- **Best run so far at 20k**: +0.23 dB / +0.006 SSIM / −0.005 LPIPS over
  015, ahead at every eval from 13k on (+0.2–0.3). Training PSNR is higher by
  about the same amount (23.05 vs 22.77, mean of 19–20k), so it fits the scene
  better rather than generalizing better (eval views are of the same scene).
- vs 512 (differs only in raw p + clamp): +0.25 dB at 20k. The raw linear
  channels seem to hurt late in training; one seed each, so treat this as weak
  evidence.
- Confound: 513 differs from 015 in PE **and** drop-path. 512 (also dp 0)
  equals 015, which hints that dp 0 alone is not worth much, but that is not
  measured.
- Global PE is a scene-location input: the gain is not expected to carry over
  to chunk-local PE or to unseen scenes.

---

## 2026-10-15 · 017 — larger Gaussians, SH 3, no augmentation

**Change.** 513's setup (sin/cos PE without raw p, drop-path 0, offsets 1.5 vox)
with: hard scale bound `head.max_scale` 1.0 → 0.4 and the scale-tail penalty
off (in 012–016 the penalty, 0.01 · Σ excess over 0.1, kept every scale ≤ 0.1);
SH degree 3, one degree per 1000 steps (was degree 1 from 10k); no
augmentation (no rotation / tilt / scale, no voxel jitter). Tests whether the
size bound and the missing view dependence cause the underfit.
`scripts/017_train_edgs_bigscale_sh3_noaug.sh`, job 514, wandb `rg5ev00l`.

| step | 513 | 017 |
|---|---|---|
| 1k | 19.99 | 24.44 |
| 3k | 21.18 | 25.84 |
| 9k | 21.84 | 27.81 |
| 15k | 22.49 / 0.786 / 0.375 | 28.53 / 0.889 / 0.256 |
| 20k | 22.59 / 0.789 / 0.373 | **28.72 / 0.891 / 0.253** |

**Findings.**
- **+6.1 dB over 513** (+0.10 SSIM, −0.12 LPIPS); ahead by 4.5 dB already
  at 1k. Best run by far; the gap to MCMC (32.8) drops from ~10 to ~4 dB.
- Train PSNR 28.25 (mean 19–20k) vs eval 28.72: still no overfitting.
- The bound is used: scale_max sits at 0.4 and ~21k of 1M Gaussians are
  above 0.1 (scale_mean 0.009); opacity_mean 0.055; offsets saturate more
  (4.1 % vs ~2 %, mean |offset| 0.0030).
- Three changes at once (scale bound / tail penalty, SH 3, no augmentation);
  which one carries the gain is not measured. The early 1k lead points at the
  scale bound or no augmentation rather than SH (degree ≥ 1 only from 1k).
- ~1.41 s/step, 42 GB reserved on the 3g.71gb slice.

---

## 2026-10-17 · 018 — 017 recipe on 120 DL3DV scenes

**Change.** The 017 setup (EDGS + RoMa, Muon, offsets 1.5 vox, sin/cos PE without
raw p, drop-path 0, max_scale 0.4 without tail penalty, SH 3) trained on many
scenes with `core/trainer_dl3dv.py`, `scripts/018_train_dl3dv_edgs.sh`:

- fixed split `data/splits/dl3dv_133.json` (133 DL3DV benchmark scenes with
  EDGS features; 7 failed extraction): 120 train, of which 10 are `eval_seen`
  (5 bounded / 5 unbounded), and 13 `eval_unseen` scenes never trained on
  (7 / 6); both evaluated on every 8th view at images_4 (960×540);
- every scene scaled so its COLMAP points' r95 = 3.56 (bonsai's). Unscaled
  DL3DV scenes are ~2× larger, and one forward OOM'd a full H200 (smoke 765).
  After scaling the EDGS r95 is 3.0–6.2 (COLMAP r95 is a loose proxy);
- all valid EDGS points in RAM; a fresh random 1M (bounded) / 1.25M
  (unbounded) per micro-batch; one fixed subset per scene for eval;
- mild augmentation (z-rot ±30°, tilt ±2°, scale 0.9–1.1, jitter);
- 8 views × grad accum 8 = 64 views per update, LR 4e-4, 60k steps, SH
  degree +1 every 2000 steps; full H200, `expandable_segments`, CUDA cache
  emptied around eval.

Smoke test (lazy mode, 4 training scenes, 100 steps, job 766): ~7.5 s/step,
peak 77 GB allocated / 82 GB reserved; eval at step 100 seen 12.48 / unseen
12.56 dB (pipeline check only).

Jobs 767 + 768 (`RESUME=latest`, after 767), 72 h each, wandb `khuy5mzi`:
startup 4 min (107.7 GiB of points in RAM), ~8.0 s/step wall (7.4 s GPU),
~78–82 GB allocated / 103.5 GB reserved. **Cancelled at ~step 100**, first
the 5-scene check below.

---

## 2026-10-17 · 018 on 5 bounded DL3DV scenes (adaptation check)

**Change.** The 018 run restricted to the first 5 bounded training scenes of
the split (`TRAIN_ENVIRONMENT=bounded TRAIN_LIMIT=5`: 0569e83f, 06da7966,
073f5a9b, 08539793, 093ef327); eval_seen = these 5, eval_unseen = the 13.
Same settings as the 120-scene run (accum 8, LR 4e-4), 20k steps, eval every
1000, train images every 250 steps (wandb `e0jhbid0`). Checks that the DL3DV adaptation (loader,
scaling, points) trains like the Mip-NeRF runs. Job 769.

~6.8 s/step, 64–67 GB allocated / 71.5 GB reserved. At step 1000:

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| eval_seen (5 bounded) | 19.16 | 0.604 | 0.488 |
| eval_unseen (13) | 13.95 | 0.515 | 0.579 |

Train PSNR ~19 at step 1230. The seen scenes train normally, so the adaptation
works. Unseen is expected to be low with 5 training scenes. **Stopped at step
~1230** to start the 120-scene run.

Restart of the 120-scene run: jobs 770 + 771 (`RESUME=latest`, after 770),
72 h each, same config as 767/768, wandb `n9w7wldz`. Step 5: 6.8 s/step,
68 GB allocated / 97.5 GB reserved.

---

## 2026-10-20 · 019c — input→head skip, per-pixel background, kNN views + frustum cull (10 scenes)

**Change.** On top of 018 (`scripts/019_train_dl3dv_edgs_skip.sh`, all new
options default off, so 018 is unchanged):

- **input → head skip** ("U-Net style"): LitePT's input `x0 = [rgb |
  feature-MLP(RoMa) | PE]` (before the stem, which has no identity path) goes
  to the head as well: `[out | GELU(Linear(x0 → 64))]` → hidden layer 128 +
  GELU → output linears (`head.skip=concat`, `head.hidden_dim=128`);
- **colour residual**: `sh0 = (rgb_in − 0.5)/C0 + Δ`, every Gaussian starts at
  its input colour instead of grey (`head.color_residual`);
- **per-pixel random background** behind training renders instead of one
  colour per view (`trainer.random_bkgd_mode=pixel`); eval still on black;
- **kNN views**: each micro-batch = the next view of the scene's shuffled
  order + its 7 nearest training views by camera center
  (`data.train_view_sampling=knn`);
- **frustum culling** of the input subset to the micro-batch's 8 views, 48 px
  (5 % of max(W, H)) margin on every side, PE radius from the unculled subset
  (`trainer.frustum_cull=subset`, `trainer.cull_margin=0.05`); eval uses the
  whole fixed subset;
- points drawn / culled / gathered in the loader workers (CPU) instead of the
  main process (`dl3dv.points_in_workers`).

Setup: the split's 10 `eval_seen` scenes (5 bounded / 5 unbounded,
`dl3dv.train_on=eval_seen`; eval_seen = them, eval_unseen = the 13), grad
accum 4 (32 views per update), LR 4e-4, 10k steps, eval every 500, per-pixel
background, kNN views, the same 500k random points drawn per micro-batch for
every scene, 3g.71gb slices (memory fraction 1.0). Three runs:

| run | skip + rgb residual | culling | job | wandb |
|---|---|---|---|---|
| R0 | – | – | 830 | `gkp90oeq` |
| R3 | ✓ | – | step in hold 808 | `fba6f624` |
| R4 | ✓ | subset (model gets the visible part of the 500k) | step in hold 807 | `8xdn5flo` |

R0 → R3: the skip + colour residual; R3 → R4: culling.

Memory on the slice: kNN + culling keeps only 73–88 % of the subset
(neighbouring video frames see almost the same region, and one view already
sees most of a bounded scene). Before settling on 500k: 1M / 1.25M and 1M for
all OOM'd at step 1 (> 68 GB); 650k fit (24–53 GB). At 500k, first steps:
R0 22–52 GB, R3 30–53 GB, R4 31–46 GB allocated (394–433k points after
culling); ~3.3–4.7 s/step. Failed / stopped attempts were deleted from wandb.

**Interim results** (checked 2026-10-21 01:45 UTC; R0 / R3 at ~5.7k steps,
R4 at ~7.5k). All healthy: no OOM / NaN, evals every 500, checkpoints saved.

| run | step | s/step (wall) | reserved | ETA (UTC, 10-21) |
|---|---|---|---|---|
| R0 (job 830), no skip | 5700 / 10k | ~4.5 | 53.4 GB | ~07:15 |
| R3 (hold 808), skip | 5685 / 10k | ~4.5 | 54.4 GB | ~07:15 |
| R4 (hold 807), skip + cull | 7495 / 10k | ~3.4 | 52.8 GB | ~04:00 |
| 790 (018 DDP, 120 scenes) | 49.4k / 60k | ~4.5 | – | ~15:00 |

R4 is faster because culling keeps only 56–88 % of the 500k points. Eval
means (PSNR / SSIM / LPIPS) at step 5499, the last step all three have
reached:

| run | eval_seen (trained on) | eval_unseen (13) |
|---|---|---|
| R0 | 19.76 / 0.656 / 0.420 | 15.77 / 0.553 / 0.519 |
| R3 | 19.88 / 0.669 / 0.396 | 16.61 / 0.571 / 0.487 |
| R4 | 19.36 / 0.662 / 0.402 | 16.64 / 0.570 / 0.485 |

eval_unseen PSNR over training:

| step | 499 | 1499 | 2499 | 3499 | 4499 | 5499 |
|---|---|---|---|---|---|---|
| R0 | 14.01 | 15.77 | 15.92 | 15.78 | 16.07 | 15.77 |
| R3 | 15.26 | 16.42 | 16.66 | 16.51 | 16.61 | 16.61 |
| R4 | 15.09 | 16.40 | 16.49 | 16.54 | 16.41 | 16.64 |

- Skip + colour residual (R0 → R3): **+0.85 dB unseen**, LPIPS better on both
  sets; the lead is there from step 500 on.
- Culling (R3 → R4): unseen the same, seen within step-to-step noise (R4 seen
  19.72 at 6999), ~25 % faster.
- All three: unseen flat after ~2k steps while seen keeps rising, the same
  pattern as job 790 (expected with 10 training scenes).

**Final results (step 10k): pending — to be added when the runs finish**
(estimated R4 ~04:00, R0 / R3 ~07:15 UTC on 2026-10-21).

**Full run (120 scenes).** The R4 config (skip + colour residual, per-pixel
background, kNN views, culling) on the whole training split, otherwise as
018 / job 790 (1M / 1.25M points, 64 views per update, LR 4e-4, 60k steps,
eval every 2000), 2 × full H200, 700G RAM. The skip should give back the
per-point detail LitePT's voxel pooling merges away. Submitted 2026-10-21 as
jobs 831 + 832 (`019_submit.sh c-full-chain`, `RESUME=latest`, 832 after
831), pending for GPUs (starts when 790 ends, ~15:00 UTC). Compare with 790.
Results: pending.

---

## 2026-10-22 · 020 — sky sphere of fake points (bonsai)

**Change.** A sphere of extra "sky" points around the scene, so sky / far
background / everything without EDGS points can get Gaussians too; the network
decides (through opacity) where it needs them. Design: `docs/chunking_implementation_plan.md`
("Sky / background").

- **Extraction** (`edgs_extract_ref.py --sky`, same file as the points): sphere
  centred at the mean μ of the valid EDGS points, radius **4σ** (σ = √(tr Σ / 3)
  of one Gaussian fitted to them), 200k Fibonacci points; frustum cull with the
  training views (sky points no view sees are dropped); greedy set cover → the
  minimum set of views; each sky point's reference view = the chosen view where
  it is most central; rgb + the 104-d RoMa / DINO features sampled there exactly
  as for the points (occlusion ignored). Appended to the same PLY / features.npz
  with `is_sky`. All 7 Mip-NeRF scenes in `experiments/edgs_feat_sky/` (job 836,
  3–5 min per scene; bonsai: 118k of 200k kept, 23 of 255 views, 0.22 % of the
  valid points outside 4σ; stump 4.4 %, bicycle 3.3 %).
- **Training** (`input.sky=true`): the sky points go in with all the scene
  points, **same voxel grid (0.005)**, same forward and head; an `is_sky` flag
  as the last feature channel (into the feature MLP).
- **PE radius** (`input.pos_radius=sigma`, both runs): r = |μ| + 4σ of the
  scene points in the training frame (sky excluded) = 0.544 + 4 × 1.218 =
  **5.418** on bonsai: the 4σ sphere's farthest point from the origin, so every
  point incl. the sky has |p| ≤ 1 (r95 would put the sky at |p| ≈ 1.4 and wrap).

Setup: `scripts/020_train_bonsai_sky.sh` (+ `jobs/ffgsplat_020.slurm`),
019c's settings on bonsai: skip concat 64 + hidden 128, colour residual,
per-pixel background, kNN views (anchor + 7 nearest; bonsai as a one-scene
multi-scene setup), frustum cull 5 %, mild augmentation, Muon 4e-4, SH 3 every
2000, max scale 0.4, offsets 1.5 vox, Fourier PE F4 without raw p, drop-path 0,
LayerNorm. Points from `edgs_feat_sky/bonsai`: a fixed 1M subset of the scene
points (seed 0) in both runs, + all 118k sky points with sky. 20k steps, 8
views / step, eval every 1000, 3g.71gb slices.

| run | sky | job | wandb |
|---|---|---|---|
| 020-sky | ✓ | 855 | `su3k9soz` |
| 020-nosky | – (sky rows dropped, same scene points, same r) | 856 | `66dy9qni` |

Smoke tests (100 steps): per step ~575k points after culling of which ~24k sky;
sky opacity 0.10 → 0.17 and scale 0.010 → 0.019 by step 100 (the sky is being
used); ~0.9 s/step (→ ~5 h for 20k), 17 GB allocated / 33 GB reserved (sky) vs
22 GB (no sky).

**Results** (bonsai test views, 37 images; 855 5 h 18 min, 856 4 h 37 min):

| run | PSNR @20k | SSIM | LPIPS | best PSNR (step) |
|---|---|---|---|---|
| 020-sky | **24.95** | **0.820** | **0.324** | 24.97 (18k) |
| 020-nosky | 24.49 | 0.811 | 0.337 | 24.56 (11k) |

| step | 1k | 2k | 3k | 5k | 10k | 15k | 20k |
|---|---|---|---|---|---|---|---|
| sky | 22.07 | 23.05 | 23.65 | 24.32 | 24.84 | 24.82 | 24.95 |
| no sky | 20.78 | 21.91 | 23.17 | 23.95 | 24.37 | 24.52 | 24.49 |

- **The sky helps: +0.47 dB / +0.009 SSIM / −0.013 LPIPS at 20k**, ahead at
  every eval, largest early (+1.3 dB at 1k, +1.1 at 2k). No-sky flattens after
  ~11k and drifts down slightly; sky keeps creeping up to the end.
- Sky use at the end: ~26k of ~644k points per step after culling are sky;
  sky opacity mean 0.03 (scene 0.08), scale mean 0.18 — few, large, faint
  Gaussians, i.e. the network uses them as a soft background.
- Caveat: r = 5.418 came from the training subset after the outlier filter,
  not from all valid points the sphere was built from (r = 5.923), so the
  farthest sky points reach |p| ≈ 1.09 (slight Fourier wrap). Fixed for 021.

---

## 2026-10-23 · 021 — complexity-based point selection (bonsai, quantile rule)

**Change.** The 1M scene points of 020-sky drawn by image complexity instead of
uniformly; sky points all kept.

- **Complexity per point** (`scripts/021_complexity_maps.py` + `edgs_extract_ref.py
  --complexity_dir`): delentropy maps (imgvibecheck, kernel 32, stride 4, "same" padding,
  float32 TIFF, `datasets/mipnerf/<scene>/delentropy_k32s4_<factor>/`), sampled at each
  row's reference pixel (points and sky). Extraction to `experiments/edgs_feat_sky_cplx/`
  (jobs 857 + 858; PLY byte-identical to `edgs_feat_sky`, the extraction is
  deterministic). Delentropy saturates at 5.0 with 32 x 32 windows (log2(1024) / 2):
  bonsai median 4.62, p1 3.46, p99 4.99.
- **Selection rule** (`input.subset=quantile`, designed in `$WORK/viewers/complexity`,
  after `docs/claude_complexity_distrib.pdf` fig. 2): complexity -> rank q in [0, 1];
  keys = beta q + Gumbel; floor: in each of 50 equal-width complexity bins the
  min(100, available) top-key points; of the rest of the 1M, 5 % uniform and 95 % the
  largest keys (= without replacement with probability ~ exp(4 q)). Bonsai: rate per
  complexity decile (x uniform) 0.17 0.23 0.30 0.42 0.58 0.81 1.13 1.56 2.09 2.71; median
  complexity of the chosen points 4.939 vs 4.627 (training log = viewer).
- Discarded on the way: min-max weights exp(rho) (koalags, `input.subset=complexity`):
  the saturation + ~0 outliers squeeze rho into ~0.9-1 (p99 / p1 weight 1.1-1.4x); with a
  p1-p99 clip 2.7x but per-point weighting follows the histogram (fig. 1 behaviour at small
  budgets); per-bin quotas: steps / dips at the saturated bin. Job 871 / 872 (clip
  variant) cancelled before the first eval.

Setup: `scripts/021_train_bonsai_complexity.sh` with `SUBSET=quantile`
(+ `jobs/ffgsplat_021.slurm`), otherwise 020-sky (sky + is_sky flag, 019c settings,
20k steps, 8 views / step, 3g.71gb). Job 935, wandb `tnghdtvw`.

Caveat for the comparison with 020-sky (no matched random run, by choice): the PE radius
differs too. 021 uses r = |mu| + 4 sigma of all valid scene points = 0.545 + 4 x 1.345 =
**5.923** (the set the sky sphere was built from: sky at |p| <= 1); 020 used 5.418 from its
subset after the trainer's far-outlier filter (sigma is sensitive to those few points), so
in 020-sky the farthest sky points reach |p| ~ 1.09.

**Results** (bonsai test views, 37 images; job 935, 5 h 18 min):

| run | PSNR @20k | SSIM | LPIPS | best PSNR (step) |
|---|---|---|---|---|
| 020-sky (uniform 1M, r 5.418) | **24.95** | **0.820** | **0.324** | 24.97 (18k) |
| 021-quant (complexity 1M, r 5.923) | 24.56 | 0.816 | 0.335 | 24.81 (15k) |
| 020-nosky | 24.49 | 0.811 | 0.337 | 24.56 (11k) |

| step | 1k | 2k | 3k | 5k | 10k | 15k | 20k |
|---|---|---|---|---|---|---|---|
| 020-sky | 22.07 | 23.05 | 23.65 | 24.32 | 24.84 | 24.82 | 24.95 |
| 021-quant | 21.70 | 22.69 | 23.33 | 24.00 | 24.68 | 24.81 | 24.56 |

- **No gain from the complexity selection here: −0.39 dB vs 020-sky at 20k**
  (−0.2 to −0.4 at almost every eval; only at 15k level). It plateaus at ~24.7
  from 10k and is noisier late (24.56–24.81); final SSIM / LPIPS between
  020-sky and no-sky.
- Sky use similar to 020-sky (~25.5k sky points per step, scale mean 0.18),
  sky opacity a bit higher (0.049 vs 0.030); more points survive the cull
  (~776k vs ~644k per step at the last step; not investigated).
- Not separable from the PE-radius change (5.923 vs 5.418): a matched run
  (uniform subset on `edgs_feat_sky_cplx` with r = 5.923) would tell which one
  costs the 0.4 dB. Not run.

---

## 2026-10-23 · 022 — sky sphere on DL3DV (10 scenes, 3 unseen)

**Change.** 020's sky sphere in the DL3DV trainer (`core/trainer_dl3dv.py`, all
defaults unchanged for 019): with `input.sky`, every sky point of the scene goes
into every forward on top of the `points_bounded / unbounded` scene points (frustum
culled the same way), flag = last feature channel; `input.pos_radius=sigma`: fixed
r = |mu| + 4 sigma of all valid scene points (as 021). New `dl3dv.eval_unseen_limit`
(first N of the 13 eval_unseen scenes).

Setup: `scripts/022_train_dl3dv_sky.sh` (+ `jobs/ffgsplat_022.slurm`) = 019c's R4
(train on the 10 eval_seen scenes, 5 bounded / 5 unbounded, 500k points, grad accum
4, kNN views, cull subset 5 %, 10k steps, eval every 500) + sky + 4 sigma PE; points
from `edgs_feat_dl3dv_sky_cplx` (job 874; complexity not used, uniform subset);
eval_unseen = 07d9f972, 0a1b7c20, 26fd2335. 1 x full H200 (memory fraction 0.9).
Job 1011, wandb `5lv2mflh`. Sky-only (no matched no-sky run, by choice); R4 is the
nearest reference but differs in the PE radius (r95 there).

Smoke test (job 1010, `jobs/ffgsplat_022_smoke.slurm`, 3g.71gb):
`scripts/022_check_sky.py` on the 13 scenes: 60k–152k sky points per scene, **none
removed by the outlier filter** (sky extent 6–37 % of the 86-unit limit), sky max |p|
<= 1.000 everywhere (r 5.0–32.1), flag correct. 100 steps: ~430k points per step after
culling (~46k sky), sky opacity 0.10 → 0.20, 4.2 s/step, peak 63 GB allocated on the
slice (R4: 46) → moved to a full H200: 1.9 s/step, 35 GB allocated / 58 reserved
(first 20 steps), ~5.5 h for 10k + evals.

**Results** (step 10k; job 1011, 6 h 36 min on a full H200, ~1.9–2.2 s/step, 64 GB
reserved). R4 = 019c R4 at 10k; its unseen number recomputed over the same 3 scenes:

| run | eval_seen (10) | bounded / unbounded PSNR | eval_unseen (3) |
|---|---|---|---|
| 019c R4 (no sky, r95 PE) | 19.88 / 0.673 / 0.389 | 20.37 / 19.39 | 17.79 / 0.638 / 0.445 |
| 022 (sky, 4 sigma PE) | **21.39 / 0.699 / 0.363** | 21.52 / 21.26 | **18.09 / 0.638 / 0.439** |

Seen PSNR per scene (R4 → 022): 032dee9f 21.13 → 22.32, 165f5af8 20.20 → 20.72,
1de58be5 20.35 → 23.56, 2beaca31 20.16 → 21.26, 3bb3bb4d 22.12 → 22.79, 9fb0588f 19.84 →
20.99, a401469c 20.65 → 20.88, d904ae29 18.68 → 19.27, e78f8ceb 18.57 → 19.07, ff592398
17.08 → 22.99. Unseen: 07d9f972 19.65 → 20.18, 0a1b7c20 19.00 → 18.98, 26fd2335 14.73 → 15.11.

- **Seen +1.51 dB**, better on all 10 scenes; +1.02 without ff592398 (R4's 17.08 there
  is an outlier, R3 had 19.45). Also +0.96 over R3 (20.42). Unbounded gains more
  (+1.87) than bounded (+1.15), as expected from a sky.
- **Unseen +0.30 dB**, LPIPS −0.006, SSIM equal; 3 scenes only, within noise. Flat
  from step 500 (18.0 → 18.1, best 18.38 at 6.5k) while seen keeps rising: the same
  10-training-scenes pattern as 019c. 26fd2335 (unbounded) stays bad (15.1).
- Seen still rising slowly at 10k (21.38 at 9k), plateauing.
- Sky use at the end: ~39–45k of ~390–450k points per step are sky, sky opacity
  ~0.08–0.09 (scene ~0.08–0.14), sky scale ~0.22–0.25: denser use than on bonsai.
- Not separable from the PE change (4 sigma vs r95), and no no-sky run with the same r.

**Full run (120 scenes).** 022 (sky + 4 sigma PE) with the 019c full setup (job 833):
train split, 1M / 1.25M points, 2 x full H200 x grad accum 4 = 64 views per update,
LR 4e-4, 60k steps, eval every 2000 on the 10 eval_seen + all 13 eval_unseen; 96 CPUs,
700G, CUDA memory fraction 0.9. Submitted 2026-10-23 as jobs 1016 + 1017 (chained,
`RESUME=latest`; env overrides on `jobs/ffgsplat_022.slurm`). Baseline 833 at 60k:
seen 19.64 / 0.680 / 0.382, unseen (13) 18.11 / 0.632 / 0.421. Results: pending.

---

## 2026-10-23 · 023 — effective receptive field (for the chunk overlap; no training)

`scripts/023_receptive_field.py` (job 1012, `jobs/ffgsplat_023_rf.slurm`): for 32
random scene points (+ 8 sky points) per scene, the gradient of one random projection
of the point's predicted Gaussian w.r.t. every input point's features (query itself
excluded); C(d) = share of that influence from points within d (~ what a chunk face
with overlap d cuts away, to first order). 4 eval_seen scenes (1de58be5, 2beaca31
bounded; 032dee9f, 9fb0588f unbounded), their fixed eval subsets. Normalized frame
(voxel 0.005). JSON in `results/023_rf_{022,019c}.json`.

Share of the influence within d (median over the 32 scene-point queries):

| ckpt | scene | 0.025 (5 vox) | 0.1 (20 vox) | 0.32 | 0.64 | 1.28 | 2.56 | 95 % radius |
|---|---|---|---|---|---|---|---|---|
| 019c full, step 50k (1M / 1.25M) | 1de58be5 | 4 % | 7 % | 22 % | 53 % | 85 % | 99 % | 1.90 |
| | 2beaca31 | 25 % | 29 % | 40 % | 63 % | 90 % | 99 % | 1.60 |
| | 032dee9f | 29 % | 32 % | 42 % | 67 % | 91 % | 99 % | 1.63 |
| | 9fb0588f | 33 % | 38 % | 48 % | 66 % | 88 % | 98 % | 1.72 |
| 022, step 2k (500k + sky) | 1de58be5 | 2 % | 19 % | 42 % | 80 % | 95 % | 100 % | 1.31 |
| | 2beaca31 | 57 % | 68 % | 78 % | 90 % | 97 % | 100 % | 0.89 |
| | 032dee9f | 39 % | 46 % | 57 % | 89 % | 98 % | 100 % | 0.93 |
| | 9fb0588f | 72 % | 81 % | 88 % | 94 % | 99 % | 100 % | 0.73 |

- The trained model (019c) is **far from local**: the attention stages (3–4) bring
  most of the influence from 0.3–1.5 away; within 20 voxels only 7–38 %. The early
  022 checkpoint is more local (less trained: the far context is learned).
- Per point the influence falls ~100× from 0.01 to 0.1 and again ~10–100× by 1.28,
  but there are many more far points, so the far field adds up.
- Sky points take their influence from far (50 % only at 1.5–2.8): sparse
  neighbourhoods, the windows of 1024 points span large distances.
- For chunking an existing checkpoint, an overlap that keeps 90 % of the influence
  would be ~1.3 (256 voxels), i.e. cells of the scene's size: exact chunked inference
  is not affordable for 019c. First-order only; the real effect is measured by
  `scripts/023_chunk_eval.py` (chunked vs full renders, `jobs/ffgsplat_023_chunk.slurm`).

---

## 2026-10-24 · 024 — EDGS v2: RoMa matching + features at the image resolution (bonsai)

**Change.** Re-extraction of bonsai (images_2, 1559 x 1039) with new opt-in flags of
`edgs_extract_ref.py` (defaults = the old extraction), motivated by the 2D comparison in
`$WORK/viewers/roma_res` / the "Potential feature problems in RoMa" section of the 200 doc:

- matching: RoMa coarse at **560 x 840** (`--roma_coarse aspect`, 3:2 instead of a squashed
  560 x 560) + RoMa's **refinement pass at 1040 x 1560** (`--roma_refine native`; EDGS used the
  560 x 560 flow, bilinearly resized);
- features: VGG roma1/2/4 on the **native image** (`--feat_native`), DINOv2 at the native aspect
  (1036 x 1554, pos embed interpolated) with **patch stride 7** (`--dino_stride 7`): 147 x 221
  tokens instead of 40 x 40;
- new PCA basis `experiments/edgs_feat/pca/mipnerf7_native_s7_basis.npz` (7 Mip-NeRF scenes x 60
  training images, same 8/16/32/48 dims), job 1020;
- extraction `experiments/edgs_feat_sky_cplx_v2/bonsai` (job 1021, 468 s vs 265 s): 4,615,802
  valid of 4,620,257 (old: 4,601,358 of 4,618,184): **3.8x fewer points rejected** by the
  reprojection tolerance (4.5k vs 16.8k); uv colour check 0.017 (old 0.011, limit 0.05).

Training: `scripts/021_train_bonsai_complexity.sh` with `SUBSET=random` (uniform 1M + all sky =
020-sky's setup, 20k steps, 3g.71gb), only `input.edgs_dir` changed; job 1022, wandb `4vmz5m1n`.
Reprojection median 0.43 px (020/021: 0.51). PE radius |mu| + 4 sigma = **5.472** (sigma 1.229
vs 1.345 on the old points: fewer far outliers), i.e. close to 020-sky's 5.418, so the comparison
with 020-sky (24.95 dB) is nearly matched in r. ~0.85 s/step.

**Results** (finished 2026-10-24 05:40 UTC, 37 test views, step 19999):

| bonsai 20k | PSNR | SSIM | LPIPS | best PSNR |
|---|---|---|---|---|
| 020-sky (old features, job 855, `su3k9soz`) | 24.95 | 0.820 | 0.324 | 24.97 @ 18k |
| **024 (v2)** | **25.07** | **0.837** | **0.306** | 25.08 @ 18k |

- SSIM +0.010…+0.018 and LPIPS −0.012…−0.022 at **every** eval (1k–20k): consistent gain in
  structure / perceptual quality.
- PSNR level: −0.17…+0.19 dB through the run, +0.11 at the end; behind mid-run (5k–12k), ahead
  from 13k on.
- One scene, one seed; the test at scale is the DL3DV v2 extraction (job 1127, trim 0.75 % sigma).
- Added to the 200 doc (roma-res section, "Result: bonsai with v2 features", `RUN_DATA.romav2`).

---

## 2026-10-24 · 025 — EDGS v2 on DL3DV (10 scenes, 3 unseen) vs 022

**Change.** Same as 022 (job 1011: R4 setup + sky 4σ, PE radius |mu| + 4σ, 10k steps, one
full H200), only the input:

- extraction `experiments/edgs_feat_dl3dv_sky_cplx_v2_trim` (job 1127, all 133 scenes, ~272 s
  each, 271 GB): RoMa coarse 560 x 994 (native aspect, no refine pass), VGG on the native
  960 x 540 image, DINO at the native aspect with stride 7, PCA `dl3dv140_native_s7_basis.npz`;
- sky sigma **trimmed** (`--sky_sigma_mode trim --sky_trim 0.0075`: drop the 0.75 % farthest
  from the per-axis median, then mean / std). The plain std blew up on far mis-triangulations
  (v2-std 165f5af8: PE radius 91, run 1079 cancelled); MAD was too tight (15–37 % outside on
  Mip-NeRF outdoor), trim 1 % 2.3–4.7 % outside. Over the 133 scenes: sigma 0.12–0.87x the old
  one (median 0.58), points outside the sphere median 4.2 % (old 1.2 %), max 7.9 % (ff592398);
  colour check median 0.024 (old 0.015), 4 scenes > 0.045 (c37109a5 0.051);
- `input.sigma_from=sky`: PE mu / sigma from the stored sphere (new option in
  `data/edgs_points.py`), so r = the sphere's radius in the training frame.

PE radii 4.20–7.94 (022: 5.00–32.06; 26fd2335 32.06 → 7.94, 032dee9f 16.08 → 7.49). Sky points
117k per seen scene on average (022: 105k). Job 1261, wandb `miw6hbr3`, ~1.7–2.2 s/step,
~58 GB reserved early on.

**Results** (finished 2026-10-24 20:44 UTC, 6 h 31 min), step 9999, PSNR / SSIM / LPIPS:

| | eval_seen (10) | eval_unseen (3) |
|---|---|---|
| 022 (1011) | 21.39 / 0.699 / 0.363 | 18.09 / 0.638 / 0.439 |
| **025** | 21.38 / 0.700 / 0.360 | **18.84** / 0.647 / 0.431 |

- Seen: a tie (−0.01 dB, SSIM / LPIPS marginally better); per scene −0.21…+0.26 dB, no pattern
  by bounded / unbounded. The bonsai v2 gain in SSIM / LPIPS (+0.017 / −0.018) does not show
  up on DL3DV at 960 x 540.
- Unseen +0.75 dB comes **entirely from 26fd2335**: 15.11 → 17.43 (+2.33 dB, LPIPS 0.604 → 0.566),
  the scene whose PE radius was 32.06 with the old sigma and is 7.94 now. 07d9f972 −0.04,
  0a1b7c20 −0.05. So the gain is most likely the robust sigma (PE scale), not the features.
- 025 is ahead on unseen from the first eval (19.0 vs 18.0 at 500 steps), again from 26fd2335.
- Next to separate the two: old features + trimmed sigma (needs a re-extraction of the sky /
  sigma only, or a sigma override computed from the old points).

---

## 2026-10-24 · 026 — wider backbone + decoder blocks (10 scenes, 30k steps)

**Finding.** The decoder never had blocks: `ModelConfig.dec_depths` stayed at LitePT's
`(0, 0, 0, 0)` since the scaffold (a45d6fd), so `dec_conv=True` did nothing and every
decoder stage was `Linear(coarse) + Linear(skip)` (+ norm / act). The finest stages were also
narrow (36 / 72 ch, stem 91 → 36). All runs underfit (025: train PSNR ~21.1 vs seen eval
21.4; 1016: ~19.3 vs 20.0), and the v2 features tied with the old ones in 025.

**Change** (overrides on 025's command, same data: v2_trim + `input.sigma_from=sky`):
`input.feature_mlp_dim` 64 → 128 (stem input 155), `model.enc_channels` (64, 128, 192, 384,
504), `model.enc_num_head` (4, 8, 12, 16, 21) (head dim 24: LitePT's 3D RoPE needs a multiple
of 6; 384 / 12 crashed, smoke job 1262), `model.dec_depths` (1, 1, 1, 1) conv blocks,
`model.dec_channels` (128, 128, 192, 384), `model.dec_num_head` (4, 4, 8, 16), `head.skip_dim`
128 (`hidden_dim` 128). 17.9M Muon + 10.3M AdamW params (025: 11.2M + 1.95M). Smoke (job 1263,
60 steps + eval): ~91 GB reserved, ~3.4 s/step (025: ~58 GB, ~2.1 s).
**30k steps** (`NUM_STEPS=30000`; LR schedule stretched over 30k, so its 10k point is not
directly comparable to 025's end), eval every 500, one full H200, 48 h limit; job 1264,
wandb `vkzauv9t`.

**Results: running** (started 2026-10-24 ~21:25 UTC, ETA ~31 h).

---

## Open questions / next

- Longer-term goal: chunked reconstruction with chunk-relative PE — see
  `docs/chunking_plan.md`.
- drop-path off for 016 → jobs 512 / 513: confirmed as the fix. Final: 512
  = 015, 513 = +0.23 dB. Open: how much of that is drop-path 0 alone (a 015
  at `drop_path=0` baseline was considered and not run); whether 015 itself
  loses something to drop-path in eval mode (not measured).
- All bonsai runs end with train PSNR ≈ eval PSNR + 0.4 (~22.8 vs ~22.4):
  the model **underfits** the training views, it does not overfit. The
  10 dB gap to MCMC (32.8) is capacity / representation (Gaussian count and
  placement, SH 1), not generalization. The old run with MCMC means as input
  reached 28.75, pointing at point placement / coverage.
- Untried option: `input.pos_norm=layer` (LayerNorm over the PE channels).
- The RoMa MLP's output is unnormalized (no norm after its last Linear); its
  range during training has not been measured.
- LitePT's stem output is not deterministic run to run for identical inputs
  (likely duplicate points per voxel in spconv's hash) — affects every run;
  not verified.
