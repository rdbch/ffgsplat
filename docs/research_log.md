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
