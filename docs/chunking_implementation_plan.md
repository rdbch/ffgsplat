# Chunked reconstruction: take + implementation plan

2026-10-22. Follows `docs/chunking_plan.md` (2026-10-15) and the "Next steps"
slides of `200_informal_presentation.html`. Base model / settings: **019c**
(job 833/834, `scripts/019_train_dl3dv_edgs_skip.sh`, variant `c`: skip concat64
+ h128 + rgb residual, per-pixel bkgd, kNN views, frustum cull subset 0.05,
global Fourier PE F4 p95, voxel 0.005, offsets 1.5 vox, Muon 4e-4, 1M / 1.25M
points, 64 views / update). Nothing here changes 019c's code paths: all new
options default off, new script `020_*.sh` (don't edit 019's while it runs).

## Take

1. **Split "chunking" in two questions.**
   - *Memory / parallelism / density* (a scene too big for one forward). This
     does **not** need a new PE: for a finite scene the global PE is
     computable (`LitePtGSModel.forward(point, pos_radius=...)` already takes
     the whole cloud's radius, as frustum culling does). So it can be tested
     **on the 019c checkpoint, without retraining**.
   - *Unbounded / "infinite" scenes* (no global centre or r95). This needs a
     chunk-local PE (or none) and **retraining**. Do it after the first one
     works.
2. **The real payoff on DL3DV is density, not size.** EDGS gives ~4.5M points
   per scene; 019c sees a random 1M / 1.25M (22–28 %). Chunking (with
   per-chunk gradient checkpointing in training) is what lets the model see
   all of them. That is the experiment worth aiming at.
3. **Fixed-size cubes aren't needed for scale consistency.** LitePT works on a
   fixed metric voxel (0.005), so the network sees the same scale whatever
   the chunk extent. Chunk size only matters for (a) the point budget
   (memory) and (b) the PE if it's chunk-relative `(x − c)/(S/2)`. Hence:
   **partition by point budget (adaptive octree), with a fixed metric halo**,
   and pick a PE that doesn't depend on chunk size (global, none, or
   distances to the faces).
4. **Start with hard assignment, not blending.** Each chunk = core + halo; the
   halo points only give context, each point's Gaussian comes from the one
   chunk whose core owns it. With halo ≥ receptive field this matches the full
   forward and keeps "one Gaussian per point" with no quaternion / SH
   averaging. Blending (partition of unity, as in the old plan) only if seams
   show up with an affordable halo.
5. **The receptive field is probably large, so measure it first.** The decoder
   is conv-only, but encoder stages 3–4 are windowed attention over 1024
   serialized points at 0.04 / 0.08 voxels. On surface-like data that's up to
   ~32×32 coarse voxels ≈ 1.3–2.5 units, vs a scene r95 of 3.56. The halo
   could be a sizeable part of the scene. If so: chunks must be big (fewer,
   more expensive) or the model must be *trained* chunked so it stops relying
   on far context.
6. **Two exactness traps** (otherwise even one chunk ≠ full forward):
   - LitePT's `grid_coord` = `(coord − coord.min()) / grid_size`, so the voxel
     / pooling grid moves with each chunk's min. Compute `grid_coord` **once
     for the whole cloud** and per chunk subtract an origin that's a
     **multiple of 16 voxels** (2^4 poolings): the same pooling clusters as in
     the full forward.
   - Attention windows are slices of the serialization order, so they differ
     between chunked and full even far from the border. Plus the known
     stem nondeterminism. → measure the **noise floor** (full forward twice)
     and compare chunked against that, not against zero.

## Spatial partitioning

Frame: the scene's normalized frame (parser `normalize` + `target_radius
3.56`), after augmentation in training. Everything below is in those units;
voxel v = 0.005, coarsest voxel 16v = 0.08.

| option | how | + | − | use |
|---|---|---|---|---|
| A. uniform cubes | grid of side S (multiple of 0.08) + halo H | simple; one S everywhere (good for a chunk-relative PE) | point counts vary a lot (empty sky cubes, dense table); unbounded far field → many tiny chunks | baseline / sanity |
| **B. point-budget octree** | root cube aligned to 0.08 around all points; split a node into 8 while its core has > `N_max` points; leaves = cores (min side 0.08 × 2^k); each leaf + halo H | balanced memory per chunk → predictable batching; dense regions → small chunks, sparse far field → big chunks (handles unbounded with no special shell); deterministic | chunk sizes vary (fine with global / face-distance PE, not with `(x−c)/(S/2)`) | **inference + training** |
| C. view-driven | chunk = points in the frustums of a cluster of cameras (what 019c's kNN + cull already does) | aligned with what a step renders | not a partition (overlapping, camera-dependent) | **training sampling** only |
| D. contracted-space grid | grid in mip-NeRF-360 contracted coords | bounded number of chunks | voxel size varies in world space | not planned (B covers far field) |

Recommended: **B for the geometry, C for choosing what to train on**.

Details for B:
- Halo H in metric units, from Phase 1 (candidates 0, 0.08, 0.16, 0.32,
  0.64, 1.28). Halo points: every point within H of the core box (L∞).
- `N_max` = core points per chunk; the forward sees core + halo. Candidates
  500k / 250k / 125k / 60k.
- Chunks are batched into one LitePT call with `Point.offset` (LitePT keeps
  the batches apart in serialization and attention, `batch << 48`), up to a
  points-per-call budget; groups run sequentially. Multi-GPU later = split the
  chunk list across ranks.
- Per chunk: global coords + global PE radius (019c), `grid_coord` from the
  global one minus a 16v-aligned origin, the cut to 2^16 voxels per chunk
  instead of per scene.
- Output: concat core predictions, ordered back to the input point order;
  rendering unchanged (global, so a border Gaussian can still extend into the
  neighbour).

## Sky / background: a sphere of fake points around the scene

Idea (2026-10-22): surround the scene with a spherical shell of synthetic points so
that sky and far background, where EDGS has no or few points, get Gaussians
too. Each ray then hits something, so we no longer rely on big far Gaussians
or the background colour.

Design:
- **Geometry: a sphere at 4σ.** Fit one Gaussian to the scene's points (mean
  μ, covariance Σ); the sphere is centred at μ with radius
  **`R_sky = 4σ`**, σ = √(tr Σ / 3) (isotropic σ of that Gaussian). A robust
  σ (1.4826 × MAD per axis) is the fallback if the far tail of unbounded
  scenes inflates σ. Points are sampled uniformly with a Fibonacci lattice (no
  poles / seams, uniform in angle). Scene points beyond 4σ (outside the sky)
  are counted and logged; they're occluded by the sky from the inside.
- **Own coarse grid, 4 points per voxel.** At v = 0.005 every shell point
  would be an isolated voxel (convs and pooling see nothing). The shell runs
  in its **own LitePT call** with its own `grid_size` g, chosen so that the
  voxels the sphere crosses hold **4 points** on average. A voxel cuts ≈ g²
  of the surface, so density 4 / g², spacing ≈ g / 2, and for N shell points
  `N = 4π R_sky² · 4 / g²` → nominal **`g = R_sky · √(16π / N)`**. E.g.
  N = 200k: g ≈ 0.0159 · R_sky. Voxels cut at a corner hold fewer points, so
  g is calibrated (bisection) until the measured mean points per occupied
  voxel is 4; mean / min / median are logged.
- **Viewer:** `$WORK/viewers/sky_sphere/` (Mip-NeRF EDGS points + the 4σ
  sphere, coloured by voxel). First numbers (EDGS points, N = 200k):

  | scene | σ fit / robust | 4σ / r95 | points outside | g calibrated (nominal) | ppv median / min / max | 1-point voxels | training frame (r95 = 3.56): R, g |
  |---|---|---|---|---|---|---|---|
  | garden | 0.94 / 0.50 | 1.30 | 1.4 % | 0.067 (0.059) | 4 / 1 / 9 | 15 % | 4.6, 0.082 |
  | bonsai | 1.11 / 0.64 | 1.38 | 0.35 % | 0.079 (0.070) | 4 / 1 / 9 | 15 % | 4.9, 0.087 |

  - 4σ of the Gaussian fit lands at only ~1.3–1.4 × r95: just outside the
    bulk of the points; 1.4 % of garden's points (far background) are
    beyond it, hidden behind the sky from the inside. Robust σ is ~half,
    so 4σ_robust would cut through the scene: not usable as is.
  - bicycle (EDGS points concentrated near the centre, r50 0.42 vs bonsai
    1.61): σ 0.70, 4σ = 2.82 < r99 3.96, **2.9 % of the points outside**.
    The sky can learn to go transparent there (render loss on its opacity),
    but its own forward doesn't see what's behind it. Fixes: R = max(4σ,
    r99), and/or a per-sky-point input "scene points beyond R in this
    direction" (count / nearest distance in a small cone), so the network
    knows where real content is behind the shell.
  - At N = 200k, g ≈ 0.08 in the training frame = LitePT's coarsest voxel
    (16 × 0.005): the shell at its own base voxel matches the scene's
    deepest level.
- **Chunks of the shell.** The same octree code (`octree_chunks`) on the shell
  points with g as the voxel (origins aligned to 16 g), budget `N_max` as for
  the scene; halo in units of g. The shell's chunks never mix with the
  scene's (different grid), so there's no near ↔ shell blending to solve.
- **Inputs: sky points are treated like the EDGS points** (decided
  2026-10-22; no panorama). Scene points: rgb | MLP(RoMa 104-d) → 64 | PE, each
  with one reference view (`ref_cam`, `ref_uv`). Sky points get the same:
  1. Fixed lattice per scene (4σ, 4 points / voxel), built in the world frame
     at extraction time; like the EDGS points, a random subset per training
     step, rotated / scaled by the augmentation.
  2. **Frustum cull** every training view (train split only, no eval views:
     no leak). Sky points seen by no view are **dropped**, as EDGS only has
     points where views see something.
  3. **Minimum set of views: greedy set cover** over the per-view culled sets.
  4. Each kept sky point's reference view = the chosen view where it's most
     central; rgb and the 104-d features sampled at its projection there
     (same VGG / DINO encoders, same global PCA basis, L2-normalized per
     block): **the same feature space and the same MLP as the EDGS points**.
     Projection of the point at R (as for points); occlusion ignored at first
     (whatever is in front is sampled).
  5. Plus the flag `is_sky`; PE as for the other points (clamped, or the
     direction).

  Estimate (pure-Python check, 20k-point lattice subsample, train views =
  every view but idx % 8 == 0):

  | scene | train views | sky seen by ≥ 1 view | per view | greedy cover 90 % / 99 % / 100 % |
  |---|---|---|---|---|
  | bonsai | 255 | 59.7 % | 6.8 % | 10 / 16 / 23 views |
  | bicycle | 169 | 81.5 % | 6.1 % | 16 / 25 / 35 views |

  So 200k lattice points → ~120k (bonsai) / ~163k (bicycle) kept, and only
  23–35 views to encode. Where it runs: a third step of `edgs_extract_ref.py`
  after the points loop (flag `--sky`), and the same code as a standalone
  script for the already extracted scenes (no RoMa matching needed, only the
  chosen views' encoder passes). Output `sky.npz` next to `features.npz`
  (xyz, rgb, feat, ref_cam, ref_uv), same layout.
- **Head bounds per point.** `max_scale` 0.4 and `max_offset` 1.5 vox are far
  too small at the shell: for shell points the caps become ∝ the spacing
  g / 2 (e.g. scale ≤ 2 × spacing, offset ≤ 0.5 × spacing, tangential to the
  sphere), quats free, opacity free.
- **Rendering.** Rendered together with the scene. Shell Gaussians sit behind
  everything, so an occluded part (e.g. behind indoor walls) gets no gradient
  and costs only memory. Opacity regularization pushes it towards 0.
- **Interaction with the per-pixel random background.** Training renders
  composite over per-pixel noise, which forces sky pixels to be covered: the
  shell takes over that role (opaque sky colour). Eval on black then stops
  penalizing uncovered sky.

Consequences:
- Changes the input width (flag) and the head → **needs training**. It can't
  be tested on the 019c checkpoint and is independent of chunking: it can be
  run as a 019c variant on its own (10-scene sub first).
- Before building it, **measure how much is uncovered today**: alpha maps of
  019c on the unbounded eval views (`alpha_mean` 0.998 in training suggests
  that large far Gaussians already cover most of the sky). It's worth it if a
  visible fraction is uncovered or the sky is blurry / wrong.
- Fallback if the points are awkward: an environment map (cubemap texture or
  small MLP on the ray direction) composited behind the render. Simpler, but
  not feed-forward per scene unless predicted from the images.

## Implementation

New, all default off:
- `models/chunking.py`
  - `octree_chunks(coord, n_max, halo, align=16*v, min_side=...)` → list of
    `(core_idx, ctx_idx, origin)`; `grid_chunks(coord, S, halo, offset)` for A.
  - `chunked_forward(model, coord, feat, grid_coord, pos_radius, chunks,
    points_per_call, checkpoint=False)` → splats for every point (hard core
    assignment). `checkpoint=True`: `torch.utils.checkpoint` per call group,
    so training activations peak at one group.
  - (later) `blend_weights` / quaternion-aware blending, only if needed.
- `core/trainer.py`: `_forward_backward` and `eval_scene` call
  `chunked_forward` when `chunk.enabled`; global `grid_coord` (with the
  augmentation's voxel jitter) computed before partitioning.
- Config block `chunk.*`: `enabled`, `mode` (octree | grid), `max_points`,
  `halo`, `size` (grid), `points_per_call`, `checkpoint`, `eval` (chunk in
  eval too), `pos_frame` (global | none | faces).
- `scripts/chunk_eval.py`: standalone inference/eval on a checkpoint (no
  training): full vs chunked, per-point Δparams, metrics, seam maps, memory,
  time.
- `scripts/020_train_dl3dv_chunked.sh` + `jobs/ffgsplat_020.slurm` (copies of
  019 with `CHUNK_*` env vars) for Phase 3.

Checks before any GPU job (unit-level, on CPU / small): one chunk covering
everything gives the same `grid_coord`, serialization and output as the full
forward (up to the noise floor); every point is in exactly one core;
halo ⊇ points within H.

## Phases

### Phase 1 — one scene, inference on the 019c checkpoint (no training)
- Checkpoint: latest 019c (`step_02xxxx.pt` now; repeat on 60k at the end).
- Scene: one `eval_seen` **bounded** DL3DV scene, then one **unbounded**.
- Input: the same fixed eval subset (1M / 1.25M, seed 0) as 019c's eval.
- Runs:
  1. noise floor: full forward ×2;
  2. one chunk = everything (must equal 1);
  3. sweep `N_max` {500k, 250k, 125k, 60k} × H {0, 0.08, 0.16, 0.32, 0.64,
     1.28}.
- Measure: |Δparams| (means, log-scales, opacity logit, SH0, quats) vs
  distance to the nearest core face → **effective RF → H**; PSNR / SSIM /
  LPIPS vs the full forward; error maps near projected core boundaries
  (seams); peak memory and time per call.
- Then the density test: chunked forward on **all 4.5M EDGS points** (same H,
  `N_max`), vs the 1M subset. Density shift (trained at ~1M), so it may also
  get worse: that tells whether density needs training (Phase 3b).
- Sky coverage (for the sky sphere): alpha maps and the fraction of pixels with
  alpha < 0.5 on the unbounded scene's eval views, full forward.
- GPU: one 2g.35gb slice (inference only; 1g.18gb if it fits).
- Output: the (N_max, H) to use, and how much chunking costs a model never
  trained for it.

### Phase 2 — DL3DV, all eval scenes, inference
- The 23 eval scenes (10 `eval_seen` + 13 `eval_unseen`), chosen (N_max, H)
  and one smaller/larger setting; 1M subset and all points.
- Compare per scene with 019c's own eval at the same step; bounded and
  unbounded separately; worst scenes looked at by hand.
- Same slice, one job looping over the scenes (or a Slurm array).

### Phase 3 — chunked training (019c settings)
Training step: scene, kNN views + frustum cull as 019c → octree partition of
the culled cloud (random root offset per step, a multiple of 16v, so borders
move) → chunks batched + gradient checkpointing per group → hard assignment →
render the 8 views as today, today's loss.

Variants:
- **3a** chunked, global PE, same 1M / 1.25M budget: isolates chunking (should
  match 019c; the model learns to live with halo-only context).
- **3b** 3a + density: draw from all 4.5M points (cull `full`), more points
  per step made affordable by the checkpointing.
- **3c** chunk-local PE (`faces`: distances to the 6 core faces, metric,
  clipped at H) or no PE: what unbounded / infinite scenes need. Compare with
  3a.
- **3d** sky sphere (see "Sky / background"): first as a plain 019c variant
  (no chunking) on the 10-scene sub, compared with R4; then combined with the
  best of 3a–3c. Measure 019c's uncovered sky (Phase 1) before.

Order, as for 019: **10-scene sub runs first** (`TRAIN_ON=eval_seen`, grad
accum 4, 10k steps, 3g.71gb slices; 019c-sub = R4 as the reference), then
the **120-scene full run** (2 × H200, 60k, chain) of the winner, compared
with 019c full.

### Phase 4 — scale (later)
Chunks across GPUs / processes; larger scenes; blending if seams remain;
optionally smaller attention windows if the RF forces huge halos.

## Open decisions
1. `N_max` and H (Phase 1 decides).
2. Hard assignment enough, or blending needed.
3. PE for 3c: faces vs none vs relative (needs fixed S → option A).
4. Whether the full-density run (3b) uses a fixed points-per-step budget or
   all points.
5. Eval for the trained models: chunked or full (both logged at first).
6. Sky sphere: N (R_sky = 4σ, g follows from 4 points per voxel), colour
   init (image median vs grey), PE of shell points (clamped vs
   direction-only).
