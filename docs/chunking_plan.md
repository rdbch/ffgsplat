# Chunked reconstruction: plan

2026-10-15

## Goal

Reconstruct scenes of any size by splitting the input point cloud into
**fixed-size, overlapping cubes**, running the point model on each cube
independently (in parallel, on several GPUs if needed), **blending** the
predictions in the overlaps, and rendering all Gaussians together as one
scene. Positional information is encoded **relative to the chunk**, not the
scene.

```
scene points ──► fixed-size cubes (+ overlap), grid placed after augmentation
                      │  each chunk = an independent "scene"
                      ▼
      per chunk: chunk-local PE → LitePT forward → GS params for its points
                      │  (independent → parallel)
                      ▼
      blend predictions of points in overlaps (partition of unity)
                      ▼
      one set of Gaussians ──► render full images (as today)
```

Only the network forward is chunked. Rendering stays global, so a Gaussian
near a chunk border can still extend into its neighbour.

## Prerequisite

Settle the eval collapse of the positional-encoding runs first (see
`research_log.md`, 016 entries): jobs 512 / 513 test whether drop-path's
train/eval mismatch is the cause. Chunking itself doesn't depend on PE.

## Design

### Chunks
- **Fixed-size cubes** of side `S` (normalized scene units), on a regular grid.
  Fixed size, so the network sees the same metric scale everywhere; point
  counts per chunk vary (accepted).
- Each chunk is extended by an **overlap** `O` on every side. A point inside
  the overlap of several chunks is processed by each of them.
- **Training:** the grid is placed *after* augmentation, with a **random
  offset** each step (chunk boundaries move every step).

### Blending (overlaps)
- Weights forming a partition of unity, from each point's distance to the
  chunk's faces:

  ```
  w_c(x) = ∏_axes ramp(dist(x, face of c) / O)     # 0 at the chunk's outer edge → 1 inside
  params(x) = Σ_c w_c(x) · params_c(x) / Σ_c w_c(x)
  ```

  A point near the outer edge of chunk A (little context in A) gets ~0
  weight from A ("alpha fading"); every point ends with exactly one Gaussian.
- Blend in a consistent space:
  - log-scales, SH, offsets: linear;
  - opacity: logit space (or post-sigmoid; pick one);
  - quaternions: align signs to a reference first (flip if ⟨q_a, q_b⟩ < 0),
    average, normalize.
- Same blending in training (gradients flow through the weights' combination),
  so the network learns to agree with itself in the overlaps.

### Overlap width = receptive field (to be measured)
- `O` must be ≥ the model's effective receptive field. Rough scale: stem 5³
  and 3³ convs, 4 poolings (coarsest voxel 16 × 0.005 = 0.08), plus attention
  windows of 1024 serialized points at stages 3–4 (their spatial extent
  depends on density). Don't guess; measure (Phase 1).

### Positional encoding, chunk-relative
- Center = the **geometric center of the cube** (not a data statistic).
- Candidates (open decision, see below):
  - relative: `p = (x − c) / (S/2)` → core in [−1, 1], overlap slightly beyond;
  - metric: `p = (x − c) / L` with a fixed `L` (meaning independent of `S`);
  - **distances to the 6 chunk faces**, metric, clipped at the receptive
    field: tells the network directly how much context it has and in which
    direction; independent of chunk size.
- With random grid offsets the same point lands at a different in-chunk
  position every step, so chunk-local PE **cannot memorize scene locations**.
  Its value is border/context awareness. Expect the +1 dB of global PE
  (likely memorization) not to carry over; judge it on seams.

### Chunk size
- **Sweep** as an experiment: e.g. `S` = 0.5 / 1 / 2 normalized units
  (quality vs parallelism; smaller → more seams, more overlap overhead).
- **Random `S` per step** as augmentation, so one model works for any chunk
  size at inference. Interacts with the PE choice (relative vs metric vs
  face distances).

### Far points (unbounded Mip-NeRF 360 background)
| option | how | notes |
|---|---|---|
| a. drop | remove points beyond radius R before chunking | trivial; loses background (PSNR hit on views that see it). For the first chunking experiments. |
| b. contraction | chunk / voxelize in mip-NeRF 360 contracted space | bounded, finite chunks; voxel size varies in world space → scale predictions inconsistent unless the network knows the local stretch. Most invasive. |
| c. background shell | near field: fixed cubes at voxel 0.005; beyond R: shell chunks with a coarser `grid_size` (4–8×) | fits LitePT (grid size per call); far content is low-res anyway; two regimes + near/shell seams. **Target solution.** |
| d. outside the network | far points as plain optimized / EDGS Gaussians | hybrid, not fully feed-forward. |

Plan: **a** first, then **c**.

## Phases

### Phase 1 — chunked inference on an existing model (no retraining)
- Model: 015 final checkpoint (bonsai, offsets 1.5 vox, no PE).
- Implement `chunk_points(coord, S, O, offset)` → per-chunk index lists + blend
  weights, and `chunked_forward(model, coord, feat, ...)` → blended splats.
- Far points: option a (drop beyond R) and also full scene, for reference.
- Measure, against the full (unchunked) forward of the same checkpoint:
  - parameter difference per point vs distance to the nearest chunk face,
    for increasing `O` → **effective receptive field**;
  - eval PSNR / SSIM / LPIPS of chunked vs full, per `S` and `O`;
  - PSNR restricted to image regions where chunk borders project (seam
    visibility).
- Output: the `O` to use, and how good chunking is with a model never trained
  for it.

### Phase 2 — chunked training
- Training forward = chunked forward + blending (chunks run sequentially:
  lower peak memory), render full images, today's loss.
- Random grid offset per step; then random `S`.
- Chunk-local PE (relative / metric / face distances) as options.
- Compare with Phase 1 (same `S`, `O`) and with 015: overall metrics and
  seam metrics.

### Phase 3 — background shell
- Points beyond R grouped into shell chunks with a coarser `grid_size`;
  blend near-field ↔ shell like any other overlap.

### Phase 4 — scale
- Parallel chunk processing (several GPUs / processes), larger scenes
  (multi-scene setup, outdoor scenes, DL3DV).

## Where it plugs into the code

- `core/trainer.py`
  - `_forward_backward` (`self.model(point)` after `augment(...)`, ~l.853):
    replace by `chunked_forward` when chunking is on.
  - `eval_scene` (~l.1018): same, with a fixed (non-random) grid.
  - `_build_scene` outlier filter (`half_extent`, 2^16-voxel limit): per chunk
    instead of per scene — chunking also removes that global limit.
- `data/augment.py`: `augment` / `jittered_grid_coord` — grid jitter becomes
  per chunk; the random chunk-grid offset is new.
- `models/point_model.py`: unchanged per chunk; chunk-local PE via
  `models/pos_encoding.py` (new center / divider options, or face distances).
- New module, e.g. `models/chunking.py`: chunk assignment, blend weights,
  quaternion-aware blending, `chunked_forward`.
- New config block, e.g. `chunk.*`: `enabled`, `size` (S), `overlap` (O),
  `size_range` (random S), `random_offset`, `far_radius` (R),
  `far_mode` (drop | shell), `shell_grid_scale`, `pos_frame`
  (relative | metric | faces).

## Open decisions

1. Chunk-local position input: relative p, metric p, face distances, or a
   combination.
2. Blend ramp shape (linear vs smooth, e.g. cos²) and opacity blend space.
3. Far-point radius R and, for the shell, its grid scale.
4. Chunk sizes for the sweep, and the random-`S` range for training.
