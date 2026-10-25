"""Chunked forward of `LitePtGSModel` on a voxel grid of cells (023).

The scene is cut into cubic cells of `cell_voxels` voxels per side (on LitePT's global
voxel grid, see `global_grid_coord`); each chunk = the points of one cell (its *core*)
+ the points within `overlap_voxels` of the cell (L-inf, its *overlap*: context only).
Every point is in exactly one core; its Gaussian is the one its chunk predicts for it
(hard assignment), so the output is one Gaussian per input point, in the input order,
exactly as the full forward. Cells with more than `max_core` points are split into 8
(up to `max_depth` times), so no chunk exceeds the budget.

Exactness with the full forward:
- `grid_coord` is computed once for the whole cloud (as LitePT would) and every chunk
  keeps those global voxel indices, so its pooling clusters (2^4 = 16 voxels at the
  coarsest stage) are the full forward's. Cell sides / origins are multiples of
  `ALIGN` = 16 voxels, so a cell never cuts a coarse voxel.
- The PE uses one radius for the whole cloud (`pos_radius`: the fixed |mu| + k sigma,
  or the full cloud's percentile radius, computed here once).
- What still differs: the context beyond the overlap, and the attention windows (each
  chunk is serialized on its own); see `docs/chunking_implementation_plan.md`.

Chunks are packed into LitePT calls of at most `points_per_call` points (core +
overlap), batched with `Point.offset` (LitePT keeps batches apart in pooling,
serialization and attention), so many small chunks (e.g. sparse sky cells) share a call.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from litept.model import Point

ALIGN = 16          # voxels: LitePT's 4 poolings of stride 2


@dataclass
class Chunk:
    lo: torch.Tensor          # [3] int64, cell box [lo, hi) in voxels (global grid)
    hi: torch.Tensor
    core: torch.Tensor        # [n_core] int64 point indices
    overlap: torch.Tensor     # [n_overlap] int64 point indices (context only)

    @property
    def num_points(self):
        return len(self.core) + len(self.overlap)


def global_grid_coord(coord: torch.Tensor, grid_size: float) -> torch.Tensor:
    """LitePT's own voxel indices of the whole cloud (`(coord - min) / grid_size`,
    truncated), [N, 3] int32: the full forward gets exactly these."""
    return torch.div(coord - coord.min(0)[0], grid_size, rounding_mode="trunc").int()


def grid_chunks(grid_coord: torch.Tensor, cell_voxels: int, overlap_voxels: int,
                max_core: Optional[int] = None, max_depth: int = 1,
                shift: Optional[torch.Tensor] = None) -> List[Chunk]:
    """Cells of `cell_voxels` (a multiple of ALIGN) voxels per side over `grid_coord`
    ([N, 3] int, global), each with its overlap points (within `overlap_voxels` of the
    cell, L-inf). `shift` ([3] int voxels, multiple of ALIGN): moves the cell grid (a
    random one per training step). Cells with more than `max_core` core points are
    split into 8 sub-cells, at most `max_depth` times (sides stay multiples of ALIGN,
    so cell_voxels must be divisible by ALIGN * 2^max_depth for full splits)."""
    if cell_voxels % ALIGN:
        raise ValueError(f"cell_voxels={cell_voxels} must be a multiple of {ALIGN}")
    g = grid_coord.long()
    if shift is not None:
        if (shift % ALIGN).any():
            raise ValueError(f"shift {shift.tolist()} must be a multiple of {ALIGN}")
        shifted = g + shift.to(g.device)
    else:
        shifted = g
    origin = torch.div(shifted.min(0)[0], cell_voxels, rounding_mode="floor") * cell_voxels
    cell = torch.div(shifted - origin, cell_voxels, rounding_mode="floor")
    keys, inverse = torch.unique(cell, dim=0, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    counts = torch.bincount(inverse, minlength=len(keys))
    starts = torch.cumsum(counts, 0) - counts
    offset = (shift.to(g.device) if shift is not None else torch.zeros(3, dtype=torch.long, device=g.device))

    boxes = []          # (lo, hi, core indices), lo / hi in unshifted voxel coordinates
    pending = []
    for i in range(len(keys)):
        lo = keys[i] * cell_voxels + origin - offset
        core = order[starts[i]: starts[i] + counts[i]]
        pending.append((lo, lo + cell_voxels, core, 0))
    while pending:
        lo, hi, core, depth = pending.pop()
        side = int(hi[0] - lo[0])
        if max_core is not None and len(core) > max_core and depth < max_depth and side // 2 >= ALIGN \
                and (side // 2) % ALIGN == 0:
            half = side // 2
            sub = torch.div(g[core] - lo, half, rounding_mode="floor")          # [n, 3] in {0, 1}
            code = sub[:, 0] * 4 + sub[:, 1] * 2 + sub[:, 2]
            for c in torch.unique(code).tolist():
                corner = torch.tensor([c >> 2 & 1, c >> 1 & 1, c & 1], device=g.device) * half
                pending.append((lo + corner, lo + corner + half, core[code == c], depth + 1))
        else:
            boxes.append((lo, hi, core))

    chunks = []
    member = torch.zeros(len(g), dtype=torch.bool, device=g.device)
    for lo, hi, core in boxes:
        inside = ((g >= lo - overlap_voxels) & (g < hi + overlap_voxels)).all(1)
        member.zero_()
        member[core] = True
        overlap = (inside & ~member).nonzero().squeeze(1)
        chunks.append(Chunk(lo, hi, core, overlap))
    return chunks


def pack_calls(chunks: List[Chunk], points_per_call: int) -> List[List[Chunk]]:
    """Greedy first-fit (largest chunk first) of chunks into calls of at most
    `points_per_call` points; a chunk larger than that gets a call of its own."""
    calls, sizes = [], []
    for chunk in sorted(chunks, key=lambda c: -c.num_points):
        n = chunk.num_points
        for i, size in enumerate(sizes):
            if size + n <= points_per_call:
                calls[i].append(chunk)
                sizes[i] += n
                break
        else:
            calls.append([chunk])
            sizes.append(n)
    return calls


def chunked_forward(model, coord: torch.Tensor, feat: torch.Tensor, grid_size: float,
                    pos_radius: Optional[float], cell_voxels: int, overlap_voxels: int,
                    points_per_call: int, max_core: Optional[int] = None, max_depth: int = 1,
                    shift: Optional[torch.Tensor] = None, return_chunks: bool = False):
    """`model(point, pos_radius)` on chunks, put back together: the same dict of [N, ...]
    tensors as the full forward (one row per input point, input order). With
    `pos_radius=None` and a PE, the full cloud's percentile radius is used for every
    chunk (what the full forward would compute)."""
    if pos_radius is None and getattr(model, "pos_encoding", "none") != "none":
        from models.pos_encoding import fourier_pos
        _, pos_radius = fourier_pos(coord, model.pos_freqs, model.pos_percentile, model.pos_include_raw,
                                    model.pos_clamp_percentile)
        pos_radius = float(pos_radius)
    grid_coord = global_grid_coord(coord, grid_size)
    chunks = grid_chunks(grid_coord, cell_voxels, overlap_voxels, max_core, max_depth, shift)
    out: Dict[str, torch.Tensor] = {}
    for call in pack_calls(chunks, points_per_call):
        idx = torch.cat([torch.cat([c.core, c.overlap]) for c in call])
        is_core = torch.cat([torch.cat([torch.ones(len(c.core), dtype=torch.bool, device=idx.device),
                                        torch.zeros(len(c.overlap), dtype=torch.bool, device=idx.device)])
                             for c in call])
        offset = torch.cumsum(torch.tensor([c.num_points for c in call], device=coord.device), 0)
        point = Point(coord=coord[idx], feat=feat[idx], grid_coord=grid_coord[idx],
                      grid_size=grid_size, offset=offset)
        splats = model(point, pos_radius=pos_radius)
        for key, value in splats.items():
            if key not in out:
                out[key] = value.new_empty((coord.shape[0], *value.shape[1:]))
            out[key][idx[is_core]] = value[is_core]
    return (out, chunks) if return_chunks else out
