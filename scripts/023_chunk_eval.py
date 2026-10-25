#!/usr/bin/env python3
"""Chunked vs full forward of a trained model on DL3DV scenes (023, inference only).

    python scripts/023_chunk_eval.py --ckpt <results/.../ckpts/step_*.pt> --scenes 1de58be5 032dee9f \
        [--cells 256 512] [--overlaps 0 5 20 64] [--points_per_call 1500000] [key=value ...]

Per scene (its fixed eval subset of points, + the sky with `input.sky`, as in training):
1. checks of the partition (every point in exactly one core, cell sides / origins on the
   16-voxel grid, overlap = exactly the points within the overlap of the cell);
2. the full forward twice (noise floor: LitePT's attention windows / stem are not
   bit-deterministic) and one chunk = everything (must match the noise floor);
3. for every (cell_voxels, overlap_voxels): `models.chunking.chunked_forward`, then
   - per-point |difference| to the full forward of opacity, rgb (degree-0 SH), log-scale
     and offset, overall and by distance of the point to the nearest face of its cell;
   - renders of the eval views: PSNR / SSIM / LPIPS vs the ground truth (and of the
     full forward), PSNR of the chunked render vs the full render;
   - chunks, calls, overlap points (share of extra points), time, peak memory.
Results to `--out` (JSON) and the log.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
from gsplat.rendering import rasterization
from omegaconf import OmegaConf
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from core.trainer import Point
from core.trainer_dl3dv import DL3DVTrainConfig, TrainerDL3DV
from models.chunking import ALIGN, chunked_forward, global_grid_coord, grid_chunks

C0 = 0.28209479177387814
FACE_BINS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]          # distance to the nearest cell face (scene units)


def full_forward(model, coord, feat, grid, pos_radius):
    point = Point(coord=coord, feat=feat, grid_size=grid, offset=torch.tensor([coord.shape[0]], device=coord.device))
    return model(point, pos_radius=pos_radius)


def activated(coord, s):
    return {"means": coord + s["offsets"] if "offsets" in s else coord, "quats": s["quats"],
            "scales": torch.exp(s["scales"]), "opacities": torch.sigmoid(s["opacities"]),
            "colors": torch.cat([s["sh0"], s["shN"]], dim=1)}


def param_diff(a, b):
    """Per-point |difference| of the interpretable parameters ([N] each)."""
    d = {"opacity": (torch.sigmoid(a["opacities"]) - torch.sigmoid(b["opacities"])).abs(),
         "rgb": (C0 * (a["sh0"] - b["sh0"])).abs().amax(dim=(1, 2)),
         "log_scale": (a["scales"] - b["scales"]).abs().amax(1)}
    if "offsets" in a:
        d["offset"] = (a["offsets"] - b["offsets"]).norm(dim=1)
    return d


def summarize_diff(d, face_dist=None):
    out = {}
    for k, v in d.items():
        v = v.float()
        row = {"mean": v.mean().item(), "p99": torch.quantile(v[torch.randperm(len(v))[:1_000_000]], 0.99).item()}
        if face_dist is not None:
            b = torch.bucketize(face_dist, torch.tensor(FACE_BINS, device=v.device))
            row["by_face_dist"] = [v[b == i].mean().item() if (b == i).any() else None for i in range(len(FACE_BINS) + 1)]
        out[k] = row
    return out


def face_distance(grid_coord, chunks, grid):
    """Distance (scene units) of every point to the nearest face of its core cell."""
    d = torch.empty(len(grid_coord), device=grid_coord.device)
    g = grid_coord.long()
    for c in chunks:
        p = g[c.core]
        d[c.core] = (torch.minimum(p - c.lo, c.hi - 1 - p).amin(1).float() + 0.5) * grid
    return d


def check_partition(grid_coord, chunks, overlap_voxels):
    n = len(grid_coord)
    count = torch.zeros(n, dtype=torch.long, device=grid_coord.device)
    for c in chunks:
        count[c.core] += 1
    errors = []
    if not (count == 1).all():
        errors.append(f"{int((count != 1).sum())} points not in exactly one core")
    g = grid_coord.long()
    for c in chunks[:50]:
        if ((c.hi - c.lo) % ALIGN).any() or (c.lo % ALIGN).any():
            errors.append(f"cell {c.lo.tolist()}-{c.hi.tolist()} off the {ALIGN}-voxel grid")
            break
        inside = ((g >= c.lo - overlap_voxels) & (g < c.hi + overlap_voxels)).all(1)
        core = ((g >= c.lo) & (g < c.hi)).all(1)
        expected = set((inside & ~core).nonzero().squeeze(1).tolist())
        if set(c.overlap.tolist()) != expected or set(c.core.tolist()) != set(core.nonzero().squeeze(1).tolist()):
            errors.append(f"cell {c.lo.tolist()}: core / overlap sets wrong")
            break
    return errors


@torch.no_grad()
def render_metrics(splats, ref, loader, device, sh_degree, near, far, metrics):
    out = {"psnr": [], "ssim": [], "lpips": [], "psnr_vs_full": []}
    for batch in loader:
        c2w, Ks = batch["camtoworld"].to(device), batch["K"].to(device)
        pixels = batch["image"].to(device) / 255.0
        h, w = pixels.shape[1:3]
        for i in range(len(pixels)):
            kw = dict(viewmats=torch.linalg.inv(c2w[i: i + 1]), Ks=Ks[i: i + 1], width=w, height=h,
                      sh_degree=sh_degree, near_plane=near, far_plane=far)
            r = rasterization(**{k: splats[k] for k in ("means", "quats", "scales", "opacities", "colors")}, **kw)[0]
            r = r.clamp(0, 1).permute(0, 3, 1, 2)
            gt = pixels[i: i + 1].permute(0, 3, 1, 2)
            out["psnr"].append(metrics["psnr"](r, gt).item())
            out["ssim"].append(metrics["ssim"](r, gt).item())
            out["lpips"].append(metrics["lpips"](r, gt).item())
            if ref is not None:
                rr = rasterization(**{k: ref[k] for k in ("means", "quats", "scales", "opacities", "colors")}, **kw)[0]
                out["psnr_vs_full"].append(metrics["psnr"](r, rr.clamp(0, 1).permute(0, 3, 1, 2)).item())
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", type=int, default=[256, 512], help="cell side, voxels (x16)")
    ap.add_argument("--overlaps", nargs="+", type=int, default=[0, 5, 20, 64], help="overlap, voxels")
    ap.add_argument("--points_per_call", type=int, default=1_500_000)
    ap.add_argument("--max_core", type=int, default=None, help="split cells above this many core points")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--out", default=None)
    args, overrides = ap.parse_known_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.merge(OmegaConf.structured(DL3DVTrainConfig), ckpt["config"], OmegaConf.from_dotlist(overrides))
    cfg.trainer.output_dir = "/tmp/unused"
    trainer = TrainerDL3DV(cfg)
    names = []
    for prefix in args.scenes:
        hits = [n for n in trainer.split["environment"] if n.startswith(prefix)]
        if len(hits) != 1:
            sys.exit(f"scene {prefix!r}: {len(hits)} matches")
        names.append(hits[0])
    built = [trainer._build_dl3dv_scene(n) for n in names]
    trainer.feature_channels = built[0][1] + int(cfg.input.sky)
    trainer.build_model()
    trainer.model.load_state_dict(ckpt["model"])
    model = trainer.model.eval()
    device, grid = trainer.device, cfg.trainer.point_grid_size
    sh_degree = min(ckpt["step"] // cfg.trainer.sh_degree_interval, cfg.head.sh_degree)
    metrics = {"psnr": PeakSignalNoiseRatio(data_range=1.0).to(device),
               "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
               "lpips": LearnedPerceptualImagePatchSimilarity(net_type=cfg.trainer.lpips_net, normalize=True).to(device)}
    near, far = cfg.trainer.near_plane, cfg.trainer.far_plane
    print(f"checkpoint {args.ckpt} (step {ckpt['step']}), voxel {grid}, sky {cfg.input.sky}, "
          f"pos_radius {cfg.input.pos_radius}, SH degree {sh_degree}")

    results = {"ckpt": args.ckpt, "step": ckpt["step"], "face_bins": FACE_BINS, "scenes": {}}
    for scene, _, _ in built:
        trainer._load_points(scene)
        view = trainer._eval_view(scene)
        scene.means = scene.feat = None
        coord, feat = view.means.to(device), view.feat.to(device)
        n = len(coord)
        print(f"\n[{scene.name[:8]}] {scene.environment}: {n} points, PE radius "
              f"{view.fixed_pos_radius if view.fixed_pos_radius is not None else 'percentile'}", flush=True)
        res = {"environment": scene.environment, "num_points": n, "runs": []}
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            full = full_forward(model, coord, feat, grid, view.fixed_pos_radius)
            torch.cuda.synchronize()
            res["full"] = {"time_s": time.time() - t, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
            full2 = full_forward(model, coord, feat, grid, view.fixed_pos_radius)
            res["noise_floor"] = summarize_diff(param_diff(full2, full))
            del full2
            full_act = activated(coord, full)
            if not args.no_render:
                res["full"].update(render_metrics(full_act, None, scene.eval_loader, device, sh_degree, near, far, metrics))
            print(f"  full: {res['full']}\n  noise floor: " + ", ".join(
                f"{k} {v['mean']:.2e}" for k, v in res["noise_floor"].items()), flush=True)

            grid_coord = global_grid_coord(coord, grid)
            extent = int(grid_coord.max()) + 1
            one_cell = ((extent + ALIGN - 1) // ALIGN) * ALIGN
            for cell, overlap in [(one_cell, 0)] + [(c, o) for c in args.cells for o in args.overlaps]:
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                out, chunks = chunked_forward(model, coord, feat, grid, view.fixed_pos_radius, cell, overlap,
                                              args.points_per_call, max_core=args.max_core, return_chunks=True)
                torch.cuda.synchronize()
                run = {"cell_voxels": cell, "overlap_voxels": overlap, "cell": cell * grid, "overlap": overlap * grid,
                       "time_s": time.time() - t, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "num_chunks": len(chunks), "max_chunk_points": max(c.num_points for c in chunks),
                       "extra_points": sum(len(c.overlap) for c in chunks) / n,
                       "check": check_partition(grid_coord, chunks, overlap)}
                run["diff"] = summarize_diff(param_diff(out, full), face_distance(grid_coord, chunks, grid))
                if not args.no_render:
                    run.update(render_metrics(activated(coord, out), full_act, scene.eval_loader, device,
                                              sh_degree, near, far, metrics))
                res["runs"].append(run)
                label = "one chunk" if cell == one_cell else f"cell {cell} vox ({cell * grid:.2f}), overlap {overlap} vox"
                print(f"  {label}: {len(chunks)} chunks, +{run['extra_points'] * 100:.1f} % points, "
                      f"max {run['max_chunk_points']} per chunk, {run['time_s']:.1f} s, {run['peak_gib']:.1f} GiB"
                      + (f", PSNR {run['psnr']:.3f} (full {res['full']['psnr']:.3f}), vs full {run['psnr_vs_full']:.2f} dB"
                         if "psnr" in run else "")
                      + "\n    |diff| mean: " + ", ".join(f"{k} {v['mean']:.2e}" for k, v in run["diff"].items())
                      + "\n    opacity |diff| by distance to the cell face " + str(
                          [None if x is None else round(x, 5) for x in run["diff"]["opacity"]["by_face_dist"]])
                      + (f"\n    CHECK FAILED: {run['check']}" if run["check"] else ""), flush=True)
                del out, chunks
                torch.cuda.empty_cache()
        results["scenes"][scene.name] = res
        del coord, feat, full, full_act
        scene.eval_view = None
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
