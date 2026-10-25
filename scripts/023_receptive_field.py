#!/usr/bin/env python3
"""Effective receptive field of a trained model (for the chunk overlap, 023).

    python scripts/023_receptive_field.py --ckpt <results/.../ckpts/step_*.pt> \
        --scenes 1de58be5 2beaca31 032dee9f 9fb0588f [--queries 32] [--sky_queries 8] [key=value ...]

For each scene (DL3DV, loaded as `TrainerDL3DV` does: its fixed eval subset of points,
+ the sky rows if the checkpoint was trained with `input.sky`), the model is run on the
whole cloud, and for a query point q the gradient of one scalar of its predicted
Gaussian (a fixed random projection of all its outputs: offsets, log-scales, quats,
opacity logit, SH) with respect to every input point's features is taken
(the "effective receptive field" of Luo et al. 2016). Influence of point j on q:
||d y_q / d feat_j||_2, the query itself excluded (the skip / colour residual make
it trivially dominant).

Per query: the cumulative share C(d) of the total influence coming from points within
distance d of q, i.e. (to first order) the share lost when everything farther than d
is cut away, which is what a chunk with overlap d does to a point on its border.
Reported per scene, over the queries: the distance at which C reaches 50 / 90 / 95 /
99 % (median and max over queries), C at fixed distances (0.025 = 5 voxels, 0.1 = 20,
0.32, 0.64, 1.28, 2.56), and the mean influence per point in distance bins
(density-free decay). Separately for scene queries and sky queries. Distances in the
normalized training frame (voxel = `trainer.point_grid_size`). Results also go to
`--out` (JSON).

The model config comes from the checkpoint (`config`); `key=value` overrides on top
(e.g. `dl3dv.root=...`). No training state is touched.
"""
import argparse
import json
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from core.trainer import Point
from core.trainer_dl3dv import DL3DVTrainConfig, TrainerDL3DV

CUTS = (0.025, 0.05, 0.1, 0.16, 0.32, 0.64, 1.28, 2.56)
LEVELS = (0.5, 0.9, 0.95, 0.99)


def query_influence(model, coord, feat, grid_size, pos_radius, q, proj):
    """||d (proj . y_q) / d feat_j|| for every input point j ([N], float64 on the CPU)."""
    feat = feat.detach().clone().requires_grad_(True)
    point = Point(coord=coord, feat=feat, grid_size=grid_size,
                  offset=torch.tensor([coord.shape[0]], device=coord.device))
    splats = model(point, pos_radius=pos_radius)
    y = torch.cat([splats[k][q].reshape(-1) for k in sorted(splats)])
    (y * proj[: y.numel()]).sum().backward()
    return feat.grad.norm(dim=1).double().cpu()


def summarize(dist, infl, edges):
    """Cumulative-share stats of one query (query itself already removed)."""
    order = torch.argsort(dist)
    d, w = dist[order], infl[order]
    cum = torch.cumsum(w, 0) / w.sum().clamp_min(1e-30)
    out = {"radius": {}, "share_within": {}}
    for level in LEVELS:
        i = int(torch.searchsorted(cum, torch.tensor(level, dtype=cum.dtype)))
        out["radius"][str(level)] = float(d[min(i, len(d) - 1)])
    for cut in CUTS:
        i = int(torch.searchsorted(d, torch.tensor(cut, dtype=d.dtype)))
        out["share_within"][str(cut)] = float(cum[i - 1]) if i > 0 else 0.0
    b = torch.bucketize(dist, edges)
    total = torch.zeros(len(edges) + 1, dtype=torch.float64).index_add_(0, b, infl)
    count = torch.zeros(len(edges) + 1, dtype=torch.float64).index_add_(0, b, torch.ones_like(infl))
    out["per_point"] = (total / count.clamp_min(1)).tolist()
    out["count"] = count.tolist()
    return out


def report(name, kind, stats, edges):
    if not stats:
        return {}
    radius = {lv: np.array([s["radius"][str(lv)] for s in stats]) for lv in LEVELS}
    share = {c: np.array([s["share_within"][str(c)] for s in stats]) for c in CUTS}
    pp = np.array([s["per_point"] for s in stats])
    pp = pp / np.maximum(pp.max(1, keepdims=True), 1e-30)    # relative to each query's largest bin
    print(f"  [{name[:8]} {kind}] {len(stats)} queries")
    print("    distance holding X of the influence (median / max over queries): " + ", ".join(
        f"{int(lv * 100)} %: {np.median(radius[lv]):.3f} / {radius[lv].max():.3f}" for lv in LEVELS))
    print("    share of the influence within d (median / min): " + ", ".join(
        f"{c:g}: {np.median(share[c]) * 100:.1f} / {share[c].min() * 100:.1f} %" for c in CUTS))
    lo = [0.0] + list(edges)
    print("    mean influence per point by distance (relative, median over queries): " + ", ".join(
        f"[{lo[i]:.3g},{edges[i]:.3g}) {np.median(pp[:, i]):.2e}" for i in range(len(edges))))
    return {"radius": {str(k): v.tolist() for k, v in radius.items()},
            "share_within": {str(k): v.tolist() for k, v in share.items()},
            "per_point_median": np.median(pp, 0).tolist(), "edges": list(edges)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes", nargs="+", required=True, help="DL3DV scene hashes (prefixes ok)")
    ap.add_argument("--queries", type=int, default=32, help="scene-point queries per scene")
    ap.add_argument("--sky_queries", type=int, default=8, help="sky-point queries per scene (if any)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args, overrides = ap.parse_known_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.merge(OmegaConf.structured(DL3DVTrainConfig), ckpt["config"],
                          OmegaConf.from_dotlist(overrides))
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
    trainer.model.eval()
    trainer.model.requires_grad_(False)                       # gradients w.r.t. the inputs only
    grid = cfg.trainer.point_grid_size
    print(f"checkpoint {args.ckpt} (step {ckpt['step']}); voxel {grid}; input.sky={cfg.input.sky}, "
          f"pos_radius={cfg.input.pos_radius}")

    edges = [0.005, 0.01, 0.025, 0.05, 0.1, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 1e9]
    edges_t = torch.tensor(edges, dtype=torch.float64)
    rng = np.random.default_rng(args.seed)
    results = {"ckpt": args.ckpt, "step": ckpt["step"], "scenes": {}}
    for scene, _, _ in built:
        trainer._load_points(scene)
        view = trainer._eval_view(scene)                       # fixed eval subset (+ sky), fp32
        scene.means = scene.feat = None
        coord = view.means.to(trainer.device)
        feat = view.feat.to(trainer.device)
        n = coord.shape[0]
        n_sky = int((view.feat[:, -1] > 0.5).sum()) if cfg.input.sky else 0
        proj = torch.randn(256, generator=torch.Generator().manual_seed(args.seed), dtype=torch.float32)
        proj = (proj / proj.norm()).to(trainer.device)
        print(f"[{scene.name[:8]}] {scene.environment}: {n} points ({n_sky} sky)", flush=True)
        scene_res = {}
        try:
            run_queries(trainer, coord, feat, grid, view, n, n_sky, args, rng, proj, edges, edges_t, scene, scene_res)
        except torch.cuda.OutOfMemoryError:
            print(f"[{scene.name[:8]}] out of GPU memory at {n} points; skipped", flush=True)
            scene_res["oom"] = True
        results["scenes"][scene.name] = {"environment": scene.environment, "num_points": n,
                                         "num_sky": n_sky, **scene_res}
        del coord, feat, view
        scene.eval_view = None
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"-> {args.out}")


def run_queries(trainer, coord, feat, grid, view, n, n_sky, args, rng, proj, edges, edges_t, scene, scene_res):
    """Scene-point and sky-point queries of one scene into `scene_res`."""
    for kind, pool, k in (("scene", np.arange(n - n_sky), args.queries),
                          ("sky", np.arange(n - n_sky, n), args.sky_queries)):
        if k == 0 or len(pool) == 0:
            continue
        stats = []
        for q in rng.choice(pool, size=min(k, len(pool)), replace=False):
            infl = query_influence(trainer.model, coord, feat, grid, view.fixed_pos_radius, int(q), proj)
            dist = (view.means.double() - view.means[q].double()).norm(dim=1)
            keep = torch.ones(n, dtype=torch.bool)
            keep[q] = False
            stats.append(summarize(dist[keep], infl[keep], edges_t))
            trainer.model.zero_grad(set_to_none=True)
        scene_res[kind] = report(scene.name, kind, stats, edges)
    print(f"    peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
