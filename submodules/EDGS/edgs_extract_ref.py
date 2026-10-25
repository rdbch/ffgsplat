#!/usr/bin/env python3
"""
COLMAP reconstruction in -> EDGS initialization + PCA-reduced RoMa features of each
point's REFERENCE view out.

Slimmed-down sibling of edgs_extract.py: one feature vector per point, sampled at the
pixel the point was triangulated from (no neighbour slots), projected with a global
basis from edgs_fit_pca.py. Splat creation is identical to corr_init /
edgs_init_from_colmap.py.

    python edgs_extract_ref.py -s data/garden -o outputs/garden --images images_4 \
        --pca_basis pca/mipnerf_basis.npz --holdout_test

Output:

    point_cloud/iteration_0/point_cloud.ply   the initialization (standard 3DGS PLY)
    features.npz                              all per-point arrays, one file:
        feat     (N, D)  float16  PCA features, blocks laid out as in meta["blocks"]
        ref_cam  (N,)    int16    index into meta["camera_names"]
        ref_uv   (N, 2)  float32  reference pixel, normalized [-1, 1] (x, y)
        valid    (N,)    bool     passed the reprojection test (False rows have zero
                                  opacity in the PLY)
        is_sky   (N,)    bool     only with --sky: True for the sky rows (appended last)
        complexity (N,)  float32  only with --complexity_dir: the image complexity map of
                                  the row's reference view sampled at ref_uv
    features_meta.json                        block spans, basis path + hash, camera
                                              names, settings, sanity checks ("sky" entry
                                              with --sky)
    cfg_args, cameras.json, input.ply          written by / for 3DGS tooling

Row i of every array is row i of the PLY (checked against the PLY xyz before saving):
COLMAP SfM points are always removed and invalid points are kept (flagged in `valid`),
so nothing shifts rows. Uncompressed: np.load reads each array lazily, on access.

--sky adds a third step after the points (`add_sky_points`): a sphere of fake points
around the scene, at sky_sigma_k x sigma of one Gaussian fitted to the valid points
(centre = their mean, sigma = sqrt(tr(Sigma) / 3)), Fibonacci lattice of sky_points
points, its voxel size g set so that occupied voxels hold sky_points_per_voxel points
on average (stored in the meta). Sky points no training view sees are dropped; a
greedy set cover picks the minimum set of views seeing the rest, and each sky point's
reference view is the chosen view where it's most central. rgb + features are sampled
there exactly as for the points (occlusion ignored). PLY rows: rgb as f_dc, scale g / 2,
identity rotation, the median opacity of the valid points.

--complexity_dir (maps from ffgsplat's scripts/021_complexity_maps.py) adds a fourth step
(`add_complexity`): every row, points and sky alike, gets the complexity of its reference
view at ref_uv (bilinear, with the maps' window-centre convention from their meta.json).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO_ROOT,
           os.path.join(REPO_ROOT, "submodules", "gaussian-splatting"),
           os.path.join(REPO_ROOT, "submodules", "RoMa")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scene import Scene, GaussianModel  # noqa: E402
from utils.sh_utils import RGB2SH  # noqa: E402
from romatch import roma_indoor, roma_outdoor  # noqa: E402

from source.corr_init import (  # noqa: E402
    aggregate_confidences_and_warps,
    extract_keypoints_and_colors,
    k_closest_vectors,
    select_cameras_kmeans,
    triangulate_points,
)
from source.data_utils import scene_cameras_train_test_split  # noqa: E402
from source.utils_aux import set_seed  # noqa: E402

from edgs_extract import (  # noqa: E402
    BlockPCA,
    MultiScaleFeatureExtractor,
    build_dataset_args,
    cam_to_pil,
    clamp_selection_params,
    load_gs_optimization_defaults,
    write_cfg_args,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = p.add_argument_group("input / output")
    io.add_argument("-s", "--source_path", required=True)
    io.add_argument("-o", "--output", required=True)
    io.add_argument("--images", default="images")
    io.add_argument("--resolution", type=int, default=-1)
    io.add_argument("--white_background", action="store_true")
    io.add_argument("--data_device", default="cuda")
    io.add_argument("--holdout_test", action="store_true",
                    help="Hold out every 8th image (3DGS / ffgsplat test split).")

    sel = p.add_argument_group("initialization (defaults = EDGS full_eval.py)")
    sel.add_argument("--num_refs", type=int, default=180)
    sel.add_argument("--nns_per_ref", type=int, default=3)
    sel.add_argument("--matches_per_ref", type=int, default=25_000)
    sel.add_argument("--roma_model", choices=["outdoors", "indoors"], default="outdoors")
    sel.add_argument("--scaling_factor", type=float, default=0.001)
    sel.add_argument("--proj_err_tolerance", type=float, default=0.01)
    sel.add_argument("--final_scale_modifier", type=float, default=0.5)
    sel.add_argument("--sh_degree", type=int, default=3)

    ft = p.add_argument_group("features")
    ft.add_argument("--pca_basis", required=True,
                    help="Global basis from edgs_fit_pca.py. Its blocks decide which "
                         "RoMa strides are sampled.")

    sky = p.add_argument_group("sky points (third step, see the module docstring)")
    sky.add_argument("--sky", action="store_true", help="Append sky points (is_sky = True).")
    sky.add_argument("--sky_sigma_k", type=float, default=4.0, help="Sphere radius in sigmas.")
    sky.add_argument("--sky_points", type=int, default=200_000, help="Lattice points before culling.")
    sky.add_argument("--sky_points_per_voxel", type=float, default=4.0)

    cx = p.add_argument_group("complexity (fourth step, see the module docstring)")
    cx.add_argument("--complexity_dir", default=None,
                    help="Folder of per-image complexity maps (<stem>.tif + meta.json), "
                         "absolute or relative to --source_path.")

    misc = p.add_argument_group("misc")
    misc.add_argument("--seed", type=int, default=228)
    misc.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def init_with_ref_features(gaussians, scene, cfg, roma_model, device, extractor, pca):
    """
    corr_init.init_gaussians_with_corr (same as edgs_extract.init_with_features with
    the neighbour slots removed) + features sampled at each point's reference pixel.
    """
    M = cfg.matches_per_ref
    upper_thresh = roma_model.sample_thresh
    viewpoint_stack = scene.getTrainCameras().copy()
    n_refs = min(cfg.num_refs, len(viewpoint_stack))
    k = min(cfg.nns_per_ref, len(viewpoint_stack) - 1)

    poses = torch.stack([c.world_view_transform.flatten() for c in viewpoint_stack], dim=0)
    selected_indices = sorted(select_cameras_kmeans(poses.detach().cpu().numpy(), n_refs))
    closest = k_closest_vectors(poses, k).detach().cpu().numpy()

    all_new_xyz, all_new_dc, all_new_rest = [], [], []
    all_new_opacity, all_new_scaling, all_new_rotation = [], [], []
    rec = {key: [] for key in ("feat", "xyz", "ref_cam", "ref_uv", "valid")}
    uv_check = []   # |image(ref_uv) - point colour| per reference: proves ref_uv is the right pixel

    for source_idx in tqdm(selected_indices, desc="references"):
        with torch.no_grad():
            (certainties_max, warps_max, certainties_max_idcs, imA, imB_compound,
             certainties_all, warps_all) = aggregate_confidences_and_warps(
                viewpoint_stack=viewpoint_stack, closest_indices=closest,
                roma_model=roma_model, source_idx=source_idx, verbose=False)

            certainty = certainties_max.clone()
            certainty[certainty > upper_thresh] = 1
            certainty = certainty.reshape(-1)
            good_samples = torch.multinomial(
                certainty, num_samples=min(M, len(certainty)), replacement=False)
            M_eff = int(good_samples.shape[0])

            tri_points, errs1, errs2 = [], [], []
            kptsA_np = kptsA_color = None
            for nn_slot in range(len(warps_all)):
                matches_nn = warps_all[nn_slot].reshape(-1, 4)[good_samples]
                kptsA_np, kptsB_np, _, kptsA_color, _ = extract_keypoints_and_colors(
                    imA, imB_compound, certainties_max, certainties_max_idcs,
                    matches_nn, roma_model)
                P1 = viewpoint_stack[source_idx].full_proj_transform
                P2 = viewpoint_stack[int(closest[source_idx, nn_slot])].full_proj_transform
                pts, e1, e2 = triangulate_points(
                    P1=torch.stack([P1] * M_eff, dim=0), P2=torch.stack([P2] * M_eff, dim=0),
                    k1_x=kptsA_np[:M_eff, 0], k1_y=kptsA_np[:M_eff, 1],
                    k2_x=kptsB_np[:M_eff, 0], k2_y=kptsB_np[:M_eff, 1])
                tri_points.append(pts)
                errs1.append(e1)
                errs2.append(e2)

            err_stack = np.maximum(np.stack(errs1, axis=0), np.stack(errs2, axis=0))
            best_nn = np.argmin(err_stack, axis=0)
            best_err = np.min(err_stack, axis=0)
            idx_pts = torch.from_numpy(best_nn).long().to(device)
            cols = torch.arange(err_stack.shape[1], device=device)
            selected_points = torch.stack(tri_points, dim=0)[idx_pts, cols, :]

            # --- splats, identical to corr_init ---
            cam = viewpoint_stack[source_idx]
            new_xyz = selected_points[:, :-1]
            N = new_xyz.shape[0]
            bad = torch.tensor(best_err > cfg.proj_err_tolerance,
                               dtype=torch.float32).unsqueeze(1).to(device)

            all_new_xyz.append(new_xyz)
            all_new_dc.append(RGB2SH(torch.tensor(kptsA_color[:N].astype(np.float32) / 255.)).unsqueeze(1))
            all_new_rest.append(torch.stack([gaussians._features_rest[-1].clone().detach() * 0.] * N, dim=0))
            all_new_opacity.append(torch.stack([gaussians._opacity[-1].clone().detach()] * N, dim=0) * 0. - bad * 1e1)
            dist = torch.linalg.norm(cam.camera_center.clone().detach() - new_xyz, dim=1, ord=2)
            all_new_scaling.append(gaussians.scaling_inverse_activation(
                (dist * cfg.scaling_factor).unsqueeze(1).repeat(1, 3)))
            all_new_rotation.append(torch.stack([gaussians._rotation[-1].clone().detach()] * N, dim=0))

            # --- reference-view features ---
            blocks = extractor.sample(source_idx, cam_to_pil(cam), kptsA_np[:N])
            rec["feat"].append(pca.project(blocks).cpu().numpy().astype(np.float16))
            rec["xyz"].append(new_xyz.cpu().numpy().astype(np.float32))
            rec["ref_cam"].append(np.full(N, source_idx, dtype=np.int16))
            rec["ref_uv"].append(np.asarray(kptsA_np[:N], dtype=np.float32))
            rec["valid"].append(best_err[:N] <= cfg.proj_err_tolerance)

            grid = torch.as_tensor(kptsA_np[:N], dtype=torch.float32, device=device).view(1, -1, 1, 2)
            img = cam.original_image.to(device).float()[None]
            rgb_at_uv = torch.nn.functional.grid_sample(img, grid, align_corners=False)[0, :, :, 0].T
            colour = torch.as_tensor(kptsA_color[:N].astype(np.float32) / 255., device=device)
            uv_check.append(float((rgb_at_uv - colour).abs().mean()))

    all_new_xyz = torch.cat(all_new_xyz, dim=0)
    gaussians.densification_postfix(
        all_new_xyz.to(device),
        torch.cat(all_new_dc, dim=0).to(device),
        torch.cat(all_new_rest, dim=0).to(device),
        torch.cat(all_new_opacity, dim=0).to(device),
        torch.cat(all_new_scaling, dim=0).to(device),
        torch.cat(all_new_rotation, dim=0).to(device),
        torch.zeros(all_new_xyz.shape[0]).to(device))

    record = {key: np.concatenate(chunks, axis=0) for key, chunks in rec.items()}
    return record, selected_indices, float(np.mean(uv_check))


def project(cam, xyz):
    """World points -> (uv in [-1, 1] (x, y), the ref_uv convention; camera depth)."""
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1)
    pc = ph @ cam.world_view_transform.to(xyz)        # 3DGS stores W2C transposed
    z = pc[:, 2]
    uv = torch.stack([pc[:, 0] / math.tan(cam.FoVx / 2), pc[:, 1] / math.tan(cam.FoVy / 2)], 1)
    return uv / z.clamp_min(1e-9)[:, None], z


def fibonacci_sphere(n, centre, radius):
    i = torch.arange(n, dtype=torch.float64, device=centre.device)
    y = 1 - 2 * (i + 0.5) / n
    r = torch.sqrt(1 - y * y)
    t = i * math.pi * (3 - math.sqrt(5))
    return centre + radius * torch.stack([r * torch.cos(t), y, r * torch.sin(t)], 1)


def sky_voxel_size(sky, origin, radius, ppv):
    """g such that occupied voxels hold `ppv` points on average (bisection; the
    mean grows with g). Returns (g, nominal R * sqrt(4 pi ppv / n), measured mean)."""
    n = sky.shape[0]
    mean = lambda g: n / torch.unique(torch.floor((sky - origin) / g).long(), dim=0).shape[0]
    nominal = radius * math.sqrt(4 * math.pi * ppv / n)
    lo, hi = nominal / 4, nominal * 4
    for _ in range(30):
        g = math.sqrt(lo * hi)
        lo, hi = (g, hi) if mean(g) < ppv else (lo, g)
    g = math.sqrt(lo * hi)
    return g, nominal, mean(g)


def add_sky_points(gaussians, record, cameras, extractor, pca, args, device):
    """Third step (--sky): append sky points to `gaussians` and `record` (see the
    module docstring); adds record["is_sky"]. Returns the meta's "sky" entry."""
    valid, n_old = record["valid"], len(record["valid"])

    # The projection below must agree with the points' reference pixels.
    rng = np.random.default_rng(0)
    rows = rng.choice(np.flatnonzero(valid), size=min(20_000, int(valid.sum())), replace=False)
    errs = []
    for cam_idx in np.unique(record["ref_cam"][rows]):
        sel = rows[record["ref_cam"][rows] == cam_idx]
        cam = cameras[cam_idx]
        uv, _ = project(cam, torch.from_numpy(record["xyz"][sel]).double().to(device))
        d = (uv.cpu().numpy() - record["ref_uv"][sel]) * 0.5 * np.array([cam.image_width, cam.image_height])
        errs.append(np.linalg.norm(d, axis=1))
    align_px = float(np.median(np.concatenate(errs)))
    print(f"      projection vs ref_uv of the points: median {align_px:.3f} px")
    if not align_px <= 2.0:
        raise RuntimeError(f"sky: projection disagrees with ref_uv ({align_px:.2f} px)")

    # Sphere + voxel size.
    pts = torch.from_numpy(record["xyz"][valid]).double().to(device)
    centre = pts.mean(0)
    sigma = float(torch.sqrt(((pts - centre) ** 2).sum(1).mean() / 3))
    radius = args.sky_sigma_k * sigma
    outside = float(((pts - centre).norm(dim=1) > radius).double().mean())
    sky = fibonacci_sphere(args.sky_points, centre, radius)
    origin = centre - radius
    g, g_nominal, ppv = sky_voxel_size(sky, origin, radius, args.sky_points_per_voxel)
    print(f"      sigma {sigma:.4f}, R = {args.sky_sigma_k:g} sigma = {radius:.4f} "
          f"({100 * outside:.2f} % of the valid points outside); voxel {g:.5f} "
          f"(nominal {g_nominal:.5f}), {ppv:.2f} points / voxel")

    # Frustum cull, then the minimum set of views (greedy set cover).
    vis = torch.zeros(len(cameras), args.sky_points, dtype=torch.bool, device=device)
    for i, cam in enumerate(cameras):
        uv, z = project(cam, sky)
        vis[i] = (z > 0) & (uv.abs() <= 1).all(1)
    seen = vis.any(0)
    n_seen, left, chosen, curve = int(seen.sum()), seen.clone(), [], []
    while left.any():
        j = int((vis & left).sum(1).argmax())
        chosen.append(j)
        left &= ~vis[j]
        curve.append(round(1 - int(left.sum()) / n_seen, 5))
    sky = sky[seen]
    m = sky.shape[0]
    print(f"      {m} of {args.sky_points} sky points seen ({100 * m / args.sky_points:.1f} %), "
          f"covered by {len(chosen)} of {len(cameras)} views")

    # Reference view = the chosen view where the point is most central.
    best = torch.full((m,), float("inf"), dtype=torch.float64, device=device)
    ref_cam = torch.full((m,), -1, dtype=torch.long, device=device)
    ref_uv = torch.zeros(m, 2, dtype=torch.float64, device=device)
    for j in chosen:
        uv, z = project(cameras[j], sky)
        inside = (z > 0) & (uv.abs() <= 1).all(1)
        c = torch.where(inside, uv.abs().amax(1), torch.full_like(z, float("inf")))
        better = c < best
        best[better], ref_cam[better], ref_uv[better] = c[better], j, uv[better]

    # rgb + features at the reference pixels, as for the points.
    feat = torch.zeros(m, pca.layout()[1], device=device)
    rgb = torch.zeros(m, 3, device=device)
    for j in tqdm(chosen, desc="sky views"):
        sel = torch.nonzero(ref_cam == j).squeeze(1)
        if len(sel) == 0:
            continue
        cam, uv = cameras[j], ref_uv[sel].float()
        with torch.no_grad():
            feat[sel] = pca.project(extractor.sample(j, cam_to_pil(cam), uv))
            img = cam.original_image.to(device).float()[None]
            rgb[sel] = F.grid_sample(img, uv.view(1, -1, 1, 2), align_corners=False)[0, :, :, 0].T

    with torch.no_grad():
        opacity = gaussians._opacity[torch.from_numpy(valid).to(gaussians._opacity.device)].median()
        gaussians.densification_postfix(
            sky.float(),
            RGB2SH(rgb.clamp(0, 1)).unsqueeze(1),
            torch.zeros(m, *gaussians._features_rest.shape[1:], device=device),
            torch.full((m, 1), float(opacity), device=device),
            torch.full((m, 3), math.log(g / 2), device=device),
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(m, 4).contiguous(),
            torch.zeros(m, device=device))
    new = {"feat": feat.cpu().numpy().astype(np.float16), "xyz": sky.float().cpu().numpy(),
           "ref_cam": ref_cam.cpu().numpy().astype(np.int16),
           "ref_uv": ref_uv.cpu().numpy().astype(np.float32), "valid": np.ones(m, dtype=bool)}
    assert sorted(new) == sorted(record), sorted(record)
    for k in record:
        record[k] = np.concatenate([record[k], new[k].astype(record[k].dtype)])
    record["is_sky"] = np.concatenate([np.zeros(n_old, dtype=bool), np.ones(m, dtype=bool)])
    return {
        "rows": [n_old, n_old + m], "num_points": m, "lattice_points": args.sky_points,
        "sigma_k": args.sky_sigma_k, "sigma": sigma, "centre": centre.cpu().numpy().tolist(),
        "radius": radius, "valid_points_outside": outside,
        "points_per_voxel_target": args.sky_points_per_voxel, "points_per_voxel": ppv,
        "voxel_size": g, "voxel_size_nominal": g_nominal, "voxel_origin": origin.cpu().numpy().tolist(),
        "views": [int(j) for j in chosen], "cover_curve": curve,
        "ply_scale": g / 2, "ply_opacity_logit": float(opacity),
        "rgb_mean": rgb.mean(0).cpu().numpy().tolist(), "align_check_px": align_px,
    }


def add_complexity(record, cameras, args):
    """Fourth step (--complexity_dir): record["complexity"] = each row's reference-view
    complexity map sampled (bilinearly) at its ref_uv. Returns the meta's entry."""
    d = args.complexity_dir if os.path.isabs(args.complexity_dir) else os.path.join(args.source_path, args.complexity_dir)
    with open(os.path.join(d, "meta.json")) as f:
        cmeta = json.load(f)
    s, k = cmeta["stride"], cmeta["kernel_size"]
    c0 = 1.0 - (k % 2) / 2.0                     # entry j's window centre: x = s * j + c0
    ref_cam, ref_uv = record["ref_cam"], record["ref_uv"]
    out = np.full(len(ref_cam), np.nan, dtype=np.float32)
    for cam_idx in np.unique(ref_cam):
        rows = np.flatnonzero(ref_cam == cam_idx)
        stem = os.path.splitext(cameras[cam_idx].image_name)[0]
        cmap = tifffile.imread(os.path.join(d, stem + ".tif"))
        h, w = cmeta["sizes"][stem]
        assert cmap.shape == ((h - 1) // s + 1, (w - 1) // s + 1), (stem, cmap.shape, h, w)
        # [-1, 1] (edges) -> continuous pixels -> fractional map indices.
        jx = np.clip(((ref_uv[rows, 0] + 1) * 0.5 * w - c0) / s, 0, cmap.shape[1] - 1)
        jy = np.clip(((ref_uv[rows, 1] + 1) * 0.5 * h - c0) / s, 0, cmap.shape[0] - 1)
        x0, y0 = np.floor(jx).astype(int), np.floor(jy).astype(int)
        x1, y1 = np.minimum(x0 + 1, cmap.shape[1] - 1), np.minimum(y0 + 1, cmap.shape[0] - 1)
        fx, fy = jx - x0, jy - y0
        out[rows] = ((1 - fy) * ((1 - fx) * cmap[y0, x0] + fx * cmap[y0, x1])
                     + fy * ((1 - fx) * cmap[y1, x0] + fx * cmap[y1, x1]))
    assert not np.isnan(out).any()
    record["complexity"] = out
    q = np.percentile(out, [0, 1, 50, 99, 100]).tolist()
    print(f"      complexity ({cmeta['metric']}, k {k}, stride {s}): min {q[0]:.3f}, p1 {q[1]:.3f}, "
          f"median {q[2]:.3f}, p99 {q[3]:.3f}, max {q[4]:.3f}")
    entry = {"dir": os.path.abspath(d), "metric": cmeta["metric"], "kernel_size": k, "stride": s,
             "min": q[0], "p1": q[1], "median": q[2], "p99": q[3], "max": q[4]}
    if "is_sky" in record:
        sky = record["is_sky"]
        entry["median_points"], entry["median_sky"] = float(np.median(out[~sky])), float(np.median(out[sky]))
    return entry


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()

    dataset = build_dataset_args(args)
    out_dir = dataset.model_path
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/5] Loading COLMAP reconstruction from {dataset.source_path}")
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    if args.holdout_test:
        scene_cameras_train_test_split(scene, verbose=True)
    cfg = SimpleNamespace(matches_per_ref=args.matches_per_ref, num_refs=args.num_refs,
                          nns_per_ref=args.nns_per_ref, scaling_factor=args.scaling_factor,
                          proj_err_tolerance=args.proj_err_tolerance)
    cameras = scene.getTrainCameras()
    clamp_selection_params(cfg, len(cameras))
    n_sfm = int(gaussians._xyz.shape[0])
    print(f"      {len(cameras)} cameras used ({len(scene.getTestCameras())} held out)")
    gaussians.tmp_radii = torch.zeros(gaussians._xyz.shape[0], device=device)
    gaussians.training_setup(load_gs_optimization_defaults())

    print(f"[2/5] Loading RoMa ({args.roma_model}) and PCA basis {args.pca_basis}")
    roma_model = (roma_indoor if args.roma_model == "indoors" else roma_outdoor)(device=device)
    roma_model.upsample_preds = False
    roma_model.symmetric = False
    pca = BlockPCA.load(args.pca_basis, device)
    strides = sorted(int(n[len("roma"):]) for n in pca.dims)
    extractor = MultiScaleFeatureExtractor(roma_model, strides, device=device, cache_size=1)
    spans, total_dim = pca.layout()
    print(f"      blocks {spans} -> {total_dim} dims")

    print(f"[3/5] Initializing: {cfg.num_refs} refs x {cfg.nns_per_ref} nns x "
          f"{cfg.matches_per_ref} matches")
    record, selected_indices, uv_err = init_with_ref_features(
        gaussians, scene, cfg, roma_model, device, extractor, pca)
    # Sanity check on the sampling location; a transposed / mis-scaled uv gives ~0.2+.
    print(f"      mean |image(ref_uv) - point colour| = {uv_err:.4f} (expect < 0.05)")

    print("[4/5] Post-processing")
    with torch.no_grad():
        n_after = int(gaussians._xyz.shape[0])
        gaussians.tmp_radii = torch.zeros(n_after).to(device)
        mask = torch.cat([torch.ones(n_sfm, dtype=torch.bool),
                          torch.zeros(n_after - n_sfm, dtype=torch.bool)], dim=0)
        gaussians.prune_points(mask)
        print(f"      removed {n_sfm} SfM points")
        if args.final_scale_modifier != 1.0:
            gaussians._scaling.data = gaussians.scaling_inverse_activation(
                gaussians.scaling_activation(gaussians._scaling.data) * args.final_scale_modifier)

    sky_meta = None
    if args.sky:
        print("[4b/5] Sky points")
        sky_meta = add_sky_points(gaussians, record, cameras, extractor, pca, args, device)

    complexity_meta = None
    if args.complexity_dir:
        print("[4c/5] Complexity")
        complexity_meta = add_complexity(record, cameras, args)

    n_final = int(gaussians._xyz.shape[0])
    assert record["feat"].shape[0] == n_final, (record["feat"].shape, n_final)
    assert np.allclose(record["xyz"], gaussians._xyz.detach().cpu().numpy(), atol=1e-5), \
        "feature rows are not aligned with the PLY rows"

    print("[5/5] Saving")
    ply_path = os.path.join(out_dir, "point_cloud", "iteration_0", "point_cloud.ply")
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    gaussians.save_ply(ply_path)
    write_cfg_args(out_dir, dataset, args.sh_degree)
    record.pop("xyz")   # only kept for the alignment assert; the PLY has it

    feat = record["feat"].astype(np.float32)
    meta = {
        "ply_path": os.path.abspath(ply_path),
        "num_points": n_final,
        "num_valid": int(record["valid"].sum()),
        "ref_uv_colour_error": uv_err,
        "feature_dim": int(total_dim),
        "feature_dtype": "float16",
        "blocks": spans,
        "block_feature_std": {n: float(feat[:, a:b].std()) for n, (a, b) in spans.items()},
        "pca_basis": os.path.abspath(args.pca_basis),
        "pca_basis_sha1": sha1(args.pca_basis),
        "pca_explained_variance": {n: float(v.sum()) for n, v in pca.explained.items()},
        "features_from": "reference view only, RoMa encoder input "
                         f"{roma_model.h_resized}x{roma_model.w_resized}, "
                         "L2-normalized per block before PCA",
        "camera_names": [c.image_name for c in cameras],
        "reference_cameras": [int(i) for i in selected_indices],
        "num_cameras_used": len(cameras),
        "num_sfm_points_removed": n_sfm,
        "seconds": round(time.time() - t0, 1),
        "args": vars(args),
    }
    if sky_meta is not None:
        meta["sky"] = sky_meta
    if complexity_meta is not None:
        meta["complexity"] = complexity_meta
    feat_path = os.path.join(out_dir, "features.npz")
    np.savez(feat_path, **record)
    with open(os.path.join(out_dir, "features_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"      {n_final} points ({meta['num_valid']} valid) x {total_dim} dims "
          f"in {meta['seconds']:.0f}s\n      {ply_path}\n      {feat_path} (+ features_meta.json)")
    return meta


if __name__ == "__main__":
    main()
