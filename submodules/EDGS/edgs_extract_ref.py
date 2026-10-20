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
    features_meta.json                        block spans, basis path + hash, camera
                                              names, settings, sanity checks
    cfg_args, cameras.json, input.ply          written by / for 3DGS tooling

Row i of every array is row i of the PLY (checked against the PLY xyz before saving):
COLMAP SfM points are always removed and invalid points are kept (flagged in `valid`),
so nothing shifts rows. Uncompressed: np.load reads each array lazily, on access.
"""

import argparse
import hashlib
import json
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
import torch  # noqa: E402
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
    feat_path = os.path.join(out_dir, "features.npz")
    np.savez(feat_path, **record)
    with open(os.path.join(out_dir, "features_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"      {n_final} points ({meta['num_valid']} valid) x {total_dim} dims "
          f"in {meta['seconds']:.0f}s\n      {ply_path}\n      {feat_path} (+ features_meta.json)")
    return meta


if __name__ == "__main__":
    main()
