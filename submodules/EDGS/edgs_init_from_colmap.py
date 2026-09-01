#!/usr/bin/env python3
"""
Create an EDGS initialization from a COLMAP reconstruction. Nothing else.

This script is the initialization stage of EDGS, stripped of everything else: no
training loop, no wandb, no hydra, no renderer, no evaluation. It takes a COLMAP /
3DGS style scene folder, runs the RoMa dense-correspondence + triangulation
initialization (`source/corr_init.py`) and writes the resulting Gaussians to a
standard 3DGS `point_cloud.ply`.

Expected input layout (same as 3DGS):

    scene_folder
    |---images
    |   |---<image 0>
    |   |---...
    |---sparse
        |---0
            |---cameras.bin
            |---images.bin
            |---points3D.bin

Output layout:

    output_folder
    |---point_cloud/iteration_0/point_cloud.ply   <- the EDGS initialization
    |---cfg_args                                  <- so 3DGS tooling/viewers can read it
    |---cameras.json, input.ply                   <- written by the 3DGS Scene loader
    |---edgs_init.json                            <- exact settings used + statistics
    |---chkpnt0.pth                               <- optional (--save_checkpoint)

Examples:

    # defaults (paper settings)
    python edgs_init_from_colmap.py -s data/garden -o outputs/garden_init

    # fast variant: 1 neighbour per reference, few references, many matches
    python edgs_init_from_colmap.py -s data/garden -o outputs/garden_init \
        --num_refs 16 --nns_per_ref 1 --matches_per_ref 20000 --scaling_factor 0.00154

    # only use 100 of the input images, hold out every 8th for eval as in the paper
    python edgs_init_from_colmap.py -s data/garden -o outputs/garden_init \
        --max_images 100 --holdout_test

The resulting checkpoint can be handed straight to the normal training entry point:

    python train.py train.gs_epochs=3000 init_wC.use=False \
        load.gs=outputs/garden_init load.gs_step=0 \
        gs.dataset.source_path=data/garden gs.dataset.model_path=outputs/garden_trained
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

# ----------------------------------------------------------------------------------
# Path setup. Has to happen before importing `source.*` or the 3DGS submodule.
# ----------------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO_ROOT,
           os.path.join(REPO_ROOT, "submodules", "gaussian-splatting"),
           os.path.join(REPO_ROOT, "submodules", "RoMa")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from scene import Scene, GaussianModel  # noqa: E402  (submodules/gaussian-splatting)
from source.corr_init import (  # noqa: E402
    init_gaussians_with_corr,
    init_gaussians_with_corr_fast,
)
from source.data_utils import scene_cameras_train_test_split  # noqa: E402
from source.utils_aux import set_seed  # noqa: E402


# ----------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="COLMAP reconstruction in, EDGS initialization out. No training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("input / output")
    io.add_argument("-s", "--source_path", required=True,
                    help="COLMAP scene folder (contains images/ and sparse/0/).")
    io.add_argument("-o", "--output", required=True,
                    help="Output folder for the initialization.")
    io.add_argument("--images", default="images",
                    help="Name of the image subfolder inside the scene folder.")
    io.add_argument("--resolution", type=int, default=-1,
                    help="3DGS resolution flag. -1 keeps the original size but "
                         "downscales images wider than 1600px; 1/2/4/8 divide by that factor.")
    io.add_argument("--white_background", action="store_true",
                    help="Treat the scene background as white.")
    io.add_argument("--data_device", default="cuda",
                    help="Where the loaded images are kept ('cuda' or 'cpu'; use 'cpu' "
                         "for many/large images).")
    io.add_argument("--ply_path", default=None,
                    help="Write the PLY here instead of <output>/point_cloud/iteration_0/point_cloud.ply.")
    io.add_argument("--save_checkpoint", action="store_true",
                    help="Also write <output>/chkpnt0.pth, loadable by train.py via load.gs / load.gs_step=0.")

    sel = p.add_argument_group("how many images / correspondences are used")
    sel.add_argument("--max_images", type=int, default=None,
                     help="Use only this many input images (evenly spaced over the "
                          "camera list). Default: use all of them.")
    sel.add_argument("--num_refs", type=int, default=180,
                     help="Number of reference frames, picked by K-means over the "
                          "camera poses. Correspondences are computed per reference.")
    sel.add_argument("--nns_per_ref", type=int, default=3,
                     help="Number of nearest-neighbour views matched against each "
                          "reference. 1 selects the fast single-pair code path.")
    sel.add_argument("--matches_per_ref", type=int, default=15_000,
                     help="Number of correspondences sampled per reference frame. "
                          "Roughly num_refs * matches_per_ref splats are produced.")
    sel.add_argument("--holdout_test", action="store_true",
                     help="Hold out every 8th image as a test set (3DGS eval protocol) "
                          "so the initialization only sees the training views.")

    ini = p.add_argument_group("initialization parameters")
    ini.add_argument("--scaling_factor", type=float, default=0.001,
                     help="Splat size relative to its distance from the reference camera.")
    ini.add_argument("--proj_err_tolerance", type=float, default=0.01,
                     help="Triangulated points with a larger reprojection error are "
                          "pushed to zero opacity (or dropped, see --drop_invalid_points).")
    ini.add_argument("--roma_model", choices=["outdoors", "indoors"], default="outdoors",
                     help="Which RoMa matcher to use.")
    ini.add_argument("--sh_degree", type=int, default=3,
                     help="Spherical harmonics degree of the produced Gaussians.")
    ini.add_argument("--add_SfM_init", action="store_true",
                     help="Keep the COLMAP SfM points in addition to the triangulated "
                          "ones. Default is to keep only the EDGS points.")
    ini.add_argument("--final_scale_modifier", type=float, default=0.5,
                     help="Multiplier applied to all splat scales at the very end "
                          "(0.5 is what the EDGS trainer does).")
    ini.add_argument("--init_opacity", type=float, default=None,
                     help="Override the opacity of the valid splats (default 0.5, as "
                          "produced by the initialization).")
    ini.add_argument("--drop_invalid_points", action="store_true",
                     help="Remove the points that failed the reprojection test instead "
                          "of leaving them at zero opacity.")

    misc = p.add_argument_group("misc")
    misc.add_argument("--seed", type=int, default=228)
    misc.add_argument("--device", default="cuda:0")
    misc.add_argument("--verbose", action="store_true",
                      help="Print intermediate information from the correspondence stage.")

    return p.parse_args(argv)


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def build_dataset_args(args):
    """The subset of 3DGS ModelParams that `Scene` actually reads."""
    return SimpleNamespace(
        source_path=os.path.abspath(args.source_path),
        model_path=os.path.abspath(args.output),
        images=args.images,
        depths="",
        resolution=args.resolution,
        white_background=args.white_background,
        data_device=args.data_device,
        eval=args.holdout_test,
        train_test_exp=False,
    )


def build_init_cfg(args):
    """The `init_wC` config block consumed by source/corr_init.py."""
    return SimpleNamespace(
        use=True,
        matches_per_ref=args.matches_per_ref,
        num_refs=args.num_refs,
        nns_per_ref=args.nns_per_ref,
        scaling_factor=args.scaling_factor,
        proj_err_tolerance=args.proj_err_tolerance,
        roma_model=args.roma_model,
        add_SfM_init=args.add_SfM_init,
    )


def load_gs_optimization_defaults():
    """
    Reuse the learning rates from configs/gs/base.yaml.

    We never take an optimizer step, but GaussianModel.training_setup() has to run:
    densification_postfix() and prune_points() both go through the optimizer state.
    """
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", "gs", "base.yaml"))
    return cfg.opt


def subsample_cameras(scene, max_images, verbose=False):
    """Keep only `max_images` training cameras, evenly spaced over the sequence."""
    if max_images is None:
        return
    for resolution in scene.train_cameras.keys():
        cams = scene.train_cameras[resolution]
        if max_images >= len(cams):
            continue
        idcs = np.unique(np.linspace(0, len(cams) - 1, max_images).round().astype(int))
        scene.train_cameras[resolution] = [cams[i] for i in idcs]
        if verbose:
            print(f"[cameras] resolution {resolution}: {len(cams)} -> "
                  f"{len(scene.train_cameras[resolution])} images")


def clamp_selection_params(cfg, n_cameras):
    """Keep num_refs / nns_per_ref within what the camera set can support."""
    if n_cameras < 2:
        raise RuntimeError(f"Need at least 2 cameras to triangulate, got {n_cameras}.")
    if cfg.num_refs > n_cameras:
        print(f"[warn] num_refs={cfg.num_refs} > {n_cameras} cameras. Clamped to {n_cameras}.")
        cfg.num_refs = n_cameras
    if cfg.nns_per_ref > n_cameras - 1:
        print(f"[warn] nns_per_ref={cfg.nns_per_ref} > {n_cameras - 1} available "
              f"neighbours. Clamped to {n_cameras - 1}.")
        cfg.nns_per_ref = n_cameras - 1


def write_cfg_args(model_path, dataset, sh_degree):
    """3DGS viewers and render.py read this file. Same format as train.py writes."""
    from argparse import Namespace
    params = {
        "sh_degree": sh_degree,
        "source_path": dataset.source_path,
        "model_path": dataset.model_path,
        "images": dataset.images,
        "depths": "",
        "resolution": dataset.resolution,
        "_white_background": dataset.white_background,
        "train_test_exp": False,
        "data_device": dataset.data_device,
        "eval": dataset.eval,
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "debug": False,
        "antialiasing": False,
    }
    with open(os.path.join(model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**params)))


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device)
    t_start = time.time()

    dataset = build_dataset_args(args)
    cfg_init = build_init_cfg(args)
    out_dir = dataset.model_path
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(dataset.source_path):
        raise FileNotFoundError(f"Scene folder not found: {dataset.source_path}")

    # --- 1. Load the COLMAP reconstruction: cameras, images, SfM points ------------
    print(f"[1/5] Loading COLMAP reconstruction from {dataset.source_path}")
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    if args.holdout_test:
        # No-op when the loader already produced a test split.
        scene_cameras_train_test_split(scene, verbose=True)
    subsample_cameras(scene, args.max_images, verbose=True)

    train_cameras = scene.getTrainCameras()
    n_cameras = len(train_cameras)
    clamp_selection_params(cfg_init, n_cameras)
    n_sfm_points = int(gaussians._xyz.shape[0])
    print(f"      {n_cameras} cameras used for initialization "
          f"({len(scene.getTestCameras())} held out), {n_sfm_points} SfM points")

    # densification_postfix()/prune_points() also index tmp_radii, and Scene populated
    # _xyz after the model was constructed, so resize it to match.
    gaussians.tmp_radii = torch.zeros(gaussians._xyz.shape[0], device=device)

    # --- 2. Optimizer state (never stepped, but required by the (de)densification API)
    gaussians.training_setup(load_gs_optimization_defaults())

    # --- 3. Correspondences -> triangulation -> Gaussians -------------------------
    init_fn = init_gaussians_with_corr_fast if cfg_init.nns_per_ref == 1 else init_gaussians_with_corr
    print(f"[2/5] Running {init_fn.__name__}: {cfg_init.num_refs} references x "
          f"{cfg_init.nns_per_ref} neighbours x {cfg_init.matches_per_ref} matches "
          f"(RoMa '{cfg_init.roma_model}')")
    t_init = time.time()
    try:
        init_fn(gaussians, scene, cfg_init, device, verbose=args.verbose, roma_model=None)
    except (IndexError, ValueError) as e:
        raise RuntimeError(
            f"Camera selection failed with num_refs={cfg_init.num_refs} and "
            f"{n_cameras} cameras. K-means could not produce that many distinct "
            f"pose clusters -- lower --num_refs (original error: {e})") from e
    t_init = time.time() - t_init
    n_after_init = int(gaussians._xyz.shape[0])
    print(f"      {n_after_init - n_sfm_points} splats created in {t_init:.1f}s")

    # --- 4. Post-processing, same as EDGSTrainer.init_with_corr does --------------
    # Order matters: every prune_points() call rebuilds the tensors from the optimizer
    # groups, so all pruning has to happen before the tensors are edited.
    print("[3/5] Post-processing")
    with torch.no_grad():
        if not args.add_SfM_init:
            # Drop the SfM points, keep only the correspondence-based ones.
            gaussians.tmp_radii = torch.zeros(gaussians._xyz.shape[0]).to(device)
            mask = torch.cat([torch.ones(n_sfm_points, dtype=torch.bool),
                              torch.zeros(n_after_init - n_sfm_points, dtype=torch.bool)], dim=0)
            gaussians.prune_points(mask)
            print(f"      removed {n_sfm_points} SfM points")

        if args.drop_invalid_points:
            # The initialization parks points that failed the reprojection test at an
            # opacity logit of -10 instead of removing them.
            gaussians.tmp_radii = torch.zeros(gaussians._xyz.shape[0]).to(device)
            prune_mask = (gaussians.get_opacity < 1e-3).squeeze()
            n_bad = int(prune_mask.sum())
            if n_bad:
                gaussians.prune_points(prune_mask)
            print(f"      dropped {n_bad} points that failed the reprojection test")

        # In-place edits from here on, so the parameters stay the ones the optimizer
        # holds and the optional checkpoint remains trainable.
        if args.init_opacity is not None:
            # Valid points come out of the initialization with a logit of exactly 0.
            p = float(np.clip(args.init_opacity, 1e-4, 1 - 1e-4))
            valid = (gaussians._opacity.data == 0.)
            gaussians._opacity.data[valid] = float(np.log(p / (1. - p)))
            print(f"      opacity of {int(valid.sum())} valid splats set to {p}")

        if args.final_scale_modifier != 1.0:
            gaussians._scaling.data = gaussians.scaling_inverse_activation(
                gaussians.scaling_activation(gaussians._scaling.data) * args.final_scale_modifier)
            print(f"      splat scales multiplied by {args.final_scale_modifier}")

    n_final = int(gaussians._xyz.shape[0])

    # --- 5. Save ------------------------------------------------------------------
    print("[4/5] Saving")
    ply_path = args.ply_path or os.path.join(out_dir, "point_cloud", "iteration_0", "point_cloud.ply")
    os.makedirs(os.path.dirname(os.path.abspath(ply_path)), exist_ok=True)
    gaussians.save_ply(ply_path)
    write_cfg_args(out_dir, dataset, args.sh_degree)

    ckpt_path = None
    if args.save_checkpoint:
        ckpt_path = os.path.join(out_dir, "chkpnt0.pth")
        torch.save((gaussians.capture(), 0), ckpt_path)

    summary = {
        "source_path": dataset.source_path,
        "ply_path": os.path.abspath(ply_path),
        "checkpoint_path": ckpt_path,
        "num_cameras_total": n_cameras + len(scene.getTestCameras()),
        "num_cameras_used": n_cameras,
        "num_reference_frames": cfg_init.num_refs,
        "num_neighbours_per_reference": cfg_init.nns_per_ref,
        "matches_per_reference": cfg_init.matches_per_ref,
        "num_sfm_points": n_sfm_points,
        "num_splats_created": n_after_init - n_sfm_points,
        "num_splats_final": n_final,
        "init_seconds": round(t_init, 2),
        "total_seconds": round(time.time() - t_start, 2),
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "edgs_init.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[5/5] Done in {summary['total_seconds']:.1f}s — {n_final} splats")
    print(f"      {ply_path}")
    if ckpt_path:
        print(f"      {ckpt_path}")
    return summary


if __name__ == "__main__":
    main()
