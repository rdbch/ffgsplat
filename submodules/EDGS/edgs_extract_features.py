#!/usr/bin/env python3
"""
EDGS initialization + multi-scale visual features per splat.

Runs the same correspondence initialization as edgs_init_from_colmap.py, but also
samples multi-scale features at every keypoint and writes a per-splat, per-view
record aligned row-for-row with the point cloud it saves.

Why one script and not two: the splats are drawn with torch.multinomial over the
certainty map, so re-running the matching in a second process would sample different
pixels. Producing the cloud and the features in the same pass makes them aligned by
construction.

Feature sources, both supplied by you (nothing is downloaded here):
  * RoMa encoder pyramid  - the fine CNN levels (stride 1/2/4/8) carry local texture
                            and shading: high resolution, low invariance.
  * DINOv2 patch tokens   - stride 14, semantic / material context.

Per splat the record holds k+1 view slots: slot 0 is the reference frame, slots
1..k are its matched neighbours.

    xyz          (N, 3)        float32   world position (same rows as the PLY)
    ref_cam      (N,)          int32     index of the reference camera
    nn_cam       (N, k)        int32     indices of the neighbour cameras
    view_dirs    (N, k+1, 3)   float16   unit vector camera -> point, world frame
    rgb          (N, k+1, 3)   uint8     observed colour in each view
    feat         (N, k+1, D)   float16   PCA-projected multi-scale features
    certainty    (N, k+1)      float16   RoMa certainty (slot 0 = 1 by definition)
    reproj_err   (N, k, 2)     float16   (error in ref frame, error in neighbour frame)
    best_nn      (N,)          int16     neighbour that won the triangulation
    valid        (N, k+1)      bool      reprojection + occlusion gates, NOT applied

The gates are stored as a mask rather than applied, so a downstream model can learn
how much to trust a marginal observation instead of inheriting a threshold.

Example:

    python edgs_extract_features.py -s data/garden -o outputs/garden_feat \
        --num_refs 30 --nns_per_ref 3 --matches_per_ref 5000 \
        --roma_scales 4,8 --use_dino --feature_dim 64
"""

import argparse
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
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scene import Scene, GaussianModel  # noqa: E402
from utils.sh_utils import RGB2SH  # noqa: E402
from romatch import roma_indoor, roma_outdoor  # noqa: E402
from romatch.utils import get_tuple_transform_ops  # noqa: E402

from source.corr_init import (  # noqa: E402
    aggregate_confidences_and_warps,
    extract_keypoints_and_colors,
    k_closest_vectors,
    select_cameras_kmeans,
    triangulate_points,
)
from source.data_utils import scene_cameras_train_test_split  # noqa: E402
from source.utils_aux import set_seed  # noqa: E402

# edgs_init_from_colmap.py is imported, never modified.
from edgs_init_from_colmap import (  # noqa: E402
    build_dataset_args,
    clamp_selection_params,
    load_gs_optimization_defaults,
    subsample_cameras,
    write_cfg_args,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="EDGS initialization with multi-scale RoMa + DINOv2 features per splat.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("input / output")
    io.add_argument("-s", "--source_path", required=True)
    io.add_argument("-o", "--output", required=True)
    io.add_argument("--images", default="images")
    io.add_argument("--resolution", type=int, default=-1)
    io.add_argument("--white_background", action="store_true")
    io.add_argument("--data_device", default="cuda")

    sel = p.add_argument_group("scene / correspondences")
    sel.add_argument("--max_images", type=int, default=None)
    sel.add_argument("--num_refs", type=int, default=30)
    sel.add_argument("--nns_per_ref", type=int, default=3,
                     help="Neighbour views per reference. Must be >= 1; this script has "
                          "no single-pair fast path because the record needs the "
                          "per-neighbour observations.")
    sel.add_argument("--matches_per_ref", type=int, default=5_000)
    sel.add_argument("--holdout_test", action="store_true")
    sel.add_argument("--roma_model", choices=["outdoors", "indoors"], default="outdoors")

    ini = p.add_argument_group("initialization")
    ini.add_argument("--scaling_factor", type=float, default=0.001)
    ini.add_argument("--proj_err_tolerance", type=float, default=0.01)
    ini.add_argument("--sh_degree", type=int, default=3)
    ini.add_argument("--final_scale_modifier", type=float, default=0.5)

    ft = p.add_argument_group("features")
    ft.add_argument("--roma_scales", default="4,8",
                    help="Comma-separated RoMa encoder pyramid strides to sample. The "
                         "coarse DINOv2 level inside RoMa is usually key 16; the fine "
                         "CNN levels are 1/2/4/8. Empty string disables RoMa features.")
    ft.add_argument("--use_dino", action="store_true",
                    help="Also sample a standalone DINOv2 model (see --dino_name).")
    ft.add_argument("--dino_name", default="dinov2_vitl14",
                    help="torch.hub entry point from facebookresearch/dinov2.")
    ft.add_argument("--dino_res", type=int, default=560,
                    help="Square resolution DINOv2 runs at. Must be a multiple of 14.")
    ft.add_argument("--feature_dim", type=int, default=64,
                    help="Total PCA output width, split evenly across the feature "
                         "blocks (coarsest blocks get the remainder).")
    ft.add_argument("--pca_images", type=int, default=20,
                    help="Random images used to fit the PCA basis.")
    ft.add_argument("--pca_pixels", type=int, default=10_000,
                    help="Random pixels sampled per image when fitting the basis.")
    ft.add_argument("--cos_gate", type=float, default=0.5,
                    help="A neighbour observation is marked invalid when its feature "
                         "cosine similarity to the reference observation falls below "
                         "this. Recorded in `valid`, never applied.")
    ft.add_argument("--feature_cache", type=int, default=8,
                    help="How many images' feature maps to keep on the GPU.")

    misc = p.add_argument_group("misc")
    misc.add_argument("--seed", type=int, default=228)
    misc.add_argument("--device", default="cuda:0")
    misc.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)
    if args.nns_per_ref < 1:
        p.error("--nns_per_ref must be at least 1")
    if args.use_dino and args.dino_res % 14 != 0:
        p.error("--dino_res must be a multiple of 14")
    return args


# ----------------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------------
class MultiScaleFeatureExtractor:
    """
    Produces a dict of feature maps {block_name: (C, h, w)} for an image, and samples
    them at normalized keypoints.

    Feature maps are cached per camera index: neighbours recur across references, so
    without a cache each image is encoded several times.
    """

    def __init__(self, roma_model, roma_scales, dino_model=None, dino_res=560,
                 device="cuda", cache_size=8):
        self.roma = roma_model
        self.roma_scales = list(roma_scales)
        self.dino = dino_model
        self.dino_res = dino_res
        self.device = device
        self.cache_size = max(1, cache_size)
        self._cache = {}
        self._order = []

        hs, ws = roma_model.h_resized, roma_model.w_resized
        # Same preprocessing RoMa itself uses for matching, so the features we sample
        # correspond to what produced the warps.
        self._roma_tf = get_tuple_transform_ops(resize=(hs, ws), normalize=True)

        self.block_names = [f"roma{s}" for s in self.roma_scales]
        if self.dino is not None:
            self.block_names.append("dino")

    # -- backbone adapters ---------------------------------------------------------
    def _roma_pyramid(self, pil_image):
        """
        Return {stride: (C, h, w)} from RoMa's encoder.

        API assumption, deliberately isolated in this one method: romatch's
        RegressionMatcher exposes its backbone through `extract_backbone_features`,
        with the encoder reachable as `.encoder`. Revisions differ in signature and in
        whether the coarse DINOv2 level is keyed 14 or 16, so we try the known entry
        points in order and surface whatever keys we actually got.
        """
        im, _ = self._roma_tf((pil_image, pil_image))
        x = im[None].to(self.device)
        batch = {"im_A": x, "im_B": x}

        pyramid = None
        errors = []
        with torch.no_grad():
            for attempt in (
                lambda: self.roma.extract_backbone_features(batch, batched=True, upsample=False)[0],
                lambda: self.roma.extract_backbone_features(batch, batched=True)[0],
                lambda: self.roma.encoder(x, upsample=False),
                lambda: self.roma.encoder(x),
            ):
                try:
                    pyramid = attempt()
                    break
                except (AttributeError, TypeError, ValueError) as e:
                    errors.append(repr(e))

        if not isinstance(pyramid, dict):
            raise RuntimeError(
                "Could not get a feature pyramid out of the RoMa model. Tried "
                f"extract_backbone_features and .encoder; errors: {errors}. Adapt "
                "MultiScaleFeatureExtractor._roma_pyramid to your romatch revision.")

        out = {}
        for key, value in pyramid.items():
            if not torch.is_tensor(value) or value.dim() != 4:
                continue
            out[int(key)] = value[0]  # (C, h, w), batch of 1
        return out

    def _dino_map(self, pil_image):
        """Return (C, h, w) DINOv2 patch tokens at stride 14."""
        x = torch.from_numpy(np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.)
        x = x.permute(2, 0, 1)[None].to(self.device)
        x = F.interpolate(x, size=(self.dino_res, self.dino_res),
                          mode="bilinear", align_corners=False)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            try:
                feat = self.dino.get_intermediate_layers(x, n=1, reshape=True)[0]
            except (AttributeError, TypeError):
                tokens = self.dino.forward_features(x)["x_norm_patchtokens"]
                g = self.dino_res // 14
                feat = tokens.reshape(1, g, g, -1).permute(0, 3, 1, 2)
        return feat[0]

    # -- public API ----------------------------------------------------------------
    def maps_for(self, cam_idx, pil_image):
        if cam_idx in self._cache:
            return self._cache[cam_idx]

        blocks = {}
        if self.roma_scales:
            pyramid = self._roma_pyramid(pil_image)
            for stride in self.roma_scales:
                if stride not in pyramid:
                    raise RuntimeError(
                        f"RoMa pyramid has no stride {stride}. Available: "
                        f"{sorted(pyramid.keys())}. Fix --roma_scales.")
                blocks[f"roma{stride}"] = pyramid[stride].float()
        if self.dino is not None:
            blocks["dino"] = self._dino_map(pil_image).float()

        self._cache[cam_idx] = blocks
        self._order.append(cam_idx)
        while len(self._order) > self.cache_size:
            del self._cache[self._order.pop(0)]
        return blocks

    def sample(self, cam_idx, pil_image, kpts_xy):
        """
        kpts_xy: (N, 2) normalized to [-1, 1] as (x, y) -- the convention
        extract_keypoints_and_colors already returns.

        Returns {block_name: (N, C)}, each block L2-normalized. Normalized coordinates
        make the sampling independent of the resolution each backbone runs at.
        """
        blocks = self.maps_for(cam_idx, pil_image)
        grid = torch.as_tensor(kpts_xy, dtype=torch.float32, device=self.device)
        grid = grid.view(1, -1, 1, 2)

        out = {}
        for name, fmap in blocks.items():
            sampled = F.grid_sample(fmap[None], grid, mode="bilinear",
                                    padding_mode="border", align_corners=False)
            sampled = sampled[0, :, :, 0].T.contiguous()  # (N, C)
            out[name] = F.normalize(sampled, dim=1)
        return out

    def clear_cache(self):
        self._cache.clear()
        self._order.clear()


class BlockPCA:
    """One PCA basis per feature block, fitted on random pixels from random images."""

    def __init__(self, dims):
        self.dims = dims                # {block: out_dim}
        self.mean = {}                  # {block: (C,)}
        self.components = {}            # {block: (out_dim, C)}
        self.explained = {}             # {block: (out_dim,)}

    def fit(self, samples):
        for name, x in samples.items():
            d = min(self.dims[name], x.shape[1], x.shape[0])
            mean = x.mean(dim=0)
            xc = x - mean
            # torch.pca_lowrank is enough here and keeps this dependency-free.
            _, s, v = torch.pca_lowrank(xc, q=min(d + 8, min(xc.shape)), center=False)
            self.mean[name] = mean
            self.components[name] = v[:, :d].T.contiguous()
            var = (s ** 2) / max(1, xc.shape[0] - 1)
            total_var = xc.pow(2).sum() / max(1, xc.shape[0] - 1)
            self.explained[name] = (var[:d] / total_var).cpu().numpy()
            self.dims[name] = d

    def project(self, blocks):
        """{block: (N, C)} -> (N, sum(dims)), blocks concatenated in a fixed order."""
        parts = []
        for name in sorted(blocks.keys()):
            x = blocks[name] - self.mean[name]
            parts.append(x @ self.components[name].T)
        return torch.cat(parts, dim=1)

    def layout(self):
        """Slice boundaries so a consumer can split the concatenated vector back up."""
        offset, spans = 0, {}
        for name in sorted(self.dims.keys()):
            spans[name] = [offset, offset + self.dims[name]]
            offset += self.dims[name]
        return spans, offset

    def save(self, path):
        payload = {}
        for name in self.mean:
            payload[f"mean_{name}"] = self.mean[name].cpu().numpy()
            payload[f"components_{name}"] = self.components[name].cpu().numpy()
            payload[f"explained_{name}"] = self.explained[name]
        np.savez_compressed(path, **payload)


def split_dims(total, block_names):
    """Split the PCA budget evenly, remainder to the coarsest (last) blocks."""
    n = len(block_names)
    base, extra = total // n, total % n
    dims = {}
    for i, name in enumerate(sorted(block_names)):
        dims[name] = base + (1 if i >= n - extra else 0)
    return dims


def cam_to_pil(cam):
    im = cam.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np.clip(im * 255, 0, 255).astype(np.uint8))


def fit_pca(extractor, cameras, n_images, n_pixels, dims, device, seed=0):
    """
    Fit the basis on random pixels of random images. Feature statistics do not depend
    on which pixels end up being keypoints, so this needs no matching -- one cheap
    pre-pass instead of holding every full-width feature in memory.
    """
    rng = np.random.default_rng(seed)
    idcs = rng.choice(len(cameras), size=min(n_images, len(cameras)), replace=False)

    pools = {name: [] for name in extractor.block_names}
    for cam_idx in idcs:
        blocks = extractor.maps_for(int(cam_idx), cam_to_pil(cameras[int(cam_idx)]))
        for name, fmap in blocks.items():
            c, h, w = fmap.shape
            flat = fmap.reshape(c, h * w).T
            take = min(n_pixels, flat.shape[0])
            sel = torch.from_numpy(rng.choice(flat.shape[0], size=take, replace=False)).to(device)
            pools[name].append(F.normalize(flat[sel], dim=1).cpu())
    extractor.clear_cache()

    samples = {name: torch.cat(chunks, dim=0).to(device) for name, chunks in pools.items()}
    pca = BlockPCA(dict(dims))
    pca.fit(samples)
    for name in sorted(pca.dims):
        print(f"      {name:>8}: {samples[name].shape[1]} -> {pca.dims[name]} dims, "
              f"{pca.explained[name].sum():.1%} variance kept")
    del samples
    torch.cuda.empty_cache()
    return pca


# ----------------------------------------------------------------------------------
# Initialization with feature recording
# ----------------------------------------------------------------------------------
def init_with_features(gaussians, scene, cfg, extractor, pca, device, verbose=False,
                       roma_model=None, cos_gate=0.5):
    """
    Mirrors source.corr_init.init_gaussians_with_corr and reuses all of its helpers,
    adding the per-view feature record. Kept here rather than patched into corr_init
    because the sampling has to happen inside the per-neighbour loop.
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
    rec = {key: [] for key in ("xyz", "ref_cam", "nn_cam", "view_dirs", "rgb",
                               "feat", "certainty", "reproj_err", "best_nn", "valid")}

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
            nn_feats, nn_rgb, nn_cert, nn_blocks = [], [], [], []
            kptsA_np = kptsA_color = None

            for nn_slot in range(len(warps_all)):
                matches_nn = warps_all[nn_slot].reshape(-1, 4)[good_samples]
                kptsA_np, kptsB_np, _, kptsA_color, _ = extract_keypoints_and_colors(
                    imA, imB_compound, certainties_max, certainties_max_idcs,
                    matches_nn, roma_model)

                nn_cam_idx = int(closest[source_idx, nn_slot])
                P1 = viewpoint_stack[source_idx].full_proj_transform
                P2 = viewpoint_stack[nn_cam_idx].full_proj_transform
                pts, e1, e2 = triangulate_points(
                    P1=torch.stack([P1] * M_eff, dim=0), P2=torch.stack([P2] * M_eff, dim=0),
                    k1_x=kptsA_np[:M_eff, 0], k1_y=kptsA_np[:M_eff, 1],
                    k2_x=kptsB_np[:M_eff, 0], k2_y=kptsB_np[:M_eff, 1])
                tri_points.append(pts)
                errs1.append(e1)
                errs2.append(e2)

                # Features observed in THIS neighbour.
                blocks = extractor.sample(nn_cam_idx, cam_to_pil(viewpoint_stack[nn_cam_idx]),
                                          kptsB_np[:M_eff])
                nn_blocks.append(blocks)
                nn_feats.append(pca.project(blocks))
                nn_cert.append(certainties_all[nn_slot].reshape(-1)[good_samples][:M_eff])

                # Colour observed in THIS neighbour. extract_keypoints_and_colors
                # returns kptsB_color from whichever neighbour won on certainty, not
                # from nn_slot, so it would silently mix views here. Sample directly.
                imB = imB_compound[nn_slot]
                H_B, W_B = imB.shape[:2]
                bx = np.clip((((kptsB_np[:M_eff, 0] + 1.) / 2.) * W_B).round().astype(int), 0, W_B - 1)
                by = np.clip((((kptsB_np[:M_eff, 1] + 1.) / 2.) * H_B).round().astype(int), 0, H_B - 1)
                nn_rgb.append(torch.as_tensor(imB[by, bx].astype(np.uint8)))

            # Winner per point: the neighbour with the lowest worst-case reprojection
            # error. select_best_keypoints computes this internally and drops the
            # argmin, so recompute it here.
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

            # --- record ---
            ref_blocks = extractor.sample(source_idx, cam_to_pil(cam), kptsA_np[:N])
            ref_feat = pca.project(ref_blocks)                       # (N, D)
            feats = torch.stack([ref_feat] + [f[:N] for f in nn_feats], dim=1)  # (N, k+1, D)

            block_order = sorted(ref_blocks.keys())
            ref_cat = torch.cat([ref_blocks[b][:N] for b in block_order], dim=1)
            nn_cat = torch.stack([torch.cat([nb[b][:N] for b in block_order], dim=1)
                                  for nb in nn_blocks], dim=1)       # (N, k, C_raw)
            cos = (nn_cat * ref_cat[:, None, :]).sum(-1) / len(block_order)

            cam_centers = torch.stack(
                [cam.camera_center.clone().detach()] +
                [viewpoint_stack[int(closest[source_idx, j])].camera_center.clone().detach()
                 for j in range(len(warps_all))], dim=0)                        # (k+1, 3)
            dirs = F.normalize(new_xyz[:, None, :] - cam_centers[None, :, :], dim=2)

            cert = torch.stack([torch.ones(N, device=device)] +
                               [c[:N].to(device) for c in nn_cert], dim=1)      # (N, k+1)
            rgb = torch.cat([torch.as_tensor(kptsA_color[:N].astype(np.uint8))[:, None, :],
                             torch.stack([c[:N] for c in nn_rgb], dim=1)], dim=1)

            err_pairs = np.stack([np.stack(errs1, axis=0), np.stack(errs2, axis=0)], axis=-1)
            err_pairs = np.transpose(err_pairs, (1, 0, 2))[:N]                  # (N, k, 2)

            # Gates: recorded, not applied. Slot 0 (the reference) is valid by
            # construction; a neighbour fails on reprojection error or on looking
            # nothing like the reference (occlusion / bad match).
            gate_err = torch.from_numpy(err_stack.T[:N] <= cfg.proj_err_tolerance).to(device)
            valid = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=device),
                               gate_err & (cos > cos_gate)], dim=1)

            rec["xyz"].append(new_xyz.cpu().numpy().astype(np.float32))
            rec["ref_cam"].append(np.full(N, source_idx, dtype=np.int32))
            rec["nn_cam"].append(np.tile(closest[source_idx][None, :], (N, 1)).astype(np.int32))
            rec["view_dirs"].append(dirs.cpu().numpy().astype(np.float16))
            rec["rgb"].append(rgb.cpu().numpy().astype(np.uint8))
            rec["feat"].append(feats.cpu().numpy().astype(np.float16))
            rec["certainty"].append(cert.cpu().numpy().astype(np.float16))
            rec["reproj_err"].append(err_pairs.astype(np.float16))
            rec["best_nn"].append(best_nn.astype(np.int16))
            rec["valid"].append(valid.cpu().numpy())

    all_new_xyz = torch.cat(all_new_xyz, dim=0)
    tmp_radii = torch.zeros(all_new_xyz.shape[0])
    gaussians.densification_postfix(
        all_new_xyz.to(device),
        torch.cat(all_new_dc, dim=0).to(device),
        torch.cat(all_new_rest, dim=0).to(device),
        torch.cat(all_new_opacity, dim=0).to(device),
        torch.cat(all_new_scaling, dim=0).to(device),
        torch.cat(all_new_rotation, dim=0).to(device),
        tmp_radii.to(device))

    record = {key: np.concatenate(chunks, axis=0) for key, chunks in rec.items()}
    return record, selected_indices, k


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()

    dataset = build_dataset_args(args)
    out_dir = dataset.model_path
    os.makedirs(out_dir, exist_ok=True)

    # --- scene ---
    print(f"[1/6] Loading COLMAP reconstruction from {dataset.source_path}")
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    if args.holdout_test:
        scene_cameras_train_test_split(scene, verbose=True)
    subsample_cameras(scene, args.max_images, verbose=True)

    cfg = SimpleNamespace(matches_per_ref=args.matches_per_ref, num_refs=args.num_refs,
                          nns_per_ref=args.nns_per_ref, scaling_factor=args.scaling_factor,
                          proj_err_tolerance=args.proj_err_tolerance)
    cameras = scene.getTrainCameras()
    clamp_selection_params(cfg, len(cameras))
    n_sfm = int(gaussians._xyz.shape[0])
    print(f"      {len(cameras)} cameras, {n_sfm} SfM points")

    gaussians.tmp_radii = torch.zeros(gaussians._xyz.shape[0], device=device)
    gaussians.training_setup(load_gs_optimization_defaults())

    # --- models ---
    print("[2/6] Loading matcher and feature backbones")
    roma_model = (roma_indoor if args.roma_model == "indoors" else roma_outdoor)(device=device)
    roma_model.upsample_preds = False
    roma_model.symmetric = False

    dino_model = None
    if args.use_dino:
        dino_model = torch.hub.load("facebookresearch/dinov2", args.dino_name)
        dino_model = dino_model.to(device).eval()

    roma_scales = [int(s) for s in args.roma_scales.split(",") if s.strip()]
    if not roma_scales and dino_model is None:
        raise SystemExit("Nothing to extract: set --roma_scales and/or --use_dino.")
    extractor = MultiScaleFeatureExtractor(roma_model, roma_scales, dino_model,
                                           args.dino_res, device, args.feature_cache)
    print(f"      blocks: {', '.join(extractor.block_names)}")

    # --- PCA basis ---
    print(f"[3/6] Fitting PCA on {args.pca_images} images x {args.pca_pixels} pixels")
    dims = split_dims(args.feature_dim, extractor.block_names)
    pca = fit_pca(extractor, cameras, args.pca_images, args.pca_pixels, dims, device, args.seed)
    spans, total_dim = pca.layout()

    # --- init + record ---
    print(f"[4/6] Initializing: {cfg.num_refs} refs x {cfg.nns_per_ref} nns x "
          f"{cfg.matches_per_ref} matches, {total_dim}-d features")
    record, selected_indices, k = init_with_features(
        gaussians, scene, cfg, extractor, pca, device,
        roma_model=roma_model, cos_gate=args.cos_gate, verbose=args.verbose)

    # --- prune SfM, keeping the record aligned ---
    print("[5/6] Post-processing")
    n_after = int(gaussians._xyz.shape[0])
    with torch.no_grad():
        gaussians.tmp_radii = torch.zeros(n_after).to(device)
        mask = torch.cat([torch.ones(n_sfm, dtype=torch.bool),
                          torch.zeros(n_after - n_sfm, dtype=torch.bool)], dim=0)
        gaussians.prune_points(mask)   # record rows already start at the first EDGS point
        if args.final_scale_modifier != 1.0:
            gaussians._scaling.data = gaussians.scaling_inverse_activation(
                gaussians.scaling_activation(gaussians._scaling.data) * args.final_scale_modifier)

    n_final = int(gaussians._xyz.shape[0])
    assert n_final == record["xyz"].shape[0], (
        f"record ({record['xyz'].shape[0]}) and cloud ({n_final}) are out of sync")

    # --- save ---
    print("[6/6] Saving")
    ply_path = os.path.join(out_dir, "point_cloud", "iteration_0", "point_cloud.ply")
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    gaussians.save_ply(ply_path)
    write_cfg_args(out_dir, dataset, args.sh_degree)

    np.savez(os.path.join(out_dir, "features.npz"), **record)
    pca.save(os.path.join(out_dir, "pca_basis.npz"))

    meta = {
        "num_splats": int(n_final),
        "num_views_per_splat": int(k + 1),
        "feature_dim": int(total_dim),
        "blocks": spans,
        "explained_variance": {n: float(v.sum()) for n, v in pca.explained.items()},
        "reference_cameras": [int(i) for i in selected_indices],
        "num_cameras_used": len(cameras),
        "slot_0_is_reference": True,
        "gates_applied": False,
        "seconds": round(time.time() - t0, 1),
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "features_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      {n_final} splats x {k + 1} views x {total_dim} dims in "
          f"{meta['seconds']:.0f}s")
    print(f"      {ply_path}")
    print(f"      {os.path.join(out_dir, 'features.npz')}")
    return meta


if __name__ == "__main__":
    main()
