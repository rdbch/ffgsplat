#!/usr/bin/env python3
"""
COLMAP reconstruction in -> EDGS initialization + per-splat multi-scale features out.

Self-contained: imports only the EDGS repo itself (source/, submodules/), nothing from
the other scripts here. No training, no wandb, no hydra, no renderer.

    python edgs_extract.py -s data/garden -o outputs/garden \
        --num_refs 30 --nns_per_ref 3 --matches_per_ref 5000 \
        --roma_scales 4,8 --use_dino --feature_dim 64

    python edgs_extract.py -s data/garden -o outputs/garden --no_features   # plain init

Expected input layout (same as 3DGS):

    scene_folder/images/<image N>
    scene_folder/sparse/0/{cameras,images,points3D}.bin

Output:

    point_cloud/iteration_0/point_cloud.ply   the initialization, standard 3DGS format
    features.npz                              per-splat, per-view record (below)
    pca_basis.npz                             mean + components, needed to project queries
    features_meta.json                        settings, block layout, row alignment
    cfg_args, cameras.json, input.ply         so 3DGS tooling can read the folder
    chkpnt0.pth                               optional, --save_checkpoint

THE RECORD
----------
Every splat is anchored to one pixel in a reference image and to one pixel in each of
its k matched neighbours, giving k+1 view slots. Slot 0 is always the reference.

    xyz          (N, 3)        float32   world position
    ref_cam      (N,)          int32     reference camera index
    nn_cam       (N, k)        int32     neighbour camera indices
    view_dirs    (N, k+1, 3)   float16   unit vector camera -> point, world frame
    rgb          (N, k+1, 3)   uint8     colour observed in each view
    feat         (N, k+1, D)   float16   PCA-projected multi-scale features
    certainty    (N, k+1)      float16   RoMa certainty (slot 0 = 1 by definition)
    reproj_err   (N, k, 2)     float16   (error in ref frame, error in neighbour frame)
    best_nn      (N,)          int16     neighbour that won the triangulation
    valid        (N, k+1)      bool      reprojection + occlusion gates, NOT applied

Gates are recorded rather than applied so a downstream model can learn how much to
trust a marginal observation instead of inheriting a threshold. `view_dirs` is what
makes the rest usable: without it the per-view appearances are unordered and view
dependence is unrecoverable.

Row i of every array corresponds to row `i + record_row_offset` of the PLY. The offset
is 0 unless --add_SfM_init keeps the COLMAP points in front; features_meta.json states
it either way.

FEATURE SOURCES
---------------
One forward pass through RoMa's encoder gives the whole multi-scale stack, because RoMa
already carries DINOv2 inside it (romatch/models/encoders.py, CNNandDinov2):

    --roma_scales   1 ->   64 ch @ H      VGG19-bn, before each max-pool: local texture
                    2 ->  128 ch @ H/2    and shading. High resolution, low invariance.
                    4 ->  256 ch @ H/4
                    8 ->  512 ch @ H/8
                   16 -> 1024 ch @ H/14   DINOv2 ViT-L/14 patch tokens. Semantic and
                                          material context. Keyed 16, but the grid is
                                          H/14 -- RoMa overwrites VGG's slot with it.

--use_dino adds a second, standalone DINOv2 through torch.hub. It is redundant with
scale 16 and only worth it to run DINOv2 at a different resolution than RoMa's.
"""

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from types import SimpleNamespace

# ----------------------------------------------------------------------------------
# Path setup. Before any repo import.
# ----------------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO_ROOT,
           os.path.join(REPO_ROOT, "submodules", "gaussian-splatting"),
           os.path.join(REPO_ROOT, "submodules", "RoMa")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==================================================================================
# Arguments
# ==================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="EDGS initialization with per-splat multi-scale RoMa + DINOv2 features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("input / output")
    io.add_argument("-s", "--source_path", required=True,
                    help="COLMAP scene folder (contains images/ and sparse/0/).")
    io.add_argument("-o", "--output", required=True, help="Output folder.")
    io.add_argument("--images", default="images",
                    help="Image subfolder name (images, images_2, images_4, ...).")
    io.add_argument("--resolution", type=int, default=-1,
                    help="3DGS resolution flag. -1 keeps the original size but downscales "
                         "images wider than 1600px; 1/2/4/8 divide by that factor.")
    io.add_argument("--white_background", action="store_true")
    io.add_argument("--data_device", default="cuda",
                    help="Where loaded images live ('cuda' or 'cpu'; 'cpu' for large sets).")
    io.add_argument("--ply_path", default=None,
                    help="Write the PLY here instead of <output>/point_cloud/iteration_0/.")
    io.add_argument("--save_checkpoint", action="store_true",
                    help="Also write <output>/chkpnt0.pth (train.py: load.gs=<output> load.gs_step=0).")

    sel = p.add_argument_group("how many images / correspondences")
    sel.add_argument("--max_images", type=int, default=None,
                     help="Use only this many input images, evenly spaced.")
    sel.add_argument("--num_refs", type=int, default=30,
                     help="Reference frames, picked by K-means over the camera poses.")
    sel.add_argument("--nns_per_ref", type=int, default=3,
                     help="Neighbour views matched against each reference. Also the "
                          "number of neighbour slots in the record.")
    sel.add_argument("--matches_per_ref", type=int, default=5_000,
                     help="Correspondences per reference. ~num_refs * this many splats.")
    sel.add_argument("--holdout_test", action="store_true",
                     help="Hold out every 8th image (3DGS eval protocol) so the "
                          "initialization only sees training views.")
    sel.add_argument("--roma_model", choices=["outdoors", "indoors"], default="outdoors")

    ini = p.add_argument_group("initialization")
    ini.add_argument("--scaling_factor", type=float, default=0.001,
                     help="Splat size as a fraction of its distance to the reference "
                          "camera, i.e. an angular size.")
    ini.add_argument("--proj_err_tolerance", type=float, default=0.01,
                     help="Points with a larger reprojection error get zero opacity "
                          "(or are dropped, see --drop_invalid_points).")
    ini.add_argument("--sh_degree", type=int, default=3)
    ini.add_argument("--final_scale_modifier", type=float, default=0.5,
                     help="Multiplier applied to all splat scales at the end.")
    ini.add_argument("--init_opacity", type=float, default=None,
                     help="Override the opacity of the valid splats (default 0.5).")
    ini.add_argument("--add_SfM_init", action="store_true",
                     help="Keep the COLMAP SfM points in front of the EDGS ones. Shifts "
                          "record_row_offset; the record still covers EDGS points only.")
    ini.add_argument("--drop_invalid_points", action="store_true",
                     help="Remove points that failed the reprojection test. The record "
                          "is masked identically.")

    ft = p.add_argument_group("features")
    ft.add_argument("--no_features", action="store_true",
                    help="Skip feature extraction entirely and just write the cloud.")
    ft.add_argument("--roma_scales", default="4,8,16",
                    help="RoMa encoder pyramid strides, comma separated. VGG19 levels "
                         "1 (64ch), 2 (128ch), 4 (256ch), 8 (512ch), and 16 which is "
                         "DINOv2 ViT-L/14 (1024ch at H/14, despite the key). Empty "
                         "string disables RoMa features.")
    ft.add_argument("--use_dino", action="store_true",
                    help="Sample a SECOND, standalone DINOv2 via torch.hub. Redundant "
                         "with --roma_scales 16, which already gives DINOv2 ViT-L/14 "
                         "from the same forward pass; use this only to run DINOv2 at a "
                         "different resolution than RoMa's.")
    ft.add_argument("--dino_name", default="dinov2_vitl14")
    ft.add_argument("--dino_res", type=int, default=560,
                    help="Square resolution DINOv2 runs at. Must be a multiple of 14.")
    ft.add_argument("--feature_dim", type=int, default=64,
                    help="Total PCA width, split evenly across blocks.")
    ft.add_argument("--pca_basis", default=None,
                    help="Load an existing pca_basis.npz instead of fitting one. Use this "
                         "to keep embeddings comparable across scenes.")
    ft.add_argument("--pca_images", type=int, default=20,
                    help="Random images used to fit the basis.")
    ft.add_argument("--pca_pixels", type=int, default=10_000,
                    help="Random pixels per image when fitting the basis.")
    ft.add_argument("--cos_gate", type=float, default=0.5,
                    help="A neighbour observation is marked invalid below this feature "
                         "cosine similarity to the reference. Recorded, never applied.")
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
    if not args.no_features and not args.roma_scales.strip() and not args.use_dino:
        p.error("nothing to extract: set --roma_scales and/or --use_dino, or --no_features")
    return args


# ==================================================================================
# Scene setup
# ==================================================================================
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


def load_gs_optimization_defaults():
    """
    Learning rates from configs/gs/base.yaml.

    No optimizer step is ever taken, but training_setup() has to run: both
    densification_postfix() and prune_points() go through the optimizer state.
    """
    return OmegaConf.load(os.path.join(REPO_ROOT, "configs", "gs", "base.yaml")).opt


def subsample_cameras(scene, max_images, verbose=False):
    """Keep only max_images training cameras, evenly spaced over the sequence."""
    if max_images is None:
        return
    for resolution in scene.train_cameras.keys():
        cams = scene.train_cameras[resolution]
        if max_images >= len(cams):
            continue
        idcs = np.unique(np.linspace(0, len(cams) - 1, max_images).round().astype(int))
        scene.train_cameras[resolution] = [cams[i] for i in idcs]
        if verbose:
            print(f"      resolution {resolution}: {len(cams)} -> "
                  f"{len(scene.train_cameras[resolution])} images")


def clamp_selection_params(cfg, n_cameras):
    """Keep num_refs / nns_per_ref within what the camera set can support."""
    if n_cameras < 2:
        raise RuntimeError(f"Need at least 2 cameras to triangulate, got {n_cameras}.")
    if cfg.num_refs > n_cameras:
        print(f"[warn] num_refs={cfg.num_refs} > {n_cameras} cameras. Clamped.")
        cfg.num_refs = n_cameras
    if cfg.nns_per_ref > n_cameras - 1:
        print(f"[warn] nns_per_ref={cfg.nns_per_ref} > {n_cameras - 1} neighbours. Clamped.")
        cfg.nns_per_ref = n_cameras - 1


def write_cfg_args(model_path, dataset, sh_degree):
    """3DGS viewers and render.py read this. Same format train.py writes."""
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


def cam_to_pil(cam):
    im = cam.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np.clip(im * 255, 0, 255).astype(np.uint8))


# ==================================================================================
# Feature extraction
# ==================================================================================
class MultiScaleFeatureExtractor:
    """
    Feature maps {block: (C, h, w)} per image, sampled at normalized keypoints.

    Maps are cached by camera index: neighbours recur across references, so without a
    cache each image gets encoded several times.
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
        # Same preprocessing RoMa uses for matching, so sampled features correspond to
        # what actually produced the warps.
        self._roma_tf = get_tuple_transform_ops(resize=(hs, ws), normalize=True)

        self.block_names = [f"roma{s}" for s in self.roma_scales]
        if self.dino is not None:
            self.block_names.append("dino")

    def _roma_pyramid(self, pil_image):
        """
        {stride: (C, h, w)} from RoMa's encoder (romatch CNNandDinov2).

        Verified against romatch/models/encoders.py: the outdoor/indoor models are built
        with use_vgg=True, so the pyramid is VGG19-bn taken before each max-pool, plus
        DINOv2 written into key 16:

            1  ->   64 ch @ H          2  ->  128 ch @ H/2
            4  ->  256 ch @ H/4        8  ->  512 ch @ H/8
            16 -> 1024 ch @ H/14       <- DINOv2 ViT-L/14 patch tokens, NOT stride 16

        Called on a single image. RegressionMatcher.extract_backbone_features would
        concatenate im_A and im_B into one batch (matcher.py:458), so going through
        self.encoder directly is both cheaper and unambiguous.
        """
        im, _ = self._roma_tf((pil_image, pil_image))
        x = im[None].to(self.device)

        pyramid, errors = None, []
        with torch.no_grad():
            for attempt in (
                lambda: self.roma.encoder(x, upsample=False),
                lambda: self.roma.encoder(x),
                lambda: self.roma.extract_backbone_features(
                    {"im_A": x, "im_B": x}, batched=False, upsample=False)[0],
            ):
                try:
                    pyramid = attempt()
                    break
                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    errors.append(repr(e))

        if not isinstance(pyramid, dict):
            raise RuntimeError(
                "Could not get a feature pyramid out of the RoMa model. Tried .encoder "
                f"and extract_backbone_features; errors: {errors}. Adapt "
                "MultiScaleFeatureExtractor._roma_pyramid to your romatch revision.")

        return {int(k): v[0] for k, v in pyramid.items()
                if torch.is_tensor(v) and v.dim() == 4}

    def _dino_map(self, pil_image):
        """(C, h, w) DINOv2 patch tokens at stride 14."""
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

        Returns {block: (N, C)}, each L2-normalized. Normalized coordinates make this
        independent of the resolution each backbone runs at.
        """
        blocks = self.maps_for(cam_idx, pil_image)
        grid = torch.as_tensor(kpts_xy, dtype=torch.float32, device=self.device)
        grid = grid.view(1, -1, 1, 2)

        out = {}
        for name, fmap in blocks.items():
            sampled = F.grid_sample(fmap[None], grid, mode="bilinear",
                                    padding_mode="border", align_corners=False)
            out[name] = F.normalize(sampled[0, :, :, 0].T.contiguous(), dim=1)
        return out

    def clear_cache(self):
        self._cache.clear()
        self._order.clear()


class BlockPCA:
    """One PCA basis per feature block, fitted on random pixels of random images."""

    def __init__(self, dims):
        self.dims = dict(dims)
        self.mean = {}
        self.components = {}
        self.explained = {}

    def fit(self, samples):
        for name, x in samples.items():
            d = min(self.dims[name], x.shape[1], x.shape[0])
            mean = x.mean(dim=0)
            xc = x - mean
            # pca_lowrank keeps this dependency-free and is plenty for a basis.
            _, s, v = torch.pca_lowrank(xc, q=min(d + 8, min(xc.shape)), center=False)
            self.mean[name] = mean
            self.components[name] = v[:, :d].T.contiguous()
            var = (s ** 2) / max(1, xc.shape[0] - 1)
            total = xc.pow(2).sum() / max(1, xc.shape[0] - 1)   # lowrank sees only q comps
            self.explained[name] = (var[:d] / total).cpu().numpy()
            self.dims[name] = d

    def project(self, blocks):
        """{block: (N, C)} -> (N, sum(dims)), blocks concatenated in sorted order."""
        parts = []
        for name in sorted(blocks.keys()):
            parts.append((blocks[name] - self.mean[name]) @ self.components[name].T)
        return torch.cat(parts, dim=1)

    def layout(self):
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

    @classmethod
    def load(cls, path, device):
        data = np.load(path)
        names = [k[len("mean_"):] for k in data.files if k.startswith("mean_")]
        pca = cls({n: int(data[f"components_{n}"].shape[0]) for n in names})
        for n in names:
            pca.mean[n] = torch.from_numpy(data[f"mean_{n}"]).to(device)
            pca.components[n] = torch.from_numpy(data[f"components_{n}"]).to(device)
            pca.explained[n] = data[f"explained_{n}"]
        return pca


def split_dims(total, block_names):
    """Split the PCA budget evenly; remainder to the last blocks in sorted order."""
    n = len(block_names)
    base, extra = total // n, total % n
    return {name: base + (1 if i >= n - extra else 0)
            for i, name in enumerate(sorted(block_names))}


def fit_pca(extractor, cameras, n_images, n_pixels, dims, device, seed=0):
    """
    Fit on random pixels of random images.

    Feature statistics do not depend on which pixels end up being keypoints, so this
    needs no matching: one cheap pre-pass instead of holding every full-width feature
    for every splat in memory.
    """
    rng = np.random.default_rng(seed)
    idcs = rng.choice(len(cameras), size=min(n_images, len(cameras)), replace=False)

    pools = {name: [] for name in extractor.block_names}
    for cam_idx in tqdm(idcs, desc="pca fit"):
        blocks = extractor.maps_for(int(cam_idx), cam_to_pil(cameras[int(cam_idx)]))
        for name, fmap in blocks.items():
            c, h, w = fmap.shape
            flat = fmap.reshape(c, h * w).T
            take = min(n_pixels, flat.shape[0])
            sel = torch.from_numpy(
                rng.choice(flat.shape[0], size=take, replace=False)).to(device)
            pools[name].append(F.normalize(flat[sel], dim=1).cpu())
    extractor.clear_cache()

    samples = {n: torch.cat(c, dim=0).to(device) for n, c in pools.items()}
    pca = BlockPCA(dims)
    pca.fit(samples)
    for name in sorted(pca.dims):
        print(f"      {name:>8}: {samples[name].shape[1]} -> {pca.dims[name]} dims, "
              f"{pca.explained[name].sum():.1%} variance kept")
    del samples, pools
    torch.cuda.empty_cache()
    return pca


# ==================================================================================
# Initialization, with the per-view record
# ==================================================================================
def init_with_features(gaussians, scene, cfg, roma_model, device,
                       extractor=None, pca=None, cos_gate=0.5):
    """
    Mirrors source.corr_init.init_gaussians_with_corr and reuses all of its helpers,
    adding the per-view feature record. Written out here rather than patched into
    corr_init because the sampling has to happen inside the per-neighbour loop.

    extractor=None runs the plain initialization and returns an empty record.
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
    rec = {key: [] for key in ("xyz", "ref_cam", "nn_cam", "view_dirs", "rgb", "feat",
                               "certainty", "reproj_err", "best_nn", "valid")}

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

                if extractor is None:
                    continue

                blocks = extractor.sample(nn_cam_idx, cam_to_pil(viewpoint_stack[nn_cam_idx]),
                                          kptsB_np[:M_eff])
                nn_blocks.append(blocks)
                nn_feats.append(pca.project(blocks))
                nn_cert.append(certainties_all[nn_slot].reshape(-1)[good_samples][:M_eff])

                # Colour observed in THIS neighbour. extract_keypoints_and_colors returns
                # kptsB_color from whichever neighbour won on certainty, not from nn_slot,
                # so using it would silently mix views. Sample the image directly.
                imB = imB_compound[nn_slot]
                H_B, W_B = imB.shape[:2]
                bx = np.clip((((kptsB_np[:M_eff, 0] + 1.) / 2.) * W_B).round().astype(int), 0, W_B - 1)
                by = np.clip((((kptsB_np[:M_eff, 1] + 1.) / 2.) * H_B).round().astype(int), 0, H_B - 1)
                nn_rgb.append(torch.as_tensor(imB[by, bx].astype(np.uint8)))

            # Winner per point: lowest worst-case reprojection error. corr_init's
            # select_best_keypoints computes this internally and drops the argmin.
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

            if extractor is None:
                continue

            # --- record ---
            ref_blocks = extractor.sample(source_idx, cam_to_pil(cam), kptsA_np[:N])
            feats = torch.stack([pca.project(ref_blocks)] + [f[:N] for f in nn_feats], dim=1)

            # Occlusion gate on the RAW features: every block is unit norm, so the
            # cosine of the concatenation is the mean of the per-block cosines. PCA
            # centring would distort it.
            order = sorted(ref_blocks.keys())
            ref_cat = torch.cat([ref_blocks[b][:N] for b in order], dim=1)
            nn_cat = torch.stack([torch.cat([nb[b][:N] for b in order], dim=1)
                                  for nb in nn_blocks], dim=1)
            cos = (nn_cat * ref_cat[:, None, :]).sum(-1) / len(order)

            centers = torch.stack(
                [cam.camera_center.clone().detach()] +
                [viewpoint_stack[int(closest[source_idx, j])].camera_center.clone().detach()
                 for j in range(len(warps_all))], dim=0)
            dirs = F.normalize(new_xyz[:, None, :] - centers[None, :, :], dim=2)

            cert = torch.stack([torch.ones(N, device=device)] +
                               [c[:N].to(device) for c in nn_cert], dim=1)
            rgb = torch.cat([torch.as_tensor(kptsA_color[:N].astype(np.uint8))[:, None, :],
                             torch.stack([c[:N] for c in nn_rgb], dim=1)], dim=1)
            err_pairs = np.transpose(
                np.stack([np.stack(errs1, 0), np.stack(errs2, 0)], axis=-1), (1, 0, 2))[:N]

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
            rec["best_nn"].append(best_nn[:N].astype(np.int16))
            rec["valid"].append(valid.cpu().numpy())

    all_new_xyz = torch.cat(all_new_xyz, dim=0)
    gaussians.densification_postfix(
        all_new_xyz.to(device),
        torch.cat(all_new_dc, dim=0).to(device),
        torch.cat(all_new_rest, dim=0).to(device),
        torch.cat(all_new_opacity, dim=0).to(device),
        torch.cat(all_new_scaling, dim=0).to(device),
        torch.cat(all_new_rotation, dim=0).to(device),
        torch.zeros(all_new_xyz.shape[0]).to(device))

    record = ({key: np.concatenate(chunks, axis=0) for key, chunks in rec.items()}
              if extractor is not None else {})
    return record, selected_indices, k


# ==================================================================================
# Main
# ==================================================================================
def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()
    want_features = not args.no_features

    dataset = build_dataset_args(args)
    out_dir = dataset.model_path
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(dataset.source_path):
        raise FileNotFoundError(f"Scene folder not found: {dataset.source_path}")

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
    print(f"      {len(cameras)} cameras used ({len(scene.getTestCameras())} held out), "
          f"{n_sfm} SfM points")

    # densification_postfix()/prune_points() index tmp_radii, and Scene filled _xyz
    # after the model was built, so resize it to match.
    gaussians.tmp_radii = torch.zeros(gaussians._xyz.shape[0], device=device)
    gaussians.training_setup(load_gs_optimization_defaults())

    # --- models ---
    print("[2/6] Loading matcher" + (" and feature backbones" if want_features else ""))
    roma_model = (roma_indoor if args.roma_model == "indoors" else roma_outdoor)(device=device)
    roma_model.upsample_preds = False
    roma_model.symmetric = False

    extractor = pca = None
    spans, total_dim = {}, 0
    if want_features:
        dino_model = None
        if args.use_dino:
            dino_model = torch.hub.load("facebookresearch/dinov2", args.dino_name)
            dino_model = dino_model.to(device).eval()
        roma_scales = [int(s) for s in args.roma_scales.split(",") if s.strip()]
        extractor = MultiScaleFeatureExtractor(roma_model, roma_scales, dino_model,
                                               args.dino_res, device, args.feature_cache)
        print(f"      blocks: {', '.join(extractor.block_names)}")

        # --- PCA basis ---
        if args.pca_basis:
            print(f"[3/6] Loading PCA basis from {args.pca_basis}")
            pca = BlockPCA.load(args.pca_basis, device)
            missing = set(extractor.block_names) - set(pca.dims)
            if missing:
                raise SystemExit(f"basis {args.pca_basis} has no block(s) {sorted(missing)}")
        else:
            print(f"[3/6] Fitting PCA on {args.pca_images} images x {args.pca_pixels} pixels")
            pca = fit_pca(extractor, cameras, args.pca_images, args.pca_pixels,
                          split_dims(args.feature_dim, extractor.block_names),
                          device, args.seed)
        spans, total_dim = pca.layout()
    else:
        print("[3/6] Features disabled (--no_features)")

    # --- init ---
    print(f"[4/6] Initializing: {cfg.num_refs} refs x {cfg.nns_per_ref} nns x "
          f"{cfg.matches_per_ref} matches" + (f", {total_dim}-d features" if want_features else ""))
    record, selected_indices, k = init_with_features(
        gaussians, scene, cfg, roma_model, device,
        extractor=extractor, pca=pca, cos_gate=args.cos_gate)

    # --- post-processing, record kept aligned through every prune ---
    print("[5/6] Post-processing")
    n_after = int(gaussians._xyz.shape[0])
    n_edgs = n_after - n_sfm
    row_offset = 0
    with torch.no_grad():
        if not args.add_SfM_init:
            gaussians.tmp_radii = torch.zeros(n_after).to(device)
            mask = torch.cat([torch.ones(n_sfm, dtype=torch.bool),
                              torch.zeros(n_edgs, dtype=torch.bool)], dim=0)
            gaussians.prune_points(mask)
            print(f"      removed {n_sfm} SfM points")
        else:
            row_offset = n_sfm

        if args.drop_invalid_points:
            gaussians.tmp_radii = torch.zeros(gaussians._xyz.shape[0]).to(device)
            prune_mask = (gaussians.get_opacity < 1e-3).squeeze()
            keep = (~prune_mask).cpu().numpy()
            n_bad = int(prune_mask.sum())
            if n_bad:
                gaussians.prune_points(prune_mask)
            if record:
                # Mask the record with the same decision, then re-derive the offset
                # from however many SfM rows survived in front of it.
                record = {key: value[keep[row_offset:]] for key, value in record.items()}
            row_offset = int(keep[:row_offset].sum())
            print(f"      dropped {n_bad} points that failed the reprojection test")

        # In-place edits from here on so the parameters stay the ones the optimizer
        # holds and --save_checkpoint stays trainable.
        if args.init_opacity is not None:
            p = float(np.clip(args.init_opacity, 1e-4, 1 - 1e-4))
            valid_op = (gaussians._opacity.data == 0.)
            gaussians._opacity.data[valid_op] = float(np.log(p / (1. - p)))
            print(f"      opacity of {int(valid_op.sum())} valid splats set to {p}")

        if args.final_scale_modifier != 1.0:
            gaussians._scaling.data = gaussians.scaling_inverse_activation(
                gaussians.scaling_activation(gaussians._scaling.data) * args.final_scale_modifier)
            print(f"      splat scales multiplied by {args.final_scale_modifier}")

    n_final = int(gaussians._xyz.shape[0])
    if record:
        assert record["xyz"].shape[0] == n_final - row_offset, (
            f"record ({record['xyz'].shape[0]}) and cloud rows "
            f"[{row_offset}:{n_final}] are out of sync")

    # --- save ---
    print("[6/6] Saving")
    ply_path = args.ply_path or os.path.join(out_dir, "point_cloud", "iteration_0",
                                             "point_cloud.ply")
    os.makedirs(os.path.dirname(os.path.abspath(ply_path)), exist_ok=True)
    gaussians.save_ply(ply_path)
    write_cfg_args(out_dir, dataset, args.sh_degree)

    ckpt_path = None
    if args.save_checkpoint:
        ckpt_path = os.path.join(out_dir, "chkpnt0.pth")
        torch.save((gaussians.capture(), 0), ckpt_path)

    if record:
        np.savez(os.path.join(out_dir, "features.npz"), **record)
        pca.save(os.path.join(out_dir, "pca_basis.npz"))

    meta = {
        "ply_path": os.path.abspath(ply_path),
        "checkpoint_path": ckpt_path,
        "num_splats": n_final,
        "record_rows": int(record["xyz"].shape[0]) if record else 0,
        "record_row_offset": row_offset,
        "num_views_per_splat": int(k + 1) if record else 0,
        "feature_dim": int(total_dim),
        "blocks": spans,
        "explained_variance": ({n: float(v.sum()) for n, v in pca.explained.items()}
                               if pca is not None else {}),
        "slot_0_is_reference": True,
        "gates_applied": False,
        "reference_cameras": [int(i) for i in selected_indices],
        "num_cameras_used": len(cameras),
        "num_sfm_points": n_sfm,
        "seconds": round(time.time() - t0, 1),
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "features_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      {n_final} splats" +
          (f" x {k + 1} views x {total_dim} dims" if record else "") +
          f" in {meta['seconds']:.0f}s")
    print(f"      {ply_path}")
    if record:
        print(f"      {os.path.join(out_dir, 'features.npz')}")
    if ckpt_path:
        print(f"      {ckpt_path}")
    return meta


if __name__ == "__main__":
    main()
