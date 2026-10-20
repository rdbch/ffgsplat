#!/usr/bin/env python3
"""
Fit ONE global PCA basis per feature block over the training images of several scenes.

The basis is shared by every scene edgs_extract_ref.py processes, so the per-point
features of all scenes live in the same coordinate system (needed for a cross-scene
model, harmless for a per-scene one).

    python edgs_fit_pca.py -o pca/mipnerf_basis.npz \
        --scenes data/bicycle:images_4 data/room:images_2 ... \
        --dims roma1=8,roma2=16,roma4=32,roma16=48

Exact PCA from streamed statistics: for every block, sum(x) and sum(x x^T) are
accumulated in float64 over ALL pixels of every image (C <= 1024, so x x^T is at most
1024 x 1024), then the covariance is eigendecomposed. No pixel subsampling, no
pca_lowrank approximation.

Each pixel vector is L2-normalized per block before the statistics, exactly as
MultiScaleFeatureExtractor.sample() does for the keypoints that get projected later.

Holdout: images are sorted by name and every 8th (idx % 8 == 0) is skipped, the same
split as 3DGS (llffhold=8) and ffgsplat's colmap.py (test_every=8). Fit only on scenes
you train on if you evaluate on held-out scenes.

Output: <out>.npz in BlockPCA's format (loadable with BlockPCA.load) plus <out>.json
with the cumulative explained variance at several widths, to choose --dims.
"""

import argparse
import json
import os
import sys
import time

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

from romatch import roma_indoor, roma_outdoor  # noqa: E402

from edgs_extract import BlockPCA, MultiScaleFeatureExtractor  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
REPORT_WIDTHS = (4, 8, 16, 32, 64, 128, 256)


def parse_dims(text):
    """'roma1=16,roma16=64' -> {'roma1': 16, 'roma16': 64}"""
    dims = {}
    for item in text.split(","):
        name, d = item.split("=")
        dims[name.strip()] = int(d)
    return dims


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-o", "--output", required=True, help="Output .npz path.")
    p.add_argument("--scenes", nargs="+", required=True,
                   help="scene_folder:image_subfolder, e.g. data/room:images_2.")
    p.add_argument("--dims", default="roma1=8,roma2=16,roma4=32,roma16=48",
                   help="PCA width per block. Block names: roma<stride>, stride in "
                        "1,2,4,8 (VGG19-bn) or 16 (DINOv2 ViT-L/14, grid H/14).")
    p.add_argument("--no_holdout", action="store_true",
                   help="Also use every 8th image (the test views).")
    p.add_argument("--max_images_per_scene", type=int, default=None,
                   help="Evenly spaced subset of each scene's training images.")
    p.add_argument("--roma_model", choices=["outdoors", "indoors"], default="outdoors")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def training_images(scene_dir, image_dir, holdout, max_images):
    folder = os.path.join(scene_dir, image_dir)
    names = sorted(f for f in os.listdir(folder) if f.endswith(IMAGE_EXTS))
    if holdout:
        names = [n for i, n in enumerate(names) if i % 8 != 0]
    if max_images is not None and max_images < len(names):
        idcs = np.unique(np.linspace(0, len(names) - 1, max_images).round().astype(int))
        names = [names[i] for i in idcs]
    return [os.path.join(folder, n) for n in names]


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    dims = parse_dims(args.dims)
    strides = sorted(int(n[len("roma"):]) for n in dims)
    t0 = time.time()

    roma_model = (roma_indoor if args.roma_model == "indoors" else roma_outdoor)(device=device)
    extractor = MultiScaleFeatureExtractor(roma_model, strides, device=device, cache_size=1)
    print(f"blocks {extractor.block_names}, RoMa input {roma_model.h_resized}x{roma_model.w_resized}")

    images = []
    for spec in args.scenes:
        scene_dir, image_dir = spec.rsplit(":", 1)
        imgs = training_images(scene_dir, image_dir, not args.no_holdout,
                               args.max_images_per_scene)
        print(f"  {scene_dir} ({image_dir}): {len(imgs)} images")
        images += imgs

    n, s1, s2 = {}, {}, {}
    for i, path in enumerate(tqdm(images, desc="statistics")):
        blocks = extractor.maps_for(i, Image.open(path).convert("RGB"))
        for name, fmap in blocks.items():
            c = fmap.shape[0]
            x = F.normalize(fmap.reshape(c, -1).T, dim=1).double()
            if name not in n:
                n[name] = 0
                s1[name] = torch.zeros(c, dtype=torch.float64, device=device)
                s2[name] = torch.zeros(c, c, dtype=torch.float64, device=device)
            n[name] += x.shape[0]
            s1[name] += x.sum(0)
            s2[name] += x.T @ x
        extractor.clear_cache()

    pca = BlockPCA(dims)
    report = {}
    for name in sorted(dims):
        mean = s1[name] / n[name]
        cov = (s2[name] - n[name] * torch.outer(mean, mean)) / (n[name] - 1)
        evals, evecs = torch.linalg.eigh(cov)                  # ascending
        evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
        ratio = (evals / evals.sum()).cpu().numpy()
        d = min(dims[name], cov.shape[0])
        pca.dims[name] = d
        pca.mean[name] = mean.float()
        pca.components[name] = evecs[:, :d].T.contiguous().float()
        pca.explained[name] = ratio[:d].astype(np.float32)
        cum = np.cumsum(ratio)
        report[name] = {
            "channels": int(cov.shape[0]),
            "pixels": int(n[name]),
            "dim": d,
            "variance_kept": float(cum[d - 1]),
            "cumulative_at": {w: float(cum[w - 1]) for w in REPORT_WIDTHS if w <= len(cum)},
        }
        print(f"  {name:>7}: {cov.shape[0]:4d} -> {d:3d} dims, {cum[d - 1]:.1%} variance kept  "
              + "  ".join(f"@{w}:{cum[w - 1]:.0%}" for w in REPORT_WIDTHS if w <= len(cum)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    spans, total = pca.layout()
    meta = {
        "blocks": spans,
        "feature_dim": total,
        "blocks_report": report,
        "num_images": len(images),
        "roma_input_hw": [int(roma_model.h_resized), int(roma_model.w_resized)],
        "seconds": round(time.time() - t0, 1),
        "args": vars(args),
    }
    pca.save(args.output)
    with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{total}-d basis from {len(images)} images in {meta['seconds']:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
