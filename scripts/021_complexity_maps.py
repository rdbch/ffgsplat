#!/usr/bin/env python3
"""Per-image complexity maps (imgvibecheck) for the EDGS point selection (021).

    python scripts/021_complexity_maps.py --scene_dir $WORK/datasets/mipnerf/bonsai --images images_2

For every image in <scene_dir>/<images>/: the local `--metric` (default delentropy) over
`--kernel_size` windows every `--stride` pixels, "same" padding (reflect), i.e. a map of
ceil(H / stride) x ceil(W / stride). Entry (i, j) is the window over pixels
[stride * j - (k - 1) // 2, stride * j - (k - 1) // 2 + k) (and the same in y): its centre
sits at x = stride * j + 1 - k % 2 / 2 in continuous pixel coordinates (pixel p spans
[p, p + 1)); `edgs_extract_ref.py --complexity_dir` samples the maps with that convention.

Written to <scene_dir>/<metric>_k<kernel>s<stride>_<images suffix>/<image stem>.tif (float32 TIFF, as KoalaGS),
through a temp file + rename; existing maps are skipped. imgvibecheck is imported from
$WORK/GitHub/imgvibecheck/src (not installed in the venv). A `meta.json` next to the maps
records the settings and every image's size ({stem: [H, W]}, needed to map normalized
coordinates to map entries).
"""
import argparse
import json
import os
import sys

IVC_SRC = "/projects/hpc/radub/GitHub/imgvibecheck/src"
if IVC_SRC not in sys.path:
    sys.path.insert(0, IVC_SRC)

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

import imgvibecheck as ivc  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def map_dir(scene_dir, images, metric, kernel_size, stride):
    suffix = images.split("_", 1)[1] if "_" in images else "1"
    return os.path.join(scene_dir, f"{metric}_k{kernel_size}s{stride}_{suffix}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--images", default="images", help="Image folder (the one EDGS uses).")
    p.add_argument("--metric", default="delentropy")
    p.add_argument("--kernel_size", type=int, default=32)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    src = os.path.join(args.scene_dir, args.images)
    out = map_dir(args.scene_dir, args.images, args.metric, args.kernel_size, args.stride)
    os.makedirs(out, exist_ok=True)
    names = sorted(f for f in os.listdir(src) if f.endswith(IMAGE_EXTS))
    todo = [f for f in names if not os.path.exists(os.path.join(out, os.path.splitext(f)[0] + ".tif"))]
    print(f"{src}: {len(names)} images, {len(todo)} maps to compute -> {out}")

    analyzer = ivc.ComplexityAnalyzer(metric=args.metric, device=args.device)
    lo, hi, shape = np.inf, -np.inf, None
    with torch.no_grad():
        for f in tqdm(todo, desc=args.metric):
            image = np.asarray(Image.open(os.path.join(src, f)).convert("RGB"))       # [H, W, 3] uint8
            cmap = analyzer.compute_map(torch.from_numpy(image).to(args.device), args.kernel_size,
                                        args.stride, padding="same")
            cmap = cmap.float().cpu().numpy().astype(np.float32)
            h, w = image.shape[:2]
            assert cmap.shape == ((h - 1) // args.stride + 1, (w - 1) // args.stride + 1), (cmap.shape, h, w)
            path = os.path.join(out, os.path.splitext(f)[0] + ".tif")
            tifffile.imwrite(path + ".tmp", cmap)        # never leave a half-written map behind
            os.replace(path + ".tmp", path)
            lo, hi, shape = min(lo, float(cmap.min())), max(hi, float(cmap.max())), (h, w, *cmap.shape)

    sizes = {}
    for f in names:
        with Image.open(os.path.join(src, f)) as im:                     # header only
            sizes[os.path.splitext(f)[0]] = [im.height, im.width]
    meta = {"metric": args.metric, "kernel_size": args.kernel_size, "stride": args.stride,
            "padding": "same", "images": args.images, "num_images": len(names),
            "window_centre": "x = stride * j + 1 - (kernel_size % 2) / 2 (continuous pixel coords)",
            "sizes": sizes}
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if shape:
        print(f"done: image {shape[0]}x{shape[1]} -> map {shape[2]}x{shape[3]}, values in [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
