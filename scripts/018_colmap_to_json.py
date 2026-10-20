"""Convert COLMAP binary models to `sparse/0/colmap.json` for fast loading.

pycolmap's `SceneManager` reads every 2D keypoint of every image and walks every
3D point's track in Python, ~3.7 s per DL3DV scene (8 min for the 133 scenes in
018). `Parser` needs none of that: only the cameras, the image poses / names and
the 3D points' xyz / colour / error (for the normalization). This writes exactly
those, read with `data.parser.colmap_json=true` (data/datasets/colmap.py).

    <sparse>/colmap.json = {
      "cameras": {"<camera_id>": {"model_id": int, "width": int, "height": int, "params": [float]}},
      "images":  [{"image_id": int, "name": str, "camera_id": int, "qvec": [w, x, y, z], "tvec": [x, y, z]}],
      "points3D": {"ids": [int], "xyz": [[x, y, z]], "rgb": [[r, g, b]], "error": [float]}
    }

Floats are written with Python's shortest round-trip repr, so the values read back
are bit-identical to the binary model's doubles. Tracks / 2D keypoints are left
out (`Parser.point_indices` stays empty: no depth supervision with this path).

Standard library only (runs on the login node). Existing files are skipped.

    python scripts/018_colmap_to_json.py --split data/splits/dl3dv_133.json \
        --root $WORK/datasets/dl3dv --subdir gaussian_splat
    python scripts/018_colmap_to_json.py <scene_dir> [<scene_dir> ...]
"""

import argparse
import json
import os
import struct

# COLMAP camera model id -> number of parameters.
NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            n = NUM_PARAMS[model_id]
            params = list(struct.unpack(f"<{n}d", f.read(8 * n)))
            cameras[str(camera_id)] = {"model_id": model_id, "width": width, "height": height, "params": params}
    return cameras


def read_images_bin(path):
    """Poses and names; the 2D keypoints (24 bytes each) are skipped."""
    images = []
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        head = struct.Struct("<I4d3dI")
        for _ in range(num):
            data = head.unpack(f.read(head.size))
            name = b"".join(iter(lambda: f.read(1), b"\x00")).decode()
            (num_points2d,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * num_points2d, os.SEEK_CUR)
            images.append({
                "image_id": data[0], "name": name, "camera_id": data[8],
                "qvec": list(data[1:5]), "tvec": list(data[5:8]),
            })
    return images


def read_points3d_bin(path):
    """xyz, colour and error of every 3D point; tracks are skipped."""
    ids, xyz, rgb, error = [], [], [], []
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        head = struct.Struct("<Q3d3BdQ")
        for _ in range(num):
            data = head.unpack(f.read(head.size))
            ids.append(data[0])
            xyz.append(list(data[1:4]))
            rgb.append(list(data[4:7]))
            error.append(data[7])
            f.seek(8 * data[8], os.SEEK_CUR)
    return {"ids": ids, "xyz": xyz, "rgb": rgb, "error": error}


def sparse_dir(scene_dir):
    """`<scene>/sparse/0` (or `<scene>/sparse`), as `Parser` looks for it."""
    d = os.path.join(scene_dir, "sparse/0")
    return d if os.path.exists(d) else os.path.join(scene_dir, "sparse")


def convert(scene_dir):
    src = sparse_dir(scene_dir)
    out = os.path.join(src, "colmap.json")
    if os.path.exists(out):
        return f"{out}: exists, skipped"
    model = {
        "cameras": read_cameras_bin(os.path.join(src, "cameras.bin")),
        "images": read_images_bin(os.path.join(src, "images.bin")),
        "points3D": read_points3d_bin(os.path.join(src, "points3D.bin")),
    }
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(model, f)
    os.rename(tmp, out)
    return f"{out}: {len(model['images'])} images, {len(model['points3D']['ids'])} points"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dirs", nargs="*", help="scene folders (containing sparse/0)")
    ap.add_argument("--split", help="split JSON (data/dl3dv.py): convert every scene listed in it")
    ap.add_argument("--root", help="with --split: folder of the scene hashes")
    ap.add_argument("--subdir", default="", help="with --split: scene folder inside <root>/<hash>")
    args = ap.parse_args()
    scene_dirs = list(args.scene_dirs)
    if args.split:
        with open(args.split) as f:
            split = json.load(f)
        names = sorted(set(split["train"]) | set(split["eval_unseen"]))
        scene_dirs += [os.path.join(args.root, name, args.subdir) for name in names]
    for i, d in enumerate(scene_dirs, 1):
        print(f"[{i}/{len(scene_dirs)}] {convert(d)}", flush=True)


if __name__ == "__main__":
    main()
