#!/usr/bin/env python3
"""
Simple calling script for edgs_init_from_colmap.py.

Edit the two blocks below and run it:

    python run_edgs_init.py

The scenes can also be given on the command line, which overrides SCENES:

    python run_edgs_init.py data/garden data/bicycle

No subprocess, no shell: this calls the initialization directly in-process.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edgs_init_from_colmap  # noqa: E402


# ----------------------------------------------------------------------------------
# What to process
# ----------------------------------------------------------------------------------
SCENES = [
    "data/garden",
]
OUTPUT_ROOT = "outputs/edgs_init"


# ----------------------------------------------------------------------------------
# Flags. None = leave at the script's own default, False = flag off, True = flag on.
# Run `python edgs_init_from_colmap.py --help` for the full list.
# ----------------------------------------------------------------------------------
SETTINGS = {
    # input
    "images": "images",           # image subfolder: images, images_2, images_4, ...
    "resolution": -1,
    "max_images": None,           # e.g. 100 to use only 100 evenly spaced images
    "holdout_test": False,        # True = hold out every 8th image, init on the rest

    # correspondences
    "num_refs": 64,               # reference frames
    "nns_per_ref": 2,             # neighbour views per reference (1 = fast path)
    "matches_per_ref": 15_000,    # correspondences per reference
    "roma_model": "outdoors",     # or "indoors"

    # initialization
    "scaling_factor": 0.001,
    "proj_err_tolerance": 0.01,
    "final_scale_modifier": 0.5,
    "init_opacity": None,
    "sh_degree": 3,
    "add_SfM_init": False,
    "drop_invalid_points": False,

    # output / misc
    "save_checkpoint": True,
    "white_background": False,
    "data_device": "cuda",
    "device": "cuda:0",
    "seed": 228,
    "verbose": False,
}


def to_argv(settings):
    """{'num_refs': 64, 'verbose': True, 'max_images': None} -> ['--num_refs', '64', '--verbose']"""
    argv = []
    for key, value in settings.items():
        if value is None or value is False:
            continue
        argv.append(f"--{key}")
        if value is not True:
            argv.append(str(value))
    return argv


def main():
    scenes = sys.argv[1:] or SCENES
    flags = to_argv(SETTINGS)

    for scene in scenes:
        name = os.path.basename(os.path.normpath(scene))
        output = os.path.join(OUTPUT_ROOT, name)
        # Note: RoMa is loaded once per scene, so a long list of scenes pays that cost
        # repeatedly. Fine for a handful of scenes.
        summary = edgs_init_from_colmap.main(["-s", scene, "-o", output] + flags)
        print(f"{name}: {summary['num_splats_final']} splats -> {summary['ply_path']}\n")


if __name__ == "__main__":
    main()
