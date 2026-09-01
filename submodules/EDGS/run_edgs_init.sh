#!/usr/bin/env bash
# Edit the paths/flags below and run from the repo root: ./run_edgs_init.sh

python edgs_init_from_colmap.py \
    -s data/garden \
    -o outputs/garden_init \
    --images images \
    --num_refs 64 \
    --nns_per_ref 2 \
    --matches_per_ref 15000 \
    --scaling_factor 0.001 \
    --save_checkpoint

# other flags, uncomment as needed:
#   --max_images 100          use only 100 of the input images
#   --holdout_test            hold out every 8th image
#   --roma_model indoors      indoor matcher
#   --drop_invalid_points     remove points that failed the reprojection test
#   --add_SfM_init            keep the COLMAP points too
#   --verbose
