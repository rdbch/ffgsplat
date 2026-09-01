#!/usr/bin/env bash
# Edit the values below and run from the repo root: ./run_edgs_extract.sh

python edgs_extract.py \
    -s data/garden \
    -o outputs/garden \
    --images images \
    --resolution -1 \
    --data_device cuda \
    \
    --num_refs 30 \
    --nns_per_ref 3 \
    --matches_per_ref 5000 \
    --roma_model outdoors \
    \
    --scaling_factor 0.001 \
    --proj_err_tolerance 0.01 \
    --sh_degree 3 \
    --final_scale_modifier 0.5 \
    \
    --roma_scales 4,8,16 \
    --dino_name dinov2_vitl14 \
    --dino_res 560 \
    --feature_dim 64 \
    --pca_images 20 \
    --pca_pixels 10000 \
    --cos_gate 0.5 \
    --feature_cache 8 \
    \
    --seed 228 \
    --device cuda:0

# flags left off, uncomment to enable:
#   --max_images 100              use only 100 of the input images, evenly spaced
#   --holdout_test                hold out every 8th image from the initialization
#   --white_background
#   --init_opacity 0.5            override the opacity of the valid splats
#   --add_SfM_init                keep the COLMAP points in front of the EDGS ones
#   --drop_invalid_points         remove points that failed the reprojection test
#   --save_checkpoint             also write chkpnt0.pth
#   --ply_path outputs/garden.ply write the PLY somewhere else
#   --pca_basis outputs/other/pca_basis.npz   reuse a basis across scenes
#   --use_dino                    second standalone DINOv2 (scale 16 already is DINOv2)
#   --no_features                 plain init, skip feature extraction
#   --verbose
