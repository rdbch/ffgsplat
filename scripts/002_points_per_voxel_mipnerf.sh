#!/usr/bin/env bash
set -euo pipefail

# Run scripts/001_points_per_voxel.py over every MipNeRF-360 scene, reading
# checkpoints written by submodules/gsplat/exp_dir/baseline/run_mipnerf.sh.

RESULT_DIR="${WORK}/experiments/mcmc_baselines"
CKPT_STEP=29999

VOXEL_SIZES=(0.001 0.005 0.01 0.05 0.1)
MIN_ALPHA=0.005

OUT_DIR="${WORK}/experiments/voxel_size_analysis_005"
mkdir -p "$OUT_DIR"

INDOOR_SCENES=(
    "bonsai"
    "counter"
    "kitchen"
    "room"
)

OUTDOOR_SCENES=(
    "bicycle"
    "flowers"
    "garden"
    "stump"
    "treehill"
)

SCENES=("${INDOOR_SCENES[@]}" "${OUTDOOR_SCENES[@]}")

for SCENE in "${SCENES[@]}"; do
    CKPT="$RESULT_DIR/$SCENE/ckpts/ckpt_${CKPT_STEP}_rank0.pt"

    if [ ! -f "$CKPT" ]; then
        echo "Skipping $SCENE: checkpoint not found at $CKPT"
        continue
    fi

    echo "Processing $SCENE"
    python scripts/001_points_per_voxel.py \
        --ckpt "$CKPT" \
        --voxel-sizes "${VOXEL_SIZES[@]}" \
        --min-alpha "$MIN_ALPHA" \
        --out-plot "$OUT_DIR/${SCENE}.png"
done
