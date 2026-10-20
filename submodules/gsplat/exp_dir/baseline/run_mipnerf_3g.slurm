#!/bin/bash
# MCMC 3DGS baselines on Mip-NeRF 360 (same configs as run_mipnerf.sh), inside the ffgsplat image.
# One array task per scene, all run in parallel (each on its own 3g.71gb slice).
# Submit from anywhere:  sbatch run_mipnerf_3g.slurm
# Needs the example deps first (tyro, viser, nerfview, fused-ssim, pycolmap fork):
#   sbatch $WORK/jobs/gsplat_examples_deps.sh

#SBATCH --job-name=gsplat_mcmc_mipnerf_3g
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200_3g.71gb:1                 # 2nd-largest slice (node has 4 of these)
#SBATCH --time=04:00:00                                    # per scene (3g.71gb allows up to 24h)
#SBATCH --array=0-2                                        # one task per entry in SCENES; add %N to cap concurrency (e.g. 0-4%2)
#SBATCH --output=/projects/hpc/radub/logs/%x_%A_%a.out    # %A = array job id, %a = task index
#SBATCH --error=/projects/hpc/radub/logs/%x_%A_%a.err

# --- Paths ---
IMAGE=/projects/hpc/radub/images/ffgsplat.sif
VENV=/projects/hpc/radub/.venvs/ffgs
EXTRA=/projects/hpc/radub/.venvs/gsplat_examples_extra     # pycolmap fork (gsplat examples only)
GSPLAT=/projects/hpc/radub/ffgsplat/submodules/gsplat

SCENE_DIR=/projects/hpc/radub/datasets/mipnerf
RESULT_DIR=/projects/hpc/radub/experiments/mcmc_baselines
RENDER_TRAJ_PATH="ellipse"

# --- Scenes: "name data_factor cap_max" (keep --array in sync with the number of entries) ---
SCENES=(
    # indoor: factor 2, cap 1M
    # "bonsai 2 1000000"
    "counter 2 1000000"
    "kitchen 2 1000000"
    # "room 2 1000000"
    # outdoor: factor 4, cap 1.5M
    # "bicycle 4 1500000"
    # "flowers 4 1500000"     # not in 360_v2.zip (needs 360_extra_scenes.zip)
    # "garden 4 1500000"
    # "stump 4 1500000"
    # "treehill 4 1500000"    # not in 360_v2.zip (needs 360_extra_scenes.zip)
)

read -r SCENE DATA_FACTOR CAP_MAX <<< "${SCENES[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$SCENE" ]; then
    echo "No scene for array index $SLURM_ARRAY_TASK_ID (only ${#SCENES[@]} entries)" >&2
    exit 1
fi

# --- Execution ---
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date) | GPU: $CUDA_VISIBLE_DEVICES"
echo "==> Running $SCENE (factor=$DATA_FACTOR, cap_max=$CAP_MAX)"

apptainer exec --nv --bind /projects/hpc/radub "$IMAGE" bash -c "
    source $VENV/bin/activate
    export PYTHONPATH=$EXTRA
    cd $GSPLAT
    python examples/simple_trainer.py mcmc --eval_steps 30000 --disable_viewer --disable_video \
        --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
"
STATUS=$?
echo "Finished $SCENE at $(date) with exit code $STATUS"
exit $STATUS
