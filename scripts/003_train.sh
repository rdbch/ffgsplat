#!/usr/bin/env bash
set -euo pipefail

# Launch core/train.py. Run as `python -m core.train` from the repo root so
# `core`, `configs` and `data` all resolve as top-level packages (see
# core/train.py's imports).
#
# Any extra `key=value` args are forwarded as OmegaConf overrides and win over
# the values below, e.g.:
#   scripts/003_train.sh optimizer.lr=5e-4 trainer.num_steps=2000
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- scene / data ----------------------------------------------------------
SCENE=bonsai
DATA_DIR="$WORK/datasets/mipnerf/$SCENE"
FACTOR=2                     # image downscale factor (baseline: 2 indoor, 4 outdoor)
NORMALIZE=true               # must be true: the reference checkpoint is in normalized space
GAUSSIAN_CKPT_PATH="$WORK/experiments/mcmc_baselines/$SCENE/ckpts/ckpt_29999_rank0.pt"

TRAIN_BATCH_SIZE=8           # views rendered per step
EVAL_BATCH_SIZE=8
NUM_WORKERS=4

# ---- model -----------------------------------------------------------------
POINT_GRID_SIZE=0.005        # LitePT voxel size, in normalized scene units
SH_DEGREE=3
INIT_SCALE=0.01              # starting (post-activation) scale of every Gaussian
INIT_OPACITY=0.1
MAX_SCALE=1.0                # upper bound on predicted scales

# ---- optimization / loss ---------------------------------------------------
LR=2e-3                      # peak LR (AdamW), reached at the end of warmup
WEIGHT_DECAY=1e-9            # minimal; 0 disables
WARMUP_PCT=0.05              # fraction of NUM_STEPS spent in linear warmup
WARMUP_START_RATIO=0.1       # warmup starts at this fraction of LR
FINAL_LR_RATIO=0.01          # cosine decay ends at this fraction of LR
GRAD_CLIP=1.0                # max grad norm; 0 disables
NUM_STEPS=10000
SSIM_LAMBDA=0.2
SH_DEGREE_INTERVAL=1000      # raise the rendered SH degree every this many steps
RANDOM_BKGD=true             # random background in training (eval is always black)

# ---- eval / logging / checkpoints ------------------------------------------
EVAL_EVERY=200
LPIPS_NET=alex                # alex | vgg
LOG_TRAIN_STEPS=5
LOG_TRAIN_IMAGES_STEPS=50
LOG_EVAL_IMAGES=8            # first N eval views logged as images; -1 = all
SAVE_EVERY=1000

RUN_NAME="${SCENE}_gs${POINT_GRID_SIZE}_lr${LR}2"
OUTPUT_DIR="results/$RUN_NAME"
RESUME=""                    # path to <output_dir>/ckpts/step_*.pt to resume from

WANDB_PROJECT=ffgsplat
WANDB_MODE=online            # online | offline | disabled

DEVICE=cuda

ARGS=(
    "data.parser.data_dir=$DATA_DIR"
    "data.parser.factor=$FACTOR"
    "data.parser.normalize=$NORMALIZE"
    "data.parser.gaussian_ckpt_path=$GAUSSIAN_CKPT_PATH"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.train_num_workers=$NUM_WORKERS"
    "data.eval_batch_size=$EVAL_BATCH_SIZE"
    "data.eval_num_workers=$NUM_WORKERS"

    "head.sh_degree=$SH_DEGREE"
    "head.init_scale=$INIT_SCALE"
    "head.init_opacity=$INIT_OPACITY"
    "head.max_scale=$MAX_SCALE"

    "optimizer.lr=$LR"
    "optimizer.weight_decay=$WEIGHT_DECAY"
    "optimizer.warmup_pct=$WARMUP_PCT"
    "optimizer.warmup_start_ratio=$WARMUP_START_RATIO"
    "optimizer.final_lr_ratio=$FINAL_LR_RATIO"
    "optimizer.grad_clip=$GRAD_CLIP"

    "trainer.device=$DEVICE"
    "trainer.num_steps=$NUM_STEPS"
    "trainer.point_grid_size=$POINT_GRID_SIZE"
    "trainer.ssim_lambda=$SSIM_LAMBDA"
    "trainer.sh_degree_interval=$SH_DEGREE_INTERVAL"
    "trainer.random_bkgd=$RANDOM_BKGD"
    "trainer.eval_every=$EVAL_EVERY"
    "trainer.lpips_net=$LPIPS_NET"
    "trainer.log_train_steps=$LOG_TRAIN_STEPS"
    "trainer.log_train_images_steps=$LOG_TRAIN_IMAGES_STEPS"
    "trainer.log_eval_images=$LOG_EVAL_IMAGES"
    "trainer.save_every=$SAVE_EVERY"
    "trainer.output_dir=$OUTPUT_DIR"

    "wandb.project=$WANDB_PROJECT"
    "wandb.name=$RUN_NAME"
    "wandb.mode=$WANDB_MODE"
)

if [ -n "$RESUME" ]; then
    ARGS+=("trainer.resume=$RESUME")
fi

python -m core.train "${ARGS[@]}" "$@"
