#!/usr/bin/env bash
set -euo pipefail

# 020: bonsai with the sky sphere, 019c's settings otherwise.
#   - Points: $WORK/experiments/edgs_feat_sky/bonsai (edgs_extract_ref.py --sky): the
#     EDGS points + ~118k sky points at 4 sigma (is_sky). Fixed 1M subset of the scene
#     points (seed 0; the single-scene trainer doesn't resample), all sky points on top.
#   - SKY=true: the sky rows go in, with an is_sky flag as the last feature channel
#     (into the feature MLP); same voxel grid (0.005) as the scene points.
#     SKY=false: sky rows dropped (the matched baseline, same scene points).
#   - PE divider r in BOTH runs (input.pos_radius=sigma): r = |mu| + 4 sigma of one
#     Gaussian fitted to the scene points (training frame, sky excluded), the
#     farthest point of the 4-sigma sphere from the origin: every point incl. the
#     sky has |p| <= 1.
#   - Otherwise 019c: input -> head skip (concat 64, hidden 128), colour residual,
#     per-pixel random background, kNN views (anchor + 7 nearest), frustum cull 5 %,
#     mild augmentation, Muon 4e-4, SH 3 every 2000, max scale 0.4, offsets 1.5 vox,
#     Fourier PE F4 without raw p, drop-path 0, LayerNorm.
#   - Single scene: 20k steps, 8 views per step (no grad accum), eval every 1000.
#     Run as a one-scene multi-scene setup (data.scenes=[bonsai]) for the kNN sampler.
#
# SKY, NUM_STEPS, EVAL_EVERY and RESUME can be set from the environment.
# Extra `key=value` args are forwarded as OmegaConf overrides.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SKY=${SKY:-true}

# ---- scene / data ----------------------------------------------------------
SCENE=bonsai
SCENE_ROOT="$WORK/datasets/mipnerf"
FACTOR=2                     # images_2, as the MCMC baseline and 015-017
NORMALIZE=true

TRAIN_BATCH_SIZE=8           # views per step
VIEW_SAMPLING=knn            # next view of the shuffled order + its 7 nearest cameras
GRAD_ACCUM=1
EVAL_BATCH_SIZE=8
NUM_WORKERS=8                # jobs request 12 CPUs

# ---- input points / features -----------------------------------------------
EDGS_DIR="$WORK/experiments/edgs_feat_sky/{scene}"
FEATURES=rgb,roma1,roma2,roma4,roma16
MAX_POINTS=1000000           # scene points (019c bounded: 1M); sky points on top
DROP_INVALID=true
POINT_SEED=0
FEATURE_MLP_DIM=64
FEATURE_MLP_LAYERS=2
POS_ENCODING=fourier
POS_FREQS=4
POS_PERCENTILE=95            # unused with POS_RADIUS=sigma
POS_RADIUS=sigma             # r = |mu| + POS_SIGMA_K sigma of the scene points
POS_SIGMA_K=4
POS_INCLUDE_RAW=false
POS_INJECT=input
POS_NORM=none
POS_CLAMP=0

# ---- model -----------------------------------------------------------------
POINT_GRID_SIZE=0.005
MODEL_NORM=layer
DROP_PATH=0.0
SH_DEGREE=3
INIT_SCALE=0.01
INIT_OPACITY=0.1
MAX_SCALE=0.4
MAX_OFFSET_VOXELS=1.5
MAX_OFFSET=$(awk "BEGIN {print $MAX_OFFSET_VOXELS * $POINT_GRID_SIZE}")
HEAD_SKIP=concat
HEAD_SKIP_DIM=64
HEAD_HIDDEN=128
COLOR_RESIDUAL=true

# ---- augmentation (mild, 019c) ---------------------------------------------
AUGMENT=true
AUG_Z_ROT_DEG=30
AUG_TILT_DEG=2
AUG_SCALE_MIN=0.9
AUG_SCALE_MAX=1.1
AUG_VOXEL_JITTER=true
AUG_MIRROR_PROB=0.0

# ---- optimization / loss (019c) --------------------------------------------
OPTIMIZER=muon
LR=4e-4
MUON_LR=null
MUON_UPDATE_SCALE=moonlight
MUON_MOMENTUM=0.95
MUON_POLAR_STEPS=6
WEIGHT_DECAY=1e-9
WARMUP_PCT=0.05
WARMUP_START_RATIO=0.1
FINAL_LR_RATIO=0.01
GRAD_CLIP=1.0
NUM_STEPS=${NUM_STEPS:-20000}
SSIM_LAMBDA=0.2
OPACITY_REG=0.01
SCALE_REG=0
SCALE_TAIL_REG=0
SCALE_TAIL_THRESHOLD=0.1
SH_DEGREE_INTERVAL=2000
RANDOM_BKGD=true
BKGD_MODE=pixel
FRUSTUM_CULL=subset          # the fixed 1M (+ sky) culled to the step's 8 views
CULL_MARGIN=0.05
EVAL_BN_BATCH_STATS=false

# ---- eval / logging / checkpoints ------------------------------------------
EVAL_EVERY=${EVAL_EVERY:-1000}
LPIPS_NET=alex
LOG_TRAIN_STEPS=5
LOG_TRAIN_IMAGES_STEPS=500
LOG_EVAL_IMAGES=8
SAVE_EVERY=1000

SKY_TAG=nosky; [ "$SKY" = true ] && SKY_TAG=sky
RUN_NAME="020_${SCENE}_${SKY_TAG}4sig_peR4sig_skipconcat64_h128_rgbres_viewsknn_bkgdpixel_cullsubset${CULL_MARGIN}_edgs1000k_v${TRAIN_BATCH_SIZE}_lr${LR}_muon${MUON_UPDATE_SCALE}_off${MAX_OFFSET_VOXELS}vox_peinputF${POS_FREQS}noraw_maxs${MAX_SCALE}_dp${DROP_PATH}_${NUM_STEPS}_augmild_ln_sh${SH_DEGREE}every${SH_DEGREE_INTERVAL}"
OUTPUT_DIR="results/$RUN_NAME"
RESUME="${RESUME:-}"

WANDB_PROJECT=ffgsplat
WANDB_MODE=online

DEVICE=cuda
CUDA_MEMORY_FRACTION=0.9

ARGS=(
    "data.scenes=[$SCENE]"
    "data.scene_root=$SCENE_ROOT"
    "data.parser.data_dir=unused"
    "data.parser.factor=$FACTOR"
    "data.parser.normalize=$NORMALIZE"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.train_view_sampling=$VIEW_SAMPLING"
    "data.train_num_workers=$NUM_WORKERS"
    "data.eval_batch_size=$EVAL_BATCH_SIZE"
    "data.eval_num_workers=$NUM_WORKERS"

    "input.source=edgs"
    "input.edgs_dir=$EDGS_DIR"
    "input.features=[$FEATURES]"
    "input.max_points=$MAX_POINTS"
    "input.drop_invalid=$DROP_INVALID"
    "input.seed=$POINT_SEED"
    "input.sky=$SKY"
    "input.feature_mlp_dim=$FEATURE_MLP_DIM"
    "input.feature_mlp_layers=$FEATURE_MLP_LAYERS"
    "input.pos_encoding=$POS_ENCODING"
    "input.pos_freqs=$POS_FREQS"
    "input.pos_percentile=$POS_PERCENTILE"
    "input.pos_radius=$POS_RADIUS"
    "input.pos_sigma_k=$POS_SIGMA_K"
    "input.pos_include_raw=$POS_INCLUDE_RAW"
    "input.pos_inject=$POS_INJECT"
    "input.pos_norm=$POS_NORM"
    "input.pos_clamp_percentile=$POS_CLAMP"

    "model.norm=$MODEL_NORM"
    "model.drop_path=$DROP_PATH"

    "head.sh_degree=$SH_DEGREE"
    "head.init_scale=$INIT_SCALE"
    "head.init_opacity=$INIT_OPACITY"
    "head.max_scale=$MAX_SCALE"
    "head.max_offset=$MAX_OFFSET"
    "head.skip=$HEAD_SKIP"
    "head.skip_dim=$HEAD_SKIP_DIM"
    "head.hidden_dim=$HEAD_HIDDEN"
    "head.color_residual=$COLOR_RESIDUAL"

    "augment.enabled=$AUGMENT"
    "augment.z_rot_deg=$AUG_Z_ROT_DEG"
    "augment.tilt_deg=$AUG_TILT_DEG"
    "augment.scale_min=$AUG_SCALE_MIN"
    "augment.scale_max=$AUG_SCALE_MAX"
    "augment.voxel_jitter=$AUG_VOXEL_JITTER"
    "augment.mirror_prob=$AUG_MIRROR_PROB"

    "optimizer.name=$OPTIMIZER"
    "optimizer.lr=$LR"
    "optimizer.muon_lr=$MUON_LR"
    "optimizer.muon_update_scale=$MUON_UPDATE_SCALE"
    "optimizer.muon_momentum=$MUON_MOMENTUM"
    "optimizer.muon_polar_steps=$MUON_POLAR_STEPS"
    "optimizer.weight_decay=$WEIGHT_DECAY"
    "optimizer.warmup_pct=$WARMUP_PCT"
    "optimizer.warmup_start_ratio=$WARMUP_START_RATIO"
    "optimizer.final_lr_ratio=$FINAL_LR_RATIO"
    "optimizer.grad_clip=$GRAD_CLIP"

    "trainer.device=$DEVICE"
    "trainer.cuda_memory_fraction=$CUDA_MEMORY_FRACTION"
    "trainer.grad_accum_steps=$GRAD_ACCUM"
    "trainer.num_steps=$NUM_STEPS"
    "trainer.point_grid_size=$POINT_GRID_SIZE"
    "trainer.ssim_lambda=$SSIM_LAMBDA"
    "trainer.opacity_reg=$OPACITY_REG"
    "trainer.scale_reg=$SCALE_REG"
    "trainer.scale_tail_reg=$SCALE_TAIL_REG"
    "trainer.scale_tail_threshold=$SCALE_TAIL_THRESHOLD"
    "trainer.sh_degree_interval=$SH_DEGREE_INTERVAL"
    "trainer.random_bkgd=$RANDOM_BKGD"
    "trainer.random_bkgd_mode=$BKGD_MODE"
    "trainer.frustum_cull=$FRUSTUM_CULL"
    "trainer.cull_margin=$CULL_MARGIN"
    "trainer.eval_bn_batch_stats=$EVAL_BN_BATCH_STATS"
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

if [ "$RESUME" = latest ]; then
    RESUME=$(ls "$OUTPUT_DIR"/ckpts/step_*.pt 2>/dev/null | sort | tail -n 1 || true)
    echo "Resuming from: ${RESUME:-nothing (fresh start)}"
fi
if [ -n "$RESUME" ]; then
    ARGS+=("trainer.resume=$RESUME")
fi

python -m core.train "${ARGS[@]}" "$@"
