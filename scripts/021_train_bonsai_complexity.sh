#!/usr/bin/env bash
set -euo pipefail

# 021: 020-sky (bonsai, sky sphere, 019c's settings) with the scene points' 1M subset
# drawn by image complexity.
#   - Points: $WORK/experiments/edgs_feat_sky_cplx/bonsai (edgs_extract_ref.py --sky
#     --complexity_dir): EDGS points + sky + each row's reference-view delentropy
#     (kernel 32, stride 4) at its ref_uv.
#   - SUBSET=complexity (default): 20 % of the 1M uniform, 80 % without replacement
#     with weights exp(rho), rho = complexity clipped to its [p1, p99] over the scene
#     points, then min-max normalized (KoalaGS's weighting, beta 1; the clip because
#     delentropy saturates at 5.0 and ~0 outliers would flatten rho); SUBSET=random:
#     uniform 1M (the matched baseline on
#     the same file). Sky points: all kept in both.
#   - SUBSET=quantile: the viewer's rule ($WORK/viewers/complexity): complexity -> rank q;
#     floor of COMPLEXITY_FLOOR (100) top-key points in each of COMPLEXITY_BINS (50)
#     equal-width complexity bins; of the rest of the 1M, SUBSET_RANDOM_FRAC uniform and
#     the others by largest key COMPLEXITY_BETA * q + Gumbel (~ exp(4 q) without
#     replacement). Defaults for quantile: random 0.05, beta 4.
#   - Everything else as 020-sky (PE radius |mu| + 4 sigma, same grid, is_sky flag, ...).
#
# SUBSET, SKY, NUM_STEPS, EVAL_EVERY and RESUME can be set from the environment.
# Extra `key=value` args are forwarded as OmegaConf overrides.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SKY=${SKY:-true}
SUBSET=${SUBSET:-complexity}   # complexity | quantile | random
if [ "$SUBSET" = quantile ]; then SUBSET_RANDOM_FRAC=0.05; COMPLEXITY_BETA=4.0; else SUBSET_RANDOM_FRAC=0.2; COMPLEXITY_BETA=1.0; fi
COMPLEXITY_BINS=50
COMPLEXITY_FLOOR=100
COMPLEXITY_CLIP=1.0            # clip rho's input to [p1, p99]

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
EDGS_DIR="$WORK/experiments/edgs_feat_sky_cplx/{scene}"
FEATURES=rgb,roma1,roma2,roma4,roma16
MAX_POINTS=1000000           # scene points (019c bounded: 1M), drawn by SUBSET; sky points on top
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
SUBSET_TAG=subsetrandom; [ "$SUBSET" = complexity ] && SUBSET_TAG="subsetcplx${SUBSET_RANDOM_FRAC}r_b${COMPLEXITY_BETA}_clip${COMPLEXITY_CLIP}"
[ "$SUBSET" = quantile ] && SUBSET_TAG="subsetquant${SUBSET_RANDOM_FRAC}r_b${COMPLEXITY_BETA}_bins${COMPLEXITY_BINS}_floor${COMPLEXITY_FLOOR}"
RUN_NAME="021_${SCENE}_${SUBSET_TAG}_${SKY_TAG}4sig_peR4sig_skipconcat64_h128_rgbres_viewsknn_bkgdpixel_cullsubset${CULL_MARGIN}_edgs1000k_v${TRAIN_BATCH_SIZE}_lr${LR}_muon${MUON_UPDATE_SCALE}_off${MAX_OFFSET_VOXELS}vox_peinputF${POS_FREQS}noraw_maxs${MAX_SCALE}_dp${DROP_PATH}_${NUM_STEPS}_augmild_ln_sh${SH_DEGREE}every${SH_DEGREE_INTERVAL}"
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
    "input.subset=$SUBSET"
    "input.subset_random_frac=$SUBSET_RANDOM_FRAC"
    "input.complexity_beta=$COMPLEXITY_BETA"
    "input.complexity_clip=$COMPLEXITY_CLIP"
    "input.complexity_bins=$COMPLEXITY_BINS"
    "input.complexity_floor=$COMPLEXITY_FLOOR"
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
