#!/usr/bin/env bash
set -euo pipefail

# 022: 019c (R4 of the 10-scene sub: skip concat 64 + hidden 128, colour residual,
# per-pixel background, kNN views, frustum cull "subset" 5 %) + the sky sphere (020):
#   - points from edgs_feat_dl3dv_sky_cplx (edgs_extract_ref.py --sky --complexity_dir,
#     job 874; the complexity column is not used: uniform subset as in 019);
#   - SKY=true: every sky point of the scene in every forward on top of the
#     POINTS_BOUNDED / POINTS_UNBOUNDED scene points, flagged in the last feature
#     channel (input.sky); SKY=false drops them (same scene points as 019);
#   - POS_RADIUS=sigma: fixed PE divider r = |mu| + 4 sigma of all valid scene points
#     (the sky sphere's radius: every point incl. the sky at |p| <= 1); "percentile" =
#     019's r95 of the drawn subset.
# Defaults = the R4 setup (TRAIN_ON=eval_seen: 5 bounded + 5 unbounded, 500k points,
# grad accum 4, 10k steps, eval every 500), but eval_unseen = only the first
# EVAL_UNSEEN_LIMIT (3) of the 13; see jobs/ffgsplat_022.slurm.
#
# SKY, POS_RADIUS and 019's env vars (HEAD_SKIP, ..., NUM_STEPS, GRAD_ACCUM, EVAL_EVERY,
# TRAIN_ON, TRAIN_ENVIRONMENT / TRAIN_LIMIT, POINTS_BOUNDED / POINTS_UNBOUNDED, RESUME)
# can be set from the environment. Extra `key=value` args are OmegaConf overrides.
# CHECK_ONLY=true: run scripts/022_check_sky.py with this config instead of training.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- scenes / data ---------------------------------------------------------
DL3DV_ROOT="$WORK/datasets/dl3dv"
SPLIT_FILE=data/splits/dl3dv_133.json
FACTOR=4                     # images_4 (960x540); the only resolution downloaded
NORMALIZE=true               # EDGS points are mapped into the parser's normalized frame
COLMAP_JSON=true             # read sparse/0/colmap.json (scripts/018_colmap_to_json.py), not the .bin files
TARGET_RADIUS=3.56           # scale every scene so its COLMAP points' r95 is this (bonsai's EDGS r95)

TRAIN_BATCH_SIZE=8           # views of one scene per micro-batch
VIEW_SAMPLING=${VIEW_SAMPLING:-knn}         # random | knn (next shuffled view + its 7 nearest cameras)
GRAD_ACCUM=${GRAD_ACCUM:-4}  # micro-batches per optimizer step
EVAL_BATCH_SIZE=8
NUM_WORKERS=16               # train loader (job has 24 CPUs; the main process gathers the points)
EVAL_NUM_WORKERS=2           # per eval scene
LOAD_THREADS=8               # scenes loaded in parallel at startup

# ---- input points / features -----------------------------------------------
EDGS_DIR="$WORK/experiments/edgs_feat_dl3dv_sky_cplx/{scene}"
SKY=${SKY:-true}
POS_RADIUS=${POS_RADIUS:-sigma}               # sigma (|mu| + 4 sigma, fixed) | percentile (r95 of the subset)
POS_SIGMA_K=4
FEATURES=rgb,roma1,roma2,roma4,roma16
POINTS_BOUNDED=${POINTS_BOUNDED:-500000}
POINTS_UNBOUNDED=${POINTS_UNBOUNDED:-500000}
RESAMPLE_POINTS=true         # fresh random subset per training micro-batch
TRAIN_ENVIRONMENT=${TRAIN_ENVIRONMENT:-null}   # bounded | unbounded | null (all): train on these scenes only
TRAIN_LIMIT=${TRAIN_LIMIT:-null}               # first N of them | null (all); with either set, eval_seen = the trained scenes
TRAIN_ON=${TRAIN_ON:-eval_seen}                    # train (120 scenes) | eval_seen (its 10: 5 bounded / 5 unbounded)
EVAL_UNSEEN_LIMIT=${EVAL_UNSEEN_LIMIT:-3}       # first N of the 13 eval_unseen scenes | null (all)
LAZY=${LAZY:-false}          # true (debugging): only LAZY_TRAIN_SCENES training scenes, points loaded on first use
LAZY_TRAIN_SCENES=4
DROP_INVALID=true
POINT_SEED=0                 # seed of the fixed eval subsets
FEATURE_MLP_DIM=64
FEATURE_MLP_LAYERS=2
POS_ENCODING=fourier
POS_FREQS=4
POS_PERCENTILE=95
POS_INCLUDE_RAW=false
POS_INJECT=input
POS_NORM=none
POS_CLAMP=0

# ---- model -----------------------------------------------------------------
POINT_GRID_SIZE=0.005
MODEL_NORM=layer
DROP_PATH=0.0                # 0.3 collapses eval with PE (504-511)
SH_DEGREE=3
INIT_SCALE=0.01
INIT_OPACITY=0.1
MAX_SCALE=0.4
MAX_OFFSET_VOXELS=1.5
MAX_OFFSET=$(awk "BEGIN {print $MAX_OFFSET_VOXELS * $POINT_GRID_SIZE}")
HEAD_SKIP=${HEAD_SKIP:-concat}                 # none | add | concat
HEAD_SKIP_DIM=${HEAD_SKIP_DIM:-64}             # concat only
HEAD_HIDDEN=${HEAD_HIDDEN:-128}                # 0 = linear head
COLOR_RESIDUAL=${COLOR_RESIDUAL:-true}

# ---- augmentation (mild) ---------------------------------------------------
AUGMENT=true
AUG_Z_ROT_DEG=30
AUG_TILT_DEG=2
AUG_SCALE_MIN=0.9
AUG_SCALE_MAX=1.1
AUG_VOXEL_JITTER=true
AUG_MIRROR_PROB=0.0

# ---- optimization / loss ---------------------------------------------------
OPTIMIZER=muon
LR=${LR:-4e-4}
MUON_LR=null                 # = LR
MUON_UPDATE_SCALE=moonlight
MUON_MOMENTUM=0.95
MUON_POLAR_STEPS=6
WEIGHT_DECAY=1e-9
WARMUP_PCT=0.05
WARMUP_START_RATIO=0.1
FINAL_LR_RATIO=0.01
GRAD_CLIP=1.0
NUM_STEPS=${NUM_STEPS:-10000}
SSIM_LAMBDA=0.2
OPACITY_REG=0.01
SCALE_REG=0
SCALE_TAIL_REG=0
SCALE_TAIL_THRESHOLD=0.1     # only for the scale_num_big stat
SH_DEGREE_INTERVAL=2000
RANDOM_BKGD=true
FRUSTUM_CULL=${FRUSTUM_CULL:-subset}             # none | subset (cull the random subset) | full (cull all, then subset)
CULL_MARGIN=${CULL_MARGIN:-0.05}               # pixels on every side = this x max(width, height) (48 px)
POINTS_IN_WORKERS=${POINTS_IN_WORKERS:-true}   # draw / cull / gather the points in the loader workers (CPU)
BKGD_MODE=${BKGD_MODE:-pixel}                  # color (per view) | pixel
EVAL_BN_BATCH_STATS=false

# ---- eval / logging / checkpoints ------------------------------------------
EVAL_EVERY=${EVAL_EVERY:-500}
LPIPS_NET=alex
LOG_TRAIN_STEPS=5
LOG_TRAIN_IMAGES_STEPS=500
LOG_EVAL_IMAGES=2            # per eval scene; metrics use every eval view
SAVE_EVERY=2000

if [ "$TRAIN_ENVIRONMENT" != null ] || [ "$TRAIN_LIMIT" != null ]; then
    SCENES_TAG="dl3dv"; [ "$TRAIN_ENVIRONMENT" = null ] || SCENES_TAG="${SCENES_TAG}${TRAIN_ENVIRONMENT}"; [ "$TRAIN_LIMIT" = null ] || SCENES_TAG="${SCENES_TAG}${TRAIN_LIMIT}"
else
    SCENES_TAG="dl3dv120"
fi
[ "$TRAIN_ON" = eval_seen ] && SCENES_TAG="${SCENES_TAG/dl3dv120/dl3dv}seen10"
[ "$EVAL_UNSEEN_LIMIT" = null ] || SCENES_TAG="${SCENES_TAG}_unseen${EVAL_UNSEEN_LIMIT}"
HEAD_TAG="skip${HEAD_SKIP}"; [ "$HEAD_SKIP" = concat ] && HEAD_TAG="${HEAD_TAG}${HEAD_SKIP_DIM}"
HEAD_TAG="${HEAD_TAG}_h${HEAD_HIDDEN}"; [ "$COLOR_RESIDUAL" = true ] && HEAD_TAG="${HEAD_TAG}_rgbres"
[ "$VIEW_SAMPLING" = random ] || HEAD_TAG="${HEAD_TAG}_views${VIEW_SAMPLING}"
HEAD_TAG="${HEAD_TAG}_bkgd${BKGD_MODE}"; [ "$FRUSTUM_CULL" = none ] || HEAD_TAG="${HEAD_TAG}_cull${FRUSTUM_CULL}${CULL_MARGIN}"
SKY_TAG=sky4sig; [ "$SKY" = true ] || SKY_TAG=nosky
PE_TAG="R4sig"; [ "$POS_RADIUS" = sigma ] || PE_TAG="p${POS_PERCENTILE}"
RUN_NAME="022_${SKY_TAG}_pe${PE_TAG}_${HEAD_TAG}_${SCENES_TAG}_edgs$(( POINTS_BOUNDED / 1000 ))k-$(( POINTS_UNBOUNDED / 1000 ))k_v${TRAIN_BATCH_SIZE}_acc${GRAD_ACCUM}_lr${LR}_muon${MUON_UPDATE_SCALE}_off${MAX_OFFSET_VOXELS}vox_pe${POS_INJECT}F${POS_FREQS}noraw_maxs${MAX_SCALE}_dp${DROP_PATH}_${NUM_STEPS}_augmild_ln_sh${SH_DEGREE}every${SH_DEGREE_INTERVAL}"
OUTPUT_DIR="results/$RUN_NAME"
RESUME="${RESUME:-}"         # <output_dir>/ckpts/step_*.pt, or "latest" = newest in OUTPUT_DIR (if any)

WANDB_PROJECT=ffgsplat
WANDB_MODE=online

DEVICE=cuda
CUDA_MEMORY_FRACTION=1.0       # one 3g.71gb slice

ARGS=(
    "dl3dv.root=$DL3DV_ROOT"
    "dl3dv.split_file=$SPLIT_FILE"
    "dl3dv.points_bounded=$POINTS_BOUNDED"
    "dl3dv.points_unbounded=$POINTS_UNBOUNDED"
    "dl3dv.resample_points=$RESAMPLE_POINTS"
    "dl3dv.train_environment=$TRAIN_ENVIRONMENT"
    "dl3dv.train_limit=$TRAIN_LIMIT"
    "dl3dv.train_on=$TRAIN_ON"
    "dl3dv.eval_unseen_limit=$EVAL_UNSEEN_LIMIT"
    "dl3dv.lazy=$LAZY"
    "dl3dv.lazy_train_scenes=$LAZY_TRAIN_SCENES"
    "dl3dv.target_radius=$TARGET_RADIUS"
    "dl3dv.load_threads=$LOAD_THREADS"
    "dl3dv.points_in_workers=$POINTS_IN_WORKERS"

    "data.parser.data_dir=unused"
    "data.parser.factor=$FACTOR"
    "data.parser.normalize=$NORMALIZE"
    "data.parser.colmap_json=$COLMAP_JSON"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.train_view_sampling=$VIEW_SAMPLING"
    "data.train_num_workers=$NUM_WORKERS"
    "data.eval_batch_size=$EVAL_BATCH_SIZE"
    "data.eval_num_workers=$EVAL_NUM_WORKERS"

    "input.source=edgs"
    "input.edgs_dir=$EDGS_DIR"
    "input.features=[$FEATURES]"
    "input.drop_invalid=$DROP_INVALID"
    "input.sky=$SKY"
    "input.pos_radius=$POS_RADIUS"
    "input.pos_sigma_k=$POS_SIGMA_K"
    "input.seed=$POINT_SEED"
    "input.feature_mlp_dim=$FEATURE_MLP_DIM"
    "input.feature_mlp_layers=$FEATURE_MLP_LAYERS"
    "input.pos_encoding=$POS_ENCODING"
    "input.pos_freqs=$POS_FREQS"
    "input.pos_percentile=$POS_PERCENTILE"
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

if [ "${CHECK_ONLY:-false}" = true ]; then
    # Pre-flight check of the inputs with the same config (scripts/022_check_sky.py).
    PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH} python scripts/022_check_sky.py "${ARGS[@]}" "$@"
    exit
fi
python -m core.train_dl3dv "${ARGS[@]}" "$@"
