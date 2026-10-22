#!/usr/bin/env bash
set -euo pipefail

# 019: 018 (DL3DV, 120 training scenes, EDGS + RoMa, Muon, offsets, PE, mild aug)
# with a skip from the model input to the GS head and a per-pixel random background:
#   - HEAD_SKIP: x0 = [rgb | projected features | PE] (LitePT's input, before the
#     stem) reaches the head directly: "concat" = GELU(Linear(x0 -> HEAD_SKIP_DIM))
#     concatenated to the backbone output, "add" = zero-init Linear(x0) added to
#     it, "none" = 018;
#   - HEAD_HIDDEN: hidden layer (+ GELU) before the output linears (0 = linear head);
#   - COLOR_RESIDUAL: sh0 = (rgb_in - 0.5) / C0 + delta, every Gaussian starts at
#     its input colour instead of grey;
#   - BKGD_MODE: "pixel" = independent random colour per pixel behind every
#     training render (018: one colour per view, "color"). Eval stays on black;
#   - FRUSTUM_CULL (default none): only the input points inside the micro-batch's
#     8 views (+ CULL_MARGIN) go through the model; "subset" culls the 1M / 1.25M
#     random subset (faster), "full" culls all points, then draws the subset from
#     the visible ones (same budget, denser). PE radius from the unculled cloud;
#   - VIEW_SAMPLING (default random): "knn" = each micro-batch is the next view of
#     the scene's shuffled order + its 7 nearest training views by camera center
#     (overlapping views: frustum culling keeps much less);
#   - POINTS_IN_WORKERS (default true): the points are drawn / culled / gathered on
#     the CPU in the train loader workers, overlapped with the GPU step; the main
#     process only copies the kept points to the GPU (018 does it all serially in
#     the main process, ~226 MB gather + copy per micro-batch).
# Variants: 019a = HEAD_SKIP=concat HEAD_HIDDEN=128 COLOR_RESIDUAL=true BKGD_MODE=color,
#           019b = 019a + BKGD_MODE=pixel (the defaults).
# The new layers don't load 018 checkpoints: trained from scratch.
#
# HEAD_SKIP, HEAD_SKIP_DIM, HEAD_HIDDEN, COLOR_RESIDUAL, BKGD_MODE, FRUSTUM_CULL, CULL_MARGIN, VIEW_SAMPLING, POINTS_IN_WORKERS, LR, NUM_STEPS, GRAD_ACCUM, EVAL_EVERY, TRAIN_ENVIRONMENT / TRAIN_LIMIT, LAZY, POINTS_BOUNDED / POINTS_UNBOUNDED and RESUME
# can be set from the environment (the run name and output dir follow them).
# Any extra `key=value` args are forwarded as OmegaConf overrides, e.g.:
#   scripts/019_train_dl3dv_edgs_skip.sh trainer.num_steps=300 trainer.eval_every=100
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- scenes / data ---------------------------------------------------------
DL3DV_ROOT="$WORK/datasets/dl3dv"
SPLIT_FILE=data/splits/dl3dv_133.json
FACTOR=4                     # images_4 (960x540); the only resolution downloaded
NORMALIZE=true               # EDGS points are mapped into the parser's normalized frame
COLMAP_JSON=true             # read sparse/0/colmap.json (scripts/018_colmap_to_json.py), not the .bin files
TARGET_RADIUS=3.56           # scale every scene so its COLMAP points' r95 is this (bonsai's EDGS r95)

TRAIN_BATCH_SIZE=8           # views of one scene per micro-batch
VIEW_SAMPLING=${VIEW_SAMPLING:-random}         # random | knn (next shuffled view + its 7 nearest cameras)
GRAD_ACCUM=${GRAD_ACCUM:-8}  # micro-batches per optimizer step
EVAL_BATCH_SIZE=8
NUM_WORKERS=16               # train loader (job has 24 CPUs; the main process gathers the points)
EVAL_NUM_WORKERS=2           # per eval scene
LOAD_THREADS=8               # scenes loaded in parallel at startup

# ---- input points / features -----------------------------------------------
EDGS_DIR="$WORK/experiments/edgs_feat_dl3dv/{scene}"
FEATURES=rgb,roma1,roma2,roma4,roma16
POINTS_BOUNDED=${POINTS_BOUNDED:-1000000}
POINTS_UNBOUNDED=${POINTS_UNBOUNDED:-1250000}
RESAMPLE_POINTS=true         # fresh random subset per training micro-batch
TRAIN_ENVIRONMENT=${TRAIN_ENVIRONMENT:-null}   # bounded | unbounded | null (all): train on these scenes only
TRAIN_LIMIT=${TRAIN_LIMIT:-null}               # first N of them | null (all); with either set, eval_seen = the trained scenes
TRAIN_ON=${TRAIN_ON:-train}                    # train (120 scenes) | eval_seen (its 10: 5 bounded / 5 unbounded)
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
NUM_STEPS=${NUM_STEPS:-60000}
SSIM_LAMBDA=0.2
OPACITY_REG=0.01
SCALE_REG=0
SCALE_TAIL_REG=0
SCALE_TAIL_THRESHOLD=0.1     # only for the scale_num_big stat
SH_DEGREE_INTERVAL=2000
RANDOM_BKGD=true
FRUSTUM_CULL=${FRUSTUM_CULL:-none}             # none | subset (cull the random subset) | full (cull all, then subset)
CULL_MARGIN=${CULL_MARGIN:-0.05}               # pixels on every side = this x max(width, height) (48 px)
POINTS_IN_WORKERS=${POINTS_IN_WORKERS:-true}   # draw / cull / gather the points in the loader workers (CPU)
BKGD_MODE=${BKGD_MODE:-pixel}                  # color (per view) | pixel
EVAL_BN_BATCH_STATS=false

# ---- eval / logging / checkpoints ------------------------------------------
EVAL_EVERY=${EVAL_EVERY:-2000}
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
HEAD_TAG="skip${HEAD_SKIP}"; [ "$HEAD_SKIP" = concat ] && HEAD_TAG="${HEAD_TAG}${HEAD_SKIP_DIM}"
HEAD_TAG="${HEAD_TAG}_h${HEAD_HIDDEN}"; [ "$COLOR_RESIDUAL" = true ] && HEAD_TAG="${HEAD_TAG}_rgbres"
[ "$VIEW_SAMPLING" = random ] || HEAD_TAG="${HEAD_TAG}_views${VIEW_SAMPLING}"
HEAD_TAG="${HEAD_TAG}_bkgd${BKGD_MODE}"; [ "$FRUSTUM_CULL" = none ] || HEAD_TAG="${HEAD_TAG}_cull${FRUSTUM_CULL}${CULL_MARGIN}"
RUN_NAME="019_${HEAD_TAG}_${SCENES_TAG}_edgs$(( POINTS_BOUNDED / 1000 ))k-$(( POINTS_UNBOUNDED / 1000 ))k_v${TRAIN_BATCH_SIZE}_acc${GRAD_ACCUM}_lr${LR}_muon${MUON_UPDATE_SCALE}_off${MAX_OFFSET_VOXELS}vox_pe${POS_INJECT}F${POS_FREQS}p${POS_PERCENTILE}noraw_maxs${MAX_SCALE}_dp${DROP_PATH}_${NUM_STEPS}_augmild_ln_sh${SH_DEGREE}every${SH_DEGREE_INTERVAL}"
OUTPUT_DIR="results/$RUN_NAME"
RESUME="${RESUME:-}"         # <output_dir>/ckpts/step_*.pt, or "latest" = newest in OUTPUT_DIR (if any)

WANDB_PROJECT=ffgsplat
WANDB_MODE=online

DEVICE=cuda
CUDA_MEMORY_FRACTION=0.9

ARGS=(
    "dl3dv.root=$DL3DV_ROOT"
    "dl3dv.split_file=$SPLIT_FILE"
    "dl3dv.points_bounded=$POINTS_BOUNDED"
    "dl3dv.points_unbounded=$POINTS_UNBOUNDED"
    "dl3dv.resample_points=$RESAMPLE_POINTS"
    "dl3dv.train_environment=$TRAIN_ENVIRONMENT"
    "dl3dv.train_limit=$TRAIN_LIMIT"
    "dl3dv.train_on=$TRAIN_ON"
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

python -m core.train_dl3dv "${ARGS[@]}" "$@"
