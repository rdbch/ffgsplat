#!/usr/bin/env bash
set -euo pipefail

# 017: 016 at its best setting (job 513: sin/cos PE without raw p, drop-path 0,
# offsets 1.5 vox; 22.59 dB at 20k) with three changes, to test whether the
# Gaussians' size bound and the missing view dependence are what limit the fit
# (all bonsai runs so far: train PSNR ~ eval PSNR + 0.4, i.e. underfitting):
#   - larger Gaussians: hard bound head.max_scale 1.0 -> 0.4 and the scale-tail
#     penalty off (SCALE_TAIL_REG=0). In 012-016 the penalty (0.01 * sum of the
#     excess over 0.1) pinned every scale at <= 0.1; now 0.4 is the only bound.
#     scale_num_big still counts Gaussians above the old 0.1 (SCALE_TAIL_THRESHOLD);
#   - SH degree 3 (was 1), one degree every SH_DEGREE_INTERVAL=1000 steps
#     (degree 3 from step 3k, gsplat's schedule; was degree 1 from 10k);
#   - no augmentation (AUGMENT=false: no rotation / tilt / scale, no voxel jitter).
#
# 016: 015 + Fourier positional encoding of the input points (input.pos_encoding=fourier,
# models/pos_encoding.py). Everything else is 015 (bonsai, EDGS + RoMa, Muon, offsets
# 1.5 vox; job 498), so the curves compare directly. POS_ENCODING=none reproduces 015.
#
# Centering / normalization: p = coord / r, centered at the scene origin (the
# normalized frame's origin = median of the COLMAP points; augmentation rotates /
# scales about it), one divider r for all axes = the POS_PERCENTILE-th percentile of
# the points' distances to the origin, computed after augmentation (so p rotates
# with the cloud and ignores the augmentation's scale). Bonsai: r95 = 3.56 units,
# 0.22% of points beyond 1.5 r. Encoding: sin / cos(2^k pi p), k < POS_FREQS, plus p
# (POS_INCLUDE_RAW): 6 * POS_FREQS + 3 channels (27 for F=4).
# POS_INJECT: input = concatenated after the feature MLP (LitePT in_channels
# 67 -> 94 for F=4) | embed = added after LitePT's stem through a zero-initialized
# Linear (starts exactly as 015) | both.
#
# 015: 012 + predicted position offsets (head.max_offset > 0): the model also predicts
# a per-point displacement, bounded by max_offset * tanh, added to every input
# point's position; zero-initialized, so it starts exactly as 012 (job 491,
# 22.07 dB at 20k). MAX_OFFSET_VOXELS sets the bound in LitePT voxels
# (POINT_GRID_SIZE); 0 disables the offset head (= 012).
#
# 012: 009 with the Muon optimizer (optimizer.name=muon, others/muon_optimizer.py):
# Muon on the hidden nn.Linear weights of LitePT / the feature MLP, AdamW on the
# rest (see Trainer._build_muon). Everything else is the 009 rgb+RoMa run
# (bonsai, 21.49 dB at 20k), so the curves compare directly. OPTIMIZER=adamw
# reproduces 009.
#
# 009: bonsai, single scene, with EDGS points as the model input instead of the MCMC
# reference Gaussians' means (input.source=edgs, data/edgs_points.py). Otherwise
# the 008 recipe: LayerNorm LitePT, augmentation without x-mirror, MCMC opacity
# regularizer + scale-tail penalty, SH degree 1 switched on at SH_DEGREE_INTERVAL.
#
# Points come from $WORK/experiments/edgs_feat/bonsai (edgs_extract_ref.py: EDGS
# init with every 8th view held out, same split as data.parser.test_every=8):
# points that failed EDGS's reprojection test are dropped, then a fixed random
# MAX_POINTS subset is taken (default 1M = the MCMC bonsai baseline's cap_max, so
# Gaussian counts match). At startup the loader checks that the EDGS cameras are
# the parser's training views and that the points project onto the pixels they
# were triangulated from (median error <= 3 px), i.e. same normalized frame.
#
# FEATURES picks the per-point input:
#   rgb                                (default) base colour only, as in 003-008
#   rgb,roma1,roma2,roma4,roma16       + RoMa features (VGG19 relu1_2/2_2/3_4 and
#                                      DINOv2; 8+16+32+48 = 104 PCA dims), each dim
#                                      standardized, then Linear(104,64)-GELU-
#                                      Linear(64,64), concatenated with rgb (67 ch
#                                      into LitePT)
#
# FEATURES, MAX_POINTS, LR, NUM_STEPS, EVAL_EVERY, SH_DEGREE_INTERVAL,
# SCALE_TAIL_REG / SCALE_TAIL_THRESHOLD and RESUME can be set from the
# environment (the run name and output dir follow them), e.g.
#   FEATURES=rgb,roma1,roma2,roma4,roma16 bash scripts/009_train_edgs_augment_ln.sh
#
# Launch core/train.py. Run as `python -m core.train` from the repo root so
# `core`, `configs` and `data` all resolve as top-level packages.
#
# Any extra `key=value` args are forwarded as OmegaConf overrides and win over
# the values below, e.g.:
#   scripts/017_train_edgs_bigscale_sh3_noaug.sh trainer.num_steps=300 trainer.eval_every=100
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- scene / data ----------------------------------------------------------
SCENE=bonsai
DATA_DIR="$WORK/datasets/mipnerf/$SCENE"
FACTOR=2                     # image downscale factor (baseline: 2 indoor, 4 outdoor)
NORMALIZE=true               # EDGS points are mapped into the parser's normalized frame

TRAIN_BATCH_SIZE=8           # views rendered per step
EVAL_BATCH_SIZE=8
NUM_WORKERS=8                # jobs request 12 CPUs

# ---- input points / features -----------------------------------------------
INPUT_SOURCE=edgs            # mcmc (003-008) | edgs
EDGS_DIR="$WORK/experiments/edgs_feat/{scene}"
FEATURES=${FEATURES:-rgb,roma1,roma2,roma4,roma16}   # comma separated: rgb, roma1, roma2, roma4, roma16
MAX_POINTS=${MAX_POINTS:-1000000}
DROP_INVALID=true            # drop points that failed EDGS's reprojection test
POINT_SEED=0                 # seed of the MAX_POINTS subset
FEATURE_MLP_DIM=64           # image features -> MLP -> this many channels (+ rgb) into LitePT
FEATURE_MLP_LAYERS=2
POS_ENCODING=${POS_ENCODING:-fourier}   # none (= 015) | fourier
POS_FREQS=${POS_FREQS:-4}               # frequencies 2^k pi, k < POS_FREQS (periods 2, 1, 0.5, 0.25 in units of r)
POS_PERCENTILE=${POS_PERCENTILE:-95}    # r = this percentile of |coord| (after augmentation)
POS_INCLUDE_RAW=${POS_INCLUDE_RAW:-false}  # also feed p itself (3 ch); 017: off (513)
POS_INJECT=${POS_INJECT:-input}         # input | embed | both
POS_NORM=${POS_NORM:-none}              # none | layer (per-point LayerNorm over the PE channels)
POS_CLAMP=${POS_CLAMP:-0}               # radial clamp of the tail at this percentile of |coord|; 0 = off

# ---- model -----------------------------------------------------------------
POINT_GRID_SIZE=0.005        # LitePT voxel size, in normalized scene units
MODEL_NORM=layer             # stem/pool/unpool norm: batch (LitePT default) | layer (no running stats)
DROP_PATH=${DROP_PATH:-0.0}  # LitePT stochastic depth; 0.3 (default) collapses eval with PE (504-511)
SH_DEGREE=${SH_DEGREE:-3}    # max SH degree predicted (017: 3, was 1)
INIT_SCALE=0.01              # starting (post-activation) scale of every Gaussian
INIT_OPACITY=0.1
MAX_SCALE=${MAX_SCALE:-0.4}  # hard upper bound on predicted scales (017: 0.4)
MAX_OFFSET_VOXELS=${MAX_OFFSET_VOXELS:-1.5}         # bound on predicted position offsets, in voxels; 0 = none
MAX_OFFSET=$(awk "BEGIN {print $MAX_OFFSET_VOXELS * $POINT_GRID_SIZE}")   # world units

# ---- augmentation ----------------------------------------------------------
AUGMENT=${AUGMENT:-false}    # 017: off (the AUG_* below are unused)
AUG_Z_ROT_DEG=180            # rotation about up (z), uniform in [-x, x]
AUG_TILT_DEG=5               # rotation about x and y, each uniform in [-x, x]
AUG_SCALE_MIN=0.8            # uniform scale, log-uniform in [min, max]
AUG_SCALE_MAX=1.25
AUG_VOXEL_JITTER=true        # random sub-voxel offset of the LitePT grid
AUG_MIRROR_PROB=0.0          # no mirroring

# ---- optimization / loss ---------------------------------------------------
OPTIMIZER=${OPTIMIZER:-muon}  # muon | adamw (= 009)
LR=${LR:-2e-3}               # peak LR (AdamW groups; Muon groups too unless MUON_LR), reached at the end of warmup
MUON_LR=${MUON_LR:-null}     # null = LR (fine with moonlight scaling: Muon update RMS ~0.2 x lr, like AdamW)
MUON_UPDATE_SCALE=${MUON_UPDATE_SCALE:-moonlight}   # moonlight | keller | none
MUON_MOMENTUM=0.95
MUON_POLAR_STEPS=6
WEIGHT_DECAY=1e-9            # minimal; 0 disables
WARMUP_PCT=0.05              # fraction of NUM_STEPS spent in linear warmup
WARMUP_START_RATIO=0.1       # warmup starts at this fraction of LR
FINAL_LR_RATIO=0.01          # cosine decay ends at this fraction of LR
GRAD_CLIP=1.0                # max grad norm; 0 disables
NUM_STEPS=${NUM_STEPS:-20000}
SSIM_LAMBDA=0.2
OPACITY_REG=0.01             # MCMC's mean-opacity penalty (gsplat MCMC preset: 0.01); 0 disables
SCALE_REG=0                  # MCMC's mean-scale penalty; off, see SCALE_TAIL_REG
SCALE_TAIL_REG=${SCALE_TAIL_REG:-0}              # penalize only scales above the threshold
SCALE_TAIL_THRESHOLD=${SCALE_TAIL_THRESHOLD:-0.1}
SH_DEGREE_INTERVAL=${SH_DEGREE_INTERVAL:-1000}      # SH degree = min(step // interval, SH_DEGREE)
RANDOM_BKGD=true             # random background in training (eval is always black)
EVAL_BN_BATCH_STATS=false    # no-op with MODEL_NORM=layer

# ---- eval / logging / checkpoints ------------------------------------------
EVAL_EVERY=${EVAL_EVERY:-1000}
LPIPS_NET=alex                # alex | vgg
LOG_TRAIN_STEPS=5
LOG_TRAIN_IMAGES_STEPS=500
LOG_EVAL_IMAGES=8            # first N eval views logged as images; -1 = all
SAVE_EVERY=1000

if [ "$SH_DEGREE_INTERVAL" -lt "$NUM_STEPS" ]; then SH_TAG="sh${SH_DEGREE}every${SH_DEGREE_INTERVAL}"; else SH_TAG=sh0; fi
if [ "$AUGMENT" = true ]; then AUG_TAG=aug_nomirror; else AUG_TAG=noaug; fi
if [ "$SCALE_TAIL_REG" != "0" ]; then TAIL_TAG="_tail${SCALE_TAIL_REG}at${SCALE_TAIL_THRESHOLD}"; else TAIL_TAG=""; fi
POINTS_TAG="edgs$(( MAX_POINTS / 1000 ))k"
FEAT_TAG=$(echo "$FEATURES" | tr ',' '+')
OPT_TAG=""
if [ "$OPTIMIZER" = muon ]; then OPT_TAG="_muon${MUON_UPDATE_SCALE}"; fi
if [ "$OPTIMIZER" = muon ] && [ "$MUON_LR" != null ]; then OPT_TAG="${OPT_TAG}${MUON_LR}"; fi
OFF_TAG=""
if [ "$MAX_OFFSET_VOXELS" != "0" ]; then OFF_TAG="_off${MAX_OFFSET_VOXELS}vox"; fi
PE_TAG=""
if [ "$POS_ENCODING" != none ]; then PE_TAG="_pe${POS_INJECT}F${POS_FREQS}p${POS_PERCENTILE}"; [ "$POS_INCLUDE_RAW" = true ] || PE_TAG="${PE_TAG}noraw"; [ "$POS_NORM" = none ] || PE_TAG="${PE_TAG}${POS_NORM}"; [ "$POS_CLAMP" = 0 ] || PE_TAG="${PE_TAG}c${POS_CLAMP}"; fi
RUN_NAME="${SCENE}_${POINTS_TAG}_${FEAT_TAG}_gs${POINT_GRID_SIZE}_lr${LR}${OPT_TAG}${OFF_TAG}${PE_TAG}_maxs${MAX_SCALE}_dp${DROP_PATH}_${NUM_STEPS}_${AUG_TAG}_ln_${SH_TAG}${TAIL_TAG}"
OUTPUT_DIR="results/$RUN_NAME"
RESUME="${RESUME:-}"         # path to <output_dir>/ckpts/step_*.pt to resume from; "latest" = newest in OUTPUT_DIR (if any)

WANDB_PROJECT=ffgsplat
WANDB_MODE=online            # online | offline | disabled

DEVICE=cuda
CUDA_MEMORY_FRACTION=0.9

ARGS=(
    "data.parser.data_dir=$DATA_DIR"
    "data.parser.factor=$FACTOR"
    "data.parser.normalize=$NORMALIZE"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.train_num_workers=$NUM_WORKERS"
    "data.eval_batch_size=$EVAL_BATCH_SIZE"
    "data.eval_num_workers=$NUM_WORKERS"

    "input.source=$INPUT_SOURCE"
    "input.edgs_dir=$EDGS_DIR"
    "input.features=[$FEATURES]"
    "input.max_points=$MAX_POINTS"
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
    "trainer.num_steps=$NUM_STEPS"
    "trainer.point_grid_size=$POINT_GRID_SIZE"
    "trainer.ssim_lambda=$SSIM_LAMBDA"
    "trainer.opacity_reg=$OPACITY_REG"
    "trainer.scale_reg=$SCALE_REG"
    "trainer.scale_tail_reg=$SCALE_TAIL_REG"
    "trainer.scale_tail_threshold=$SCALE_TAIL_THRESHOLD"
    "trainer.sh_degree_interval=$SH_DEGREE_INTERVAL"
    "trainer.random_bkgd=$RANDOM_BKGD"
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
    RESUME=$(ls "$OUTPUT_DIR"/ckpts/step_*.pt 2>/dev/null | sort | tail -n 1)
    echo "Resuming from: ${RESUME:-nothing (fresh start)}"
fi
if [ -n "$RESUME" ]; then
    ARGS+=("trainer.resume=$RESUME")
fi

python -m core.train "${ARGS[@]}" "$@"
