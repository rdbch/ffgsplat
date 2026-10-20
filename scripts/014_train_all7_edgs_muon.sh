#!/usr/bin/env bash
set -euo pipefail

# 011 with the Muon optimizer and the 013 batch recipe: Muon (optimizer.name=muon,
# others/muon_optimizer.py, as 012/013: Muon on the hidden nn.Linear weights of
# LitePT / the feature MLP, AdamW on the rest, see Trainer._build_muon) and
# GRAD_ACCUM=8 x 8 views = 64 views per update at LR 4e-3 (shared by both groups
# via moonlight scaling), 30k steps, SH from 15k, on a full H200
# (jobs/ffgsplat_014.slurm). 011 (5vlrx5ul) used 32 views per update at LR 3e-3
# with AdamW, so it is a reference, not a matched baseline.
#
# 011: 010 on all seven Mip-NeRF 360 scenes: the four indoor ones (bonsai, counter,
# kitchen, room) at image factor 2 with 1M EDGS points each, and the three
# outdoor ones (bicycle, garden, stump) at factor 4 (as the MCMC baselines)
# with 1.5M points each (data.scene_parser / input.scene_max_points). EDGS
# points + RoMa features as input (as 009) and gradient accumulation
# (trainer.grad_accum_steps): every optimizer step averages the gradients of
# GRAD_ACCUM micro-batches, each TRAIN_BATCH_SIZE views of one random scene with
# its own random augmentation (4 x 8 = 32 views per update on one GPU; under
# torchrun every rank does that). Meant for one full H200 (jobs/ffgsplat_011.slurm). Otherwise the 008 recipe:
# LayerNorm LitePT, augmentation without x-mirror, MCMC opacity regularizer +
# scale-tail penalty, SH degree 1 switched on at SH_DEGREE_INTERVAL. Every scene
# is evaluated on its own val split (eval/<scene>/*, plus the mean), split
# across the ranks.
#
# Points come from $WORK/experiments/edgs_feat/<scene> (edgs_extract_ref.py: EDGS
# init with every 8th view held out, same split as data.parser.test_every=8):
# points that failed EDGS's reprojection test are dropped, then a fixed random
# MAX_POINTS subset is taken per scene (1M indoor, OUTDOOR_MAX_POINTS outdoor). At startup the loader checks that the EDGS cameras are
# the parser's training views and that the points project onto the pixels they
# were triangulated from (median error <= 3 px), i.e. same normalized frame.
#
# FEATURES picks the per-point input:
#   rgb                                base colour only, as in 003-008
#   rgb,roma1,roma2,roma4,roma16       (default) + RoMa features (VGG19 relu1_2/2_2/3_4 and
#                                      DINOv2; 8+16+32+48 = 104 PCA dims), each dim
#                                      standardized, then Linear(104,64)-GELU-
#                                      Linear(64,64), concatenated with rgb (67 ch
#                                      into LitePT)
#
# FEATURES, MAX_POINTS, TRAIN_BATCH_SIZE, GRAD_ACCUM, GRAD_CHECKPOINT, LR,
# NUM_STEPS, EVAL_EVERY, SH_DEGREE_INTERVAL, SCALE_TAIL_REG / SCALE_TAIL_THRESHOLD
# and RESUME can be set from the environment (the run name and output dir follow them), e.g.
#   FEATURES=rgb NUM_STEPS=20000 bash scripts/011_train_all7_edgs_gradaccum.sh
#
# Launch core/train.py. Run as `python -m core.train` from the repo root so
# `core`, `configs` and `data` all resolve as top-level packages.
#
# Any extra `key=value` args are forwarded as OmegaConf overrides and win over
# the values below, e.g.:
#   scripts/011_train_all7_edgs_gradaccum.sh trainer.num_steps=300 trainer.eval_every=100
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- scene / data ----------------------------------------------------------
SCENES="[bonsai,counter,kitchen,room,bicycle,garden,stump]"
SCENE_ROOT="$WORK/datasets/mipnerf"
FACTOR=2                     # image downscale factor (baseline: 2 indoor, 4 outdoor)
OUTDOOR_FACTOR=4
SCENE_PARSER="{bicycle: {factor: $OUTDOOR_FACTOR}, garden: {factor: $OUTDOOR_FACTOR}, stump: {factor: $OUTDOOR_FACTOR}}"
NORMALIZE=true               # EDGS points are mapped into the parser's normalized frame

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}   # views of one scene per micro-batch (per rank)
GRAD_ACCUM=${GRAD_ACCUM:-8}  # micro-batches per optimizer step (1 = no accumulation)
EVAL_BATCH_SIZE=8
NUM_WORKERS=2                # per rank (jobs get 2 CPUs, even on a full H200)

# ---- input points / features -----------------------------------------------
INPUT_SOURCE=edgs            # mcmc (003-008) | edgs
EDGS_DIR="$WORK/experiments/edgs_feat/{scene}"
FEATURES=${FEATURES:-rgb,roma1,roma2,roma4,roma16}   # comma separated: rgb, roma1, roma2, roma4, roma16
MAX_POINTS=${MAX_POINTS:-1000000}                  # indoor
OUTDOOR_MAX_POINTS=${OUTDOOR_MAX_POINTS:-1500000}  # outdoor
SCENE_MAX_POINTS="{bicycle: $OUTDOOR_MAX_POINTS, garden: $OUTDOOR_MAX_POINTS, stump: $OUTDOOR_MAX_POINTS}"
DROP_INVALID=true            # drop points that failed EDGS's reprojection test
POINT_SEED=0                 # seed of the MAX_POINTS subset
FEATURE_MLP_DIM=64           # image features -> MLP -> this many channels (+ rgb) into LitePT
FEATURE_MLP_LAYERS=2

# ---- model -----------------------------------------------------------------
POINT_GRID_SIZE=0.005        # LitePT voxel size, in normalized scene units
MODEL_NORM=layer             # stem/pool/unpool norm: batch (LitePT default) | layer (no running stats)
GRAD_CHECKPOINT=${GRAD_CHECKPOINT:-false}  # recompute LitePT block activations in backward (less memory, slower); off
SH_DEGREE=1                  # max SH degree predicted (degree 1: linear view dependence)
INIT_SCALE=0.01              # starting (post-activation) scale of every Gaussian
INIT_OPACITY=0.1
MAX_SCALE=1.0                # upper bound on predicted scales

# ---- augmentation ----------------------------------------------------------
AUGMENT=true
AUG_Z_ROT_DEG=180            # rotation about up (z), uniform in [-x, x]
AUG_TILT_DEG=5               # rotation about x and y, each uniform in [-x, x]
AUG_SCALE_MIN=0.8            # uniform scale, log-uniform in [min, max]
AUG_SCALE_MAX=1.25
AUG_VOXEL_JITTER=true        # random sub-voxel offset of the LitePT grid
AUG_MIRROR_PROB=0.0          # no mirroring

# ---- optimization / loss ---------------------------------------------------
OPTIMIZER=${OPTIMIZER:-muon}  # muon | adamw
LR=${LR:-4e-3}               # peak LR (AdamW groups; Muon groups too unless MUON_LR), reached at the end of warmup (64 views per update, as 010/013)
MUON_LR=${MUON_LR:-null}     # null = LR (fine with moonlight scaling: Muon update RMS ~0.2 x lr, like AdamW)
MUON_UPDATE_SCALE=${MUON_UPDATE_SCALE:-moonlight}   # moonlight | keller | none
MUON_MOMENTUM=0.95
MUON_POLAR_STEPS=6
WEIGHT_DECAY=1e-9            # minimal; 0 disables
WARMUP_PCT=0.05              # fraction of NUM_STEPS spent in linear warmup
WARMUP_START_RATIO=0.1       # warmup starts at this fraction of LR
FINAL_LR_RATIO=0.01          # cosine decay ends at this fraction of LR
GRAD_CLIP=1.0                # max grad norm; 0 disables
# Optimizer steps (32 views each); set from the smoke test's step time to fit
# the 48 h limit.
NUM_STEPS=${NUM_STEPS:-30000}
SSIM_LAMBDA=0.2
OPACITY_REG=0.01             # MCMC's mean-opacity penalty (gsplat MCMC preset: 0.01); 0 disables
SCALE_REG=0                  # MCMC's mean-scale penalty; off, see SCALE_TAIL_REG
SCALE_TAIL_REG=${SCALE_TAIL_REG:-0.01}              # penalize only scales above the threshold
SCALE_TAIL_THRESHOLD=${SCALE_TAIL_THRESHOLD:-0.3}   # 009: 0.1, which pinned scale_max there
SH_DEGREE_INTERVAL=${SH_DEGREE_INTERVAL:-$(( NUM_STEPS / 2 ))}   # with SH_DEGREE=1: the step SH switches on (half way, as 009)
RANDOM_BKGD=true             # random background in training (eval is always black)
EVAL_BN_BATCH_STATS=false    # no-op with MODEL_NORM=layer

# ---- eval / logging / checkpoints ------------------------------------------
EVAL_EVERY=${EVAL_EVERY:-2000}   # 7 scenes, ~300 eval views
LPIPS_NET=alex                # alex | vgg
LOG_TRAIN_STEPS=5
LOG_TRAIN_IMAGES_STEPS=500
LOG_EVAL_IMAGES=8            # first N eval views logged as images; -1 = all
SAVE_EVERY=1000

if [ "$SH_DEGREE_INTERVAL" -lt "$NUM_STEPS" ]; then SH_TAG="sh${SH_DEGREE}from${SH_DEGREE_INTERVAL}"; else SH_TAG=sh0; fi
if [ "$SCALE_TAIL_REG" != "0" ]; then TAIL_TAG="_tail${SCALE_TAIL_REG}at${SCALE_TAIL_THRESHOLD}"; else TAIL_TAG=""; fi
POINTS_TAG="edgs$(( MAX_POINTS / 1000 ))k-$(( OUTDOOR_MAX_POINTS / 1000 ))k"
FEAT_TAG=$(echo "$FEATURES" | tr ',' '+')
if [ "$GRAD_CHECKPOINT" = true ]; then GC_TAG="_gc"; else GC_TAG=""; fi
if [ "$GRAD_ACCUM" != 1 ]; then ACC_TAG="_acc${GRAD_ACCUM}"; else ACC_TAG=""; fi
NGPUS=${WORLD_SIZE:-1}       # set by torchrun
if [ "$NGPUS" != 1 ]; then DDP_TAG="_ddp${NGPUS}"; else DDP_TAG=""; fi
OPT_TAG=""
if [ "$OPTIMIZER" = muon ]; then OPT_TAG="_muon${MUON_UPDATE_SCALE}"; fi
if [ "$OPTIMIZER" = muon ] && [ "$MUON_LR" != null ]; then OPT_TAG="${OPT_TAG}${MUON_LR}"; fi
RUN_NAME="all7_${POINTS_TAG}_${FEAT_TAG}_v${TRAIN_BATCH_SIZE}${ACC_TAG}${DDP_TAG}_gs${POINT_GRID_SIZE}_lr${LR}${OPT_TAG}_${NUM_STEPS}_aug_nomirror_ln_${SH_TAG}${TAIL_TAG}${GC_TAG}"
OUTPUT_DIR="results/$RUN_NAME"
RESUME="${RESUME:-}"         # path to <output_dir>/ckpts/step_*.pt to resume from; "latest" = newest in OUTPUT_DIR (if any)

WANDB_PROJECT=ffgsplat
WANDB_MODE=online            # online | offline | disabled

DEVICE=cuda
CUDA_MEMORY_FRACTION=0.9

ARGS=(
    "data.scenes=$SCENES"
    "data.scene_root=$SCENE_ROOT"
    "data.scene_parser=$SCENE_PARSER"
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
    "input.scene_max_points=$SCENE_MAX_POINTS"
    "input.drop_invalid=$DROP_INVALID"
    "input.seed=$POINT_SEED"
    "input.feature_mlp_dim=$FEATURE_MLP_DIM"
    "input.feature_mlp_layers=$FEATURE_MLP_LAYERS"

    "model.norm=$MODEL_NORM"

    "head.sh_degree=$SH_DEGREE"
    "head.init_scale=$INIT_SCALE"
    "head.init_opacity=$INIT_OPACITY"
    "head.max_scale=$MAX_SCALE"

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
    "trainer.grad_checkpoint=$GRAD_CHECKPOINT"
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
