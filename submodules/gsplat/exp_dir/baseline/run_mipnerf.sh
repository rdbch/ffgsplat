SCENE_DIR="$WORK/datasets/mipnerf"
RESULT_DIR="$WORK/experiments/mcmc_baselines"
RENDER_TRAJ_PATH="ellipse"

INDOOR_SCENES=(
    "bonsai"  # already run
    # "counter"
    # "kitchen"
    "room"
)

OUTDOOR_SCENES=(
    "bicycle"
    "flowers"
    "garden"
    "stump"
    "treehill"
)

for SCENE in "${INDOOR_SCENES[@]}"; do
    DATA_FACTOR=2
    CAP_MAX=1000000
    echo "Running $SCENE (indoor, factor=$DATA_FACTOR, cap_max=$CAP_MAX)"

    CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer.py mcmc --eval_steps 30000 --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

done

for SCENE in "${OUTDOOR_SCENES[@]}"; do
    DATA_FACTOR=4
    CAP_MAX=1500000
    echo "Running $SCENE (outdoor, factor=$DATA_FACTOR, cap_max=$CAP_MAX)"

    CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer.py mcmc --eval_steps 30000 --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

done
