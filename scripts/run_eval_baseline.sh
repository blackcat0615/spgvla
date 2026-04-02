#!/bin/bash

# 基础参数配置
POLICY_TYPE="lerobot"
POLICY_PATH="/root/lehome-challenge/outputs/train/smolvla_top_long/checkpoints/last/pretrained_model"
DATASET_ROOT="/root/gpufree-data/lehome/Datasets/example/four_types_merged"
NUM_EPISODES=5
DEVICE="cpu"
ENABLE_CAMERAS="--enable_cameras"   # 若要禁用摄像头，可删除此行或设为空字符串

# 定义四个场景
scenarios=("top_long" "top_short" "pant_long" "pant_short")

# 循环执行评测
for scenario in "${scenarios[@]}"; do
    echo "========================================"
    echo "Starting evaluation for scenario: $scenario"
    echo "========================================"
    python -m scripts.eval \
        --policy_type "$POLICY_TYPE" \
        --policy_path "$POLICY_PATH" \
        --garment_type "$scenario" \
        --dataset_root "$DATASET_ROOT" \
        --num_episodes "$NUM_EPISODES" \
        --device "$DEVICE" \
        $ENABLE_CAMERAS
    
    # 检查执行状态
    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for scenario $scenario"
        exit 1
    fi
    echo "Completed evaluation for scenario: $scenario"
    echo
done

echo "All scenarios evaluated successfully."