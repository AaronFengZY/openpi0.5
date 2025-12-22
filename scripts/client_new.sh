#!/bin/bash
set -e

# ================= 配置路径 =================
PROJECT_ROOT="/home/v-zhifeng/HPE/openpi"
DATA_ROOT="${PROJECT_ROOT}/data"

# 统计文件与索引
NORM_STATS_FILE="${PROJECT_ROOT}/assets/pi05_agiworld/agibot_full/dataset_stats_mp_q01q99_static.json"
INDEX_FILE="${DATA_ROOT}/episodic_dataset_fixed_static.npy"

# 相机与模型参数 (参考你提供的 FK 直接验证路径)
CAMERA_NPY="/home/v-zhifeng/HPE/v-zhifeng/agibot_beta_split_500/camera_param.npy"
URDF="/home/v-zhifeng/HPE/agirobot/G1/A2D_120s/A2D.urdf"

# 导出环境变量
export PYTHONPATH=$PYTHONPATH:"${PROJECT_ROOT}"
export NORM_STATS_FILE="$NORM_STATS_FILE"
export AGIBOT_INDEX_FILE="$INDEX_FILE"

echo "🚀 Launching Visual Client with Physics-based FK Projection..."

uv run scripts/client_new.py \
    --ws "ws://127.0.0.1:8001" \
    --dataset_root "$DATA_ROOT" \
    --norm_stats_file "$NORM_STATS_FILE" \
    --index_file "$INDEX_FILE" \
    --camera_npy "$CAMERA_NPY" \
    --urdf "$URDF"