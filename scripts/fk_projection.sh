#!/bin/bash
set -e

# ================= 配置路径 =================
DATA_ROOT="/home/v-zhifeng/HPE/openpi/data"
# 注意：确保这个路径是正确的，指向那个 70w+ 数据的 npy 文件
# ▼▼▼▼▼▼▼▼▼▼ [修改] 拼接完整路径 ▼▼▼▼▼▼▼▼▼▼
INDEX_FILE="${DATA_ROOT}/episodic_dataset_fixed_static.npy"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
CAMERA_NPY="/home/v-zhifeng/HPE/v-zhifeng/agibot_beta_split_500/camera_param.npy"
URDF="/home/v-zhifeng/HPE/agirobot/G1/A2D_120s/A2D.urdf"
OUTPUT_DIR="./outputs_fk_vis"

# ================= 目标索引 =================
# 留空 ""  => 随机选择
# 填数字 => 指定测试 (例如 750134)
TARGET_INDEX=""

# ================= 运行 =================
echo "🚀 Starting FK Verification (Indexed + Direct Mode)..."
export PYTHONPATH=$PYTHONPATH:. 

CMD_ARGS=(
  --dataset_root "$DATA_ROOT"
  --index_file "$INDEX_FILE"
  --camera_npy "$CAMERA_NPY"
  --urdf "$URDF"
  --output_dir "$OUTPUT_DIR"
)

if [ -n "$TARGET_INDEX" ]; then
  echo "🎯 Using specific index: $TARGET_INDEX"
  CMD_ARGS+=(--index "$TARGET_INDEX")
else
  echo "🎲 Using RANDOM selection mode"
fi

uv run fk_projection.py "${CMD_ARGS[@]}"