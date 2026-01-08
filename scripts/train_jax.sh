#!/bin/bash

# ==============================================================================
# Script Name: Train Pi05 Agibot (JAX/FSDP)
# Description: Sets up environment variables and launches JAX distributed training.
# Usage:       ./train.sh [BATCH_SIZE]
# Example:     ./train.sh 64
# ==============================================================================

# Stop on error
set -e

# ==============================================================================
# [1] Configuration: Hardware & Hyperparameters
# ==============================================================================

# --- GPU Setup ---
# 1. Detect Local GPUs (per node)
DETECTED_LOCAL_GPU_COUNT=$(nvidia-smi -L | wc -l)

# 2. Detect Node Count
# AMLT/MPI sets WORLD_SIZE. If not present, default to 1 (Single Node).
NODE_COUNT=${WORLD_SIZE:-1}

# 3. Calculate Global GPU Count
# Single Node: 8 * 1 = 8
# Multi Node:  8 * 2 = 16
AUTO_TOTAL_GPU_COUNT=$((DETECTED_LOCAL_GPU_COUNT * NODE_COUNT))

# Allow override via env for debugging, otherwise use auto-calculated value
TOTAL_GPU_COUNT=${GPU_COUNT_OVERRIDE:-$AUTO_TOTAL_GPU_COUNT}
LOCAL_GPU_COUNT=$DETECTED_LOCAL_GPU_COUNT  # 8

echo "🖥️ Detected Local GPU Count : $DETECTED_LOCAL_GPU_COUNT"

export JAX_PLATFORMS=cuda

# --- Training Parameters ---
BATCH_SIZE=${1:-64}
ACTION=${2:-"resume"}  # Default action is resume

# [修改] 直接赋值，不再从参数读取
DOWNSAMPLE=1

RESUME_EXP_NAME=${2:-""}
NUM_WORKERS=64

WARMUP_STEPS=1000
PEAK_LR=5e-5
DECAY_LR=1e-5
DECAY_STEPS=1000000  # 通常设为跟 NUM_TRAIN_STEPS 一样，或者更长

NUM_TRAIN_STEPS=1000000
SAVE_INTERVAL=10000
KEEP_PERIOD=2

# Validation: Batch size must be divisible by GLOBAL device count
if (( BATCH_SIZE % TOTAL_GPU_COUNT != 0 )); then
    echo "❌ Error: Batch Size ($BATCH_SIZE) must be divisible by Total Global GPU Count ($TOTAL_GPU_COUNT)."
    echo "   (Nodes: $NODE_COUNT, Local GPUs: $DETECTED_LOCAL_GPU_COUNT)"
    exit 1
fi

# ==============================================================================
# [2] Configuration: Paths & Environment
# ==============================================================================

# Locate script and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set Python Path to include project root
export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT"

# --- Dataset Paths ---
# Define where the raw data and statistics file are located
export AGIBOT_DATA_ROOT="$PROJECT_ROOT/data"
export NORM_STATS_FILE="$PROJECT_ROOT/assets/pi05_agiworld/agibot_full/dataset_stats_mp_q01q99_static.json"
export AGIBOT_INDEX_FILE="episodic_dataset_fixed_static.npy"

# [修改] 将硬编码的 DOWNSAMPLE 导出给 Python
export AGIBOT_DOWNSAMPLE_RATE="$DOWNSAMPLE"

echo "⏰ FORCED JAX Timeout: $JAX_COORDINATION_SERVICE_TIMEOUT_SEC seconds"
echo "⏭️  Downsample Rate : $AGIBOT_DOWNSAMPLE_RATE (Fixed)"

# ==============================================================================
# [3] Configuration: Logging & Experiments
# ==============================================================================

CONFIG_NAME="pi05_agiworld"
TIMESTAMP=$(date +%m%d_%H%M)


# ---------------------------------------------------------------------
# [关键修改] Resume 逻辑判断
# ---------------------------------------------------------------------
# --- Experiment Naming ---
if [[ -z "$EXP_NAME_ENV" ]] || [[ "$EXP_NAME_ENV" == "default" ]]; then
    BASE_NAME="pi05_fsdp_run_${TIMESTAMP}"
else
    BASE_NAME="$EXP_NAME_ENV"
fi
SUFFIX="_bs${BATCH_SIZE}_downsample${DOWNSAMPLE}_pk${PEAK_LR}_decay${DECAY_LR}_iter${NUM_TRAIN_STEPS}"
EXP_NAME="${BASE_NAME}${SUFFIX}"

# --- Logic Mapping ---
# We map the string intent to specific flags for the Python script
if [[ "$ACTION" == "overwrite" ]]; then
    echo "⚠️  ACTION: OVERWRITE - Forcing a fresh start."
    MODE_FLAGS="--overwrite --no-resume"
elif [[ "$ACTION" == "resume" ]]; then
    echo "🔄 ACTION: RESUME - Defaulting to continuation if data exists."
    MODE_FLAGS="--resume --no-overwrite"
else
    echo "❌ Error: Unknown action '$ACTION'. Use 'resume' or 'overwrite'."
    exit 1
fi

# --- WandB Naming Strategy ---
# Format: Pi05_Agibot_BS{BatchSize}_{Date_Time}
CUSTOM_WANDB_NAME="Pi05_Agibot_BS${BATCH_SIZE}_${TIMESTAMP}"

# Export to ENV so Python script (train_jax.py) can read it
export WANDB_RUN_NAME="$EXP_NAME"

# ==============================================================================
# [4] Execution Summary
# ==============================================================================

echo "================================================================"
echo "🚀 Launching Training Job"
echo "================================================================"
echo "📂 Project Root : $PROJECT_ROOT"
echo "📂 Data Root    : $AGIBOT_DATA_ROOT"
echo "📊 Stats File   : $NORM_STATS_FILE"
echo "📑 Index File   : $AGIBOT_INDEX_FILE"  # <--- 打印出来确认一下
echo "----------------------------------------------------------------"
echo "🔧 Config Name  : $CONFIG_NAME"
echo "🏷️  Exp Name     : $EXP_NAME"
echo "⏭️  Downsample   : $DOWNSAMPLE (Fixed)"
echo "🔢 MODE_FLAGS   : $MODE_FLAGS"
echo "----------------------------------------------------------------"
echo "🖥️  GPUs Used    : $GPU_COUNT (IDs: $CUDA_VISIBLE_DEVICES)"
echo "📦 Batch Size   : $BATCH_SIZE"
echo "🧵 Num Workers  : $NUM_WORKERS"
echo "🏃 Total Steps  : $NUM_TRAIN_STEPS"
echo "📈 LR Schedule  : Warmup=$WARMUP_STEPS | Peak=$PEAK_LR | End=$DECAY_LR"
echo "💾 Save Interval: Every $SAVE_INTERVAL steps"
echo "♻️  Keep Last    : $KEEP_PERIOD checkpoints"
echo "----------------------------------------------------------------"
echo "📈 WandB Name   : $WANDB_RUN_NAME"
echo "================================================================"

# ==============================================================================
echo "🔧 Patching OpenPI DataLoader to allow multi-node..."

TARGET_FILE="$PROJECT_ROOT/src/openpi/training/data_loader.py"

# 检查文件是否存在，防止路径错误
if [ -f "$TARGET_FILE" ]; then
    sed -i 's/raise NotImplementedError("Data loading with multiple processes is not supported.")/pass # Hotfix for multi-node/g' "$TARGET_FILE"
    echo "✅ Successfully patched data_loader.py"
else
    echo "⚠️ Warning: Could not find data_loader.py at $TARGET_FILE. Check path if error persists."
fi


# Added --num_train_steps argument
uv run "$SCRIPT_DIR/train_jax.py" "$CONFIG_NAME" \
    --exp_name "$EXP_NAME" \
    --fsdp_devices "$LOCAL_GPU_COUNT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_train_steps "$NUM_TRAIN_STEPS" \
    --save_interval "$SAVE_INTERVAL" \
    --keep_period "$KEEP_PERIOD" \
    $MODE_FLAGS \
    --wandb_enabled \
    --log_interval 100 \
    --lr_schedule.warmup_steps "$WARMUP_STEPS" \
    --lr_schedule.peak_lr "$PEAK_LR" \
    --lr_schedule.decay_lr "$DECAY_LR" \
    --lr_schedule.decay_steps "$DECAY_STEPS"