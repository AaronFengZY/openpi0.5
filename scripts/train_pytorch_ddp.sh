#!/bin/bash

# =================================================================
# 1. 基础配置 (Configuration)
# =================================================================

# 项目源码路径 (请确保指向 openpi 根目录)
export WORKSPACE_DIR="/home/v-zhifeng/HPE/openpi"

# 数据集根目录 (agibot_dataset.py 会读取这个环境变量)
export AGIBOT_DATA_ROOT="/home/v-zhifeng/HPE/v-zhifeng/agibot_beta_split_500"

# 统计文件路径 (Dataset 内部归一化需要，假设你已经放在了 dataset 目录下，或者你可以在 dataset 代码里写死)
# 这里的路径对应你之前提到的 .json
export NORM_STATS_FILE="/home/v-zhifeng/HPE/openpi/assets/pi05_agiworld/agibotworld_lerobot_test/norm_stats.json"

# Python 路径设置 (确保能 import openpi)
export PYTHONPATH="${WORKSPACE_DIR}/src:$PYTHONPATH"

# =================================================================
# 2. 训练参数 (Training Args)
# =================================================================

EXP_NAME="pi05_agibot_fsdp_run1"
CONFIG_NAME="pi05_agiworld"

# 显卡配置
NUM_NODES=1
NUM_GPUS_PER_NODE=4  # 请根据实际显卡数量修改 (例如 4 或 8)
MASTER_PORT=29500

# =================================================================
# 3. 性能与调试优化 (Optimization)
# =================================================================

# 显存优化: 减少 FSDP 训练时的显存碎片
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# WandB 设置: offline 模式防止网络问题中断训练
export WANDB_MODE="offline"
# export WANDB_PROJECT="openpi_agibot"

# 忽略一些无关紧要的警告
export TORCH_DISTRIBUTED_DEBUG="INFO" 

# =================================================================
# 4. 启动命令 (Launch)
# =================================================================

echo ">>> Starting Training..."
echo "Data Root: $AGIBOT_DATA_ROOT"
echo "Config: $CONFIG_NAME"
echo "GPUs: $NUM_GPUS_PER_NODE"

torchrun \
    --standalone \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    scripts/train_pytorch_ddp.py \
    $CONFIG_NAME \
    --exp_name $EXP_NAME \
    --save_interval 2000