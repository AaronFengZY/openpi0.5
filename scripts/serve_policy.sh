# 首先设置 PYTHONPATH，确保能找到项目源码
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 设置数据集相关的环境变量（这是你的 Dataset 类初始化所必需的）
export AGIBOT_DATA_ROOT="/home/v-zhifeng/HPE/openpi/data"
export NORM_STATS_FILE="/home/v-zhifeng/HPE/openpi/assets/pi05_agiworld/agibot_full/dataset_stats_mp_q01q99_static.json"
export CUDA_VISIBLE_DEVICES=1


# 执行运行命令
uv run scripts/serve_policy.py \
    --port 8001 \
    policy:checkpoint \
    --policy.config="pi05_agiworld" \
    --policy.dir="checkpoints/pi05_agiworld/formal_pi05_hsdp_4n8gA100_20251220_bs512_lr5e-5_step1000000/20000" \