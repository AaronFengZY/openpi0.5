# 设置变量（参考你的 train.sh）
DATA_ROOT="/home/v-zhifeng/HPE/openpi/data"
STATS_FILE="/home/v-zhifeng/HPE/openpi/assets/pi05_agiworld/agibot_full/dataset_stats_mp_q01q99_static.json"
INDEX_FILE="episodic_dataset_fixed_static.npy"

# 运行 Client (无需指定 index，它会随机选)
uv run client.py \
  --ws ws://127.0.0.1:8001 \
  --dataset_root "$DATA_ROOT" \
  --norm_stats_file "$STATS_FILE" \
  --index_file "$INDEX_FILE" \
  --action_horizon 30