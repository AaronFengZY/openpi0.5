#!/bin/bash
set -e

# ================= 路径配置 =================
VIDEO_ROOT="/home/v-zhifeng/HPE/openpi/data/videos_h264"
DATA_ROOT="/home/v-zhifeng/HPE/openpi/data"
INDEX_FILE="${DATA_ROOT}/episodic_dataset_fixed_static.npy"
ACTION_ROOT="/home/v-zhifeng/HPE/openpi/data/actions_gaussian"

# ✅ 1. 定义相机和 URDF 路径
CAMERA_NPY="/home/v-zhifeng/HPE/v-zhifeng/agibot_beta_split_500/camera_param.npy"
URDF="/home/v-zhifeng/HPE/agirobot/G1/A2D_120s/A2D.urdf"

OUTPUT_DIR="./outputs_fk_inference_vis"
HOST="127.0.0.1"
PORT=8001
INTERVAL=10

echo "🎲 Randomly selecting an episode from index file..."

# ==========================================
# 🆕 新增：使用 Python 临时读取 npy 并随机选择
# ==========================================
# 这段 Python 代码会输出: "EPISODE_ID EPISODE_LEN" (例如: "750134 426")
read -r TARGET_INDEX MAX_LEN <<< $(python3 -c "
import numpy as np
import os
import sys

try:
    path = '${INDEX_FILE}'
    raw = np.load(path, allow_pickle=True)
    data = raw.item() if hasattr(raw, 'ndim') and raw.ndim == 0 else raw
    
    paths = data.get('video_path', None)
    start_ends = data.get('start_end', None) # 修改这里

    if paths is None or start_ends is None:
        raise KeyError(f'Missing keys. Available: {list(data.keys())}')

    count = len(paths)
    idx = np.random.randint(0, count)
    
    # 1. 提取 ID
    path_raw = paths[idx]
    path_str = path_raw.decode('utf-8') if isinstance(path_raw, bytes) else str(path_raw)
    episode_id = os.path.basename(path_str.rstrip('/'))
    
    # 2. 提取长度 (end - start)
    se = start_ends[idx]
    # 假设 se 是 [start, end] 格式
    ep_len = int(se[1] - se[0])
    
    print(f'{episode_id} {ep_len}')

except Exception as e:
    sys.stderr.write(f'\n🐍 Python Error: {str(e)}\n')
    print('ERROR 0') 
")

if [ "$TARGET_INDEX" == "ERROR" ]; then
    echo "❌ Failed to read index file."
    exit 1
fi

# ✅ 自动设置起止帧
START_FRAME=0
# 为了安全，可以将结束帧稍微减去一点，或者直接用全长
END_FRAME=$MAX_LEN

echo "🎯 Selected Episode: $TARGET_INDEX"
echo "📏 Episode Length: $MAX_LEN (Setting range: $START_FRAME -> $END_FRAME)"

# ==========================================
# 🚀 启动 Client
# ==========================================

echo "🚀 Starting Inference + FK Visualization Client..."
export PYTHONPATH=$PYTHONPATH:. 

CMD_ARGS=(
  --host "$HOST"
  --port "$PORT"
  --index "$TARGET_INDEX"
  --data_path "$INDEX_FILE"
  --action_root "$ACTION_ROOT"
  --video_folder "$VIDEO_ROOT"
  --start_frame "$START_FRAME"
  --end_frame "$END_FRAME"
  --interval "$INTERVAL"
  --camera_npy "$CAMERA_NPY"
  --urdf "$URDF"
)

echo "Running command: uv run client_fk_video_gen.py ${CMD_ARGS[@]}"
uv run client_fk_video_gen.py "${CMD_ARGS[@]}"