#!/usr/bin/env bash
set -euo pipefail

# ================= 配置区域 =================

# azcopy命令路径
AZCOPY_BIN="/scratch/amlt_code/azcopy"

# --- URL 配置 ---
# 提取公共根路径 (去掉具体的 videos_h264，保留到 resize_224)
COMMON_ROOT_URL="https://igshare.blob.core.windows.net/v-zhifeng/agibot_beta_split_500_resize_224"

# SAS token（保持原样）
SAS_TOKEN="?sv=2023-01-03&spr=https%2Chttp&st=2025-12-10T05%3A26%3A07Z&se=2025-12-17T05%3A26%3A00Z&skoid=3950ba63-725c-441f-8f24-b3af6f933a15&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-12-10T05%3A26%3A07Z&ske=2025-12-17T05%3A26%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=zjJ1IKYmCQTzcMbRQWyHHmb6CMQBJFqBcoZeH9qkF98%3D"

# --- 本地路径配置 ---
# JSON list 文件
LIST_FILE="videos_h264_list.json"

# 设置根输出目录 (对应 agibot_beta_split_500_resize_224)
ROOT_OUTPUT_DIR="$(pwd)/agibot_beta_split_500_resize_224"
# 设置视频子目录
VIDEO_OUTPUT_DIR="${ROOT_OUTPUT_DIR}/videos_h264"

# ================= 执行区域 =================

if [[ ! -f "$LIST_FILE" ]]; then
    echo "❌ ERROR: List file $LIST_FILE not found!"
    exit 1
fi

# 创建根目录和视频目录
mkdir -p "$ROOT_OUTPUT_DIR"
mkdir -p "$VIDEO_OUTPUT_DIR"

echo "📂 Root Output dir: $ROOT_OUTPUT_DIR"

# ---------------------------------------------------------
# Part 1: 下载额外的静态文件 (.npy 和 actions_gaussian)
# ---------------------------------------------------------
echo -e "\n⬇️  正在下载静态基础文件..."

# 1. 下载 episodic_dataset_fixed_static.npy
echo "   -> Downloading: episodic_dataset_fixed_static.npy"
"$AZCOPY_BIN" copy "${COMMON_ROOT_URL}/episodic_dataset_fixed_static.npy${SAS_TOKEN}" "$ROOT_OUTPUT_DIR/"

# 2. 下载 actions_gaussian (递归下载文件夹)
echo "   -> Downloading: actions_gaussian (recursive)"
"$AZCOPY_BIN" copy "${COMMON_ROOT_URL}/actions_gaussian${SAS_TOKEN}" "$ROOT_OUTPUT_DIR/" --recursive

echo "✅ 静态文件下载完成。"

# ---------------------------------------------------------
# Part 2: 循环下载视频列表
# ---------------------------------------------------------
echo -e "\n⬇️  开始处理视频列表..."

# 用 Python 解析 JSON
mapfile -t ITEMS < <(python3 << 'PYCODE'
import json
with open("videos_h264_list.json", "r") as f:
    data = json.load(f)
for x in data:
    print(str(x))
PYCODE
)

# 遍历数组
for ITEM in "${ITEMS[@]}"; do
    echo -e "\n=============================="
    echo "🚀 开始下载视频片段: $ITEM"
    echo "=============================="

    # 拼接视频特定的 URL
    URL="${COMMON_ROOT_URL}/videos_h264/${ITEM}${SAS_TOKEN}"

    # 注意：这里输出目录用的是 VIDEO_OUTPUT_DIR
    "$AZCOPY_BIN" copy "$URL" "$VIDEO_OUTPUT_DIR" --recursive

    if [[ $? -eq 0 ]]; then
        echo "✅ 下载完成: $ITEM"
    else
        echo "❌ 下载失败: $ITEM (已跳过)"
    fi
done

echo -e "\n🎉 全部任务执行完毕！"