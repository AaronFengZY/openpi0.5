#!/usr/bin/env bash
set -euo pipefail

echo "=== ⚙️ Setting up OpenPI (Pi0.5) training environment with uv ==="

############################################
# 0) 定位仓库根目录（你把 repo 放在 /scratch/amlt_code/openpi）
############################################
BASE_DIR="/scratch/amlt_code/openpi"
if [[ ! -d "${BASE_DIR}" ]]; then
  # 允许你把 repo 直接扔到 /scratch/amlt_code/（没有 openpi 子目录）
  if [[ -f "/scratch/amlt_code/pyproject.toml" ]]; then
    BASE_DIR="/scratch/amlt_code"
  else
    echo "❌ Cannot find openpi repo. Expected at /scratch/amlt_code/openpi or /scratch/amlt_code."
    exit 1
  fi
fi
echo "→ Repo dir: ${BASE_DIR}"
cd "${BASE_DIR}"

############################################
# 1) 安装 / 校验 uv
############################################
if ! command -v uv >/dev/null 2>&1; then
  echo "→ Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | bash
  # uv 默认装在 ~/.local/bin 或 ~/.cargo/bin
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"   # ✅ 确保立刻生效
else
  echo "→ uv found: $(uv --version)"
fi

# 永久写入 .bashrc，防止下次登录失效
if ! grep -q '\/.local\/bin' "${HOME}/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> "${HOME}/.bashrc"
fi

############################################
# 2) Git 子模块 + Git LFS（LeRobot 需要）
############################################
if command -v git >/dev/null 2>&1; then
  # 如果 .git 存在，初始化子模块
  if [[ -d ".git" ]]; then
    echo "→ Initializing submodules ..."
    git submodule update --init --recursive || true
  else
    echo "⚠️ No .git directory. Skipping submodule init (zip/rsync 上传通常无 .git)"
  fi

  # 安装 Git LFS（某些镜像没有）
  if ! git lfs version >/dev/null 2>&1; then
    echo "→ Installing Git LFS ..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -y
      apt-get install -y git-lfs
    else
      echo "⚠️ apt-get not available; skipping git-lfs system install"
    fi
  fi

  # 即便没装 LFS，也强制跳过 smudge，避免 LeRobot 依赖卡住
  git config --global filter.lfs.smudge "git-lfs smudge --skip -- %f" || true
  export GIT_LFS_SKIP_SMUDGE=1
else
  echo "⚠️ git not found; skipping submodule & LFS steps"
  export GIT_LFS_SKIP_SMUDGE=1
fi

############################################
# 3) Python / JAX / HF 环境变量（训练友好）
############################################
# JAX 显存：不预分配 + 最大占比
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# 关闭不必要的 NCCL info（多卡更干净）
export NCCL_DEBUG=WARN
# HF：离线/本地缓存更稳（可按需改为0/1）
export HF_HUB_OFFLINE=1
export HF_HOME="${BASE_DIR}/.hf_cache"
mkdir -p "${HF_HOME}"

############################################
# 4) 用 uv 同步 Python 依赖（创建 .venv）
############################################
if [[ ! -f "pyproject.toml" ]]; then
  echo "❌ pyproject.toml not found in ${BASE_DIR}"
  exit 1
fi

############################################
# 4) 用 uv 创建 Python 3.11 的 venv，并安装本地包
############################################

echo "→ Creating fresh Python 3.11 venv with uv ..."
# 防止之前残留的是 3.10 的 .venv
rm -rf .venv

echo "→ Syncing deps with uv (Python >=3.11, full dependency tree) ..."
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "→ Python version (via uv): $(uv run python -V)"

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [关键修改] ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
echo "→ Installing decord into the .venv ..."
# 使用 --python .venv 强制告诉 uv：不管当前激活了谁，给我装进 .venv 里！
uv pip install decord --python .venv

echo "→ Verifying decord installation ..."
# 立即验证，如果这里报错，任务直接停止，不用等到跑训练才发现
uv run python -c "import decord; print(f'✅ Decord installed at: {decord.__file__}')"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


############################################
# 5) （可选）修复 JAX CUDA 轮子不匹配的情况
# 通常不需要；如果你在运行时报 CUDA 版本/Driver 兼容错误，再打开下面注释。
############################################
# echo "→ (Optional) Repair JAX CUDA wheels"
# uv pip install --reinstall --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

############################################
# 6) 打印关键信息
############################################
echo "Python: $(uv run python -V)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "GIT_LFS_SKIP_SMUDGE=${GIT_LFS_SKIP_SMUDGE}"

# ############################################
# # ✅ 新增：安装 OpenCV 运行依赖（libgthread等）
# ############################################
# echo "→ Installing OpenCV system dependencies ..."
# sudo apt-get update
# sudo apt-get install -y libglib2.0-0 libgl1 libsm6 libxext6 libxrender1


echo "=== ✅ Environment ready. Use the following to train: ==="
echo "cd ${BASE_DIR}"
echo 'CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1 uv run scripts/train.py pi05_agiworld --exp-name=agibotworld_run1_shard --fsdp-devices=4 --overwrite'
echo "=========================================================="