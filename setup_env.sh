#!/usr/bin/env bash
set -euo pipefail

echo "=== ⚙️ Setting up uv Python environment ==="

# 1️⃣ 检查或安装 uv
if ! command -v uv >/dev/null 2>&1; then
  echo "→ Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | bash
  export PATH="$HOME/.cargo/bin:$PATH"
else
  echo "→ uv already installed: $(uv --version)"
fi

# 2️⃣ 将 uv 路径永久加入 ~/.bashrc
if ! grep -q "uv" ~/.bashrc 2>/dev/null; then
  echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
  echo "→ Added uv to ~/.bashrc"
fi

# 3️⃣ 打印基础信息
echo "Python: $(python3 -V)"
echo "uv: $(uv --version)"
echo "PATH: $PATH"

# 4️⃣（可选）设置 JAX / HF 环境变量
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HF_HUB_ENABLE_HF_TRANSFER=1

# 5️⃣ 安装依赖（假设当前目录有 pyproject.toml）
if [[ -f "pyproject.toml" ]]; then
  echo "→ Installing dependencies..."
  uv sync
else
  echo "⚠️ pyproject.toml not found, skipping uv sync"
fi

echo "=== ✅ uv environment ready ==="
