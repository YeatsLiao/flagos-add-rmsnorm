#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/flagos-add-rmsnorm"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_DIR/src"

mkdir -p "$PROJECT/logs" "$PROJECT/src"
if [ "$SRC" != "$PROJECT/src" ]; then
    cp "$SRC"/*.py "$PROJECT/src/"
fi
cd "$PROJECT"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "[0/6] install uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "[1/6] create venv with uv"
uv venv .venv --python 3.14 --clear
source .venv/bin/activate
python -V

echo "[2/6] install C compiler and Python headers"
# Triton compiles driver.c at first kernel launch
if ! command -v gcc >/dev/null 2>&1 || [ ! -f /usr/include/python3.14/Python.h ]; then
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        apt-get install -y gcc python3-dev
    else
        echo "missing gcc or Python headers; run: sudo apt-get install -y gcc python3-dev"
        exit 1
    fi
fi

echo "[3/6] install torch 2.9.1+cu128, triton 3.5.1 and numpy"
uv pip install "torch==2.9.1+cu128" "triton==3.5.1" numpy \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple \
    --index-strategy unsafe-best-match

echo "[4/6] version and gpu check"
python - <<'PY'
import torch
import triton
import torch.nn.functional as F

print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("cc", torch.cuda.get_device_capability(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
print("rms_norm", hasattr(F, "rms_norm"))
PY

echo "[5/6] triton smoke test"
cd "$PROJECT/src"
python smoke_triton.py

echo "[6/6] setup done"
