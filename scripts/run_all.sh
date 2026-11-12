#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/flagos-add-rmsnorm"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"

cd "$PROJECT"
source .venv/bin/activate

mkdir -p "$LOG_DIR"

echo "[1/4] correctness + benchmark"
cd "$PROJECT/src"
python bench.py 2>&1 | tee "$LOG_DIR/bench.log"

echo "[2/4] block and num_warps sweep"
python block_sweep.py 2>&1 | tee "$LOG_DIR/block_sweep.log"

echo "[3/4] kernel launch overhead"
python launch_overhead.py 2>&1 | tee "$LOG_DIR/launch_overhead.log"

echo "[4/4] backward correctness and timing"
python test_backward.py --shapes 64x256,4096x4096 --time 2>&1 | tee "$LOG_DIR/backward.log"

echo "logs written to $LOG_DIR"
echo "FlagGems comparison is optional: install g++ and flag-gems==5.0.2, then run src/bench_flag_gems.py"
