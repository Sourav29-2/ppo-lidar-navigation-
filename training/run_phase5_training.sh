#!/bin/bash
# Phase 4 Training Launcher — Nav2 + PPO Hybrid (50D observation space)
# Ensures clean environment for macOS Apple Silicon
#
# ─── IMPORTANT ───────────────────────────────────────────────────────────────
# Do NOT source ros2_study/install/setup.bash here.
# The pixi env already contains a full ROS Humble stack (rclpy, nav_msgs, etc.)
# Sourcing ros2_study WILL pollute DYLD_LIBRARY_PATH with duplicate Qt/OpenMP
# libraries, causing a libomp.dylib double-init deadlock when PyTorch loads.
#
# All env vars (KMP_DUPLICATE_LIB_OK, OMP_NUM_THREADS, etc.) are declared in
# pixi.toml [activation.env] and fire before any Python code runs. They are
# also set explicitly here as a belt-and-suspenders fallback.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.pixi/envs/default/bin/python"

# Belt-and-suspenders: set env vars even if pixi activation didn't fire
export ROS_LOCALHOST_ONLY=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

echo "=============================================="
echo "  Phase 4: Nav2 + PPO Hybrid Training"
echo "=============================================="
echo "  Python  : $PYTHON"
echo "  Script  : $SCRIPT_DIR/train_phase5_safety.py"
echo "  Started : $(date)"
echo "----------------------------------------------"
echo "  Source  : checkpoints/wall_ft_best.pt"
echo "  Output  : checkpoints/hybrid_phase4/"
echo "  Budget  : 50,000 steps (HARD STOP)"
echo "  Eval    : every ~5k steps"
echo "=============================================="
echo ""

exec pixi run python -u "$SCRIPT_DIR/train_phase5_safety.py" 2>&1
