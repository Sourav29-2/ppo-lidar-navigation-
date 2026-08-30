#!/bin/bash
# Stage D training launcher script
# Ensures clean environment for macOS Apple Silicon M2
#
# IMPORTANT: Do NOT source ros2_study/install/setup.bash here.
# The pixi env already has a full ROS Humble stack. Sourcing ros2_study
# pollutes DYLD_LIBRARY_PATH with duplicate Qt/OpenMP libs, causing
# libomp.dylib double-init deadlocks when PyTorch is imported.
#
# All required env vars (KMP_DUPLICATE_LIB_OK, OMP_NUM_THREADS etc.) are
# declared in pixi.toml [activation.env] and fire automatically.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.pixi/envs/default/bin/python"

echo "=== Stage D Training Launcher ==="
echo "Python: $PYTHON"
echo "Working Dir: $SCRIPT_DIR"
echo "Starting at: $(date)"
echo "================================="

exec "$PYTHON" -u "$SCRIPT_DIR/train_stage_d.py" 2>&1
