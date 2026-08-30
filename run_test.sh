#!/bin/bash
# Diagnostic test launcher for PPO Lidar Navigation Phase 3+
#
# IMPORTANT: Do NOT source ros2_study/install/setup.bash here.
# The pixi env already has a full ROS Humble stack. Sourcing ros2_study
# pollutes DYLD_LIBRARY_PATH with duplicate Qt/OpenMP libs, causing
# libomp.dylib double-init deadlocks when PyTorch is imported.
#
# All required env vars (KMP_DUPLICATE_LIB_OK, OMP_NUM_THREADS etc.) are
# now declared in pixi.toml [activation.env] and are set automatically.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.pixi/envs/default/bin/python"

echo "=== Hybrid Observation Diagnostic ==="
echo "Python: $PYTHON"
echo "Working Dir: $SCRIPT_DIR"
echo "Starting at: $(date)"
echo "======================================"

exec "$PYTHON" -u "$SCRIPT_DIR/diagnostics/test_hybrid_observation.py" 2>&1
