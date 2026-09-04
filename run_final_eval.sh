#!/bin/bash
# Final Phase 5 Evaluation Launcher
# Uses the same pixi environment as training to ensure all deps are available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROS_LOCALHOST_ONLY=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

echo "============================================================"
echo "  FINAL HYBRID PPO VALIDATION — Phase 5 Safety Model"
echo "============================================================"
echo "  Script  : $SCRIPT_DIR/eval_final_phase5.py"
echo "  Started : $(date)"
echo "  Checkpoint: checkpoints/hybrid_phase5/best_success.pt"
echo "  Scenarios : 150 deterministic (benchmark_150_scenarios.json)"
echo "  Threshold : 0.25 m strict"
echo "============================================================"
echo ""

exec pixi run python -u "$SCRIPT_DIR/eval_final_phase5.py" 2>&1
