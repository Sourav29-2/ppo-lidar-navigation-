#!/usr/bin/env bash
# =============================================================================
#  demo.sh — One-command launcher for PPO LiDAR Navigation Interactive Demo
# =============================================================================
#
#  USAGE:
#    ./demo.sh
#
#  WHAT IT DOES:
#    1. Sources ROS2 workspace (ros2_study) for Gazebo / Nav2 / RViz
#    2. Sources this project's pixi environment for PyTorch / PPO
#    3. Opens Gazebo Classic with the indoor map
#    4. Starts SLAM Toolbox
#    5. Starts Nav2 stack (global planner + costmaps)
#    6. Opens RViz with the pre-configured demo view
#    7. Starts demo.py — waits for your RViz clicks
#
#  HOW TO USE THE DEMO:
#    → Click the "Nav2 Goal" button in the RViz toolbar (arrow icon)
#    → Click anywhere on the map
#    → Watch the robot navigate there!
#    → Click another point to set a new goal anytime
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_WS="$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   PPO LiDAR Navigation — Interactive Demo Launcher  ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Ensure ROS 2 is available ────────────────────────────────────────────────
if ! command -v ros2 &> /dev/null; then
    echo "⚙️  Activating pixi environment for ROS 2..."
    eval "$(pixi shell-hook -e default 2>/dev/null)" || true
fi
if [ ! -f "$PROJECT_WS/checkpoints/hybrid_phase5/best_success.pt" ]; then
    echo "❌ Champion checkpoint not found."
    echo "   Run: python training/train_phase5_safety.py"
    echo "   Or download it from: https://github.com/Sourav29-2/ppo-lidar-navigation-/releases"
    exit 1
fi

# ── Environment variables ─────────────────────────────────────────────────────
export ROS_LOCALHOST_ONLY=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Source this project's colcon build
source "$PROJECT_WS/install/setup.bash" 2>/dev/null || true

echo -e "${GREEN}[1/5]${NC} Starting Gazebo simulation..."
ros2 launch urdf_test launch_sim_gazebo_test_robo.launch.py &
GAZEBO_PID=$!
sleep 5

# (SLAM is disabled during navigation because we are using the pre-saved full map + AMCL)

echo -e "${GREEN}[3/5]${NC} Starting Nav2 stack..."
ros2 launch urdf_test nav2.launch.py &
NAV2_PID=$!

# (AMCL now initializes automatically via set_initial_pose in nav2_params.yaml)

echo -e "${GREEN}[4/5]${NC} Opening RViz..."
RVIZ_CONFIG="$PROJECT_WS/src/urdf_test/config/rviz_demo.rviz"
ros2 run rviz2 rviz2 -d "$RVIZ_CONFIG" &
RVIZ_PID=$!
sleep 3

echo -e "${GREEN}[5/5]${NC} Starting PPO Demo (inference mode)..."
echo ""
echo -e "${YELLOW}  ┌────────────────────────────────────────────────┐"
echo "  │  Click 'Nav2 Goal' in RViz toolbar, then      │"
echo "  │  click anywhere on the map to navigate there! │"
echo "  └────────────────────────────────────────────────┘${NC}"
echo ""

# Activate pixi env for PyTorch / PPO inference
eval "$(pixi shell-hook -e default 2>/dev/null)" || true
python "$PROJECT_WS/demo.py"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
echo ""
echo "Shutting down simulation stack..."
kill $RVIZ_PID $NAV2_PID $GAZEBO_PID 2>/dev/null || true
wait 2>/dev/null
echo "✅ All processes stopped. Goodbye!"
