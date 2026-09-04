#!/usr/bin/env bash
# =============================================================================
#  setup_mac.sh — One-command setup for macOS (Apple Silicon or Intel)
# =============================================================================
#
#  USAGE (copy-paste into your terminal):
#    git clone https://github.com/Sourav29-2/ppo-lidar-nav.git
#    cd ppo-lidar-nav
#    bash setup/setup_mac.sh
#
# =============================================================================

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${GREEN}"
echo "  ╔════════════════════════════════════════════╗"
echo "  ║   PPO LiDAR Navigation — macOS Setup       ║"
echo "  ╚════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: Install pixi (if not already installed) ──────────────────────────
if ! command -v pixi &> /dev/null; then
    echo -e "${YELLOW}[1/4]${NC} Installing pixi package manager..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
    echo "✅ pixi installed."
else
    echo -e "${GREEN}[1/4]${NC} pixi already installed ($(pixi --version))."
fi

# ── Step 2: Install all dependencies ─────────────────────────────────────────
echo -e "${YELLOW}[2/4]${NC} Installing ROS2 Humble + PyTorch + Nav2 dependencies..."
echo "    (This may take 5–15 minutes on first run — packages are large)"
pixi install
echo "✅ All dependencies installed."

# ── Step 3: Build the ROS2 package ───────────────────────────────────────────
echo -e "${YELLOW}[3/4]${NC} Building ROS2 package (colcon)..."
pixi run -e default -- bash -c "
    source \$(pixi run -e default -- bash -c 'echo \$CONDA_PREFIX')/setup.bash 2>/dev/null || true
    colcon build --symlink-install --packages-select urdf_test
"
echo "✅ ROS2 package built."

# ── Step 4: Verify checkpoint ─────────────────────────────────────────────────
echo -e "${YELLOW}[4/4]${NC} Checking for trained model checkpoint..."
if [ -f "checkpoints/hybrid_phase5/best_success.pt" ]; then
    echo "✅ Champion checkpoint found."
else
    echo -e "${RED}⚠️  Checkpoint not found.${NC}"
    echo "   The trained model is available in GitHub Releases."
    echo "   Download best_success.pt and place it at:"
    echo "   checkpoints/hybrid_phase5/best_success.pt"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Setup complete!${NC}"
echo ""
echo "  To run the interactive demo:"
echo "    1. Start your ROS2 simulation workspace (ros2_study)"
echo "    2. Run:  ./demo.sh"
echo ""
echo "  Or run training from scratch:"
echo "    pixi run python training/train_phase5_safety.py"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
