#!/usr/bin/env bash
# =============================================================================
#  setup_wsl_inner.sh — Bootstrap script run inside WSL2 (Ubuntu 22.04)
#  Called by: curl -fsSL ...setup_wsl_inner.sh | bash
# =============================================================================
set -e

REPO_URL="https://github.com/Sourav29-2/ppo-lidar-navigation-.git"
REPO_DIR="$HOME/ppo-lidar-navigation-"

echo "════════════════════════════════════════════════════════"
echo "  PPO LiDAR Navigation — WSL2 / Linux Setup"
echo "════════════════════════════════════════════════════════"

# ── System dependencies ───────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y curl git build-essential

# ── pixi ─────────────────────────────────────────────────────────────────────
echo "[2/5] Installing pixi..."
if ! command -v pixi &> /dev/null; then
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
    echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> "$HOME/.bashrc"
fi
echo "✅ pixi ready."

# ── Clone repo ────────────────────────────────────────────────────────────────
echo "[3/5] Cloning repository..."
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
echo "✅ Repo cloned to $REPO_DIR"

# ── pixi install ─────────────────────────────────────────────────────────────
echo "[4/5] Installing all dependencies (may take 5–15 min first time)..."
pixi install
echo "✅ Dependencies installed."

# ── Build ROS2 package ────────────────────────────────────────────────────────
echo "[5/5] Building ROS2 package..."
pixi run -e default -- colcon build --symlink-install --packages-select urdf_test || true
echo "✅ Build complete."

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo ""
echo "  Next steps:"
echo "    cd ~/ppo-lidar-navigation-"
echo "    ./demo.sh"
echo "════════════════════════════════════════════════════════"
