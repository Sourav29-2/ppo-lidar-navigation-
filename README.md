# PPO LiDAR Navigation

**Hybrid Deep RL + Nav2 indoor robot navigation — 82.7% success rate over 150 benchmark scenarios**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Gazebo](https://img.shields.io/badge/Gazebo-Classic-orange)](http://gazebosim.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

> 🎥 **[Watch full demo recording (YouTube)](https://youtube.com/YOUR_DEMO_LINK)**  
> 📊 **[Full results & failure analysis](RESULTS.md)**  
> 🏗️ **[System architecture deep-dive](ARCHITECTURE.md)**

---

## What This Is

A trained robot navigation system that combines **Nav2 global path planning** with
a **PPO (Proximal Policy Optimisation) deep RL agent** for local navigation and
reactive obstacle avoidance.

The robot navigates a 16 × 16 m simulated apartment with rooms, corridors, and
furniture — using only a **360° LiDAR sensor** and the Nav2 global path.
No camera. No object detection. Pure sensor-to-action end-to-end control.

**You can click anywhere in RViz and the robot will navigate there.**

---

## System Architecture

```
User clicks in RViz
        │
        ▼ /goal_pose
Nav2 Global Planner (SLAM map)
        │
        ▼ /plan  (global path)
┌───────────────────────────────────┐
│   PPO Actor  (50D → 2D)           │
│                                   │
│  36D  LiDAR sectors               │
│   2D  Goal waypoint               │
│   2D  Current velocity            │
│  10D  Nav2 path features  ──────► │──► /cmd_vel_nav ──► Robot
└───────────────────────────────────┘
```

The PPO agent sees a **50-dimensional observation** and outputs smooth velocity
commands at 10 Hz, handling tight turns, wall-blocked corridors, and complex
obstacle clusters that trip up classical planners alone.

---

## Results

| Model | Success Rate | Collision | Timeout |
|---|---|---|---|
| PPO-only (baseline) | 52.0% | 32.7% | 15.3% |
| Hybrid Nav2 + PPO Phase 4 | 64.2% | 26.3% | 9.5% |
| **Phase 5 Safety (this repo)** | **82.7%** | **16.0%** | **1.3%** |

**150 scenarios** across 8 categories — open spaces, wall-blocked corridors,
complex obstacle clusters, and random start/goal pairs.
Zero oscillations. Zero fallback triggers.

→ [Full breakdown by category](RESULTS.md)

---

## Quick Demo (Mac)

### Prerequisites
- macOS with Apple Silicon or Intel
- ~8 GB free disk space (ROS2 + dependencies)

### 1. Clone & Setup

```bash
git clone https://github.com/Sourav29-2/ppo-lidar-nav.git
cd ppo-lidar-nav
bash setup/setup_mac.sh
```

This installs everything: ROS2 Humble, Nav2, SLAM Toolbox, PyTorch, Gymnasium.
Takes 5–15 minutes on first run.

### 2. Launch

```bash
./demo.sh
```

Gazebo, SLAM, Nav2, and RViz open automatically.

### 3. Navigate

1. In RViz, click the **"Nav2 Goal"** button in the toolbar (arrow icon)
2. Click anywhere on the apartment map
3. Watch the robot navigate there!
4. Click again to set a new destination anytime

---

## Quick Demo (Windows)

### Step 1 — Install WSL2 (PowerShell, as Administrator)

```powershell
wsl --install -d Ubuntu-22.04
```

Restart when prompted.

### Step 2 — Inside the Ubuntu WSL2 terminal

```bash
curl -fsSL https://raw.githubusercontent.com/Sourav29-2/ppo-lidar-nav/master/setup/setup_wsl_inner.sh | bash
```

### Step 3 — Run

```bash
cd ~/ppo-lidar-nav
./demo.sh
```

→ [Full Windows setup guide](setup/setup_windows_wsl.md)

---

## Repository Structure

```
ppo-lidar-nav/
│
├── demo.py                  # Interactive RViz demo (inference only)
├── demo.sh                  # One-command launcher
│
├── ppo/                     # PPO algorithm implementation
│   ├── actor.py             # Policy network (Gaussian, 50D → 2D)
│   ├── critic.py            # Value network
│   ├── ppo_trainer.py       # PPO update step
│   └── rollout_buffer.py    # Experience collection
│
├── src/urdf_test/           # ROS2 package
│   ├── scripts/
│   │   └── gazebo_nav_env.py  # Gymnasium environment (GazeboMacNavEnv)
│   ├── launch/              # Gazebo + Nav2 + RViz launch files
│   ├── config/
│   │   ├── nav2_params.yaml
│   │   └── rviz_demo.rviz   # Pre-configured RViz layout
│   ├── maps/                # SLAM-generated apartment map
│   └── urdf/                # Robot URDF model
│
├── training/                # Training scripts
│   ├── train_phase5_safety.py
│   └── run_phase5_training.sh
│
├── evaluation/              # Evaluation scripts
│   ├── eval_final_phase5.py
│   └── run_final_eval.sh
│
├── checkpoints/
│   └── hybrid_phase5/
│       └── best_success.pt  # Final champion model
│
├── results/                 # Clean evaluation data
│   ├── final_evaluation.csv
│   ├── final_evaluation_report.txt
│   └── benchmark_150_scenarios.json
│
├── setup/                   # Platform setup scripts
│   ├── setup_mac.sh
│   ├── setup_windows_wsl.md
│   └── setup_wsl_inner.sh
│
├── ARCHITECTURE.md          # System design deep-dive
├── RESULTS.md               # Full benchmark results
├── ROADMAP.md               # Planned extensions
└── CONTRIBUTING.md          # How to extend the project
```

---

## Training From Scratch

If you want to retrain the agent:

```bash
# Install dependencies
bash setup/setup_mac.sh

# Start simulation (in your ros2_study workspace)
# Then in a second terminal:
pixi run python training/train_phase5_safety.py
```

Training logs to `runs/` (TensorBoard). Checkpoints save every 5k steps.

See [ARCHITECTURE.md](ARCHITECTURE.md) for hyperparameters and curriculum details.

---

## Running the Benchmark

```bash
# Run the full 150-scenario evaluation
pixi run bash evaluation/run_final_eval.sh

# Results saved to results/final_evaluation.csv
```

---

## Future Plans

| Extension | Status |
|---|---|
| Real robot deployment (hardware interface) | 🔜 Planned |
| RGB-D camera + YOLOv8 object detection | 🔜 Planned |
| Voice command interface (Whisper STT) | 💡 Idea |
| Dynamic obstacle avoidance | 💡 Idea |

→ [Full roadmap](ROADMAP.md)

---

## Contact

**Sourav Kumar**  
📧 sourav9835359245@gmail.com  
🐙 [GitHub](https://github.com/Sourav29-2)

Open to robotics engineering roles, autonomous systems research, and collaborations.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
