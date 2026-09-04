# System Architecture

## Overview

This project implements a **Hybrid Nav2 + PPO** indoor navigation system for a
differential-drive robot in a simulated 16 × 16 m apartment environment.

The key insight: instead of replacing Nav2, the PPO agent is trained to **refine
and execute** Nav2's global plan locally — combining the strengths of traditional
planning (global path, map awareness) with deep RL (reactive obstacle avoidance,
smooth motion).

---

## Full Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Simulation Layer                             │
│  Gazebo Classic ──► Robot URDF ──► /scan (LiDAR) + /odom + /clock  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                       Mapping & Planning                             │
│  SLAM Toolbox ──► /map ──► Nav2 Global Planner ──► /plan (Path)    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ 10D path features
┌────────────────────────────────▼────────────────────────────────────┐
│                    PPO Actor (50D → 2D)                             │
│                                                                     │
│  Inputs:                                                            │
│    [0:36]  36 LiDAR sectors (min of 10 rays each, normalized)       │
│    [36:38] 2D lookahead waypoint (distance + heading error)         │
│    [38:40] Current linear + angular velocity                        │
│    [40:50] 5 Nav2 path waypoints in robot frame (x,y) × 5          │
│                                                                     │
│  Output:                                                            │
│    [0] Linear velocity  ∈ [-1, 1] → scaled to [-0.22, 0.33] m/s   │
│    [1] Angular velocity ∈ [-1, 1] → scaled to [-2.84, 2.84] rad/s  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ /cmd_vel_nav
┌────────────────────────────────▼────────────────────────────────────┐
│  Twist Mux → /cmd_vel → Robot                                       │
│  (PPO output at priority 10; teleop override at priority 100)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PPO Network Architecture

```python
Actor(obs_dim=50, action_dim=2)
  Linear(50 → 256) + Tanh
  Linear(256 → 256) + Tanh
  Linear(256 → 2)    # mean of Gaussian policy
  log_std (learnable parameter, init -0.5)

Critic(obs_dim=50)
  Linear(50 → 256) + Tanh
  Linear(256 → 256) + Tanh
  Linear(256 → 1)    # state value
```

---

## Observation Space (50D)

| Index | Feature | Description |
|---|---|---|
| 0–35 | LiDAR sectors | 36 sectors, each = min of 10 raw rays, normalized to [0,1] |
| 36 | Waypoint distance | Distance to 0.5m lookahead point, normalized by 8m |
| 37 | Heading error | Heading to waypoint, normalized by π |
| 38 | Linear velocity | Current linear vel, normalized by max_linear_vel |
| 39 | Angular velocity | Current angular vel, normalized by max_angular_vel |
| 40–49 | Nav2 path features | 5 waypoints at 0.5/1.0/1.5/2.0/2.5m along plan, in robot frame, normalized by 3m |

---

## Reward Function (Phase 5)

| Component | Weight | Purpose |
|---|---|---|
| Progress reward | ×15 | Reward distance reduction to goal |
| Clearance reward | ×8 | Reward maintaining distance from obstacles |
| Safety margin penalty | ×6 | Penalise approaching walls faster than safe speed |
| Oscillation penalty | ×4 | Penalise rapid left-right turns |
| Time penalty | −0.05/step | Encourage efficiency |
| Inflation zone penalty | variable | Penalise entering 0.35m obstacle inflation zone |
| Reverse penalty | ×2 | Discourage unnecessary reversing |
| Collision penalty | −200 | Terminal penalty on collision |
| Success reward | +300 | Terminal reward on reaching goal (≤ 0.5m) |

---

## Training Curriculum

| Phase | Description | Steps | Success Rate |
|---|---|---|---|
| Stage A | Basic goal seeking, random goals | 200k | ~46% |
| Stage C | Oscillation removal, wall-following | 50k | ~48% |
| Stage D | Safe collision training, 40D obs | 100k | ~52% |
| Hybrid Phase 4 | Nav2 path features added (50D), Nav2+PPO integration | 100k | ~64% |
| **Hybrid Phase 5** | Safety margin fine-tuning, speed control | 50k | **82.7%** |

---

## PPO Hyperparameters

| Parameter | Value |
|---|---|
| Rollout size | 4096 steps |
| Mini-batch size | 256 |
| Epochs per rollout | 10 |
| Learning rate | 3×10⁻⁴ |
| Discount (γ) | 0.99 |
| GAE lambda (λ) | 0.95 |
| Clip ε | 0.2 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Gradient clip | 0.5 |
