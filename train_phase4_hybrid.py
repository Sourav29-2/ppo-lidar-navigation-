"""Phase 4 — Hybrid Nav2 + PPO Training Script (50D observation space).

Architecture:
    SLAM → Nav2 Global Planner → /plan → 10D path features
                                                        ↓
    existing 40D PPO observation ──────────────────→ 50D PPO → cmd_vel

Rules (enforced in this file):
  • Loads from wall_ft_best.pt — NEVER overwrites the original Champion files.
  • Saves ONLY to checkpoints/hybrid_phase4/ (isolated directory).
  • Hard stop at exactly PHASE4_TOTAL_STEPS (50,000). No auto-extension.
  • No early stopping. Must reach 50k.
  • No Champion replacement based on intermediate evals.
  • Reward function, LiDAR preprocessing, action space, PPO architecture: UNCHANGED.
  • Curriculum: 40% wall-blocked / 30% complex-obstacle / 30% random.
  • Full 9-metric evaluation at every 5k steps.
"""

from __future__ import annotations

import os
# ─── OpenMP deadlock prevention — must be before ALL other imports ─────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

import csv
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# ─── Configure Python path ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

from gazebo_nav_env import GazeboMacNavEnv
from rl.actor import Actor
from rl.critic import Critic
from rl.ppo_trainer import PPOTrainer
from rl.rollout_buffer import RolloutBuffer
from rl.checkpointing import (
    CHECKPOINT_CONFIG,
    build_checkpoint,
    load_checkpoint,
    restore_from_checkpoint,
    save_checkpoint,
)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 constants — edit ONLY these to change run configuration
# ─────────────────────────────────────────────────────────────────────────────
PHASE4_TOTAL_STEPS   = 50_000   # hard stop — no auto-extension
PHASE4_EVAL_INTERVAL = 4_096    # evaluate every ~1 rollout (~5k steps effective)
PHASE4_ROLLOUT_SIZE  = 4_096
PHASE4_EVAL_EPISODES = 50       # deterministic benchmark (same as all previous models)
PHASE4_WALL_EVAL_EPS = 15       # additional wall-blocked-specific episodes for sub-metrics
PHASE4_SEED          = 42       # same eval seed as all previous models

# Source Champion checkpoint (read-only — never overwritten)
CHAMPION_CKPT = PROJECT_ROOT / "checkpoints" / "wall_ft_best.pt"

# Phase 4 output directory (isolated — never touches originals)
PHASE4_DIR = PROJECT_ROOT / "checkpoints" / "hybrid_phase4"

# Phase 4 training log
PHASE4_LOG  = PROJECT_ROOT / "checkpoints" / "hybrid_phase4" / "training_log.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Map geometry — obstacles from obstacles.world
# ─────────────────────────────────────────────────────────────────────────────
BOX_OBSTACLES = [
    (0.0,  8.0, 16.0, 0.2),    # North wall
    (0.0, -8.0, 16.0, 0.2),    # South wall
    (8.0,  0.0,  0.2, 16.0),   # East wall
    (-8.0, 0.0,  0.2, 16.0),   # West wall
    (-2.0,  5.5,  0.2, 5.0),   # br1_wall_v
    (-5.6,  3.0,  4.8, 0.2),   # br1_wall_h
    ( 2.0,  5.5,  0.2, 5.0),   # br2_wall_v
    ( 5.6,  3.0,  4.8, 0.2),   # br2_wall_h
    (-2.0, -5.5,  0.2, 5.0),   # br3_wall_v
    (-5.6, -3.0,  4.8, 0.2),   # br3_wall_h
    ( 3.0, -5.5,  0.2, 5.0),   # kitchen_wall_v
    ( 6.1, -3.0,  3.8, 0.2),   # kitchen_wall_h
    (-6.0,  6.0,  2.0, 1.8),   # br1_bed
    ( 6.0,  6.0,  2.0, 1.8),   # br2_bed
    (-6.0, -6.0,  2.0, 1.8),   # br3_bed
    ( 7.0, -5.5,  1.0, 3.5),   # kitchen_counter
    ( 0.0,  1.5,  3.0, 0.8),   # living_sofa
    ( 1.2,  1.2,  1.2, 0.6),   # coffee_table
    ( 1.0, -1.5,  0.4, 0.4),   # hall_box_obstacle
]
CYLINDER_OBSTACLES = [
    (-5.0, 0.0, 0.25),   # hall_pillar_1
    ( 4.5, 1.0, 0.25),   # hall_pillar_2
    (-2.0, 1.0, 0.20),   # hall_cylinder_obstacle
]

# ─────────────────────────────────────────────────────────────────────────────
# Goal classifiers
# ─────────────────────────────────────────────────────────────────────────────
def _is_clear_of_obstacles(x: float, y: float, margin: float = 0.45) -> bool:
    for cx, cy, sx, sy in BOX_OBSTACLES:
        if (cx - sx/2 - margin) <= x <= (cx + sx/2 + margin) and \
           (cy - sy/2 - margin) <= y <= (cy + sy/2 + margin):
            return False
    for cx, cy, r in CYLINDER_OBSTACLES:
        if np.linalg.norm([x - cx, y - cy]) <= r + margin:
            return False
    return True


def is_valid_goal(x: float, y: float, robot_pos: np.ndarray,
                  min_dist: float = 1.5) -> bool:
    if np.linalg.norm(np.array([x, y]) - robot_pos) < min_dist:
        return False
    return _is_clear_of_obstacles(x, y)


# ── Wall-blocked goals: inside bedrooms / behind room walls ───────────────────
# These goals require Nav2 to plan around a wall — the direct heading is blocked.
WALL_BLOCKED_GOALS = [
    # Bedroom 1 (behind br1_wall_v at x=-2, needs detour through doorway)
    np.array([-5.0,  7.0], dtype=np.float32),
    np.array([-6.5,  7.0], dtype=np.float32),
    np.array([-4.0,  7.0], dtype=np.float32),
    np.array([-5.5,  5.5], dtype=np.float32),
    np.array([-6.5,  4.5], dtype=np.float32),
    # Bedroom 2 (behind br2_wall_v at x=+2, needs detour through doorway)
    np.array([ 5.0,  7.0], dtype=np.float32),
    np.array([ 6.5,  7.0], dtype=np.float32),
    np.array([ 4.0,  7.0], dtype=np.float32),
    np.array([ 5.5,  5.5], dtype=np.float32),
    np.array([ 6.5,  4.5], dtype=np.float32),
    # Bedroom 3 (behind br3_wall_v at x=-2 lower half)
    np.array([-5.0, -7.0], dtype=np.float32),
    np.array([-6.5, -7.0], dtype=np.float32),
    np.array([-4.0, -7.0], dtype=np.float32),
    np.array([-5.5, -5.0], dtype=np.float32),
    # Kitchen (behind kitchen_wall_v at x=+3, lower)
    np.array([ 5.5, -7.0], dtype=np.float32),
    np.array([ 4.5, -7.0], dtype=np.float32),
    np.array([ 6.5, -6.5], dtype=np.float32),
    np.array([ 5.0, -6.5], dtype=np.float32),
    np.array([ 4.0, -5.0], dtype=np.float32),
    np.array([ 6.5, -4.5], dtype=np.float32),
]

# ── Complex obstacle goals: near pillars / furniture clusters ─────────────────
COMPLEX_GOALS = [
    np.array([-4.5,  0.5], dtype=np.float32),   # near hall_pillar_1
    np.array([-5.5,  0.5], dtype=np.float32),
    np.array([ 5.0,  1.5], dtype=np.float32),   # near hall_pillar_2
    np.array([ 5.5,  2.0], dtype=np.float32),
    np.array([-2.5,  1.5], dtype=np.float32),   # near hall_cylinder_obstacle
    np.array([-1.5,  2.0], dtype=np.float32),
    np.array([ 2.0,  0.5], dtype=np.float32),   # near sofa/coffee table cluster
    np.array([-0.5,  2.5], dtype=np.float32),
    np.array([ 2.5, -1.0], dtype=np.float32),   # near hall_box_obstacle
    np.array([ 0.5, -2.0], dtype=np.float32),
    np.array([-3.5,  2.5], dtype=np.float32),
    np.array([ 3.5, -3.5], dtype=np.float32),
    np.array([-4.5, -2.5], dtype=np.float32),
    np.array([ 4.5,  4.5], dtype=np.float32),
    np.array([-4.5,  4.5], dtype=np.float32),
]


def sample_random_goal(rng: np.random.Generator, robot_pos: np.ndarray,
                       min_dist: float = 1.5) -> np.ndarray:
    for _ in range(1000):
        x = rng.uniform(-7.5, 7.5)
        y = rng.uniform(-7.5, 7.5)
        if is_valid_goal(x, y, robot_pos, min_dist):
            return np.array([x, y], dtype=np.float32)
    return np.array([3.5, 6.0], dtype=np.float32)


def sample_curriculum_goal(rng: np.random.Generator,
                            robot_pos: np.ndarray) -> tuple[np.ndarray, str]:
    """Sample a goal from the mixed curriculum.

    Returns (goal_array, class_label) where class_label is one of:
      'wall_blocked', 'complex', 'random'
    """
    roll = rng.random()
    if roll < 0.40:
        # 40% wall-blocked
        valid = [g for g in WALL_BLOCKED_GOALS
                 if np.linalg.norm(g - robot_pos) >= 1.5]
        if valid:
            idx = rng.integers(len(valid))
            return valid[idx].copy(), "wall_blocked"
    elif roll < 0.70:
        # 30% complex obstacle
        valid = [g for g in COMPLEX_GOALS
                 if is_valid_goal(float(g[0]), float(g[1]), robot_pos)]
        if valid:
            idx = rng.integers(len(valid))
            return valid[idx].copy(), "complex"
    # 30% random (fallthrough from any failed sample above)
    return sample_random_goal(rng, robot_pos), "random"


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic evaluation goals — identical to all previous models (seed=42)
# ─────────────────────────────────────────────────────────────────────────────
def generate_eval_goals(num_episodes: int = PHASE4_EVAL_EPISODES) -> list[np.ndarray]:
    rng = np.random.default_rng(PHASE4_SEED)
    goals: list[np.ndarray] = []
    robot_start = np.array([0.0, 0.0])
    for _ in range(num_episodes):
        goal = sample_random_goal(rng, robot_start, min_dist=1.5)
        goals.append(goal)
    return goals


def generate_wall_eval_goals(num: int = PHASE4_WALL_EVAL_EPS) -> list[np.ndarray]:
    """Fixed set of wall-blocked scenarios for sub-metric measurement."""
    # Use first `num` from the WALL_BLOCKED_GOALS list — deterministic
    return [g.copy() for g in WALL_BLOCKED_GOALS[:num]]


# ─────────────────────────────────────────────────────────────────────────────
# Episode telemetry dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EpisodeResult:
    success: bool = False
    collision: bool = False
    timeout: bool = False
    steps: int = 0
    path_length: float = 0.0
    goal_class: str = "random"
    angular_history: list[float] = field(default_factory=list)
    oscillation_detected: bool = False
    no_turn_detected: bool = False   # angular vel never exceeded threshold


def _detect_oscillation(angular_history: list[float],
                         window: int = 20, threshold: float = 1.0) -> bool:
    """Detect oscillation: rapid direction reversals in a sliding window."""
    if len(angular_history) < window:
        return False
    for i in range(len(angular_history) - window):
        seg = angular_history[i:i + window]
        pos = sum(1 for v in seg if v > threshold)
        neg = sum(1 for v in seg if v < -threshold)
        if pos >= 3 and neg >= 3:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation harness — full 9-metric + sub-class breakdown
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(
    env,
    actor: nn.Module,
    eval_goals: list[np.ndarray],
    wall_eval_goals: list[np.ndarray],
    device: torch.device,
    step_label: int,
) -> dict[str, float]:
    """Run evaluation on fixed goals and return all required metrics."""
    actor.eval()

    def _run_episodes(goals: list[np.ndarray],
                      goal_class: str) -> list[EpisodeResult]:
        results = []
        for ep_idx, goal in enumerate(goals):
            obs, _ = env.reset(options={"target_position": goal})
            done = False
            result = EpisodeResult(goal_class=goal_class)
            last_pos = env.robot_pos.copy()

            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    dist = actor(obs_t)
                    action = torch.clamp(dist.mean, -1.0, 1.0).cpu().numpy()[0]

                obs, _, terminated, truncated, info = env.step(action)
                result.steps += 1
                result.path_length += float(np.linalg.norm(env.robot_pos - last_pos))
                last_pos = env.robot_pos.copy()
                result.angular_history.append(float(env.current_angular_vel))
                done = terminated or truncated

            result.success   = bool(info.get("is_success",   False))
            result.collision = bool(info.get("is_collision",  False))
            result.timeout   = not result.success and not result.collision

            # Oscillation & no-turn detection
            result.oscillation_detected = _detect_oscillation(result.angular_history)
            max_abs_angular = max((abs(v) for v in result.angular_history), default=0.0)
            result.no_turn_detected = max_abs_angular < 0.3   # never meaningfully turned

            status = ("SUCCESS" if result.success else
                      "COLLISION" if result.collision else "TIMEOUT")
            print(f"    [Eval {goal_class} ep{ep_idx+1:02d}] {status} | "
                  f"steps={result.steps} | path={result.path_length:.2f}m | "
                  f"osc={'Y' if result.oscillation_detected else 'N'} | "
                  f"no_turn={'Y' if result.no_turn_detected else 'N'}", flush=True)
            results.append(result)
        return results

    print(f"\n{'='*60}", flush=True)
    print(f"EVALUATION AT STEP {step_label:,}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Run deterministic benchmark (same 50 goals as all previous models) ──
    print(f"\n  [A] Deterministic benchmark ({len(eval_goals)} episodes):", flush=True)
    bench_results = _run_episodes(eval_goals, "benchmark")

    # ── Run wall-blocked sub-eval ────────────────────────────────────────────
    print(f"\n  [B] Wall-blocked sub-eval ({len(wall_eval_goals)} episodes):", flush=True)
    wall_results = _run_episodes(wall_eval_goals, "wall_blocked")

    actor.train()

    # ── Compute aggregate metrics ────────────────────────────────────────────
    def _metrics(res: list[EpisodeResult]) -> dict:
        n = len(res)
        if n == 0:
            return {"success": 0.0, "collision": 0.0, "timeout": 0.0,
                    "no_turn": 0.0, "oscillation": 0.0}
        return {
            "success":     sum(r.success    for r in res) / n * 100,
            "collision":   sum(r.collision  for r in res) / n * 100,
            "timeout":     sum(r.timeout    for r in res) / n * 100,
            "no_turn":     sum(r.no_turn_detected       for r in res),
            "oscillation": sum(r.oscillation_detected   for r in res),
        }

    bm = _metrics(bench_results)
    wl = _metrics(wall_results)

    # Wall bypass: wall-blocked episodes that succeeded
    wall_bypass_successes = sum(r.success for r in wall_results)

    metrics = {
        # ── Overall (deterministic benchmark) ──────────────────────────────
        "success_rate":                bm["success"],
        "collision_rate":              bm["collision"],
        "timeout_rate":                bm["timeout"],
        "no_turn_events":              bm["no_turn"],
        "oscillation_events":          bm["oscillation"],

        # ── Wall-blocked sub-class ──────────────────────────────────────────
        "wall_blocked_success_rate":   wl["success"],
        "wall_blocked_collision_rate": wl["collision"],
        "wall_blocked_timeout_rate":   wl["timeout"],
        "wall_bypass_successes":       float(wall_bypass_successes),

        # ── Path statistics ─────────────────────────────────────────────────
        "mean_path_length": np.mean([r.path_length for r in bench_results]),
        "mean_episode_steps": np.mean([r.steps for r in bench_results]),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print(f"  Overall    Success={metrics['success_rate']:.1f}%  "
          f"Collision={metrics['collision_rate']:.1f}%  "
          f"Timeout={metrics['timeout_rate']:.1f}%", flush=True)
    print(f"  Wall-Block Success={metrics['wall_blocked_success_rate']:.1f}%  "
          f"Bypass={wall_bypass_successes}/{len(wall_results)}", flush=True)
    print(f"  No-Turn events: {int(metrics['no_turn_events'])}  "
          f"Oscillation events: {int(metrics['oscillation_events'])}", flush=True)
    print(f"  Mean path: {metrics['mean_path_length']:.2f}m  "
          f"Mean steps: {metrics['mean_episode_steps']:.0f}", flush=True)
    print(f"{'─'*60}\n", flush=True)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CSV logger for hybrid_phase4
# ─────────────────────────────────────────────────────────────────────────────
_LOG_COLUMNS = [
    "total_steps", "update_idx",
    "avg_train_reward", "actor_loss", "critic_loss", "entropy",
    "success_rate", "collision_rate", "timeout_rate",
    "wall_blocked_success_rate", "wall_blocked_collision_rate",
    "wall_bypass_successes",
    "no_turn_events", "oscillation_events",
    "mean_path_length", "mean_episode_steps",
]


def _append_log(path: Path, row: dict) -> None:
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({col: row.get(col, "") for col in _LOG_COLUMNS})


# ─────────────────────────────────────────────────────────────────────────────
# TurtleBotEnv subclass (target_position override support)
# ─────────────────────────────────────────────────────────────────────────────
class TurtleBotEnv(GazeboMacNavEnv):
    def reset(self, seed=None, options=None):
        if options is not None and "target_position" in options:
            orig = self.safe_goals
            self.safe_goals = [np.array(options["target_position"])]
            obs, info = super().reset(seed=seed, options=options)
            self.safe_goals = orig
            return obs, info
        return super().reset(seed=seed, options=options)


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────
def train_phase4() -> None:
    # ── 1. Setup ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("PHASE 4 — HYBRID NAV2 + PPO TRAINING")
    print(f"{'='*60}")
    print(f"  Device       : {device}")
    print(f"  Total steps  : {PHASE4_TOTAL_STEPS:,}")
    print(f"  Eval interval: every ~{PHASE4_EVAL_INTERVAL:,} steps")
    print(f"  Source ckpt  : {CHAMPION_CKPT.name}")
    print(f"  Output dir   : {PHASE4_DIR}")
    print(f"  Curriculum   : 40% wall-blocked / 30% complex / 30% random")
    print(f"{'='*60}\n")

    # ── 2. Isolate Champion checkpoint ────────────────────────────────────────
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    seed_ckpt_path = PHASE4_DIR / "seed_champion.pt"
    if not seed_ckpt_path.exists():
        if not CHAMPION_CKPT.exists():
            raise FileNotFoundError(
                f"Champion checkpoint not found: {CHAMPION_CKPT}\n"
                f"Available checkpoints: {list(Path(PROJECT_ROOT / 'checkpoints').glob('*.pt'))}"
            )
        shutil.copy2(CHAMPION_CKPT, seed_ckpt_path)
        print(f"  [ckpt] Copied Champion → {seed_ckpt_path.name} (original untouched)")
    else:
        print(f"  [ckpt] Seed already present: {seed_ckpt_path.name}")

    # ── 3. Initialize environment and networks ────────────────────────────────
    env = TurtleBotEnv()
    env.debug_mode = False  # reduce noise during training
    obs_dim    = env.observation_space.shape[0]   # must be 50
    action_dim = env.action_space.shape[0]         # must be 2
    assert obs_dim == 50, f"Expected 50D obs, got {obs_dim}"
    print(f"  Observation dim: {obs_dim}  Action dim: {action_dim}\n")

    # Hyperparameters — must match CHECKPOINT_CONFIG
    ppo_epochs      = 10
    mini_batch_size = 64
    gamma           = 0.99
    gae_lambda      = 0.95
    clip_eps        = 0.2
    value_coef      = 0.5
    entropy_coef    = 0.003
    actor_lr        = 3e-4
    critic_lr       = 3e-4

    actor  = Actor(observation_dim=obs_dim, action_dim=action_dim, hidden_sizes=(256, 256)).to(device)
    critic = Critic(observation_dim=obs_dim, hidden_sizes=(256, 256)).to(device)
    buffer = RolloutBuffer(rollout_size=PHASE4_ROLLOUT_SIZE, observation_dim=obs_dim,
                           action_dim=action_dim, device=device)
    trainer = PPOTrainer(
        actor=actor, critic=critic, buffer=buffer, device=device,
        actor_lr=actor_lr, critic_lr=critic_lr,
        gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
        value_coef=value_coef, entropy_coef=entropy_coef,
        ppo_epochs=ppo_epochs, batch_size=mini_batch_size,
    )

    # ── 4. TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(PROJECT_ROOT / "runs" / "hybrid_phase4"))

    # ── 5. Eval goal sets (deterministic, same as all previous models) ────────
    eval_goals      = generate_eval_goals()
    wall_eval_goals = generate_wall_eval_goals()
    print(f"  Deterministic eval goals  : {len(eval_goals)}")
    print(f"  Wall-blocked eval episodes: {len(wall_eval_goals)}\n")

    # ── 6. Load from seed checkpoint ─────────────────────────────────────────
    rng = np.random.default_rng(7)

    # Try resuming an existing Phase 4 run first
    ckpt = load_checkpoint(PHASE4_DIR, current_config=CHECKPOINT_CONFIG)
    if ckpt is not None:
        print("  [ckpt] Resuming existing Phase 4 run...", flush=True)
    else:
        # First run — load from isolated seed copy
        ckpt = load_checkpoint(PHASE4_DIR / "..", current_config=None)  # don't auto-pick
        ckpt = None  # force load from seed

        import torch as _torch
        raw = _torch.load(seed_ckpt_path, map_location="cpu", weights_only=False)
        # Manually stitch weights (same logic as restore_from_checkpoint)
        from rl.checkpointing import _stitch_weights
        _stitch_weights(actor,  raw["actor"],  "actor")
        _stitch_weights(critic, raw["critic"], "critic")
        actor.to(device)
        critic.to(device)
        print("  [ckpt] Loaded and stitched weights from Champion seed", flush=True)
        ckpt = None  # training state starts fresh for Phase 4

    # State vars — always start step counter at 0 for this Phase 4 run
    total_steps       = 0
    update_idx        = 0
    best_p4_success   = -1.0
    best_p4_metrics: dict = {}
    episode_rewards: list[float] = []
    episode_reward    = 0.0
    _shutdown_requested = False

    # Resume from an existing Phase 4 checkpoint if available
    existing = load_checkpoint(PHASE4_DIR, current_config=CHECKPOINT_CONFIG)
    if existing is not None:
        state = restore_from_checkpoint(
            existing, actor=actor, critic=critic,
            actor_optimizer=trainer.actor_optimizer,
            critic_optimizer=trainer.critic_optimizer,
            rng=rng, device=device,
        )
        total_steps     = state["total_steps"]
        update_idx      = state["update_idx"]
        best_p4_success = state["best_success_rate"]
        best_p4_metrics = state["best_eval_metrics"]
        print(f"\n  [ckpt] Resumed Phase 4 at step {total_steps:,} / update {update_idx}")
        if total_steps >= PHASE4_TOTAL_STEPS:
            print(f"\n  ✋ Phase 4 already completed ({total_steps:,} steps). Exiting.")
            env.close(); writer.close()
            return

    # ── 7. SIGINT / SIGTERM graceful shutdown ─────────────────────────────────
    def _shutdown(signum, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            return
        _shutdown_requested = True
        print("\n\nSignal received — saving emergency checkpoint...", flush=True)
        try:
            ckpt_e = build_checkpoint(
                actor=actor, critic=critic,
                actor_optimizer=trainer.actor_optimizer,
                critic_optimizer=trainer.critic_optimizer,
                update_idx=update_idx, total_steps=total_steps,
                best_success_rate=best_p4_success,
                best_eval_metrics=best_p4_metrics,
                consecutive_success_runs=0, rng=rng,
            )
            save_checkpoint(ckpt_e, PHASE4_DIR)
            print(f"  Emergency checkpoint saved at step {total_steps:,}", flush=True)
        except Exception as exc:
            print(f"  Emergency save FAILED: {exc}", flush=True)
        try:
            env.close(); writer.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 8. Initial reset ──────────────────────────────────────────────────────
    robot_start = np.array([0.0, 0.0])
    goal, goal_class = sample_curriculum_goal(rng, robot_start)
    observation, _ = env.reset(options={"target_position": goal})

    # Track per-rollout curriculum class counts for logging
    class_counts: dict[str, int] = {"wall_blocked": 0, "complex": 0, "random": 0}

    print(f"\n{'='*60}")
    print(f"TRAINING START — target: {PHASE4_TOTAL_STEPS:,} steps")
    print(f"{'='*60}\n")

    # ── 9. Main PPO loop ──────────────────────────────────────────────────────
    # Evaluate before training begins (step=0 baseline)
    _eval_metrics = evaluate_policy(env, actor, eval_goals, wall_eval_goals, device, total_steps)
    _append_log(PHASE4_LOG, {
        "total_steps": total_steps, "update_idx": update_idx,
        "avg_train_reward": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
        **_eval_metrics,
    })
    for k, v in _eval_metrics.items():
        writer.add_scalar(f"Phase4/{k}", v, global_step=total_steps)

    next_eval_at = PHASE4_EVAL_INTERVAL

    while total_steps < PHASE4_TOTAL_STEPS:
        # ── Collect one rollout ───────────────────────────────────────────────
        for _ in range(PHASE4_ROLLOUT_SIZE):
            total_steps += 1

            action, log_prob, value = trainer.select_action(observation)
            env_action = torch.clamp(action, -1.0, 1.0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(env_action)

            # Bootstrap truncated episodes
            if truncated and not terminated:
                with torch.no_grad():
                    next_t = torch.as_tensor(next_obs, dtype=torch.float32,
                                              device=device).unsqueeze(0)
                    reward += gamma * critic(next_t).item()

            done = terminated or truncated
            trainer.store_transition(
                observation=torch.as_tensor(observation, dtype=torch.float32),
                action=action, reward=reward, done=done,
                log_prob=log_prob, value=value,
            )
            episode_reward += reward
            observation = next_obs

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                goal, goal_class = sample_curriculum_goal(rng, env.robot_pos)
                class_counts[goal_class] = class_counts.get(goal_class, 0) + 1
                observation, _ = env.reset(options={"target_position": goal})

            # Hard stop check inside rollout
            if total_steps >= PHASE4_TOTAL_STEPS:
                break

        # ── PPO update ────────────────────────────────────────────────────────
        trainer.finish_rollout(torch.as_tensor(observation, dtype=torch.float32))
        returns_np = buffer.returns[:PHASE4_ROLLOUT_SIZE].cpu().numpy()
        values_np  = buffer.values[:PHASE4_ROLLOUT_SIZE].cpu().numpy()
        expl_var   = 1.0 - np.var(returns_np - values_np) / (np.var(returns_np) + 1e-8)

        ppo_metrics = trainer.update()
        update_idx += 1

        avg_rew = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0

        # TensorBoard
        writer.add_scalar("Phase4/Loss/actor",      ppo_metrics["actor_loss"],  total_steps)
        writer.add_scalar("Phase4/Loss/critic",     ppo_metrics["critic_loss"], total_steps)
        writer.add_scalar("Phase4/Policy/entropy",  ppo_metrics["entropy"],     total_steps)
        writer.add_scalar("Phase4/Value/expl_var",  expl_var,                   total_steps)
        writer.add_scalar("Phase4/Reward/avg",      avg_rew,                    total_steps)
        writer.add_scalar("Phase4/Curriculum/wall_blocked",
                          class_counts.get("wall_blocked", 0), total_steps)

        print(
            f"[Update {update_idx:04d}] Steps: {total_steps:,}/{PHASE4_TOTAL_STEPS:,} | "
            f"ActorLoss: {ppo_metrics['actor_loss']:.4f} | "
            f"CriticLoss: {ppo_metrics['critic_loss']:.4f} | "
            f"Entropy: {ppo_metrics['entropy']:.4f} | "
            f"ExplVar: {expl_var:.3f} | "
            f"AvgRew(10): {avg_rew:.2f} | "
            f"Curriculum: WB={class_counts.get('wall_blocked',0)} "
            f"CX={class_counts.get('complex',0)} "
            f"RN={class_counts.get('random',0)}",
            flush=True,
        )

        # ── Periodic checkpoint ───────────────────────────────────────────────
        ckpt_out = build_checkpoint(
            actor=actor, critic=critic,
            actor_optimizer=trainer.actor_optimizer,
            critic_optimizer=trainer.critic_optimizer,
            update_idx=update_idx, total_steps=total_steps,
            best_success_rate=best_p4_success,
            best_eval_metrics=best_p4_metrics,
            consecutive_success_runs=0, rng=rng,
        )
        save_checkpoint(ckpt_out, PHASE4_DIR)

        # ── Periodic evaluation ───────────────────────────────────────────────
        if total_steps >= next_eval_at or total_steps >= PHASE4_TOTAL_STEPS:
            eval_m = evaluate_policy(
                env, actor, eval_goals, wall_eval_goals, device, total_steps
            )
            for k, v in eval_m.items():
                writer.add_scalar(f"Phase4/{k}", v, global_step=total_steps)

            _append_log(PHASE4_LOG, {
                "total_steps":       total_steps,
                "update_idx":        update_idx,
                "avg_train_reward":  avg_rew,
                "actor_loss":        ppo_metrics["actor_loss"],
                "critic_loss":       ppo_metrics["critic_loss"],
                "entropy":           ppo_metrics["entropy"],
                **eval_m,
            })

            # Track best Phase 4 model (DOES NOT overwrite Champion)
            if eval_m["success_rate"] > best_p4_success:
                best_p4_success = eval_m["success_rate"]
                best_p4_metrics = {**eval_m, "total_steps": total_steps,
                                   "update_idx": update_idx}
                ckpt_best = build_checkpoint(
                    actor=actor, critic=critic,
                    actor_optimizer=trainer.actor_optimizer,
                    critic_optimizer=trainer.critic_optimizer,
                    update_idx=update_idx, total_steps=total_steps,
                    best_success_rate=best_p4_success,
                    best_eval_metrics=best_p4_metrics,
                    consecutive_success_runs=0, rng=rng,
                )
                save_checkpoint(ckpt_best, PHASE4_DIR, is_best=True)
                print(f"  [Phase4] New best success rate: {best_p4_success:.1f}% "
                      f"→ hybrid_phase4/best_success.pt", flush=True)
                print(f"  NOTE: Champion checkpoints/{CHAMPION_CKPT.name} is UNCHANGED.",
                      flush=True)

            next_eval_at += PHASE4_EVAL_INTERVAL

        # Hard stop after 50k
        if total_steps >= PHASE4_TOTAL_STEPS:
            break

    # ── 10. Final checkpoint + summary ────────────────────────────────────────
    ckpt_final = build_checkpoint(
        actor=actor, critic=critic,
        actor_optimizer=trainer.actor_optimizer,
        critic_optimizer=trainer.critic_optimizer,
        update_idx=update_idx, total_steps=total_steps,
        best_success_rate=best_p4_success,
        best_eval_metrics=best_p4_metrics,
        consecutive_success_runs=0, rng=rng,
    )
    save_checkpoint(ckpt_final, PHASE4_DIR, is_final=True)

    print(f"\n{'='*60}", flush=True)
    print("PHASE 4 TRAINING COMPLETE — 50,000 STEPS REACHED", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Best Phase 4 success rate : {best_p4_success:.1f}%", flush=True)
    print(f"  Best at step              : {best_p4_metrics.get('total_steps', '?'):,}", flush=True)
    print(f"  Wall-blocked success      : {best_p4_metrics.get('wall_blocked_success_rate', 0):.1f}%",
          flush=True)
    print(f"  Wall bypass successes     : {int(best_p4_metrics.get('wall_bypass_successes', 0))}/"
          f"{PHASE4_WALL_EVAL_EPS}", flush=True)
    print(f"\n  Output directory          : {PHASE4_DIR}", flush=True)
    print(f"  Training log              : {PHASE4_LOG}", flush=True)
    print(f"\n  Champion ({CHAMPION_CKPT.name}) : UNCHANGED ✅", flush=True)
    print(f"\n  ⏸  WAITING FOR REVIEW BEFORE ANY FURTHER ACTION.", flush=True)
    print(f"{'='*60}\n", flush=True)

    env.close()
    writer.close()


if __name__ == "__main__":
    train_phase4()
