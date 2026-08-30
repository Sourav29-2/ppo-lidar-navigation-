"""Stage C — Mixed Curriculum Experiment (100k steps)"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from shapely.geometry import LineString, box, Point
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

from gazebo_nav_env import GazeboMacNavEnv
from rl.actor import Actor
from rl.critic import Critic
from rl.ppo_trainer import PPOTrainer
from rl.rollout_buffer import RolloutBuffer
from rl.checkpointing import build_checkpoint

# Bounding box obstacles from obstacles.world
BOX_OBSTACLES = [
    (0.0, 8.0, 16.0, 0.2),    # North wall
    (0.0, -8.0, 16.0, 0.2),   # South wall
    (8.0, 0.0, 0.2, 16.0),    # East wall
    (-8.0, 0.0, 0.2, 16.0),   # West wall
    (-2.0, 5.5, 0.2, 5.0),    # br1_wall_v
    (-5.6, 3.0, 4.8, 0.2),    # br1_wall_h
    (2.0, 5.5, 0.2, 5.0),     # br2_wall_v
    (5.6, 3.0, 4.8, 0.2),     # br2_wall_h
    (-2.0, -5.5, 0.2, 5.0),   # br3_wall_v
    (-5.6, -3.0, 4.8, 0.2),   # br3_wall_h
    (3.0, -5.5, 0.2, 5.0),    # kitchen_wall_v
    (6.1, -3.0, 3.8, 0.2),    # kitchen_wall_h
    (-6.0, 6.0, 2.0, 1.8),    # br1_bed
    (6.0, 6.0, 2.0, 1.8),     # br2_bed
    (-6.0, -6.0, 2.0, 1.8),   # br3_bed
    (7.0, -5.5, 1.0, 3.5),    # kitchen_counter
    (0.0, 1.5, 3.0, 0.8),     # living_sofa
    (1.2, 1.2, 1.2, 0.6),     # coffee_table
    (1.0, -1.5, 0.4, 0.4),    # hall_box_obstacle
]

CYLINDER_OBSTACLES = [
    (-5.0, 0.0, 0.25),        # hall_pillar_1
    (4.5, 1.0, 0.25),         # hall_pillar_2
    (-2.0, 1.0, 0.20),        # hall_cylinder_obstacle
]

class TurtleBotEnv(GazeboMacNavEnv):
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if options is not None and "target_position" in options:
            orig_safe_goals = self.safe_goals
            self.safe_goals = [np.array(options["target_position"])]
            obs, info = super().reset(seed=seed, options=options)
            self.safe_goals = orig_safe_goals
            return obs, info
        return super().reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 1. COLLISION PENALTY
        if info.get('is_collision', False):
            # Base environment already applies -100.0, we add -50.0 to make it slightly stronger.
            reward -= 50.0
            
        # 2. CLEARANCE SHAPING
        try:
            min_laser = float(np.min(self.laser_ranges))
            if min_laser < 0.25:
                # Dangerously close: penalty up to -0.5
                reward -= (0.25 - min_laser) * 2.0
            elif min_laser < 0.5:
                # Moderate clearance: small bonus up to +0.25
                reward += (min_laser - 0.25) * 1.0
            else:
                # Safe clearance: plateaued bonus
                reward += 0.25
        except Exception:
            pass
            
        return obs, reward, terminated, truncated, info

def is_valid_goal(x: float, y: float, robot_pos: np.ndarray, min_dist: float = 1.5, margin: float = 0.45) -> bool:
    if np.linalg.norm(np.array([x, y]) - robot_pos) < min_dist:
        return False
    for cx, cy, sx, sy in BOX_OBSTACLES:
        left, right = cx - sx/2.0 - margin, cx + sx/2.0 + margin
        bottom, top = cy - sy/2.0 - margin, cy + sy/2.0 + margin
        if left <= x <= right and bottom <= y <= top:
            return False
    for cx, cy, r in CYLINDER_OBSTACLES:
        dist = np.linalg.norm(np.array([x, y]) - np.array([cx, cy]))
        if dist <= r + margin:
            return False
    return True

def sample_random_goal(rng: np.random.Generator, robot_pos: np.ndarray, min_dist: float = 1.5) -> np.ndarray:
    for _ in range(1000):
        x = rng.uniform(-7.5, 7.5)
        y = rng.uniform(-7.5, 7.5)
        if is_valid_goal(x, y, robot_pos, min_dist):
            return np.array([x, y], dtype=np.float32)
    return np.array([3.5, 6.0], dtype=np.float32)

def classify_goal_difficulty(sx, sy, gx, gy):
    path_line = LineString([(sx, sy), (gx, gy)])
    blocking_obstacles = 0
    
    for cx, cy, szx, szy in BOX_OBSTACLES:
        b = box(cx - szx/2, cy - szy/2, cx + szx/2, cy + szy/2)
        if path_line.intersects(b):
            blocking_obstacles += 1
                
    for cx, cy, r in CYLINDER_OBSTACLES:
        c = Point(cx, cy).buffer(r)
        if path_line.intersects(c):
            blocking_obstacles += 1
            
    direct_path_blocked = blocking_obstacles > 0
    
    if not direct_path_blocked:
        return "EASY"
    elif direct_path_blocked and blocking_obstacles == 1:
        return "MODERATE"
    elif blocking_obstacles >= 3:
        return "COMPLEX"
    else:
        return "WALL-BLOCKED"

def sample_curriculum_goal(rng: np.random.Generator, robot_pos: np.ndarray, min_dist: float = 1.5) -> tuple[np.ndarray, str]:
    r = rng.random()
    if r < 0.2: target_category = "EASY"
    elif r < 0.4: target_category = "MODERATE"
    elif r < 0.8: target_category = "WALL-BLOCKED"
    else: target_category = "COMPLEX"
    
    for _ in range(1000):
        x = rng.uniform(-7.5, 7.5)
        y = rng.uniform(-7.5, 7.5)
        if is_valid_goal(x, y, robot_pos, min_dist):
            cat = classify_goal_difficulty(robot_pos[0], robot_pos[1], x, y)
            if cat == target_category:
                return np.array([x, y], dtype=np.float32), cat
                
    # Fallback if impossible
    fallback = sample_random_goal(rng, robot_pos, min_dist)
    cat = classify_goal_difficulty(robot_pos[0], robot_pos[1], fallback[0], fallback[1])
    return fallback, cat

def generate_eval_goals(rng) -> list[tuple[np.ndarray, str]]:
    eval_goals = []
    robot_start = np.array([0.0, 0.0])
    for _ in range(10): eval_goals.append((sample_curriculum_goal(rng, robot_start)[0], "EASY"))
    for _ in range(10): eval_goals.append((sample_curriculum_goal(rng, robot_start)[0], "MODERATE"))
    for _ in range(10): eval_goals.append((sample_curriculum_goal(rng, robot_start)[0], "WALL-BLOCKED"))
    for _ in range(10): eval_goals.append((sample_curriculum_goal(rng, robot_start)[0], "COMPLEX"))
    return eval_goals

def evaluate_policy_detailed(
    env: TurtleBotEnv,
    actor: nn.Module,
    eval_goals: list[tuple[np.ndarray, str]],
    device: torch.device,
) -> dict:
    actor.eval()
    
    results = {
        "total": 0, "success": 0, "collision": 0, "timeout": 0,
        "stagnation_events": 0,
        "corners": {
            "EASY": {"total": 0, "success": 0},
            "MODERATE": {"total": 0, "success": 0},
            "WALL-BLOCKED": {"total": 0, "success": 0},
            "COMPLEX": {"total": 0, "success": 0}
        },
        "front_collisions": 0,
        "no_turns": 0,
        "successful_avoidances": 0,
        "critical_recoveries": 0,
        "angular_vel_under_045": [],
        "min_clearance": []
    }
    
    num_episodes = len(eval_goals)
    
    for ep in range(num_episodes):
        goal, g_type = eval_goals[ep]
        obs, info = env.reset(options={"target_position": goal})
        done = False
        
        ep_min_clearance = float('inf')
        in_danger = False
        danger_angular_vels = []
        max_ang_vel_in_danger = 0.0
        
        ep_stagnation_events = 0
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                distribution = actor(obs_tensor)
                action = torch.clamp(distribution.mean, -1.0, 1.0).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            
            # STAGNATION TRACKING
            if info.get("stagnation", 0.0) < -0.1:
                ep_stagnation_events += 1
                
            lidar_sectors = obs[:36] * 12.0
            min_c = float(np.min(lidar_sectors))
            if min_c < ep_min_clearance:
                ep_min_clearance = min_c
                
            if min_c < 0.45:
                in_danger = True
                danger_angular_vels.append(abs(env.current_angular_vel))
                if abs(env.current_angular_vel) > max_ang_vel_in_danger:
                    max_ang_vel_in_danger = abs(env.current_angular_vel)
            
            done = terminated or truncated

        is_success = bool(info.get("is_success", False))
        is_collision = bool(info.get("is_collision", False))
        
        results["total"] += 1
        results["corners"][g_type]["total"] += 1
        results["min_clearance"].append(ep_min_clearance)
        results["stagnation_events"] += ep_stagnation_events
        
        if in_danger:
            results["angular_vel_under_045"].extend(danger_angular_vels)
            
        if is_success:
            results["success"] += 1
            results["corners"][g_type]["success"] += 1
            if in_danger:
                results["successful_avoidances"] += 1
            if ep_min_clearance < 0.35:
                results["critical_recoveries"] += 1
        elif is_collision:
            results["collision"] += 1
            
            # Did it turn? (No-turn logic)
            if in_danger and max_ang_vel_in_danger < 0.5:
                results["no_turns"] += 1
                
            # Front collision logic
            front_min = min(obs[35]*12, obs[0]*12, obs[1]*12)
            if front_min < 0.2:
                results["front_collisions"] += 1
        else:
            results["timeout"] += 1

    actor.train()
    return results

def run_fine_tuning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = TurtleBotEnv(num_scan_features=360)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # === HYPERPARAMETERS ===
    TOTAL_FT_STEPS     = 100_000
    rollout_size       = 2048
    ppo_epochs         = 10
    mini_batch_size    = 64
    gamma              = 0.99
    gae_lambda         = 0.95
    clip_eps           = 0.2
    value_coef         = 0.5
    entropy_coef       = 0.003
    eval_every_updates = 5
    save_every_updates = 5
    next_milestone     = 60000

    # Networks, buffer, trainer
    actor  = Actor(observation_dim=obs_dim, action_dim=action_dim, hidden_sizes=(256, 256))
    critic = Critic(observation_dim=obs_dim, hidden_sizes=(256, 256))
    buffer = RolloutBuffer(rollout_size=rollout_size, observation_dim=obs_dim, action_dim=action_dim, device=device)
    trainer = PPOTrainer(
        actor=actor, critic=critic, buffer=buffer, device=device,
        actor_lr=3e-5, critic_lr=3e-5,
        gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
        value_coef=value_coef, entropy_coef=entropy_coef,
        ppo_epochs=ppo_epochs, batch_size=mini_batch_size,
    )

    # Load starting checkpoint (Oscillation Latest)
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    start_path = checkpoints_dir / "hybrid_curriculum_latest.pt"
    if not start_path.exists():
        raise RuntimeError(f"Starting checkpoint not found at {start_path}")
        
    print(f"Loading starting checkpoint: {start_path}")
    checkpoint = torch.load(start_path, map_location=device, weights_only=True)
    
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    trainer.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    if "critic_optimizer" in checkpoint:
        trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    # Overwrite LR explicitly
    for param_group in trainer.actor_optimizer.param_groups:
        param_group["lr"] = 3e-5
    for param_group in trainer.critic_optimizer.param_groups:
        param_group["lr"] = 3e-5

    writer = SummaryWriter(log_dir="runs/stage_d_safe_collision")
    rng = np.random.default_rng(42)
    eval_goals = generate_eval_goals(rng)

    # print("\n========================================")
    # print("CHAMPION BASELINE METRICS")
    # print("========================================")
    # print(f"Success: {base_metrics['success']/base_metrics['total']*100:.1f}%")
    # print(f"Collision: {base_metrics['collision']/base_metrics['total']*100:.1f}%")
    # print(f"Timeout: {base_metrics['timeout']/base_metrics['total']*100:.1f}%")
    # print(f"Stagnation Events: {base_metrics['stagnation_events']}")
    # print("========================================\n")

    ft_steps = 0
    update_idx = 0
    
    _shutdown_requested = False
    def _signal_handler(signum, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested: return
        _shutdown_requested = True
        print("\\nEmergency save...")
        sys.exit(0)
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    robot_start = np.array([0.0, 0.0])
    goal, _ = sample_curriculum_goal(rng, robot_start, min_dist=1.5)
    observation, _ = env.reset(options={"target_position": goal})
    is_first_ep_reset = False
    actual_goal_distribution = []
    print("=== STARTING FINE TUNING LOOP ===", flush=True)

    # ── Main Fine-Tuning Loop ────────────────────────────────────────
    while ft_steps < TOTAL_FT_STEPS:
        current_lr = 3e-5 - (3e-5 - 3e-6) * (ft_steps / TOTAL_FT_STEPS)
        for param_group in trainer.actor_optimizer.param_groups:
            param_group["lr"] = current_lr
        for param_group in trainer.critic_optimizer.param_groups:
            param_group["lr"] = current_lr
            
        stagnation_penalties = []
        for _ in range(rollout_size):
            ft_steps += 1
            
            action, log_prob, value = trainer.select_action(observation)
            env_action = torch.clamp(action, -1.0, 1.0).cpu().numpy()
            next_observation, reward, terminated, truncated, info = env.step(env_action)
            
            if info.get("stagnation", 0.0) < -0.1:
                stagnation_penalties.append(info["stagnation"])
            
            if truncated and not terminated:
                with torch.no_grad():
                    next_obs_tensor = torch.as_tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)
                    next_value = critic(next_obs_tensor).item()
                    reward += gamma * next_value

            done = terminated or truncated

            trainer.store_transition(
                observation=torch.as_tensor(observation, dtype=torch.float32),
                action=action, reward=reward, done=done, log_prob=log_prob, value=value
            )

            observation = next_observation

            if done:
                if is_first_ep_reset:
                    goal, cat = sample_curriculum_goal(np.random.default_rng(), np.array([0.0, 0.0]))
                    observation, info = env.reset(options={"target_position": goal})
                    is_first_ep_reset = False
                else:
                    robot_pos = observation[:2] if isinstance(observation, np.ndarray) else np.array([0.0, 0.0])
                    goal, cat = sample_curriculum_goal(np.random.default_rng(), robot_pos)
                    observation, info = env.reset(options={"target_position": goal})
                    
                actual_goal_distribution.append({
                    "step": ft_steps,
                    "category": cat,
                    "goal_x": goal[0],
                    "goal_y": goal[1]
                })

            if np.isnan(reward) or np.isinf(reward):
                print(f"NaN reward detected at step {ft_steps}. Stopping.")
                return

        trainer.finish_rollout(torch.as_tensor(observation, dtype=torch.float32))
        metrics = trainer.update()
        update_idx += 1
        
        if np.isnan(metrics['actor_loss']) or np.isnan(metrics['critic_loss']) or metrics['critic_loss'] > 1e6:
            print(f"Numerical instability detected (Losses: Actor {metrics['actor_loss']}, Critic {metrics['critic_loss']}). Stopping.")
            return

        writer.add_scalar("StageC/actor_loss", metrics["actor_loss"], global_step=ft_steps)
        writer.add_scalar("StageC/critic_loss", metrics["critic_loss"], global_step=ft_steps)
        writer.add_scalar("StageC/learning_rate", current_lr, global_step=ft_steps)
        writer.add_scalar("StageC/stagnation_penalty_total", sum(stagnation_penalties), global_step=ft_steps)
        
        print(f"[Update {update_idx:03d}] Steps: {ft_steps}/{TOTAL_FT_STEPS} | LR: {current_lr:.2e} | "
              f"Stag Penalty: {sum(stagnation_penalties):.1f} | "
              f"Actor Loss: {metrics['actor_loss']:.4f} | Critic Loss: {metrics['critic_loss']:.4f}")

        # Checkpoint every 10k
        if update_idx % save_every_updates == 0 or ft_steps >= TOTAL_FT_STEPS:
            ckpt_data = build_checkpoint(
                actor=actor, critic=critic, actor_optimizer=trainer.actor_optimizer, critic_optimizer=trainer.critic_optimizer,
                update_idx=update_idx, total_steps=ft_steps, best_success_rate=0.0, best_eval_metrics={}, consecutive_success_runs=0, rng=rng,
            )
            torch.save(ckpt_data, checkpoints_dir / "safe_collision_latest.pt")
            print(f"[{ft_steps}/{TOTAL_FT_STEPS}] Saved safe_collision_latest.pt")
            
            # Custom milestone saves
            if ft_steps >= next_milestone and next_milestone <= TOTAL_FT_STEPS:
                milestone = next_milestone // 1000
                torch.save(ckpt_data, checkpoints_dir / f"safe_collision_{milestone}k.pt")
                print(f"[{ft_steps}/{TOTAL_FT_STEPS}] Saved safe_collision_{milestone}k.pt")
                next_milestone += 10000
                
        # Evaluate every 10k
        if update_idx % eval_every_updates == 0 or ft_steps >= TOTAL_FT_STEPS:
            print(f"\\nRunning Detailed Evaluation at step {ft_steps}...")
            res = evaluate_policy_detailed(env, actor, eval_goals, device)
            
            suc_rate = (res['success'] / res['total']) * 100.0
            col_rate = (res['collision'] / res['total']) * 100.0
            to_rate = (res['timeout'] / res['total']) * 100.0
            
            writer.add_scalar("Eval/SuccessRate", suc_rate, global_step=ft_steps)
            writer.add_scalar("Eval/CollisionRate", col_rate, global_step=ft_steps)
            writer.add_scalar("Eval/TimeoutRate", to_rate, global_step=ft_steps)
            writer.add_scalar("Eval/StagnationEvents", res['stagnation_events'], global_step=ft_steps)
            
            print("=" * 40)
            print(f"EVALUATION — {ft_steps // 1000}k")
            print("=" * 40)
            print(f"Overall:\\nSuccess: {suc_rate:.1f}%\\nCollision: {col_rate:.1f}%\\nTimeout: {to_rate:.1f}%")
            print(f"Stagnation Events: {res['stagnation_events']}")
            print("========================================\\n")

    env.close()
    writer.close()

if __name__ == "__main__":
    run_fine_tuning()
