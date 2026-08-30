"""
tests/diagnose_reward.py

Short Gazebo diagnostic (1-3 episodes) with debug_mode=True.
Checks whether backward exploitation or spinning exploitation
generates reward without meaningful navigation progress.

Run with:  pixi run python3 -u tests/diagnose_reward.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

import rclpy
from train import TurtleBotEnv, generate_eval_goals, Actor

NUM_EPISODES = 3

def run():
    rclpy.init(args=None)
    env = TurtleBotEnv(num_scan_features=360)
    env.debug_mode = True           # ← enable full reward component logging
    env.debug_log_interval = 50     # print every 50 steps (not too noisy)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Fresh actor — no checkpoint loaded
    actor = Actor(observation_dim=obs_dim, action_dim=action_dim, hidden_sizes=(256, 256))
    actor.eval()

    eval_goals = generate_eval_goals(NUM_EPISODES)

    # Aggregate stats across episodes
    agg = dict(
        total_reward=[], path_len=[], ep_steps=[],
        progress=[], clearance=[], turning=[], time_pen=[],
        inflation=[], reverse=[], success=0, collisions=0,
        reverse_attempts=[], reverse_rewarded=[],
        high_angular=[],
    )

    for ep in range(NUM_EPISODES):
        goal = eval_goals[ep]
        print(f"\n{'='*56}")
        print(f"DIAGNOSTIC EPISODE {ep+1}/{NUM_EPISODES} | Goal: {goal}")
        print(f"{'='*56}")

        obs, _ = env.reset(options={"target_position": goal})
        done = False
        ep_reward = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist = actor(obs_t)
                action = torch.clamp(dist.mean, -1.0, 1.0).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        # Collect episode-level diagnostics from env counters
        steps = max(env.step_count, 1)
        agg["total_reward"].append(env._diag_total_reward)
        agg["path_len"].append(info["path_length"])
        agg["ep_steps"].append(env.step_count)
        agg["progress"].append(env._diag_total_progress / steps)
        agg["clearance"].append(env._diag_total_clearance / steps)
        agg["turning"].append(env._diag_total_turning / steps)
        agg["time_pen"].append(env._diag_total_time / steps)
        agg["inflation"].append(env._diag_total_inflation / steps)
        agg["reverse"].append(env._diag_total_reverse / steps)
        agg["reverse_attempts"].append(env._diag_reverse_attempts)
        agg["reverse_rewarded"].append(env._diag_reverse_rewarded)
        agg["high_angular"].append(env._diag_high_angular_steps)
        if info.get("is_success"):
            agg["success"] += 1
        if info.get("is_collision"):
            agg["collisions"] += 1

    env.close()

    # ─── Final aggregate summary ────────────────────────────────────────────
    n = len(agg["total_reward"])
    print(f"\n{'='*56}")
    print(f"FINAL GAZEBO DIAGNOSTIC REPORT ({NUM_EPISODES} episodes)")
    print(f"{'='*56}")
    print(f"  1. Avg Total Reward:            {np.mean(agg['total_reward']):+.4f}")
    print(f"  2. Avg Progress Reward/step:    {np.mean(agg['progress']):+.4f}")
    print(f"  3. Avg Clearance Reward/step:   {np.mean(agg['clearance']):+.4f}")
    print(f"  4. Avg Turning Penalty/step:    {np.mean(agg['turning']):+.4f}")
    print(f"  5. Avg Time Penalty/step:       {np.mean(agg['time_pen']):+.4f}")
    print(f"  6. Avg Inflation Penalty/step:  {np.mean(agg['inflation']):+.4f}")
    print(f"  7. Total Reverse Recovery:      {np.sum(agg['reverse']) * np.mean(agg['ep_steps']):.4f}")
    print(f"  8. Reverse Attempts (total):    {sum(agg['reverse_attempts'])}")
    print(f"  9. Reverse Rewarded (total):    {sum(agg['reverse_rewarded'])}")
    print(f" 10. High Angular-vel Steps:      {sum(agg['high_angular'])}")
    print(f" 11. Collisions:                  {agg['collisions']}")
    print(f" 12. Successes:                   {agg['success']}")
    spinning_observed = sum(agg["high_angular"]) > (0.3 * sum(agg["ep_steps"]))
    backward_exploited = (sum(agg["reverse_attempts"]) > 0
                          and sum(agg["reverse_rewarded"]) == 0)
    print(f" 13. Continuous Spinning Observed: {spinning_observed}")
    print(f" 14. Backward Exploitation Observed: {backward_exploited}")
    print(f"{'='*56}")

if __name__ == "__main__":
    run()
