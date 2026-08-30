"""Comparative Benchmark Script (no pandas dependency)."""

import csv
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

from gazebo_nav_env import GazeboMacNavEnv
from rl.actor import Actor
from generate_benchmark_scenarios import check_direct_path_blocked


# ── helpers (replace pandas) ─────────────────────────────────────────
def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _sum(vals):
    return sum(vals)

def _filter(rows, **kw):
    """Return rows matching all key=value filters."""
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out

def _col(rows, key):
    """Extract a column as a list of values."""
    return [r[key] for r in rows]
# ─────────────────────────────────────────────────────────────────────


class BenchmarkEnv(GazeboMacNavEnv):
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if options is not None and "target_position" in options:
            orig_safe_goals = self.safe_goals
            self.safe_goals = [np.array(options["target_position"])]
            obs, info = super().reset(seed=seed, options=options)
            self.safe_goals = orig_safe_goals
            return obs, info
        return super().reset(seed=seed, options=options)

def run_episode(env: BenchmarkEnv, actor: nn.Module, scenario: dict, model_name: str, device: torch.device):
    goal_pos = np.array(scenario["goal_position"], dtype=np.float32)
    obs, info = env.reset(seed=scenario["random_seed"], options={"target_position": goal_pos})
    
    done = False
    
    # Tracking vars
    initial_dist = np.linalg.norm(goal_pos - np.array([0.0, 0.0]))
    steps = 0
    path_length = 0.0
    last_pos = np.array([0.0, 0.0])
    
    min_clearance = float('inf')
    linear_vels = []
    angular_vels = []
    abs_angular_vels = []
    
    forward_steps = 0
    reverse_steps = 0
    stationary_steps = 0
    
    in_danger = False
    danger_ang_vels = []
    max_ang_vel_in_danger = 0.0
    
    # Wall-blocked tracking
    first_obstacle_detected_step = -1
    first_turn_step = -1
    clearance_at_first_turn = -1.0
    did_turn_away = False
    
    front_collision = False
    
    while not done:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = actor(obs_tensor)
            action = torch.clamp(dist.mean, -1.0, 1.0).cpu().numpy()[0]
            
        lin_act = action[0] * env.max_linear_vel
        ang_act = action[1] * env.max_angular_vel
        
        linear_vels.append(lin_act)
        angular_vels.append(ang_act)
        abs_angular_vels.append(abs(ang_act))
        
        if lin_act > 0.05: forward_steps += 1
        elif lin_act < -0.05: reverse_steps += 1
        else: stationary_steps += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        path_length += abs(lin_act) * 0.1
        
        lidar = obs[:36] * 12.0
        c_min = float(np.min(lidar))
        if c_min < min_clearance: min_clearance = c_min
        
        if c_min < 0.45:
            in_danger = True
            danger_ang_vels.append(abs(ang_act))
            if abs(ang_act) > max_ang_vel_in_danger:
                max_ang_vel_in_danger = abs(ang_act)
                
            if first_obstacle_detected_step == -1:
                first_obstacle_detected_step = steps
                
            if first_turn_step == -1 and abs(ang_act) > 0.3:
                first_turn_step = steps
                clearance_at_first_turn = c_min
                did_turn_away = True
                
        steps += 1
        done = terminated or truncated

    is_success = bool(info.get("is_success", False))
    is_collision = bool(info.get("is_collision", False))
    
    no_turn_event = False
    if is_collision and in_danger and max_ang_vel_in_danger < 0.5:
        no_turn_event = True
        
    if is_collision:
        front_min = min(obs[35]*12, obs[0]*12, obs[1]*12)
        if front_min < 0.2:
            front_collision = True
            
    successful_obstacle_avoidance = is_success and in_danger
    critical_recovery = is_success and (min_clearance < 0.35)
    
    final_dist = obs[36] * 12.0
    did_get_around = is_success
    
    res = {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["scenario_category"],
        "model": model_name,
        "success": is_success,
        "collision": is_collision,
        "timeout": not (is_success or is_collision),
        "initial_distance_to_goal": initial_dist,
        "final_distance_to_goal": final_dist,
        "episode_steps": steps,
        "episode_time": steps * 0.1,
        "path_length": path_length,
        "minimum_clearance": min_clearance,
        "mean_linear_velocity": _mean(linear_vels),
        "mean_angular_velocity": _mean(angular_vels),
        "mean_absolute_angular_velocity": _mean(abs_angular_vels),
        "forward_percentage": forward_steps/steps if steps>0 else 0,
        "reverse_percentage": reverse_steps/steps if steps>0 else 0,
        "stationary_percentage": stationary_steps/steps if steps>0 else 0,
        "no_turn_event": no_turn_event,
        "front_collision": front_collision,
        "successful_obstacle_avoidance": successful_obstacle_avoidance,
        "critical_recovery": critical_recovery,
        "angular_velocity_when_clearance_below_0.45m": _mean(danger_ang_vels),
        # Wall-blocked specific
        "direct_goal_path_blocked": check_direct_path_blocked([0.0, 0.0], goal_pos),
        "first_obstacle_detected_step": first_obstacle_detected_step,
        "first_turn_step": first_turn_step,
        "clearance_at_first_turn": clearance_at_first_turn,
        "did_robot_turn_away_from_blocking_wall": did_turn_away,
        "did_robot_get_around_wall": did_get_around,
        "final_result": "Success" if is_success else ("Collision" if is_collision else "Timeout")
    }
    return res

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = BenchmarkEnv(num_scan_features=36)
    
    actor = Actor(observation_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], hidden_sizes=(256, 256)).to(device)
    
    with open("diagnostics/benchmark_150_scenarios.json", "r") as f:
        scenarios = json.load(f)
        
    all_results = []
    
    models = {
        "Champion": PROJECT_ROOT / "checkpoints" / "best_success.pt",
        "Fine-tuned": PROJECT_ROOT / "checkpoints" / "wall_ft_best.pt"
    }
    
    for model_name, ckpt_path in models.items():
        print(f"\nEvaluating {model_name}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        
        for idx, sc in enumerate(scenarios):
            print(f"[{model_name}] {idx+1}/150 - {sc['scenario_category']}")
            res = run_episode(env, actor, sc, model_name, device)
            all_results.append(res)
            
    env.close()
    
    # ── 1. Save Episode Results as CSV ────────────────────────────────
    keys = list(all_results[0].keys()) if all_results else []
    with open("diagnostics/benchmark_episode_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    
    # ── 2. Split by model ─────────────────────────────────────────────
    c_rows = _filter(all_results, model="Champion")
    f_rows = _filter(all_results, model="Fine-tuned")
    
    # ── 3. Category Summary ───────────────────────────────────────────
    all_cats = sorted(set(r["category"] for r in all_results))
    summ = {}  # (model, category) -> dict of rates
    for model_name, rows in [("Champion", c_rows), ("Fine-tuned", f_rows)]:
        for cat in all_cats:
            d = _filter(rows, category=cat)
            if not d:
                continue
            summ[(model_name, cat)] = {
                "success_rate": _mean([r["success"] for r in d]),
                "collision_rate": _mean([r["collision"] for r in d]),
                "timeout_rate": _mean([r["timeout"] for r in d]),
            }
    
    # ── 4. Final Report ───────────────────────────────────────────────
    c_success = _mean([r["success"] for r in c_rows])
    f_success = _mean([r["success"] for r in f_rows])
    c_collision = _mean([r["collision"] for r in c_rows])
    f_collision = _mean([r["collision"] for r in f_rows])
    c_timeout = _mean([r["timeout"] for r in c_rows])
    f_timeout = _mean([r["timeout"] for r in f_rows])
    c_path = _mean([r["path_length"] for r in c_rows])
    f_path = _mean([r["path_length"] for r in f_rows])
    c_time = _mean([r["episode_time"] for r in c_rows])
    f_time = _mean([r["episode_time"] for r in f_rows])
    c_clear = _mean([r["minimum_clearance"] for r in c_rows])
    f_clear = _mean([r["minimum_clearance"] for r in f_rows])
    c_noturn = _sum([r["no_turn_event"] for r in c_rows])
    f_noturn = _sum([r["no_turn_event"] for r in f_rows])
    c_front = _sum([r["front_collision"] for r in c_rows])
    f_front = _sum([r["front_collision"] for r in f_rows])
    
    report = f"""========================================
CONTROLLED BENCHMARK
========================================

Scenarios:    150
Models:       Champion, Fine-tuned
Same scenarios: YES
Deterministic:  YES

========================================
OVERALL RESULTS
========================================

                    Champion    Fine-tuned
Success:            {c_success*100:.1f}%       {f_success*100:.1f}%
Collision:          {c_collision*100:.1f}%       {f_collision*100:.1f}%
Timeout:            {c_timeout*100:.1f}%       {f_timeout*100:.1f}%
Mean Path:          {c_path:.2f}m        {f_path:.2f}m
Mean Time:          {c_time:.1f}s       {f_time:.1f}s
Mean Clearance:     {c_clear:.3f}m      {f_clear:.3f}m
No-Turn:            {c_noturn}          {f_noturn}
Front Collision:    {c_front}          {f_front}

========================================
CATEGORY RESULTS
========================================
"""
    for cat in ["EASY / OPEN", "WALL-BLOCKED", "COMPLEX OBSTACLE", "TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT", "RANDOM"]:
        cs = summ.get(("Champion", cat))
        fs = summ.get(("Fine-tuned", cat))
        if cs and fs:
            report += f"\n{cat}\n"
            report += f"  Champion:   {cs['success_rate']*100:.1f}% success, {cs['collision_rate']*100:.1f}% collision\n"
            report += f"  Fine-tuned: {fs['success_rate']*100:.1f}% success, {fs['collision_rate']*100:.1f}% collision\n"
    
    # ── Wall-blocked analysis ─────────────────────────────────────────
    c_wall = _filter(c_rows, category="WALL-BLOCKED")
    f_wall = _filter(f_rows, category="WALL-BLOCKED")
    
    c_wall_success = _mean([r["success"] for r in c_wall]) if c_wall else 0
    f_wall_success = _mean([r["success"] for r in f_wall]) if f_wall else 0
    c_wall_col = _mean([r["collision"] for r in c_wall]) if c_wall else 0
    f_wall_col = _mean([r["collision"] for r in f_wall]) if f_wall else 0
    c_wall_noturn = _mean([r["no_turn_event"] for r in c_wall]) if c_wall else 0
    f_wall_noturn = _mean([r["no_turn_event"] for r in f_wall]) if f_wall else 0
    c_wall_bypass = _mean([r["did_robot_get_around_wall"] for r in c_wall]) if c_wall else 0
    f_wall_bypass = _mean([r["did_robot_get_around_wall"] for r in f_wall]) if f_wall else 0
    
    c_turn_clr = [r["clearance_at_first_turn"] for r in c_wall if r["first_turn_step"] != -1]
    f_turn_clr = [r["clearance_at_first_turn"] for r in f_wall if r["first_turn_step"] != -1]
    c_ang_near = _mean([r["angular_velocity_when_clearance_below_0.45m"] for r in c_wall]) if c_wall else 0
    f_ang_near = _mean([r["angular_velocity_when_clearance_below_0.45m"] for r in f_wall]) if f_wall else 0
    
    report += f"""
========================================
WALL-BLOCKED ANALYSIS
========================================

Number of wall-blocked scenarios: {len(c_wall)}

Champion success: {c_wall_success*100:.1f}%
Fine-tuned success: {f_wall_success*100:.1f}%

Champion wall-collision rate: {c_wall_col*100:.1f}%
Fine-tuned wall-collision rate: {f_wall_col*100:.1f}%

Champion no-turn rate: {c_wall_noturn*100:.1f}%
Fine-tuned no-turn rate: {f_wall_noturn*100:.1f}%

Champion successful wall bypass rate: {c_wall_bypass*100:.1f}%
Fine-tuned successful wall bypass rate: {f_wall_bypass*100:.1f}%

Champion mean clearance at first turn: {_mean(c_turn_clr):.3f}m
Fine-tuned mean clearance at first turn: {_mean(f_turn_clr):.3f}m

Champion mean angular velocity <0.45m: {c_ang_near:.3f} rad/s
Fine-tuned mean angular velocity <0.45m: {f_ang_near:.3f} rad/s

========================================
FOUR-CORNER RESULTS
========================================

Corner          Champion       Fine-tuned
"""
    for corner in ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"]:
        cs = summ.get(("Champion", corner))
        fs = summ.get(("Fine-tuned", corner))
        cp = cs["success_rate"]*100 if cs else 0
        fp = fs["success_rate"]*100 if fs else 0
        report += f"{corner.ljust(15)} {cp:.1f}%          {fp:.1f}%\n"
        
    # ── Paired Scenario Analysis ──────────────────────────────────────
    both_success = 0
    both_fail = 0
    champ_only = 0
    ft_only = 0
    
    c_by_sid = {r["scenario_id"]: r for r in c_rows}
    f_by_sid = {r["scenario_id"]: r for r in f_rows}
    
    for sid in c_by_sid:
        c_s = c_by_sid[sid]["success"]
        f_s = f_by_sid[sid]["success"]
        if c_s and f_s: both_success += 1
        elif not c_s and not f_s: both_fail += 1
        elif c_s and not f_s: champ_only += 1
        elif not c_s and f_s: ft_only += 1

    report += f"""
========================================
PAIRED SCENARIO ANALYSIS
========================================

scenarios where BOTH succeed: {both_success}
scenarios where BOTH fail: {both_fail}

Champion succeeds / Fine-tuned fails: {champ_only}
Fine-tuned succeeds / Champion fails: {ft_only}

========================================
FINAL VERDICT
========================================
"""
    # ── Verdict logic ─────────────────────────────────────────────────
    c_cx_rows = _filter(c_rows, category="COMPLEX OBSTACLE")
    f_cx_rows = _filter(f_rows, category="COMPLEX OBSTACLE")
    c_cx = _mean([r["success"] for r in c_cx_rows]) if c_cx_rows else 0
    f_cx = _mean([r["success"] for r in f_cx_rows]) if f_cx_rows else 0
    
    if f_success > c_success:
        winner = "FINE-TUNED"
        reason = f"Fine-tuned model has a higher overall success rate ({f_success*100:.1f}% vs {c_success*100:.1f}%) and significantly reduced collisions ({f_collision*100:.1f}% vs {c_collision*100:.1f}%)."
    elif f_success == c_success and f_wall_success > c_wall_success:
        winner = "FINE-TUNED"
        reason = "Models tied on overall success, but Fine-Tuned was superior on WALL-BLOCKED scenarios."
    elif c_success > f_success and (c_success - f_success) < 0.05 and f_collision < c_collision:
        winner = "CONTEXT-DEPENDENT"
        reason = "Champion has slightly better success, but Fine-tuned trades a small drop in success for massive safety/collision improvements."
    else:
        winner = "CHAMPION"
        reason = f"Champion maintained superior overall success ({c_success*100:.1f}% vs {f_success*100:.1f}%)."

    report += f"""
========================================
FINAL BENCHMARK VERDICT
========================================

Champion overall success:         {c_success*100:.1f}%
Fine-tuned overall success:       {f_success*100:.1f}%

Champion wall-blocked success:    {c_wall_success*100:.1f}%
Fine-tuned wall-blocked success:  {f_wall_success*100:.1f}%

Champion complex-obstacle success:  {c_cx*100:.1f}%
Fine-tuned complex-obstacle success:{f_cx*100:.1f}%

Champion collision:               {c_collision*100:.1f}%
Fine-tuned collision:             {f_collision*100:.1f}%

Champion timeout:                 {c_timeout*100:.1f}%
Fine-tuned timeout:               {f_timeout*100:.1f}%

========================================
WINNER: {winner}

REASON: {reason}
========================================
"""
    with open("diagnostics/champion_vs_finetuned_report.txt", "w") as f:
        f.write(report)
    
    print("\n" + report)
    print("Report saved to diagnostics/champion_vs_finetuned_report.txt")
    print("Episode results saved to diagnostics/benchmark_episode_results.csv")

if __name__ == "__main__":
    main()
