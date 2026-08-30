"""Champion Failure Diagnostic Script."""

import csv
import json
import os
import sys
import math
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

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class DiagnosticEnv(GazeboMacNavEnv):
    def get_robot_pose(self):
        try:
            return self.robot_pos, self.robot_yaw
        except AttributeError:
            return np.array([0.0, 0.0]), 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["robot_x"] = float(self.robot_pos[0]) if hasattr(self, "robot_pos") else 0.0
        info["robot_y"] = float(self.robot_pos[1]) if hasattr(self, "robot_pos") else 0.0
        info["robot_yaw"] = float(self.robot_yaw) if hasattr(self, "robot_yaw") else 0.0
        info["waypoint"] = None
        return obs, reward, terminated, truncated, info

    def odom_callback(self, msg):
        super().odom_callback(msg)
        q = msg.pose.pose.orientation
        self.robot_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if options is not None and "target_position" in options:
            orig_safe_goals = self.safe_goals
            self.safe_goals = [np.array(options["target_position"])]
            obs, info = super().reset(seed=seed, options=options)
            self.safe_goals = orig_safe_goals
            return obs, info
        return super().reset(seed=seed, options=options)

def angle_diff(a, b):
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    return diff

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DiagnosticEnv(num_scan_features=36)
    
    actor = Actor(observation_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], hidden_sizes=(256, 256)).to(device)
    
    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_success.pt"
    print(f"Loading Champion model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    
    with open("diagnostics/benchmark_150_scenarios.json", "r") as f:
        scenarios = json.load(f)
        
    all_trajectories = []
    episode_summaries = []
    classifications = []
    
    for idx, sc in enumerate(scenarios):
        sc_id = sc["scenario_id"]
        cat = sc["scenario_category"]
        print(f"[{idx+1}/150] {cat} - {sc_id}")
        
        goal_pos = np.array(sc["goal_position"], dtype=np.float32)
        obs, info = env.reset(seed=sc["random_seed"], options={"target_position": goal_pos})
        
        done = False
        steps = 0
        traj = []
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist = actor(obs_tensor)
                action = torch.clamp(dist.mean, -1.0, 1.0).cpu().numpy()[0]
                
            lin_act = action[0] * env.max_linear_vel
            ang_act = action[1] * env.max_angular_vel
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            lidar = obs[:36] * 12.0
            min_lidar = float(np.min(lidar))
            front_min = float(np.min([lidar[35], lidar[0], lidar[1]]))
            left_min = float(np.min(lidar[7:11]))
            right_min = float(np.min(lidar[26:30]))
            
            goal_x, goal_y = goal_pos[0], goal_pos[1]
            dist_to_goal = math.hypot(goal_x - info.get("robot_x", 0.0), goal_y - info.get("robot_y", 0.0))
            
            wp = info.get("waypoint")
            if wp is not None:
                wp_x, wp_y = wp[0], wp[1]
                dist_to_wp = math.hypot(wp_x - info.get("robot_x", 0.0), wp_y - info.get("robot_y", 0.0))
            else:
                wp_x, wp_y, dist_to_wp = 0.0, 0.0, 0.0
                
            r_comps = info.get("reward_components", {})
            
            t_row = {
                "episode_id": sc_id,
                "step": steps,
                "timestamp": steps * 0.1,
                "robot_x": info.get("robot_x", 0.0),
                "robot_y": info.get("robot_y", 0.0),
                "robot_yaw": info.get("robot_yaw", 0.0),
                "goal_x": goal_x,
                "goal_y": goal_y,
                "distance_to_goal": dist_to_goal,
                "waypoint_x": wp_x,
                "waypoint_y": wp_y,
                "distance_to_waypoint": dist_to_wp,
                "linear_velocity": lin_act,
                "angular_velocity": ang_act,
                "action_linear": action[0],
                "action_angular": action[1],
                "minimum_lidar_distance": min_lidar,
                "front_sector_minimum": front_min,
                "left_sector_minimum": left_min,
                "right_sector_minimum": right_min,
                "progress_reward": r_comps.get("progress_reward", 0.0),
                "clearance_reward": r_comps.get("clearance_reward", 0.0),
                "turning_penalty": r_comps.get("turning_penalty", 0.0),
                "time_penalty": r_comps.get("time_penalty", 0.0),
                "inflation_penalty": r_comps.get("inflation_penalty", 0.0),
                "reverse_recovery_reward": r_comps.get("reverse_recovery_reward", 0.0),
                "total_reward": reward
            }
            traj.append(t_row)
            all_trajectories.extend([t_row])
            steps += 1
            
        # Post-episode summary calculation
        is_succ = info.get("is_success", False)
        is_coll = info.get("is_collision", False)
        outcome = "SUCCESS" if is_succ else ("COLLISION" if is_coll else "TIMEOUT")
        
        initial_goal_dist = traj[0]["distance_to_goal"]
        final_goal_dist = traj[-1]["distance_to_goal"]
        total_dist_reduction = initial_goal_dist - final_goal_dist
        
        def get_dist_change(w_steps):
            if len(traj) <= w_steps: return 0.0
            return traj[0]["distance_to_goal"] - traj[w_steps]["distance_to_goal"]
            
        def calc_reversals(key, threshold):
            revs = 0
            if len(traj) < 2: return 0
            last_sign = math.copysign(1, traj[0][key]) if abs(traj[0][key]) > threshold else 0
            for r in traj[1:]:
                v = r[key]
                if abs(v) > threshold:
                    s = math.copysign(1, v)
                    if last_sign != 0 and s != last_sign:
                        revs += 1
                    last_sign = s
            return revs
            
        ang_revs = calc_reversals("angular_velocity", 0.1)
        lin_revs = calc_reversals("linear_velocity", 0.05)
        
        # spinning
        abs_ang = [abs(t["angular_velocity"]) for t in traj]
        cum_ang = sum(abs_ang) * 0.1
        pct_turning = sum(1 for t in traj if abs(t["angular_velocity"]) > 0.2) / len(traj)
        
        # Last 10 seconds
        last_10 = traj[-100:]
        avg_lin_last10 = sum(t["linear_velocity"] for t in last_10) / len(last_10) if last_10 else 0
        avg_ang_last10 = sum(t["angular_velocity"] for t in last_10) / len(last_10) if last_10 else 0
        
        summ_row = {
            "episode_id": sc_id,
            "category": cat,
            "outcome": outcome,
            "initial_goal_distance": initial_goal_dist,
            "final_goal_distance": final_goal_dist,
            "total_distance_reduction": total_dist_reduction,
            "angular_reversals": ang_revs,
            "linear_reversals": lin_revs,
            "cumulative_angular_rotation": cum_ang,
            "percent_turning": pct_turning,
            "avg_lin_last10": avg_lin_last10,
            "avg_ang_last10": avg_ang_last10
        }
        episode_summaries.append(summ_row)
        
        # Automatic Classification
        fail_mode = "N/A"
        if not is_succ:
            if is_coll and sum(t["front_sector_minimum"] < 0.3 for t in last_10) > 5 and abs(avg_ang_last10) < 0.2:
                fail_mode = "DIRECT_WALL_COMMITMENT"
            elif ang_revs > 15 or lin_revs > 10:
                fail_mode = "OSCILLATION"
            elif pct_turning > 0.7 or cum_ang > 15.0:
                fail_mode = "SPINNING"
            elif avg_lin_last10 < -0.05:
                fail_mode = "REVERSE_CREEP"
            elif is_coll and pct_turning < 0.2:
                fail_mode = "INSUFFICIENT_TURN"
            elif total_dist_reduction < 1.0:
                fail_mode = "LOW_PROGRESS"
            elif is_coll:
                fail_mode = "COLLISION"
            else:
                fail_mode = "TIMEOUT"
                
        classifications.append({
            "episode_id": sc_id,
            "category": cat,
            "outcome": outcome,
            "failure_mode": fail_mode,
            "supporting_metrics": f"lin_rev={lin_revs} ang_rev={ang_revs} turn_pct={pct_turning:.2f} dist_red={total_dist_reduction:.2f}"
        })

    env.close()
    
    # Save files
    os.makedirs("diagnostics", exist_ok=True)
    
    with open("diagnostics/champion_failure_trajectories.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_trajectories[0].keys())
        writer.writeheader()
        writer.writerows(all_trajectories)
        
    with open("diagnostics/champion_failure_episode_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_summaries[0].keys())
        writer.writeheader()
        writer.writerows(episode_summaries)
        
    with open("diagnostics/champion_failure_classification.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=classifications[0].keys())
        writer.writeheader()
        writer.writerows(classifications)
        
    # Generate Report
    total = len(episode_summaries)
    succ_cnt = sum(1 for c in classifications if c["outcome"] == "SUCCESS")
    coll_cnt = sum(1 for c in classifications if c["outcome"] == "COLLISION")
    time_cnt = sum(1 for c in classifications if c["outcome"] == "TIMEOUT")
    fail_cnt = coll_cnt + time_cnt
    
    fm_counts = {}
    for c in classifications:
        if c["failure_mode"] != "N/A":
            fm_counts[c["failure_mode"]] = fm_counts.get(c["failure_mode"], 0) + 1
            
    def pct(c): return f"{c/fail_cnt*100:.1f}%" if fail_cnt>0 else "0%"
    
    report = f"""========================================
CHAMPION FAILURE DIAGNOSTIC
========================================

Total episodes: {total}

Successful: {succ_cnt}
Collision: {coll_cnt}
Timeout: {time_cnt}

----------------------------------------
FAILURE MODES (across all {fail_cnt} failures)
----------------------------------------

OSCILLATION:                {fm_counts.get('OSCILLATION', 0)} / {pct(fm_counts.get('OSCILLATION', 0))}
SPINNING:                   {fm_counts.get('SPINNING', 0)} / {pct(fm_counts.get('SPINNING', 0))}
INSUFFICIENT_TURN:          {fm_counts.get('INSUFFICIENT_TURN', 0)} / {pct(fm_counts.get('INSUFFICIENT_TURN', 0))}
WRONG_DIRECTION:            {fm_counts.get('WRONG_DIRECTION', 0)} / {pct(fm_counts.get('WRONG_DIRECTION', 0))}
REPEATED_OBSTACLE_APPROACH: {fm_counts.get('REPEATED_OBSTACLE_APPROACH', 0)} / {pct(fm_counts.get('REPEATED_OBSTACLE_APPROACH', 0))}
LOW_PROGRESS:               {fm_counts.get('LOW_PROGRESS', 0)} / {pct(fm_counts.get('LOW_PROGRESS', 0))}
REVERSE_CREEP:              {fm_counts.get('REVERSE_CREEP', 0)} / {pct(fm_counts.get('REVERSE_CREEP', 0))}
DIRECT_WALL_COMMITMENT:     {fm_counts.get('DIRECT_WALL_COMMITMENT', 0)} / {pct(fm_counts.get('DIRECT_WALL_COMMITMENT', 0))}
COLLISION (Generic):        {fm_counts.get('COLLISION', 0)} / {pct(fm_counts.get('COLLISION', 0))}
TIMEOUT (Generic):          {fm_counts.get('TIMEOUT', 0)} / {pct(fm_counts.get('TIMEOUT', 0))}

"""

    for t_cat in ["WALL-BLOCKED", "COMPLEX OBSTACLE"]:
        cat_eps = [c for c in classifications if c["category"] == t_cat]
        c_tot = len(cat_eps)
        if c_tot == 0: continue
        c_suc = sum(1 for c in cat_eps if c["outcome"] == "SUCCESS")
        c_col = sum(1 for c in cat_eps if c["outcome"] == "COLLISION")
        c_tim = sum(1 for c in cat_eps if c["outcome"] == "TIMEOUT")
        c_fail = c_col + c_tim
        
        report += f"""----------------------------------------
{t_cat}
----------------------------------------

Total: {c_tot}
Success: {c_suc}
Collision: {c_col}
Timeout: {c_tim}

Failure mode distribution:
"""
        c_fms = {}
        for c in cat_eps:
            if c["failure_mode"] != "N/A":
                c_fms[c["failure_mode"]] = c_fms.get(c["failure_mode"], 0) + 1
                
        for k, v in c_fms.items():
            report += f"{k}: {v} ({v/c_fail*100:.1f}%)\\n"
        report += "\\n"
        
    s_rows = [s for s in episode_summaries if s["outcome"] == "SUCCESS"]
    f_rows = [s for s in episode_summaries if s["outcome"] != "SUCCESS"]
    
    s_prog = sum(r["total_distance_reduction"] for r in s_rows) / len(s_rows) if s_rows else 0
    f_prog = sum(r["total_distance_reduction"] for r in f_rows) / len(f_rows) if f_rows else 0
    s_turn = sum(r["percent_turning"] for r in s_rows) / len(s_rows) if s_rows else 0
    f_turn = sum(r["percent_turning"] for r in f_rows) / len(f_rows) if f_rows else 0
    s_angr = sum(r["angular_reversals"] for r in s_rows) / len(s_rows) if s_rows else 0
    f_angr = sum(r["angular_reversals"] for r in f_rows) / len(f_rows) if f_rows else 0

    report += f"""----------------------------------------
SUCCESS VS FAILURE
----------------------------------------
Mean Progress: Success = {s_prog:.2f}m | Failure = {f_prog:.2f}m
Mean Turn %:   Success = {s_turn*100:.1f}% | Failure = {f_turn*100:.1f}%
Mean Ang Revs: Success = {s_angr:.1f} | Failure = {f_angr:.1f}

========================================
FINAL DIAGNOSIS
========================================

"""
    sorted_fms = sorted(fm_counts.items(), key=lambda x: x[1], reverse=True)
    first_fm = sorted_fms[0][0] if sorted_fms else "N/A"
    second_fm = sorted_fms[1][0] if len(sorted_fms) > 1 else "N/A"
    
    report += f"""1. Is the dominant problem oscillation?
   {"YES" if first_fm == "OSCILLATION" else "NO"}

2. Is the dominant problem spinning?
   {"YES" if first_fm == "SPINNING" else "NO"}

3. Is the dominant problem insufficient turning?
   {"YES" if first_fm == "INSUFFICIENT_TURN" else "NO"}

4. Is the robot repeatedly committing toward blocked goal directions?
   {"YES" if first_fm == "DIRECT_WALL_COMMITMENT" else "NO"}

8. What is the SINGLE most common failure mode?
   {first_fm}

9. What is the SECOND most common failure mode?
   {second_fm}

10. What behavior should the next experiment specifically target?
    {first_fm}
"""

    with open("diagnostics/champion_failure_report.txt", "w") as f:
        f.write(report)
        
    print(report)

if __name__ == "__main__":
    main()
