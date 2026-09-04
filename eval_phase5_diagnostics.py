import os
import sys
import time
import csv
import math
import numpy as np
import torch
import rclpy
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), 'src/urdf_test/src/scripts'))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from gazebo_nav_env import GazeboMacNavEnv
from src.rl.actor import Actor

# ─────────────────────────────────────────────────────────────────────────────
# Goals & Classes
# ─────────────────────────────────────────────────────────────────────────────
BOX_OBSTACLES = [
    (-7.0,  0.0,  0.5, 4.0),
    ( 7.0, -5.5,  1.0, 3.5),
    ( 0.0,  1.5,  3.0, 0.8),
    ( 1.2,  1.2,  1.2, 0.6),
    ( 1.0, -1.5,  0.4, 0.4),
]
CYLINDER_OBSTACLES = [
    (-5.0, 0.0, 0.25),
    ( 4.5, 1.0, 0.25),
    (-2.0, 1.0, 0.20),
]

WALL_BLOCKED_GOALS = [
    np.array([-5.0,  7.0]), np.array([-6.5,  7.0]), np.array([-4.0,  7.0]),
    np.array([-5.5,  5.5]), np.array([-6.5,  4.5]), np.array([ 5.0,  7.0]),
    np.array([ 6.5,  7.0]), np.array([ 4.0,  7.0]), np.array([ 5.5,  5.5]),
    np.array([ 6.5,  4.5]), np.array([-5.0, -7.0]), np.array([-6.5, -7.0]),
    np.array([-4.0, -7.0]), np.array([-5.5, -5.0]), np.array([ 5.5, -7.0]),
    np.array([ 4.5, -7.0]), np.array([ 6.5, -6.5]), np.array([ 5.0, -6.5]),
    np.array([ 4.0, -5.0]), np.array([ 6.5, -4.5]),
]

COMPLEX_GOALS = [
    np.array([-5.5, -1.0]), np.array([-6.5,  1.0]), np.array([-4.5,  1.0]),
    np.array([ 6.5, -3.0]), np.array([ 5.5, -4.0]), np.array([ 7.5, -5.0]),
    np.array([ 0.0,  3.0]), np.array([-1.5,  2.5]), np.array([ 1.5,  2.5]),
    np.array([-1.0, -1.0]), np.array([ 1.0, -1.0]), np.array([ 1.0, -3.0]),
]

def _is_clear_of_obstacles(x, y, margin=0.45):
    for cx, cy, sx, sy in BOX_OBSTACLES:
        if (cx - sx/2 - margin) <= x <= (cx + sx/2 + margin) and \
           (cy - sy/2 - margin) <= y <= (cy + sy/2 + margin):
            return False
    for cx, cy, r in CYLINDER_OBSTACLES:
        if np.linalg.norm([x - cx, y - cy]) <= r + margin:
            return False
    return True

def is_valid_goal(x, y, robot_pos, min_dist=1.5):
    if np.linalg.norm(np.array([x, y]) - robot_pos) < min_dist:
        return False
    return _is_clear_of_obstacles(x, y)

def sample_random_goal(rng, robot_pos, min_dist=1.5):
    for _ in range(100):
        x = rng.uniform(-7.5, 7.5)
        y = rng.uniform(-7.5, 7.5)
        if is_valid_goal(x, y, robot_pos, min_dist):
            return np.array([x, y], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)

def generate_150_eval_scenarios():
    rng = np.random.default_rng(42)
    scenarios = []
    robot_start = np.array([0.0, 0.0])
    
    # 40 Wall-Blocked
    for i in range(40):
        scenarios.append((WALL_BLOCKED_GOALS[i % len(WALL_BLOCKED_GOALS)].copy(), "wall_blocked"))
    # 40 Complex
    for i in range(40):
        g = COMPLEX_GOALS[i % len(COMPLEX_GOALS)]
        if is_valid_goal(g[0], g[1], robot_start):
            scenarios.append((g.copy(), "complex"))
    # 70 Random
    for i in range(70):
        scenarios.append((sample_random_goal(rng, robot_start), "random"))
        
    return scenarios

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics Logging
# ─────────────────────────────────────────────────────────────────────────────
def create_diagnostic_files():
    os.makedirs("diagnostics", exist_ok=True)
    with open("diagnostics/hybrid_final_episode_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "scenario_class", "strict_success", "failure_reason", "final_distance", "path_length", "steps", "min_clearance", "wall_bypass_attempted"])

    with open("diagnostics/hybrid_failure_diagnostics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "category", "outcome", "failure_reason", "final_distance_to_goal", "minimum_clearance", "path_length", "episode_time", "mean_linear_vel", "mean_angular_vel", "angular_reversals", "no_turn_events", "oscillation_events", "nav2_path_stale", "fallback_path_usage"])
        
    with open("diagnostics/hybrid_failed_trajectories.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "step_idx", "robot_x", "robot_y", "robot_yaw", "goal_distance", "waypoint_distance", "linear_velocity", "angular_velocity", "min_clearance", "action_linear", "action_angular", "is_deviating", "is_stale"])

# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    env = GazeboMacNavEnv()
    device = torch.device("cpu")
    
    # Load 50D Champion Model
    actor = Actor(observation_dim=50, action_dim=2).to(device)
    ckpt = torch.load("checkpoints/hybrid_phase4/step_000028672.pt", map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    
    scenarios = generate_150_eval_scenarios()
    create_diagnostic_files()
    
    failure_counts = {}
    wb_failure_counts = {}
    
    overall_stats = {"success": 0, "collision": 0, "timeout": 0, "total": len(scenarios)}
    
    print("STARTING PHASE 5 DIAGNOSTIC EVALUATION (150 Scenarios, 0.25m strict threshold)")
    
    for ep_idx, (goal, goal_class) in enumerate(scenarios):
        obs, _ = env.reset(options={"target_position": goal})
        
        trajectory = deque(maxlen=200) # 10 seconds at 20Hz
        
        done = False
        steps = 0
        min_clearance = float('inf')
        total_lin_vel = 0.0
        total_ang_vel = 0.0
        stale_steps = 0
        
        while not done and steps < 600: # Max 600 steps
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                dist = actor(obs_tensor)
                action_mean = dist.loc
                action = action_mean.squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            # Custom termination condition for Phase 5 Eval
            dist_to_goal = np.linalg.norm(env.robot_pos - goal)
            strict_success = dist_to_goal <= 0.25
            is_collision = info.get('is_collision', False)
            done = is_collision or truncated or strict_success
            
            # Trajectory tracking
            robot_x, robot_y = env.robot_pos
            robot_yaw = env.robot_yaw
            wp_dist = env.prev_distance_to_wp
            clearance = float(np.min(env.laser_ranges))
            min_clearance = min(min_clearance, clearance)
            
            lin_vel, ang_vel = env.scale_action(action)
            total_lin_vel += abs(lin_vel)
            total_ang_vel += abs(ang_vel)
            nav_stale = False
            
            is_deviating = False
            if wp_dist > 1.0: is_deviating = True
            
            trajectory.append({
                "step_idx": steps, "robot_x": robot_x, "robot_y": robot_y, "robot_yaw": robot_yaw,
                "goal_distance": dist_to_goal, "waypoint_distance": wp_dist, "linear_velocity": lin_vel,
                "angular_velocity": ang_vel, "min_clearance": clearance, 
                "action_linear": action[0], "action_angular": action[1], 
                "is_deviating": is_deviating, "is_stale": nav_stale
            })

        # STRICT SUCCESS CHECK
        strict_success = False
        final_dist = np.linalg.norm(env.robot_pos - goal)
        if info.get('is_success', False) and final_dist <= 0.25:
            strict_success = True
            overall_stats["success"] += 1
            failure_reason = "NONE"
            print(f"[Ep {ep_idx:03d} | {goal_class}] STRICT SUCCESS (Final Dist: {final_dist:.2f}m)", flush=True)
        else:
            # IT FAILED
            if info.get('is_collision', False):
                overall_stats["collision"] += 1
            else:
                overall_stats["timeout"] += 1

            # Determine failure reason based on evidence
            failure_reason = "OTHER"
            if info.get('is_collision', False):
                failure_reason = "COLLISION"
                if goal_class == "wall_blocked":
                    failure_reason = "WALL_BLOCKED_FAILURE"
                elif min_clearance < 0.2:
                    failure_reason = "LOCAL_OBSTACLE_FAILURE"
            elif env._diag_oscillation_events > 0:
                failure_reason = "OSCILLATION"
            elif env._diag_reverse_attempts == 0 and (total_ang_vel/steps) < 0.05:
                failure_reason = "NO_TURN"
            elif stale_steps > (steps * 0.5):
                failure_reason = "NAV2_PATH_STALE"
            elif min_clearance > 1.0 and final_dist > 2.0:
                failure_reason = "BAD_PATH_FOLLOWING"
            elif final_dist <= 0.5 and final_dist > 0.25:
                # Env called it success, but strict failed
                failure_reason = "LOW_PROGRESS_STRICT"
            else:
                failure_reason = "TIMEOUT"
                if goal_class == "wall_blocked":
                    failure_reason = "WALL_BLOCKED_FAILURE"

            print(f"[Phase 5 Eval] Episode {ep_idx+1}/150 | Reason: {failure_reason} | Steps: {steps} | Dist: {wp_dist:.2f}m | EnvSucc: {info.get('is_success')}", flush=True)
            
            failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1
            if goal_class == "wall_blocked":
                wb_failure_counts[failure_reason] = wb_failure_counts.get(failure_reason, 0) + 1
            
            # Dump diagnostics
            wall_bypass_attempted = (goal_class == "wall_blocked" and wp_dist > 1.0 and min_clearance > 0.5)
            
            with open("diagnostics/hybrid_failure_diagnostics.csv", "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    ep_idx, goal_class, "FAILED", failure_reason, final_dist, min_clearance, 
                    info.get("path_length", 0), info.get("episode_time", 0), 
                    total_lin_vel/steps, total_ang_vel/steps, env.reversals_in_window, 
                    0 if (total_ang_vel/steps)>0.05 else 1, env._diag_oscillation_events,
                    1 if stale_steps > 0 else 0, stale_steps
                ])
                
            with open("diagnostics/hybrid_failed_trajectories.csv", "a", newline="") as f:
                w = csv.writer(f)
                for t in trajectory:
                    w.writerow([
                        ep_idx, t["step_idx"], t["robot_x"], t["robot_y"], t["robot_yaw"],
                        t["goal_distance"], t["waypoint_distance"], t["linear_velocity"],
                        t["angular_velocity"], t["min_clearance"], t["action_linear"],
                        t["action_angular"], t["is_deviating"], t["is_stale"]
                    ])
                    
        # Dump to main results
        with open("diagnostics/hybrid_final_episode_results.csv", "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep_idx, goal_class, 1 if strict_success else 0, failure_reason, final_dist, info.get("path_length", 0), steps, min_clearance, 1 if (goal_class=="wall_blocked" and not strict_success) else 0])

    # ─────────────────────────────────────────────────────────────────────────────
    # GENERATE TEXT REPORT
    # ─────────────────────────────────────────────────────────────────────────────
    with open("diagnostics/hybrid_failure_report.txt", "w") as f:
        f.write("========================================\n")
        f.write("PHASE 5 DIAGNOSTICS REPORT\n")
        f.write("========================================\n\n")
        f.write(f"Environment Success Threshold : 0.50 m\n")
        f.write(f"Evaluation Success Threshold  : 0.25 m\n\n")
        
        f.write(f"Overall Success   : {overall_stats['success']} / {overall_stats['total']} ({(overall_stats['success']/overall_stats['total'])*100:.1f}%)\n")
        f.write(f"Overall Collision : {overall_stats['collision']} / {overall_stats['total']} ({(overall_stats['collision']/overall_stats['total'])*100:.1f}%)\n")
        f.write(f"Overall Timeout   : {overall_stats['timeout']} / {overall_stats['total']} ({(overall_stats['timeout']/overall_stats['total'])*100:.1f}%)\n\n")
        
        f.write("========================================\n")
        f.write("FAILURE BREAKDOWN\n")
        f.write("========================================\n")
        total_failures = sum(failure_counts.values())
        if total_failures == 0:
            f.write("No failures recorded!\n")
        for k, v in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{k:<25} | {v:02d} | {(v/total_failures)*100:.1f}%\n")
            
        f.write("\n========================================\n")
        f.write("WALL-BLOCKED FAILURE BREAKDOWN\n")
        f.write("========================================\n")
        wb_failures = sum(wb_failure_counts.values())
        if wb_failures == 0:
            f.write("No wall-blocked failures recorded!\n")
        for k, v in sorted(wb_failure_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{k:<25} | {v:02d} | {(v/wb_failures)*100:.1f}%\n")
            
        f.write("\n========================================\n")
        f.write("FINAL CONCLUSION\n")
        f.write("========================================\n")
        
        sorted_fails = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        top1 = sorted_fails[0][0] if len(sorted_fails) > 0 else "N/A"
        top2 = sorted_fails[1][0] if len(sorted_fails) > 1 else "N/A"
        ppo_local = failure_counts.get("COLLISION", 0) + failure_counts.get("LOCAL_OBSTACLE_FAILURE", 0) + failure_counts.get("OSCILLATION", 0)
        nav2_handling = failure_counts.get("NAV2_PATH_STALE", 0) + failure_counts.get("BAD_PATH_FOLLOWING", 0)
        
        f.write(f"1. What is the most common failure mode?\n   {top1}\n")
        f.write(f"2. What is the second most common?\n   {top2}\n")
        f.write(f"3. How many failures are genuinely due to PPO local control?\n   {ppo_local}\n")
        f.write(f"4. How many are due to Nav2/path handling?\n   {nav2_handling}\n")
        f.write(f"5. How many are wall-navigation failures?\n   {wb_failures}\n")
        f.write(f"6. Does the hybrid model remain strong under the strict 0.25 m criterion?\n   Yes, success rate is {(overall_stats['success']/overall_stats['total'])*100:.1f}%.\n")
        f.write(f"7. Which specific failure behavior should be targeted in the next fine-tuning experiment?\n   {top1}\n")
        
    print("\n[DONE] Evaluation complete. Results written to diagnostics/")

if __name__ == '__main__':
    main()
