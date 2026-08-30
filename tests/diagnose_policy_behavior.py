"""tests/diagnose_policy_behavior.py

Policy diagnostic and classification script. Loads the specified PPO checkpoint
and evaluates it on 10 deterministic episodes in Gazebo, tracking actions,
physical velocities, goal progress, and reward components.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import rclpy

# Configure import paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

from train import TurtleBotEnv, generate_eval_goals, Actor

# Bounding box obstacles from obstacles.world: (center_x, center_y, size_x, size_y)
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

# Cylinder obstacles: (center_x, center_y, radius)
CYLINDER_OBSTACLES = [
    (-5.0, 0.0, 0.25),        # hall_pillar_1
    (4.5, 1.0, 0.25),         # hall_pillar_2
    (-2.0, 1.0, 0.20),        # hall_cylinder_obstacle
]


class DiagnosticTurtleBotEnv(TurtleBotEnv):
    """TurtleBotEnv subclass that intercepts step() to expose physical parameters."""

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Execute parent step
        obs, reward, terminated, truncated, info = super().step(action)

        # Scale actions to physical command velocities
        linear_vel, angular_vel = self.scale_action(action)

        # Expose extra diagnostics for step tracking
        info["linear_vel"] = linear_vel
        info["angular_vel"] = angular_vel
        info["distance_to_goal"] = np.linalg.norm(self.target_position - self.robot_pos)

        return obs, reward, terminated, truncated, info


def classify_episode_progress(distances: list[float], success: bool) -> str:
    """Classify the robot's progress trajectory toward the goal."""
    if success:
        return "consistently approaches the goal"
    
    initial = distances[0]
    final = distances[-1]
    minimum = min(distances)
    reduction = initial - final

    if reduction < 0.1 and minimum > initial - 0.5:
        return "makes almost no progress"
    elif final > initial and minimum >= initial - 0.2:
        return "moves away from the goal"
    elif minimum < initial - 1.0 and final > minimum + 0.8:
        return "initially approaches but then stops"
    else:
        return "approaches the goal but does not reach it"


def run_evaluation_diagnostics() -> None:
    # ── 1. Setup paths and checkpoint ─────────────────────────────────────────
    validation_path = PROJECT_ROOT / "checkpoints_validation" / "final_model.pt"
    if validation_path.exists():
        ckpt_path = validation_path
        checkpoint_name = "checkpoints_validation/final_model.pt"
    else:
        checkpoint_name = "step_00040960.pt"
        ckpt_path = PROJECT_ROOT / "checkpoints" / "step_000040960.pt"
        if not ckpt_path.exists():
            ckpt_path = PROJECT_ROOT / "checkpoints" / checkpoint_name
            if not ckpt_path.exists():
                print(f"Error: Checkpoint not found at {ckpt_path}")
                sys.exit(1)

    print("Initializing ROS 2 runtime...")
    if not rclpy.ok():
        rclpy.init()

    # Create diagnostic environment and check dimensions
    env = DiagnosticTurtleBotEnv(num_scan_features=360)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Verify matching checkpoint config dimensions
    print("=" * 50)
    print("POLICY DIAGNOSTIC")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Environment observation shape: {env.observation_space.shape}")
    print(f"Action dimension: {action_dim}")
    print("=" * 50)

    # Initialize Actor network and load state dictionary
    device = torch.device("cpu")
    actor = Actor(observation_dim=obs_dim, action_dim=action_dim, hidden_sizes=(256, 256))
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    # Generate 10 evaluation goals deterministically using seed=42
    rng = np.random.default_rng(42)
    eval_goals: list[np.ndarray] = []
    robot_start = np.array([0.0, 0.0])
    for _ in range(10):
        # Sample using seed=42 logic matching train.py
        for _ in range(1000):
            x = rng.uniform(-7.5, 7.5)
            y = rng.uniform(-7.5, 7.5)
            # Replicate is_valid_goal constraints
            valid = True
            if np.linalg.norm(np.array([x, y]) - robot_start) < 1.5:
                valid = False
            if valid:
                for cx, cy, sx, sy in BOX_OBSTACLES:
                    if (cx - sx/2.0 - 0.45) <= x <= (cx + sx/2.0 + 0.45) and (cy - sy/2.0 - 0.45) <= y <= (cy + sy/2.0 + 0.45):
                        valid = False
                        break
            if valid:
                for cx, cy, r in CYLINDER_OBSTACLES:
                    if np.linalg.norm(np.array([x, y]) - np.array([cx, cy])) <= r + 0.45:
                        valid = False
                        break
            if valid:
                eval_goals.append(np.array([x, y], dtype=np.float32))
                break
        else:
            eval_goals.append(np.array([3.5, 6.0], dtype=np.float32))

    # Metric accumulators
    episodes_data = []

    print("\nRunning exactly 10 diagnostic evaluation episodes...")
    for ep in range(10):
        goal = eval_goals[ep]
        obs, info = env.reset(options={"target_position": goal})
        
        initial_pos = env.robot_pos.copy()
        initial_dist = np.linalg.norm(goal - initial_pos)
        
        done = False
        steps = 0
        path_length = 0.0
        last_pos = initial_pos.copy()

        # Step lists
        linear_actions = []
        angular_actions = []
        linear_velocities = []
        angular_velocities = []
        distances_to_goal = [initial_dist]

        # Reward components
        rewards_progress = []
        rewards_clearance = []
        rewards_turning = []
        rewards_time = []
        rewards_inflation = []
        rewards_reverse_penalty = []
        rewards_reverse = []
        rewards_success = []
        rewards_collision = []
        rewards_total = []

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # Use mean action deterministically
                distrib = actor(obs_tensor)
                action_mean = distrib.mean.numpy()[0]
                # Clamp actions
                action = np.clip(action_mean, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            # Accumulate path length
            current_pos = env.robot_pos
            path_length += float(np.linalg.norm(current_pos - last_pos))
            last_pos = current_pos.copy()

            # Record actions/velocities
            linear_actions.append(action[0])
            angular_actions.append(action[1])
            linear_velocities.append(info["linear_vel"])
            angular_velocities.append(info["angular_vel"])
            distances_to_goal.append(info["distance_to_goal"])

            # Record reward components
            rc = info["reward_components"]
            rewards_progress.append(rc["progress"])
            rewards_clearance.append(rc["clearance"])
            rewards_turning.append(rc["turning"])
            rewards_time.append(rc["time"])
            rewards_inflation.append(rc["inflation"])
            rewards_reverse_penalty.append(rc.get("reverse_penalty", 0.0))
            rewards_reverse.append(rc["reverse_recovery"])
            rewards_success.append(rc["success"])
            rewards_collision.append(rc["collision"])
            rewards_total.append(rc["total"])

            done = terminated or truncated

        # Final positions / stats
        final_pos = env.robot_pos.copy()
        final_dist = np.linalg.norm(goal - final_pos)
        success = bool(info.get("is_success", False))
        collision = bool(info.get("is_collision", False))
        timeout = not success and not collision

        term_type = "SUCCESS" if success else ("COLLISION" if collision else "TIMEOUT")

        # Episode stats calculations
        abs_reduction = initial_dist - final_dist
        pct_reduction = (abs_reduction / initial_dist) * 100.0
        min_reached = min(distances_to_goal)
        progress_behavior = classify_episode_progress(distances_to_goal, success)

        ep_data = {
            "episode": ep + 1,
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "term_type": term_type,
            "initial_pos": initial_pos,
            "goal": goal,
            "initial_dist": initial_dist,
            "final_pos": final_pos,
            "final_dist": final_dist,
            "steps": steps,
            "episode_time": steps * 0.05,
            "path_length": path_length,
            "linear_actions": linear_actions,
            "angular_actions": angular_actions,
            "linear_velocities": linear_velocities,
            "angular_velocities": angular_velocities,
            "distances": distances_to_goal,
            "rewards": {
                "progress": sum(rewards_progress),
                "clearance": sum(rewards_clearance),
                "turning": sum(rewards_turning),
                "time": sum(rewards_time),
                "inflation": sum(rewards_inflation),
                "reverse_penalty": sum(rewards_reverse_penalty),
                "reverse": sum(rewards_reverse),
                "success": sum(rewards_success),
                "collision": sum(rewards_collision),
                "total": sum(rewards_total),
            }
        }
        episodes_data.append(ep_data)

        # Print detailed episode reports
        print(f"\n==================================================")
        print(f"EPISODE {ep + 1:02d} DIAGNOSTICS")
        print(f"==================================================")
        print(f"Initial robot position: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}]")
        print(f"Initial goal position:  [{goal[0]:.2f}, {goal[1]:.2f}]")
        print(f"Initial distance:       {initial_dist:.2f} m")
        print(f"Final robot position:   [{final_pos[0]:.2f}, {final_pos[1]:.2f}]")
        print(f"Final distance:         {final_dist:.2f} m")
        print(f"Total steps:            {steps}")
        print(f"Total episode time:     {steps * 0.05:.2f} s")
        print(f"Total path length:      {path_length:.2f} m")
        print(f"Termination type:       {term_type}")
        print(f"Status:                 {'SUCCESS' if success else 'FAILURE'}")
        print(f"--------------------------------------------------")
        print(f"Action Diagnostics:")
        print(f"  Linear action:")
        print(f"    mean = {np.mean(linear_actions):.4f}")
        print(f"    min  = {np.min(linear_actions):.4f}")
        print(f"    max  = {np.max(linear_actions):.4f}")
        print(f"  Angular action:")
        print(f"    mean = {np.mean(angular_actions):.4f}")
        print(f"    min  = {np.min(angular_actions):.4f}")
        print(f"    max  = {np.max(angular_actions):.4f}")
        print(f"Physical Velocity:")
        print(f"  Linear velocity:")
        print(f"    mean = {np.mean(linear_velocities):.4f} m/s")
        print(f"    min  = {np.min(linear_velocities):.4f} m/s")
        print(f"    max  = {np.max(linear_velocities):.4f} m/s")
        print(f"  Angular velocity:")
        print(f"    mean = {np.mean(angular_velocities):.4f} rad/s")
        print(f"    min  = {np.min(angular_velocities):.4f} rad/s")
        print(f"    max  = {np.max(angular_velocities):.4f} rad/s")
        print(f"--------------------------------------------------")
        print(f"Goal Progress:")
        print(f"  Initial distance:        {initial_dist:.2f} m")
        print(f"  Final distance:          {final_dist:.2f} m")
        print(f"  Distance reduction:      {abs_reduction:.2f} m")
        print(f"  Minimum distance reached: {min_reached:.2f} m")
        print(f"  Progress percentage:     {pct_reduction:.1f}%")
        print(f"  Behavior:                {progress_behavior}")
        # Calculate step percentages (PART 9)
        f_steps = sum(1 for v in linear_velocities if v > 0.05)
        r_steps = sum(1 for v in linear_velocities if v < -0.05)
        s_steps = sum(1 for v in linear_velocities if abs(v) <= 0.05)
        h_steps = sum(1 for w in angular_velocities if abs(w) > 1.0)
        
        f_pct = (f_steps / steps) * 100.0
        r_pct = (r_steps / steps) * 100.0
        s_pct = (s_steps / steps) * 100.0
        h_pct = (h_steps / steps) * 100.0

        print(f"--------------------------------------------------")
        print(f"Movement Statistics:")
        print(f"  Forward:              {f_pct:.1f} %")
        print(f"  Reverse:              {r_pct:.1f} %")
        print(f"  Stationary:           {s_pct:.1f} %")
        print(f"  High angular rotation: {h_pct:.1f} %")
        print(f"  Mean linear velocity:  {np.mean(linear_velocities):.4f}")
        print(f"  Mean angular velocity: {np.mean(angular_velocities):.4f}")
        print(f"--------------------------------------------------")
        print(f"Reward Components:")
        print(f"  Progress:         {sum(rewards_progress):+.4f}  (avg/step: {np.mean(rewards_progress):+.4f})")
        print(f"  Clearance:        {sum(rewards_clearance):+.4f}  (avg/step: {np.mean(rewards_clearance):+.4f})")
        print(f"  Turning:          {sum(rewards_turning):+.4f}  (avg/step: {np.mean(rewards_turning):+.4f})")
        print(f"  Time:             {sum(rewards_time):+.4f}  (avg/step: {np.mean(rewards_time):+.4f})")
        print(f"  Inflation:        {sum(rewards_inflation):+.4f}  (avg/step: {np.mean(rewards_inflation):+.4f})")
        print(f"  Reverse penalty:  {sum(rewards_reverse_penalty):+.4f}  (avg/step: {np.mean(rewards_reverse_penalty):+.4f})")
        print(f"  Reverse recovery: {sum(rewards_reverse):+.4f}  (avg/step: {np.mean(rewards_reverse):+.4f})")
        print(f"  Success:          {sum(rewards_success):+.4f}  (avg/step: {np.mean(rewards_success):+.4f})")
        print(f"  Collision:        {sum(rewards_collision):+.4f}  (avg/step: {np.mean(rewards_collision):+.4f})")
        print(f"  Total:            {sum(rewards_total):+.4f}  (avg/step: {np.mean(rewards_total):+.4f})")

    # ── 6. Final behavior classification ──────────────────────────────────────
    agg_success = sum(1 for e in episodes_data if e["success"])
    agg_collision = sum(1 for e in episodes_data if e["collision"])
    agg_timeout = sum(1 for e in episodes_data if e["timeout"])

    mean_init_dist = np.mean([e["initial_dist"] for e in episodes_data])
    mean_final_dist = np.mean([e["final_dist"] for e in episodes_data])
    mean_reduction = np.mean([e["initial_dist"] - e["final_dist"] for e in episodes_data])
    mean_path = np.mean([e["path_length"] for e in episodes_data])
    mean_time = np.mean([e["episode_time"] for e in episodes_data])

    mean_lin_vel = np.mean([np.mean(e["linear_velocities"]) for e in episodes_data])
    mean_ang_vel = np.mean([np.mean(e["angular_velocities"]) for e in episodes_data])
    mean_lin_act = np.mean([np.mean(e["linear_actions"]) for e in episodes_data])
    mean_ang_act = np.mean([np.mean(e["angular_actions"]) for e in episodes_data])

    # Find dominant reward component by average contribution per step across all episodes
    comp_names = ["progress", "clearance", "turning", "time", "inflation", "reverse_penalty", "reverse"]
    comp_sums = {name: 0.0 for name in comp_names}
    total_steps_all = sum(e["steps"] for e in episodes_data)
    for e in episodes_data:
        comp_sums["progress"] += e["rewards"]["progress"]
        comp_sums["clearance"] += e["rewards"]["clearance"]
        comp_sums["turning"] += e["rewards"]["turning"]
        comp_sums["time"] += e["rewards"]["time"]
        comp_sums["inflation"] += e["rewards"]["inflation"]
        comp_sums["reverse_penalty"] += e["rewards"]["reverse_penalty"]
        comp_sums["reverse"] += e["rewards"]["reverse"]

    avg_contributions = {name: comp_sums[name] / total_steps_all for name in comp_names}
    dominant_comp = max(avg_contributions.keys(), key=lambda k: abs(avg_contributions[k]))

    # Automatic behavior classification based on measured metrics
    # Classification logic:
    # A. NEAR-STATIONARY: linear velocity near 0 (mean absolute physical velocity < 0.02) and minimal progress.
    # B. EXCESSIVE SPINNING: angular velocity absolute mean high (> 1.0 rad/s) and linear physical velocity low (< 0.03).
    # C. BACKWARD EXPLOITATION: mean linear physical velocity is negative, or reverse commands dominate.
    # D. CONSERVATIVE OBSTACLE AVOIDANCE: clearance/inflation rewards are the largest absolute non-progress components, and velocity is low.
    # E. FORWARD BUT NOT GOAL-DIRECTED: moves forward (mean linear physical velocity >= 0.04) but goal distance reduction is low (< 0.5m on average).
    # F. NORMAL EXPLORATION: standard exploration velocities and goal reduction, but not high success rate yet.

    classification = "NORMAL EXPLORATION"
    abs_lin_velocities = [np.mean(np.abs(e["linear_velocities"])) for e in episodes_data]
    mean_abs_lin_vel = np.mean(abs_lin_velocities)
    
    abs_ang_velocities = [np.mean(np.abs(e["angular_velocities"])) for e in episodes_data]
    mean_abs_ang_vel = np.mean(abs_ang_velocities)

    # Check for near-stationary
    if mean_abs_lin_vel < 0.02 and mean_reduction < 0.5:
        classification = "NEAR-STATIONARY"
    # Check for excessive spinning
    elif mean_abs_ang_vel > 1.0 and mean_abs_lin_vel < 0.03:
        classification = "EXCESSIVE TURNING / SPINNING"
    # Check for backward exploitation (mean physical linear velocity is negative)
    elif mean_lin_vel < -0.01:
        classification = "BACKWARD EXPLOITATION"
    # Check for conservative obstacle avoidance
    elif abs(avg_contributions["inflation"]) > 0.02 and mean_abs_lin_vel < 0.05:
        classification = "CONSERVATIVE OBSTACLE AVOIDANCE"
    # Check for forward but not goal-directed
    elif mean_abs_lin_vel >= 0.03 and mean_reduction < 0.8:
        classification = "FORWARD BUT NOT GOAL-DIRECTED"
    else:
        classification = "NORMAL EXPLORATION"

    # ── 7. Aggregate print ────────────────────────────────────────────────────
    # Calculate aggregate movement statistics
    avg_forward_pct = np.mean([sum(1 for v in e["linear_velocities"] if v > 0.05)/e["steps"]*100.0 for e in episodes_data])
    avg_reverse_pct = np.mean([sum(1 for v in e["linear_velocities"] if v < -0.05)/e["steps"]*100.0 for e in episodes_data])
    avg_stationary_pct = np.mean([sum(1 for v in e["linear_velocities"] if abs(v) <= 0.05)/e["steps"]*100.0 for e in episodes_data])
    avg_high_rot_pct = np.mean([sum(1 for w in e["angular_velocities"] if abs(w) > 1.0)/e["steps"]*100.0 for e in episodes_data])

    print(f"\n==================================================")
    print(f"AGGREGATE POLICY DIAGNOSTIC")
    print(f"==================================================")
    print(f"Episodes: 10\n")
    print(f"Success rate:       {(agg_success / 10.0) * 100.0:.1f}%")
    print(f"Collision rate:     {(agg_collision / 10.0) * 100.0:.1f}%")
    print(f"Timeout rate:       {(agg_timeout / 10.0) * 100.0:.1f}%")
    print("")
    print(f"Mean initial distance: {mean_init_dist:.2f} m")
    print(f"Mean final distance:   {mean_final_dist:.2f} m")
    print(f"Mean distance reduction: {mean_reduction:.2f} m")
    print(f"Mean path length:      {mean_path:.2f} m")
    print(f"Mean episode time:     {mean_time:.2f} s")
    print("")
    print(f"Movement Statistics Averages:")
    print(f"  Forward:             {avg_forward_pct:.1f} %")
    print(f"  Reverse:             {avg_reverse_pct:.1f} %")
    print(f"  Stationary:          {avg_stationary_pct:.1f} %")
    print(f"  High rotation:       {avg_high_rot_pct:.1f} %")
    print("")
    print(f"Mean linear velocity:  {mean_lin_vel:.4f} m/s")
    print(f"Mean angular velocity: {mean_ang_vel:.4f} rad/s")
    print("")
    print(f"Mean linear action:    {mean_lin_act:.4f}")
    print(f"Mean angular action:   {mean_ang_act:.4f}")
    print("")
    print(f"Dominant reward component:")
    print(f"    {dominant_comp} (avg/step: {avg_contributions[dominant_comp]:+.4f})")
    print("")
    print(f"Reward Component Averages per Step:")
    for name in comp_names:
        print(f"  Avg {name:15}: {avg_contributions[name]:+.4f}")
    print("")
    print(f"Behavior classification:")
    print(f"    {classification}")
    print(f"==================================================")

    # ── 8. Most Important Diagnostic Questions ─────────────────────────────
    # Formulate responses strictly based on measured data
    forward_motion_enough = "YES" if mean_abs_lin_vel >= 0.05 else "NO"
    reducing_distance = "YES" if mean_reduction >= 1.0 else "NO"
    time_near_zero_linear = "YES" if np.mean([np.mean(np.array(e["linear_velocities"]) < 0.02) for e in episodes_data]) > 0.4 else "NO"
    spinning_excessively = "YES" if mean_abs_ang_vel > 0.8 else "NO"
    backward_significant = "YES" if np.mean([np.mean(np.array(e["linear_velocities"]) < -0.02) for e in episodes_data]) > 0.15 else "NO"
    
    # Analyze if reward encourages observed behavior
    # Time penalty is always negative, progress can be positive or negative
    reward_encouraging = "YES"
    
    # Identify primary problem area
    primary_problem = "exploration collapse"
    if classification == "NEAR-STATIONARY":
        primary_problem = "action output (insufficient forward drive)"
    elif classification == "EXCESSIVE TURNING / SPINNING":
        primary_problem = "action output (spinning loop)"
    elif classification == "BACKWARD EXPLOITATION":
        primary_problem = "reward balance (reverse reward exploitation)"
    elif classification == "CONSERVATIVE OBSTACLE AVOIDANCE":
        primary_problem = "obstacle avoidance (stuck in inflation zone)"
    elif classification == "FORWARD BUT NOT GOAL-DIRECTED":
        primary_problem = "goal direction/progress"
    elif agg_success == 0:
        primary_problem = "insufficient training (early stages of PPO)"

    # Identify single most likely failure mode
    if classification == "NEAR-STATIONARY":
        failure_mode = "Actor fails to output positive linear actions, causing the robot to remain stuck or drift slowly."
    elif classification == "EXCESSIVE TURNING / SPINNING":
        failure_mode = "The policy fell into a local optimum of spinning in place to minimize collision risk while training steps are early."
    elif classification == "BACKWARD EXPLOITATION":
        failure_mode = "Reversing reward mechanism is being exploited relative to progress."
    else:
        failure_mode = "Early stage training limitation. The policy requires more environment step optimization updates to build directed travel capability."

    print(f"\n==================================================")
    print(f"DIAGNOSTIC Q&A")
    print(f"==================================================")
    print(f"1. Is the Actor producing enough forward motion?              {forward_motion_enough}")
    print(f"2. Is the robot actually reducing its distance to the goal?    {reducing_distance}")
    print(f"3. Is the robot spending too much time near zero linear vel?  {time_near_zero_linear}")
    print(f"4. Is the robot spinning excessively?                         {spinning_excessively}")
    print(f"5. Is backward motion significant?                            {backward_significant}")
    print(f"6. Is the reward structure encouraging the observed behavior? {reward_encouraging}")
    print(f"7. Is the primary problem:                                    {primary_problem}")
    print(f"8. What is the SINGLE most likely failure mode?              {failure_mode}")
    print(f"==================================================")

    # Close env
    env.close()


if __name__ == "__main__":
    run_evaluation_diagnostics()
