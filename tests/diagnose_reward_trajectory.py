import os
import sys
import rclpy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

from gazebo_nav_env import GazeboMacNavEnv
from rl.actor import Actor

def main():
    rclpy.init()
    env = GazeboMacNavEnv()
    
    checkpoint_path = "checkpoints/step_000819200.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = Actor(observation_dim=obs_dim, action_dim=action_dim, hidden_sizes=(256, 256))
    ckpt = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.to(device)
    actor.eval()
    
    num_episodes = 20
    max_steps = 500
    
    all_data = []
    
    print(f"Starting reward diagnostic for {num_episodes} episodes...")
    
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        
        ep_data = []
        step_count = 0
        
        while not done and step_count < max_steps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = actor(obs_tensor)
                action = torch.clamp(dist.mean, -1.0, 1.0).cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract basic geometry from env state
            robot_x = env.robot_pos[0]
            robot_y = env.robot_pos[1]
            goal_x = env.target_position[0]
            goal_y = env.target_position[1]
            dist_to_goal = np.linalg.norm([goal_x - robot_x, goal_y - robot_y])
            prev_dist_to_goal = env.prev_distance_to_goal
            
            min_lidar = float(np.min(env.laser_ranges))
            
            # Action and speed
            lin_act, ang_act = action[0], action[1]
            lin_vel = float(env.current_linear_vel)
            ang_vel = float(env.current_angular_vel)
            
            rc = info.get("reward_components", {})
            
            step_record = {
                "episode": ep,
                "step": step_count,
                "robot_x": robot_x,
                "robot_y": robot_y,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "distance_to_goal": dist_to_goal,
                "previous_distance_to_goal": prev_dist_to_goal,
                "min_lidar_distance": min_lidar,
                "clearance_change": rc.get("clearance", 0.0) / 5.0, # reverse engineer clearance change
                "linear_action": lin_act,
                "angular_action": ang_act,
                "linear_velocity": lin_vel,
                "angular_velocity": ang_vel,
                "progress_reward": rc.get("progress", 0.0),
                "clearance_reward": rc.get("clearance", 0.0),
                "turning_penalty": rc.get("turning", 0.0),
                "time_penalty": rc.get("time", 0.0),
                "inflation_penalty": rc.get("inflation", 0.0),
                "reverse_reward_or_penalty": rc.get("reverse_penalty", 0.0) + rc.get("reverse_recovery", 0.0),
                "success_reward": rc.get("success", 0.0),
                "collision_penalty": rc.get("collision", 0.0),
                "total_reward": rc.get("total", reward),
                "is_near_obstacle": rc.get("inflation", 0.0) < 0.0,
                "collision": info.get("is_collision", False),
                "success": info.get("is_success", False),
            }
            ep_data.append(step_record)
            all_data.append(step_record)
            step_count += 1
            
        final_dist = ep_data[-1]["distance_to_goal"]
        if info.get("is_success", False):
            result = "SUCCESS"
        elif info.get("is_collision", False):
            result = "COLLISION"
        else:
            result = "TIMEOUT"
            
        print(f"Episode {ep:2d} finished in {step_count:3d} steps | Result: {result:10s} | Final Dist: {final_dist:.2f}m")
        
    df = pd.DataFrame(all_data)
    os.makedirs("diagnostics", exist_ok=True)
    df.to_csv("diagnostics/diagnostic_reward_trajectory.csv", index=False)
    
    analyze_and_plot(df, num_episodes)
    
    env.destroy_node()
    rclpy.shutdown()

def analyze_and_plot(df, num_episodes):
    print("\n" + "="*40)
    print("REWARD TRAJECTORY DIAGNOSTIC")
    print("="*40)
    
    successes = len(df[df['success'] == True]['episode'].unique())
    collisions = len(df[df['collision'] == True]['episode'].unique())
    timeouts = num_episodes - successes - collisions
    
    print(f"Episodes evaluated: {num_episodes}")
    print(f"Successes: {successes}")
    print(f"Collisions: {collisions}")
    print(f"Timeouts: {timeouts}")
    
    # Identify detour segments and difficult episodes
    difficult_episodes = []
    detour_episodes = []
    
    print("\n" + "="*40)
    print("DIFFICULT EPISODE ANALYSIS")
    print("="*40)
    
    for ep in range(1, num_episodes + 1):
        ep_df = df[df['episode'] == ep].copy()
        if len(ep_df) == 0:
            continue
            
        initial_dist = ep_df.iloc[0]['distance_to_goal']
        min_dist = ep_df['distance_to_goal'].min()
        final_dist = ep_df.iloc[-1]['distance_to_goal']
        
        # Difficult if it got stuck, collided, or if the path was highly non-monotonic
        max_dist_increase = 0
        current_increase = 0
        detour_segment = None
        
        in_detour = False
        detour_start_idx = 0
        
        for i in range(1, len(ep_df)):
            prev = ep_df.iloc[i-1]
            curr = ep_df.iloc[i]
            
            dist_diff = curr['distance_to_goal'] - prev['distance_to_goal']
            
            # If distance is increasing AND clearance is improving, we're in a detour
            if dist_diff > 0.005 and curr['clearance_reward'] > 0:
                if not in_detour:
                    in_detour = True
                    detour_start_idx = i - 1
                current_increase += dist_diff
            else:
                if in_detour:
                    in_detour = False
                    if current_increase > max_dist_increase:
                        max_dist_increase = current_increase
                        detour_segment = (detour_start_idx, i)
                    current_increase = 0
                    
        # Check if episode was difficult
        is_difficult = (ep_df['min_lidar_distance'] < 0.5).any() and (
            ep_df.iloc[-1]['collision'] or ep_df.iloc[-1]['success'] == False or max_dist_increase > 0.3
        )
        
        if is_difficult:
            difficult_episodes.append(ep)
            if max_dist_increase > 0.1:
                detour_episodes.append(ep)
            
            print(f"\nEpisode {ep}")
            print(f"Initial distance: {initial_dist:.2f} m")
            print(f"Minimum distance: {min_dist:.2f} m")
            print(f"Final distance: {final_dist:.2f} m")
            print(f"Maximum detour distance increase: {max_dist_increase:.2f} m")
            
            if detour_segment:
                s, e = detour_segment
                detour_df = ep_df.iloc[s:e]
                prog = detour_df['progress_reward'].sum()
                clr = detour_df['clearance_reward'].sum()
                trn = detour_df['turning_penalty'].sum()
                tim = detour_df['time_penalty'].sum()
                
                print("\nDuring maximal detour:")
                print(f"Progress reward: {prog:.3f}")
                print(f"Clearance reward: {clr:.3f}")
                print(f"Turning penalty: {trn:.3f}")
                print(f"Time penalty: {tim:.3f}")
                
                print("\nConclusion:")
                if prog < 0 and clr > 0 and abs(prog) > abs(clr):
                    print("- Reward heavily favors progress, overriding clearance (Detour heavily penalized)")
                elif prog < 0 and clr > 0 and abs(prog) <= abs(clr):
                    print("- Reward sufficiently rewards clearance to offset progress loss (Detour viable)")
                else:
                    print("- Inconclusive")
                    
            # Generate Plot for difficult episodes
            plot_episode(ep_df, ep)
            
    print("\n" + "="*40)
    print("FINAL DIAGNOSIS")
    print("="*40)
    
    # 1. Is progress reward overpowering obstacle-clearance reward?
    # Analyze all detour segments
    total_detour_prog = 0
    total_detour_clr = 0
    detour_count = 0
    
    for ep in detour_episodes:
        ep_df = df[df['episode'] == ep]
        for i in range(1, len(ep_df)):
            curr = ep_df.iloc[i]
            prev = ep_df.iloc[i-1]
            dist_diff = curr['distance_to_goal'] - prev['distance_to_goal']
            if dist_diff > 0.005 and curr['clearance_reward'] > 0:
                total_detour_prog += curr['progress_reward']
                total_detour_clr += curr['clearance_reward']
                detour_count += 1
                
    if detour_count > 0:
        avg_prog = total_detour_prog / detour_count
        avg_clr = total_detour_clr / detour_count
        
        print("\n1. Is progress reward overpowering obstacle-clearance reward?")
        if abs(avg_prog) > abs(avg_clr):
            print(f"   YES. During detours, avg progress loss ({avg_prog:.3f}) overpowers clearance gain ({avg_clr:.3f}).")
        else:
            print(f"   NO. During detours, avg progress loss ({avg_prog:.3f}) is offset by clearance gain ({avg_clr:.3f}).")
            
        print("\n2. Does the robot receive a net negative reward when making a useful detour?")
        if (avg_prog + avg_clr) < 0:
            print(f"   YES. The net core movement reward (Prog+Clr) is {avg_prog+avg_clr:.3f} per step.")
        else:
            print(f"   NO. The net core movement reward (Prog+Clr) is positive {avg_prog+avg_clr:.3f} per step.")
            
        print("\n3. Does the robot actually improve clearance when it turns away from the goal?")
        print(f"   YES, by definition of these segments, clearance improved by an average reward value of {avg_clr:.3f}.")
        
        print("\n4. Does the robot attempt detours?")
        print(f"   It attempted detours of >0.1m in {len(detour_episodes)} out of {len(difficult_episodes)} difficult episodes.")
        
    else:
        print("\n1-4. Inconclusive. The robot almost never made meaningful detours where clearance actually improved.")
        
    print("\n5. Does it fail because of reward incentives, insufficient turning behavior, insufficient LiDAR information, or something else?")
    if detour_count > 0 and abs(total_detour_prog) > abs(total_detour_clr):
        print("   Based on the data, the reward incentives heavily penalize moving away from the goal, dominating the clearance reward. This discourages detours.")
    elif len(detour_episodes) == 0:
        print("   The robot almost never attempts to move away from the goal to improve clearance, suggesting the policy learned a strict goal-seeking behavior, likely because turning away was heavily penalized during training.")
    else:
        print("   Requires further manual inspection.")
        
    if len(difficult_episodes) > 0:
        print(f"\nBest episodes to manually inspect in Gazebo: {difficult_episodes}")

def plot_episode(df, ep_num):
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    
    steps = df['step'].values
    
    # 1. Distances
    axs[0].plot(steps, df['distance_to_goal'], label='Goal Distance (m)', color='blue')
    axs[0].plot(steps, df['min_lidar_distance'], label='Min LiDAR Clearance (m)', color='red')
    axs[0].set_ylabel('Distance (m)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title(f'Episode {ep_num} Diagnostics')
    
    # 2. Rewards (Progress vs Clearance)
    axs[1].plot(steps, df['progress_reward'], label='Progress Reward', color='green')
    axs[1].plot(steps, df['clearance_reward'], label='Clearance Reward', color='orange')
    axs[1].set_ylabel('Reward')
    axs[1].legend()
    axs[1].grid(True)
    
    # 3. Total Reward
    axs[2].plot(steps, df['total_reward'], label='Total Reward', color='purple')
    axs[2].set_ylabel('Total Reward')
    axs[2].legend()
    axs[2].grid(True)
    
    # 4. Velocities
    axs[3].plot(steps, df['linear_velocity'], label='Linear Velocity (m/s)', color='cyan')
    axs[3].plot(steps, df['angular_velocity'], label='Angular Velocity (rad/s)', color='magenta')
    axs[3].set_xlabel('Step')
    axs[3].set_ylabel('Velocity')
    axs[3].legend()
    axs[3].grid(True)
    
    # Mark events
    col_steps = df[df['collision'] == True]['step'].values
    succ_steps = df[df['success'] == True]['step'].values
    for ax in axs:
        for cs in col_steps:
            ax.axvline(x=cs, color='r', linestyle='--', alpha=0.5, label='Collision' if ax == axs[0] else "")
        for ss in succ_steps:
            ax.axvline(x=ss, color='g', linestyle='--', alpha=0.5, label='Success' if ax == axs[0] else "")
            
    plt.tight_layout()
    plt.savefig(f"diagnostics/ep_{ep_num}_diagnostic.png")
    plt.close()

if __name__ == "__main__":
    main()
