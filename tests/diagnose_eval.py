# tests/diagnose_eval.py
import sys
import numpy as np
import torch
import rclpy
from pathlib import Path

# Add project paths to import modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

from train import TurtleBotEnv, generate_eval_goals, Actor, Critic
from rl.rollout_buffer import RolloutBuffer
from rl.ppo_trainer import PPOTrainer

def run_action_pipeline_tests(env):
    print("\n" + "=" * 50)
    print("RUNNING ACTION PIPELINE MAPPING TESTS")
    print("=" * 50)
    
    test_actions = [
        np.array([-1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    
    for action in test_actions:
        # Scale using environment's unified method
        scaled_linear, scaled_angular = env.scale_action(action)
        
        print(f"Normalized Action:            [{action[0]:.4f}, {action[1]:.4f}]")
        print(f"  - Scaled linear.x:          {scaled_linear:.4f} m/s (expected limits: [-0.22, 0.33])")
        print(f"  - Scaled angular.z:         {scaled_angular:.4f} rad/s (expected limits: [-2.84, 2.84])")
        print(f"  - Published cmd_vel matches: linear.x={scaled_linear:.4f}, angular.z={scaled_angular:.4f}")
        print("-" * 40)

def run_diagnostic():
    print("Initializing ROS 2 Node for environment...")
    rclpy.init(args=None)
    
    # 1. Initialize environment (Observation size: 40, Action size: 2)
    env = TurtleBotEnv(num_scan_features=360)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Run action scaling checks first
    run_action_pipeline_tests(env)
    
    # 2. Fresh Model Verification
    print("\n" + "=" * 50)
    print("FRESH MODEL CONFIGURATION VERIFICATION")
    print("=" * 50)
    actor = Actor(observation_dim=obs_dim, action_dim=action_dim, hidden_sizes=(256, 256))
    critic = Critic(observation_dim=obs_dim, hidden_sizes=(256, 256))
    
    print(f"  - Actor Input Dimension:   {actor.backbone[0].in_features}")
    print(f"  - Critic Input Dimension:  {critic.backbone[0].in_features}")
    print(f"  - Actor Output Dimension:  {actor.mean_head.out_features}")
    print(f"  - Critic Output Dimension: 1")
    
    assert actor.backbone[0].in_features == 40
    assert critic.backbone[0].in_features == 40
    assert actor.mean_head.out_features == 2
    
    actor.eval()
    
    # 3. Generate 5 evaluation goals
    eval_goals = generate_eval_goals(5)
    
    # Run 5 episodes to verify behavior and physical movement
    print("\n" + "=" * 50)
    print("RUNNING 5 SANITY EVALUATION EPISODES")
    print("=" * 50)
    
    for ep in range(5):
        goal = eval_goals[ep]
        print(f"\nEpisode {ep + 1}/5 | Goal: {goal}")
        print("-" * 40)
        
        obs, info = env.reset(options={"target_position": goal})
        
        initial_dist = np.linalg.norm(goal - env.robot_pos)
        path_length = 0.0
        last_pos = env.robot_pos.copy()
        
        done = False
        step = 0
        ep_reward = 0.0
        
        linear_actions = []
        angular_actions = []
        
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                distribution = actor(obs_tensor)
                action_mean = distribution.mean.cpu().numpy()[0]
                action = torch.clamp(distribution.mean, -1.0, 1.0).cpu().numpy()[0]
                
            pub_linear_vel, pub_angular_vel = env.scale_action(action)
            linear_actions.append(action[0])
            angular_actions.append(action[1])
            
            # Log every 100 steps
            if step % 100 == 0:
                dist_to_goal = np.linalg.norm(goal - env.robot_pos)
                print(f"[Step {step:03d}]")
                print(f"  - Actor Mean (before scaling): [{action_mean[0]:.4f}, {action_mean[1]:.4f}]")
                print(f"  - Final action (clamped):      [{action[0]:.4f}, {action[1]:.4f}]")
                print(f"  - Published cmd_vel:          linear.x: {pub_linear_vel:.4f} m/s | angular.z: {pub_angular_vel:.4f} rad/s")
                print(f"  - Robot Position:              [{env.robot_pos[0]:.4f}, {env.robot_pos[1]:.4f}]")
                print(f"  - Distance to Goal:            {dist_to_goal:.4f} m")
                print(f"  - Cumulative Reward:           {ep_reward:.4f}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            
            # Path tracking
            current_pos = env.robot_pos
            path_length += float(np.linalg.norm(current_pos - last_pos))
            last_pos = current_pos.copy()
            
            done = terminated or truncated
            
        final_dist = np.linalg.norm(goal - env.robot_pos)
        mean_abs_linear = np.mean(np.abs(linear_actions))
        mean_abs_angular = np.mean(np.abs(angular_actions))
        min_linear, max_linear = np.min(linear_actions), np.max(linear_actions)
        min_angular, max_angular = np.min(angular_actions), np.max(angular_actions)
        
        print(f"\n--- Episode {ep + 1} Performance Summary ---")
        print(f"  - Mean Absolute Actions:  Linear: {mean_abs_linear:.4f} | Angular: {mean_abs_angular:.4f}")
        print(f"  - Action Ranges:          Linear: [{min_linear:.4f}, {max_linear:.4f}] | Angular: [{min_angular:.4f}, {max_angular:.4f}]")
        print(f"  - Goal Distance change:   Initial: {initial_dist:.4f} m | Final: {final_dist:.4f} m")
        print(f"  - Total Path Length:      {path_length:.4f} m")
        print(f"  - Total Steps:            {step}")
        print(f"  - Termination Type:       {'SUCCESS' if info.get('is_success') else ('COLLISION' if info.get('is_collision') else 'TIMEOUT')}")
    
    # 4. PPO Update Step Sanity Check
    print("\n" + "=" * 50)
    print("VERIFYING PPO UPDATE SUCCESS")
    print("=" * 50)
    
    rollout_size = 64
    buffer = RolloutBuffer(rollout_size=rollout_size, observation_dim=obs_dim, action_dim=action_dim, device=torch.device("cpu"))
    trainer = PPOTrainer(
        actor=actor,
        critic=critic,
        buffer=buffer,
        device=torch.device("cpu"),
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=2,
        batch_size=16
    )
    
    obs, info = env.reset()
    for _ in range(rollout_size):
        action, log_prob, value = trainer.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        trainer.store_transition(
            observation=torch.as_tensor(obs, dtype=torch.float32),
            action=action,
            reward=reward,
            done=(terminated or truncated),
            log_prob=log_prob,
            value=value
        )
        obs = next_obs
        
    trainer.finish_rollout(torch.as_tensor(obs, dtype=torch.float32))
    print("Starting optimization update...")
    metrics = trainer.update()
    print("PPO update completed successfully!")
    print(f"Metrics: actor_loss={metrics['actor_loss']:.4f} | critic_loss={metrics['critic_loss']:.4f} | entropy={metrics['entropy']:.4f}")
    
    env.close()

if __name__ == "__main__":
    run_diagnostic()
