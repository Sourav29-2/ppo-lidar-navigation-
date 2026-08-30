import os
import sys

# Set environment variable to bypass duplicate OpenMP library loading error on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Guarantee that the script directory is in the Python search path for direct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from gazebo_nav_env import GazeboMacNavEnv

if __name__ == '__main__':
    # Launch the custom Mac-Gazebo bridge environment (with 24 scan features for fast convergence)
    env = GazeboMacNavEnv(num_scan_features=24)
    
    # Custom policy architecture: set initial action std to 0.50 (log_std_init = -0.7) for smooth velocity execution
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-0.7
    )
    
    # Instantiate Stable-Baselines3 PPO policy with custom parameters for mobile navigation
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,            # Long-term horizon scaling factor
        gae_lambda=0.95,
        clip_range=0.2,        # PPO clipping threshold bound
        ent_coef=0.01          # Entropy coefficient to encourage room door exploration
    )
    
    print("Starting baseline PPO training inside native Gazebo...")
    model.learn(total_timesteps=150000)
    
    # Archive final weights
    model.save("native_mac_gazebo_policy")
    print("Policy saved successfully.")
    
    env.close()
