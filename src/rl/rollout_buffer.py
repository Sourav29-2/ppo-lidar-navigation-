import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, rollout_size: int, observation_dim: int, action_dim: int, device: torch.device):
        self.rollout_size = rollout_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        
        self.observations = torch.zeros((rollout_size, observation_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((rollout_size, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros(rollout_size, dtype=torch.float32).to(device)
        self.dones = torch.zeros(rollout_size, dtype=torch.float32).to(device)
        self.log_probs = torch.zeros(rollout_size, dtype=torch.float32).to(device)
        self.values = torch.zeros(rollout_size, dtype=torch.float32).to(device)
        
        self.advantages = torch.zeros(rollout_size, dtype=torch.float32).to(device)
        self.returns = torch.zeros(rollout_size, dtype=torch.float32).to(device)
        
        self.step = 0

    def store_transition(self, observation: torch.Tensor, action: torch.Tensor, reward: float, done: bool, log_prob: torch.Tensor, value: torch.Tensor):
        self.observations[self.step] = observation.to(self.device)
        self.actions[self.step] = action.to(self.device)
        self.rewards[self.step] = float(reward)
        self.dones[self.step] = 1.0 if done else 0.0
        self.log_probs[self.step] = log_prob.to(self.device)
        self.values[self.step] = value.to(self.device)
        self.step += 1

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        last_gae_lam = 0
        for step in reversed(range(self.rollout_size)):
            if step == self.rollout_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def reset(self):
        self.step = 0
