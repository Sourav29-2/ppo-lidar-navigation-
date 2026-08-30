"""PPO trainer for continuous TurtleBot3 velocity control."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam

from rl.rollout_buffer import RolloutBuffer


class PPOTrainer:
    """Train actor and critic networks with clipped PPO updates."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        buffer: RolloutBuffer,
        device: torch.device | str = "cpu",
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 10,
        batch_size: int = 64,
    ) -> None:
        if actor_lr <= 0.0:
            raise ValueError("actor_lr must be positive")
        if critic_lr <= 0.0:
            raise ValueError("critic_lr must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if clip_eps <= 0.0:
            raise ValueError("clip_eps must be positive")
        if ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.buffer = buffer

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample one action and return action, log probability, and value."""
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        if observation_tensor.ndim == 1:
            observation_tensor = observation_tensor.unsqueeze(0)

        with torch.no_grad():
            distribution = self.actor(observation_tensor)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
            value = self.critic(observation_tensor).reshape(-1)

        return action.squeeze(0), log_prob.reshape(()), value.reshape(())

    def store_transition(
        self,
        observation: Tensor,
        action: Tensor,
        reward: float | Tensor,
        done: bool | float | Tensor,
        log_prob: Tensor,
        value: Tensor,
    ) -> None:
        """Store a single environment transition in the rollout buffer."""
        self.buffer.add(observation, action, reward, done, log_prob, value)

    def finish_rollout(self, last_observation: Tensor) -> None:
        """Bootstrap the final value and compute GAE targets."""
        last_observation_tensor = torch.as_tensor(
            last_observation, dtype=torch.float32, device=self.device
        )
        if last_observation_tensor.ndim == 1:
            last_observation_tensor = last_observation_tensor.unsqueeze(0)

        with torch.no_grad():
            last_value = self.critic(last_observation_tensor).reshape(-1)[0]

        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def update(self) -> dict[str, float]:
        """Run PPO optimization epochs and clear the rollout buffer."""
        rollout_length = self._rollout_length()
        if rollout_length == 0:
            raise ValueError("cannot update PPOTrainer with an empty buffer")

        self._normalize_advantages(rollout_length)

        metrics: dict[str, float] = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        update_steps = 0

        for _ in range(self.ppo_epochs):
            for (
                observations,
                actions,
                old_log_probs,
                _old_values,
                advantages,
                returns,
            ) in self.buffer.get_minibatches(self.batch_size):
                distribution = self.actor(observations)
                new_log_probs = distribution.log_prob(actions).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1).mean()
                values = self.critic(observations).reshape(-1)

                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                unclipped_objective = ratio * advantages
                clipped_objective = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * advantages
                )

                actor_loss = -torch.min(
                    unclipped_objective, clipped_objective
                ).mean()
                critic_loss = F.mse_loss(values, returns)
                total_loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                metrics["actor_loss"] += actor_loss.item()
                metrics["critic_loss"] += critic_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["total_loss"] += total_loss.item()
                update_steps += 1

        self.buffer.clear()

        if update_steps == 0:
            return metrics
        return {key: value / update_steps for key, value in metrics.items()}

    def _rollout_length(self) -> int:
        return self.buffer.rollout_size if self.buffer.full else self.buffer.position

    def _normalize_advantages(self, rollout_length: int) -> None:
        advantages = self.buffer.advantages[:rollout_length]
        advantage_std = advantages.std(unbiased=False)
        self.buffer.advantages[:rollout_length] = (
            advantages - advantages.mean()
        ) / (advantage_std + 1e-8)
