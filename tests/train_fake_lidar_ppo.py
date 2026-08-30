"""Train PPO on a lightweight fake LiDAR navigation environment.

This script is a local integration demo for the PPO pipeline:
FakeLidarEnv -> Actor/Critic -> RolloutBuffer -> GAE -> PPOTrainer update.
It does not use ROS2, Gazebo, camera, or real robot code.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rl.actor import Actor  # noqa: E402
from rl.critic import Critic  # noqa: E402
from rl.ppo_trainer import PPOTrainer  # noqa: E402
from rl.rollout_buffer import RolloutBuffer  # noqa: E402


class FakeLidarEnv:
    """Small 2D continuous-control navigation task with 360 fake LiDAR rays."""

    observation_dim = 364
    action_dim = 2

    def __init__(
        self,
        seed: int = 7,
        max_steps: int = 250,
        world_size: float = 6.0,
        max_lidar_range: float = 6.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.world_size = world_size
        self.half_world = world_size / 2.0
        self.max_lidar_range = max_lidar_range
        self.goal_radius = 0.25
        self.robot_radius = 0.15
        self.dt = 0.1
        self.action_low = np.array([-0.35, -1.5], dtype=np.float32)
        self.action_high = np.array([0.55, 1.5], dtype=np.float32)
        self.ray_angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        self.obstacles = np.array(
            [
                [-1.20, -0.50, 0.45],
                [0.65, 0.85, 0.40],
                [1.35, -1.05, 0.35],
                [-0.45, 1.35, 0.30],
            ],
            dtype=np.float32,
        )
        self.position = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.previous_distance = 0.0

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        """Reset robot and goal to opposite sides of the map."""
        self.position = np.array([-2.4, -2.2], dtype=np.float32)
        self.position += self.rng.normal(0.0, 0.12, size=2).astype(np.float32)
        self.heading = float(self.rng.uniform(-0.25, 0.25))
        self.goal = np.array([2.35, 2.15], dtype=np.float32)
        self.goal += self.rng.normal(0.0, 0.12, size=2).astype(np.float32)
        self.steps = 0
        self.previous_distance = self._distance_to_goal()
        return self._observation(), {"distance_to_goal": self.previous_distance}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        """Apply linear/angular velocity and return a Gymnasium-like step tuple."""
        linear_velocity, angular_velocity = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_low,
            self.action_high,
        )

        self.heading = self._wrap_angle(self.heading + float(angular_velocity) * self.dt)
        direction = np.array(
            [math.cos(self.heading), math.sin(self.heading)],
            dtype=np.float32,
        )
        self.position = self.position + direction * float(linear_velocity) * self.dt
        self.steps += 1

        distance = self._distance_to_goal()
        progress = self.previous_distance - distance
        self.previous_distance = distance

        collision = self._has_collision()
        reached_goal = distance <= self.goal_radius
        truncated = self.steps >= self.max_steps
        terminated = collision or reached_goal

        reward = 8.0 * progress - 0.01
        reward -= 0.015 * abs(float(angular_velocity))
        if reached_goal:
            reward += 10.0
        if collision:
            reward -= 10.0

        info = {
            "distance_to_goal": distance,
            "reached_goal": float(reached_goal),
            "collision": float(collision),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        lidar = self._fake_lidar() / self.max_lidar_range
        goal_vector = self.goal - self.position
        distance = np.linalg.norm(goal_vector)
        goal_angle = math.atan2(float(goal_vector[1]), float(goal_vector[0]))
        relative_bearing = self._wrap_angle(goal_angle - self.heading)
        normalized_distance = distance / (math.sqrt(2.0) * self.world_size)

        robot_state = np.array(
            [
                normalized_distance,
                math.cos(relative_bearing),
                math.sin(relative_bearing),
                self.heading / math.pi,
            ],
            dtype=np.float32,
        )
        return np.concatenate([lidar.astype(np.float32), robot_state])

    def _fake_lidar(self) -> np.ndarray:
        angles = self.heading + self.ray_angles
        ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        distances = self._wall_distances(ray_dirs)

        for obstacle_x, obstacle_y, obstacle_radius in self.obstacles:
            center = np.array([obstacle_x, obstacle_y], dtype=np.float32)
            offset = self.position - center
            b = 2.0 * (ray_dirs @ offset)
            c = float(offset @ offset - obstacle_radius**2)
            discriminant = b * b - 4.0 * c
            hit_mask = discriminant >= 0.0
            roots = np.full_like(distances, self.max_lidar_range)
            roots[hit_mask] = (-b[hit_mask] - np.sqrt(discriminant[hit_mask])) / 2.0
            valid_roots = roots > 0.0
            distances = np.minimum(distances, np.where(valid_roots, roots, distances))

        return np.clip(distances, 0.0, self.max_lidar_range)

    def _wall_distances(self, ray_dirs: np.ndarray) -> np.ndarray:
        distances = np.full(360, self.max_lidar_range, dtype=np.float32)
        x, y = float(self.position[0]), float(self.position[1])

        for axis, boundary in ((0, self.half_world), (0, -self.half_world)):
            direction = ray_dirs[:, axis]
            valid = np.abs(direction) > 1e-6
            t = np.full(360, self.max_lidar_range, dtype=np.float32)
            t[valid] = (boundary - x) / direction[valid]
            distances = np.minimum(distances, np.where(t > 0.0, t, distances))

        for axis, boundary in ((1, self.half_world), (1, -self.half_world)):
            direction = ray_dirs[:, axis]
            valid = np.abs(direction) > 1e-6
            t = np.full(360, self.max_lidar_range, dtype=np.float32)
            t[valid] = (boundary - y) / direction[valid]
            distances = np.minimum(distances, np.where(t > 0.0, t, distances))

        return distances

    def _has_collision(self) -> bool:
        outside_world = np.any(np.abs(self.position) >= self.half_world)
        if outside_world:
            return True

        for obstacle_x, obstacle_y, obstacle_radius in self.obstacles:
            obstacle = np.array([obstacle_x, obstacle_y], dtype=np.float32)
            if np.linalg.norm(self.position - obstacle) <= obstacle_radius + self.robot_radius:
                return True
        return False

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rolling_average(values: list[float], window: int) -> np.ndarray:
    """Return rolling average values with a fixed window."""
    if len(values) < window:
        return np.array([], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(np.asarray(values, dtype=np.float32), kernel, mode="valid")


def plot_rewards(episode_rewards: list[float], output_path: Path) -> None:
    """Save episode reward and rolling average reward curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, color="tab:blue", alpha=0.25, label="Episode reward")

    for window, color in ((10, "tab:orange"), (25, "tab:red")):
        averages = rolling_average(episode_rewards, window)
        if len(averages) > 0:
            plt.plot(
                range(window - 1, window - 1 + len(averages)),
                averages,
                color=color,
                linewidth=2,
                label=f"{window}-episode rolling average",
            )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Fake LiDAR Navigation Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train(args: argparse.Namespace) -> None:
    """Train PPO against FakeLidarEnv and print reward/distance progress."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    env = FakeLidarEnv(seed=args.seed, max_steps=args.max_episode_steps)

    actor = Actor(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_sizes=(256, 256),
    )
    critic = Critic(
        observation_dim=env.observation_dim,
        hidden_sizes=(256, 256),
    )
    buffer = RolloutBuffer(
        rollout_size=args.rollout_size,
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        device=device,
    )
    trainer = PPOTrainer(
        actor=actor,
        critic=critic,
        buffer=buffer,
        device=device,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
    )

    observation, info = env.reset()
    episode_reward = 0.0
    episode_rewards: list[float] = []
    final_distances: list[float] = []

    for iteration in range(args.iterations):
        # Linear learning rate decay
        frac = 1.0 - (iteration / args.iterations)
        for param_group in trainer.actor_optimizer.param_groups:
            param_group["lr"] = args.actor_lr * frac
        for param_group in trainer.critic_optimizer.param_groups:
            param_group["lr"] = args.critic_lr * frac

        for _step in range(args.rollout_size):
            action, log_prob, value = trainer.select_action(observation)

            next_observation, reward, terminated, truncated, info = env.step(
                action.detach().cpu().numpy()
            )
            
            # Fix Time-Limit Truncation Bug
            if truncated and not terminated:
                with torch.no_grad():
                    next_obs_tensor = torch.as_tensor(
                        next_observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    next_value = critic(next_obs_tensor).item()
                    reward += trainer.gamma * next_value

            done = terminated or truncated

            trainer.store_transition(
                observation=torch.as_tensor(observation, dtype=torch.float32),
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
            )

            episode_reward += reward
            observation = next_observation

            if done:
                episode_rewards.append(episode_reward)
                final_distances.append(float(info["distance_to_goal"]))
                episode_reward = 0.0
                observation, info = env.reset()

        trainer.finish_rollout(torch.as_tensor(observation, dtype=torch.float32))
        metrics = trainer.update()

        recent_rewards = episode_rewards[-10:] or [episode_reward]
        recent_distances = final_distances[-10:] or [float(info["distance_to_goal"])]
        print(
            "iteration={iteration:03d} "
            "avg_reward_10={reward:.2f} "
            "avg_final_distance_10={distance:.3f} "
            "actor_loss={actor_loss:.4f} "
            "critic_loss={critic_loss:.4f} "
            "entropy={entropy:.4f}".format(
                iteration=iteration + 1,
                reward=float(np.mean(recent_rewards)),
                distance=float(np.mean(recent_distances)),
                actor_loss=metrics["actor_loss"],
                critic_loss=metrics["critic_loss"],
                entropy=metrics["entropy"],
            )
        )

    if episode_rewards:
        plot_rewards(episode_rewards, PROJECT_ROOT / "outputs" / "fake_lidar_rewards.png")

    if len(final_distances) >= 20:
        early_distance = float(np.mean(final_distances[:10]))
        late_distance = float(np.mean(final_distances[-10:]))
        print(f"Early avg final distance: {early_distance:.3f}")
        print(f"Late avg final distance: {late_distance:.3f}")
        if late_distance < early_distance:
            print("Result: agent is getting closer to the goal over time.")
        else:
            print("Result: goal distance did not improve yet; train longer or tune rewards.")

    print("Saved reward plot to outputs/fake_lidar_rewards.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--rollout-size", type=int, default=2048)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
