"""Actor network for continuous mobile robot control.

The actor maps an observation vector to a Gaussian policy over two actions:

1. ``linear.x``  - forward/backward velocity command.
2. ``angular.z`` - yaw rotation velocity command.

This module intentionally returns a ``torch.distributions.Normal`` object instead
of sampled actions. PPO-specific sampling, log-probability storage, clipping, and
advantage logic should live outside this file.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Gaussian actor network for two continuous robot velocity actions.

    Args:
        observation_dim: Number of scalar values in one observation.
        hidden_sizes: Width of each hidden layer in the shared MLP backbone.
        activation: Activation module used after each hidden layer.
        log_std_init: Initial value for the learnable log standard deviation.
        min_log_std: Lower clamp used for numerical stability.
        max_log_std: Upper clamp used for numerical stability.

    Notes:
        ``log_std`` is a state-independent learnable parameter. This is a common
        first PPO design because it is simple, stable, and keeps exploration
        separate from the observation-dependent mean head. We can switch to an
        observation-dependent log-std head later if the robot needs more adaptive
        exploration.
    """


    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        action_dim: int = 1,
        activation: type[nn.Module] = nn.Tanh,
        log_std_init: float = -0.5,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Shared feature extractor for LiDAR/robot state observations.
        self.backbone = self._build_mlp(
            input_dim=observation_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        # Mean action head: [linear.x mean, angular.z mean].
        self.mean_head = self._layer_init(nn.Linear(hidden_sizes[-1], self.action_dim), std=0.01)

        # Learnable, state-independent log standard deviation for both actions.
        self.log_std = nn.Parameter(torch.full((self.action_dim,), log_std_init))

    def forward(self, observation: Tensor) -> Normal:
        """Build the Gaussian action distribution for a batch of observations.

        Args:
            observation: Tensor with shape ``[batch_size, observation_dim]`` or
                ``[observation_dim]`` for a single observation.

        Returns:
            A Normal distribution whose mean and standard deviation have shape
            ``[batch_size, 2]``. The two action dimensions correspond to
            ``linear.x`` and ``angular.z``.
        """
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2:
            raise ValueError("observation must have shape [batch_size, observation_dim]")

        features = self.backbone(observation)
        mean = self.mean_head(features)

        # Clamping keeps std away from zero and very large values during training.
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        std = log_std.exp().expand_as(mean)

        return Normal(loc=mean, scale=std)

    @staticmethod
    def _layer_init(layer: nn.Module, std: float = 2**0.5, bias_const: float = 0.0) -> nn.Module:
        """Initialize layers with orthogonal weights for PPO stability."""
        if isinstance(layer, nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_sizes: Sequence[int],
        activation: type[nn.Module],
    ) -> nn.Sequential:
        """Create the shared MLP backbone used before policy output heads."""
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_sizes:
            if hidden_dim <= 0:
                raise ValueError("all hidden layer sizes must be positive")
            layers.append(Actor._layer_init(nn.Linear(previous_dim, hidden_dim)))
            layers.append(activation())
            previous_dim = hidden_dim

        return nn.Sequential(*layers)
