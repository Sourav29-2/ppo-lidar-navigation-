"""Critic network for PPO value estimation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Critic(nn.Module):
    """Value network that maps an observation to a scalar state value."""

    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")

        self.backbone = self._build_mlp(
            input_dim=observation_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
        self.value_head = self._layer_init(nn.Linear(hidden_sizes[-1], 1), std=1.0)

    def forward(self, observation: Tensor) -> Tensor:
        """Estimate state value for one observation or a batch of observations."""
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2:
            raise ValueError(
                "observation must have shape [batch_size, observation_dim]"
            )

        features = self.backbone(observation)
        value = self.value_head(features)
        return value

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
        """Create the shared MLP backbone used before the value head."""
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_sizes:
            if hidden_dim <= 0:
                raise ValueError("all hidden layer sizes must be positive")
            layers.append(Critic._layer_init(nn.Linear(previous_dim, hidden_dim)))
            layers.append(activation())
            previous_dim = hidden_dim

        return nn.Sequential(*layers)
