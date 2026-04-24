from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.activation = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = x
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        return self.activation(out + residual)


class Connect4CNNExtractor(BaseFeaturesExtractor):
    """Residual CNN backbone followed by a projection layer."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        channels: int = 64,
        num_res_blocks: int = 6,
    ):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")

        blocks: list[nn.Module] = [
            nn.Conv2d(
                n_input_channels, channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        ]
        blocks.extend(ResidualBlock(channels) for _ in range(num_res_blocks))
        blocks.append(nn.Flatten())
        self.cnn = nn.Sequential(
            *blocks
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
