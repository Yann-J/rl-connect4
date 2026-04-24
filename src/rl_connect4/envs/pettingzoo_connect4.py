from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.classic import connect_four_v3


OpponentPolicy = Callable[[np.ndarray, np.ndarray], int]
OpponentSampler = Callable[[], OpponentPolicy]


def random_legal_policy(_obs: np.ndarray, action_mask: np.ndarray) -> int:
    legal_actions = np.flatnonzero(action_mask)
    return int(np.random.choice(legal_actions))


@dataclass
class Connect4Config:
    train_agent: str = "player_0"
    render_mode: str | None = None
    symmetry_augmentation: bool = False
    randomize_train_agent: bool = False


class PettingZooConnect4GymEnv(gym.Env[np.ndarray, int]):
    """Single-agent Gym wrapper around PettingZoo Connect Four."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        opponent_policy: OpponentPolicy | None = None,
        opponent_sampler: OpponentSampler | None = None,
        config: Connect4Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config or Connect4Config()
        self.opponent_policy = opponent_policy or random_legal_policy
        self.opponent_sampler = opponent_sampler
        self.env = connect_four_v3.env(render_mode=self.config.render_mode)
        self._mirror_episode = False
        self._train_agent = self.config.train_agent

        self.action_space = spaces.Discrete(7)
        # 2 channels x 6 rows x 7 columns
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, 6, 7),
            dtype=np.float32,
        )

    def set_opponent_policy(self, policy: OpponentPolicy) -> None:
        self.opponent_policy = policy

    def _is_done(self) -> bool:
        return any(self.env.terminations.values()) or any(
            self.env.truncations.values()
        )

    def _raw_obs(self, agent: str) -> dict:
        return self.env.observe(agent)

    def _format_obs(self, raw_obs: dict) -> np.ndarray:
        board = raw_obs["observation"]
        # PettingZoo provides (6, 7, 2); SB3 CNN expects (C, H, W)
        obs = np.transpose(board, (2, 0, 1)).astype(np.float32)
        if self._mirror_episode:
            obs = obs[:, :, ::-1].copy()
        return obs

    def _action_mask(self, raw_obs: dict) -> np.ndarray:
        mask = np.asarray(raw_obs["action_mask"], dtype=np.int8)
        if self._mirror_episode:
            mask = mask[::-1].copy()
        return mask

    def _to_env_action(self, action: int) -> int:
        if self._mirror_episode:
            return self.action_space.n - 1 - int(action)
        return int(action)

    def _advance_until_train_turn(self) -> None:
        while (
            not self._is_done()
            and self.env.agent_selection != self._train_agent
        ):
            agent = self.env.agent_selection
            raw_obs = self._raw_obs(agent)
            action_mask = self._action_mask(raw_obs)
            action = self.opponent_policy(
                self._format_obs(raw_obs), action_mask
            )
            if action_mask[action] == 0:
                action = int(np.flatnonzero(action_mask)[0])
            self.env.step(self._to_env_action(action))

    def _current_info(self) -> dict:
        if self._is_done():
            return {
                "action_mask": np.zeros(
                    self.action_space.n, dtype=np.int8
                )
            }
        raw_obs = self._raw_obs(self._train_agent)
        return {"action_mask": self._action_mask(raw_obs)}

    def action_masks(self) -> np.ndarray:
        return self._current_info()["action_mask"].astype(bool)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._mirror_episode = bool(
            self.config.symmetry_augmentation and np.random.random() < 0.5
        )
        if self.config.randomize_train_agent:
            self._train_agent = str(
                np.random.choice(["player_0", "player_1"])
            )
        else:
            self._train_agent = self.config.train_agent
        if self.opponent_sampler is not None:
            self.opponent_policy = self.opponent_sampler()
        self.env.reset(seed=seed, options=options)
        self._advance_until_train_turn()
        raw_obs = self._raw_obs(self._train_agent)
        return self._format_obs(raw_obs), self._current_info()

    def step(self, action: int):
        if self._is_done():
            raise RuntimeError(
                "Cannot step() a finished episode. Call reset()."
            )
        if self.env.agent_selection != self._train_agent:
            raise RuntimeError(
                "Environment desynced: not training agent turn."
            )

        raw_obs = self._raw_obs(self._train_agent)
        action_mask = self._action_mask(raw_obs)
        if action_mask[action] == 0:
            raise ValueError(
                f"Illegal action {action} for current board state."
            )

        self.env.step(self._to_env_action(action))
        self._advance_until_train_turn()

        terminated = self._is_done()
        truncated = False
        reward = (
            float(self.env.rewards[self._train_agent])
            if terminated
            else 0.0
        )
        if terminated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = self._format_obs(self._raw_obs(self._train_agent))
        return next_obs, reward, terminated, truncated, self._current_info()

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()
