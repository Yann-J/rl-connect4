from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sb3_contrib import MaskablePPO

from rl_connect4.envs.pettingzoo_connect4 import (
    OpponentPolicy,
    random_legal_policy,
)
from rl_connect4.mcts.mcts import make_mcts_policy
from rl_connect4.policies.rule_based import rule_based_policy


class PredictModel(Protocol):
    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
        **kwargs,
    ): ...


def _masked_predict(
    model: PredictModel,
    obs: np.ndarray,
    action_mask: np.ndarray,
    *,
    deterministic: bool,
) -> int:
    action, _ = model.predict(
        obs,
        deterministic=deterministic,
        action_masks=action_mask.reshape(1, -1),
    )
    return int(np.asarray(action).item())


def make_model_policy(
    model: PredictModel, *, deterministic: bool = True
) -> OpponentPolicy:
    def _policy(obs: np.ndarray, action_mask: np.ndarray) -> int:
        return _masked_predict(
            model,
            obs,
            action_mask,
            deterministic=deterministic,
        )

    return _policy


def make_checkpoint_policy(
    checkpoint_path: str | Path, *, deterministic: bool = True
) -> OpponentPolicy:
    model = MaskablePPO.load(str(checkpoint_path))

    def _policy(obs: np.ndarray, action_mask: np.ndarray) -> int:
        return _masked_predict(
            model,
            obs,
            action_mask,
            deterministic=deterministic,
        )

    return _policy


@dataclass
class OpponentMix:
    current: float = 0.5
    historical: float = 0.3
    random: float = 0.1
    mcts: float = 0.05
    rule_based: float = 0.05


class OpponentPool:
    def __init__(self, mix: OpponentMix, mcts_simulations: int = 100) -> None:
        probs = np.array(
            [
                mix.current,
                mix.historical,
                mix.random,
                mix.mcts,
                mix.rule_based,
            ],
            dtype=np.float64,
        )
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError("Opponent mix probabilities must sum to 1.0")
        self._probs = probs
        self._mcts_simulations = mcts_simulations
        self._mcts_policy = make_mcts_policy(
            simulations=self._mcts_simulations
        )
        self._current_policy: OpponentPolicy | None = None
        self._historical_policies: list[OpponentPolicy] = []
        self._historical_sampling_probs: np.ndarray = np.array(
            [], dtype=np.float64
        )
        self._elite_policy: OpponentPolicy | None = None

    def set_current_model(self, model: PredictModel) -> None:
        self._current_policy = make_model_policy(model, deterministic=True)

    def refresh_historical(self, checkpoint_paths: list[Path]) -> None:
        self._historical_policies = [
            make_checkpoint_policy(path, deterministic=True)
            for path in checkpoint_paths
        ]
        n_policies = len(self._historical_policies)
        if n_policies == 0:
            self._historical_sampling_probs = np.array([], dtype=np.float64)
            return
        # Recency bias: newer checkpoints are sampled more often.
        weights = np.arange(1, n_policies + 1, dtype=np.float64)
        self._historical_sampling_probs = weights / weights.sum()

    def set_elite_checkpoint(self, checkpoint_path: Path | None) -> None:
        self._elite_policy = (
            None
            if checkpoint_path is None
            else make_checkpoint_policy(checkpoint_path, deterministic=True)
        )

    def set_mix(self, mix: OpponentMix) -> None:
        probs = np.array(
            [
                mix.current,
                mix.historical,
                mix.random,
                mix.mcts,
                mix.rule_based,
            ],
            dtype=np.float64,
        )
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError("Opponent mix probabilities must sum to 1.0")
        self._probs = probs

    def set_mcts_simulations(self, simulations: int) -> None:
        self._mcts_simulations = int(simulations)
        self._mcts_policy = make_mcts_policy(
            simulations=self._mcts_simulations
        )

    def sample(self) -> OpponentPolicy:
        choices = ["current", "historical", "random", "mcts", "rule_based"]
        choice = np.random.choice(choices, p=self._probs)
        if choice == "current" and self._current_policy is not None:
            return self._current_policy
        if choice == "historical":
            candidates = list(self._historical_policies)
            probs = (
                self._historical_sampling_probs.copy()
                if self._historical_sampling_probs.size > 0
                else np.array([], dtype=np.float64)
            )
            if self._elite_policy is not None:
                candidates.append(self._elite_policy)
                elite_weight = (
                    float(self._historical_sampling_probs[-1])
                    if self._historical_sampling_probs.size > 0
                    else 1.0
                )
                probs = (
                    np.append(probs, elite_weight)
                    if probs.size > 0
                    else np.array([elite_weight], dtype=np.float64)
                )
            if candidates:
                if probs.size != len(candidates):
                    probs = np.ones(len(candidates), dtype=np.float64)
                probs /= probs.sum()
                idx = int(np.random.choice(len(candidates), p=probs))
                return candidates[idx]
        if choice == "mcts":
            return self._mcts_policy
        if choice == "rule_based":
            return rule_based_policy
        return random_legal_policy
