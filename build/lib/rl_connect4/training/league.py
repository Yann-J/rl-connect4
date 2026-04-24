from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from rl_connect4.envs.pettingzoo_connect4 import PettingZooConnect4GymEnv
from rl_connect4.training.opponent_pool import make_checkpoint_policy


@dataclass
class LeagueConfig:
    n_games_per_pair: int = 5
    max_policies: int = 10
    initial_elo: float = 1000.0
    k_factor: float = 24.0


def _predict_masked(model, obs: np.ndarray, action_mask: np.ndarray) -> int:
    action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=action_mask.reshape(1, -1),
    )
    return int(np.asarray(action).item())


def _play_model_vs_policy(model, opponent_policy, n_games: int) -> float:
    env = PettingZooConnect4GymEnv(opponent_policy=opponent_policy)
    score = 0.0
    for ep in range(n_games):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            action = _predict_masked(model, obs, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        if ep_reward > 0:
            score += 1.0
        elif ep_reward == 0:
            score += 0.5
    env.close()
    return score / n_games


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def run_checkpoint_league(
    checkpoint_paths: list[Path],
    config: LeagueConfig,
) -> dict[Path, float]:
    pool = checkpoint_paths[-config.max_policies:]
    if not pool:
        return {}

    elos: dict[Path, float] = {path: config.initial_elo for path in pool}
    loaded_models = {path: MaskablePPO.load(str(path)) for path in pool}
    loaded_policies = {path: make_checkpoint_policy(path) for path in pool}

    for path_a, path_b in combinations(pool, 2):
        score_a = _play_model_vs_policy(
            loaded_models[path_a],
            loaded_policies[path_b],
            n_games=config.n_games_per_pair,
        )
        expected_a = _expected_score(elos[path_a], elos[path_b])
        delta = config.k_factor * (score_a - expected_a)
        elos[path_a] += delta
        elos[path_b] -= delta

    return elos
