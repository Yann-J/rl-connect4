from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from rl_connect4.envs.pettingzoo_connect4 import PettingZooConnect4GymEnv
from rl_connect4.envs.pettingzoo_connect4 import Connect4Config
from rl_connect4.training.opponent_pool import (
    load_policy_model,
    make_checkpoint_policy,
)


@dataclass
class LeagueConfig:
    n_games_per_pair: int = 5
    max_policies: int = 10
    initial_elo: float = 1000.0
    k_factor: float = 24.0
    # Match the training/eval game protocol so elite selection is consistent.
    symmetry_augmentation: bool = False
    randomize_train_agent: bool = False
    empty_cell_ratio_terminal_reward: bool = False


def _predict_masked(model, obs: np.ndarray, action_mask: np.ndarray) -> int:
    action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=action_mask.reshape(1, -1),
    )
    return int(np.asarray(action).item())


def _play_model_vs_policy(
    model,
    opponent_policy,
    n_games: int,
    *,
    seed_offset: int = 0,
    connect4_config: Connect4Config,
) -> float:
    env = PettingZooConnect4GymEnv(
        opponent_policy=opponent_policy,
        config=connect4_config,
    )
    score = 0.0
    for ep in range(n_games):
        obs, info = env.reset(seed=seed_offset + ep)
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


def _pair_score(
    model_a,
    policy_a,
    model_b,
    policy_b,
    n_games_per_pair: int,
    *,
    connect4_config: Connect4Config,
) -> float:
    """Return score for A in [0, 1], with both seating directions balanced."""
    games_a_as_train = (n_games_per_pair + 1) // 2
    games_b_as_train = n_games_per_pair // 2

    wins_a = 0.0
    if games_a_as_train > 0:
        # A plays as the train-side player.
        score_a_forward = _play_model_vs_policy(
            model_a,
            policy_b,
            n_games=games_a_as_train,
            seed_offset=0,
            connect4_config=connect4_config,
        )
        wins_a += score_a_forward * games_a_as_train
    if games_b_as_train > 0:
        # B plays as the train-side player; convert to A score.
        score_b_forward = _play_model_vs_policy(
            model_b,
            policy_a,
            n_games=games_b_as_train,
            seed_offset=games_a_as_train,
            connect4_config=connect4_config,
        )
        wins_a += (1.0 - score_b_forward) * games_b_as_train
    return wins_a / n_games_per_pair


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def run_checkpoint_league(
    checkpoint_paths: list[Path],
    config: LeagueConfig,
) -> dict[Path, float]:
    connect4_config = Connect4Config(
        symmetry_augmentation=config.symmetry_augmentation,
        randomize_train_agent=config.randomize_train_agent,
        empty_cell_ratio_terminal_reward=(
            config.empty_cell_ratio_terminal_reward
        ),
    )
    pool = checkpoint_paths[-config.max_policies:]
    if not pool:
        return {}

    elos: dict[Path, float] = {path: config.initial_elo for path in pool}
    loaded_models = {path: load_policy_model(path) for path in pool}
    loaded_policies = {path: make_checkpoint_policy(path) for path in pool}

    for path_a, path_b in combinations(pool, 2):
        score_a = _pair_score(
            loaded_models[path_a],
            loaded_policies[path_a],
            loaded_models[path_b],
            loaded_policies[path_b],
            n_games_per_pair=config.n_games_per_pair,
            connect4_config=connect4_config,
        )
        expected_a = _expected_score(elos[path_a], elos[path_b])
        delta = config.k_factor * (score_a - expected_a)
        elos[path_a] += delta
        elos[path_b] -= delta

    return elos
