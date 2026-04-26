from __future__ import annotations

import numpy as np
from numpy.random import Generator

from rl_connect4.envs.pettingzoo_connect4 import (
    Connect4Config,
    OpponentPolicy,
    PettingZooConnect4GymEnv,
)


def _predict_masked(model, obs: np.ndarray, action_mask: np.ndarray, deterministic: bool = True) -> int:
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_mask.reshape(1, -1))
    return int(np.asarray(action).item())


def evaluate_vs_opponent(
    model,
    opponent_policy: OpponentPolicy,
    n_episodes: int,
    *,
    random_symmetry: bool = True,
    random_side: bool = True,
    random_episode_seeds: bool = True,
    rng_seed: int | None = None,
    episode_seed_rng: Generator | None = None,
) -> dict[str, float]:
    """Run ``n_episodes`` against ``opponent_policy``.

    By default each episode uses fresh gym seeds plus random horizontal mirroring
    and random first player so aggregate win rates are meaningful.

    Pass ``episode_seed_rng`` to share one stream across several eval batches
    (e.g. MCTS then PUCT) so episode seeds do not repeat when ``rng_seed`` is fixed.
    """
    config = Connect4Config(
        symmetry_augmentation=random_symmetry,
        randomize_train_agent=random_side,
    )
    env = PettingZooConnect4GymEnv(opponent_policy=opponent_policy, config=config)
    wins = 0
    draws = 0
    losses = 0
    rewards: list[float] = []

    master = (
        episode_seed_rng
        if episode_seed_rng is not None
        else np.random.default_rng(rng_seed)
    )
    for ep in range(n_episodes):
        if random_episode_seeds:
            ep_seed = int(master.integers(0, 2**31 - 1))
        else:
            ep_seed = None if rng_seed is None else int(rng_seed) + ep
        obs, info = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        while not done:
            action = _predict_masked(model, obs, info["action_mask"], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        if ep_reward > 0:
            wins += 1
        elif ep_reward < 0:
            losses += 1
        else:
            draws += 1

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
    }

