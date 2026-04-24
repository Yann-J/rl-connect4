from __future__ import annotations

import numpy as np

from rl_connect4.envs.pettingzoo_connect4 import OpponentPolicy, PettingZooConnect4GymEnv


def _predict_masked(model, obs: np.ndarray, action_mask: np.ndarray, deterministic: bool = True) -> int:
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_mask.reshape(1, -1))
    return int(np.asarray(action).item())


def evaluate_vs_opponent(
    model,
    opponent_policy: OpponentPolicy,
    n_episodes: int,
    seed: int | None = None,
) -> dict[str, float]:
    env = PettingZooConnect4GymEnv(opponent_policy=opponent_policy)
    wins = 0
    draws = 0
    losses = 0
    rewards: list[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=None if seed is None else seed + ep)
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

