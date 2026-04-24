import numpy as np

from rl_connect4.policies.rule_based import rule_based_policy


def _empty_obs() -> np.ndarray:
    return np.zeros((2, 6, 7), dtype=np.float32)


def test_rule_policy_wins_when_possible():
    obs = _empty_obs()
    # Current player has three in a row at bottom row, can win with col 3.
    obs[0, 5, 0] = 1
    obs[0, 5, 1] = 1
    obs[0, 5, 2] = 1
    mask = np.ones((7,), dtype=np.int8)
    assert rule_based_policy(obs, mask) == 3


def test_rule_policy_blocks_immediate_loss():
    obs = _empty_obs()
    # Opponent has three in a row at bottom row, must block at col 3.
    obs[1, 5, 0] = 1
    obs[1, 5, 1] = 1
    obs[1, 5, 2] = 1
    mask = np.ones((7,), dtype=np.int8)
    assert rule_based_policy(obs, mask) == 3
