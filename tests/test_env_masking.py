import pytest

from rl_connect4.envs.pettingzoo_connect4 import PettingZooConnect4GymEnv


def fixed_policy(_obs, action_mask):
    for a, valid in enumerate(action_mask):
        if valid:
            return a
    return 0


def test_action_mask_blocks_full_column():
    env = PettingZooConnect4GymEnv(opponent_policy=fixed_policy)
    obs, info = env.reset(seed=0)
    assert obs.shape == (2, 6, 7)
    assert info["action_mask"].shape == (7,)

    for _ in range(10):
        if info["action_mask"][0] == 0:
            break
        obs, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break

    if info["action_mask"][0] == 0:
        with pytest.raises(ValueError):
            env.step(0)
    env.close()

