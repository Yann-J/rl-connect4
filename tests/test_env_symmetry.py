import numpy as np

from rl_connect4.envs.pettingzoo_connect4 import (
    Connect4Config,
    PettingZooConnect4GymEnv,
)


def test_symmetry_transforms_obs_mask_and_actions():
    env = PettingZooConnect4GymEnv(
        config=Connect4Config(symmetry_augmentation=True)
    )
    env._mirror_episode = True

    raw_obs = {
        "observation": np.arange(6 * 7 * 2, dtype=np.float32).reshape(6, 7, 2),
        "action_mask": np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.int8),
    }

    obs = env._format_obs(raw_obs)
    mask = env._action_mask(raw_obs)

    assert obs.shape == (2, 6, 7)
    assert np.array_equal(mask, np.array([1, 0, 0, 1, 1, 0, 1], dtype=np.int8))
    assert env._to_env_action(0) == 6
    assert env._to_env_action(2) == 4
    assert env._to_env_action(6) == 0
    env.close()
