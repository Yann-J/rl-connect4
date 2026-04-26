from rl_connect4.envs.pettingzoo_connect4 import (
    Connect4Config,
    PettingZooConnect4GymEnv,
)


def opponent_far_column(_obs, action_mask):
    return 6 if action_mask[6] else next(i for i, v in enumerate(action_mask) if v)


def test_sparse_reward_terminal_win():
    env = PettingZooConnect4GymEnv(opponent_policy=opponent_far_column)
    obs, info = env.reset(seed=42)
    rewards = []
    done = False

    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        rewards.append(reward)
        done = terminated or truncated

    assert rewards[-1] == 1.0
    assert all(r == 0.0 for r in rewards[:-1])
    env.close()


def test_empty_cell_ratio_terminal_reward_increases_win_bonus():
    env = PettingZooConnect4GymEnv(
        opponent_policy=opponent_far_column,
        config=Connect4Config(empty_cell_ratio_terminal_reward=True),
    )
    obs, info = env.reset(seed=42)
    rewards = []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        rewards.append(reward)
        done = terminated or truncated
    assert rewards[-1] > 1.0
    env.close()

