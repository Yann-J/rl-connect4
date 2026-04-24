from rl_connect4.envs.pettingzoo_connect4 import PettingZooConnect4GymEnv


def opponent_far_column(_obs, action_mask):
    return 6 if action_mask[6] else next(i for i, v in enumerate(action_mask) if v)


def test_sparse_reward_terminal_win():
    env = PettingZooConnect4GymEnv(opponent_policy=opponent_far_column)
    _, info = env.reset(seed=42)
    rewards = []
    done = False

    while not done:
        _, reward, terminated, truncated, info = env.step(0)
        rewards.append(reward)
        done = terminated or truncated

    assert rewards[-1] == 1.0
    assert all(r == 0.0 for r in rewards[:-1])
    env.close()

