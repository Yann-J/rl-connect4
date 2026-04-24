import numpy as np

from rl_connect4.training.opponent_pool import OpponentMix, OpponentPool


def test_pool_can_sample_mcts_policy():
    pool = OpponentPool(
        OpponentMix(
            current=0.0,
            historical=0.0,
            random=0.0,
            mcts=1.0,
            rule_based=0.0,
        ),
        mcts_simulations=10,
    )
    policy = pool.sample()
    obs = np.zeros((2, 6, 7), dtype=np.float32)
    mask = np.ones((7,), dtype=np.int8)
    action = policy(obs, mask)
    assert 0 <= action <= 6


def test_pool_can_sample_rule_based_policy():
    pool = OpponentPool(
        OpponentMix(
            current=0.0,
            historical=0.0,
            random=0.0,
            mcts=0.0,
            rule_based=1.0,
        ),
        mcts_simulations=10,
    )
    policy = pool.sample()
    obs = np.zeros((2, 6, 7), dtype=np.float32)
    mask = np.ones((7,), dtype=np.int8)
    action = policy(obs, mask)
    assert 0 <= action <= 6
