import numpy as np

from rl_connect4.training.opponent_pool import (
    OpponentMix,
    OpponentPool,
    make_model_policy,
)


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


def test_make_model_policy_is_deterministic_for_opponents():
    class DummyModel:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def predict(self, _observation, deterministic=False, **_kwargs):
            self.calls.append(bool(deterministic))
            return np.array([3], dtype=np.int64), None

    model = DummyModel()
    policy = make_model_policy(model, deterministic=True)
    obs = np.zeros((2, 6, 7), dtype=np.float32)
    mask = np.ones((7,), dtype=np.int8)
    action = policy(obs, mask)

    assert action == 3
    assert model.calls == [True]


def test_historical_sampling_is_recency_biased():
    pool = OpponentPool(
        OpponentMix(
            current=0.0,
            historical=1.0,
            random=0.0,
            mcts=0.0,
            rule_based=0.0,
        ),
        mcts_simulations=10,
    )
    # Inject synthetic policies to isolate sampling behavior.
    pool._historical_policies = [lambda *_: 0, lambda *_: 1, lambda *_: 2]
    pool._historical_sampling_probs = np.array(
        [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]
    )

    counts = [0, 0, 0]
    obs = np.zeros((2, 6, 7), dtype=np.float32)
    mask = np.ones((7,), dtype=np.int8)
    for _ in range(2000):
        sampled = pool.sample()
        action = sampled(obs, mask)
        counts[action] += 1

    assert counts[2] > counts[1] > counts[0]
