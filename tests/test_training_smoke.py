from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_connect4.envs.pettingzoo_connect4 import (
    PettingZooConnect4GymEnv,
    random_legal_policy,
)
from rl_connect4.policies.cnn_policy import Connect4CNNExtractor


def make_env():
    env = PettingZooConnect4GymEnv(opponent_policy=random_legal_policy)
    return ActionMasker(env, lambda e: e.action_masks())


def test_short_training_smoke():
    env = DummyVecEnv([make_env])
    model = MaskablePPO(
        policy="CnnPolicy",
        env=env,
        n_steps=64,
        batch_size=64,
        learning_rate=3e-4,
        policy_kwargs={
            "features_extractor_class": Connect4CNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": 64,
                "channels": 32,
                "num_res_blocks": 2,
            },
        },
        verbose=0,
    )
    model.learn(total_timesteps=128)

    obs = env.reset()
    masks = env.envs[0].action_masks().reshape(1, -1)
    action, _ = model.predict(obs, action_masks=masks)
    assert 0 <= int(action[0]) <= 6
