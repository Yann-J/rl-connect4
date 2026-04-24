from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rl_connect4.envs.pettingzoo_connect4 import (
    Connect4Config,
    PettingZooConnect4GymEnv,
)
from rl_connect4.policies.cnn_policy import Connect4CNNExtractor
from rl_connect4.training.callbacks import (
    CurriculumPhase,
    SelfPlayEvalCallback,
)
from rl_connect4.training.checkpoints import CheckpointManager
from rl_connect4.training.league import LeagueConfig
from rl_connect4.training.opponent_pool import OpponentMix, OpponentPool


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(pool: OpponentPool, env_cfg: dict, rank: int = 0):
    def _factory():
        config = Connect4Config(
            symmetry_augmentation=bool(env_cfg.get("symmetry_augmentation", False))
        )
        env = PettingZooConnect4GymEnv(
            opponent_sampler=pool.sample,
            config=config,
        )
        env.reset(seed=1000 + rank)
        return ActionMasker(env, lambda e: e.action_masks())

    return _factory


def parse_curriculum(
    self_play_cfg: dict, total_timesteps: int
) -> list[CurriculumPhase]:
    phases = self_play_cfg.get("phases", [])
    parsed: list[CurriculumPhase] = []
    for phase in phases:
        mix_cfg = phase["mix"]
        if "start_percent" in phase:
            start_percent = float(phase["start_percent"])
            if not 0.0 <= start_percent <= 100.0:
                raise ValueError("phase.start_percent must be in [0, 100]")
            start_timestep = int(total_timesteps * (start_percent / 100.0))
        elif "start_timestep" in phase:
            start_timestep = int(phase["start_timestep"])
        else:
            raise KeyError(
                "Each phase must define either start_percent or start_timestep"
            )
        parsed.append(
            CurriculumPhase(
                start_timestep=start_timestep,
                mix=OpponentMix(
                    current=float(mix_cfg["current"]),
                    historical=float(mix_cfg["historical"]),
                    random=float(mix_cfg["random"]),
                    mcts=float(mix_cfg.get("mcts", 0.0)),
                    rule_based=float(mix_cfg.get("rule_based", 0.0)),
                ),
                mcts_simulations=int(phase["mcts_simulations"]),
            )
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_self_play.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg["run_name"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    self_play_cfg = cfg["self_play"]
    env_cfg = cfg.get("env", {})
    league_cfg = cfg.get("league", {})
    n_envs = int(train_cfg.get("num_envs", 1))

    opponent_pool = OpponentPool(
        OpponentMix(
            current=float(self_play_cfg["mix"]["current"]),
            historical=float(self_play_cfg["mix"]["historical"]),
            random=float(self_play_cfg["mix"]["random"]),
            mcts=float(self_play_cfg["mix"].get("mcts", 0.0)),
            rule_based=float(self_play_cfg["mix"].get("rule_based", 0.0)),
        ),
        mcts_simulations=int(
            self_play_cfg.get("mcts_simulations", eval_cfg["mcts_simulations"])
        ),
    )

    env = DummyVecEnv([make_env(opponent_pool, env_cfg, rank=i) for i in range(n_envs)])

    policy_kwargs = {
        "features_extractor_class": Connect4CNNExtractor,
        "features_extractor_kwargs": {
            "features_dim": int(train_cfg["features_dim"]),
            "channels": int(train_cfg.get("cnn_channels", 64)),
            "num_res_blocks": int(train_cfg.get("cnn_res_blocks", 6)),
        },
        "net_arch": {
            "pi": [256, 128],
            "vf": [256, 128],
        },
    }
    model = MaskablePPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=float(train_cfg["learning_rate"]),
        n_steps=int(train_cfg["n_steps"]),
        batch_size=int(train_cfg["batch_size"]),
        gamma=float(train_cfg["gamma"]),
        policy_kwargs=policy_kwargs,
        tensorboard_log="runs",
        verbose=1,
    )
    opponent_pool.set_current_model(model)

    checkpoint_manager = CheckpointManager(
        root_dir=Path("checkpoints") / run_name,
        max_checkpoints=int(self_play_cfg["max_checkpoints"]),
    )

    callback = SelfPlayEvalCallback(
        opponent_pool=opponent_pool,
        checkpoint_manager=checkpoint_manager,
        eval_freq=int(eval_cfg["eval_freq"]),
        checkpoint_freq=int(self_play_cfg["checkpoint_freq"]),
        n_eval_episodes=int(eval_cfg["n_eval_episodes"]),
        mcts_simulations=int(eval_cfg["mcts_simulations"]),
        curriculum=parse_curriculum(
            self_play_cfg, total_timesteps=int(train_cfg["total_timesteps"])
        ),
        league_config=LeagueConfig(
            n_games_per_pair=int(league_cfg.get("n_games_per_pair", 5)),
            max_policies=int(league_cfg.get("max_policies", 10)),
            initial_elo=float(league_cfg.get("initial_elo", 1000.0)),
            k_factor=float(league_cfg.get("k_factor", 24.0)),
        ),
    )

    model.learn(
        total_timesteps=int(train_cfg["total_timesteps"]),
        callback=callback,
        tb_log_name=run_name,
    )
    model.save(Path("checkpoints") / run_name / "final_model.zip")


if __name__ == "__main__":
    main()
