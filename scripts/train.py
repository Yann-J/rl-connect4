from __future__ import annotations

import argparse
import re
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


def parse_numeric_literal(value: int | float | str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"Expected int, float, or str; got {type(value).__name__}")

    raw = value.strip()
    match = re.fullmatch(
        (
            r"([+-]?(?:\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?|\.\d+(?:_\d+)*)"
            r"(?:[eE][+-]?\d+(?:_\d+)*)?)([kKmMbB]?)"
        ),
        raw,
    )
    if not match:
        raise ValueError(f"Invalid numeric literal: {value!r}")

    number_part = match.group(1).replace("_", "")
    suffix = match.group(2).lower()
    multiplier = {"": 1.0, "k": 1e3, "m": 1e6, "b": 1e9}[suffix]
    return float(number_part) * multiplier


def cfg_float(value: int | float | str) -> float:
    return float(parse_numeric_literal(value))


def cfg_int(value: int | float | str) -> int:
    parsed = parse_numeric_literal(value)
    if not float(parsed).is_integer():
        raise ValueError(f"Expected integer-compatible value, got {value!r}")
    return int(parsed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(pool: OpponentPool, env_cfg: dict, rank: int = 0):
    def _factory():
        config = Connect4Config(
            symmetry_augmentation=bool(env_cfg.get("symmetry_augmentation", False)),
            randomize_train_agent=bool(env_cfg.get("randomize_train_agent", False)),
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
            start_timestep = cfg_int(phase["start_timestep"])
        else:
            raise KeyError(
                "Each phase must define either start_percent or start_timestep"
            )
        parsed.append(
            CurriculumPhase(
                start_timestep=start_timestep,
                mix=OpponentMix(
                    current=cfg_float(mix_cfg["current"]),
                    historical=cfg_float(mix_cfg["historical"]),
                    random=cfg_float(mix_cfg["random"]),
                    mcts=cfg_float(mix_cfg.get("mcts", 0.0)),
                    rule_based=cfg_float(mix_cfg.get("rule_based", 0.0)),
                ),
                mcts_simulations=cfg_int(phase["mcts_simulations"]),
            )
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg["run_name"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    self_play_cfg = cfg["self_play"]
    env_cfg = cfg.get("env", {})
    league_cfg = cfg.get("league", {})
    n_envs = cfg_int(train_cfg.get("num_envs", 1))

    opponent_pool = OpponentPool(
        OpponentMix(
            current=cfg_float(self_play_cfg["mix"]["current"]),
            historical=cfg_float(self_play_cfg["mix"]["historical"]),
            random=cfg_float(self_play_cfg["mix"]["random"]),
            mcts=cfg_float(self_play_cfg["mix"].get("mcts", 0.0)),
            rule_based=cfg_float(self_play_cfg["mix"].get("rule_based", 0.0)),
        ),
        mcts_simulations=cfg_int(
            self_play_cfg.get("mcts_simulations", eval_cfg["mcts_simulations"])
        ),
    )

    env = DummyVecEnv([make_env(opponent_pool, env_cfg, rank=i) for i in range(n_envs)])

    policy_kwargs = {
        "features_extractor_class": Connect4CNNExtractor,
        "features_extractor_kwargs": {
            "features_dim": cfg_int(train_cfg["features_dim"]),
            "channels": cfg_int(train_cfg.get("cnn_channels", 64)),
            "num_res_blocks": cfg_int(train_cfg.get("cnn_res_blocks", 6)),
        },
        "net_arch": {
            "pi": [256, 128],
            "vf": [256, 128],
        },
    }
    model = MaskablePPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=cfg_float(train_cfg["learning_rate"]),
        n_steps=cfg_int(train_cfg["n_steps"]),
        batch_size=cfg_int(train_cfg["batch_size"]),
        gamma=cfg_float(train_cfg["gamma"]),
        policy_kwargs=policy_kwargs,
        tensorboard_log="runs",
        verbose=1,
    )
    opponent_pool.set_current_model(model)

    checkpoint_manager = CheckpointManager(
        root_dir=Path("checkpoints") / run_name,
        max_checkpoints=cfg_int(self_play_cfg["max_checkpoints"]),
    )

    callback = SelfPlayEvalCallback(
        opponent_pool=opponent_pool,
        checkpoint_manager=checkpoint_manager,
        eval_freq=cfg_int(eval_cfg["eval_freq"]),
        checkpoint_freq=cfg_int(self_play_cfg["checkpoint_freq"]),
        n_eval_episodes=cfg_int(eval_cfg["n_eval_episodes"]),
        mcts_simulations=cfg_int(eval_cfg["mcts_simulations"]),
        curriculum=parse_curriculum(
            self_play_cfg, total_timesteps=cfg_int(train_cfg["total_timesteps"])
        ),
        league_config=LeagueConfig(
            n_games_per_pair=cfg_int(league_cfg.get("n_games_per_pair", 5)),
            max_policies=cfg_int(league_cfg.get("max_policies", 10)),
            initial_elo=cfg_float(league_cfg.get("initial_elo", 1000.0)),
            k_factor=cfg_float(league_cfg.get("k_factor", 24.0)),
        ),
    )

    model.learn(
        total_timesteps=cfg_int(train_cfg["total_timesteps"]),
        callback=callback,
        tb_log_name=run_name,
    )
    model.save(Path("checkpoints") / run_name / "final_model.zip")


if __name__ == "__main__":
    main()
