from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_connect4.envs.pettingzoo_connect4 import random_legal_policy
from rl_connect4.eval.evaluate import evaluate_vs_opponent
from rl_connect4.mcts.mcts import make_mcts_policy
from rl_connect4.policies.rule_based import rule_based_policy
from rl_connect4.training.checkpoints import CheckpointManager
from rl_connect4.training.league import LeagueConfig, run_checkpoint_league
from rl_connect4.training.opponent_pool import OpponentMix, OpponentPool


@dataclass
class CurriculumPhase:
    start_timestep: int
    mix: OpponentMix
    mcts_simulations: int


class SelfPlayEvalCallback(BaseCallback):
    def __init__(
        self,
        opponent_pool: OpponentPool,
        checkpoint_manager: CheckpointManager,
        eval_freq: int = 5_000,
        checkpoint_freq: int = 10_000,
        n_eval_episodes: int = 50,
        mcts_simulations: int = 100,
        curriculum: list[CurriculumPhase] | None = None,
        league_config: LeagueConfig | None = None,
        rolling_window: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.opponent_pool = opponent_pool
        self.checkpoint_manager = checkpoint_manager
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.n_eval_episodes = n_eval_episodes
        self.mcts_simulations = mcts_simulations
        self.curriculum = sorted(
            curriculum or [],
            key=lambda phase: phase.start_timestep,
        )
        self.league_config = league_config or LeagueConfig()
        self._applied_phase_idx = -1
        self.rolling_rewards = deque(maxlen=rolling_window)

    def _on_training_start(self) -> None:
        self.opponent_pool.set_current_model(self.model)
        self._apply_curriculum_phase(force=True)

    def _apply_curriculum_phase(self, force: bool = False) -> None:
        if not self.curriculum:
            return
        active_idx = 0
        for idx, phase in enumerate(self.curriculum):
            if self.num_timesteps >= phase.start_timestep:
                active_idx = idx
        if force or active_idx != self._applied_phase_idx:
            phase = self.curriculum[active_idx]
            self.opponent_pool.set_mix(phase.mix)
            self.opponent_pool.set_mcts_simulations(phase.mcts_simulations)
            self._applied_phase_idx = active_idx
            self.logger.record("train/curriculum_phase", float(active_idx))
            self.logger.record(
                "train/curriculum_mcts_simulations",
                float(phase.mcts_simulations),
            )

    def _on_step(self) -> bool:
        self._apply_curriculum_phase()
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is not None and dones is not None:
            for reward, done in zip(
                np.array(rewards).reshape(-1),
                np.array(dones).reshape(-1),
            ):
                if bool(done):
                    self.rolling_rewards.append(float(reward))

        if self.n_calls % self.checkpoint_freq == 0:
            new_checkpoint = self.checkpoint_manager.save(
                self.model,
                self.num_timesteps,
            )
            self.opponent_pool.refresh_historical(
                self.checkpoint_manager.paths
            )
            self.opponent_pool.set_current_model(self.model)
            self.opponent_pool.set_elite_checkpoint(new_checkpoint)

            league_elos = run_checkpoint_league(
                self.checkpoint_manager.paths,
                self.league_config,
            )
            if league_elos:
                best_path, best_elo = max(
                    league_elos.items(),
                    key=lambda item: item[1],
                )
                self.opponent_pool.set_elite_checkpoint(best_path)
                self.logger.record("league/best_elo", float(best_elo))
                self.logger.record(
                    "league/mean_elo",
                    float(np.mean(list(league_elos.values()))),
                )
                self.logger.record(
                    "league/selected_checkpoint_step",
                    float(best_path.stem.split("_")[-1]),
                )

        if self.n_calls % self.eval_freq == 0:
            random_metrics = evaluate_vs_opponent(
                self.model,
                opponent_policy=random_legal_policy,
                n_episodes=self.n_eval_episodes,
            )
            mcts_metrics = evaluate_vs_opponent(
                self.model,
                opponent_policy=make_mcts_policy(self.mcts_simulations),
                n_episodes=self.n_eval_episodes,
            )
            rule_based_metrics = evaluate_vs_opponent(
                self.model,
                opponent_policy=rule_based_policy,
                n_episodes=self.n_eval_episodes,
            )
            self.logger.record(
                "eval/random_win_rate",
                random_metrics["win_rate"],
            )
            self.logger.record("eval/mcts_win_rate", mcts_metrics["win_rate"])
            self.logger.record(
                "eval/rule_based_win_rate",
                rule_based_metrics["win_rate"],
            )
            self.logger.record(
                "eval/random_mean_reward",
                random_metrics["mean_reward"],
            )
            self.logger.record(
                "eval/mcts_mean_reward",
                mcts_metrics["mean_reward"],
            )
            self.logger.record(
                "eval/rule_based_mean_reward",
                rule_based_metrics["mean_reward"],
            )

            if self.rolling_rewards:
                win_rate = float(np.mean(np.array(self.rolling_rewards) > 0))
                self.logger.record("train/rolling_win_rate", win_rate)
                self.logger.record(
                    "train/rolling_reward",
                    float(np.mean(self.rolling_rewards)),
                )

        return True
