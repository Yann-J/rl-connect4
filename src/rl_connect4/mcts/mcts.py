from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _obs_to_board(obs: np.ndarray) -> np.ndarray:
    # obs shape: (2, 6, 7), channel 0 = current player, channel 1 = opponent.
    return obs[0].astype(np.int8) - obs[1].astype(np.int8)


def _legal_actions(board: np.ndarray) -> np.ndarray:
    return np.flatnonzero(board[0] == 0)


def _drop(board: np.ndarray, action: int, player: int) -> np.ndarray:
    out = board.copy()
    for row in range(5, -1, -1):
        if out[row, action] == 0:
            out[row, action] = player
            return out
    return out


def _winner(board: np.ndarray) -> int:
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            token = board[r, c]
            if token == 0:
                continue
            if c + 3 < cols and all(
                board[r, c + i] == token for i in range(4)
            ):
                return token
            if r + 3 < rows and all(
                board[r + i, c] == token for i in range(4)
            ):
                return token
            if r + 3 < rows and c + 3 < cols and all(
                board[r + i, c + i] == token for i in range(4)
            ):
                return token
            if r + 3 < rows and c - 3 >= 0 and all(
                board[r + i, c - i] == token for i in range(4)
            ):
                return token
    return 0


def _is_terminal(board: np.ndarray) -> tuple[bool, float]:
    w = _winner(board)
    if w != 0:
        return True, float(w)
    if _legal_actions(board).size == 0:
        return True, 0.0
    return False, 0.0


def _center_bias(action: int) -> float:
    # Favor center columns, usually stronger in Connect-4.
    return 3.0 - abs(action - 3)


def _score_windows(board: np.ndarray, player: int) -> float:
    score = 0.0
    rows, cols = board.shape
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                r_end = r + 3 * dr
                c_end = c + 3 * dc
                if not (0 <= r_end < rows and 0 <= c_end < cols):
                    continue
                window = np.array(
                    [board[r + i * dr, c + i * dc] for i in range(4)]
                )
                own = int(np.count_nonzero(window == player))
                opp = int(np.count_nonzero(window == -player))
                empty = int(np.count_nonzero(window == 0))
                if own > 0 and opp > 0:
                    continue
                if own == 3 and empty == 1:
                    score += 8.0
                elif own == 2 and empty == 2:
                    score += 2.0
                elif own == 1 and empty == 3:
                    score += 0.25
                if opp == 3 and empty == 1:
                    score -= 9.0
                elif opp == 2 and empty == 2:
                    score -= 2.5
    return score


def _evaluate(board: np.ndarray, to_play: int) -> float:
    winner = _winner(board)
    if winner != 0:
        return 1.0 if winner == to_play else -1.0
    if _legal_actions(board).size == 0:
        return 0.0

    center = np.array(board[:, 3])
    center_own = int(np.count_nonzero(center == to_play))
    center_opp = int(np.count_nonzero(center == -to_play))
    center_score = 0.2 * (center_own - center_opp)
    pattern_score = _score_windows(board, to_play) / 30.0
    value = center_score + pattern_score
    return float(np.clip(value, -0.99, 0.99))


def _best_tactical_action(
    board: np.ndarray, legal_actions: np.ndarray
) -> int | None:
    for action in legal_actions:
        a = int(action)
        if _winner(_drop(board, a, 1)) == 1:
            return a
    for action in legal_actions:
        a = int(action)
        if _winner(_drop(board, a, -1)) == -1:
            return a
    return None


@dataclass
class Node:
    board: np.ndarray
    player: int
    parent: "Node | None" = None
    action: int | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def ucb(self, c: float = 1.4) -> float:
        if self.visits == 0 or self.parent is None:
            return float("inf")
        return self.value + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def expand(self) -> None:
        for action in _legal_actions(self.board):
            a = int(action)
            if a not in self.children:
                self.children[a] = Node(
                    board=_drop(self.board, a, self.player),
                    player=-self.player,
                    parent=self,
                    action=a,
                )


def mcts_action(
    obs: np.ndarray, action_mask: np.ndarray, simulations: int = 100
) -> int:
    board = _obs_to_board(obs)
    root = Node(board=board, player=1)
    legal = np.flatnonzero(action_mask)
    if legal.size == 1:
        return int(legal[0])
    tactical = _best_tactical_action(board, legal)
    if tactical is not None:
        return tactical

    root.expand()
    for _ in range(simulations):
        node = root
        while node.children:
            node = max(node.children.values(), key=lambda n: n.ucb())
        terminal, value = _is_terminal(node.board)
        if not terminal:
            node.expand()
            if node.children:
                node = np.random.choice(list(node.children.values()))
            value = _evaluate(node.board, node.player)
        # Backprop: value is from player_1 perspective.
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value * cur.player
            cur = cur.parent

    best = max(
        root.children.values(),
        key=lambda n: (
            n.visits,
            n.value,
            _center_bias(int(n.action if n.action is not None else 3)),
        ),
    )
    return int(best.action)


def make_mcts_policy(simulations: int = 100):
    def _policy(obs: np.ndarray, action_mask: np.ndarray) -> int:
        return mcts_action(obs, action_mask, simulations=simulations)

    return _policy

