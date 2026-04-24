from __future__ import annotations

import numpy as np


def _board_from_obs(obs: np.ndarray) -> np.ndarray:
    # obs: (2, 6, 7), channel 0 current player, channel 1 opponent
    return obs[0].astype(np.int8) - obs[1].astype(np.int8)


def _drop(board: np.ndarray, action: int, player: int) -> np.ndarray | None:
    out = board.copy()
    for row in range(5, -1, -1):
        if out[row, action] == 0:
            out[row, action] = player
            return out
    return None


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
                return int(token)
            if r + 3 < rows and all(
                board[r + i, c] == token for i in range(4)
            ):
                return int(token)
            if r + 3 < rows and c + 3 < cols:
                if all(board[r + i, c + i] == token for i in range(4)):
                    return int(token)
            if r + 3 < rows and c - 3 >= 0:
                if all(board[r + i, c - i] == token for i in range(4)):
                    return int(token)
    return 0


def _best_extend_action(board: np.ndarray, legal_actions: np.ndarray) -> int:
    center = 3
    best_action = int(legal_actions[0])
    best_score = -1_000
    for action in legal_actions:
        placed = _drop(board, int(action), player=1)
        if placed is None:
            continue
        own_tokens = int(np.count_nonzero(placed == 1))
        opp_tokens = int(np.count_nonzero(placed == -1))
        # Simple extension heuristic: encourage own presence and center control.
        score = own_tokens - opp_tokens - abs(center - int(action))
        if score > best_score:
            best_score = score
            best_action = int(action)
    return best_action


def rule_based_policy(obs: np.ndarray, action_mask: np.ndarray) -> int:
    board = _board_from_obs(obs)
    legal_actions = np.flatnonzero(action_mask)
    if legal_actions.size == 1:
        return int(legal_actions[0])

    # 1) Win immediately if possible.
    for action in legal_actions:
        placed = _drop(board, int(action), player=1)
        if placed is not None and _winner(placed) == 1:
            return int(action)

    # 2) Block opponent immediate winning move.
    for action in legal_actions:
        placed = _drop(board, int(action), player=-1)
        if placed is not None and _winner(placed) == -1:
            return int(action)

    # 3) Otherwise extend own structure, biased to center.
    return _best_extend_action(board, legal_actions)
