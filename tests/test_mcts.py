import numpy as np

from rl_connect4.mcts.mcts import mcts_action


def _board_to_obs(board: np.ndarray, to_play: int = 1) -> np.ndarray:
    if to_play == 1:
        cur = (board == 1).astype(np.float32)
        opp = (board == -1).astype(np.float32)
    else:
        cur = (board == -1).astype(np.float32)
        opp = (board == 1).astype(np.float32)
    return np.stack([cur, opp], axis=0)


def _all_legal_mask(board: np.ndarray) -> np.ndarray:
    return (board[0] == 0).astype(np.int8)


def test_mcts_takes_immediate_winning_move():
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0] = 1
    board[5, 1] = 1
    board[5, 2] = 1
    obs = _board_to_obs(board, to_play=1)
    mask = _all_legal_mask(board)

    action = mcts_action(obs, mask, simulations=20)
    assert action == 3


def test_mcts_blocks_opponent_immediate_win():
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0] = -1
    board[5, 1] = -1
    board[5, 2] = -1
    obs = _board_to_obs(board, to_play=1)
    mask = _all_legal_mask(board)

    action = mcts_action(obs, mask, simulations=20)
    assert action == 3
