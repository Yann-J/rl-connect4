from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch


def _obs_to_board(obs: np.ndarray) -> np.ndarray:
    # obs shape: (2, 6, 7), channel 0 = current player, channel 1 = opponent.
    return obs[0].astype(np.int8) - obs[1].astype(np.int8)


def _board_to_obs(board: np.ndarray, player: int) -> np.ndarray:
    own = (board == player).astype(np.float32)
    opp = (board == -player).astype(np.float32)
    return np.stack([own, opp], axis=0)


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
                return int(token)
            if r + 3 < rows and all(
                board[r + i, c] == token for i in range(4)
            ):
                return int(token)
            if r + 3 < rows and c + 3 < cols and all(
                board[r + i, c + i] == token for i in range(4)
            ):
                return int(token)
            if r + 3 < rows and c - 3 >= 0 and all(
                board[r + i, c - i] == token for i in range(4)
            ):
                return int(token)
    return 0


def _is_terminal(board: np.ndarray) -> tuple[bool, float]:
    winner = _winner(board)
    if winner != 0:
        return True, float(winner)
    if _legal_actions(board).size == 0:
        return True, 0.0
    return False, 0.0


def _center_bias(action: int) -> float:
    return 3.0 - abs(action - 3)


class PuctModel(Protocol):
    policy: object


def _policy_prior_and_value(
    model: PuctModel,
    board: np.ndarray,
    action_mask: np.ndarray,
    player: int,
) -> tuple[np.ndarray, float]:
    obs = _board_to_obs(board, player)
    obs_tensor = torch.as_tensor(obs[None], dtype=torch.float32)
    mask_tensor = torch.as_tensor(
        action_mask[None].astype(bool), dtype=torch.bool
    )

    with torch.no_grad():
        distribution = model.policy.get_distribution(
            obs_tensor,
            action_masks=mask_tensor,
        )
        probs = distribution.distribution.probs.squeeze(0).cpu().numpy()
        value_tensor = model.policy.predict_values(obs_tensor)
        value = float(value_tensor.squeeze().cpu().item())

    priors = np.asarray(probs, dtype=np.float64)
    priors *= action_mask.astype(np.float64)
    total = float(priors.sum())
    if total <= 0:
        legal = np.flatnonzero(action_mask)
        if legal.size == 0:
            return np.zeros_like(priors), value
        priors = np.zeros_like(priors)
        priors[legal] = 1.0 / float(legal.size)
        return priors, value
    return priors / total, value


@dataclass
class Node:
    board: np.ndarray
    to_play: int
    parent: "Node | None" = None
    action: int | None = None
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


def _expand_node(model: PuctModel, node: Node) -> float:
    legal = _legal_actions(node.board)
    if legal.size == 0:
        node.expanded = True
        return 0.0
    mask = np.zeros(7, dtype=np.int8)
    mask[legal] = 1
    priors, value = _policy_prior_and_value(
        model,
        node.board,
        mask,
        player=node.to_play,
    )

    for action in legal:
        a = int(action)
        if a in node.children:
            continue
        child_board = _drop(node.board, a, node.to_play)
        node.children[a] = Node(
            board=child_board,
            to_play=-node.to_play,
            parent=node,
            action=a,
            prior=float(priors[a]),
        )
    node.expanded = True
    return value


def _select_child(node: Node, c_puct: float) -> Node:
    sqrt_parent_visits = math.sqrt(max(1, node.visits))
    best_score = -float("inf")
    best_child: Node | None = None
    for child in node.children.values():
        q = -child.mean_value
        u = c_puct * child.prior * sqrt_parent_visits / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    if best_child is None:
        raise RuntimeError("Cannot select child from empty node")
    return best_child


def _backprop(node: Node, value: float) -> None:
    cur: Node | None = node
    cur_value = float(value)
    while cur is not None:
        cur.visits += 1
        cur.value_sum += cur_value
        cur_value = -cur_value
        cur = cur.parent


def puct_action(
    model: PuctModel,
    obs: np.ndarray,
    action_mask: np.ndarray,
    simulations: int = 128,
    c_puct: float = 1.5,
) -> int:
    legal = np.flatnonzero(action_mask)
    if legal.size == 0:
        return 0
    if legal.size == 1:
        return int(legal[0])

    root = Node(board=_obs_to_board(obs), to_play=1)
    _expand_node(model, root)

    for _ in range(simulations):
        node = root
        while node.expanded and node.children:
            node = _select_child(node, c_puct=c_puct)

        terminal, terminal_value = _is_terminal(node.board)
        if terminal:
            if terminal_value == 0.0:
                value = 0.0
            else:
                value = 1.0 if terminal_value == float(node.to_play) else -1.0
        else:
            value = _expand_node(model, node)

        _backprop(node, value)

    best_child = max(
        root.children.values(),
        key=lambda child: (
            child.visits,
            child.mean_value,
            _center_bias(int(child.action if child.action is not None else 3)),
        ),
    )
    return int(
        best_child.action if best_child.action is not None else legal[0]
    )


def make_puct_policy(
    model: PuctModel,
    simulations: int = 128,
    c_puct: float = 1.5,
):
    def _policy(obs: np.ndarray, action_mask: np.ndarray) -> int:
        return puct_action(
            model=model,
            obs=obs,
            action_mask=action_mask,
            simulations=simulations,
            c_puct=c_puct,
        )

    return _policy
