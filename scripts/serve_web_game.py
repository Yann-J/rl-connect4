from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from pettingzoo.classic import connect_four_v3
from sb3_contrib import MaskablePPO


def _transpose_obs(raw_obs: dict[str, Any]) -> np.ndarray:
    board = raw_obs["observation"]  # (6, 7, 2)
    return np.transpose(board, (2, 0, 1)).astype(np.float32)


def _action_mask(raw_obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(raw_obs["action_mask"], dtype=np.int8)


def _board_for_player_zero(env) -> list[list[int]]:
    raw_obs = env.observe("player_0")
    p0 = np.asarray(raw_obs["observation"][:, :, 0], dtype=np.int8)
    p1 = np.asarray(raw_obs["observation"][:, :, 1], dtype=np.int8)
    board = p0 + (2 * p1)
    return board.astype(int).tolist()


@dataclass
class GameState:
    env: Any
    model: Any
    human_agent: str
    ai_agent: str

    @classmethod
    def create(cls, model: Any, human_starts: bool) -> "GameState":
        env = connect_four_v3.env(render_mode=None)
        env.reset()
        human_agent = "player_0" if human_starts else "player_1"
        ai_agent = "player_1" if human_starts else "player_0"
        state = cls(
            env=env,
            model=model,
            human_agent=human_agent,
            ai_agent=ai_agent,
        )
        state._play_ai_until_human_turn()
        return state

    def close(self) -> None:
        self.env.close()

    def _is_done(self) -> bool:
        return (
            any(self.env.terminations.values())
            or any(self.env.truncations.values())
        )

    def _winner(self) -> str | None:
        rewards = self.env.rewards
        if rewards.get(self.human_agent, 0.0) > 0:
            return "human"
        if rewards.get(self.ai_agent, 0.0) > 0:
            return "ai"
        if self._is_done():
            return "draw"
        return None

    def status_payload(self, error: str | None = None) -> dict[str, Any]:
        current = self.env.agent_selection if not self._is_done() else None
        return {
            "board": _board_for_player_zero(self.env),
            "done": self._is_done(),
            "winner": self._winner(),
            "human_agent": self.human_agent,
            "ai_agent": self.ai_agent,
            "current_turn": current,
            "error": error,
        }

    def _pick_ai_action(self) -> int:
        raw_obs = self.env.observe(self.ai_agent)
        mask = _action_mask(raw_obs)
        obs = _transpose_obs(raw_obs)
        action, _ = self.model.predict(
            obs,
            deterministic=True,
            action_masks=mask.reshape(1, -1),
        )
        picked = int(np.asarray(action).item())
        if mask[picked] == 0:
            legal_actions = np.flatnonzero(mask)
            return int(legal_actions[0])
        return picked

    def _play_ai_until_human_turn(self) -> None:
        while not self._is_done() and self.env.agent_selection == self.ai_agent:
            self.env.step(self._pick_ai_action())

    def human_move(self, action: int) -> dict[str, Any]:
        if self._is_done():
            return self.status_payload(
                error="Game already finished. Start a new game."
            )

        if self.env.agent_selection != self.human_agent:
            return self.status_payload(error="It is not your turn.")

        raw_obs = self.env.observe(self.human_agent)
        mask = _action_mask(raw_obs)
        if action < 0 or action >= 7 or mask[action] == 0:
            return self.status_payload(error="Illegal move.")

        self.env.step(action)
        self._play_ai_until_human_turn()
        return self.status_payload()


class Connect4RequestHandler(BaseHTTPRequestHandler):
    state: GameState | None = None
    static_dir: Path | None = None
    model: Any = None

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found.")
            return
        raw = path.read_bytes()
        if path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        else:
            content_type = "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(self.static_dir / "index.html")
            return
        if self.path == "/api/state":
            if self.state is None:
                self.state = GameState.create(model=self.model, human_starts=True)
            self._send_json(self.state.status_payload())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown route.")

    def do_POST(self) -> None:
        if self.path == "/api/new-game":
            body = self._read_json_body()
            human_starts = bool(body.get("human_starts", True))
            if self.state is not None:
                self.state.close()
            self.state = GameState.create(
                model=self.model,
                human_starts=human_starts,
            )
            self._send_json(self.state.status_payload())
            return

        if self.path == "/api/move":
            if self.state is None:
                self.state = GameState.create(model=self.model, human_starts=True)
            body = self._read_json_body()
            action = int(body.get("column", -1))
            self._send_json(self.state.human_move(action))
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown route.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a static Connect-4 UI against a trained policy."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a MaskablePPO checkpoint (.zip).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    # Import registers the class path used in checkpoints.
    from rl_connect4.policies.cnn_policy import Connect4CNNExtractor

    model = MaskablePPO.load(
        str(model_path),
        custom_objects={"features_extractor_class": Connect4CNNExtractor},
    )
    static_dir = Path(__file__).resolve().parents[1] / "web"
    if not static_dir.exists():
        raise FileNotFoundError(f"Static directory not found: {static_dir}")

    handler_cls = Connect4RequestHandler
    handler_cls.static_dir = static_dir
    handler_cls.model = model
    handler_cls.state = None

    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Serving Connect-4 web game at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if handler_cls.state is not None:
            handler_cls.state.close()
        server.server_close()


if __name__ == "__main__":
    main()
