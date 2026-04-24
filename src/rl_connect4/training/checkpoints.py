from __future__ import annotations

from collections import deque
from pathlib import Path


class CheckpointManager:
    def __init__(self, root_dir: str | Path, max_checkpoints: int = 10) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self._paths: deque[Path] = deque(maxlen=max_checkpoints)

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    def save(self, model, step: int) -> Path:
        path = self.root_dir / f"model_{step}.zip"
        model.save(path)
        self._paths.append(path)
        return path

