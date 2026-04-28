from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rl_connect4.training.league import (  # noqa: E402
    LeagueConfig,
    run_checkpoint_league,
)


def _extract_step(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    return (_extract_step(path), path.stat().st_mtime, path.name)


def _collect_checkpoints(
    checkpoint_paths: list[str],
    checkpoint_dirs: list[str],
) -> list[Path]:
    collected: list[Path] = []

    for raw_path in checkpoint_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {raw_path}")
        collected.append(path)

    for raw_dir in checkpoint_dirs:
        directory = Path(raw_dir).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Checkpoint directory not found: {raw_dir}"
            )
        for pattern in ("*.zip", "*.onnx"):
            collected.extend(
                sorted(directory.glob(pattern), key=_checkpoint_sort_key)
            )

    # Keep insertion order while deduplicating.
    deduped = list(dict.fromkeys(collected))
    return sorted(deduped, key=_checkpoint_sort_key)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an Elo league between candidate checkpoints and rank them."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "Path to a model checkpoint (.zip or .onnx). "
            "Repeat for multiple candidates."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=[],
        help=(
            "Directory containing checkpoint .zip/.onnx files. "
            "All files are added as candidates."
        ),
    )
    parser.add_argument(
        "--max-policies",
        type=int,
        default=None,
        help=(
            "Keep only the latest N checkpoints after sorting "
            "(default: keep all provided candidates)."
        ),
    )
    parser.add_argument(
        "--n-games-per-pair",
        type=int,
        default=10,
        help="Number of games played for each checkpoint pair.",
    )
    parser.add_argument(
        "--initial-elo",
        type=float,
        default=1000.0,
        help="Initial Elo assigned to each candidate.",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=24.0,
        help="Elo K-factor.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to save the final ranking as JSON.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    candidates = _collect_checkpoints(
        checkpoint_paths=args.checkpoint,
        checkpoint_dirs=args.checkpoint_dir,
    )
    if args.max_policies is not None:
        if args.max_policies <= 1:
            raise ValueError("--max-policies must be >= 2")
        candidates = candidates[-args.max_policies:]
    if len(candidates) < 2:
        raise ValueError(
            "Need at least 2 candidate checkpoints. "
            "Use --checkpoint and/or --checkpoint-dir."
        )

    config = LeagueConfig(
        n_games_per_pair=args.n_games_per_pair,
        max_policies=len(candidates),
        initial_elo=args.initial_elo,
        k_factor=args.k_factor,
    )
    elos = run_checkpoint_league(candidates, config)
    ranking = sorted(elos.items(), key=lambda item: item[1], reverse=True)

    print(f"League completed with {len(candidates)} candidates.")
    print(f"Games per pair: {args.n_games_per_pair}")
    print("")
    print(f"{'#':>2}  {'ELO':>8}  {'STEP':>8}  CHECKPOINT")
    for rank, (path, elo) in enumerate(ranking, start=1):
        step = _extract_step(path)
        step_label = str(step) if step >= 0 else "-"
        print(f"{rank:>2}  {elo:>8.1f}  {step_label:>8}  {path}")

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "rank": idx,
                "elo": score,
                "step": _extract_step(path),
                "checkpoint": str(path),
            }
            for idx, (path, score) in enumerate(ranking, start=1)
        ]
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("")
        print(f"Saved ranking JSON to {output_path}")


if __name__ == "__main__":
    main()
