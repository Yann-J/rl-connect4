from __future__ import annotations

import argparse
import json
from pathlib import Path

from tbparse import SummaryReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build static TensorBoard report HTML."
    )
    parser.add_argument(
        "--log-root", default="runs", help="TensorBoard log root directory."
    )
    parser.add_argument(
        "--report-dir",
        default="web/tensorboard",
        help="Output directory for the static report.",
    )
    parser.add_argument(
        "--template",
        default="scripts/tensorboard/template.html",
        help="HTML template containing __RUNS_DATA__ placeholder.",
    )
    return parser.parse_args()


def collect_runs_data(
    log_root: Path,
) -> dict[str, dict[str, list[dict[str, float]]]]:
    runs_data: dict[str, dict[str, list[dict[str, float]]]] = {}

    if not log_root.exists() or not any(log_root.rglob("events.out.tfevents.*")):
        return runs_data

    scalars = SummaryReader(str(log_root), pivot=False).scalars
    if scalars.empty:
        return runs_data

    run_column = next(
        (
            column
            for column in ("dir_name", "dir", "run", "run_name", "logdir", "path")
            if column in scalars.columns
        ),
        None,
    )

    grouped_runs = (
        scalars.groupby(run_column)
        if run_column is not None
        else [("default", scalars)]
    )

    for run_name, run_df in grouped_runs:
        tags: dict[str, list[dict[str, float]]] = {}
        for tag_name, tag_df in run_df.groupby("tag"):
            rows = tag_df.sort_values("step")[["step", "value"]].to_dict(
                orient="records"
            )
            tags[str(tag_name)] = rows
        runs_data[str(run_name)] = tags

    return runs_data


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root)
    report_dir = Path(args.report_dir)
    template_path = Path(args.template)

    report_dir.mkdir(parents=True, exist_ok=True)
    template = template_path.read_text(encoding="utf-8")
    payload = json.dumps(collect_runs_data(log_root))
    html = template.replace("__RUNS_DATA__", payload)

    output_path = report_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Static TensorBoard report generated at {output_path}")


if __name__ == "__main__":
    main()
