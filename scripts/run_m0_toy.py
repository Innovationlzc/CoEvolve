#!/usr/bin/env python3
"""Run M0: HALyPO projection on a toy two-agent differentiable game."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coevolve.toys.two_agent_game import run_toy_game, save_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--rotation", type=float, default=1.25)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/m0_toy"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    histories = {}
    for method in ("independent", "halypo"):
        summary, history = run_toy_game(
            method=method,
            steps=args.steps,
            lr=args.lr,
            rotation=args.rotation,
            sigma=args.sigma,
        )
        summaries.append(summary.to_dict())
        histories[method] = history

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    history_path = args.output_dir / "history.csv"
    fieldnames = ["method", "step", "x", "y", "v", "field_alignment", "update_alignment", "constraint_violation"]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, history in histories.items():
            for row in history:
                writer.writerow({name: method if name == "method" else row[name] for name in fieldnames})

    plot_path = args.output_dir / "m0_toy_curves.png"
    save_plot(histories, plot_path)

    print(json.dumps(summaries, indent=2))
    print(f"[M0] wrote {metrics_path}")
    print(f"[M0] wrote {history_path}")
    print(f"[M0] wrote {plot_path}")


if __name__ == "__main__":
    main()
