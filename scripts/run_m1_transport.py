#!/usr/bin/env python3
"""Run M1: 2D two-agent transport with independent vs HALyPO updates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coevolve.envs.two_agent_transport import run_transport_training, save_transport_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--conflict-strength", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/m1_transport"))
    parser.add_argument("--show", action="store_true", help="Open the saved plot with matplotlib after running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    histories = {}
    for method in ("independent", "halypo"):
        summary, history = run_transport_training(
            method=method,
            updates=args.updates,
            lr=args.lr,
            sigma=args.sigma,
            conflict_strength=args.conflict_strength,
        )
        summaries.append(summary.to_dict())
        histories[method] = history

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    history_path = args.output_dir / "history.csv"
    fieldnames = [
        "method",
        "update",
        "team_return",
        "final_distance",
        "mismatch",
        "final_x",
        "final_y",
        "v",
        "field_alignment",
        "update_alignment",
        "constraint_violation",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, history in histories.items():
            for row in history:
                writer.writerow({name: method if name == "method" else row[name] for name in fieldnames})

    plot_path = args.output_dir / "m1_transport_curves.png"
    save_transport_plot(histories, plot_path)

    print(json.dumps(summaries, indent=2))
    print(f"[M1] wrote {metrics_path}")
    print(f"[M1] wrote {history_path}")
    print(f"[M1] wrote {plot_path}")

    if args.show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        plt.imshow(mpimg.imread(plot_path))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
