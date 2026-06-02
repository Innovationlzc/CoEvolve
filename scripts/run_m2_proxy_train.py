#!/usr/bin/env python3
"""Run M2 proxy training: independent vs HALyPO on the Isaac 4D box-push action."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coevolve.envs.m2_box_push_proxy import run_m2_proxy_training, save_m2_proxy_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1.0e-10)
    parser.add_argument("--conflict-strength", type=float, default=0.4)
    parser.add_argument("--velocity-scale", type=float, default=0.47)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("results/m2_proxy_training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    histories = {}
    policies = {}
    for method in ("independent", "halypo"):
        summary, history, theta, action = run_m2_proxy_training(
            method=method,
            updates=args.updates,
            lr=args.lr,
            sigma=args.sigma,
            eps=args.eps,
            conflict_strength=args.conflict_strength,
            velocity_scale=args.velocity_scale,
            horizon=args.horizon,
        )
        summaries.append(summary.to_dict())
        histories[method] = history
        policies[method] = {
            "method": method,
            "theta": theta,
            "action": action,
            "action_space": "M2 proxy [agent_a_x, agent_a_y, agent_b_x, agent_b_y]",
            "velocity_scale": args.velocity_scale,
        }
        (args.output_dir / f"{method}_policy.json").write_text(
            json.dumps(policies[method], indent=2), encoding="utf-8"
        )

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
        "action_a_x",
        "action_a_y",
        "action_b_x",
        "action_b_y",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, history in histories.items():
            for row in history:
                writer.writerow({name: method if name == "method" else row[name] for name in fieldnames})

    plot_path = args.output_dir / "m2_proxy_curves.png"
    save_m2_proxy_plot(histories, plot_path)

    print(json.dumps(summaries, indent=2))
    print(f"[M2-PROXY] wrote {metrics_path}")
    print(f"[M2-PROXY] wrote {history_path}")
    print(f"[M2-PROXY] wrote {plot_path}")
    print(f"[M2-PROXY] wrote {args.output_dir / 'halypo_policy.json'}")


if __name__ == "__main__":
    main()
