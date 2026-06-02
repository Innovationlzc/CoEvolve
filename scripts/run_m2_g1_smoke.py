#!/usr/bin/env python3
"""Launch an upstream Isaac Lab G1 smoke/eval scene from CoEvolve."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


ISAACLAB_ROOT = Path("/home/admin1/Project/IsaacLab")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-Play-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--viz", default="newton", choices=["newton", "none"])
    parser.add_argument("--timeout", type=int, default=0, help="Optional timeout in seconds; 0 means no timeout.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without launching Isaac Lab.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        str(ISAACLAB_ROOT / "isaaclab.sh"),
        "-p",
        "scripts/environments/random_agent.py",
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
    ]
    if args.viz != "none":
        cmd.extend(["--viz", args.viz])

    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    print(" ".join(cmd))
    if args.dry_run:
        return

    run_cmd = cmd
    if args.timeout > 0:
        run_cmd = ["timeout", str(args.timeout), *cmd]
    subprocess.run(run_cmd, cwd=ISAACLAB_ROOT, env=env, check=False)


if __name__ == "__main__":
    main()
