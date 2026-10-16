#!/usr/bin/env python3
"""Rebuild same-action executable exemplars from accepted rollout evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.cap_policy.action_memory import VerifiedActionMemoryStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("RAW_JSONL", "ARTIFACT_ROOT"),
        default=[],
    )
    parser.add_argument(
        "--exclude",
        nargs=2,
        action="append",
        default=[],
        metavar=("RAW_JSONL", "CHECKPOINT_AFTER"),
        help="Exclude a known-invalid accepted record without deleting raw evidence.",
    )
    parser.add_argument(
        "--revoke-existing-checkpoint",
        action="append",
        default=[],
        help=(
            "Remove executable examples created by or descending from this checkpoint "
            "without rebuilding the rest of the store."
        ),
    )
    parser.add_argument(
        "--revoke-source-artifact",
        action="append",
        default=[],
        help=(
            "Remove an executable example from one exact retired rollout artifact "
            "directory. Unlike checkpoint ids, this is safe across experiment runs."
        ),
    )
    args = parser.parse_args()

    store = VerifiedActionMemoryStore(args.output)
    if args.source:
        counts = store.rebuild_from_raw(
            ((Path(raw), Path(root)) for raw, root in args.source),
            excluded_evidence=(
                (Path(raw), checkpoint) for raw, checkpoint in args.exclude
            ),
        )
    elif args.revoke_existing_checkpoint or args.revoke_source_artifact:
        counts = {
            "removed_by_checkpoint": store.revoke_checkpoints(
                args.revoke_existing_checkpoint
            ),
            "removed_by_source": store.revoke_source_artifacts(
                args.revoke_source_artifact
            ),
            "retained": len(store.exemplars),
        }
    else:
        parser.error(
            "provide --source, --revoke-existing-checkpoint, or "
            "--revoke-source-artifact"
        )
    print(
        "[verified-action-memory] "
        + " ".join(f"{key}={value}" for key, value in counts.items())
        + f" output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
