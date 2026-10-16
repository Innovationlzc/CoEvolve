#!/usr/bin/env python3
"""Promote raw rollout evidence into compact sync/async strategy cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.cap_policy.strategy_memory import (  # noqa: E402
    StrategyMemoryStore,
    load_raw_attempts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-attempt-memory", type=Path, required=True)
    parser.add_argument("--strategy-memory", type=Path, required=True)
    parser.add_argument(
        "--default-phase",
        default="corridor_transport",
        help="Semantic phase assigned only when an older raw record lacks phase context.",
    )
    parser.add_argument(
        "--revoke-checkpoint",
        action="append",
        default=[],
        help="Exclude a known-invalid accepted checkpoint while preserving its raw record.",
    )
    parser.add_argument(
        "--seed-memory",
        type=Path,
        action="append",
        default=[],
        help=(
            "Merge an already validated, manually screened strategy-memory file "
            "after rebuilding verifier evidence."
        ),
    )
    args = parser.parse_args()

    records = load_raw_attempts(args.raw_attempt_memory)
    store = StrategyMemoryStore(args.strategy_memory)
    counts = store.rebuild_from_raw(
        records,
        default_phase=args.default_phase,
        revoked_checkpoint_ids=args.revoke_checkpoint,
    )
    seeded_ids: list[str] = []
    if args.seed_memory:
        rebuilt_specs = list(store.specs)
        merged_specs: list[dict] = []
        seen_ids: set[str] = set()
        for seed_path in args.seed_memory:
            for spec in StrategyMemoryStore(seed_path).specs:
                spec_id = str(spec.get("id", ""))
                if spec_id in seen_ids:
                    continue
                merged_specs.append(spec)
                seen_ids.add(spec_id)
                seeded_ids.append(spec_id)
        for spec in rebuilt_specs:
            spec_id = str(spec.get("id", ""))
            if spec_id in seen_ids:
                continue
            merged_specs.append(spec)
            seen_ids.add(spec_id)
        store.data["specs"] = merged_specs
        store.data.setdefault("migration", {})["seed_memories"] = [
            str(path) for path in args.seed_memory
        ]
        store.data["migration"]["seeded_spec_ids"] = seeded_ids
        store.save()
    print(
        json.dumps(
            {
                "raw_attempt_memory": str(args.raw_attempt_memory),
                "strategy_memory": str(args.strategy_memory),
                "spec_count": len(store.specs),
                "seeded_spec_ids": seeded_ids,
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
