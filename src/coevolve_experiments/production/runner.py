#!/usr/bin/env python3
"""Fail-closed entrypoint for the future post-Table-1 Praxis runner.

The subprocess bridge invokes this file with exactly one configuration token.
It currently performs the complete immutable preflight and exits before any
physical or model dispatch because the audited V83 runtime adapter has not yet
been migrated.  Returning success here would incorrectly turn a smoke harness
into paper evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

# The live bridge executes this repository-relative file directly rather than
# with ``python -m``.  In that mode Python puts this directory, not the
# repository's ``src`` directory, on ``sys.path``.  Bootstrap only that fixed
# parent so the same byte-pinned entrypoint works before the package is
# installed; no caller-controlled path is accepted.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coevolve_experiments.serialization import to_primitive
from coevolve_experiments.production.preflight import (
    ProductionPreflightError,
    preflight_production_launch,
)


EXIT_CONFIG_INVALID = 64
EXIT_RUNTIME_UNAVAILABLE = 78


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="coevolve-post-table-production-runner",
        allow_abbrev=False,
        description="Consume one immutable post-Table-1 launch configuration.",
    )
    parser.add_argument("--launch-config", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.launch_config.is_absolute():
        parser.error("--launch-config must be an absolute path")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, bundle = preflight_production_launch(args.launch_config)
    except ProductionPreflightError as exc:
        print(
            json.dumps(
                {
                    "kind": "coevolve_production_preflight_error",
                    "schema_version": 1,
                    "launch_config_path": str(args.launch_config),
                    "error": str(exc),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return EXIT_CONFIG_INVALID

    print(
        json.dumps(
            {
                "kind": "coevolve_production_preflight_report",
                "schema_version": 1,
                "report": to_primitive(report),
                "injection_channels": [item.name for item in bundle.channels],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    if not report.available:
        return EXIT_RUNTIME_UNAVAILABLE
    # No implementation may reach this branch until a reviewed adapter both
    # consumes every bundle channel and emits the v3 evidence contract.
    return EXIT_RUNTIME_UNAVAILABLE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_CONFIG_INVALID",
    "EXIT_RUNTIME_UNAVAILABLE",
    "main",
    "parse_args",
]
