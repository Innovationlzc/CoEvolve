#!/usr/bin/env python3
"""Prepare or audit the frozen CoEvolve backbone benchmark matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.cap_policy.formal_eval import (  # noqa: E402
    FORMAL_BACKBONES,
    FORMAL_EVAL_PROTOCOL_ID,
    FORMAL_MODEL_ATTEMPT_BUDGET,
    FORMAL_PHYSICAL_BUDGET,
    FORMAL_RUN_INDICES,
    FORMAL_SEEDS,
    formal_memory_paths,
    formal_run_output_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create empty run directories and run_plan.json files. No physics is run.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if a planned run directory contains mixed or stale formal artifacts.",
    )
    args = parser.parse_args()

    plans = planned_runs(REPO_ROOT)
    failures: list[str] = []
    for plan in plans:
        output_dir = Path(plan["output_dir"])
        status, reason = inspect_run_directory(output_dir, expected_plan=plan)
        if args.create:
            status, reason = prepare_run_directory(output_dir, plan)
        if args.check and status not in {"absent", "prepared_empty"}:
            failures.append(f"{output_dir}: {reason}")
        print(
            f"[{plan['model_key']}] run={plan['run_index']} seed={plan['seed']:03d} "
            f"status={status} path={output_dir}"
        )

    print(
        f"matrix={len(FORMAL_BACKBONES)} models x {len(FORMAL_RUN_INDICES)} runs "
        f"= {len(plans)}; physical_budget={FORMAL_PHYSICAL_BUDGET}/run; "
        f"protocol={FORMAL_EVAL_PROTOCOL_ID}"
    )
    if failures:
        raise SystemExit("formal matrix check failed:\n" + "\n".join(failures))


def planned_runs(repo_root: Path) -> list[dict[str, object]]:
    plans: list[dict[str, object]] = []
    for backbone in FORMAL_BACKBONES:
        for run_index in FORMAL_RUN_INDICES:
            output_dir = formal_run_output_dir(
                repo_root,
                model_or_key=backbone.key,
                run_index=run_index,
            )
            plans.append(
                {
                    "kind": "coevolve_formal_run_plan",
                    "protocol_id": FORMAL_EVAL_PROTOCOL_ID,
                    "benchmark": "backbone",
                    "model": backbone.requested_model,
                    "model_key": backbone.key,
                    "provider": backbone.provider,
                    "planner_backend": backbone.planner_backend,
                    "run_index": run_index,
                    "seed": FORMAL_SEEDS[run_index],
                    "physical_budget": FORMAL_PHYSICAL_BUDGET,
                    "model_attempt_budget": FORMAL_MODEL_ATTEMPT_BUDGET,
                    "initial_memory_condition": "empty_dynamic_and_action_memory",
                    "output_dir": str(output_dir),
                    "memory_paths": {
                        key: str(path)
                        for key, path in formal_memory_paths(output_dir).items()
                    },
                    "backbone_spec": asdict(backbone),
                }
            )
    return plans


def inspect_run_directory(
    output_dir: Path,
    *,
    expected_plan: dict[str, object] | None = None,
) -> tuple[str, str]:
    if not output_dir.exists():
        return "absent", "ready to prepare"
    allowed = {"run_plan.json"}
    entries = {item.name for item in output_dir.iterdir()}
    unexpected = sorted(entries - allowed)
    if unexpected:
        return "not_fresh", f"contains formal runtime artifacts: {unexpected[:8]}"
    plan_path = output_dir / "run_plan.json"
    if not plan_path.exists():
        return "prepared_empty", "empty directory"
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid_plan", f"cannot read run_plan.json: {exc}"
    if payload.get("protocol_id") != FORMAL_EVAL_PROTOCOL_ID:
        return "invalid_plan", "run_plan.json uses another protocol"
    for key in (
        "model",
        "run_index",
        "seed",
        "physical_budget",
        "model_attempt_budget",
        "initial_memory_condition",
        "output_dir",
        "memory_paths",
    ):
        if expected_plan is not None and payload.get(key) != expected_plan.get(key):
            return "invalid_plan", f"run_plan.json has stale or mismatched {key}"
    return "prepared_empty", "run plan exists; no experiment artifacts"


def prepare_run_directory(
    output_dir: Path,
    plan: dict[str, object],
) -> tuple[str, str]:
    """Create one immutable plan without touching a non-fresh run directory."""

    status, reason = inspect_run_directory(output_dir, expected_plan=plan)
    if status not in {"absent", "prepared_empty"}:
        return status, reason
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inspect_run_directory(output_dir, expected_plan=plan)


if __name__ == "__main__":
    main()
