"""Resolve and bootstrap one Table-4 target from an external suite plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from coevolve_experiments.contracts import MemoryCondition, RunSpec, TableId
from coevolve_experiments.scenes import load_scene, validate_scene
from coevolve_experiments.serialization import to_primitive
from coevolve_experiments.suite_plan import (
    SUITE_PLAN_KIND,
    SUITE_PLAN_SCHEMA_VERSION,
    SuitePlanError,
    load_verified_suite_plan,
)

from .bootstrap import TransferMemoryBootstrap, bootstrap_transfer_memory
from .validation import validate_strategy_bank


class FormalTransferPreparationError(ValueError):
    """The external plan or selected transfer artifacts are not formally pinned."""


@dataclass(frozen=True)
class ResolvedFormalRun:
    run_spec: RunSpec
    plan_path: Path
    plan_sha256: str


@dataclass(frozen=True)
class FormalTransferPreparation:
    run_spec: RunSpec
    plan_path: Path
    plan_sha256: str
    run_dir: Path
    memory_dir: Path
    bootstrap: TransferMemoryBootstrap


def resolve_formal_run(
    plan_path: str | Path,
    run_id: str,
) -> ResolvedFormalRun:
    """Rebuild the suite and resolve exactly one unmodified planned RunSpec."""

    try:
        verified = load_verified_suite_plan(plan_path)
    except (OSError, TypeError, ValueError, SuitePlanError) as exc:
        raise FormalTransferPreparationError(str(exc)) from exc
    if verified.execution_preflight != "passed":
        raise FormalTransferPreparationError(
            "formal prepare-transfer requires execution_preflight='passed'"
        )
    try:
        selected = verified.resolve_run(run_id)
    except SuitePlanError as exc:
        raise FormalTransferPreparationError(str(exc)) from exc
    _rerun_selected_execution_preflight(
        selected,
        verified.protocol_lock,
        list(verified.execution_reports),
    )
    return ResolvedFormalRun(
        run_spec=selected,
        plan_path=verified.path,
        plan_sha256=verified.sha256,
    )


def prepare_transfer_from_plan(
    *,
    plan_path: str | Path,
    run_id: str,
    results_root: str | Path,
    bank_path: str | Path,
    provenance_path: str | Path,
) -> FormalTransferPreparation:
    """Prepare one formal target without free-form condition/model/hash flags."""

    resolved = resolve_formal_run(plan_path, run_id)
    run = resolved.run_spec
    if run.table_id is not TableId.TABLE_4:
        raise FormalTransferPreparationError("prepare-transfer accepts only Table-4 RunSpecs")
    if run.memory_condition not in {
        MemoryCondition.GPT_TO_GPT,
        MemoryCondition.GEMINI_TO_GPT,
    }:
        raise FormalTransferPreparationError(
            "prepare-transfer accepts only new strategy-transfer conditions"
        )
    if run.strategy_bank is None:
        raise FormalTransferPreparationError("Table-4 RunSpec has no strategy-bank pin")
    root = Path(results_root)
    if not root.is_absolute():
        raise FormalTransferPreparationError("formal results root must be absolute")
    run_dir = (root / run.output_relpath).resolve()
    try:
        run_dir.relative_to(root.resolve())
    except ValueError as exc:
        raise FormalTransferPreparationError("planned output path escapes results root") from exc
    memory_dir = run_dir / "memory"

    validation = validate_strategy_bank(Path(bank_path))
    if validation.bank_sha256 != run.strategy_bank.sha256:
        raise FormalTransferPreparationError(
            "strategy bank does not match the hash pinned by the selected RunSpec"
        )
    condition = {
        MemoryCondition.GPT_TO_GPT: "GPT→GPT",
        MemoryCondition.GEMINI_TO_GPT: "Gemini→GPT",
    }[run.memory_condition]
    formal_binding = _formal_binding(run, resolved)
    bootstrap = bootstrap_transfer_memory(
        memory_dir,
        condition,
        seed_bank=Path(bank_path),
        seed_provenance=Path(provenance_path),
        source_model=run.strategy_bank.source_backbone,
        target_model=run.target_backbone,
        formal_binding=formal_binding,
        expected_strategy_bank_pin=run.strategy_bank,
    )
    return FormalTransferPreparation(
        run_spec=run,
        plan_path=resolved.plan_path,
        plan_sha256=resolved.plan_sha256,
        run_dir=run_dir,
        memory_dir=memory_dir,
        bootstrap=bootstrap,
    )


def _formal_binding(
    run: RunSpec,
    resolved: ResolvedFormalRun,
) -> dict[str, str]:
    # RunSpec identity pins are included explicitly; this resolver never
    # reconstructs a stale hand-written subset.
    binding = {
        "run_id": run.run_id,
        "protocol_id": run.protocol_id,
        "condition_fingerprint": run.condition_fingerprint,
        "strategy_bank_sha256": run.strategy_bank.sha256,
        "source_model": run.strategy_bank.source_backbone,
        "source_provider_route_id": run.strategy_bank.source_provider_route_id,
        "source_provider_route_fingerprint": (
            run.strategy_bank.source_provider_route_fingerprint
        ),
        "target_model": run.target_backbone,
        "provider_route_id": run.provider_route_id,
        "provider_route_fingerprint": run.provider_route_fingerprint,
        "plan_path": str(resolved.plan_path),
        "plan_sha256": resolved.plan_sha256,
    }
    for name in (
        "observation_contract_fingerprint",
        "task_instruction_fingerprint",
        "shared_prompt_contract_fingerprint",
        "decision_prompt_fingerprint",
        "skill_api_fingerprint",
        "full_method_compatibility_fingerprint",
    ):
        if hasattr(run, name):
            binding[name] = str(getattr(run, name))
    return binding


def _rerun_selected_execution_preflight(
    run: RunSpec,
    protocol_lock: object,
    planned_reports: object,
) -> None:
    """Re-hash current selected-scene evidence; never trust a plan status flag."""

    if not isinstance(planned_reports, list):
        raise FormalTransferPreparationError("suite plan execution_reports must be an array")
    matching = [
        item
        for item in planned_reports
        if isinstance(item, Mapping) and item.get("scene_id") == run.scene_id.value
    ]
    if len(matching) != 1:
        raise FormalTransferPreparationError(
            "suite plan must contain exactly one execution report for the selected scene"
        )
    try:
        current = validate_scene(
            load_scene(run.scene_id),
            mode="execution",
            protocol_lock=protocol_lock,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise FormalTransferPreparationError(
            f"cannot rerun selected-scene execution preflight: {exc}"
        ) from exc
    current_value = to_primitive(current)
    if current_value != matching[0]:
        raise FormalTransferPreparationError(
            "selected-scene execution evidence changed after suite planning"
        )
    if not current.execution_ready:
        reasons = current.errors or (
            "execution report lacks a complete host-derived evidence fingerprint",
        )
        raise FormalTransferPreparationError(
            "selected-scene execution preflight is not ready: "
            + "; ".join(map(str, reasons))
        )


__all__ = [
    "FormalTransferPreparation",
    "FormalTransferPreparationError",
    "ResolvedFormalRun",
    "SUITE_PLAN_KIND",
    "SUITE_PLAN_SCHEMA_VERSION",
    "prepare_transfer_from_plan",
    "resolve_formal_run",
]
