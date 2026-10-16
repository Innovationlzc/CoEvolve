"""Fail-closed batch preparation and serial execution for Tables 2--4.

The single-run bridge remains the authority for every physical launch.  This
module only adds a deterministic, plan-derived queue around that bridge:

* every new ``RunSpec`` in one preregistered table partition appears exactly
  once (or every new run for the ``full`` partition);
* every Table-2/3 run receives a plan-bound empty-memory bootstrap;
* Table-4 bootstrap memory is prepared from the two lock-pinned banks;
* launch configurations are staged and published as one immutable batch; and
* completed cells are re-attested before they are skipped on resume.

There is intentionally no ``continue_on_error`` switch.  An incomplete or
infrastructure-invalid cell must be inspected and archived under the formal
attempt policy before the canonical output path can be used again.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Optional, Tuple

from ..contracts import MemoryCondition, RunSpec, TableId
from ..memory.formal_prepare import prepare_transfer_from_plan
from ..serialization import sha256_json, to_primitive
from ..suite_plan import load_verified_suite_plan
from .bridge import (
    LIVE_BRIDGE_RECEIPT_FILENAME,
    live_run_reservation_path,
    launch_live_subprocess,
    load_subprocess_launch_config,
    validate_live_bridge_receipt,
    write_subprocess_launch_config,
)
from .capability import load_runner_capability
from .contracts import LiveRunReceipt
from .launch import (
    FROM_SCRATCH_BOOTSTRAP_MISSING_CODE,
    CompiledLiveLaunch,
    compile_live_launch,
)
from .preflight import LIVE_BRIDGE_UNAVAILABLE_CODE


LIVE_BATCH_MANIFEST_KIND = "coevolve_praxis_live_batch"
LIVE_BATCH_MANIFEST_SCHEMA_VERSION = 5
LIVE_BATCH_MANIFEST_FILENAME = "batch_manifest.json"

LIVE_BATCH_PARTITIONS = ("full", "table2", "table3", "table4")
_PARTITION_TABLE: Mapping[str, TableId | None] = {
    "full": None,
    "table2": TableId.TABLE_2,
    "table3": TableId.TABLE_3,
    "table4": TableId.TABLE_4,
}

_TRANSFER_BOOTSTRAP_MISSING = "TRANSFER_MEMORY_BOOTSTRAP_NOT_PREPARED"
_BATCH_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "plan_path",
        "plan_sha256",
        "results_root",
        "capability_manifest_path",
        "capability_manifest_sha256",
        "base_method_commit",
        "runner_integration_commit",
        "full_method_compatibility_fingerprint",
        "partition",
        "run_count",
        "cells",
        "batch_fingerprint",
    }
)
_CELL_FIELDS = frozenset(
    {
        "ordinal",
        "run_id",
        "output_dir",
        "launch_config_path",
        "launch_config_sha256",
        "config_fingerprint",
        "launch_fingerprint",
    }
)


class LiveBatchError(ValueError):
    """Raised before dispatch when a batch is incomplete or has drifted."""


@dataclass(frozen=True, slots=True)
class TransferArtifactPaths:
    bank_path: Path
    provenance_path: Path


@dataclass(frozen=True, slots=True)
class LiveBatchCell:
    ordinal: int
    run_id: str
    output_dir: Path
    launch_config_path: Path
    launch_config_sha256: str
    config_fingerprint: str
    launch_fingerprint: str


@dataclass(frozen=True, slots=True)
class VerifiedLiveBatch:
    manifest_path: Path
    manifest_sha256: str
    batch_fingerprint: str
    plan_path: Path
    plan_sha256: str
    results_root: Path
    capability_manifest_path: Path
    capability_manifest_sha256: str
    base_method_commit: str
    runner_integration_commit: str
    full_method_compatibility_fingerprint: str
    partition: str
    cells: Tuple[LiveBatchCell, ...]


def prepare_live_batch(
    *,
    plan_path: str | Path,
    results_root: str | Path,
    capability_manifest_path: str | Path,
    batch_root: str | Path,
    gpt_strategy_bank_path: str | Path | None = None,
    gpt_strategy_provenance_path: str | Path | None = None,
    gemini_strategy_bank_path: str | Path | None = None,
    gemini_strategy_provenance_path: str | Path | None = None,
    partition: str = "full",
) -> VerifiedLiveBatch:
    """Prepare one fixed paper-table partition without starting execution.

    Partitions are closed over the registry: callers may select Table 2,
    Table 3, Table 4, or the complete suite, but never arbitrary cells or
    seeds.  This permits independent tables to run as soon as their declared
    inputs are available without enabling favorable-result selection.  The
    two transfer banks are required only for a partition containing Table 4.
    """

    plan = _load_execution_plan(plan_path)
    selected_runs = _runs_for_partition(plan.suite.new_runs, partition)
    root = _absolute_normalized_path(results_root, "formal results root")
    destination = _absolute_normalized_path(batch_root, "live batch root")
    if destination.exists():
        raise LiveBatchError("live batch root already exists; refusing to overwrite it")
    if destination.parent.exists() and destination.parent.is_symlink():
        raise LiveBatchError("live batch parent must not be a symlink")

    commits = {run.base_method_commit for run in selected_runs}
    if len(commits) != 1:
        raise LiveBatchError("suite contains more than one base method commit")
    compatibility_fingerprints = {
        run.full_method_compatibility_fingerprint
        for run in selected_runs
    }
    if len(compatibility_fingerprints) != 1:
        raise LiveBatchError(
            "suite contains more than one Full-method compatibility identity"
        )
    capability_path = _absolute_existing_file(
        capability_manifest_path, "runner capability manifest"
    )
    try:
        capability = load_runner_capability(
            capability_path,
            expected_base_method_commit=next(iter(commits)),
            expected_full_method_compatibility_fingerprint=next(
                iter(compatibility_fingerprints)
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise LiveBatchError(f"runner capability is not valid: {exc}") from exc

    transfer_paths = _transfer_paths_for_runs(
        selected_runs,
        gpt_strategy_bank_path=gpt_strategy_bank_path,
        gpt_strategy_provenance_path=gpt_strategy_provenance_path,
        gemini_strategy_bank_path=gemini_strategy_bank_path,
        gemini_strategy_provenance_path=gemini_strategy_provenance_path,
    )

    # Validate every scene, output binding, bank, and provenance before the
    # first bootstrap directory is created.  Missing bootstrap snapshots are
    # the sole expected pre-preparation blocker.
    preliminary = tuple(
        _compile_for_run(
            plan.path,
            run,
            results_root=root,
            transfer_paths=transfer_paths,
        )
        for run in selected_runs
    )
    for launch in preliminary:
        allowed = {
            LIVE_BRIDGE_UNAVAILABLE_CODE,
            FROM_SCRATCH_BOOTSTRAP_MISSING_CODE,
        }
        if launch.run_spec.strategy_bank is not None:
            allowed.add(_TRANSFER_BOOTSTRAP_MISSING)
        unexpected = tuple(item for item in launch.blockers if item not in allowed)
        if unexpected:
            raise LiveBatchError(
                f"{launch.run_spec.run_id} is not batch-ready: "
                + ", ".join(unexpected)
            )

    # Both initializers are fresh-write-only.  Existing valid bootstraps were
    # recognized by the preliminary compile and are left byte-for-byte
    # unchanged, making an interrupted preparation safely resumable without
    # allowing a previous run's memory to become a new repetition's input.
    for launch in preliminary:
        run = launch.run_spec
        if (
            run.strategy_bank is not None
            and _TRANSFER_BOOTSTRAP_MISSING in launch.blockers
        ):
            artifacts = transfer_paths[run.memory_condition]
            try:
                prepare_transfer_from_plan(
                    plan_path=plan.path,
                    run_id=run.run_id,
                    results_root=root,
                    bank_path=artifacts.bank_path,
                    provenance_path=artifacts.provenance_path,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise LiveBatchError(
                    f"cannot prepare transfer memory for {run.run_id}: {exc}"
                ) from exc
        elif (
            run.strategy_bank is None
            and FROM_SCRATCH_BOOTSTRAP_MISSING_CODE in launch.blockers
        ):
            try:
                _prepare_from_scratch_for_batch(
                    run,
                    launch.output_dir / "memory",
                    plan_path=plan.path,
                    plan_sha256=plan.sha256,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise LiveBatchError(
                    f"cannot prepare from-scratch memory for {run.run_id}: {exc}"
                ) from exc

    compiled = tuple(
        _compile_for_run(
            plan.path,
            run,
            results_root=root,
            transfer_paths=transfer_paths,
        )
        for run in selected_runs
    )
    for launch in compiled:
        unexpected = tuple(
            item
            for item in launch.blockers
            if item != LIVE_BRIDGE_UNAVAILABLE_CODE
        )
        if unexpected:
            raise LiveBatchError(
                f"{launch.run_spec.run_id} is not launchable after preparation: "
                + ", ".join(unexpected)
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=str(destination.parent),
        )
    )
    published = False
    try:
        config_dir = staging / "configs"
        config_dir.mkdir()
        cells = []
        for ordinal, launch in enumerate(compiled):
            filename = f"{ordinal:03d}__{launch.run_spec.run_id}.json"
            staged_config = (config_dir / filename).resolve()
            verified = write_subprocess_launch_config(
                launch,
                capability_manifest_path=capability_path,
                destination=staged_config,
            )
            cells.append(
                LiveBatchCell(
                    ordinal=ordinal,
                    run_id=launch.run_spec.run_id,
                    output_dir=launch.output_dir,
                    launch_config_path=(
                        destination / "configs" / filename
                    ).resolve(strict=False),
                    launch_config_sha256=verified.sha256,
                    config_fingerprint=verified.config_fingerprint,
                    launch_fingerprint=verified.launch.launch_fingerprint,
                )
            )

        fingerprint_payload = {
            "kind": LIVE_BATCH_MANIFEST_KIND,
            "schema_version": LIVE_BATCH_MANIFEST_SCHEMA_VERSION,
            "plan_path": str(plan.path),
            "plan_sha256": plan.sha256,
            "results_root": str(root),
            "capability_manifest_path": str(capability.manifest_path),
            "capability_manifest_sha256": capability.manifest_sha256,
            "base_method_commit": capability.base_method_commit,
            "runner_integration_commit": capability.runner_integration_commit,
            "full_method_compatibility_fingerprint": (
                capability.full_method_compatibility.fingerprint
            ),
            "partition": partition,
            "run_count": len(cells),
            "cells": [to_primitive(cell) for cell in cells],
        }
        document = {
            **fingerprint_payload,
            "batch_fingerprint": sha256_json(fingerprint_payload),
        }
        _write_exclusive_json(staging / LIVE_BATCH_MANIFEST_FILENAME, document)
        os.replace(staging, destination)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    return load_live_batch(destination / LIVE_BATCH_MANIFEST_FILENAME)


def _prepare_from_scratch_for_batch(
    run: RunSpec,
    memory_dir: Path,
    *,
    plan_path: Path,
    plan_sha256: str,
) -> object:
    # Local import avoids the memory -> methods -> live package initialization
    # cycle while keeping this boundary directly testable.
    from ..memory.from_scratch import prepare_from_scratch_memory

    return prepare_from_scratch_memory(
        run,
        memory_dir,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
    )


def load_live_batch(path: str | Path) -> VerifiedLiveBatch:
    """Rebuild a published batch from its plan and every launch config."""

    manifest_path = _absolute_existing_file(path, "live batch manifest")
    raw, payload = _read_exact_json_object(manifest_path, "live batch manifest")
    _require_exact_fields(payload, _BATCH_FIELDS, "live batch manifest")
    if payload.get("kind") != LIVE_BATCH_MANIFEST_KIND:
        raise LiveBatchError("live batch manifest has the wrong kind")
    _require_exact_integer(
        payload.get("schema_version"),
        LIVE_BATCH_MANIFEST_SCHEMA_VERSION,
        "live batch manifest schema_version",
    )
    fingerprint_payload = dict(payload)
    fingerprint = fingerprint_payload.pop("batch_fingerprint", None)
    if not isinstance(fingerprint, str) or fingerprint != sha256_json(
        fingerprint_payload
    ):
        raise LiveBatchError("live batch fingerprint differs")

    plan_path = _payload_absolute_path(payload, "plan_path", "suite plan")
    plan = _load_execution_plan(plan_path)
    if payload.get("plan_sha256") != plan.sha256:
        raise LiveBatchError("live batch plan_sha256 differs from current plan bytes")
    results_root = _payload_absolute_path(
        payload, "results_root", "formal results root"
    )
    capability_path = _payload_absolute_path(
        payload,
        "capability_manifest_path",
        "runner capability manifest",
        must_exist=True,
    )
    commits = {run.base_method_commit for run in plan.suite.new_runs}
    if len(commits) != 1:
        raise LiveBatchError("suite contains more than one base method commit")
    compatibility_fingerprints = {
        run.full_method_compatibility_fingerprint
        for run in plan.suite.new_runs
    }
    if len(compatibility_fingerprints) != 1:
        raise LiveBatchError(
            "suite contains more than one Full-method compatibility identity"
        )
    try:
        capability = load_runner_capability(
            capability_path,
            expected_base_method_commit=next(iter(commits)),
            expected_full_method_compatibility_fingerprint=next(
                iter(compatibility_fingerprints)
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise LiveBatchError(f"runner capability is not valid: {exc}") from exc
    if payload.get("capability_manifest_sha256") != capability.manifest_sha256:
        raise LiveBatchError("live batch capability hash differs from current bytes")
    if payload.get("base_method_commit") != capability.base_method_commit:
        raise LiveBatchError(
            "live batch base method commit differs from current capability"
        )
    if (
        payload.get("runner_integration_commit")
        != capability.runner_integration_commit
    ):
        raise LiveBatchError(
            "live batch runner integration commit differs from current capability"
        )
    if (
        payload.get("full_method_compatibility_fingerprint")
        != capability.full_method_compatibility.fingerprint
    ):
        raise LiveBatchError(
            "live batch Full-method identity differs from current capability"
        )

    values = payload.get("cells")
    if not isinstance(values, list):
        raise LiveBatchError("live batch cells must be an array")
    if payload.get("run_count") != len(values):
        raise LiveBatchError("live batch run_count differs from its cells")
    partition = _validated_partition(payload.get("partition"))
    runs = _runs_for_partition(plan.suite.new_runs, partition)
    if len(values) != len(runs):
        raise LiveBatchError(
            "live batch must contain every run in its registered partition exactly once"
        )

    cells = []
    for ordinal, (value, run) in enumerate(zip(values, runs)):
        if not isinstance(value, Mapping):
            raise LiveBatchError(f"live batch cell {ordinal} must be an object")
        _require_exact_fields(value, _CELL_FIELDS, f"live batch cell {ordinal}")
        if value.get("ordinal") != ordinal:
            raise LiveBatchError(f"live batch cell {ordinal} has the wrong ordinal")
        if value.get("run_id") != run.run_id:
            raise LiveBatchError(
                f"live batch cell {ordinal} differs from plan run order"
            )
        config_path = _payload_absolute_path(
            value,
            "launch_config_path",
            f"live batch cell {ordinal} launch config",
            must_exist=True,
        )
        expected_config_parent = (manifest_path.parent / "configs").resolve()
        if config_path.parent != expected_config_parent:
            raise LiveBatchError(
                f"live batch cell {ordinal} config is outside the batch config directory"
            )
        try:
            verified = load_subprocess_launch_config(
                config_path,
                output_may_exist=True,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise LiveBatchError(
                f"live batch cell {ordinal} launch config is invalid: {exc}"
            ) from exc
        output_dir = _payload_absolute_path(
            value,
            "output_dir",
            f"live batch cell {ordinal} output",
        )
        expected_output = (results_root / run.output_relpath).resolve(strict=False)
        if output_dir != expected_output or verified.launch.output_dir != expected_output:
            raise LiveBatchError(
                f"live batch cell {ordinal} output differs from its RunSpec"
            )
        expected_values = {
            "launch_config_sha256": verified.sha256,
            "config_fingerprint": verified.config_fingerprint,
            "launch_fingerprint": verified.launch.launch_fingerprint,
        }
        for key, expected in expected_values.items():
            if value.get(key) != expected:
                raise LiveBatchError(
                    f"live batch cell {ordinal} {key} differs from current config"
                )
        cells.append(
            LiveBatchCell(
                ordinal=ordinal,
                run_id=run.run_id,
                output_dir=output_dir,
                launch_config_path=config_path,
                launch_config_sha256=verified.sha256,
                config_fingerprint=verified.config_fingerprint,
                launch_fingerprint=verified.launch.launch_fingerprint,
            )
        )

    return VerifiedLiveBatch(
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        batch_fingerprint=fingerprint,
        plan_path=plan.path,
        plan_sha256=plan.sha256,
        results_root=results_root,
        capability_manifest_path=capability.manifest_path,
        capability_manifest_sha256=capability.manifest_sha256,
        base_method_commit=capability.base_method_commit,
        runner_integration_commit=capability.runner_integration_commit,
        full_method_compatibility_fingerprint=(
            capability.full_method_compatibility.fingerprint
        ),
        partition=partition,
        cells=tuple(cells),
    )


def inspect_live_batch(path: str | Path) -> dict[str, Any]:
    """Return a revalidated, non-mutating status snapshot."""

    batch = load_live_batch(path)
    rows = []
    counts = {"completed": 0, "ready": 0, "blocked": 0}
    for cell in batch.cells:
        receipt_path = cell.output_dir / LIVE_BRIDGE_RECEIPT_FILENAME
        if receipt_path.is_file():
            try:
                receipt = validate_live_bridge_receipt(
                    cell.output_dir,
                    expected_run_id=cell.run_id,
                )
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                status = "blocked"
                detail = f"invalid completion receipt: {exc}"
            else:
                status = "completed"
                detail = str(receipt.get("terminal_status", ""))
        else:
            reservation_path = live_run_reservation_path(cell.output_dir)
            if reservation_path.exists() or reservation_path.is_symlink():
                status = "blocked"
                detail = (
                    "live-run reservation exists; failed or concurrent attempt "
                    "requires operator review"
                )
            else:
                try:
                    load_subprocess_launch_config(
                        cell.launch_config_path,
                        output_may_exist=False,
                    )
                except (OSError, TypeError, ValueError, RuntimeError) as exc:
                    status = "blocked"
                    detail = str(exc)
                else:
                    status = "ready"
                    detail = ""
        counts[status] += 1
        rows.append(
            {
                "ordinal": cell.ordinal,
                "run_id": cell.run_id,
                "status": status,
                "detail": detail,
            }
        )
    return {
        "kind": "coevolve_praxis_live_batch_status",
        "batch_manifest_path": str(batch.manifest_path),
        "batch_manifest_sha256": batch.manifest_sha256,
        "batch_fingerprint": batch.batch_fingerprint,
        "partition": batch.partition,
        "run_count": len(batch.cells),
        "counts": counts,
        "cells": rows,
    }


def execute_live_batch(
    path: str | Path,
    *,
    timeout_seconds: float | None = None,
    max_runs: int | None = None,
) -> dict[str, Any]:
    """Run pending cells serially and stop at the first invalid attempt."""

    if max_runs is not None and (
        isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs <= 0
    ):
        raise LiveBatchError("max_runs must be a positive integer")
    batch = load_live_batch(path)
    completed_before = 0
    launched: list[LiveRunReceipt] = []
    for cell in batch.cells:
        receipt_path = cell.output_dir / LIVE_BRIDGE_RECEIPT_FILENAME
        if receipt_path.is_file():
            validate_live_bridge_receipt(
                cell.output_dir,
                expected_run_id=cell.run_id,
            )
            completed_before += 1
            continue
        if max_runs is not None and len(launched) >= max_runs:
            break
        try:
            receipt = launch_live_subprocess(
                cell.launch_config_path,
                timeout_seconds=timeout_seconds,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise LiveBatchError(
                f"batch stopped at {cell.run_id}; canonical output was not "
                f"overwritten and requires operator review: {exc}"
            ) from exc
        launched.append(receipt)
    return {
        "kind": "coevolve_praxis_live_batch_execution",
        "batch_manifest_path": str(batch.manifest_path),
        "batch_manifest_sha256": batch.manifest_sha256,
        "batch_fingerprint": batch.batch_fingerprint,
        "partition": batch.partition,
        "completed_before": completed_before,
        "launched_count": len(launched),
        "launched": [to_primitive(item) for item in launched],
        "status": inspect_live_batch(batch.manifest_path),
    }


def _compile_for_run(
    plan_path: Path,
    run: RunSpec,
    *,
    results_root: Path,
    transfer_paths: Mapping[MemoryCondition, TransferArtifactPaths],
) -> CompiledLiveLaunch:
    artifacts: Optional[TransferArtifactPaths] = None
    if run.strategy_bank is not None:
        artifacts = transfer_paths.get(run.memory_condition)
        if artifacts is None:
            raise LiveBatchError(
                f"no transfer artifacts are bound for {run.memory_condition.value}"
            )
    try:
        return compile_live_launch(
            plan_path,
            run.run_id,
            results_root=results_root,
            strategy_bank_path=artifacts.bank_path if artifacts else None,
            strategy_bank_provenance_path=(
                artifacts.provenance_path if artifacts else None
            ),
            require_launchable=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise LiveBatchError(f"cannot compile {run.run_id}: {exc}") from exc


def _validated_partition(value: object) -> str:
    if not isinstance(value, str) or value not in _PARTITION_TABLE:
        raise LiveBatchError(
            "live batch partition must be one of " + ", ".join(LIVE_BATCH_PARTITIONS)
        )
    return value


def _runs_for_partition(
    runs: Tuple[RunSpec, ...] | tuple[object, ...],
    partition: object,
) -> tuple:
    """Return a complete registry-defined partition; never accept run IDs."""

    normalized = _validated_partition(partition)
    table = _PARTITION_TABLE[normalized]
    selected = tuple(
        run
        for run in runs
        if table is None or getattr(run, "table_id", None) is table
    )
    if not selected:
        raise LiveBatchError(f"registered partition {normalized} contains no new runs")
    expected_counts = {
        "full": 45,
        "table2": 20,
        "table3": 15,
        "table4": 10,
    }
    # Real RunSpecs use the paper's fixed cardinalities.  Lightweight unit-test
    # fixtures may omit ``table_id`` and are accepted only for the full scope.
    if all(isinstance(run, RunSpec) for run in runs):
        expected = expected_counts[normalized]
        if len(selected) != expected:
            raise LiveBatchError(
                f"registered partition {normalized} has {len(selected)} runs; "
                f"expected {expected}"
            )
    return selected


def _transfer_paths_for_runs(
    runs: tuple,
    *,
    gpt_strategy_bank_path: str | Path | None,
    gpt_strategy_provenance_path: str | Path | None,
    gemini_strategy_bank_path: str | Path | None,
    gemini_strategy_provenance_path: str | Path | None,
) -> Mapping[MemoryCondition, TransferArtifactPaths]:
    required = {
        run.memory_condition
        for run in runs
        if getattr(run, "strategy_bank", None) is not None
    }
    supplied = {
        MemoryCondition.GPT_TO_GPT: (
            gpt_strategy_bank_path,
            gpt_strategy_provenance_path,
            "GPT",
        ),
        MemoryCondition.GEMINI_TO_GPT: (
            gemini_strategy_bank_path,
            gemini_strategy_provenance_path,
            "Gemini",
        ),
    }
    result: dict[MemoryCondition, TransferArtifactPaths] = {}
    for condition, (bank, provenance, label) in supplied.items():
        if condition not in required:
            if bank is not None or provenance is not None:
                raise LiveBatchError(
                    f"{label} transfer artifacts are not permitted for this partition"
                )
            continue
        if bank is None or provenance is None:
            raise LiveBatchError(
                f"{label} strategy bank and provenance are required for this partition"
            )
        result[condition] = TransferArtifactPaths(
            bank_path=_absolute_existing_file(bank, f"{label} strategy bank"),
            provenance_path=_absolute_existing_file(
                provenance, f"{label} strategy provenance"
            ),
        )
    unknown = required.difference(supplied)
    if unknown:
        raise LiveBatchError(
            "partition uses unsupported transfer conditions: "
            + ", ".join(sorted(item.value for item in unknown))
        )
    return result


def _load_execution_plan(path: str | Path):
    try:
        return load_verified_suite_plan(path, require_execution_ready=True)
    except (OSError, TypeError, ValueError) as exc:
        raise LiveBatchError(f"suite plan is not execution-ready: {exc}") from exc


def _absolute_normalized_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise LiveBatchError(f"{label} path must be absolute")
    if path.is_symlink():
        raise LiveBatchError(f"{label} path must not be a symlink")
    resolved = path.resolve(strict=False)
    if resolved != path:
        raise LiveBatchError(
            f"{label} path must be normalized and contain no symlink components"
        )
    return resolved


def _absolute_existing_file(value: str | Path, label: str) -> Path:
    path = _absolute_normalized_path(value, label)
    if not path.is_file() or path.is_symlink():
        raise LiveBatchError(f"{label} must be an existing regular file")
    return path.resolve(strict=True)


def _payload_absolute_path(
    payload: Mapping[str, Any],
    key: str,
    label: str,
    *,
    must_exist: bool = False,
) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise LiveBatchError(f"{label} must be a non-empty path string")
    path = _absolute_normalized_path(Path(value), label)
    if must_exist and not path.exists():
        raise LiveBatchError(f"{label} does not exist")
    return path


def _read_exact_json_object(
    path: Path, label: str
) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LiveBatchError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LiveBatchError(f"{label} must be a JSON object")
    return raw, payload


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _require_exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise LiveBatchError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_exact_integer(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise LiveBatchError(f"{label} must equal integer {expected}")


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise LiveBatchError(f"cannot write immutable batch manifest {path}: {exc}") from exc


__all__ = [
    "LIVE_BATCH_MANIFEST_FILENAME",
    "LIVE_BATCH_MANIFEST_KIND",
    "LIVE_BATCH_MANIFEST_SCHEMA_VERSION",
    "LIVE_BATCH_PARTITIONS",
    "LiveBatchCell",
    "LiveBatchError",
    "TransferArtifactPaths",
    "VerifiedLiveBatch",
    "execute_live_batch",
    "inspect_live_batch",
    "load_live_batch",
    "prepare_live_batch",
]
