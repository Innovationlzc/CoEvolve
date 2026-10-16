"""Strict, reusable reader for an externally frozen Tables 2--4 plan."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .contracts import ExperimentSuite, ProtocolLock, RunSpec
from .lockfile import protocol_lock_from_mapping
from .registry import build_suite
from .scenes import load_scene, validate_scene
from .serialization import strict_json_loads, to_primitive


SUITE_PLAN_KIND = "coevolve_tables_2_4_suite_plan"
SUITE_PLAN_SCHEMA_VERSION = 2
_ROOT_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "planning_validated",
        "execution_preflight",
        "protocol_lock",
        "suite",
        "execution_reports",
    }
)


class SuitePlanError(ValueError):
    """Raised when a plan is self-described, stale, or otherwise untrusted."""


@dataclass(frozen=True, slots=True)
class VerifiedSuiteDocument:
    """Registry-rebuilt content shared by every plan-consuming boundary."""

    protocol_lock: ProtocolLock
    suite: ExperimentSuite
    execution_preflight: str
    execution_reports: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VerifiedSuitePlan:
    path: Path
    sha256: str
    protocol_lock: ProtocolLock
    suite: ExperimentSuite
    execution_preflight: str
    execution_reports: Tuple[Mapping[str, Any], ...]

    def resolve_run(self, run_id: str) -> RunSpec:
        matches = tuple(run for run in self.suite.new_runs if run.run_id == run_id)
        if len(matches) != 1:
            raise SuitePlanError(
                f"run_id must resolve exactly once in suite plan: {run_id!r}"
            )
        return matches[0]


def load_verified_suite_plan(
    path: str | Path,
    *,
    require_execution_ready: bool = False,
    protocol_root: Optional[Path] = None,
    scene_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> VerifiedSuitePlan:
    """Read, rebuild, and optionally re-run every scene execution preflight.

    A plan is not trusted merely because it says ``planning_validated`` or
    ``execution_preflight=passed``.  Its suite is rebuilt from the current
    declarative registry and lock, and execution evidence is rehashed when a
    launch-capable caller requests it.
    """

    source = Path(path)
    if not source.is_absolute():
        raise SuitePlanError("formal suite plan path must be absolute")
    if source.is_symlink():
        raise SuitePlanError("formal suite plan must not be a symlink")
    try:
        raw = source.read_bytes()
        payload = strict_json_loads(raw)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SuitePlanError(f"cannot read suite plan {source}: {exc}") from exc
    document = validate_suite_plan_document(
        payload,
        require_execution_ready=require_execution_ready,
        protocol_root=protocol_root,
        scene_root=scene_root,
        repo_root=repo_root,
    )
    return VerifiedSuitePlan(
        path=source.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        protocol_lock=document.protocol_lock,
        suite=document.suite,
        execution_preflight=document.execution_preflight,
        execution_reports=document.execution_reports,
    )


def validate_suite_plan_document(
    payload: object,
    *,
    require_execution_ready: bool = False,
    protocol_root: Optional[Path] = None,
    scene_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> VerifiedSuiteDocument:
    """Validate an already decoded plan with the same rules as the file reader."""

    if not isinstance(payload, Mapping):
        raise SuitePlanError("suite plan root must be an object")
    actual_fields = frozenset(payload)
    if actual_fields != _ROOT_FIELDS:
        raise SuitePlanError(
            "suite plan fields differ: "
            f"missing={sorted(_ROOT_FIELDS - actual_fields)}, "
            f"unknown={sorted(actual_fields - _ROOT_FIELDS)}"
        )
    if payload.get("kind") != SUITE_PLAN_KIND:
        raise SuitePlanError("suite plan has the wrong kind")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SUITE_PLAN_SCHEMA_VERSION
    ):
        raise SuitePlanError("suite plan has unsupported schema_version")
    if payload.get("planning_validated") is not True:
        raise SuitePlanError("suite plan is not planning validated")
    execution_preflight = payload.get("execution_preflight")
    if execution_preflight not in {"not_requested", "passed"}:
        raise SuitePlanError("suite plan has an invalid execution_preflight state")

    lock_payload = payload.get("protocol_lock")
    suite_payload = payload.get("suite")
    reports_payload = payload.get("execution_reports")
    if not isinstance(lock_payload, Mapping):
        raise SuitePlanError("suite plan protocol_lock is malformed")
    if not isinstance(suite_payload, Mapping):
        raise SuitePlanError("suite plan suite is malformed")
    if not isinstance(reports_payload, list) or not all(
        isinstance(item, Mapping) for item in reports_payload
    ):
        raise SuitePlanError("suite plan execution_reports must be an array of objects")

    try:
        lock = protocol_lock_from_mapping(lock_payload)
        run_plan = suite_payload.get("run_plan")
        if not isinstance(run_plan, Mapping):
            raise SuitePlanError("suite plan run_plan is malformed")
        rebuilt = build_suite(
            str(run_plan.get("tier", "")),
            lock,
            protocol_root=protocol_root,
            scene_root=scene_root,
        )
    except SuitePlanError:
        raise
    except (TypeError, ValueError) as exc:
        raise SuitePlanError(f"cannot rebuild suite plan: {exc}") from exc
    if suite_payload != to_primitive(rebuilt):
        raise SuitePlanError(
            "suite plan does not exactly match the registry rebuilt from its lock"
        )

    reports = tuple(dict(item) for item in reports_payload)
    if require_execution_ready:
        if execution_preflight != "passed":
            raise SuitePlanError("formal launch requires execution_preflight='passed'")
        _require_current_execution_evidence(
            rebuilt,
            lock,
            reports,
            scene_root=scene_root,
            repo_root=repo_root,
        )

    return VerifiedSuiteDocument(
        protocol_lock=lock,
        suite=rebuilt,
        execution_preflight=str(execution_preflight),
        execution_reports=reports,
    )


def _require_current_execution_evidence(
    suite: ExperimentSuite,
    lock: ProtocolLock,
    planned_reports: Tuple[Mapping[str, Any], ...],
    *,
    scene_root: Optional[Path],
    repo_root: Optional[Path],
) -> None:
    scene_ids = tuple(
        sorted({run.scene_id for run in suite.new_runs}, key=lambda item: item.value)
    )
    current = tuple(
        validate_scene(
            load_scene(scene_id, root=scene_root),
            mode="execution",
            protocol_lock=lock,
            repo_root=repo_root,
            scene_root=scene_root,
        )
        for scene_id in scene_ids
    )
    current_payload = tuple(to_primitive(report) for report in current)
    if tuple(planned_reports) != current_payload:
        raise SuitePlanError(
            "suite plan execution reports do not match current hash-pinned preflight"
        )
    if any(not report.execution_ready for report in current):
        raise SuitePlanError("one or more selected scenes are not execution ready")


__all__ = [
    "SUITE_PLAN_KIND",
    "SUITE_PLAN_SCHEMA_VERSION",
    "SuitePlanError",
    "VerifiedSuiteDocument",
    "VerifiedSuitePlan",
    "load_verified_suite_plan",
    "validate_suite_plan_document",
]
