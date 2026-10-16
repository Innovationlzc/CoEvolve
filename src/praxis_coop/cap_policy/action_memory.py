"""Verifier-gated executable exemplars and accepted episode manifests."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import EvaluationReport, ExecutionTrace, PolicyScript


ACTION_MEMORY_SCHEMA_VERSION = 1
EPISODE_MANIFEST_SCHEMA_VERSION = 1
_OBJECTIVE_KEYS = {
    "coordinated_linear_transport": ("direction",),
    "coordinated_arc_transport": ("turn_direction", "travel_direction"),
    "coordinated_pivot": ("turn_direction", "travel_direction"),
    "coordinated_alignment": ("axis",),
    "carry_forward": (),
    "carry_backward": (),
    "carry_lateral_left": (),
    "carry_lateral_right": (),
    "carry_turn_in_place": ("yaw_deg",),
    "place_crouch": (),
    "recover_stand": (),
}


class ActionMemoryError(ValueError):
    """Raised when executable memory is malformed or not verifier-grounded."""


class VerifiedActionMemoryStore:
    """Persist successful robot scripts under an exact action-context key.

    Retrieval happens only after the Agent has selected a primitive. A stored
    exemplar is evidence that a transaction executed safely in its source
    context; it is never evidence that the same direction is correct now.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self._load_or_default()

    @property
    def exemplars(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self.data.get("exemplars", ()))

    def save(self) -> None:
        self._validate_document(self.data)
        _atomic_write_json(self.path, self.data)

    def reset(self) -> None:
        self.data = _empty_action_memory()

    def revoke_checkpoints(self, checkpoint_ids: Iterable[str]) -> int:
        """Remove executable examples created by or descending from revoked states."""

        revoked = {str(item).strip() for item in checkpoint_ids if str(item).strip()}
        if not revoked:
            return 0
        exemplars = list(self.data.get("exemplars", ()))
        retained = []
        for exemplar in exemplars:
            source = exemplar.get("source", {})
            source = source if isinstance(source, dict) else {}
            if {
                str(source.get("checkpoint_before", "")),
                str(source.get("checkpoint_after", "")),
            } & revoked:
                continue
            retained.append(exemplar)
        removed = len(exemplars) - len(retained)
        self.data["exemplars"] = retained
        self.data["revoked_checkpoint_ids"] = sorted(
            set(self.data.get("revoked_checkpoint_ids", ())) | revoked
        )
        self.save()
        return removed

    def revoke_source_artifacts(self, artifact_dirs: Iterable[str | Path]) -> int:
        """Remove examples produced by exact retired rollout artifacts.

        Checkpoint ids are local to one Term1 process and are reused across
        experiments. Artifact directories identify the retired transition
        without accidentally deleting valid examples from another run.
        """

        revoked = {
            _normalize_artifact_dir(item)
            for item in artifact_dirs
            if str(item).strip()
        }
        if not revoked:
            return 0
        exemplars = list(self.data.get("exemplars", ()))
        retained = []
        removed_sources: set[str] = set()
        for exemplar in exemplars:
            source = exemplar.get("source", {})
            source = source if isinstance(source, dict) else {}
            artifact_dir = _normalize_artifact_dir(source.get("artifact_dir", ""))
            if artifact_dir in revoked:
                removed_sources.add(artifact_dir)
                continue
            retained.append(exemplar)
        removed = len(exemplars) - len(retained)
        self.data["exemplars"] = retained
        self.data["revoked_source_artifacts"] = sorted(
            set(self.data.get("revoked_source_artifacts", ())) | removed_sources
        )
        self.save()
        return removed

    def retrieve(
        self,
        *,
        method: str,
        semantic_objective: dict[str, object] | None = None,
        coordination_mode: str = "",
        phase: str = "",
        limit: int = 2,
    ) -> list[dict[str, Any]]:
        """Return only same-method, same-direction verified examples."""

        method = str(method).strip()
        mode = str(coordination_mode).strip().lower()
        objective = _normalize_objective(semantic_objective or {})
        phase = str(phase).strip().lower()
        scored: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        for index, exemplar in enumerate(self.data.get("exemplars", ())):
            if str(exemplar.get("method", "")) != method:
                continue
            exemplar_mode = str(exemplar.get("coordination_mode", "")).lower()
            if mode and exemplar_mode != mode:
                continue
            exemplar_objective = _normalize_objective(
                exemplar.get("semantic_objective", {})
            )
            if any(exemplar_objective.get(key) != value for key, value in objective.items()):
                continue
            exemplar_phase = str(exemplar.get("strategy_phase", "")).lower()
            score = (
                2.0 if objective and exemplar_objective == objective else 1.0,
                1.0 if phase and exemplar_phase == phase else 0.0,
                float(exemplar.get("accepted_at_unix", 0.0)),
                float(index),
            )
            scored.append((score, exemplar))
        selected = [
            _prompt_exemplar(item)
            for _score, item in sorted(scored, key=lambda item: item[0], reverse=True)[
                : max(0, int(limit))
            ]
        ]
        return selected

    def record_accepted(
        self,
        *,
        policy: PolicyScript,
        trace: ExecutionTrace,
        report: EvaluationReport,
        checkpoint_before: str,
        checkpoint_after: str,
        strategy_phase: str,
        source_artifact_dir: str | Path,
    ) -> bool:
        """Store one transaction only after checkpoint round-trip acceptance."""

        if (
            not report.success
            or not _trace_has_completed_team_publish(trace)
            or not checkpoint_after
            or checkpoint_after == checkpoint_before
        ):
            return False
        transition = report.evidence.get("transition_decision", {})
        if not isinstance(transition, dict) or str(
            transition.get("outcome", "")
        ).strip() not in {"ACCEPT_PROGRESS", "TASK_COMPLETE"}:
            return False
        if transition.get("positive_route_memory_allowed") is not True:
            return False
        route_quality = report.evidence.get("accepted_route_quality", {})
        if (
            isinstance(route_quality, dict)
            and route_quality.get("positive_memory_eligible") is False
        ):
            return False
        robot_calls = _robot_task_calls(policy, trace)
        if set(robot_calls) != {"front", "back"}:
            return False
        method = _primary_method(robot_calls)
        primary_parameters = next(
            item["parameters"]
            for item in robot_calls.values()
            if item["method"] == method
        )
        objective = semantic_objective_for_call(method, primary_parameters)
        mode = _policy_coordination_mode(policy)
        evidence_key = _evidence_key(
            source=str(source_artifact_dir),
            checkpoint_before=checkpoint_before,
            checkpoint_after=checkpoint_after,
            script_id=policy.script_id,
        )
        if any(
            item.get("evidence_key") == evidence_key
            for item in self.data.get("exemplars", ())
        ):
            return False
        exemplar = {
            "id": f"action_{evidence_key[:16]}",
            "evidence_key": evidence_key,
            "status": "verifier_accepted_at_source_checkpoint",
            "method": method,
            "semantic_objective": objective,
            "coordination_mode": mode,
            "strategy_phase": str(strategy_phase),
            "robot_calls": robot_calls,
            "source": {
                "artifact_dir": str(source_artifact_dir),
                "script_id": policy.script_id,
                "attempt": policy.attempt,
                "checkpoint_before": checkpoint_before,
                "checkpoint_after": checkpoint_after,
            },
            "measured_outcome": _compact_metrics(report.metrics),
            "accepted_at_unix": max(
                (float(call.finished_at_unix) for call in trace.skill_calls),
                default=0.0,
            ),
            "reuse_contract": (
                "Execution exemplar only. Reuse after the Agent independently selects the same "
                "primitive, semantic direction, and coordination mode in comparable loaded "
                "geometry; it does not prove that direction is correct in the current scene."
            ),
        }
        self.data.setdefault("exemplars", []).append(exemplar)
        self.data["exemplars"] = _bounded_exemplars(self.data["exemplars"])
        self.save()
        return True

    def rebuild_from_raw(
        self,
        sources: Iterable[tuple[Path, Path]],
        *,
        excluded_evidence: Iterable[tuple[Path, str]] = (),
    ) -> dict[str, int]:
        """Rebuild from accepted raw records and their generated policy artifacts."""

        excluded = {
            (str(Path(path).resolve()), str(checkpoint))
            for path, checkpoint in excluded_evidence
        }
        self.reset()
        counts = {"seen": 0, "accepted": 0, "excluded": 0, "ignored": 0}
        for raw_path, artifact_root in sources:
            resolved_raw = str(Path(raw_path).resolve())
            for record in _read_jsonl(raw_path):
                counts["seen"] += 1
                if str(record.get("kind", "")) != "physical_rollout_finalized":
                    counts["ignored"] += 1
                    continue
                verifier = record.get("verifier", {})
                verifier = verifier if isinstance(verifier, dict) else {}
                if not bool(record.get("physically_dispatched", False)) or not bool(
                    record.get("physical_transaction_completed", False)
                ):
                    counts["ignored"] += 1
                    continue
                if str(verifier.get("transition_outcome", "")) == "ACCEPT_SETUP":
                    counts["ignored"] += 1
                    continue
                if verifier.get("positive_route_memory_allowed") is False:
                    counts["ignored"] += 1
                    continue
                checkpoint_after = str(verifier.get("checkpoint_after", ""))
                if (resolved_raw, checkpoint_after) in excluded:
                    counts["excluded"] += 1
                    continue
                if not bool(verifier.get("step_accepted", False)) or not checkpoint_after:
                    counts["ignored"] += 1
                    continue
                exemplar = _exemplar_from_raw_record(
                    record,
                    raw_path=Path(raw_path),
                    artifact_root=Path(artifact_root),
                )
                if exemplar is None:
                    counts["ignored"] += 1
                    continue
                self.data.setdefault("exemplars", []).append(exemplar)
                counts["accepted"] += 1
        self.data["exemplars"] = _bounded_exemplars(self.data.get("exemplars", []))
        self.data["rebuild"] = {
            "sources": [str(Path(item[0])) for item in sources],
            "excluded_evidence": [
                {"raw_path": str(path), "checkpoint_after": checkpoint}
                for path, checkpoint in excluded_evidence
            ],
            "counts": counts,
        }
        self.save()
        return counts

    def _load_or_default(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_action_memory()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ActionMemoryError(f"invalid action memory JSON: {exc}") from exc
        self._validate_document(payload)
        return payload

    @staticmethod
    def _validate_document(payload: object) -> None:
        if not isinstance(payload, dict):
            raise ActionMemoryError("action memory root must be an object")
        if payload.get("kind") != "verified_action_exemplar_memory":
            raise ActionMemoryError("unexpected action memory kind")
        if int(payload.get("schema_version", -1)) != ACTION_MEMORY_SCHEMA_VERSION:
            raise ActionMemoryError("unsupported action memory schema")
        exemplars = payload.get("exemplars")
        if not isinstance(exemplars, list):
            raise ActionMemoryError("action memory exemplars must be a list")
        ids: set[str] = set()
        for exemplar in exemplars:
            if not isinstance(exemplar, dict):
                raise ActionMemoryError("each action exemplar must be an object")
            required = {
                "id",
                "status",
                "method",
                "semantic_objective",
                "coordination_mode",
                "robot_calls",
                "source",
                "reuse_contract",
            }
            if missing := required - set(exemplar):
                raise ActionMemoryError(f"action exemplar missing fields: {sorted(missing)}")
            if exemplar["id"] in ids:
                raise ActionMemoryError(f"duplicate action exemplar id: {exemplar['id']}")
            ids.add(str(exemplar["id"]))
            if exemplar.get("status") != "verifier_accepted_at_source_checkpoint":
                raise ActionMemoryError("unverified exemplar cannot enter action memory")
            if set(exemplar.get("robot_calls", {})) != {"front", "back"}:
                raise ActionMemoryError("action exemplar requires front/back robot calls")


class AcceptedEpisodeManifest:
    """A contiguous list of scripts that created accepted checkpoints."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self._load_or_default()

    def initialize(
        self,
        *,
        checkpoint_id: str,
        initial_state: dict[str, Any],
        reset: bool,
    ) -> None:
        if reset:
            self.data = _empty_episode_manifest()
        if self.data.get("steps"):
            if self.latest_checkpoint != checkpoint_id:
                raise ActionMemoryError(
                    f"episode manifest ends at {self.latest_checkpoint}, not {checkpoint_id}"
                )
            return
        self.data["initial_checkpoint"] = checkpoint_id
        self.data["latest_checkpoint"] = checkpoint_id
        self.data["initial_state"] = _compact_initial_state(initial_state)
        self.save()

    @property
    def latest_checkpoint(self) -> str:
        return str(self.data.get("latest_checkpoint", ""))

    def rewind_to_checkpoint(self, checkpoint_id: str) -> int:
        """Retire accepted descendants after restoring an ancestor checkpoint."""

        checkpoint_id = str(checkpoint_id).strip()
        initial_checkpoint = str(self.data.get("initial_checkpoint", "")).strip()
        steps = list(self.data.get("steps", ()))
        if checkpoint_id == initial_checkpoint:
            retained_steps: list[dict[str, Any]] = []
        else:
            retained_steps = []
            for index, step in enumerate(steps):
                if str(step.get("checkpoint_after", "")) == checkpoint_id:
                    retained_steps = steps[: index + 1]
                    break
            else:
                raise ActionMemoryError(
                    f"checkpoint {checkpoint_id!r} is not in the accepted episode lineage"
                )

        retired_count = len(steps) - len(retained_steps)
        self.data["steps"] = retained_steps
        self.data["latest_checkpoint"] = checkpoint_id
        self.save()
        return retired_count

    def append_accepted(
        self,
        *,
        policy: PolicyScript,
        trace: ExecutionTrace,
        report: EvaluationReport,
        checkpoint_before: str,
        checkpoint_after: str,
        source_artifact_dir: str | Path,
    ) -> dict[str, Any]:
        if not report.success or not checkpoint_after or checkpoint_after == checkpoint_before:
            raise ActionMemoryError("episode steps must create a verified checkpoint")
        if self.latest_checkpoint != checkpoint_before:
            raise ActionMemoryError(
                f"non-contiguous episode lineage: {self.latest_checkpoint} -> {checkpoint_before}"
            )
        robot_calls = _robot_task_calls(policy, trace)
        if set(robot_calls) != {"front", "back"}:
            raise ActionMemoryError("episode step requires front/back executable calls")
        scripts = {item.robot_id: asdict(item) for item in policy.robot_scripts}
        step = {
            "index": len(self.data.get("steps", ())) + 1,
            "script_id": policy.script_id,
            "attempt": policy.attempt,
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
            "coordination_mode": _policy_coordination_mode(policy),
            "robot_calls": robot_calls,
            "robot_scripts": scripts,
            "verification_contract": dict(policy.verification_contract),
            "measured_outcome": _compact_metrics(report.metrics),
            "source_artifact_dir": str(source_artifact_dir),
        }
        transition = report.evidence.get("transition_decision", {})
        if isinstance(transition, dict) and transition:
            step["transition_outcome"] = str(transition.get("outcome", ""))
            step["positive_route_memory_eligible"] = bool(
                transition.get("positive_route_memory_allowed", False)
            )
        previous_steps = self.data.get("steps", ())
        previous_step = (
            previous_steps[-1]
            if isinstance(previous_steps, list) and previous_steps
            else {}
        )
        route_quality = _accepted_route_quality(
            previous_step,
            step,
            report=report,
        )
        if route_quality:
            step["route_quality"] = route_quality
        self.data.setdefault("steps", []).append(step)
        self.data["latest_checkpoint"] = checkpoint_after
        self.save()
        return dict(route_quality)

    def save(self) -> None:
        _validate_episode_manifest(self.data)
        _atomic_write_json(self.path, self.data)

    def _load_or_default(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_episode_manifest()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        _validate_episode_manifest(payload)
        return payload


def semantic_objective_for_call(
    method: str,
    parameters: dict[str, Any],
) -> dict[str, object]:
    keys = _OBJECTIVE_KEYS.get(str(method), ())
    objective = {key: parameters[key] for key in keys if key in parameters}
    if method == "carry_forward":
        objective["direction"] = "forward"
    elif method == "carry_backward":
        objective["direction"] = "backward"
    elif method == "carry_lateral_left":
        objective["direction"] = "left"
    elif method == "carry_lateral_right":
        objective["direction"] = "right"
    return _normalize_objective(objective)


def _accepted_route_quality(
    previous_step: object,
    current_step: object,
    *,
    report: EvaluationReport,
) -> dict[str, Any]:
    """Separate restorable setup checkpoints from reusable route progress."""

    transition = report.evidence.get("transition_decision", {})
    transition = transition if isinstance(transition, dict) else {}
    if transition.get("positive_route_memory_allowed") is False:
        return {
            "classification": "accepted_setup_without_route_progress",
            "transition_outcome": str(transition.get("outcome", "ACCEPT_SETUP")),
            "physical_checkpoint_retained": True,
            "positive_memory_eligible": False,
            "reason": (
                "The transition verifier classified this endpoint as restorable "
                "setup rather than connected-route progress."
            ),
        }

    previous_direction = _episode_linear_direction(previous_step)
    current_direction = _episode_linear_direction(current_step)
    opposite = {
        "forward": "backward",
        "backward": "forward",
        "left": "right",
        "right": "left",
    }.get(previous_direction, "")
    if opposite and current_direction == opposite:
        return {
            "classification": "accepted_immediate_linear_reversal",
            "physical_checkpoint_retained": True,
            "positive_memory_eligible": False,
            "earlier_direction": previous_direction,
            "latest_direction": current_direction,
            "reason": (
                "The endpoint is physically restorable, but this action immediately "
                "reversed the preceding accepted body-frame translation and therefore "
                "does not establish reusable route progress."
            ),
        }

    method = _episode_primary_method(current_step)
    if method not in {
        "coordinated_alignment",
        "coordinated_arc_transport",
        "coordinated_pivot",
        "turn_clockwise",
        "turn_counterclockwise",
    }:
        return {}
    transition = report.evidence.get("transition_decision", {})
    transition = transition if isinstance(transition, dict) else {}
    evidence = transition.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    progress = evidence.get("active_phase_progress", {})
    progress = progress if isinstance(progress, dict) else {}
    try:
        phase_delta_m = float(progress["delta_m"])
        target_delta_m = float(report.metrics.get("payload_target_progress_m", 0.0))
    except (KeyError, TypeError, ValueError):
        return {}
    if phase_delta_m > 0.0 or target_delta_m > 0.0:
        return {}
    return {
        "classification": "accepted_local_setup_without_route_progress",
        "physical_checkpoint_retained": True,
        "positive_memory_eligible": False,
        "active_phase_delta_m": phase_delta_m,
        "target_progress_m": target_delta_m,
        "reason": (
            "The local correction is a restorable setup checkpoint, but neither "
            "active-phase nor target progress was measured. A later connected-passage "
            "probe must validate the maneuver before it becomes reusable route memory."
        ),
    }


def _episode_primary_method(step: object) -> str:
    if not isinstance(step, dict):
        return ""
    robot_calls = step.get("robot_calls", {})
    if not isinstance(robot_calls, dict):
        return ""
    methods = {
        str(call.get("method", "")).strip()
        for call in robot_calls.values()
        if isinstance(call, dict) and str(call.get("method", "")).strip()
    }
    return next(iter(methods)) if len(methods) == 1 else ""


def _episode_linear_direction(step: object) -> str:
    if not isinstance(step, dict):
        return ""
    robot_calls = step.get("robot_calls", {})
    if not isinstance(robot_calls, dict):
        return ""
    representative = next(
        (
            robot_calls.get(robot_id)
            for robot_id in ("front", "back")
            if isinstance(robot_calls.get(robot_id), dict)
        ),
        {},
    )
    if str(representative.get("method", "")).strip() != "coordinated_linear_transport":
        return ""
    parameters = representative.get("parameters", {})
    if not isinstance(parameters, dict):
        return ""
    direction = str(parameters.get("direction", "")).strip()
    return direction if direction in {"forward", "backward", "left", "right"} else ""


def selected_subtask_objective(selected_subtask: dict[str, Any]) -> dict[str, object]:
    parameter = str(selected_subtask.get("motion_objective_parameter", "")).strip()
    value = selected_subtask.get("motion_objective")
    return {parameter: value} if parameter and value is not None and value != "" else {}


def _robot_task_calls(
    policy: PolicyScript,
    trace: ExecutionTrace,
) -> dict[str, dict[str, Any]]:
    scripts = {item.robot_id: item for item in policy.robot_scripts}
    calls: dict[str, dict[str, Any]] = {}
    for execution in trace.robot_executions:
        task_calls = [call for call in execution.planned_calls if call.method != "stop_all"]
        if len(task_calls) != 1:
            continue
        call = task_calls[0]
        script = scripts.get(execution.robot_id)
        calls[execution.robot_id] = {
            "method": call.method,
            "parameters": dict(call.parameters),
            "code": "" if script is None else script.code,
        }
    return calls


def _trace_has_completed_team_publish(trace: ExecutionTrace) -> bool:
    """Require a real, completed front/back dispatch before learning an action."""

    if trace.status != "success":
        return False
    executions = {item.robot_id: item for item in trace.robot_executions}
    return set(executions) == {"front", "back"} and all(
        item.status == "command_publish_completed" for item in executions.values()
    )


def _primary_method(robot_calls: dict[str, dict[str, Any]]) -> str:
    methods = [str(item.get("method", "")) for item in robot_calls.values()]
    movers = [method for method in methods if method != "hold_position"]
    if not movers:
        return "hold_position"
    if len(set(movers)) != 1:
        raise ActionMemoryError(f"accepted transaction has multiple mover methods: {methods}")
    return movers[0]


def _policy_coordination_mode(policy: PolicyScript) -> str:
    modes = {
        str(item.coordination_mode).strip().lower()
        for item in policy.robot_scripts
        if str(item.coordination_mode).strip()
    }
    return next(iter(modes)) if len(modes) == 1 else "sync"


def _prompt_exemplar(exemplar: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": exemplar["status"],
        "method": exemplar["method"],
        "semantic_objective": exemplar.get("semantic_objective", {}),
        "coordination_mode": exemplar.get("coordination_mode", ""),
        "strategy_phase": exemplar.get("strategy_phase", ""),
        "robot_calls": exemplar.get("robot_calls", {}),
        "measured_outcome": exemplar.get("measured_outcome", {}),
        "source": exemplar.get("source", {}),
        "reuse_contract": exemplar.get("reuse_contract", ""),
    }


def _exemplar_from_raw_record(
    record: dict[str, Any],
    *,
    raw_path: Path,
    artifact_root: Path,
) -> dict[str, Any] | None:
    verifier = record.get("verifier", {})
    verifier = verifier if isinstance(verifier, dict) else {}
    parameters = record.get("policy_execution_parameters", ())
    if not isinstance(parameters, list):
        return None
    robot_calls: dict[str, dict[str, Any]] = {}
    attempt = int(record.get("attempt", 0))
    policy_path = artifact_root / f"attempt_{attempt:02d}" / "policy.json"
    policy_payload = _read_json(policy_path)
    script_by_robot = {
        str(item.get("robot_id", "")): str(item.get("code", ""))
        for item in policy_payload.get("robot_scripts", ())
        if isinstance(item, dict)
    }
    for item in parameters:
        if not isinstance(item, dict):
            continue
        robot_id = str(item.get("robot_id", ""))
        if robot_id not in {"front", "back"}:
            continue
        robot_calls[robot_id] = {
            "method": str(item.get("skill", "")),
            "parameters": dict(item.get("parameters", {})),
            "code": script_by_robot.get(robot_id, ""),
        }
    if set(robot_calls) != {"front", "back"}:
        return None
    method = _primary_method(robot_calls)
    primary = next(item for item in robot_calls.values() if item["method"] == method)
    checkpoint_before = str(verifier.get("checkpoint_before", ""))
    checkpoint_after = str(verifier.get("checkpoint_after", ""))
    evidence_key = _evidence_key(
        source=str(raw_path.resolve()),
        checkpoint_before=checkpoint_before,
        checkpoint_after=checkpoint_after,
        script_id=str(record.get("script_id", "")),
    )
    context = record.get("strategy_context", {})
    context = context if isinstance(context, dict) else {}
    return {
        "id": f"action_{evidence_key[:16]}",
        "evidence_key": evidence_key,
        "status": "verifier_accepted_at_source_checkpoint",
        "method": method,
        "semantic_objective": semantic_objective_for_call(
            method,
            primary["parameters"],
        ),
        "coordination_mode": str(record.get("coordination_mode", "sync")),
        "strategy_phase": str(context.get("phase", "")),
        "robot_calls": robot_calls,
        "source": {
            "raw_path": str(raw_path),
            "artifact_dir": str(policy_path.parent),
            "script_id": str(record.get("script_id", "")),
            "attempt": attempt,
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
        },
        "measured_outcome": _compact_metrics(verifier.get("metrics", {})),
        "accepted_at_unix": 0.0,
        "reuse_contract": (
            "Execution exemplar only. Reuse after the Agent independently selects the same "
            "primitive, semantic direction, and coordination mode in comparable loaded "
            "geometry; it does not prove that direction is correct in the current scene."
        ),
    }


def _bounded_exemplars(exemplars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for exemplar in exemplars:
        key = (
            str(exemplar.get("method", "")),
            json.dumps(
                _normalize_objective(exemplar.get("semantic_objective", {})),
                sort_keys=True,
            ),
            str(exemplar.get("coordination_mode", "")),
        )
        grouped.setdefault(key, []).append(exemplar)
    bounded: list[dict[str, Any]] = []
    for items in grouped.values():
        bounded.extend(items[-5:])
    return bounded


def _compact_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    keys = (
        "payload_x_m",
        "payload_y_m",
        "payload_z_m",
        "payload_step_displacement_xy_m",
        "payload_step_signed_yaw_change_deg",
        "team_step_signed_yaw_change_deg",
        "payload_target_progress_m",
        "front_back_step_vector_desync_m",
        "robot_pair_longitudinal_stagger_m",
        "robot_pair_formation_skew_deg",
        "robot_heading_disagreement_after_deg",
        "payload_formation_slip_m",
        "payload_support_center_error_m",
        "payload_tilt_deg",
        "wall_contact_count",
    )
    compact: dict[str, float] = {}
    for key in keys:
        try:
            compact[key] = float(metrics[key])
        except (KeyError, TypeError, ValueError):
            continue
    return compact


def _compact_initial_state(state: dict[str, Any]) -> dict[str, Any]:
    payload = state.get("payload", {})
    robots = state.get("robots", {})
    return {
        "payload": dict(payload) if isinstance(payload, dict) else {},
        "robots": dict(robots) if isinstance(robots, dict) else {},
        "source": state.get("source", ""),
    }


def _normalize_objective(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in sorted(value.items())
        if item is not None and item != ""
    }


def _evidence_key(
    *,
    source: str,
    checkpoint_before: str,
    checkpoint_after: str,
    script_id: str,
) -> str:
    payload = "|".join((source, checkpoint_before, checkpoint_after, script_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _empty_action_memory() -> dict[str, Any]:
    return {
        "kind": "verified_action_exemplar_memory",
        "schema_version": ACTION_MEMORY_SCHEMA_VERSION,
        "selection_policy": (
            "Retrieve only after primitive selection; exact method, semantic direction, and "
            "coordination mode must match. Source success is not a current route instruction."
        ),
        "exemplars": [],
    }


def _normalize_artifact_dir(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return Path(text).as_posix().rstrip("/")


def _empty_episode_manifest() -> dict[str, Any]:
    return {
        "kind": "accepted_checkpoint_episode",
        "schema_version": EPISODE_MANIFEST_SCHEMA_VERSION,
        "initial_checkpoint": "",
        "latest_checkpoint": "",
        "initial_state": {},
        "steps": [],
    }


def _validate_episode_manifest(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ActionMemoryError("episode manifest root must be an object")
    if payload.get("kind") != "accepted_checkpoint_episode":
        raise ActionMemoryError("unexpected episode manifest kind")
    if int(payload.get("schema_version", -1)) != EPISODE_MANIFEST_SCHEMA_VERSION:
        raise ActionMemoryError("unsupported episode manifest schema")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ActionMemoryError("episode steps must be a list")
    expected = str(payload.get("initial_checkpoint", ""))
    for step in steps:
        if not isinstance(step, dict) or step.get("checkpoint_before") != expected:
            raise ActionMemoryError("episode checkpoint lineage is not contiguous")
        if set(step.get("robot_scripts", {})) != {"front", "back"}:
            raise ActionMemoryError("episode step must contain front/back scripts")
        expected = str(step.get("checkpoint_after", ""))
    if steps and expected != str(payload.get("latest_checkpoint", "")):
        raise ActionMemoryError("episode latest checkpoint does not match its lineage")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "AcceptedEpisodeManifest",
    "ActionMemoryError",
    "VerifiedActionMemoryStore",
    "selected_subtask_objective",
    "semantic_objective_for_call",
]
