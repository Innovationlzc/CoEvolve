#!/usr/bin/env python3
"""Export and model-free replay a trusted accepted-checkpoint episode."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "deployment"
    / "cap_box_carry_wall_route_20260726_run01"
)
ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")
ATTEMPT_TOKEN_RE = re.compile(r"attempt[_-](\d+)", re.IGNORECASE)
GENERATED_SCHEMA_VERSION = 1


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _relative(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_artifact_path(value: object, *, run_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists() or str(path).startswith("results/"):
        return repo_candidate
    return run_dir / path


def _manifest_path(run_dir: Path) -> Path:
    candidates = (
        run_dir / "memory" / "accepted_episode_manifest.json",
        run_dir / "accepted_episode_manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"accepted episode manifest not found under {run_dir}"
    )


def _slug(value: object, *, limit: int = 72) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return (normalized or "step")[:limit]


def _api_call(code: str) -> tuple[str, dict[str, Any]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "", {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "api"
            and function.attr != "stop_all"
        ):
            parameters: dict[str, Any] = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                try:
                    parameters[keyword.arg] = ast.literal_eval(keyword.value)
                except (TypeError, ValueError):
                    parameters[keyword.arg] = ast.unparse(keyword.value)
            return function.attr, parameters
    return "", {}


def _compact_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    keys = (
        "payload_step_displacement_xy_m",
        "payload_target_progress_m",
        "front_step_displacement_m",
        "back_step_displacement_m",
        "front_back_step_vector_desync_m",
        "payload_formation_slip_m",
        "payload_support_center_error_m",
        "payload_tilt_deg",
        "payload_target_xy_error_m",
        "payload_target_z_error_m",
        "payload_target_contact",
        "payload_on_target",
        "min_robot_target_distance_m",
        "robots_clear_after_place",
        "task_complete",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _checkpoint_number(checkpoint_id: object) -> str:
    match = re.search(r"(\d+)$", str(checkpoint_id))
    return f"{int(match.group(1)):02d}" if match else "unknown"


def _copy_step_visual(
    *,
    source_dir: Path,
    checkpoint_after: str,
    attempt: int,
    output_dir: Path,
    sequence_index: int,
) -> str:
    visual_root = source_dir.parent / "accepted_step_visuals"
    checkpoint_number = _checkpoint_number(checkpoint_after)
    patterns = (
        f"accepted_step_{checkpoint_number}_attempt_{attempt:02d}_vlm_global_and_payload.png",
        f"accepted_step_{checkpoint_number}_attempt_{attempt:02d}_mujoco_global.png",
    )
    for pattern in patterns:
        candidate = visual_root / pattern
        if candidate.is_file():
            suffix = "vlm_global_and_payload" if "_vlm_" in candidate.name else "mujoco_global"
            target = output_dir / "visuals" / (
                f"{sequence_index:02d}_{checkpoint_after}_attempt_{attempt:02d}_{suffix}.png"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            return str(target.relative_to(output_dir))
    return ""


def _export_trajectory(
    *,
    run_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    steps = manifest.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("accepted episode manifest contains no steps")

    exported_steps: list[dict[str, Any]] = []
    for sequence_index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            raise TypeError(f"manifest step {sequence_index} is not an object")
        source_dir = _resolve_artifact_path(
            raw_step.get("source_artifact_dir", ""),
            run_dir=run_dir,
        )
        attempt = int(raw_step.get("attempt", 0))
        script_id = str(raw_step.get("script_id", f"step_{sequence_index:02d}"))
        step_dir = output_dir / "steps" / (
            f"{sequence_index:02d}_attempt_{attempt:02d}_{_slug(script_id)}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)

        robot_scripts = raw_step.get("robot_scripts", {})
        if not isinstance(robot_scripts, dict):
            raise TypeError(f"manifest step {sequence_index} has invalid robot_scripts")
        normalized_scripts: dict[str, dict[str, Any]] = {}
        actions: dict[str, dict[str, Any]] = {}
        for robot_id in ("front", "back"):
            script = robot_scripts.get(robot_id, {})
            if not isinstance(script, dict) or not str(script.get("code", "")).strip():
                raise ValueError(
                    f"manifest step {sequence_index} is missing {robot_id} code"
                )
            code = str(script["code"]).rstrip() + "\n"
            (step_dir / f"{robot_id}.py").write_text(code, encoding="utf-8")
            method, parameters = _api_call(code)
            normalized_scripts[robot_id] = {
                **script,
                "code": code.rstrip(),
            }
            actions[robot_id] = {
                "method": method,
                "parameters": parameters,
            }

        policy = _read_json(source_dir / "policy.json", {})
        evaluation = _read_json(source_dir / "evaluation.json", {})
        execution = _read_json(source_dir / "execution_trace.json", {})
        if policy:
            _write_json(step_dir / "policy.json", policy)
        if evaluation:
            _write_json(
                step_dir / "accepted_evaluation.json",
                {
                    "success": bool(evaluation.get("success", False)),
                    "failure_type": evaluation.get("failure_type", ""),
                    "failure_level": evaluation.get("failure_level", ""),
                    "diagnosis": evaluation.get("diagnosis", ""),
                    "visual_summary": evaluation.get("visual_summary", ""),
                    "metrics": _compact_metrics(evaluation),
                },
            )
        if execution:
            _write_json(step_dir / "accepted_execution_trace.json", execution)

        visual_path = _copy_step_visual(
            source_dir=source_dir,
            checkpoint_after=str(raw_step.get("checkpoint_after", "")),
            attempt=attempt,
            output_dir=output_dir,
            sequence_index=sequence_index,
        )
        exported = {
            **raw_step,
            "sequence_index": sequence_index,
            "source_artifact_dir": _relative(source_dir),
            "exported_step_dir": str(step_dir.relative_to(output_dir)),
            "exported_visual": visual_path,
            "robot_scripts": normalized_scripts,
            "actions": actions,
            "accepted_evaluation": {
                "success": bool(evaluation.get("success", False)),
                "failure_type": str(evaluation.get("failure_type", "")),
                "metrics": _compact_metrics(evaluation),
            },
        }
        _write_json(step_dir / "metadata.json", exported)
        exported_steps.append(exported)

    trajectory = {
        "schema_version": GENERATED_SCHEMA_VERSION,
        "kind": "trusted_accepted_episode_export",
        "generated_at_unix": time.time(),
        "source_run_dir": _relative(run_dir),
        "source_manifest": _relative(_manifest_path(run_dir)),
        "initial_checkpoint": manifest.get("initial_checkpoint", ""),
        "latest_checkpoint": manifest.get("latest_checkpoint", ""),
        "accepted_action_count": len(exported_steps),
        "checkpoint_numbering_note": (
            "Checkpoint numbers may skip after branch rewinds. Replay strictly follows "
            "this steps array and never infers order from checkpoint numbers."
        ),
        "steps": exported_steps,
    }
    _write_json(output_dir / "trajectory_manifest.json", trajectory)
    return trajectory


def _attempt_directories(run_dir: Path) -> list[Path]:
    directories: set[Path] = set()
    evidence_files = (
        "policy.json",
        "evaluation.json",
        "execution_trace.json",
        "evolution_record.json",
        "team_plan.json",
    )
    for path in run_dir.rglob("attempt_*"):
        if (
            path.is_dir()
            and ATTEMPT_RE.match(path.name)
            and any((path / name).is_file() for name in evidence_files)
        ):
            directories.add(path.resolve())
    return sorted(directories, key=lambda item: _relative(item, run_dir))


def _trace_was_dispatched(trace: dict[str, Any]) -> bool:
    for call in trace.get("skill_calls", ()):
        if not isinstance(call, dict):
            continue
        status = str(call.get("status", ""))
        if status and status != "dry_run":
            return True
    return str(trace.get("status", "")) in {"success", "script_error"}


def _attempt_times(trace: dict[str, Any]) -> tuple[float | None, float | None, float]:
    starts: list[float] = []
    finishes: list[float] = []
    active_s = 0.0
    for call in trace.get("skill_calls", ()):
        if not isinstance(call, dict):
            continue
        try:
            started = float(call["started_at_unix"])
            finished = float(call["finished_at_unix"])
        except (KeyError, TypeError, ValueError):
            continue
        starts.append(started)
        finishes.append(finished)
        active_s += max(0.0, finished - started)
    return (
        min(starts) if starts else None,
        max(finishes) if finishes else None,
        active_s,
    )


def _attempt_record(
    *,
    attempt_dir: Path,
    run_dir: Path,
    trusted_sources: set[Path],
) -> dict[str, Any]:
    policy = _read_json(attempt_dir / "policy.json", {})
    evaluation = _read_json(attempt_dir / "evaluation.json", {})
    trace = _read_json(attempt_dir / "execution_trace.json", {})
    evolution = _read_json(attempt_dir / "evolution_record.json", {})
    distribution = _read_json(attempt_dir / "task_distribution.json", {})
    outcome = (
        evaluation.get("evidence", {}).get("checkpoint_outcome", {})
        if isinstance(evaluation.get("evidence", {}), dict)
        else {}
    )
    outcome = outcome if isinstance(outcome, dict) else {}
    attempt_match = ATTEMPT_RE.match(attempt_dir.name)
    artifact_attempt = int(attempt_match.group(1)) if attempt_match else 0
    attempt = int(
        policy.get(
            "attempt",
            artifact_attempt,
        )
    )
    has_evaluation = bool(evaluation)
    success = bool(evaluation.get("success", False)) if has_evaluation else False
    if "physically_dispatched" in outcome:
        physically_dispatched = bool(outcome["physically_dispatched"])
    else:
        physically_dispatched = _trace_was_dispatched(trace)
    rollback = bool(
        outcome.get("rollback_performed", False)
        or evolution.get("rollback_performed", False)
        or distribution.get("checkpoint", {}).get("rollback_performed", False)
    )
    trusted = attempt_dir.resolve() in trusted_sources
    if trusted:
        classification = "trusted_accepted"
    elif not has_evaluation:
        classification = "interrupted_or_incomplete"
    elif success and physically_dispatched:
        classification = "retired_branch_accepted"
    elif success:
        classification = "accepted_without_physical_dispatch"
    elif physically_dispatched:
        classification = "rejected_physical_rollout"
    else:
        classification = "rejected_before_physical_dispatch"

    robot_scripts = policy.get("robot_scripts", [])
    actions: dict[str, dict[str, Any]] = {}
    if isinstance(robot_scripts, list):
        for script in robot_scripts:
            if not isinstance(script, dict):
                continue
            robot_id = str(script.get("robot_id", ""))
            method, parameters = _api_call(str(script.get("code", "")))
            if robot_id:
                actions[robot_id] = {"method": method, "parameters": parameters}
    started, finished, active_s = _attempt_times(trace)
    checkpoint_before = str(
        outcome.get("checkpoint_before")
        or evolution.get("checkpoint_before")
        or distribution.get("checkpoint", {}).get("before", "")
    )
    checkpoint_after = str(
        outcome.get("checkpoint_after")
        or evolution.get("checkpoint_after")
        or distribution.get("checkpoint", {}).get("after", "")
    )
    return {
        "artifact_dir": _relative(attempt_dir),
        "batch": _relative(attempt_dir.parent, run_dir),
        "artifact_attempt": artifact_attempt,
        "attempt": attempt,
        "script_id": str(policy.get("script_id", "")),
        "coordination_mode": (
            str(robot_scripts[0].get("coordination_mode", ""))
            if isinstance(robot_scripts, list) and robot_scripts
            and isinstance(robot_scripts[0], dict)
            else ""
        ),
        "actions": actions,
        "classification": classification,
        "trusted_main_path": trusted,
        "has_evaluation": has_evaluation,
        "evaluation_success": success,
        "failure_type": str(evaluation.get("failure_type", "")),
        "failure_level": str(evaluation.get("failure_level", "")),
        "diagnosis": str(evaluation.get("diagnosis", "")),
        "physically_dispatched": physically_dispatched,
        "trace_status": str(trace.get("status", "")),
        "trace_error": str(trace.get("error", "")),
        "rollback_performed": rollback,
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "checkpoint_disposition": str(outcome.get("checkpoint_disposition", "")),
        "metrics": _compact_metrics(evaluation),
        "started_at_unix": started,
        "finished_at_unix": finished,
        "active_physical_command_s": active_s,
    }


def _summarize_attempts(
    *,
    run_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    trusted_sources = {
        _resolve_artifact_path(step.get("source_artifact_dir", ""), run_dir=run_dir).resolve()
        for step in manifest.get("steps", [])
        if isinstance(step, dict)
    }
    attempts = [
        _attempt_record(
            attempt_dir=attempt_dir,
            run_dir=run_dir,
            trusted_sources=trusted_sources,
        )
        for attempt_dir in _attempt_directories(run_dir)
    ]
    class_counts = Counter(row["classification"] for row in attempts)
    failure_counts = Counter(
        row["failure_type"]
        for row in attempts
        if row["has_evaluation"] and not row["evaluation_success"]
    )
    batch_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in attempts:
        batch_counts[row["batch"]]["attempt_artifacts"] += 1
        batch_counts[row["batch"]][row["classification"]] += 1
        if row["rollback_performed"]:
            batch_counts[row["batch"]]["rollbacks"] += 1

    summary = {
        "schema_version": GENERATED_SCHEMA_VERSION,
        "source_run_dir": _relative(run_dir),
        "totals": {
            "attempt_artifacts": len(attempts),
            "evaluated_attempts": sum(row["has_evaluation"] for row in attempts),
            "evaluation_successes": sum(
                row["has_evaluation"] and row["evaluation_success"] for row in attempts
            ),
            "evaluation_failures": sum(
                row["has_evaluation"] and not row["evaluation_success"]
                for row in attempts
            ),
            "trusted_accepted_actions": class_counts["trusted_accepted"],
            "retired_branch_acceptances": class_counts["retired_branch_accepted"],
            "rejected_physical_rollouts": class_counts["rejected_physical_rollout"],
            "rejected_before_physical_dispatch": class_counts[
                "rejected_before_physical_dispatch"
            ],
            "interrupted_or_incomplete": class_counts["interrupted_or_incomplete"],
            "rollbacks": sum(row["rollback_performed"] for row in attempts),
            "active_physical_command_s": sum(
                float(row["active_physical_command_s"]) for row in attempts
            ),
        },
        "classification_counts": dict(sorted(class_counts.items())),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "batch_counts": {
            batch: dict(sorted(counts.items()))
            for batch, counts in sorted(batch_counts.items())
        },
        "attempts": attempts,
    }
    _write_json(output_dir / "attempt_summary.json", summary)
    return summary


def _usage_key(usage: dict[str, Any]) -> str:
    payload = {
        "model_name": usage.get("model_name", ""),
        "kind": usage.get("kind", ""),
        "started_at_unix": usage.get("started_at_unix"),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "image_path": usage.get("image_path", ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _role_call_key(call: dict[str, Any], *, index_path: Path, position: int) -> str:
    artifact_path = str(call.get("artifact_path", "")).strip()
    if artifact_path:
        material = (
            f"{artifact_path}|{call.get('response_index', 0)}|"
            f"{call.get('role', '')}|{call.get('context_sha256', '')}"
        )
    else:
        material = f"{index_path.resolve()}|{position}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _compact_model_result(call: dict[str, Any]) -> dict[str, Any]:
    parsed = call.get("parsed_response", {})
    if not isinstance(parsed, dict):
        parsed = {}
    role = str(call.get("role", ""))
    failure_analysis = parsed.get("failure_analysis", {})
    if not isinstance(failure_analysis, dict):
        failure_analysis = {}
    selected: dict[str, Any] = {}
    for key in (
        "summary",
        "status",
        "action",
        "decision",
        "reason",
        "rationale",
        "plan",
        "task_goal",
        "current_phase_id",
        "current_phase_status",
        "required_change",
        "repair_level",
        "mutation_level",
        "strategy_pair",
        "motion_objective",
        "coordination_mode",
        "script_id",
        "robot_id",
        "success",
        "failure_type",
        "confidence",
        "promote",
        "strategy_name",
        "memory_kind",
        "spec_candidate",
        "failure_hypothesis",
        "learned_rule",
    ):
        if key in parsed:
            selected[key] = parsed[key]

    global_plan = parsed.get("global_plan", ())
    if isinstance(global_plan, list):
        selected["global_plan"] = [
            {
                key: phase[key]
                for key in (
                    "phase_id",
                    "abstract_goal",
                    "status",
                    "completion_evidence",
                )
                if key in phase
            }
            for phase in global_plan
            if isinstance(phase, dict)
        ]

    current_subtask = parsed.get("current_subtask", {})
    if isinstance(current_subtask, dict) and current_subtask:
        selected["current_subtask"] = {
            key: current_subtask[key]
            for key in (
                "subtask_id",
                "objective",
                "guidance",
                "strategy",
                "team_assignment",
                "completion_condition",
                "preserve",
            )
            if key in current_subtask
        }

    for key in (
        "constraints",
        "evidence",
        "preserve",
        "mutate",
        "forbidden",
        "success_test",
        "inheritance",
        "applicability",
        "parent_strategy",
        "child_strategy",
        "avoid",
        "evidence_refs",
        "changed_dimensions",
        "conflicts",
    ):
        if key in parsed:
            selected[key] = parsed[key]

    robot_scripts = parsed.get("robot_scripts", {})
    if isinstance(robot_scripts, dict):
        robot_actions: dict[str, Any] = {}
        for robot_id, script in robot_scripts.items():
            if not isinstance(script, dict):
                continue
            method, parameters = _api_call(str(script.get("code", "")))
            robot_actions[str(robot_id)] = {
                "method": method,
                "parameters": parameters,
                "coordination_mode": script.get("coordination_mode", ""),
            }
        if robot_actions:
            selected["robot_actions"] = robot_actions

    generated_code = parsed.get("code", "")
    if str(generated_code).strip():
        method, parameters = _api_call(str(generated_code))
        selected["generated_action"] = {
            "method": method,
            "parameters": parameters,
        }

    repaired_code = parsed.get("repaired_code", {})
    if role == "code_repair" and isinstance(repaired_code, dict):
        repaired_actions: dict[str, Any] = {}
        for robot_id, code in repaired_code.items():
            if not str(code).strip():
                continue
            method, parameters = _api_call(str(code))
            repaired_actions[str(robot_id)] = {
                "method": method,
                "parameters": parameters,
            }
        if repaired_actions:
            selected["repaired_actions"] = repaired_actions

    if failure_analysis:
        selected["failure_analysis"] = {
            key: failure_analysis[key]
            for key in (
                "outcome",
                "primary_failure",
                "causal_chain",
                "physical_evidence",
                "visual_evidence",
                "strategy_defect",
                "coordination_defect",
                "magnitude_defect",
                "reusable_context",
                "confidence",
            )
            if key in failure_analysis
        }
    return selected


def _attempt_from_model_call(call: dict[str, Any]) -> int | None:
    for value in (
        call.get("artifact_id", ""),
        call.get("artifact_path", ""),
        call.get("image_path", ""),
    ):
        match = ATTEMPT_TOKEN_RE.search(str(value))
        if match:
            return int(match.group(1))
    return None


def _summarize_model_calls(
    *,
    run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    role_calls_by_key: dict[str, dict[str, Any]] = {}
    invocations_by_key: dict[str, dict[str, Any]] = {}

    index_paths = sorted(run_dir.rglob("model_roles/role_call_index.json"))
    for index_path in index_paths:
        index = _read_json(index_path, {})
        calls = index.get("calls", []) if isinstance(index, dict) else []
        if not isinstance(calls, list):
            continue
        batch_dir = index_path.parent.parent
        for position, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            key = _role_call_key(call, index_path=index_path, position=position)
            if key in role_calls_by_key:
                continue
            usage = call.get("usage", {})
            usage = usage if isinstance(usage, dict) else {}
            row = {
                "call_key": key,
                "role": str(call.get("role", "")),
                "model_name": str(call.get("model_name", index.get("model_name", ""))),
                "batch": _relative(batch_dir, run_dir),
                "attempt": _attempt_from_model_call(call),
                "artifact_id": str(call.get("artifact_id", "")),
                "artifact_path": str(call.get("artifact_path", "")),
                "response_index": call.get("response_index", 0),
                "uses_image": bool(call.get("uses_image", False)),
                "image_path": str(call.get("image_path", "")),
                "valid": bool(call.get("valid", False)),
                "validation_error": str(call.get("validation_error", "")),
                "latency_s": float(call.get("latency_s", usage.get("latency_s", 0.0)) or 0.0),
                "usage": usage,
                "result": _compact_model_result(call),
                "source_index": _relative(index_path),
            }
            role_calls_by_key[key] = row
            if usage:
                invocation_key = _usage_key(usage)
                invocation = invocations_by_key.setdefault(
                    invocation_key,
                    {
                        **usage,
                        "invocation_key": invocation_key,
                        "roles": [],
                        "sources": [],
                    },
                )
                role = str(call.get("role", ""))
                if role and role not in invocation["roles"]:
                    invocation["roles"].append(role)
                source = str(call.get("artifact_path", "") or index_path)
                if source not in invocation["sources"]:
                    invocation["sources"].append(source)

    usage_paths = sorted(run_dir.rglob("model_usage.json"))
    for usage_path in usage_paths:
        payload = _read_json(usage_path, {})
        for channel in ("text", "vision"):
            channel_payload = payload.get(channel, {}) if isinstance(payload, dict) else {}
            calls = (
                channel_payload.get("calls", [])
                if isinstance(channel_payload, dict)
                else []
            )
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                invocation_key = _usage_key(call)
                invocation = invocations_by_key.setdefault(
                    invocation_key,
                    {
                        **call,
                        "invocation_key": invocation_key,
                        "roles": [],
                        "sources": [],
                    },
                )
                source = f"{_relative(usage_path)}::{channel}"
                if source not in invocation["sources"]:
                    invocation["sources"].append(source)

    role_calls = sorted(
        role_calls_by_key.values(),
        key=lambda row: (
            float(row.get("usage", {}).get("started_at_unix", 0.0) or 0.0),
            row["source_index"],
            int(row.get("response_index", 0) or 0),
        ),
    )
    invocations = sorted(
        invocations_by_key.values(),
        key=lambda row: float(row.get("started_at_unix", 0.0) or 0.0),
    )

    by_role: dict[str, dict[str, Any]] = {}
    for role in sorted({row["role"] for row in role_calls}):
        rows = [row for row in role_calls if row["role"] == role]
        by_role[role] = {
            "call_count": len(rows),
            "valid_responses": sum(row["valid"] for row in rows),
            "invalid_responses": sum(not row["valid"] for row in rows),
            "input_tokens": sum(
                int(row["usage"].get("input_tokens", 0) or 0) for row in rows
            ),
            "output_tokens": sum(
                int(row["usage"].get("output_tokens", 0) or 0) for row in rows
            ),
            "latency_s": sum(float(row["latency_s"]) for row in rows),
        }
    by_model: dict[str, dict[str, Any]] = {}
    for model in sorted(
        {str(row.get("model_name", "")) for row in invocations}
    ):
        rows = [row for row in invocations if str(row.get("model_name", "")) == model]
        by_model[model] = {
            "inference_count": len(rows),
            "input_tokens": sum(int(row.get("input_tokens", 0) or 0) for row in rows),
            "output_tokens": sum(int(row.get("output_tokens", 0) or 0) for row in rows),
            "latency_s": sum(float(row.get("latency_s", 0.0) or 0.0) for row in rows),
            "network_retries": sum(
                int(row.get("network_retries", 0) or 0) for row in rows
            ),
        }
    by_batch: dict[str, Counter[str]] = defaultdict(Counter)
    for row in role_calls:
        by_batch[row["batch"]]["role_calls"] += 1
        by_batch[row["batch"]]["valid_responses"] += int(row["valid"])
        by_batch[row["batch"]]["invalid_responses"] += int(not row["valid"])

    by_attempt_rows: list[dict[str, Any]] = []
    grouped_by_attempt: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    unattributed_rows: list[dict[str, Any]] = []
    for row in role_calls:
        attempt = row.get("attempt")
        if attempt is None:
            unattributed_rows.append(row)
            continue
        grouped_by_attempt[(str(row["batch"]), int(attempt))].append(row)
    for (batch, attempt), rows in sorted(
        grouped_by_attempt.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        role_counts = Counter(str(row["role"]) for row in rows)
        role_valid = Counter(
            str(row["role"]) for row in rows if bool(row["valid"])
        )
        role_invalid = Counter(
            str(row["role"]) for row in rows if not bool(row["valid"])
        )
        latest_valid_results: dict[str, Any] = {}
        validation_errors = Counter()
        for row in rows:
            role = str(row["role"])
            if row["valid"] and row["result"]:
                latest_valid_results[role] = row["result"]
            if not row["valid"] and row["validation_error"]:
                validation_errors[str(row["validation_error"])] += 1
        call_starts = [
            float(row.get("usage", {}).get("started_at_unix", 0.0) or 0.0)
            for row in rows
            if row.get("usage", {}).get("started_at_unix")
        ]
        call_finishes = [
            float(row.get("usage", {}).get("started_at_unix", 0.0) or 0.0)
            + float(row.get("latency_s", 0.0) or 0.0)
            for row in rows
            if row.get("usage", {}).get("started_at_unix")
        ]
        by_attempt_rows.append(
            {
                "batch": batch,
                "attempt": attempt,
                "role_calls": len(rows),
                "valid_responses": sum(bool(row["valid"]) for row in rows),
                "invalid_responses": sum(not bool(row["valid"]) for row in rows),
                "input_tokens": sum(
                    int(row["usage"].get("input_tokens", 0) or 0)
                    for row in rows
                ),
                "output_tokens": sum(
                    int(row["usage"].get("output_tokens", 0) or 0)
                    for row in rows
                ),
                "latency_s": sum(float(row["latency_s"]) for row in rows),
                "first_call_unix": min(call_starts) if call_starts else None,
                "last_call_finished_unix": (
                    max(call_finishes) if call_finishes else None
                ),
                "role_counts": dict(sorted(role_counts.items())),
                "role_valid_responses": dict(sorted(role_valid.items())),
                "role_invalid_responses": dict(sorted(role_invalid.items())),
                "validation_error_counts": dict(
                    sorted(validation_errors.items())
                ),
                "latest_valid_results": latest_valid_results,
            }
        )

    starts = [
        float(row.get("started_at_unix", 0.0))
        for row in invocations
        if row.get("started_at_unix")
    ]
    finishes = [
        float(row.get("started_at_unix", 0.0))
        + float(row.get("latency_s", 0.0) or 0.0)
        for row in invocations
        if row.get("started_at_unix")
    ]
    summary = {
        "schema_version": GENERATED_SCHEMA_VERSION,
        "source_run_dir": _relative(run_dir),
        "source_role_indices": [_relative(path) for path in index_paths],
        "source_usage_files": [_relative(path) for path in usage_paths],
        "totals": {
            "role_decision_calls": len(role_calls),
            "valid_structured_responses": sum(row["valid"] for row in role_calls),
            "invalid_structured_responses": sum(not row["valid"] for row in role_calls),
            "unique_model_inferences": len(invocations),
            "input_tokens": sum(
                int(row.get("input_tokens", 0) or 0) for row in invocations
            ),
            "output_tokens": sum(
                int(row.get("output_tokens", 0) or 0) for row in invocations
            ),
            "model_latency_s": sum(
                float(row.get("latency_s", 0.0) or 0.0) for row in invocations
            ),
            "network_retries": sum(
                int(row.get("network_retries", 0) or 0) for row in invocations
            ),
            "attempt_attributed_role_calls": sum(
                row.get("attempt") is not None for row in role_calls
            ),
            "unattributed_role_calls": len(unattributed_rows),
            "first_inference_unix": min(starts) if starts else None,
            "last_inference_finished_unix": max(finishes) if finishes else None,
            "wall_clock_span_s": (
                max(finishes) - min(starts) if starts and finishes else 0.0
            ),
        },
        "by_role": by_role,
        "by_model": by_model,
        "by_batch": {
            batch: dict(sorted(counts.items()))
            for batch, counts in sorted(by_batch.items())
        },
        "by_attempt": by_attempt_rows,
        "unattributed": {
            "role_calls": len(unattributed_rows),
            "role_counts": dict(
                sorted(Counter(str(row["role"]) for row in unattributed_rows).items())
            ),
            "artifact_ids": dict(
                sorted(
                    Counter(
                        str(row["artifact_id"]) for row in unattributed_rows
                    ).items()
                )
            ),
            "note": (
                "These calls are batch-start initial observations without an attempt "
                "number; they are real model inferences and remain included in totals."
            ),
        },
    }
    _write_json(output_dir / "model_call_summary.json", summary)
    _write_jsonl(output_dir / "model_calls.jsonl", role_calls)
    _write_jsonl(output_dir / "model_invocations.jsonl", invocations)
    return summary


def _format_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _format_time(value: object) -> str:
    try:
        return datetime.fromtimestamp(float(value)).astimezone().isoformat(
            timespec="seconds"
        )
    except (TypeError, ValueError, OSError):
        return ""


def _format_duration(value: object) -> str:
    try:
        total = max(0.0, float(value))
    except (TypeError, ValueError):
        return ""
    hours, remainder = divmod(total, 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    parts: list[str] = []
    if hours >= 1:
        parts.append(f"{int(hours)}h")
    if minutes >= 1 or hours >= 1:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.1f}s")
    return " ".join(parts)


def _action_label(action: dict[str, Any]) -> str:
    method = str(action.get("method", ""))
    parameters = action.get("parameters", {})
    if not isinstance(parameters, dict) or not parameters:
        return method
    rendered = ", ".join(
        f"{key}={value!r}" for key, value in parameters.items()
    )
    return f"{method}({rendered})"


def _markdown_digest(value: object, *, limit: int = 900) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    text = re.sub(r"\s+", " ", text).replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _write_reports(
    *,
    run_dir: Path,
    output_dir: Path,
    trajectory: dict[str, Any],
    attempts: dict[str, Any],
    models: dict[str, Any],
) -> None:
    action_counts = Counter(
        step.get("actions", {}).get("front", {}).get("method", "")
        for step in trajectory["steps"]
    )
    action_counts.pop("", None)
    phase_counts = Counter(
        str(
            step.get("robot_scripts", {})
            .get("front", {})
            .get("phase_id", "")
        )
        for step in trajectory["steps"]
    )
    phase_counts.pop("", None)
    trajectory_lines = [
        "# Trusted Successful Trajectory",
        "",
        (
            f"Source: `{_relative(run_dir)}`. The trusted manifest contains "
            f"**{trajectory['accepted_action_count']} accepted actions**, from "
            f"`{trajectory['initial_checkpoint']}` to `{trajectory['latest_checkpoint']}`."
        ),
        "",
        "## Composition",
        "",
        "| Action | Accepted transactions |",
        "|---|---:|",
    ]
    trajectory_lines.extend(
        f"| `{action}` | {count} |"
        for action, count in action_counts.most_common()
    )
    trajectory_lines.extend(
        [
            "",
            "| Phase | Accepted transactions |",
            "|---|---:|",
        ]
    )
    trajectory_lines.extend(
        f"| `{phase}` | {count} |"
        for phase, count in sorted(phase_counts.items())
    )
    trajectory_lines.extend(
        [
        "",
        "## Ordered Actions",
        "",
        "Checkpoint numbering gaps are intentional branch rewinds. Replay uses this table order.",
        "",
        "| # | Attempt | Checkpoint | Mode | Front action | Back action | Payload step (m) | Target progress (m) |",
        "|---:|---:|---|---|---|---|---:|---:|",
        ]
    )
    for step in trajectory["steps"]:
        front = step["actions"]["front"]
        back = step["actions"]["back"]
        metrics = step.get("accepted_evaluation", {}).get("metrics", {})
        route_metric_applicable = front["method"] != "place_payload_on_arms"
        trajectory_lines.append(
            "| {index} | {attempt} | `{before}` -> `{after}` | {mode} | "
            "`{front}` | `{back}` | {step_m} | {progress_m} |".format(
                index=step["sequence_index"],
                attempt=step["attempt"],
                before=step["checkpoint_before"],
                after=step["checkpoint_after"],
                mode=step["coordination_mode"],
                front=_action_label(front),
                back=_action_label(back),
                step_m=(
                    _format_float(metrics.get("payload_step_displacement_xy_m"))
                    if route_metric_applicable
                    else "n/a"
                ),
                progress_m=(
                    _format_float(metrics.get("payload_target_progress_m"))
                    if route_metric_applicable
                    else "n/a"
                ),
            )
        )
    trajectory_lines.extend(
        [
            "",
            (
                "`place_payload_on_arms` route metrics are shown as `n/a`: its raw "
                "displacement compares the off-scene deferred staging pose with the live "
                "support pose and is not transport distance. Raw values remain in the "
                "accepted evaluation JSON."
            ),
        ]
    )
    (output_dir / "TRAJECTORY.md").write_text(
        "\n".join(trajectory_lines) + "\n",
        encoding="utf-8",
    )

    model_attempts = {
        (str(row["batch"]), int(row["attempt"])): row
        for row in models.get("by_attempt", ())
        if isinstance(row, dict)
    }
    attempt_lines = [
        "# Attempt and Rollback Ledger",
        "",
        (
            "The artifact number and logical attempt can differ where a resumed runner "
            "wrote an attempt into an older directory name; both are retained here."
        ),
        "",
        "| Artifact | Attempt | Class | Action | Result | Rollback | Physical time | Model calls (valid/invalid) | Model latency | Checkpoint |",
        "|---|---:|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in attempts["attempts"]:
        front_action = row.get("actions", {}).get("front", {}).get("method", "")
        result = (
            "accepted"
            if row["evaluation_success"]
            else row["failure_type"] or row["trace_status"] or "incomplete"
        )
        checkpoint = (
            f"{row['checkpoint_before']} -> {row['checkpoint_after']}"
            if row["checkpoint_before"] or row["checkpoint_after"]
            else ""
        )
        model_stats = model_attempts.get((row["batch"], int(row["attempt"])), {})
        model_calls = int(model_stats.get("role_calls", 0) or 0)
        valid = int(model_stats.get("valid_responses", 0) or 0)
        invalid = int(model_stats.get("invalid_responses", 0) or 0)
        attempt_lines.append(
            f"| `{row['artifact_dir']}` | {row['attempt']} | "
            f"`{row['classification']}` | `{front_action}` | `{result}` | "
            f"{'yes' if row['rollback_performed'] else 'no'} | "
            f"{row['active_physical_command_s']:.1f}s | "
            f"{model_calls} ({valid}/{invalid}) | "
            f"{_format_duration(model_stats.get('latency_s', 0.0))} | "
            f"`{checkpoint}` |"
        )
    (output_dir / "ATTEMPTS.md").write_text(
        "\n".join(attempt_lines) + "\n",
        encoding="utf-8",
    )

    model_lines = [
        "# Model Call Summary",
        "",
        (
            f"Unique model inferences: **{models['totals']['unique_model_inferences']}**; "
            f"role decisions: **{models['totals']['role_decision_calls']}**; "
            f"valid structured responses: **{models['totals']['valid_structured_responses']}**; "
            f"invalid structured responses: **{models['totals']['invalid_structured_responses']}**."
        ),
        (
            f"Attempt-attributed role calls: "
            f"**{models['totals']['attempt_attributed_role_calls']}**; "
            f"batch-start initial-observation calls without an attempt number: "
            f"**{models['totals']['unattributed_role_calls']}**."
        ),
        (
            f"Summed inference latency: "
            f"**{_format_duration(models['totals']['model_latency_s'])}**; "
            f"first-to-last wall-clock span including pauses and resumes: "
            f"**{_format_duration(models['totals']['wall_clock_span_s'])}**."
        ),
        "",
        "| Role | Calls | Valid | Invalid | Input tokens | Output tokens | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role, stats in models["by_role"].items():
        model_lines.append(
            f"| `{role}` | {stats['call_count']} | {stats['valid_responses']} | "
            f"{stats['invalid_responses']} | {stats['input_tokens']} | "
            f"{stats['output_tokens']} | {stats['latency_s']:.1f} |"
        )
    model_lines.extend(
        [
            "",
            "## Models",
            "",
            "| Model | Inferences | Input tokens | Output tokens | Latency | Network retries |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, stats in models["by_model"].items():
        model_lines.append(
            f"| `{model}` | {stats['inference_count']} | "
            f"{stats['input_tokens']} | {stats['output_tokens']} | "
            f"{_format_duration(stats['latency_s'])} | "
            f"{stats['network_retries']} |"
        )
    model_lines.extend(
        [
            "",
            "The detailed index is `model_calls.jsonl`; it omits full prompts and raw responses,",
            "but preserves each role, result summary, validation error, usage, and source artifact.",
            "The original role artifacts remain in their source run directories.",
            "",
            "`MODEL_DECISIONS.md` gives the latest valid semantic result for every role in",
            "each attributed attempt. `model_attempt_summary.json` provides the same data",
            "without Markdown truncation.",
        ]
    )
    (output_dir / "MODEL_CALLS.md").write_text(
        "\n".join(model_lines) + "\n",
        encoding="utf-8",
    )

    decision_lines = [
        "# Per-Attempt Model Decisions",
        "",
        (
            "Each row aggregates all calls for one model role in one logical attempt. "
            "The digest is the latest valid structured result; invalid retries and every "
            "individual response remain indexed in `model_calls.jsonl`."
        ),
        "",
        "| Batch / Attempt | Role | Calls | Valid | Invalid | Latest valid result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for aggregate in models.get("by_attempt", ()):
        if not isinstance(aggregate, dict):
            continue
        roles = aggregate.get("role_counts", {})
        valid_roles = aggregate.get("role_valid_responses", {})
        invalid_roles = aggregate.get("role_invalid_responses", {})
        results = aggregate.get("latest_valid_results", {})
        for role in sorted(roles):
            decision_lines.append(
                f"| `{aggregate['batch']} / {aggregate['attempt']}` | `{role}` | "
                f"{roles.get(role, 0)} | {valid_roles.get(role, 0)} | "
                f"{invalid_roles.get(role, 0)} | "
                f"{_markdown_digest(results.get(role, {}))} |"
            )
    decision_lines.extend(
        [
            "",
            "## Calls Without Attempt Number",
            "",
            (
                f"There are **{models['unattributed']['role_calls']}** batch-start "
                "initial-observation calls. They are included in all global token and "
                "latency totals but cannot honestly be assigned to a physical attempt."
            ),
            "",
            _markdown_digest(models["unattributed"], limit=5000),
        ]
    )
    (output_dir / "MODEL_DECISIONS.md").write_text(
        "\n".join(decision_lines) + "\n",
        encoding="utf-8",
    )
    _write_json(
        output_dir / "model_attempt_summary.json",
        {
            "schema_version": GENERATED_SCHEMA_VERSION,
            "source_run_dir": _relative(run_dir),
            "attempts": models.get("by_attempt", []),
            "unattributed": models.get("unattributed", {}),
        },
    )

    totals = attempts["totals"]
    model_totals = models["totals"]
    readme_lines = [
        "# Accepted Episode Package",
        "",
        "This directory is generated from the accepted episode manifest. It separates the",
        "trusted main trajectory from accepted-but-retired branches and rejected attempts.",
        "",
        "## Trusted result",
        "",
        f"- Accepted main-path actions: **{trajectory['accepted_action_count']}**",
        f"- Initial checkpoint: `{trajectory['initial_checkpoint']}`",
        f"- Final checkpoint: `{trajectory['latest_checkpoint']}`",
        "- Final task result: payload on the target, both robots clear and upright",
        "",
        "## Experiment accounting",
        "",
        f"- Attempt artifacts: **{totals['attempt_artifacts']}**",
        f"- Evaluated attempts: **{totals['evaluated_attempts']}**",
        f"- Evaluator successes: **{totals['evaluation_successes']}**",
        f"- Evaluator failures: **{totals['evaluation_failures']}**",
        f"- Trusted accepted actions: **{totals['trusted_accepted_actions']}**",
        f"- Accepted actions later retired by branch rewind: **{totals['retired_branch_acceptances']}**",
        f"- Rejected physical rollouts: **{totals['rejected_physical_rollouts']}**",
        f"- Rejected before physical dispatch: **{totals['rejected_before_physical_dispatch']}**",
        f"- Interrupted/incomplete attempt directories: **{totals['interrupted_or_incomplete']}**",
        f"- Recorded rollbacks: **{totals['rollbacks']}**",
        f"- Traced active physical command time: **{_format_duration(totals['active_physical_command_s'])}**",
        f"- Unique model inferences: **{model_totals['unique_model_inferences']}**",
        f"- Total model input/output tokens: **{model_totals['input_tokens']} / {model_totals['output_tokens']}**",
        f"- Summed model latency: **{_format_duration(model_totals['model_latency_s'])}**",
        f"- Inference wall-clock span including pauses/resumes: **{_format_duration(model_totals['wall_clock_span_s'])}**",
        f"- First/last inference: `{_format_time(model_totals['first_inference_unix'])}` / "
        f"`{_format_time(model_totals['last_inference_finished_unix'])}`",
        "",
        "## Contents",
        "",
        "- `TRAJECTORY.md`: ordered trusted trajectory",
        "- `steps/`: exact front/back scripts and accepted evidence for each trusted action",
        "- `visuals/`: ordered accepted global-view images when available",
        "- `ATTEMPTS.md` and `attempt_summary.json`: every discovered attempt and rollback",
        "- `MODEL_CALLS.md`, `model_call_summary.json`, and `model_calls.jsonl`: model accounting",
        "- `MODEL_DECISIONS.md` and `model_attempt_summary.json`: per-attempt Agent decisions",
        "- `trajectory_manifest.json`: machine-readable replay source",
        "- `replay_successful_trajectory.sh`: model-free physical replay command",
        "- `REPLAY_RESULTS.md`: outcomes from model-free reproducibility trials",
        "",
        "## Replay",
        "",
        "Keep the original Term1 and both SONIC deploy backends running, then execute:",
        "",
        "```bash",
        f"cd {REPO_ROOT}",
        f"bash {_relative(output_dir / 'replay_successful_trajectory.sh')}",
        "```",
        "",
        "Replay restores `accepted_step_00` and executes the manifest in order. By default,",
        "it round-trips each accepted replay checkpoint to reset both SONIC planners, matching",
        "the evolution checkpoint boundary. Pass `--no-planner-reset-each-step` to retain",
        "continuous planner history. It writes only new `replay_*` checkpoints; a hard physical",
        "failure restores the latest replay checkpoint and stops.",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )

    replay_script = output_dir / "replay_successful_trajectory.sh"
    replay_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {REPO_ROOT}\n"
        'PY="${COEVOLVE_PYTHON:-python3}"\n'
        'if [[ ! -x "$PY" ]]; then PY=python3; fi\n'
        "PYTHONPATH=src \"$PY\" scripts/manage_accepted_episode.py replay "
        f"--run-dir {_relative(run_dir)} --execute \"$@\"\n",
        encoding="utf-8",
    )
    replay_script.chmod(0o755)


def _summarize_replays(output_dir: Path) -> dict[str, Any]:
    replay_root = output_dir / "replays"
    runs: list[dict[str, Any]] = []
    if replay_root.is_dir():
        for replay_dir in sorted(replay_root.glob("replay_*")):
            summary_path = replay_dir / "replay_summary.json"
            if not summary_path.is_file():
                continue
            summary = _read_json(summary_path, {})
            config = _read_json(replay_dir / "replay_config.json", {})
            results = summary.get("results", [])
            results = results if isinstance(results, list) else []
            successful = [
                row
                for row in results
                if isinstance(row, dict) and bool(row.get("physical_success", False))
            ]
            failed = next(
                (
                    row
                    for row in reversed(results)
                    if isinstance(row, dict)
                    and not bool(row.get("physical_success", False))
                ),
                {},
            )
            deviations = [
                float(value)
                for row in successful
                if isinstance(row, dict)
                for value in (
                    row.get("accepted_endpoint_deviation", {}).get(
                        "payload_xy_error_m"
                    ),
                )
                if value is not None
            ]
            verifier = failed.get("current_verifier", {})
            verifier = verifier if isinstance(verifier, dict) else {}
            metrics = verifier.get("metrics", {})
            metrics = metrics if isinstance(metrics, dict) else {}
            trace_error = str(failed.get("trace_error", "")).strip()
            runs.append(
                {
                    "replay_dir": _relative(replay_dir),
                    "status": str(summary.get("status", "")),
                    "planner_reset_each_step": bool(
                        config.get("planner_reset_each_step", False)
                    ),
                    "steps_requested": int(config.get("steps_requested", 0) or 0),
                    "physically_safe_steps": len(successful),
                    "completed_steps": int(summary.get("completed_steps", 0) or 0),
                    "failed_step": summary.get("failed_step"),
                    "failed_action": str(failed.get("action", "")),
                    "failure_type": str(verifier.get("failure_type", "")),
                    "hard_failure_reasons": list(
                        failed.get("hard_physical_gate", {}).get("reasons", ())
                    ),
                    "trace_error_first_line": (
                        trace_error.splitlines()[0] if trace_error else ""
                    ),
                    "payload_target_xy_error_m": metrics.get(
                        "payload_target_xy_error_m"
                    ),
                    "max_accepted_endpoint_xy_error_m": (
                        max(deviations) if deviations else None
                    ),
                    "last_accepted_endpoint_xy_error_m": (
                        deviations[-1] if deviations else None
                    ),
                    "latest_replay_checkpoint": str(
                        summary.get("latest_replay_checkpoint", "")
                    ),
                    "rollback_performed": bool(
                        failed.get("rollback_performed", False)
                    ),
                    "elapsed_s": float(summary.get("elapsed_s", 0.0) or 0.0),
                    "saved_visual_count": len(
                        list((replay_dir / "visuals").glob("*.png"))
                    ),
                }
            )

    aggregate = {
        "schema_version": GENERATED_SCHEMA_VERSION,
        "trial_count": len(runs),
        "full_replay_successes": sum(row["status"] == "completed" for row in runs),
        "trials": runs,
    }
    _write_json(output_dir / "replay_results.json", aggregate)

    lines = [
        "# Model-Free Replay Results",
        "",
        "These trials execute the trusted manifest without any model calls. They use new",
        "`replay_*` checkpoints and never overwrite the accepted checkpoint lineage.",
        "",
        "| Trial | Planner reset each step | Safe steps | Failed step/action | Failure | Target XY error (m) | Max endpoint divergence (m) | Rollback | Time (s) |",
        "|---|---|---:|---|---|---:|---:|---|---:|",
    ]
    for row in runs:
        failed = (
            f"{row['failed_step']} / `{row['failed_action']}`"
            if row["failed_step"] is not None
            else "none"
        )
        failure = row["failure_type"] or row["status"]
        lines.append(
            f"| `{Path(row['replay_dir']).name}` | "
            f"{'yes' if row['planner_reset_each_step'] else 'no'} | "
            f"{row['physically_safe_steps']} | {failed} | `{failure}` | "
            f"{_format_float(row['payload_target_xy_error_m'])} | "
            f"{_format_float(row['max_accepted_endpoint_xy_error_m'])} | "
            f"{'yes' if row['rollback_performed'] else 'no'} | "
            f"{row['elapsed_s']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Trial Details",
            "",
        ]
    )
    for row in runs:
        lines.extend(
            [
                f"### `{Path(row['replay_dir']).name}`",
                "",
                (
                    f"- Safe prefix: **{row['physically_safe_steps']}** of "
                    f"**{row['steps_requested']}** requested actions"
                ),
                (
                    f"- Stop: step **{row['failed_step']}**, "
                    f"`{row['failed_action']}`, "
                    f"`{row['failure_type'] or row['status']}`"
                ),
                (
                    "- Hard gate: "
                    + (
                        ", ".join(f"`{reason}`" for reason in row["hard_failure_reasons"])
                        if row["hard_failure_reasons"]
                        else "none"
                    )
                ),
                (
                    f"- Last safe replay checkpoint: "
                    f"`{row['latest_replay_checkpoint']}`"
                ),
                (
                    f"- Rollback performed: "
                    f"{'yes' if row['rollback_performed'] else 'no'}"
                ),
            ]
        )
        if row["trace_error_first_line"]:
            lines.append(
                f"- Dispatch detail: `{row['trace_error_first_line']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "A trusted accepted path proves that every child action passed the verifier "
                "from its recorded parent checkpoint. It does not by itself prove that the "
                "same open-loop action list is deterministic under fresh SONIC gait phases."
            ),
            (
                "The per-step-reset trial matches the original checkpoint boundary more "
                "closely. Comparing it with the continuous trial isolates planner-history "
                "effects from remaining policy and physics nondeterminism."
            ),
        ]
    )
    (output_dir / "REPLAY_RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return aggregate


def export_episode(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(_manifest_path(run_dir), {})
    trajectory = _export_trajectory(
        run_dir=run_dir,
        output_dir=output_dir,
        manifest=manifest,
    )
    attempts = _summarize_attempts(
        run_dir=run_dir,
        output_dir=output_dir,
        manifest=manifest,
    )
    models = _summarize_model_calls(run_dir=run_dir, output_dir=output_dir)
    _write_reports(
        run_dir=run_dir,
        output_dir=output_dir,
        trajectory=trajectory,
        attempts=attempts,
        models=models,
    )
    replay_results = _summarize_replays(output_dir)
    package = {
        "output_dir": str(output_dir),
        "accepted_actions": trajectory["accepted_action_count"],
        "attempt_totals": attempts["totals"],
        "model_totals": models["totals"],
        "replay_totals": {
            "trial_count": replay_results["trial_count"],
            "full_replay_successes": replay_results["full_replay_successes"],
        },
    }
    _write_json(output_dir / "package_summary.json", package)
    return package


def _load_replay_imports() -> dict[str, Any]:
    source_root = REPO_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from praxis_coop.cap_policy.backends import StaticSonicSkillBackend
    from praxis_coop.cap_policy.checkpoints import SimulatorCheckpointClient
    from praxis_coop.cap_policy.contracts import (
        BoxCarryTask,
        PolicyScript,
        RobotPolicyScript,
    )
    from praxis_coop.cap_policy.evaluator import BoxCarryEvaluator
    from praxis_coop.cap_policy.policy_executor import (
        PolicyExecutor,
        validate_team_policy_scripts,
    )
    from praxis_coop.cap_policy.skill_api import G1SkillAPI
    from praxis_coop.cap_policy.visual_observation import ExternalCameraObserver

    return {
        "StaticSonicSkillBackend": StaticSonicSkillBackend,
        "SimulatorCheckpointClient": SimulatorCheckpointClient,
        "BoxCarryTask": BoxCarryTask,
        "PolicyScript": PolicyScript,
        "RobotPolicyScript": RobotPolicyScript,
        "BoxCarryEvaluator": BoxCarryEvaluator,
        "PolicyExecutor": PolicyExecutor,
        "validate_team_policy_scripts": validate_team_policy_scripts,
        "G1SkillAPI": G1SkillAPI,
        "ExternalCameraObserver": ExternalCameraObserver,
    }


def _policy_from_step(
    step: dict[str, Any],
    *,
    imports: dict[str, Any],
) -> Any:
    RobotPolicyScript = imports["RobotPolicyScript"]
    PolicyScript = imports["PolicyScript"]
    scripts = []
    for robot_id in ("front", "back"):
        raw = step["robot_scripts"][robot_id]
        scripts.append(
            RobotPolicyScript(
                robot_id=robot_id,
                script_id=f"replay_{step['sequence_index']:02d}_{raw.get('script_id', robot_id)}",
                code=str(raw["code"]).rstrip() + "\n",
                synchronization_group=str(
                    raw.get("synchronization_group", "atomic_team_activation")
                ),
                phase_id=str(raw.get("phase_id", "")),
                contact_intent=str(raw.get("contact_intent", "unspecified")),
                coordination_mode=str(
                    raw.get("coordination_mode", step.get("coordination_mode", "sync"))
                ),
            )
        )
    return PolicyScript(
        script_id=f"replay_{step['sequence_index']:02d}_{step['script_id']}",
        attempt=int(step["sequence_index"]),
        code=scripts[0].code,
        robot_scripts=tuple(scripts),
        rationale=(
            f"Model-free replay of trusted attempt {step['attempt']} from "
            f"{step['checkpoint_before']} to {step['checkpoint_after']}."
        ),
        verification_contract=dict(step.get("verification_contract", {})),
    )


def _runtime_state(path: Path) -> dict[str, Any]:
    payload = _read_json(path, {})
    return payload if isinstance(payload, dict) else {}


def _physical_gate(
    state: dict[str, Any],
    *,
    require_loaded_support: bool,
    require_target_support: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    control = state.get("control_health", {})
    if not isinstance(control, dict) or not bool(control.get("healthy", False)):
        reasons.append("control_health_not_healthy")
    if str(state.get("runtime_status", "")) != "running":
        reasons.append("runtime_not_running")
    if not bool(state.get("elastic_band_dropped", False)):
        reasons.append("robots_not_dropped_into_physics")

    robots = state.get("robots", {})
    robots = robots if isinstance(robots, dict) else {}
    robot_summary: dict[str, Any] = {}
    positions: dict[str, tuple[float, float]] = {}
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        robot = robot if isinstance(robot, dict) else {}
        try:
            pelvis = float(robot.get("pelvis_height_m", robot.get("z", 0.0)))
            x = float(robot["x"])
            y = float(robot["y"])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{robot_id}_state_missing")
            continue
        robot_summary[robot_id] = {"x": x, "y": y, "pelvis_height_m": pelvis}
        positions[robot_id] = (x, y)
        if pelvis < 0.35:
            reasons.append(f"{robot_id}_fallen")
    separation = None
    if len(positions) == 2:
        separation = math.dist(positions["front"], positions["back"])
        if separation < 0.61:
            reasons.append("robot_pair_too_close")

    payload = state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    active = bool(payload.get("active", False))
    released = bool(payload.get("released", False))
    floor = bool(contacts.get("floor", False))
    walls = sorted(str(item) for item in contacts.get("walls", ()))
    support_robots = sorted(str(item) for item in contacts.get("support_robots", ()))
    target = bool(contacts.get("target", False))
    try:
        payload_z = float(payload.get("z", 0.0))
    except (TypeError, ValueError):
        payload_z = 0.0

    # The deferred payload is deliberately hidden in an off-scene staging
    # location whose floor contact is not a task failure.
    if active and floor:
        reasons.append("payload_floor_contact")
    if active and walls:
        reasons.append("payload_wall_contact")
    if require_loaded_support:
        if not active:
            reasons.append("payload_not_active")
        if released:
            reasons.append("payload_released_too_early")
        if payload_z < 0.65:
            reasons.append("payload_too_low")
        if not {"front", "back"}.issubset(set(support_robots)):
            reasons.append("dual_support_missing")
    if require_target_support:
        if not active or not released:
            reasons.append("payload_not_released")
        if not target:
            reasons.append("payload_not_on_target")

    summary = {
        "payload": {
            "active": active,
            "released": released,
            "x": payload.get("x"),
            "y": payload.get("y"),
            "z": payload_z,
            "yaw_deg": payload.get("yaw_deg"),
            "tilt_deg": payload.get("tilt_deg"),
            "support_robots": support_robots,
            "floor_contact": floor,
            "wall_contacts": walls,
            "target_contact": target,
        },
        "robots": robot_summary,
        "robot_pair_xy_separation_m": separation,
    }
    return not reasons, reasons, summary


def _wait_for_gate(
    runtime_state_file: Path,
    *,
    require_loaded_support: bool,
    require_target_support: bool,
    timeout_s: float,
    stable_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    stable_since: float | None = None
    last: dict[str, Any] = {}
    while True:
        state = _runtime_state(runtime_state_file)
        passed, reasons, summary = _physical_gate(
            state,
            require_loaded_support=require_loaded_support,
            require_target_support=require_target_support,
        )
        last = {"success": passed, "reasons": reasons, "state": summary}
        now = time.monotonic()
        if passed:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= stable_s:
                return last
        else:
            stable_since = None
        if now >= deadline:
            return {**last, "success": False, "timeout_s": timeout_s}
        time.sleep(min(0.1, max(0.01, deadline - now)))


def _wait_checkpoint_release(
    runtime_state_file: Path,
    *,
    checkpoint_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while True:
        last = _runtime_state(runtime_state_file)
        checkpoint = last.get("checkpoint", {})
        if (
            isinstance(checkpoint, dict)
            and str(checkpoint.get("checkpoint_id", "")) == checkpoint_id
            and not bool(checkpoint.get("resync_active", False))
        ):
            return last
        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(
                f"checkpoint {checkpoint_id} did not leave resync hold"
            )
        time.sleep(min(0.1, max(0.01, deadline - now)))


def _restore_with_planner_reset(
    *,
    checkpoint_client: Any,
    backend: Any,
    checkpoint_id: str,
    output_dir: Path,
    heartbeat_file: Path,
    runtime_state_file: Path,
    require_loaded_support: bool,
    require_target_support: bool,
) -> dict[str, Any]:
    first = checkpoint_client.restore(checkpoint_id)
    resync = backend.resynchronize_idle(
        output_dir=output_dir / "planner_resync",
        heartbeat_file=heartbeat_file,
        runtime_state_file=runtime_state_file,
        duration_s=1.0,
        stand_phase_id=f"replay_restore_{checkpoint_id}",
    )
    second = checkpoint_client.restore(checkpoint_id)
    released_state = _wait_checkpoint_release(
        runtime_state_file,
        checkpoint_id=checkpoint_id,
        timeout_s=10.0,
    )
    gate = _wait_for_gate(
        runtime_state_file,
        require_loaded_support=require_loaded_support,
        require_target_support=require_target_support,
        timeout_s=12.0,
        stable_s=0.5,
    )
    if not gate["success"] and (require_loaded_support or require_target_support):
        raise RuntimeError(
            f"restored checkpoint {checkpoint_id} failed physical gate: "
            f"{gate['reasons']}"
        )
    return {
        "initial_restore": asdict(first),
        "planner_resync": resync,
        "reassert_restore": asdict(second),
        "released_runtime_checkpoint": released_state.get("checkpoint", {}),
        "physical_gate": gate,
    }


def _pose_deviation(
    *,
    replay_state: dict[str, Any],
    accepted_evaluation: dict[str, Any],
) -> dict[str, Any]:
    evidence = accepted_evaluation.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    accepted_payload = evidence.get("payload", {})
    accepted_payload = accepted_payload if isinstance(accepted_payload, dict) else {}
    accepted_robots = evidence.get("robots", {})
    accepted_robots = accepted_robots if isinstance(accepted_robots, dict) else {}
    replay_payload = replay_state.get("payload", {})
    replay_payload = replay_payload if isinstance(replay_payload, dict) else {}
    replay_robots = replay_state.get("robots", {})
    replay_robots = replay_robots if isinstance(replay_robots, dict) else {}
    result: dict[str, Any] = {"available": False, "robots": {}}
    try:
        result["payload_xy_error_m"] = math.hypot(
            float(replay_payload["x"]) - float(accepted_payload["x"]),
            float(replay_payload["y"]) - float(accepted_payload["y"]),
        )
        result["payload_z_error_m"] = abs(
            float(replay_payload["z"]) - float(accepted_payload["z"])
        )
        result["available"] = True
    except (KeyError, TypeError, ValueError):
        pass
    for robot_id in ("front", "back"):
        accepted = accepted_robots.get(robot_id, {})
        replay = replay_robots.get(robot_id, {})
        if not isinstance(accepted, dict) or not isinstance(replay, dict):
            continue
        try:
            result["robots"][robot_id] = {
                "xy_error_m": math.hypot(
                    float(replay["x"]) - float(accepted["x"]),
                    float(replay["y"]) - float(accepted["y"]),
                )
            }
        except (KeyError, TypeError, ValueError):
            continue
    return result


def replay_episode(
    *,
    run_dir: Path,
    package_dir: Path,
    execute: bool,
    from_step: int,
    through_step: int,
    gate_timeout_s: float,
    planner_reset_each_step: bool,
) -> int:
    trajectory_path = package_dir / "trajectory_manifest.json"
    if not trajectory_path.is_file():
        export_episode(run_dir, package_dir)
    trajectory = _read_json(trajectory_path, {})
    steps = trajectory.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory package has no replay steps")
    if from_step <= 0:
        from_step = 1
    if through_step <= 0:
        through_step = len(steps)
    if from_step > len(steps):
        raise ValueError(
            f"--from-step={from_step} exceeds {len(steps)} accepted actions"
        )
    if through_step > len(steps):
        raise ValueError(
            f"--through-step={through_step} exceeds {len(steps)} accepted actions"
        )
    if from_step > through_step:
        raise ValueError(
            f"--from-step={from_step} must not exceed --through-step={through_step}"
        )

    selected_steps = steps[from_step - 1 : through_step]

    imports = _load_replay_imports()
    validate_team_policy_scripts = imports["validate_team_policy_scripts"]
    policies = [
        _policy_from_step(step, imports=imports)
        for step in selected_steps
    ]
    for policy in policies:
        validate_team_policy_scripts(policy.robot_scripts)

    validation = {
        "status": "validated",
        "step_count": len(policies),
        "execute": bool(execute),
        "validated_at_unix": time.time(),
    }
    _write_json(package_dir / "replay_preflight.json", validation)
    if not execute:
        print(
            f"[accepted-replay] validated {len(policies)} trusted team transactions; "
            "no physical commands dispatched"
        )
        return 0

    runtime_state_file = run_dir / "runtime_state.json"
    heartbeat_file = run_dir / "planner_heartbeat.json"
    live = _runtime_state(runtime_state_file)
    available = live.get("checkpoint", {}).get("available_ids", [])
    initial_checkpoint = str(
        selected_steps[0].get("checkpoint_before", "")
        or trajectory.get("initial_checkpoint", "accepted_step_00")
    )
    if initial_checkpoint not in available:
        raise RuntimeError(
            f"live Term1 does not hold {initial_checkpoint}; available={available}"
        )

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    replay_dir = package_dir / "replays" / f"replay_{timestamp}"
    replay_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_prefix = f"replay_{timestamp}"

    StaticSonicSkillBackend = imports["StaticSonicSkillBackend"]
    SimulatorCheckpointClient = imports["SimulatorCheckpointClient"]
    G1SkillAPI = imports["G1SkillAPI"]
    PolicyExecutor = imports["PolicyExecutor"]
    BoxCarryTask = imports["BoxCarryTask"]
    BoxCarryEvaluator = imports["BoxCarryEvaluator"]
    ExternalCameraObserver = imports["ExternalCameraObserver"]

    backend = StaticSonicSkillBackend(
        execute=True,
        default_start_pulse_seconds=0.0,
        default_stand_seconds=0.5,
        minimum_transport_window_s=1.0,
        minimum_lateral_transport_window_s=0.4,
        nominal_distance_duration_gain=1.0,
        nominal_distance_duration_baseline_m=2.0,
        minimum_transport_ramp_s=1.5,
        post_transport_idle_s=1.0,
        payload_safety_profile="task",
        team_sync_enabled=True,
        team_sync_active_catchup=True,
        team_sync_longitudinal_kp=2.0,
        team_sync_cross_track_kp=0.6,
        team_sync_payload_kp=1.2,
        team_sync_error_filter_alpha=0.35,
        team_sync_pause_leader_error_m=0.008,
        team_sync_resume_error_m=0.003,
        team_sync_lagging_boost_mps=0.06,
        team_sync_max_correction_mps=0.06,
        team_sync_min_active_speed_ratio=0.0,
        team_sync_hard_desync_m=0.50,
    )
    checkpoint_client = SimulatorCheckpointClient(
        heartbeat_file=heartbeat_file,
        runtime_state_file=runtime_state_file,
        artifact_dir=replay_dir / "checkpoint_events",
        timeout_s=10.0,
        restore_settle_s=4.0,
    )
    camera = ExternalCameraObserver(
        output_dir=replay_dir / "visuals",
        tracking_inset=True,
    )
    task = BoxCarryTask(
        task_id="two_g1_long_box_carry_manual_layout",
        target=(
            "place the long box on the cylindrical target, retreat while crouched, "
            "then recover both robots upright and clear"
        ),
        task_stage="full_scene",
        route_discovery=True,
        min_payload_z_m=0.65,
        min_robot_pelvis_height_m=0.35,
        min_robot_xy_separation_m=0.61,
        require_payload_on_target=True,
        require_robots_exit=True,
    )
    evaluator = BoxCarryEvaluator(
        runtime_state_file=runtime_state_file,
        safety_profile="task",
        require_visual=False,
    )

    _write_json(
        replay_dir / "replay_config.json",
        {
            "source_trajectory": _relative(trajectory_path),
            "from_step": from_step,
            "through_step": through_step,
            "steps_requested": len(policies),
            "checkpoint_prefix": checkpoint_prefix,
            "initial_checkpoint": initial_checkpoint,
            "planner_reset_each_step": planner_reset_each_step,
            "backend": {
                "stand_s": 0.5,
                "active_catchup": True,
                "lagging_boost_mps": 0.06,
                "hard_desync_m": 0.50,
                "safety_profile": "task",
            },
        },
    )

    print(
        f"[accepted-replay] restoring {initial_checkpoint} and resetting both planners",
        flush=True,
    )
    restore_record = _restore_with_planner_reset(
        checkpoint_client=checkpoint_client,
        backend=backend,
        checkpoint_id=initial_checkpoint,
        output_dir=replay_dir / "initial_restore",
        heartbeat_file=heartbeat_file,
        runtime_state_file=runtime_state_file,
        require_loaded_support=False,
        require_target_support=False,
    )
    _write_json(replay_dir / "initial_restore.json", restore_record)
    initial_state = _runtime_state(runtime_state_file)
    initial_gate = _physical_gate(
        initial_state,
        require_loaded_support=False,
        require_target_support=False,
    )
    if not initial_gate[0]:
        raise RuntimeError(f"initial replay state is not safe: {initial_gate[1]}")
    initial_replay_checkpoint = f"{checkpoint_prefix}_00"
    checkpoint_client.save(initial_replay_checkpoint, reset_metrics=True)
    initial_visual = camera.observe(label="replay_step_00_initial", runtime_state=initial_state)
    _write_json(replay_dir / "initial_visual.json", asdict(initial_visual))

    results: list[dict[str, Any]] = []
    latest_replay_checkpoint = initial_replay_checkpoint
    released = False
    started_at = time.time()
    for step, policy in zip(selected_steps, policies):
        index = int(step["sequence_index"])
        attempt = int(step["attempt"])
        step_dir = replay_dir / "steps" / f"{index:02d}_attempt_{attempt:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        before_state = _runtime_state(runtime_state_file)
        front_action = step["actions"]["front"]["method"]
        print(
            f"[accepted-replay] step={index:02d}/{through_step:02d} "
            f"source_attempt={attempt:02d} action={front_action} "
            f"mode={step['coordination_mode']}",
            flush=True,
        )
        api = G1SkillAPI(
            backend=backend,
            output_dir=step_dir / "skill_calls",
            heartbeat_file=heartbeat_file,
            runtime_state_file=runtime_state_file,
        )
        released_before = released
        trace = PolicyExecutor(step_dir).execute(policy, api)
        require_target = released or front_action in {
            "release_payload",
            "exit_forward",
            "exit_backward",
            "exit_lateral_left",
            "exit_lateral_right",
            "recover_stand",
        }
        require_loaded = not require_target

        gate = _wait_for_gate(
            runtime_state_file,
            require_loaded_support=require_loaded,
            require_target_support=require_target,
            timeout_s=gate_timeout_s,
            stable_s=0.5,
        )
        after_state = _runtime_state(runtime_state_file)
        report = evaluator.evaluate(
            task=task,
            trace=trace,
            before_state=before_state,
            state_override=after_state,
            stepwise=True,
            route_phase_id=str(
                step["robot_scripts"]["front"].get("phase_id", "")
            ),
        )
        accepted_evaluation = _read_json(
            _resolve_artifact_path(
                step.get("source_artifact_dir", ""),
                run_dir=run_dir,
            )
            / "evaluation.json",
            {},
        )
        deviation = _pose_deviation(
            replay_state=after_state,
            accepted_evaluation=accepted_evaluation,
        )
        physical_success = bool(trace.status == "success" and gate["success"])
        result = {
            "sequence_index": index,
            "source_attempt": attempt,
            "source_checkpoint_before": step["checkpoint_before"],
            "source_checkpoint_after": step["checkpoint_after"],
            "action": front_action,
            "coordination_mode": step["coordination_mode"],
            "trace_status": trace.status,
            "trace_error": trace.error,
            "hard_physical_gate": gate,
            "current_verifier": asdict(report),
            "accepted_endpoint_deviation": deviation,
            "physical_success": physical_success,
            "replay_checkpoint_before": latest_replay_checkpoint,
            "replay_checkpoint_after": "",
            "rollback_performed": False,
        }
        if not physical_success:
            print(
                f"[accepted-replay] HARD FAILURE at step={index:02d}: "
                f"trace={trace.status} reasons={gate.get('reasons', [])}; "
                f"restoring {latest_replay_checkpoint}",
                flush=True,
            )
            result["rollback_performed"] = True
            result["rollback"] = _restore_with_planner_reset(
                checkpoint_client=checkpoint_client,
                backend=backend,
                checkpoint_id=latest_replay_checkpoint,
                output_dir=step_dir / "rollback",
                heartbeat_file=heartbeat_file,
                runtime_state_file=runtime_state_file,
                require_loaded_support=not released_before,
                require_target_support=released_before,
            )
            rollback_state = _runtime_state(runtime_state_file)
            rollback_visual = camera.observe(
                label=f"replay_step_{index:02d}_rollback",
                runtime_state=rollback_state,
            )
            result["rollback_visual"] = asdict(rollback_visual)
            _write_json(step_dir / "step_result.json", result)
            results.append(result)
            summary = {
                "status": "failed_and_rolled_back",
                "completed_steps": index - 1,
                "failed_step": index,
                "latest_replay_checkpoint": latest_replay_checkpoint,
                "elapsed_s": time.time() - started_at,
                "results": results,
            }
            _write_json(replay_dir / "replay_summary.json", summary)
            _summarize_replays(package_dir)
            print(f"[accepted-replay] artifacts={replay_dir}", flush=True)
            return 2

        released = bool(
            after_state.get("payload", {}).get("released", released_before)
            if isinstance(after_state.get("payload", {}), dict)
            else released_before
        )
        previous_replay_checkpoint = latest_replay_checkpoint
        next_checkpoint = f"{checkpoint_prefix}_{index:02d}"
        checkpoint_client.save(next_checkpoint, reset_metrics=True)
        latest_replay_checkpoint = next_checkpoint
        if planner_reset_each_step and index < through_step:
            try:
                result["planner_reset_after_step"] = _restore_with_planner_reset(
                    checkpoint_client=checkpoint_client,
                    backend=backend,
                    checkpoint_id=next_checkpoint,
                    output_dir=step_dir / "accepted_checkpoint_roundtrip",
                    heartbeat_file=heartbeat_file,
                    runtime_state_file=runtime_state_file,
                    require_loaded_support=not released,
                    require_target_support=released,
                )
                reset_state = _runtime_state(runtime_state_file)
                result["next_step_start_deviation"] = _pose_deviation(
                    replay_state=reset_state,
                    accepted_evaluation=accepted_evaluation,
                )
            except Exception as exc:  # noqa: BLE001 - preserve and roll back replay evidence
                print(
                    f"[accepted-replay] CHECKPOINT ROUNDTRIP FAILURE at step={index:02d}: "
                    f"{type(exc).__name__}: {exc}; restoring "
                    f"{previous_replay_checkpoint}",
                    flush=True,
                )
                result["physical_success"] = False
                result["checkpoint_roundtrip_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                result["rollback_performed"] = True
                result["rollback"] = _restore_with_planner_reset(
                    checkpoint_client=checkpoint_client,
                    backend=backend,
                    checkpoint_id=previous_replay_checkpoint,
                    output_dir=step_dir / "roundtrip_failure_rollback",
                    heartbeat_file=heartbeat_file,
                    runtime_state_file=runtime_state_file,
                    require_loaded_support=not released_before,
                    require_target_support=released_before,
                )
                latest_replay_checkpoint = previous_replay_checkpoint
                rollback_state = _runtime_state(runtime_state_file)
                rollback_visual = camera.observe(
                    label=f"replay_step_{index:02d}_roundtrip_rollback",
                    runtime_state=rollback_state,
                )
                result["rollback_visual"] = asdict(rollback_visual)
                _write_json(step_dir / "step_result.json", result)
                results.append(result)
                summary = {
                    "status": "checkpoint_roundtrip_failed_and_rolled_back",
                    "completed_steps": index - 1,
                    "failed_step": index,
                    "latest_replay_checkpoint": latest_replay_checkpoint,
                    "elapsed_s": time.time() - started_at,
                    "results": results,
                }
                _write_json(replay_dir / "replay_summary.json", summary)
                _summarize_replays(package_dir)
                print(f"[accepted-replay] artifacts={replay_dir}", flush=True)
                return 2
        visual = camera.observe(
            label=f"replay_step_{index:02d}_attempt_{attempt:02d}",
            runtime_state=_runtime_state(runtime_state_file),
        )
        result["replay_checkpoint_after"] = next_checkpoint
        result["visual"] = asdict(visual)
        _write_json(step_dir / "step_result.json", result)
        results.append(result)
        semantic = "accepted" if report.success else f"warning:{report.failure_type}"
        payload_error = deviation.get("payload_xy_error_m")
        print(
            f"[accepted-replay] step={index:02d} physically_safe checkpoint="
            f"{next_checkpoint} current_verifier={semantic} "
            f"accepted_endpoint_xy_error_m={_format_float(payload_error)}",
            flush=True,
        )

    final_state = _runtime_state(runtime_state_file)
    final_gate = _physical_gate(
        final_state,
        require_loaded_support=False,
        require_target_support=released,
    )
    summary = {
        "status": "completed" if final_gate[0] else "completed_with_final_gate_failure",
        "completed_steps": len(results),
        "latest_replay_checkpoint": latest_replay_checkpoint,
        "elapsed_s": time.time() - started_at,
        "final_physical_gate": {
            "success": final_gate[0],
            "reasons": final_gate[1],
            "state": final_gate[2],
        },
        "results": results,
    }
    _write_json(replay_dir / "replay_summary.json", summary)
    _summarize_replays(package_dir)
    print(
        f"[accepted-replay] completed={len(results)} final_gate={final_gate[0]} "
        f"checkpoint={latest_replay_checkpoint} artifacts={replay_dir}",
        flush=True,
    )
    return 0 if final_gate[0] else 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export all trusted accepted scripts and audit/replay one CAP episode."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="Export trusted scripts and aggregate attempts/model calls.",
    )
    export_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    export_parser.add_argument("--output-dir", type=Path, default=None)

    replay_parser = subparsers.add_parser(
        "replay",
        help="Validate or physically replay the exported trusted trajectory.",
    )
    replay_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    replay_parser.add_argument("--package-dir", type=Path, default=None)
    replay_parser.add_argument(
        "--execute",
        action="store_true",
        help="Dispatch the scripts after restoring the initial checkpoint.",
    )
    replay_parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        help=(
            "Start at this sequence index and restore that step's trusted "
            "checkpoint_before; defaults to 1."
        ),
    )
    replay_parser.add_argument(
        "--through-step",
        type=int,
        default=0,
        help="Replay through this sequence index; 0 means the complete trajectory.",
    )
    replay_parser.add_argument(
        "--gate-timeout-s",
        type=float,
        default=10.0,
        help="Wait this long for each endpoint hard physical gate.",
    )
    replay_parser.add_argument(
        "--planner-reset-each-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Round-trip each replay checkpoint through the same dual-planner reset "
            "boundary used by the accepted evolution loop."
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    if args.command == "export":
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else run_dir / "all_scripts"
        )
        summary = export_episode(run_dir, output_dir)
        print(
            f"[accepted-export] actions={summary['accepted_actions']} "
            f"attempts={summary['attempt_totals']['attempt_artifacts']} "
            f"model_inferences={summary['model_totals']['unique_model_inferences']} "
            f"output={output_dir}"
        )
        return 0

    package_dir = (
        args.package_dir.resolve()
        if args.package_dir is not None
        else run_dir / "all_scripts"
    )
    return replay_episode(
        run_dir=run_dir,
        package_dir=package_dir,
        execute=bool(args.execute),
        from_step=int(args.from_step),
        through_step=int(args.through_step),
        gate_timeout_s=max(1.0, float(args.gate_timeout_s)),
        planner_reset_each_step=bool(args.planner_reset_each_step),
    )


if __name__ == "__main__":
    raise SystemExit(main())
