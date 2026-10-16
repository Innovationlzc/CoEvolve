"""Human-oriented, non-destructive views over evolution run artifacts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any


ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")
DIRECT_ATTEMPT_RE = re.compile(r"^attempt_(\d+)(?:_|$)")

PLANNING_FILES = {
    "policy.json",
    "team_plan.json",
    "task_distribution.json",
}
EXECUTION_FILES = {
    "execution_trace.json",
    "pre_dispatch_recovery.json",
}
VERIFICATION_FILES = {
    "evaluation.json",
    "corrected_evaluation.json",
    "resume_evaluation.json",
    "preliminary_physics_evaluation.json",
    "visual_evaluation.json",
    "transition_decision.json",
    "post_action_stability_gate.json",
    "precommit_stability_gate.json",
    "checkpoint_roundtrip_stability_gate.json",
    "checkpoint_roundtrip_semantic_evaluation.json",
}
EVOLUTION_FILES = {
    "evolution_controller_decision.json",
    "evolution_record.json",
    "model_evolution_error.json",
}

CATEGORY_DIRS = {
    "planning": "01_planning",
    "code": "02_generated_code",
    "execution": "03_execution",
    "verification": "04_verification",
    "checkpoint": "05_checkpoint",
    "evolution": "06_evolution",
    "model_calls": "07_model_calls",
    "visuals": "08_visuals",
    "other": "99_other",
}


def build_run_review(
    run_dir: str | Path,
    *,
    review_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a disposable review tree without changing machine-readable evidence."""

    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    if review_dir is None:
        destination = run_dir / "review"
    else:
        requested = Path(review_dir).expanduser()
        destination = (
            requested if requested.is_absolute() else run_dir / requested
        ).resolve()
    if destination == run_dir or run_dir not in destination.parents:
        raise ValueError("review directory must be a child of the run directory")

    attempts = _attempt_directories(run_dir)
    role_calls = _read_role_calls(run_dir)
    call_assignments, unassigned_calls = _assign_role_calls(attempts, role_calls)
    summaries = [
        _summarize_attempt(
            run_dir,
            attempt_dir,
            role_calls=call_assignments.get(_attempt_number(attempt_dir), []),
        )
        for attempt_dir in attempts
    ]
    run_summary = _summarize_run(
        run_dir,
        summaries,
        role_calls=role_calls,
        unassigned_calls=unassigned_calls,
    )

    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    _remove_generated_tree(staging)
    staging.mkdir(parents=True)
    try:
        for attempt_dir, summary in zip(attempts, summaries):
            target = staging / "attempts" / attempt_dir.name
            _build_attempt_view(
                run_dir,
                attempt_dir,
                target,
                summary=summary,
                role_calls=call_assignments.get(summary["attempt"], []),
            )
        _build_run_level_views(run_dir, staging)
        _write_json(staging / "run_summary.json", run_summary)
        _write_attempt_csv(staging / "attempts.csv", summaries)
        (staging / "ARTIFACT_GUIDE.md").write_text(
            _artifact_guide_markdown(),
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            _run_readme_markdown(run_dir, run_summary, summaries),
            encoding="utf-8",
        )
        if unassigned_calls:
            _link_model_calls(
                run_dir,
                staging / "unassigned_model_calls",
                unassigned_calls,
            )
        _remove_generated_tree(destination)
        staging.rename(destination)
    except BaseException:
        _remove_generated_tree(staging)
        raise
    return run_summary


def _attempt_directories(run_dir: Path) -> list[Path]:
    attempts = [
        child
        for child in run_dir.iterdir()
        if child.is_dir() and ATTEMPT_RE.fullmatch(child.name)
    ]
    return sorted(attempts, key=_attempt_number)


def _attempt_number(path: Path) -> int:
    match = ATTEMPT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not an attempt directory: {path}")
    return int(match.group(1))


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_role_calls(run_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(run_dir / "model_roles" / "role_call_index.json", {})
    calls = payload.get("calls", []) if isinstance(payload, dict) else []
    return [dict(call) for call in calls if isinstance(call, dict)]


def _assign_role_calls(
    attempts: list[Path],
    role_calls: list[dict[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_attempt: dict[int, list[dict[str, Any]]] = {
        _attempt_number(path): [] for path in attempts
    }
    script_to_attempt: dict[str, int] = {}
    for attempt_dir in attempts:
        attempt = _attempt_number(attempt_dir)
        policy = _read_json(attempt_dir / "policy.json", {})
        if not isinstance(policy, dict):
            policy = _read_json(attempt_dir / "team_plan.json", {})
        if not isinstance(policy, dict):
            continue
        script_id = str(policy.get("script_id", "")).strip()
        if script_id:
            script_to_attempt[script_id] = attempt
        for robot_script in policy.get("robot_scripts", []):
            if not isinstance(robot_script, dict):
                continue
            robot_script_id = str(robot_script.get("script_id", "")).strip()
            if robot_script_id:
                script_to_attempt[robot_script_id] = attempt

    unassigned: list[dict[str, Any]] = []
    for call in role_calls:
        artifact_id = str(call.get("artifact_id", "")).strip()
        match = DIRECT_ATTEMPT_RE.match(artifact_id)
        attempt = int(match.group(1)) if match else script_to_attempt.get(artifact_id)
        if attempt not in by_attempt:
            attempt = None
        if attempt is None:
            unassigned.append(call)
        else:
            by_attempt[attempt].append(call)
    return by_attempt, unassigned


def _summarize_attempt(
    run_dir: Path,
    attempt_dir: Path,
    *,
    role_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    attempt = _attempt_number(attempt_dir)
    policy = _read_json(attempt_dir / "policy.json", {})
    if not isinstance(policy, dict):
        policy = _read_json(attempt_dir / "team_plan.json", {})
    policy = policy if isinstance(policy, dict) else {}
    distribution = _object(_read_json(attempt_dir / "task_distribution.json", {}))
    trace = _object(_read_json(attempt_dir / "execution_trace.json", {}))
    evaluation = _object(_read_json(attempt_dir / "evaluation.json", {}))
    transition = _object(
        _object(evaluation.get("evidence", {})).get("transition_decision", {})
    )
    if not transition:
        transition = _object(
            _read_json(attempt_dir / "transition_decision.json", {})
        )
    controller = _object(
        _read_json(attempt_dir / "evolution_controller_decision.json", {})
    )
    record = _object(_read_json(attempt_dir / "evolution_record.json", {}))

    assignments = [
        item for item in policy.get("assignments", []) if isinstance(item, dict)
    ]
    robot_scripts = [
        item for item in policy.get("robot_scripts", []) if isinstance(item, dict)
    ]
    methods: dict[str, str] = {}
    parameters: dict[str, dict[str, Any]] = {}
    robot_executions = trace.get("robot_executions", [])
    if isinstance(robot_executions, list):
        for execution in robot_executions:
            if not isinstance(execution, dict):
                continue
            robot_id = str(execution.get("robot_id", "")).strip()
            planned = execution.get("planned_calls", [])
            first = planned[0] if isinstance(planned, list) and planned else {}
            if robot_id and isinstance(first, dict):
                methods[robot_id] = str(first.get("method", ""))
                parameters[robot_id] = _object(first.get("parameters", {}))
    if not methods:
        for script in robot_scripts:
            robot_id = str(script.get("robot_id", "")).strip()
            method, parsed_parameters = _first_api_call(str(script.get("code", "")))
            if robot_id and method:
                methods[robot_id] = method
                parameters[robot_id] = parsed_parameters

    metrics = _object(evaluation.get("metrics", {}))
    checkpoint = _object(distribution.get("checkpoint", {}))
    checkpoint_before = str(
        checkpoint.get("before", record.get("checkpoint_before", ""))
    )
    checkpoint_after = str(
        checkpoint.get("after", record.get("checkpoint_after", ""))
    )
    rollback = bool(
        checkpoint.get("rollback_performed", record.get("rollback_performed", False))
    )
    completed = bool(evaluation)
    accepted = bool(evaluation.get("success", False)) if completed else None
    dispatched = _trace_was_dispatched(trace)
    phase = _policy_phase(policy)
    primitive = next((value for value in methods.values() if value), "")
    if not primitive:
        primitive = _strategy_field(policy, "primitive")
    objective = _strategy_field(policy, "semantic_objective")
    if not objective:
        for values in parameters.values():
            for key in ("turn_direction", "direction", "axis"):
                candidate = str(values.get(key, "")).strip()
                if candidate:
                    objective = candidate
                    break
            if objective:
                break
    coordination_mode = str(distribution.get("coordination_mode", "")).strip()
    if not coordination_mode and assignments:
        coordination_mode = str(assignments[0].get("coordination_mode", "")).strip()

    call_usage = _summarize_call_usage(role_calls)
    visuals = _attempt_visuals(run_dir, attempt)
    status = (
        "accepted"
        if accepted is True
        else "rejected"
        if accepted is False and completed
        else "incomplete"
    )
    return {
        "attempt": attempt,
        "attempt_dir": attempt_dir.name,
        "script_id": str(policy.get("script_id", "")),
        "phase_id": phase,
        "primitive": primitive,
        "semantic_objective": objective,
        "coordination_mode": coordination_mode,
        "assignments": assignments,
        "robot_scripts": robot_scripts,
        "methods": methods,
        "parameters": parameters,
        "execution_status": str(trace.get("status", "not_recorded")),
        "physically_dispatched": dispatched,
        "status": status,
        "accepted": accepted,
        "failure_type": str(evaluation.get("failure_type", "")),
        "failure_level": str(evaluation.get("failure_level", "")),
        "diagnosis": str(evaluation.get("diagnosis", "")),
        "visual_summary": str(evaluation.get("visual_summary", "")),
        "transition_outcome": (
            str(transition.get("outcome", ""))
            if accepted is not False
            else str(
                _object(
                    _object(evaluation.get("evidence", {})).get(
                        "checkpoint_outcome", {}
                    )
                ).get("transition_outcome", "")
                or transition.get("outcome", "")
            )
        ),
        "transition_reason": str(transition.get("reason", "")),
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "rollback_performed": rollback,
        "controller_action": str(controller.get("action", "")),
        "controller_repair_level": str(controller.get("repair_level", "")),
        "next_script_id": str(record.get("next_script_id", "")),
        "metrics": _compact_metrics(metrics),
        "model_calls": call_usage,
        "visuals": visuals,
    }


def _summarize_run(
    run_dir: Path,
    attempts: list[dict[str, Any]],
    *,
    role_calls: list[dict[str, Any]],
    unassigned_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = _object(_read_json(run_dir / "run_manifest.json", {}))
    metrics = _object(_read_json(run_dir / "metrics.json", {}))
    accepted = sum(item["status"] == "accepted" for item in attempts)
    rejected = sum(item["status"] == "rejected" for item in attempts)
    incomplete = len(attempts) - accepted - rejected
    dispatched = sum(bool(item["physically_dispatched"]) for item in attempts)
    rollbacks = sum(bool(item["rollback_performed"]) for item in attempts)
    latest_checkpoint = ""
    for item in attempts:
        if item["status"] == "accepted" and item["checkpoint_after"]:
            latest_checkpoint = item["checkpoint_after"]
    terminal = _object(metrics.get("terminal", {}))
    status = str(terminal.get("status", "")).strip()
    if not status:
        status = "interrupted_or_in_progress" if incomplete else "not_finalized"
    usage = _summarize_call_usage(role_calls)
    protocol = _object(manifest.get("protocol", {}))
    codebase = _object(manifest.get("codebase", {}))
    model = _object(manifest.get("model", {}))
    return {
        "schema_version": 1,
        "kind": "coevolve_human_review_summary",
        "source_run_dir": str(run_dir),
        "review_is_derived": True,
        "review_can_be_regenerated": True,
        "status": status,
        "protocol_id": str(protocol.get("protocol_id", "")),
        "planner_model": str(
            model.get("requested", "")
            or _object(manifest.get("effective_arguments", {})).get(
                "planner_model", ""
            )
        ),
        "seed": manifest.get("seed"),
        "git_head": str(codebase.get("git_head", "")),
        "attempt_count": len(attempts),
        "completed_attempt_count": accepted + rejected,
        "physical_dispatch_count": dispatched,
        "accepted_attempt_count": accepted,
        "rejected_attempt_count": rejected,
        "incomplete_attempt_count": incomplete,
        "rollback_count": rollbacks,
        "latest_trusted_checkpoint": latest_checkpoint,
        "model_calls": usage,
        "unassigned_model_call_count": len(unassigned_calls),
        "attempts": [
            {
                key: item[key]
                for key in (
                    "attempt",
                    "script_id",
                    "phase_id",
                    "primitive",
                    "semantic_objective",
                    "coordination_mode",
                    "status",
                    "failure_type",
                    "checkpoint_before",
                    "checkpoint_after",
                    "rollback_performed",
                )
            }
            for item in attempts
        ],
    }


def _build_attempt_view(
    run_dir: Path,
    attempt_dir: Path,
    target: Path,
    *,
    summary: dict[str, Any],
    role_calls: list[dict[str, Any]],
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for category_dir in CATEGORY_DIRS.values():
        (target / category_dir).mkdir()

    for source in sorted(attempt_dir.iterdir(), key=lambda item: item.name):
        category = _attempt_artifact_category(source)
        _relative_symlink(source, target / CATEGORY_DIRS[category] / source.name)

    _link_model_calls(
        run_dir,
        target / CATEGORY_DIRS["model_calls"],
        role_calls,
    )
    for source in _visual_sources(run_dir, summary["attempt"]):
        _relative_symlink(source, target / CATEGORY_DIRS["visuals"] / source.name)

    _write_json(target / "summary.json", summary)
    (target / "README.md").write_text(
        _attempt_readme_markdown(summary, role_calls),
        encoding="utf-8",
    )


def _attempt_artifact_category(path: Path) -> str:
    name = path.name
    if path.is_file() and path.suffix == ".py":
        return "code"
    if name in PLANNING_FILES:
        return "planning"
    if name in EXECUTION_FILES or name == "skill_calls" or name.startswith(
        "pre_dispatch_"
    ):
        return "execution"
    if name in VERIFICATION_FILES:
        return "verification"
    if name in EVOLUTION_FILES or name.startswith("role_generation_stall_"):
        return "evolution"
    if name.startswith("checkpoint_") or name == "trusted_checkpoint_observation.json":
        return "checkpoint"
    return "other"


def _build_run_level_views(run_dir: Path, target: Path) -> None:
    groups = {
        "run_metadata": (
            "run_manifest.json",
            "run_plan.json",
            "formal_arguments.json",
            "initial_observation.json",
            "initial_physical_gate.json",
            "initial_checkpoint_physical_gate.json",
            "term1_summary.json",
        ),
        "final_results": (
            "metrics.json",
            "final_summary.json",
            "model_usage.json",
            "billing_reconciliation.json",
            "evolution_report.html",
            "evolution_records.jsonl",
            "raw_rollout_evidence.jsonl",
        ),
        "trusted_state": (
            "accepted_step_visuals",
            "checkpoints",
            "checkpoint_events",
            "memory",
        ),
        "live_runtime": (
            "runtime_state.json",
            "planner_heartbeat.json",
            "latest_task_distribution.json",
            "turn_episode.json",
            "straight_heading_references.json",
            "vlm_global_frame.png",
        ),
        "raw_model_roles": ("model_roles",),
        "raw_visual_observations": ("visual_observations",),
        "physical_action_videos": ("action_videos",),
    }
    root = target / "run_artifacts"
    for group, names in groups.items():
        group_dir = root / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = run_dir / name
            if source.exists():
                _relative_symlink(source, group_dir / name)


def _link_model_calls(
    run_dir: Path,
    target: Path,
    calls: list[dict[str, Any]],
) -> None:
    for index, call in enumerate(calls, start=1):
        role = _slug(str(call.get("role", "unknown"))) or "unknown"
        source = _role_call_source(run_dir, call)
        if source is None:
            continue
        role_dir = target / role
        role_dir.mkdir(parents=True, exist_ok=True)
        name = source.name
        link = role_dir / name
        if link.exists() or link.is_symlink():
            link = role_dir / f"{index:03d}_{name}"
        _relative_symlink(source, link)


def _role_call_source(run_dir: Path, call: dict[str, Any]) -> Path | None:
    artifact = str(call.get("artifact_path", "")).strip()
    if artifact:
        candidate = run_dir / "model_roles" / Path(artifact).name
        if candidate.is_file():
            return candidate
    return None


def _attempt_visuals(run_dir: Path, attempt: int) -> list[str]:
    return [str(path.relative_to(run_dir)) for path in _visual_sources(run_dir, attempt)]


def _visual_sources(run_dir: Path, attempt: int) -> list[Path]:
    sources: set[Path] = set()
    visual_root = run_dir / "visual_observations"
    if visual_root.is_dir():
        sources.update(visual_root.glob(f"attempt_{attempt:02d}_*"))
    accepted_root = run_dir / "accepted_step_visuals"
    if accepted_root.is_dir():
        sources.update(accepted_root.glob(f"*_attempt_{attempt:02d}.*"))
        sources.update(accepted_root.glob(f"*_attempt_{attempt:02d}_*"))
    return sorted(path for path in sources if path.is_file())


def _summarize_call_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    known_cost_microusd = 0
    billed = 0
    roles: dict[str, int] = {}
    valid = 0
    invalid = 0
    for call in calls:
        usage = _object(call.get("usage", {}))
        input_tokens += _integer(usage.get("input_tokens"))
        output_tokens += _integer(usage.get("output_tokens"))
        amount = usage.get("billing_amount_microusd")
        if amount is not None:
            known_cost_microusd += _integer(amount)
            billed += 1
        role = str(call.get("role", "unknown")) or "unknown"
        roles[role] = roles.get(role, 0) + 1
        if bool(call.get("valid", False)):
            valid += 1
        else:
            invalid += 1
    return {
        "call_count": len(calls),
        "valid_call_count": valid,
        "invalid_call_count": invalid,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "known_billing_call_count": billed,
        "unknown_billing_call_count": len(calls) - billed,
        "known_billing_amount_microusd": known_cost_microusd,
        "known_billing_amount_usd": known_cost_microusd / 1_000_000.0,
        "roles": dict(sorted(roles.items())),
    }


def _trace_was_dispatched(trace: dict[str, Any]) -> bool:
    for call in trace.get("skill_calls", []):
        if isinstance(call, dict) and str(call.get("status", "")) not in {
            "",
            "dry_run",
        }:
            return True
    return bool(trace.get("robot_executions")) and str(trace.get("status", "")) in {
        "success",
        "script_error",
    }


def _policy_phase(policy: dict[str, Any]) -> str:
    assignments = policy.get("assignments", [])
    if isinstance(assignments, list):
        for assignment in assignments:
            if isinstance(assignment, dict) and assignment.get("phase_id"):
                return str(assignment["phase_id"])
    return ""


def _strategy_field(policy: dict[str, Any], field: str) -> str:
    task_plan = policy.get("task_plan", policy.get("team_plan", {}))
    if not isinstance(task_plan, dict):
        return ""
    subtask = task_plan.get("current_subtask", {})
    if not isinstance(subtask, dict):
        return ""
    strategy = subtask.get("strategy", {})
    return str(strategy.get(field, "")) if isinstance(strategy, dict) else ""


def _first_api_call(code: str) -> tuple[str, dict[str, Any]]:
    match = re.search(r"api\.([A-Za-z_][A-Za-z0-9_]*)\(([^\n]*)\)", code)
    if match is None:
        return "", {}
    return match.group(1), {"raw_arguments": match.group(2)}


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "payload_x_m",
        "payload_y_m",
        "payload_z_m",
        "payload_step_displacement_m",
        "payload_target_progress_m",
        "payload_target_xy_error_m",
        "payload_tilt_deg",
        "wall_contact_count",
        "payload_support_surface_contact_count",
        "robot_pair_xy_separation_m",
        "front_back_step_vector_desync_m",
        "task_complete",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _write_attempt_csv(path: Path, attempts: list[dict[str, Any]]) -> None:
    fields = (
        "attempt",
        "phase_id",
        "primitive",
        "semantic_objective",
        "coordination_mode",
        "status",
        "failure_type",
        "checkpoint_before",
        "checkpoint_after",
        "rollback_performed",
        "controller_action",
        "next_script_id",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for attempt in attempts:
            writer.writerow({field: attempt.get(field, "") for field in fields})


def _run_readme_markdown(
    run_dir: Path,
    summary: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> str:
    lines = [
        "# CoEvolve 实验结果 Review",
        "",
        "> 本目录是自动生成的人工阅读层。原始证据仍保存在上一级目录；",
        "> 这里的分类文件是相对软链接，删除整个 `review/` 不会删除实验证据。",
        "",
        "## 本轮概览",
        "",
        f"- Source: `{run_dir}`",
        f"- Protocol: `{summary['protocol_id'] or 'unknown'}`",
        f"- Model: `{summary['planner_model'] or 'unknown'}`",
        f"- Status: `{summary['status']}`",
        f"- Attempts: `{summary['attempt_count']}`",
        f"- Physical dispatches: `{summary['physical_dispatch_count']}`",
        f"- Accepted / rejected / incomplete: `{summary['accepted_attempt_count']} / "
        f"{summary['rejected_attempt_count']} / {summary['incomplete_attempt_count']}`",
        f"- Rollbacks: `{summary['rollback_count']}`",
        f"- Latest trusted checkpoint: `{summary['latest_trusted_checkpoint'] or 'none'}`",
        f"- Model calls: `{summary['model_calls']['call_count']}` "
        f"(valid `{summary['model_calls']['valid_call_count']}`, "
        f"invalid `{summary['model_calls']['invalid_call_count']}`)",
        f"- Known API cost: `${summary['model_calls']['known_billing_amount_usd']:.6f}`; "
        f"unknown-price calls: `{summary['model_calls']['unknown_billing_call_count']}`",
        "",
        "详细文件含义见 [ARTIFACT_GUIDE.md](ARTIFACT_GUIDE.md)，表格数据见 "
        "[attempts.csv](attempts.csv)，机器可读汇总见 "
        "[run_summary.json](run_summary.json)。",
        "",
        "## Attempt 索引",
        "",
        "| Attempt | Phase | Action | Mode | Result | Failure | Checkpoint | Rollback | Calls |",
        "|---:|---|---|---|---|---|---|:---:|---:|",
    ]
    for item in attempts:
        attempt = item["attempt"]
        action = item["primitive"]
        if item["semantic_objective"]:
            action += f":{item['semantic_objective']}"
        checkpoint = f"{item['checkpoint_before']} -> {item['checkpoint_after']}"
        lines.append(
            f"| [{attempt:02d}](attempts/attempt_{attempt:02d}/README.md) | "
            f"{_md(item['phase_id'])} | {_md(action)} | "
            f"{_md(item['coordination_mode'])} | `{item['status']}` | "
            f"{_md(item['failure_type'])} | {_md(checkpoint)} | "
            f"{'yes' if item['rollback_performed'] else 'no'} | "
            f"{item['model_calls']['call_count']} |"
        )
    lines.extend(
        [
            "",
            "## 整轮目录",
            "",
            "```text",
            "review/",
            "├── README.md                 # 本轮总览和 Attempt 索引",
            "├── ARTIFACT_GUIDE.md          # 每类文件用途",
            "├── attempts.csv               # 一行一个 Attempt",
            "├── run_summary.json           # 机器可读汇总",
            "├── attempts/",
            "│   └── attempt_XX/            # 每轮按职责分类的阅读视图",
            "├── run_artifacts/             # 初始化、最终结果、可信状态等整轮产物",
            "└── unassigned_model_calls/    # 无法可靠归属某次 Attempt 的模型调用",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _attempt_readme_markdown(
    summary: dict[str, Any],
    role_calls: list[dict[str, Any]],
) -> str:
    attempt = summary["attempt"]
    action = summary["primitive"]
    if summary["semantic_objective"]:
        action += f":{summary['semantic_objective']}"
    lines = [
        f"# Attempt {attempt:02d}",
        "",
        "## 结论",
        "",
        f"- Script: `{summary['script_id']}`",
        f"- Phase: `{summary['phase_id']}`",
        f"- Strategy: `{action}`",
        f"- Coordination: `{summary['coordination_mode']}`",
        f"- Execution: `{summary['execution_status']}`",
        f"- Result: `{summary['status']}`",
        f"- Failure: `{summary['failure_type'] or 'none'}` / "
        f"`{summary['failure_level'] or 'none'}`",
        f"- Checkpoint: `{summary['checkpoint_before']}` -> "
        f"`{summary['checkpoint_after']}`",
        f"- Rollback: `{'yes' if summary['rollback_performed'] else 'no'}`",
        f"- Transition: `{summary['transition_outcome'] or 'not_recorded'}`",
        "",
    ]
    if summary["diagnosis"]:
        lines.extend(["Verifier diagnosis:", "", f"> {_md(summary['diagnosis'])}", ""])
    lines.extend(
        [
            "## 双机脚本",
            "",
            "| Robot | Role | Method | Parameters |",
            "|---|---|---|---|",
        ]
    )
    assignment_by_robot = {
        str(item.get("robot_id", "")): item for item in summary["assignments"]
    }
    for robot_id in ("front", "back"):
        assignment = assignment_by_robot.get(robot_id, {})
        parameters = json.dumps(
            summary["parameters"].get(robot_id, {}),
            ensure_ascii=False,
            sort_keys=True,
        )
        lines.append(
            f"| `{robot_id}` | {_md(assignment.get('role', ''))} | "
            f"`{summary['methods'].get(robot_id, '')}` | `{_md(parameters)}` |"
        )
    lines.extend(
        [
            "",
            "## 模型调用",
            "",
            "| Role | Artifact | Valid | Input | Output | Known cost |",
            "|---|---|:---:|---:|---:|---:|",
        ]
    )
    for call in role_calls:
        usage = _object(call.get("usage", {}))
        amount_microusd = usage.get("billing_amount_microusd")
        if amount_microusd is not None:
            amount = _integer(amount_microusd) / 1_000_000.0
        else:
            amount = usage.get("billing_amount_usd")
        cost = "unknown" if amount is None else f"${float(amount):.6f}"
        source = Path(str(call.get("artifact_path", ""))).name
        role = str(call.get("role", ""))
        artifact_id = str(call.get("artifact_id", ""))
        link = f"07_model_calls/{_slug(role)}/{source}"
        lines.append(
            f"| `{role}` | [{_md(artifact_id)}]({link}) | "
            f"{'yes' if call.get('valid') else 'no'} | "
            f"{_integer(usage.get('input_tokens'))} | "
            f"{_integer(usage.get('output_tokens'))} | {cost} |"
        )
    if not role_calls:
        lines.append("| none | - | - | 0 | 0 | - |")
    lines.extend(
        [
            "",
            "## 分类目录",
            "",
            "| Directory | Contents |",
            "|---|---|",
            "| `01_planning/` | 团队策略、角色分配、dispatch 元数据 |",
            "| `02_generated_code/` | Team/front/back 的实际 Python Skill API 脚本 |",
            "| `03_execution/` | 执行 trace、SONIC 命令、stdout/stderr |",
            "| `04_verification/` | 物理、视觉、transition 和稳定性判定 |",
            "| `05_checkpoint/` | checkpoint round-trip、恢复和位姿校验 |",
            "| `06_evolution/` | Controller 决策、完整 EvolutionRecord、下一步关系 |",
            "| `07_model_calls/` | 本 Attempt 各模型角色的原始 prompt/response/usage |",
            "| `08_visuals/` | 本动作执行前后及 accepted checkpoint 图像 |",
            "| `99_other/` | 尚未归入稳定类别的诊断产物 |",
            "",
            "这些文件均链接到原始结果；`summary.json` 和本页是自动生成摘要。",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_guide_markdown() -> str:
    return """# CoEvolve Result Artifact Guide

## 两层结构

- 上一级 run 目录是 **raw audit layer**：执行、恢复、动态经验和论文统计读取它，文件名保持稳定。
- `review/` 是 **human review layer**：自动分类、摘要和相对软链接，可随时删除并重新生成。

## 一个 Attempt 的执行链

```text
Task Orchestrator plan
        -> Coder front/back scripts
        -> team Code Repair/review
        -> host validation and atomic dispatch
        -> SONIC execution trace
        -> physics + VLM evaluation
        -> transition/checkpoint decision
        -> Evolution Controller
        -> Memory Update and next policy
```

## 主要文件

| Raw artifact | Purpose |
|---|---|
| `policy.json` | 最终编译后的团队策略，是本 Attempt 的 canonical policy。 |
| `team_plan.json` | dispatch 阶段保存的兼容副本；正常情况下与 `policy.json` 内容相同。 |
| `task_distribution.json` | front/back 角色、端口、脚本路径、planned calls、checkpoint 结果。 |
| `*.py` | 模型生成并通过 AST/contract 校验的实际 Skill API 脚本。异步时通常有 team/front/back 三份；真正执行的是 front/back 两份。 |
| `execution_trace.json` | 脚本是否发布成功、每台机器人实际 planned calls、Skill backend 记录。 |
| `skill_calls/` | 发给 SONIC backend 的命令、编译 schedule、stdout/stderr 和 backend 返回值。 |
| `preliminary_physics_evaluation.json` | 动作刚结束后的第一轮物理判定，尚未完成全部视觉与 checkpoint 流程。 |
| `visual_evaluation.json` | VLM Evaluator/Failure Analyzer 对固定 global view 的结构化判断。 |
| `evaluation.json` | 合并物理、视觉、稳定性和 checkpoint 语义后的最终 Attempt 判定。 |
| `transition_decision.json` | 是否允许把动作结果作为路线进展和可信 checkpoint。 |
| `*_stability_gate.json` | 动作后、提交前、恢复后的连续稳定性采样结果。 |
| `checkpoint_candidate_roundtrip/` | 候选状态保存、恢复、heading resync 和位姿 round-trip 证据。 |
| `evolution_controller_decision.json` | continue、换策略、改协同或调参数，以及剩余物理预算。 |
| `evolution_record.json` | 本 Attempt 的聚合记录：输入 observation、policy、execution、evaluation、checkpoint 和 next script。 |
| `model_roles/*.json` | 每次模型角色调用的完整 system/user prompt、原始响应、结构化结果、token、延迟和已知费用。 |
| `accepted_step_visuals/` | 仅 accepted checkpoint 的固定 global 图，便于快速查看可信轨迹。 |
| `action_videos/` | 仅物理动作与可见 rollback/rewind 的逐段 MP4、时间线 manifest 和完整拼接视频；模型推理与等待不录制。 |

## 为什么会有三份 Python

异步协同时，Team policy 描述共同事务，front/back Coder 分别生成各自角色脚本。例如 outer robot 执行圆弧、inner robot 保持支点。Team 文件用于总体策略审计，真正原子下发的是 front/back 两份脚本。同步协同时两台机器人参数应一致，框架仍保留按机器人展开的记录，便于排查 backend 分化。
"""


def _relative_symlink(source: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    relative = os.path.relpath(source.resolve(), start=link.parent.resolve())
    link.symlink_to(relative, target_is_directory=source.is_dir())


def _remove_generated_tree(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", value.strip().lower()).strip("_")


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


__all__ = ["build_run_review"]
