#!/usr/bin/env python3
"""Aggregate the frozen CoEvolve backbone matrix without running models or physics."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.cap_policy.contracts import write_json  # noqa: E402
from praxis_coop.cap_policy.action_memory import (  # noqa: E402
    ActionMemoryError,
    VerifiedActionMemoryStore,
    _validate_episode_manifest,
)
from praxis_coop.cap_policy.formal_eval import (  # noqa: E402
    FORMAL_BACKBONES,
    FORMAL_BENCHMARK_NAME,
    FORMAL_EVAL_PROTOCOL_ID,
    FORMAL_PHYSICAL_BUDGET,
    FORMAL_RUN_INDICES,
    FORMAL_SEEDS,
    FORMAL_TRANSACTION_JOURNAL_SCHEMA_VERSION,
    FORMAL_TRANSACTION_JOURNAL_STAGES,
    PROTOCOL,
    formal_run_output_dir,
    formal_memory_paths,
)
from praxis_coop.cap_policy.strategy_memory import (  # noqa: E402
    StrategyMemoryError,
    StrategyMemoryStore,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "results" / "deployment" / "formal" / FORMAL_EVAL_PROTOCOL_ID
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail unless every frozen run contains metrics, is protocol-valid, "
            "and has complete reconciled API costs."
        ),
    )
    args = parser.parse_args()

    rows = collect_rows(REPO_ROOT)
    summaries = [
        summarize_rows(
            [row for row in rows if row["requested_model"] == backbone.requested_model]
        )
        for backbone in FORMAL_BACKBONES
    ]
    fingerprints = sorted(
        {
            str(row["evaluation_fingerprint_sha256"])
            for row in rows
            if row["metrics_available"] and row["evaluation_fingerprint_sha256"]
        }
    )
    cohort_consistent = len(fingerprints) <= 1
    payload = {
        "kind": "coevolve_formal_backbone_summary",
        "protocol_id": FORMAL_EVAL_PROTOCOL_ID,
        "benchmark": FORMAL_BENCHMARK_NAME,
        "expected_runs": len(FORMAL_BACKBONES) * len(FORMAL_RUN_INDICES),
        "available_runs": sum(row["metrics_available"] for row in rows),
        "protocol_valid_runs": sum(row["protocol_valid"] for row in rows),
        "cohort_consistent": cohort_consistent,
        "evaluation_fingerprints": fingerprints,
        "models": summaries,
        "runs": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "formal_summary.json", payload)
    write_json(
        args.output_dir / "formal_runs.json",
        {
            "kind": "coevolve_formal_run_reconciliation",
            "protocol_id": FORMAL_EVAL_PROTOCOL_ID,
            "runs": rows,
        },
    )
    _write_csv(args.output_dir / "formal_runs.csv", rows)
    _write_csv(args.output_dir / "formal_summary.csv", summaries)

    for summary in summaries:
        print(
            f"[{summary['requested_model']}] "
            f"runs={summary['available_runs']}/{summary['expected_runs']} "
            f"valid={summary['protocol_valid_runs']} "
            f"SR@80={_display(summary['success_rate'])} "
            f"CPT@80={_display(summary['cpt_at_80'])}",
            flush=True,
        )

    invalid = [
        row for row in rows if row["metrics_available"] and not row["protocol_valid"]
    ]
    missing = [row for row in rows if not row["metrics_available"]]
    cost_incomplete = [
        row
        for row in rows
        if row["metrics_available"]
        and row["protocol_valid"]
        and not row["cost_complete"]
    ]
    if args.require_complete and (
        invalid or missing or cost_incomplete or not cohort_consistent
    ):
        raise SystemExit(
            "formal matrix is not complete and protocol-valid: "
            f"missing={len(missing)}, invalid={len(invalid)}, "
            f"cost_incomplete={len(cost_incomplete)}, "
            f"evaluation_fingerprints={len(fingerprints)}"
        )


def collect_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backbone in FORMAL_BACKBONES:
        for run_index in FORMAL_RUN_INDICES:
            output_dir = formal_run_output_dir(
                repo_root,
                model_or_key=backbone.key,
                run_index=run_index,
            )
            metrics = _read_json(output_dir / "metrics.json")
            manifest = _read_json(output_dir / "run_manifest.json")
            success = bool(metrics.get("task_success", False))
            physical_iterations = _integer(metrics.get("physical_iterations"))
            phase = metrics.get("phase_completion", {})
            phase = phase if isinstance(phase, dict) else {}
            model_metrics = metrics.get("model", {})
            model_metrics = model_metrics if isinstance(model_metrics, dict) else {}
            manifest_model = manifest.get("model", {})
            manifest_model = manifest_model if isinstance(manifest_model, dict) else {}
            resolved = str(manifest_model.get("resolved", ""))
            evaluation_fingerprint = str(
                manifest.get("evaluation_fingerprint_sha256", "")
            )
            metrics_available = bool(metrics)
            protocol_invalid_reasons = _protocol_invalid_reasons(
                metrics=metrics,
                manifest=manifest,
                expected_run_index=run_index,
                expected_seed=FORMAL_SEEDS[run_index],
                expected_model=backbone.requested_model,
                output_dir=output_dir,
            )
            protocol_valid = not protocol_invalid_reasons
            cost_complete = _is_cost_complete(
                metrics=metrics,
                manifest=manifest,
                output_dir=output_dir,
            )
            rows.append(
                {
                    "requested_model": backbone.requested_model,
                    "run_index": run_index,
                    "seed": FORMAL_SEEDS[run_index],
                    "output_dir": str(output_dir),
                    "metrics_available": metrics_available,
                    "protocol_valid": protocol_valid,
                    "protocol_invalid_reasons": protocol_invalid_reasons,
                    "cost_complete": cost_complete,
                    "manifest_status": str(manifest.get("status", "missing")),
                    "resolved_model": resolved,
                    "evaluation_fingerprint_sha256": evaluation_fingerprint,
                    "task_success": success,
                    "physical_iterations": physical_iterations,
                    "cpt_at_80": (
                        physical_iterations
                        if success and physical_iterations is not None
                        else FORMAL_PHYSICAL_BUDGET
                        if metrics_available
                        else None
                    ),
                    "phase_completion": _number(phase.get("fraction")),
                    "rejected_transactions": _integer(
                        metrics.get("rejected_step_count")
                    ),
                    "fm_calls": _sum_optional_ints(
                        model_metrics.get("llm_calls"),
                        model_metrics.get("vlm_calls"),
                    ),
                    "input_tokens": _integer(model_metrics.get("input_tokens")),
                    "output_tokens": _integer(model_metrics.get("output_tokens")),
                    "model_latency_s": _number(model_metrics.get("latency_s")),
                    "api_cost_usd": _number(model_metrics.get("billing_amount_usd")),
                    "wall_time_s": _number(metrics.get("elapsed_s")),
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["metrics_available"]]
    valid = [row for row in available if row["protocol_valid"]]
    cost_complete = [row for row in valid if row.get("cost_complete") is True]
    all_valid_costs_complete = len(cost_complete) == len(valid)
    successes = sum(bool(row["task_success"]) for row in valid)
    success_rate = successes / len(valid) if valid else None
    lower, upper = _wilson_interval(successes, len(valid))
    return {
        "requested_model": str(rows[0]["requested_model"]) if rows else "",
        "expected_runs": len(rows),
        "available_runs": len(available),
        "protocol_valid_runs": len(valid),
        "cost_complete_runs": len(cost_complete),
        "successful_runs": successes,
        "success_rate": success_rate,
        "success_rate_wilson_95_low": lower,
        "success_rate_wilson_95_high": upper,
        "cpt_at_80": _mean(row["cpt_at_80"] for row in valid),
        "phase_completion": _mean(row["phase_completion"] for row in valid),
        "rejected_transactions": _mean(row["rejected_transactions"] for row in valid),
        "fm_calls": _mean(row["fm_calls"] for row in valid),
        "input_tokens": _mean(row["input_tokens"] for row in valid),
        "output_tokens": _mean(row["output_tokens"] for row in valid),
        "model_latency_s": _mean(row["model_latency_s"] for row in valid),
        "api_cost_usd_total": (
            _sum(row["api_cost_usd"] for row in valid)
            if all_valid_costs_complete
            else None
        ),
        "wall_time_s": _mean(row["wall_time_s"] for row in valid),
    }


def _is_protocol_valid_run(
    *,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    expected_run_index: int,
    expected_seed: int,
    expected_model: str,
    output_dir: Path | None = None,
) -> bool:
    """Compatibility wrapper around the evidence-backed reconciliation audit.

    Callers that provide ``output_dir`` receive the complete formal artifact
    audit.  The optional path preserves the pre-ledger helper API for unit
    tests and non-formal tooling; ``collect_rows`` always supplies it.
    """

    return not _protocol_invalid_reasons(
        metrics=metrics,
        manifest=manifest,
        expected_run_index=expected_run_index,
        expected_seed=expected_seed,
        expected_model=expected_model,
        output_dir=output_dir,
    )


def _protocol_invalid_reasons(
    *,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    expected_run_index: int,
    expected_seed: int,
    expected_model: str,
    output_dir: Path | None = None,
) -> list[str]:
    reasons: list[str] = []
    if not metrics or not manifest:
        if not metrics:
            reasons.append("MISSING_OR_INVALID_METRICS")
        if not manifest:
            reasons.append("MISSING_OR_INVALID_RUN_MANIFEST")
        return reasons
    manifest_model = manifest.get("model", {})
    manifest_model = manifest_model if isinstance(manifest_model, dict) else {}
    protocol = manifest.get("protocol", {})
    protocol = protocol if isinstance(protocol, dict) else {}
    codebase = manifest.get("codebase", {})
    codebase = codebase if isinstance(codebase, dict) else {}
    memory = manifest.get("memory", {})
    memory = memory if isinstance(memory, dict) else {}
    status = str(manifest.get("status", ""))
    success = bool(metrics.get("task_success", False))
    physical_iterations = _integer(metrics.get("physical_iterations"))
    model_metrics = metrics.get("model", {})
    model_metrics = model_metrics if isinstance(model_metrics, dict) else {}
    model_calls = _sum_optional_ints(
        model_metrics.get("llm_calls"),
        model_metrics.get("vlm_calls"),
    )
    terminal_status_matches = (status == "SUCCESS" and success) or (
        status in {"PHYSICAL_BUDGET_EXHAUSTED", "SYSTEM_STALL"} and not success
    )
    terminal = metrics.get("terminal", {})
    terminal = terminal if isinstance(terminal, dict) else {}
    checks = (
        (protocol == asdict(PROTOCOL), "PROTOCOL_SNAPSHOT_MISMATCH"),
        (
            _integer(manifest.get("run_index")) == expected_run_index,
            "RUN_INDEX_MISMATCH",
        ),
        (_integer(manifest.get("seed")) == expected_seed, "SEED_MISMATCH"),
        (
            str(manifest_model.get("requested", "")) == expected_model,
            "REQUESTED_MODEL_MISMATCH",
        ),
        (
            str(manifest_model.get("resolved", "")) == expected_model,
            "RESOLVED_MODEL_MISMATCH",
        ),
        (manifest_model.get("identity_verified") is True, "MODEL_IDENTITY_UNVERIFIED"),
        (
            bool(str(manifest.get("evaluation_fingerprint_sha256", ""))),
            "MISSING_EVALUATION_FINGERPRINT",
        ),
        (not codebase.get("dirty_entries", []), "DIRTY_CODEBASE"),
        (memory.get("cross_run_retrieval") is False, "CROSS_RUN_MEMORY_ENABLED"),
        (
            manifest.get("budgets", {}).get("physical_transactions")
            == FORMAL_PHYSICAL_BUDGET,
            "PHYSICAL_BUDGET_MISMATCH",
        ),
        (
            manifest.get("budgets", {}).get("model_attempts")
            == PROTOCOL.model_attempt_budget,
            "MODEL_BUDGET_MISMATCH",
        ),
        (
            physical_iterations is not None
            and 0 <= physical_iterations <= FORMAL_PHYSICAL_BUDGET,
            "INVALID_PHYSICAL_ITERATION_COUNT",
        ),
        (model_calls is not None and model_calls > 0, "MISSING_MODEL_CALLS"),
        (terminal.get("codebase_unchanged") is True, "CODEBASE_CHANGED_DURING_RUN"),
        (str(terminal.get("status", "")) == status, "TERMINAL_STATUS_MISMATCH"),
        (terminal_status_matches, "TERMINAL_OUTCOME_MISMATCH"),
        (_manifest_hash_is_valid(manifest), "INVALID_MANIFEST_HASH"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    if output_dir is not None:
        reasons.extend(_terminal_invalid_reasons(status=status, terminal=terminal))
        reasons.extend(
            _artifact_protocol_invalid_reasons(
                output_dir=Path(output_dir),
                metrics=metrics,
                manifest=manifest,
            )
        )
    return list(dict.fromkeys(reasons))


def _terminal_invalid_reasons(
    *,
    status: str,
    terminal: dict[str, Any],
) -> list[str]:
    """Reject infrastructure/operator exits and incoherent method terminals."""

    allowed = {"SUCCESS", "PHYSICAL_BUDGET_EXHAUSTED", "SYSTEM_STALL"}
    reasons: list[str] = []
    if status not in allowed:
        reasons.append(f"NON_EVALUATION_TERMINAL_STATUS:{status or 'missing'}")
        return reasons
    if terminal.get("runtime_healthy_at_termination") is not True:
        reasons.append("RUNTIME_UNHEALTHY_AT_TERMINATION")
    runtime_fault = terminal.get("runtime_fault_at_termination", {})
    if isinstance(runtime_fault, dict) and runtime_fault:
        reasons.append("RUNTIME_FAULT_AT_TERMINATION")
    if status == "SUCCESS":
        if any(
            str(terminal.get(field, "")).strip()
            for field in ("error_type", "decision_terminal_error_type")
        ):
            reasons.append("SUCCESS_HAS_TERMINAL_EXCEPTION")
    elif status == "SYSTEM_STALL":
        error_types = (
            str(terminal.get("error_type", "")).strip(),
            str(terminal.get("decision_terminal_error_type", "")).strip(),
        )
        if error_types != ("SystemStallError", "SystemStallError"):
            reasons.append("SYSTEM_STALL_EXCEPTION_MISMATCH")
    return reasons


def _artifact_protocol_invalid_reasons(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    """Reconcile canonical physics, checkpoints, memory, and terminal evidence."""

    reasons: list[str] = []
    raw_events, raw_errors = _read_jsonl(output_dir / "raw_rollout_evidence.jsonl")
    evolution_records, evolution_errors = _read_jsonl(
        output_dir / "evolution_records.jsonl"
    )
    memory_events, memory_errors = _read_jsonl(
        output_dir / "memory" / "raw_rollouts.jsonl"
    )
    reasons.extend(f"RAW_ROLLOUT_LEDGER_INVALID:{error}" for error in raw_errors)
    reasons.extend(f"EVOLUTION_LEDGER_INVALID:{error}" for error in evolution_errors)
    reasons.extend(f"MEMORY_ROLLOUT_LEDGER_INVALID:{error}" for error in memory_errors)

    canonical = [
        event
        for event in raw_events
        if event.get("kind") == "physical_rollout_finalized"
        and event.get("physically_dispatched") is True
    ]
    rollout_ids = [str(event.get("rollout_id", "")).strip() for event in canonical]
    if any(not rollout_id for rollout_id in rollout_ids):
        reasons.append("CANONICAL_ROLLOUT_ID_MISSING")
    if len(rollout_ids) != len(set(rollout_ids)):
        reasons.append("DUPLICATE_CANONICAL_ROLLOUT_ID")

    journals = []
    for path in sorted(output_dir.glob("attempt_*/physical_transaction_commit.json")):
        journal = _read_json(path)
        if not journal:
            reasons.append(f"TRANSACTION_JOURNAL_INVALID:{path.parent.name}")
            continue
        journals.append((path, journal))
    finalized_journals = [
        (path, journal)
        for path, journal in journals
        if journal.get("outcome_finalized") is True
    ]
    if len(finalized_journals) != len(canonical):
        reasons.append(
            "FINALIZED_TRANSACTION_JOURNAL_COUNT_MISMATCH:"
            f"raw={len(canonical)},finalized_journals={len(finalized_journals)}"
        )

    raw_by_attempt: dict[int, dict[str, Any]] = {}
    for event in canonical:
        attempt = _integer(event.get("attempt"))
        if attempt is None or attempt in raw_by_attempt:
            reasons.append("CANONICAL_ROLLOUT_ATTEMPT_INVALID_OR_DUPLICATE")
        else:
            raw_by_attempt[attempt] = event
    memory_physical_by_id = _unique_events_by_rollout_id(
        memory_events,
        kind="physical_rollout_finalized",
        reasons=reasons,
        duplicate_reason="DUPLICATE_MEMORY_PHYSICAL_ROLLOUT_ID",
    )
    memory_update_by_id = _unique_events_by_rollout_id(
        memory_events,
        kind="memory_update_result",
        reasons=reasons,
        duplicate_reason="DUPLICATE_MEMORY_UPDATE_ROLLOUT_ID",
    )
    for path, journal in journals:
        label = path.parent.name
        stages = journal.get("stages", {})
        schema_valid = (
            journal.get("schema_version")
            == FORMAL_TRANSACTION_JOURNAL_SCHEMA_VERSION
        )
        stage_contract_valid = bool(
            isinstance(stages, dict)
            and set(stages) == set(FORMAL_TRANSACTION_JOURNAL_STAGES)
        )
        if not schema_valid:
            reasons.append(f"TRANSACTION_JOURNAL_SCHEMA_MISMATCH:{label}")
        if not stage_contract_valid:
            reasons.append(f"TRANSACTION_JOURNAL_STAGE_CONTRACT_MISMATCH:{label}")
        if (
            journal.get("status") != "sealed"
            or journal.get("outcome_finalized") is not True
            or not stage_contract_valid
            or not all(value is True for value in stages.values())
        ):
            reasons.append(f"TRANSACTION_JOURNAL_UNSEALED:{label}")
        attempt = _integer(journal.get("attempt"))
        raw = raw_by_attempt.get(attempt) if attempt is not None else None
        if raw is None:
            if journal.get("outcome_finalized") is True:
                reasons.append(f"TRANSACTION_JOURNAL_HAS_NO_RAW_ROLLOUT:{label}")
            continue
        details = journal.get("stage_details", {})
        details = details if isinstance(details, dict) else {}
        raw_detail = details.get("raw_rollout", {})
        memory_detail = details.get("memory_update", {})
        raw_detail = raw_detail if isinstance(raw_detail, dict) else {}
        memory_detail = memory_detail if isinstance(memory_detail, dict) else {}
        raw_id = str(raw.get("rollout_id", "")).strip()
        journal_raw_id = str(raw_detail.get("rollout_id", "")).strip()
        journal_memory_id = str(memory_detail.get("rollout_id", "")).strip()
        if raw_id != journal_raw_id or raw_id != journal_memory_id:
            reasons.append(f"TRANSACTION_ROLLOUT_ID_MISMATCH:{label}")
        if raw_id not in memory_physical_by_id:
            reasons.append(f"MEMORY_PHYSICAL_ROLLOUT_MISSING:{label}")
        if bool(journal.get("memory_required", False)) and raw_id not in memory_update_by_id:
            reasons.append(f"MEMORY_UPDATE_RESULT_MISSING:{label}")
        verifier = raw.get("verifier", {})
        verifier = verifier if isinstance(verifier, dict) else {}
        identity_matches = (
            str(journal.get("script_id", "")) == str(raw.get("script_id", ""))
            and str(journal.get("checkpoint_before", ""))
            == str(verifier.get("checkpoint_before", ""))
            and str(journal.get("checkpoint_after", ""))
            == str(verifier.get("checkpoint_after", ""))
        )
        if not identity_matches:
            reasons.append(f"TRANSACTION_IDENTITY_MISMATCH:{label}")

    physical_count = _integer(metrics.get("physical_iterations"))
    evolution_physical = [
        record for record in evolution_records if _evolution_record_is_physical(record)
    ]
    if physical_count is None or physical_count != len(journals):
        reasons.append(
            "PHYSICAL_DISPATCH_COUNT_MISMATCH:"
            f"journals={len(journals)},metrics={physical_count}"
        )
    if not (
        len(canonical)
        == len(evolution_physical)
        == len(finalized_journals)
    ):
        reasons.append(
            "PHYSICAL_COUNT_MISMATCH:"
            f"raw={len(canonical)},evolution={len(evolution_physical)},"
            f"finalized_journals={len(finalized_journals)}"
        )
    canonical_identities = {_rollout_identity(event) for event in canonical}
    evolution_identities = {
        _evolution_identity(record) for record in evolution_physical
    }
    if evolution_identities != canonical_identities:
        reasons.append("EVOLUTION_RAW_IDENTITY_MISMATCH")

    accepted_raw = [
        event
        for event in canonical
        if isinstance(event.get("verifier"), dict)
        and event["verifier"].get("step_accepted") is True
    ]
    accepted_outcomes = {
        str(event.get("verifier", {}).get("transition_outcome", ""))
        for event in accepted_raw
        if isinstance(event.get("verifier"), dict)
    }
    allowed_accepted_outcomes = {
        "ACCEPT_SETUP",
        "ACCEPT_PROGRESS",
        "TASK_COMPLETE",
    }
    if not accepted_outcomes.issubset(allowed_accepted_outcomes):
        reasons.append("INVALID_ACCEPTED_TRANSITION_OUTCOME")
    if str(manifest.get("status", "")) == "SUCCESS":
        final_outcome = ""
        if canonical:
            verifier = canonical[-1].get("verifier", {})
            verifier = verifier if isinstance(verifier, dict) else {}
            final_outcome = str(verifier.get("transition_outcome", ""))
        if final_outcome != "TASK_COMPLETE":
            reasons.append("SUCCESS_FINAL_ROLLOUT_NOT_TASK_COMPLETE")
    episode = _read_json(output_dir / "memory" / "accepted_episode_manifest.json")
    steps = episode.get("steps", []) if isinstance(episode, dict) else []
    steps = steps if isinstance(steps, list) else []
    accepted_metric = _integer(metrics.get("accepted_step_count"))
    if not episode:
        reasons.append("MISSING_ACCEPTED_EPISODE_MANIFEST")
    else:
        try:
            _validate_episode_manifest(episode)
        except ActionMemoryError:
            reasons.append("INVALID_ACCEPTED_EPISODE_MANIFEST_SCHEMA")
    if not (
        accepted_metric is not None
        and len(accepted_raw) == len(steps) == accepted_metric
    ):
        reasons.append(
            "ACCEPTED_COUNT_MISMATCH:"
            f"raw={len(accepted_raw)},manifest={len(steps)},metrics={accepted_metric}"
        )
    rejected_metric = _integer(metrics.get("rejected_step_count"))
    if rejected_metric is None or rejected_metric != len(canonical) - len(accepted_raw):
        reasons.append("REJECTED_COUNT_MISMATCH")
    reasons.extend(_checkpoint_lineage_invalid_reasons(accepted_raw, episode, steps))

    positive = _positive_rollout_identities(canonical)
    action_memory = _read_json(output_dir / "memory" / "action_exemplars.json")
    strategy_memory = _read_json(output_dir / "memory" / "strategy_specs.json")
    if not _is_json_object_file(output_dir / "memory" / "action_exemplars.json"):
        reasons.append("MISSING_OR_INVALID_ACTION_EXEMPLAR_MEMORY")
    else:
        try:
            VerifiedActionMemoryStore._validate_document(action_memory)
        except ActionMemoryError:
            reasons.append("INVALID_ACTION_EXEMPLAR_MEMORY_SCHEMA")
    if not _is_json_object_file(output_dir / "memory" / "strategy_specs.json"):
        reasons.append("MISSING_OR_INVALID_STRATEGY_MEMORY")
    else:
        try:
            StrategyMemoryStore._validate_document(strategy_memory)
        except (StrategyMemoryError, TypeError, ValueError):
            reasons.append("INVALID_STRATEGY_MEMORY_SCHEMA")
    reasons.extend(_action_memory_invalid_reasons(action_memory, positive))
    reasons.extend(_strategy_memory_invalid_reasons(strategy_memory, positive))

    artifact_hashes = manifest.get("artifact_hashes", {})
    artifact_hashes = artifact_hashes if isinstance(artifact_hashes, dict) else {}
    expected_memory_hash = str(
        artifact_hashes.get("final_online_memory_sha256", "")
    ).strip()
    if not expected_memory_hash or expected_memory_hash != _online_memory_sha256(
        output_dir
    ):
        reasons.append("FINAL_ONLINE_MEMORY_HASH_MISMATCH")
    result_hashes = manifest.get("result_artifact_hashes", {})
    result_hashes = result_hashes if isinstance(result_hashes, dict) else {}
    expected_summary_hash = str(
        result_hashes.get("final_summary_sha256", "")
    ).strip()
    if not expected_summary_hash or expected_summary_hash != _sha256_file(
        output_dir / "final_summary.json"
    ):
        reasons.append("FINAL_SUMMARY_HASH_MISMATCH")

    if str(manifest.get("status", "")) == "SYSTEM_STALL":
        terminal = metrics.get("terminal", {})
        terminal = terminal if isinstance(terminal, dict) else {}
        reasons.extend(
            _system_stall_evidence_invalid_reasons(
                output_dir,
                terminal=terminal,
            )
        )
    if (
        str(manifest.get("status", "")) == "PHYSICAL_BUDGET_EXHAUSTED"
        and physical_count != FORMAL_PHYSICAL_BUDGET
    ):
        reasons.append("PHYSICAL_BUDGET_TERMINAL_BEFORE_BUDGET_EXHAUSTION")
    return reasons


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [type(exc).__name__]
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line_{index}_json")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line_{index}_not_object")
            continue
        records.append(payload)
    return records, errors


def _is_json_object_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, json.JSONDecodeError):
        return False


def _unique_events_by_rollout_id(
    events: list[dict[str, Any]],
    *,
    kind: str,
    reasons: list[str],
    duplicate_reason: str,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("kind") != kind:
            continue
        rollout_id = str(event.get("rollout_id", "")).strip()
        if not rollout_id:
            reasons.append(f"{kind.upper()}_ID_MISSING")
        elif rollout_id in selected:
            reasons.append(duplicate_reason)
        else:
            selected[rollout_id] = event
    return selected


def _evolution_record_is_physical(record: dict[str, Any]) -> bool:
    evaluation = record.get("evaluation", {})
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    evidence = evaluation.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    checkpoint = evidence.get("checkpoint_outcome", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    if "physically_dispatched" in checkpoint:
        return checkpoint.get("physically_dispatched") is True
    controller = record.get("controller_decision", {})
    controller = controller if isinstance(controller, dict) else {}
    return str(controller.get("status", "")).startswith("physical_transaction_")


def _evolution_identity(
    record: dict[str, Any],
) -> tuple[int | None, str, str, str]:
    policy = record.get("policy", {})
    policy = policy if isinstance(policy, dict) else {}
    return (
        _integer(policy.get("attempt")),
        str(policy.get("script_id", "")),
        str(record.get("checkpoint_before", "")),
        str(record.get("checkpoint_after", "")),
    )


def _rollout_identity(payload: dict[str, Any]) -> tuple[int | None, str, str, str]:
    verifier = payload.get("verifier", {})
    verifier = verifier if isinstance(verifier, dict) else {}
    return (
        _integer(payload.get("attempt")),
        str(payload.get("script_id", "")),
        str(verifier.get("checkpoint_before", "")),
        str(verifier.get("checkpoint_after", "")),
    )


def _step_identity(payload: dict[str, Any]) -> tuple[int | None, str, str, str]:
    return (
        _integer(payload.get("attempt")),
        str(payload.get("script_id", "")),
        str(payload.get("checkpoint_before", "")),
        str(payload.get("checkpoint_after", "")),
    )


def _checkpoint_lineage_invalid_reasons(
    accepted_raw: list[dict[str, Any]],
    episode: dict[str, Any],
    steps: list[Any],
) -> list[str]:
    reasons: list[str] = []
    raw_identities = {_rollout_identity(event) for event in accepted_raw}
    step_objects = [step for step in steps if isinstance(step, dict)]
    if len(step_objects) != len(steps):
        reasons.append("ACCEPTED_MANIFEST_STEP_INVALID")
    if {_step_identity(step) for step in step_objects} != raw_identities:
        reasons.append("ACCEPTED_MANIFEST_RAW_IDENTITY_MISMATCH")
    previous = str(episode.get("initial_checkpoint", ""))
    for index, step in enumerate(step_objects, start=1):
        expected_after = f"accepted_step_{index:02d}"
        if (
            _integer(step.get("index")) != index
            or str(step.get("checkpoint_before", "")) != previous
            or str(step.get("checkpoint_after", "")) != expected_after
        ):
            reasons.append("ACCEPTED_CHECKPOINT_LINEAGE_INVALID")
            break
        previous = expected_after
    expected_latest = previous or "accepted_step_00"
    if str(episode.get("latest_checkpoint", "")) != expected_latest:
        reasons.append("LATEST_CHECKPOINT_MISMATCH")
    return reasons


def _positive_rollout_identities(
    canonical: list[dict[str, Any]],
) -> set[tuple[int | None, str, str, str]]:
    positive = set()
    for event in canonical:
        verifier = event.get("verifier", {})
        verifier = verifier if isinstance(verifier, dict) else {}
        if (
            verifier.get("step_accepted") is True
            and str(verifier.get("transition_outcome", ""))
            in {"ACCEPT_PROGRESS", "TASK_COMPLETE"}
        ):
            positive.add(_rollout_identity(event))
    return positive


def _action_memory_invalid_reasons(
    memory: dict[str, Any],
    positive: set[tuple[int | None, str, str, str]],
) -> list[str]:
    exemplars = memory.get("exemplars", []) if isinstance(memory, dict) else []
    if not isinstance(exemplars, list):
        return ["ACTION_EXEMPLAR_MEMORY_INVALID"]
    reasons: list[str] = []
    for exemplar in exemplars:
        source = exemplar.get("source", {}) if isinstance(exemplar, dict) else {}
        source = source if isinstance(source, dict) else {}
        if _step_identity(source) not in positive:
            reasons.append("ACTION_EXEMPLAR_NOT_FROM_POSITIVE_CANONICAL_ROLLOUT")
            break
    return reasons


def _strategy_memory_invalid_reasons(
    memory: dict[str, Any],
    positive: set[tuple[int | None, str, str, str]],
) -> list[str]:
    specs = memory.get("specs", []) if isinstance(memory, dict) else []
    if not isinstance(specs, list):
        return ["STRATEGY_MEMORY_INVALID"]
    for spec in specs:
        evidence = spec.get("evidence", {}) if isinstance(spec, dict) else {}
        evidence = evidence if isinstance(evidence, dict) else {}
        refs = evidence.get("rollout_refs", [])
        refs = refs if isinstance(refs, list) else []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            if str(ref.get("outcome", "")).lower() in {"accepted", "success", "positive"}:
                if _step_identity(ref) not in positive:
                    return ["STRATEGY_POSITIVE_EVIDENCE_NOT_FROM_CANONICAL_PROGRESS"]
    return []


def _system_stall_evidence_invalid_reasons(
    output_dir: Path,
    *,
    terminal: dict[str, Any],
) -> list[str]:
    payload = _read_json(output_dir / "generation_stalls.json")
    events = payload.get("events", []) if isinstance(payload, dict) else []
    events = events if isinstance(events, list) else []
    tail = [event for event in events[-3:] if isinstance(event, dict)]
    if len(tail) != 3:
        return ["SYSTEM_STALL_MISSING_THREE_COMPLETE_CYCLES"]
    checkpoints = {str(event.get("checkpoint_id", "")) for event in tail}
    cycles = [_integer(event.get("generation_cycle")) for event in tail]
    consecutive_cycles = bool(
        all(cycle is not None for cycle in cycles)
        and cycles == list(range(int(cycles[0]), int(cycles[0]) + 3))
    )
    inert = all(
        event.get("physical_dispatch") is False
        and event.get("checkpoint_change") is False
        for event in tail
    )
    common_complete = (
        len(checkpoints) == 1
        and "" not in checkpoints
        and consecutive_cycles
        and inert
    )
    outcomes = [str(event.get("cycle_outcome", "")) for event in tail]
    infeasible_levels = {
        str(event.get("selected_mutation_level", "")).strip() for event in tail
    }
    known_levels = {
        "parameter_tuning",
        "coordination_mutation",
        "strategy_mutation",
    }
    all_levels_proved_empty = bool(
        outcomes == ["MUTATION_SCOPE_INFEASIBLE"] * 3
        and infeasible_levels == known_levels
    )
    complete = bool(
        common_complete
        and tail[-1].get("system_stall") is True
        and all(
            event.get("counts_as_model_stall") is True
            or event.get("cycle_outcome") == "MUTATION_SCOPE_INFEASIBLE"
            for event in tail
        )
        and not all_levels_proved_empty
    )
    return [] if complete else ["SYSTEM_STALL_EVIDENCE_INCOMPLETE_OR_INCOHERENT"]


def _billing_call_identity(scope: str, record: dict[str, Any]) -> tuple[str, ...]:
    """Return the stable fields shared by model usage and the flat ledger."""

    return (
        str(scope),
        str(record.get("kind", "")),
        str(record.get("role", "")),
        str(record.get("artifact_id", "")),
        json.dumps(record.get("response_index"), sort_keys=True),
        str(record.get("request_id", "")).strip(),
        str(record.get("requested_model", "")),
        str(record.get("resolved_model", "")),
        json.dumps(record.get("started_at_unix"), sort_keys=True),
    )


def _successful_call_lacks_request_id(record: dict[str, Any]) -> bool:
    """Reject a completed response disguised as a zero-cost transport round."""

    if str(record.get("request_id", "")).strip():
        return False
    status = _integer(record.get("http_status"))
    token_evidence = any(
        (_integer(record.get(field)) or 0) > 0
        for field in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "image_input_tokens",
            "reasoning_tokens",
        )
    )
    response_identity = any(
        str(record.get(field, "")).strip()
        for field in ("resolved_model", "response_model", "canonical_resolved_model")
    )
    return bool(
        (status is not None and 200 <= status < 300)
        or token_evidence
        or response_identity
        or record.get("billing_amount_microusd") is not None
    )


def _money_matches(value: object, amount_microusd: int) -> bool:
    number = _number(value)
    return bool(
        number is not None
        and math.isclose(
            number,
            amount_microusd / 1_000_000.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )


def _billing_artifact_invalid_reasons(
    *,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Cross-check every provider call and exact charge across all artifacts."""

    target = Path(output_dir)
    usage = _read_json(target / "model_usage.json")
    ledger = _read_json(target / "api_call_costs.json")
    reconciliation = _read_json(target / "billing_reconciliation.json")
    post = _read_json(target / "billing_post_reconciliation.json")
    missing = [
        name
        for name, payload in (
            ("MODEL_USAGE", usage),
            ("API_COST_LEDGER", ledger),
            ("BILLING_RECONCILIATION", reconciliation),
            ("BILLING_POST_RECONCILIATION", post),
        )
        if not payload
    ]
    if missing:
        return [f"MISSING_OR_INVALID_{name}" for name in missing]

    reasons: list[str] = []
    usage_records: list[tuple[str, dict[str, Any]]] = []
    scope_counts: dict[str, int] = {}
    for scope in ("text", "vision"):
        bucket = usage.get(scope, {})
        if not isinstance(bucket, dict) or not isinstance(bucket.get("calls"), list):
            reasons.append(f"MODEL_USAGE_{scope.upper()}_SCHEMA_INVALID")
            continue
        records = bucket["calls"]
        if not all(isinstance(record, dict) for record in records):
            reasons.append(f"MODEL_USAGE_{scope.upper()}_CALL_INVALID")
            continue
        scope_counts[scope] = len(records)
        usage_records.extend((scope, record) for record in records)
        completed = [
            record
            for record in records
            if str(record.get("request_id", "")).strip()
        ]
        billed = [
            record
            for record in completed
            if record.get("billing_amount_microusd") is not None
        ]
        amount = sum(
            int(record.get("billing_amount_microusd", 0) or 0)
            for record in billed
        )
        expected = {
            "call_count": len(records),
            "billed_call_count": len(billed),
            "unbilled_call_count": len(completed) - len(billed),
            "failed_transport_round_count": len(records) - len(completed),
            "billing_amount_microusd": amount,
        }
        for field, value in expected.items():
            if _integer(bucket.get(field)) != value:
                reasons.append(f"MODEL_USAGE_{scope.upper()}_{field.upper()}_MISMATCH")
        if not _money_matches(bucket.get("billing_amount_usd"), amount):
            reasons.append(f"MODEL_USAGE_{scope.upper()}_USD_MISMATCH")

    usage_request_ids = [
        str(record.get("request_id", "")).strip()
        for _, record in usage_records
        if str(record.get("request_id", "")).strip()
    ]
    if len(usage_request_ids) != len(set(usage_request_ids)):
        reasons.append("MODEL_USAGE_DUPLICATE_REQUEST_ID")
    if any(_successful_call_lacks_request_id(record) for _, record in usage_records):
        reasons.append("COMPLETED_MODEL_RESPONSE_MISSING_REQUEST_ID")

    ledger_calls = ledger.get("calls", [])
    if not isinstance(ledger_calls, list) or not all(
        isinstance(record, dict) for record in ledger_calls
    ):
        reasons.append("API_COST_LEDGER_CALL_SCHEMA_INVALID")
        ledger_calls = []
    usage_identities = Counter(
        _billing_call_identity(scope, record) for scope, record in usage_records
    )
    ledger_identities = Counter(
        _billing_call_identity(str(record.get("api_scope", "")), record)
        for record in ledger_calls
    )
    if usage_identities != ledger_identities:
        reasons.append("MODEL_USAGE_LEDGER_CALL_IDENTITY_MISMATCH")

    ledger_request_ids = [
        str(record.get("request_id", "")).strip()
        for record in ledger_calls
        if str(record.get("request_id", "")).strip()
    ]
    if len(ledger_request_ids) != len(set(ledger_request_ids)):
        reasons.append("API_COST_LEDGER_DUPLICATE_REQUEST_ID")
    if Counter(usage_request_ids) != Counter(ledger_request_ids):
        reasons.append("MODEL_USAGE_LEDGER_REQUEST_ID_MISMATCH")

    usage_charges = {
        str(record.get("request_id", "")).strip(): _integer(
            record.get("billing_amount_microusd")
        )
        for _, record in usage_records
        if str(record.get("request_id", "")).strip()
    }
    ledger_charges = {
        str(record.get("request_id", "")).strip(): _integer(
            record.get("billing_amount_microusd")
        )
        for record in ledger_calls
        if str(record.get("request_id", "")).strip()
    }
    if usage_charges != ledger_charges:
        reasons.append("MODEL_USAGE_LEDGER_CALL_CHARGE_MISMATCH")

    completed = 0
    billed = 0
    pending = 0
    failed = 0
    amount_microusd = 0
    for record in ledger_calls:
        request_id = str(record.get("request_id", "")).strip()
        amount = _integer(record.get("billing_amount_microusd"))
        if request_id and amount is not None:
            completed += 1
            billed += 1
            amount_microusd += amount
            expected_status = "exact_provider_billing"
            if not _money_matches(record.get("billing_amount_usd"), amount):
                reasons.append("API_COST_LEDGER_CALL_USD_MISMATCH")
        elif request_id:
            completed += 1
            pending += 1
            expected_status = "pending_provider_billing"
        else:
            failed += 1
            expected_status = "zero_no_completed_model_response"
            if amount != 0 or not _money_matches(record.get("billing_amount_usd"), 0):
                reasons.append("API_COST_LEDGER_FAILED_ROUND_NONZERO_CHARGE")
        if str(record.get("charge_status", "")) != expected_status:
            reasons.append("API_COST_LEDGER_CHARGE_STATUS_MISMATCH")

    summary = ledger.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    expected_summary = {
        "recorded_api_rounds": len(ledger_calls),
        "completed_model_calls": completed,
        "failed_transport_rounds": failed,
        "exact_billed_call_count": billed,
        "pending_billing_call_count": pending,
        "billing_amount_microusd": amount_microusd,
    }
    for field, value in expected_summary.items():
        if _integer(summary.get(field)) != value:
            reasons.append(f"API_COST_LEDGER_{field.upper()}_MISMATCH")
    if summary.get("billing_complete") is not (pending == 0):
        reasons.append("API_COST_LEDGER_COMPLETE_FLAG_MISMATCH")
    if not _money_matches(summary.get("billing_amount_usd"), amount_microusd):
        reasons.append("API_COST_LEDGER_USD_MISMATCH")
    if ledger.get("kind") != "coevolve_api_call_cost_ledger":
        reasons.append("API_COST_LEDGER_KIND_MISMATCH")
    if _integer(ledger.get("schema_version")) != 1 or ledger.get("currency") != "USD":
        reasons.append("API_COST_LEDGER_SCHEMA_OR_CURRENCY_MISMATCH")

    model_metrics = metrics.get("model", {})
    model_metrics = model_metrics if isinstance(model_metrics, dict) else {}
    expected_metrics = {
        "llm_calls": scope_counts.get("text"),
        "vlm_calls": scope_counts.get("vision"),
        "billed_call_count": billed,
        "unbilled_call_count": pending,
        "failed_transport_round_count": failed,
        "billing_amount_microusd": amount_microusd,
    }
    for field, value in expected_metrics.items():
        if value is None or _integer(model_metrics.get(field)) != value:
            reasons.append(f"METRICS_MODEL_{field.upper()}_MISMATCH")
    if not _money_matches(model_metrics.get("billing_amount_usd"), amount_microusd):
        reasons.append("METRICS_MODEL_BILLING_AMOUNT_USD_MISMATCH")
    if model_metrics.get("billing_reconciliation_status") != "complete":
        reasons.append("METRICS_MODEL_BILLING_STATUS_INCOMPLETE")

    if usage.get("billing_reconciliation") != reconciliation:
        reasons.append("MODEL_USAGE_RECONCILIATION_REPORT_MISMATCH")
    if manifest.get("model_usage") != usage:
        reasons.append("MANIFEST_MODEL_USAGE_MISMATCH")
    for label, report in (
        ("BILLING_RECONCILIATION", reconciliation),
        ("BILLING_POST_RECONCILIATION", post),
    ):
        if (
            report.get("status") != "complete"
            or _integer(report.get("remaining")) != 0
            or report.get("ledger_billing_complete") is not True
            or _integer(report.get("billing_amount_microusd"))
            != amount_microusd
            or not _money_matches(report.get("billing_amount_usd"), amount_microusd)
        ):
            reasons.append(f"{label}_TOTALS_MISMATCH")
    return list(dict.fromkeys(reasons))


def _is_cost_complete(
    *,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path | None = None,
) -> bool:
    """Accounting completeness is independent of evaluation validity."""

    if not metrics or not manifest:
        return False
    model_metrics = metrics.get("model", {})
    model_metrics = model_metrics if isinstance(model_metrics, dict) else {}
    unbilled_calls = _integer(model_metrics.get("unbilled_call_count"))
    base_complete = bool(
        unbilled_calls == 0
        and model_metrics.get("billing_reconciliation_status") == "complete"
        and bool(
            manifest.get("result_artifact_hashes", {}).get(
                "billing_reconciliation_sha256"
            )
        )
    )
    if not base_complete or output_dir is None:
        return base_complete

    target = Path(output_dir)
    ledger = _read_json(target / "api_call_costs.json")
    post = _read_json(target / "billing_post_reconciliation.json")
    ledger_summary = ledger.get("summary", {})
    ledger_summary = ledger_summary if isinstance(ledger_summary, dict) else {}
    result_hashes = manifest.get("result_artifact_hashes", {})
    result_hashes = result_hashes if isinstance(result_hashes, dict) else {}
    required_hashes = {
        "billing_reconciliation_sha256": target / "billing_reconciliation.json",
        "billing_post_reconciliation_sha256": (
            target / "billing_post_reconciliation.json"
        ),
        "api_call_costs_json_sha256": target / "api_call_costs.json",
        "api_call_costs_csv_sha256": target / "api_call_costs.csv",
        "model_usage_sha256": target / "model_usage.json",
        "metrics_sha256": target / "metrics.json",
    }
    hashes_complete = bool(
        ledger_summary.get("billing_complete") is True
        and _integer(ledger_summary.get("pending_billing_call_count")) == 0
        and post.get("ledger_billing_complete") is True
        and all(
            str(result_hashes.get(key, "")) == _sha256_file(path)
            and bool(str(result_hashes.get(key, "")))
            for key, path in required_hashes.items()
        )
    )
    return bool(
        hashes_complete
        and not _billing_artifact_invalid_reasons(
            metrics=metrics,
            manifest=manifest,
            output_dir=target,
        )
    )


def _manifest_hash_is_valid(manifest: dict[str, Any]) -> bool:
    expected = str(manifest.get("manifest_sha256", ""))
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return bool(expected) and hashlib.sha256(encoded).hexdigest() == expected


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _online_memory_sha256(output_dir: Path) -> str:
    """Recompute the exact memory digest written by the formal manifest."""

    files = {
        path.name: _sha256_file(path)
        for path in sorted(
            formal_memory_paths(Path(output_dir)).values(),
            key=lambda item: item.name,
        )
    }
    encoded = json.dumps(
        files,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _wilson_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - margin, center + margin


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _integer(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _number(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _sum_optional_ints(*values: object) -> int | None:
    converted = [_integer(value) for value in values]
    present = [value for value in converted if value is not None]
    return sum(present) if present else None


def _mean(values: Iterable[object]) -> float | None:
    present = [number for value in values if (number := _number(value)) is not None]
    return statistics.fmean(present) if present else None


def _sum(values: Iterable[object]) -> float | None:
    present = [number for value in values if (number := _number(value)) is not None]
    return sum(present) if present else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, ensure_ascii=True, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def _display(value: object) -> str:
    number = _number(value)
    return "NA" if number is None else f"{number:.3f}"


if __name__ == "__main__":
    main()
