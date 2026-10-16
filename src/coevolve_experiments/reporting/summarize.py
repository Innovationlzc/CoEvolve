"""Aggregate Tables 2--4 and Appendix A1--A3 from reconciled runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Iterable

from .collect import RunRecord
from .metrics import mean, median_iqr, sample_std, wilson_interval


PAPER_RUN_SEEDS = (200, 201, 202, 203, 204)


class IncompleteConditionError(RuntimeError):
    """Raised instead of silently changing the registered denominator."""


class MixedConditionError(ValueError):
    """Raised when one condition call spans more than one frozen cohort."""


_COHORT_FIELDS = (
    "table",
    "condition",
    "protocol_id",
    "target_model",
    "provider",
    "provider_route_id",
    "provider_route_fingerprint",
    "source_model",
    "method",
    "scene",
    "memory_condition",
    "base_method_commit",
    "condition_fingerprint",
    "invariant_fingerprint",
    "scene_fingerprint",
    "terminal_fingerprint",
    "observation_contract_fingerprint",
    "task_instruction_fingerprint",
    "shared_prompt_contract_fingerprint",
    "decision_prompt_fingerprint",
    "skill_api_fingerprint",
    "full_method_compatibility_fingerprint",
    "strategy_bank_sha256",
)


def summarize_condition(
    records: Iterable[RunRecord],
    *,
    expected_runs: int = 5,
    expected_seeds: Iterable[int] | None = None,
    physical_budget: int = 80,
    require_complete: bool = False,
) -> dict[str, Any]:
    rows = list(records)
    identities = {_cohort_key(row) for row in rows}
    if len(identities) > 1:
        varying = [
            field
            for index, field in enumerate(_COHORT_FIELDS)
            if len({identity[index] for identity in identities}) > 1
        ]
        raise MixedConditionError(
            "condition rows have incompatible frozen identities: "
            f"varying={varying}"
        )
    valid = [row for row in rows if row.protocol_valid]
    invalid_infra = [row for row in rows if row.infrastructure_invalid]
    seeds = [row.seed for row in valid if row.seed is not None]
    duplicate_seeds = sorted(seed for seed, count in Counter(seeds).items() if count > 1)
    expected_seed_tuple = _resolve_expected_seeds(expected_runs, expected_seeds)
    expected_seed_set = set(expected_seed_tuple)
    observed_seed_set = set(seeds)
    complete = (
        len(valid) == expected_runs
        and not duplicate_seeds
        and observed_seed_set == expected_seed_set
    )
    if require_complete and not complete:
        raise IncompleteConditionError(
            "condition is not complete: "
            f"valid={len(valid)}/{expected_runs}, duplicate_seeds={duplicate_seeds}, "
            f"observed_seeds={sorted(observed_seed_set)}, "
            f"infrastructure_invalid={len(invalid_infra)}"
        )
    observed_successes = sum(row.task_success for row in valid)
    successes = observed_successes if complete else None
    success_rate = observed_successes / len(valid) if complete else None
    low, high = (
        wilson_interval(observed_successes, len(valid))
        if complete
        else (None, None)
    )
    capped = [
        min(int(row.physical_transactions or 0), physical_budget)
        if row.task_success and row.physical_transactions is not None
        else physical_budget
        for row in valid
    ]
    phase_values = [row.phase_completion for row in valid]
    rejected_values = [row.rejected_transactions for row in valid]
    fm_call_values = [row.fm_calls for row in valid]
    billing_complete_runs = sum(
        row.cost_complete and row.cost_usd is not None for row in valid
    )
    billing_complete = bool(
        complete and valid and billing_complete_runs == len(valid)
    )
    condition = rows[0].condition if rows else ""
    result = {
        "condition": condition,
        "table": rows[0].table if rows else "",
        "cohort_identity": _cohort_identity(rows[0]) if rows else {},
        "expected_runs": len(expected_seed_tuple),
        "observed_attempts": len(rows),
        "protocol_valid_runs": len(valid),
        "infrastructure_invalid_attempts": len(invalid_infra),
        "complete": complete,
        "duplicate_valid_seeds": duplicate_seeds,
        "expected_seeds": sorted(expected_seed_set),
        "observed_valid_seeds": sorted(observed_seed_set),
        "successful_runs": successes,
        "observed_successful_runs": observed_successes,
        "success_over_expected": (
            f"{observed_successes}/{len(expected_seed_tuple)}" if complete else None
        ),
        "success_rate": success_rate,
        "success_rate_wilson_95_low": low,
        "success_rate_wilson_95_high": high,
        "cpt_at_80": mean(capped) if complete else None,
        "cpt_at_80_std": sample_std(capped) if complete else None,
        "phase_completion": mean(phase_values) if complete else None,
        "phase_completion_std": sample_std(phase_values) if complete else None,
        "rejected_transactions": mean(rejected_values) if complete else None,
        "rejected_transactions_std": sample_std(rejected_values) if complete else None,
        "fm_calls": mean(fm_call_values) if complete else None,
        "fm_calls_std": sample_std(fm_call_values) if complete else None,
        "billing_complete": billing_complete,
        "billing_complete_runs": billing_complete_runs,
        "distribution": (
            {
                "cpt_at_80": median_iqr(capped),
                "phase_completion": median_iqr(phase_values),
                "rejected_transactions": median_iqr(rejected_values),
                "fm_calls": median_iqr(fm_call_values),
            }
            if complete
            else None
        ),
        "cost_usd_total": (
            sum(
                float(row.cost_usd)
                for row in valid
                if row.cost_usd is not None
            )
            if billing_complete
            else None
        ),
        "observed_partial_diagnostics": (
            None
            if complete
            else {
                "valid_runs": len(valid),
                "successful_runs": observed_successes,
                "cpt_at_80": mean(capped),
                "phase_completion": mean(phase_values),
                "rejected_transactions": mean(rejected_values),
                "fm_calls": mean(fm_call_values),
                "billing_complete_runs": billing_complete_runs,
            }
        ),
    }
    return result


def summarize_suite(
    records: Iterable[RunRecord],
    *,
    expected_runs: int = 5,
    expected_seeds: Iterable[int] | None = None,
    physical_budget: int = 80,
    require_complete: bool = False,
) -> dict[str, Any]:
    rows = list(records)
    grouped: dict[tuple[str, ...], list[RunRecord]] = {}
    for row in rows:
        grouped.setdefault(_cohort_key(row), []).append(row)
    summaries = [
        summarize_condition(
            grouped[key],
            expected_runs=expected_runs,
            expected_seeds=expected_seeds,
            physical_budget=physical_budget,
            require_complete=require_complete,
        )
        for key in sorted(grouped)
    ]
    return {
        "kind": "coevolve_non_table1_summary",
        "conditions": summaries,
        "appendix": build_appendix_tables(rows),
    }


def build_appendix_tables(records: Iterable[RunRecord]) -> dict[str, Any]:
    rows = list(records)
    a1 = [
        {
            **asdict(row),
            "failure_counts": dict(row.failure_counts),
            "mutation_counts": dict(row.mutation_counts),
            "artifact_hashes": dict(row.artifact_hashes),
        }
        for row in rows
        if row.protocol_valid
    ]
    infra_by_cell: dict[tuple[Any, ...], int] = Counter(
        (*_cohort_key(row), row.seed)
        for row in rows
        if row.infrastructure_invalid
    )
    infra_seen: Counter[tuple[Any, ...]] = Counter()
    a2 = []
    for row in rows:
        if not row.infrastructure_invalid:
            continue
        cell = (*_cohort_key(row), row.seed)
        infra_seen[cell] += 1
        a2.append(
            {
                "run_dir": row.run_dir,
                "table": row.table,
                "condition": row.condition,
                "model": row.target_model,
                "seed": row.seed,
                "cohort_identity": _cohort_identity(row),
                "terminal_reason": row.terminal_status,
                "invalid_reasons": list(row.invalid_reasons),
                "invalid_attempt_index": infra_seen[cell],
                "invalid_attempt_count_for_cell": infra_by_cell[cell],
            }
        )
    failure_counts: Counter[str] = Counter()
    mutation_counts: Counter[str] = Counter()
    for row in rows:
        if not row.protocol_valid:
            continue
        failure_counts.update(row.failure_counts)
        mutation_counts.update(row.mutation_counts)
    return {
        "A1_per_run_results": a1,
        "A2_infrastructure_invalid_attempts": a2,
        "protocol_invalid_non_infrastructure": [
            {
                "run_dir": row.run_dir,
                "table": row.table,
                "condition": row.condition,
                "seed": row.seed,
                "cohort_identity": _cohort_identity(row),
                "terminal_reason": row.terminal_status,
                "invalid_reasons": list(row.invalid_reasons),
            }
            for row in rows
            if not row.protocol_valid and not row.infrastructure_invalid
        ],
        "A3_failure_and_evolution_statistics": {
            "failure_counts": dict(sorted(failure_counts.items())),
            "mutation_counts": dict(sorted(mutation_counts.items())),
        },
    }


def _cohort_key(row: RunRecord) -> tuple[str, ...]:
    return tuple(str(getattr(row, field)) for field in _COHORT_FIELDS)


def _resolve_expected_seeds(
    expected_runs: int,
    expected_seeds: Iterable[int] | None,
) -> tuple[int, ...]:
    if isinstance(expected_runs, bool) or not isinstance(expected_runs, int):
        raise TypeError("expected_runs must be an integer")
    if expected_runs <= 0:
        raise ValueError("expected_runs must be positive")
    if expected_seeds is None:
        # This module is paper-facing for the post-Table-1 matrix.  Keep the
        # old range behavior only for explicitly non-five-run diagnostics.
        seeds = PAPER_RUN_SEEDS if expected_runs == len(PAPER_RUN_SEEDS) else tuple(
            range(expected_runs)
        )
    else:
        seeds = tuple(expected_seeds)
    if len(seeds) != expected_runs:
        raise ValueError("expected_seeds length differs from expected_runs")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("expected_seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must be unique")
    return seeds


def _cohort_identity(row: RunRecord) -> dict[str, str]:
    return {field: str(getattr(row, field)) for field in _COHORT_FIELDS}
