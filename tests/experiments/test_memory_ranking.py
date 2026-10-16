from __future__ import annotations

from coevolve_experiments.memory.candidate_extraction import StrategyCandidate
from coevolve_experiments.memory.ranking import deduplicate_and_cap
from coevolve_experiments.memory.retrieval import select_strategy_records


def _candidate(
    index: int,
    outcome: str,
    *,
    evidence_weight: int,
) -> StrategyCandidate:
    return StrategyCandidate(
        record={
            "phase_trigger": f"phase token {index}",
            "failure_condition": "verifier state",
            "coordination_pattern": f"role-compatible support pattern {index}",
            "verified_outcome": outcome,  # type: ignore[typeddict-item]
            "avoid_rule": "avoid unsupported motion",
            "repair_direction": "preserve cooperative support",
        },
        source_index=0,
        source_record_index=index,
        source_record_id=f"record-{index}",
        evidence_weight=evidence_weight,
        sanitization_notes=(),
    )


def test_frozen_ranking_applies_rejected_and_total_caps() -> None:
    candidates = [
        _candidate(index, "rejected", evidence_weight=100 - index)
        for index in range(20)
    ] + [
        _candidate(index + 20, "accepted", evidence_weight=1)
        for index in range(40)
    ]

    selected, dispositions = deduplicate_and_cap(candidates)

    assert len(selected) == 48
    assert sum(
        item.record["verified_outcome"] == "rejected" for item in selected
    ) == 16
    reasons = [item["reason"] for item in dispositions]
    assert reasons.count("frozen_rejected_counterexample_cap") == 4
    assert reasons.count("frozen_total_record_cap") == 8


def test_semantic_duplicates_keep_stable_source_representative() -> None:
    later = _candidate(1, "accepted", evidence_weight=9)
    earlier = StrategyCandidate(
        record=dict(later.record),  # type: ignore[arg-type]
        source_index=0,
        source_record_index=0,
        source_record_id="earlier",
        evidence_weight=1,
        sanitization_notes=(),
    )

    selected, dispositions = deduplicate_and_cap([later, earlier])

    assert selected == [earlier]
    assert {item["reason"] for item in dispositions} == {
        "verifier_backed_unique_record",
        f"deterministic_semantic_duplicate_of:{earlier.digest}",
    }


def test_frozen_bank_cap_is_distinct_from_v83_query_top_three() -> None:
    candidates = [
        _candidate(index, "accepted", evidence_weight=100 - index)
        for index in range(60)
    ]
    selected, _dispositions = deduplicate_and_cap(candidates)
    exact_phase_records = []
    for candidate in selected:
        record = dict(candidate.record)
        record["phase_trigger"] = "during supported payload transport"
        exact_phase_records.append(record)

    retrieved = select_strategy_records(
        exact_phase_records,
        active_phase="carry",
    )

    assert len(selected) == 48  # construction-time frozen bank bound
    assert len(retrieved) == 3  # independent V83 query-time bound
    assert [item["coordination_pattern"] for item in retrieved] == [
        "role-compatible support pattern 47",
        "role-compatible support pattern 46",
        "role-compatible support pattern 45",
    ]
