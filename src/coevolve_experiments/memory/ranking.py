"""Frozen semantic deduplication and ranking policy for strategy candidates."""

from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

from .candidate_extraction import StrategyCandidate
from .schema import FROZEN_STRATEGY_BANK_POLICY


class CandidateDisposition(TypedDict):
    source_index: int
    source_record_index: int
    kept: bool
    reason: str


def deduplicate_and_cap(
    candidates: list[StrategyCandidate],
) -> tuple[list[StrategyCandidate], list[CandidateDisposition]]:
    """Construct a bounded frozen bank; this is not retrieval-time top-k."""

    groups: dict[str, list[StrategyCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.digest].append(candidate)

    representatives: list[tuple[StrategyCandidate, int, int]] = []
    dispositions: dict[tuple[int, int], CandidateDisposition] = {}
    for digest, group in groups.items():
        ordered = sorted(
            group,
            key=lambda item: (
                item.source_index,
                item.source_record_index,
                item.source_record_id,
            ),
        )
        representative = ordered[0]
        run_support = len({item.source_index for item in group})
        evidence = sum(item.evidence_weight for item in group)
        representatives.append((representative, run_support, evidence))
        dispositions[
            (representative.source_index, representative.source_record_index)
        ] = {
            "source_index": representative.source_index,
            "source_record_index": representative.source_record_index,
            "kept": True,
            "reason": "verifier_backed_unique_record",
        }
        for duplicate in ordered[1:]:
            dispositions[
                (duplicate.source_index, duplicate.source_record_index)
            ] = {
                "source_index": duplicate.source_index,
                "source_record_index": duplicate.source_record_index,
                "kept": False,
                "reason": f"deterministic_semantic_duplicate_of:{digest}",
            }

    ranked = sorted(
        representatives,
        key=lambda item: (-item[1], -item[2], item[0].digest),
    )
    rejected = [
        item
        for item in ranked
        if item[0].record["verified_outcome"] == "rejected"
    ]
    rejected_cap = FROZEN_STRATEGY_BANK_POLICY.maximum_rejected_counterexamples
    allowed_rejected = {item[0].digest for item in rejected[:rejected_cap]}
    for candidate, _support, _evidence in rejected[rejected_cap:]:
        disposition = dispositions[
            (candidate.source_index, candidate.source_record_index)
        ]
        disposition.update(
            kept=False,
            reason="frozen_rejected_counterexample_cap",
        )

    eligible = [
        item
        for item in ranked
        if item[0].record["verified_outcome"] == "accepted"
        or item[0].digest in allowed_rejected
    ]
    record_cap = FROZEN_STRATEGY_BANK_POLICY.maximum_records
    selected_items = eligible[:record_cap]
    selected_digests = {item[0].digest for item in selected_items}
    for candidate, _support, _evidence in eligible[record_cap:]:
        disposition = dispositions[
            (candidate.source_index, candidate.source_record_index)
        ]
        disposition.update(kept=False, reason="frozen_total_record_cap")

    # If a representative is removed by a cap, its duplicates stay marked as
    # duplicates; the sidecar still identifies the canonical group digest.
    selected = [
        item[0]
        for item in selected_items
        if item[0].digest in selected_digests
    ]
    return selected, list(dispositions.values())


__all__ = ["CandidateDisposition", "deduplicate_and_cap"]
