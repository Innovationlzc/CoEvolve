from __future__ import annotations

import pytest

from coevolve_experiments.memory.leak_check import scan_memory_leaks
from coevolve_experiments.memory.online_memory import (
    OnlineMemoryValidationError,
    strategy_entry_from_rollout,
    validate_online_strategy_overlay,
)
from coevolve_experiments.memory.retrieval import (
    canonical_strategy_phase,
    select_strategy_records,
)
from coevolve_experiments.memory.schema import FROZEN_STRATEGY_RETRIEVAL_POLICY
from coevolve_experiments.runtime import (
    CheckpointRef,
    Observation,
    PolicyValidator,
    RolloutRecord,
    VerifierOutcome,
    make_team_proposal,
)
from coevolve_experiments.runtime.contracts import PhysicalDispatchResult


def _record(phase: str, label: str, *, failure: str = "") -> dict[str, str]:
    return {
        "phase_trigger": phase,
        "failure_condition": failure or "no verifier-reported failure",
        "coordination_pattern": label,
        "verified_outcome": "rejected" if failure else "accepted",
        "avoid_rule": f"avoid {label}",
        "repair_direction": f"repair {label}",
    }


def test_exact_phase_uses_frozen_top_three_and_later_source_tie_break() -> None:
    records = [
        *[
            _record("during supported payload transport", f"carry-{index}")
            for index in range(7)
        ],
        _record("during terminal placement coordination", "place"),
        _record("when the applicable cooperative phase is active", "generic"),
    ]

    selected = select_strategy_records(records, active_phase="carry")

    assert len(selected) == FROZEN_STRATEGY_RETRIEVAL_POLICY.limit == 3
    assert [item["coordination_pattern"] for item in selected] == [
        "carry-6",
        "carry-5",
        "carry-4",
    ]


def test_exact_alias_and_generic_records_are_jointly_ranked() -> None:
    corridor = _record("corridor transport", "corridor")
    generic = _record("when the applicable cooperative phase is active", "generic")
    exact = _record("corridor entry", "entry")

    with_exact = select_strategy_records(
        [corridor, generic, exact], active_phase="corridor entry"
    )
    alias_only = select_strategy_records(
        [corridor, generic], active_phase="corridor entry"
    )
    generic_only = select_strategy_records(
        [generic], active_phase="unregistered lifecycle"
    )

    assert [item["coordination_pattern"] for item in with_exact] == [
        "entry",
        "corridor",
        "generic",
    ]
    assert [item["coordination_pattern"] for item in alias_only] == [
        "corridor",
        "generic",
    ]
    assert [item["coordination_pattern"] for item in generic_only] == ["generic"]


def test_balancing_prioritizes_two_failures_and_reserves_a_success() -> None:
    records = [
        _record("carry", "positive"),
        _record("carry", "wall", failure="payload wall contact"),
        _record("carry", "drop", failure="bilateral support payload drop"),
    ]

    selected = select_strategy_records(
        records,
        active_phase="supported payload transport",
        failure_code="PAYLOAD_DROP",
        failure_report="the payload lost support",
    )

    assert [item["coordination_pattern"] for item in selected] == [
        "drop",
        "wall",
        "positive",
    ]


def test_without_a_success_fallback_can_fill_the_third_slot_with_a_failure() -> None:
    records = [
        _record("carry", f"failure-{index}", failure=f"failure condition {index}")
        for index in range(4)
    ]

    selected = select_strategy_records(records, active_phase="carry")

    assert [item["coordination_pattern"] for item in selected] == [
        "failure-3",
        "failure-2",
        "failure-1",
    ]


def test_rich_v83_fields_rank_by_context_confidence_evidence_then_source() -> None:
    def rich(
        label: str,
        *,
        failure_type: str,
        confidence: float,
        evidence: int,
    ) -> dict[str, object]:
        return {
            "id": label,
            "phase": "corridor_transport",
            "kind": "verified_failure_counterexample",
            "failure_types": [failure_type],
            "confidence": confidence,
            "evidence": {
                "accepted_count": 0,
                "rejected_count": evidence,
            },
        }

    records = [
        rich(
            "matching-low-confidence",
            failure_type="payload_drop",
            confidence=0.2,
            evidence=1,
        ),
        rich(
            "nonmatching-high-confidence",
            failure_type="wall_contact",
            confidence=0.9,
            evidence=9,
        ),
        rich(
            "matching-more-evidence-earlier",
            failure_type="payload_drop",
            confidence=0.7,
            evidence=2,
        ),
        rich(
            "matching-more-evidence-later",
            failure_type="payload_drop",
            confidence=0.7,
            evidence=2,
        ),
        {
            "id": "positive",
            "phase": "corridor_transport",
            "kind": "verified_motion_success",
            "confidence": 0.1,
            "evidence": {"accepted_count": 1, "rejected_count": 0},
        },
    ]

    selected = select_strategy_records(
        records,
        active_phase="corridor transport",
        failure_code="PAYLOAD_DROP",
    )

    assert [item["id"] for item in selected] == [
        "matching-more-evidence-later",
        "matching-more-evidence-earlier",
        "positive",
    ]


def test_failure_level_is_ranked_separately_before_confidence() -> None:
    def rich(
        label: str,
        *,
        failure_level: str,
        confidence: float,
    ) -> dict[str, object]:
        return {
            "id": label,
            "phase": "corridor_transport",
            "kind": "verified_failure_counterexample",
            "failure_types": ["payload_drop"],
            "failure_level": failure_level,
            "confidence": confidence,
            "evidence": {"accepted_count": 0, "rejected_count": 1},
        }

    records = [
        rich("wrong-level-high-confidence", failure_level="strategy", confidence=0.99),
        rich("matching-level-low-confidence", failure_level="coordination", confidence=0.10),
        {
            "id": "positive",
            "phase": "corridor_transport",
            "kind": "verified_motion_success",
            "failure_level": "none",
            "confidence": 0.1,
            "evidence": {"accepted_count": 1, "rejected_count": 0},
        },
    ]

    selected = select_strategy_records(
        records,
        active_phase="corridor transport",
        failure_code="PAYLOAD_DROP",
        failure_level="coordination",
    )

    assert [item["id"] for item in selected] == [
        "matching-level-low-confidence",
        "wrong-level-high-confidence",
        "positive",
    ]


def test_six_field_projection_is_semantic_not_byte_identity() -> None:
    policy = FROZEN_STRATEGY_RETRIEVAL_POLICY

    assert policy.as_dict() == {
        "limit": 3,
        "prioritized_counterexample_limit": 2,
        "reserve_success_when_available": True,
        "six_field_confidence_equivalent": 0.50,
        "six_field_evidence_equivalent": 1,
        "source_order_policy": "later_record_first",
    }


def test_online_entry_phase_and_failure_keys_are_host_derived() -> None:
    proposal = make_team_proposal(
        "advance",
        lifecycle_phase="first corridor entry",
    )
    policy = PolicyValidator().validate(proposal)
    rollout = RolloutRecord(
        transaction_index=1,
        checkpoint=CheckpointRef("c0"),
        observation_before=Observation(0),
        policy=policy,
        dispatch=PhysicalDispatchResult(True, True, Observation(1)),
        verifier=VerifierOutcome(
            False,
            failure_code="PAYLOAD_DROP",
            report="dropped",
        ),
        observation_after=Observation(1),
    )

    entry = strategy_entry_from_rollout(
        {
            "phase_trigger": "terminal placement",
            "failure_code": "WALL_CONTACT",
        },
        rollout,
    )

    assert entry["retrieval"] == {
        "phase": "corridor_entry",
        "failure_code": "PAYLOAD_DROP",
    }
    assert canonical_strategy_phase(entry["proposal"]["phase_trigger"]) == (
        "corridor_entry"
    )
    assert entry["proposal"]["verified_outcome"] == "rejected"


def test_online_strategy_promotion_abstracts_route_values_and_applies_leak_gate() -> None:
    proposal = make_team_proposal(
        "turn",
        lifecycle_phase="first corner",
    )
    policy = PolicyValidator().validate(proposal)
    rollout = RolloutRecord(
        transaction_index=2,
        checkpoint=CheckpointRef("checkpoint_17"),
        observation_before=Observation(1),
        policy=policy,
        dispatch=PhysicalDispatchResult(True, True, Observation(2)),
        verifier=VerifierOutcome(
            False,
            failure_code="WALL_CONTACT",
            report="at x=1.25 m the left robot hit the red cylinder",
        ),
        observation_after=Observation(2),
    )

    entry = strategy_entry_from_rollout(
        {
            "phase_trigger": "waypoint_3 beside the red cylinder",
            "failure_condition": "left robot at x=1.25 m touched the east wall",
            "coordination_pattern": "turn clockwise by 12 deg at checkpoint_17",
            "avoid_rule": "do not replay /tmp/run_204/script.py",
            "repair_direction": "move right by 0.30 m toward waypoint_4",
        },
        rollout,
    )

    assert entry["retrieval"] == {
        "phase": "heading_transition",
        "failure_code": "WALL_CONTACT",
    }
    assert entry["proposal"]["phase_trigger"] == "heading_transition"
    assert entry["proposal"]["verified_outcome"] == "rejected"
    assert not scan_memory_leaks(entry["proposal"])
    rendered = " ".join(entry["proposal"].values()).lower()
    for forbidden in (
        "1.25",
        "0.30",
        "left",
        "right",
        "clockwise",
        "red cylinder",
        "checkpoint_17",
        "waypoint_4",
        "/tmp/run_204/script.py",
    ):
        assert forbidden not in rendered


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "coordination_pattern",
            "move left by 0.30 m at checkpoint_7",
            "leaks route-local data",
        ),
        ("verified_outcome", "accepted", "outcome differs"),
        ("phase_trigger", "terminal placement", "phase differs"),
    ],
)
def test_online_overlay_validation_rejects_reintroduced_leaks_or_model_routing(
    field: str,
    value: str,
    message: str,
) -> None:
    proposal = make_team_proposal("turn", lifecycle_phase="first corner")
    rollout = RolloutRecord(
        transaction_index=3,
        checkpoint=CheckpointRef("c0"),
        observation_before=Observation(2),
        policy=PolicyValidator().validate(proposal),
        dispatch=PhysicalDispatchResult(True, True, Observation(3)),
        verifier=VerifierOutcome(False, failure_code="WALL_CONTACT", report="wall"),
        observation_after=Observation(3),
    )
    entry = strategy_entry_from_rollout({"strategy": "hold support"}, rollout)
    entry["proposal"][field] = value

    with pytest.raises(OnlineMemoryValidationError, match=message):
        validate_online_strategy_overlay(
            {
                "kind": "run_local_strategy_overlay",
                "schema_version": 2,
                "entries": [entry],
            }
        )
