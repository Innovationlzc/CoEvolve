from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from coevolve_experiments.serialization import to_primitive
from coevolve_experiments.reporting.terminal_aux_evidence import (
    FROZEN_PAYLOAD_DROP_SAFETY_PROFILE,
    PAYLOAD_DROP_EVALUATION_KIND,
    PAYLOAD_DROP_EXECUTION_TRACE_KIND,
    PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
    PAYLOAD_DROP_POLICY_KIND,
    PHYSICAL_TRANSACTION_RECEIPT_KIND,
    RAW_AUX_EVIDENCE_SCHEMA_VERSION,
    capability_bound_auxiliary_producer_allowlist,
)
from coevolve_experiments.reporting.transaction_accounting import (
    ACCEPTED_MANIFEST_PIN,
    ACCEPTED_MANIFEST_RELPATH,
    FROZEN_PHASE_PLAN_KIND,
    FROZEN_PHASE_PLAN_PIN,
    FROZEN_PHASE_PLAN_SCHEMA_VERSION,
    MUTATION_COMMIT_PRODUCER_ID,
    MUTATION_COMMIT_RECEIPT_KIND,
    MUTATION_COMMIT_RECEIPT_SCHEMA_VERSION,
    TRANSACTION_ACCOUNTING_LEDGER_KIND,
    TRANSACTION_ACCOUNTING_LEDGER_PIN,
    TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
    recompute_transaction_accounting_ledger,
    validate_transaction_accounting_evidence,
)


RUN_ID = "table2/full/seed_200"
CAPABILITY_SHA = "a" * 64
COMPILED_MODEL_SHA = "b" * 64
SCENE_REALIZATION_SHA = "c" * 64
COMMITTED_MUTATION_SHA = "d" * 64
PRODUCERS = capability_bound_auxiliary_producer_allowlist(CAPABILITY_SHA)
PHASE_IDS = ("phase_1", "phase_2")


def _producer(family: str) -> dict[str, str]:
    return dict(PRODUCERS[family])


def _write(root: Path, relative_path: str, value: object) -> dict[str, object]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _physical_sample(*, transaction: int, event: str, payload_drop: bool) -> dict:
    contacts = []
    payload_z = 0.80
    if payload_drop:
        payload_z = 0.59
        contacts.append(
            {
                "geom1_id": 11,
                "geom2_id": 12,
                "geom1_role": "payload",
                "geom2_role": "floor",
                "geom1_name": "long_box_collision",
                "geom2_name": "manual_floor",
                "body1_name": "long_box",
                "body2_name": "manual_floor",
                "distance_m": -0.001,
                "normal_force_n": 5.0,
            }
        )
    return {
        "kind": PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
        "schema_version": RAW_AUX_EVIDENCE_SCHEMA_VERSION,
        "producer": _producer("payload_drop.physical_samples"),
        "run_id": RUN_ID,
        "physical_transaction": transaction,
        "checkpoint_id": "",
        "checkpoint_event_id": event,
        "simulator": {
            "name": "MuJoCo",
            "version": "3.3.5",
            "compiled_model_sha256": COMPILED_MODEL_SHA,
            "scene_realization_sha256": SCENE_REALIZATION_SHA,
        },
        "drop_predicate": {
            "safety_profile": FROZEN_PAYLOAD_DROP_SAFETY_PROFILE,
            "minimum_payload_z_m": 0.65,
            "height_margin_m": 0.05,
            "drop_z_threshold_m": 0.60,
            "contact_distance_tolerance_m": 1.0e-6,
        },
        "samples": [
            {
                "sample_index": 0,
                "time_s": float(transaction),
                "payload_active": True,
                "payload_pose": {
                    "position_xyz_m": [0.0, 0.0, payload_z],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "contacts": contacts,
                "support": {
                    "available": True,
                    "supporting_robots": [],
                    "normal_force_n_by_robot": {"front": 0.0, "back": 0.0},
                    "total_normal_force_n": 0.0,
                },
            }
        ],
    }


def _fixture(
    root: Path,
    *,
    phase_regression: bool = False,
    manifest_mismatch: bool = False,
    drop_counter_mismatch: bool = False,
    receipt_hash_chain_mismatch: bool = False,
    checkpoint_event_chain_mismatch: bool = False,
    forged_mutation_producer: bool = False,
) -> dict[str, object]:
    phase_ref = _write(
        root,
        "evidence/accounting/frozen_phase_plan.json",
        {
            "kind": FROZEN_PHASE_PLAN_KIND,
            "schema_version": FROZEN_PHASE_PLAN_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "phase_ids": list(PHASE_IDS),
        },
    )
    specifications = [
        {
            "checkpoint_before": "c0",
            "checkpoint_after": "c1",
            "outcome": "ACCEPT_SETUP",
            "failure": "none",
            "phase_before": [],
            "phase_after": [],
            "drop": False,
            "counter_before": 0,
            "counter_after": 0,
        },
        {
            "checkpoint_before": "c1",
            "checkpoint_after": "c1",
            "outcome": "REJECT",
            "failure": "payload_drop",
            "phase_before": [],
            "phase_after": [],
            "drop": True,
            "counter_before": 0,
            "counter_after": 0 if drop_counter_mismatch else 1,
        },
        {
            "checkpoint_before": "c1",
            "checkpoint_after": "c2",
            "outcome": "ACCEPT_PROGRESS",
            "failure": "none",
            "phase_before": [],
            "phase_after": ["phase_1"],
            "drop": False,
            "counter_before": 1,
            "counter_after": 0,
        },
        {
            "checkpoint_before": "c2",
            "checkpoint_after": "c3",
            "outcome": "TASK_COMPLETE",
            "failure": "none",
            "phase_before": ["phase_1"],
            "phase_after": [] if phase_regression else list(PHASE_IDS),
            "drop": False,
            "counter_before": 0,
            "counter_after": 0,
        },
    ]
    receipt_refs = []
    previous_receipt_sha = None
    for transaction, specification in enumerate(specifications, start=1):
        before_event = f"event_{transaction - 1}"
        if checkpoint_event_chain_mismatch and transaction == 3:
            before_event = "forged_event"
        after_event = f"event_{transaction}"
        common = {
            "schema_version": RAW_AUX_EVIDENCE_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "physical_transaction": transaction,
        }
        checkpoint_before = str(specification["checkpoint_before"])
        checkpoint_after = str(specification["checkpoint_after"])
        script = f"execute_transaction_{transaction}()"
        policy_ref = _write(
            root,
            f"evidence/transactions/transaction_{transaction:03d}_policy.json",
            {
                "kind": PAYLOAD_DROP_POLICY_KIND,
                "producer": _producer("payload_drop.policy"),
                **common,
                "checkpoint_id": checkpoint_before,
                "checkpoint_event_id": before_event,
                "action_script": script,
                "action_script_sha256": hashlib.sha256(script.encode()).hexdigest(),
            },
        )
        accepted = specification["outcome"] != "REJECT"
        evaluation_ref = _write(
            root,
            f"evidence/transactions/transaction_{transaction:03d}_evaluation.json",
            {
                "kind": PAYLOAD_DROP_EVALUATION_KIND,
                "producer": _producer("payload_drop.evaluation"),
                **common,
                "checkpoint_id": checkpoint_before,
                "checkpoint_event_id": before_event,
                "accepted": accepted,
                "failure_type": specification["failure"],
                "transition_outcome": specification["outcome"],
                "completed_phase_ids_before": specification["phase_before"],
                "completed_phase_ids_after": specification["phase_after"],
            },
        )
        execution_ref = _write(
            root,
            f"evidence/transactions/transaction_{transaction:03d}_execution_trace.json",
            {
                "kind": PAYLOAD_DROP_EXECUTION_TRACE_KIND,
                "producer": _producer("payload_drop.execution_trace"),
                **common,
                "checkpoint_id_before": checkpoint_before,
                "checkpoint_event_id_before": before_event,
                "checkpoint_id_after": checkpoint_after,
                "checkpoint_event_id_after": after_event,
                "physically_dispatched": True,
                "rollback_performed": not accepted,
                "physics_step_count": 50,
                "dispatch_started_s": float(transaction),
                "dispatch_finished_s": float(transaction) + 0.5,
            },
        )
        physical = _physical_sample(
            transaction=transaction,
            event=before_event,
            payload_drop=bool(specification["drop"]),
        )
        physical["checkpoint_id"] = checkpoint_before
        physical_ref = _write(
            root,
            f"evidence/transactions/transaction_{transaction:03d}_physical_samples.json",
            physical,
        )
        previous = previous_receipt_sha
        if receipt_hash_chain_mismatch and transaction == 4:
            previous = "f" * 64
        receipt_ref = _write(
            root,
            f"evidence/transactions/transaction_{transaction:03d}.json",
            {
                "kind": PHYSICAL_TRANSACTION_RECEIPT_KIND,
                "producer": _producer("payload_drop.transaction_receipt"),
                **common,
                "checkpoint_id_before": checkpoint_before,
                "checkpoint_event_id_before": before_event,
                "checkpoint_id_after": checkpoint_after,
                "checkpoint_event_id_after": after_event,
                "physically_dispatched": True,
                "verifier": {
                    "accepted": accepted,
                    "failure_type": specification["failure"],
                },
                "rollback_performed": not accepted,
                "drop_counter_before": specification["counter_before"],
                "drop_counter_after": specification["counter_after"],
                "policy": policy_ref,
                "evaluation": evaluation_ref,
                "execution_trace": execution_ref,
                "physical_samples": physical_ref,
                "previous_receipt_sha256": previous,
            },
        )
        receipt_refs.append(receipt_ref)
        previous_receipt_sha = str(receipt_ref["sha256"])

    manifest_steps = [
        {
            "index": 1,
            "attempt": 1,
            "script_id": "setup",
            "checkpoint_before": "c0",
            "checkpoint_after": "c1",
            "transition_outcome": "ACCEPT_SETUP",
            "robot_scripts": {"front": {}, "back": {}},
        },
        {
            "index": 2,
            "attempt": 3,
            "script_id": "progress",
            "checkpoint_before": "c1",
            "checkpoint_after": "c2",
            "transition_outcome": "ACCEPT_PROGRESS",
            "robot_scripts": {"front": {}, "back": {}},
        },
        {
            "index": 3,
            "attempt": 4,
            "script_id": "complete",
            "checkpoint_before": "c2",
            "checkpoint_after": "c3",
            "transition_outcome": "TASK_COMPLETE",
            "robot_scripts": {"front": {}, "back": {}},
        },
    ]
    manifest_ref = _write(
        root,
        ACCEPTED_MANIFEST_RELPATH,
        {
            "kind": "accepted_checkpoint_episode",
            "schema_version": 1,
            "initial_checkpoint": "c0",
            "latest_checkpoint": "wrong" if manifest_mismatch else "c3",
            "initial_state": {},
            "steps": manifest_steps,
        },
    )
    mutation_producer = {
        "id": MUTATION_COMMIT_PRODUCER_ID,
        "source_sha256": CAPABILITY_SHA,
    }
    if forged_mutation_producer:
        mutation_producer["source_sha256"] = "e" * 64
    mutation_ref = _write(
        root,
        "evidence/mutations/mutation_commit_001.json",
        {
            "kind": MUTATION_COMMIT_RECEIPT_KIND,
            "schema_version": MUTATION_COMMIT_RECEIPT_SCHEMA_VERSION,
            "producer": mutation_producer,
            "run_id": RUN_ID,
            "mutation_ordinal": 1,
            "source_rejected_physical_transaction": 2,
            "checkpoint_id": "c1",
            "mutation_level": "strategy_mutation",
            "committed_mutation_sha256": COMMITTED_MUTATION_SHA,
            "previous_commit_receipt_sha256": None,
        },
    )
    return {
        "receipt_refs": receipt_refs,
        "phase_ref": phase_ref,
        "manifest_ref": manifest_ref,
        "mutation_refs": [mutation_ref],
    }


def _derive(root: Path, fixture: dict[str, object], **overrides):
    arguments = {
        "transaction_receipt_refs": fixture["receipt_refs"],
        "phase_plan_ref": fixture["phase_ref"],
        "accepted_manifest_ref": fixture["manifest_ref"],
        "mutation_commit_receipt_refs": fixture["mutation_refs"],
        "expected_run_id": RUN_ID,
        "expected_physical_transactions": 4,
        "physical_budget": 80,
        "expected_compiled_model_sha256": COMPILED_MODEL_SHA,
        "expected_scene_realization_sha256": SCENE_REALIZATION_SHA,
        "expected_producers": PRODUCERS,
    }
    arguments.update(overrides)
    return recompute_transaction_accounting_ledger(root, **arguments)


def _claimed(**overrides) -> dict[str, object]:
    value = {
        "physical_transactions": 4,
        "accepted_transactions": 3,
        "rejected_transactions": 1,
        "accepted_setup_transactions": 1,
        "accepted_progress_transactions": 2,
        "phase_completed": 2,
        "phase_total": 2,
        "failure_counts": {"payload_drop": 1},
        "mutation_counts": {"strategy_mutation": 1},
    }
    value.update(overrides)
    return value


def _publish(
    root: Path,
    fixture: dict[str, object],
    *,
    receipt_refs=None,
) -> None:
    receipts = fixture["receipt_refs"] if receipt_refs is None else receipt_refs
    derivation = recompute_transaction_accounting_ledger(
        root,
        transaction_receipt_refs=receipts,
        phase_plan_ref=fixture["phase_ref"],
        accepted_manifest_ref=fixture["manifest_ref"],
        mutation_commit_receipt_refs=fixture["mutation_refs"],
        expected_run_id=RUN_ID,
        expected_physical_transactions=len(receipts),
        physical_budget=80,
        expected_compiled_model_sha256=COMPILED_MODEL_SHA,
        expected_scene_realization_sha256=SCENE_REALIZATION_SHA,
        expected_producers=PRODUCERS,
    )
    assert derivation.valid, derivation.invalid_reasons
    ledger_ref = _write(
        root,
        TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
        to_primitive(derivation.ledger),
    )
    _write(
        root,
        "run_manifest.json",
        {
            "status": "SUCCESS",
            "experiment": {"run_id": RUN_ID},
            "result_artifact_hashes": {
                TRANSACTION_ACCOUNTING_LEDGER_PIN: ledger_ref["sha256"],
                ACCEPTED_MANIFEST_PIN: fixture["manifest_ref"]["sha256"],
                FROZEN_PHASE_PLAN_PIN: fixture["phase_ref"]["sha256"],
            },
        },
    )


def _validate_published(root: Path):
    return validate_transaction_accounting_evidence(
        root,
        expected_run_id=RUN_ID,
        physical_budget=80,
        expected_compiled_model_sha256=COMPILED_MODEL_SHA,
        expected_scene_realization_sha256=SCENE_REALIZATION_SHA,
        expected_producers=PRODUCERS,
    )


def test_recomputes_all_accounting_from_receipts_and_freezes_ledger(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _derive(tmp_path, fixture, claimed_accounting=_claimed())

    assert result.valid, result.invalid_reasons
    assert result.ledger is not None
    assert result.ledger["kind"] == TRANSACTION_ACCOUNTING_LEDGER_KIND
    assert result.ledger["physical_transactions"] == 4
    assert result.ledger["accepted_transactions"] == 3
    assert result.ledger["rejected_transactions"] == 1
    assert result.ledger["accepted_setup_transactions"] == 1
    assert result.ledger["accepted_progress_transactions"] == 2
    assert result.ledger["failure_counts"] == {"payload_drop": 1}
    assert result.ledger["mutation_counts"] == {"strategy_mutation": 1}
    assert result.ledger["phase_completion"]["completed_phase_ids"] == PHASE_IDS
    assert result.ledger["drop_accounting"] == {
        "final_consecutive_drop_count": 0,
        "maximum_consecutive_drop_count": 1,
    }
    assert len(result.source_artifacts) == 23
    with pytest.raises(TypeError):
        result.ledger["physical_transactions"] = 9  # type: ignore[index]
    with pytest.raises(TypeError):
        result.ledger["failure_counts"]["payload_drop"] = 9  # type: ignore[index]


def test_supports_zero_physical_transaction_stall(tmp_path: Path) -> None:
    phase_ref = _write(
        tmp_path,
        "evidence/accounting/frozen_phase_plan.json",
        {
            "kind": FROZEN_PHASE_PLAN_KIND,
            "schema_version": FROZEN_PHASE_PLAN_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "phase_ids": list(PHASE_IDS),
        },
    )
    manifest_ref = _write(
        tmp_path,
        "memory/accepted_episode_manifest.json",
        {
            "kind": "accepted_checkpoint_episode",
            "schema_version": 1,
            "initial_checkpoint": "c0",
            "latest_checkpoint": "c0",
            "initial_state": {},
            "steps": [],
        },
    )
    result = recompute_transaction_accounting_ledger(
        tmp_path,
        transaction_receipt_refs=(),
        phase_plan_ref=phase_ref,
        accepted_manifest_ref=manifest_ref,
        mutation_commit_receipt_refs=(),
        expected_run_id=RUN_ID,
        expected_physical_transactions=0,
        physical_budget=80,
        expected_compiled_model_sha256=COMPILED_MODEL_SHA,
        expected_scene_realization_sha256=SCENE_REALIZATION_SHA,
        expected_producers=PRODUCERS,
        claimed_accounting={
            "physical_transactions": 0,
            "accepted_transactions": 0,
            "rejected_transactions": 0,
            "accepted_setup_transactions": 0,
            "accepted_progress_transactions": 0,
            "phase_completed": 0,
            "phase_total": 2,
            "failure_counts": {},
            "mutation_counts": {},
        },
    )

    assert result.valid, result.invalid_reasons
    assert result.ledger is not None
    assert result.ledger["receipt_chain"]["coverage_start_transaction"] is None
    assert result.ledger["physical_transactions"] == 0
    assert result.ledger["phase_completion"]["fraction"] == 0.0


def test_rejects_tampered_raw_transaction_leaf(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = tmp_path / "evidence/transactions/transaction_002_physical_samples.json"
    path.write_bytes(path.read_bytes() + b" ")

    result = _derive(tmp_path, fixture)

    assert not result.valid
    assert any("PHYSICAL_SAMPLES_MISSING_OR_INVALID" in reason for reason in result.invalid_reasons)


def test_rejects_missing_or_noncontiguous_receipt_coverage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    refs = list(fixture["receipt_refs"])
    del refs[1]

    result = _derive(tmp_path, fixture, transaction_receipt_refs=refs)

    assert not result.valid
    assert "TRANSACTION_ACCOUNTING_RECEIPT_COVERAGE_INCOMPLETE" in result.invalid_reasons
    assert any(
        "PATH_INVALID" in reason or "SEQUENCE_INVALID" in reason
        for reason in result.invalid_reasons
    )


def test_rejects_forged_aggregate_count(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    result = _derive(
        tmp_path,
        fixture,
        claimed_accounting=_claimed(accepted_transactions=4),
    )

    assert not result.valid
    assert (
        "TRANSACTION_ACCOUNTING_CLAIM_ACCEPTED_TRANSACTIONS_MISMATCH"
        in result.invalid_reasons
    )
    assert result.ledger is not None
    assert result.ledger["accepted_transactions"] == 3


def test_rejects_completed_phase_regression(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, phase_regression=True)

    result = _derive(tmp_path, fixture)

    assert not result.valid
    assert any("COMPLETED_PHASE_HISTORY_REGRESSED" in reason for reason in result.invalid_reasons)
    assert "TRANSACTION_ACCOUNTING_TRANSACTION_4_PHASE_HISTORY_REGRESSION" in result.invalid_reasons


def test_rejects_manifest_lineage_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, manifest_mismatch=True)

    result = _derive(tmp_path, fixture)

    assert not result.valid
    assert "TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_LATEST_MISMATCH" in result.invalid_reasons
    assert "TRANSACTION_ACCOUNTING_MANIFEST_LATEST_CHECKPOINT_MISMATCH" in result.invalid_reasons


def test_drop_counter_is_recomputed_from_raw_mujoco_samples(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, drop_counter_mismatch=True)

    result = _derive(tmp_path, fixture)

    assert not result.valid
    assert any("RAW_DERIVED_DROP_COUNTER_MISMATCH" in reason for reason in result.invalid_reasons)
    assert (
        "TRANSACTION_ACCOUNTING_TRANSACTION_2_DROP_COUNTER_NOT_RAW_DERIVED"
        in result.invalid_reasons
    )
    assert result.ledger is not None
    assert result.ledger["drop_accounting"]["maximum_consecutive_drop_count"] == 1


def test_rejects_receipt_and_checkpoint_event_chain_tampering(tmp_path: Path) -> None:
    hash_fixture = _fixture(tmp_path / "hash", receipt_hash_chain_mismatch=True)
    hash_result = _derive(tmp_path / "hash", hash_fixture)
    event_fixture = _fixture(
        tmp_path / "event",
        checkpoint_event_chain_mismatch=True,
    )
    event_result = _derive(tmp_path / "event", event_fixture)

    assert any("RECEIPT_HASH_CHAIN_MISMATCH" in reason for reason in hash_result.invalid_reasons)
    assert (
        "TRANSACTION_ACCOUNTING_TRANSACTION_3_CHECKPOINT_EVENT_CHAIN_MISMATCH"
        in event_result.invalid_reasons
    )


def test_mutation_commits_must_be_capability_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, forged_mutation_producer=True)

    result = _derive(tmp_path, fixture)

    assert not result.valid
    assert (
        "TRANSACTION_ACCOUNTING_MUTATION_COMMIT_1_PRODUCER_NOT_CAPABILITY_BOUND"
        in result.invalid_reasons
    )


def test_mutation_commit_reference_requires_canonical_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original_path = tmp_path / "evidence/mutations/mutation_commit_001.json"
    value = json.loads(original_path.read_text())
    fixture["mutation_refs"] = [
        _write(tmp_path, "evidence/mutations/renamed.json", value)
    ]

    result = _derive(tmp_path, fixture)

    assert not result.valid
    assert (
        "TRANSACTION_ACCOUNTING_MUTATION_COMMIT_1_PATH_INVALID"
        in result.invalid_reasons
    )


def test_validates_manifest_published_ledger_by_full_recomputation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _publish(tmp_path, fixture)

    result = _validate_published(tmp_path)

    assert result.valid, result.invalid_reasons
    assert result.mismatched_fields == ()
    assert result.published_ledger == result.recomputed_ledger
    assert len(result.source_artifacts) == 23


def test_published_ledger_field_tamper_fails_even_when_manifest_pin_is_updated(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _publish(tmp_path, fixture)
    ledger_path = tmp_path / TRANSACTION_ACCOUNTING_LEDGER_RELPATH
    ledger = json.loads(ledger_path.read_text())
    ledger["accepted_transactions"] = 4
    ledger_ref = _write(tmp_path, TRANSACTION_ACCOUNTING_LEDGER_RELPATH, ledger)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["result_artifact_hashes"][TRANSACTION_ACCOUNTING_LEDGER_PIN] = (
        ledger_ref["sha256"]
    )
    _write(tmp_path, "run_manifest.json", manifest)

    result = _validate_published(tmp_path)

    assert not result.valid
    assert "$.accepted_transactions" in result.mismatched_fields
    assert (
        "TRANSACTION_ACCOUNTING_PUBLISHED_LEDGER_NOT_RECOMPUTED"
        in result.invalid_reasons
    )


def test_manifest_pin_detects_published_ledger_byte_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _publish(tmp_path, fixture)
    ledger_path = tmp_path / TRANSACTION_ACCOUNTING_LEDGER_RELPATH
    ledger_path.write_bytes(ledger_path.read_bytes() + b" ")

    result = _validate_published(tmp_path)

    assert not result.valid
    assert (
        "TRANSACTION_ACCOUNTING_PUBLISHED_LEDGER_PIN_MISMATCH"
        in result.invalid_reasons
    )


def test_published_ledger_requires_unambiguous_strict_json(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _publish(tmp_path, fixture)
    ledger_path = tmp_path / TRANSACTION_ACCOUNTING_LEDGER_RELPATH
    raw = ledger_path.read_text()
    raw = raw.replace(
        '"physical_budget": 80,',
        '"physical_budget": 80,\n  "physical_budget": 80,',
        1,
    )
    ledger_path.write_text(raw)
    digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["result_artifact_hashes"][TRANSACTION_ACCOUNTING_LEDGER_PIN] = digest
    _write(tmp_path, "run_manifest.json", manifest)

    result = _validate_published(tmp_path)

    assert not result.valid
    assert (
        "TRANSACTION_ACCOUNTING_PUBLISHED_LEDGER_NOT_STRICT_JSON"
        in result.invalid_reasons
    )


def test_unreferenced_canonical_receipt_tail_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = tmp_path / ACCEPTED_MANIFEST_RELPATH
    manifest = json.loads(manifest_path.read_text())
    manifest["steps"] = manifest["steps"][:2]
    manifest["latest_checkpoint"] = "c2"
    fixture["manifest_ref"] = _write(
        tmp_path,
        ACCEPTED_MANIFEST_RELPATH,
        manifest,
    )
    first_three = list(fixture["receipt_refs"])[:3]
    _publish(tmp_path, fixture, receipt_refs=first_three)

    result = _validate_published(tmp_path)

    assert not result.valid
    assert (
        "TRANSACTION_ACCOUNTING_UNREFERENCED_RECEIPT_TAIL"
        in result.invalid_reasons
    )
    assert (
        "TRANSACTION_ACCOUNTING_UNREFERENCED_CANONICAL_RECEIPT"
        in result.invalid_reasons
    )


def test_canonical_receipt_directory_gap_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _publish(tmp_path, fixture)
    (tmp_path / "evidence/transactions/transaction_002.json").unlink()

    result = _validate_published(tmp_path)

    assert not result.valid
    assert (
        "TRANSACTION_ACCOUNTING_CANONICAL_RECEIPT_GAP"
        in result.invalid_reasons
    )
    assert (
        "TRANSACTION_ACCOUNTING_REFERENCED_RECEIPT_NOT_CANONICAL"
        in result.invalid_reasons
    )


def test_unreferenced_canonical_mutation_tail_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _publish(tmp_path, fixture)
    _write(
        tmp_path,
        "evidence/mutations/mutation_commit_002.json",
        {"not": "referenced"},
    )

    result = _validate_published(tmp_path)

    assert not result.valid
    assert (
        "TRANSACTION_ACCOUNTING_UNREFERENCED_MUTATION_TAIL"
        in result.invalid_reasons
    )
    assert (
        "TRANSACTION_ACCOUNTING_UNREFERENCED_CANONICAL_MUTATION"
        in result.invalid_reasons
    )


def test_canonical_mutation_directory_gap_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _publish(tmp_path, fixture)
    (tmp_path / "evidence/mutations/mutation_commit_001.json").unlink()

    result = _validate_published(tmp_path)

    assert not result.valid
    assert (
        "TRANSACTION_ACCOUNTING_CANONICAL_MUTATION_GAP"
        in result.invalid_reasons
    )
    assert (
        "TRANSACTION_ACCOUNTING_REFERENCED_MUTATION_NOT_CANONICAL"
        in result.invalid_reasons
    )
