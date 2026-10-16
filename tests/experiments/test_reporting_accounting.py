from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE
from coevolve_experiments.reporting import collect_run
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
    FROZEN_PHASE_PLAN_RELPATH,
    FROZEN_PHASE_PLAN_SCHEMA_VERSION,
    TRANSACTION_ACCOUNTING_LEDGER_PIN,
    TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
    recompute_transaction_accounting_ledger,
)
from coevolve_experiments.serialization import to_primitive


RUN_ID = "table2_method_comparison__coevolve_full__seed_200"
CAPABILITY_SHA256 = "a" * 64
COMPILED_MODEL_SHA256 = "b" * 64
SCENE_REALIZATION_SHA256 = "c" * 64
PHASE_ID = "task_complete"
PRODUCERS = capability_bound_auxiliary_producer_allowlist(CAPABILITY_SHA256)


def _write_json_ref(
    root: Path,
    relative_path: str,
    payload: object,
) -> dict[str, object]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return {
        "relative_path": relative_path,
        "sha256": sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _producer(family: str) -> dict[str, str]:
    return dict(PRODUCERS[family])


def _write_accounted_run(root: Path) -> dict[str, object]:
    """Publish one minimal SUCCESS run with independently replayable accounting."""

    root.mkdir(parents=True)
    expected: dict[str, object] = {
        "run_id": RUN_ID,
        "table": "table2_method_comparison",
        "condition": "coevolve_full",
        "method": "coevolve_full",
        "scene": "base",
        "memory_condition": "online_only",
        "target_model": "gpt-exact",
        "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        "seed": 200,
        "protocol_id": "paper-contract-v1",
        "base_method_commit": "1" * 40,
        "condition_fingerprint": "2" * 64,
        "invariant_fingerprint": "3" * 64,
        "scene_fingerprint": "4" * 64,
        "terminal_fingerprint": "5" * 64,
        "observation_contract_fingerprint": "6" * 64,
        "task_instruction_fingerprint": "7" * 64,
        "shared_prompt_contract_fingerprint": "8" * 64,
        "decision_prompt_fingerprint": "9" * 64,
        "skill_api_fingerprint": "d" * 64,
        "full_method_compatibility_fingerprint": "e" * 64,
        "compiled_model_sha256": COMPILED_MODEL_SHA256,
        "scene_realization_sha256": SCENE_REALIZATION_SHA256,
        "require_transaction_accounting_evidence": True,
    }
    metrics = {
        "task_success": True,
        "physical_iterations": 1,
        "accepted_step_count": 1,
        "rejected_step_count": 0,
        "accepted_setup_count": 0,
        "accepted_progress_count": 1,
        "elapsed_s": 1.0,
        "phase_completion": {"completed": 1, "total": 1, "fraction": 1.0},
        "failure_counts": {},
        "mutation_counts": {},
        "model": {
            "llm_calls": 1,
            "vlm_calls": 0,
            "input_tokens": 10,
            "output_tokens": 2,
            "billing_amount_usd": 0.01,
            "billing_reconciliation_status": "complete",
        },
        "terminal": {"status": "SUCCESS"},
    }
    _write_json_ref(root, "run_plan.json", expected)
    _write_json_ref(root, "metrics.json", metrics)

    phase_ref = _write_json_ref(
        root,
        FROZEN_PHASE_PLAN_RELPATH,
        {
            "kind": FROZEN_PHASE_PLAN_KIND,
            "schema_version": FROZEN_PHASE_PLAN_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "phase_ids": [PHASE_ID],
        },
    )
    accepted_manifest_ref = _write_json_ref(
        root,
        ACCEPTED_MANIFEST_RELPATH,
        {
            "kind": "accepted_checkpoint_episode",
            "schema_version": 1,
            "initial_checkpoint": "checkpoint_0",
            "latest_checkpoint": "checkpoint_1",
            "initial_state": {},
            "steps": [
                {
                    "index": 1,
                    "attempt": 1,
                    "script_id": "complete",
                    "checkpoint_before": "checkpoint_0",
                    "checkpoint_after": "checkpoint_1",
                    "transition_outcome": "TASK_COMPLETE",
                    "robot_scripts": {"front": {}, "back": {}},
                }
            ],
        },
    )

    common = {
        "schema_version": RAW_AUX_EVIDENCE_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "physical_transaction": 1,
    }
    script = "execute_verified_task_complete()"
    policy_ref = _write_json_ref(
        root,
        "evidence/transactions/transaction_001_policy.json",
        {
            "kind": PAYLOAD_DROP_POLICY_KIND,
            "producer": _producer("payload_drop.policy"),
            **common,
            "checkpoint_id": "checkpoint_0",
            "checkpoint_event_id": "event_0",
            "action_script": script,
            "action_script_sha256": sha256(script.encode("utf-8")).hexdigest(),
        },
    )
    evaluation_ref = _write_json_ref(
        root,
        "evidence/transactions/transaction_001_evaluation.json",
        {
            "kind": PAYLOAD_DROP_EVALUATION_KIND,
            "producer": _producer("payload_drop.evaluation"),
            **common,
            "checkpoint_id": "checkpoint_0",
            "checkpoint_event_id": "event_0",
            "accepted": True,
            "failure_type": "none",
            "transition_outcome": "TASK_COMPLETE",
            "completed_phase_ids_before": [],
            "completed_phase_ids_after": [PHASE_ID],
        },
    )
    execution_ref = _write_json_ref(
        root,
        "evidence/transactions/transaction_001_execution_trace.json",
        {
            "kind": PAYLOAD_DROP_EXECUTION_TRACE_KIND,
            "producer": _producer("payload_drop.execution_trace"),
            **common,
            "checkpoint_id_before": "checkpoint_0",
            "checkpoint_event_id_before": "event_0",
            "checkpoint_id_after": "checkpoint_1",
            "checkpoint_event_id_after": "event_1",
            "physically_dispatched": True,
            "rollback_performed": False,
            "physics_step_count": 50,
            "dispatch_started_s": 0.0,
            "dispatch_finished_s": 0.5,
        },
    )
    physical_ref = _write_json_ref(
        root,
        "evidence/transactions/transaction_001_physical_samples.json",
        {
            "kind": PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
            "producer": _producer("payload_drop.physical_samples"),
            **common,
            "checkpoint_id": "checkpoint_0",
            "checkpoint_event_id": "event_0",
            "simulator": {
                "name": "MuJoCo",
                "version": "3.3.5",
                "compiled_model_sha256": COMPILED_MODEL_SHA256,
                "scene_realization_sha256": SCENE_REALIZATION_SHA256,
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
                    "time_s": 0.0,
                    "payload_active": True,
                    "payload_pose": {
                        "position_xyz_m": [0.0, 0.0, 0.8],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    "contacts": [],
                    "support": {
                        "available": True,
                        "supporting_robots": [],
                        "normal_force_n_by_robot": {"front": 0.0, "back": 0.0},
                        "total_normal_force_n": 0.0,
                    },
                }
            ],
        },
    )
    receipt_ref = _write_json_ref(
        root,
        "evidence/transactions/transaction_001.json",
        {
            "kind": PHYSICAL_TRANSACTION_RECEIPT_KIND,
            "producer": _producer("payload_drop.transaction_receipt"),
            **common,
            "checkpoint_id_before": "checkpoint_0",
            "checkpoint_event_id_before": "event_0",
            "checkpoint_id_after": "checkpoint_1",
            "checkpoint_event_id_after": "event_1",
            "physically_dispatched": True,
            "verifier": {"accepted": True, "failure_type": "none"},
            "rollback_performed": False,
            "drop_counter_before": 0,
            "drop_counter_after": 0,
            "policy": policy_ref,
            "evaluation": evaluation_ref,
            "execution_trace": execution_ref,
            "physical_samples": physical_ref,
            "previous_receipt_sha256": None,
        },
    )
    derivation = recompute_transaction_accounting_ledger(
        root,
        transaction_receipt_refs=[receipt_ref],
        phase_plan_ref=phase_ref,
        accepted_manifest_ref=accepted_manifest_ref,
        expected_run_id=RUN_ID,
        expected_physical_transactions=1,
        physical_budget=80,
        expected_compiled_model_sha256=COMPILED_MODEL_SHA256,
        expected_scene_realization_sha256=SCENE_REALIZATION_SHA256,
        expected_producers=PRODUCERS,
    )
    assert derivation.valid, derivation.invalid_reasons
    assert derivation.ledger is not None
    ledger_ref = _write_json_ref(
        root,
        TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
        to_primitive(derivation.ledger),
    )

    experiment = {
        key: value
        for key, value in expected.items()
        if key
        in {
            "run_id",
            "table",
            "condition",
            "method",
            "scene",
            "memory_condition",
            "target_model",
            "provider_route_id",
            "provider_route_fingerprint",
            "seed",
            "protocol_id",
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
        }
    }
    _write_json_ref(
        root,
        "run_manifest.json",
        {
            "status": "SUCCESS",
            "experiment": experiment,
            "codebase": {"git_head": expected["base_method_commit"]},
            "model": {
                "provider": "provider_a",
                "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
                "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
                "requested": "gpt-exact",
                "resolved": "gpt-exact",
            },
            "live_launch": {"entrypoint_sha256": CAPABILITY_SHA256},
            "result_artifact_hashes": {
                TRANSACTION_ACCOUNTING_LEDGER_PIN: ledger_ref["sha256"],
                ACCEPTED_MANIFEST_PIN: accepted_manifest_ref["sha256"],
                FROZEN_PHASE_PLAN_PIN: phase_ref["sha256"],
            },
        },
    )
    return expected


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _assert_accounting_fail_closed(record) -> None:
    assert not record.protocol_valid
    assert record.physical_transactions is None
    assert record.accepted_transactions is None
    assert record.rejected_transactions is None
    assert record.phase_completed is None
    assert record.failure_counts == {}
    assert record.mutation_counts == {}


def test_collect_run_recomputes_and_rejects_forged_metrics_aggregate(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    expected = _write_accounted_run(run)
    baseline = collect_run(run, expected=expected)
    assert baseline.protocol_valid, baseline.invalid_reasons
    assert baseline.physical_transactions == 1

    metrics_path = run / "metrics.json"
    metrics = _load_json(metrics_path)
    metrics["physical_iterations"] = 2
    metrics["accepted_step_count"] = 2
    metrics["accepted_progress_count"] = 2
    _write_json_ref(run, "metrics.json", metrics)

    record = collect_run(run, expected=expected)

    _assert_accounting_fail_closed(record)
    assert {
        "TRANSACTION_ACCOUNTING_CLAIM_PHYSICAL_TRANSACTIONS_MISMATCH",
        "TRANSACTION_ACCOUNTING_CLAIM_ACCEPTED_TRANSACTIONS_MISMATCH",
        "TRANSACTION_ACCOUNTING_CLAIM_ACCEPTED_PROGRESS_TRANSACTIONS_MISMATCH",
    }.issubset(record.invalid_reasons)
    # Raw receipt hashes are exposed only after the independent reconstruction
    # ran; the forged metrics aggregate itself is never published on the row.
    assert (
        "evidence/transactions/transaction_001.json" in record.artifact_hashes
    )


@pytest.mark.parametrize("tamper_mode", ["bytes", "field"])
def test_collect_run_rejects_published_ledger_tamper(
    tmp_path: Path,
    tamper_mode: str,
) -> None:
    run = tmp_path / tamper_mode
    expected = _write_accounted_run(run)
    ledger_path = run / TRANSACTION_ACCOUNTING_LEDGER_RELPATH
    if tamper_mode == "bytes":
        ledger_path.write_bytes(ledger_path.read_bytes() + b" ")
    else:
        ledger = _load_json(ledger_path)
        ledger["accepted_transactions"] = 2
        ledger_ref = _write_json_ref(
            run,
            TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
            ledger,
        )
        manifest = _load_json(run / "run_manifest.json")
        result_hashes = manifest["result_artifact_hashes"]
        assert isinstance(result_hashes, dict)
        result_hashes[TRANSACTION_ACCOUNTING_LEDGER_PIN] = ledger_ref["sha256"]
        _write_json_ref(run, "run_manifest.json", manifest)

    record = collect_run(run, expected=expected)

    _assert_accounting_fail_closed(record)
    if tamper_mode == "bytes":
        assert (
            "TRANSACTION_ACCOUNTING_PUBLISHED_LEDGER_PIN_MISMATCH"
            in record.invalid_reasons
        )
    else:
        assert (
            "TRANSACTION_ACCOUNTING_PUBLISHED_LEDGER_NOT_RECOMPUTED"
            in record.invalid_reasons
        )
        assert (
            "TRANSACTION_ACCOUNTING_LEDGER_FIELD_MISMATCH"
            in record.invalid_reasons
        )


def test_collect_run_rejects_receipt_tail_omitted_from_ledger_refs(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    expected = _write_accounted_run(run)
    _write_json_ref(
        run,
        "evidence/transactions/transaction_002.json",
        {"unreferenced": "canonical receipt tail"},
    )

    record = collect_run(run, expected=expected)

    _assert_accounting_fail_closed(record)
    assert "TRANSACTION_ACCOUNTING_UNREFERENCED_RECEIPT_TAIL" in record.invalid_reasons
    assert (
        "TRANSACTION_ACCOUNTING_UNREFERENCED_CANONICAL_RECEIPT"
        in record.invalid_reasons
    )


def test_collect_run_rejects_accepted_manifest_pin_mismatch(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    expected = _write_accounted_run(run)
    manifest = _load_json(run / "run_manifest.json")
    result_hashes = manifest["result_artifact_hashes"]
    assert isinstance(result_hashes, dict)
    result_hashes[ACCEPTED_MANIFEST_PIN] = "0" * 64
    _write_json_ref(run, "run_manifest.json", manifest)

    record = collect_run(run, expected=expected)

    _assert_accounting_fail_closed(record)
    assert (
        "TRANSACTION_ACCOUNTING_PUBLISHED_ACCEPTED_MANIFEST_REF_PIN_MISMATCH"
        in record.invalid_reasons
    )
