from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from coevolve_experiments.memory import (
    FROZEN_STRATEGY_BANK_POLICY,
    MemoryBootstrapError,
    SourceRun,
    SourceRunValidationError,
    StrategyBankError,
    StrategyBankValidationError,
    TRANSFER_RECORD_FIELDS,
    TRANSFER_BOOTSTRAP_MUTABLE_FILES,
    bootstrap_transfer_memory,
    export_strategy_bank,
    remove_memory_leaks,
    scan_memory_leaks,
    validate_strategy_bank,
    validate_transfer_bootstrap_snapshot,
    verify_seed_bank_immutable,
)
from coevolve_experiments.memory.schema import semantic_sha256
from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE


def _spec(
    *,
    identifier: str = "source_record",
    kind: str = "verified_motion_success",
    accepted: int = 1,
    rejected: int = 0,
    child: str = "synchronized bilateral support retention",
) -> dict[str, object]:
    count = rejected if rejected else accepted
    rollout_outcome = "formation_desync" if rejected else "accepted"
    attempt_base = 7 + int(hashlib.sha256(identifier.encode()).hexdigest()[:6], 16) % 1000
    rollout_refs = [
        {
            "script_id": f"subtask_{attempt_base + index}",
            "attempt": attempt_base + index,
            "checkpoint_before": "cp4",
            "checkpoint_after": "" if rejected else f"cp{5 + index}",
            "outcome": rollout_outcome,
        }
        for index in range(count)
    ]
    rollout_keys = [
        _rollout_key(reference, _rollout_metrics(int(reference["attempt"])))
        for reference in rollout_refs
    ]
    return {
        "id": identifier,
        "phase": "route_phase_0",
        "kind": kind,
        "observed_coordination_mode": "sync",
        "observed_coordination_primitive": "coordinated_linear_transport",
        "condition": "Use waypoint (1.2, 3.4) in the source scene.",
        "diagnosis": (
            "The script from attempt 7 moved 0.5 m at 0.2 m/s from "
            "checkpoint cp4 with world X=1.2."
        ),
        "forbidden": [
            "Do not replay route segment 3.",
            "Do not load the accepted manifest or action exemplar.",
        ],
        "repair": "Restore checkpoint cp4 and change yaw by 15 deg.",
        "failure_types": ["formation_desync"] if rejected else [],
        "strategy_genome": {"child_strategy": child},
        "evidence": {
            "source": "physics_verifier",
            "accepted_count": accepted,
            "rejected_count": rejected,
            "failure_counts": {"formation_desync": rejected} if rejected else {},
            "rollout_refs": rollout_refs,
            "rollout_keys": rollout_keys,
        },
    }


def _rollout_metrics(attempt: int) -> dict[str, float]:
    return {"payload_step_displacement_m": 0.01 * attempt}


def _rollout_key(reference: dict[str, object], metrics: dict[str, float]) -> str:
    fingerprint = {**reference, "metrics": metrics}
    return hashlib.sha256(
        json.dumps(
            fingerprint,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _run(
    specs: list[dict[str, object]],
    *,
    model: str = "gpt-source",
    scene: str = "base",
    protocol: str = "table4-frozen",
    seed: int = 0,
    method_commit: str = "a" * 40,
    condition_fingerprint: str = "b" * 64,
) -> SourceRun:
    metrics = {
        "physical_accounting": {
            "invariant_satisfied": True,
            "pending_physical_evaluation": 0,
            "accepted_physical": 0,
            "rejected_physical": 0,
        }
    }
    bound_records = []
    evidence_receipts: dict[str, object] = {}
    for spec in specs:
        evidence = spec.get("evidence")
        if not isinstance(evidence, dict):
            continue
        if evidence.get("source") != "physics_verifier":
            continue
        kind = str(spec.get("kind", ""))
        rejected = int(evidence.get("rejected_count", 0))
        accepted = int(evidence.get("accepted_count", 0))
        outcome = (
            "rejected"
            if "failure" in kind or "counterexample" in kind
            else "accepted"
        )
        count = rejected if outcome == "rejected" else accepted
        if count > 0:
            metrics["physical_accounting"][f"{outcome}_physical"] += count
            spec_digest = semantic_sha256(spec)
            receipt_refs = []
            rollout_refs = evidence.get("rollout_refs", [])
            rollout_keys = evidence.get("rollout_keys", [])
            for receipt_index in range(count):
                rollout_ref = dict(rollout_refs[receipt_index])
                rollout_key = str(rollout_keys[receipt_index])
                attempt = int(rollout_ref["attempt"])
                failure_type = (
                    "none" if outcome == "accepted" else str(rollout_ref["outcome"])
                )
                host_evidence = {
                    "success": outcome == "accepted",
                    "failure_type": failure_type,
                    "metrics": _rollout_metrics(attempt),
                }
                host_raw = json.dumps(
                    host_evidence,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                host_evidence_sha256 = hashlib.sha256(host_raw).hexdigest()
                host_relpath = f"attempt_{attempt:02d}/evaluation.json"
                evidence_receipts[host_relpath] = host_evidence
                receipt = {
                    "kind": "strategy_source_verifier_receipt",
                    "schema_version": 2,
                    "source_record_sha256": spec_digest,
                    "verified_outcome": outcome,
                    "rollout_ref": rollout_ref,
                    "rollout_key": rollout_key,
                    "host_evidence": {
                        "relative_path": host_relpath,
                        "sha256": host_evidence_sha256,
                        "size_bytes": len(host_raw),
                    },
                }
                relpath = (
                    "memory/strategy_export_receipts/"
                    f"{spec_digest}_{receipt_index}.json"
                )
                evidence_receipts[relpath] = receipt
                receipt_refs.append(
                    {
                        "relpath": relpath,
                        "sha256": semantic_sha256(receipt),
                        "host_evidence_sha256": host_evidence_sha256,
                    }
                )
            bound_records.append(
                {
                    "source_record_sha256": spec_digest,
                    "verified_outcome": outcome,
                    "evidence_count": count,
                    "evidence_receipts": receipt_refs,
                }
            )
    strategy_specs = {"specs": specs}
    evidence_binding = {
        "kind": "strategy_source_evidence_binding",
        "schema_version": 2,
        "strategy_specs_sha256": semantic_sha256(strategy_specs),
        "metrics_sha256": semantic_sha256(metrics),
        "records": bound_records,
    }
    return SourceRun(
        strategy_specs=strategy_specs,
        metrics=metrics,
        manifest={
            "kind": "coevolve_formal_run_manifest",
            "model": {"resolved": model, "identity_verified": True},
            "scene_id": scene,
            "codebase": {"git_head": method_commit},
            "condition_fingerprint": condition_fingerprint,
            "budgets": {"physical_transactions": 80},
            "effective_arguments": {
                "eval_protocol": protocol,
                "execute": True,
                "seed": seed,
            },
            "result_artifact_hashes": {
                "strategy_export_evidence_sha256": semantic_sha256(
                    evidence_binding
                )
            },
        },
        audit={"protocol_valid": True},
        evidence_binding=evidence_binding,
        evidence_receipts=evidence_receipts,
    )


def _five_runs(*specs: dict[str, object], model: str = "gpt-source") -> list[SourceRun]:
    source_specs = list(specs) or [_spec()]
    return [_run(source_specs, model=model, seed=seed) for seed in range(5)]


def test_export_is_six_field_sanitized_deduplicated_and_audited(tmp_path: Path):
    runs = _five_runs(_spec())
    result = export_strategy_bank(
        runs,
        output_path=tmp_path / "strategy_bank.json",
        expected_source_model="gpt-source",
        expected_protocol="table4-frozen",
    )

    assert set(result.bank) == {"schema_version", "records"}
    assert len(result["records"]) == 1
    assert set(result["records"][0]) == set(TRANSFER_RECORD_FIELDS)
    assert not scan_memory_leaks(result.bank["records"])
    serialized_bank = json.dumps(result.bank, sort_keys=True).lower()
    for forbidden in (
        "checkpoint",
        "waypoint",
        "route",
        "segment",
        "scene",
        "script",
        "attempt",
        "rollout",
        "manifest",
        "action exemplar",
        "world x",
        "yaw",
        "0.5",
    ):
        assert forbidden not in serialized_bank

    assert "source_model" not in result.bank
    assert "sources" not in result.bank
    assert result.provenance["source_model"] == "gpt-source"
    assert result.provenance["policy"] == FROZEN_STRATEGY_BANK_POLICY.as_dict()
    assert result.provenance["summary"]["kept_records"] == 1
    assert result.provenance["summary"]["dropped_source_records"] == 4
    for source in result.provenance["sources"]:
        assert set(source["documents"]) == {
            "strategy_specs",
            "metrics",
            "manifest",
            "evidence_binding",
        }
        assert all(
            len(document["sha256"]) == 64
            for document in source["documents"].values()
        )
        assert source["kept"] or source["dropped"]
        assert source["frozen_audit"]["protocol_valid"] is True
        assert len(source["frozen_audit_sha256"]) == 64

    assert validate_strategy_bank(result)
    assert result.bank_path == tmp_path / "strategy_bank.json"
    assert result.provenance_path == tmp_path / "strategy_bank.provenance.json"


def test_export_order_is_deterministic_and_rejections_require_verifier_evidence():
    success = _spec(identifier="success", child="stable support retention")
    rejected = _spec(
        identifier="rejected",
        kind="verified_failure_counterexample",
        accepted=0,
        rejected=1,
        child="role-compatible repair",
    )
    unverified = _spec(identifier="unverified", child="unsafe suggestion")
    unverified["evidence"] = {"source": "model", "accepted_count": 1}

    forward = export_strategy_bank(_five_runs(success, rejected, unverified))
    reverse = export_strategy_bank(
        list(reversed(_five_runs(unverified, rejected, success)))
    )
    assert forward.bank == reverse.bank
    assert {item["verified_outcome"] for item in forward["records"]} == {
        "accepted",
        "rejected",
    }
    dropped_reasons = [
        item["reason"]
        for source in forward.provenance["sources"]
        for item in source["dropped"]
    ]
    assert "source_record_not_evidence_bound" in dropped_reasons
    assert any("deterministic_semantic_duplicate" in item for item in dropped_reasons)


def test_source_cohort_requires_exactly_five_same_model_base_protocol_runs():
    with pytest.raises(SourceRunValidationError, match="exactly 5"):
        export_strategy_bank(_five_runs()[:4])

    mixed_model = _five_runs()
    mixed_model[-1] = _run([_spec()], model="gemini-source", seed=4)
    with pytest.raises(SourceRunValidationError, match="different models"):
        export_strategy_bank(mixed_model)

    mirrored = _five_runs()
    mirrored[-1] = _run([_spec()], scene="mirrored", seed=4)
    with pytest.raises(SourceRunValidationError, match="expected base scene"):
        export_strategy_bank(mirrored)

    invalid_audit = [{"protocol_valid": True} for _ in range(5)]
    invalid_audit[-1] = {"protocol_valid": False}
    with pytest.raises(SourceRunValidationError, match="protocol_valid=true"):
        export_strategy_bank(_five_runs(), audit_metadata=invalid_audit)

    duplicate_seed = _five_runs()
    duplicate_seed[-1] = _run([_spec()], seed=3)
    with pytest.raises(SourceRunValidationError, match="five unique seeds"):
        export_strategy_bank(duplicate_seed)

    one_run = _run([_spec()], seed=0)
    forged_seed_audits = [
        {"protocol_valid": True, "seed": seed} for seed in range(5)
    ]
    with pytest.raises(SourceRunValidationError, match="audit seed conflicts"):
        export_strategy_bank([one_run] * 5, audit_metadata=forged_seed_audits)

    forged_model_audit = {
        "protocol_valid": True,
        "source_model": "gemini-forged",
    }
    with pytest.raises(SourceRunValidationError, match="audit model conflicts"):
        export_strategy_bank(
            _five_runs(model="gpt-source"),
            audit_metadata=forged_model_audit,
            expected_source_model="gemini-forged",
        )

    with pytest.raises(
        SourceRunValidationError,
        match="audit physical budget conflicts with manifest",
    ):
        export_strategy_bank(
            _five_runs(),
            audit_metadata={"protocol_valid": True, "physical_budget": 1},
        )


def test_source_cohort_seeds_are_lock_driven_not_hardcoded():
    runs = [_run([_spec()], seed=seed) for seed in range(200, 205)]

    exported = export_strategy_bank(
        runs,
        expected_source_seeds=(200, 201, 202, 203, 204),
    )

    assert exported.provenance["source_seeds"] == [200, 201, 202, 203, 204]
    with pytest.raises(
        SourceRunValidationError,
        match="seeds differ from the locked source cohort",
    ):
        export_strategy_bank(
            runs,
            expected_source_seeds=(0, 1, 2, 3, 4),
        )


def test_v3_provenance_binds_complete_locked_source_identity():
    method_commit = "c" * 40
    condition_fingerprint = "d" * 64
    result_manifest_hash = "e" * 64
    seeds = (200, 201, 202, 203, 204)
    runs = [
        _run(
            [_spec()],
            seed=seed,
            method_commit=method_commit,
            condition_fingerprint=condition_fingerprint,
        )
        for seed in seeds
    ]

    exported = export_strategy_bank(
        runs,
        audit_metadata={
            "protocol_valid": True,
            "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
            "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        },
        expected_source_model="gpt-source",
        expected_protocol="table4-frozen",
        expected_source_seeds=seeds,
        expected_source_method_commit=method_commit,
        expected_source_condition_fingerprint=condition_fingerprint,
        expected_source_result_manifest_hash=result_manifest_hash,
        expected_source_provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        expected_source_provider_route_fingerprint=(
            PRIMARY_GPT_PROVIDER_ROUTE.fingerprint
        ),
    )

    assert exported.provenance["schema_version"] == 3
    assert exported.provenance["source_method_commit"] == method_commit
    assert exported.provenance["source_seeds"] == list(seeds)
    assert (
        exported.provenance["source_condition_fingerprint"]
        == condition_fingerprint
    )
    assert (
        exported.provenance["source_result_manifest_hash"]
        == result_manifest_hash
    )
    assert (
        exported.provenance["source_provider_route_id"]
        == PRIMARY_GPT_PROVIDER_ROUTE.route_id
    )
    assert (
        exported.provenance["source_provider_route_fingerprint"]
        == PRIMARY_GPT_PROVIDER_ROUTE.fingerprint
    )


@pytest.mark.parametrize(
    ("run_field", "expected_name", "expected_value", "message"),
    (
        ("method_commit", "expected_source_method_commit", "c" * 40, "method commit"),
        (
            "condition_fingerprint",
            "expected_source_condition_fingerprint",
            "d" * 64,
            "condition fingerprint",
        ),
    ),
)
def test_v3_export_rejects_source_identity_relabelling(
    run_field, expected_name, expected_value, message
):
    kwargs = {
        "method_commit": "a" * 40,
        "condition_fingerprint": "b" * 64,
    }
    runs = [
        _run([_spec()], seed=seed, **kwargs)
        for seed in range(200, 205)
    ]
    expected = {
        "expected_source_model": "gpt-source",
        "expected_protocol": "table4-frozen",
        "expected_source_seeds": (200, 201, 202, 203, 204),
        "expected_source_method_commit": "a" * 40,
        "expected_source_condition_fingerprint": "b" * 64,
        "expected_source_result_manifest_hash": "e" * 64,
        "expected_source_provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "expected_source_provider_route_fingerprint": (
            PRIMARY_GPT_PROVIDER_ROUTE.fingerprint
        ),
    }
    expected[expected_name] = expected_value

    with pytest.raises(SourceRunValidationError, match=message):
        export_strategy_bank(
            runs,
            audit_metadata={
                "protocol_valid": True,
                "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
                "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            },
            **expected,
        )


def test_locked_bank_hash_is_checked_before_any_output_is_written(tmp_path: Path):
    destination = tmp_path / "should-not-exist.json"
    with pytest.raises(StrategyBankError, match="protocol-lock hash"):
        export_strategy_bank(
            _five_runs(),
            bank_path=destination,
            expected_bank_sha256="0" * 64,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "expected_seeds",
    (
        (200, 201, 202, 203),
        (200, 201, 202, 203, 203),
        (200, 201, 202, 203, True),
    ),
)
def test_locked_source_seed_identity_requires_five_unique_integers(expected_seeds):
    with pytest.raises(SourceRunValidationError, match="expected source cohort"):
        export_strategy_bank(
            _five_runs(),
            expected_source_seeds=expected_seeds,
        )


def test_source_budget_uses_manifest_first_and_audit_only_fills_missing():
    runs = _five_runs()
    legacy_runs = []
    for run in runs:
        manifest = dict(run.manifest)
        manifest.pop("budgets")
        legacy_runs.append(
            SourceRun(
                strategy_specs=run.strategy_specs,
                metrics=run.metrics,
                manifest=manifest,
                audit=run.audit,
                evidence_binding=run.evidence_binding,
                evidence_receipts=run.evidence_receipts,
            )
        )

    result = export_strategy_bank(
        legacy_runs,
        audit_metadata={"protocol_valid": True, "physical_budget": 80},
    )
    assert result.provenance["summary"]["source_runs"] == 5

    forged = []
    for run in runs:
        manifest = dict(run.manifest)
        manifest["budgets"] = {"physical_transactions": 1}
        forged.append(
            SourceRun(
                strategy_specs=run.strategy_specs,
                metrics=run.metrics,
                manifest=manifest,
                audit=run.audit,
                evidence_binding=run.evidence_binding,
                evidence_receipts=run.evidence_receipts,
            )
        )
    with pytest.raises(
        SourceRunValidationError,
        match="audit physical budget conflicts with manifest",
    ):
        export_strategy_bank(
            forged,
            audit_metadata={"protocol_valid": True, "physical_budget": 80},
        )


def test_recursive_leak_checker_and_remover_cover_local_execution_state():
    contaminated = {
        "nested": [
            {"world_x": 1.0, "checkpoint_pose": [1, 2, 3]},
            "waypoint cp3 on route gate segment 4 at yaw 30 deg",
            {"accepted_manifest": {}, "failure_cache": {}, "action_exemplars": []},
            {"artifact_path": "/tmp/run_4/attempt.json", "rollout_id": "r4"},
        ]
    }
    findings = scan_memory_leaks(contaminated)
    categories = {item.category for item in findings}
    assert {
        "world_coordinate",
        "pose_or_yaw",
        "checkpoint_reference",
        "waypoint_reference",
        "scene_topology",
        "numeric_unit",
        "execution_reference",
        "manifest_reference",
        "failure_cache",
        "action_exemplar",
    } <= categories
    assert not scan_memory_leaks(remove_memory_leaks(contaminated))


def test_validator_rejects_extra_fields_duplicates_caps_and_text_leaks():
    valid_record = {
        "phase_trigger": "during supported payload transport",
        "failure_condition": "the verifier reports formation loss",
        "coordination_pattern": "synchronous supported translation",
        "verified_outcome": "rejected",
        "avoid_rule": "do not repeat the rejected coordination pattern",
        "repair_direction": "change role compatibility before retrying",
    }
    assert validate_strategy_bank({"schema_version": 1, "records": [valid_record]})

    leaked = dict(valid_record, repair_direction="return to checkpoint cp3 at 0.5 m/s")
    with pytest.raises(StrategyBankValidationError, match="leaks"):
        validate_strategy_bank({"schema_version": 1, "records": [leaked]})

    extra = dict(valid_record, source_path="/tmp/source.json")
    with pytest.raises(StrategyBankValidationError, match="fields must be exactly"):
        validate_strategy_bank({"schema_version": 1, "records": [extra]})

    with pytest.raises(StrategyBankValidationError, match="duplicates"):
        validate_strategy_bank(
            {"schema_version": 1, "records": [valid_record, dict(valid_record)]}
        )


@pytest.mark.parametrize(
    ("condition", "source_model"),
    [("GPT→GPT", "gpt-source"), ("Gemini->GPT", "gemini-source")],
)
def test_bootstrap_keeps_seed_immutable_and_all_target_local_state_empty(
    tmp_path: Path,
    condition: str,
    source_model: str,
):
    exported = export_strategy_bank(_five_runs(model=source_model))
    target = tmp_path / condition.replace("→", "-").replace(">", "")
    bootstrap = bootstrap_transfer_memory(
        target,
        condition,
        seed_bank=exported,
        seed_provenance=exported,
        target_model="gpt-target",
    )

    assert verify_seed_bank_immutable(bootstrap)
    assert bootstrap.seed_bank_path.stat().st_mode & 0o222 == 0
    assert bootstrap.audit_path.stat().st_mode & 0o222 == 0
    assert bootstrap.snapshot_path.stat().st_mode & 0o222 == 0
    snapshot = validate_transfer_bootstrap_snapshot(
        bootstrap.seed_bank_path.parent,
        require_initial_state=True,
    )
    assert snapshot.file_sha256 == bootstrap.snapshot_sha256
    assert bootstrap.seed_bank_path != bootstrap.online_overlay_path
    assert json.loads(bootstrap.online_overlay_path.read_text())["entries"] == []
    assert json.loads(bootstrap.action_exemplars_path.read_text())["exemplars"] == []
    assert json.loads(bootstrap.accepted_manifest_path.read_text())["steps"] == []
    assert json.loads(bootstrap.failure_cache_path.read_text())["failures"] == []
    audit = json.loads(bootstrap.audit_path.read_text())
    assert audit["seed"]["immutable"] is True
    assert audit["online_overlay"]["separate_from_seed"] is True
    assert all(
        audit["fresh_target_state"][field] is True
        for field in (
            "action_exemplars_empty",
            "accepted_manifest_steps_empty",
            "failure_cache_empty",
        )
    )


def test_bootstrap_snapshot_separates_prelaunch_and_postdispatch_mutability(
    tmp_path: Path,
):
    exported = export_strategy_bank(_five_runs(model="gpt-source"))

    before = bootstrap_transfer_memory(
        tmp_path / "before",
        "GPT→GPT",
        seed_bank=exported,
        seed_provenance=exported,
        target_model="gpt-target",
    )
    before.online_overlay_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MemoryBootstrapError, match="initial bootstrap bytes changed"):
        validate_transfer_bootstrap_snapshot(
            before.seed_bank_path.parent,
            require_initial_state=True,
        )

    after = bootstrap_transfer_memory(
        tmp_path / "after",
        "GPT→GPT",
        seed_bank=exported,
        seed_provenance=exported,
        target_model="gpt-target",
    )
    paths = {
        "online_strategy_overlay.json": after.online_overlay_path,
        "action_exemplars.json": after.action_exemplars_path,
        "accepted_episode_manifest.json": after.accepted_manifest_path,
        "failure_cache.json": after.failure_cache_path,
    }
    assert set(paths) == set(TRANSFER_BOOTSTRAP_MUTABLE_FILES)
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    validate_transfer_bootstrap_snapshot(
        after.seed_bank_path.parent,
        require_initial_state=False,
    )

    after.audit_path.chmod(0o644)
    after.audit_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MemoryBootstrapError, match="formal binding differs|immutable bootstrap bytes changed"):
        validate_transfer_bootstrap_snapshot(
            after.seed_bank_path.parent,
            require_initial_state=False,
        )


def test_empty_bootstrap_and_cross_model_label_guards(tmp_path: Path):
    empty = bootstrap_transfer_memory(tmp_path / "empty", "Empty", target_model="GPT")
    assert json.loads(empty.seed_bank_path.read_text())["records"] == []

    gpt_bank = export_strategy_bank(_five_runs(model="gpt-source"))
    with pytest.raises(MemoryBootstrapError, match="gemini source provenance"):
        bootstrap_transfer_memory(
            tmp_path / "wrong-source",
            "Gemini→GPT",
            seed_bank=gpt_bank,
            seed_provenance=gpt_bank,
        )
    with pytest.raises(MemoryBootstrapError, match="cannot preload"):
        bootstrap_transfer_memory(
            tmp_path / "wrong-empty",
            "Empty",
            seed_bank=gpt_bank,
        )

    with pytest.raises(MemoryBootstrapError, match="provenance sidecar"):
        bootstrap_transfer_memory(
            tmp_path / "missing-provenance",
            "GPT→GPT",
            seed_bank=gpt_bank.bank,
            source_model="gpt-source",
        )

    with pytest.raises(MemoryBootstrapError, match="does not match seed provenance"):
        bootstrap_transfer_memory(
            tmp_path / "forged-source-label",
            "Gemini→GPT",
            seed_bank=gpt_bank,
            seed_provenance=gpt_bank,
            source_model="gemini-forged",
        )

    missing_hash = dict(gpt_bank.provenance)
    missing_hash.pop("bank_sha256")
    with pytest.raises(MemoryBootstrapError, match="bank_sha256 is missing"):
        bootstrap_transfer_memory(
            tmp_path / "missing-bank-hash",
            "GPT→GPT",
            seed_bank=gpt_bank,
            seed_provenance=missing_hash,
            source_model="gpt-source",
        )
