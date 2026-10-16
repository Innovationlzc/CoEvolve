from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from coevolve_experiments.memory import (
    SourceRunValidationError,
    StrategyBankError,
    export_strategy_bank,
)
from coevolve_experiments.memory.schema import semantic_sha256


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _source_spec(seed: int) -> dict[str, object]:
    rollout_ref, rollout_key, _ = _rollout(seed)
    return {
        "id": f"source_strategy_{seed}",
        "phase": "transport",
        "kind": "verified_motion_success",
        "observed_coordination_mode": "sync",
        "observed_coordination_primitive": "coordinated_linear_transport",
        "strategy_genome": {
            "child_strategy": "synchronized bilateral support retention"
        },
        "forbidden": "avoid unsupported motion",
        "repair": "preserve role-compatible support",
        # This model-facing block is intentionally present.  Export authority
        # comes only from strategy_export_evidence.json below.
        "evidence": {
            "source": "physics_verifier",
            "accepted_count": 1,
            "rejected_count": 0,
            "failure_counts": {},
            "rollout_refs": [rollout_ref],
            "rollout_keys": [rollout_key],
        },
    }


def _rollout(seed: int) -> tuple[dict[str, object], str, dict[str, object]]:
    metrics = {"payload_step_displacement_m": 0.1 + seed * 0.001}
    rollout_ref: dict[str, object] = {
        "script_id": f"transport_{seed}",
        "attempt": 1,
        "checkpoint_before": "accepted_step_00",
        "checkpoint_after": "accepted_step_01",
        "outcome": "accepted",
    }
    fingerprint = {
        "script_id": rollout_ref["script_id"],
        "attempt": rollout_ref["attempt"],
        "checkpoint_before": rollout_ref["checkpoint_before"],
        "checkpoint_after": rollout_ref["checkpoint_after"],
        "outcome": rollout_ref["outcome"],
        "metrics": metrics,
    }
    rollout_key = hashlib.sha256(
        json.dumps(
            fingerprint,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    evaluation = {"success": True, "failure_type": "none", "metrics": metrics}
    return rollout_ref, rollout_key, evaluation


def _file_source_run(
    parent: Path,
    seed: int,
    *,
    pin_in_manifest: bool = True,
    bind_record: bool = True,
) -> tuple[Path, str]:
    root = parent / f"seed_{seed:03d}"
    spec = _source_spec(seed)
    strategy = {"specs": [spec]}
    metrics = {
        "physical_accounting": {
            "accepted_physical": 1,
            "rejected_physical": 0,
            "invariant_satisfied": True,
            "pending_physical_evaluation": 0,
        }
    }
    strategy_hash = _write_json(root / "memory" / "strategy_specs.json", strategy)
    metrics_hash = _write_json(root / "metrics.json", metrics)
    rollout_ref, rollout_key, evaluation = _rollout(seed)
    evaluation_path = root / "attempt_01" / "evaluation.json"
    host_evidence_sha256 = _write_json(evaluation_path, evaluation)
    host_evidence_size = evaluation_path.stat().st_size
    spec_digest = semantic_sha256(spec)
    receipt = {
        "kind": "strategy_source_verifier_receipt",
        "schema_version": 2,
        "source_record_sha256": spec_digest,
        "verified_outcome": "accepted",
        "rollout_ref": rollout_ref,
        "rollout_key": rollout_key,
        "host_evidence": {
            "relative_path": "attempt_01/evaluation.json",
            "sha256": host_evidence_sha256,
            "size_bytes": host_evidence_size,
        },
    }
    receipt_relpath = (
        "memory/strategy_export_receipts/"
        f"{spec_digest}.json"
    )
    receipt_hash = _write_json(root / receipt_relpath, receipt)
    records = (
        [
            {
                "source_record_sha256": spec_digest,
                "verified_outcome": "accepted",
                "evidence_count": 1,
                "evidence_receipts": [
                    {
                        "relpath": receipt_relpath,
                        "sha256": receipt_hash,
                        "host_evidence_sha256": host_evidence_sha256,
                    }
                ],
            }
        ]
        if bind_record
        else []
    )
    binding = {
        "kind": "strategy_source_evidence_binding",
        "schema_version": 2,
        "strategy_specs_sha256": strategy_hash,
        "metrics_sha256": metrics_hash,
        "records": records,
    }
    binding_hash = _write_json(
        root / "memory" / "strategy_export_evidence.json",
        binding,
    )
    manifest: dict[str, object] = {
        "kind": "coevolve_formal_run_manifest",
        "model": {"resolved": "gpt-source", "identity_verified": True},
        "scene_id": "base",
        "budgets": {"physical_transactions": 80},
        "effective_arguments": {
            "eval_protocol": "table4-frozen",
            "execute": True,
            "seed": seed,
        },
    }
    if pin_in_manifest:
        manifest["result_artifact_hashes"] = {
            "strategy_export_evidence_sha256": binding_hash
        }
    _write_json(root / "run_manifest.json", manifest)
    return root, binding_hash


def _five_file_runs(
    tmp_path: Path,
    *,
    pin_in_manifest: bool = True,
    bind_record: bool = True,
) -> tuple[list[Path], list[str]]:
    pairs = [
        _file_source_run(
            tmp_path,
            seed,
            pin_in_manifest=pin_in_manifest,
            bind_record=bind_record,
        )
        for seed in range(5)
    ]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def test_file_backed_export_uses_manifest_pinned_evidence_binding(
    tmp_path: Path,
) -> None:
    runs, _ = _five_file_runs(tmp_path)

    result = export_strategy_bank(
        runs,
        audit_metadata={"protocol_valid": True},
        expected_source_model="gpt-source",
        expected_protocol="table4-frozen",
    )

    assert result.bank["records"]
    for source in result.provenance["sources"]:
        binding = source["documents"]["evidence_binding"]
        assert binding["pinned_by"] == "manifest"
        assert len(binding["sha256"]) == 64


def test_file_backed_export_blocks_when_binding_pin_is_missing(
    tmp_path: Path,
) -> None:
    runs, _ = _five_file_runs(tmp_path, pin_in_manifest=False)

    with pytest.raises(
        SourceRunValidationError,
        match="SOURCE_EVIDENCE_BINDING_REQUIRED",
    ):
        export_strategy_bank(runs, audit_metadata={"protocol_valid": True})


def test_file_backed_export_blocks_when_binding_artifact_is_missing(
    tmp_path: Path,
) -> None:
    runs, _ = _five_file_runs(tmp_path)
    (runs[0] / "memory" / "strategy_export_evidence.json").unlink()

    with pytest.raises(
        SourceRunValidationError,
        match="SOURCE_EVIDENCE_BINDING_REQUIRED",
    ):
        export_strategy_bank(runs, audit_metadata={"protocol_valid": True})


def test_frozen_audit_can_fill_but_cannot_override_manifest_binding_pin(
    tmp_path: Path,
) -> None:
    unpinned_runs, binding_hashes = _five_file_runs(
        tmp_path / "legacy",
        pin_in_manifest=False,
    )
    audits = [
        {
            "protocol_valid": True,
            "strategy_export_evidence_sha256": digest,
        }
        for digest in binding_hashes
    ]

    result = export_strategy_bank(unpinned_runs, audit_metadata=audits)
    assert {
        source["documents"]["evidence_binding"]["pinned_by"]
        for source in result.provenance["sources"]
    } == {"audit"}

    pinned_runs, _ = _five_file_runs(tmp_path / "current")
    with pytest.raises(SourceRunValidationError, match="audit .* conflicts"):
        export_strategy_bank(
            pinned_runs,
            audit_metadata={
                "protocol_valid": True,
                "strategy_export_evidence_sha256": "f" * 64,
            },
        )


def test_binding_or_bound_document_tampering_fails_closed(tmp_path: Path) -> None:
    runs, _ = _five_file_runs(tmp_path / "binding")
    binding_path = runs[0] / "memory" / "strategy_export_evidence.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["records"][0]["evidence_count"] = 2
    _write_json(binding_path, binding)

    with pytest.raises(
        SourceRunValidationError,
        match="binding SHA-256 does not match",
    ):
        export_strategy_bank(runs, audit_metadata={"protocol_valid": True})

    metric_runs, _ = _five_file_runs(tmp_path / "metrics")
    metrics_path = metric_runs[0] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["physical_accounting"]["accepted_physical"] = 2
    _write_json(metrics_path, metrics)

    with pytest.raises(SourceRunValidationError, match="metrics_sha256 does not match"):
        export_strategy_bank(metric_runs, audit_metadata={"protocol_valid": True})


def test_raw_host_evaluation_bytes_are_bound_and_rechecked(tmp_path: Path) -> None:
    runs, _ = _five_file_runs(tmp_path)
    evaluation_path = runs[0] / "attempt_01" / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["metrics"]["payload_step_displacement_m"] = 9.9
    _write_json(evaluation_path, evaluation)

    with pytest.raises(
        SourceRunValidationError,
        match="host-evidence reference differs from loaded bytes",
    ):
        export_strategy_bank(runs, audit_metadata={"protocol_valid": True})


def test_rehashed_chain_still_rejects_rollout_key_inconsistent_with_raw_metrics(
    tmp_path: Path,
) -> None:
    """Re-signing every container cannot rewrite the strategy's rollout fact."""

    runs, _ = _five_file_runs(tmp_path)
    root = runs[0]
    evaluation_path = root / "attempt_01" / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["metrics"]["payload_step_displacement_m"] = 9.9
    host_hash = _write_json(evaluation_path, evaluation)

    binding_path = root / "memory" / "strategy_export_evidence.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    receipt_ref = binding["records"][0]["evidence_receipts"][0]
    receipt_path = root / receipt_ref["relpath"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["host_evidence"].update(
        {
            "sha256": host_hash,
            "size_bytes": evaluation_path.stat().st_size,
        }
    )
    # Deliberately retain receipt.rollout_key and the corresponding strategy
    # rollout key.  They describe the old metrics and must fail semantic
    # recomputation even after every enclosing byte hash is refreshed.
    receipt_hash = _write_json(receipt_path, receipt)
    receipt_ref.update(
        {
            "sha256": receipt_hash,
            "host_evidence_sha256": host_hash,
        }
    )
    binding_hash = _write_json(binding_path, binding)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result_artifact_hashes"][
        "strategy_export_evidence_sha256"
    ] = binding_hash
    _write_json(manifest_path, manifest)

    with pytest.raises(
        SourceRunValidationError,
        match="rollout_key does not match raw verifier metrics",
    ):
        export_strategy_bank(runs, audit_metadata={"protocol_valid": True})


def test_spec_self_attestation_cannot_replace_independent_binding(
    tmp_path: Path,
) -> None:
    runs, _ = _five_file_runs(tmp_path, bind_record=False)

    with pytest.raises(StrategyBankError, match="no verifier-backed"):
        export_strategy_bank(runs, audit_metadata={"protocol_valid": True})


def test_each_bound_record_requires_an_exact_host_verifier_receipt(
    tmp_path: Path,
) -> None:
    missing_runs, _ = _five_file_runs(tmp_path / "missing")
    missing_root = missing_runs[0]
    missing_binding = json.loads(
        (missing_root / "memory" / "strategy_export_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    missing_relpath = missing_binding["records"][0]["evidence_receipts"][0][
        "relpath"
    ]
    (missing_root / missing_relpath).unlink()

    with pytest.raises(
        SourceRunValidationError,
        match="SOURCE_EVIDENCE_RECEIPT_REQUIRED",
    ):
        export_strategy_bank(
            missing_runs,
            audit_metadata={"protocol_valid": True},
        )

    tampered_runs, _ = _five_file_runs(tmp_path / "tampered")
    tampered_root = tampered_runs[0]
    tampered_binding = json.loads(
        (tampered_root / "memory" / "strategy_export_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    tampered_relpath = tampered_binding["records"][0]["evidence_receipts"][0][
        "relpath"
    ]
    receipt_path = tampered_root / tampered_relpath
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["host_evidence"]["sha256"] = "a" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(
        SourceRunValidationError,
        match="sha256 does not match receipt bytes",
    ):
        export_strategy_bank(
            tampered_runs,
            audit_metadata={"protocol_valid": True},
        )


def test_hash_consistent_binding_cannot_inflate_metrics_evidence_count(
    tmp_path: Path,
) -> None:
    runs, _ = _five_file_runs(tmp_path)
    root = runs[0]
    binding_path = root / "memory" / "strategy_export_evidence.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["records"][0]["evidence_count"] = 2
    new_binding_hash = _write_json(binding_path, binding)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result_artifact_hashes"][
        "strategy_export_evidence_sha256"
    ] = new_binding_hash
    _write_json(manifest_path, manifest)

    with pytest.raises(
        SourceRunValidationError,
        match="SOURCE_EVIDENCE_COUNT_EXCEEDS_METRICS",
    ):
        export_strategy_bank(runs, audit_metadata={"protocol_valid": True})
