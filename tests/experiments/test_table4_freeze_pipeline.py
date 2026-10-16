from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import coevolve_experiments.cli as cli_module
import coevolve_experiments.reporting.table1_source as table1_source_module
from coevolve_experiments.cli import main
from coevolve_experiments.contracts import Table1SourceCohort
from coevolve_experiments.memory import (
    SourceRunValidationError,
    StrategyBankError,
    export_audited_table1_strategy_bank,
    load_posthoc_source_evidence,
    produce_posthoc_source_evidence,
    validate_locked_strategy_bank_candidate,
    write_strategy_bank_candidate_attestation,
)
from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE
from coevolve_experiments.reporting import Table1CohortAudit, Table1RunEvidence
from coevolve_experiments.full_method_compatibility import (
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
)
from coevolve_experiments.lockfile import LOCK_SCHEMA_VERSION


def _full_method() -> FullMethodCompatibilityPin:
    return FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": "1" * 64,
            "prompt_sha256": "e" * 64,
            "route_spec_sha256": "2" * 64,
            "scene_sha256": "3" * 64,
            "skill_api_sha256": "4" * 64,
            "static_memory_sha256": "5" * 64,
            "verifier_config_sha256": "6" * 64,
            "verifier_source_sha256": "7" * 64,
        },
        behavior_artifacts={path: "8" * 64 for path in V83_FULL_BEHAVIOR_RELPATHS},
    )


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _source_fixture(tmp_path: Path) -> tuple[list[Path], Table1CohortAudit]:
    full_method = _full_method()
    aggregate = tmp_path / "strict_result_manifest.json"
    aggregate.write_text("{}\n", encoding="utf-8")
    aggregate_hash = hashlib.sha256(aggregate.read_bytes()).hexdigest()
    source = Table1SourceCohort(
        backbone="gpt-source",
        protocol_id="frozen-source-v1",
        method_commit="a" * 40,
        seeds=(200, 201, 202, 203, 204),
        condition_fingerprint="b" * 64,
        result_manifest_hash=aggregate_hash,
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        shared_prompt_contract_fingerprint="d" * 64,
        decision_prompt_fingerprint="e" * 64,
        full_method_compatibility_fingerprint=full_method.fingerprint,
    )
    paths: list[Path] = []
    runs: list[Table1RunEvidence] = []
    for run_index, seed in enumerate(source.seeds):
        root = tmp_path / "formal" / f"seed_{seed:03d}"
        evaluation_metrics = {"payload_step_displacement_m": 0.1 + run_index}
        rollout_ref = {
            "script_id": f"transport_{seed}",
            "attempt": 1,
            "checkpoint_before": "accepted_step_00",
            "checkpoint_after": "accepted_step_01",
            "outcome": "accepted",
        }
        rollout_key = _semantic_hash({**rollout_ref, "metrics": evaluation_metrics})
        strategy = {
            "specs": [
                {
                    "id": f"source_strategy_{seed}",
                    "phase": "transport",
                    "kind": "verified_motion_success",
                    "observed_coordination_mode": "sync",
                    "observed_coordination_primitive": (
                        "coordinated_linear_transport"
                    ),
                    "strategy_genome": {
                        "child_strategy": "synchronized bilateral support retention"
                    },
                    "forbidden": "avoid unsupported motion",
                    "repair": "preserve role-compatible support",
                    "evidence": {
                        "source": "physics_verifier",
                        "accepted_count": 1,
                        "rejected_count": 0,
                        "failure_counts": {},
                        "rollout_refs": [rollout_ref],
                        "rollout_keys": [rollout_key],
                    },
                }
            ]
        }
        metrics = {
            "physical_accounting": {
                "accepted_physical": 1,
                "rejected_physical": 0,
                "invariant_satisfied": True,
                "pending_physical_evaluation": 0,
            },
            "terminal": {"status": "SUCCESS"},
        }
        _write_json(root / "memory/strategy_specs.json", strategy)
        _write_json(root / "memory/action_exemplars.json", {"exemplars": []})
        _write_json(root / "memory/accepted_episode_manifest.json", {"episodes": []})
        (root / "memory/raw_rollouts.jsonl").write_text("", encoding="utf-8")
        _write_json(
            root / "attempt_01/evaluation.json",
            {"success": True, "failure_type": "none", "metrics": evaluation_metrics},
        )
        metrics_hash = _write_json(root / "metrics.json", metrics)
        memory_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "memory").iterdir(), key=lambda item: item.name)
        }
        final_memory_hash = _semantic_hash(memory_hashes)
        manifest = {
            "kind": "coevolve_formal_run_manifest",
            "benchmark": "backbone",
            "model": {
                "resolved": source.backbone,
                "provider": "provider_a",
                "identity_verified": True,
            },
            "codebase": {"git_head": source.method_commit},
            "budgets": {"physical_transactions": 80},
            "effective_arguments": {
                "eval_protocol": source.protocol_id,
                "execute": True,
                "seed": seed,
                "planner_base_url": PRIMARY_GPT_PROVIDER_ROUTE.base_url,
            },
            "artifact_hashes": {
                "final_online_memory_sha256": final_memory_hash,
            },
        }
        manifest_hash = _write_json(root / "run_manifest.json", manifest)
        paths.append(root)
        runs.append(
            Table1RunEvidence(
                run_dir=str(root.resolve()),
                seed=seed,
                run_index=run_index,
                backbone=source.backbone,
                provider="provider_a",
                provider_route_id=source.provider_route_id,
                provider_route_fingerprint=source.provider_route_fingerprint,
                protocol_id=source.protocol_id,
                method_commit=source.method_commit,
                condition_fingerprint=source.condition_fingerprint,
                shared_prompt_contract_fingerprint=(
                    source.shared_prompt_contract_fingerprint
                ),
                decision_prompt_fingerprint=source.decision_prompt_fingerprint,
                full_method_compatibility_fingerprint=(
                    source.full_method_compatibility_fingerprint
                ),
                terminal_status="SUCCESS",
                task_success=True,
                physical_transactions=1,
                accepted_transactions=1,
                rejected_transactions=0,
                phase_completed=6,
                phase_total=6,
                phase_completion=1.0,
                fm_calls=1,
                input_tokens=1,
                output_tokens=1,
                wall_time_s=1.0,
                cost_usd=0.01,
                artifact_hashes={
                    "run_manifest.json": manifest_hash,
                    "metrics.json": metrics_hash,
                },
            )
        )
    audit = Table1CohortAudit(
        source=source,
        manifest_path=str(aggregate.resolve()),
        manifest_sha256=aggregate_hash,
        manifest_internal_sha256="c" * 64,
        manifest_kind="coevolve_v83_strict_result_manifest",
        provider="provider_a",
        provider_route_id=source.provider_route_id,
        provider_route_fingerprint=source.provider_route_fingerprint,
        provider_only_companion=False,
        parent_protocol_id="",
        parent_method_commit="",
        parent_result_manifest_hash="",
        shared_method_artifact_hashes={},
        full_method_compatibility=full_method,
        runs=tuple(runs),
    )
    return paths, audit


def _inventory(paths: list[Path]) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in paths
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_posthoc_package_is_external_and_supports_two_stage_freeze(
    tmp_path: Path,
) -> None:
    sources, audit = _source_fixture(tmp_path)
    before = _inventory(sources)
    package = produce_posthoc_source_evidence(
        list(reversed(sources)), audit, tmp_path / "evidence"
    )
    assert _inventory(sources) == before
    assert len(package.runs) == 5

    candidate = export_audited_table1_strategy_bank(
        sources,
        audit,
        posthoc_evidence_package=package.root,
    )
    assert candidate.provenance["schema_version"] == 3
    assert (
        candidate.provenance["source_provider_route_id"]
        == audit.source.provider_route_id
    )
    assert (
        candidate.provenance["source_provider_route_fingerprint"]
        == audit.source.provider_route_fingerprint
    )
    attestation = write_strategy_bank_candidate_attestation(
        candidate,
        audit,
        package,
        "gpt",
        tmp_path / "candidate.json",
    )
    locked_hash = candidate.provenance["bank_sha256"]
    locked = export_audited_table1_strategy_bank(
        sources,
        audit,
        expected_bank_sha256=locked_hash,
        posthoc_evidence_package=package.root,
    )
    validate_locked_strategy_bank_candidate(
        locked,
        audit,
        package,
        "gpt",
        attestation.path,
        locked_bank_sha256=locked_hash,
    )
    assert candidate.bank == locked.bank
    assert candidate.provenance == locked.provenance


def test_posthoc_package_rejects_source_and_evidence_tampering(tmp_path: Path) -> None:
    sources, audit = _source_fixture(tmp_path)
    package = produce_posthoc_source_evidence(sources, audit, tmp_path / "evidence")
    evaluation = sources[0] / "attempt_01/evaluation.json"
    evaluation.write_bytes(evaluation.read_bytes() + b"\n")

    with pytest.raises(SourceRunValidationError, match="Host evaluation binding"):
        load_posthoc_source_evidence(package.root, audit)


def test_posthoc_production_rejects_unpinned_final_memory(tmp_path: Path) -> None:
    sources, audit = _source_fixture(tmp_path)
    strategy = sources[0] / "memory/strategy_specs.json"
    strategy.write_bytes(strategy.read_bytes() + b"\n")

    with pytest.raises(SourceRunValidationError, match="final online-memory digest"):
        produce_posthoc_source_evidence(sources, audit, tmp_path / "evidence")


def test_locked_export_rejects_different_candidate_or_lock_hash(tmp_path: Path) -> None:
    sources, audit = _source_fixture(tmp_path)
    package = produce_posthoc_source_evidence(sources, audit, tmp_path / "evidence")
    candidate = export_audited_table1_strategy_bank(
        sources, audit, posthoc_evidence_package=package.root
    )
    attestation = write_strategy_bank_candidate_attestation(
        candidate, audit, package, "gpt", tmp_path / "candidate.json"
    )

    with pytest.raises(StrategyBankError, match="lock bank hash differs"):
        validate_locked_strategy_bank_candidate(
            candidate,
            audit,
            package,
            "gpt",
            attestation.path,
            locked_bank_sha256="f" * 64,
        )


def test_posthoc_package_requires_exact_audited_five_run_set(tmp_path: Path) -> None:
    sources, audit = _source_fixture(tmp_path)
    sources[-1] = sources[0]
    with pytest.raises(SourceRunValidationError, match="must be unique"):
        produce_posthoc_source_evidence(sources, audit, tmp_path / "evidence")


def test_historical_candidate_audit_is_unavailable_without_private_pins(tmp_path):
    # Do not read or create evidence using fabricated qualification identities.
    manifest = tmp_path / "absent.json"
    with pytest.raises(table1_source_module.Table1SourceAuditError, match="not distributed"):
        table1_source_module.audit_frozen_table1_candidate(manifest, "gpt")
    assert not manifest.exists()


def test_posthoc_and_candidate_cli_complete_prelock_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    sources, audit = _source_fixture(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "audit_frozen_table1_candidate",
        lambda *args, **kwargs: audit,
    )
    common: list[str] = []
    for source in sources:
        common.extend(("--source-run", str(source)))
    common.extend(
        (
            "--source-manifest",
            audit.manifest_path,
            "--source-cohort",
            "gpt",
        )
    )
    evidence = tmp_path / "external-evidence"
    assert main(
        [
            "build-posthoc-strategy-evidence",
            *common,
            "--output-dir",
            str(evidence),
        ]
    ) == 0
    build_output = json.loads(capsys.readouterr().out)
    assert build_output["run_count"] == 5
    assert build_output["paper_evidence"] is False

    bank = tmp_path / "candidate" / "bank.json"
    provenance = tmp_path / "candidate" / "provenance.json"
    attestation = tmp_path / "candidate" / "attestation.json"
    assert main(
        [
            "candidate-export-strategy-bank",
            *common,
            "--evidence-package",
            str(evidence),
            "--output",
            str(bank),
            "--provenance-output",
            str(provenance),
            "--candidate-output",
            str(attestation),
        ]
    ) == 0
    candidate_output = json.loads(capsys.readouterr().out)
    assert candidate_output["protocol_lock_field"] == "gpt_strategy_bank_hash"
    assert candidate_output["protocol_lock_schema_version"] == LOCK_SCHEMA_VERSION
    assert len(candidate_output["bank_sha256"]) == 64
    assert bank.is_file() and provenance.is_file() and attestation.is_file()
