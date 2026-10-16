from __future__ import annotations

from pathlib import Path

import pytest

import coevolve_experiments.memory.exporter as exporter_module
from coevolve_experiments.contracts import Table1SourceCohort
from coevolve_experiments.memory import (
    SourceRunValidationError,
    export_audited_table1_strategy_bank,
)
from coevolve_experiments.reporting import Table1CohortAudit, Table1RunEvidence
from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE
from coevolve_experiments.full_method_compatibility import (
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
)


def _full_method() -> FullMethodCompatibilityPin:
    return FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": "1" * 64,
            "prompt_sha256": "f" * 64,
            "route_spec_sha256": "2" * 64,
            "scene_sha256": "3" * 64,
            "skill_api_sha256": "4" * 64,
            "static_memory_sha256": "5" * 64,
            "verifier_config_sha256": "6" * 64,
            "verifier_source_sha256": "7" * 64,
        },
        behavior_artifacts={path: "8" * 64 for path in V83_FULL_BEHAVIOR_RELPATHS},
    )


def _audit(tmp_path: Path) -> Table1CohortAudit:
    full_method = _full_method()
    source = Table1SourceCohort(
        backbone="gpt-source",
        protocol_id="frozen-source-v1",
        method_commit="a" * 40,
        seeds=(200, 201, 202, 203, 204),
        condition_fingerprint="b" * 64,
        result_manifest_hash="c" * 64,
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        shared_prompt_contract_fingerprint="f" * 64,
        decision_prompt_fingerprint="f" * 64,
        full_method_compatibility_fingerprint=full_method.fingerprint,
    )
    runs = []
    for index, seed in enumerate(source.seeds):
        run_dir = tmp_path / f"seed_{seed:03d}"
        run_dir.mkdir()
        runs.append(
            Table1RunEvidence(
                run_dir=str(run_dir.resolve()),
                seed=seed,
                run_index=index,
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
                fm_calls=3,
                input_tokens=10,
                output_tokens=2,
                wall_time_s=1.0,
                cost_usd=0.1,
                artifact_hashes={"metrics.json": "d" * 64},
            )
        )
    return Table1CohortAudit(
        source=source,
        manifest_path=str((tmp_path / "aggregate.json").resolve()),
        manifest_sha256=source.result_manifest_hash,
        manifest_internal_sha256="e" * 64,
        manifest_kind="strict",
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


def test_formal_export_derives_identity_and_canonical_order_from_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _audit(tmp_path)
    captured = {}
    sentinel = object()

    def fake_export(source_runs, **kwargs):
        captured["source_runs"] = tuple(source_runs)
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(exporter_module, "export_strategy_bank", fake_export)
    supplied = [Path(run.run_dir) for run in reversed(audit.runs)]

    result = export_audited_table1_strategy_bank(
        supplied,
        audit,
        bank_path=tmp_path / "bank.json",
        expected_bank_sha256="f" * 64,
    )

    assert result is sentinel
    assert captured["source_runs"] == tuple(Path(run.run_dir) for run in audit.runs)
    assert captured["expected_source_model"] == audit.source.backbone
    assert captured["expected_protocol"] == audit.source.protocol_id
    assert captured["expected_source_seeds"] == audit.source.seeds
    assert captured["expected_source_method_commit"] == audit.source.method_commit
    assert (
        captured["expected_source_condition_fingerprint"]
        == audit.source.condition_fingerprint
    )
    assert captured["expected_source_result_manifest_hash"] == audit.manifest_sha256
    assert [item["seed"] for item in captured["audit_metadata"]] == list(
        audit.source.seeds
    )


def test_formal_export_rejects_source_directory_not_in_audited_cohort(
    tmp_path: Path,
) -> None:
    audit = _audit(tmp_path)
    supplied = [Path(run.run_dir) for run in audit.runs]
    replacement = tmp_path / "seed_999"
    replacement.mkdir()
    supplied[-1] = replacement

    with pytest.raises(SourceRunValidationError, match="differ from the audited"):
        export_audited_table1_strategy_bank(
            supplied,
            audit,
            bank_path=tmp_path / "bank.json",
            expected_bank_sha256="f" * 64,
        )
