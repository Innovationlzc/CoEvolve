from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import coevolve_experiments.cli as cli_module
from coevolve_experiments.cli import main
from coevolve_experiments.contracts import (
    MethodDecisionPromptPin,
    MethodVariant,
    ProtocolLock,
    Table1SourceCohort,
)
from coevolve_experiments.provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE,
    SECONDARY_GEMINI_PROVIDER_ROUTE,
)
from coevolve_experiments.full_method_compatibility import (
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
)
from coevolve_experiments.registry import build_suite
from coevolve_experiments.reporting import RunRecord
from coevolve_experiments.reporting.collect import collect_run
from coevolve_experiments.reporting.paper_summary import (
    IncompletePaperSummaryError,
    PaperSummaryError,
    summarize_paper_tables,
)
from coevolve_experiments.reporting.table1_source import (
    Table1CohortAudit,
    Table1RunEvidence,
)
from coevolve_experiments.scenes import load_scene


SEEDS = (200, 201, 202, 203, 204)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _lock() -> ProtocolLock:
    shared_prompt = _digest("shared-prompt")
    full_prompt = _digest("full-prompt")
    skill_api = _digest("skill-api")
    full_method = FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": _digest("initial-memory"),
            "prompt_sha256": full_prompt,
            "route_spec_sha256": _digest("route"),
            "scene_sha256": _digest("scene"),
            "skill_api_sha256": skill_api,
            "static_memory_sha256": _digest("static-memory"),
            "verifier_config_sha256": _digest("verifier-config"),
            "verifier_source_sha256": _digest("verifier-source"),
        },
        behavior_artifacts={
            path: _digest(f"fixture:{path}")
            for path in V83_FULL_BEHAVIOR_RELPATHS
        },
    )
    gpt = Table1SourceCohort(
        backbone="gpt-5.4-mini-2026-03-17",
        protocol_id="coevolve-eval-v83-v33-efficient",
        method_commit="b" * 40,
        seeds=SEEDS,
        condition_fingerprint=_digest("table1-gpt-condition"),
        result_manifest_hash=_digest("table1-gpt-results"),
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        shared_prompt_contract_fingerprint=shared_prompt,
        decision_prompt_fingerprint=full_prompt,
        full_method_compatibility_fingerprint=full_method.fingerprint,
    )
    gemini = Table1SourceCohort(
        backbone="gemini-3.5-flash",
        protocol_id="coevolve-eval-v83-v33-efficient-gemini",
        method_commit="c" * 40,
        seeds=SEEDS,
        condition_fingerprint=_digest("table1-gemini-condition"),
        result_manifest_hash=_digest("table1-gemini-results"),
        provider_route_id=SECONDARY_GEMINI_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=SECONDARY_GEMINI_PROVIDER_ROUTE.fingerprint,
        shared_prompt_contract_fingerprint=shared_prompt,
        decision_prompt_fingerprint=full_prompt,
        full_method_compatibility_fingerprint=full_method.fingerprint,
    )
    decision_pins = tuple(
        MethodDecisionPromptPin(
            method_variant=method,
            decision_prompt_fingerprint=(
                full_prompt
                if method is MethodVariant.COEVOLE_FULL
                else _digest(f"decision-{method.value}")
            ),
        )
        for method in MethodVariant
    )
    return ProtocolLock(
        protocol_id="tables-2-4-paper-summary-test",
        invariant_fingerprint=_digest("invariant"),
        terminal_fingerprint=_digest("terminal"),
        base_method_commit="a" * 40,
        table1_gpt_source=gpt,
        table1_gemini_source=gemini,
        gpt_strategy_bank_hash=_digest("gpt-bank"),
        gemini_strategy_bank_hash=_digest("gemini-bank"),
        target_provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        target_provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        observation_contract_fingerprint=_digest("observation"),
        task_instruction_fingerprint=_digest("instruction"),
        shared_prompt_contract_fingerprint=shared_prompt,
        method_decision_prompt_pins=decision_pins,
        skill_api_fingerprint=skill_api,
        full_method_compatibility=full_method,
    )


def _source_audit(lock: ProtocolLock) -> Table1CohortAudit:
    scene = load_scene("base")
    source = lock.table1_gpt_source
    runs = tuple(
        Table1RunEvidence(
            run_dir=f"/immutable/table1/gpt/seed_{seed:03d}",
            seed=seed,
            run_index=index,
            backbone=source.backbone,
            provider=PRIMARY_GPT_PROVIDER_ROUTE.provider,
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
            physical_transactions=20 + index,
            accepted_transactions=19 + index,
            rejected_transactions=1,
            phase_completed=6,
            phase_total=6,
            phase_completion=1.0,
            fm_calls=10 + index,
            input_tokens=100 + index,
            output_tokens=20 + index,
            wall_time_s=30.0 + index,
            cost_usd=0.1 + index / 100,
            artifact_hashes={"metrics.json": _digest(f"metrics-{seed}")},
        )
        for index, seed in enumerate(SEEDS)
    )
    return Table1CohortAudit(
        source=source,
        manifest_path="/immutable/table1/strict_result_manifest.json",
        manifest_sha256=source.result_manifest_hash,
        manifest_internal_sha256=_digest("internal-manifest"),
        manifest_kind="coevolve_v83_strict_result_manifest",
        provider=PRIMARY_GPT_PROVIDER_ROUTE.provider,
        provider_route_id=source.provider_route_id,
        provider_route_fingerprint=source.provider_route_fingerprint,
        provider_only_companion=False,
        parent_protocol_id="",
        parent_method_commit="",
        parent_result_manifest_hash="",
        shared_method_artifact_hashes={
            "scene_sha256": scene.asset.source_sha256,
            "skill_api_sha256": lock.skill_api_fingerprint,
            "prompt_sha256": source.shared_prompt_contract_fingerprint,
        },
        full_method_compatibility=lock.full_method_compatibility,
        runs=runs,
    )


def _record_for_run(run, *, valid: bool = True) -> RunRecord:
    condition = run.output_relpath.split("/")[1]
    source_model = run.strategy_bank.source_backbone if run.strategy_bank else ""
    return RunRecord(
        run_dir=f"/paper-results/{run.output_relpath}",
        table=run.table_id.value if valid else "",
        condition=condition if valid else "",
        method=run.method_variant.value if valid else "",
        scene=run.scene_id.value if valid else "",
        memory_condition=run.memory_condition.value if valid else "",
        source_model=source_model if valid else "",
        target_model=run.target_backbone if valid else "",
        provider=PRIMARY_GPT_PROVIDER_ROUTE.provider if valid else "",
        provider_route_id=run.provider_route_id if valid else "",
        provider_route_fingerprint=(
            run.provider_route_fingerprint if valid else ""
        ),
        strategy_bank_sha256=(
            run.strategy_bank.sha256 if valid and run.strategy_bank else ""
        ),
        seed=run.seed if valid else None,
        protocol_id=run.protocol_id if valid else "",
        base_method_commit=run.base_method_commit if valid else "",
        condition_fingerprint=run.condition_fingerprint if valid else "",
        invariant_fingerprint=run.invariant_fingerprint if valid else "",
        scene_fingerprint=run.scene_fingerprint if valid else "",
        terminal_fingerprint=run.terminal_fingerprint if valid else "",
        observation_contract_fingerprint=(
            run.observation_contract_fingerprint if valid else ""
        ),
        task_instruction_fingerprint=(
            run.task_instruction_fingerprint if valid else ""
        ),
        shared_prompt_contract_fingerprint=(
            run.shared_prompt_contract_fingerprint if valid else ""
        ),
        decision_prompt_fingerprint=(
            run.decision_prompt_fingerprint if valid else ""
        ),
        skill_api_fingerprint=run.skill_api_fingerprint if valid else "",
        full_method_compatibility_fingerprint=(
            run.full_method_compatibility_fingerprint if valid else ""
        ),
        terminal_status="SUCCESS" if valid else "MISSING",
        metrics_available=valid,
        protocol_valid=valid,
        invalid_reasons=() if valid else (
            "MISSING_OR_INVALID_METRICS",
            "MISSING_OR_INVALID_RUN_MANIFEST",
        ),
        task_success=valid,
        physical_transactions=25 if valid else None,
        accepted_transactions=24 if valid else None,
        rejected_transactions=1 if valid else None,
        accepted_setup_transactions=2 if valid else None,
        accepted_progress_transactions=22 if valid else None,
        phase_completed=6 if valid else None,
        phase_total=6 if valid else None,
        phase_completion=1.0 if valid else None,
        fm_calls=12 if valid else None,
        input_tokens=100 if valid else None,
        output_tokens=20 if valid else None,
        wall_time_s=30.0 if valid else None,
        cost_usd=0.1 if valid else None,
        cost_complete=valid,
    )


def _run_map(suite):
    return {run.run_id: _record_for_run(run) for run in suite.new_runs}


def _row(document, table_id: str, condition: str):
    table = next(item for item in document["tables"] if item["table_id"] == table_id)
    return next(item for item in table["rows"] if item["condition"] == condition)


def test_complete_summary_joins_all_three_reuses_without_copying_results():
    lock = _lock()
    suite = build_suite("full", lock)
    audit = _source_audit(lock)
    records = _run_map(suite)

    document = summarize_paper_tables(
        suite,
        records,
        table1_source_audit=audit,
        require_complete=True,
    )

    assert document["complete"] is True
    assert [len(table["rows"]) for table in document["tables"]] == [5, 4, 3]
    assert all(
        row["observed_valid_seeds"] == list(SEEDS)
        for table in document["tables"]
        for row in table["rows"]
    )
    table2_full = _row(document, "table2_method_comparison", "coevolve_full")
    table3_base = _row(document, "table3_scene_generalization", "base")
    table3_mirrored = _row(document, "table3_scene_generalization", "mirrored")
    table4_empty = _row(document, "table4_memory_transfer", "empty")
    assert table2_full["source_reference"]["source_manifest"]["sha256"] == (
        audit.manifest_sha256
    )
    assert table3_base["source_reference"]["source_manifest"] == (
        table2_full["source_reference"]["source_manifest"]
    )
    assert table4_empty["source_reference"]["source_run_dirs"] == [
        records[run.run_id].run_dir
        for run in suite.new_runs
        if run.table_id.value == "table3_scene_generalization"
        and run.scene_id.value == "mirrored"
    ]
    assert table4_empty["cpt_at_80"] == table3_mirrored["cpt_at_80"]
    assert document["reuse_policy"] == {
        "mode": "reference_only",
        "result_directories_copied": False,
        "registered_reuse_rows": 3,
    }
    assert len(document["appendix"]["A4_reference_only_reuse_index"]) == 3


def test_partial_mirrored_row_propagates_null_to_table4_empty_with_seed_reason():
    lock = _lock()
    suite = build_suite("core", lock)
    records = _run_map(suite)
    missing_run = next(
        run
        for run in suite.new_runs
        if run.table_id.value == "table3_scene_generalization"
        and run.scene_id.value == "mirrored"
        and run.seed == 204
    )
    records[missing_run.run_id] = _record_for_run(missing_run, valid=False)

    document = summarize_paper_tables(
        suite,
        records,
        table1_source_audit=_source_audit(lock),
    )

    mirrored = _row(document, "table3_scene_generalization", "mirrored")
    empty = _row(document, "table4_memory_transfer", "empty")
    for row in (mirrored, empty):
        assert row["complete"] is False
        assert row["successful_runs"] is None
        assert row["success_rate"] is None
        assert row["cpt_at_80"] is None
        assert row["phase_completion"] is None
        assert row["rejected_transactions"] is None
        assert row["fm_calls"] is None
        assert row["observed_valid_seeds"] == [200, 201, 202, 203]
        assert row["missing_valid_seeds"] == [204]
        assert row["invalid_seed_details"][0]["seed"] == 204
        assert "MISSING_OR_INVALID_METRICS" in row["invalid_seed_details"][0][
            "reasons"
        ]
    assert empty["source_reference"]["aliased_valid_seeds"] == [200, 201, 202, 203]
    assert document["complete"] is False
    with pytest.raises(IncompletePaperSummaryError, match="table3.*mirrored"):
        summarize_paper_tables(
            suite,
            records,
            table1_source_audit=_source_audit(lock),
            require_complete=True,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("provider_route_fingerprint", "provider route fingerprint"),
        ("shared_prompt_contract_fingerprint", "shared prompt contract"),
        ("decision_prompt_fingerprint", "decision prompt"),
    ),
)
def test_external_reuse_rejects_hardened_source_identity_drift(field, message):
    lock = _lock()
    suite = build_suite("core", lock)
    audit = _source_audit(lock)
    if field == "provider_route_fingerprint":
        changed = replace(
            audit,
            provider_route_fingerprint=_digest(f"wrong-{field}"),
        )
    else:
        changed_source = replace(
            audit.source, **{field: _digest(f"wrong-{field}")}
        )
        changed = replace(audit, source=changed_source)

    with pytest.raises(PaperSummaryError, match=message):
        summarize_paper_tables(
            suite,
            _run_map(suite),
            table1_source_audit=changed,
        )


def test_internal_alias_rejects_prompt_identity_drift_even_if_marked_valid():
    lock = _lock()
    suite = build_suite("core", lock)
    records = _run_map(suite)
    mirrored = next(
        run
        for run in suite.new_runs
        if run.table_id.value == "table3_scene_generalization"
        and run.scene_id.value == "mirrored"
        and run.seed == 200
    )
    records[mirrored.run_id] = replace(
        records[mirrored.run_id],
        decision_prompt_fingerprint=_digest("tampered-decision-prompt"),
    )

    with pytest.raises(PaperSummaryError, match="decision_prompt"):
        summarize_paper_tables(
            suite,
            records,
            table1_source_audit=_source_audit(lock),
        )


def test_formal_summary_cli_requires_explicit_source_aggregate(capsys):
    assert main(
        [
            "summarize",
            "--results-root",
            "/tmp/table-results",
            "--plan",
            "/tmp/suite-plan.json",
        ]
    ) == 2
    error = json.loads(capsys.readouterr().err)
    assert "--table1-source-manifest" in error["error"]


def test_formal_summary_cli_threads_plan_records_and_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = build_suite("core", _lock())
    records = list(_run_map(suite).values())
    audit = _source_audit(_lock())
    plan = tmp_path / "suite-plan.json"
    plan.write_text("{}", encoding="utf-8")
    source = tmp_path / "strict-result-manifest.json"
    source.write_text("{}", encoding="utf-8")
    output = tmp_path / "summary.json"
    captured = {}

    monkeypatch.setattr(
        cli_module,
        "validate_suite_plan_document",
        lambda payload: SimpleNamespace(suite=suite),
    )
    monkeypatch.setattr(
        cli_module,
        "_collect_planned_runs",
        lambda results_root, plan_path: records,
    )

    def fake_audit(selected_suite, manifest_path, *, parent_manifest_path=None):
        captured["audit"] = (
            selected_suite,
            manifest_path,
            parent_manifest_path,
        )
        return audit

    def fake_summary(selected_suite, record_map, **kwargs):
        captured["summary"] = (selected_suite, record_map, kwargs)
        return {"kind": "test-paper-summary"}

    monkeypatch.setattr(cli_module, "audit_external_reuse_source", fake_audit)
    monkeypatch.setattr(cli_module, "summarize_paper_tables", fake_summary)

    assert main(
        [
            "summarize",
            "--results-root",
            str(tmp_path / "results"),
            "--plan",
            str(plan),
            "--table1-source-manifest",
            str(source),
            "--require-complete",
            "--output",
            str(output),
        ]
    ) == 0

    assert captured["audit"] == (suite, source, None)
    assert set(captured["summary"][1]) == {run.run_id for run in suite.new_runs}
    assert captured["summary"][2] == {
        "table1_source_audit": audit,
        "require_complete": True,
    }
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["suite_plan"]["path"] == str(plan.resolve())
    assert payload["results_root"] == str((tmp_path / "results").resolve())


def test_formal_collector_rejects_zero_total_fm_calls(tmp_path: Path) -> None:
    root = tmp_path / "zero-fm-calls"
    root.mkdir()
    expected = {
        "table": "table2_method_comparison",
        "condition": "remac_adapted",
        "method": "remac_adapted",
        "scene": "base",
        "memory_condition": "empty",
        "source_model": "",
        "target_model": "gpt-5.4-mini-2026-03-17",
        "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        "seed": 200,
        "protocol_id": "formal-test",
        "base_method_commit": "a" * 40,
        "condition_fingerprint": _digest("condition"),
        "invariant_fingerprint": _digest("invariant"),
        "scene_fingerprint": _digest("scene"),
        "terminal_fingerprint": _digest("terminal"),
        "observation_contract_fingerprint": _digest("observation"),
        "task_instruction_fingerprint": _digest("instruction"),
        "shared_prompt_contract_fingerprint": _digest("shared"),
            "decision_prompt_fingerprint": _digest("decision"),
            "skill_api_fingerprint": _digest("skill"),
            "full_method_compatibility_fingerprint": _digest("full-method"),
        }
    experiment = {
        **expected,
        "target_model": expected["target_model"],
    }
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "experiment": experiment,
                "protocol": {"protocol_id": expected["protocol_id"]},
                "codebase": {"git_head": expected["base_method_commit"]},
                "model": {
                    "resolved": expected["target_model"],
                    "provider": PRIMARY_GPT_PROVIDER_ROUTE.provider,
                    "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
                    "provider_route_fingerprint": (
                        PRIMARY_GPT_PROVIDER_ROUTE.fingerprint
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "task_success": True,
                "physical_iterations": 1,
                "accepted_step_count": 1,
                "rejected_step_count": 0,
                "accepted_setup_count": 0,
                "accepted_progress_count": 1,
                "elapsed_s": 1.0,
                "phase_completion": {"completed": 6, "total": 6, "fraction": 1.0},
                "model": {
                    "llm_calls": 0,
                    "vlm_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "billing_amount_usd": 0.0,
                    "billing_reconciliation_status": "complete",
                },
                "terminal": {"status": "SUCCESS"},
            }
        ),
        encoding="utf-8",
    )

    record = collect_run(root, expected=expected)

    assert not record.protocol_valid
    assert record.invalid_reasons == ("FM_CALL_COUNT_ZERO",)


def test_paper_summary_rechecks_positive_fm_calls_on_claimed_valid_record():
    lock = _lock()
    suite = build_suite("core", lock)
    records = _run_map(suite)
    run = suite.new_runs[0]
    records[run.run_id] = replace(records[run.run_id], fm_calls=0)

    with pytest.raises(PaperSummaryError, match="positive FM-call"):
        summarize_paper_tables(
            suite,
            records,
            table1_source_audit=_source_audit(lock),
        )
