from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import coevolve_experiments.cli as cli_module
import pytest
from coevolve_experiments.cli import main
from coevolve_experiments.contracts import (
    MethodDecisionPromptPin,
    MethodVariant,
    ProtocolLock,
    Table1SourceCohort,
)
from coevolve_experiments.lockfile import protocol_lock_document
from coevolve_experiments.full_method_compatibility import (
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
)
from coevolve_experiments.registry import build_suite
from coevolve_experiments.scenes import (
    EVIDENCE_CHECKS,
    EvidenceReceiptBinding,
    RawArtifactBinding,
    SceneValidationReport,
)
from coevolve_experiments.serialization import to_primitive
from coevolve_experiments.provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE,
    SECONDARY_GEMINI_PROVIDER_ROUTE,
)


def _lock() -> ProtocolLock:
    prompt = "9" * 64
    skill_api = "a" * 64
    full_method = FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": "1" * 64,
            "prompt_sha256": prompt,
            "route_spec_sha256": "2" * 64,
            "scene_sha256": "3" * 64,
            "skill_api_sha256": skill_api,
            "static_memory_sha256": "4" * 64,
            "verifier_config_sha256": "5" * 64,
            "verifier_source_sha256": "6" * 64,
        },
        behavior_artifacts={
            path: sha256(f"fixture:{path}".encode("utf-8")).hexdigest()
            for path in V83_FULL_BEHAVIOR_RELPATHS
        },
    )
    return ProtocolLock(
        protocol_id="post-table1-test-v1",
        invariant_fingerprint="1" * 64,
        terminal_fingerprint="2" * 64,
        base_method_commit="a" * 40,
        table1_gpt_source=Table1SourceCohort(
            backbone="gpt-5.4-mini-2026-03-17",
            protocol_id="coevolve-eval-v83-v33-efficient",
            method_commit="b" * 40,
            seeds=(200, 201, 202, 203, 204),
            condition_fingerprint="3" * 64,
            result_manifest_hash="4" * 64,
            provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
            provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            shared_prompt_contract_fingerprint=prompt,
            decision_prompt_fingerprint=prompt,
            full_method_compatibility_fingerprint=full_method.fingerprint,
        ),
        table1_gemini_source=Table1SourceCohort(
            backbone="gemini-3.5-flash",
            protocol_id="coevolve-eval-v83-v33-efficient-gemini",
            method_commit="c" * 40,
            seeds=(200, 201, 202, 203, 204),
            condition_fingerprint="b" * 64,
            result_manifest_hash="c" * 64,
            provider_route_id=SECONDARY_GEMINI_PROVIDER_ROUTE.route_id,
            provider_route_fingerprint=SECONDARY_GEMINI_PROVIDER_ROUTE.fingerprint,
            shared_prompt_contract_fingerprint=prompt,
            decision_prompt_fingerprint=prompt,
            full_method_compatibility_fingerprint=full_method.fingerprint,
        ),
        gpt_strategy_bank_hash="5" * 64,
        gemini_strategy_bank_hash="6" * 64,
        target_provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        target_provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        observation_contract_fingerprint="7" * 64,
        task_instruction_fingerprint="8" * 64,
        shared_prompt_contract_fingerprint=prompt,
        method_decision_prompt_pins=tuple(
            MethodDecisionPromptPin(variant, prompt) for variant in MethodVariant
        ),
        skill_api_fingerprint=skill_api,
        full_method_compatibility=full_method,
    )


def _write_lock(path: Path) -> None:
    path.write_text(
        json.dumps(protocol_lock_document(_lock())), encoding="utf-8"
    )


def test_plan_cli_defaults_to_complete_full_paper_matrix(tmp_path: Path):
    lock = tmp_path / "lock.json"
    output = tmp_path / "plan.json"
    _write_lock(lock)

    assert main(["plan", "--lock", str(lock), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["kind"] == "coevolve_tables_2_4_suite_plan"
    assert payload["protocol_lock"]["schema_version"] == 5
    assert payload["protocol_lock"]["table1_gpt_source"]["seeds"] == [
        200,
        201,
        202,
        203,
        204,
    ]
    assert payload["execution_preflight"] == "not_requested"
    assert payload["suite"]["run_plan"]["tier"] == "full"
    assert len(payload["suite"]["run_plan"]["new_runs"]) == 45
    assert len(payload["suite"]["run_plan"]["reuses"]) == 3
    with pytest.raises(ValueError, match="execution_preflight='passed'"):
        cli_module._verified_formal_suite(payload)


def test_scene_and_adapter_smoke_cli_never_claim_physics(capsys):
    assert main(["validate-scenes", "--mode", "planning"]) == 0
    planning = json.loads(capsys.readouterr().out)
    assert planning["ok"] is True

    assert main(["adapter-smoke", "--method", "all"]) == 0
    smoke = json.loads(capsys.readouterr().out)
    assert smoke["ok"] is True
    assert smoke["physics_dispatched"] is False
    assert smoke["execution_capability"] == "contract_smoke_only"
    assert smoke["paper_evidence"] is False
    assert len(smoke["results"]) == 5
    assert all(item["paper_evidence"] is False for item in smoke["results"])
    assert all(
        item["policy_representation"] == "team_intent_ir"
        for item in smoke["results"]
    )


def test_bind_live_launch_cli_writes_but_does_not_execute(monkeypatch, capsys):
    compiled = object()
    captured = {}
    monkeypatch.setattr(cli_module, "compile_live_launch", lambda *args, **kwargs: compiled)

    def fake_write(value, **kwargs):
        captured["compiled"] = value
        captured.update(kwargs)
        return SimpleNamespace(
            path=Path("/tmp/bound-launch.json"),
            sha256="1" * 64,
            config_fingerprint="2" * 64,
            launch=SimpleNamespace(run_spec=SimpleNamespace(run_id="run-1")),
            capability=SimpleNamespace(manifest_sha256="3" * 64),
        )

    monkeypatch.setattr(cli_module, "write_subprocess_launch_config", fake_write)

    assert main(
        [
            "bind-live-launch",
            "--plan",
            "/tmp/plan.json",
            "--run-id",
            "run-1",
            "--results-root",
            "/tmp/results",
            "--capability",
            "/tmp/capability.json",
            "--launch-config",
            "/tmp/bound-launch.json",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert captured["compiled"] is compiled
    assert captured["capability_manifest_path"] == Path("/tmp/capability.json")
    assert payload["paper_evidence"] is False
    assert payload["run_id"] == "run-1"


def test_run_live_cli_emits_only_attested_bridge_result(monkeypatch, capsys):
    captured = {}

    def fake_launch(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return {
            "run_id": "run-1",
            "paper_evidence_eligible": True,
        }

    monkeypatch.setattr(cli_module, "launch_live_subprocess", fake_launch)

    assert main(
        [
            "run-live",
            "--launch-config",
            "/tmp/bound-launch.json",
            "--timeout-seconds",
            "3600",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert captured == {
        "path": Path("/tmp/bound-launch.json"),
        "timeout_seconds": 3600.0,
    }
    assert payload["receipt"]["paper_evidence_eligible"] is True


def test_execution_ready_plan_fails_closed_without_sim_evidence(
    tmp_path: Path, capsys
):
    lock = tmp_path / "lock.json"
    _write_lock(lock)

    assert (
        main(
            [
                "plan",
                "--lock",
                str(lock),
                "--require-execution-ready",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert "execution preflight failed" in error["error"]
    assert "missing simulator evidence" in error["error"]
    assert "FULL_CODE_POLICY_BRIDGE_UNAVAILABLE" not in error["error"]


def test_execution_ready_plan_freezes_scenes_without_global_bridge(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "lock.json"
    output = tmp_path / "plan.json"
    _write_lock(lock_path)
    suite = build_suite("full", _lock())
    scene_ids = sorted(
        {run.scene_id for run in suite.new_runs},
        key=lambda item: item.value,
    )
    reports = {
        scene_id: _ready_report(scene_id, f"{index + 1:x}" * 64)
        for index, scene_id in enumerate(scene_ids)
    }
    monkeypatch.setattr(cli_module, "load_scene", lambda scene_id: scene_id)
    monkeypatch.setattr(
        cli_module,
        "validate_scene",
        lambda scene_id, **_kwargs: reports[scene_id],
    )

    assert main(
        [
            "plan",
            "--lock",
            str(lock_path),
            "--require-execution-ready",
            "--output",
            str(output),
        ]
    ) == 0
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execution_preflight"] == "passed"
    assert payload["execution_reports"] == [
        to_primitive(reports[scene_id]) for scene_id in scene_ids
    ]


def _ready_report(scene_id, fingerprint: str) -> SceneValidationReport:
    receipts = tuple(
        EvidenceReceiptBinding(
            check=check,
            receipt_path=f"evidence/{scene_id.value}/{check}.json",
            receipt_sha256="1" * 64,
            normalized_receipt_sha256="2" * 64,
            producer_id=f"synthetic.{check}",
            producer_version="test",
            producer_commit="a" * 40,
            command_sha256="3" * 64,
            config_sha256="4" * 64,
            measurements_sha256="5" * 64,
            raw_artifacts=(
                RawArtifactBinding(
                    artifact_type="trace",
                    relative_path=f"evidence/{scene_id.value}/{check}.jsonl",
                    sha256="6" * 64,
                    size_bytes=1,
                ),
            ),
        )
        for check in EVIDENCE_CHECKS
    )
    return SceneValidationReport(
        scene_id=scene_id,
        mode="execution",
        errors=(),
        execution_evidence_fingerprint=fingerprint,
        evidence_receipts=receipts,
    )


def test_formal_suite_freezes_execution_evidence_identity(monkeypatch) -> None:
    lock = _lock()
    suite = build_suite("core", lock)
    scene_ids = sorted(
        {run.scene_id for run in suite.new_runs},
        key=lambda item: item.value,
    )
    reports = {
        scene_id: _ready_report(scene_id, f"{index + 1:x}" * 64)
        for index, scene_id in enumerate(scene_ids)
    }
    payload = {
        "kind": "coevolve_tables_2_4_suite_plan",
        "schema_version": 2,
        "planning_validated": True,
        "execution_preflight": "passed",
        "protocol_lock": protocol_lock_document(lock),
        "suite": to_primitive(suite),
        "execution_reports": [to_primitive(reports[scene_id]) for scene_id in scene_ids],
    }
    monkeypatch.setattr(cli_module, "load_scene", lambda scene_id: scene_id)
    monkeypatch.setattr(
        cli_module,
        "validate_scene",
        lambda scene_id, **_kwargs: reports[scene_id],
    )

    assert cli_module._verified_formal_suite(payload) == payload["suite"]

    changed = json.loads(json.dumps(payload))
    changed["execution_reports"][0]["execution_evidence_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="do not match current hash-pinned preflight"):
        cli_module._verified_formal_suite(changed)


def test_lock_cli_rejects_unknown_fields(tmp_path: Path, capsys):
    lock = tmp_path / "bad-lock.json"
    payload = protocol_lock_document(_lock())
    payload["typo_hash"] = "7" * 64
    lock.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["plan", "--lock", str(lock)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "unknown=['typo_hash']" in error["error"]


def test_lock_cli_rejects_unknown_source_identity_fields(tmp_path: Path, capsys):
    lock = tmp_path / "bad-source-lock.json"
    payload = protocol_lock_document(_lock())
    payload["table1_gpt_source"]["typo"] = "unsafe"
    lock.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["plan", "--lock", str(lock)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "table1_gpt_source fields differ" in error["error"]
    assert "unknown=['typo']" in error["error"]


def test_export_cli_uses_lock_bound_table1_audit(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    lock_path = tmp_path / "lock.json"
    audit_path = tmp_path / "audit.json"
    output_path = tmp_path / "bank.json"
    evidence_path = tmp_path / "evidence"
    candidate_path = tmp_path / "candidate.json"
    _write_lock(lock_path)
    audit_path.write_text("{}", encoding="utf-8")
    captured = {}
    cohort_audit = object()
    package = object()
    candidate = SimpleNamespace(file_sha256="6" * 64)

    class ExportResult:
        bank = {"records": []}
        provenance = {"bank_sha256": "5" * 64}
        bank_path = output_path
        provenance_path = tmp_path / "bank.provenance.json"

    def fake_audit(manifest_path, source, **kwargs):
        captured["manifest_path"] = manifest_path
        captured["source"] = source
        captured["audit_kwargs"] = kwargs
        return cohort_audit

    def fake_export(source_runs, audit, **kwargs):
        captured["source_runs"] = source_runs
        captured["cohort_audit"] = audit
        captured.update(kwargs)
        return ExportResult()

    def fake_load_package(path, audit):
        captured["evidence_package"] = path
        captured["package_audit"] = audit
        return package

    def fake_validate_candidate(result, audit, loaded_package, cohort, path, **kwargs):
        captured["candidate"] = (result, audit, loaded_package, cohort, path, kwargs)
        return candidate

    def fake_write(result, bank_path, provenance_path):
        captured["persisted"] = (result, bank_path, provenance_path)
        return result

    monkeypatch.setattr(cli_module, "audit_table1_source_cohort", fake_audit)
    monkeypatch.setattr(
        cli_module,
        "export_audited_table1_strategy_bank",
        fake_export,
    )
    monkeypatch.setattr(cli_module, "load_posthoc_source_evidence", fake_load_package)
    monkeypatch.setattr(
        cli_module,
        "validate_locked_strategy_bank_candidate",
        fake_validate_candidate,
    )
    monkeypatch.setattr(cli_module, "write_strategy_bank_export", fake_write)
    args = ["export-strategy-bank"]
    for index in range(5):
        args.extend(("--source-run", str(tmp_path / f"seed_{index}")))
    args.extend(
        (
            "--audit",
            str(audit_path),
            "--lock",
            str(lock_path),
            "--source-cohort",
            "gpt",
            "--evidence-package",
            str(evidence_path),
            "--candidate-attestation",
            str(candidate_path),
            "--output",
            str(output_path),
        )
    )

    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert captured["manifest_path"] == audit_path
    assert captured["source"] == _lock().table1_gpt_source
    assert captured["audit_kwargs"] == {}
    assert captured["cohort_audit"] is cohort_audit
    assert captured["source_runs"] == [
        tmp_path / f"seed_{index}" for index in range(5)
    ]
    assert captured["posthoc_evidence_package"] == evidence_path
    assert captured["expected_bank_sha256"] == "5" * 64
    assert captured["evidence_package"] == evidence_path
    assert captured["candidate"][2] is package
    assert captured["candidate"][3:] == (
        "gpt",
        candidate_path,
        {"locked_bank_sha256": "5" * 64},
    )
    assert captured["persisted"] == (captured["candidate"][0], output_path, None)


def test_lock_cli_rejects_bare_unversioned_mapping(tmp_path: Path, capsys):
    lock = tmp_path / "unversioned-lock.json"
    payload = protocol_lock_document(_lock())
    payload.pop("kind")
    payload.pop("schema_version")
    lock.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["plan", "--lock", str(lock)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "requires kind and schema_version" in error["error"]


@pytest.mark.parametrize("legacy_schema", (3, 4))
def test_legacy_lock_cannot_guess_full_method_identity(
    tmp_path: Path,
    capsys,
    legacy_schema: int,
):
    lock = tmp_path / f"legacy-v{legacy_schema}-lock.json"
    payload = protocol_lock_document(_lock())
    payload["schema_version"] = legacy_schema
    lock.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["plan", "--lock", str(lock)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "cannot be inferred from an older lock" in error["error"]


def test_planned_collector_forwards_table4_strategy_bank_hash(
    tmp_path: Path,
    monkeypatch,
):
    expected_hash = "7" * 64
    plan = tmp_path / "plan.json"
    suite = {
        "run_plan": {
            "new_runs": [
                {
                    "table_id": "table4_memory_transfer",
                    "output_relpath": (
                        "table4_memory_transfer/gpt_to_gpt/seed_000"
                    ),
                    "method_variant": "coevolve_full",
                    "scene_id": "mirrored",
                    "memory_condition": "gpt_to_gpt",
                    "strategy_bank": {
                        "source_backbone": "gpt-source",
                        "sha256": expected_hash,
                    },
                    "target_backbone": "gpt-target",
                    "seed": 0,
                    "physical_budget": 80,
                }
            ]
        }
    }
    plan.write_text(
        json.dumps({"suite": suite, "execution_reports": []}),
        encoding="utf-8",
    )
    captured = []

    def fake_collect(path, *, expected, physical_budget):
        captured.append((path, expected, physical_budget))
        return object()

    monkeypatch.setattr(cli_module, "collect_run", fake_collect)
    monkeypatch.setattr(
        cli_module,
        "_verified_formal_suite",
        lambda payload: suite,
    )
    monkeypatch.setattr(
        "coevolve_experiments.reporting.scene_result_binding."
        "resolve_scene_result_binding",
        lambda *args, **kwargs: SimpleNamespace(
            scene_realization_sha256="a" * 64,
            scene_execution_evidence_fingerprint="b" * 64,
            compiled_model_sha256="c" * 64,
        ),
    )

    records = cli_module._collect_planned_runs(tmp_path / "results", plan)

    assert len(records) == 1
    assert captured[0][1]["strategy_bank_sha256"] == expected_hash
    assert captured[0][1]["source_model"] == "gpt-source"
