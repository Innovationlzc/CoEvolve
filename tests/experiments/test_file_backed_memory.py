from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from coevolve_experiments.cli import main
from coevolve_experiments.contracts import (
    MemoryCondition,
    MethodDecisionPromptPin,
    MethodVariant,
    ProtocolLock,
    Table1SourceCohort,
)
from coevolve_experiments.provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE,
    SECONDARY_GEMINI_PROVIDER_ROUTE,
)
from coevolve_experiments.lockfile import protocol_lock_document
from coevolve_experiments.full_method_compatibility import (
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
)
from coevolve_experiments.memory import (
    FROZEN_STRATEGY_BANK_POLICY,
    MemoryRuntimeError,
    bind_table4_adapter,
    prepare_transfer_from_plan,
)
from coevolve_experiments.memory.schema import semantic_sha256
from coevolve_experiments.memory import formal_prepare as formal_prepare_module
from coevolve_experiments.methods import (
    ActionExemplarQuery,
    AdapterContext,
    ScriptedRoleBackend,
)
from coevolve_experiments.registry import build_suite
from coevolve_experiments.reporting import collect_run
from coevolve_experiments.runtime import (
    CheckpointRef,
    ExperimentEngine,
    Observation,
    PolicyValidator,
    RolloutRecord,
    RunAccounting,
    ScriptedRuntime,
    VerifierOutcome,
    make_team_proposal,
)
from coevolve_experiments.runtime.contracts import PhysicalDispatchResult
from coevolve_experiments.serialization import to_primitive
from coevolve_experiments.scenes import (
    EVIDENCE_CHECKS,
    EvidenceReceiptBinding,
    RawArtifactBinding,
    SceneValidationReport,
)


def _record(label: str = "stable") -> dict[str, str]:
    return {
        "phase_trigger": "during supported payload transport",
        "failure_condition": "the verifier reports cooperative instability",
        "coordination_pattern": f"synchronous support pattern {label}",
        "verified_outcome": "accepted",
        "avoid_rule": "avoid repeating an unstable coordination pattern",
        "repair_direction": "change role compatibility while preserving support",
    }


def _write_formal_inputs(tmp_path: Path):
    bank = {"schema_version": 1, "records": [_record()]}
    bank_sha256 = semantic_sha256(bank)
    prompt = "9" * 64
    skill_api = "b" * 64
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
    lock = ProtocolLock(
        protocol_id="post-table1-memory-test-v1",
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
            condition_fingerprint="c" * 64,
            result_manifest_hash="d" * 64,
            provider_route_id=SECONDARY_GEMINI_PROVIDER_ROUTE.route_id,
            provider_route_fingerprint=SECONDARY_GEMINI_PROVIDER_ROUTE.fingerprint,
            shared_prompt_contract_fingerprint=prompt,
            decision_prompt_fingerprint=prompt,
            full_method_compatibility_fingerprint=full_method.fingerprint,
        ),
        gpt_strategy_bank_hash=bank_sha256,
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
    suite = build_suite("core", lock)
    run = next(
        item
        for item in suite.new_runs
        if item.memory_condition is MemoryCondition.GPT_TO_GPT and item.seed == 200
    )
    plan = {
        "kind": "coevolve_tables_2_4_suite_plan",
        "schema_version": 2,
        "planning_validated": True,
        "execution_preflight": "passed",
        "protocol_lock": protocol_lock_document(lock),
        "suite": to_primitive(suite),
        "execution_reports": [],
    }
    plan_path = (tmp_path / "suite_plan.json").resolve()
    bank_path = tmp_path / "strategy_bank.json"
    provenance_path = tmp_path / "strategy_bank.provenance.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    bank_path.write_text(json.dumps(bank, sort_keys=True), encoding="utf-8")
    provenance = {
        "kind": "strategy_transfer_provenance",
        "schema_version": 3,
        "bank_sha256": bank_sha256,
        "source_model": run.strategy_bank.source_backbone,
        "source_provider_route_id": run.strategy_bank.source_provider_route_id,
        "source_provider_route_fingerprint": (
            run.strategy_bank.source_provider_route_fingerprint
        ),
        "source_scene": "base",
        "source_protocol": run.strategy_bank.source_protocol_id,
        "source_method_commit": run.strategy_bank.source_method_commit,
        "source_seeds": list(run.strategy_bank.source_seeds),
        "source_condition_fingerprint": (
            run.strategy_bank.source_condition_fingerprint
        ),
        "source_result_manifest_hash": (
            run.strategy_bank.source_result_manifest_hash
        ),
        "policy": FROZEN_STRATEGY_BANK_POLICY.as_dict(),
        "summary": {},
        "sources": [],
    }
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True),
        encoding="utf-8",
    )
    return run, plan_path, bank_path, provenance_path


def _install_synthetic_execution_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run,
    plan_path: Path,
) -> None:
    """Bind synthetic unit fixtures to one internally consistent preflight."""

    receipts = tuple(
        EvidenceReceiptBinding(
            check=check,
            receipt_path=f"synthetic/{check}.json",
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
                    artifact_type="synthetic",
                    relative_path=f"synthetic/{check}.raw",
                    sha256="6" * 64,
                    size_bytes=1,
                ),
            ),
        )
        for check in EVIDENCE_CHECKS
    )
    report = SceneValidationReport(
        scene_id=run.scene_id,
        mode="execution",
        errors=(),
        execution_evidence_fingerprint="f" * 64,
        evidence_receipts=receipts,
    )
    assert report.execution_ready
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["execution_reports"] = [to_primitive(report)]
    plan_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        formal_prepare_module,
        "validate_scene",
        lambda *_args, **_kwargs: report,
    )


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run, plan, bank, provenance = _write_formal_inputs(tmp_path)
    _install_synthetic_execution_preflight(
        monkeypatch,
        run=run,
        plan_path=plan,
    )
    prepared = prepare_transfer_from_plan(
        plan_path=plan,
        run_id=run.run_id,
        results_root=(tmp_path / "results").resolve(),
        bank_path=bank,
        provenance_path=provenance,
    )
    return run, prepared


def test_plan_pinned_store_is_retrieved_promoted_and_engine_finalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run, prepared = _prepare(tmp_path, monkeypatch)
    proposal = make_team_proposal("advance")
    backend = ScriptedRoleBackend(
        [
            (
                "task_orchestrator",
                {
                    "updated_plan": {"plan": "p0"},
                    "committed_strategy": {
                        "primitive": "cooperative_move",
                        "semantic_objective": "advance",
                        "coordination_mode": "synchronized",
                    },
                },
            ),
            ("policy_generator", proposal),
            ("memory_update", {"strategy": "retain verified cooperative support"}),
        ]
    )
    binding = bind_table4_adapter(run, backend, prepared.memory_dir)
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=True, report="done")],
        initial_observation=Observation(0, {"phase": "carry"}),
    )

    result = ExperimentEngine(binding.adapter, runtime.callbacks()).run("carry")

    assert result.success
    overlay = json.loads(prepared.bootstrap.online_overlay_path.read_text())
    actions = json.loads(prepared.bootstrap.action_exemplars_path.read_text())
    audit = json.loads(binding.memory.runtime_audit_path.read_text())
    assert len(overlay["entries"]) == 1
    assert overlay["entries"][0]["proposal"] == {
        "phase_trigger": "corridor_transport",
        "failure_condition": (
            "no verifier-reported failure under the applicable cooperative state"
        ),
        "coordination_pattern": "retain verified cooperative support",
        "verified_outcome": "accepted",
        "avoid_rule": (
            "do not repeat a verifier-rejected coordination pattern without a "
            "material repair"
        ),
        "repair_direction": (
            "change the coordination abstraction while preserving verified support"
        ),
    }
    assert overlay["entries"][0]["provenance"]["verifier_outcome"] == "accepted"
    assert len(actions["exemplars"]) == 1
    assert audit["lifecycle_state"] == "finalized"
    assert audit["finalized"] is True
    assert audit["violations"] == []
    assert audit["seed"]["preflight_sha256"] == run.strategy_bank.sha256
    assert audit["seed"]["final_sha256"] == run.strategy_bank.sha256
    assert audit["online_overlay"]["initial_record_count"] == 0
    assert audit["online_overlay"]["final_record_count"] == 1
    assert audit["fairness_fingerprints"] == {
        "observation_contract_fingerprint": run.observation_contract_fingerprint,
        "task_instruction_fingerprint": run.task_instruction_fingerprint,
        "shared_prompt_contract_fingerprint": (
            run.shared_prompt_contract_fingerprint
        ),
        "decision_prompt_fingerprint": run.decision_prompt_fingerprint,
            "provider_route_fingerprint": run.provider_route_fingerprint,
            "skill_api_fingerprint": run.skill_api_fingerprint,
            "full_method_compatibility_fingerprint": (
                run.full_method_compatibility_fingerprint
            ),
        }
    # Retrieval is sequential: the Orchestrator sees strategy records, while
    # only the Coder sees post-commit executable examples.
    assert backend.calls[0].payload["strategy_records"] == [_record()]
    assert "action_exemplars" not in backend.calls[0].payload
    assert "strategy_records" not in backend.calls[1].payload
    assert backend.calls[1].payload["action_exemplars"] == []
    assert backend.calls[1].payload["action_exemplar_query"] == {
            "primitive": "cooperative_move",
            "semantic_objective": "advance",
            "coordination_mode": "synchronized",
            "active_phase": "carry",
        }

    _write_minimal_result(prepared.run_dir, run)
    expected = {
        "run_id": run.run_id,
        "table": run.table_id.value,
        "condition": "gpt_to_gpt",
        "method": run.method_variant.value,
        "scene": run.scene_id.value,
        "memory_condition": run.memory_condition.value,
        "source_model": run.strategy_bank.source_backbone,
        "source_provider_route_id": run.strategy_bank.source_provider_route_id,
        "source_provider_route_fingerprint": (
            run.strategy_bank.source_provider_route_fingerprint
        ),
        "strategy_bank_sha256": run.strategy_bank.sha256,
        "strategy_bank": to_primitive(run.strategy_bank),
        "target_model": run.target_backbone,
        "provider_route_id": run.provider_route_id,
        "provider_route_fingerprint": run.provider_route_fingerprint,
        "seed": run.seed,
        "protocol_id": run.protocol_id,
        "base_method_commit": run.base_method_commit,
        "condition_fingerprint": run.condition_fingerprint,
        "invariant_fingerprint": run.invariant_fingerprint,
        "scene_fingerprint": run.scene_fingerprint,
        "terminal_fingerprint": run.terminal_fingerprint,
        "observation_contract_fingerprint": run.observation_contract_fingerprint,
        "task_instruction_fingerprint": run.task_instruction_fingerprint,
        "shared_prompt_contract_fingerprint": (
            run.shared_prompt_contract_fingerprint
        ),
        "decision_prompt_fingerprint": run.decision_prompt_fingerprint,
        "skill_api_fingerprint": run.skill_api_fingerprint,
        "full_method_compatibility_fingerprint": (
            run.full_method_compatibility_fingerprint
        ),
        "plan_path": str(prepared.plan_path),
        "plan_sha256": prepared.plan_sha256,
    }
    collected = collect_run(prepared.run_dir, expected=expected)
    assert collected.protocol_valid, collected.invalid_reasons
    assert collected.strategy_bank_sha256 == run.strategy_bank.sha256
    assert {
        "memory_bootstrap_audit",
        "memory_bootstrap_snapshot",
        "memory_runtime_audit",
    } <= set(collected.artifact_hashes)


def test_action_exemplar_retrieval_is_exact_phase_ranked_and_capped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _run, prepared = _prepare(tmp_path, monkeypatch)
    binding = bind_table4_adapter(
        prepared.run_spec,
        ScriptedRoleBackend([]),
        prepared.memory_dir,
    )

    def promote(
        index: int,
        *,
        primitive: str = "cooperative_move",
        phase: str,
    ) -> None:
        proposal = make_team_proposal(
            f"step-{index}",
            primitive=primitive,
            objective="advance",
            mode="synchronized",
            lifecycle_phase=phase,
        )
        policy = PolicyValidator().validate(proposal)
        rollout = RolloutRecord(
            transaction_index=index,
            checkpoint=CheckpointRef("c0"),
            observation_before=Observation(index - 1, {"phase": phase}),
            policy=policy,
            dispatch=PhysicalDispatchResult(
                True,
                True,
                Observation(index, {"phase": phase}),
            ),
            verifier=VerifierOutcome(True, report="accepted"),
            observation_after=Observation(index, {"phase": phase}),
        )
        binding.memory.promote_action_exemplar({}, rollout)

    promote(1, phase="carry")
    promote(2, phase="place")
    promote(3, phase="place")
    promote(4, primitive="coordinated_pivot", phase="carry")
    context = AdapterContext(
        instruction="carry",
        observation=Observation(4, {"phase": "carry"}),
        checkpoint=CheckpointRef("c0"),
        remaining_physical_budget=76,
        accepted_manifest=(),
        latest_physical_failure=None,
        blocked_ast_fingerprints=frozenset(),
        accounting=RunAccounting(),
    )
    query = ActionExemplarQuery(
        primitive="cooperative_move",
        semantic_objective="advance",
        coordination_mode="synchronized",
        active_phase="carry",
    )

    retrieved = binding.memory.retrieve_action_exemplars(context, query)

    assert [item["transaction_index"] for item in retrieved] == [1, 3]
    assert binding.memory.retrieve_action_exemplars(
        context,
        ActionExemplarQuery(
            primitive="cooperative_move",
            semantic_objective="different-objective",
            coordination_mode="synchronized",
            active_phase="carry",
        ),
    ) == ()


def test_seed_writable_or_drift_marks_lifecycle_violated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run, prepared = _prepare(tmp_path, monkeypatch)
    binding = bind_table4_adapter(run, ScriptedRoleBackend([]), prepared.memory_dir)
    prepared.bootstrap.seed_bank_path.chmod(0o644)

    with pytest.raises(MemoryRuntimeError, match="became writable"):
        binding.memory.retrieve_strategies(None)  # type: ignore[arg-type]

    audit = json.loads(binding.memory.runtime_audit_path.read_text())
    assert audit["lifecycle_state"] == "violated"
    assert audit["finalized"] is False
    assert audit["violations"][0]["operation"].endswith(":pre")


def test_runtime_rejects_symlinked_memory_ancestor_before_audit_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    run, prepared = _prepare(real_parent, monkeypatch)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_memory = linked_parent / prepared.memory_dir.relative_to(real_parent)

    with pytest.raises(MemoryRuntimeError, match="ancestors is a symlink"):
        bind_table4_adapter(run, ScriptedRoleBackend([]), linked_memory)

    assert not (prepared.memory_dir / "memory_runtime_audit.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("physical_state", "s1", "bootstrap snapshot preflight failed"),
        ("checkpoint", "c1", "bootstrap snapshot preflight failed"),
    ),
)
def test_runtime_requires_exact_fresh_s0_c0_before_audit_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
):
    run, prepared = _prepare(tmp_path, monkeypatch)
    bootstrap_path = prepared.memory_dir / "memory_bootstrap_audit.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["fresh_target_state"][field] = value
    bootstrap_path.chmod(0o644)
    bootstrap_path.write_text(json.dumps(bootstrap, sort_keys=True), encoding="utf-8")

    with pytest.raises(MemoryRuntimeError, match=message):
        bind_table4_adapter(run, ScriptedRoleBackend([]), prepared.memory_dir)

    assert not (prepared.memory_dir / "memory_runtime_audit.json").exists()


def test_runtime_rejects_uppercase_provenance_hash_before_audit_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run, prepared = _prepare(tmp_path, monkeypatch)
    bootstrap_path = prepared.memory_dir / "memory_bootstrap_audit.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["seed"]["provenance_sha256"] = bootstrap["seed"][
        "provenance_sha256"
    ].upper()
    bootstrap_path.chmod(0o644)
    bootstrap_path.write_text(json.dumps(bootstrap, sort_keys=True), encoding="utf-8")

    with pytest.raises(MemoryRuntimeError, match="bootstrap snapshot preflight failed"):
        bind_table4_adapter(run, ScriptedRoleBackend([]), prepared.memory_dir)

    assert not (prepared.memory_dir / "memory_runtime_audit.json").exists()


def test_runtime_finalize_rejects_promotion_counter_store_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run, prepared = _prepare(tmp_path, monkeypatch)
    binding = bind_table4_adapter(run, ScriptedRoleBackend([]), prepared.memory_dir)
    binding.memory._audit.payload["operations"]["strategy_promotions"] = 1

    with pytest.raises(MemoryRuntimeError, match="strategy promotion counter"):
        binding.memory.finalize()

    audit = json.loads(binding.memory.runtime_audit_path.read_text(encoding="utf-8"))
    assert audit["lifecycle_state"] == "violated"
    assert audit["finalized"] is False


def test_formal_resolver_rejects_tampered_runspec_before_creating_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run, plan_path, bank_path, provenance_path = _write_formal_inputs(tmp_path)
    payload = json.loads(plan_path.read_text())
    selected = next(
        item
        for item in payload["suite"]["run_plan"]["new_runs"]
        if item["run_id"] == run.run_id
    )
    selected["strategy_bank"]["sha256"] = "f" * 64
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    results = (tmp_path / "results").resolve()

    with pytest.raises(ValueError, match="does not exactly match"):
        prepare_transfer_from_plan(
            plan_path=plan_path,
            run_id=run.run_id,
            results_root=results,
            bank_path=bank_path,
            provenance_path=provenance_path,
        )

    assert not results.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_protocol", "cross-cohort-protocol"),
        ("source_method_commit", "f" * 40),
        ("source_seeds", [200, 201, 202, 203, 999]),
        ("source_condition_fingerprint", "e" * 64),
        ("source_result_manifest_hash", "d" * 64),
    ),
)
def test_formal_prepare_rejects_rehashed_provenance_identity_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
):
    run, plan_path, bank_path, provenance_path = _write_formal_inputs(tmp_path)
    _install_synthetic_execution_preflight(
        monkeypatch,
        run=run,
        plan_path=plan_path,
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance[field] = replacement
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    results = (tmp_path / "results").resolve()

    with pytest.raises(ValueError, match=field):
        prepare_transfer_from_plan(
            plan_path=plan_path,
            run_id=run.run_id,
            results_root=results,
            bank_path=bank_path,
            provenance_path=provenance_path,
        )

    assert not results.exists()


def test_formal_prepare_requires_scene_evidence_not_global_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run, plan_path, bank_path, provenance_path = _write_formal_inputs(tmp_path)
    _install_synthetic_execution_preflight(
        monkeypatch,
        run=run,
        plan_path=plan_path,
    )
    results = (tmp_path / "results").resolve()

    prepared = prepare_transfer_from_plan(
        plan_path=plan_path,
        run_id=run.run_id,
        results_root=results,
        bank_path=bank_path,
        provenance_path=provenance_path,
    )

    assert prepared.run_spec == run
    assert prepared.memory_dir.is_dir()


def test_prepare_transfer_cli_derives_identity_from_plan(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
):
    run, plan_path, bank_path, provenance_path = _write_formal_inputs(tmp_path)
    _install_synthetic_execution_preflight(
        monkeypatch,
        run=run,
        plan_path=plan_path,
    )
    results = (tmp_path / "results").resolve()

    exit_code = main(
        [
            "prepare-transfer",
            "--plan",
            str(plan_path),
            "--run-id",
            run.run_id,
            "--results-root",
            str(results),
            "--bank",
            str(bank_path),
            "--provenance",
            str(provenance_path),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["run_id"] == run.run_id
    assert Path(output["memory_dir"]) == results / run.output_relpath / "memory"
    assert output["seed_bank_sha256"] == run.strategy_bank.sha256


def test_formal_prepare_rejects_missing_selected_scene_execution_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run, plan_path, bank_path, provenance_path = _write_formal_inputs(tmp_path)
    with pytest.raises(ValueError, match="exactly one execution report"):
        prepare_transfer_from_plan(
            plan_path=plan_path,
            run_id=run.run_id,
            results_root=(tmp_path / "results").resolve(),
            bank_path=bank_path,
            provenance_path=provenance_path,
        )


def test_low_level_bootstrap_cannot_open_as_formal_without_plan_binding(tmp_path: Path):
    from coevolve_experiments.memory import bootstrap_transfer_memory

    run, _plan, bank_path, provenance_path = _write_formal_inputs(tmp_path)
    low_level = bootstrap_transfer_memory(
        tmp_path / "low-level",
        "GPT→GPT",
        seed_bank=bank_path,
        seed_provenance=provenance_path,
        source_model=run.strategy_bank.source_backbone,
        target_model=run.target_backbone,
    )

    with pytest.raises(MemoryRuntimeError, match="formal_binding"):
        bind_table4_adapter(run, ScriptedRoleBackend([]), low_level.audit_path.parent)


def _write_minimal_result(run_dir: Path, run) -> None:
    experiment = {
        "run_id": run.run_id,
        "table": run.table_id.value,
        "condition": "gpt_to_gpt",
        "method": run.method_variant.value,
        "scene": run.scene_id.value,
        "memory_condition": run.memory_condition.value,
        "source_model": run.strategy_bank.source_backbone,
        "source_provider_route_id": run.strategy_bank.source_provider_route_id,
        "source_provider_route_fingerprint": (
            run.strategy_bank.source_provider_route_fingerprint
        ),
        "target_model": run.target_backbone,
        "provider_route_id": run.provider_route_id,
        "provider_route_fingerprint": run.provider_route_fingerprint,
        "seed": run.seed,
        "protocol_id": run.protocol_id,
        "base_method_commit": run.base_method_commit,
        "condition_fingerprint": run.condition_fingerprint,
        "invariant_fingerprint": run.invariant_fingerprint,
        "scene_fingerprint": run.scene_fingerprint,
        "terminal_fingerprint": run.terminal_fingerprint,
        "observation_contract_fingerprint": run.observation_contract_fingerprint,
        "task_instruction_fingerprint": run.task_instruction_fingerprint,
        "shared_prompt_contract_fingerprint": (
            run.shared_prompt_contract_fingerprint
        ),
        "decision_prompt_fingerprint": run.decision_prompt_fingerprint,
        "skill_api_fingerprint": run.skill_api_fingerprint,
        "full_method_compatibility_fingerprint": (
            run.full_method_compatibility_fingerprint
        ),
        "output_relpath": run.output_relpath,
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
        "model": {
            "llm_calls": 3,
            "vlm_calls": 0,
            "input_tokens": 10,
            "output_tokens": 4,
            "billing_amount_usd": 0.01,
            "billing_reconciliation_status": "complete",
        },
        "terminal": {"status": "SUCCESS"},
    }
    manifest = {
        "status": "SUCCESS",
        "experiment": experiment,
        "codebase": {"git_head": run.base_method_commit},
        "model": {
            "requested": run.target_backbone,
            "resolved": run.target_backbone,
            "provider": "provider_a",
            "provider_route_id": run.provider_route_id,
            "provider_route_fingerprint": run.provider_route_fingerprint,
        },
    }
    (run_dir / "run_plan.json").write_text(
        json.dumps(to_primitive(run), sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
