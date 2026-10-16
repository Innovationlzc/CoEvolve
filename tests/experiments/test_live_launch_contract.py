from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

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
from coevolve_experiments.live.launch import (
    FROM_SCRATCH_BOOTSTRAP_MISSING_CODE,
    LiveLaunchConfigError,
    compile_live_launch,
)
from coevolve_experiments.lockfile import protocol_lock_document
from coevolve_experiments.memory import prepare_from_scratch_memory
from coevolve_experiments.registry import build_suite
from coevolve_experiments.serialization import to_primitive, write_json
from coevolve_experiments.suite_plan import load_verified_suite_plan


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _lock() -> ProtocolLock:
    prompt = _digest("prompts")
    skill_api = _digest("skills")
    full_method = FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": _digest("initial-memory"),
            "prompt_sha256": prompt,
            "route_spec_sha256": _digest("route"),
            "scene_sha256": _digest("base-scene"),
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
    return ProtocolLock(
        protocol_id="post-table1-test-v1",
        invariant_fingerprint=_digest("invariant"),
        terminal_fingerprint=_digest("terminal"),
        base_method_commit="a" * 40,
        table1_gpt_source=Table1SourceCohort(
            backbone="gpt-5.4-mini-2026-03-17",
            protocol_id="coevolve-eval-v83-v33-efficient",
            method_commit="b" * 40,
            seeds=(200, 201, 202, 203, 204),
            condition_fingerprint=_digest("table1-gpt-condition"),
            result_manifest_hash=_digest("table1-gpt-results"),
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
            condition_fingerprint=_digest("table1-gemini-condition"),
            result_manifest_hash=_digest("table1-gemini-results"),
            provider_route_id=SECONDARY_GEMINI_PROVIDER_ROUTE.route_id,
            provider_route_fingerprint=SECONDARY_GEMINI_PROVIDER_ROUTE.fingerprint,
            shared_prompt_contract_fingerprint=prompt,
            decision_prompt_fingerprint=prompt,
            full_method_compatibility_fingerprint=full_method.fingerprint,
        ),
        gpt_strategy_bank_hash=_digest("gpt-bank"),
        gemini_strategy_bank_hash=_digest("gemini-bank"),
        target_provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        target_provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        observation_contract_fingerprint=_digest("observation"),
        task_instruction_fingerprint=_digest("instruction"),
        shared_prompt_contract_fingerprint=prompt,
        method_decision_prompt_pins=tuple(
            MethodDecisionPromptPin(variant, prompt) for variant in MethodVariant
        ),
        skill_api_fingerprint=skill_api,
        full_method_compatibility=full_method,
    )


def _plan(tmp_path: Path) -> tuple[Path, object]:
    suite = build_suite("core", _lock())
    path = tmp_path / "core_plan.json"
    write_json(
        path,
        {
            "kind": "coevolve_tables_2_4_suite_plan",
            "schema_version": 2,
            "planning_validated": True,
            "execution_preflight": "not_requested",
            "protocol_lock": protocol_lock_document(suite.protocol_lock),
            "suite": to_primitive(suite),
            "execution_reports": [],
        },
    )
    return path, suite


def _run_id(suite: object, variant: MethodVariant) -> str:
    return next(
        run.run_id
        for run in suite.new_runs  # type: ignore[attr-defined]
        if run.method_variant is variant and run.seed == 200
    )


def test_plan_is_rebuilt_and_one_run_resolves_exactly(tmp_path: Path) -> None:
    plan_path, suite = _plan(tmp_path)
    loaded = load_verified_suite_plan(plan_path)
    run_id = _run_id(suite, MethodVariant.REMAC_ADAPTED)

    assert loaded.resolve_run(run_id).run_id == run_id
    assert loaded.sha256
    with pytest.raises(ValueError, match="exactly once"):
        loaded.resolve_run("missing")


def test_compile_is_plan_bound_and_fails_closed_before_physics(tmp_path: Path) -> None:
    plan_path, suite = _plan(tmp_path)
    run_id = _run_id(suite, MethodVariant.REMAC_ADAPTED)
    result = compile_live_launch(
        plan_path,
        run_id,
        results_root=tmp_path / "formal_results",
    )

    assert result.run_spec.seed == 200
    assert result.output_dir == (
        tmp_path / "formal_results" / result.run_spec.output_relpath
    ).resolve()
    assert result.method.decision_structure == "task_plan_reflection_and_replanning"
    assert result.method.policy_representation.value == "praxis_code_as_policy"
    assert result.method.common_physics_verifier
    assert result.method.common_checkpoint_rollback
    assert result.scene.injection_channels == (
        "term1_simulator_asset",
        "host_clearance_geometry",
        "vlm_scene_prior",
        "route_and_progress_gates",
        "formal_run_manifest",
    )
    assert not result.paper_evidence_eligible
    assert "SCENE_EXECUTION_PREFLIGHT_NOT_FROZEN" in result.blockers
    assert "FULL_CODE_POLICY_BRIDGE_UNAVAILABLE" in result.blockers
    assert FROM_SCRATCH_BOOTSTRAP_MISSING_CODE in result.blockers


def test_compile_binds_plan_derived_from_scratch_bootstrap(tmp_path: Path) -> None:
    plan_path, suite = _plan(tmp_path)
    run_id = _run_id(suite, MethodVariant.FLAT_REFLECTIVE_REPAIR)
    run = next(item for item in suite.new_runs if item.run_id == run_id)
    results_root = tmp_path / "formal_results"
    output_dir = (results_root / run.output_relpath).resolve()
    plan_sha = sha256(plan_path.read_bytes()).hexdigest()
    prepared = prepare_from_scratch_memory(
        run,
        output_dir / "memory",
        plan_path=plan_path.resolve(),
        plan_sha256=plan_sha,
    )

    result = compile_live_launch(
        plan_path,
        run_id,
        results_root=results_root,
    )

    assert result.memory.bootstrap_snapshot_path == prepared.bootstrap_path
    assert result.memory.bootstrap_snapshot_sha256 == prepared.bootstrap_sha256
    assert result.memory.bootstrap_snapshot_fingerprint
    assert result.memory.bootstrap_formal_binding_sha256
    assert result.memory.bootstrap_initial_file_sha256s
    assert FROM_SCRATCH_BOOTSTRAP_MISSING_CODE not in result.blockers
    assert "PLANNED_OUTPUT_DIRECTORY_NOT_EMPTY" not in result.blockers


def test_no_dynamic_closes_only_the_four_memory_gates(tmp_path: Path) -> None:
    plan_path, suite = _plan(tmp_path)
    result = compile_live_launch(
        plan_path,
        _run_id(suite, MethodVariant.NO_DYNAMIC_EXPERIENCE),
        results_root=tmp_path / "formal_results",
    )

    method = result.method
    assert method.hierarchical_evolution
    assert not method.strategy_retrieval
    assert not method.strategy_promotion
    assert not method.action_exemplar_retrieval
    assert not method.action_exemplar_promotion
    assert method.common_physics_verifier
    assert method.common_checkpoint_rollback
    assert method.common_terminal_predicate
    assert method.frozen_sonic_skills


def test_transfer_launch_requires_hash_bound_bank_files(tmp_path: Path) -> None:
    plan_path, suite = _plan(tmp_path)
    transfer = next(
        run
        for run in suite.new_runs
        if run.strategy_bank is not None and run.seed == 200
    )
    result = compile_live_launch(
        plan_path,
        transfer.run_id,
        results_root=tmp_path / "formal_results",
    )

    assert result.memory.strategy_bank_expected_sha256 == transfer.strategy_bank.sha256
    assert "TRANSFER_STRATEGY_BANK_FILES_NOT_BOUND" in result.blockers


def test_tampered_plan_never_becomes_a_launch_config(tmp_path: Path) -> None:
    plan_path, suite = _plan(tmp_path)
    document = __import__("json").loads(plan_path.read_text(encoding="utf-8"))
    document["suite"]["run_plan"]["new_runs"][0]["physical_budget"] = 79
    write_json(plan_path, document)

    with pytest.raises(LiveLaunchConfigError, match="does not exactly match"):
        compile_live_launch(
            plan_path,
            _run_id(suite, MethodVariant.REMAC_ADAPTED),
            results_root=tmp_path / "formal_results",
        )


def test_launchable_mode_rejects_a_planning_only_plan(tmp_path: Path) -> None:
    plan_path, suite = _plan(tmp_path)
    with pytest.raises(LiveLaunchConfigError, match="execution_preflight"):
        compile_live_launch(
            plan_path,
            _run_id(suite, MethodVariant.REMAC_ADAPTED),
            results_root=tmp_path / "formal_results",
            require_launchable=True,
        )


def test_compile_launch_cli_writes_a_non_evidence_readiness_record(
    tmp_path: Path,
) -> None:
    plan_path, suite = _plan(tmp_path)
    output = tmp_path / "launch.json"
    assert (
        main(
            [
                "compile-launch",
                "--plan",
                str(plan_path),
                "--run-id",
                _run_id(suite, MethodVariant.FLAT_REFLECTIVE_REPAIR),
                "--results-root",
                str(tmp_path / "formal_results"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert payload["kind"] == "coevolve_compiled_live_launch"
    assert payload["launch"]["paper_evidence_eligible"] is False
    assert payload["launch"]["method"]["policy_representation"] == (
        "praxis_code_as_policy"
    )
