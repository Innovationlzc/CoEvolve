from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from coevolve_experiments.contracts import (
    MemoryCondition,
    MemoryPolicy,
    MethodVariant,
    RunSpec,
    SceneId,
    TableId,
)
from coevolve_experiments.memory import (
    FromScratchMemoryError,
    bind_from_scratch_adapter,
    prepare_from_scratch_memory,
    scan_memory_leaks,
    validate_from_scratch_bootstrap,
)
from coevolve_experiments.methods import ScriptedRoleBackend
from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE
from coevolve_experiments.reporting import validate_from_scratch_memory_audit
from coevolve_experiments.runtime import (
    ExperimentEngine,
    Observation,
    ScriptedRuntime,
    VerifierOutcome,
    make_team_proposal,
)


def _run(
    *,
    method: MethodVariant = MethodVariant.COEVOLE_FULL,
    condition: MemoryCondition = MemoryCondition.EMPTY,
    policy: MemoryPolicy = MemoryPolicy.ONLINE_FROM_EMPTY,
) -> RunSpec:
    return RunSpec(
        run_id=f"table3__{method.value}__seed_200",
        table_id=TableId.TABLE_3,
        seed=200,
        method_variant=method,
        scene_id=SceneId.MIRRORED,
        memory_condition=condition,
        memory_policy=policy,
        target_backbone="gpt-5.4-mini-2026-03-17",
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        physical_budget=80,
        condition_fingerprint="1" * 64,
        invariant_fingerprint="2" * 64,
        base_method_commit="a" * 40,
        protocol_id="post-table1-v1",
        output_relpath=f"table3/{method.value}/seed_200",
        scene_fingerprint="3" * 64,
        terminal_fingerprint="4" * 64,
        observation_contract_fingerprint="5" * 64,
        task_instruction_fingerprint="6" * 64,
        shared_prompt_contract_fingerprint="7" * 64,
        decision_prompt_fingerprint="8" * 64,
        skill_api_fingerprint="9" * 64,
        full_method_compatibility_fingerprint="b" * 64,
    )


def _plan(tmp_path: Path) -> tuple[Path, str]:
    path = (tmp_path / "suite_plan.json").resolve()
    path.write_text('{"kind":"test-suite"}\n', encoding="utf-8")
    return path, sha256(path.read_bytes()).hexdigest()


def _strategy() -> dict[str, object]:
    return {
        "updated_plan": {"phase": "carry"},
        "committed_strategy": {
            "primitive": "cooperative_move",
            "semantic_objective": "advance",
            "coordination_mode": "synchronized",
        },
    }


def test_online_from_empty_lifecycle_is_bound_promoted_and_finalized(
    tmp_path: Path,
) -> None:
    run = _run()
    plan, plan_sha = _plan(tmp_path)
    prepared = prepare_from_scratch_memory(
        run,
        tmp_path / "run" / "memory",
        plan_path=plan,
        plan_sha256=plan_sha,
    )
    backend = ScriptedRoleBackend(
        [
            ("task_orchestrator", _strategy()),
            ("policy_generator", make_team_proposal("advance", lifecycle_phase="carry")),
            (
                "memory_update",
                {
                    "coordination_pattern": "turn left 0.30 m at checkpoint_7",
                    "repair_direction": "continue toward the red cylinder",
                },
            ),
        ]
    )
    binding = bind_from_scratch_adapter(run, backend, prepared.memory_dir)
    runtime = ScriptedRuntime(
        [
            VerifierOutcome(
                True,
                terminal=True,
                report="done",
                evidence={"route_progress": True},
            )
        ],
        initial_observation=Observation(0, {"phase": "carry"}),
    )

    result = ExperimentEngine(binding.adapter, runtime.callbacks()).run("carry")

    assert result.success
    overlay = json.loads(prepared.online_overlay_path.read_text(encoding="utf-8"))
    actions = json.loads(prepared.action_exemplars_path.read_text(encoding="utf-8"))
    audit = json.loads(binding.memory.runtime_audit_path.read_text(encoding="utf-8"))
    assert overlay["schema_version"] == 2
    assert len(overlay["entries"]) == len(actions["exemplars"]) == 1
    assert not scan_memory_leaks(overlay["entries"][0]["proposal"])
    assert audit["lifecycle_state"] == "finalized"
    assert audit["initial_state_verified"] is True
    assert audit["cross_run_retrieval"] is False
    assert audit["operations"] == {
        "strategy_retrievals": 1,
        "exemplar_retrievals": 1,
        "strategy_promotions": 1,
        "exemplar_promotions": 1,
        "promotion_noops": 0,
    }
    assert audit["online_overlay"]["initial_count"] == 0
    assert audit["online_overlay"]["final_count"] == 1
    assert audit["run_binding"]["plan_sha256"] == plan_sha
    validation = validate_from_scratch_memory_audit(
        prepared.memory_dir.parent,
        expected_run_binding=audit["run_binding"],
        expected_memory_condition="empty",
        expected_memory_policy="online_from_empty",
    )
    assert validation.invalid_reasons == ()
    assert all(validation.artifact_hashes.values())


def test_no_dynamic_lifecycle_proves_all_four_gates_closed(tmp_path: Path) -> None:
    base = _run()
    run = replace(
        base,
        run_id="table2__no_dynamic__seed_200",
        table_id=TableId.TABLE_2,
        method_variant=MethodVariant.NO_DYNAMIC_EXPERIENCE,
        scene_id=SceneId.BASE,
        memory_condition=MemoryCondition.NO_DYNAMIC,
        memory_policy=MemoryPolicy.DISABLED,
        output_relpath="table2/no_dynamic/seed_200",
    )
    plan, plan_sha = _plan(tmp_path)
    prepared = prepare_from_scratch_memory(
        run,
        tmp_path / "run" / "memory",
        plan_path=plan,
        plan_sha256=plan_sha,
    )
    backend = ScriptedRoleBackend(
        [
            ("task_orchestrator", _strategy()),
            ("policy_generator", make_team_proposal("advance")),
        ]
    )
    binding = bind_from_scratch_adapter(run, backend, prepared.memory_dir)
    result = ExperimentEngine(
        binding.adapter,
        ScriptedRuntime([VerifierOutcome(True, terminal=True)]).callbacks(),
    ).run("carry")

    assert result.success
    audit = json.loads(binding.memory.runtime_audit_path.read_text(encoding="utf-8"))
    assert audit["operation_gates"] == {
        "strategy_retrieval": False,
        "strategy_promotion": False,
        "action_exemplar_retrieval": False,
        "action_exemplar_promotion": False,
    }
    assert not any(audit["operations"].values())
    assert audit["online_overlay"]["final_count"] == 0
    assert audit["action_exemplars"]["final_count"] == 0
    assert "memory_update" not in backend.roles
    validation = validate_from_scratch_memory_audit(
        prepared.memory_dir.parent,
        expected_run_binding=audit["run_binding"],
        expected_memory_condition="no_dynamic",
        expected_memory_policy="disabled",
    )
    assert validation.invalid_reasons == ()


def test_disabled_port_fails_closed_if_a_method_crosses_a_memory_gate(
    tmp_path: Path,
) -> None:
    base = _run()
    run = replace(
        base,
        run_id="table2__remac__seed_200",
        table_id=TableId.TABLE_2,
        method_variant=MethodVariant.REMAC_ADAPTED,
        scene_id=SceneId.BASE,
        memory_policy=MemoryPolicy.DISABLED,
        output_relpath="table2/remac/seed_200",
    )
    plan, plan_sha = _plan(tmp_path)
    prepared = prepare_from_scratch_memory(
        run,
        tmp_path / "run" / "memory",
        plan_path=plan,
        plan_sha256=plan_sha,
    )
    binding = bind_from_scratch_adapter(run, ScriptedRoleBackend([]), prepared.memory_dir)

    with pytest.raises(FromScratchMemoryError, match="gate is disabled"):
        binding.memory.retrieve_strategies(None)  # type: ignore[arg-type]

    audit = json.loads(binding.memory.runtime_audit_path.read_text(encoding="utf-8"))
    assert audit["lifecycle_state"] == "violated"
    assert audit["violations"][0]["operation"] == "retrieve_strategies"


def test_from_scratch_bootstrap_is_non_overwriting_and_rejects_table4(
    tmp_path: Path,
) -> None:
    run = _run()
    plan, plan_sha = _plan(tmp_path)
    memory_dir = tmp_path / "run" / "memory"
    prepare_from_scratch_memory(
        run,
        memory_dir,
        plan_path=plan,
        plan_sha256=plan_sha,
    )
    snapshot = validate_from_scratch_bootstrap(
        memory_dir,
        run,
        require_initial_state=True,
        expected_plan_path=plan,
        expected_plan_sha256=plan_sha,
    )
    assert snapshot.path == (memory_dir / "from_scratch_memory_bootstrap.json").resolve()
    assert snapshot.file_sha256
    assert snapshot.snapshot_fingerprint
    assert snapshot.formal_binding_sha256
    assert tuple(name for name, _digest in snapshot.initial_file_sha256s) == (
        "online_strategy_overlay.json",
        "action_exemplars.json",
        "accepted_episode_manifest.json",
        "failure_cache.json",
    )

    with pytest.raises(FromScratchMemoryError, match="already exists"):
        prepare_from_scratch_memory(
            run,
            memory_dir,
            plan_path=plan,
            plan_sha256=plan_sha,
        )
    with pytest.raises(FromScratchMemoryError, match="restricted to Table 2/3"):
        prepare_from_scratch_memory(
            replace(run, table_id=TableId.TABLE_4),
            tmp_path / "other" / "memory",
            plan_path=plan,
            plan_sha256=plan_sha,
        )


def test_from_scratch_collector_audit_rejects_relabel_and_store_tampering(
    tmp_path: Path,
) -> None:
    base = _run()
    run = replace(
        base,
        run_id="table2__no_dynamic__seed_200",
        table_id=TableId.TABLE_2,
        method_variant=MethodVariant.NO_DYNAMIC_EXPERIENCE,
        scene_id=SceneId.BASE,
        memory_condition=MemoryCondition.NO_DYNAMIC,
        memory_policy=MemoryPolicy.DISABLED,
        output_relpath="table2/no_dynamic/seed_200",
    )
    plan, plan_sha = _plan(tmp_path)
    prepared = prepare_from_scratch_memory(
        run,
        tmp_path / "run" / "memory",
        plan_path=plan,
        plan_sha256=plan_sha,
    )
    binding = bind_from_scratch_adapter(run, ScriptedRoleBackend([]), prepared.memory_dir)
    audit = binding.memory.finalize()

    wrong_binding = dict(audit["run_binding"])
    wrong_binding["seed"] = 204
    relabelled = validate_from_scratch_memory_audit(
        prepared.memory_dir.parent,
        expected_run_binding=wrong_binding,
        expected_memory_condition="no_dynamic",
        expected_memory_policy="disabled",
    )
    assert "FROM_SCRATCH_BOOTSTRAP_RUN_BINDING_MISMATCH" in relabelled.invalid_reasons
    assert "FROM_SCRATCH_RUNTIME_RUN_BINDING_MISMATCH" in relabelled.invalid_reasons

    overlay = json.loads(prepared.online_overlay_path.read_text(encoding="utf-8"))
    overlay["entries"].append({"not": "a valid strategy record"})
    prepared.online_overlay_path.write_text(json.dumps(overlay), encoding="utf-8")
    tampered = validate_from_scratch_memory_audit(
        prepared.memory_dir.parent,
        expected_run_binding=audit["run_binding"],
        expected_memory_condition="no_dynamic",
        expected_memory_policy="disabled",
    )
    assert "FROM_SCRATCH_OVERLAY_INVALID" in tampered.invalid_reasons
    assert "FROM_SCRATCH_DISABLED_MEMORY_NOT_EMPTY" not in tampered.invalid_reasons
