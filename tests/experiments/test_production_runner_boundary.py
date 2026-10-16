from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from coevolve_experiments.contracts import (
    MemoryCondition,
    MemoryPolicy,
    MethodVariant,
    RunSpec,
    SceneId,
    TableId,
)
from coevolve_experiments.live.bridge import VerifiedSubprocessLaunchConfig
from coevolve_experiments.live.capability import (
    RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
)
from coevolve_experiments.live.contracts import (
    ExecutionCapability,
    PolicyRepresentation,
)
from coevolve_experiments.live.launch import (
    LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
    CompiledLiveLaunch,
    MemoryExecutionBinding,
    MethodExecutionContract,
    SceneExecutionBinding,
)
from coevolve_experiments.memory import (
    prepare_from_scratch_memory,
    validate_from_scratch_bootstrap,
)
from coevolve_experiments.production import (
    PRODUCTION_RUNTIME_ADAPTER_BLOCKER,
    ProductionChannelBinding,
    ProductionInjectionBundle,
    ProductionPreflightError,
    ProductionPreflightReport,
    REQUIRED_PRODUCTION_CHANNELS,
    V83_POST_TABLE_MIGRATION_REQUIREMENTS,
    audit_frozen_v83_checkout,
    compile_production_injection_bundle,
)
from coevolve_experiments.production import preflight as preflight_module
from coevolve_experiments.production import runner as runner_module
from coevolve_experiments.production.v83_audit import FrozenV83AuditError
from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE
from coevolve_experiments.scenes import load_scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _verified(tmp_path: Path) -> VerifiedSubprocessLaunchConfig:
    scene = load_scene(SceneId.BASE)
    run = RunSpec(
        run_id="table2_method_comparison__coevolve_full__seed_200",
        table_id=TableId.TABLE_2,
        seed=200,
        method_variant=MethodVariant.COEVOLE_FULL,
        scene_id=SceneId.BASE,
        memory_condition=MemoryCondition.EMPTY,
        memory_policy=MemoryPolicy.ONLINE_FROM_EMPTY,
        target_backbone="gpt-5.4-mini-2026-03-17",
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        physical_budget=80,
        condition_fingerprint=_digest("condition"),
        invariant_fingerprint=_digest("invariant"),
        base_method_commit="a" * 40,
        protocol_id="post-table1-test",
        output_relpath="table2_method_comparison/coevolve_full/seed_200",
        scene_fingerprint=scene.scene_hash,
        terminal_fingerprint=_digest("terminal"),
        observation_contract_fingerprint=_digest("observation"),
        task_instruction_fingerprint=_digest("instruction"),
        shared_prompt_contract_fingerprint=_digest("shared-prompt"),
        decision_prompt_fingerprint=_digest("decision-prompt"),
        skill_api_fingerprint=_digest("skill-api"),
        full_method_compatibility_fingerprint=_digest("full-method"),
    )
    method = MethodExecutionContract(
        method_variant=run.method_variant,
        decision_structure="hierarchical_plan_strategy_code_with_verifier_feedback",
        policy_representation=PolicyRepresentation.PRAXIS_CODE_AS_POLICY,
        hierarchical_evolution=True,
        strategy_retrieval=True,
        strategy_promotion=True,
        action_exemplar_retrieval=True,
        action_exemplar_promotion=True,
    )
    scene_binding = SceneExecutionBinding(
        scene_id=scene.scene_id.value,
        scene_fingerprint=scene.scene_hash,
        registry_source_path=scene.asset.source_path,
        registry_source_sha256=scene.asset.source_sha256,
        execution_path=scene.asset.execution_path,
        execution_sha256=scene.asset.execution_sha256,
        execution_composite_sha256=scene.asset.execution_composite_sha256,
        execution_evidence_fingerprint=_digest("execution-evidence"),
        compiled_model_sha256=_digest("compiled-model"),
        target_center_xyz_m=scene.target.center.as_tuple(),
        route_agent_visible=scene.route.agent_visible,
    )
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    results_root = (tmp_path / "formal").resolve()
    output_dir = (results_root / run.output_relpath).resolve()
    plan_sha256 = sha256(plan.read_bytes()).hexdigest()
    prepare_from_scratch_memory(
        run,
        output_dir / "memory",
        plan_path=plan.resolve(),
        plan_sha256=plan_sha256,
    )
    snapshot = validate_from_scratch_bootstrap(
        output_dir / "memory",
        run,
        require_initial_state=True,
        expected_plan_path=plan.resolve(),
        expected_plan_sha256=plan_sha256,
    )
    memory = MemoryExecutionBinding(
        memory_condition=run.memory_condition,
        memory_policy=run.memory_policy.value,
        strategy_bank_source_backbone=None,
        strategy_bank_source_protocol_id=None,
        strategy_bank_source_method_commit=None,
        strategy_bank_source_seeds=None,
        strategy_bank_source_condition_fingerprint=None,
        strategy_bank_source_result_manifest_hash=None,
        strategy_bank_expected_sha256=None,
        strategy_bank_path=None,
        strategy_bank_actual_sha256=None,
        strategy_bank_provenance_path=None,
        strategy_bank_provenance_sha256=None,
        bootstrap_snapshot_path=snapshot.path,
        bootstrap_snapshot_sha256=snapshot.file_sha256,
        bootstrap_snapshot_fingerprint=snapshot.snapshot_fingerprint,
        bootstrap_formal_binding_sha256=snapshot.formal_binding_sha256,
        bootstrap_initial_file_sha256s=snapshot.initial_file_sha256s,
        target_online_overlay_starts_empty=True,
        target_action_exemplars_start_empty=True,
        target_manifest_starts_empty=True,
        target_failure_cache_starts_empty=True,
    )
    launch = CompiledLiveLaunch(
        schema_version=LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
        run_spec=run,
        plan_path=plan.resolve(),
        plan_sha256=plan_sha256,
        results_root=results_root,
        output_dir=output_dir,
        method=method,
        scene=scene_binding,
        memory=memory,
        execution_capability=ExecutionCapability.PAPER_FORMAL,
        paper_evidence_eligible=True,
        blockers=(),
        launch_fingerprint=_digest("launch"),
    )
    capability = SimpleNamespace(
        runner_repo=REPO_ROOT,
        result_contract_schema_version=RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
        manifest_sha256=_digest("capability"),
        base_method_commit=run.base_method_commit,
        runner_integration_commit="b" * 40,
        entrypoint_sha256=_digest("entrypoint"),
        full_method_compatibility=SimpleNamespace(
            fingerprint=run.full_method_compatibility_fingerprint,
        ),
    )
    config = tmp_path / "launch.json"
    config.write_text("{}\n", encoding="utf-8")
    return VerifiedSubprocessLaunchConfig(
        path=config.resolve(),
        sha256=sha256(config.read_bytes()).hexdigest(),
        config_fingerprint=_digest("config"),
        launch=launch,
        capability=capability,  # type: ignore[arg-type]
    )


def test_bundle_derives_every_runtime_injection_without_output_mutation(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    bundle = compile_production_injection_bundle(
        verified,
        scene_registry_root=REPO_ROOT / "src/coevolve_experiments/scenes",
    )

    assert tuple(item.name for item in bundle.channels) == REQUIRED_PRODUCTION_CHANNELS
    assert bundle.channel("term1_simulator_asset").payload["execution_path"] == str(
        (REPO_ROOT / "assets/manual_scenes/two_g1_box_push_layout.usda").resolve()
    )
    contact_roles = bundle.channel("term1_simulator_asset").payload[
        "contact_role_registry"
    ]
    assert contact_roles["source_execution_sha256"] == load_scene(
        SceneId.BASE
    ).asset.execution_sha256
    assert contact_roles["wall_prim_names"] == [
        "wall_01",
        "wall_03",
        "wall_04",
        "wall_05",
        "wall_06",
        "wall_07",
        "wall_08",
    ]
    assert bundle.channel("provider_route").payload["credential_env"] == (
        "PRAXIS_REMOTE_API_KEY"
    )
    assert bundle.channel("method_gates").payload["method_variant"] == "coevolve_full"
    assert bundle.channel("formal_output").payload["physical_budget"] == 80
    assert bundle.channel("formal_output").payload[
        "required_result_artifact_pins"
    ]["native_memory_audit"] == "v83_native_memory_runtime_audit_sha256"
    assert bundle.channel("formal_output").payload[
        "required_result_artifact_pins"
    ]["transaction_accounting_ledger"] == (
        "transaction_accounting_ledger_sha256"
    )
    assert bundle.channel("formal_output").payload[
        "required_result_artifact_pins"
    ]["accepted_episode_manifest"] == "accepted_episode_manifest_sha256"
    assert bundle.channel("formal_output").payload[
        "required_result_artifact_pins"
    ]["frozen_phase_plan"] == "frozen_phase_plan_sha256"
    result_binding = bundle.channel("formal_run_manifest").payload["result_binding"]
    assert result_binding["schema_version"] == RUNNER_RESULT_CONTRACT_SCHEMA_VERSION
    assert result_binding["base_method_commit"] == "a" * 40
    assert result_binding["runner_integration_commit"] == "b" * 40
    assert "runner_git_commit" not in result_binding
    runtime_contract = bundle.channel("memory_gates").payload["runtime_contract"]
    assert runtime_contract["cross_run_retrieval"] is False
    assert all(runtime_contract["operation_gates"].values())
    assert runtime_contract["runtime_audit"]["path"] == str(
        verified.launch.output_dir
        / "memory"
        / "v83_native"
        / "v83_native_memory_runtime_audit.json"
    )
    assert runtime_contract["runtime_audit"]["path_source"] == (
        "output_dir/memory/v83_native/fixed_filename"
    )
    assert runtime_contract["bootstrap"]["is_production_memory_evidence"] is False
    assert runtime_contract["native_facade"] == {
        "adapter": "V83NativeMemoryBridge",
        "mode": "from_empty",
        "root": str(verified.launch.output_dir / "memory" / "v83_native"),
        "strategy_store_path": str(
            verified.launch.output_dir / "memory" / "v83_native" / "strategy_specs.json"
        ),
        "action_exemplar_store_path": str(
            verified.launch.output_dir / "memory" / "v83_native" / "action_exemplars.json"
        ),
        "raw_rollout_path": str(
            verified.launch.output_dir / "memory" / "v83_native" / "raw_rollouts.jsonl"
        ),
        "accepted_episode_manifest_path": str(
            verified.launch.output_dir
            / "memory"
            / "v83_native"
            / "accepted_episode_manifest.json"
        ),
        "transfer_seed_path": None,
        "strategy_retrieval_limit": 3,
        "action_exemplar_retrieval_limit": 2,
        "run_binding": {
            "run_id": verified.launch.run_spec.run_id,
            "table_id": "table2_method_comparison",
            "seed": 200,
            "protocol_id": "post-table1-test",
            "base_method_commit": "a" * 40,
            "condition_fingerprint": _digest("condition"),
            "invariant_fingerprint": _digest("invariant"),
            "scene_fingerprint": verified.launch.run_spec.scene_fingerprint,
            "terminal_fingerprint": _digest("terminal"),
            "memory_condition": "empty",
            "memory_policy": "online_from_empty",
            "target_backbone": "gpt-5.4-mini-2026-03-17",
            "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
            "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            "strategy_bank_sha256": "",
            "plan_path": str(verified.launch.plan_path),
            "plan_sha256": verified.launch.plan_sha256,
            "observation_contract_fingerprint": _digest("observation"),
            "task_instruction_fingerprint": _digest("instruction"),
            "shared_prompt_contract_fingerprint": _digest("shared-prompt"),
            "decision_prompt_fingerprint": _digest("decision-prompt"),
            "skill_api_fingerprint": _digest("skill-api"),
            "full_method_compatibility_fingerprint": _digest("full-method"),
        },
        "run_binding_sha256": runtime_contract["native_facade"][
            "run_binding_sha256"
        ],
    }
    assert len(runtime_contract["native_facade"]["run_binding_sha256"]) == 64
    assert runtime_contract["initial_state"]["initial_checkpoint"] == "accepted_step_00"
    assert all(runtime_contract["ungated_v83_lifecycle"].values())
    assert verified.launch.output_dir.is_dir()
    assert not (verified.launch.output_dir / "run_manifest.json").exists()


@pytest.mark.parametrize(
    ("memory_condition", "memory_policy", "expected_mode", "has_seed"),
    [
        (MemoryCondition.NO_DYNAMIC, MemoryPolicy.DISABLED, "disabled", False),
        (
            MemoryCondition.GPT_TO_GPT,
            MemoryPolicy.ONLINE_FROM_PRELOADED_STRATEGIES,
            "preloaded_seed",
            True,
        ),
    ],
)
def test_native_memory_contract_modes_are_closed_world(
    tmp_path: Path,
    memory_condition: MemoryCondition,
    memory_policy: MemoryPolicy,
    expected_mode: str,
    has_seed: bool,
) -> None:
    output_dir = (tmp_path / "formal/run").resolve()
    launch = SimpleNamespace(
        output_dir=output_dir,
        plan_path=(tmp_path / "plan.json").resolve(),
        plan_sha256=_digest("plan"),
        memory=SimpleNamespace(
            bootstrap_snapshot_path=(tmp_path / "bootstrap.json").resolve(),
            bootstrap_snapshot_sha256=_digest("bootstrap"),
            bootstrap_snapshot_fingerprint=_digest("bootstrap-fingerprint"),
        ),
    )
    run = SimpleNamespace(
        run_id="native-mode-test",
        table_id=TableId.TABLE_2,
        seed=200,
        protocol_id="post-table1-test",
        base_method_commit="a" * 40,
        condition_fingerprint=_digest("condition"),
        invariant_fingerprint=_digest("invariant"),
        scene_fingerprint=_digest("scene"),
        terminal_fingerprint=_digest("terminal"),
        memory_condition=memory_condition,
        memory_policy=memory_policy,
        target_backbone="gpt-5.4-mini-2026-03-17",
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        strategy_bank=(
            SimpleNamespace(sha256=_digest("bank")) if has_seed else None
        ),
        observation_contract_fingerprint=_digest("observation"),
        task_instruction_fingerprint=_digest("instruction"),
        shared_prompt_contract_fingerprint=_digest("shared-prompt"),
        decision_prompt_fingerprint=_digest("decision-prompt"),
        skill_api_fingerprint=_digest("skill-api"),
        full_method_compatibility_fingerprint=_digest("full-method"),
    )

    contract = preflight_module._memory_runtime_contract(run, launch)

    assert contract["native_facade"]["mode"] == expected_mode
    assert contract["cross_run_retrieval"] is has_seed
    assert (
        contract["native_facade"]["transfer_seed_path"] is not None
    ) is has_seed
    if memory_policy is MemoryPolicy.DISABLED:
        assert not any(contract["operation_gates"].values())
        assert contract["final_state"]["strategy_store_must_be_empty"] is True
        assert contract["final_state"]["action_exemplar_store_must_be_empty"] is True
        assert contract["final_state"]["operation_counts_must_be_zero"] is True
        assert contract["final_state"]["memory_writer_calls_must_be_zero"] is True
    else:
        assert all(contract["operation_gates"].values())


def test_bundle_rejects_scene_and_method_binding_drift(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    bad_scene = replace(
        verified.launch.scene,
        execution_sha256=_digest("tampered-execution"),
    )
    with pytest.raises(ProductionPreflightError, match="execution_sha256"):
        compile_production_injection_bundle(
            replace(verified, launch=replace(verified.launch, scene=bad_scene)),
            scene_registry_root=REPO_ROOT / "src/coevolve_experiments/scenes",
        )
    bad_method = replace(
        verified.launch.method,
        method_variant=MethodVariant.REMAC_ADAPTED,
    )
    with pytest.raises(ProductionPreflightError, match="method execution contract"):
        compile_production_injection_bundle(
            replace(verified, launch=replace(verified.launch, method=bad_method)),
            scene_registry_root=REPO_ROOT / "src/coevolve_experiments/scenes",
        )


def test_v83_audit_is_read_only_and_fails_on_dirty_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "v83"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "test"], check=True,
    )
    paths = ("scripts/runner.py", "src/formal_eval.py")
    for relpath in paths:
        path = repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relpath}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "tag", "v83-test"], check=True)
    hashes = {
        relpath: sha256((repo / relpath).read_bytes()).hexdigest()
        for relpath in paths
    }
    audit = audit_frozen_v83_checkout(
        repo.resolve(),
        expected_commit=commit,
        expected_tag="v83-test",
        expected_file_hashes=hashes,
    )
    assert audit.commit == commit
    assert not (repo / "audit.json").exists()

    (repo / paths[0]).write_text("dirty\n", encoding="utf-8")
    with pytest.raises(FrozenV83AuditError, match="not clean"):
        audit_frozen_v83_checkout(
            repo.resolve(),
            expected_commit=commit,
            expected_tag="v83-test",
            expected_file_hashes=hashes,
        )


def _blocked_report_and_bundle(tmp_path: Path):
    channels = tuple(
        ProductionChannelBinding.create(name, {"fixture": name})
        for name in REQUIRED_PRODUCTION_CHANNELS
    )
    bundle = ProductionInjectionBundle.create(
        run_id="run",
        launch_config_path=str((tmp_path / "launch.json").resolve()),
        launch_config_sha256=_digest("config"),
        launch_fingerprint=_digest("launch"),
        channels=channels,
    )
    report = ProductionPreflightReport(
        available=False,
        run_id="run",
        launch_config_path=bundle.launch_config_path,
        launch_config_sha256=bundle.launch_config_sha256,
        injection_bundle_fingerprint=bundle.fingerprint,
        blocker_codes=(PRODUCTION_RUNTIME_ADAPTER_BLOCKER,),
        requirements=V83_POST_TABLE_MIGRATION_REQUIREMENTS,
    )
    return report, bundle


def test_runner_has_one_argument_and_never_claims_smoke_as_formal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    launch = (tmp_path / "launch.json").resolve()
    launch.write_text("{}\n", encoding="utf-8")
    report, bundle = _blocked_report_and_bundle(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "preflight_production_launch",
        lambda path: (report, bundle),
    )
    assert runner_module.main(["--launch-config", str(launch)]) == (
        runner_module.EXIT_RUNTIME_UNAVAILABLE
    )
    emitted = json.loads(capsys.readouterr().err)
    assert emitted["report"]["available"] is False
    assert emitted["injection_channels"] == list(REQUIRED_PRODUCTION_CHANNELS)
    with pytest.raises(SystemExit):
        runner_module.parse_args(
            ["--launch-config", str(launch), "--method", "coevolve_full"]
        )
    with pytest.raises(SystemExit):
        runner_module.parse_args(["--launch-config", "relative.json"])


def test_executable_runner_bootstraps_package_and_fails_closed(
    tmp_path: Path,
) -> None:
    """The live bridge invokes the entrypoint as a file, not ``python -m``."""

    missing = (tmp_path / "missing-launch.json").resolve()
    runner = REPO_ROOT / "src/coevolve_experiments/production/runner.py"
    completed = subprocess.run(
        [str(runner), "--launch-config", str(missing)],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )

    assert completed.returncode == runner_module.EXIT_CONFIG_INVALID
    emitted = json.loads(completed.stderr)
    assert emitted["kind"] == "coevolve_production_preflight_error"
    assert emitted["launch_config_path"] == str(missing)


def test_preflight_exposes_every_migration_blocker_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified(tmp_path)
    report_fixture, bundle = _blocked_report_and_bundle(tmp_path)
    del report_fixture
    monkeypatch.setattr(
        preflight_module,
        "load_subprocess_launch_config",
        lambda path: verified,
    )
    monkeypatch.setattr(
        preflight_module,
        "compile_production_injection_bundle",
        lambda value: bundle,
    )
    report, returned = preflight_module.preflight_production_launch(
        verified.path
    )
    assert returned is bundle
    assert report.available is False
    assert report.blocker_codes[0] == PRODUCTION_RUNTIME_ADAPTER_BLOCKER
    assert report.blocker_codes[1:] == tuple(
        item.code for item in V83_POST_TABLE_MIGRATION_REQUIREMENTS
    )
