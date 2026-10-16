from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
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
from coevolve_experiments.decision_prompts import (
    FROZEN_V83_FULL_DECISION_PROMPT_SHA256,
    load_production_prompt_registry,
)
from coevolve_experiments.live.bridge import VerifiedSubprocessLaunchConfig
from coevolve_experiments.live.capability import RUNNER_RESULT_CONTRACT_SCHEMA_VERSION
from coevolve_experiments.live.contracts import ExecutionCapability, PolicyRepresentation
from coevolve_experiments.live.launch import (
    LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
    CompiledLiveLaunch,
    MemoryExecutionBinding,
    MethodExecutionContract,
    SceneExecutionBinding,
)
from coevolve_experiments.memory import (
    V83NativeMemoryBridge,
    prepare_from_scratch_memory,
    validate_from_scratch_bootstrap,
)
from coevolve_experiments.production import compile_production_injection_bundle
from coevolve_experiments.production.contracts import (
    ProductionChannelBinding,
    ProductionInjectionBundle,
    REQUIRED_PRODUCTION_CHANNELS,
)
from coevolve_experiments.production.session import (
    ProductionSessionBindingError,
    bind_production_session,
)
from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE
from coevolve_experiments.scenes import load_scene
from coevolve_experiments.serialization import sha256_json, to_primitive


REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _bundle(
    tmp_path: Path,
    *,
    variant: MethodVariant = MethodVariant.NO_DYNAMIC_EXPERIENCE,
    scene_id: SceneId = SceneId.BASE,
    runner_integration_commit: str = "b" * 40,
    runner_capability_sha256: str | None = None,
    runner_capability_entrypoint_sha256: str | None = None,
) -> ProductionInjectionBundle:
    artifact = (
        None
        if variant is MethodVariant.COEVOLE_FULL
        else load_production_prompt_registry().artifact_for(variant)
    )
    enabled = variant in {
        MethodVariant.COEVOLE_FULL,
        MethodVariant.FLAT_REFLECTIVE_REPAIR,
    }
    scene = load_scene(scene_id)
    table_id = TableId.TABLE_2 if scene_id is SceneId.BASE else TableId.TABLE_3
    table_name = table_id.value
    run = RunSpec(
        run_id=f"{table_name}__{scene_id.value}__{variant.value}__seed_200",
        table_id=table_id,
        seed=200,
        method_variant=variant,
        scene_id=scene_id,
        memory_condition=(
            MemoryCondition.NO_DYNAMIC
            if variant is MethodVariant.NO_DYNAMIC_EXPERIENCE
            else MemoryCondition.EMPTY
        ),
        memory_policy=(
            MemoryPolicy.ONLINE_FROM_EMPTY if enabled else MemoryPolicy.DISABLED
        ),
        target_backbone="gpt-5.4-mini-2026-03-17",
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        physical_budget=80,
        condition_fingerprint=_digest("condition"),
        invariant_fingerprint=_digest("invariant"),
        base_method_commit="a" * 40,
        protocol_id="post-table1-test",
        output_relpath=f"{table_name}/{scene_id.value}_{variant.value}/seed_200",
        scene_fingerprint=scene.scene_hash,
        terminal_fingerprint=_digest("terminal"),
        observation_contract_fingerprint=_digest("observation"),
        task_instruction_fingerprint=_digest("instruction"),
        shared_prompt_contract_fingerprint=_digest("shared-prompt"),
        decision_prompt_fingerprint=(
            FROZEN_V83_FULL_DECISION_PROMPT_SHA256
            if artifact is None
            else artifact.decision_prompt_fingerprint
        ),
        skill_api_fingerprint=_digest("skill-api"),
        full_method_compatibility_fingerprint=_digest("full-method"),
    )
    method = MethodExecutionContract(
        method_variant=run.method_variant,
        decision_structure=(
            "hierarchical_evolution"
            if artifact is None
            else artifact.decision_structure
        ),
        policy_representation=PolicyRepresentation.PRAXIS_CODE_AS_POLICY,
        hierarchical_evolution=(
            True if artifact is None else artifact.hierarchical_evolution
        ),
        strategy_retrieval=enabled,
        strategy_promotion=enabled,
        action_exemplar_retrieval=enabled,
        action_exemplar_promotion=enabled,
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
    plan = (tmp_path / "plan.json").resolve()
    plan.write_text("{}\n", encoding="utf-8")
    plan_sha = sha256(plan.read_bytes()).hexdigest()
    results_root = (tmp_path / "formal").resolve()
    output_dir = results_root / run.output_relpath
    prepare_from_scratch_memory(
        run,
        output_dir / "memory",
        plan_path=plan,
        plan_sha256=plan_sha,
    )
    snapshot = validate_from_scratch_bootstrap(
        output_dir / "memory",
        run,
        require_initial_state=True,
        expected_plan_path=plan,
        expected_plan_sha256=plan_sha,
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
        plan_path=plan,
        plan_sha256=plan_sha,
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
        manifest_sha256=(
            _digest("capability")
            if runner_capability_sha256 is None
            else runner_capability_sha256
        ),
        base_method_commit=run.base_method_commit,
        runner_integration_commit=runner_integration_commit,
        entrypoint_sha256=(
            _digest("entrypoint")
            if runner_capability_entrypoint_sha256 is None
            else runner_capability_entrypoint_sha256
        ),
        full_method_compatibility=SimpleNamespace(
            fingerprint=run.full_method_compatibility_fingerprint
        ),
    )
    config = (tmp_path / "launch.json").resolve()
    config.write_text("{}\n", encoding="utf-8")
    verified = VerifiedSubprocessLaunchConfig(
        path=config,
        sha256=sha256(config.read_bytes()).hexdigest(),
        config_fingerprint=_digest("config"),
        launch=launch,
        capability=capability,  # type: ignore[arg-type]
    )
    return compile_production_injection_bundle(
        verified,
        scene_registry_root=REPO_ROOT / "src/coevolve_experiments/scenes",
    )


def _tree_bytes(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        str(path.relative_to(root)): (path.stat().st_mode, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def _replace_channels(
    bundle: ProductionInjectionBundle,
    replacements: dict[str, dict[str, object]],
) -> ProductionInjectionBundle:
    channels = tuple(
        ProductionChannelBinding.create(
            name,
            replacements.get(name, to_primitive(bundle.channel(name).payload)),
        )
        for name in REQUIRED_PRODUCTION_CHANNELS
    )
    return ProductionInjectionBundle.create(
        run_id=bundle.run_id,
        launch_config_path=bundle.launch_config_path,
        launch_config_sha256=bundle.launch_config_sha256,
        launch_fingerprint=bundle.launch_fingerprint,
        channels=channels,
    )


def test_session_consumes_and_binds_without_writing(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    before = _tree_bytes(tmp_path)

    session = bind_production_session(bundle, simulator_version="3.3.5")

    assert _tree_bytes(tmp_path) == before
    assert session.execution_ready is False
    assert session.run_spec.run_id == bundle.run_id
    assert session.prompt.artifact is not None
    assert sha256(session.prompt.registry_bytes).hexdigest() == session.prompt.registry_sha256
    assert sha256(session.prompt.manifest_bytes).hexdigest() == session.prompt.manifest_sha256
    assert session.prompt.audit_identity["source"] == "production_prompt_registry"
    assert session.memory.mode == "disabled"
    assert session.memory.gates.all_disabled
    assert session.runtime_trace.run_id == bundle.run_id
    assert session.evidence_bundle.fingerprint == bundle.fingerprint
    with pytest.raises(TypeError):
        session.channels.method_gates["method_variant"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        session.memory.open()  # type: ignore[call-arg]


@pytest.mark.parametrize("scene_id", tuple(SceneId))
def test_all_scene_channels_round_trip_to_one_registry_identity(
    tmp_path: Path,
    scene_id: SceneId,
) -> None:
    bundle = _bundle(tmp_path, scene_id=scene_id)

    session = bind_production_session(bundle, simulator_version="3.3.5")
    scene = load_scene(scene_id)

    assert session.run_spec.scene_id is scene_id
    assert session.run_spec.scene_fingerprint == scene.scene_hash
    assert to_primitive(session.channels.term1_simulator_asset["target"]) == (
        to_primitive(scene.target)
    )
    assert to_primitive(session.channels.host_clearance_geometry["corridor"]) == (
        to_primitive(scene.corridor)
    )
    assert to_primitive(session.channels.vlm_scene_prior["camera"]) == (
        to_primitive(scene.camera)
    )
    assert session.channels.route_and_progress_gates["scene_id"] == (
        scene.route.scene_id
    )


@pytest.mark.parametrize(
    ("channel", "field"),
    (
        ("term1_simulator_asset", "target"),
        ("host_clearance_geometry", "corridor"),
        ("vlm_scene_prior", "mission_topology"),
        ("route_and_progress_gates", "segments"),
    ),
)
def test_scene_consumer_rejects_cross_channel_geometry_drift(
    tmp_path: Path,
    channel: str,
    field: str,
) -> None:
    bundle = _bundle(tmp_path)
    changed = to_primitive(bundle.channel(channel).payload)
    changed[field] = [] if isinstance(changed[field], list) else {"tampered": True}
    tampered = _replace_channels(bundle, {channel: changed})

    with pytest.raises(
        ProductionSessionBindingError,
        match="(scene|geometry|prior|route/progress gates)",
    ):
        bind_production_session(tampered, simulator_version="3.3.5")


def test_memory_open_requires_native_factories_and_is_explicit(tmp_path: Path) -> None:
    session = bind_production_session(_bundle(tmp_path), simulator_version="3.3.5")

    bridge = session.memory.open(
        strategy_store_factory=lambda path: SimpleNamespace(specs=[]),
        action_store_factory=lambda path: SimpleNamespace(exemplars=[]),
    )

    assert isinstance(bridge, V83NativeMemoryBridge)
    assert session.memory.audit_path.is_file()
    bridge.mark_incomplete(reason="unit-test explicit open")


def test_full_method_uses_only_the_frozen_v83_prompt_pin(tmp_path: Path) -> None:
    session = bind_production_session(
        _bundle(tmp_path, variant=MethodVariant.COEVOLE_FULL),
        simulator_version="3.3.5",
    )

    assert session.prompt.artifact is None
    assert session.prompt.registry_bytes == b""
    assert session.prompt.decision_prompt_fingerprint == (
        FROZEN_V83_FULL_DECISION_PROMPT_SHA256
    )
    assert session.prompt.audit_identity["source"] == "frozen_v83_prompts_py"
    assert session.memory.mode == "from_empty"
    assert session.memory.gates == type(session.memory.gates).enabled()


def test_prompt_artifact_fingerprint_drift_fails_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bad_digest = _digest("different-decision-prompt")
    formal = to_primitive(bundle.channel("formal_run_manifest").payload)
    formal["run_spec"]["decision_prompt_fingerprint"] = bad_digest
    formal["result_binding"]["run_spec_sha256"] = sha256_json(formal["run_spec"])
    method = to_primitive(bundle.channel("method_gates").payload)
    method["decision_prompt_fingerprint"] = bad_digest
    memory = to_primitive(bundle.channel("memory_gates").payload)
    run_binding = memory["runtime_contract"]["native_facade"]["run_binding"]
    run_binding["decision_prompt_fingerprint"] = bad_digest
    memory["runtime_contract"]["native_facade"]["run_binding_sha256"] = sha256_json(
        run_binding
    )
    tampered = _replace_channels(
        bundle,
        {
            "formal_run_manifest": formal,
            "method_gates": method,
            "memory_gates": memory,
        },
    )

    with pytest.raises(
        ProductionSessionBindingError,
        match="artifact fingerprint differs from RunSpec",
    ):
        bind_production_session(tampered, simulator_version="3.3.5")
    assert not (
        Path(bundle.channel("formal_output").payload["output_dir"])
        / "memory/v83_native"
    ).exists()


def test_prompt_and_memory_gate_disagreement_fails_before_open(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    memory = to_primitive(bundle.channel("memory_gates").payload)
    memory["runtime_contract"]["operation_gates"]["strategy_retrieval"] = True
    tampered = _replace_channels(bundle, {"memory_gates": memory})

    with pytest.raises(
        ProductionSessionBindingError,
        match="method gates and native memory operation gates differ",
    ):
        bind_production_session(tampered, simulator_version="3.3.5")


def test_bootstrap_drift_between_bind_and_open_fails_without_audit(tmp_path: Path) -> None:
    session = bind_production_session(_bundle(tmp_path), simulator_version="3.3.5")
    mutable = session.memory.bootstrap_path.parent / "failure_cache.json"
    mutable.chmod(0o644)
    mutable.write_text('{"kind":"run_local_failure_cache","schema_version":1,"failures":[{}]}\n')

    with pytest.raises(ValueError, match="initial bytes changed"):
        session.memory.open(
            strategy_store_factory=lambda path: SimpleNamespace(specs=[]),
            action_store_factory=lambda path: SimpleNamespace(exemplars=[]),
        )
    assert not session.memory.audit_path.exists()
