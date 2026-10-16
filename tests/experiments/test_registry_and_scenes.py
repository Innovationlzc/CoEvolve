from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType

import pytest

import coevolve_experiments.scenes.evidence as scene_evidence

from coevolve_experiments.contracts import (
    MemoryCondition,
    MemoryPolicy,
    MethodDecisionPromptPin,
    MethodVariant,
    ProtocolLock,
    ReuseMode,
    SceneId,
    SuiteTier,
    Table1SourceCohort,
    TableId,
)
from coevolve_experiments.provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE,
    SECONDARY_GEMINI_PROVIDER_ROUTE,
)
from coevolve_experiments.full_method_compatibility import (
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
)
from coevolve_experiments.protocols import load_protocol_bundle
from coevolve_experiments.registry import build_run_plan, build_suite
from coevolve_experiments.scenes import (
    ALLOWED_EVIDENCE_PRODUCERS,
    FrozenControllerPin,
    compute_asset_composite_hash,
    compute_scene_hash,
    directed_route_document,
    load_scene,
    validate_scene,
)
from coevolve_experiments.scenes.integrity import _evidence_errors
from coevolve_experiments.scenes.schema import (
    CorridorSegment,
    DirectedRouteSegment,
    EvidencePin,
    ExecutionEvidence,
    Pose,
    ProgressGate,
    RealizationParameter,
    TargetGeometry,
    Vec2,
    Vec3,
)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture
def protocol_lock() -> ProtocolLock:
    shared_prompt = _digest("prompt-bundle")
    skill_api = _digest("skill-api")
    full_method = FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": _digest("initial-memory"),
            "prompt_sha256": shared_prompt,
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
    decision_prompts = tuple(
        MethodDecisionPromptPin(
            method_variant=variant,
            decision_prompt_fingerprint=(
                shared_prompt
                if variant is MethodVariant.COEVOLE_FULL
                else _digest(f"decision-prompt:{variant.value}")
            ),
        )
        for variant in MethodVariant
    )
    return ProtocolLock(
        protocol_id="aaai27-post-table1-v1",
        invariant_fingerprint=_digest("common-invariant"),
        terminal_fingerprint=_digest("terminal-predicate"),
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
            shared_prompt_contract_fingerprint=shared_prompt,
            decision_prompt_fingerprint=shared_prompt,
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
            shared_prompt_contract_fingerprint=shared_prompt,
            decision_prompt_fingerprint=shared_prompt,
            full_method_compatibility_fingerprint=full_method.fingerprint,
        ),
        gpt_strategy_bank_hash=_digest("gpt-strategy-bank"),
        gemini_strategy_bank_hash=_digest("gemini-strategy-bank"),
        target_provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        target_provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        observation_contract_fingerprint=_digest("observation-contract"),
        task_instruction_fingerprint=_digest("task-instruction"),
        shared_prompt_contract_fingerprint=shared_prompt,
        method_decision_prompt_pins=decision_prompts,
        skill_api_fingerprint=skill_api,
        full_method_compatibility=full_method,
    )


def test_protocol_contracts_are_frozen_and_complete(protocol_lock: ProtocolLock) -> None:
    assert len(tuple(TableId)) == 3
    assert len(tuple(MethodVariant)) == 5
    assert len(tuple(SceneId)) == 4
    assert {
        MemoryCondition.EMPTY,
        MemoryCondition.GPT_TO_GPT,
        MemoryCondition.GEMINI_TO_GPT,
    }.issubset(set(MemoryCondition))
    with pytest.raises(FrozenInstanceError):
        protocol_lock.protocol_id = "changed"  # type: ignore[misc]
    assert protocol_lock.protocol_id != protocol_lock.table1_gpt_source.protocol_id
    assert protocol_lock.base_method_commit != protocol_lock.table1_gpt_source.method_commit
    assert protocol_lock.table1_gpt_source.protocol_id != (
        protocol_lock.table1_gemini_source.protocol_id
    )


def test_source_cohort_accepts_real_table1_seeds_but_rejects_bad_cohorts() -> None:
    source = Table1SourceCohort(
        backbone="gpt",
        protocol_id="v83",
        method_commit="a" * 40,
        seeds=(200, 201, 202, 203, 204),
        condition_fingerprint="1" * 64,
        result_manifest_hash="2" * 64,
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        shared_prompt_contract_fingerprint="3" * 64,
        decision_prompt_fingerprint="3" * 64,
        full_method_compatibility_fingerprint="4" * 64,
    )
    assert source.seeds == (200, 201, 202, 203, 204)
    with pytest.raises(ValueError, match="exactly five unique"):
        replace(source, seeds=(200, 201, 202, 203))
    with pytest.raises(ValueError, match="must be unique"):
        replace(source, seeds=(200, 201, 202, 203, 203))


@pytest.mark.parametrize(("tier", "expected"), [(SuiteTier.CORE, 35), (SuiteTier.FULL, 45)])
def test_suite_has_only_new_unique_runs(
    protocol_lock: ProtocolLock, tier: SuiteTier, expected: int
) -> None:
    suite = build_suite(tier, protocol_lock)
    plan = build_run_plan(tier, protocol_lock)

    assert suite.unique_run_count == expected
    assert plan.unique_run_count == expected
    assert len({run.run_id for run in plan.new_runs}) == expected
    assert len({run.output_relpath for run in plan.new_runs}) == expected
    assert {run.seed for run in plan.new_runs} == {200, 201, 202, 203, 204}
    assert {run.physical_budget for run in plan.new_runs} == {80}
    assert {run.target_backbone for run in plan.new_runs} == {
        "gpt-5.4-mini-2026-03-17"
    }
    assert {run.invariant_fingerprint for run in plan.new_runs} == {
        protocol_lock.invariant_fingerprint
    }
    assert {run.base_method_commit for run in plan.new_runs} == {
        protocol_lock.base_method_commit
    }
    assert {run.protocol_id for run in plan.new_runs} == {protocol_lock.protocol_id}
    assert {run.observation_contract_fingerprint for run in plan.new_runs} == {
        protocol_lock.observation_contract_fingerprint
    }
    assert {run.task_instruction_fingerprint for run in plan.new_runs} == {
        protocol_lock.task_instruction_fingerprint
    }
    assert {run.provider_route_id for run in plan.new_runs} == {
        PRIMARY_GPT_PROVIDER_ROUTE.route_id
    }
    assert {run.provider_route_fingerprint for run in plan.new_runs} == {
        PRIMARY_GPT_PROVIDER_ROUTE.fingerprint
    }
    assert {run.shared_prompt_contract_fingerprint for run in plan.new_runs} == {
        protocol_lock.shared_prompt_contract_fingerprint
    }
    assert {
        (run.method_variant, run.decision_prompt_fingerprint)
        for run in plan.new_runs
    } == {
        (
            variant,
            protocol_lock.decision_prompt_fingerprint_for(variant),
        )
        for variant in {
            run.method_variant for run in plan.new_runs
        }
    }
    assert {run.skill_api_fingerprint for run in plan.new_runs} == {
        protocol_lock.skill_api_fingerprint
    }


def test_core_and_full_condition_accounting(protocol_lock: ProtocolLock) -> None:
    core = build_run_plan("core", protocol_lock)
    full = build_run_plan("full", protocol_lock)

    def count(table: TableId, plan: object) -> int:
        return sum(run.table_id is table for run in plan.new_runs)  # type: ignore[attr-defined]

    assert [count(table, core) for table in TableId] == [15, 10, 10]
    assert [count(table, full) for table in TableId] == [20, 15, 10]
    assert not any(run.method_variant is MethodVariant.COHERENT_ADAPTED for run in core.new_runs)
    assert not any(run.scene_id is SceneId.ALTERED_TOPOLOGY for run in core.new_runs)
    assert sum(run.method_variant is MethodVariant.COHERENT_ADAPTED for run in full.new_runs) == 5
    assert sum(run.scene_id is SceneId.ALTERED_TOPOLOGY for run in full.new_runs) == 5


def test_suite_compiler_rejects_count_preserving_paper_row_drift(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
) -> None:
    source = Path("src/coevolve_experiments/protocols")
    for name in ("common.yaml", "table2.yaml", "table3.yaml", "table4.yaml"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    table2 = json.loads((tmp_path / "table2.yaml").read_text(encoding="utf-8"))
    table2["conditions"][0]["label"] = "Count-preserving protocol drift"
    (tmp_path / "table2.yaml").write_text(json.dumps(table2), encoding="utf-8")

    with pytest.raises(ValueError, match="registered paper contract"):
        build_suite("core", protocol_lock, protocol_root=tmp_path)


def test_full_base_empty_are_hash_pinned_references(protocol_lock: ProtocolLock) -> None:
    plan = build_run_plan("full", protocol_lock)

    assert [reuse.target_label for reuse in plan.reuses] == ["Full CoEvolve", "Base", "Empty"]
    assert all(reuse.mode is ReuseMode.REFERENCE_ONLY for reuse in plan.reuses)
    assert all(not reuse.copies_results for reuse in plan.reuses)
    for reuse in plan.reuses:
        names = {item.name for item in reuse.expected_hashes}
        assert {
            "condition",
            "invariant",
            "scene",
            "terminal",
            "selector",
            "provider_route",
            "observation_contract",
            "task_instruction",
            "shared_prompt_contract",
            "decision_prompt",
            "skill_api",
            "full_method_compatibility",
        }.issubset(names)
        if reuse.source_selector.external:
            assert "result_manifest" in names

    full_row, base_row, empty_row = plan.reuses
    with pytest.raises(ValueError, match="terminal"):
        replace(
            empty_row,
            expected_hashes=tuple(
                item for item in empty_row.expected_hashes if item.name != "terminal"
            ),
        )
    with pytest.raises(ValueError, match="result_manifest"):
        replace(
            full_row,
            expected_hashes=tuple(
                item
                for item in full_row.expected_hashes
                if item.name != "result_manifest"
            ),
        )
    assert full_row.source_selector == base_row.source_selector
    assert full_row.source_selector.external
    assert (
        full_row.source_selector.protocol_id
        == protocol_lock.table1_gpt_source.protocol_id
    )
    assert (
        full_row.source_selector.method_commit
        == protocol_lock.table1_gpt_source.method_commit
    )
    assert full_row.source_selector.seeds == protocol_lock.table1_gpt_source.seeds
    assert (
        full_row.source_selector.full_method_compatibility_fingerprint
        == protocol_lock.full_method_compatibility.fingerprint
    )
    assert full_row.source_selector.method_commit != protocol_lock.base_method_commit
    assert {
        item.name: item.sha256 for item in base_row.expected_hashes
    }["full_method_compatibility"] == protocol_lock.full_method_compatibility.fingerprint
    assert not empty_row.source_selector.external
    assert empty_row.source_selector.source_table == TableId.TABLE_3.value
    assert empty_row.source_selector.protocol_id == protocol_lock.protocol_id
    assert empty_row.source_selector.method_commit == protocol_lock.base_method_commit
    assert empty_row.source_selector.seeds == (200, 201, 202, 203, 204)


def test_five_seeds_share_a_condition_fingerprint(protocol_lock: ProtocolLock) -> None:
    plan = build_run_plan("full", protocol_lock)
    grouped: dict[tuple[TableId, MethodVariant, SceneId, MemoryCondition], set[str]] = {}
    for run in plan.new_runs:
        key = (run.table_id, run.method_variant, run.scene_id, run.memory_condition)
        grouped.setdefault(key, set()).add(run.condition_fingerprint)
    assert all(len(fingerprints) == 1 for fingerprints in grouped.values())


def test_live_contract_pins_are_part_of_condition_identity(
    protocol_lock: ProtocolLock,
) -> None:
    original = build_run_plan("core", protocol_lock).new_runs[0]
    changed_pins = tuple(
        replace(
            pin,
            decision_prompt_fingerprint=_digest("different-decision-prompt"),
        )
        if pin.method_variant is original.method_variant
        else pin
        for pin in protocol_lock.method_decision_prompt_pins
    )
    changed_lock = replace(
        protocol_lock,
        method_decision_prompt_pins=changed_pins,
    )
    changed = build_run_plan("core", changed_lock).new_runs[0]

    assert changed.shared_prompt_contract_fingerprint == (
        original.shared_prompt_contract_fingerprint
    )
    assert changed.decision_prompt_fingerprint != original.decision_prompt_fingerprint
    assert changed.condition_fingerprint != original.condition_fingerprint


def test_table1_source_identity_changes_only_dependent_plan_entries(
    protocol_lock: ProtocolLock,
) -> None:
    original = build_run_plan("core", protocol_lock)
    changed_lock = replace(
        protocol_lock,
        table1_gpt_source=replace(
            protocol_lock.table1_gpt_source,
            protocol_id="coevolve-eval-v83-v33-efficient-repinned",
        ),
    )
    changed = build_run_plan("core", changed_lock)

    original_by_id = {run.run_id: run for run in original.new_runs}
    changed_by_id = {run.run_id: run for run in changed.new_runs}
    for run_id, before in original_by_id.items():
        after = changed_by_id[run_id]
        if before.memory_condition is MemoryCondition.GPT_TO_GPT:
            assert after.condition_fingerprint != before.condition_fingerprint
        else:
            assert after.condition_fingerprint == before.condition_fingerprint
    assert changed.reuses[0].source_selector.protocol_id == (
        "coevolve-eval-v83-v33-efficient-repinned"
    )
    assert changed.reuses[0].source_selector.protocol_id != (
        original.reuses[0].source_selector.protocol_id
    )


def test_baseline_and_ablation_memory_boundaries(protocol_lock: ProtocolLock) -> None:
    table2_runs = [
        run
        for run in build_run_plan("full", protocol_lock).new_runs
        if run.table_id is TableId.TABLE_2
    ]
    policies = {
        method: {run.memory_policy for run in table2_runs if run.method_variant is method}
        for method in MethodVariant
    }
    assert policies[MethodVariant.REMAC_ADAPTED] == {MemoryPolicy.DISABLED}
    assert policies[MethodVariant.COHERENT_ADAPTED] == {MemoryPolicy.DISABLED}
    assert policies[MethodVariant.NO_DYNAMIC_EXPERIENCE] == {MemoryPolicy.DISABLED}
    assert policies[MethodVariant.FLAT_REFLECTIVE_REPAIR] == {MemoryPolicy.ONLINE_FROM_EMPTY}

    remac = next(run for run in table2_runs if run.method_variant is MethodVariant.REMAC_ADAPTED)
    with pytest.raises(ValueError, match="cannot enable durable memory"):
        replace(remac, memory_policy=MemoryPolicy.ONLINE_FROM_EMPTY)


def test_transfer_runs_pin_only_the_declared_source_bank(protocol_lock: ProtocolLock) -> None:
    table4 = [
        run
        for run in build_run_plan("core", protocol_lock).new_runs
        if run.table_id is TableId.TABLE_4
    ]
    assert len(table4) == 10
    assert {run.target_backbone for run in table4} == {"gpt-5.4-mini-2026-03-17"}
    gpt_runs = [run for run in table4 if run.memory_condition is MemoryCondition.GPT_TO_GPT]
    gemini_runs = [run for run in table4 if run.memory_condition is MemoryCondition.GEMINI_TO_GPT]
    assert {run.strategy_bank.sha256 for run in gpt_runs if run.strategy_bank} == {
        protocol_lock.gpt_strategy_bank_hash
    }
    assert {run.strategy_bank.source_backbone for run in gemini_runs if run.strategy_bank} == {
        "gemini-3.5-flash"
    }
    assert all(
        run.strategy_bank.source_protocol_id
        == protocol_lock.table1_gpt_source.protocol_id
        and run.strategy_bank.source_method_commit
        == protocol_lock.table1_gpt_source.method_commit
        and run.strategy_bank.source_seeds == protocol_lock.table1_gpt_source.seeds
        for run in gpt_runs
        if run.strategy_bank
    )
    assert all(
        run.strategy_bank.source_protocol_id
        == protocol_lock.table1_gemini_source.protocol_id
        and run.strategy_bank.source_method_commit
        == protocol_lock.table1_gemini_source.method_commit
        and run.strategy_bank.source_seeds == protocol_lock.table1_gemini_source.seeds
        for run in gemini_runs
        if run.strategy_bank
    )
    assert all(run.memory_policy is MemoryPolicy.ONLINE_FROM_PRELOADED_STRATEGIES for run in table4)


def test_protocol_yaml_freezes_shared_budget_models_and_sources() -> None:
    bundle = load_protocol_bundle()
    assert bundle.common.seeds == (200, 201, 202, 203, 204)
    assert bundle.common.physical_budget == 80
    assert bundle.common.target_backbone == "gpt-5.4-mini-2026-03-17"
    assert bundle.common.source_backbones == (
        "gpt-5.4-mini-2026-03-17",
        "gemini-3.5-flash",
    )


def test_all_scene_candidates_are_planning_valid_but_not_claimed_ready(
    protocol_lock: ProtocolLock,
) -> None:
    for scene_id in SceneId:
        scene = load_scene(scene_id)
        planning = validate_scene(scene, mode="planning")
        execution = validate_scene(scene, mode="execution", protocol_lock=protocol_lock)

        assert planning.ok, planning.errors
        assert scene.protocol_invariant_ref == "lock://invariant_fingerprint"
        assert scene.terminal_predicate_ref == "lock://terminal_fingerprint"
        assert scene.route.mission_topology
        assert scene.route.progress_gates
        assert not execution.ok
        assert any("missing simulator evidence" in error for error in execution.errors)


def test_base_directed_route_is_exact_v83_route_contract() -> None:
    scene = load_scene(SceneId.BASE)
    document = directed_route_document(scene)
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    assert sha256(canonical).hexdigest() == (
        "c189b4353798b364023f80935e48610bd841db25a688a95f05920ce35c568c50"
    )
    assert set(document) == {
        "schema_version",
        "kind",
        "scene_id",
        "mission_topology",
        "progress_gates",
        "segments",
    }
    assert document["schema_version"] == 1
    assert document["kind"] == "directed_connected_route"
    assert document["scene_id"] == "two_g1_box_push_layout"
    assert [tuple(item.values()) for item in document["progress_gates"]] == [
        ("first_corner", 1.0, 0.0, 1.15),
        ("cross_passage", 0.0, 1.0, 2.85),
        ("second_corner", 0.0, 1.0, 7.35),
        ("final_straight", -1.0, 0.0, -0.85),
    ]
    assert [tuple(item.values()) for item in document["segments"]] == [
        ("initial_straight", 1.0, 0.0, -1.55, 2.28, 1.35, 2.35, -0.10, 2.32, 1.45, 0.35),
        ("first_corner_entry", 0.0, 1.0, 1.35, 2.35, 1.35, 2.85, 1.35, 2.60, 1.30, 0.35),
        ("cross_passage", 0.0, 1.0, 1.35, 2.85, 1.35, 7.35, 1.35, 5.10, 1.30, 0.35),
        ("second_corner_entry", -1.0, 0.0, 1.35, 7.35, 0.85, 7.60, 1.10, 7.48, 1.45, 0.35),
        ("final_straight", -1.0, 0.0, 0.85, 7.60, -1.50, 7.60, -0.33, 7.60, 1.45, 0.35),
    ]


def test_route_visibility_truthfully_preserves_model_action_authority() -> None:
    for scene_id in SceneId:
        route = load_scene(scene_id).route
        assert route.agent_visible
        assert route.host_validated_reference
        assert not route.host_selects_primitive
        assert not route.host_selects_coordination_mode
        assert not route.host_selects_robot_roles
        assert not route.host_selects_bounded_parameters

        with pytest.raises(ValueError, match="route Host may not select"):
            replace(route, host_selects_primitive=True)
        with pytest.raises(ValueError, match="model-visible"):
            replace(route, agent_visible=False)


def test_mirrored_directed_route_is_a_strict_x_reflection() -> None:
    base = load_scene(SceneId.BASE).route
    mirrored = load_scene(SceneId.MIRRORED).route
    for original, reflected in zip(base.segments, mirrored.segments):
        assert reflected.segment_id == original.segment_id
        assert reflected.direction_x == -original.direction_x
        assert reflected.direction_y == original.direction_y
        assert reflected.start_x_m == -original.start_x_m
        assert reflected.start_y_m == original.start_y_m
        assert reflected.goal_x_m == -original.goal_x_m
        assert reflected.goal_y_m == original.goal_y_m
        assert reflected.center_x_m == -original.center_x_m
        assert reflected.center_y_m == original.center_y_m
        assert reflected.half_width_m == original.half_width_m
        assert reflected.detour_margin_m == original.detour_margin_m
    for original, reflected in zip(base.progress_gates, mirrored.progress_gates):
        assert reflected.phase_name == original.phase_name
        assert reflected.normal_x == -original.normal_x
        assert reflected.normal_y == original.normal_y
        assert reflected.threshold_m == original.threshold_m


def test_narrow_route_preserves_centers_and_tracks_registered_clearance() -> None:
    base = load_scene(SceneId.BASE)
    narrow = load_scene(SceneId.NARROW)
    for original, reduced, base_corridor, narrow_corridor in zip(
        base.route.segments,
        narrow.route.segments,
        base.corridor.segments,
        narrow.corridor.segments,
    ):
        assert reduced.center_x_m == original.center_x_m
        assert reduced.center_y_m == original.center_y_m
        assert reduced.start_x_m == original.start_x_m
        assert reduced.start_y_m == original.start_y_m
        assert reduced.goal_x_m == original.goal_x_m
        assert reduced.goal_y_m == original.goal_y_m
        assert reduced.direction_x == original.direction_x
        assert reduced.direction_y == original.direction_y
        assert reduced.detour_margin_m == original.detour_margin_m
        assert reduced.half_width_m == pytest.approx(
            original.half_width_m
            - 0.5 * (base_corridor.clear_width_m - narrow_corridor.clear_width_m)
        )

    changed = replace(narrow.route.segments[0], half_width_m=1.24)
    bad_route = replace(
        narrow.route,
        segments=(changed,) + narrow.route.segments[1:],
    )
    tampered = replace(narrow, route=bad_route)
    tampered = replace(tampered, scene_hash=compute_scene_hash(tampered))
    report = validate_scene(tampered)
    assert any("half-width must track" in error for error in report.errors)


def test_altered_topology_carries_a_complete_explicit_directed_route() -> None:
    scene = load_scene(SceneId.ALTERED_TOPOLOGY)
    document = directed_route_document(scene)

    assert document["scene_id"] == "two_g1_box_push_layout_altered_topology"
    assert len(document["segments"]) == 6
    assert len(document["progress_gates"]) == 5
    assert len(document["mission_topology"]) == 7
    assert [segment.segment_id for segment in scene.route.segments] == [
        "initial_straight",
        "north_leg",
        "west_connector",
        "upper_north_leg",
        "east_target_approach",
        "target_entry",
    ]
    assert [
        (segment.direction_x, segment.direction_y)
        for segment in scene.route.segments
    ] == [
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.0),
    ]
    final = scene.route.segments[-1]
    assert (final.goal_x_m, final.goal_y_m) == (
        scene.target.center.x,
        scene.target.center.y,
    )


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_scene_loader_rejects_nonfinite_json_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "coevolve_experiments"
        / "scenes"
        / "base.yaml"
    ).read_text(encoding="utf-8")
    poisoned = source.replace('"radius_m": 0.3', f'"radius_m": {constant}', 1)
    assert poisoned != source
    (tmp_path / "base.yaml").write_text(poisoned, encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_scene(SceneId.BASE, root=tmp_path)


def test_scene_loader_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "coevolve_experiments"
        / "scenes"
        / "base.yaml"
    ).read_text(encoding="utf-8")
    poisoned = source.replace(
        '  "label": "Authored base corridor",',
        '  "label": "ambiguous",\n  "label": "Authored base corridor",',
        1,
    )
    assert poisoned != source
    (tmp_path / "base.yaml").write_text(poisoned, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key: label"):
        load_scene(SceneId.BASE, root=tmp_path)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize(
    ("field_name", "factory"),
    (
        ("Vec2.x", lambda value: Vec2(value, 0.0)),
        ("Vec2.y", lambda value: Vec2(0.0, value)),
        ("Vec3.x", lambda value: Vec3(value, 0.0, 0.0)),
        ("Vec3.y", lambda value: Vec3(0.0, value, 0.0)),
        ("Vec3.z", lambda value: Vec3(0.0, 0.0, value)),
        ("Pose.yaw_rad", lambda value: Pose(Vec3(0.0, 0.0, 0.0), value)),
        (
            "realization parameter values[6]",
            lambda value: RealizationParameter(
                parameter_id="wall",
                operation="set_box_transform_xyz_scale_xyz_yaw",
                targets=("/World/wall",),
                values=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, value),
            ),
        ),
        (
            "target radius_m",
            lambda value: TargetGeometry(Vec3(0.0, 0.0, 0.0), value),
        ),
        (
            "corridor clear_width_m",
            lambda value: CorridorSegment(
                "segment", Vec2(0.0, 0.0), Vec2(1.0, 0.0), value
            ),
        ),
        (
            "progress gate normal_x",
            lambda value: ProgressGate("phase", value, 0.0, 0.0),
        ),
        (
            "progress gate normal_y",
            lambda value: ProgressGate("phase", 0.0, value, 0.0),
        ),
        (
            "progress gate threshold_m",
            lambda value: ProgressGate("phase", 1.0, 0.0, value),
        ),
        (
            "directed route segment direction_x",
            lambda value: DirectedRouteSegment(
                "segment",
                value,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.5,
                0.0,
                1.0,
                0.35,
            ),
        ),
    ),
)
def test_scene_schema_rejects_every_nonfinite_physical_field(
    field_name: str,
    factory,
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(f"{field_name} must be a finite number"),
    ):
        factory(value)


def _copy_base_source(repo_root: Path) -> bytes:
    scene = load_scene(SceneId.BASE)
    source = Path(__file__).resolve().parents[2] / scene.asset.source_path
    content = source.read_bytes()
    destination = repo_root / scene.asset.source_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return content


@pytest.mark.parametrize("symlink_kind", ("leaf", "ancestor"))
def test_planning_preflight_rejects_source_asset_symlink_components(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    scene = load_scene(SceneId.BASE)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    external_asset = external / "source.usda"
    external_asset.write_bytes(
        (Path(__file__).resolve().parents[2] / scene.asset.source_path).read_bytes()
    )

    if symlink_kind == "leaf":
        source = repo_root / scene.asset.source_path
        source.parent.mkdir(parents=True)
        source.symlink_to(external_asset)
    else:
        external_assets = external / "assets"
        external_source = external_assets / "manual_scenes" / external_asset.name
        external_source.parent.mkdir(parents=True)
        external_source.write_bytes(external_asset.read_bytes())
        (repo_root / "assets").symlink_to(external_assets, target_is_directory=True)

    report = validate_scene(scene, mode="planning", repo_root=repo_root)

    assert any("source scene asset must not traverse a symlink" in error for error in report.errors)


@pytest.mark.parametrize("symlink_kind", ("leaf", "ancestor"))
def test_execution_preflight_rejects_execution_asset_symlink_components(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
    symlink_kind: str,
) -> None:
    scene = load_scene(SceneId.BASE)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    content = _copy_base_source(repo_root)
    execution_relative = "generated/base.usda"
    execution_digest = sha256(content).hexdigest()
    asset = replace(
        scene.asset,
        execution_path=execution_relative,
        execution_sha256=execution_digest,
        execution_composite_sha256="0" * 64,
    )
    asset = replace(
        asset,
        execution_composite_sha256=compute_asset_composite_hash(asset),
    )
    scene = replace(scene, asset=asset)
    scene = replace(scene, scene_hash=compute_scene_hash(scene))
    external = tmp_path / "external-execution.usda"
    external.write_bytes(content)

    if symlink_kind == "leaf":
        execution = repo_root / execution_relative
        execution.parent.mkdir(parents=True)
        execution.symlink_to(external)
    else:
        external_generated = tmp_path / "external-generated"
        external_generated.mkdir()
        (external_generated / "base.usda").write_bytes(content)
        (repo_root / "generated").symlink_to(
            external_generated,
            target_is_directory=True,
        )

    report = validate_scene(
        scene,
        mode="execution",
        protocol_lock=protocol_lock,
        repo_root=repo_root,
    )

    assert any(
        "execution scene asset must not traverse a symlink" in error
        for error in report.errors
    )


def test_candidate_execution_preflight_accepts_pinned_assets_but_requires_simulator_evidence(
    protocol_lock: ProtocolLock,
) -> None:
    for scene_id in (SceneId.MIRRORED, SceneId.NARROW, SceneId.ALTERED_TOPOLOGY):
        scene = load_scene(scene_id)
        assert scene.asset.execution_path is not None
        assert scene.asset.execution_sha256 is not None
        assert scene.asset.execution_composite_sha256 is not None
        assert compute_asset_composite_hash(scene.asset) == (
            scene.asset.execution_composite_sha256
        )
        report = validate_scene(
            scene, mode="execution", protocol_lock=protocol_lock
        )
        assert not any(
            "realized execution asset is not pinned" in error
            for error in report.errors
        )
        assert {
            error.removeprefix("missing simulator evidence: ")
            for error in report.errors
            if error.startswith("missing simulator evidence: ")
        } == {
            "simulator_integrity",
            "initial_collision",
            "scripted_solvability",
        }
        assert report.execution_ready is False

    base = load_scene(SceneId.BASE)
    assert compute_asset_composite_hash(base.asset) == base.asset.execution_composite_sha256


def _validate_evidence_payload(
    tmp_path: Path,
    payload: object,
    protocol_lock: ProtocolLock,
    evidence_name: str = "simulator_integrity",
) -> tuple[str, ...]:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    path = tmp_path / f"{evidence_name}.json"
    path.write_bytes(encoded)
    pin = EvidencePin(path=path.name, sha256=sha256(encoded).hexdigest())
    return tuple(
        _evidence_errors(
            tmp_path,
            pin,
            evidence_name,
            load_scene(SceneId.BASE),
            protocol_lock,
        )
    )


_ARTIFACT_TYPES = {
    "simulator_integrity": (
        "command",
        "config",
        "compiled_model",
        "model_inventory",
    ),
    "initial_collision": (
        "command",
        "config",
        "contact_trace",
        "support_trace",
    ),
    "scripted_solvability": (
        "command",
        "config",
        "script",
        "controller",
        "trajectory_trace",
        "verifier_trace",
    ),
}

_TEST_CONTROLLER_ID = "tests.frozen-scripted-controller"
_TEST_CONTROLLER_VERSION = "1.0.0"
_TEST_CONTROLLER_BYTES = b"frozen scripted controller fixture v1\n"


def _allow_test_controller(
    monkeypatch: pytest.MonkeyPatch,
    protocol_lock: ProtocolLock,
) -> None:
    pin = FrozenControllerPin(
        controller_id=_TEST_CONTROLLER_ID,
        controller_version=_TEST_CONTROLLER_VERSION,
        skill_api_fingerprint=protocol_lock.skill_api_fingerprint,
        controller_sha256=sha256(_TEST_CONTROLLER_BYTES).hexdigest(),
    )
    monkeypatch.setattr(
        scene_evidence,
        "ALLOWED_SCRIPTED_SOLVABILITY_CONTROLLERS",
        MappingProxyType({(_TEST_CONTROLLER_ID, _TEST_CONTROLLER_VERSION): pin}),
    )


def _valid_model_inventory(compiled_model_sha256: str) -> dict[str, object]:
    return {
        "kind": "coevolve_mujoco_model_inventory",
        "schema_version": 1,
        "compiled_model_sha256": compiled_model_sha256,
        "bodies": [
            {"id": 0, "name": "world"},
            {"id": 1, "name": "front_robot"},
            {"id": 2, "name": "back_robot"},
            {"id": 3, "name": "payload"},
        ],
        "joints": [
            {"id": 0, "name": "front_root"},
            {"id": 1, "name": "back_root"},
        ],
        "actuators": [
            {"id": 0, "name": "front_drive"},
            {"id": 1, "name": "back_drive"},
        ],
        "geoms": [
            {"id": 0, "name": "front_support"},
            {"id": 1, "name": "back_support"},
            {"id": 2, "name": "payload_geom"},
        ],
        "role_bindings": [
            {
                "role": role,
                "object_type": "body",
                "object_id": object_id,
                "object_name": role,
            }
            for role, object_id in (
                ("front_robot", 1),
                ("back_robot", 2),
                ("payload", 3),
            )
        ],
    }


def _valid_measurements(
    evidence_name: str,
    artifact_hashes: dict[str, str],
    protocol_lock: ProtocolLock,
) -> dict[str, object]:
    if evidence_name == "simulator_integrity":
        scene = load_scene(SceneId.BASE)
        return {
            "simulator_name": "MuJoCo",
            "simulator_version": "3.3.5",
            "simulator_build": "test-build",
            "model_loaded": True,
            "compiled_model_format": "mjb",
            "compiled_model_sha256": artifact_hashes["compiled_model"],
            "model_inventory_sha256": artifact_hashes["model_inventory"],
            "scene_realization_sha256": scene.asset.execution_composite_sha256,
            "body_count": 4,
            "joint_count": 2,
            "actuator_count": 2,
            "geom_count": 3,
            "invalid_model_object_count": 0,
        }
    if evidence_name == "initial_collision":
        return {
            "sample_count": 60,
            "stable_window_s": 2.0,
            "nan_sample_count": 0,
            "disallowed_contact_count": 0,
            "support_loss_sample_count": 0,
            "max_initial_penetration_m": 0.001,
            "max_robot_base_drift_m": 0.001,
            "max_payload_drift_m": 0.001,
            "max_linear_speed_mps": 0.01,
            "max_angular_speed_radps": 0.01,
        }
    return {
        "script_sha256": artifact_hashes["script"],
        "controller_id": _TEST_CONTROLLER_ID,
        "controller_version": _TEST_CONTROLLER_VERSION,
        "controller_sha256": artifact_hashes["controller"],
        "skill_api_fingerprint": protocol_lock.skill_api_fingerprint,
        "trajectory_sample_count": 120,
        "verifier_sample_count": 60,
        "physical_iteration_count": 20,
        "terminal_reached": True,
        "terminal_success": True,
        "invariant_violation_count": 0,
        "nan_sample_count": 0,
        "final_payload_target_distance_m": 0.1,
    }


def _valid_evidence_payload(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
    evidence_name: str = "simulator_integrity",
    *,
    namespace: str = "default",
) -> dict[str, object]:
    raw_artifacts: list[dict[str, object]] = []
    artifact_hashes: dict[str, str] = {}
    artifact_contents: dict[str, bytes] = {}
    if evidence_name == "simulator_integrity":
        artifact_contents["compiled_model"] = (
            b"MuJoCo MJB compiled model test fixture " + namespace.encode("utf-8") + b"\n"
        )
        compiled_hash = sha256(artifact_contents["compiled_model"]).hexdigest()
        artifact_contents["model_inventory"] = (
            json.dumps(
                _valid_model_inventory(compiled_hash),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    if evidence_name == "scripted_solvability":
        artifact_contents["controller"] = _TEST_CONTROLLER_BYTES
    for artifact_type in _ARTIFACT_TYPES[evidence_name]:
        suffix = {
            "compiled_model": ".mjb",
            "model_inventory": ".json",
            "controller": ".py",
        }.get(artifact_type, ".jsonl")
        relative_path = Path("raw") / namespace / evidence_name / f"{artifact_type}{suffix}"
        content = artifact_contents.get(
            artifact_type,
            f"{namespace}:{evidence_name}:{artifact_type}\n".encode("utf-8"),
        )
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = sha256(content).hexdigest()
        artifact_hashes[artifact_type] = digest
        raw_artifacts.append(
            {
                "artifact_type": artifact_type,
                "relative_path": relative_path.as_posix(),
                "sha256": digest,
                "size_bytes": len(content),
            }
        )
    producer_id, producer_version = ALLOWED_EVIDENCE_PRODUCERS[evidence_name]
    scene = load_scene(SceneId.BASE)
    return {
        "kind": "coevolve_scene_execution_evidence",
        "schema_version": 2,
        "check": evidence_name,
        "scene_hash": scene.scene_hash,
        "execution_sha256": scene.asset.execution_sha256,
        "execution_composite_sha256": scene.asset.execution_composite_sha256,
        "invariant_fingerprint": protocol_lock.invariant_fingerprint,
        "terminal_fingerprint": protocol_lock.terminal_fingerprint,
        "skill_api_fingerprint": protocol_lock.skill_api_fingerprint,
        "protocol_id": protocol_lock.protocol_id,
        "producer_id": producer_id,
        "producer_version": producer_version,
        "producer_commit": "a" * 40,
        "command_sha256": artifact_hashes["command"],
        "config_sha256": artifact_hashes["config"],
        "measurements": _valid_measurements(
            evidence_name,
            artifact_hashes,
            protocol_lock,
        ),
        "raw_artifacts": raw_artifacts,
    }


def test_execution_evidence_accepts_only_the_strict_audit_contract(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
) -> None:
    valid = _valid_evidence_payload(tmp_path, protocol_lock)
    assert not _validate_evidence_payload(tmp_path, valid, protocol_lock)

    caller_claim = dict(valid, passed=True)
    assert any(
        "must not contain caller-supplied passed" in error
        for error in _validate_evidence_payload(tmp_path, caller_claim, protocol_lock)
    )

    missing = dict(valid)
    del missing["check"]
    assert any(
        "missing=['check']" in error
        for error in _validate_evidence_payload(tmp_path, missing, protocol_lock)
    )

    unknown = dict(valid, model_claimed_pass=True)
    assert any(
        "unknown=['model_claimed_pass']" in error
        for error in _validate_evidence_payload(tmp_path, unknown, protocol_lock)
    )

    invalid_measurements = dict(valid, measurements=[])
    assert any(
        "measurements must be a JSON object" in error
        for error in _validate_evidence_payload(tmp_path, invalid_measurements, protocol_lock)
    )

    wrong_producer = dict(valid, producer_id="untrusted.external.producer")
    assert any(
        "producer is not allowlisted" in error
        for error in _validate_evidence_payload(tmp_path, wrong_producer, protocol_lock)
    )

    wrong_commit = dict(valid, producer_commit="b" * 40)
    assert any(
        "producer_commit does not match the protocol lock" in error
        for error in _validate_evidence_payload(tmp_path, wrong_commit, protocol_lock)
    )


def test_execution_evidence_is_host_derived_and_check_specific(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
) -> None:
    simulator_payload = _valid_evidence_payload(tmp_path, protocol_lock)
    relabelled = dict(simulator_payload, check="initial_collision")
    errors = _validate_evidence_payload(
        tmp_path,
        relabelled,
        protocol_lock,
        "initial_collision",
    )
    assert any("producer is not allowlisted" in error for error in errors)
    assert any("measurements fields differ" in error for error in errors)
    assert any("raw artifact types differ" in error for error in errors)

    scripted = _valid_evidence_payload(
        tmp_path,
        protocol_lock,
        "scripted_solvability",
    )
    failed_measurements = dict(scripted["measurements"], terminal_success=False)
    failed_scripted = dict(scripted, measurements=failed_measurements)
    assert any(
        "terminal_success was not observed" in error
        for error in _validate_evidence_payload(
            tmp_path,
            failed_scripted,
            protocol_lock,
            "scripted_solvability",
        )
    )


def test_simulator_integrity_requires_mujoco_and_semantic_model_inventory(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
) -> None:
    payload = _valid_evidence_payload(tmp_path, protocol_lock)
    old_backend = dict(
        payload,
        measurements=dict(payload["measurements"], simulator_name="Isaac Sim"),
    )
    assert any(
        "require simulator_name='MuJoCo'" in error
        for error in _validate_evidence_payload(tmp_path, old_backend, protocol_lock)
    )

    inventory_artifact = next(
        item
        for item in payload["raw_artifacts"]
        if item["artifact_type"] == "model_inventory"
    )
    inventory_path = tmp_path / inventory_artifact["relative_path"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["role_bindings"][0]["object_name"] = "wrong_body"
    content = (
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    inventory_path.write_bytes(content)
    digest = sha256(content).hexdigest()
    inventory_artifact["sha256"] = digest
    inventory_artifact["size_bytes"] = len(content)
    payload["measurements"]["model_inventory_sha256"] = digest

    errors = _validate_evidence_payload(tmp_path, payload, protocol_lock)

    assert any("object_name does not match the inventoried body" in error for error in errors)


def test_scripted_solvability_fails_closed_until_controller_bytes_are_allowlisted(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_evidence_payload(
        tmp_path,
        protocol_lock,
        "scripted_solvability",
    )
    errors = _validate_evidence_payload(
        tmp_path,
        payload,
        protocol_lock,
        "scripted_solvability",
    )
    assert any("is not in the reviewed controller allowlist" in error for error in errors)

    _allow_test_controller(monkeypatch, protocol_lock)
    assert not _validate_evidence_payload(
        tmp_path,
        payload,
        protocol_lock,
        "scripted_solvability",
    )

    wrong_api = dict(
        payload,
        measurements=dict(
            payload["measurements"],
            skill_api_fingerprint=_digest("different-skill-api"),
        ),
    )
    wrong_api_errors = _validate_evidence_payload(
        tmp_path,
        wrong_api,
        protocol_lock,
        "scripted_solvability",
    )
    assert any("does not match the protocol lock" in error for error in wrong_api_errors)



def test_execution_evidence_rehashes_raw_artifacts(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
) -> None:
    payload = _valid_evidence_payload(tmp_path, protocol_lock)
    artifact = next(
        item
        for item in payload["raw_artifacts"]
        if item["artifact_type"] == "model_inventory"
    )
    (tmp_path / artifact["relative_path"]).write_text("tampered\n", encoding="utf-8")

    errors = _validate_evidence_payload(tmp_path, payload, protocol_lock)

    assert any("raw artifact model_inventory SHA-256 mismatch" in error for error in errors)


def _scene_with_evidence(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
    namespace: str,
):
    scene = load_scene(SceneId.BASE)
    pins: dict[str, EvidencePin] = {}
    for evidence_name in _ARTIFACT_TYPES:
        payload = _valid_evidence_payload(
            tmp_path,
            protocol_lock,
            evidence_name,
            namespace=namespace,
        )
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        relative_path = Path("evidence") / namespace / f"{evidence_name}.json"
        evidence_path = tmp_path / relative_path
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_bytes(encoded)
        pins[evidence_name] = EvidencePin(
            path=relative_path.as_posix(),
            sha256=sha256(encoded).hexdigest(),
        )
    return replace(scene, evidence=ExecutionEvidence(**pins))


def test_execution_evidence_identity_is_frozen_into_report(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_test_controller(monkeypatch, protocol_lock)
    base = load_scene(SceneId.BASE)
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / base.asset.source_path
    copied_source = tmp_path / base.asset.source_path
    copied_source.parent.mkdir(parents=True)
    copied_source.write_bytes(source.read_bytes())

    first = validate_scene(
        _scene_with_evidence(tmp_path, protocol_lock, "first"),
        mode="execution",
        protocol_lock=protocol_lock,
        repo_root=tmp_path,
    )
    second = validate_scene(
        _scene_with_evidence(tmp_path, protocol_lock, "second"),
        mode="execution",
        protocol_lock=protocol_lock,
        repo_root=tmp_path,
    )

    assert first.execution_ready and second.execution_ready
    assert first.execution_evidence_fingerprint != second.execution_evidence_fingerprint
    assert first.evidence_receipts != second.evidence_receipts


def test_execution_preflight_passes_with_all_strict_hash_pinned_evidence(
    tmp_path: Path,
    protocol_lock: ProtocolLock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_test_controller(monkeypatch, protocol_lock)
    scene = load_scene(SceneId.BASE)
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / scene.asset.source_path
    copied_source = tmp_path / scene.asset.source_path
    copied_source.parent.mkdir(parents=True)
    copied_source.write_bytes(source.read_bytes())

    pins: dict[str, EvidencePin] = {}
    for evidence_name in (
        "simulator_integrity",
        "initial_collision",
        "scripted_solvability",
    ):
        payload = _valid_evidence_payload(tmp_path, protocol_lock, evidence_name)
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        relative_path = Path("evidence") / f"{evidence_name}.json"
        evidence_path = tmp_path / relative_path
        evidence_path.parent.mkdir(exist_ok=True)
        evidence_path.write_bytes(encoded)
        pins[evidence_name] = EvidencePin(
            path=relative_path.as_posix(),
            sha256=sha256(encoded).hexdigest(),
        )

    scene = replace(scene, evidence=ExecutionEvidence(**pins))
    report = validate_scene(
        scene,
        mode="execution",
        protocol_lock=protocol_lock,
        repo_root=tmp_path,
    )
    assert report.execution_ready, report.errors
    assert report.execution_evidence_fingerprint is not None
    assert {item.check for item in report.evidence_receipts} == set(_ARTIFACT_TYPES)
    assert all(item.raw_artifacts for item in report.evidence_receipts)


def test_mirrored_scene_checks_poses_camera_route_and_oriented_gates() -> None:
    mirrored = load_scene(SceneId.MIRRORED)
    assert validate_scene(mirrored).ok

    bad_target = replace(mirrored.target, center=Vec3(1.6, 7.6, 0.3))
    tampered = replace(mirrored, target=bad_target)
    tampered = replace(tampered, scene_hash=compute_scene_hash(tampered))
    report = validate_scene(tampered)
    assert not report.ok
    assert any("mirrored target" in error for error in report.errors)

    first_gate = replace(mirrored.route.progress_gates[0], threshold_m=1.10)
    bad_route = replace(
        mirrored.route,
        progress_gates=(first_gate,) + mirrored.route.progress_gates[1:],
    )
    tampered_gate = replace(mirrored, route=bad_route)
    tampered_gate = replace(tampered_gate, scene_hash=compute_scene_hash(tampered_gate))
    gate_report = validate_scene(tampered_gate)
    assert any("mirrored progress gate" in error for error in gate_report.errors)


def test_scene_documents_are_self_hashed_and_do_not_import_private_runners() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "coevolve_experiments"
    for scene_id in SceneId:
        scene = load_scene(scene_id)
        assert compute_scene_hash(scene) == scene.scene_hash
        assert scene.asset.realization_parameters
    python_text = "\n".join(path.read_text(encoding="utf-8") for path in package_root.rglob("*.py"))
    assert "import scripts." not in python_text
    assert "from scripts." not in python_text
    assert "import praxis_coop.cap_policy.formal_eval" not in python_text
    assert "from praxis_coop.cap_policy.formal_eval" not in python_text
