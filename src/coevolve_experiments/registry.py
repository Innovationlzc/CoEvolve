"""Compile declarative Tables 2--4 into one immutable run plan."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from .contracts import (
    ExperimentSuite,
    MemoryCondition,
    MemoryPolicy,
    MethodVariant,
    ProtocolLock,
    ReuseSpec,
    RunPlan,
    RunSpec,
    SceneId,
    SourceSelector,
    StrategyBankPin,
    SuiteTier,
    TableId,
    hash_expectations,
)
from .protocols import ProtocolCondition, load_protocol_bundle
from .scenes import SceneSpec, load_scene, validate_scene


def build_suite(
    tier: Union[SuiteTier, str],
    protocol_lock: ProtocolLock,
    *,
    protocol_root: Optional[Path] = None,
    scene_root: Optional[Path] = None,
) -> ExperimentSuite:
    """Build the core (35) or full (45) set of *new unique runs*.

    Reused Table-1/Table-3 rows are represented separately by
    :class:`ReuseSpec`; they never consume a new output path or inflate the run
    count.  Candidate scene construction/evidence is an execution concern, so
    suite compilation requires planning validity but does not claim simulator
    readiness.
    """

    normalized_tier = tier if isinstance(tier, SuiteTier) else SuiteTier(tier)
    # This is a paper-facing compiler, not a generic matrix generator. Exact
    # row identity/order must be checked before deriving any run or reuse;
    # matching only the aggregate 35/45 count would allow claim drift.
    from .paper_contract import build_paper_contract

    build_paper_contract(normalized_tier, protocol_root=protocol_root)
    bundle = load_protocol_bundle(protocol_root)
    if bundle.common.target_provider_route_id != protocol_lock.target_provider_route_id:
        raise ValueError("protocol target provider route differs from the lock")
    locked_source_routes = (
        protocol_lock.table1_gpt_source.provider_route_id,
        protocol_lock.table1_gemini_source.provider_route_id,
    )
    if bundle.common.source_provider_route_ids != locked_source_routes:
        raise ValueError("protocol source provider routes differ from the lock")
    scenes = _load_planning_scenes(scene_root)
    runs = []
    condition_fingerprints: Dict[tuple[TableId, str], str] = {}

    for table in bundle.tables:
        for condition in table.conditions:
            if normalized_tier is SuiteTier.CORE and not condition.core:
                continue
            if condition.execution == "reuse":
                continue
            scene = scenes[condition.scene_id]
            memory_policy = _memory_policy(condition)
            strategy_bank = _strategy_bank(condition, bundle.common.source_backbones, protocol_lock)
            condition_fingerprint = _condition_fingerprint(
                condition=condition,
                scene=scene,
                memory_policy=memory_policy,
                strategy_bank=strategy_bank,
                target_backbone=bundle.common.target_backbone,
                budget=bundle.common.physical_budget,
                protocol_lock=protocol_lock,
            )
            condition_fingerprints[(table.table_id, condition.condition_id)] = condition_fingerprint
            for seed in bundle.common.seeds:
                output_relpath = (
                    f"{table.table_id.value}/{condition.condition_id}/seed_{seed:03d}"
                )
                runs.append(
                    RunSpec(
                        run_id=f"{table.table_id.value}__{condition.condition_id}__seed_{seed:03d}",
                        table_id=table.table_id,
                        seed=seed,
                        method_variant=condition.method_variant,
                        scene_id=condition.scene_id,
                        memory_condition=condition.memory_condition,
                        memory_policy=memory_policy,
                        target_backbone=bundle.common.target_backbone,
                        provider_route_id=protocol_lock.target_provider_route_id,
                        provider_route_fingerprint=(
                            protocol_lock.target_provider_route_fingerprint
                        ),
                        physical_budget=bundle.common.physical_budget,
                        condition_fingerprint=condition_fingerprint,
                        invariant_fingerprint=protocol_lock.invariant_fingerprint,
                        base_method_commit=protocol_lock.base_method_commit,
                        protocol_id=protocol_lock.protocol_id,
                        output_relpath=output_relpath,
                        scene_fingerprint=scene.scene_hash,
                        terminal_fingerprint=protocol_lock.terminal_fingerprint,
                        observation_contract_fingerprint=(
                            protocol_lock.observation_contract_fingerprint
                        ),
                        task_instruction_fingerprint=(
                            protocol_lock.task_instruction_fingerprint
                        ),
                        shared_prompt_contract_fingerprint=(
                            protocol_lock.shared_prompt_contract_fingerprint
                        ),
                        decision_prompt_fingerprint=(
                            protocol_lock.decision_prompt_fingerprint_for(
                                condition.method_variant
                            )
                        ),
                        skill_api_fingerprint=protocol_lock.skill_api_fingerprint,
                        full_method_compatibility_fingerprint=(
                            protocol_lock.full_method_compatibility.fingerprint
                        ),
                        strategy_bank=strategy_bank,
                    )
                )

    reuses = _build_reuses(
        bundle=bundle,
        tier=normalized_tier,
        lock=protocol_lock,
        scenes=scenes,
        condition_fingerprints=condition_fingerprints,
    )
    plan = RunPlan(
        tier=normalized_tier,
        protocol_id=protocol_lock.protocol_id,
        new_runs=tuple(runs),
        reuses=reuses,
    )
    expected_count = 35 if normalized_tier is SuiteTier.CORE else 45
    if plan.unique_run_count != expected_count:
        raise ValueError(
            f"{normalized_tier.value} protocol compiled to {plan.unique_run_count} new runs; "
            f"expected {expected_count}"
        )
    if len(plan.reuses) != 3:
        raise ValueError("the frozen protocol must contain Full/Base/Empty reuse rows")
    return ExperimentSuite(protocol_lock=protocol_lock, run_plan=plan)


def build_run_plan(
    tier: Union[SuiteTier, str],
    protocol_lock: ProtocolLock,
    **kwargs: Any,
) -> RunPlan:
    """Convenience wrapper for callers that need only the plan."""

    return build_suite(tier, protocol_lock, **kwargs).run_plan


def _load_planning_scenes(scene_root: Optional[Path]) -> Mapping[SceneId, SceneSpec]:
    scenes: Dict[SceneId, SceneSpec] = {}
    for scene_id in SceneId:
        scene = load_scene(scene_id, root=scene_root)
        report = validate_scene(scene, mode="planning", scene_root=scene_root)
        if not report.ok:
            report.require_ok()
        scenes[scene_id] = scene
    return scenes


def _memory_policy(condition: ProtocolCondition) -> MemoryPolicy:
    if condition.memory_condition is MemoryCondition.NO_DYNAMIC:
        return MemoryPolicy.DISABLED
    if condition.method_variant in {
        MethodVariant.REMAC_ADAPTED,
        MethodVariant.COHERENT_ADAPTED,
    }:
        return MemoryPolicy.DISABLED
    if condition.memory_condition is MemoryCondition.EMPTY:
        return MemoryPolicy.ONLINE_FROM_EMPTY
    return MemoryPolicy.ONLINE_FROM_PRELOADED_STRATEGIES


def _strategy_bank(
    condition: ProtocolCondition,
    source_backbones: Tuple[str, ...],
    lock: ProtocolLock,
) -> Optional[StrategyBankPin]:
    if condition.memory_condition is MemoryCondition.GPT_TO_GPT:
        source_backbone = source_backbones[0]
        source = lock.table1_gpt_source
        digest = lock.gpt_strategy_bank_hash
    elif condition.memory_condition is MemoryCondition.GEMINI_TO_GPT:
        source_backbone = source_backbones[1]
        source = lock.table1_gemini_source
        digest = lock.gemini_strategy_bank_hash
    else:
        return None
    if source.backbone != source_backbone:
        raise ValueError(
            "protocol source backbone differs from its locked Table-1 cohort: "
            f"{source_backbone!r} != {source.backbone!r}"
        )
    return StrategyBankPin(
        source_backbone=source_backbone,
        source_protocol_id=source.protocol_id,
        source_method_commit=source.method_commit,
        source_seeds=source.seeds,
        source_condition_fingerprint=source.condition_fingerprint,
        source_result_manifest_hash=source.result_manifest_hash,
        source_provider_route_id=source.provider_route_id,
        source_provider_route_fingerprint=source.provider_route_fingerprint,
        selector=(
            f"table1/{source.protocol_id}/base/{source_backbone}/"
            "route_abstracted_strategy_bank"
        ),
        sha256=digest,
    )


def _condition_fingerprint(
    *,
    condition: ProtocolCondition,
    scene: SceneSpec,
    memory_policy: MemoryPolicy,
    strategy_bank: Optional[StrategyBankPin],
    target_backbone: str,
    budget: int,
    protocol_lock: ProtocolLock,
) -> str:
    # Table ID, seed, and output path are deliberately absent: this is a
    # semantic condition identity, not a reporting-location identity.
    payload = {
        "protocol_id": protocol_lock.protocol_id,
        "invariant_fingerprint": protocol_lock.invariant_fingerprint,
        "terminal_fingerprint": protocol_lock.terminal_fingerprint,
        "base_method_commit": protocol_lock.base_method_commit,
        "method_variant": condition.method_variant.value,
        "scene_id": condition.scene_id.value,
        "scene_fingerprint": scene.scene_hash,
        "memory_condition": condition.memory_condition.value,
        "memory_policy": memory_policy.value,
        "target_backbone": target_backbone,
        "provider_route_id": protocol_lock.target_provider_route_id,
        "provider_route_fingerprint": (
            protocol_lock.target_provider_route_fingerprint
        ),
        "physical_budget": budget,
        "observation_contract_fingerprint": (
            protocol_lock.observation_contract_fingerprint
        ),
        "task_instruction_fingerprint": protocol_lock.task_instruction_fingerprint,
        "shared_prompt_contract_fingerprint": (
            protocol_lock.shared_prompt_contract_fingerprint
        ),
        "decision_prompt_fingerprint": (
            protocol_lock.decision_prompt_fingerprint_for(
                condition.method_variant
            )
        ),
        "skill_api_fingerprint": protocol_lock.skill_api_fingerprint,
        "full_method_compatibility_fingerprint": (
            protocol_lock.full_method_compatibility.fingerprint
        ),
        "strategy_bank": asdict(strategy_bank) if strategy_bank is not None else None,
    }
    return _digest(payload)


def _build_reuses(
    *,
    bundle: Any,
    tier: SuiteTier,
    lock: ProtocolLock,
    scenes: Mapping[SceneId, SceneSpec],
    condition_fingerprints: Mapping[tuple[TableId, str], str],
) -> Tuple[ReuseSpec, ...]:
    reuses = []
    for table in bundle.tables:
        for condition in table.conditions:
            if tier is SuiteTier.CORE and not condition.core:
                continue
            if condition.execution != "reuse":
                continue
            if condition.source == "table1_gpt_base":
                source = lock.table1_gpt_source
                if source.backbone != bundle.common.target_backbone:
                    raise ValueError(
                        "Table-1 GPT source backbone differs from target_backbone: "
                        f"{source.backbone!r} != {bundle.common.target_backbone!r}"
                    )
                selector = SourceSelector(
                    source_table="table1_backbone_robustness",
                    protocol_id=source.protocol_id,
                    method_commit=source.method_commit,
                    backbone=source.backbone,
                    method_variant=MethodVariant.COEVOLE_FULL,
                    scene_id=SceneId.BASE,
                    memory_condition=MemoryCondition.EMPTY,
                    seeds=source.seeds,
                    provider_route_id=source.provider_route_id,
                    provider_route_fingerprint=source.provider_route_fingerprint,
                    full_method_compatibility_fingerprint=(
                        source.full_method_compatibility_fingerprint
                    ),
                    output_relpath=(
                        "table1_backbone_robustness/"
                        f"{bundle.common.target_backbone}/base"
                    ),
                    external=True,
                )
                expected_items = [
                    ("condition", source.condition_fingerprint),
                    ("invariant", lock.invariant_fingerprint),
                    ("scene", scenes[SceneId.BASE].scene_hash),
                    ("terminal", lock.terminal_fingerprint),
                    ("provider_route", source.provider_route_fingerprint),
                    ("observation_contract", lock.observation_contract_fingerprint),
                    ("task_instruction", lock.task_instruction_fingerprint),
                    (
                        "shared_prompt_contract",
                        lock.shared_prompt_contract_fingerprint,
                    ),
                    (
                        "decision_prompt",
                        lock.decision_prompt_fingerprint_for(
                            MethodVariant.COEVOLE_FULL
                        ),
                    ),
                    ("skill_api", lock.skill_api_fingerprint),
                    (
                        "full_method_compatibility",
                        source.full_method_compatibility_fingerprint,
                    ),
                    ("result_manifest", source.result_manifest_hash),
                ]
            elif condition.source == "table3_mirrored":
                selector = SourceSelector(
                    source_table=TableId.TABLE_3.value,
                    protocol_id=lock.protocol_id,
                    method_commit=lock.base_method_commit,
                    backbone=bundle.common.target_backbone,
                    method_variant=MethodVariant.COEVOLE_FULL,
                    scene_id=SceneId.MIRRORED,
                    memory_condition=MemoryCondition.EMPTY,
                    seeds=bundle.common.seeds,
                    provider_route_id=lock.target_provider_route_id,
                    provider_route_fingerprint=(
                        lock.target_provider_route_fingerprint
                    ),
                    full_method_compatibility_fingerprint=(
                        lock.full_method_compatibility.fingerprint
                    ),
                    output_relpath=f"{TableId.TABLE_3.value}/mirrored",
                    external=False,
                )
                try:
                    mirrored_condition = condition_fingerprints[(TableId.TABLE_3, "mirrored")]
                except KeyError as exc:
                    raise ValueError("Table-4 Empty reuse requires Table-3 Mirrored new runs") from exc
                expected_items = [
                    ("condition", mirrored_condition),
                    ("invariant", lock.invariant_fingerprint),
                    ("scene", scenes[SceneId.MIRRORED].scene_hash),
                    ("terminal", lock.terminal_fingerprint),
                    (
                        "provider_route",
                        lock.target_provider_route_fingerprint,
                    ),
                    ("observation_contract", lock.observation_contract_fingerprint),
                    ("task_instruction", lock.task_instruction_fingerprint),
                    (
                        "shared_prompt_contract",
                        lock.shared_prompt_contract_fingerprint,
                    ),
                    (
                        "decision_prompt",
                        lock.decision_prompt_fingerprint_for(
                            MethodVariant.COEVOLE_FULL
                        ),
                    ),
                    ("skill_api", lock.skill_api_fingerprint),
                    (
                        "full_method_compatibility",
                        lock.full_method_compatibility.fingerprint,
                    ),
                ]
            else:
                raise ValueError(f"unknown reuse source: {condition.source}")
            expected_items.append(("selector", _selector_fingerprint(selector)))
            reuses.append(
                ReuseSpec(
                    reuse_id=f"{table.table_id.value}__{condition.condition_id}",
                    target_table=table.table_id,
                    target_label=condition.label,
                    source_selector=selector,
                    expected_hashes=hash_expectations(expected_items),
                )
            )
    return tuple(reuses)


def _selector_fingerprint(selector: SourceSelector) -> str:
    payload = {
        "source_table": selector.source_table,
        "protocol_id": selector.protocol_id,
        "method_commit": selector.method_commit,
        "backbone": selector.backbone,
        "method_variant": selector.method_variant.value,
        "scene_id": selector.scene_id.value,
        "memory_condition": selector.memory_condition.value,
        "seeds": selector.seeds,
        "provider_route_id": selector.provider_route_id,
        "provider_route_fingerprint": selector.provider_route_fingerprint,
        "full_method_compatibility_fingerprint": (
            selector.full_method_compatibility_fingerprint
        ),
        "output_relpath": selector.output_relpath,
        "external": selector.external,
    }
    return _digest(payload)


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["build_run_plan", "build_suite"]
