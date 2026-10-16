"""Compile one plan-bound request for the production Code-as-Policy runner.

This module intentionally stops before subprocess launch.  It proves that a
single ``run_id`` resolves to one frozen scene/method/memory configuration and
lists every missing production capability.  The eventual runner must consume
this object as a whole; accepting independent scene, method, or budget CLI
overrides would reopen the fairness contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from ..contracts import MemoryCondition, MethodVariant, RunSpec
from ..scenes import load_scene, validate_scene
from ..serialization import sha256_json
from ..suite_plan import load_verified_suite_plan
from .contracts import ExecutionCapability, PolicyRepresentation
from .preflight import current_live_bridge_preflight, require_current_live_bridge


LIVE_LAUNCH_CONFIG_SCHEMA_VERSION = 3
FROM_SCRATCH_BOOTSTRAP_MISSING_CODE = (
    "FROM_SCRATCH_MEMORY_BOOTSTRAP_NOT_PREPARED"
)
_SCENE_INJECTION_CHANNELS = (
    "term1_simulator_asset",
    "host_clearance_geometry",
    "vlm_scene_prior",
    "route_and_progress_gates",
    "formal_run_manifest",
)


class LiveLaunchConfigError(ValueError):
    """Raised before any physical process or output mutation occurs."""


@dataclass(frozen=True, slots=True)
class MethodExecutionContract:
    method_variant: MethodVariant
    decision_structure: str
    policy_representation: PolicyRepresentation
    hierarchical_evolution: bool
    strategy_retrieval: bool
    strategy_promotion: bool
    action_exemplar_retrieval: bool
    action_exemplar_promotion: bool
    common_physics_verifier: bool = True
    common_checkpoint_rollback: bool = True
    common_terminal_predicate: bool = True
    frozen_sonic_skills: bool = True


@dataclass(frozen=True, slots=True)
class SceneExecutionBinding:
    scene_id: str
    scene_fingerprint: str
    registry_source_path: str
    registry_source_sha256: str
    execution_path: Optional[str]
    execution_sha256: Optional[str]
    execution_composite_sha256: Optional[str]
    execution_evidence_fingerprint: Optional[str]
    compiled_model_sha256: Optional[str]
    target_center_xyz_m: Tuple[float, float, float]
    route_agent_visible: bool
    injection_channels: Tuple[str, ...] = _SCENE_INJECTION_CHANNELS


@dataclass(frozen=True, slots=True)
class MemoryExecutionBinding:
    memory_condition: MemoryCondition
    memory_policy: str
    strategy_bank_source_backbone: Optional[str]
    strategy_bank_source_protocol_id: Optional[str]
    strategy_bank_source_method_commit: Optional[str]
    strategy_bank_source_seeds: Optional[Tuple[int, ...]]
    strategy_bank_source_condition_fingerprint: Optional[str]
    strategy_bank_source_result_manifest_hash: Optional[str]
    strategy_bank_expected_sha256: Optional[str]
    strategy_bank_path: Optional[Path]
    strategy_bank_actual_sha256: Optional[str]
    strategy_bank_provenance_path: Optional[Path]
    strategy_bank_provenance_sha256: Optional[str]
    bootstrap_snapshot_path: Optional[Path] = None
    bootstrap_snapshot_sha256: Optional[str] = None
    bootstrap_snapshot_fingerprint: Optional[str] = None
    bootstrap_formal_binding_sha256: Optional[str] = None
    bootstrap_initial_file_sha256s: Optional[Tuple[Tuple[str, str], ...]] = None
    target_online_overlay_starts_empty: bool = False
    target_action_exemplars_start_empty: bool = False
    target_manifest_starts_empty: bool = False
    target_failure_cache_starts_empty: bool = False


@dataclass(frozen=True, slots=True)
class CompiledLiveLaunch:
    schema_version: int
    run_spec: RunSpec
    plan_path: Path
    plan_sha256: str
    results_root: Path
    output_dir: Path
    method: MethodExecutionContract
    scene: SceneExecutionBinding
    memory: MemoryExecutionBinding
    execution_capability: ExecutionCapability
    paper_evidence_eligible: bool
    blockers: Tuple[str, ...]
    launch_fingerprint: str


def compile_live_launch(
    plan_path: str | Path,
    run_id: str,
    *,
    results_root: str | Path,
    strategy_bank_path: str | Path | None = None,
    strategy_bank_provenance_path: str | Path | None = None,
    require_launchable: bool = False,
) -> CompiledLiveLaunch:
    """Resolve one immutable launch without touching the result directory."""

    try:
        verified = load_verified_suite_plan(
            plan_path,
            require_execution_ready=require_launchable,
        )
        run = verified.resolve_run(run_id)
    except (OSError, TypeError, ValueError) as exc:
        raise LiveLaunchConfigError(str(exc)) from exc

    root = _absolute_nonsymlink_path(results_root, "formal results root", must_exist=False)
    output_dir = (root / run.output_relpath).resolve(strict=False)
    try:
        output_dir.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise LiveLaunchConfigError("planned output path escapes results root") from exc

    scene = load_scene(run.scene_id)
    if scene.scene_hash != run.scene_fingerprint:
        raise LiveLaunchConfigError("RunSpec scene fingerprint differs from registry")
    planning = validate_scene(scene, mode="planning")
    if not planning.ok:
        raise LiveLaunchConfigError(
            "selected scene is not planning valid: " + "; ".join(planning.errors)
        )

    method_binding = _method_contract(run.method_variant)
    memory_binding, memory_blockers = _memory_binding(
        run,
        plan_path=verified.path,
        plan_sha256=verified.sha256,
        output_dir=output_dir,
        strategy_bank_path=strategy_bank_path,
        strategy_bank_provenance_path=strategy_bank_provenance_path,
    )
    evidence_fingerprint, compiled_model_sha256 = _scene_execution_evidence_identity(
        verified.execution_reports,
        scene.scene_id.value,
    )
    scene_binding = SceneExecutionBinding(
        scene_id=scene.scene_id.value,
        scene_fingerprint=scene.scene_hash,
        registry_source_path=scene.asset.source_path,
        registry_source_sha256=scene.asset.source_sha256,
        execution_path=scene.asset.execution_path,
        execution_sha256=scene.asset.execution_sha256,
        execution_composite_sha256=scene.asset.execution_composite_sha256,
        execution_evidence_fingerprint=evidence_fingerprint,
        compiled_model_sha256=compiled_model_sha256,
        target_center_xyz_m=scene.target.center.as_tuple(),
        route_agent_visible=scene.route.agent_visible,
    )

    blockers = list(memory_blockers)
    if verified.execution_preflight != "passed":
        blockers.append("SCENE_EXECUTION_PREFLIGHT_NOT_FROZEN")
    else:
        execution = validate_scene(
            scene,
            mode="execution",
            protocol_lock=verified.protocol_lock,
        )
        if not execution.execution_ready:
            blockers.append("CURRENT_SCENE_EXECUTION_EVIDENCE_NOT_READY")
        if (
            scene_binding.execution_evidence_fingerprint is None
            or scene_binding.compiled_model_sha256 is None
        ):
            blockers.append("SCENE_RESULT_EVIDENCE_IDENTITY_NOT_BOUND")
    bridge = current_live_bridge_preflight()
    if not bridge.available:
        blockers.append(bridge.reason_code)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not _is_pristine_prepared_output(output_dir, run, memory_binding):
            blockers.append("PLANNED_OUTPUT_DIRECTORY_NOT_EMPTY")

    unique_blockers = tuple(dict.fromkeys(blockers))
    paper_evidence_eligible = not unique_blockers
    if require_launchable:
        if unique_blockers:
            raise LiveLaunchConfigError(
                "formal launch is blocked: " + ", ".join(unique_blockers)
            )
        # Keep this final check adjacent to the future subprocess boundary.
        require_current_live_bridge()

    fingerprint_payload = {
        "schema_version": LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
        "run_spec": run,
        "plan_sha256": verified.sha256,
        "results_root": root,
        "output_dir": output_dir,
        "method": method_binding,
        "scene": scene_binding,
        "memory": memory_binding,
        "execution_capability": bridge.execution_capability,
    }
    return CompiledLiveLaunch(
        schema_version=LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
        run_spec=run,
        plan_path=verified.path,
        plan_sha256=verified.sha256,
        results_root=root,
        output_dir=output_dir,
        method=method_binding,
        scene=scene_binding,
        memory=memory_binding,
        execution_capability=bridge.execution_capability,
        paper_evidence_eligible=paper_evidence_eligible,
        blockers=unique_blockers,
        launch_fingerprint=sha256_json(fingerprint_payload),
    )


def _method_contract(variant: MethodVariant) -> MethodExecutionContract:
    common = {
        "policy_representation": PolicyRepresentation.PRAXIS_CODE_AS_POLICY,
    }
    if variant is MethodVariant.COEVOLE_FULL:
        return MethodExecutionContract(
            method_variant=variant,
            decision_structure="hierarchical_evolution",
            hierarchical_evolution=True,
            strategy_retrieval=True,
            strategy_promotion=True,
            action_exemplar_retrieval=True,
            action_exemplar_promotion=True,
            **common,
        )
    if variant is MethodVariant.NO_DYNAMIC_EXPERIENCE:
        return MethodExecutionContract(
            method_variant=variant,
            decision_structure="hierarchical_evolution",
            hierarchical_evolution=True,
            strategy_retrieval=False,
            strategy_promotion=False,
            action_exemplar_retrieval=False,
            action_exemplar_promotion=False,
            **common,
        )
    if variant is MethodVariant.FLAT_REFLECTIVE_REPAIR:
        return MethodExecutionContract(
            method_variant=variant,
            decision_structure="flat_failure_conditioned_code_regeneration",
            hierarchical_evolution=False,
            strategy_retrieval=True,
            strategy_promotion=True,
            action_exemplar_retrieval=True,
            action_exemplar_promotion=True,
            **common,
        )
    if variant is MethodVariant.REMAC_ADAPTED:
        return MethodExecutionContract(
            method_variant=variant,
            decision_structure="task_plan_reflection_and_replanning",
            hierarchical_evolution=False,
            strategy_retrieval=False,
            strategy_promotion=False,
            action_exemplar_retrieval=False,
            action_exemplar_promotion=False,
            **common,
        )
    if variant is MethodVariant.COHERENT_ADAPTED:
        return MethodExecutionContract(
            method_variant=variant,
            decision_structure="central_proposal_execution_feedback_adjustment",
            hierarchical_evolution=False,
            strategy_retrieval=False,
            strategy_promotion=False,
            action_exemplar_retrieval=False,
            action_exemplar_promotion=False,
            **common,
        )
    raise LiveLaunchConfigError(f"unsupported method variant: {variant}")


def _scene_execution_evidence_identity(
    reports: Tuple[Mapping[str, Any], ...],
    scene_id: str,
) -> tuple[Optional[str], Optional[str]]:
    """Read the execution-evidence and compiled-model pins from one report."""

    selected = tuple(
        report
        for report in reports
        if isinstance(report, Mapping) and report.get("scene_id") == scene_id
    )
    if len(selected) != 1:
        return None, None
    report = selected[0]
    fingerprint = report.get("execution_evidence_fingerprint")
    receipts = report.get("evidence_receipts")
    if not isinstance(fingerprint, str) or not isinstance(receipts, list):
        return None, None
    simulator = tuple(
        receipt
        for receipt in receipts
        if isinstance(receipt, Mapping) and receipt.get("check") == "simulator_integrity"
    )
    if len(simulator) != 1 or not isinstance(simulator[0].get("raw_artifacts"), list):
        return None, None
    compiled = tuple(
        artifact.get("sha256")
        for artifact in simulator[0]["raw_artifacts"]
        if isinstance(artifact, Mapping) and artifact.get("artifact_type") == "compiled_model"
    )
    if len(compiled) != 1 or not isinstance(compiled[0], str):
        return None, None
    return fingerprint, compiled[0]


def _memory_binding(
    run: RunSpec,
    *,
    plan_path: Path,
    plan_sha256: str,
    output_dir: Path,
    strategy_bank_path: str | Path | None,
    strategy_bank_provenance_path: str | Path | None,
) -> tuple[MemoryExecutionBinding, Tuple[str, ...]]:
    transfer = run.strategy_bank is not None
    if not transfer:
        if strategy_bank_path is not None or strategy_bank_provenance_path is not None:
            raise LiveLaunchConfigError(
                "strategy-bank files are valid only for a transfer RunSpec"
            )
        from ..memory.from_scratch import (
            FROM_SCRATCH_BOOTSTRAP_FILENAME,
            FromScratchMemoryError,
            validate_from_scratch_bootstrap,
        )

        snapshot_path: Optional[Path] = None
        snapshot_sha: Optional[str] = None
        snapshot_fingerprint: Optional[str] = None
        formal_binding_sha: Optional[str] = None
        initial_file_hashes: Optional[Tuple[Tuple[str, str], ...]] = None
        blockers = []
        candidate = output_dir / "memory" / FROM_SCRATCH_BOOTSTRAP_FILENAME
        if not candidate.is_file() or candidate.is_symlink():
            blockers.append(FROM_SCRATCH_BOOTSTRAP_MISSING_CODE)
        else:
            try:
                snapshot = validate_from_scratch_bootstrap(
                    candidate.parent,
                    run,
                    require_initial_state=False,
                    expected_plan_path=plan_path,
                    expected_plan_sha256=plan_sha256,
                )
            except FromScratchMemoryError as exc:
                raise LiveLaunchConfigError(
                    f"from-scratch memory bootstrap is invalid: {exc}"
                ) from exc
            snapshot_path = snapshot.path
            snapshot_sha = snapshot.file_sha256
            snapshot_fingerprint = snapshot.snapshot_fingerprint
            formal_binding_sha = snapshot.formal_binding_sha256
            initial_file_hashes = snapshot.initial_file_sha256s
        return (
            MemoryExecutionBinding(
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
                bootstrap_snapshot_path=snapshot_path,
                bootstrap_snapshot_sha256=snapshot_sha,
                bootstrap_snapshot_fingerprint=snapshot_fingerprint,
                bootstrap_formal_binding_sha256=formal_binding_sha,
                bootstrap_initial_file_sha256s=initial_file_hashes,
                target_online_overlay_starts_empty=snapshot_path is not None,
                target_action_exemplars_start_empty=snapshot_path is not None,
                target_manifest_starts_empty=snapshot_path is not None,
                target_failure_cache_starts_empty=snapshot_path is not None,
            ),
            tuple(blockers),
        )

    blockers = []
    bank_path: Optional[Path] = None
    provenance_path: Optional[Path] = None
    provenance_sha: Optional[str] = None
    actual_sha: Optional[str] = None
    snapshot_path: Optional[Path] = None
    snapshot_sha: Optional[str] = None
    snapshot_fingerprint: Optional[str] = None
    formal_binding_sha: Optional[str] = None
    initial_file_hashes: Optional[Tuple[Tuple[str, str], ...]] = None
    if strategy_bank_path is None or strategy_bank_provenance_path is None:
        blockers.append("TRANSFER_STRATEGY_BANK_FILES_NOT_BOUND")
    else:
        # Local import avoids a live <-> memory package initialization cycle;
        # the memory package's formal preparation boundary imports live.
        from ..memory.validation import validate_strategy_bank

        bank_path = _absolute_nonsymlink_path(
            strategy_bank_path, "strategy bank", must_exist=True
        )
        provenance_path = _absolute_nonsymlink_path(
            strategy_bank_provenance_path,
            "strategy bank provenance",
            must_exist=True,
        )
        validation = validate_strategy_bank(bank_path)
        actual_sha = validation.bank_sha256
        if actual_sha != run.strategy_bank.sha256:
            raise LiveLaunchConfigError(
                "strategy bank does not match the selected RunSpec hash"
            )
        provenance_sha = _validate_bank_provenance(
            provenance_path, run, actual_sha
        )
        from ..memory.bootstrap import (
            TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME,
            MemoryBootstrapError,
            validate_transfer_bootstrap_snapshot,
        )

        expected_binding = _expected_transfer_formal_binding(
            run,
            plan_path=plan_path,
            plan_sha256=plan_sha256,
        )
        candidate = output_dir / "memory" / TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME
        if not candidate.is_file() or candidate.is_symlink():
            blockers.append("TRANSFER_MEMORY_BOOTSTRAP_NOT_PREPARED")
        else:
            try:
                snapshot = validate_transfer_bootstrap_snapshot(
                    candidate.parent,
                    require_initial_state=False,
                    expected_formal_binding=expected_binding,
                    expected_condition=run.memory_condition.value,
                    expected_target_model=run.target_backbone,
                    expected_seed_bank_sha256=run.strategy_bank.sha256,
                )
            except MemoryBootstrapError as exc:
                raise LiveLaunchConfigError(
                    f"transfer memory bootstrap is invalid: {exc}"
                ) from exc
            snapshot_path = snapshot.path
            snapshot_sha = snapshot.file_sha256
            snapshot_fingerprint = snapshot.snapshot_fingerprint
            formal_binding_sha = snapshot.formal_binding_sha256
            initial_file_hashes = snapshot.initial_file_sha256s

    return (
        MemoryExecutionBinding(
            memory_condition=run.memory_condition,
            memory_policy=run.memory_policy.value,
            strategy_bank_source_backbone=run.strategy_bank.source_backbone,
            strategy_bank_source_protocol_id=run.strategy_bank.source_protocol_id,
            strategy_bank_source_method_commit=(
                run.strategy_bank.source_method_commit
            ),
            strategy_bank_source_seeds=run.strategy_bank.source_seeds,
            strategy_bank_source_condition_fingerprint=(
                run.strategy_bank.source_condition_fingerprint
            ),
            strategy_bank_source_result_manifest_hash=(
                run.strategy_bank.source_result_manifest_hash
            ),
            strategy_bank_expected_sha256=run.strategy_bank.sha256,
            strategy_bank_path=bank_path,
            strategy_bank_actual_sha256=actual_sha,
            strategy_bank_provenance_path=provenance_path,
            strategy_bank_provenance_sha256=provenance_sha,
            bootstrap_snapshot_path=snapshot_path,
            bootstrap_snapshot_sha256=snapshot_sha,
            bootstrap_snapshot_fingerprint=snapshot_fingerprint,
            bootstrap_formal_binding_sha256=formal_binding_sha,
            bootstrap_initial_file_sha256s=initial_file_hashes,
            target_online_overlay_starts_empty=snapshot_path is not None,
            target_action_exemplars_start_empty=snapshot_path is not None,
            target_manifest_starts_empty=snapshot_path is not None,
            target_failure_cache_starts_empty=snapshot_path is not None,
        ),
        tuple(blockers),
    )


def _validate_bank_provenance(
    path: Path, run: RunSpec, expected_bank_sha256: str
) -> str:
    if run.strategy_bank is None:  # defensive; this is called only for transfer runs
        raise LiveLaunchConfigError("transfer RunSpec has no strategy-bank pin")
    # Local import avoids a live <-> memory package initialization cycle.
    from ..memory.provenance import (
        StrategyBankProvenanceError,
        validate_strategy_bank_provenance,
    )

    try:
        validation = validate_strategy_bank_provenance(
            path,
            strategy_bank_pin=run.strategy_bank,
            expected_bank_sha256=expected_bank_sha256,
        )
    except StrategyBankProvenanceError as exc:
        raise LiveLaunchConfigError(str(exc)) from exc
    # Exact provenance bytes are recorded in the launch configuration even
    # though the bank's semantic content hash remains the protocol pin.
    return validation.file_sha256


def _expected_transfer_formal_binding(
    run: RunSpec,
    *,
    plan_path: Path,
    plan_sha256: str,
) -> dict[str, str]:
    if run.strategy_bank is None:
        raise LiveLaunchConfigError("transfer RunSpec has no strategy-bank pin")
    return {
        "run_id": run.run_id,
        "protocol_id": run.protocol_id,
        "condition_fingerprint": run.condition_fingerprint,
        "strategy_bank_sha256": run.strategy_bank.sha256,
        "source_model": run.strategy_bank.source_backbone,
        "source_provider_route_id": run.strategy_bank.source_provider_route_id,
        "source_provider_route_fingerprint": (
            run.strategy_bank.source_provider_route_fingerprint
        ),
        "target_model": run.target_backbone,
        "provider_route_id": run.provider_route_id,
        "provider_route_fingerprint": run.provider_route_fingerprint,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
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
    }


def _is_pristine_transfer_output(
    output_dir: Path,
    run: RunSpec,
    memory: MemoryExecutionBinding,
) -> bool:
    """Allow only the plan-bound Table-4 bootstrap before dispatch."""

    if run.strategy_bank is None or memory.bootstrap_snapshot_path is None:
        return False
    try:
        if output_dir.is_symlink() or not output_dir.is_dir():
            return False
        children = tuple(sorted(child.name for child in output_dir.iterdir()))
        if children != ("memory",):
            return False
        memory_dir = output_dir / "memory"
        from ..memory.bootstrap import (
            TRANSFER_BOOTSTRAP_INITIAL_FILES,
            TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME,
            validate_transfer_bootstrap_snapshot,
        )

        expected_children = tuple(
            sorted((*TRANSFER_BOOTSTRAP_INITIAL_FILES, TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME))
        )
        actual_children = tuple(sorted(child.name for child in memory_dir.iterdir()))
        if actual_children != expected_children:
            return False
        snapshot = validate_transfer_bootstrap_snapshot(
            memory_dir,
            require_initial_state=True,
            expected_condition=run.memory_condition.value,
            expected_target_model=run.target_backbone,
            expected_seed_bank_sha256=run.strategy_bank.sha256,
        )
        return bool(
            snapshot.path == memory.bootstrap_snapshot_path
            and snapshot.file_sha256 == memory.bootstrap_snapshot_sha256
            and snapshot.snapshot_fingerprint
            == memory.bootstrap_snapshot_fingerprint
            and snapshot.initial_file_sha256s
            == memory.bootstrap_initial_file_sha256s
        )
    except (OSError, TypeError, ValueError):
        return False


def _is_pristine_from_scratch_output(
    output_dir: Path,
    run: RunSpec,
    memory: MemoryExecutionBinding,
) -> bool:
    """Allow only the exact plan-bound empty Table-2/3 bootstrap."""

    if run.strategy_bank is not None or memory.bootstrap_snapshot_path is None:
        return False
    try:
        if output_dir.is_symlink() or not output_dir.is_dir():
            return False
        if tuple(sorted(child.name for child in output_dir.iterdir())) != ("memory",):
            return False
        memory_dir = output_dir / "memory"
        from ..memory.from_scratch import (
            FROM_SCRATCH_BOOTSTRAP_FILENAME,
            FROM_SCRATCH_INITIAL_FILES,
            validate_from_scratch_bootstrap,
        )

        expected_children = tuple(
            sorted((*FROM_SCRATCH_INITIAL_FILES, FROM_SCRATCH_BOOTSTRAP_FILENAME))
        )
        actual_children = tuple(sorted(child.name for child in memory_dir.iterdir()))
        if actual_children != expected_children:
            return False
        snapshot = validate_from_scratch_bootstrap(
            memory_dir,
            run,
            require_initial_state=True,
        )
        return bool(
            snapshot.path == memory.bootstrap_snapshot_path
            and snapshot.file_sha256 == memory.bootstrap_snapshot_sha256
            and snapshot.snapshot_fingerprint
            == memory.bootstrap_snapshot_fingerprint
            and snapshot.formal_binding_sha256
            == memory.bootstrap_formal_binding_sha256
            and snapshot.initial_file_sha256s
            == memory.bootstrap_initial_file_sha256s
        )
    except (OSError, TypeError, ValueError):
        return False


def _is_pristine_prepared_output(
    output_dir: Path,
    run: RunSpec,
    memory: MemoryExecutionBinding,
) -> bool:
    if run.strategy_bank is not None:
        return _is_pristine_transfer_output(output_dir, run, memory)
    return _is_pristine_from_scratch_output(output_dir, run, memory)


def _absolute_nonsymlink_path(
    value: str | Path, label: str, *, must_exist: bool
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise LiveLaunchConfigError(f"{label} path must be absolute")
    if path.is_symlink():
        raise LiveLaunchConfigError(f"{label} path must not be a symlink")
    if must_exist and (not path.exists() or not path.is_file()):
        raise LiveLaunchConfigError(f"{label} path must be an existing regular file")
    return path.resolve(strict=must_exist)


__all__ = [
    "FROM_SCRATCH_BOOTSTRAP_MISSING_CODE",
    "LIVE_LAUNCH_CONFIG_SCHEMA_VERSION",
    "CompiledLiveLaunch",
    "LiveLaunchConfigError",
    "MemoryExecutionBinding",
    "MethodExecutionContract",
    "SceneExecutionBinding",
    "compile_live_launch",
]
