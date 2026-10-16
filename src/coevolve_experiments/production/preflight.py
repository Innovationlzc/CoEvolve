"""No-side-effect compilation of a production Praxis runner handoff.

The existing deterministic TeamIntent engine is intentionally excluded.  This
module validates and expands an already signed subprocess launch into every
runtime injection needed by the frozen Praxis Code-as-Policy loop.  Execution
remains fail-closed until the audited V83 loop consumes all channels and emits
the v3 evidence contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from ..contracts import MemoryCondition, MemoryPolicy, MethodVariant
from ..live.bridge import (
    VerifiedSubprocessLaunchConfig,
    load_subprocess_launch_config,
)
from ..live.capability import REQUIRED_SCENE_INJECTION_CHANNELS
from ..live.contracts import ExecutionCapability, PolicyRepresentation
from ..provider_routes import require_provider_route
from ..reporting.transaction_accounting import (
    ACCEPTED_MANIFEST_PIN,
    FROZEN_PHASE_PLAN_PIN,
    TRANSACTION_ACCOUNTING_LEDGER_PIN,
    TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION,
)
from ..reporting.usage_evidence import FM_USAGE_LEDGER_SCHEMA_VERSION
from ..scenes import compute_asset_composite_hash, directed_route_document, load_scene
from ..serialization import sha256_json
from .contracts import (
    ProductionChannelBinding,
    ProductionInjectionBundle,
    ProductionPreflightReport,
    REQUIRED_PRODUCTION_CHANNELS,
)
from .v83_audit import V83_POST_TABLE_MIGRATION_REQUIREMENTS


PRODUCTION_RUNTIME_ADAPTER_BLOCKER = "PRODUCTION_PRAXIS_RUNTIME_ADAPTER_NOT_IMPLEMENTED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_USD_PRIM_NAME_RE = re.compile(
    r'\b(?:def|over)\s+[A-Za-z_][A-Za-z0-9_]*\s+"([^"]+)"'
)
_PROVIDER_CREDENTIAL_ENV = {
    "provider_a.openai-compatible.chat-completions.v1": "PRAXIS_REMOTE_API_KEY",
    "provider_b.openai-compatible.chat-completions.v1": "PRAXIS_GEMINI_API_KEY",
}


class ProductionPreflightError(ValueError):
    """A launch cannot be converted into a complete immutable handoff."""


def preflight_production_launch(
    launch_config_path: str | Path,
) -> tuple[ProductionPreflightReport, ProductionInjectionBundle]:
    """Validate a launch and return a blocked, side-effect-free handoff.

    This function intentionally never creates the output directory and never
    invokes MuJoCo, SONIC, or a foundation-model endpoint.
    """

    try:
        verified = load_subprocess_launch_config(launch_config_path)
        bundle = compile_production_injection_bundle(verified)
    except (OSError, TypeError, ValueError) as exc:
        raise ProductionPreflightError(str(exc)) from exc
    requirement_codes = tuple(item.code for item in V83_POST_TABLE_MIGRATION_REQUIREMENTS)
    blockers = (PRODUCTION_RUNTIME_ADAPTER_BLOCKER, *requirement_codes)
    report = ProductionPreflightReport(
        available=False,
        run_id=verified.launch.run_spec.run_id,
        launch_config_path=str(verified.path),
        launch_config_sha256=verified.sha256,
        injection_bundle_fingerprint=bundle.fingerprint,
        blocker_codes=blockers,
        requirements=V83_POST_TABLE_MIGRATION_REQUIREMENTS,
    )
    return report, bundle


def compile_production_injection_bundle(
    verified: VerifiedSubprocessLaunchConfig,
    *,
    scene_registry_root: str | Path | None = None,
) -> ProductionInjectionBundle:
    """Derive all production channels from one verified launch."""

    launch = verified.launch
    run = launch.run_spec
    capability = verified.capability
    if launch.execution_capability is not ExecutionCapability.PAPER_FORMAL:
        raise ProductionPreflightError("launch is not paper_formal")
    if launch.method.policy_representation is not PolicyRepresentation.PRAXIS_CODE_AS_POLICY:
        raise ProductionPreflightError("launch is not Praxis Code-as-Policy")
    if launch.blockers or launch.paper_evidence_eligible is not True:
        raise ProductionPreflightError("launch was not activated for paper evidence")
    if tuple(launch.scene.injection_channels) != REQUIRED_SCENE_INJECTION_CHANNELS:
        raise ProductionPreflightError("scene injection channel set is incomplete")

    registry_root = (
        Path(scene_registry_root)
        if scene_registry_root is not None
        else capability.runner_repo / "src" / "coevolve_experiments" / "scenes"
    )
    scene = load_scene(run.scene_id, root=registry_root)
    _validate_scene_binding(launch.scene, scene)
    _validate_method_binding(run.method_variant, launch.method)
    _validate_memory_binding(run, launch)

    route = require_provider_route(
        run.provider_route_id,
        run.provider_route_fingerprint,
    )
    credential_env = _PROVIDER_CREDENTIAL_ENV.get(route.route_id)
    if not credential_env:
        raise ProductionPreflightError("provider route has no reviewed credential binding")
    _validate_output_binding(launch.results_root, launch.output_dir, run.output_relpath)

    asset_source = _resolve_repo_file(
        capability.runner_repo,
        launch.scene.registry_source_path,
        launch.scene.registry_source_sha256,
        "scene source asset",
    )
    if launch.scene.execution_path is None or launch.scene.execution_sha256 is None:
        raise ProductionPreflightError("scene execution asset is not hash-bound")
    execution_asset = _resolve_repo_file(
        capability.runner_repo,
        launch.scene.execution_path,
        launch.scene.execution_sha256,
        "scene execution asset",
    )
    contact_role_registry = _contact_role_registry(execution_asset)

    result_binding = {
        "kind": "coevolve_praxis_live_result_binding",
        "schema_version": capability.result_contract_schema_version,
        "run_id": run.run_id,
        "run_spec_sha256": sha256_json(run),
        "launch_fingerprint": launch.launch_fingerprint,
        "config_fingerprint": verified.config_fingerprint,
        "launch_config_sha256": verified.sha256,
        "plan_sha256": launch.plan_sha256,
        "runner_capability_sha256": capability.manifest_sha256,
        "base_method_commit": capability.base_method_commit,
        "runner_integration_commit": capability.runner_integration_commit,
        "entrypoint_sha256": capability.entrypoint_sha256,
        "full_method_compatibility_fingerprint": (
            capability.full_method_compatibility.fingerprint
        ),
    }
    channel_payloads: Mapping[str, Mapping[str, Any]] = {
        "term1_simulator_asset": {
            "scene_id": scene.scene_id.value,
            "source_path": str(asset_source),
            "source_sha256": scene.asset.source_sha256,
            "execution_path": str(execution_asset),
            "execution_sha256": scene.asset.execution_sha256,
            "execution_composite_sha256": scene.asset.execution_composite_sha256,
            "execution_evidence_fingerprint": (
                launch.scene.execution_evidence_fingerprint
            ),
            "compiled_model_sha256": launch.scene.compiled_model_sha256,
            "contact_role_registry": contact_role_registry,
            "initial_state": scene.initial_state,
            "target": scene.target,
        },
        "host_clearance_geometry": {
            "scene_id": scene.scene_id.value,
            "scene_fingerprint": scene.scene_hash,
            "corridor": scene.corridor,
        },
        "vlm_scene_prior": {
            "scene_id": scene.scene_id.value,
            "label": scene.label,
            "camera": scene.camera,
            "initial_state": scene.initial_state,
            "target": scene.target,
            "mission_topology": scene.route.mission_topology,
            "route_agent_visible": scene.route.agent_visible,
        },
        "route_and_progress_gates": directed_route_document(scene),
        "formal_run_manifest": {
            "run_spec": run,
            "plan_path": str(launch.plan_path),
            "plan_sha256": launch.plan_sha256,
            "launch_config_path": str(verified.path),
            "launch_config_sha256": verified.sha256,
            "launch_fingerprint": launch.launch_fingerprint,
            "result_binding": result_binding,
        },
        "method_gates": {
            "method_variant": run.method_variant.value,
            "contract": launch.method,
            "decision_prompt_fingerprint": run.decision_prompt_fingerprint,
            "shared_prompt_contract_fingerprint": (
                run.shared_prompt_contract_fingerprint
            ),
            "skill_api_fingerprint": run.skill_api_fingerprint,
            "full_method_compatibility_fingerprint": (
                run.full_method_compatibility_fingerprint
            ),
        },
        "memory_gates": {
            "memory_condition": run.memory_condition.value,
            "memory_policy": run.memory_policy.value,
            "strategy_bank": run.strategy_bank,
            "binding": launch.memory,
            "runtime_contract": _memory_runtime_contract(run, launch),
        },
        "provider_route": {
            "route_id": route.route_id,
            "route_fingerprint": route.fingerprint,
            "provider": route.provider,
            "base_url": route.base_url,
            "api_family": route.api_family,
            "target_model": run.target_backbone,
            "credential_env": credential_env,
        },
        "formal_output": {
            "results_root": str(launch.results_root),
            "output_dir": str(launch.output_dir),
            "output_relpath": run.output_relpath,
            "seed": run.seed,
            "physical_budget": run.physical_budget,
            "result_contract_schema_version": (
                capability.result_contract_schema_version
            ),
            "terminal_evidence_schema_version": 3,
            "fm_usage_ledger_schema_version": FM_USAGE_LEDGER_SCHEMA_VERSION,
            "transaction_accounting_ledger_schema_version": (
                TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION
            ),
            "required_result_artifact_pins": {
                "native_memory_audit": (
                    "v83_native_memory_runtime_audit_sha256"
                ),
                "terminal_evidence": "terminal_verifier_evidence_sha256",
                "fm_usage_ledger": "fm_usage_ledger_sha256",
                "transaction_accounting_ledger": (
                    TRANSACTION_ACCOUNTING_LEDGER_PIN
                ),
                "accepted_episode_manifest": ACCEPTED_MANIFEST_PIN,
                "frozen_phase_plan": FROZEN_PHASE_PLAN_PIN,
            },
        },
    }
    channels = tuple(
        ProductionChannelBinding.create(name, channel_payloads[name])
        for name in REQUIRED_PRODUCTION_CHANNELS
    )
    return ProductionInjectionBundle.create(
        run_id=run.run_id,
        launch_config_path=str(verified.path),
        launch_config_sha256=verified.sha256,
        launch_fingerprint=launch.launch_fingerprint,
        channels=channels,
    )


def _validate_scene_binding(binding: Any, scene: Any) -> None:
    if binding.scene_id != scene.scene_id.value:
        raise ProductionPreflightError("launch scene ID differs from registry")
    if binding.scene_fingerprint != scene.scene_hash:
        raise ProductionPreflightError("launch scene fingerprint differs from registry")
    expected = {
        "registry_source_path": scene.asset.source_path,
        "registry_source_sha256": scene.asset.source_sha256,
        "execution_path": scene.asset.execution_path,
        "execution_sha256": scene.asset.execution_sha256,
        "execution_composite_sha256": scene.asset.execution_composite_sha256,
        "target_center_xyz_m": scene.target.center.as_tuple(),
        "route_agent_visible": scene.route.agent_visible,
    }
    for field, wanted in expected.items():
        if getattr(binding, field) != wanted:
            raise ProductionPreflightError(
                f"launch scene {field} differs from registry"
            )
    if (
        not _sha(binding.execution_evidence_fingerprint)
        or not _sha(binding.compiled_model_sha256)
    ):
        raise ProductionPreflightError("scene simulator evidence identity is incomplete")
    if scene.asset.execution_composite_sha256 != compute_asset_composite_hash(scene.asset):
        raise ProductionPreflightError("scene execution composite hash differs")
def _validate_method_binding(variant: MethodVariant, binding: Any) -> None:
    if binding.method_variant is not variant:
        raise ProductionPreflightError("method execution contract differs from RunSpec")
    if not all(
        getattr(binding, field) is True
        for field in (
            "common_physics_verifier",
            "common_checkpoint_rollback",
            "common_terminal_predicate",
            "frozen_sonic_skills",
        )
    ):
        raise ProductionPreflightError("method disabled a common physical invariant")


def _validate_memory_binding(run: Any, launch: Any) -> None:
    binding = launch.memory
    if binding.memory_condition is not run.memory_condition:
        raise ProductionPreflightError("memory condition differs from RunSpec")
    if binding.memory_policy != run.memory_policy.value:
        raise ProductionPreflightError("memory policy differs from RunSpec")
    transfer = run.memory_condition in {
        MemoryCondition.GPT_TO_GPT,
        MemoryCondition.GEMINI_TO_GPT,
    }
    if transfer:
        required = (
            binding.strategy_bank_expected_sha256,
            binding.strategy_bank_actual_sha256,
            binding.strategy_bank_path,
            binding.strategy_bank_provenance_path,
            binding.strategy_bank_provenance_sha256,
            binding.bootstrap_snapshot_path,
            binding.bootstrap_snapshot_sha256,
            binding.bootstrap_snapshot_fingerprint,
            binding.bootstrap_formal_binding_sha256,
            binding.bootstrap_initial_file_sha256s,
        )
        if run.strategy_bank is None or any(item is None for item in required):
            raise ProductionPreflightError("transfer memory binding is incomplete")
        if binding.strategy_bank_actual_sha256 != run.strategy_bank.sha256:
            raise ProductionPreflightError("transfer strategy bank hash differs")
        from ..memory.bootstrap import (
            TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME,
            MemoryBootstrapError,
            validate_transfer_bootstrap_snapshot,
        )

        if binding.bootstrap_snapshot_path.name != TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME:
            raise ProductionPreflightError(
                "transfer bootstrap path has the wrong artifact kind"
            )
        try:
            snapshot = validate_transfer_bootstrap_snapshot(
                binding.bootstrap_snapshot_path.parent,
                require_initial_state=True,
                expected_condition=run.memory_condition.value,
                expected_target_model=run.target_backbone,
                expected_seed_bank_sha256=run.strategy_bank.sha256,
            )
        except (MemoryBootstrapError, OSError, TypeError, ValueError) as exc:
            raise ProductionPreflightError(
                f"transfer bootstrap kind/schema/state is invalid: {exc}"
            ) from exc
        _require_bootstrap_binding_matches(binding, snapshot, "transfer")
    elif run.strategy_bank is not None:
        raise ProductionPreflightError("non-transfer run unexpectedly binds a strategy bank")
    else:
        required = (
            binding.bootstrap_snapshot_path,
            binding.bootstrap_snapshot_sha256,
            binding.bootstrap_snapshot_fingerprint,
            binding.bootstrap_formal_binding_sha256,
            binding.bootstrap_initial_file_sha256s,
        )
        if any(item is None for item in required):
            raise ProductionPreflightError(
                "from-scratch memory bootstrap binding is incomplete"
            )
        from ..memory.from_scratch import (
            FROM_SCRATCH_BOOTSTRAP_FILENAME,
            FromScratchMemoryError,
            validate_from_scratch_bootstrap,
        )

        if binding.bootstrap_snapshot_path.name != FROM_SCRATCH_BOOTSTRAP_FILENAME:
            raise ProductionPreflightError(
                "from-scratch bootstrap path has the wrong artifact kind"
            )
        try:
            snapshot = validate_from_scratch_bootstrap(
                binding.bootstrap_snapshot_path.parent,
                run,
                require_initial_state=True,
                expected_plan_path=launch.plan_path,
                expected_plan_sha256=launch.plan_sha256,
            )
        except (FromScratchMemoryError, OSError, TypeError, ValueError) as exc:
            raise ProductionPreflightError(
                f"from-scratch bootstrap kind/schema/state is invalid: {exc}"
            ) from exc
        _require_bootstrap_binding_matches(binding, snapshot, "from-scratch")
    if not all(
        getattr(binding, field) is True
        for field in (
            "target_online_overlay_starts_empty",
            "target_action_exemplars_start_empty",
            "target_manifest_starts_empty",
            "target_failure_cache_starts_empty",
        )
    ):
        raise ProductionPreflightError("target run-local memory is not fresh")


def _require_bootstrap_binding_matches(binding: Any, snapshot: Any, label: str) -> None:
    expected = {
        "bootstrap_snapshot_path": snapshot.path,
        "bootstrap_snapshot_sha256": snapshot.file_sha256,
        "bootstrap_snapshot_fingerprint": snapshot.snapshot_fingerprint,
        "bootstrap_formal_binding_sha256": snapshot.formal_binding_sha256,
        "bootstrap_initial_file_sha256s": snapshot.initial_file_sha256s,
    }
    for field, wanted in expected.items():
        if getattr(binding, field) != wanted:
            raise ProductionPreflightError(
                f"{label} bootstrap {field} differs from current bytes"
            )


def _memory_runtime_contract(run: Any, launch: Any) -> Mapping[str, Any]:
    """Derive the closed-world memory lifecycle the production Host must use."""

    from ..memory.v83_native import (
        V83_NATIVE_ACTION_LIMIT,
        V83_NATIVE_AUDIT_FILENAME,
        V83_NATIVE_AUDIT_KIND,
        V83_NATIVE_AUDIT_SCHEMA_VERSION,
        V83_NATIVE_STRATEGY_LIMIT,
    )

    transfer = run.memory_condition in {
        MemoryCondition.GPT_TO_GPT,
        MemoryCondition.GEMINI_TO_GPT,
    }
    enabled = run.memory_policy in {
        MemoryPolicy.ONLINE_FROM_EMPTY,
        MemoryPolicy.ONLINE_FROM_PRELOADED_STRATEGIES,
    }
    operation_gates = {
        "strategy_retrieval": enabled,
        "strategy_promotion": enabled,
        "action_exemplar_retrieval": enabled,
        "action_exemplar_promotion": enabled,
    }
    if transfer:
        from ..memory.bootstrap import (
            TRANSFER_BOOTSTRAP_SNAPSHOT_KIND,
            TRANSFER_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION,
        )
        bootstrap_kind = TRANSFER_BOOTSTRAP_SNAPSHOT_KIND
        bootstrap_schema = TRANSFER_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION
    else:
        from ..memory.from_scratch import (
            FROM_SCRATCH_BOOTSTRAP_KIND,
            FROM_SCRATCH_SCHEMA_VERSION,
        )

        bootstrap_kind = FROM_SCRATCH_BOOTSTRAP_KIND
        bootstrap_schema = FROM_SCRATCH_SCHEMA_VERSION

    disabled = run.memory_policy is MemoryPolicy.DISABLED
    mode = "preloaded_seed" if transfer else "disabled" if disabled else "from_empty"
    memory_root = (launch.output_dir / "memory").resolve(strict=False)
    native_root = (memory_root / "v83_native").resolve(strict=False)
    strategy_path = native_root / "strategy_specs.json"
    action_path = native_root / "action_exemplars.json"
    raw_rollout_path = native_root / "raw_rollouts.jsonl"
    accepted_manifest_path = native_root / "accepted_episode_manifest.json"
    runtime_audit_path = native_root / V83_NATIVE_AUDIT_FILENAME
    transfer_seed_path = memory_root / "seed_strategy_bank.json" if transfer else None
    native_run_binding = _native_memory_run_binding(run, launch)
    return {
        "bootstrap": {
            "kind": bootstrap_kind,
            "schema_version": bootstrap_schema,
            "path": str(launch.memory.bootstrap_snapshot_path),
            "sha256": launch.memory.bootstrap_snapshot_sha256,
            "fingerprint": launch.memory.bootstrap_snapshot_fingerprint,
            "role": "immutable_launch_binding_only",
            "is_production_memory_evidence": False,
        },
        "native_facade": {
            "adapter": "V83NativeMemoryBridge",
            "mode": mode,
            "root": str(native_root),
            "strategy_store_path": str(strategy_path),
            "action_exemplar_store_path": str(action_path),
            "raw_rollout_path": str(raw_rollout_path),
            "accepted_episode_manifest_path": str(accepted_manifest_path),
            "transfer_seed_path": (
                str(transfer_seed_path) if transfer_seed_path is not None else None
            ),
            "strategy_retrieval_limit": V83_NATIVE_STRATEGY_LIMIT,
            "action_exemplar_retrieval_limit": V83_NATIVE_ACTION_LIMIT,
            "run_binding": native_run_binding,
            "run_binding_sha256": sha256_json(native_run_binding),
        },
        "cross_run_retrieval": transfer,
        "operation_gates": operation_gates,
        "initial_state": {
            "native_strategy_store_empty": True,
            "native_action_exemplars_empty": True,
            "accepted_episode_manifest_empty": True,
            "checkpoint_local_failure_cache_empty": True,
            "initial_checkpoint": "accepted_step_00",
        },
        "ungated_v83_lifecycle": {
            "raw_rollout_persistence": True,
            "accepted_episode_manifest": True,
            "checkpoint_and_rollback": True,
            "exact_rejected_transaction_blocker": True,
        },
        "runtime_audit": {
            "kind": V83_NATIVE_AUDIT_KIND,
            "schema_version": V83_NATIVE_AUDIT_SCHEMA_VERSION,
            "path": str(runtime_audit_path),
            "path_source": "output_dir/memory/v83_native/fixed_filename",
            "finalize_on_task_terminal": True,
            "mark_incomplete_on_infrastructure_terminal": True,
        },
        "final_state": {
            "strategy_store_must_be_empty": disabled,
            "action_exemplar_store_must_be_empty": disabled,
            "operation_counts_must_be_zero": disabled,
            "memory_writer_calls_must_be_zero": disabled,
            "transfer_seed_bytes_must_be_unchanged": transfer,
        },
    }


def _native_memory_run_binding(run: Any, launch: Any) -> Mapping[str, Any]:
    """Return the exact plan-derived identity signed by the native audit."""

    strategy_bank_sha256 = ""
    if run.strategy_bank is not None:
        strategy_bank_sha256 = str(run.strategy_bank.sha256)
    return {
        "run_id": run.run_id,
        "table_id": run.table_id.value,
        "seed": run.seed,
        "protocol_id": run.protocol_id,
        "base_method_commit": run.base_method_commit,
        "condition_fingerprint": run.condition_fingerprint,
        "invariant_fingerprint": run.invariant_fingerprint,
        "scene_fingerprint": run.scene_fingerprint,
        "terminal_fingerprint": run.terminal_fingerprint,
        "memory_condition": run.memory_condition.value,
        "memory_policy": run.memory_policy.value,
        "target_backbone": run.target_backbone,
        "provider_route_id": run.provider_route_id,
        "provider_route_fingerprint": run.provider_route_fingerprint,
        "strategy_bank_sha256": strategy_bank_sha256,
        "plan_path": str(launch.plan_path),
        "plan_sha256": launch.plan_sha256,
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


def _validate_output_binding(results_root: Path, output_dir: Path, relpath: str) -> None:
    root = Path(results_root)
    output = Path(output_dir)
    if not root.is_absolute() or not output.is_absolute():
        raise ProductionPreflightError("formal output paths must be absolute")
    expected = (root / PurePosixPath(relpath)).resolve(strict=False)
    if output.resolve(strict=False) != expected:
        raise ProductionPreflightError("formal output path differs from RunSpec")
    try:
        output.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ProductionPreflightError("formal output path escapes results root") from exc


def _resolve_repo_file(repo: Path, relpath: str, expected_sha256: str, label: str) -> Path:
    relative = PurePosixPath(relpath)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ProductionPreflightError(f"{label} path is unsafe")
    candidate = repo / relative.as_posix()
    chain = (candidate, *candidate.parents[: len(relative.parts) - 1])
    if any(path.is_symlink() for path in chain) or not candidate.is_file():
        raise ProductionPreflightError(f"{label} is missing or symlinked")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repo.resolve(strict=True))
    except ValueError as exc:
        raise ProductionPreflightError(f"{label} escapes runner checkout") from exc
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ProductionPreflightError(f"{label} byte hash differs")
    return resolved


def _contact_role_registry(execution_asset: Path) -> Mapping[str, Any]:
    """Derive reviewed contact-role names from the hash-pinned USD asset.

    The terminal recorder must not guess that every geom containing ``wall``
    is a prohibited wall.  Instead, preflight extracts the exact wall prim
    families from the already hash-verified execution asset and injects that
    closed set into Term1.  MuJoCo conversion may suffix a prim name for a
    generated geom, so the runtime accepts either the exact name or a name
    beginning with ``<prim>_``; it accepts no unrelated substring match.
    """

    try:
        text = execution_asset.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProductionPreflightError(
            "scene execution asset is not readable USDA text"
        ) from exc
    names = tuple(
        sorted(
            {
                name
                for name in _USD_PRIM_NAME_RE.findall(text)
                if name.startswith("wall_") or name.startswith("topology_wall_")
            }
        )
    )
    if not names:
        raise ProductionPreflightError(
            "scene execution asset exposes no reviewed wall prim names"
        )
    return {
        "schema_version": 1,
        "source_execution_sha256": hashlib.sha256(
            execution_asset.read_bytes()
        ).hexdigest(),
        "wall_prim_names": list(names),
        "target_geom_names": ["target_marker"],
        "floor_geom_names": ["floor", "manual_floor"],
    }


def _sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


__all__ = [
    "PRODUCTION_RUNTIME_ADAPTER_BLOCKER",
    "ProductionPreflightError",
    "compile_production_injection_bundle",
    "preflight_production_launch",
]
