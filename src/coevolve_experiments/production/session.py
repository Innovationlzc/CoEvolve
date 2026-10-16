"""Read-only binding seam for a future production V83 driver.

This module turns the nine independently fingerprinted production channels
into one validated object.  Binding is deliberately side-effect free: it does
not create the V83 memory facade, read a provider credential, import MuJoCo,
call a model, or authorize dispatch.  The still-missing physical/role loop
must explicitly open the memory facade with the frozen V83 store factories.

The seam is intentionally stricter than a collection of convenience getters.
It reconstructs the typed ``RunSpec`` and method contract, checks all prompt
and memory-gate identities against each other, revalidates the immutable
memory bootstrap, and derives the terminal trace identity from the same
consumed values later retained for the evidence writer.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..contracts import (
    MemoryCondition,
    MemoryPolicy,
    MethodVariant,
    RunSpec,
    SceneId,
    StrategyBankPin,
    TableId,
)
from ..decision_prompts import (
    FROZEN_V83_FULL_DECISION_PROMPT_SHA256,
    ProductionDecisionPromptArtifact,
    ProductionPromptRegistry,
    PromptMemoryGates,
    load_production_prompt_registry,
)
from ..live.capability import RUNNER_RESULT_CONTRACT_SCHEMA_VERSION
from ..live.contracts import PolicyRepresentation
from ..live.launch import MethodExecutionContract
from ..memory import (
    ActionStoreFactory,
    StrategyStoreFactory,
    V83MemoryOperationGates,
    V83NativeMemoryBridge,
    V83_NATIVE_ACTION_LIMIT,
    V83_NATIVE_AUDIT_FILENAME,
    V83_NATIVE_STRATEGY_LIMIT,
    select_strategy_records,
    validate_from_scratch_bootstrap,
    validate_transfer_bootstrap_snapshot,
)
from ..memory.v83_native import StrategyRenderer, StrategySelector
from ..provider_routes import require_provider_route
from ..reporting.terminal_evidence import TERMINAL_EVIDENCE_SCHEMA_VERSION
from ..reporting.transaction_accounting import (
    TRANSACTION_ACCOUNTING_LEDGER_PIN,
    TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION,
    ACCEPTED_MANIFEST_PIN,
    FROZEN_PHASE_PLAN_PIN,
)
from ..reporting.usage_evidence import FM_USAGE_LEDGER_SCHEMA_VERSION
from ..scenes import directed_route_document, load_scene
from ..serialization import sha256_json, to_primitive
from .contracts import (
    ProductionChannelBinding,
    ProductionInjectionBundle,
    REQUIRED_PRODUCTION_CHANNELS,
)
from .injections import ProductionInjectionConsumptionGate
from .runtime_instrumentation import RuntimeTraceBinding


_RUN_FIELDS = frozenset(item.name for item in fields(RunSpec))
_STRATEGY_BANK_FIELDS = frozenset(item.name for item in fields(StrategyBankPin))
_METHOD_CONTRACT_FIELDS = frozenset(
    item.name for item in fields(MethodExecutionContract)
)
_FORMAL_MANIFEST_FIELDS = frozenset(
    {
        "run_spec",
        "plan_path",
        "plan_sha256",
        "launch_config_path",
        "launch_config_sha256",
        "launch_fingerprint",
        "result_binding",
    }
)
_METHOD_GATE_FIELDS = frozenset(
    {
        "method_variant",
        "contract",
        "decision_prompt_fingerprint",
        "shared_prompt_contract_fingerprint",
        "skill_api_fingerprint",
        "full_method_compatibility_fingerprint",
    }
)
_MEMORY_GATE_FIELDS = frozenset(
    {
        "memory_condition",
        "memory_policy",
        "strategy_bank",
        "binding",
        "runtime_contract",
    }
)
_MEMORY_RUNTIME_FIELDS = frozenset(
    {
        "bootstrap",
        "native_facade",
        "cross_run_retrieval",
        "operation_gates",
        "initial_state",
        "ungated_v83_lifecycle",
        "runtime_audit",
        "final_state",
    }
)
_NATIVE_FACADE_FIELDS = frozenset(
    {
        "adapter",
        "mode",
        "root",
        "strategy_store_path",
        "action_exemplar_store_path",
        "raw_rollout_path",
        "accepted_episode_manifest_path",
        "transfer_seed_path",
        "strategy_retrieval_limit",
        "action_exemplar_retrieval_limit",
        "run_binding",
        "run_binding_sha256",
    }
)
_NATIVE_RUN_BINDING_FIELDS = frozenset(
    {
        "run_id",
        "table_id",
        "seed",
        "protocol_id",
        "base_method_commit",
        "condition_fingerprint",
        "invariant_fingerprint",
        "scene_fingerprint",
        "terminal_fingerprint",
        "memory_condition",
        "memory_policy",
        "target_backbone",
        "provider_route_id",
        "provider_route_fingerprint",
        "strategy_bank_sha256",
        "plan_path",
        "plan_sha256",
        "observation_contract_fingerprint",
        "task_instruction_fingerprint",
        "shared_prompt_contract_fingerprint",
        "decision_prompt_fingerprint",
        "skill_api_fingerprint",
        "full_method_compatibility_fingerprint",
    }
)
_PROVIDER_FIELDS = frozenset(
    {
        "route_id",
        "route_fingerprint",
        "provider",
        "base_url",
        "api_family",
        "target_model",
        "credential_env",
    }
)
_FORMAL_OUTPUT_FIELDS = frozenset(
    {
        "results_root",
        "output_dir",
        "output_relpath",
        "seed",
        "physical_budget",
        "result_contract_schema_version",
        "terminal_evidence_schema_version",
        "fm_usage_ledger_schema_version",
        "transaction_accounting_ledger_schema_version",
        "required_result_artifact_pins",
    }
)
_RESULT_BINDING_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "run_id",
        "run_spec_sha256",
        "launch_fingerprint",
        "config_fingerprint",
        "launch_config_sha256",
        "plan_sha256",
        "runner_capability_sha256",
        "base_method_commit",
        "runner_integration_commit",
        "entrypoint_sha256",
        "full_method_compatibility_fingerprint",
    }
)
_OPERATION_GATE_FIELDS = frozenset(
    {
        "strategy_retrieval",
        "strategy_promotion",
        "action_exemplar_retrieval",
        "action_exemplar_promotion",
    }
)


class ProductionSessionBindingError(ValueError):
    """The consumed production channels cannot describe one safe session."""


@dataclass(frozen=True, slots=True)
class ConsumedProductionChannels:
    """The recursively read-only snapshots returned by the existing gate."""

    term1_simulator_asset: Mapping[str, Any]
    host_clearance_geometry: Mapping[str, Any]
    vlm_scene_prior: Mapping[str, Any]
    route_and_progress_gates: Mapping[str, Any]
    formal_run_manifest: Mapping[str, Any]
    method_gates: Mapping[str, Any]
    memory_gates: Mapping[str, Any]
    provider_route: Mapping[str, Any]
    formal_output: Mapping[str, Any]

    def channel(self, name: str) -> Mapping[str, Any]:
        if name not in REQUIRED_PRODUCTION_CHANNELS:
            raise KeyError(name)
        return getattr(self, name)


@dataclass(frozen=True, slots=True)
class ProductionMethodPromptBinding:
    """Method prompt bytes and identities already checked against the run."""

    method_variant: MethodVariant
    decision_prompt_fingerprint: str
    shared_prompt_contract_fingerprint: str
    frozen_v83_prompt_sha256: str
    memory_gates: PromptMemoryGates
    artifact: ProductionDecisionPromptArtifact | None
    registry_path: Path | None
    registry_sha256: str | None
    registry_bytes: bytes
    manifest_path: Path | None
    manifest_sha256: str | None
    manifest_bytes: bytes

    @property
    def audit_identity(self) -> Mapping[str, Any]:
        """JSON-safe prompt identity for a future driver-owned audit."""

        return MappingProxyType(
            {
                "method_variant": self.method_variant.value,
                "decision_prompt_fingerprint": self.decision_prompt_fingerprint,
                "shared_prompt_contract_fingerprint": (
                    self.shared_prompt_contract_fingerprint
                ),
                "frozen_v83_prompt_sha256": self.frozen_v83_prompt_sha256,
                "source": (
                    "frozen_v83_prompts_py"
                    if self.artifact is None
                    else "production_prompt_registry"
                ),
                "registry_path": (
                    str(self.registry_path) if self.registry_path is not None else None
                ),
                "registry_sha256": self.registry_sha256,
                "manifest_path": (
                    str(self.manifest_path) if self.manifest_path is not None else None
                ),
                "manifest_sha256": self.manifest_sha256,
                "role_prompt_sha256": {
                    role.role: role.prompt_sha256
                    for role in (() if self.artifact is None else self.artifact.roles)
                },
            }
        )


@dataclass(frozen=True, slots=True)
class V83NativeMemoryBootstrapBinding:
    """Validated constructor inputs for ``V83NativeMemoryBridge``.

    Creating this record is read-only.  ``open`` is the explicit transition
    that rechecks the bootstrap and then lets the existing bridge create its
    runtime audit.  Both native factories are required keyword arguments so a
    smoke adapter cannot silently stand in for the frozen V83 stores.
    """

    run_spec: RunSpec
    plan_path: Path
    plan_sha256: str
    bootstrap_path: Path
    bootstrap_kind: str
    bootstrap_schema_version: int
    bootstrap_sha256: str
    bootstrap_fingerprint: str
    mode: str
    gates: V83MemoryOperationGates
    native_root: Path
    strategy_store_path: Path
    action_exemplar_store_path: Path
    raw_rollout_path: Path
    accepted_episode_manifest_path: Path
    audit_path: Path
    transfer_seed_path: Path | None
    run_binding: Mapping[str, Any]

    def open(
        self,
        *,
        strategy_store_factory: StrategyStoreFactory,
        action_store_factory: ActionStoreFactory,
        native_strategy_renderer: StrategyRenderer | None = None,
        strategy_selector: StrategySelector = select_strategy_records,
    ) -> V83NativeMemoryBridge:
        """Revalidate initial bytes, then open the real V83 memory facade."""

        if not callable(strategy_store_factory) or not callable(action_store_factory):
            raise ProductionSessionBindingError(
                "native V83 strategy/action store factories must be callable"
            )
        self._revalidate_initial_state()
        kwargs: dict[str, Any] = {
            "strategy_path": self.strategy_store_path,
            "action_exemplars_path": self.action_exemplar_store_path,
            "audit_path": self.audit_path,
            "mode": self.mode,
            "gates": self.gates,
            "strategy_store_factory": strategy_store_factory,
            "action_store_factory": action_store_factory,
            "run_binding": self.run_binding,
            "transfer_seed_path": self.transfer_seed_path,
            "strategy_selector": strategy_selector,
        }
        if native_strategy_renderer is not None:
            kwargs["native_strategy_renderer"] = native_strategy_renderer
        return V83NativeMemoryBridge(**kwargs)

    def _revalidate_initial_state(self) -> None:
        _require_fresh_native_root(self.native_root)
        if self.transfer_seed_path is None:
            snapshot = validate_from_scratch_bootstrap(
                self.bootstrap_path.parent,
                self.run_spec,
                require_initial_state=True,
                expected_plan_path=self.plan_path,
                expected_plan_sha256=self.plan_sha256,
            )
            seed_sha = None
        else:
            if self.run_spec.strategy_bank is None:
                raise ProductionSessionBindingError(
                    "transfer memory lost its RunSpec strategy-bank pin"
                )
            snapshot = validate_transfer_bootstrap_snapshot(
                self.bootstrap_path.parent,
                require_initial_state=True,
                expected_condition=self.run_spec.memory_condition.value,
                expected_target_model=self.run_spec.target_backbone,
                expected_seed_bank_sha256=self.run_spec.strategy_bank.sha256,
            )
            seed_sha = snapshot.seed_bank_sha256
        checks = {
            "bootstrap path": (snapshot.path, self.bootstrap_path),
            "bootstrap file SHA-256": (snapshot.file_sha256, self.bootstrap_sha256),
            "bootstrap fingerprint": (
                snapshot.snapshot_fingerprint,
                self.bootstrap_fingerprint,
            ),
        }
        mismatches = tuple(name for name, pair in checks.items() if pair[0] != pair[1])
        if mismatches:
            raise ProductionSessionBindingError(
                "memory bootstrap changed after session binding: "
                + ", ".join(mismatches)
            )
        if seed_sha is not None and seed_sha != self.run_spec.strategy_bank.sha256:
            raise ProductionSessionBindingError(
                "transfer seed differs from the RunSpec strategy-bank pin"
            )


@dataclass(frozen=True, slots=True)
class ProductionSessionBinding:
    """Single read-only handoff to the future physical/role driver.

    This object is *not* an execution-ready receipt and does not remove any
    production preflight blocker.  It only proves that configuration parsing
    and identity alignment are complete.
    """

    run_spec: RunSpec
    method_contract: MethodExecutionContract
    prompt: ProductionMethodPromptBinding
    memory: V83NativeMemoryBootstrapBinding
    runtime_trace: RuntimeTraceBinding
    channels: ConsumedProductionChannels
    evidence_bundle: ProductionInjectionBundle

    @property
    def execution_ready(self) -> bool:
        """Always false until a separately reviewed physical/role loop exists."""

        return False


def bind_production_session(
    bundle: ProductionInjectionBundle,
    *,
    simulator_version: str,
    prompt_registry_root: str | Path | None = None,
) -> ProductionSessionBinding:
    """Consume all nine channels and build one side-effect-free session seam."""

    try:
        gate = ProductionInjectionConsumptionGate(bundle)
        consumed = {
            name: gate.consume(name) for name in REQUIRED_PRODUCTION_CHANNELS
        }
        gate.assert_ready_for_dispatch()
        evidence_bundle = _readonly_bundle_copy(bundle, consumed)
        channels = ConsumedProductionChannels(
            **{name: evidence_bundle.channel(name).payload for name in REQUIRED_PRODUCTION_CHANNELS}
        )

        manifest = _exact_mapping(
            channels.formal_run_manifest,
            _FORMAL_MANIFEST_FIELDS,
            "formal run manifest",
        )
        run = _parse_run_spec(manifest.get("run_spec"))
        _validate_bundle_and_manifest_identity(bundle, manifest, run)
        method = _parse_method_contract(
            _exact_mapping(
                channels.method_gates,
                _METHOD_GATE_FIELDS,
                "method gates",
            ).get("contract")
        )
        method_payload = _mapping(channels.method_gates, "method gates")
        memory_payload = _exact_mapping(
            channels.memory_gates,
            _MEMORY_GATE_FIELDS,
            "memory gates",
        )
        _validate_method_run_identity(run, method, method_payload)
        prompt = _bind_prompt(
            run,
            method,
            method_payload,
            memory_payload,
            prompt_registry_root=prompt_registry_root,
        )
        output_dir = _validate_output_and_provider(
            run,
            channels.formal_output,
            channels.provider_route,
        )
        _validate_scene_channels(run, channels)
        memory = _bind_memory(
            run,
            manifest,
            method,
            memory_payload,
            output_dir=output_dir,
        )
        runtime_trace = RuntimeTraceBinding.from_consumed_channels(
            channels.term1_simulator_asset,
            channels.formal_run_manifest,
            simulator_version=simulator_version,
        )
        return ProductionSessionBinding(
            run_spec=run,
            method_contract=method,
            prompt=prompt,
            memory=memory,
            runtime_trace=runtime_trace,
            channels=channels,
            evidence_bundle=evidence_bundle,
        )
    except ProductionSessionBindingError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ProductionSessionBindingError(str(exc)) from exc


def _readonly_bundle_copy(
    source: ProductionInjectionBundle,
    consumed: Mapping[str, Mapping[str, Any]],
) -> ProductionInjectionBundle:
    bindings = []
    for name in REQUIRED_PRODUCTION_CHANNELS:
        created = ProductionChannelBinding.create(name, consumed[name])
        bindings.append(replace(created, payload=consumed[name]))
    copied = ProductionInjectionBundle.create(
        run_id=source.run_id,
        launch_config_path=source.launch_config_path,
        launch_config_sha256=source.launch_config_sha256,
        launch_fingerprint=source.launch_fingerprint,
        channels=tuple(bindings),
    )
    if copied.fingerprint != source.fingerprint:
        raise ProductionSessionBindingError(
            "detached production bundle fingerprint differs from its source"
        )
    return copied


def _parse_run_spec(value: object) -> RunSpec:
    payload = _exact_mapping(value, _RUN_FIELDS, "formal RunSpec")
    strategy_value = payload.get("strategy_bank")
    strategy: StrategyBankPin | None
    if strategy_value is None:
        strategy = None
    else:
        strategy_payload = _exact_mapping(
            strategy_value,
            _STRATEGY_BANK_FIELDS,
            "RunSpec strategy bank",
        )
        strategy = StrategyBankPin(
            **{
                **strategy_payload,
                "source_seeds": _integer_tuple(
                    strategy_payload.get("source_seeds"),
                    "strategy-bank source seeds",
                ),
            }
        )
    try:
        run = RunSpec(
            **{
                **payload,
                "table_id": TableId(payload["table_id"]),
                "method_variant": MethodVariant(payload["method_variant"]),
                "scene_id": SceneId(payload["scene_id"]),
                "memory_condition": MemoryCondition(payload["memory_condition"]),
                "memory_policy": MemoryPolicy(payload["memory_policy"]),
                "strategy_bank": strategy,
            }
        )
    except (TypeError, ValueError) as exc:
        raise ProductionSessionBindingError(f"formal RunSpec is invalid: {exc}") from exc
    if to_primitive(run) != dict(payload):
        raise ProductionSessionBindingError("formal RunSpec is not canonical")
    return run


def _parse_method_contract(value: object) -> MethodExecutionContract:
    payload = _exact_mapping(
        value,
        _METHOD_CONTRACT_FIELDS,
        "method execution contract",
    )
    try:
        contract = MethodExecutionContract(
            **{
                **payload,
                "method_variant": MethodVariant(payload["method_variant"]),
                "policy_representation": PolicyRepresentation(
                    payload["policy_representation"]
                ),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ProductionSessionBindingError(
            f"method execution contract is invalid: {exc}"
        ) from exc
    if to_primitive(contract) != dict(payload):
        raise ProductionSessionBindingError(
            "method execution contract is not canonical"
        )
    boolean_fields = (
        "hierarchical_evolution",
        "strategy_retrieval",
        "strategy_promotion",
        "action_exemplar_retrieval",
        "action_exemplar_promotion",
        "common_physics_verifier",
        "common_checkpoint_rollback",
        "common_terminal_predicate",
        "frozen_sonic_skills",
    )
    if not all(type(getattr(contract, name)) is bool for name in boolean_fields):
        raise ProductionSessionBindingError(
            "method execution contract gates must be booleans"
        )
    if contract.policy_representation is not PolicyRepresentation.PRAXIS_CODE_AS_POLICY:
        raise ProductionSessionBindingError(
            "method execution contract is not Praxis Code-as-Policy"
        )
    if not all(
        getattr(contract, name)
        for name in (
            "common_physics_verifier",
            "common_checkpoint_rollback",
            "common_terminal_predicate",
            "frozen_sonic_skills",
        )
    ):
        raise ProductionSessionBindingError(
            "method execution contract disabled a common physical invariant"
        )
    return contract


def _validate_bundle_and_manifest_identity(
    bundle: ProductionInjectionBundle,
    manifest: Mapping[str, Any],
    run: RunSpec,
) -> None:
    if bundle.run_id != run.run_id:
        raise ProductionSessionBindingError("bundle run_id differs from RunSpec")
    expected = {
        "launch_config_path": bundle.launch_config_path,
        "launch_config_sha256": bundle.launch_config_sha256,
        "launch_fingerprint": bundle.launch_fingerprint,
    }
    for name, wanted in expected.items():
        if manifest.get(name) != wanted:
            raise ProductionSessionBindingError(
                f"formal manifest {name} differs from the production bundle"
            )
    launch_path = _absolute_file(bundle.launch_config_path, "launch config")
    _require_file_sha(launch_path, bundle.launch_config_sha256, "launch config")
    plan_path = _absolute_file(manifest.get("plan_path"), "suite plan")
    plan_sha = _sha_value(manifest.get("plan_sha256"), "suite plan SHA-256")
    _require_file_sha(plan_path, plan_sha, "suite plan")
    result = _exact_mapping(
        manifest.get("result_binding"),
        _RESULT_BINDING_FIELDS,
        "formal result binding",
    )
    checks = {
        "kind": "coevolve_praxis_live_result_binding",
        "run_id": run.run_id,
        "run_spec_sha256": sha256_json(run),
        "launch_fingerprint": bundle.launch_fingerprint,
        "launch_config_sha256": bundle.launch_config_sha256,
        "plan_sha256": plan_sha,
        "base_method_commit": run.base_method_commit,
        "full_method_compatibility_fingerprint": (
            run.full_method_compatibility_fingerprint
        ),
    }
    for name, wanted in checks.items():
        if result.get(name) != wanted:
            raise ProductionSessionBindingError(
                f"formal result binding {name} differs from the consumed run"
            )


def _validate_method_run_identity(
    run: RunSpec,
    contract: MethodExecutionContract,
    method_payload: Mapping[str, Any],
) -> None:
    checks = {
        "method_variant": run.method_variant.value,
        "decision_prompt_fingerprint": run.decision_prompt_fingerprint,
        "shared_prompt_contract_fingerprint": (
            run.shared_prompt_contract_fingerprint
        ),
        "skill_api_fingerprint": run.skill_api_fingerprint,
        "full_method_compatibility_fingerprint": (
            run.full_method_compatibility_fingerprint
        ),
    }
    for name, wanted in checks.items():
        if method_payload.get(name) != wanted:
            raise ProductionSessionBindingError(
                f"method gate {name} differs from RunSpec"
            )
    if contract.method_variant is not run.method_variant:
        raise ProductionSessionBindingError(
            "method execution contract variant differs from RunSpec"
        )


def _bind_prompt(
    run: RunSpec,
    contract: MethodExecutionContract,
    method_payload: Mapping[str, Any],
    memory_payload: Mapping[str, Any],
    *,
    prompt_registry_root: str | Path | None,
) -> ProductionMethodPromptBinding:
    del method_payload
    operation_gates = _operation_gates(memory_payload)
    expected = PromptMemoryGates(
        strategy_retrieval=contract.strategy_retrieval,
        strategy_promotion=contract.strategy_promotion,
        action_exemplar_retrieval=contract.action_exemplar_retrieval,
        action_exemplar_promotion=contract.action_exemplar_promotion,
        memory_writer_call=contract.strategy_promotion,
    )
    if operation_gates != V83MemoryOperationGates(
        expected.strategy_retrieval,
        expected.strategy_promotion,
        expected.action_exemplar_retrieval,
        expected.action_exemplar_promotion,
    ):
        raise ProductionSessionBindingError(
            "method gates and native memory operation gates differ"
        )

    if run.method_variant is MethodVariant.COEVOLE_FULL:
        if run.decision_prompt_fingerprint != FROZEN_V83_FULL_DECISION_PROMPT_SHA256:
            raise ProductionSessionBindingError(
                "Full CoEvolve decision prompt is not the frozen V83 prompts.py hash"
            )
        if expected != PromptMemoryGates(True, True, True, True, True):
            raise ProductionSessionBindingError(
                "Full CoEvolve method gates differ from the frozen V83 contract"
            )
        if contract.decision_structure != "hierarchical_evolution" or not contract.hierarchical_evolution:
            raise ProductionSessionBindingError(
                "Full CoEvolve hierarchy differs from the frozen method contract"
            )
        return ProductionMethodPromptBinding(
            method_variant=run.method_variant,
            decision_prompt_fingerprint=run.decision_prompt_fingerprint,
            shared_prompt_contract_fingerprint=run.shared_prompt_contract_fingerprint,
            frozen_v83_prompt_sha256=FROZEN_V83_FULL_DECISION_PROMPT_SHA256,
            memory_gates=expected,
            artifact=None,
            registry_path=None,
            registry_sha256=None,
            registry_bytes=b"",
            manifest_path=None,
            manifest_sha256=None,
            manifest_bytes=b"",
        )

    registry = load_production_prompt_registry(prompt_registry_root)
    if not isinstance(registry, ProductionPromptRegistry):
        raise ProductionSessionBindingError("prompt registry loader returned wrong type")
    artifact = registry.artifact_for(run.method_variant)
    if artifact.decision_prompt_fingerprint != run.decision_prompt_fingerprint:
        raise ProductionSessionBindingError(
            "method decision prompt artifact fingerprint differs from RunSpec"
        )
    if artifact.memory_gates != expected:
        raise ProductionSessionBindingError(
            "method decision prompt artifact memory gates differ from runtime gates"
        )
    structure_checks = {
        "decision_structure": (artifact.decision_structure, contract.decision_structure),
        "hierarchical_evolution": (
            artifact.hierarchical_evolution,
            contract.hierarchical_evolution,
        ),
    }
    mismatches = tuple(
        name for name, pair in structure_checks.items() if pair[0] != pair[1]
    )
    if mismatches:
        raise ProductionSessionBindingError(
            "method prompt artifact differs from method contract in: "
            + ", ".join(mismatches)
        )
    registry_bytes = registry.registry_path.read_bytes()
    manifest_bytes = artifact.manifest_path.read_bytes()
    if sha256(registry_bytes).hexdigest() != registry.registry_sha256:
        raise ProductionSessionBindingError("prompt registry bytes changed after load")
    if sha256(manifest_bytes).hexdigest() != artifact.manifest_sha256:
        raise ProductionSessionBindingError("prompt manifest bytes changed after load")
    return ProductionMethodPromptBinding(
        method_variant=run.method_variant,
        decision_prompt_fingerprint=run.decision_prompt_fingerprint,
        shared_prompt_contract_fingerprint=run.shared_prompt_contract_fingerprint,
        frozen_v83_prompt_sha256=FROZEN_V83_FULL_DECISION_PROMPT_SHA256,
        memory_gates=artifact.memory_gates,
        artifact=artifact,
        registry_path=registry.registry_path,
        registry_sha256=registry.registry_sha256,
        registry_bytes=registry_bytes,
        manifest_path=artifact.manifest_path,
        manifest_sha256=artifact.manifest_sha256,
        manifest_bytes=manifest_bytes,
    )


def _bind_memory(
    run: RunSpec,
    manifest: Mapping[str, Any],
    method: MethodExecutionContract,
    memory_payload: Mapping[str, Any],
    *,
    output_dir: Path,
) -> V83NativeMemoryBootstrapBinding:
    if memory_payload.get("memory_condition") != run.memory_condition.value:
        raise ProductionSessionBindingError("memory condition differs from RunSpec")
    if memory_payload.get("memory_policy") != run.memory_policy.value:
        raise ProductionSessionBindingError("memory policy differs from RunSpec")
    if memory_payload.get("strategy_bank") != to_primitive(run.strategy_bank):
        raise ProductionSessionBindingError("memory strategy-bank pin differs from RunSpec")
    runtime = _exact_mapping(
        memory_payload.get("runtime_contract"),
        _MEMORY_RUNTIME_FIELDS,
        "memory runtime contract",
    )
    facade = _exact_mapping(
        runtime.get("native_facade"),
        _NATIVE_FACADE_FIELDS,
        "native memory facade",
    )
    if facade.get("adapter") != "V83NativeMemoryBridge":
        raise ProductionSessionBindingError("native memory adapter is not V83NativeMemoryBridge")
    if facade.get("strategy_retrieval_limit") != V83_NATIVE_STRATEGY_LIMIT:
        raise ProductionSessionBindingError("native strategy retrieval limit differs")
    if facade.get("action_exemplar_retrieval_limit") != V83_NATIVE_ACTION_LIMIT:
        raise ProductionSessionBindingError("native action retrieval limit differs")

    transfer = run.strategy_bank is not None
    expected_mode = (
        "preloaded_seed"
        if transfer
        else "disabled"
        if run.memory_policy is MemoryPolicy.DISABLED
        else "from_empty"
    )
    if facade.get("mode") != expected_mode:
        raise ProductionSessionBindingError("native memory mode differs from RunSpec")
    if runtime.get("cross_run_retrieval") is not transfer:
        raise ProductionSessionBindingError("native cross-run retrieval gate differs")
    gates = _operation_gates(memory_payload)
    method_gates = V83MemoryOperationGates(
        method.strategy_retrieval,
        method.strategy_promotion,
        method.action_exemplar_retrieval,
        method.action_exemplar_promotion,
    )
    if gates != method_gates:
        raise ProductionSessionBindingError("method and memory operation gates differ")
    if run.memory_policy is MemoryPolicy.DISABLED:
        if not gates.all_disabled:
            raise ProductionSessionBindingError("disabled memory has an open operation gate")
    elif gates != V83MemoryOperationGates.enabled():
        raise ProductionSessionBindingError("online memory does not enable all four gates")

    plan_path = _absolute_file(manifest.get("plan_path"), "suite plan")
    plan_sha = _sha_value(manifest.get("plan_sha256"), "suite plan SHA-256")
    run_binding = _exact_mapping(
        facade.get("run_binding"),
        _NATIVE_RUN_BINDING_FIELDS,
        "native memory run binding",
    )
    expected_binding = _native_run_binding(run, plan_path, plan_sha)
    if dict(run_binding) != expected_binding:
        raise ProductionSessionBindingError("native memory run binding differs from RunSpec")
    if facade.get("run_binding_sha256") != sha256_json(expected_binding):
        raise ProductionSessionBindingError("native memory run-binding hash differs")

    expected_root = output_dir / "memory" / "v83_native"
    native_root = _absolute_path(facade.get("root"), "native memory root")
    if native_root != expected_root:
        raise ProductionSessionBindingError("native memory root escapes the formal output")
    expected_paths = {
        "strategy_store_path": native_root / "strategy_specs.json",
        "action_exemplar_store_path": native_root / "action_exemplars.json",
        "raw_rollout_path": native_root / "raw_rollouts.jsonl",
        "accepted_episode_manifest_path": native_root / "accepted_episode_manifest.json",
    }
    resolved: dict[str, Path] = {}
    for name, wanted in expected_paths.items():
        actual = _absolute_path(facade.get(name), name.replace("_", " "))
        if actual != wanted:
            raise ProductionSessionBindingError(f"native {name} is not canonical")
        resolved[name] = actual
    runtime_audit = _mapping(runtime.get("runtime_audit"), "native runtime audit")
    audit_path = _absolute_path(runtime_audit.get("path"), "native audit path")
    if (
        audit_path != native_root / V83_NATIVE_AUDIT_FILENAME
        or runtime_audit.get("path_source")
        != "output_dir/memory/v83_native/fixed_filename"
    ):
        raise ProductionSessionBindingError("native audit path is not canonical")
    _require_fresh_native_root(native_root)

    transfer_seed_value = facade.get("transfer_seed_path")
    if transfer:
        transfer_seed_path = _absolute_file(
            transfer_seed_value,
            "native transfer seed",
        )
        if transfer_seed_path != output_dir / "memory" / "seed_strategy_bank.json":
            raise ProductionSessionBindingError("native transfer seed path is not canonical")
    else:
        if transfer_seed_value is not None:
            raise ProductionSessionBindingError("non-transfer run binds a transfer seed")
        transfer_seed_path = None

    bootstrap = _mapping(runtime.get("bootstrap"), "memory bootstrap binding")
    bootstrap_path = _absolute_file(bootstrap.get("path"), "memory bootstrap")
    bootstrap_sha = _sha_value(bootstrap.get("sha256"), "memory bootstrap SHA-256")
    bootstrap_fingerprint = _sha_value(
        bootstrap.get("fingerprint"),
        "memory bootstrap fingerprint",
    )
    if bootstrap.get("role") != "immutable_launch_binding_only" or bootstrap.get(
        "is_production_memory_evidence"
    ) is not False:
        raise ProductionSessionBindingError("memory bootstrap role is not canonical")
    binding = _mapping(memory_payload.get("binding"), "launch memory binding")
    launch_checks = {
        "bootstrap_snapshot_path": str(bootstrap_path),
        "bootstrap_snapshot_sha256": bootstrap_sha,
        "bootstrap_snapshot_fingerprint": bootstrap_fingerprint,
    }
    for name, wanted in launch_checks.items():
        if binding.get(name) != wanted:
            raise ProductionSessionBindingError(
                f"launch memory {name} differs from runtime bootstrap"
            )

    result = V83NativeMemoryBootstrapBinding(
        run_spec=run,
        plan_path=plan_path,
        plan_sha256=plan_sha,
        bootstrap_path=bootstrap_path,
        bootstrap_kind=_nonempty(bootstrap.get("kind"), "memory bootstrap kind"),
        bootstrap_schema_version=_integer(
            bootstrap.get("schema_version"), "memory bootstrap schema version"
        ),
        bootstrap_sha256=bootstrap_sha,
        bootstrap_fingerprint=bootstrap_fingerprint,
        mode=expected_mode,
        gates=gates,
        native_root=native_root,
        strategy_store_path=resolved["strategy_store_path"],
        action_exemplar_store_path=resolved["action_exemplar_store_path"],
        raw_rollout_path=resolved["raw_rollout_path"],
        accepted_episode_manifest_path=resolved[
            "accepted_episode_manifest_path"
        ],
        audit_path=audit_path,
        transfer_seed_path=transfer_seed_path,
        run_binding=run_binding,
    )
    result._revalidate_initial_state()
    return result


def _operation_gates(memory_payload: Mapping[str, Any]) -> V83MemoryOperationGates:
    runtime = _mapping(memory_payload.get("runtime_contract"), "memory runtime contract")
    payload = _exact_mapping(
        runtime.get("operation_gates"),
        _OPERATION_GATE_FIELDS,
        "native memory operation gates",
    )
    if not all(type(value) is bool for value in payload.values()):
        raise ProductionSessionBindingError("native memory operation gates must be booleans")
    return V83MemoryOperationGates(**payload)


def _validate_output_and_provider(
    run: RunSpec,
    output_value: Mapping[str, Any],
    provider_value: Mapping[str, Any],
) -> Path:
    output = _exact_mapping(output_value, _FORMAL_OUTPUT_FIELDS, "formal output")
    root = _absolute_path(output.get("results_root"), "formal results root")
    output_dir = _absolute_path(output.get("output_dir"), "formal output directory")
    expected = (root / PurePosixPath(run.output_relpath)).resolve(strict=False)
    if output_dir != expected:
        raise ProductionSessionBindingError("formal output path differs from RunSpec")
    checks = {
        "output_relpath": run.output_relpath,
        "seed": run.seed,
        "physical_budget": run.physical_budget,
    }
    for name, wanted in checks.items():
        if output.get(name) != wanted:
            raise ProductionSessionBindingError(f"formal output {name} differs from RunSpec")
    schema_checks = {
        "result_contract_schema_version": RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
        "terminal_evidence_schema_version": TERMINAL_EVIDENCE_SCHEMA_VERSION,
        "fm_usage_ledger_schema_version": FM_USAGE_LEDGER_SCHEMA_VERSION,
        "transaction_accounting_ledger_schema_version": (
            TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION
        ),
    }
    for name, wanted in schema_checks.items():
        if output.get(name) != wanted:
            raise ProductionSessionBindingError(
                f"formal output {name} differs from the reviewed schema"
            )
    required_pins = _exact_mapping(
        output.get("required_result_artifact_pins"),
        frozenset(
            {
                "native_memory_audit",
                "terminal_evidence",
                "fm_usage_ledger",
                "transaction_accounting_ledger",
                "accepted_episode_manifest",
                "frozen_phase_plan",
            }
        ),
        "required result artifact pins",
    )
    expected_pins = {
        "native_memory_audit": "v83_native_memory_runtime_audit_sha256",
        "terminal_evidence": "terminal_verifier_evidence_sha256",
        "fm_usage_ledger": "fm_usage_ledger_sha256",
        "transaction_accounting_ledger": TRANSACTION_ACCOUNTING_LEDGER_PIN,
        "accepted_episode_manifest": ACCEPTED_MANIFEST_PIN,
        "frozen_phase_plan": FROZEN_PHASE_PLAN_PIN,
    }
    if dict(required_pins) != expected_pins:
        raise ProductionSessionBindingError(
            "formal output required result pins differ from the reviewed contract"
        )

    provider = _exact_mapping(provider_value, _PROVIDER_FIELDS, "provider route")
    route = require_provider_route(run.provider_route_id, run.provider_route_fingerprint)
    provider_checks = {
        "route_id": route.route_id,
        "route_fingerprint": route.fingerprint,
        "provider": route.provider,
        "base_url": route.base_url,
        "api_family": route.api_family,
        "target_model": run.target_backbone,
    }
    for name, wanted in provider_checks.items():
        if provider.get(name) != wanted:
            raise ProductionSessionBindingError(f"provider {name} differs from RunSpec")
    _nonempty(provider.get("credential_env"), "provider credential environment name")
    return output_dir


def _validate_scene_channels(run: RunSpec, channels: ConsumedProductionChannels) -> None:
    term1 = _exact_mapping(
        channels.term1_simulator_asset,
        frozenset(
            {
                "scene_id",
                "source_path",
                "source_sha256",
                "execution_path",
                "execution_sha256",
                "execution_composite_sha256",
                "execution_evidence_fingerprint",
                "compiled_model_sha256",
                "contact_role_registry",
                "initial_state",
                "target",
            }
        ),
        "Term1 simulator asset",
    )
    clearance = _exact_mapping(
        channels.host_clearance_geometry,
        frozenset({"scene_id", "scene_fingerprint", "corridor"}),
        "Host clearance geometry",
    )
    vlm = _exact_mapping(
        channels.vlm_scene_prior,
        frozenset(
            {
                "scene_id",
                "label",
                "camera",
                "initial_state",
                "target",
                "mission_topology",
                "route_agent_visible",
            }
        ),
        "VLM scene prior",
    )
    route = _exact_mapping(
        channels.route_and_progress_gates,
        frozenset(
            {
                "schema_version",
                "kind",
                "scene_id",
                "mission_topology",
                "progress_gates",
                "segments",
            }
        ),
        "route and progress gates",
    )
    scene = load_scene(run.scene_id)
    if scene.scene_hash != run.scene_fingerprint:
        raise ProductionSessionBindingError(
            "RunSpec scene fingerprint differs from the checked-in registry"
        )

    expected_term1 = {
        "scene_id": scene.scene_id.value,
        "source_sha256": scene.asset.source_sha256,
        "execution_sha256": scene.asset.execution_sha256,
        "execution_composite_sha256": scene.asset.execution_composite_sha256,
        "initial_state": to_primitive(scene.initial_state),
        "target": to_primitive(scene.target),
    }
    for name, expected in expected_term1.items():
        if to_primitive(term1.get(name)) != expected:
            raise ProductionSessionBindingError(
                f"Term1 scene {name} differs from the checked-in registry"
            )
    expected_clearance = {
        "scene_id": scene.scene_id.value,
        "scene_fingerprint": scene.scene_hash,
        "corridor": to_primitive(scene.corridor),
    }
    if to_primitive(clearance) != expected_clearance:
        raise ProductionSessionBindingError(
            "Host clearance geometry differs from the checked-in scene"
        )
    expected_vlm = {
        "scene_id": scene.scene_id.value,
        "label": scene.label,
        "camera": to_primitive(scene.camera),
        "initial_state": to_primitive(scene.initial_state),
        "target": to_primitive(scene.target),
        "mission_topology": to_primitive(scene.route.mission_topology),
        "route_agent_visible": scene.route.agent_visible,
    }
    if to_primitive(vlm) != expected_vlm:
        raise ProductionSessionBindingError(
            "VLM scene prior differs from the checked-in scene"
        )
    if to_primitive(route) != to_primitive(directed_route_document(scene)):
        raise ProductionSessionBindingError(
            "route/progress gates differ from the checked-in scene"
        )
    source = _absolute_file(term1.get("source_path"), "scene source asset")
    execution = _absolute_file(term1.get("execution_path"), "scene execution asset")
    _require_file_sha(
        source,
        _sha_value(term1.get("source_sha256"), "scene source SHA-256"),
        "scene source asset",
    )
    _require_file_sha(
        execution,
        _sha_value(term1.get("execution_sha256"), "scene execution SHA-256"),
        "scene execution asset",
    )


def _native_run_binding(run: RunSpec, plan_path: Path, plan_sha: str) -> dict[str, Any]:
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
        "strategy_bank_sha256": (
            "" if run.strategy_bank is None else run.strategy_bank.sha256
        ),
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
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


def _require_fresh_native_root(path: Path) -> None:
    if path.is_symlink():
        raise ProductionSessionBindingError("native memory root must not be a symlink")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ProductionSessionBindingError(
                "native memory root is not pristine; formal resume is forbidden"
            )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionSessionBindingError(f"{label} must be an object")
    return value


def _exact_mapping(
    value: object,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    payload = _mapping(value, label)
    actual = frozenset(payload)
    if actual != expected:
        raise ProductionSessionBindingError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return payload


def _absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProductionSessionBindingError(f"{label} path must be a string")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ProductionSessionBindingError(
            f"{label} path must be absolute and non-symlinked"
        )
    return path.resolve(strict=False)


def _absolute_file(value: object, label: str) -> Path:
    path = _absolute_path(value, label)
    if not path.is_file():
        raise ProductionSessionBindingError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _require_file_sha(path: Path, expected: str, label: str) -> None:
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ProductionSessionBindingError(f"{label} byte hash differs")


def _sha_value(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProductionSessionBindingError(f"{label} must be SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProductionSessionBindingError(f"{label} must be SHA-256") from exc
    if value.lower() != value:
        raise ProductionSessionBindingError(f"{label} must be lowercase SHA-256")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionSessionBindingError(f"{label} must be non-empty")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProductionSessionBindingError(f"{label} must be an integer")
    return value


def _integer_tuple(value: object, label: str) -> Tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ProductionSessionBindingError(f"{label} must be an integer array")
    return tuple(value)


__all__ = [
    "ConsumedProductionChannels",
    "ProductionMethodPromptBinding",
    "ProductionSessionBinding",
    "ProductionSessionBindingError",
    "V83NativeMemoryBootstrapBinding",
    "bind_production_session",
]
