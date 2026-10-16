"""Audited run-local memory for Table-2/3 from-scratch conditions.

This lifecycle is intentionally separate from Table 4's preloaded-bank
bootstrap.  It proves that an ``empty`` or ``no_dynamic`` run starts with no
strategy/action records and that only protocol-authorized online operations
occur.  Safety lineage files are initialized alongside memory, but remain
owned by the production Host after dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from coevolve_experiments.contracts import (
    MemoryCondition,
    MemoryPolicy,
    RunSpec,
    TableId,
)
from coevolve_experiments.methods.base import (
    ActionExemplarQuery,
    AdapterContext,
    AdapterMemoryPort,
    MethodAdapter,
    RoleBackend,
)
from coevolve_experiments.methods.factory import create_method_adapter
from coevolve_experiments.runtime.contracts import RolloutRecord

from .json_io import atomic_write_json, read_json_object
from .online_memory import (
    action_exemplar_from_rollout,
    empty_action_exemplar_memory,
    empty_online_strategy_overlay,
    strategy_entry_from_rollout,
    validate_action_exemplar_memory,
    validate_online_strategy_overlay,
)
from .retrieval import select_strategy_records
from .schema import semantic_sha256


FROM_SCRATCH_BOOTSTRAP_FILENAME = "from_scratch_memory_bootstrap.json"
FROM_SCRATCH_RUNTIME_AUDIT_FILENAME = "from_scratch_memory_runtime_audit.json"
FROM_SCRATCH_BOOTSTRAP_KIND = "coevolve_from_scratch_memory_bootstrap"
FROM_SCRATCH_RUNTIME_AUDIT_KIND = "coevolve_from_scratch_memory_runtime_audit"
FROM_SCRATCH_SCHEMA_VERSION = 1
FROM_SCRATCH_INITIAL_FILES = (
    "online_strategy_overlay.json",
    "action_exemplars.json",
    "accepted_episode_manifest.json",
    "failure_cache.json",
)
MAX_RETRIEVED_ACTION_EXEMPLARS = 2


class FromScratchMemoryError(ValueError):
    """A from-scratch memory identity or lifecycle invariant was violated."""


@dataclass(frozen=True)
class FromScratchMemoryPreparation:
    run_spec: RunSpec
    memory_dir: Path
    bootstrap_path: Path
    online_overlay_path: Path
    action_exemplars_path: Path
    accepted_manifest_path: Path
    failure_cache_path: Path
    bootstrap_sha256: str


@dataclass(frozen=True)
class FromScratchBootstrapSnapshot:
    """Verified immutable identity of one Table-2/3 empty bootstrap.

    The live-launch schema already has generic ``bootstrap_*`` fields.  This
    view lets Table 2/3 reuse those fields without introducing a second launch
    schema or pretending that a from-scratch bootstrap is a Table-4 transfer
    snapshot.
    """

    path: Path
    file_sha256: str
    snapshot_fingerprint: str
    initial_file_sha256s: tuple[tuple[str, str], ...]
    formal_binding_sha256: str
    operation_gates: Mapping[str, bool]


@dataclass(frozen=True)
class FromScratchAdapterBinding:
    run_spec: RunSpec
    adapter: MethodAdapter
    memory: "FromScratchAdapterMemory"

    def finalize(self) -> Mapping[str, Any]:
        return self.memory.finalize()

    def __enter__(self) -> "FromScratchAdapterBinding":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.memory.finalize()
        return False


def empty_accepted_episode_manifest() -> dict[str, Any]:
    return {
        "kind": "accepted_checkpoint_episode",
        "schema_version": 1,
        "initial_checkpoint": "c0",
        "latest_checkpoint": "c0",
        "initial_state": {},
        "steps": [],
    }


def empty_failure_cache() -> dict[str, Any]:
    return {
        "kind": "run_local_failure_cache",
        "schema_version": 1,
        "failures": [],
    }


def prepare_from_scratch_memory(
    run_spec: RunSpec,
    memory_dir: str | Path,
    *,
    plan_path: str | Path,
    plan_sha256: str,
) -> FromScratchMemoryPreparation:
    """Create a non-overwriting, plan-bound empty-memory bootstrap."""

    _validate_from_scratch_run(run_spec)
    plan = Path(plan_path)
    if not plan.is_absolute() or not plan.is_file() or plan.is_symlink():
        raise FromScratchMemoryError("plan_path must be an absolute regular file")
    if not _is_sha256(plan_sha256) or _file_sha256(plan) != plan_sha256:
        raise FromScratchMemoryError("plan_sha256 does not match plan_path bytes")

    root = Path(memory_dir).absolute()
    if root.exists():
        raise FromScratchMemoryError(
            "from-scratch memory directory already exists; refusing resume/overwrite"
        )
    if _path_or_ancestor_is_symlink(root.parent):
        raise FromScratchMemoryError("memory directory ancestry must not contain symlinks")
    root.mkdir(parents=True, exist_ok=False)

    overlay_path = root / "online_strategy_overlay.json"
    actions_path = root / "action_exemplars.json"
    accepted_path = root / "accepted_episode_manifest.json"
    failure_path = root / "failure_cache.json"
    bootstrap_path = root / FROM_SCRATCH_BOOTSTRAP_FILENAME
    initial_values = {
        overlay_path.name: empty_online_strategy_overlay(),
        actions_path.name: empty_action_exemplar_memory(),
        accepted_path.name: empty_accepted_episode_manifest(),
        failure_path.name: empty_failure_cache(),
    }
    for path in (overlay_path, actions_path, accepted_path, failure_path):
        _write_fresh_json(path, initial_values[path.name])

    bootstrap = {
        "kind": FROM_SCRATCH_BOOTSTRAP_KIND,
        "schema_version": FROM_SCRATCH_SCHEMA_VERSION,
        "run_binding": _run_binding(
            run_spec,
            plan_path=plan,
            plan_sha256=plan_sha256,
        ),
        "memory_condition": run_spec.memory_condition.value,
        "memory_policy": run_spec.memory_policy.value,
        "cross_run_retrieval": False,
        "operation_gates": _operation_gates(run_spec.memory_policy),
        "initial_state": {
            "physical_state": "s0",
            "checkpoint": "c0",
            "strategy_records": 0,
            "action_exemplars": 0,
            "accepted_manifest_steps": 0,
            "failure_cache_records": 0,
        },
        "initial_files": [
            {
                "filename": filename,
                "sha256": _file_sha256(root / filename),
                "mutable_after_dispatch": True,
            }
            for filename in FROM_SCRATCH_INITIAL_FILES
        ],
    }
    _write_fresh_json(bootstrap_path, bootstrap)
    bootstrap_path.chmod(0o444)
    bootstrap_sha = _file_sha256(bootstrap_path)
    _validate_bootstrap(
        root,
        run_spec,
        require_initial_state=True,
        expected_plan_path=plan,
        expected_plan_sha256=plan_sha256,
    )
    return FromScratchMemoryPreparation(
        run_spec=run_spec,
        memory_dir=root.resolve(strict=True),
        bootstrap_path=bootstrap_path,
        online_overlay_path=overlay_path,
        action_exemplars_path=actions_path,
        accepted_manifest_path=accepted_path,
        failure_cache_path=failure_path,
        bootstrap_sha256=bootstrap_sha,
    )


def validate_from_scratch_bootstrap(
    memory_dir: str | Path,
    run_spec: RunSpec,
    *,
    require_initial_state: bool,
    expected_plan_path: str | Path | None = None,
    expected_plan_sha256: str | None = None,
) -> FromScratchBootstrapSnapshot:
    """Validate and project one immutable Table-2/3 bootstrap identity.

    ``require_initial_state=True`` is used immediately before dispatch and
    proves that all four mutable run-local files still contain their canonical
    empty values.  Post-run config reconstruction uses ``False`` so the same
    immutable bootstrap can be checked after legitimate online updates.
    """

    _validate_from_scratch_run(run_spec)
    root = Path(memory_dir).absolute()
    plan = None if expected_plan_path is None else Path(expected_plan_path)
    bootstrap = _validate_bootstrap(
        root,
        run_spec,
        require_initial_state=require_initial_state,
        expected_plan_path=plan,
        expected_plan_sha256=expected_plan_sha256,
    )
    binding = bootstrap["run_binding"]
    initial_files = bootstrap["initial_files"]
    return FromScratchBootstrapSnapshot(
        path=(root / FROM_SCRATCH_BOOTSTRAP_FILENAME).resolve(strict=True),
        file_sha256=_file_sha256(root / FROM_SCRATCH_BOOTSTRAP_FILENAME),
        snapshot_fingerprint=semantic_sha256(bootstrap),
        initial_file_sha256s=tuple(
            (str(item["filename"]), str(item["sha256"]))
            for item in initial_files
        ),
        formal_binding_sha256=semantic_sha256(binding),
        operation_gates=dict(bootstrap["operation_gates"]),
    )


def from_scratch_run_binding(
    run_spec: RunSpec,
    *,
    plan_path: str | Path,
    plan_sha256: str,
) -> Mapping[str, Any]:
    """Build the external expectation for bootstrap/final-audit validation."""

    _validate_from_scratch_run(run_spec)
    plan = Path(plan_path)
    if not plan.is_absolute() or not plan.is_file() or plan.is_symlink():
        raise FromScratchMemoryError("plan_path must be an absolute regular file")
    if not _is_sha256(plan_sha256) or _file_sha256(plan) != plan_sha256:
        raise FromScratchMemoryError("plan_sha256 does not match plan_path bytes")
    return _run_binding(run_spec, plan_path=plan, plan_sha256=plan_sha256)


class FromScratchAdapterMemory(AdapterMemoryPort):
    """File-backed memory port with explicit enabled/disabled gates."""

    def __init__(self, run_spec: RunSpec, memory_dir: str | Path) -> None:
        _validate_from_scratch_run(run_spec)
        self.run_spec = run_spec
        self.root = Path(memory_dir).absolute().resolve(strict=True)
        self.bootstrap_path = self.root / FROM_SCRATCH_BOOTSTRAP_FILENAME
        self.runtime_audit_path = self.root / FROM_SCRATCH_RUNTIME_AUDIT_FILENAME
        self.online_overlay_path = self.root / "online_strategy_overlay.json"
        self.action_exemplars_path = self.root / "action_exemplars.json"
        self._disabled = run_spec.memory_policy is MemoryPolicy.DISABLED

        bootstrap = _validate_bootstrap(
            self.root,
            run_spec,
            require_initial_state=True,
        )
        if self.runtime_audit_path.exists():
            raise FromScratchMemoryError(
                "from-scratch runtime audit already exists; refusing resume/overwrite"
            )
        overlay = self._load_overlay()
        actions = self._load_actions()
        self._audit: dict[str, Any] = {
            "kind": FROM_SCRATCH_RUNTIME_AUDIT_KIND,
            "schema_version": FROM_SCRATCH_SCHEMA_VERSION,
            "lifecycle_state": "active",
            "run_binding": dict(bootstrap["run_binding"]),
            "memory_condition": run_spec.memory_condition.value,
            "memory_policy": run_spec.memory_policy.value,
            "cross_run_retrieval": False,
            "operation_gates": _operation_gates(run_spec.memory_policy),
            "bootstrap_path": self.bootstrap_path.name,
            "bootstrap_sha256": _file_sha256(self.bootstrap_path),
            "initial_state_verified": True,
            "online_overlay": {
                "path": self.online_overlay_path.name,
                "initial_sha256": semantic_sha256(overlay),
                "current_sha256": semantic_sha256(overlay),
                "final_sha256": "",
                "initial_count": 0,
                "current_count": 0,
                "final_count": None,
            },
            "action_exemplars": {
                "path": self.action_exemplars_path.name,
                "initial_sha256": semantic_sha256(actions),
                "current_sha256": semantic_sha256(actions),
                "final_sha256": "",
                "initial_count": 0,
                "current_count": 0,
                "final_count": None,
            },
            "operations": {
                "strategy_retrievals": 0,
                "exemplar_retrievals": 0,
                "strategy_promotions": 0,
                "exemplar_promotions": 0,
                "promotion_noops": 0,
            },
            "violations": [],
            "finalized": False,
        }
        self._persist()

    def retrieve_strategies(
        self, context: AdapterContext
    ) -> Sequence[Mapping[str, Any]]:
        self._begin("retrieve_strategies", gate="strategy_retrieval")
        if not isinstance(context, AdapterContext):
            self._violate("retrieve_strategies", "AdapterContext is required")
        overlay = self._load_overlay_and_require_current()
        failure = context.latest_physical_failure
        selected = select_strategy_records(
            [dict(item) for item in overlay["entries"]],
            active_phase=context.observation.state.get("phase", "unspecified"),
            failure_code="" if failure is None else failure.failure_code,
            failure_report="" if failure is None else failure.report,
        )
        self._record_operation("strategy_retrievals")
        return selected

    def retrieve_action_exemplars(
        self,
        context: AdapterContext,
        query: ActionExemplarQuery,
    ) -> Sequence[Mapping[str, Any]]:
        self._begin("retrieve_action_exemplars", gate="action_exemplar_retrieval")
        if not isinstance(context, AdapterContext) or not isinstance(
            query, ActionExemplarQuery
        ):
            self._violate(
                "retrieve_action_exemplars",
                "AdapterContext and ActionExemplarQuery are required",
            )
        actions = self._load_actions_and_require_current()
        compatible = [
            dict(item)
            for item in actions["exemplars"]
            if _action_exemplar_matches(item, query)
        ]
        compatible.sort(key=lambda item: _action_sort_key(item, query))
        self._record_operation("exemplar_retrievals")
        return tuple(compatible[:MAX_RETRIEVED_ACTION_EXEMPLARS])

    def promote_strategy(self, proposal: Any, rollout: RolloutRecord) -> None:
        self._begin("promote_strategy", gate="strategy_promotion")
        try:
            entry = strategy_entry_from_rollout(proposal, rollout)
            overlay = self._load_overlay_and_require_current()
            identity = _strategy_identity(entry)
            if identity in {_strategy_identity(item) for item in overlay["entries"]}:
                self._record_operation("promotion_noops")
                return
            updated = {**overlay, "entries": [*overlay["entries"], entry]}
            updated["entries"].sort(key=_strategy_identity)
            validate_online_strategy_overlay(updated)
            atomic_write_json(self.online_overlay_path, updated)
            persisted = self._load_overlay()
            self._audit["online_overlay"].update(
                current_sha256=semantic_sha256(persisted),
                current_count=len(persisted["entries"]),
            )
            self._record_operation("strategy_promotions")
        except Exception as exc:
            self._operation_failed("promote_strategy", exc)

    def promote_action_exemplar(
        self, proposal: Any, rollout: RolloutRecord
    ) -> None:
        del proposal
        self._begin("promote_action_exemplar", gate="action_exemplar_promotion")
        try:
            exemplar = action_exemplar_from_rollout(rollout)
            actions = self._load_actions_and_require_current()
            identity = _action_identity(exemplar)
            if identity in {_action_identity(item) for item in actions["exemplars"]}:
                self._record_operation("promotion_noops")
                return
            updated = {**actions, "exemplars": [*actions["exemplars"], exemplar]}
            updated["exemplars"].sort(key=_action_identity)
            validate_action_exemplar_memory(updated)
            atomic_write_json(self.action_exemplars_path, updated)
            persisted = self._load_actions()
            self._audit["action_exemplars"].update(
                current_sha256=semantic_sha256(persisted),
                current_count=len(persisted["exemplars"]),
            )
            self._record_operation("exemplar_promotions")
        except Exception as exc:
            self._operation_failed("promote_action_exemplar", exc)

    def finalize(self) -> Mapping[str, Any]:
        if self._audit.get("finalized") is True:
            return _json_copy(self._audit)
        self._ensure_active("finalize")
        try:
            _validate_bootstrap(self.root, self.run_spec, require_initial_state=False)
            overlay = self._load_overlay_and_require_current()
            actions = self._load_actions_and_require_current()
            operations = self._audit["operations"]
            if self._disabled:
                if overlay["entries"] or actions["exemplars"] or any(
                    operations.values()
                ):
                    raise FromScratchMemoryError(
                        "disabled memory condition performed an operation or changed stores"
                    )
            else:
                if operations["strategy_promotions"] != len(overlay["entries"]):
                    raise FromScratchMemoryError(
                        "strategy promotion count differs from final overlay"
                    )
                if operations["exemplar_promotions"] != len(actions["exemplars"]):
                    raise FromScratchMemoryError(
                        "exemplar promotion count differs from final action store"
                    )
            for section, payload, field in (
                ("online_overlay", overlay, "entries"),
                ("action_exemplars", actions, "exemplars"),
            ):
                digest = semantic_sha256(payload)
                self._audit[section].update(
                    current_sha256=digest,
                    final_sha256=digest,
                    current_count=len(payload[field]),
                    final_count=len(payload[field]),
                )
            self._audit["lifecycle_state"] = "finalized"
            self._audit["finalized"] = True
            self._persist()
            return _json_copy(self._audit)
        except Exception as exc:
            self._operation_failed("finalize", exc)

    def _begin(self, operation: str, *, gate: str) -> None:
        self._ensure_active(operation)
        _validate_bootstrap(self.root, self.run_spec, require_initial_state=False)
        if self._audit["operation_gates"].get(gate) is not True:
            self._violate(operation, f"memory gate is disabled: {gate}")

    def _ensure_active(self, operation: str) -> None:
        if self._audit.get("lifecycle_state") != "active":
            raise FromScratchMemoryError(
                f"cannot {operation}: memory lifecycle is "
                f"{self._audit.get('lifecycle_state')}"
            )

    def _load_overlay(self) -> dict[str, Any]:
        return validate_online_strategy_overlay(
            read_json_object(self.online_overlay_path, "online strategy overlay")
        )

    def _load_actions(self) -> dict[str, Any]:
        return validate_action_exemplar_memory(
            read_json_object(self.action_exemplars_path, "action exemplars")
        )

    def _load_overlay_and_require_current(self) -> dict[str, Any]:
        value = self._load_overlay()
        self._require_current("online_overlay", semantic_sha256(value))
        return value

    def _load_actions_and_require_current(self) -> dict[str, Any]:
        value = self._load_actions()
        self._require_current("action_exemplars", semantic_sha256(value))
        return value

    def _require_current(self, section: str, digest: str) -> None:
        if self._audit[section]["current_sha256"] != digest:
            self._violate(section, "memory store changed outside the audited port")

    def _record_operation(self, name: str) -> None:
        self._audit["operations"][name] += 1
        self._persist()

    def _operation_failed(self, operation: str, exc: Exception) -> None:
        if self._audit.get("lifecycle_state") == "violated":
            if isinstance(exc, FromScratchMemoryError):
                raise exc
            raise FromScratchMemoryError(str(exc)) from exc
        self._violate(operation, str(exc))

    def _violate(self, operation: str, detail: str) -> None:
        self._audit["violations"].append(
            {
                "sequence": len(self._audit["violations"]),
                "operation": operation,
                "detail": detail,
            }
        )
        self._audit["lifecycle_state"] = "violated"
        self._audit["finalized"] = False
        self._persist()
        raise FromScratchMemoryError(detail)

    def _persist(self) -> None:
        atomic_write_json(self.runtime_audit_path, self._audit)


def bind_from_scratch_adapter(
    run_spec: RunSpec,
    backend: RoleBackend,
    memory_dir: str | Path,
) -> FromScratchAdapterBinding:
    """Bind every Table-2/3 method to an audited (even if disabled) port."""

    memory = FromScratchAdapterMemory(run_spec, memory_dir)
    adapter = create_method_adapter(run_spec.method_variant, backend, memory=memory)
    return FromScratchAdapterBinding(run_spec, adapter, memory)


def _validate_from_scratch_run(run: RunSpec) -> None:
    if not isinstance(run, RunSpec):
        raise FromScratchMemoryError("run_spec must be a RunSpec")
    if run.table_id not in {TableId.TABLE_2, TableId.TABLE_3}:
        raise FromScratchMemoryError("from-scratch lifecycle is restricted to Table 2/3")
    if run.memory_condition not in {MemoryCondition.EMPTY, MemoryCondition.NO_DYNAMIC}:
        raise FromScratchMemoryError("from-scratch lifecycle cannot preload a bank")
    if run.strategy_bank is not None:
        raise FromScratchMemoryError("from-scratch lifecycle cannot carry a strategy bank")
    if run.memory_policy not in {MemoryPolicy.ONLINE_FROM_EMPTY, MemoryPolicy.DISABLED}:
        raise FromScratchMemoryError("unsupported from-scratch memory policy")


def _operation_gates(policy: MemoryPolicy) -> dict[str, bool]:
    enabled = policy is MemoryPolicy.ONLINE_FROM_EMPTY
    return {
        "strategy_retrieval": enabled,
        "strategy_promotion": enabled,
        "action_exemplar_retrieval": enabled,
        "action_exemplar_promotion": enabled,
    }


def _run_binding(
    run: RunSpec,
    *,
    plan_path: Path,
    plan_sha256: str,
) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "table_id": run.table_id.value,
        "method_variant": run.method_variant.value,
        "scene_id": run.scene_id.value,
        "seed": run.seed,
        "protocol_id": run.protocol_id,
        "base_method_commit": run.base_method_commit,
        "condition_fingerprint": run.condition_fingerprint,
        "target_model": run.target_backbone,
        "provider_route_id": run.provider_route_id,
        "provider_route_fingerprint": run.provider_route_fingerprint,
        "observation_contract_fingerprint": run.observation_contract_fingerprint,
        "task_instruction_fingerprint": run.task_instruction_fingerprint,
        "shared_prompt_contract_fingerprint": run.shared_prompt_contract_fingerprint,
        "decision_prompt_fingerprint": run.decision_prompt_fingerprint,
        "skill_api_fingerprint": run.skill_api_fingerprint,
        "full_method_compatibility_fingerprint": (
            run.full_method_compatibility_fingerprint
        ),
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
    }


def _validate_bootstrap(
    root: Path,
    run: RunSpec,
    *,
    require_initial_state: bool,
    expected_plan_path: Path | None = None,
    expected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    _require_root_integrity(root)
    bootstrap_path = root / FROM_SCRATCH_BOOTSTRAP_FILENAME
    bootstrap = read_json_object(bootstrap_path, "from-scratch memory bootstrap")
    if set(bootstrap) != {
        "kind",
        "schema_version",
        "run_binding",
        "memory_condition",
        "memory_policy",
        "cross_run_retrieval",
        "operation_gates",
        "initial_state",
        "initial_files",
    }:
        raise FromScratchMemoryError("from-scratch bootstrap fields are not canonical")
    if (
        bootstrap.get("kind") != FROM_SCRATCH_BOOTSTRAP_KIND
        or bootstrap.get("schema_version") != FROM_SCRATCH_SCHEMA_VERSION
    ):
        raise FromScratchMemoryError("from-scratch bootstrap kind/schema differs")
    if bootstrap_path.stat().st_mode & 0o222:
        raise FromScratchMemoryError("from-scratch bootstrap must remain read-only")
    binding = bootstrap.get("run_binding")
    if not isinstance(binding, Mapping):
        raise FromScratchMemoryError("from-scratch run binding is missing")
    plan_path_value = expected_plan_path or Path(str(binding.get("plan_path", "")))
    plan_sha_value = expected_plan_sha256 or str(binding.get("plan_sha256", ""))
    if not plan_path_value.is_absolute() or not plan_path_value.is_file():
        raise FromScratchMemoryError("bound suite plan is missing")
    if _file_sha256(plan_path_value) != plan_sha_value:
        raise FromScratchMemoryError("bound suite plan changed after memory bootstrap")
    expected_binding = _run_binding(
        run,
        plan_path=plan_path_value,
        plan_sha256=plan_sha_value,
    )
    if dict(binding) != expected_binding:
        raise FromScratchMemoryError("from-scratch bootstrap differs from RunSpec")
    if (
        bootstrap.get("memory_condition") != run.memory_condition.value
        or bootstrap.get("memory_policy") != run.memory_policy.value
        or bootstrap.get("cross_run_retrieval") is not False
        or bootstrap.get("operation_gates") != _operation_gates(run.memory_policy)
    ):
        raise FromScratchMemoryError("from-scratch memory policy binding differs")
    if bootstrap.get("initial_state") != {
        "physical_state": "s0",
        "checkpoint": "c0",
        "strategy_records": 0,
        "action_exemplars": 0,
        "accepted_manifest_steps": 0,
        "failure_cache_records": 0,
    }:
        raise FromScratchMemoryError("from-scratch initial-state attestation differs")
    initial_files = bootstrap.get("initial_files")
    if not isinstance(initial_files, list) or [
        item.get("filename") if isinstance(item, Mapping) else None
        for item in initial_files
    ] != list(FROM_SCRATCH_INITIAL_FILES):
        raise FromScratchMemoryError("from-scratch initial file registry differs")
    for item in initial_files:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"filename", "sha256", "mutable_after_dispatch"}
            or not _is_sha256(item.get("sha256"))
            or item.get("mutable_after_dispatch") is not True
        ):
            raise FromScratchMemoryError("from-scratch initial file record is invalid")
        if require_initial_state:
            path = root / str(item["filename"])
            if _file_sha256(path) != item["sha256"]:
                raise FromScratchMemoryError(
                    f"from-scratch initial bytes changed for {path.name}"
                )
    if require_initial_state:
        if validate_online_strategy_overlay(
            read_json_object(root / "online_strategy_overlay.json", "online overlay")
        ) != empty_online_strategy_overlay():
            raise FromScratchMemoryError("strategy overlay does not start empty")
        if validate_action_exemplar_memory(
            read_json_object(root / "action_exemplars.json", "action exemplars")
        ) != empty_action_exemplar_memory():
            raise FromScratchMemoryError("action exemplars do not start empty")
        if read_json_object(
            root / "accepted_episode_manifest.json", "accepted manifest"
        ) != empty_accepted_episode_manifest():
            raise FromScratchMemoryError("accepted manifest does not start empty")
        if read_json_object(root / "failure_cache.json", "failure cache") != empty_failure_cache():
            raise FromScratchMemoryError("failure cache does not start empty")
    return bootstrap


def _action_exemplar_matches(
    entry: Mapping[str, Any], query: ActionExemplarQuery
) -> bool:
    intent = entry.get("intent")
    commands = intent.get("commands") if isinstance(intent, Mapping) else None
    if not isinstance(commands, list) or not commands:
        return False
    expected = (query.primitive, query.semantic_objective, query.coordination_mode)
    return all(
        isinstance(command, Mapping)
        and (
            command.get("primitive"),
            command.get("objective"),
            command.get("mode"),
        )
        == expected
        for command in commands
    )


def _action_sort_key(
    entry: Mapping[str, Any], query: ActionExemplarQuery
) -> tuple[int, int, str]:
    intent = entry.get("intent")
    phase = intent.get("lifecycle_phase") if isinstance(intent, Mapping) else None
    return (
        0 if phase == query.active_phase else 1,
        -int(entry["transaction_index"]),
        str(entry["ast_fingerprint"]),
    )


def _strategy_identity(entry: Mapping[str, Any]) -> tuple[int, str]:
    provenance = entry["provenance"]
    return int(provenance["transaction_index"]), str(provenance["ast_fingerprint"])


def _action_identity(entry: Mapping[str, Any]) -> tuple[int, str]:
    return int(entry["transaction_index"]), str(entry["ast_fingerprint"])


def _write_fresh_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FromScratchMemoryError(f"refusing to overwrite {path}")
    atomic_write_json(path, value)


def _require_root_integrity(root: Path) -> None:
    if not root.is_absolute() or not root.is_dir() or _path_or_ancestor_is_symlink(root):
        raise FromScratchMemoryError("from-scratch memory path is missing or not isolated")
    for filename in (
        *FROM_SCRATCH_INITIAL_FILES,
        FROM_SCRATCH_BOOTSTRAP_FILENAME,
        FROM_SCRATCH_RUNTIME_AUDIT_FILENAME,
    ):
        if (root / filename).is_symlink():
            raise FromScratchMemoryError(f"memory artifact is a symlink: {filename}")


def _path_or_ancestor_is_symlink(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


__all__ = [
    "FROM_SCRATCH_BOOTSTRAP_FILENAME",
    "FROM_SCRATCH_BOOTSTRAP_KIND",
    "FROM_SCRATCH_INITIAL_FILES",
    "FROM_SCRATCH_RUNTIME_AUDIT_FILENAME",
    "FROM_SCRATCH_RUNTIME_AUDIT_KIND",
    "FROM_SCRATCH_SCHEMA_VERSION",
    "FromScratchAdapterBinding",
    "FromScratchAdapterMemory",
    "FromScratchBootstrapSnapshot",
    "FromScratchMemoryError",
    "FromScratchMemoryPreparation",
    "bind_from_scratch_adapter",
    "empty_accepted_episode_manifest",
    "empty_failure_cache",
    "from_scratch_run_binding",
    "prepare_from_scratch_memory",
    "validate_from_scratch_bootstrap",
]
