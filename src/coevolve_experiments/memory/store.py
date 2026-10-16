"""Formal file-backed implementation of the experiment AdapterMemoryPort."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from coevolve_experiments.contracts import (
    MemoryCondition,
    MemoryPolicy,
    MethodVariant,
    RunSpec,
    TableId,
)
from coevolve_experiments.methods.base import (
    ActionExemplarQuery,
    AdapterContext,
    AdapterMemoryPort,
)
from coevolve_experiments.runtime.contracts import RolloutRecord

from .json_io import (
    atomic_write_json,
    read_json_object,
    resolve_local_artifact,
)
from .lifecycle import MemoryLifecycleAudit, MemoryLifecycleError
from .online_memory import (
    action_exemplar_from_rollout,
    empty_online_strategy_overlay,
    strategy_entry_from_rollout,
    validate_action_exemplar_memory,
    validate_online_strategy_overlay,
)
from .schema import semantic_sha256
from .provenance import (
    StrategyBankProvenanceError,
    validate_strategy_bank_provenance,
)
from .retrieval import select_strategy_records
from .seed_guard import SeedBankGuard, SeedIntegrityError
from .validation import load_strategy_bank, validate_strategy_bank


class MemoryRuntimeError(ValueError):
    """A formal run-local memory invariant was violated."""


# Full executable examples are substantially larger than abstract strategy
# records.  This frozen bound is part of the adapter contract: compatibility
# is exact and phase affects ordering only.
MAX_RETRIEVED_ACTION_EXEMPLARS = 2


class FileBackedAdapterMemory(AdapterMemoryPort):
    """Read an immutable seed and atomically update separate online memories.

    ``open_for_run`` requires the hash/model/condition pins already present in
    the selected RunSpec.  It never accepts those identities as independent
    caller-provided strings.
    """

    def __init__(self, root: Path, run_spec: RunSpec) -> None:
        _require_memory_root_integrity(root)
        self.root = root.absolute().resolve(strict=True)
        self.run_spec = run_spec
        self.bootstrap_audit_path = self.root / "memory_bootstrap_audit.json"
        self.seed_bank_path = self.root / "seed_strategy_bank.json"
        self.online_overlay_path = self.root / "online_strategy_overlay.json"
        self.action_exemplars_path = self.root / "action_exemplars.json"
        self._validate_run_spec()

        # This is the last check before the runtime creates its mutable audit.
        # It proves that prepare-transfer and bind-live-launch saw the same
        # pristine target memory, while keeping later online updates legal.
        from .bootstrap import validate_transfer_bootstrap_snapshot
        from .schema import MemoryBootstrapError

        try:
            validate_transfer_bootstrap_snapshot(
                self.root,
                require_initial_state=True,
                expected_condition=run_spec.memory_condition.value,
                expected_target_model=run_spec.target_backbone,
                expected_seed_bank_sha256=run_spec.strategy_bank.sha256,
            )
        except MemoryBootstrapError as exc:
            raise MemoryRuntimeError(
                f"transfer bootstrap snapshot preflight failed: {exc}"
            ) from exc

        bootstrap = read_json_object(
            self.bootstrap_audit_path,
            "memory bootstrap audit",
        )
        binding = self._validate_bootstrap(bootstrap)
        expected_seed = run_spec.strategy_bank.sha256
        self._seed_guard = SeedBankGuard(self.seed_bank_path, expected_seed)
        seed_digest = self._seed_guard.verify("preflight")
        overlay, overlay_digest = self._load_overlay()
        actions, action_digest = self._load_actions()
        if overlay["entries"]:
            raise MemoryRuntimeError(
                "formal memory preflight requires an empty online overlay"
            )
        if actions["exemplars"]:
            raise MemoryRuntimeError(
                "formal memory preflight requires empty action exemplars"
            )
        self._audit = MemoryLifecycleAudit.create(
            root=self.root,
            run_id=run_spec.run_id,
            protocol_id=run_spec.protocol_id,
            condition=run_spec.memory_condition.value,
            condition_fingerprint=run_spec.condition_fingerprint,
            target_model=run_spec.target_backbone,
            bootstrap_audit_sha256=semantic_sha256(bootstrap),
            plan_path=str(binding["plan_path"]),
            plan_sha256=str(binding["plan_sha256"]),
            seed_sha256=seed_digest,
            overlay_sha256=overlay_digest,
            action_sha256=action_digest,
            fairness_fingerprints={
                name: str(getattr(run_spec, name))
                for name in (
                    "observation_contract_fingerprint",
                    "task_instruction_fingerprint",
                    "shared_prompt_contract_fingerprint",
                    "decision_prompt_fingerprint",
                    "provider_route_fingerprint",
                    "skill_api_fingerprint",
                    "full_method_compatibility_fingerprint",
                )
            },
        )
        self.runtime_audit_path = self._audit.path
        self._guard_seed("runtime_audit_initialized")

    @classmethod
    def open_for_run(
        cls,
        run_spec: RunSpec,
        memory_dir: str | Path,
    ) -> "FileBackedAdapterMemory":
        return cls(Path(memory_dir), run_spec)

    def retrieve_strategies(
        self,
        context: AdapterContext,
    ) -> Sequence[Mapping[str, Any]]:
        self._begin("retrieve_strategies")
        try:
            if not isinstance(context, AdapterContext):
                raise MemoryRuntimeError("strategy retrieval requires AdapterContext")
            seed = load_strategy_bank(self.seed_bank_path)
            validate_strategy_bank(seed)
            overlay, digest = self._load_overlay()
            self._require_current("online_overlay", digest)
            # Frozen seed records and native run-local experience stay visibly
            # distinct.  Both are mappings on the same public retrieval port.
            records = [dict(item) for item in seed["records"]]
            records.extend(dict(item) for item in overlay["entries"])
            failure = context.latest_physical_failure
            selected = select_strategy_records(
                records,
                active_phase=context.observation.state.get("phase", "unspecified"),
                failure_code="" if failure is None else failure.failure_code,
                failure_report="" if failure is None else failure.report,
            )
            self._guard_seed("retrieve_strategies:post")
            self._audit.record_operation("strategy_retrievals")
            self._audit.persist()
            return selected
        except Exception as exc:
            self._operation_failed("retrieve_strategies", exc)

    def retrieve_action_exemplars(
        self,
        context: AdapterContext,
        query: ActionExemplarQuery,
    ) -> Sequence[Mapping[str, Any]]:
        if not isinstance(context, AdapterContext):
            raise MemoryRuntimeError(
                "action exemplar retrieval requires AdapterContext"
            )
        if not isinstance(query, ActionExemplarQuery):
            raise MemoryRuntimeError(
                "action exemplar retrieval requires a committed-strategy query"
            )
        self._begin("retrieve_action_exemplars")
        try:
            actions, digest = self._load_actions()
            self._require_current("action_exemplars", digest)
            compatible = [
                dict(item)
                for item in actions["exemplars"]
                if _action_exemplar_matches(item, query)
            ]
            compatible.sort(
                key=lambda item: _action_retrieval_sort_key(item, query)
            )
            self._guard_seed("retrieve_action_exemplars:post")
            self._audit.record_operation("exemplar_retrievals")
            self._audit.persist()
            return tuple(compatible[:MAX_RETRIEVED_ACTION_EXEMPLARS])
        except Exception as exc:
            self._operation_failed("retrieve_action_exemplars", exc)

    def promote_strategy(self, proposal: Any, rollout: RolloutRecord) -> None:
        self._begin("promote_strategy")
        try:
            entry = strategy_entry_from_rollout(proposal, rollout)
            overlay, digest = self._load_overlay()
            self._require_current("online_overlay", digest)
            identity = _entry_identity(entry)
            if identity in {_entry_identity(item) for item in overlay["entries"]}:
                self._guard_seed("promote_strategy:duplicate:post")
                self._audit.record_noop()
                self._audit.persist()
                return
            updated = {
                "kind": overlay["kind"],
                "schema_version": overlay["schema_version"],
                "entries": [*overlay["entries"], entry],
            }
            updated["entries"].sort(key=_entry_sort_key)
            validate_online_strategy_overlay(updated)
            atomic_write_json(self.online_overlay_path, updated)
            persisted, persisted_digest = self._load_overlay()
            self._guard_seed("promote_strategy:post")
            self._audit.update_overlay(
                persisted_digest,
                len(persisted["entries"]),
            )
            self._audit.record_operation("strategy_promotions")
            self._audit.persist()
        except Exception as exc:
            self._operation_failed("promote_strategy", exc)

    def promote_action_exemplar(
        self,
        proposal: Any,
        rollout: RolloutRecord,
    ) -> None:
        del proposal
        self._begin("promote_action_exemplar")
        try:
            exemplar = action_exemplar_from_rollout(rollout)
            actions, digest = self._load_actions()
            self._require_current("action_exemplars", digest)
            identity = _action_identity(exemplar)
            if identity in {_action_identity(item) for item in actions["exemplars"]}:
                self._guard_seed("promote_action_exemplar:duplicate:post")
                self._audit.record_noop()
                self._audit.persist()
                return
            updated = {
                "kind": actions["kind"],
                "schema_version": actions["schema_version"],
                "exemplars": [*actions["exemplars"], exemplar],
            }
            updated["exemplars"].sort(key=_action_sort_key)
            validate_action_exemplar_memory(updated)
            atomic_write_json(self.action_exemplars_path, updated)
            persisted, persisted_digest = self._load_actions()
            self._guard_seed("promote_action_exemplar:post")
            self._audit.update_actions(
                persisted_digest,
                len(persisted["exemplars"]),
            )
            self._audit.record_operation("exemplar_promotions")
            self._audit.persist()
        except Exception as exc:
            self._operation_failed("promote_action_exemplar", exc)

    def finalize(self) -> Mapping[str, Any]:
        if self._audit.finalized:
            return self._audit.snapshot()
        _require_memory_root_integrity(self.root)
        self._audit.ensure_active("finalize")
        try:
            self._guard_seed("finalize:pre")
            overlay, overlay_digest = self._load_overlay()
            actions, action_digest = self._load_actions()
            self._require_current("online_overlay", overlay_digest)
            self._require_current("action_exemplars", action_digest)
            operations = self._audit.payload["operations"]
            if operations["strategy_promotions"] != len(overlay["entries"]):
                raise MemoryRuntimeError(
                    "strategy promotion counter does not match online overlay"
                )
            if operations["exemplar_promotions"] != len(actions["exemplars"]):
                raise MemoryRuntimeError(
                    "exemplar promotion counter does not match action memory"
                )
            final_seed = self._guard_seed("finalize:post")
            return self._audit.finalize(
                seed_sha256=final_seed,
                overlay_sha256=overlay_digest,
                overlay_count=len(overlay["entries"]),
                action_sha256=action_digest,
                action_count=len(actions["exemplars"]),
            )
        except Exception as exc:
            self._operation_failed("finalize", exc)

    def __enter__(self) -> "FileBackedAdapterMemory":
        self._audit.ensure_active("enter")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc, traceback
        if self._audit.state == "active":
            self.finalize()
        return False

    def _validate_run_spec(self) -> None:
        run = self.run_spec
        if run.table_id is not TableId.TABLE_4:
            raise MemoryRuntimeError("file-backed transfer memory requires a Table-4 RunSpec")
        if run.method_variant is not MethodVariant.COEVOLE_FULL:
            raise MemoryRuntimeError("Table-4 transfer memory requires Full CoEvolve")
        if run.memory_condition not in {
            MemoryCondition.GPT_TO_GPT,
            MemoryCondition.GEMINI_TO_GPT,
        }:
            raise MemoryRuntimeError("file-backed transfer memory requires a transfer condition")
        if run.memory_policy is not MemoryPolicy.ONLINE_FROM_PRELOADED_STRATEGIES:
            raise MemoryRuntimeError("transfer RunSpec has the wrong memory policy")
        if run.strategy_bank is None:
            raise MemoryRuntimeError("transfer RunSpec is missing its strategy-bank pin")

    def _validate_bootstrap(self, bootstrap: Mapping[str, Any]) -> Mapping[str, Any]:
        required = {
            "kind",
            "schema_version",
            "condition",
            "source_model",
            "source_model_family",
            "target_model",
            "target_model_family",
            "seed",
            "online_overlay",
            "fresh_target_state",
            "formal_binding",
        }
        if set(bootstrap) != required:
            raise MemoryRuntimeError(
                "formal bootstrap audit fields differ: "
                f"missing={sorted(required - set(bootstrap))}, "
                f"unknown={sorted(set(bootstrap) - required)}"
            )
        if bootstrap.get("kind") != "table4_memory_bootstrap_audit":
            raise MemoryRuntimeError("memory bootstrap audit has the wrong kind")
        schema_version = bootstrap.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            raise MemoryRuntimeError("memory bootstrap audit has unsupported schema_version")
        expected_label = {
            MemoryCondition.GPT_TO_GPT: "GPT→GPT",
            MemoryCondition.GEMINI_TO_GPT: "Gemini→GPT",
        }[self.run_spec.memory_condition]
        expected = {
            "condition": expected_label,
            "target_model": self.run_spec.target_backbone,
            "source_model": self.run_spec.strategy_bank.source_backbone,
        }
        for name, value in expected.items():
            if bootstrap.get(name) != value:
                raise MemoryRuntimeError(f"bootstrap {name} does not match RunSpec")

        seed = _mapping(bootstrap.get("seed"), "bootstrap seed")
        overlay = _mapping(bootstrap.get("online_overlay"), "bootstrap overlay")
        fresh = _mapping(bootstrap.get("fresh_target_state"), "bootstrap fresh state")
        binding = _mapping(bootstrap.get("formal_binding"), "formal binding")
        if set(fresh) != {
            "physical_state",
            "checkpoint",
            "action_exemplars_empty",
            "accepted_manifest_steps_empty",
            "failure_cache_empty",
        }:
            raise MemoryRuntimeError(
                "bootstrap fresh-state fields are not canonical"
            )
        resolve_local_artifact(self.root, seed.get("path"), self.seed_bank_path.name)
        resolve_local_artifact(self.root, overlay.get("path"), self.online_overlay_path.name)
        if seed.get("sha256") != self.run_spec.strategy_bank.sha256:
            raise MemoryRuntimeError("bootstrap seed hash does not match RunSpec")
        if seed.get("immutable") is not True or seed.get("mode") != "0444":
            raise MemoryRuntimeError("bootstrap does not attest a read-only seed")
        if overlay.get("sha256_at_bootstrap") != semantic_sha256(
            empty_online_strategy_overlay()
        ):
            raise MemoryRuntimeError("bootstrap overlay does not start from the frozen empty hash")
        if overlay.get("record_count_at_bootstrap") != 0:
            raise MemoryRuntimeError("bootstrap overlay record count is not zero")
        if overlay.get("separate_from_seed") is not True:
            raise MemoryRuntimeError("bootstrap overlay is not separate from the seed")
        for name in (
            "action_exemplars_empty",
            "accepted_manifest_steps_empty",
            "failure_cache_empty",
        ):
            if fresh.get(name) is not True:
                raise MemoryRuntimeError(f"bootstrap fresh-state flag is false: {name}")
        if fresh.get("physical_state") != "s0":
            raise MemoryRuntimeError("bootstrap physical_state is not s0")
        if fresh.get("checkpoint") != "c0":
            raise MemoryRuntimeError("bootstrap checkpoint is not c0")

        provenance_path = seed.get("provenance_path")
        provenance_sha256 = seed.get("provenance_sha256")
        if not isinstance(provenance_path, str) or not Path(
            provenance_path
        ).is_absolute():
            raise MemoryRuntimeError(
                "bootstrap seed provenance_path must be absolute"
            )
        if not _is_sha256(provenance_sha256):
            raise MemoryRuntimeError(
                "bootstrap seed provenance_sha256 must be lowercase SHA-256"
            )
        provenance = read_json_object(
            Path(provenance_path),
            "strategy-bank provenance",
        )
        if semantic_sha256(provenance) != provenance_sha256:
            raise MemoryRuntimeError(
                "strategy-bank provenance changed after bootstrap"
            )
        try:
            validate_strategy_bank_provenance(
                provenance,
                strategy_bank_pin=self.run_spec.strategy_bank,
                expected_bank_sha256=self.run_spec.strategy_bank.sha256,
            )
        except StrategyBankProvenanceError as exc:
            raise MemoryRuntimeError(str(exc)) from exc

        binding_expected = {
            "run_id": self.run_spec.run_id,
            "protocol_id": self.run_spec.protocol_id,
            "condition_fingerprint": self.run_spec.condition_fingerprint,
            "strategy_bank_sha256": self.run_spec.strategy_bank.sha256,
            "source_model": self.run_spec.strategy_bank.source_backbone,
            "target_model": self.run_spec.target_backbone,
            "source_provider_route_id": (
                self.run_spec.strategy_bank.source_provider_route_id
            ),
            "source_provider_route_fingerprint": (
                self.run_spec.strategy_bank.source_provider_route_fingerprint
            ),
            "provider_route_id": self.run_spec.provider_route_id,
            "provider_route_fingerprint": (
                self.run_spec.provider_route_fingerprint
            ),
        }
        for name in (
            "observation_contract_fingerprint",
            "task_instruction_fingerprint",
            "shared_prompt_contract_fingerprint",
            "decision_prompt_fingerprint",
            "skill_api_fingerprint",
            "full_method_compatibility_fingerprint",
        ):
            if hasattr(self.run_spec, name):
                binding_expected[name] = str(getattr(self.run_spec, name))
        for name, value in binding_expected.items():
            if binding.get(name) != value:
                raise MemoryRuntimeError(
                    f"bootstrap formal binding {name} does not match RunSpec"
                )
        plan_path = binding.get("plan_path")
        plan_sha256 = binding.get("plan_sha256")
        if not isinstance(plan_path, str) or not Path(plan_path).is_absolute():
            raise MemoryRuntimeError("formal binding requires an absolute plan_path")
        if not _is_sha256(plan_sha256):
            raise MemoryRuntimeError("formal binding requires plan_sha256")
        try:
            actual_plan_hash = hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise MemoryRuntimeError(f"cannot re-read pinned suite plan: {exc}") from exc
        if actual_plan_hash != plan_sha256:
            raise MemoryRuntimeError("suite plan hash changed after memory bootstrap")
        return binding

    def _begin(self, operation: str) -> None:
        _require_memory_root_integrity(self.root)
        self._audit.ensure_active(operation)
        self._guard_seed(f"{operation}:pre")

    def _guard_seed(self, stage: str) -> str:
        try:
            digest = self._seed_guard.verify(stage)
        except SeedIntegrityError as exc:
            self._violate(stage, str(exc))
        self._audit.set_seed_current(digest)
        return digest

    def _load_overlay(self) -> tuple[dict[str, Any], str]:
        payload = read_json_object(self.online_overlay_path, "online strategy overlay")
        validated = validate_online_strategy_overlay(payload)
        return validated, semantic_sha256(validated)

    def _load_actions(self) -> tuple[dict[str, Any], str]:
        payload = read_json_object(self.action_exemplars_path, "action exemplars")
        validated = validate_action_exemplar_memory(payload)
        return validated, semantic_sha256(validated)

    def _require_current(self, section: str, actual: str) -> None:
        expected = self._audit.current_hash(section)
        if actual != expected:
            self._violate(
                section,
                f"external mutation detected: {actual!r} != {expected!r}",
            )

    def _operation_failed(self, operation: str, exc: Exception) -> None:
        if self._audit.state == "violated":
            if isinstance(exc, MemoryRuntimeError):
                raise exc
            raise MemoryRuntimeError(str(exc)) from exc
        self._violate(operation, str(exc))

    def _violate(self, operation: str, detail: str) -> None:
        try:
            self._audit.violate(operation, detail)
        except MemoryLifecycleError as exc:
            raise MemoryRuntimeError(str(exc)) from exc


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MemoryRuntimeError(f"{label} must be an object")
    return value


def _entry_identity(entry: Mapping[str, Any]) -> tuple[int, str]:
    provenance = entry["provenance"]
    return int(provenance["transaction_index"]), str(provenance["ast_fingerprint"])


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[int, str]:
    return _entry_identity(entry)


def _action_identity(entry: Mapping[str, Any]) -> tuple[int, str]:
    return int(entry["transaction_index"]), str(entry["ast_fingerprint"])


def _action_sort_key(entry: Mapping[str, Any]) -> tuple[int, str]:
    return _action_identity(entry)


def _action_exemplar_matches(
    entry: Mapping[str, Any],
    query: ActionExemplarQuery,
) -> bool:
    intent = entry.get("intent")
    if not isinstance(intent, Mapping):
        return False
    commands = intent.get("commands")
    if not isinstance(commands, (list, tuple)) or not commands:
        return False
    expected = (
        _exact_text(query.primitive),
        _exact_text(query.semantic_objective),
        _exact_text(query.coordination_mode),
    )
    actual: list[tuple[str, str, str]] = []
    for command in commands:
        if not isinstance(command, Mapping):
            return False
        triple = (
            _exact_text(command.get("primitive")),
            _exact_text(command.get("objective")),
            _exact_text(command.get("mode")),
        )
        if not all(triple):
            return False
        actual.append(triple)
    return all(triple == expected for triple in actual)


def _action_retrieval_sort_key(
    entry: Mapping[str, Any],
    query: ActionExemplarQuery,
) -> tuple[int, int, str]:
    intent = entry.get("intent")
    phase = intent.get("lifecycle_phase") if isinstance(intent, Mapping) else None
    phase_rank = 0 if _exact_text(phase) == _exact_text(query.active_phase) else 1
    # Prefer recent verifier-accepted code within each phase rank.  The AST is
    # only a deterministic tiebreaker and never a semantic similarity proxy.
    transaction_index, ast_fingerprint = _action_identity(entry)
    return phase_rank, -transaction_index, ast_fingerprint


def _exact_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_memory_root_integrity(root: Path) -> None:
    absolute = root.absolute()
    if not absolute.exists() or not absolute.is_dir():
        raise MemoryRuntimeError("formal memory directory is missing")
    if any(path.is_symlink() for path in (absolute, *absolute.parents)):
        raise MemoryRuntimeError(
            "formal memory directory or one of its ancestors is a symlink"
        )
    for filename in (
        "memory_bootstrap_audit.json",
        "memory_bootstrap_snapshot.json",
        "seed_strategy_bank.json",
        "online_strategy_overlay.json",
        "action_exemplars.json",
        "accepted_episode_manifest.json",
        "failure_cache.json",
        "memory_runtime_audit.json",
    ):
        path = absolute / filename
        if path.is_symlink():
            raise MemoryRuntimeError(
                f"formal memory artifact is a symlink: {filename}"
            )


__all__ = [
    "FileBackedAdapterMemory",
    "MAX_RETRIEVED_ACTION_EXEMPLARS",
    "MemoryRuntimeError",
]
