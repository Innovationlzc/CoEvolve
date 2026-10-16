"""Audited operation gates for the frozen V83 native memory stores.

This module is deliberately an adapter, not a second memory implementation.
For ordinary Table-2/3 runs it delegates retrieval and promotion to the real
V83 ``StrategyMemoryStore`` and ``VerifiedActionMemoryStore`` supplied by the
production runner.  Table-4 transfer is the sole exception: its immutable
public six-field strategy bank cannot be represented as a native V83 spec
without inventing coordination mode, failure level, and evidence.  A
composite query therefore ranks the immutable seed records together with the
run-local native specs and renders the two schemas separately.  Promotions
always go to the run-local native stores.

The adapter owns only memory-operation policy and evidence.  It never imports
the frozen checkout, invokes a model, writes raw rollout evidence, mutates an
accepted-episode manifest, or participates in checkpoint/rollback logic.
Those remain responsibilities of the V83 runner.  Store factories are
injected so a reviewed production shim can use the exact frozen classes and
unit tests can use side-effect-accounting fakes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .json_io import atomic_write_json
from .retrieval import canonical_strategy_phase, select_strategy_records
from .validation import load_strategy_bank, validate_strategy_bank


V83_NATIVE_AUDIT_KIND = "coevolve_v83_memory_runtime_audit"
V83_NATIVE_AUDIT_SCHEMA_VERSION = 1
V83_NATIVE_AUDIT_FILENAME = "v83_native_memory_runtime_audit.json"
V83_NATIVE_STRATEGY_LIMIT = 3
V83_NATIVE_ACTION_LIMIT = 2

_VALID_MODES = frozenset({"from_empty", "disabled", "preloaded_seed"})
_VALID_TERMINALS = frozenset(
    {
        "SUCCESS",
        "SYSTEM_STALL",
        "PHYSICAL_BUDGET_EXHAUSTED",
        "CONSECUTIVE_PAYLOAD_DROP_LIMIT",
    }
)
_SOURCE_FIELD = "__coevolve_v83_bridge_source__"
_SOURCE_INDEX_FIELD = "__coevolve_v83_bridge_source_index__"


class V83NativeMemoryError(ValueError):
    """The native-memory operation contract or its evidence was violated."""


class NativeStrategyStore(Protocol):
    """Structural subset used from the frozen ``StrategyMemoryStore``."""

    @property
    def specs(self) -> Sequence[Mapping[str, Any]]:
        ...

    def retrieve(self, context: object, *, limit: int) -> list[str]:
        ...

    def update_from_rollout(
        self,
        record: dict[str, Any],
        *,
        default_phase: str,
        model_proposal: dict[str, Any] | None,
    ) -> bool:
        ...

    def save(self) -> None:
        ...


class NativeActionStore(Protocol):
    """Structural subset used from V83 ``VerifiedActionMemoryStore``."""

    @property
    def exemplars(self) -> Sequence[Mapping[str, Any]]:
        ...

    def retrieve(
        self,
        *,
        method: str,
        semantic_objective: dict[str, object] | None,
        coordination_mode: str,
        phase: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        ...

    def record_accepted(self, **kwargs: Any) -> bool:
        ...


StrategyStoreFactory = Callable[[Path], NativeStrategyStore]
ActionStoreFactory = Callable[[Path], NativeActionStore]
StrategyRenderer = Callable[[Mapping[str, Any]], str]
StrategySelector = Callable[..., tuple[Mapping[str, Any], ...]]
MemoryWriter = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class V83MemoryOperationGates:
    """The only four dynamic-memory switches admitted by the adapter."""

    strategy_retrieval: bool
    strategy_promotion: bool
    action_exemplar_retrieval: bool
    action_exemplar_promotion: bool

    @classmethod
    def enabled(cls) -> "V83MemoryOperationGates":
        return cls(True, True, True, True)

    @classmethod
    def disabled(cls) -> "V83MemoryOperationGates":
        return cls(False, False, False, False)

    @property
    def all_disabled(self) -> bool:
        return not any(asdict(self).values())


@dataclass(frozen=True, slots=True)
class _FileState:
    exists: bool
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"exists": self.exists, "sha256": self.sha256}


class V83NativeMemoryBridge:
    """Gate and audit the two real V83 run-local memory stores.

    ``mode`` has three meanings:

    * ``from_empty``: Table 2/3 online memory, initially empty;
    * ``disabled``: No-Dynamic/baseline memory, with all four gates closed;
    * ``preloaded_seed``: Table 4 immutable six-field seed plus an initially
      empty, independently mutable native online store.

    The native paths may either be absent (the V83 constructors' default) or
    contain valid empty native documents.  Non-empty native state is rejected
    for every new target run, including Table 4: transfer records live only in
    ``transfer_seed_path`` and are never copied into ``strategy_specs.json``.
    """

    def __init__(
        self,
        *,
        strategy_path: str | Path,
        action_exemplars_path: str | Path,
        audit_path: str | Path,
        mode: str,
        gates: V83MemoryOperationGates,
        strategy_store_factory: StrategyStoreFactory,
        action_store_factory: ActionStoreFactory,
        run_binding: Mapping[str, Any],
        transfer_seed_path: str | Path | None = None,
        native_strategy_renderer: StrategyRenderer | None = None,
        strategy_selector: StrategySelector = select_strategy_records,
    ) -> None:
        self.strategy_path = _checked_output_path(strategy_path, "strategy memory")
        self.action_path = _checked_output_path(
            action_exemplars_path, "action exemplar memory"
        )
        self.audit_path = _checked_output_path(audit_path, "runtime audit")
        if len({self.strategy_path, self.action_path, self.audit_path}) != 3:
            raise V83NativeMemoryError("native memory and audit paths must be distinct")
        if self.audit_path.exists():
            raise V83NativeMemoryError(
                f"native memory audit already exists: {self.audit_path}"
            )
        if mode not in _VALID_MODES:
            raise V83NativeMemoryError(f"unsupported native memory mode: {mode!r}")
        if not isinstance(gates, V83MemoryOperationGates):
            raise V83NativeMemoryError("gates must be V83MemoryOperationGates")
        if mode == "disabled" and not gates.all_disabled:
            raise V83NativeMemoryError("disabled memory requires all four gates closed")
        if mode != "disabled" and gates.all_disabled:
            raise V83NativeMemoryError(
                "all-closed gates must use the explicit disabled memory mode"
            )

        self.mode = mode
        self.gates = gates
        self._strategy_factory = strategy_store_factory
        self._action_factory = action_store_factory
        self._native_renderer = native_strategy_renderer or render_native_strategy_spec
        self._selector = strategy_selector
        self._audit_sha256 = ""

        self.transfer_seed_path: Path | None = None
        self._seed_records: tuple[dict[str, Any], ...] = ()
        self._seed_file_state: _FileState | None = None
        self._seed_semantic_sha256 = ""
        if mode == "preloaded_seed":
            if transfer_seed_path is None:
                raise V83NativeMemoryError(
                    "preloaded_seed mode requires an immutable transfer seed"
                )
            seed_path = _checked_input_path(transfer_seed_path, "transfer seed")
            if seed_path in {self.strategy_path, self.action_path, self.audit_path}:
                raise V83NativeMemoryError("transfer seed must be a separate file")
            validation = validate_strategy_bank(seed_path)
            bank = load_strategy_bank(seed_path)
            self.transfer_seed_path = seed_path
            self._seed_records = tuple(
                _json_object_copy(item) for item in bank.get("records", ())
            )
            self._seed_file_state = _file_state(seed_path)
            self._seed_semantic_sha256 = validation.bank_sha256
        elif transfer_seed_path is not None:
            raise V83NativeMemoryError(
                "a transfer seed is admitted only in preloaded_seed mode"
            )

        self._strategy_file_state = _file_state(self.strategy_path)
        self._action_file_state = _file_state(self.action_path)
        strategy_count = self._record_count(
            self.strategy_path,
            self._strategy_factory,
            "specs",
        )
        action_count = self._record_count(
            self.action_path,
            self._action_factory,
            "exemplars",
        )
        if strategy_count or action_count:
            raise V83NativeMemoryError(
                "a formal target must begin with empty native strategy/action stores"
            )

        binding = _json_object_copy(run_binding)
        self._audit: dict[str, Any] = {
            "kind": V83_NATIVE_AUDIT_KIND,
            "schema_version": V83_NATIVE_AUDIT_SCHEMA_VERSION,
            "lifecycle_state": "active",
            "finalized": False,
            "mode": mode,
            "run_binding": binding,
            "run_binding_sha256": _semantic_sha256(binding),
            "cross_run_retrieval": mode == "preloaded_seed",
            "operation_gates": asdict(gates),
            "initial_state": {
                "strategy_store": {
                    "path": str(self.strategy_path),
                    **self._strategy_file_state.as_dict(),
                    "record_count": strategy_count,
                },
                "action_exemplar_store": {
                    "path": str(self.action_path),
                    **self._action_file_state.as_dict(),
                    "record_count": action_count,
                },
                "transfer_seed": self._seed_audit_state(initial=True),
            },
            "operations": {
                "strategy_retrieval": {
                    "invocations": 0,
                    "returned_records": 0,
                    "native_records_returned": 0,
                    "transfer_seed_records_returned": 0,
                },
                "strategy_promotion": {
                    "attempts": 0,
                    "promotions": 0,
                    "noops": 0,
                    "errors": 0,
                },
                "action_exemplar_retrieval": {
                    "invocations": 0,
                    "returned_exemplars": 0,
                },
                "action_exemplar_promotion": {
                    "attempts": 0,
                    "promotions": 0,
                    "noops": 0,
                    "errors": 0,
                },
                "memory_writer": {"calls": 0, "failures": 0},
            },
            # Suppressed boundary calls prove a disabled route was exercised,
            # but are intentionally not counted as memory operations.
            "suppressed_boundary_calls": {
                "strategy_retrieval": 0,
                "strategy_promotion": 0,
                "action_exemplar_retrieval": 0,
                "action_exemplar_promotion": 0,
            },
            "operation_errors": [],
            "violations": [],
            "final_state": None,
            "terminal_reason": "",
        }
        self._persist_audit(initial=True)

    @property
    def audit(self) -> Mapping[str, Any]:
        """Return a detached JSON snapshot; callers cannot mutate the audit."""

        return _json_object_copy(self._audit)

    def retrieve_strategies(self, context: object) -> list[str]:
        """Return the frozen V83 top-three strategy prompt records."""

        self._ensure_active("retrieve strategies")
        if not self.gates.strategy_retrieval:
            self._record_suppressed("strategy_retrieval")
            return []
        self._guard_external_state("strategy retrieval")
        counters = self._audit["operations"]["strategy_retrieval"]
        counters["invocations"] += 1
        try:
            store = self._strategy_factory(self.strategy_path)
            if self.transfer_seed_path is None:
                rendered = list(
                    store.retrieve(context, limit=V83_NATIVE_STRATEGY_LIMIT)
                )
                native_returned = len(rendered)
                seed_returned = 0
            else:
                native_records = tuple(
                    _json_object_copy(item) for item in store.specs
                )
                rendered, native_returned, seed_returned = self._composite_retrieve(
                    context,
                    native_records=native_records,
                )
            if len(rendered) > V83_NATIVE_STRATEGY_LIMIT or not all(
                isinstance(item, str) and item.strip() for item in rendered
            ):
                self._violate(
                    "strategy retrieval",
                    "native/composite retrieval violated the frozen top-3 text contract",
                )
            self._assert_no_store_write("strategy retrieval")
            counters["returned_records"] += len(rendered)
            counters["native_records_returned"] += native_returned
            counters["transfer_seed_records_returned"] += seed_returned
            self._persist_audit()
            return rendered
        except V83NativeMemoryError:
            raise
        except Exception as exc:
            self._assert_no_store_write("failed strategy retrieval")
            self._record_operation_error("strategy retrieval", exc)
            raise V83NativeMemoryError(
                f"native strategy retrieval failed: {type(exc).__name__}: {exc}"
            ) from exc

    def retrieve_action_exemplars(
        self,
        *,
        method: str,
        semantic_objective: dict[str, object] | None = None,
        coordination_mode: str = "",
        phase: str = "",
    ) -> list[dict[str, Any]]:
        """Delegate the exact V83 action query with its fixed top-two cap."""

        self._ensure_active("retrieve action exemplars")
        if not self.gates.action_exemplar_retrieval:
            self._record_suppressed("action_exemplar_retrieval")
            return []
        self._guard_external_state("action exemplar retrieval")
        counters = self._audit["operations"]["action_exemplar_retrieval"]
        counters["invocations"] += 1
        try:
            store = self._action_factory(self.action_path)
            selected = list(
                store.retrieve(
                    method=method,
                    semantic_objective=semantic_objective,
                    coordination_mode=coordination_mode,
                    phase=phase,
                    limit=V83_NATIVE_ACTION_LIMIT,
                )
            )
            if len(selected) > V83_NATIVE_ACTION_LIMIT or not all(
                isinstance(item, Mapping) for item in selected
            ):
                self._violate(
                    "action exemplar retrieval",
                    "native action retrieval violated the frozen top-2 object contract",
                )
            self._assert_no_store_write("action exemplar retrieval")
            counters["returned_exemplars"] += len(selected)
            self._persist_audit()
            return [_json_object_copy(item) for item in selected]
        except V83NativeMemoryError:
            raise
        except Exception as exc:
            self._assert_no_store_write("failed action exemplar retrieval")
            self._record_operation_error("action exemplar retrieval", exc)
            raise V83NativeMemoryError(
                f"native action retrieval failed: {type(exc).__name__}: {exc}"
            ) from exc

    def promote_strategy(
        self,
        record: Mapping[str, Any],
        *,
        default_phase: str,
        raw_evidence_persisted: bool,
        memory_writer: MemoryWriter | None = None,
        memory_writer_payload: Mapping[str, Any] | None = None,
        model_proposal: Mapping[str, Any] | None = None,
    ) -> bool:
        """Run the optional Writer and native V83 promotion behind one gate.

        The caller must first durably append the authoritative raw rollout.
        Writer failures follow V83 semantics: they are counted, converted to a
        ``promote=false`` proposal, and do not fail the physical task.
        """

        self._ensure_active("promote strategy")
        if not self.gates.strategy_promotion:
            self._record_suppressed("strategy_promotion")
            return False
        if raw_evidence_persisted is not True:
            raise V83NativeMemoryError(
                "strategy promotion requires durable raw physical evidence first"
            )
        if memory_writer is not None and model_proposal is not None:
            raise V83NativeMemoryError(
                "supply either a Memory Writer or a precomputed proposal, not both"
            )
        self._guard_external_state("strategy promotion")
        counters = self._audit["operations"]["strategy_promotion"]
        counters["attempts"] += 1
        proposal = (
            _json_object_copy(model_proposal)
            if model_proposal is not None
            else None
        )
        if memory_writer is not None:
            writer_counters = self._audit["operations"]["memory_writer"]
            writer_counters["calls"] += 1
            try:
                candidate = memory_writer(
                    _json_object_copy(memory_writer_payload or {})
                )
                proposal = (
                    _json_object_copy(candidate) if candidate is not None else None
                )
            except Exception as exc:  # V83 keeps Writer outages non-fatal.
                writer_counters["failures"] += 1
                proposal = {
                    "promote": False,
                    "source": "memory_update_unavailable",
                }
                self._audit["operation_errors"].append(
                    _error_entry("memory writer", exc)
                )
        before = self._strategy_file_state
        try:
            store = self._strategy_factory(self.strategy_path)
            changed = bool(
                store.update_from_rollout(
                    _json_object_copy(record),
                    default_phase=str(default_phase),
                    model_proposal=proposal,
                )
            )
            if changed:
                store.save()
            after = _file_state(self.strategy_path)
            self._guard_seed("strategy promotion postcondition")
            if not changed and after != before:
                self._violate(
                    "strategy promotion",
                    "native store changed bytes while reporting a promotion no-op",
                )
            if changed and (not after.exists or after == before):
                self._violate(
                    "strategy promotion",
                    "native store reported promotion without a durable byte change",
                )
            self._strategy_file_state = after
            if changed:
                self._validated_native_count("strategy", expected_nonempty=True)
                counters["promotions"] += 1
            else:
                counters["noops"] += 1
            self._persist_audit()
            return changed
        except V83NativeMemoryError:
            raise
        except Exception as exc:
            counters["errors"] += 1
            self._assert_failed_promotion_did_not_write(
                "failed strategy promotion",
                expected_strategy=before,
                expected_action=self._action_file_state,
            )
            self._record_operation_error("strategy promotion", exc)
            # Match V83: invalid/unavailable active-memory promotion cannot
            # retroactively fail the already finalized physical rollout.
            return False

    def promote_action_exemplar(
        self,
        *,
        raw_evidence_persisted: bool,
        record_kwargs: Mapping[str, Any],
    ) -> bool:
        """Delegate V83 ``record_accepted`` behind the exemplar gate."""

        self._ensure_active("promote action exemplar")
        if not self.gates.action_exemplar_promotion:
            self._record_suppressed("action_exemplar_promotion")
            return False
        if raw_evidence_persisted is not True:
            raise V83NativeMemoryError(
                "action-exemplar promotion requires durable raw evidence first"
            )
        self._guard_external_state("action exemplar promotion")
        counters = self._audit["operations"]["action_exemplar_promotion"]
        counters["attempts"] += 1
        before = self._action_file_state
        try:
            store = self._action_factory(self.action_path)
            changed = bool(store.record_accepted(**dict(record_kwargs)))
            after = _file_state(self.action_path)
            self._guard_seed("action exemplar promotion postcondition")
            if not changed and after != before:
                self._violate(
                    "action exemplar promotion",
                    "native store changed bytes while reporting a promotion no-op",
                )
            if changed and (not after.exists or after == before):
                self._violate(
                    "action exemplar promotion",
                    "native store reported promotion without a durable byte change",
                )
            self._action_file_state = after
            if changed:
                self._validated_native_count("action", expected_nonempty=True)
                counters["promotions"] += 1
            else:
                counters["noops"] += 1
            self._persist_audit()
            return changed
        except V83NativeMemoryError:
            raise
        except Exception as exc:
            counters["errors"] += 1
            self._assert_failed_promotion_did_not_write(
                "failed action exemplar promotion",
                expected_strategy=self._strategy_file_state,
                expected_action=before,
            )
            self._record_operation_error("action exemplar promotion", exc)
            return False

    def finalize(self, *, terminal_reason: str) -> Mapping[str, Any]:
        """Finalize a receipt only for a protocol-valid evaluation terminal."""

        if self._audit.get("finalized") is True:
            if terminal_reason != self._audit.get("terminal_reason"):
                raise V83NativeMemoryError(
                    "finalized native audit cannot change terminal reason"
                )
            return self.audit
        self._ensure_active("finalize native memory audit")
        if terminal_reason not in _VALID_TERMINALS:
            raise V83NativeMemoryError(
                "infrastructure/contract fatals cannot receive a finalized memory receipt"
            )
        self._guard_external_state("finalization")
        strategy_count = self._validated_native_count("strategy")
        action_count = self._validated_native_count("action")
        self._validate_final_invariants(strategy_count, action_count)
        self._audit["final_state"] = {
            "strategy_store": {
                "path": str(self.strategy_path),
                **self._strategy_file_state.as_dict(),
                "record_count": strategy_count,
            },
            "action_exemplar_store": {
                "path": str(self.action_path),
                **self._action_file_state.as_dict(),
                "record_count": action_count,
            },
            "transfer_seed": self._seed_audit_state(initial=False),
        }
        self._audit["terminal_reason"] = terminal_reason
        self._audit["lifecycle_state"] = "finalized"
        self._audit["finalized"] = True
        self._persist_audit()
        return self.audit

    def mark_incomplete(self, *, reason: str) -> Mapping[str, Any]:
        """Close an invalid infrastructure attempt without signing a receipt."""

        self._ensure_active("mark native memory audit incomplete")
        if not str(reason).strip():
            raise V83NativeMemoryError("incomplete audit requires a reason")
        self._audit["lifecycle_state"] = "incomplete"
        self._audit["terminal_reason"] = str(reason)
        self._audit["finalized"] = False
        self._persist_audit()
        return self.audit

    def _composite_retrieve(
        self,
        context: object,
        *,
        native_records: Sequence[Mapping[str, Any]],
    ) -> tuple[list[str], int, int]:
        seed_candidates: list[dict[str, Any]] = []
        for index, record in enumerate(self._seed_records):
            seed_candidates.append(
                {
                    "proposal": _json_object_copy(record),
                    "retrieval": {
                        "phase": canonical_strategy_phase(
                            record.get("phase_trigger", "any")
                        )
                    },
                    "provenance": {
                        "verifier_outcome": record.get("verified_outcome", "")
                    },
                    _SOURCE_FIELD: "transfer_seed",
                    _SOURCE_INDEX_FIELD: index,
                }
            )
        online_candidates: list[dict[str, Any]] = []
        for index, record in enumerate(native_records):
            candidate = _json_object_copy(record)
            candidate[_SOURCE_FIELD] = "native_online"
            candidate[_SOURCE_INDEX_FIELD] = index
            online_candidates.append(candidate)
        # Seed comes first, native online records second.  V83 resolves a full
        # tie by later source order, so newly learned run-local evidence wins a
        # tie without changing any higher-priority criterion.
        selected = self._selector(
            (*seed_candidates, *online_candidates),
            active_phase=_context_field(context, "phase"),
            failure_code=_context_field(context, "last_failure_type"),
            failure_level=_context_field(context, "last_failure_level"),
            failure_report=_context_field(context, "task_stage"),
            limit=V83_NATIVE_STRATEGY_LIMIT,
        )
        rendered: list[str] = []
        native_count = 0
        seed_count = 0
        for selected_record in selected:
            source = str(selected_record.get(_SOURCE_FIELD, ""))
            if source == "transfer_seed":
                payload = selected_record.get("proposal")
                if not isinstance(payload, Mapping):
                    raise V83NativeMemoryError(
                        "composite selector lost the six-field seed payload"
                    )
                rendered.append(render_transfer_strategy_record(payload))
                seed_count += 1
            elif source == "native_online":
                native = {
                    key: value
                    for key, value in selected_record.items()
                    if key not in {_SOURCE_FIELD, _SOURCE_INDEX_FIELD}
                }
                rendered.append(self._native_renderer(native))
                native_count += 1
            else:
                raise V83NativeMemoryError(
                    "composite selector returned a record from an unknown source"
                )
        return rendered, native_count, seed_count

    def _record_suppressed(self, operation: str) -> None:
        # Deliberately no store/seed hash, factory construction, or Writer
        # invocation occurs on this path.  Only the independent audit changes.
        self._audit["suppressed_boundary_calls"][operation] += 1
        self._persist_audit()

    def _guard_external_state(self, operation: str) -> None:
        self._assert_audit_intact()
        try:
            strategy = _file_state(self.strategy_path)
            action = _file_state(self.action_path)
        except V83NativeMemoryError as exc:
            self._violate(operation, str(exc))
        if strategy != self._strategy_file_state:
            self._violate(operation, "external strategy-store byte change detected")
        if action != self._action_file_state:
            self._violate(operation, "external action-store byte change detected")
        self._guard_seed(operation)

    def _guard_seed(self, operation: str) -> None:
        if self.transfer_seed_path is None:
            return
        try:
            current = _file_state(self.transfer_seed_path)
        except V83NativeMemoryError as exc:
            self._violate(operation, str(exc))
        if current != self._seed_file_state:
            self._violate(operation, "immutable transfer-seed byte change detected")

    def _assert_no_store_write(self, operation: str) -> None:
        strategy = _file_state(self.strategy_path)
        action = _file_state(self.action_path)
        if strategy != self._strategy_file_state or action != self._action_file_state:
            self._violate(operation, "a retrieval operation wrote native store bytes")
        self._guard_seed(f"{operation} postcondition")

    def _assert_failed_promotion_did_not_write(
        self,
        operation: str,
        *,
        expected_strategy: _FileState,
        expected_action: _FileState,
    ) -> None:
        if (
            _file_state(self.strategy_path) != expected_strategy
            or _file_state(self.action_path) != expected_action
        ):
            self._violate(
                operation,
                "a failed native promotion left an unaudited store-byte change",
            )
        self._guard_seed(f"{operation} postcondition")

    def _validated_native_count(
        self,
        which: str,
        *,
        expected_nonempty: bool = False,
    ) -> int:
        if which == "strategy":
            path = self.strategy_path
            factory: Callable[[Path], object] = self._strategy_factory
            attribute = "specs"
        elif which == "action":
            path = self.action_path
            factory = self._action_factory
            attribute = "exemplars"
        else:  # pragma: no cover - private programming error
            raise AssertionError(which)
        state = _file_state(path)
        count = self._record_count(path, factory, attribute)
        if expected_nonempty and count < 1:
            self._violate(
                f"validate {which} promotion",
                f"{which} promotion changed bytes but produced no native records",
            )
        if which == "strategy":
            self._strategy_file_state = state
        else:
            self._action_file_state = state
        return count

    @staticmethod
    def _record_count(
        path: Path,
        factory: Callable[[Path], object],
        attribute: str,
    ) -> int:
        if not path.exists():
            return 0
        store = factory(path)
        records = getattr(store, attribute, None)
        if isinstance(records, (str, bytes, Mapping)) or not isinstance(
            records, Sequence
        ):
            raise V83NativeMemoryError(
                f"native store {attribute} must expose a record sequence"
            )
        if not all(isinstance(item, Mapping) for item in records):
            raise V83NativeMemoryError(
                f"native store {attribute} contains a non-object record"
            )
        return len(records)

    def _validate_final_invariants(
        self,
        strategy_count: int,
        action_count: int,
    ) -> None:
        if self.mode != "disabled":
            return
        operations = self._audit["operations"]
        primary_counts = (
            operations["strategy_retrieval"]["invocations"],
            operations["strategy_promotion"]["attempts"],
            operations["action_exemplar_retrieval"]["invocations"],
            operations["action_exemplar_promotion"]["attempts"],
            operations["memory_writer"]["calls"],
        )
        if any(primary_counts) or strategy_count or action_count:
            self._violate(
                "finalize disabled memory",
                "No-Dynamic must finish with zero memory operations, Writer calls, and records",
            )

    def _seed_audit_state(self, *, initial: bool) -> Mapping[str, Any] | None:
        if self.transfer_seed_path is None or self._seed_file_state is None:
            return None
        return {
            "path": str(self.transfer_seed_path),
            "file_sha256": self._seed_file_state.sha256,
            "semantic_sha256": self._seed_semantic_sha256,
            "record_count": len(self._seed_records),
            "schema_projection": "immutable_public_six_field_strategy_bank",
            "native_mode_invented": False,
            "immutable_verified": True,
            "verification_point": "initial" if initial else "final",
        }

    def _ensure_active(self, operation: str) -> None:
        state = str(self._audit.get("lifecycle_state", ""))
        if state != "active":
            raise V83NativeMemoryError(
                f"cannot {operation}: native memory lifecycle is {state}"
            )

    def _record_operation_error(self, operation: str, exc: Exception) -> None:
        self._audit["operation_errors"].append(_error_entry(operation, exc))
        try:
            self._persist_audit()
        except V83NativeMemoryError:
            raise

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
        self._persist_audit()
        raise V83NativeMemoryError(detail)

    def _assert_audit_intact(self) -> None:
        if not self._audit_sha256:
            return
        current = _file_state(self.audit_path)
        if not current.exists or current.sha256 != self._audit_sha256:
            raise V83NativeMemoryError(
                "native runtime audit was externally modified; refusing overwrite"
            )

    def _persist_audit(self, *, initial: bool = False) -> None:
        if not initial:
            self._assert_audit_intact()
        atomic_write_json(self.audit_path, self._audit)
        self._audit_sha256 = _file_state(self.audit_path).sha256


def render_native_strategy_spec(spec: Mapping[str, Any]) -> str:
    """Render one native record exactly as frozen V83 ``render_strategy_spec``."""

    evidence = spec.get("evidence", {})
    evidence = evidence if isinstance(evidence, Mapping) else {}
    failures = evidence.get("failure_counts", {})
    failures = failures if isinstance(failures, Mapping) else {}
    failure_text = ", ".join(
        f"{key}:{value}" for key, value in sorted(failures.items())
    )
    forbidden = "; ".join(str(item) for item in spec.get("forbidden", ()))
    return (
        f"[deployment_spec id={spec['id']} phase={spec['phase']} "
        f"observed_mode={spec['observed_coordination_mode']} "
        f"primitive={spec.get('observed_coordination_primitive') or 'legacy'} "
        f"failure_level={spec['failure_level']} "
        f"confidence={float(spec.get('confidence', 0.0)):.2f}]\n"
        f"Condition: {spec['condition']}\n"
        f"Diagnosis: {spec['diagnosis']}\n"
        f"Decision: {spec['decision']}\n"
        f"Forbidden: {forbidden}\n"
        f"Repair: {spec['repair']}\n"
        "Verifier evidence: "
        f"accepted={int(evidence.get('accepted_count', 0))}, "
        f"rejected={int(evidence.get('rejected_count', 0))}"
        + (f", failures={failure_text}" if failure_text else "")
    )


def render_transfer_strategy_record(record: Mapping[str, Any]) -> str:
    """Render a public Table-4 record without fabricating native-only fields."""

    required = {
        "phase_trigger",
        "failure_condition",
        "coordination_pattern",
        "verified_outcome",
        "avoid_rule",
        "repair_direction",
    }
    if set(record) != required:
        raise V83NativeMemoryError("transfer record is not the exact six-field schema")
    return (
        "[transfer_strategy schema=public_six_field "
        f"phase={canonical_strategy_phase(record['phase_trigger'])} "
        f"verified_outcome={record['verified_outcome']}]\n"
        f"Phase trigger: {record['phase_trigger']}\n"
        f"Failure condition: {record['failure_condition']}\n"
        f"Coordination pattern: {record['coordination_pattern']}\n"
        f"Avoid: {record['avoid_rule']}\n"
        f"Repair direction: {record['repair_direction']}"
    )


def _checked_output_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise V83NativeMemoryError(f"{label} path must be absolute")
    if path.is_symlink():
        raise V83NativeMemoryError(f"{label} path must not be a symlink")
    if path.exists() and not path.is_file():
        raise V83NativeMemoryError(f"{label} path must be a regular file")
    return path.resolve(strict=False)


def _checked_input_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise V83NativeMemoryError(
            f"{label} path must be an absolute non-symlink regular file"
        )
    return path.resolve(strict=True)


def _file_state(path: Path) -> _FileState:
    if path.is_symlink():
        raise V83NativeMemoryError(f"memory artifact became a symlink: {path}")
    if not path.exists():
        return _FileState(False, "")
    if not path.is_file():
        raise V83NativeMemoryError(f"memory artifact is not a regular file: {path}")
    return _FileState(True, hashlib.sha256(path.read_bytes()).hexdigest())


def _context_field(context: object, field: str) -> str:
    if isinstance(context, Mapping):
        return str(context.get(field, "")).strip()
    return str(getattr(context, field, "")).strip()


def _json_object_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V83NativeMemoryError("audit/native-memory value must be an object")
    try:
        copied = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise V83NativeMemoryError(f"value is not finite JSON: {exc}") from exc
    if not isinstance(copied, dict):  # pragma: no cover - guaranteed by input
        raise V83NativeMemoryError("JSON object copy changed root type")
    return copied


def _semantic_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _error_entry(operation: str, exc: Exception) -> dict[str, str]:
    return {
        "operation": operation,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


__all__ = [
    "ActionStoreFactory",
    "MemoryWriter",
    "NativeActionStore",
    "NativeStrategyStore",
    "StrategyStoreFactory",
    "V83MemoryOperationGates",
    "V83NativeMemoryBridge",
    "V83NativeMemoryError",
    "V83_NATIVE_ACTION_LIMIT",
    "V83_NATIVE_AUDIT_FILENAME",
    "V83_NATIVE_AUDIT_KIND",
    "V83_NATIVE_AUDIT_SCHEMA_VERSION",
    "V83_NATIVE_STRATEGY_LIMIT",
    "render_native_strategy_spec",
    "render_transfer_strategy_record",
]
