"""Closed-world lifecycle seam for the post-Table-1 V83 production driver.

This module deliberately does not make :mod:`production.runner` executable.
It supplies the orchestration boundary that the eventual reviewed worker
entrypoints must implement:

* all four workers receive the same, already verified ``--launch-config`` and
  no caller-authored argument tail;
* Term1 is given a runtime class whose evidence recorder arguments are bound by
  a closure (the frozen launcher does not pass those arguments itself);
* Term4 proves the exact method/prompt/memory identity it ran;
* logical FM requests and their individual HTTP/transport attempts are
  collected as records, never as a caller-authored aggregate;
* the native V83 memory audit and MuJoCo terminal trace are revalidated before
  one immutable evidence handoff; and
* every started worker is stopped in reverse order on every exit path.

The concrete V83 worker entrypoints, readiness probe, and Term4 raw-artifact
collector are intentionally injected.  Until those production implementations
exist and pass end-to-end collector tests, ``ProductionDriverBinding`` remains
``execution_ready == False`` and ``production.runner`` must retain exit 78.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import time
from types import MappingProxyType
from typing import Any, BinaryIO, Mapping, Protocol, Sequence, runtime_checkable

from ..contracts import MethodVariant
from ..decision_prompts import FROZEN_V83_PROMPT_BASE_SYMBOLS
from ..live.contracts import PolicyRepresentation
from ..memory.v83_native import V83MemoryOperationGates
from ..reporting.v83_native_memory_audit import (
    validate_v83_native_memory_audit,
)
from ..serialization import sha256_json, to_primitive
from .evidence_writer import (
    EvidenceProducer,
    FoundationModelCall,
    FoundationModelTransportAttempt,
    ManifestArtifactPin,
    PayloadDropEvidence,
    ProductionEvidenceWriteResult,
    ProductionRunAccounting,
    SystemStallEvidence,
    TransactionAccountingEvidence,
    write_production_result_evidence,
)
from .runtime_instrumentation import (
    MuJoCoTerminalTraceRecorder,
    RuntimeTraceBinding,
    TerminalEvidenceDocuments,
    instrument_runtime_class,
    load_live_terminal_snapshot,
)
from .session import ProductionSessionBinding


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VALID_TERMINALS = frozenset(
    {
        "SUCCESS",
        "SYSTEM_STALL",
        "PHYSICAL_BUDGET_EXHAUSTED",
        "CONSECUTIVE_PAYLOAD_DROP_LIMIT",
    }
)
_FULL_METHOD_ROLES = (
    "task_orchestrator",
    "coder",
    "evolution_manager",
    "mutation_operator",
    "memory_update",
)
_COMMON_RUNTIME_ROLES = ("vlm_failure_analyzer", "code_repair")
_COMMON_ROLE_OUTPUT_CONTRACTS: Mapping[str, str] = MappingProxyType(
    {
        "vlm_failure_analyzer": "v83_vlm_failure_analyzer_json_v1",
        "code_repair": "v83_team_policy_json_v1",
    }
)
_FULL_ROLE_PROMPT_BASES: Mapping[str, str] = MappingProxyType(
    {
        "task_orchestrator": "frozen_v83_task_orchestrator",
        "coder": "frozen_v83_policy_coder",
        "evolution_manager": "frozen_v83_evolution_manager",
        "mutation_operator": "frozen_v83_mutation_operator",
        "memory_update": "frozen_v83_memory_update",
        "vlm_failure_analyzer": "frozen_v83_vlm_failure_analyzer",
        "code_repair": "frozen_v83_code_repair",
    }
)


class ProductionDriverContractError(ValueError):
    """A worker/collector result does not match the immutable launch."""


class ProductionDriverInfrastructureError(RuntimeError):
    """A managed worker or infrastructure collector failed."""


class ProductionWorkerRole(str, Enum):
    TERM1 = "term1"
    TERM2_FRONT = "term2_front"
    TERM3_BACK = "term3_back"
    TERM4 = "term4"


_START_ORDER = (
    ProductionWorkerRole.TERM1,
    ProductionWorkerRole.TERM2_FRONT,
    ProductionWorkerRole.TERM3_BACK,
    ProductionWorkerRole.TERM4,
)
_CANONICAL_WORKER_ENTRYPOINT_RELPATHS: Mapping[
    ProductionWorkerRole, PurePosixPath
] = MappingProxyType(
    {
        ProductionWorkerRole.TERM1: PurePosixPath(
            "src/coevolve_experiments/production/workers/term1.py"
        ),
        ProductionWorkerRole.TERM2_FRONT: PurePosixPath(
            "src/coevolve_experiments/production/workers/term2_front.py"
        ),
        ProductionWorkerRole.TERM3_BACK: PurePosixPath(
            "src/coevolve_experiments/production/workers/term3_back.py"
        ),
        ProductionWorkerRole.TERM4: PurePosixPath(
            "src/coevolve_experiments/production/workers/term4.py"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class LiveRolePromptBinding:
    """One enabled role and the exact prompt source it must use."""

    role: str
    prompt_base: str
    output_contract: str
    overlay_sha256: str | None

    def __post_init__(self) -> None:
        for name in ("role", "prompt_base", "output_contract"):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ).strip():
                raise ProductionDriverContractError(
                    f"live role {name} must be non-empty"
                )
        if self.overlay_sha256 is not None and not _is_sha(self.overlay_sha256):
            raise ProductionDriverContractError(
                "live role overlay_sha256 must be SHA-256 or None"
            )


@dataclass(frozen=True, slots=True)
class LiveMethodPipelineBinding:
    """Method-specific Praxis Code-as-Policy identity consumed by Term4."""

    method_variant: MethodVariant
    policy_representation: PolicyRepresentation
    decision_structure: str
    decision_prompt_fingerprint: str
    shared_prompt_contract_fingerprint: str
    frozen_v83_prompt_sha256: str
    roles: tuple[LiveRolePromptBinding, ...]
    common_runtime_roles: tuple[str, ...]
    forbidden_method_roles: tuple[str, ...]
    memory_operation_gates: V83MemoryOperationGates
    memory_writer_call: bool

    def __post_init__(self) -> None:
        if self.policy_representation is not PolicyRepresentation.PRAXIS_CODE_AS_POLICY:
            raise ProductionDriverContractError(
                "live method pipeline must retain Praxis Code-as-Policy"
            )
        if not self.decision_structure.strip():
            raise ProductionDriverContractError(
                "live method decision structure must be non-empty"
            )
        for label, value in (
            ("decision prompt", self.decision_prompt_fingerprint),
            ("shared prompt", self.shared_prompt_contract_fingerprint),
            ("frozen V83 prompt", self.frozen_v83_prompt_sha256),
        ):
            if not _is_sha(value):
                raise ProductionDriverContractError(f"{label} must be SHA-256")
        role_names = tuple(item.role for item in self.roles)
        if not role_names or len(role_names) != len(set(role_names)):
            raise ProductionDriverContractError(
                "live method roles must be non-empty and unique"
            )
        if len(self.common_runtime_roles) != len(set(self.common_runtime_roles)):
            raise ProductionDriverContractError("common runtime roles must be unique")
        if len(self.forbidden_method_roles) != len(set(self.forbidden_method_roles)):
            raise ProductionDriverContractError("forbidden method roles must be unique")
        if set(role_names).intersection(self.forbidden_method_roles):
            raise ProductionDriverContractError(
                "a live method role is also declared forbidden"
            )
        if not isinstance(self.memory_operation_gates, V83MemoryOperationGates):
            raise ProductionDriverContractError(
                "live method memory gates have the wrong type"
            )
        if type(self.memory_writer_call) is not bool:
            raise ProductionDriverContractError(
                "live method memory_writer_call must be boolean"
            )

    @property
    def fingerprint(self) -> str:
        return sha256_json(to_primitive(self))


@dataclass(frozen=True, slots=True)
class ProductionWorkerInvocation:
    """One fixed worker identity derived only from the consumed session."""

    role: ProductionWorkerRole
    run_id: str
    launch_config_path: Path
    launch_config_sha256: str
    injection_bundle_fingerprint: str
    method_pipeline_fingerprint: str
    log_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.role, ProductionWorkerRole):
            raise ProductionDriverContractError("worker role is invalid")
        if not self.run_id.strip():
            raise ProductionDriverContractError("worker run_id is empty")
        if not self.launch_config_path.is_absolute():
            raise ProductionDriverContractError(
                "worker launch-config path must be absolute"
            )
        for label, value in (
            ("launch config", self.launch_config_sha256),
            ("injection bundle", self.injection_bundle_fingerprint),
            ("method pipeline", self.method_pipeline_fingerprint),
        ):
            if not _is_sha(value):
                raise ProductionDriverContractError(f"worker {label} hash is invalid")
        if not self.log_path.is_absolute():
            raise ProductionDriverContractError("worker log path must be absolute")

    @property
    def fingerprint(self) -> str:
        return sha256_json(to_primitive(self))


@dataclass(frozen=True, slots=True)
class ProductionDriverBinding:
    """Read-only driver input compiled from one consumed production session."""

    session: ProductionSessionBinding
    output_dir: Path
    terminal_snapshot_path: Path
    method_pipeline: LiveMethodPipelineBinding
    workers: tuple[ProductionWorkerInvocation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.session, ProductionSessionBinding):
            raise ProductionDriverContractError(
                "driver binding requires a ProductionSessionBinding"
            )
        if tuple(item.role for item in self.workers) != _START_ORDER:
            raise ProductionDriverContractError(
                "driver worker set/order must be exact Term1, Term2, Term3, Term4"
            )
        if len({item.fingerprint for item in self.workers}) != len(self.workers):
            raise ProductionDriverContractError("driver worker bindings are not unique")

    @property
    def execution_ready(self) -> bool:
        """Remain false until the actual V83 workers and collectors are reviewed."""

        return False

    def worker(self, role: ProductionWorkerRole) -> ProductionWorkerInvocation:
        selected = tuple(item for item in self.workers if item.role is role)
        if len(selected) != 1:
            raise ProductionDriverContractError(
                f"worker role does not resolve exactly once: {role.value}"
            )
        return selected[0]


@dataclass(frozen=True, slots=True)
class ProductionTerm4Completion:
    """Raw-collector result after a zero-exit Term4 worker."""

    run_id: str
    launch_config_sha256: str
    injection_bundle_fingerprint: str
    method_pipeline_fingerprint: str
    terminal_status: str
    accounting: ProductionRunAccounting
    transaction_accounting_evidence: TransactionAccountingEvidence
    system_stall: SystemStallEvidence | None = None
    payload_drop: PayloadDropEvidence | None = None
    additional_artifact_pins: tuple[ManifestArtifactPin, ...] = ()


@dataclass(frozen=True, slots=True)
class ProductionEvidenceHandoff:
    """Complete, still-unpublished input to the immutable evidence writer."""

    binding: ProductionDriverBinding
    completion: ProductionTerm4Completion
    terminal_documents: TerminalEvidenceDocuments
    fm_calls: tuple[FoundationModelCall, ...]


@dataclass(frozen=True, slots=True)
class ProductionDriverReceipt:
    run_id: str
    terminal_status: str
    worker_start_order: tuple[ProductionWorkerRole, ...]
    worker_stop_order: tuple[ProductionWorkerRole, ...]
    evidence: ProductionEvidenceWriteResult


@dataclass(frozen=True, slots=True)
class ProductionDriverFailure:
    """Secret-free infrastructure failure passed to an immutable archiver."""

    run_id: str
    launch_config_sha256: str
    injection_bundle_fingerprint: str
    stage: str
    error_type: str
    error: str
    started_roles: tuple[str, ...]
    stopped_roles: tuple[str, ...]


@runtime_checkable
class ManagedProductionWorker(Protocol):
    role: ProductionWorkerRole

    def poll(self) -> int | None:
        ...

    def wait(self, timeout_s: float | None = None) -> int:
        ...

    def terminate(self) -> None:
        ...

    def kill(self) -> None:
        ...


@runtime_checkable
class ProductionWorkerLauncher(Protocol):
    def start(
        self,
        invocation: ProductionWorkerInvocation,
        *,
        session: ProductionSessionBinding,
    ) -> ManagedProductionWorker:
        ...


@runtime_checkable
class ProductionStackReadiness(Protocol):
    def wait_ready(
        self,
        binding: ProductionDriverBinding,
        workers: Mapping[ProductionWorkerRole, ManagedProductionWorker],
    ) -> None:
        ...


@runtime_checkable
class ProductionTerm4Collector(Protocol):
    def collect(
        self,
        binding: ProductionDriverBinding,
    ) -> ProductionTerm4Completion:
        ...


@runtime_checkable
class FoundationModelCallCollector(Protocol):
    def collect(
        self,
        binding: ProductionDriverBinding,
    ) -> Sequence[FoundationModelCall]:
        ...


@runtime_checkable
class ProductionTerminalCollector(Protocol):
    def collect(
        self,
        binding: ProductionDriverBinding,
    ) -> TerminalEvidenceDocuments:
        ...


@runtime_checkable
class ProductionEvidencePublisher(Protocol):
    def publish(
        self,
        handoff: ProductionEvidenceHandoff,
    ) -> ProductionEvidenceWriteResult:
        ...


@runtime_checkable
class ProductionFailureArchiver(Protocol):
    def archive(
        self,
        binding: ProductionDriverBinding,
        failure: ProductionDriverFailure,
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ProductionDriverPorts:
    launcher: ProductionWorkerLauncher
    readiness: ProductionStackReadiness
    term4_collector: ProductionTerm4Collector
    fm_call_collector: FoundationModelCallCollector
    terminal_collector: ProductionTerminalCollector
    evidence_publisher: ProductionEvidencePublisher
    failure_archiver: ProductionFailureArchiver
    term4_timeout_s: float | None = None
    shutdown_grace_s: float = 10.0

    def __post_init__(self) -> None:
        if self.term4_timeout_s is not None and (
            isinstance(self.term4_timeout_s, bool)
            or not isinstance(self.term4_timeout_s, (int, float))
            or self.term4_timeout_s <= 0
        ):
            raise ProductionDriverContractError(
                "Term4 timeout must be positive or None"
            )
        if (
            isinstance(self.shutdown_grace_s, bool)
            or not isinstance(self.shutdown_grace_s, (int, float))
            or self.shutdown_grace_s <= 0
        ):
            raise ProductionDriverContractError("shutdown grace must be positive")


def compile_production_driver_binding(
    session: ProductionSessionBinding,
) -> ProductionDriverBinding:
    """Derive every worker input from the already consumed nine channels."""

    if not isinstance(session, ProductionSessionBinding):
        raise ProductionDriverContractError(
            "compile_production_driver_binding requires a production session"
        )
    manifest = session.channels.formal_run_manifest
    output = session.channels.formal_output
    launch_path = _absolute_file(
        manifest.get("launch_config_path"), "launch config"
    )
    launch_sha = _sha(manifest.get("launch_config_sha256"), "launch config")
    if sha256(launch_path.read_bytes()).hexdigest() != launch_sha:
        raise ProductionDriverContractError("launch config changed after session binding")
    if launch_path != Path(session.evidence_bundle.launch_config_path):
        raise ProductionDriverContractError(
            "session and manifest launch-config paths differ"
        )
    if launch_sha != session.evidence_bundle.launch_config_sha256:
        raise ProductionDriverContractError(
            "session and manifest launch-config hashes differ"
        )
    output_dir = _absolute_path(output.get("output_dir"), "formal output")
    pipeline = _method_pipeline_binding(session)
    log_names = {
        ProductionWorkerRole.TERM1: "term1.log",
        ProductionWorkerRole.TERM2_FRONT: "term2_front.log",
        ProductionWorkerRole.TERM3_BACK: "term3_back.log",
        ProductionWorkerRole.TERM4: "term4.log",
    }
    workers = tuple(
        ProductionWorkerInvocation(
            role=role,
            run_id=session.run_spec.run_id,
            launch_config_path=launch_path,
            launch_config_sha256=launch_sha,
            injection_bundle_fingerprint=session.evidence_bundle.fingerprint,
            method_pipeline_fingerprint=pipeline.fingerprint,
            log_path=output_dir / "operator_logs" / log_names[role],
        )
        for role in _START_ORDER
    )
    return ProductionDriverBinding(
        session=session,
        output_dir=output_dir,
        terminal_snapshot_path=output_dir / "evidence" / "live_terminal_snapshot.json",
        method_pipeline=pipeline,
        workers=workers,
    )


def bind_term1_instrumented_runtime_class(
    base_class: type,
    *,
    trace_binding: RuntimeTraceBinding,
    snapshot_path: str | Path,
    sample_interval_s: float = 0.10,
    retention_window_s: float = 4.0,
) -> type:
    """Bind recorder kwargs absent from the frozen Term1 launcher's constructor.

    The frozen ``run_dual_sonic_manual_layout_sim.py`` stores
    ``DualG1SonicMujocoRuntime`` in a module global and constructs it with only
    its legacy arguments.  A worker must replace that global with the returned
    closure-bound class before calling the frozen ``main``.  Passing the raw
    class returned by :func:`instrument_runtime_class` would instead raise a
    constructor ``TypeError``.
    """

    recorder = MuJoCoTerminalTraceRecorder(
        trace_binding,
        sample_interval_s=sample_interval_s,
        retention_window_s=retention_window_s,
    )
    snapshot = _absolute_path(snapshot_path, "terminal snapshot")
    try:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProductionDriverInfrastructureError(
            f"cannot create terminal snapshot directory: {exc}"
        ) from exc
    if snapshot.parent.is_symlink() or not snapshot.parent.is_dir():
        raise ProductionDriverContractError(
            "terminal snapshot parent must be a real directory"
        )
    instrumented = instrument_runtime_class(base_class)

    class BoundInstrumentedRuntime(instrumented):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if "terminal_trace_recorder" in kwargs or "terminal_trace_snapshot_path" in kwargs:
                raise ProductionDriverContractError(
                    "Term1 recorder bindings cannot be overridden"
                )
            super().__init__(
                *args,
                terminal_trace_recorder=recorder,
                terminal_trace_snapshot_path=snapshot,
                **kwargs,
            )

    BoundInstrumentedRuntime.__name__ = "BoundInstrumentedDualG1SonicMujocoRuntime"
    BoundInstrumentedRuntime.__qualname__ = BoundInstrumentedRuntime.__name__
    return BoundInstrumentedRuntime


class LiveSnapshotTerminalCollector:
    """Collect the fixed Term1 rolling snapshot after the stack is quiescent."""

    def collect(self, binding: ProductionDriverBinding) -> TerminalEvidenceDocuments:
        return load_live_terminal_snapshot(binding.terminal_snapshot_path)


@dataclass(frozen=True, slots=True)
class ImmutableProductionEvidencePublisher:
    """Concrete handoff to the existing immutable evidence writer."""

    terminal_producer: EvidenceProducer
    fm_producer: EvidenceProducer

    def publish(
        self,
        handoff: ProductionEvidenceHandoff,
    ) -> ProductionEvidenceWriteResult:
        completion = handoff.completion
        documents = handoff.terminal_documents
        return write_production_result_evidence(
            handoff.binding.session.evidence_bundle,
            terminal_status=completion.terminal_status,
            terminal_sample_trace=documents.terminal_sample_trace,
            terminal_state=documents.terminal_state,
            accounting=completion.accounting,
            transaction_accounting_evidence=(
                completion.transaction_accounting_evidence
            ),
            fm_calls=handoff.fm_calls,
            terminal_producer=self.terminal_producer,
            fm_producer=self.fm_producer,
            system_stall=completion.system_stall,
            payload_drop=completion.payload_drop,
            additional_artifact_pins=completion.additional_artifact_pins,
        )


@dataclass(frozen=True, slots=True)
class ProductionWorkerCommand:
    """Byte-pinned entrypoint in the audited runner integration checkout.

    A command is intentionally more than an arbitrary absolute path.  Its
    executable must be a tracked, canonical file in the exact clean git
    checkout named by the launch capability.  The identity is checked both
    here and immediately before a future exec; construction is not an
    authorization to dispatch.
    """

    role: ProductionWorkerRole
    runner_repo: Path
    runner_integration_commit: str
    runner_capability_sha256: str
    runner_capability_entrypoint_sha256: str
    entrypoint: Path
    entrypoint_sha256: str
    cwd: Path
    environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, ProductionWorkerRole):
            raise ProductionDriverContractError("worker command role is invalid")
        for label, value in (
            ("runner capability", self.runner_capability_sha256),
            (
                "runner capability entrypoint",
                self.runner_capability_entrypoint_sha256,
            ),
            ("worker entrypoint", self.entrypoint_sha256),
        ):
            if not _is_sha(value):
                raise ProductionDriverContractError(
                    f"{self.role.value} {label} hash must be SHA-256"
                )
        if not isinstance(self.runner_integration_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", self.runner_integration_commit
        ):
            raise ProductionDriverContractError(
                f"{self.role.value} runner integration commit is invalid"
            )
        if self.environment is not None:
            if not isinstance(self.environment, Mapping) or not all(
                isinstance(key, str)
                and key
                and isinstance(value, str)
                for key, value in self.environment.items()
            ):
                raise ProductionDriverContractError(
                    f"{self.role.value} environment must be a string mapping"
                )
            object.__setattr__(
                self,
                "environment",
                MappingProxyType(dict(self.environment)),
            )
        _revalidate_worker_command_files(self)


class _SubprocessWorker:
    def __init__(
        self,
        role: ProductionWorkerRole,
        process: subprocess.Popen[bytes],
        log_stream: BinaryIO,
    ) -> None:
        self.role = role
        self._process = process
        self._log_stream = log_stream
        self._closed = False
        # ``start_new_session=True`` makes the child the leader of a fresh
        # process group.  Retaining that group identity is what lets cleanup
        # stop grandchildren as well as the direct Popen child.
        self._process_group_id = process.pid
        if self._process_group_id <= 1 or self._process_group_id == os.getpgrp():
            raise ProductionDriverInfrastructureError(
                "worker did not receive an isolated process group"
            )
        try:
            actual_process_group_id = os.getpgid(process.pid)
        except ProcessLookupError as exc:
            raise ProductionDriverInfrastructureError(
                "worker exited before its isolated process group was attested"
            ) from exc
        if actual_process_group_id != self._process_group_id:
            raise ProductionDriverInfrastructureError(
                "worker is not the leader of its start_new_session process group"
            )
        self._group_observed_gone = False

    def poll(self) -> int | None:
        result = self._process.poll()
        if result is None:
            return None
        if self._process_group_exists():
            # The leader exited but one of its descendants is still alive.
            # Treat the managed worker as live until the complete group is
            # quiescent so reverse cleanup cannot silently orphan children.
            return None
        if result is not None:
            self._close_log()
        return result

    def wait(self, timeout_s: float | None = None) -> int:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        try:
            result = int(self._process.wait(timeout=timeout_s))
            while self._process_group_exists():
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(
                        self._process.args,
                        timeout_s,
                    )
                time.sleep(0.01)
            return result
        finally:
            if self._process.poll() is not None and not self._process_group_exists():
                self._close_log()

    def terminate(self) -> None:
        self._signal_process_group(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_process_group(signal.SIGKILL)

    def _signal_process_group(self, sig: signal.Signals) -> None:
        if not self._process_group_exists():
            return
        try:
            os.killpg(self._process_group_id, sig)
        except ProcessLookupError:
            self._group_observed_gone = True
        except OSError as exc:
            raise ProductionDriverInfrastructureError(
                f"cannot signal {self.role.value} process group: {exc}"
            ) from exc

    def _process_group_exists(self) -> bool:
        if self._group_observed_gone:
            return False
        try:
            os.killpg(self._process_group_id, 0)
        except ProcessLookupError:
            self._group_observed_gone = True
            return False
        except PermissionError:
            return True
        return True

    def _close_log(self) -> None:
        if not self._closed:
            self._log_stream.close()
            self._closed = True


class SubprocessWorkerLauncher:
    """Concrete no-shell launcher for four distinct, fixed worker entrypoints."""

    def __init__(self, commands: Sequence[ProductionWorkerCommand]) -> None:
        frozen = tuple(commands)
        if tuple(item.role for item in frozen) != _START_ORDER:
            raise ProductionDriverContractError(
                "worker commands must be exact Term1, Term2, Term3, Term4 order"
            )
        if len({item.entrypoint for item in frozen}) != len(_START_ORDER):
            raise ProductionDriverContractError(
                "the four production worker entrypoints must be distinct"
            )
        shared_identities = {
            (
                item.runner_repo,
                item.runner_integration_commit,
                item.runner_capability_sha256,
                item.runner_capability_entrypoint_sha256,
            )
            for item in frozen
        }
        if len(shared_identities) != 1:
            raise ProductionDriverContractError(
                "worker commands do not share one audited runner identity"
            )
        self._commands = {item.role: item for item in frozen}

    def start(
        self,
        invocation: ProductionWorkerInvocation,
        *,
        session: ProductionSessionBinding,
    ) -> ManagedProductionWorker:
        if not isinstance(session, ProductionSessionBinding):
            raise ProductionDriverContractError(
                "subprocess dispatch requires a production session binding"
            )
        if invocation.run_id != session.run_spec.run_id:
            raise ProductionDriverContractError(
                "worker invocation differs from the bound session"
            )
        _revalidate_launch_token(invocation)
        command = self._commands[invocation.role]
        _revalidate_worker_command(command, session)
        # This is the concrete OS-process boundary.  Fake launchers remain
        # useful for orchestration unit tests, but a caller cannot bypass
        # production.runner's exit-78 guard by instantiating this launcher.
        # A reviewed multi-worker capability and an execution-ready session
        # must be introduced together before this branch can be enabled.
        if session.execution_ready is not True:
            raise ProductionDriverContractError(
                "production subprocess dispatch is blocked: reviewed runtime "
                "capability is not execution-ready"
            )
        invocation.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = invocation.log_path.open("xb")
        except OSError as exc:
            raise ProductionDriverInfrastructureError(
                f"cannot create {invocation.role.value} log: {exc}"
            ) from exc
        argv = (
            str(command.entrypoint),
            "--launch-config",
            str(invocation.launch_config_path),
        )
        environment = None
        if command.environment is not None:
            environment = dict(command.environment)
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(command.cwd),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            stream.close()
            invocation.log_path.unlink(missing_ok=True)
            raise ProductionDriverInfrastructureError(
                f"cannot start {invocation.role.value}: {exc}"
            ) from exc
        try:
            return _SubprocessWorker(invocation.role, process, stream)
        except BaseException:
            # Popen succeeded, so this launcher owns cleanup even if wrapping
            # or process-group validation fails before a handle can be
            # returned to the driver.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                try:
                    process.wait(timeout=5.0)
                finally:
                    stream.close()
            raise


class ImmutableProductionFailureArchiver:
    """Write one non-paper failure marker without creating a result manifest."""

    filename = "production_driver_failure.json"

    def archive(
        self,
        binding: ProductionDriverBinding,
        failure: ProductionDriverFailure,
    ) -> None:
        path = binding.output_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "coevolve_production_driver_infrastructure_failure",
            "schema_version": 1,
            "paper_evidence": False,
            **to_primitive(failure),
        }
        raw = (
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except OSError as exc:
            raise ProductionDriverInfrastructureError(
                f"cannot archive production driver failure: {exc}"
            ) from exc
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise


class ProductionDriver:
    """Single-use Term1–4 coordinator with reverse-order cleanup."""

    def __init__(
        self,
        binding: ProductionDriverBinding,
        ports: ProductionDriverPorts,
    ) -> None:
        if not isinstance(binding, ProductionDriverBinding):
            raise ProductionDriverContractError("driver binding has the wrong type")
        if not isinstance(ports, ProductionDriverPorts):
            raise ProductionDriverContractError("driver ports have the wrong type")
        self.binding = binding
        self.ports = ports
        self._used = False

    def run(self) -> ProductionDriverReceipt:
        if self._used:
            raise ProductionDriverContractError("a production driver is single-use")
        self._used = True
        # Register every value returned by ``launcher.start`` before checking
        # its claimed role/type.  Otherwise a malformed-but-live handle leaks
        # precisely on the contract-error path intended to reject it.
        handles: dict[ProductionWorkerRole, object] = {}
        started: list[ProductionWorkerRole] = []
        stopped: list[ProductionWorkerRole] = []
        stage = "revalidate_launch"
        try:
            _revalidate_driver_binding(self.binding)
            for role in _START_ORDER[:3]:
                stage = f"start_{role.value}"
                handle = self.ports.launcher.start(
                    self.binding.worker(role),
                    session=self.binding.session,
                )
                handles[role] = handle
                started.append(role)
                _validate_handle(handle, role)
                _require_running(handle, role, "startup")

            stage = "wait_stack_ready"
            self.ports.readiness.wait_ready(self.binding, MappingProxyType(handles))
            for role, handle in handles.items():
                _require_running(handle, role, "readiness")

            role = ProductionWorkerRole.TERM4
            stage = "start_term4"
            term4 = self.ports.launcher.start(
                self.binding.worker(role),
                session=self.binding.session,
            )
            handles[role] = term4
            started.append(role)
            _validate_handle(term4, role)
            stage = "wait_term4"
            return_code = term4.wait(timeout_s=self.ports.term4_timeout_s)
            if return_code != 0:
                raise ProductionDriverInfrastructureError(
                    f"Term4 exited with status {return_code}"
                )

            stage = "collect_term4"
            completion = self.ports.term4_collector.collect(self.binding)
            _validate_completion(self.binding, completion)
            stage = "collect_fm_calls"
            fm_calls = _validated_fm_calls(
                self.binding,
                self.ports.fm_call_collector.collect(self.binding),
            )
        except BaseException as exc:
            cleanup_error = _stop_workers(
                handles,
                stopped,
                grace_s=self.ports.shutdown_grace_s,
            )
            failure = _failure_record(
                self.binding,
                stage=stage,
                error=cleanup_error or exc,
                started=started,
                stopped=stopped,
            )
            try:
                self.ports.failure_archiver.archive(self.binding, failure)
            except BaseException as archive_error:
                raise ProductionDriverInfrastructureError(
                    f"{type(exc).__name__}: {exc}; failure archive also failed: "
                    f"{type(archive_error).__name__}: {archive_error}"
                ) from exc
            if cleanup_error is not None:
                raise ProductionDriverInfrastructureError(
                    f"{type(exc).__name__}: {exc}; cleanup failed: {cleanup_error}"
                ) from exc
            raise

        stage = "stop_stack"
        cleanup_error = _stop_workers(
            handles,
            stopped,
            grace_s=self.ports.shutdown_grace_s,
        )
        if cleanup_error is not None:
            failure = _failure_record(
                self.binding,
                stage=stage,
                error=cleanup_error,
                started=started,
                stopped=stopped,
            )
            self.ports.failure_archiver.archive(self.binding, failure)
            raise ProductionDriverInfrastructureError(str(cleanup_error))

        try:
            stage = "validate_native_memory"
            _validate_native_memory(self.binding, completion)
            stage = "collect_terminal_trace"
            terminal_documents = self.ports.terminal_collector.collect(self.binding)
            if not isinstance(terminal_documents, TerminalEvidenceDocuments):
                raise ProductionDriverContractError(
                    "terminal collector returned the wrong type"
                )
            stage = "publish_evidence"
            evidence = self.ports.evidence_publisher.publish(
                ProductionEvidenceHandoff(
                    binding=self.binding,
                    completion=completion,
                    terminal_documents=terminal_documents,
                    fm_calls=fm_calls,
                )
            )
            if not isinstance(evidence, ProductionEvidenceWriteResult):
                raise ProductionDriverContractError(
                    "evidence publisher returned the wrong type"
                )
        except BaseException as exc:
            failure = _failure_record(
                self.binding,
                stage=stage,
                error=exc,
                started=started,
                stopped=stopped,
            )
            try:
                self.ports.failure_archiver.archive(self.binding, failure)
            except BaseException as archive_error:
                raise ProductionDriverInfrastructureError(
                    f"{type(exc).__name__}: {exc}; failure archive also failed: "
                    f"{type(archive_error).__name__}: {archive_error}"
                ) from exc
            raise
        return ProductionDriverReceipt(
            run_id=self.binding.session.run_spec.run_id,
            terminal_status=completion.terminal_status,
            worker_start_order=tuple(started),
            worker_stop_order=tuple(stopped),
            evidence=evidence,
        )


def _method_pipeline_binding(
    session: ProductionSessionBinding,
) -> LiveMethodPipelineBinding:
    prompt = session.prompt
    contract = session.method_contract
    if prompt.artifact is None:
        method_roles = tuple(
            LiveRolePromptBinding(
                role=role,
                prompt_base=_FULL_ROLE_PROMPT_BASES[role],
                output_contract=(
                    "v83_task_orchestrator_json_v1"
                    if role == "task_orchestrator"
                    else "v83_team_policy_json_v1"
                    if role == "coder"
                    else "v83_evolution_manager_json_v1"
                    if role == "evolution_manager"
                    else "v83_mutation_operator_json_v1"
                    if role == "mutation_operator"
                    else "v83_memory_update_json_v1"
                ),
                overlay_sha256=None,
            )
            for role in _FULL_METHOD_ROLES
        )
        common = _COMMON_RUNTIME_ROLES
        forbidden: tuple[str, ...] = ()
    else:
        artifact = prompt.artifact
        method_roles = tuple(
            LiveRolePromptBinding(
                role=item.role,
                prompt_base=item.prompt_base,
                output_contract=item.output_contract,
                overlay_sha256=item.prompt_sha256,
            )
            for item in artifact.roles
        )
        common = artifact.common_runtime_roles
        forbidden = artifact.forbidden_method_roles
    common_roles = tuple(
        LiveRolePromptBinding(
            role=role,
            prompt_base=_FULL_ROLE_PROMPT_BASES[role],
            output_contract=_COMMON_ROLE_OUTPUT_CONTRACTS[role],
            overlay_sha256=None,
        )
        for role in common
    )
    roles = (*method_roles, *common_roles)
    gates = V83MemoryOperationGates(
        prompt.memory_gates.strategy_retrieval,
        prompt.memory_gates.strategy_promotion,
        prompt.memory_gates.action_exemplar_retrieval,
        prompt.memory_gates.action_exemplar_promotion,
    )
    result = LiveMethodPipelineBinding(
        method_variant=session.run_spec.method_variant,
        policy_representation=contract.policy_representation,
        decision_structure=contract.decision_structure,
        decision_prompt_fingerprint=prompt.decision_prompt_fingerprint,
        shared_prompt_contract_fingerprint=(
            prompt.shared_prompt_contract_fingerprint
        ),
        frozen_v83_prompt_sha256=prompt.frozen_v83_prompt_sha256,
        roles=roles,
        common_runtime_roles=common,
        forbidden_method_roles=forbidden,
        memory_operation_gates=gates,
        memory_writer_call=prompt.memory_gates.memory_writer_call,
    )
    # Every named frozen base must resolve to a symbol in the closed registry.
    for item in (*result.roles,):
        if item.prompt_base != "none" and item.prompt_base not in FROZEN_V83_PROMPT_BASE_SYMBOLS:
            raise ProductionDriverContractError(
                f"unregistered frozen prompt base: {item.prompt_base}"
            )
    return result


def _validate_completion(
    binding: ProductionDriverBinding,
    value: ProductionTerm4Completion,
) -> None:
    if not isinstance(value, ProductionTerm4Completion):
        raise ProductionDriverContractError("Term4 collector returned the wrong type")
    expected = {
        "run_id": binding.session.run_spec.run_id,
        "launch_config_sha256": binding.session.evidence_bundle.launch_config_sha256,
        "injection_bundle_fingerprint": binding.session.evidence_bundle.fingerprint,
        "method_pipeline_fingerprint": binding.method_pipeline.fingerprint,
    }
    for name, wanted in expected.items():
        if getattr(value, name) != wanted:
            raise ProductionDriverContractError(
                f"Term4 completion {name} differs from the driver binding"
            )
    if value.terminal_status not in _VALID_TERMINALS:
        raise ProductionDriverContractError(
            f"Term4 terminal status is invalid: {value.terminal_status!r}"
        )
    if not isinstance(value.accounting, ProductionRunAccounting):
        raise ProductionDriverContractError("Term4 accounting has the wrong type")
    if not isinstance(
        value.transaction_accounting_evidence,
        TransactionAccountingEvidence,
    ):
        raise ProductionDriverContractError(
            "Term4 transaction accounting evidence has the wrong type"
        )
    if value.terminal_status == "SYSTEM_STALL":
        if value.system_stall is None or value.payload_drop is not None:
            raise ProductionDriverContractError(
                "SYSTEM_STALL completion requires only stall evidence"
            )
    elif value.system_stall is not None:
        raise ProductionDriverContractError(
            "stall evidence is present for a non-stall terminal"
        )
    if value.terminal_status == "CONSECUTIVE_PAYLOAD_DROP_LIMIT":
        if value.payload_drop is None or value.system_stall is not None:
            raise ProductionDriverContractError(
                "drop-limit completion requires only payload-drop evidence"
            )
    elif value.payload_drop is not None:
        raise ProductionDriverContractError(
            "payload-drop evidence is present for a non-drop terminal"
        )


def _validated_fm_calls(
    binding: ProductionDriverBinding,
    values: Sequence[FoundationModelCall],
) -> tuple[FoundationModelCall, ...]:
    if isinstance(values, (str, bytes)):
        raise ProductionDriverContractError("FM call collector returned a string")
    try:
        calls = tuple(values)
    except TypeError as exc:
        raise ProductionDriverContractError(
            "FM call collector did not return a sequence"
        ) from exc
    if not calls or not all(isinstance(item, FoundationModelCall) for item in calls):
        raise ProductionDriverContractError(
            "FM call collector must return individual logical-request records"
        )
    model = binding.session.run_spec.target_backbone
    request_ids: set[str] = set()
    for call in calls:
        if not call.request_id.strip() or call.request_id in request_ids:
            raise ProductionDriverContractError(
                "FM request IDs must be non-empty and unique"
            )
        request_ids.add(call.request_id)
        if call.modality not in {"text", "vision"}:
            raise ProductionDriverContractError("FM modality must be text or vision")
        if call.requested_model != model:
            raise ProductionDriverContractError(
                "FM requested model differs from the launch"
            )
        if call.resolved_model is not None and call.resolved_model != model:
            raise ProductionDriverContractError(
                "FM resolved model differs from the launch"
            )
        attempts = call.transport_attempts
        if attempts:
            final_success = False
            final_http_status: int | None = None
            for ordinal, attempt in enumerate(attempts, 1):
                if not isinstance(attempt, FoundationModelTransportAttempt):
                    raise ProductionDriverContractError(
                        "FM transport attempt collector returned the wrong type"
                    )
                if attempt.outcome == "http_response":
                    status = attempt.http_status
                    if (
                        isinstance(status, bool)
                        or not isinstance(status, int)
                        or not 100 <= status <= 599
                        or attempt.error_type is not None
                    ):
                        raise ProductionDriverContractError(
                            "FM HTTP attempt is invalid"
                        )
                    succeeded = 200 <= status < 300
                    current_status: int | None = status
                elif attempt.outcome == "transport_error":
                    if (
                        attempt.http_status is not None
                        or not isinstance(attempt.error_type, str)
                        or not attempt.error_type.strip()
                    ):
                        raise ProductionDriverContractError(
                            "FM transport-error attempt is invalid"
                        )
                    succeeded = False
                    current_status = None
                else:
                    raise ProductionDriverContractError(
                        "FM transport attempt outcome is invalid"
                    )
                if succeeded and ordinal != len(attempts):
                    raise ProductionDriverContractError(
                        "FM transport retry appears after success"
                    )
                if ordinal == len(attempts):
                    final_success = succeeded
                    final_http_status = current_status
            if call.http_status != final_http_status:
                raise ProductionDriverContractError(
                    "FM final HTTP status differs from transport attempts"
                )
        else:
            status = call.http_status
            if (
                isinstance(status, bool)
                or not isinstance(status, int)
                or not 100 <= status <= 599
            ):
                raise ProductionDriverContractError(
                    "FM call without attempt records requires an HTTP status"
                )
            final_success = 200 <= status < 300
        inferred_outcome = "success" if final_success else "failure"
        if call.outcome is not None and call.outcome != inferred_outcome:
            raise ProductionDriverContractError(
                "FM logical outcome differs from transport outcome"
            )
        if final_success:
            if call.resolved_model != model or call.response_model != model:
                raise ProductionDriverContractError(
                    "successful FM response model differs from the launch"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in (call.input_tokens, call.output_tokens)
            ):
                raise ProductionDriverContractError(
                    "successful FM response token usage is invalid"
                )
        elif (
            call.response_model is not None
            or call.input_tokens is not None
            or call.output_tokens is not None
        ):
            raise ProductionDriverContractError(
                "failed FM request contains fabricated response usage"
            )
        if call.billing_amount_usd is not None and (
            isinstance(call.billing_amount_usd, bool)
            or not isinstance(call.billing_amount_usd, (int, float))
            or not math.isfinite(float(call.billing_amount_usd))
            or call.billing_amount_usd < 0
        ):
            raise ProductionDriverContractError(
                "FM billing value must be non-negative finite data or None"
            )
    return calls


def _validate_native_memory(
    binding: ProductionDriverBinding,
    completion: ProductionTerm4Completion,
) -> None:
    memory = binding.session.memory
    validation = validate_v83_native_memory_audit(
        binding.output_dir,
        expected_run_binding=memory.run_binding,
        expected_mode=memory.mode,
        expected_operation_gates=to_primitive(memory.gates),
        expected_terminal_reason=completion.terminal_status,
        expected_transfer_bank=memory.transfer_seed_path,
        expected_strategy_bank_sha256=(
            ""
            if binding.session.run_spec.strategy_bank is None
            else binding.session.run_spec.strategy_bank.sha256
        ),
    )
    if not validation.valid:
        raise ProductionDriverContractError(
            "native V83 memory evidence is invalid: "
            + ", ".join(validation.invalid_reasons)
        )


def _revalidate_driver_binding(binding: ProductionDriverBinding) -> None:
    if binding.session.execution_ready is not False or binding.execution_ready is not False:
        raise ProductionDriverContractError(
            "unreviewed code changed the fail-closed execution readiness flag"
        )
    expected = compile_production_driver_binding(binding.session)
    if (
        expected.output_dir != binding.output_dir
        or expected.terminal_snapshot_path != binding.terminal_snapshot_path
        or expected.method_pipeline != binding.method_pipeline
        or expected.workers != binding.workers
    ):
        raise ProductionDriverContractError(
            "driver binding differs from the consumed production session"
        )
    for invocation in binding.workers:
        _revalidate_launch_token(invocation)


def _revalidate_launch_token(invocation: ProductionWorkerInvocation) -> None:
    path = invocation.launch_config_path
    if path.is_symlink() or not path.is_file():
        raise ProductionDriverContractError("worker launch config is unavailable")
    if sha256(path.read_bytes()).hexdigest() != invocation.launch_config_sha256:
        raise ProductionDriverContractError("worker launch config bytes changed")


def _revalidate_worker_command(
    command: ProductionWorkerCommand,
    session: ProductionSessionBinding,
) -> None:
    """Re-attest command bytes/git identity against the consumed capability."""

    _revalidate_worker_command_files(command)
    result = session.channels.formal_run_manifest.get("result_binding")
    if not isinstance(result, Mapping):
        raise ProductionDriverContractError(
            "production session has no formal runner result binding"
        )
    checks = {
        "runner_integration_commit": command.runner_integration_commit,
        "runner_capability_sha256": command.runner_capability_sha256,
        "entrypoint_sha256": command.runner_capability_entrypoint_sha256,
    }
    for name, actual in checks.items():
        if result.get(name) != actual:
            raise ProductionDriverContractError(
                f"{command.role.value} worker runner identity differs from "
                f"session {name}"
            )


def _revalidate_worker_command_files(command: ProductionWorkerCommand) -> None:
    """Require canonical, tracked worker bytes in one exact clean checkout."""

    repo = _canonical_existing_directory(
        command.runner_repo,
        f"{command.role.value} runner repository",
    )
    cwd = _canonical_existing_directory(
        command.cwd,
        f"{command.role.value} cwd",
    )
    if cwd != repo:
        raise ProductionDriverContractError(
            f"{command.role.value} cwd must equal the audited runner repository"
        )
    entrypoint = _canonical_existing_file(
        command.entrypoint,
        f"{command.role.value} entrypoint",
    )
    try:
        relpath = entrypoint.relative_to(repo)
    except ValueError as exc:
        raise ProductionDriverContractError(
            f"{command.role.value} entrypoint escapes the runner repository"
        ) from exc
    canonical_relpath = _CANONICAL_WORKER_ENTRYPOINT_RELPATHS[command.role]
    if PurePosixPath(relpath.as_posix()) != canonical_relpath:
        raise ProductionDriverContractError(
            f"{command.role.value} entrypoint is not the canonical audited worker path"
        )
    if not stat.S_ISREG(entrypoint.stat().st_mode) or not os.access(
        entrypoint, os.X_OK
    ):
        raise ProductionDriverContractError(
            f"{command.role.value} entrypoint must be an executable regular file"
        )
    actual_sha = sha256(entrypoint.read_bytes()).hexdigest()
    if actual_sha != command.entrypoint_sha256:
        raise ProductionDriverContractError(
            f"{command.role.value} entrypoint byte hash changed"
        )

    head = _git_text(repo, ("rev-parse", "--verify", "HEAD^{commit}"))
    if head != command.runner_integration_commit:
        raise ProductionDriverContractError(
            f"{command.role.value} runner HEAD differs from the audited integration commit"
        )
    dirty = _git_text(
        repo,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        allow_empty=True,
    )
    if dirty:
        raise ProductionDriverContractError(
            f"{command.role.value} runner repository is not clean"
        )
    reltext = relpath.as_posix()
    tracked = _git_text(
        repo,
        ("ls-files", "--stage", "--error-unmatch", "--", reltext),
    )
    if not tracked.startswith("100755 "):
        raise ProductionDriverContractError(
            f"{command.role.value} worker entrypoint is not tracked executable content"
        )
    committed = _git_bytes(repo, ("show", f"HEAD:{reltext}"))
    if sha256(committed).hexdigest() != command.entrypoint_sha256:
        raise ProductionDriverContractError(
            f"{command.role.value} entrypoint pin differs from committed bytes"
        )


def _validate_handle(
    handle: object,
    role: ProductionWorkerRole,
) -> None:
    if not isinstance(handle, ManagedProductionWorker) or handle.role is not role:
        raise ProductionDriverContractError(
            f"launcher returned the wrong handle for {role.value}"
        )


def _require_running(
    handle: ManagedProductionWorker,
    role: ProductionWorkerRole,
    stage: str,
) -> None:
    return_code = handle.poll()
    if return_code is not None:
        raise ProductionDriverInfrastructureError(
            f"{role.value} exited with status {return_code} during {stage}"
        )


def _stop_workers(
    handles: Mapping[ProductionWorkerRole, object],
    stopped: list[ProductionWorkerRole],
    *,
    grace_s: float,
) -> BaseException | None:
    first_error: BaseException | None = None
    for role in reversed(_START_ORDER):
        handle = handles.get(role)
        if handle is None or role in stopped:
            continue
        try:
            poll = getattr(handle, "poll", None)
            wait = getattr(handle, "wait", None)
            terminate = getattr(handle, "terminate", None)
            kill = getattr(handle, "kill", None)
            lifecycle = (poll, wait, terminate, kill)
            if not all(callable(item) for item in lifecycle):
                # Still attempt every cleanup operation the malformed value
                # exposes before reporting that quiescence was unverifiable.
                if callable(terminate):
                    terminate()
                if callable(kill):
                    kill()
                raise ProductionDriverContractError(
                    f"launcher returned an unmanageable handle for {role.value}"
                )
            if poll() is None:
                terminate()
                try:
                    wait(timeout_s=grace_s)
                except (TimeoutError, subprocess.TimeoutExpired):
                    kill()
                    wait(timeout_s=grace_s)
            stopped.append(role)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    return first_error


def _canonical_existing_directory(value: object, label: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ProductionDriverContractError(f"{label} must be an absolute Path")
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise ProductionDriverContractError(f"{label} is unavailable: {exc}") from exc
    if resolved != value or value.is_symlink() or not value.is_dir():
        raise ProductionDriverContractError(
            f"{label} must be canonical, non-symlinked, and existing"
        )
    return value


def _canonical_existing_file(value: object, label: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ProductionDriverContractError(f"{label} must be an absolute Path")
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise ProductionDriverContractError(f"{label} is unavailable: {exc}") from exc
    if resolved != value or value.is_symlink() or not value.is_file():
        raise ProductionDriverContractError(
            f"{label} must be a canonical non-symlink regular file"
        )
    return value


def _git_text(
    repo: Path,
    args: Sequence[str],
    *,
    allow_empty: bool = False,
) -> str:
    raw = _git_bytes(repo, args)
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ProductionDriverContractError(
            "runner git identity is not UTF-8"
        ) from exc
    if not value and not allow_empty:
        raise ProductionDriverContractError("runner git identity is empty")
    return value


def _git_bytes(repo: Path, args: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repo), *args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ProductionDriverContractError(
            f"cannot inspect runner git identity: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ProductionDriverContractError(
            f"cannot verify runner git identity: {detail or completed.returncode}"
        )
    return completed.stdout


def _failure_record(
    binding: ProductionDriverBinding,
    *,
    stage: str,
    error: BaseException,
    started: Sequence[ProductionWorkerRole],
    stopped: Sequence[ProductionWorkerRole],
) -> ProductionDriverFailure:
    detail = str(error)
    if len(detail) > 4000:
        detail = detail[-4000:]
    return ProductionDriverFailure(
        run_id=binding.session.run_spec.run_id,
        launch_config_sha256=binding.session.evidence_bundle.launch_config_sha256,
        injection_bundle_fingerprint=binding.session.evidence_bundle.fingerprint,
        stage=stage,
        error_type=type(error).__name__,
        error=detail,
        started_roles=tuple(item.value for item in started),
        stopped_roles=tuple(item.value for item in stopped),
    )


def _absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ProductionDriverContractError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ProductionDriverContractError(
            f"{label} path must be absolute and non-symlinked"
        )
    return path.resolve(strict=False)


def _absolute_file(value: object, label: str) -> Path:
    path = _absolute_path(value, label)
    if not path.is_file():
        raise ProductionDriverContractError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _sha(value: object, label: str) -> str:
    if not _is_sha(value):
        raise ProductionDriverContractError(f"{label} must be SHA-256")
    return str(value)


__all__ = [
    "FoundationModelCallCollector",
    "ImmutableProductionEvidencePublisher",
    "ImmutableProductionFailureArchiver",
    "LiveMethodPipelineBinding",
    "LiveRolePromptBinding",
    "LiveSnapshotTerminalCollector",
    "ManagedProductionWorker",
    "ProductionDriver",
    "ProductionDriverBinding",
    "ProductionDriverContractError",
    "ProductionDriverFailure",
    "ProductionDriverInfrastructureError",
    "ProductionDriverPorts",
    "ProductionDriverReceipt",
    "ProductionEvidenceHandoff",
    "ProductionEvidencePublisher",
    "ProductionFailureArchiver",
    "ProductionStackReadiness",
    "ProductionTerm4Collector",
    "ProductionTerm4Completion",
    "ProductionTerminalCollector",
    "ProductionWorkerCommand",
    "ProductionWorkerInvocation",
    "ProductionWorkerLauncher",
    "ProductionWorkerRole",
    "SubprocessWorkerLauncher",
    "bind_term1_instrumented_runtime_class",
    "compile_production_driver_binding",
]
