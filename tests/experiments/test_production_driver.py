from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from coevolve_experiments.contracts import MethodVariant
from coevolve_experiments.production.driver import (
    ImmutableProductionFailureArchiver,
    ProductionDriver,
    ProductionDriverBinding,
    ProductionDriverContractError,
    ProductionDriverInfrastructureError,
    ProductionDriverPorts,
    ProductionEvidenceHandoff,
    ProductionTerm4Completion,
    ProductionWorkerCommand,
    ProductionWorkerRole,
    SubprocessWorkerLauncher,
    bind_term1_instrumented_runtime_class,
    compile_production_driver_binding,
)
from coevolve_experiments.production.evidence_writer import (
    FoundationModelCall,
    FoundationModelTransportAttempt,
    ProductionArtifactRef,
    ProductionEvidenceWriteResult,
    ProductionRunAccounting,
    TransactionAccountingEvidence,
)
from coevolve_experiments.production.runtime_instrumentation import (
    TerminalEvidenceDocuments,
)
from coevolve_experiments.production.session import bind_production_session
import coevolve_experiments.production.driver as driver_module
from test_production_session import _bundle


def _session(
    tmp_path: Path,
    *,
    variant: MethodVariant = MethodVariant.NO_DYNAMIC_EXPERIENCE,
    **bundle_kwargs: Any,
):
    return bind_production_session(
        _bundle(tmp_path, variant=variant, **bundle_kwargs),
        simulator_version="3.3.5",
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _worker_repo(tmp_path: Path) -> tuple[Path, str, dict[ProductionWorkerRole, Path]]:
    repo = (tmp_path / "audited-runner").resolve()
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Production Driver Tests")
    worker_root = repo / "src/coevolve_experiments/production/workers"
    worker_root.mkdir(parents=True)
    entrypoints = {}
    for role in ProductionWorkerRole:
        path = worker_root / f"{role.value}.py"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "print(json.dumps(sys.argv[1:]), flush=True)\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        entrypoints[role] = path
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "pin four production workers")
    return repo, _git(repo, "rev-parse", "HEAD"), entrypoints


def _worker_commands(
    repo: Path,
    commit: str,
    entrypoints: dict[ProductionWorkerRole, Path],
    *,
    capability_sha256: str,
    capability_entrypoint_sha256: str,
) -> tuple[ProductionWorkerCommand, ...]:
    return tuple(
        ProductionWorkerCommand(
            role=role,
            runner_repo=repo,
            runner_integration_commit=commit,
            runner_capability_sha256=capability_sha256,
            runner_capability_entrypoint_sha256=capability_entrypoint_sha256,
            entrypoint=entrypoints[role],
            entrypoint_sha256=sha256(entrypoints[role].read_bytes()).hexdigest(),
            cwd=repo,
        )
        for role in ProductionWorkerRole
    )


def _accounting() -> ProductionRunAccounting:
    return ProductionRunAccounting(
        physical_transactions=1,
        accepted_transactions=1,
        rejected_transactions=0,
        accepted_setup_transactions=0,
        accepted_progress_transactions=1,
        elapsed_s=1.5,
        phase_completed=6,
        phase_total=6,
    )


def _transaction_accounting_evidence() -> TransactionAccountingEvidence:
    return TransactionAccountingEvidence(
        transaction_receipts=(
            ProductionArtifactRef(
                "evidence/transactions/transaction_001.json",
                "1" * 64,
                1,
            ),
        ),
        frozen_phase_plan=ProductionArtifactRef(
            "evidence/accounting/frozen_phase_plan.json",
            "2" * 64,
            1,
        ),
        accepted_episode_manifest=ProductionArtifactRef(
            "memory/v83_native/accepted_episode_manifest.json",
            "3" * 64,
            1,
        ),
    )


def _call(model: str, *, request_id: str = "req-1") -> FoundationModelCall:
    return FoundationModelCall(
        request_id=request_id,
        role="coder",
        modality="text",
        requested_model=model,
        resolved_model=model,
        response_model=model,
        http_status=200,
        input_tokens=12,
        output_tokens=4,
        billing_amount_usd=0.01,
    )


class _Handle:
    def __init__(self, role: ProductionWorkerRole, events: list[str]) -> None:
        self.role = role
        self.events = events
        self.return_code: int | None = None

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout_s: float | None = None) -> int:
        del timeout_s
        if self.role is ProductionWorkerRole.TERM4 and self.return_code is None:
            self.return_code = 0
        if self.return_code is None:
            raise TimeoutError("worker still running")
        self.events.append(f"wait:{self.role.value}")
        return self.return_code

    def terminate(self) -> None:
        self.events.append(f"terminate:{self.role.value}")
        self.return_code = 0

    def kill(self) -> None:
        self.events.append(f"kill:{self.role.value}")
        self.return_code = -9


class _Launcher:
    def __init__(self, events: list[str], *, finalize_memory: bool = True) -> None:
        self.events = events
        self.finalize_memory = finalize_memory
        self.launch_paths: list[Path] = []

    def start(self, invocation, *, session):
        assert invocation.run_id == session.run_spec.run_id
        self.events.append(f"start:{invocation.role.value}")
        self.launch_paths.append(invocation.launch_config_path)
        if invocation.role is ProductionWorkerRole.TERM4 and self.finalize_memory:
            memory = session.memory.open(
                strategy_store_factory=lambda path: SimpleNamespace(specs=[]),
                action_store_factory=lambda path: SimpleNamespace(exemplars=[]),
            )
            memory.finalize(terminal_reason="SUCCESS")
        return _Handle(invocation.role, self.events)


class _Ready:
    def __init__(self, events: list[str], *, error: Exception | None = None) -> None:
        self.events = events
        self.error = error

    def wait_ready(self, binding, workers) -> None:
        assert binding.session.run_spec.run_id
        assert tuple(workers) == (
            ProductionWorkerRole.TERM1,
            ProductionWorkerRole.TERM2_FRONT,
            ProductionWorkerRole.TERM3_BACK,
        )
        self.events.append("ready")
        if self.error is not None:
            raise self.error


class _Term4Collector:
    def __init__(
        self,
        events: list[str],
        *,
        mutate_identity: bool = False,
    ) -> None:
        self.events = events
        self.mutate_identity = mutate_identity

    def collect(self, binding: ProductionDriverBinding) -> ProductionTerm4Completion:
        self.events.append("collect:term4")
        return ProductionTerm4Completion(
            run_id=(
                "wrong-run" if self.mutate_identity else binding.session.run_spec.run_id
            ),
            launch_config_sha256=binding.session.evidence_bundle.launch_config_sha256,
            injection_bundle_fingerprint=binding.session.evidence_bundle.fingerprint,
            method_pipeline_fingerprint=binding.method_pipeline.fingerprint,
            terminal_status="SUCCESS",
            accounting=_accounting(),
            transaction_accounting_evidence=_transaction_accounting_evidence(),
        )


class _FMCollector:
    def __init__(self, events: list[str], *, duplicate: bool = False) -> None:
        self.events = events
        self.duplicate = duplicate

    def collect(self, binding):
        self.events.append("collect:fm")
        call = _call(binding.session.run_spec.target_backbone)
        return (call, call) if self.duplicate else (call,)


class _TerminalCollector:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def collect(self, binding):
        self.events.append("collect:terminal")
        return TerminalEvidenceDocuments(
            terminal_sample_trace=MappingProxyType({"run_id": binding.session.run_spec.run_id}),
            terminal_state=MappingProxyType({"run_id": binding.session.run_spec.run_id}),
        )


class _Publisher:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.handoff: ProductionEvidenceHandoff | None = None

    def publish(self, handoff):
        self.events.append("publish")
        self.handoff = handoff
        root = handoff.binding.output_dir
        return ProductionEvidenceWriteResult(
            run_root=root,
            run_manifest_path=root / "run_manifest.json",
            metrics_path=root / "metrics.json",
            terminal_evidence_path=root / "terminal_verifier_evidence.json",
            fm_usage_ledger_path=root / "fm_usage_ledger.json",
            transaction_accounting_ledger_path=(
                root / "evidence/transaction_accounting_ledger.json"
            ),
            result_artifact_hashes=MappingProxyType({}),
            usage_totals=MappingProxyType({}),
            terminal_validation=None,  # type: ignore[arg-type]
            usage_validation=None,  # type: ignore[arg-type]
            accounting_validation=None,  # type: ignore[arg-type]
        )


class _Archiver:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.failures = []

    def archive(self, binding, failure) -> None:
        assert failure.run_id == binding.session.run_spec.run_id
        self.events.append(f"archive:{failure.stage}")
        self.failures.append(failure)


def _ports(
    events: list[str],
    *,
    readiness_error: Exception | None = None,
    mutate_identity: bool = False,
    duplicate_calls: bool = False,
    finalize_memory: bool = True,
):
    publisher = _Publisher(events)
    archiver = _Archiver(events)
    launcher = _Launcher(events, finalize_memory=finalize_memory)
    return (
        ProductionDriverPorts(
            launcher=launcher,
            readiness=_Ready(events, error=readiness_error),
            term4_collector=_Term4Collector(
                events,
                mutate_identity=mutate_identity,
            ),
            fm_call_collector=_FMCollector(events, duplicate=duplicate_calls),
            terminal_collector=_TerminalCollector(events),
            evidence_publisher=publisher,
            failure_archiver=archiver,
            shutdown_grace_s=0.1,
        ),
        launcher,
        publisher,
        archiver,
    )


def test_driver_binding_uses_one_launch_token_and_remains_blocked(tmp_path: Path) -> None:
    session = _session(tmp_path)
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    binding = compile_production_driver_binding(session)

    after = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert binding.execution_ready is False
    assert tuple(item.role for item in binding.workers) == tuple(ProductionWorkerRole)
    assert {item.launch_config_path for item in binding.workers} == {
        Path(session.evidence_bundle.launch_config_path)
    }
    assert {item.launch_config_sha256 for item in binding.workers} == {
        session.evidence_bundle.launch_config_sha256
    }
    assert binding.terminal_snapshot_path == (
        binding.output_dir / "evidence/live_terminal_snapshot.json"
    )


@pytest.mark.parametrize("variant", tuple(MethodVariant))
def test_driver_binds_exact_praxis_method_roles(
    tmp_path: Path,
    variant: MethodVariant,
) -> None:
    binding = compile_production_driver_binding(_session(tmp_path, variant=variant))
    pipeline = binding.method_pipeline

    assert pipeline.method_variant is variant
    assert pipeline.policy_representation.value == "praxis_code_as_policy"
    assert {"vlm_failure_analyzer", "code_repair"}.issubset(
        {item.role for item in pipeline.roles}
    )
    assert pipeline.common_runtime_roles == (
        "vlm_failure_analyzer",
        "code_repair",
    )
    assert not set(pipeline.forbidden_method_roles).intersection(
        {item.role for item in pipeline.roles}
    )
    if variant is MethodVariant.COEVOLE_FULL:
        assert "task_orchestrator" in {item.role for item in pipeline.roles}
        assert pipeline.memory_writer_call is True
    elif variant is MethodVariant.NO_DYNAMIC_EXPERIENCE:
        assert "memory_update" not in {item.role for item in pipeline.roles}
        assert pipeline.memory_operation_gates.all_disabled


def test_bound_term1_runtime_injects_recorder_kwargs_automatically(tmp_path: Path) -> None:
    binding = compile_production_driver_binding(_session(tmp_path))

    class FrozenRuntime:
        def __init__(self, value: str) -> None:
            self.value = value

        def _write_runtime_state(self) -> None:
            return None

    bound = bind_term1_instrumented_runtime_class(
        FrozenRuntime,
        trace_binding=binding.session.runtime_trace,
        snapshot_path=binding.terminal_snapshot_path,
    )
    runtime = bound("legacy-constructor")

    assert runtime.value == "legacy-constructor"
    assert runtime._coevolve_terminal_trace_recorder.binding == (  # type: ignore[attr-defined]
        binding.session.runtime_trace
    )
    assert runtime._coevolve_terminal_trace_snapshot_path == (  # type: ignore[attr-defined]
        binding.terminal_snapshot_path
    )
    with pytest.raises(
        ProductionDriverContractError,
        match="cannot be overridden",
    ):
        bound(
            "bad",
            terminal_trace_recorder=object(),
        )


def test_driver_runs_four_roles_cleans_up_and_hands_off_once(tmp_path: Path) -> None:
    binding = compile_production_driver_binding(_session(tmp_path))
    events: list[str] = []
    ports, launcher, publisher, archiver = _ports(events)

    receipt = ProductionDriver(binding, ports).run()

    assert receipt.worker_start_order == tuple(ProductionWorkerRole)
    assert receipt.worker_stop_order == tuple(reversed(tuple(ProductionWorkerRole)))
    assert receipt.terminal_status == "SUCCESS"
    assert len(set(launcher.launch_paths)) == 1
    assert publisher.handoff is not None
    assert tuple(call.request_id for call in publisher.handoff.fm_calls) == ("req-1",)
    assert archiver.failures == []
    assert events.index("collect:term4") < events.index("terminate:term3_back")
    assert events.index("terminate:term1") < events.index("collect:terminal")
    assert events[-1] == "publish"


def test_readiness_failure_stops_started_stack_and_archives(tmp_path: Path) -> None:
    binding = compile_production_driver_binding(_session(tmp_path))
    events: list[str] = []
    ports, _launcher, publisher, archiver = _ports(
        events,
        readiness_error=RuntimeError("not ready"),
        finalize_memory=False,
    )

    with pytest.raises(RuntimeError, match="not ready"):
        ProductionDriver(binding, ports).run()

    assert publisher.handoff is None
    assert len(archiver.failures) == 1
    assert archiver.failures[0].stage == "wait_stack_ready"
    assert archiver.failures[0].started_roles == (
        "term1",
        "term2_front",
        "term3_back",
    )
    assert archiver.failures[0].stopped_roles == (
        "term3_back",
        "term2_front",
        "term1",
    )


@pytest.mark.parametrize(
    ("mutate_identity", "duplicate_calls", "match"),
    (
        (True, False, "completion run_id differs"),
        (False, True, "request IDs must be non-empty and unique"),
    ),
)
def test_collector_drift_fails_after_reverse_cleanup(
    tmp_path: Path,
    mutate_identity: bool,
    duplicate_calls: bool,
    match: str,
) -> None:
    binding = compile_production_driver_binding(_session(tmp_path))
    events: list[str] = []
    ports, _launcher, publisher, archiver = _ports(
        events,
        mutate_identity=mutate_identity,
        duplicate_calls=duplicate_calls,
    )

    with pytest.raises(ProductionDriverContractError, match=match):
        ProductionDriver(binding, ports).run()

    assert publisher.handoff is None
    assert len(archiver.failures) == 1
    assert events.index("terminate:term1") < events.index(
        f"archive:{archiver.failures[0].stage}"
    )


def test_driver_accepts_retried_success_and_recorded_failed_logical_call(
    tmp_path: Path,
) -> None:
    binding = compile_production_driver_binding(_session(tmp_path))
    model = binding.session.run_spec.target_backbone
    success = replace(
        _call(model),
        transport_attempts=(
            FoundationModelTransportAttempt(
                outcome="http_response", http_status=503
            ),
            FoundationModelTransportAttempt(
                outcome="http_response", http_status=200
            ),
        ),
    )
    failure = FoundationModelCall(
        request_id="req-failed",
        role="reliable_feedback",
        modality="vision",
        requested_model=model,
        resolved_model=model,
        response_model=None,
        http_status=None,
        input_tokens=None,
        output_tokens=None,
        billing_amount_usd=None,
        outcome="failure",
        transport_attempts=(
            FoundationModelTransportAttempt(
                outcome="transport_error", error_type="timeout"
            ),
        ),
    )

    result = driver_module._validated_fm_calls(binding, (success, failure))

    assert result == (success, failure)


def test_driver_binding_drift_is_rejected_before_worker_start(tmp_path: Path) -> None:
    binding = compile_production_driver_binding(_session(tmp_path))
    bad_worker = replace(
        binding.workers[0],
        method_pipeline_fingerprint="0" * 64,
    )
    poisoned = replace(binding, workers=(bad_worker, *binding.workers[1:]))
    events: list[str] = []
    ports, _launcher, _publisher, archiver = _ports(events, finalize_memory=False)

    with pytest.raises(
        ProductionDriverContractError,
        match="differs from the consumed production session",
    ):
        ProductionDriver(poisoned, ports).run()

    assert not any(event.startswith("start:") for event in events)
    assert archiver.failures[0].stage == "revalidate_launch"


def test_subprocess_launcher_revalidates_pins_and_cannot_bypass_exit_78(
    tmp_path: Path,
) -> None:
    repo, commit, entrypoints = _worker_repo(tmp_path)
    capability_sha = sha256(b"reviewed-capability").hexdigest()
    capability_entrypoint_sha = sha256(b"reviewed-runner-entrypoint").hexdigest()
    binding = compile_production_driver_binding(
        _session(
            tmp_path,
            runner_integration_commit=commit,
            runner_capability_sha256=capability_sha,
            runner_capability_entrypoint_sha256=capability_entrypoint_sha,
        )
    )
    commands = _worker_commands(
        repo,
        commit,
        entrypoints,
        capability_sha256=capability_sha,
        capability_entrypoint_sha256=capability_entrypoint_sha,
    )
    launcher = SubprocessWorkerLauncher(commands)

    with pytest.raises(
        ProductionDriverContractError,
        match="reviewed runtime capability is not execution-ready",
    ):
        launcher.start(
            binding.worker(ProductionWorkerRole.TERM1),
            session=binding.session,
        )
    assert not binding.worker(ProductionWorkerRole.TERM1).log_path.exists()

    # A pin that was valid at launcher construction is not treated as a
    # durable authorization: exact bytes are checked again on every start.
    entrypoints[ProductionWorkerRole.TERM1].write_text(
        "#!/usr/bin/env python3\nraise SystemExit(99)\n",
        encoding="utf-8",
    )
    with pytest.raises(ProductionDriverContractError, match="byte hash changed"):
        launcher.start(
            binding.worker(ProductionWorkerRole.TERM1),
            session=binding.session,
        )


def test_worker_entrypoint_must_be_canonical_tracked_and_byte_pinned(
    tmp_path: Path,
) -> None:
    repo, commit, entrypoints = _worker_repo(tmp_path)
    target = entrypoints[ProductionWorkerRole.TERM1]
    common = {
        "role": ProductionWorkerRole.TERM1,
        "runner_repo": repo,
        "runner_integration_commit": commit,
        "runner_capability_sha256": sha256(b"capability").hexdigest(),
        "runner_capability_entrypoint_sha256": sha256(b"runner").hexdigest(),
        "entrypoint_sha256": sha256(target.read_bytes()).hexdigest(),
        "cwd": repo,
    }

    outside = (tmp_path / "outside.py").resolve()
    outside.write_bytes(target.read_bytes())
    outside.chmod(0o755)
    with pytest.raises(ProductionDriverContractError, match="escapes"):
        ProductionWorkerCommand(entrypoint=outside, **common)

    alias = repo / "alias.py"
    alias.symlink_to(target)
    with pytest.raises(ProductionDriverContractError, match="canonical non-symlink"):
        ProductionWorkerCommand(entrypoint=alias, **common)
    alias.unlink()

    with pytest.raises(ProductionDriverContractError, match="byte hash changed"):
        ProductionWorkerCommand(
            entrypoint=target,
            **{**common, "entrypoint_sha256": "0" * 64},
        )


class _RolelessHandle:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.return_code = None

    def poll(self):
        return self.return_code

    def wait(self, timeout_s=None):
        del timeout_s
        self.events.append("wait:roleless")
        return 0

    def terminate(self):
        self.events.append("terminate:roleless")
        self.return_code = 0

    def kill(self):
        self.events.append("kill:roleless")
        self.return_code = -9


class _MalformedHandleLauncher:
    def __init__(self, events: list[str], *, wrong_role: bool) -> None:
        self.events = events
        self.wrong_role = wrong_role

    def start(self, invocation, *, session):
        del session
        self.events.append(f"start:{invocation.role.value}")
        if self.wrong_role:
            return _Handle(ProductionWorkerRole.TERM4, self.events)
        return _RolelessHandle(self.events)


@pytest.mark.parametrize(
    ("wrong_role", "cleanup_event"),
    ((True, "terminate:term4"), (False, "terminate:roleless")),
)
def test_malformed_returned_handle_is_owned_and_cleaned(
    tmp_path: Path,
    wrong_role: bool,
    cleanup_event: str,
) -> None:
    binding = compile_production_driver_binding(_session(tmp_path))
    events: list[str] = []
    ports, _launcher, publisher, archiver = _ports(events, finalize_memory=False)
    ports = replace(
        ports,
        launcher=_MalformedHandleLauncher(events, wrong_role=wrong_role),
    )

    with pytest.raises(ProductionDriverContractError, match="wrong handle"):
        ProductionDriver(binding, ports).run()

    assert cleanup_event in events
    assert publisher.handoff is None
    assert len(archiver.failures) == 1
    assert archiver.failures[0].started_roles == ("term1",)
    assert archiver.failures[0].stopped_roles == ("term1",)


def test_subprocess_worker_terminates_the_complete_process_group(
    tmp_path: Path,
) -> None:
    child_pid_path = (tmp_path / "child.pid").resolve()
    script = (tmp_path / "process_tree.py").resolve()
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    log_path = (tmp_path / "tree.log").resolve()
    stream = log_path.open("wb")
    process = subprocess.Popen(
        (sys.executable, str(script), str(child_pid_path)),
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    worker = driver_module._SubprocessWorker(  # noqa: SLF001
        ProductionWorkerRole.TERM1,
        process,
        stream,
    )
    deadline = time.monotonic() + 5.0
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert child_pid > 1

    worker.terminate()
    worker.wait(timeout_s=5.0)

    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_immutable_failure_archiver_never_overwrites(tmp_path: Path) -> None:
    binding = compile_production_driver_binding(_session(tmp_path))
    failure = SimpleNamespace(
        run_id=binding.session.run_spec.run_id,
        launch_config_sha256=binding.session.evidence_bundle.launch_config_sha256,
        injection_bundle_fingerprint=binding.session.evidence_bundle.fingerprint,
        stage="test",
        error_type="RuntimeError",
        error="boom",
        started_roles=("term1",),
        stopped_roles=("term1",),
    )
    # Exercise the concrete writer with the real frozen record type produced by
    # the driver, rather than permitting an arbitrary mapping.
    from coevolve_experiments.production.driver import ProductionDriverFailure

    record = ProductionDriverFailure(**vars(failure))
    archiver = ImmutableProductionFailureArchiver()
    archiver.archive(binding, record)

    path = binding.output_dir / archiver.filename
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["paper_evidence"] is False
    assert document["stage"] == "test"
    with pytest.raises(
        ProductionDriverInfrastructureError,
        match="cannot archive",
    ):
        archiver.archive(binding, record)
