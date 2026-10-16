"""Hash-bound subprocess bridge for the production Praxis runner.

This module never discovers a runner from the environment and never accepts a
free-form argv tail.  A caller first writes one closed-world launch document,
the bridge re-reads and reconstructs it from the frozen suite plan, and the
runner receives exactly ``--launch-config /absolute/path``.  Only a zero-exit
process whose two result files pass the protocol-neutral collector receives a
paper-evidence receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import time
from typing import Any, Mapping, Optional
import uuid

from ..contracts import RunSpec, TableId
from ..serialization import sha256_json, to_primitive
from .capability import (
    REQUIRED_SCENE_INJECTION_CHANNELS,
    RUNNER_LAUNCH_CONFIG_SCHEMA_VERSION,
    RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
    RunnerCapability,
    RunnerCapabilityError,
    load_runner_capability,
)
from .contracts import ExecutionCapability, LiveRunReceipt, PolicyRepresentation
from .launch import (
    LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
    CompiledLiveLaunch,
    _is_pristine_prepared_output,
    compile_live_launch,
)
from .preflight import LIVE_BRIDGE_UNAVAILABLE_CODE


SUBPROCESS_LAUNCH_CONFIG_KIND = "coevolve_praxis_subprocess_launch"
SUBPROCESS_LAUNCH_CONFIG_SCHEMA_VERSION = RUNNER_LAUNCH_CONFIG_SCHEMA_VERSION
LIVE_RESULT_BINDING_KIND = "coevolve_praxis_live_result_binding"
LIVE_BRIDGE_RECEIPT_KIND = "coevolve_praxis_live_bridge_receipt"
LIVE_BRIDGE_RECEIPT_SCHEMA_VERSION = 4
LIVE_BRIDGE_RECEIPT_FILENAME = "live_bridge_receipt.json"
LIVE_RUN_RESERVATION_KIND = "coevolve_praxis_live_run_reservation"
LIVE_RUN_RESERVATION_SCHEMA_VERSION = 1
LIVE_RUN_RESERVATION_SUFFIX = ".coevolve-live-run-reservation.json"

_PROCESS_GROUP_TERM_GRACE_SECONDS = 0.5

_OUTPUT_ALREADY_PRESENT_CODE = "PLANNED_OUTPUT_DIRECTORY_NOT_EMPTY"
_LAUNCH_CONFIG_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "launch",
        "runner_capability",
        "config_fingerprint",
    }
)
_CAPABILITY_BINDING_FIELDS = frozenset(
    {
        "manifest_path",
        "manifest_sha256",
        "runner_repo",
        "base_method_commit",
        "runner_integration_commit",
        "entrypoint_path",
        "entrypoint_sha256",
        "full_method_compatibility_fingerprint",
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
_BRIDGE_RECEIPT_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "paper_evidence_eligible",
        "protocol_valid",
        "run_id",
        "launch_fingerprint",
        "launch_config_path",
        "launch_config_sha256",
        "runner_capability_sha256",
        "base_method_commit",
        "runner_integration_commit",
        "full_method_compatibility_fingerprint",
        "run_manifest_sha256",
        "metrics_sha256",
        "terminal_status",
        "task_success",
        "receipt_fingerprint",
    }
)
_RUN_RESERVATION_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "run_id",
        "output_dir",
        "launch_config_path",
        "launch_config_sha256",
        "launch_fingerprint",
        "owner",
        "created_unix_ns",
        "reservation_fingerprint",
    }
)


class LiveBridgeConfigError(ValueError):
    """Raised before starting a subprocess when a launch is not exact."""


class LiveSubprocessError(RuntimeError):
    """Raised when the production subprocess cannot complete cleanly."""


class LiveResultValidationError(RuntimeError):
    """Raised when process outputs are not admissible paper evidence."""


@dataclass(frozen=True, slots=True)
class VerifiedSubprocessLaunchConfig:
    path: Path
    sha256: str
    config_fingerprint: str
    launch: CompiledLiveLaunch
    capability: RunnerCapability


@dataclass(frozen=True, slots=True)
class LiveRunReservation:
    """One process-owned, atomically-created claim on a canonical output.

    A reservation is deliberately outside ``output_dir`` so the exact V83
    memory bootstrap remains pristine.  Failed reservations are durable
    operator-review markers; only a successful, fully attested launch releases
    its own reservation automatically.
    """

    path: Path
    owner: str
    encoded: bytes
    descriptor: int
    device: int
    inode: int
    run_id: str
    launch_config_sha256: str


@dataclass(frozen=True, slots=True)
class PraxisSubprocessBridge:
    """Protocol implementation with no scene/method/memory override surface."""

    policy_representation: PolicyRepresentation = (
        PolicyRepresentation.PRAXIS_CODE_AS_POLICY
    )
    execution_capability: ExecutionCapability = ExecutionCapability.PAPER_FORMAL

    def launch(self, launch_config_path: Path) -> LiveRunReceipt:
        return launch_live_subprocess(launch_config_path)


def write_subprocess_launch_config(
    compiled: CompiledLiveLaunch,
    *,
    capability_manifest_path: str | Path,
    destination: str | Path,
) -> VerifiedSubprocessLaunchConfig:
    """Create one immutable runner input and immediately verify it from disk."""

    capability = _load_capability_for_launch(
        capability_manifest_path,
        compiled.run_spec,
    )
    activated = _activate_launch(compiled, capability, output_may_exist=False)
    _require_launch_fingerprint(activated)
    destination_path = _new_absolute_config_path(destination, activated.output_dir)
    capability_binding = _capability_binding(capability)
    fingerprint_payload = {
        "kind": SUBPROCESS_LAUNCH_CONFIG_KIND,
        "schema_version": SUBPROCESS_LAUNCH_CONFIG_SCHEMA_VERSION,
        "launch": to_primitive(activated),
        "runner_capability": capability_binding,
    }
    document = {
        **fingerprint_payload,
        "config_fingerprint": sha256_json(fingerprint_payload),
    }
    _write_exclusive_json(destination_path, document)
    try:
        return load_subprocess_launch_config(destination_path)
    except Exception:
        # This exact file was created by this call and has never been handed to
        # a runner.  Do not leave a malformed launch token behind.
        destination_path.unlink(missing_ok=True)
        raise


def load_subprocess_launch_config(
    path: str | Path,
    *,
    output_may_exist: bool = False,
) -> VerifiedSubprocessLaunchConfig:
    """Re-read, reconstruct, and compare every launch binding from source."""

    config_path = _absolute_regular_path(path, "launch config")
    raw, payload = _read_json_object(config_path, "launch config")
    _require_exact_fields(payload, _LAUNCH_CONFIG_FIELDS, "launch config")
    if payload.get("kind") != SUBPROCESS_LAUNCH_CONFIG_KIND:
        raise LiveBridgeConfigError("launch config has the wrong kind")
    _require_exact_integer(
        payload.get("schema_version"),
        SUBPROCESS_LAUNCH_CONFIG_SCHEMA_VERSION,
        "launch config schema_version",
    )
    launch_payload = payload.get("launch")
    capability_payload = payload.get("runner_capability")
    if not isinstance(launch_payload, Mapping):
        raise LiveBridgeConfigError("launch config launch must be an object")
    if not isinstance(capability_payload, Mapping):
        raise LiveBridgeConfigError(
            "launch config runner_capability must be an object"
        )
    _require_exact_fields(
        capability_payload,
        _CAPABILITY_BINDING_FIELDS,
        "launch config runner_capability",
    )

    fingerprint_payload = {
        "kind": payload["kind"],
        "schema_version": payload["schema_version"],
        "launch": launch_payload,
        "runner_capability": capability_payload,
    }
    expected_config_fingerprint = sha256_json(fingerprint_payload)
    if payload.get("config_fingerprint") != expected_config_fingerprint:
        raise LiveBridgeConfigError("launch config fingerprint differs")

    run_payload = launch_payload.get("run_spec")
    if not isinstance(run_payload, Mapping):
        raise LiveBridgeConfigError("launch run_spec must be an object")
    run_id = _required_string(run_payload, "run_id", "launch run_spec")
    base_method_commit = _required_string(
        run_payload,
        "base_method_commit",
        "launch run_spec",
    )
    full_method_compatibility_fingerprint = _required_string(
        run_payload,
        "full_method_compatibility_fingerprint",
        "launch run_spec",
    )
    manifest_path = _absolute_path_string(
        capability_payload,
        "manifest_path",
        "runner capability manifest",
    )
    capability = _load_capability_for_commit(
        manifest_path,
        base_method_commit,
        full_method_compatibility_fingerprint,
    )
    if dict(capability_payload) != _capability_binding(capability):
        raise LiveBridgeConfigError(
            "launch runner capability binding differs from current bytes or checkout"
        )

    plan_path = _absolute_path_string(
        launch_payload,
        "plan_path",
        "suite plan",
    )
    results_root = _absolute_path_string(
        launch_payload,
        "results_root",
        "formal results root",
    )
    memory_payload = launch_payload.get("memory")
    if not isinstance(memory_payload, Mapping):
        raise LiveBridgeConfigError("launch memory must be an object")
    strategy_bank_path = _optional_absolute_path_string(
        memory_payload,
        "strategy_bank_path",
        "strategy bank",
    )
    provenance_path = _optional_absolute_path_string(
        memory_payload,
        "strategy_bank_provenance_path",
        "strategy bank provenance",
    )

    try:
        rebuilt = compile_live_launch(
            plan_path,
            run_id,
            results_root=results_root,
            strategy_bank_path=strategy_bank_path,
            strategy_bank_provenance_path=provenance_path,
            require_launchable=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise LiveBridgeConfigError(
            f"cannot reconstruct launch config from frozen inputs: {exc}"
        ) from exc
    activated = _activate_launch(
        rebuilt,
        capability,
        output_may_exist=output_may_exist,
    )
    try:
        current_plan_sha256 = hashlib.sha256(activated.plan_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise LiveBridgeConfigError(f"cannot re-read frozen suite plan: {exc}") from exc
    if current_plan_sha256 != activated.plan_sha256:
        raise LiveBridgeConfigError("suite plan bytes differ from plan_sha256")
    _require_launch_fingerprint(activated)
    if to_primitive(activated) != dict(launch_payload):
        raise LiveBridgeConfigError(
            "serialized launch differs from the launch reconstructed from plan/run/output/memory"
        )
    if not output_may_exist:
        _require_pristine_output(activated)
    _require_config_outside_output(config_path, activated.output_dir)
    return VerifiedSubprocessLaunchConfig(
        path=config_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        config_fingerprint=expected_config_fingerprint,
        launch=activated,
        capability=capability,
    )


def launch_live_subprocess(
    launch_config_path: str | Path,
    *,
    timeout_seconds: float | None = None,
) -> LiveRunReceipt:
    """Execute exactly one config-bound process and attest its result files.

    The output is reserved with an ``O_EXCL`` sibling file before the runner
    starts.  The launch is then reconstructed a second time while that claim
    is held, closing the pristine-check/exec race.  A reservation is removed
    only after a paper receipt has been written successfully, or when failure
    occurs before a child was started.  Once execution starts, any timeout,
    non-zero exit, or result-validation failure retains the reservation for an
    explicit audited archive instead of silently making the cell runnable
    again.
    """

    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise LiveBridgeConfigError("timeout_seconds must be positive or None")
    initial = load_subprocess_launch_config(launch_config_path)
    output_dir = initial.launch.output_dir
    receipt_path = output_dir / LIVE_BRIDGE_RECEIPT_FILENAME
    reservation = _acquire_run_reservation(initial)
    process: subprocess.Popen[str] | None = None
    try:
        # The reservation lives beside, rather than inside, output_dir, so a
        # second closed-world load must still observe the exact pristine
        # bootstrap.  It also catches launch/capability drift after the first
        # check but before process creation.
        verified = load_subprocess_launch_config(launch_config_path)
        if (
            verified.sha256 != initial.sha256
            or verified.config_fingerprint != initial.config_fingerprint
            or verified.launch.launch_fingerprint
            != initial.launch.launch_fingerprint
            or verified.launch.output_dir != initial.launch.output_dir
        ):
            raise LiveBridgeConfigError(
                "launch config changed while acquiring its output reservation"
            )
        command = [
            str(verified.capability.entrypoint_path),
            "--launch-config",
            str(verified.path),
        ]
        process = _start_runner_process(
            command,
            cwd=str(verified.capability.runner_repo),
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            # Closing the whole group makes inherited stdout/stderr pipes
            # finite.  Reap the direct child before reporting the timeout.
            try:
                process.communicate(timeout=_PROCESS_GROUP_TERM_GRACE_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                _terminate_process_group(process, term_grace_seconds=0.0)
            raise LiveSubprocessError(
                f"production runner exceeded timeout_seconds={timeout_seconds}"
            ) from exc
        except BaseException:
            _terminate_process_group(process)
            raise

        if process.returncode != 0:
            _terminate_process_group(process)
            detail = (stderr or stdout or "").strip()
            if len(detail) > 4000:
                detail = detail[-4000:]
            suffix = f": {detail}" if detail else ""
            raise LiveSubprocessError(
                f"production runner exited with status {process.returncode}{suffix}"
            )
        # A zero-exit leader is not sufficient if it orphaned a worker that
        # can continue mutating evidence.  Kill that group and reject the run
        # before collector/receipt publication.
        if _process_group_exists(process.pid):
            _terminate_process_group(process)
            raise LiveSubprocessError(
                "production runner exited while child processes remained alive"
            )

        post_run = load_subprocess_launch_config(
            verified.path,
            output_may_exist=True,
        )
        if post_run.sha256 != verified.sha256:
            raise LiveResultValidationError(
                "runner mutated the launch config during execution"
            )
        receipt = _validate_and_attest_results(post_run)
        _release_run_reservation(reservation)
        return receipt
    except BaseException as exc:
        _discard_untrusted_receipt(receipt_path)
        if process is None:
            try:
                _release_run_reservation(reservation)
            except BaseException as release_exc:
                raise LiveSubprocessError(
                    f"{type(exc).__name__}: {exc}; pre-start reservation "
                    f"release failed: {type(release_exc).__name__}: {release_exc}"
                ) from exc
        else:
            # A child was created: retain the immutable reservation so this
            # canonical output cannot be retried until the failed attempt is
            # explicitly inspected and archived.
            _retain_run_reservation(reservation)
        if isinstance(
            exc,
            (LiveBridgeConfigError, LiveResultValidationError, LiveSubprocessError),
        ):
            raise
        if isinstance(exc, OSError):
            raise LiveSubprocessError(
                f"production runner did not complete: {exc}"
            ) from exc
        raise


def _validate_and_attest_results(
    verified: VerifiedSubprocessLaunchConfig,
) -> LiveRunReceipt:
    output_dir = verified.launch.output_dir
    (
        manifest_path,
        metrics_path,
        manifest_raw,
        metrics_raw,
        manifest,
        _metrics,
    ) = _read_and_validate_result_bindings(verified)

    # Local import avoids making the fail-closed live package initialize the
    # reporting/memory package graph merely to compile a launch.
    from ..reporting.collect import collect_run

    expected = _collector_expectation(verified)
    record = collect_run(
        output_dir,
        expected=expected,
        physical_budget=verified.launch.run_spec.physical_budget,
    )
    _require_from_scratch_memory_evidence(verified, manifest)
    if not record.protocol_valid:
        raise LiveResultValidationError(
            "runner results are not protocol-valid paper evidence: "
            + ", ".join(record.invalid_reasons)
        )
    if record.infrastructure_invalid:
        raise LiveResultValidationError(
            "infrastructure-invalid attempts cannot become paper evidence"
        )

    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    metrics_sha256 = hashlib.sha256(metrics_raw).hexdigest()
    receipt_path = output_dir / LIVE_BRIDGE_RECEIPT_FILENAME
    receipt_payload = {
        "kind": LIVE_BRIDGE_RECEIPT_KIND,
        "schema_version": LIVE_BRIDGE_RECEIPT_SCHEMA_VERSION,
        "paper_evidence_eligible": True,
        "protocol_valid": True,
        "run_id": verified.launch.run_spec.run_id,
        "launch_fingerprint": verified.launch.launch_fingerprint,
        "launch_config_path": str(verified.path),
        "launch_config_sha256": verified.sha256,
        "runner_capability_sha256": verified.capability.manifest_sha256,
        "base_method_commit": verified.capability.base_method_commit,
        "runner_integration_commit": verified.capability.runner_integration_commit,
        "full_method_compatibility_fingerprint": (
            verified.capability.full_method_compatibility.fingerprint
        ),
        "run_manifest_sha256": manifest_sha256,
        "metrics_sha256": metrics_sha256,
        "terminal_status": record.terminal_status,
        "task_success": record.task_success,
    }
    receipt_fingerprint = sha256_json(receipt_payload)
    receipt_document = {
        **receipt_payload,
        "receipt_fingerprint": receipt_fingerprint,
    }
    _write_exclusive_json(receipt_path, receipt_document)
    receipt_raw = receipt_path.read_bytes()
    return LiveRunReceipt(
        run_id=verified.launch.run_spec.run_id,
        output_dir=output_dir,
        run_manifest_path=manifest_path,
        metrics_path=metrics_path,
        launch_config_path=verified.path,
        bridge_receipt_path=receipt_path,
        launch_config_sha256=verified.sha256,
        run_manifest_sha256=manifest_sha256,
        metrics_sha256=metrics_sha256,
        bridge_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        launch_fingerprint=verified.launch.launch_fingerprint,
        base_method_commit=verified.capability.base_method_commit,
        runner_integration_commit=verified.capability.runner_integration_commit,
        terminal_status=record.terminal_status,
        task_success=record.task_success,
        protocol_valid=True,
        paper_evidence_eligible=True,
    )


def validate_live_bridge_receipt(
    run_dir: str | Path,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Revalidate one durable bridge receipt without issuing a new receipt.

    Formal aggregation uses this boundary instead of consulting a process-global
    bridge availability flag.  A receipt remains valid only while its frozen
    launch configuration reconstructs and its bound result bytes are unchanged.
    """

    output_dir = Path(run_dir).resolve(strict=False)
    reservation_path = live_run_reservation_path(output_dir)
    if reservation_path.exists() or reservation_path.is_symlink():
        raise LiveResultValidationError(
            "live bridge receipt has a residual run reservation requiring "
            "operator review"
        )
    receipt_path = _result_file(output_dir, LIVE_BRIDGE_RECEIPT_FILENAME)
    _receipt_raw, receipt = _read_result_object(
        receipt_path,
        "live bridge receipt",
    )
    if frozenset(receipt) != _BRIDGE_RECEIPT_FIELDS:
        raise LiveResultValidationError(
            "live bridge receipt does not use the exact receipt schema"
        )
    if receipt.get("kind") != LIVE_BRIDGE_RECEIPT_KIND:
        raise LiveResultValidationError("live bridge receipt has the wrong kind")
    try:
        _require_exact_integer(
            receipt.get("schema_version"),
            LIVE_BRIDGE_RECEIPT_SCHEMA_VERSION,
            "live bridge receipt schema_version",
        )
    except LiveBridgeConfigError as exc:
        raise LiveResultValidationError(str(exc)) from exc
    if receipt.get("paper_evidence_eligible") is not True:
        raise LiveResultValidationError(
            "live bridge receipt is not paper-evidence eligible"
        )
    if receipt.get("protocol_valid") is not True:
        raise LiveResultValidationError(
            "live bridge receipt is not protocol valid"
        )

    fingerprint_payload = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_fingerprint"
    }
    if receipt.get("receipt_fingerprint") != sha256_json(fingerprint_payload):
        raise LiveResultValidationError("live bridge receipt fingerprint differs")

    launch_config_value = receipt.get("launch_config_path")
    if not isinstance(launch_config_value, str) or not launch_config_value:
        raise LiveResultValidationError(
            "live bridge receipt launch_config_path must be a non-empty string"
        )
    try:
        verified = load_subprocess_launch_config(
            Path(launch_config_value),
            output_may_exist=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise LiveResultValidationError(
            f"live bridge receipt launch config is not valid: {exc}"
        ) from exc
    if verified.launch.output_dir != output_dir:
        raise LiveResultValidationError(
            "live bridge receipt output directory differs from the launch config"
        )
    run_id = verified.launch.run_spec.run_id
    if expected_run_id is not None and run_id != expected_run_id:
        raise LiveResultValidationError(
            "live bridge receipt run_id differs from the planned run"
        )
    if receipt.get("run_id") != run_id:
        raise LiveResultValidationError(
            "live bridge receipt run_id differs from the launch config"
        )
    expected_receipt_binding = {
        "launch_fingerprint": verified.launch.launch_fingerprint,
        "launch_config_path": str(verified.path),
        "launch_config_sha256": verified.sha256,
        "runner_capability_sha256": verified.capability.manifest_sha256,
        "base_method_commit": verified.capability.base_method_commit,
        "runner_integration_commit": verified.capability.runner_integration_commit,
        "full_method_compatibility_fingerprint": (
            verified.capability.full_method_compatibility.fingerprint
        ),
    }
    for key, expected in expected_receipt_binding.items():
        if receipt.get(key) != expected:
            raise LiveResultValidationError(
                f"live bridge receipt {key} differs from the executed launch"
            )

    (
        _manifest_path,
        _metrics_path,
        manifest_raw,
        metrics_raw,
        manifest,
        metrics,
    ) = _read_and_validate_result_bindings(verified)
    if receipt.get("run_manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest():
        raise LiveResultValidationError(
            "live bridge receipt run_manifest_sha256 differs from current bytes"
        )
    if receipt.get("metrics_sha256") != hashlib.sha256(metrics_raw).hexdigest():
        raise LiveResultValidationError(
            "live bridge receipt metrics_sha256 differs from current bytes"
        )

    # A durable receipt binds the two top-level result files, but those files
    # in turn reference raw terminal evidence.  Re-run the protocol-neutral
    # collector so batch resume cannot accept a receipt after a referenced
    # MuJoCo state/ledger has been removed or modified.
    from ..reporting.collect import collect_run

    record = collect_run(
        output_dir,
        expected=_collector_expectation(verified),
        physical_budget=verified.launch.run_spec.physical_budget,
    )
    _require_from_scratch_memory_evidence(verified, manifest)
    if not record.protocol_valid or record.infrastructure_invalid:
        detail = ", ".join(record.invalid_reasons) or record.terminal_status
        raise LiveResultValidationError(
            "live bridge receipt no longer resolves to protocol-valid paper "
            f"evidence: {detail}"
        )
    if receipt.get("terminal_status") != record.terminal_status:
        raise LiveResultValidationError(
            "live bridge receipt terminal_status differs from current results"
        )
    if receipt.get("task_success") is not record.task_success:
        raise LiveResultValidationError(
            "live bridge receipt task_success differs from current metrics"
        )
    return dict(receipt)


def _require_from_scratch_memory_evidence(
    verified: VerifiedSubprocessLaunchConfig,
    manifest: Mapping[str, Any],
) -> None:
    """Require finalized empty-start evidence before issuing/reusing a receipt."""

    run = verified.launch.run_spec
    if run.table_id not in {TableId.TABLE_2, TableId.TABLE_3}:
        return
    from ..memory.from_scratch import from_scratch_run_binding
    from ..reporting.from_scratch_memory_audit import (
        validate_from_scratch_memory_audit,
    )

    expected_binding = from_scratch_run_binding(
        run,
        plan_path=verified.launch.plan_path,
        plan_sha256=verified.launch.plan_sha256,
    )
    validation = validate_from_scratch_memory_audit(
        verified.launch.output_dir,
        expected_run_binding=expected_binding,
        expected_memory_condition=run.memory_condition.value,
        expected_memory_policy=run.memory_policy.value,
    )
    if validation.invalid_reasons:
        raise LiveResultValidationError(
            "from-scratch memory evidence is not protocol valid: "
            + ", ".join(validation.invalid_reasons)
        )
    declared = manifest.get(
        "result_artifact_hashes",
        manifest.get("artifact_hashes"),
    )
    if not isinstance(declared, Mapping):
        raise LiveResultValidationError(
            "run manifest does not bind from-scratch memory evidence"
        )
    for name, digest in validation.artifact_hashes.items():
        if not digest or declared.get(name) != digest:
            raise LiveResultValidationError(
                f"run manifest {name} differs from finalized memory evidence"
            )


def _read_and_validate_result_bindings(
    verified: VerifiedSubprocessLaunchConfig,
) -> tuple[Path, Path, bytes, bytes, Mapping[str, Any], Mapping[str, Any]]:
    output_dir = verified.launch.output_dir
    manifest_path = _result_file(output_dir, "run_manifest.json")
    metrics_path = _result_file(output_dir, "metrics.json")
    manifest_raw, manifest = _read_result_object(manifest_path, "run manifest")
    metrics_raw, metrics = _read_result_object(metrics_path, "metrics")
    expected_binding = _expected_result_binding(verified)
    for label, payload in (("run manifest", manifest), ("metrics", metrics)):
        binding = payload.get("live_launch")
        if not isinstance(binding, Mapping):
            raise LiveResultValidationError(
                f"{label} is missing its live_launch result binding"
            )
        if frozenset(binding) != _RESULT_BINDING_FIELDS:
            raise LiveResultValidationError(
                f"{label} live_launch binding does not use the exact result schema"
            )
        if dict(binding) != expected_binding:
            raise LiveResultValidationError(
                f"{label} live_launch binding differs from the executed launch"
            )
    return (
        manifest_path,
        metrics_path,
        manifest_raw,
        metrics_raw,
        manifest,
        metrics,
    )


def _activate_launch(
    compiled: CompiledLiveLaunch,
    capability: RunnerCapability,
    *,
    output_may_exist: bool,
) -> CompiledLiveLaunch:
    if compiled.schema_version != LIVE_LAUNCH_CONFIG_SCHEMA_VERSION:
        raise LiveBridgeConfigError("compiled launch uses an unsupported schema")
    if compiled.run_spec.base_method_commit != capability.base_method_commit:
        raise LiveBridgeConfigError(
            "runner capability base commit differs from RunSpec.base_method_commit"
        )
    if (
        compiled.run_spec.full_method_compatibility_fingerprint
        != capability.full_method_compatibility.fingerprint
    ):
        raise LiveBridgeConfigError(
            "runner capability Full-method identity differs from RunSpec"
        )
    if compiled.method.policy_representation is not PolicyRepresentation.PRAXIS_CODE_AS_POLICY:
        raise LiveBridgeConfigError("compiled method is not Praxis Code-as-Policy")
    if capability.execution_capability is not ExecutionCapability.PAPER_FORMAL:
        raise LiveBridgeConfigError("runner capability is not paper_formal")
    if compiled.run_spec.method_variant.value not in capability.supported_method_variants:
        raise LiveBridgeConfigError("runner does not support the selected method variant")
    if compiled.run_spec.scene_id.value not in capability.supported_scene_ids:
        raise LiveBridgeConfigError("runner does not support the selected scene")
    if tuple(compiled.scene.injection_channels) != REQUIRED_SCENE_INJECTION_CHANNELS:
        raise LiveBridgeConfigError(
            "compiled scene does not bind every required injection channel"
        )
    if tuple(compiled.scene.injection_channels) != capability.scene_injection_channels:
        raise LiveBridgeConfigError(
            "compiled scene injection channels differ from runner capability"
        )

    allowed = {LIVE_BRIDGE_UNAVAILABLE_CODE}
    if output_may_exist:
        allowed.add(_OUTPUT_ALREADY_PRESENT_CODE)
    other_blockers = tuple(item for item in compiled.blockers if item not in allowed)
    if other_blockers:
        raise LiveBridgeConfigError(
            "formal launch has unresolved blockers: " + ", ".join(other_blockers)
        )
    return replace(
        compiled,
        execution_capability=capability.execution_capability,
        paper_evidence_eligible=True,
        blockers=(),
    )


def _require_launch_fingerprint(compiled: CompiledLiveLaunch) -> None:
    fingerprint_payload = {
        "schema_version": compiled.schema_version,
        "run_spec": compiled.run_spec,
        "plan_sha256": compiled.plan_sha256,
        "results_root": compiled.results_root,
        "output_dir": compiled.output_dir,
        "method": compiled.method,
        "scene": compiled.scene,
        "memory": compiled.memory,
        "execution_capability": compiled.execution_capability,
    }
    expected = sha256_json(fingerprint_payload)
    if compiled.launch_fingerprint != expected:
        raise LiveBridgeConfigError("compiled launch fingerprint differs")


def _load_capability_for_launch(
    path: str | Path, run: RunSpec
) -> RunnerCapability:
    return _load_capability_for_commit(
        path,
        run.base_method_commit,
        run.full_method_compatibility_fingerprint,
    )


def _load_capability_for_commit(
    path: str | Path,
    commit: str,
    full_method_compatibility_fingerprint: str,
) -> RunnerCapability:
    try:
        return load_runner_capability(
            path,
            expected_base_method_commit=commit,
            expected_full_method_compatibility_fingerprint=(
                full_method_compatibility_fingerprint
            ),
        )
    except (OSError, TypeError, ValueError, RunnerCapabilityError) as exc:
        raise LiveBridgeConfigError(f"runner capability is not valid: {exc}") from exc


def _capability_binding(capability: RunnerCapability) -> dict[str, Any]:
    return {
        "manifest_path": str(capability.manifest_path),
        "manifest_sha256": capability.manifest_sha256,
        "runner_repo": str(capability.runner_repo),
        "base_method_commit": capability.base_method_commit,
        "runner_integration_commit": capability.runner_integration_commit,
        "entrypoint_path": str(capability.entrypoint_path),
        "entrypoint_sha256": capability.entrypoint_sha256,
        "full_method_compatibility_fingerprint": (
            capability.full_method_compatibility.fingerprint
        ),
    }


def _expected_result_binding(
    verified: VerifiedSubprocessLaunchConfig,
) -> dict[str, Any]:
    return {
        "kind": LIVE_RESULT_BINDING_KIND,
        "schema_version": RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
        "run_id": verified.launch.run_spec.run_id,
        "run_spec_sha256": sha256_json(verified.launch.run_spec),
        "launch_fingerprint": verified.launch.launch_fingerprint,
        "config_fingerprint": verified.config_fingerprint,
        "launch_config_sha256": verified.sha256,
        "plan_sha256": verified.launch.plan_sha256,
        "runner_capability_sha256": verified.capability.manifest_sha256,
        "base_method_commit": verified.capability.base_method_commit,
        "runner_integration_commit": verified.capability.runner_integration_commit,
        "entrypoint_sha256": verified.capability.entrypoint_sha256,
        "full_method_compatibility_fingerprint": (
            verified.capability.full_method_compatibility.fingerprint
        ),
    }


def _collector_expectation(
    verified: VerifiedSubprocessLaunchConfig,
) -> dict[str, Any]:
    run = verified.launch.run_spec
    # These values were resolved from the plan's revalidated scene report when
    # the immutable launch was compiled; the runner may not self-declare them.
    scene_result = verified.launch.scene
    parts = Path(run.output_relpath).parts
    condition = parts[1] if len(parts) >= 2 else ""
    strategy_bank_sha256 = run.strategy_bank.sha256 if run.strategy_bank else ""
    source_model = run.strategy_bank.source_backbone if run.strategy_bank else ""
    source_provider_route_id = (
        run.strategy_bank.source_provider_route_id if run.strategy_bank else ""
    )
    source_provider_route_fingerprint = (
        run.strategy_bank.source_provider_route_fingerprint
        if run.strategy_bank
        else ""
    )
    return {
        "run_id": run.run_id,
        "table": run.table_id.value,
        "condition": condition,
        "method": run.method_variant.value,
        "scene": run.scene_id.value,
        "memory_condition": run.memory_condition.value,
        "memory_policy": run.memory_policy.value,
        "source_model": source_model,
        "source_provider_route_id": source_provider_route_id,
        "source_provider_route_fingerprint": source_provider_route_fingerprint,
        "target_model": run.target_backbone,
        "provider_route_id": run.provider_route_id,
        "provider_route_fingerprint": run.provider_route_fingerprint,
        "seed": run.seed,
        "protocol_id": run.protocol_id,
        "base_method_commit": run.base_method_commit,
        "condition_fingerprint": run.condition_fingerprint,
        "invariant_fingerprint": run.invariant_fingerprint,
        "scene_fingerprint": run.scene_fingerprint,
        "scene_realization_sha256": scene_result.execution_composite_sha256,
        "scene_execution_evidence_fingerprint": (
            scene_result.execution_evidence_fingerprint
        ),
        "compiled_model_sha256": scene_result.compiled_model_sha256,
        "target_center_xyz_m": list(scene_result.target_center_xyz_m),
        "terminal_fingerprint": run.terminal_fingerprint,
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
        "strategy_bank_sha256": strategy_bank_sha256,
        "plan_path": str(verified.launch.plan_path),
        "plan_sha256": verified.launch.plan_sha256,
        # A production bridge receipt is stronger than generic/ad-hoc
        # collection: it must be backed by the Host-owned terminal verifier
        # receipt and its hash-bound raw MuJoCo artifacts.
        "require_terminal_evidence": True,
        "require_usage_evidence": True,
        "require_native_memory_evidence": True,
        "require_transaction_accounting_evidence": True,
    }


def live_run_reservation_path(output_dir: str | Path) -> Path:
    """Return the sole reservation path for one canonical output directory."""

    output = Path(output_dir)
    if not output.is_absolute() or output.resolve(strict=False) != output:
        raise LiveBridgeConfigError(
            "reservation output path must be absolute and normalized"
        )
    if not output.name:
        raise LiveBridgeConfigError("reservation output path has no final component")
    return output.parent / f".{output.name}{LIVE_RUN_RESERVATION_SUFFIX}"


def _acquire_run_reservation(
    verified: VerifiedSubprocessLaunchConfig,
) -> LiveRunReservation:
    """Atomically claim a run/output pair before process creation.

    Existing reservations are never recovered automatically, even if their
    recorded PID is gone.  A dead owner is indistinguishable from a process
    that wrote partial evidence, so recovery belongs to the audited attempt
    archive workflow.
    """

    output_dir = verified.launch.output_dir
    reservation_path = live_run_reservation_path(output_dir)
    reservation_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_parent = reservation_path.parent.resolve(strict=True)
    except OSError as exc:
        raise LiveBridgeConfigError(
            f"cannot resolve reservation parent: {exc}"
        ) from exc
    if (
        reservation_path.parent.is_symlink()
        or resolved_parent != reservation_path.parent
    ):
        raise LiveBridgeConfigError(
            "reservation parent must be normalized and contain no symlink components"
        )

    owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
    payload = {
        "kind": LIVE_RUN_RESERVATION_KIND,
        "schema_version": LIVE_RUN_RESERVATION_SCHEMA_VERSION,
        "run_id": verified.launch.run_spec.run_id,
        "output_dir": str(output_dir),
        "launch_config_path": str(verified.path),
        "launch_config_sha256": verified.sha256,
        "launch_fingerprint": verified.launch.launch_fingerprint,
        "owner": owner,
        "created_unix_ns": time.time_ns(),
    }
    document = {
        **payload,
        "reservation_fingerprint": sha256_json(payload),
    }
    if frozenset(document) != _RUN_RESERVATION_FIELDS:
        raise LiveBridgeConfigError("internal live-run reservation schema drift")
    encoded = _encode_json_document(document)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(reservation_path, flags, 0o600)
    except FileExistsError as exc:
        raise LiveBridgeConfigError(
            "planned output already has a live-run reservation; automatic "
            "stale-lock recovery is forbidden"
        ) from exc
    except OSError as exc:
        raise LiveBridgeConfigError(
            f"cannot create live-run reservation: {exc}"
        ) from exc
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating reservation")
            view = view[written:]
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise LiveBridgeConfigError("live-run reservation is not a regular file")
        if reservation_path.read_bytes() != encoded:
            raise LiveBridgeConfigError(
                "live-run reservation bytes changed during atomic publication"
            )
        _fsync_directory(reservation_path.parent)
        return LiveRunReservation(
            path=reservation_path,
            owner=owner,
            encoded=encoded,
            descriptor=descriptor,
            device=info.st_dev,
            inode=info.st_ino,
            run_id=verified.launch.run_spec.run_id,
            launch_config_sha256=verified.sha256,
        )
    except BaseException:
        # The path may contain a partial claim.  Keep it fail-closed; only
        # close this process's descriptor and never guess that unlink is safe.
        os.close(descriptor)
        raise


def _release_run_reservation(reservation: LiveRunReservation) -> None:
    """Release only the exact inode and owner bytes created by this process."""

    descriptor = reservation.descriptor
    try:
        held = os.fstat(descriptor)
        try:
            current = reservation.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise LiveBridgeConfigError(
                "owned live-run reservation disappeared before release"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino)
            != (reservation.device, reservation.inode)
            or (held.st_dev, held.st_ino)
            != (reservation.device, reservation.inode)
        ):
            raise LiveBridgeConfigError(
                "live-run reservation ownership changed; refusing to remove it"
            )
        if reservation.path.read_bytes() != reservation.encoded:
            raise LiveBridgeConfigError(
                "live-run reservation owner bytes changed; refusing to remove it"
            )
        reservation.path.unlink()
        _fsync_directory(reservation.path.parent)
    finally:
        os.close(descriptor)


def _retain_run_reservation(reservation: LiveRunReservation) -> None:
    """Close the local handle while preserving a failed-attempt marker."""

    try:
        os.close(reservation.descriptor)
    except OSError:
        pass


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_runner_process(
    command: list[str],
    *,
    cwd: str,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        shell=False,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    term_grace_seconds: float = _PROCESS_GROUP_TERM_GRACE_SECONDS,
) -> None:
    """Terminate every process in the runner's dedicated POSIX session."""

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise LiveSubprocessError(
            f"cannot terminate production runner process group: {exc}"
        ) from exc

    deadline = time.monotonic() + max(0.0, term_grace_seconds)
    while _process_group_exists(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise LiveSubprocessError(
                f"cannot kill production runner process group: {exc}"
            ) from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _new_absolute_config_path(value: str | Path, output_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise LiveBridgeConfigError("launch config path must be absolute")
    if path.exists() or path.is_symlink():
        raise LiveBridgeConfigError("launch config destination must not already exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=False)
    if resolved != path:
        raise LiveBridgeConfigError(
            "launch config path must be normalized and contain no symlink components"
        )
    _require_config_outside_output(path, output_dir)
    return path


def _absolute_regular_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise LiveBridgeConfigError(f"{label} path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise LiveBridgeConfigError(
            f"{label} path must be a non-symlink existing regular file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise LiveBridgeConfigError(
            f"{label} path must be normalized and contain no symlink components"
        )
    return resolved


def _absolute_path_string(
    payload: Mapping[str, Any], key: str, label: str
) -> Path:
    value = _required_string(payload, key, label)
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise LiveBridgeConfigError(
            f"{label} path must be absolute, normalized, and non-symlinked"
        )
    return path


def _optional_absolute_path_string(
    payload: Mapping[str, Any], key: str, label: str
) -> Optional[Path]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise LiveBridgeConfigError(f"{label} path must be null or a non-empty string")
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise LiveBridgeConfigError(
            f"{label} path must be absolute, normalized, and non-symlinked"
        )
    return path


def _require_pristine_output(compiled: CompiledLiveLaunch) -> None:
    output_dir = compiled.output_dir
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise LiveBridgeConfigError(
                "planned output must be absent or a non-symlink empty directory"
            )
        if any(output_dir.iterdir()):
            if _is_pristine_prepared_output(
                output_dir,
                compiled.run_spec,
                compiled.memory,
            ):
                return
            raise LiveBridgeConfigError(
                "planned output directory is neither empty nor an exact "
                "plan-bound memory bootstrap"
            )


def _require_config_outside_output(config_path: Path, output_dir: Path) -> None:
    try:
        config_path.relative_to(output_dir)
    except ValueError:
        return
    raise LiveBridgeConfigError("launch config must be outside the planned output")


def _result_file(output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    if path.is_symlink() or not path.is_file():
        raise LiveResultValidationError(
            f"runner did not produce a regular non-symlink {filename}"
        )
    return path


def _read_result_object(
    path: Path, label: str
) -> tuple[bytes, Mapping[str, Any]]:
    try:
        return _read_json_object(path, label)
    except LiveBridgeConfigError as exc:
        raise LiveResultValidationError(str(exc)) from exc


def _read_json_object(
    path: Path, label: str
) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LiveBridgeConfigError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LiveBridgeConfigError(f"{label} must be a JSON object")
    return raw, payload


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _required_string(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise LiveBridgeConfigError(f"{label} {key} must be a non-empty string")
    return value


def _require_exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise LiveBridgeConfigError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_exact_integer(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise LiveBridgeConfigError(f"{label} must equal integer {expected}")


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _encode_json_document(payload)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise LiveBridgeConfigError(f"refusing to overwrite existing file: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _encode_json_document(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            to_primitive(payload),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _discard_untrusted_receipt(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


__all__ = [
    "LIVE_BRIDGE_RECEIPT_FILENAME",
    "LIVE_BRIDGE_RECEIPT_KIND",
    "LIVE_BRIDGE_RECEIPT_SCHEMA_VERSION",
    "LIVE_RESULT_BINDING_KIND",
    "LIVE_RUN_RESERVATION_KIND",
    "LIVE_RUN_RESERVATION_SCHEMA_VERSION",
    "LIVE_RUN_RESERVATION_SUFFIX",
    "SUBPROCESS_LAUNCH_CONFIG_KIND",
    "SUBPROCESS_LAUNCH_CONFIG_SCHEMA_VERSION",
    "LiveBridgeConfigError",
    "LiveResultValidationError",
    "LiveSubprocessError",
    "PraxisSubprocessBridge",
    "VerifiedSubprocessLaunchConfig",
    "live_run_reservation_path",
    "launch_live_subprocess",
    "load_subprocess_launch_config",
    "validate_live_bridge_receipt",
    "write_subprocess_launch_config",
]
