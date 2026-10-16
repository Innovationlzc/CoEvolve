from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

from coevolve_experiments.memory.v83_native import (
    V83MemoryOperationGates,
    V83NativeMemoryBridge,
    V83_NATIVE_AUDIT_FILENAME,
)
from coevolve_experiments.memory.validation import validate_strategy_bank
from coevolve_experiments.reporting.v83_native_memory_audit import (
    V83_NATIVE_MEMORY_RELATIVE_ROOT,
    validate_v83_native_memory_audit,
)


_RUN_BINDING = {
    "run_id": "table2-method-seed-200",
    "plan_sha256": "a" * 64,
}
_VALID_TERMINAL = "SUCCESS"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _native_root(run_dir: Path) -> Path:
    return run_dir / V83_NATIVE_MEMORY_RELATIVE_ROOT


def _audit_path(run_dir: Path) -> Path:
    return _native_root(run_dir) / V83_NATIVE_AUDIT_FILENAME


class _StrategyStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = _read(path) if path.exists() else {
            "kind": "deployment_decision_memory",
            "schema_version": 2,
            "description": "audit-test native strategy store",
            "specs": [],
        }

    @property
    def specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self.data["specs"])

    def retrieve(self, context: object, *, limit: int) -> list[str]:
        del context
        return [str(item["id"]) for item in self.data["specs"][:limit]]

    def update_from_rollout(
        self,
        record: dict[str, Any],
        *,
        default_phase: str,
        model_proposal: dict[str, Any] | None,
    ) -> bool:
        del default_phase, model_proposal
        self.data["specs"].append(dict(record))
        return True

    def save(self) -> None:
        _write(self.path, self.data)


class _ActionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = _read(path) if path.exists() else {
            "kind": "verified_action_exemplar_memory",
            "schema_version": 1,
            "selection_policy": "audit-test native action store",
            "exemplars": [],
        }

    @property
    def exemplars(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self.data["exemplars"])

    def retrieve(self, **kwargs: Any) -> list[dict[str, Any]]:
        limit = int(kwargs["limit"])
        return [dict(item) for item in self.data["exemplars"][:limit]]

    def record_accepted(self, **kwargs: Any) -> bool:
        self.data["exemplars"].append(dict(kwargs))
        _write(self.path, self.data)
        return True


def _bridge(
    run_dir: Path,
    *,
    mode: str,
    gates: V83MemoryOperationGates,
    transfer_seed: Path | None = None,
) -> V83NativeMemoryBridge:
    memory = _native_root(run_dir)
    return V83NativeMemoryBridge(
        strategy_path=(memory / "strategy_specs.json").resolve(),
        action_exemplars_path=(memory / "action_exemplars.json").resolve(),
        audit_path=(memory / V83_NATIVE_AUDIT_FILENAME).resolve(),
        mode=mode,
        gates=gates,
        strategy_store_factory=_StrategyStore,
        action_store_factory=_ActionStore,
        run_binding=_RUN_BINDING,
        transfer_seed_path=transfer_seed,
    )


def _validate(
    run_dir: Path,
    *,
    mode: str = "from_empty",
    gates: V83MemoryOperationGates | None = None,
    terminal: str = _VALID_TERMINAL,
    transfer_seed: Path | None = None,
    strategy_bank_sha256: str = "",
):
    selected_gates = gates or V83MemoryOperationGates.enabled()
    return validate_v83_native_memory_audit(
        run_dir,
        expected_run_binding=_RUN_BINDING,
        expected_mode=mode,
        expected_operation_gates=asdict(selected_gates),
        expected_terminal_reason=terminal,
        expected_transfer_bank=transfer_seed,
        expected_strategy_bank_sha256=strategy_bank_sha256,
    )


def _seed(path: Path) -> Path:
    _write(
        path,
        {
            "schema_version": 1,
            "records": [
                {
                    "phase_trigger": "corridor transport",
                    "failure_condition": "payload support becomes asymmetric",
                    "coordination_pattern": "restore bilateral support",
                    "verified_outcome": "accepted",
                    "avoid_rule": "avoid unilateral transport",
                    "repair_direction": "recenter the cooperative support frame",
                }
            ],
        },
    )
    return path.resolve()


def test_from_empty_finalized_audit_matches_fixed_native_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    native = _native_root(run_dir)
    _write(
        native / "strategy_specs.json",
        {
            "kind": "deployment_decision_memory",
            "schema_version": 2,
            "description": "empty",
            "specs": [],
        },
    )
    _write(
        native / "action_exemplars.json",
        {
            "kind": "verified_action_exemplar_memory",
            "schema_version": 1,
            "selection_policy": "empty",
            "exemplars": [],
        },
    )
    bridge = _bridge(
        run_dir,
        mode="from_empty",
        gates=V83MemoryOperationGates.enabled(),
    )
    bridge.finalize(terminal_reason=_VALID_TERMINAL)

    result = _validate(run_dir)
    assert result.valid, result.invalid_reasons
    assert result.loaded_seed_sha256 == ""
    assert result.artifact_hashes[
        "memory/v83_native/v83_native_memory_runtime_audit.json"
    ]
    assert result.artifact_hashes["memory/v83_native/strategy_specs.json"]


def test_from_empty_accepts_audited_native_promotions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bridge = _bridge(
        run_dir,
        mode="from_empty",
        gates=V83MemoryOperationGates.enabled(),
    )
    assert bridge.promote_strategy(
        {"id": "verified-strategy"},
        default_phase="corridor_transport",
        raw_evidence_persisted=True,
        model_proposal={"promote": True},
    )
    assert bridge.promote_action_exemplar(
        raw_evidence_persisted=True,
        record_kwargs={"id": "verified-action", "accepted": True},
    )
    bridge.finalize(terminal_reason=_VALID_TERMINAL)

    result = _validate(run_dir)
    assert result.valid, result.invalid_reasons


def test_disabled_allows_absent_empty_files_but_rejects_activity(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    gates = V83MemoryOperationGates.disabled()
    bridge = _bridge(run_dir, mode="disabled", gates=gates)
    bridge.retrieve_strategies({"phase": "carry"})
    bridge.retrieve_action_exemplars(method="carry_forward")
    bridge.finalize(terminal_reason="SYSTEM_STALL")

    valid = _validate(
        run_dir,
        mode="disabled",
        gates=gates,
        terminal="SYSTEM_STALL",
    )
    assert valid.valid, valid.invalid_reasons

    audit = _read(_audit_path(run_dir))
    audit["operations"]["memory_writer"]["calls"] = 1
    audit["operations"]["strategy_promotion"]["attempts"] = 1
    audit["operations"]["strategy_promotion"]["noops"] = 1
    _write(_audit_path(run_dir), audit)
    invalid = _validate(
        run_dir,
        mode="disabled",
        gates=gates,
        terminal="SYSTEM_STALL",
    )
    assert "V83_NATIVE_DISABLED_OPERATION_DETECTED" in invalid.invalid_reasons
    assert "V83_NATIVE_CLOSED_GATE_WRITER_CALL_DETECTED" in invalid.invalid_reasons


def test_preloaded_seed_matches_external_bank_initial_and_final(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    seed = _seed(tmp_path / "frozen-bank.json")
    bridge = _bridge(
        run_dir,
        mode="preloaded_seed",
        gates=V83MemoryOperationGates.enabled(),
        transfer_seed=seed,
    )
    bridge.finalize(terminal_reason="PHYSICAL_BUDGET_EXHAUSTED")

    valid = _validate(
        run_dir,
        mode="preloaded_seed",
        terminal="PHYSICAL_BUDGET_EXHAUSTED",
        transfer_seed=seed,
        strategy_bank_sha256=validate_strategy_bank(seed).bank_sha256,
    )
    assert valid.valid, valid.invalid_reasons
    assert valid.loaded_seed_sha256 == validate_strategy_bank(seed).bank_sha256

    wrong_identity = _validate(
        run_dir,
        mode="preloaded_seed",
        terminal="PHYSICAL_BUDGET_EXHAUSTED",
        transfer_seed=seed,
        strategy_bank_sha256="f" * 64,
    )
    assert "V83_NATIVE_TRANSFER_SEED_SEMANTIC_HASH_MISMATCH" in (
        wrong_identity.invalid_reasons
    )

    seed.write_text(seed.read_text(encoding="utf-8") + " ", encoding="utf-8")
    invalid = _validate(
        run_dir,
        mode="preloaded_seed",
        terminal="PHYSICAL_BUDGET_EXHAUSTED",
        transfer_seed=seed,
        strategy_bank_sha256=validate_strategy_bank(seed).bank_sha256,
    )
    assert {
        "V83_NATIVE_TRANSFER_SEED_INITIAL_STATE_MISMATCH",
        "V83_NATIVE_TRANSFER_SEED_FINAL_STATE_MISMATCH",
    }.issubset(invalid.invalid_reasons)


def test_native_store_byte_tamper_and_path_substitution_fail_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bridge = _bridge(
        run_dir,
        mode="from_empty",
        gates=V83MemoryOperationGates.enabled(),
    )
    bridge.finalize(terminal_reason=_VALID_TERMINAL)
    audit_path = _audit_path(run_dir)
    audit = _read(audit_path)
    audit["final_state"]["strategy_store"]["path"] = str(
        (tmp_path / "outside.json").resolve()
    )
    _write(audit_path, audit)
    substituted = _validate(run_dir)
    assert "V83_NATIVE_STRATEGY_STORE_FINAL_PATH_MISMATCH" in (
        substituted.invalid_reasons
    )

    audit["final_state"]["strategy_store"]["path"] = str(
        (_native_root(run_dir) / "strategy_specs.json").resolve()
    )
    _write(audit_path, audit)
    strategy = _native_root(run_dir) / "strategy_specs.json"
    _write(
        strategy,
        {
            "kind": "deployment_decision_memory",
            "schema_version": 2,
            "description": "injected after finalization",
            "specs": [],
        },
    )
    tampered = _validate(run_dir)
    assert "V83_NATIVE_STRATEGY_STORE_FINAL_CURRENT_STATE_MISMATCH" in (
        tampered.invalid_reasons
    )


def test_native_store_and_transfer_bank_use_strict_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    native = _native_root(run_dir)
    _write(
        native / "strategy_specs.json",
        {
            "kind": "deployment_decision_memory",
            "schema_version": 2,
            "description": "empty",
            "specs": [],
        },
    )
    bridge = _bridge(
        run_dir,
        mode="from_empty",
        gates=V83MemoryOperationGates.enabled(),
    )
    bridge.finalize(terminal_reason=_VALID_TERMINAL)
    strategy = native / "strategy_specs.json"
    strategy.write_text(
        '{"kind":"deployment_decision_memory","schema_version":2,'
        '"specs":[],"specs":[]}\n',
        encoding="utf-8",
    )
    invalid_store = _validate(run_dir)
    assert "V83_NATIVE_STRATEGY_STORE_INVALID" in invalid_store.invalid_reasons

    transfer_run = tmp_path / "transfer-run"
    seed = _seed(tmp_path / "strict-bank.json")
    expected_sha = validate_strategy_bank(seed).bank_sha256
    transfer = _bridge(
        transfer_run,
        mode="preloaded_seed",
        gates=V83MemoryOperationGates.enabled(),
        transfer_seed=seed,
    )
    transfer.finalize(terminal_reason="PHYSICAL_BUDGET_EXHAUSTED")
    seed.write_text('{"schema_version":1,"records":[],"poison":NaN}\n')
    invalid_seed = _validate(
        transfer_run,
        mode="preloaded_seed",
        terminal="PHYSICAL_BUDGET_EXHAUSTED",
        transfer_seed=seed,
        strategy_bank_sha256=expected_sha,
    )
    assert "V83_NATIVE_EXPECTED_TRANSFER_BANK_INVALID" in (
        invalid_seed.invalid_reasons
    )


@pytest.mark.parametrize("poison", ["duplicate", "nan"])
def test_audit_reader_rejects_duplicate_keys_and_nan(
    tmp_path: Path,
    poison: str,
) -> None:
    run_dir = tmp_path / "run"
    bridge = _bridge(
        run_dir,
        mode="from_empty",
        gates=V83MemoryOperationGates.enabled(),
    )
    bridge.finalize(terminal_reason=_VALID_TERMINAL)
    path = _audit_path(run_dir)
    text = path.read_text(encoding="utf-8")
    if poison == "duplicate":
        text = text.replace("{\n", '{\n  "kind": "duplicate",\n', 1)
    else:
        text = text.replace('"schema_version": 1', '"schema_version": NaN', 1)
    path.write_text(text, encoding="utf-8")

    result = _validate(run_dir)
    assert result.invalid_reasons == ("V83_NATIVE_AUDIT_MISSING_OR_INVALID",)


def test_incomplete_infrastructure_attempt_is_not_formal_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bridge = _bridge(
        run_dir,
        mode="from_empty",
        gates=V83MemoryOperationGates.enabled(),
    )
    bridge.mark_incomplete(reason="BACKEND_FATAL")

    result = _validate(run_dir)
    assert "V83_NATIVE_AUDIT_NOT_FINALIZED" in result.invalid_reasons
    assert "V83_NATIVE_AUDIT_TERMINAL_NOT_TASK_TERMINAL" in result.invalid_reasons


def test_run_binding_and_transfer_seed_null_contract_are_external(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bridge = _bridge(
        run_dir,
        mode="from_empty",
        gates=V83MemoryOperationGates.enabled(),
    )
    bridge.finalize(terminal_reason=_VALID_TERMINAL)
    audit = _read(_audit_path(run_dir))
    audit["run_binding"]["run_id"] = "relabelled"
    audit["initial_state"]["transfer_seed"] = {"self_described": True}
    _write(_audit_path(run_dir), audit)

    result = _validate(run_dir)
    assert "V83_NATIVE_RUN_BINDING_MISMATCH" in result.invalid_reasons
    assert "V83_NATIVE_UNEXPECTED_TRANSFER_SEED_STATE" in result.invalid_reasons
