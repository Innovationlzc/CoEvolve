from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from coevolve_experiments.memory.v83_native import (
    V83MemoryOperationGates,
    V83NativeMemoryBridge,
    V83NativeMemoryError,
    V83_NATIVE_ACTION_LIMIT,
    V83_NATIVE_AUDIT_KIND,
    V83_NATIVE_STRATEGY_LIMIT,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_spec(
    identifier: str,
    *,
    kind: str = "verified_motion_success",
    phase: str = "corridor_transport",
    confidence: float = 0.8,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "phase": phase,
        "kind": kind,
        "failure_level": "none" if "success" in kind else "coordination",
        "observed_coordination_mode": "sync",
        "observed_coordination_primitive": "coordinated_linear_transport",
        "condition": "The verifier observed a comparable supported carry phase.",
        "diagnosis": "The selected coordination pattern was physically grounded.",
        "decision": "Reuse only after checking the current physical geometry.",
        "forbidden": ["Do not replay this record outside comparable support."],
        "repair": "Select a verifier-compatible coordination repair.",
        "failure_types": (
            ["payload_slip"] if kind == "verified_failure_counterexample" else []
        ),
        "confidence": confidence,
        "evidence": {
            "source": "unit_test",
            "accepted_count": 1 if "success" in kind else 0,
            "rejected_count": 1 if kind == "verified_failure_counterexample" else 0,
            "failure_counts": (
                {"payload_slip": 1}
                if kind == "verified_failure_counterexample"
                else {}
            ),
            "rollout_refs": [],
            "rollout_keys": [identifier],
        },
    }


class _FakeStrategyStore:
    calls: list[tuple[str, Any]] = []

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = (
            _read(path)
            if path.exists()
            else {
                "kind": "deployment_decision_memory",
                "schema_version": 2,
                "description": "test native store",
                "specs": [],
            }
        )

    @property
    def specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self.data["specs"])

    def retrieve(self, context: object, *, limit: int) -> list[str]:
        self.calls.append(("retrieve", limit))
        del context
        return [f"native:{item['id']}" for item in self.data["specs"][:limit]]

    def update_from_rollout(
        self,
        record: dict[str, Any],
        *,
        default_phase: str,
        model_proposal: dict[str, Any] | None,
    ) -> bool:
        self.calls.append(("update", model_proposal))
        if model_proposal is not None and model_proposal.get("promote") is False:
            return False
        if record.get("promote") is False:
            return False
        specs = record.get("specs")
        if not isinstance(specs, list):
            specs = [
                _native_spec(
                    str(record.get("id", "online-native")),
                    kind=str(record.get("kind", "verified_motion_success")),
                    phase=default_phase,
                    confidence=float(record.get("confidence", 0.9)),
                )
            ]
        self.data["specs"].extend(specs)
        return True

    def save(self) -> None:
        self.calls.append(("save", len(self.data["specs"])))
        _write(self.path, self.data)


class _FakeActionStore:
    calls: list[tuple[str, Any]] = []

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = (
            _read(path)
            if path.exists()
            else {
                "kind": "verified_action_exemplar_memory",
                "schema_version": 1,
                "selection_policy": "test native store",
                "exemplars": [],
            }
        )

    @property
    def exemplars(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self.data["exemplars"])

    def retrieve(
        self,
        *,
        method: str,
        semantic_objective: dict[str, object] | None,
        coordination_mode: str,
        phase: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(("retrieve", limit))
        del semantic_objective, phase
        selected = [
            item
            for item in self.data["exemplars"]
            if item["method"] == method
            and (not coordination_mode or item["coordination_mode"] == coordination_mode)
        ]
        return [dict(item) for item in selected[:limit]]

    def record_accepted(self, **kwargs: Any) -> bool:
        self.calls.append(("record", kwargs.get("id")))
        if kwargs.get("accepted", True) is not True:
            return False
        item = {
            "id": str(kwargs["id"]),
            "method": str(kwargs.get("method", "coordinated_linear_transport")),
            "coordination_mode": str(kwargs.get("coordination_mode", "sync")),
        }
        if any(existing["id"] == item["id"] for existing in self.data["exemplars"]):
            return False
        self.data["exemplars"].append(item)
        _write(self.path, self.data)
        return True


@pytest.fixture(autouse=True)
def _reset_fake_calls() -> None:
    _FakeStrategyStore.calls = []
    _FakeActionStore.calls = []


def _bridge(
    tmp_path: Path,
    *,
    mode: str = "from_empty",
    gates: V83MemoryOperationGates | None = None,
    transfer_seed_path: Path | None = None,
    strategy_factory: Any = _FakeStrategyStore,
    action_factory: Any = _FakeActionStore,
) -> V83NativeMemoryBridge:
    return V83NativeMemoryBridge(
        strategy_path=tmp_path / "memory/strategy_specs.json",
        action_exemplars_path=tmp_path / "memory/action_exemplars.json",
        audit_path=tmp_path / "memory/v83_native_memory_runtime_audit.json",
        mode=mode,
        gates=gates or V83MemoryOperationGates.enabled(),
        strategy_store_factory=strategy_factory,
        action_store_factory=action_factory,
        run_binding={"run_id": "table-test-seed-200", "plan_sha256": "a" * 64},
        transfer_seed_path=transfer_seed_path,
    )


def test_from_empty_delegates_native_top3_top2_and_finalizes(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    writer_calls: list[Mapping[str, Any]] = []

    def writer(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        writer_calls.append(payload)
        return {"promote": True, "strategy_name": "safe carry"}

    assert bridge.promote_strategy(
        {
            "specs": [
                _native_spec(f"s{index}", confidence=0.9 - index / 100)
                for index in range(5)
            ]
        },
        default_phase="corridor_transport",
        raw_evidence_persisted=True,
        memory_writer=writer,
        memory_writer_payload={"rollout_id": "r1"},
    )
    strategies = bridge.retrieve_strategies({"phase": "corridor_transport"})
    assert len(strategies) == V83_NATIVE_STRATEGY_LIMIT
    assert ("retrieve", V83_NATIVE_STRATEGY_LIMIT) in _FakeStrategyStore.calls
    assert writer_calls == [{"rollout_id": "r1"}]

    for index in range(4):
        assert bridge.promote_action_exemplar(
            raw_evidence_persisted=True,
            record_kwargs={
                "id": f"a{index}",
                "method": "coordinated_linear_transport",
                "coordination_mode": "sync",
            },
        )
    exemplars = bridge.retrieve_action_exemplars(
        method="coordinated_linear_transport",
        semantic_objective={"direction": "forward"},
        coordination_mode="sync",
        phase="corridor_transport",
    )
    assert len(exemplars) == V83_NATIVE_ACTION_LIMIT
    assert ("retrieve", V83_NATIVE_ACTION_LIMIT) in _FakeActionStore.calls

    receipt = bridge.finalize(terminal_reason="SUCCESS")
    assert receipt["kind"] == V83_NATIVE_AUDIT_KIND
    assert receipt["lifecycle_state"] == "finalized"
    assert receipt["operations"]["strategy_retrieval"] == {
        "invocations": 1,
        "returned_records": 3,
        "native_records_returned": 3,
        "transfer_seed_records_returned": 0,
    }
    assert receipt["operations"]["memory_writer"] == {"calls": 1, "failures": 0}
    assert receipt["final_state"]["strategy_store"]["record_count"] == 5
    assert receipt["final_state"]["action_exemplar_store"]["record_count"] == 4


def test_disabled_boundary_never_constructs_stores_or_calls_writer(tmp_path: Path) -> None:
    constructed: list[str] = []
    writer_calls: list[Mapping[str, Any]] = []

    def forbidden_strategy_factory(_path: Path) -> Any:
        constructed.append("strategy")
        raise AssertionError("disabled boundary constructed strategy store")

    def forbidden_action_factory(_path: Path) -> Any:
        constructed.append("action")
        raise AssertionError("disabled boundary constructed action store")

    def forbidden_writer(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        writer_calls.append(payload)
        raise AssertionError("disabled boundary called Memory Writer")

    bridge = _bridge(
        tmp_path,
        mode="disabled",
        gates=V83MemoryOperationGates.disabled(),
        strategy_factory=forbidden_strategy_factory,
        action_factory=forbidden_action_factory,
    )
    assert bridge.retrieve_strategies({"phase": "corridor_transport"}) == []
    assert bridge.retrieve_action_exemplars(method="carry_forward") == []
    assert not bridge.promote_strategy(
        {"id": "never"},
        default_phase="corridor_transport",
        raw_evidence_persisted=True,
        memory_writer=forbidden_writer,
    )
    assert not bridge.promote_action_exemplar(
        raw_evidence_persisted=True,
        record_kwargs={"id": "never"},
    )
    receipt = bridge.finalize(terminal_reason="SYSTEM_STALL")

    assert constructed == []
    assert writer_calls == []
    operations = receipt["operations"]
    assert operations["strategy_retrieval"]["invocations"] == 0
    assert operations["strategy_promotion"]["attempts"] == 0
    assert operations["action_exemplar_retrieval"]["invocations"] == 0
    assert operations["action_exemplar_promotion"]["attempts"] == 0
    assert operations["memory_writer"]["calls"] == 0
    assert set(receipt["suppressed_boundary_calls"].values()) == {1}
    assert receipt["final_state"]["strategy_store"]["record_count"] == 0
    assert receipt["final_state"]["action_exemplar_store"]["record_count"] == 0
    assert not (tmp_path / "memory/strategy_specs.json").exists()
    assert not (tmp_path / "memory/action_exemplars.json").exists()


def test_transfer_seed_is_jointly_ranked_separately_rendered_and_immutable(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "gpt_transfer_bank.json"
    _write(
        seed,
        {
            "schema_version": 1,
            "records": [
                {
                    "phase_trigger": "corridor transport",
                    "failure_condition": "payload slip under uneven support",
                    "coordination_pattern": "close support before advancing",
                    "verified_outcome": "rejected",
                    "avoid_rule": "avoid widening the support formation",
                    "repair_direction": "restore symmetric payload support",
                },
                {
                    "phase_trigger": "corridor transport",
                    "failure_condition": "payload drift after a heading change",
                    "coordination_pattern": "recenter support before translation",
                    "verified_outcome": "rejected",
                    "avoid_rule": "avoid translating with divergent headings",
                    "repair_direction": "realign the cooperative support frame",
                },
            ],
        },
    )
    original_seed_sha = _sha(seed)
    bridge = _bridge(
        tmp_path,
        mode="preloaded_seed",
        transfer_seed_path=seed,
    )
    assert bridge.promote_strategy(
        {"id": "online-success", "kind": "verified_motion_success"},
        default_phase="corridor_transport",
        raw_evidence_persisted=True,
        model_proposal={"promote": True},
    )

    rendered = bridge.retrieve_strategies(
        {
            "phase": "corridor_transport",
            "last_failure_type": "payload_slip",
            "last_failure_level": "coordination",
        }
    )
    assert len(rendered) == 3
    transfer = [item for item in rendered if item.startswith("[transfer_strategy")]
    native = [item for item in rendered if item.startswith("[deployment_spec")]
    assert len(transfer) == 2
    assert len(native) == 1
    assert all("observed_mode=" not in item for item in transfer)
    assert all("mode=sync" not in item and "mode=async" not in item for item in transfer)
    assert "observed_mode=sync" in native[0]
    assert _sha(seed) == original_seed_sha

    receipt = bridge.finalize(terminal_reason="PHYSICAL_BUDGET_EXHAUSTED")
    assert receipt["operations"]["strategy_retrieval"][
        "transfer_seed_records_returned"
    ] == 2
    assert receipt["operations"]["strategy_retrieval"][
        "native_records_returned"
    ] == 1
    final_seed = receipt["final_state"]["transfer_seed"]
    assert final_seed["file_sha256"] == original_seed_sha
    assert final_seed["native_mode_invented"] is False
    assert final_seed["immutable_verified"] is True


def test_writer_failure_is_nonfatal_and_does_not_promote(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)

    def unavailable(_payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise TimeoutError("provider timeout")

    assert not bridge.promote_strategy(
        {"id": "not-promoted"},
        default_phase="corridor_transport",
        raw_evidence_persisted=True,
        memory_writer=unavailable,
    )
    audit = bridge.audit
    assert audit["operations"]["memory_writer"] == {"calls": 1, "failures": 1}
    assert audit["operations"]["strategy_promotion"]["noops"] == 1
    assert not (tmp_path / "memory/strategy_specs.json").exists()
    assert audit["lifecycle_state"] == "active"


def test_external_native_store_tamper_is_fail_closed_and_audited(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    strategy = tmp_path / "memory/strategy_specs.json"
    _write(
        strategy,
        {
            "kind": "deployment_decision_memory",
            "schema_version": 2,
            "description": "externally injected",
            "specs": [],
        },
    )

    with pytest.raises(V83NativeMemoryError, match="external strategy-store"):
        bridge.retrieve_strategies({"phase": "corridor_transport"})
    audit = _read(tmp_path / "memory/v83_native_memory_runtime_audit.json")
    assert audit["lifecycle_state"] == "violated"
    assert audit["violations"][0]["operation"] == "strategy retrieval"


def test_external_transfer_seed_tamper_is_fail_closed(tmp_path: Path) -> None:
    seed = tmp_path / "seed.json"
    _write(
        seed,
        {
            "schema_version": 1,
            "records": [
                {
                    "phase_trigger": "target alignment",
                    "failure_condition": "support becomes asymmetric",
                    "coordination_pattern": "restore cooperative support",
                    "verified_outcome": "accepted",
                    "avoid_rule": "avoid unilateral release",
                    "repair_direction": "reestablish shared support",
                }
            ],
        },
    )
    bridge = _bridge(
        tmp_path,
        mode="preloaded_seed",
        transfer_seed_path=seed,
    )
    seed.write_text(seed.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(V83NativeMemoryError, match="immutable transfer-seed"):
        bridge.retrieve_strategies({"phase": "target_alignment"})


def test_raw_evidence_order_and_terminal_receipt_are_enforced(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    with pytest.raises(V83NativeMemoryError, match="raw physical evidence"):
        bridge.promote_strategy(
            {"id": "too-early"},
            default_phase="corridor_transport",
            raw_evidence_persisted=False,
        )
    with pytest.raises(V83NativeMemoryError, match="cannot receive a finalized"):
        bridge.finalize(terminal_reason="BACKEND_FATAL")

    incomplete = bridge.mark_incomplete(reason="BACKEND_FATAL")
    assert incomplete["lifecycle_state"] == "incomplete"
    assert incomplete["finalized"] is False
    with pytest.raises(V83NativeMemoryError, match="lifecycle is incomplete"):
        bridge.retrieve_strategies({"phase": "corridor_transport"})


def test_external_audit_tamper_is_never_overwritten(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    audit_path = tmp_path / "memory/v83_native_memory_runtime_audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(V83NativeMemoryError, match="audit was externally modified"):
        bridge.retrieve_strategies({"phase": "corridor_transport"})
    assert audit_path.read_text(encoding="utf-8") == "{}\n"

