"""Persistent lifecycle audit for one run-local adapter memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .json_io import atomic_write_json


RUNTIME_AUDIT_FILENAME = "memory_runtime_audit.json"
RUNTIME_AUDIT_KIND = "table4_memory_runtime_audit"
RUNTIME_AUDIT_SCHEMA_VERSION = 1


class MemoryLifecycleError(ValueError):
    """An operation was attempted after finalization or a recorded violation."""


class MemoryLifecycleAudit:
    """Own the active/finalized/violated state machine and durable counters."""

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        run_id: str,
        protocol_id: str,
        condition: str,
        condition_fingerprint: str,
        target_model: str,
        bootstrap_audit_sha256: str,
        plan_path: str,
        plan_sha256: str,
        seed_sha256: str,
        overlay_sha256: str,
        action_sha256: str,
        fairness_fingerprints: Mapping[str, str],
    ) -> "MemoryLifecycleAudit":
        path = root / RUNTIME_AUDIT_FILENAME
        if path.exists():
            raise MemoryLifecycleError(
                "memory runtime audit already exists; refusing to resume or overwrite "
                f"{path}"
            )
        payload: dict[str, Any] = {
            "kind": RUNTIME_AUDIT_KIND,
            "schema_version": RUNTIME_AUDIT_SCHEMA_VERSION,
            "lifecycle_state": "active",
            "condition": condition,
            "target_model": target_model,
            "run_id": run_id,
            "protocol_id": protocol_id,
            "condition_fingerprint": condition_fingerprint,
            "fairness_fingerprints": dict(fairness_fingerprints),
            "bootstrap_audit_path": "memory_bootstrap_audit.json",
            # This is the canonical/semantic JSON hash, deliberately
            # independent of indentation or key presentation.
            "bootstrap_audit_sha256": bootstrap_audit_sha256,
            "plan_path": plan_path,
            "plan_sha256": plan_sha256,
            "seed": {
                "path": "seed_strategy_bank.json",
                "expected_sha256": seed_sha256,
                "preflight_sha256": seed_sha256,
                "current_sha256": seed_sha256,
                "final_sha256": "",
                "immutable_preflight": True,
                "immutable_final": False,
            },
            "online_overlay": {
                "path": "online_strategy_overlay.json",
                "initial_sha256": overlay_sha256,
                "current_sha256": overlay_sha256,
                "final_sha256": "",
                "initial_record_count": 0,
                "current_record_count": 0,
                "final_record_count": None,
            },
            "action_exemplars": {
                "path": "action_exemplars.json",
                "initial_sha256": action_sha256,
                "current_sha256": action_sha256,
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
        audit = cls(path, payload)
        audit.persist()
        return audit

    @property
    def state(self) -> str:
        return str(self.payload.get("lifecycle_state", "not_initialized"))

    @property
    def finalized(self) -> bool:
        return self.payload.get("finalized") is True

    def ensure_active(self, operation: str) -> None:
        if self.state != "active":
            raise MemoryLifecycleError(
                f"cannot {operation}: memory lifecycle is {self.state}"
            )

    def current_hash(self, section: str) -> str:
        value = self.payload[section]["current_sha256"]
        return str(value)

    def set_seed_current(self, digest: str) -> None:
        self.payload["seed"]["current_sha256"] = digest

    def record_operation(self, name: str) -> None:
        self.ensure_active(name)
        self.payload["operations"][name] += 1

    def record_noop(self) -> None:
        self.ensure_active("record promotion no-op")
        self.payload["operations"]["promotion_noops"] += 1

    def update_overlay(self, digest: str, count: int) -> None:
        self.payload["online_overlay"].update(
            current_sha256=digest,
            current_record_count=count,
        )

    def update_actions(self, digest: str, count: int) -> None:
        self.payload["action_exemplars"].update(
            current_sha256=digest,
            current_count=count,
        )

    def finalize(
        self,
        *,
        seed_sha256: str,
        overlay_sha256: str,
        overlay_count: int,
        action_sha256: str,
        action_count: int,
    ) -> Mapping[str, Any]:
        if self.finalized:
            return self.snapshot()
        self.ensure_active("finalize")
        self.payload["seed"].update(
            current_sha256=seed_sha256,
            final_sha256=seed_sha256,
            immutable_final=True,
        )
        self.payload["online_overlay"].update(
            current_sha256=overlay_sha256,
            final_sha256=overlay_sha256,
            current_record_count=overlay_count,
            final_record_count=overlay_count,
        )
        self.payload["action_exemplars"].update(
            current_sha256=action_sha256,
            final_sha256=action_sha256,
            current_count=action_count,
            final_count=action_count,
        )
        self.payload["lifecycle_state"] = "finalized"
        self.payload["finalized"] = True
        self.persist()
        return self.snapshot()

    def violate(self, operation: str, detail: str) -> None:
        entry = {
            "sequence": len(self.payload["violations"]),
            "operation": operation,
            "detail": str(detail),
        }
        self.payload["violations"].append(entry)
        self.payload["lifecycle_state"] = "violated"
        self.payload["finalized"] = False
        try:
            self.persist()
        except Exception as exc:
            raise MemoryLifecycleError(
                f"{detail}; additionally failed to persist violation audit: {exc}"
            ) from exc
        raise MemoryLifecycleError(str(detail))

    def persist(self) -> None:
        atomic_write_json(self.path, self.payload)

    def snapshot(self) -> Mapping[str, Any]:
        return json.loads(json.dumps(self.payload))


__all__ = [
    "MemoryLifecycleAudit",
    "MemoryLifecycleError",
    "RUNTIME_AUDIT_FILENAME",
    "RUNTIME_AUDIT_KIND",
    "RUNTIME_AUDIT_SCHEMA_VERSION",
]
