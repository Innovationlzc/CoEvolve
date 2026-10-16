"""Host-owned audit trail and checkpoint-local safety state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .contracts import AuditEvent


class AuditSinkError(RuntimeError):
    """Durable audit write failed after the event was retained in memory."""


@dataclass(slots=True)
class ExactASTBlocker:
    """Rejected ASTs are blocked only at the checkpoint that disproved them."""

    _blocked: dict[str, dict[str, Mapping[str, Any]]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def add(
        self,
        checkpoint_id: str,
        ast_fingerprint: str,
        evidence: Mapping[str, Any],
    ) -> None:
        self._blocked[checkpoint_id][ast_fingerprint] = dict(evidence)

    def contains(self, checkpoint_id: str, ast_fingerprint: str) -> bool:
        return ast_fingerprint in self._blocked.get(checkpoint_id, {})

    def fingerprints(self, checkpoint_id: str) -> frozenset[str]:
        return frozenset(self._blocked.get(checkpoint_id, {}))

    def evidence(
        self, checkpoint_id: str, ast_fingerprint: str
    ) -> Mapping[str, Any] | None:
        return self._blocked.get(checkpoint_id, {}).get(ast_fingerprint)

    def clear_checkpoint(self, checkpoint_id: str) -> None:
        self._blocked.pop(checkpoint_id, None)

    def snapshot(self) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
        return {
            checkpoint: dict(entries)
            for checkpoint, entries in self._blocked.items()
        }


class AuditTrail:
    """Append-only in-memory audit with an optional durable sink callback."""

    def __init__(self, sink: Callable[[AuditEvent], None] | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._sink = sink

    def append(
        self,
        kind: str,
        *,
        checkpoint_id: str | None,
        transaction_index: int | None,
        payload: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            sequence=len(self._events),
            kind=kind,
            checkpoint_id=checkpoint_id,
            transaction_index=transaction_index,
            payload={} if payload is None else dict(payload),
        )
        self._events.append(event)
        if self._sink is not None:
            try:
                self._sink(event)
            except Exception as exc:
                # Disable the failed sink so recovery/audit bookkeeping cannot
                # be interrupted a second time.  The failure itself remains in
                # the in-memory raw audit returned to the caller.
                self._sink = None
                self._events.append(
                    AuditEvent(
                        sequence=len(self._events),
                        kind="audit_sink_failed",
                        checkpoint_id=checkpoint_id,
                        transaction_index=transaction_index,
                        payload={"event_kind": kind, "error": str(exc)},
                    )
                )
                raise AuditSinkError(
                    f"audit sink failed while writing {kind}: {exc}"
                ) from exc
        return event

    def snapshot(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)
