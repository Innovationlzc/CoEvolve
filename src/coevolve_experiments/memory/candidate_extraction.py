"""Convert bound source strategy specs into leak-free transfer candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .leak_check import sanitize_transfer_text, scan_memory_leaks
from .schema import (
    SourceRunValidationError,
    TransferStrategyRecord,
    semantic_sha256,
)
from .source_evidence import BoundSourceEvidence


@dataclass(frozen=True)
class StrategyCandidate:
    """One route-abstracted record plus source/ranking audit metadata."""

    record: TransferStrategyRecord
    source_index: int
    source_record_index: int
    source_record_id: str
    evidence_weight: int
    sanitization_notes: tuple[str, ...]

    @property
    def digest(self) -> str:
        return semantic_sha256(self.record)


def extract_strategy_specs(value: object) -> list[dict[str, Any]]:
    """Load the supported source-memory envelope into independent mappings."""

    raw = (
        value.get("specs", value.get("records"))
        if isinstance(value, Mapping)
        else value
    )
    if not isinstance(raw, list):
        raise SourceRunValidationError(
            "strategy_specs must be a list or an object containing specs"
        )
    if not all(isinstance(item, Mapping) for item in raw):
        raise SourceRunValidationError("every source strategy spec must be an object")
    return [dict(item) for item in raw]


def candidate_from_spec(
    spec: Mapping[str, Any],
    *,
    source_index: int,
    source_record_index: int,
    bound_evidence: Mapping[str, BoundSourceEvidence],
) -> tuple[StrategyCandidate | None, str]:
    """Build one candidate, returning a stable disposition reason on drop."""

    outcome, evidence_weight, error = _outcome_and_evidence(spec, bound_evidence)
    if outcome is None:
        return None, error

    notes: list[str] = []
    phase_trigger = _abstract_phase_trigger(
        spec.get("phase", spec.get("phase_trigger"))
    )

    if outcome == "rejected":
        failure_source: object = spec.get("diagnosis") or spec.get(
            "failure_condition"
        )
        if spec.get("failure_types"):
            failure_source = [spec.get("failure_types"), failure_source]
        failure_condition, note = _safe_text(
            failure_source,
            "the verifier reports loss of cooperative stability",
            "failure_condition",
        )
    else:
        failure_condition = (
            "no verifier-reported failure under the applicable cooperative state"
        )
        note = None
    if note:
        notes.append(note)

    mode = str(
        spec.get("observed_coordination_mode", spec.get("coordination_mode", ""))
    ).strip().lower()
    mode_text = (
        "synchronous"
        if mode == "sync"
        else "asynchronous"
        if mode == "async"
        else "cooperative"
    )
    primitive = str(
        spec.get(
            "observed_coordination_primitive",
            spec.get("coordination_primitive", ""),
        )
    ).strip()
    primitive_text = _PRIMITIVE_ABSTRACTIONS.get(
        primitive,
        sanitize_transfer_text(primitive),
    )
    genome = spec.get("strategy_genome")
    genome = genome if isinstance(genome, Mapping) else {}
    child = genome.get("child_strategy") or spec.get("coordination_pattern")
    coordination_pattern, note = _safe_text(
        [mode_text, primitive_text, child],
        "preserve coordinated support and role-compatible motion",
        "coordination_pattern",
    )
    if note:
        notes.append(note)

    avoid_rule, note = _safe_text(
        spec.get("forbidden") or spec.get("avoid_rule") or genome.get("avoid"),
        "do not repeat a verifier-rejected coordination pattern without a "
        "material repair",
        "avoid_rule",
    )
    if note:
        notes.append(note)
    repair_direction, note = _safe_text(
        spec.get("repair") or spec.get("repair_direction") or spec.get("decision"),
        "change the coordination abstraction while preserving verified support",
        "repair_direction",
    )
    if note:
        notes.append(note)

    record: TransferStrategyRecord = {
        "phase_trigger": phase_trigger,
        "failure_condition": failure_condition,
        "coordination_pattern": coordination_pattern,
        "verified_outcome": outcome,  # type: ignore[typeddict-item]
        "avoid_rule": avoid_rule,
        "repair_direction": repair_direction,
    }
    leaks = scan_memory_leaks(record)
    if leaks:
        return None, "sanitized_record_failed_leak_check:" + ",".join(
            sorted({item.category for item in leaks})
        )
    return (
        StrategyCandidate(
            record=record,
            source_index=source_index,
            source_record_index=source_record_index,
            source_record_id=str(spec.get("id", "")).strip(),
            evidence_weight=evidence_weight,
            sanitization_notes=tuple(notes),
        ),
        "",
    )


def _outcome_and_evidence(
    spec: Mapping[str, Any],
    bound_evidence: Mapping[str, BoundSourceEvidence],
) -> tuple[str | None, int, str]:
    binding = bound_evidence.get(semantic_sha256(dict(spec)))
    if binding is None:
        return None, 0, "source_record_not_evidence_bound"

    explicit = str(spec.get("verified_outcome", "")).strip().lower()
    kind = str(spec.get("kind", "")).strip().lower()
    declared: str | None = None
    if explicit in {"accepted", "rejected"}:
        declared = explicit
    elif "success" in kind:
        declared = "accepted"
    elif "counterexample" in kind or "failure" in kind:
        declared = "rejected"
    if declared is not None and declared != binding.outcome:
        return None, 0, "source_spec_outcome_conflicts_with_evidence_binding"
    return binding.outcome, binding.count, ""


def _abstract_phase_trigger(value: object) -> str:
    phase = str(value or "").strip().lower().replace("-", "_")
    if any(token in phase for token in ("place", "target", "release", "align")):
        return "during terminal placement coordination"
    if any(token in phase for token in ("pivot", "turn", "corner")):
        return "during a cooperative heading transition"
    if any(
        token in phase for token in ("transport", "corridor", "passage", "straight")
    ):
        return "during supported payload transport"
    if any(token in phase for token in ("setup", "bootstrap", "lift", "support")):
        return "during payload support establishment"
    if any(token in phase for token in ("recover", "repair", "failure")):
        return "after verifier-reported instability"
    return "when the applicable cooperative phase is active"


_PRIMITIVE_ABSTRACTIONS = {
    "coordinated_linear_transport": "coordinated supported translation",
    "coordinated_arc_transport": "coordinated curved transport",
    "coordinated_pivot": "coordinated heading transition",
    "coordinated_alignment": "coordinated relative alignment",
    "hold_position": "coordinated support hold",
    "place_crouch": "coordinated supported placement",
    "recover_stand": "coordinated support recovery",
}


def _safe_text(value: object, fallback: str, label: str) -> tuple[str, str | None]:
    sanitized = sanitize_transfer_text(value)
    if not sanitized:
        return fallback, f"{label}:empty_after_sanitization"
    leaks = scan_memory_leaks(sanitized)
    if leaks:
        categories = ",".join(sorted({item.category for item in leaks}))
        return fallback, f"{label}:fallback_after_{categories}"
    return sanitized, None


__all__ = [
    "StrategyCandidate",
    "candidate_from_spec",
    "extract_strategy_specs",
]
