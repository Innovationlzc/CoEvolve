"""Deterministic V83-compatible retrieval for verified strategy records.

The frozen V83 runner requested three records from one jointly ranked pool of
exact-phase, permitted phase-alias, and phase-generic candidates. Retrieval is
balanced: it initially prioritises at most two verified counterexamples and
reserves a slot for a verified success whenever one exists. Phase relevance,
recent failure context, confidence, verifier-evidence count, and stable source
order determine ranking; later source records win the final tie as in V83.

Table-4 transfer records expose only the public six-field semantic schema.
They therefore use the neutral confidence and one-record evidence equivalents
declared in :mod:`coevolve_experiments.memory.schema`. This preserves the V83
query structure without claiming that a sanitized transfer record is
byte-identical to the richer private deployment-memory record.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .schema import FROZEN_STRATEGY_RETRIEVAL_POLICY, TRANSFER_RECORD_FIELDS


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def canonical_strategy_phase(value: object) -> str:
    """Map deployment and transfer labels onto stable retrieval phases."""

    text = " ".join(_TOKEN_RE.findall(str(value or "").lower()))
    if (
        not text
        or text in {"any", "all", "unknown", "unspecified"}
        or "applicable cooperative phase" in text
    ):
        return "any"
    if any(token in text for token in ("target", "place", "release", "alignment")):
        return "target_alignment"
    if any(token in text for token in ("pivot", "turn", "corner", "heading")):
        return "heading_transition"
    if any(token in text for token in ("recover", "repair", "failure", "instability")):
        return "recovery"
    if any(token in text for token in ("bootstrap", "initialize", "activation", "lift")):
        return "support_establishment"
    if "support" in text and not any(
        token in text for token in ("transport", "carry", "corridor", "passage")
    ):
        return "support_establishment"
    if "corridor" in text or "passage" in text:
        return "corridor_entry" if "entry" in text else "corridor_transport"
    if "straight" in text:
        return "straight_transport"
    if any(token in text for token in ("transport", "carry", "advance", "move")):
        return "corridor_transport"
    return text.replace(" ", "_")


def select_strategy_records(
    records: Sequence[Mapping[str, Any]],
    *,
    active_phase: object,
    failure_code: object = "",
    failure_level: object = "",
    failure_report: object = "",
    limit: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return the balanced V83 top-three strategy set.

    ``limit`` exists for contract tests and non-formal diagnostics. Formal
    callers omit it and consequently use the frozen value of three. Exact,
    permitted alias, and ``any`` records compete in one ranked candidate pool;
    aliases are not an empty-exact fallback.
    """

    phase = canonical_strategy_phase(active_phase)
    aliases = _phase_aliases(phase)
    context_tokens = _meaningful_tokens(f"{failure_code} {failure_report}")
    normalized_failure_code = _normalized_code(failure_code)
    normalized_failure_level = _normalized_code(failure_level)
    ranked: list[tuple[tuple[float, ...], Mapping[str, Any]]] = []
    for source_index, source_record in enumerate(records):
        record = dict(source_record)
        record_phase = _record_phase(record)
        if (
            record_phase != phase
            and record_phase not in aliases
            and record_phase != "any"
        ):
            continue
        score = (
            2.0
            if record_phase == phase
            else 1.0
            if record_phase in aliases
            else 0.0,
            1.0
            if _failure_matches(
                record,
                context_tokens=context_tokens,
                normalized_failure_code=normalized_failure_code,
            )
            else 0.0,
            1.0
            if _failure_level_matches(
                record,
                normalized_failure_level=normalized_failure_level,
            )
            else 0.0,
            _record_confidence(record),
            float(_record_evidence_total(record)),
            float(source_index),
        )
        ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)

    maximum = (
        FROZEN_STRATEGY_RETRIEVAL_POLICY.limit
        if limit is None
        else max(0, int(limit))
    )
    if maximum == 0:
        return ()

    failures = [
        record
        for _score, record in ranked
        if _record_kind(record) == "verified_failure_counterexample"
    ]
    successes = [
        record
        for _score, record in ranked
        if _record_kind(record)
        in {"verified_motion_success", "verified_support_success"}
    ]
    selected: list[Mapping[str, Any]] = []

    # This is intentionally the V83 balancing sequence. "At most two" refers
    # to the priority pass. If no positive record exists, the later fallback
    # pass may fill an otherwise empty third slot with another counterexample.
    negative_limit = maximum
    if (
        successes
        and FROZEN_STRATEGY_RETRIEVAL_POLICY.reserve_success_when_available
    ):
        negative_limit = max(0, maximum - 1)
    selected.extend(
        failures[
            : min(
                FROZEN_STRATEGY_RETRIEVAL_POLICY.prioritized_counterexample_limit,
                negative_limit,
            )
        ]
    )
    if successes and len(selected) < maximum:
        selected.append(successes[0])

    def add_matching(predicate: Any) -> None:
        for _score, record in ranked:
            if len(selected) >= maximum:
                return
            if predicate(record) and record not in selected:
                selected.append(record)

    if context_tokens and len(selected) < maximum:
        add_matching(
            lambda record: _failure_matches(
                record,
                context_tokens=context_tokens,
                normalized_failure_code=normalized_failure_code,
            )
        )
    add_matching(
        lambda record: _record_kind(record) == "verified_failure_counterexample"
    )
    add_matching(lambda record: _record_kind(record) == "verified_motion_success")
    add_matching(lambda record: _record_kind(record) == "verified_support_success")
    add_matching(lambda _record: True)
    return tuple(selected)


def _record_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    proposal = record.get("proposal")
    return proposal if isinstance(proposal, Mapping) else record


def _record_phase(record: Mapping[str, Any]) -> str:
    retrieval = record.get("retrieval")
    if isinstance(retrieval, Mapping):
        return canonical_strategy_phase(retrieval.get("phase"))
    payload = _record_payload(record)
    return canonical_strategy_phase(
        record.get(
            "phase",
            payload.get("phase_trigger", payload.get("phase", "any")),
        )
    )


def _phase_aliases(phase: str) -> frozenset[str]:
    # Match V83's deliberately narrow, asymmetric alias relation.
    if phase == "corridor_transport":
        return frozenset({"straight_transport"})
    if phase == "corridor_entry":
        return frozenset({"corridor_transport"})
    return frozenset()


def _record_kind(record: Mapping[str, Any]) -> str:
    payload = _record_payload(record)
    kind = str(record.get("kind", payload.get("kind", ""))).strip().lower()
    if kind in {
        "verified_failure_counterexample",
        "verified_motion_success",
        "verified_support_success",
    }:
        return kind
    outcome = str(payload.get("verified_outcome", "")).strip().lower()
    if outcome == "rejected":
        return "verified_failure_counterexample"
    if outcome == "accepted":
        # Six-field records do not retain the private motion/support subtype.
        return "verified_motion_success"
    return kind


def _record_confidence(record: Mapping[str, Any]) -> float:
    payload = _record_payload(record)
    for candidate in (record.get("confidence"), payload.get("confidence")):
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            continue
        value = float(candidate)
        if math.isfinite(value):
            return value
    if _is_six_field_record(payload):
        return FROZEN_STRATEGY_RETRIEVAL_POLICY.six_field_confidence_equivalent
    return 0.0


def _record_evidence_total(record: Mapping[str, Any]) -> int:
    payload = _record_payload(record)
    for candidate in (record.get("evidence"), payload.get("evidence")):
        if not isinstance(candidate, Mapping):
            continue
        accepted = _nonnegative_int(candidate.get("accepted_count"))
        rejected = _nonnegative_int(candidate.get("rejected_count"))
        return accepted + rejected
    provenance = record.get("provenance")
    if isinstance(provenance, Mapping) and str(
        provenance.get("verifier_outcome", "")
    ).lower() in {"accepted", "rejected"}:
        return FROZEN_STRATEGY_RETRIEVAL_POLICY.six_field_evidence_equivalent
    if _is_six_field_record(payload):
        return FROZEN_STRATEGY_RETRIEVAL_POLICY.six_field_evidence_equivalent
    return 0


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(float(value)):
        return 0
    return max(0, int(value))


def _is_six_field_record(record: Mapping[str, Any]) -> bool:
    return set(record) == set(TRANSFER_RECORD_FIELDS)


def _failure_matches(
    record: Mapping[str, Any],
    *,
    context_tokens: set[str],
    normalized_failure_code: str,
) -> bool:
    if not context_tokens and not normalized_failure_code:
        return False
    retrieval = record.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, Mapping) else {}
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    payload = _record_payload(record)
    failure_types = record.get("failure_types", payload.get("failure_types", ()))
    if not isinstance(failure_types, (list, tuple, set, frozenset)):
        failure_types = (failure_types,)
    explicit_codes = {
        _normalized_code(value)
        for value in (
            retrieval.get("failure_code"),
            provenance.get("failure_code"),
            *failure_types,
        )
        if _normalized_code(value)
    }
    if normalized_failure_code and normalized_failure_code in explicit_codes:
        return True
    candidate = " ".join(
        str(value or "")
        for value in (
            *explicit_codes,
            record.get("failure_condition"),
            record.get("diagnosis"),
            payload.get("failure_condition"),
            payload.get("diagnosis"),
        )
    )
    # One generic token such as ``payload`` is too weak to claim compatibility.
    return len(context_tokens & _meaningful_tokens(candidate)) >= 2


def _failure_level_matches(
    record: Mapping[str, Any],
    *,
    normalized_failure_level: str,
) -> bool:
    """Match V83's independent failure-level ranking component.

    Public six-field transfer records deliberately carry no failure level, so
    they remain neutral instead of receiving a fabricated native-only value.
    """

    if not normalized_failure_level:
        return False
    payload = _record_payload(record)
    if _is_six_field_record(payload):
        return False
    return _normalized_code(
        record.get("failure_level", payload.get("failure_level", ""))
    ) == normalized_failure_level


def _normalized_code(value: object) -> str:
    return "_".join(_TOKEN_RE.findall(str(value or "").lower()))


def _meaningful_tokens(value: object) -> set[str]:
    ignored = {"and", "for", "from", "the", "verifier", "with"}
    return {
        token
        for token in _TOKEN_RE.findall(str(value or "").lower())
        if len(token) >= 3 and token not in ignored
    }


__all__ = ["canonical_strategy_phase", "select_strategy_records"]
