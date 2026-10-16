"""Run-local online memory schema shared by Full CoEvolve conditions.

Strategy proposals are route-abstracted *before* they enter the mutable
overlay.  The writer may emit a richer JSON object, but only the same public
six-field strategy vocabulary used by transfer banks survives promotion.
Host-owned rollout provenance and lookup keys remain outside that proposal.
This keeps later within-run retrieval useful without turning the overlay into
an unbounded channel for coordinates, action magnitudes, or source artifacts.
"""

from __future__ import annotations

from typing import Any, Mapping

from coevolve_experiments.runtime.contracts import RolloutRecord
from coevolve_experiments.serialization import to_primitive

from .leak_check import sanitize_transfer_text, scan_memory_leaks
from .retrieval import canonical_strategy_phase
from .schema import TRANSFER_RECORD_FIELDS, semantic_sha256


ONLINE_OVERLAY_KIND = "run_local_strategy_overlay"
ONLINE_OVERLAY_SCHEMA_VERSION = 2
ACTION_MEMORY_KIND = "verified_action_exemplar_memory"
ACTION_MEMORY_SCHEMA_VERSION = 1

# At most one strategy/action promotion can follow one completed physical
# transaction, and the frozen formal budget is 80.  The caps prevent malformed
# files from becoming an unbounded prompt channel without changing valid runs.
MAX_RUN_LOCAL_STRATEGY_ENTRIES = 80
MAX_RUN_LOCAL_ACTION_EXEMPLARS = 80


class OnlineMemoryValidationError(ValueError):
    """Malformed or unauditable run-local online memory."""


def empty_online_strategy_overlay() -> dict[str, Any]:
    return {
        "kind": ONLINE_OVERLAY_KIND,
        "schema_version": ONLINE_OVERLAY_SCHEMA_VERSION,
        "entries": [],
    }


def empty_action_exemplar_memory() -> dict[str, Any]:
    return {
        "kind": ACTION_MEMORY_KIND,
        "schema_version": ACTION_MEMORY_SCHEMA_VERSION,
        "exemplars": [],
    }


def validate_online_strategy_overlay(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OnlineMemoryValidationError("online strategy overlay root must be an object")
    payload = dict(value)
    expected_root = {"kind", "schema_version", "entries"}
    if set(payload) != expected_root:
        raise OnlineMemoryValidationError(
            "online strategy overlay has non-canonical root fields"
        )
    if payload.get("kind") != ONLINE_OVERLAY_KIND:
        raise OnlineMemoryValidationError("online strategy overlay has the wrong kind")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != ONLINE_OVERLAY_SCHEMA_VERSION
    ):
        raise OnlineMemoryValidationError(
            "online strategy overlay has unsupported schema_version"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise OnlineMemoryValidationError("online strategy overlay entries must be an array")
    if len(entries) > MAX_RUN_LOCAL_STRATEGY_ENTRIES:
        raise OnlineMemoryValidationError("online strategy overlay exceeds its frozen cap")
    identities: set[tuple[int, str]] = set()
    for index, entry in enumerate(entries):
        _validate_strategy_entry(entry, index=index)
        provenance = entry["provenance"]
        identity = (provenance["transaction_index"], provenance["ast_fingerprint"])
        if identity in identities:
            raise OnlineMemoryValidationError(
                "online strategy entries duplicate a transaction/AST identity"
            )
        identities.add(identity)
    _require_finite_json(payload, "online strategy overlay")
    return payload


def validate_action_exemplar_memory(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OnlineMemoryValidationError("action exemplar memory root must be an object")
    payload = dict(value)
    expected_root = {"kind", "schema_version", "exemplars"}
    if set(payload) != expected_root:
        raise OnlineMemoryValidationError(
            "action exemplar memory has non-canonical root fields"
        )
    if payload.get("kind") != ACTION_MEMORY_KIND:
        raise OnlineMemoryValidationError("action exemplar memory has the wrong kind")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != ACTION_MEMORY_SCHEMA_VERSION
    ):
        raise OnlineMemoryValidationError(
            "action exemplar memory has unsupported schema_version"
        )
    exemplars = payload.get("exemplars")
    if not isinstance(exemplars, list):
        raise OnlineMemoryValidationError("action exemplar memory exemplars must be an array")
    if len(exemplars) > MAX_RUN_LOCAL_ACTION_EXEMPLARS:
        raise OnlineMemoryValidationError("action exemplar memory exceeds its frozen cap")
    identities: set[tuple[int, str]] = set()
    for index, exemplar in enumerate(exemplars):
        _validate_action_exemplar(exemplar, index=index)
        identity = (exemplar["transaction_index"], exemplar["ast_fingerprint"])
        if identity in identities:
            raise OnlineMemoryValidationError(
                "action exemplars duplicate a transaction/AST identity"
            )
        identities.add(identity)
    _require_finite_json(payload, "action exemplar memory")
    return payload


def strategy_entry_from_rollout(proposal: Any, rollout: RolloutRecord) -> dict[str, Any]:
    """Route-abstract a writer proposal and bind Host verifier provenance."""

    if not isinstance(proposal, Mapping):
        raise OnlineMemoryValidationError("memory-update proposal must be an object")
    proposal_json = _require_finite_json(proposal, "memory-update proposal")
    if not isinstance(proposal_json, Mapping):
        raise OnlineMemoryValidationError("memory-update proposal must remain an object")
    provenance = _rollout_provenance(rollout)
    abstracted = _route_abstract_strategy_proposal(
        proposal_json,
        rollout=rollout,
        provenance=provenance,
    )
    entry = {
        "proposal": abstracted,
        "provenance": provenance,
        "retrieval": {
            # These lookup keys come from the executed Host-owned transaction;
            # model-authored memory text cannot route itself into later prompts.
            "phase": canonical_strategy_phase(
                rollout.policy.proposal.intent.lifecycle_phase
            ),
            "failure_code": provenance["failure_code"],
        },
    }
    _validate_strategy_entry(entry, index=0)
    return entry


def action_exemplar_from_rollout(rollout: RolloutRecord) -> dict[str, Any]:
    """Serialize the compiler-owned intent of one accepted physical rollout."""

    if not rollout.verifier.accepted:
        raise OnlineMemoryValidationError(
            "action exemplars require a host-verifier accepted rollout"
        )
    provenance = _rollout_provenance(rollout)
    intent = _require_finite_json(
        rollout.policy.proposal.intent,
        "action exemplar intent",
    )
    exemplar = {
        "transaction_index": provenance["transaction_index"],
        "ast_fingerprint": provenance["ast_fingerprint"],
        "intent": intent,
        "verifier_outcome": provenance["verifier_outcome"],
        "terminal": provenance["terminal"],
        "phase_completion": provenance["phase_completion"],
        "verifier_evidence_sha256": provenance["verifier_evidence_sha256"],
    }
    _validate_action_exemplar(exemplar, index=0)
    return exemplar


def _rollout_provenance(rollout: RolloutRecord) -> dict[str, Any]:
    if not isinstance(rollout, RolloutRecord):
        raise OnlineMemoryValidationError("promotion requires RolloutRecord")
    if not rollout.dispatch.physically_dispatched:
        raise OnlineMemoryValidationError("promotion requires physical dispatch")
    if not rollout.dispatch.physical_transaction_completed:
        raise OnlineMemoryValidationError(
            "promotion requires a completed physical transaction"
        )
    evidence = _require_finite_json(
        rollout.verifier.evidence,
        "host verifier evidence",
    )
    return {
        "transaction_index": rollout.transaction_index,
        "checkpoint_id": rollout.checkpoint.checkpoint_id,
        "ast_fingerprint": rollout.policy.ast_fingerprint,
        "verifier_outcome": (
            "accepted" if rollout.verifier.accepted else "rejected"
        ),
        "terminal": rollout.verifier.terminal,
        "failure_code": rollout.verifier.failure_code or "",
        "phase_completion": rollout.verifier.phase_completion,
        "verifier_evidence_sha256": semantic_sha256(evidence),
    }


def _validate_strategy_entry(value: object, *, index: int) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "proposal",
        "provenance",
        "retrieval",
    }:
        raise OnlineMemoryValidationError(
            f"online strategy entries[{index}] fields are not canonical"
        )
    proposal = value.get("proposal")
    if not isinstance(proposal, Mapping):
        raise OnlineMemoryValidationError(
            f"online strategy entries[{index}].proposal must be an object"
        )
    if set(proposal) != set(TRANSFER_RECORD_FIELDS):
        raise OnlineMemoryValidationError(
            f"online strategy entries[{index}].proposal must use the canonical "
            "six-field route abstraction"
        )
    provenance = value.get("provenance")
    _validate_rollout_provenance(provenance, label=f"entries[{index}].provenance")
    retrieval = value.get("retrieval")
    if not isinstance(retrieval, Mapping) or set(retrieval) != {
        "phase",
        "failure_code",
    }:
        raise OnlineMemoryValidationError(
            f"online strategy entries[{index}].retrieval fields are not canonical"
        )
    phase = retrieval.get("phase")
    if (
        not isinstance(phase, str)
        or not phase.strip()
        or phase != canonical_strategy_phase(phase)
    ):
        raise OnlineMemoryValidationError(
            f"online strategy entries[{index}].retrieval.phase is invalid"
        )
    failure = retrieval.get("failure_code")
    if not isinstance(failure, str) or failure != provenance.get("failure_code"):
        raise OnlineMemoryValidationError(
            f"online strategy entries[{index}].retrieval.failure_code differs from provenance"
        )
    expected_outcome = provenance.get("verifier_outcome")
    if proposal.get("verified_outcome") != expected_outcome:
        raise OnlineMemoryValidationError(
            f"online strategy entries[{index}].proposal outcome differs from provenance"
        )
    if canonical_strategy_phase(proposal.get("phase_trigger")) != phase:
        raise OnlineMemoryValidationError(
            f"online strategy entries[{index}].proposal phase differs from Host routing"
        )
    leaks = scan_memory_leaks(proposal)
    if leaks:
        categories = ",".join(sorted({item.category for item in leaks}))
        raise OnlineMemoryValidationError(
            f"online strategy entries[{index}].proposal leaks route-local data: "
            f"{categories}"
        )


def _route_abstract_strategy_proposal(
    proposal: Mapping[str, Any],
    *,
    rollout: RolloutRecord,
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    """Project arbitrary writer JSON onto the leak-free strategy schema.

    Phase and outcome are derived from the executed transaction, never from
    writer-authored routing fields.  Semantic text is deterministically
    sanitized; empty or fully redacted fields receive stable abstractions.
    """

    phase = canonical_strategy_phase(rollout.policy.proposal.intent.lifecycle_phase)
    outcome = str(provenance["verifier_outcome"])
    if outcome == "rejected":
        failure_source = _first_present(
            proposal,
            "failure_condition",
            "diagnosis",
            "failure",
            "failure_code",
        )
        if failure_source is None:
            failure_source = rollout.verifier.report
        failure_condition = _abstract_text(
            failure_source,
            "the verifier reports loss of cooperative stability",
        )
    else:
        failure_condition = (
            "no verifier-reported failure under the applicable cooperative state"
        )

    coordination_pattern = _abstract_text(
        _first_present(
            proposal,
            "coordination_pattern",
            "strategy",
            "coordination",
            "decision",
        )
        or proposal,
        "preserve coordinated support and role-compatible motion",
    )
    avoid_rule = _abstract_text(
        _first_present(proposal, "avoid_rule", "avoid", "forbidden"),
        "do not repeat a verifier-rejected coordination pattern without a "
        "material repair",
    )
    repair_direction = _abstract_text(
        _first_present(
            proposal,
            "repair_direction",
            "repair",
            "recommendation",
            "decision",
        ),
        "change the coordination abstraction while preserving verified support",
    )
    abstracted = {
        "phase_trigger": phase,
        "failure_condition": failure_condition,
        "coordination_pattern": coordination_pattern,
        "verified_outcome": outcome,
        "avoid_rule": avoid_rule,
        "repair_direction": repair_direction,
    }
    leaks = scan_memory_leaks(abstracted)
    if leaks:
        categories = ",".join(sorted({item.category for item in leaks}))
        raise OnlineMemoryValidationError(
            "route abstraction failed its final leak gate: " + categories
        )
    return abstracted


def _first_present(proposal: Mapping[str, Any], *keys: str) -> object | None:
    for key in keys:
        value = proposal.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _abstract_text(value: object, fallback: str) -> str:
    text = sanitize_transfer_text(value)
    if not text or scan_memory_leaks(text):
        text = fallback
    if scan_memory_leaks(text):  # defensive: constants are part of the contract
        raise OnlineMemoryValidationError("route-abstraction fallback is not leak free")
    return text


def _validate_rollout_provenance(value: object, *, label: str) -> None:
    expected = {
        "transaction_index",
        "checkpoint_id",
        "ast_fingerprint",
        "verifier_outcome",
        "terminal",
        "failure_code",
        "phase_completion",
        "verifier_evidence_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise OnlineMemoryValidationError(f"{label} fields are not canonical")
    transaction = value.get("transaction_index")
    if isinstance(transaction, bool) or not isinstance(transaction, int) or transaction <= 0:
        raise OnlineMemoryValidationError(f"{label}.transaction_index is invalid")
    checkpoint = value.get("checkpoint_id")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise OnlineMemoryValidationError(f"{label}.checkpoint_id is invalid")
    fingerprint = value.get("ast_fingerprint")
    if not _is_sha256(fingerprint):
        raise OnlineMemoryValidationError(f"{label}.ast_fingerprint is invalid")
    if value.get("verifier_outcome") not in {"accepted", "rejected"}:
        raise OnlineMemoryValidationError(f"{label}.verifier_outcome is invalid")
    if not isinstance(value.get("terminal"), bool):
        raise OnlineMemoryValidationError(f"{label}.terminal is invalid")
    failure = value.get("failure_code")
    if not isinstance(failure, str):
        raise OnlineMemoryValidationError(f"{label}.failure_code is invalid")
    phase = value.get("phase_completion")
    if phase is not None and (
        isinstance(phase, bool)
        or not isinstance(phase, (int, float))
        or not 0.0 <= float(phase) <= 1.0
    ):
        raise OnlineMemoryValidationError(f"{label}.phase_completion is invalid")
    evidence_hash = value.get("verifier_evidence_sha256")
    if not _is_sha256(evidence_hash):
        raise OnlineMemoryValidationError(
            f"{label}.verifier_evidence_sha256 is invalid"
        )


def _validate_action_exemplar(value: object, *, index: int) -> None:
    expected = {
        "transaction_index",
        "ast_fingerprint",
        "intent",
        "verifier_outcome",
        "terminal",
        "phase_completion",
        "verifier_evidence_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise OnlineMemoryValidationError(
            f"action exemplars[{index}] fields are not canonical"
        )
    synthetic = {
        "transaction_index": value.get("transaction_index"),
        "checkpoint_id": "run-local",
        "ast_fingerprint": value.get("ast_fingerprint"),
        "verifier_outcome": value.get("verifier_outcome"),
        "terminal": value.get("terminal"),
        "failure_code": "",
        "phase_completion": value.get("phase_completion"),
        "verifier_evidence_sha256": value.get("verifier_evidence_sha256"),
    }
    _validate_rollout_provenance(
        synthetic,
        label=f"action exemplars[{index}]",
    )
    if value.get("verifier_outcome") != "accepted":
        raise OnlineMemoryValidationError(
            f"action exemplars[{index}] is not verifier accepted"
        )
    if not isinstance(value.get("intent"), Mapping):
        raise OnlineMemoryValidationError(f"action exemplars[{index}].intent is invalid")


def _require_finite_json(value: object, label: str) -> Any:
    try:
        primitive = to_primitive(value)
        # semantic_sha256 uses allow_nan=False and therefore also rejects NaN
        # and infinities, which cannot be stable protocol evidence.
        semantic_sha256(primitive)
    except (TypeError, ValueError) as exc:
        raise OnlineMemoryValidationError(f"{label} is not finite JSON: {exc}") from exc
    return primitive


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ACTION_MEMORY_KIND",
    "ACTION_MEMORY_SCHEMA_VERSION",
    "MAX_RUN_LOCAL_ACTION_EXEMPLARS",
    "MAX_RUN_LOCAL_STRATEGY_ENTRIES",
    "ONLINE_OVERLAY_KIND",
    "ONLINE_OVERLAY_SCHEMA_VERSION",
    "OnlineMemoryValidationError",
    "action_exemplar_from_rollout",
    "empty_action_exemplar_memory",
    "empty_online_strategy_overlay",
    "strategy_entry_from_rollout",
    "validate_action_exemplar_memory",
    "validate_online_strategy_overlay",
]
