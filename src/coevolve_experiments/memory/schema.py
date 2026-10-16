"""Public, experiment-only schema for Table 4 strategy transfer.

The transfer schema is deliberately independent from the deployment memory
schema in :mod:`praxis_coop`.  Deployment records contain local evidence and
execution references which are useful inside one run but are not admissible
inputs to a held-out transfer run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, TypedDict


STRATEGY_BANK_SCHEMA_VERSION = 1

# These are the *only* fields carried by a transferable record.  Keep this a
# tuple (rather than deriving it from annotations) so ordering and validation
# remain stable across Python versions.
TRANSFER_RECORD_FIELDS = (
    "phase_trigger",
    "failure_condition",
    "coordination_pattern",
    "verified_outcome",
    "avoid_rule",
    "repair_direction",
)

VerifiedOutcome = Literal["accepted", "rejected"]


class TransferStrategyRecord(TypedDict):
    phase_trigger: str
    failure_condition: str
    coordination_pattern: str
    verified_outcome: VerifiedOutcome
    avoid_rule: str
    repair_direction: str


class StrategyBankDocument(TypedDict):
    schema_version: int
    records: list[TransferStrategyRecord]


@dataclass(frozen=True)
class StrategyBankPolicy:
    """Frozen pre-registration policy for constructing a Table 4 bank.

    ``maximum_records`` and ``maximum_rejected_counterexamples`` bound the
    one-time source-bank construction before target evaluation.  They are
    distinct from the three-record query policy in
    :class:`StrategyRetrievalPolicy`.
    """

    required_source_runs: int = 5
    source_scene: str = "base"
    counterexample_policy: str = "verifier_backed_rejections_only"
    deduplication_policy: str = "canonical_six_field_sha256"
    ranking_policy: str = "run_support_then_evidence_then_digest"
    maximum_records: int = 48
    maximum_rejected_counterexamples: int = 16

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_source_runs": self.required_source_runs,
            "source_scene": self.source_scene,
            "counterexample_policy": self.counterexample_policy,
            "deduplication_policy": self.deduplication_policy,
            "ranking_policy": self.ranking_policy,
            "maximum_records": self.maximum_records,
            "maximum_rejected_counterexamples": (
                self.maximum_rejected_counterexamples
            ),
        }


# Callers cannot tune these values per result.  Changing them is a protocol
# revision and should consequently be a reviewed source change.
FROZEN_STRATEGY_BANK_POLICY = StrategyBankPolicy()


@dataclass(frozen=True)
class StrategyRetrievalPolicy:
    """Frozen V83-compatible query policy for strategy memory.

    The deployment V83 path jointly ranked exact-phase, permitted phase-alias,
    and phase-generic candidates and requested three rendered records.  Its
    balancing pass first prioritised no more than two verified failure
    counterexamples and reserved one slot for a verified success whenever one
    was available.

    Table-4 transfer records deliberately retain only the public six semantic
    fields, so their source confidence and aggregate evidence counts cannot be
    reconstructed byte-for-byte.  The adapter therefore gives every verified
    six-field record the same neutral confidence and one-record evidence
    equivalent.  Those ties are resolved by the stable frozen-bank/run-local
    source order, matching V83's later-record-first tie-break.  This is an
    explicit semantic projection, not a claim that the sanitized transfer
    records are byte-identical to V83's private deployment specs.
    """

    limit: int = 3
    prioritized_counterexample_limit: int = 2
    reserve_success_when_available: bool = True
    six_field_confidence_equivalent: float = 0.50
    six_field_evidence_equivalent: int = 1
    source_order_policy: str = "later_record_first"

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "prioritized_counterexample_limit": (
                self.prioritized_counterexample_limit
            ),
            "reserve_success_when_available": (
                self.reserve_success_when_available
            ),
            "six_field_confidence_equivalent": (
                self.six_field_confidence_equivalent
            ),
            "six_field_evidence_equivalent": self.six_field_evidence_equivalent,
            "source_order_policy": self.source_order_policy,
        }


FROZEN_STRATEGY_RETRIEVAL_POLICY = StrategyRetrievalPolicy()


class StrategyBankError(ValueError):
    """Base error for malformed inputs or a violated transfer protocol."""


class SourceRunValidationError(StrategyBankError):
    """Raised when the five source runs are not comparable formal evidence."""


class StrategyBankValidationError(StrategyBankError):
    """Raised when a sanitized bank violates the public transfer schema."""


class MemoryBootstrapError(StrategyBankError):
    """Raised when target memory cannot be initialized without contamination."""


def empty_strategy_bank() -> StrategyBankDocument:
    """Return a new empty bank; no mutable state is shared between calls."""

    return {
        "schema_version": STRATEGY_BANK_SCHEMA_VERSION,
        "records": [],
    }


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value for identities, never presentation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
