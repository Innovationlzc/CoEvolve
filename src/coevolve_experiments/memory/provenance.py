"""Strict source-identity binding for frozen Table-4 strategy banks.

The six-field strategy-bank document deliberately contains no source metadata.
Its sidecar is therefore the only artifact that binds those route-abstracted
records to the exact audited Table-1 cohort.  Formal consumers must validate
the complete sidecar against the :class:`~coevolve_experiments.contracts.StrategyBankPin`
rebuilt from the external protocol lock; checking only the bank digest or the
source model is insufficient because two cohorts can produce identical bank
content.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from ..contracts import StrategyBankPin
from ..serialization import strict_json_loads
from .schema import FROZEN_STRATEGY_BANK_POLICY, semantic_sha256


STRATEGY_TRANSFER_PROVENANCE_KIND = "strategy_transfer_provenance"
STRATEGY_TRANSFER_PROVENANCE_SCHEMA_VERSION = 3

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ROOT_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "bank_sha256",
        "source_model",
        "source_provider_route_id",
        "source_provider_route_fingerprint",
        "source_scene",
        "source_protocol",
        "source_method_commit",
        "source_seeds",
        "source_condition_fingerprint",
        "source_result_manifest_hash",
        "policy",
        "summary",
        "sources",
    }
)


class StrategyBankProvenanceError(ValueError):
    """A strategy-bank sidecar is incomplete, malformed, or misbound."""


@dataclass(frozen=True, slots=True)
class StrategyBankProvenanceValidation:
    payload: Mapping[str, Any]
    semantic_sha256: str
    file_sha256: str


def load_strategy_bank_provenance(
    value: object,
) -> tuple[dict[str, Any], str]:
    """Load a sidecar and return its payload plus exact-byte hash when present."""

    if hasattr(value, "provenance"):
        value = getattr(value, "provenance")
    file_sha256 = ""
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            raw = path.read_bytes()
            value = strict_json_loads(raw)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise StrategyBankProvenanceError(
                f"cannot read strategy bank provenance {path}: {exc}"
            ) from exc
        file_sha256 = hashlib.sha256(raw).hexdigest()
    if not isinstance(value, Mapping):
        raise StrategyBankProvenanceError(
            "strategy bank provenance root must be an object"
        )
    return dict(value), file_sha256


def validate_strategy_bank_provenance(
    provenance: object,
    *,
    strategy_bank_pin: StrategyBankPin,
    expected_bank_sha256: str | None = None,
    expected_source_scene: str = "base",
) -> StrategyBankProvenanceValidation:
    """Validate the entire source identity against one plan-derived bank pin.

    ``strategy_bank_pin`` is expected to come from a :class:`RunSpec` rebuilt
    from its embedded :class:`ProtocolLock`.  The provenance document is never
    allowed to choose or fill any of these expected values.
    """

    if not isinstance(strategy_bank_pin, StrategyBankPin):
        raise StrategyBankProvenanceError(
            "strategy_bank_pin must be a plan-derived StrategyBankPin"
        )
    payload, file_sha256 = load_strategy_bank_provenance(provenance)
    actual_fields = frozenset(payload)
    if actual_fields != _ROOT_FIELDS:
        raise StrategyBankProvenanceError(
            "strategy bank provenance fields differ: "
            f"missing={sorted(_ROOT_FIELDS - actual_fields)}, "
            f"unknown={sorted(actual_fields - _ROOT_FIELDS)}"
        )
    if payload.get("kind") != STRATEGY_TRANSFER_PROVENANCE_KIND:
        raise StrategyBankProvenanceError(
            "strategy bank provenance has the wrong kind"
        )
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != STRATEGY_TRANSFER_PROVENANCE_SCHEMA_VERSION
    ):
        raise StrategyBankProvenanceError(
            "strategy bank provenance schema_version 3 is required for full "
            "Table-1 source and provider-route identity binding"
        )

    bank_sha256 = expected_bank_sha256 or strategy_bank_pin.sha256
    if not isinstance(bank_sha256, str) or not _SHA256_RE.fullmatch(bank_sha256):
        raise StrategyBankProvenanceError(
            "expected strategy bank hash must be lowercase SHA-256"
        )
    expected = {
        "bank_sha256": bank_sha256,
        "source_model": strategy_bank_pin.source_backbone,
        "source_provider_route_id": strategy_bank_pin.source_provider_route_id,
        "source_provider_route_fingerprint": (
            strategy_bank_pin.source_provider_route_fingerprint
        ),
        "source_scene": expected_source_scene,
        "source_protocol": strategy_bank_pin.source_protocol_id,
        "source_method_commit": strategy_bank_pin.source_method_commit,
        "source_seeds": list(strategy_bank_pin.source_seeds),
        "source_condition_fingerprint": (
            strategy_bank_pin.source_condition_fingerprint
        ),
        "source_result_manifest_hash": (
            strategy_bank_pin.source_result_manifest_hash
        ),
        "policy": FROZEN_STRATEGY_BANK_POLICY.as_dict(),
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise StrategyBankProvenanceError(
                f"strategy bank provenance {field} differs from selected RunSpec"
            )

    # Give malformed source identities a precise failure even if a caller ever
    # constructs a non-dataclass expectation at a future boundary.
    if not _GIT_COMMIT_RE.fullmatch(str(payload["source_method_commit"])):
        raise StrategyBankProvenanceError(
            "strategy bank provenance source_method_commit is not a full commit"
        )
    for field in (
        "bank_sha256",
        "source_condition_fingerprint",
        "source_result_manifest_hash",
        "source_provider_route_fingerprint",
    ):
        if not _SHA256_RE.fullmatch(str(payload[field])):
            raise StrategyBankProvenanceError(
                f"strategy bank provenance {field} is not lowercase SHA-256"
            )
    if not isinstance(payload.get("summary"), Mapping):
        raise StrategyBankProvenanceError(
            "strategy bank provenance summary must be an object"
        )
    if not isinstance(payload.get("sources"), list):
        raise StrategyBankProvenanceError(
            "strategy bank provenance sources must be an array"
        )

    return StrategyBankProvenanceValidation(
        payload=payload,
        semantic_sha256=semantic_sha256(payload),
        file_sha256=file_sha256,
    )


__all__ = [
    "STRATEGY_TRANSFER_PROVENANCE_KIND",
    "STRATEGY_TRANSFER_PROVENANCE_SCHEMA_VERSION",
    "StrategyBankProvenanceError",
    "StrategyBankProvenanceValidation",
    "load_strategy_bank_provenance",
    "validate_strategy_bank_provenance",
]
