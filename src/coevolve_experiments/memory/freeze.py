"""Two-stage candidate/locked freeze receipts for Table-4 strategy banks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any

from ..serialization import strict_json_loads, to_primitive
from .exporter import StrategyBankExport
from .posthoc_evidence import PosthocEvidencePackage
from .schema import StrategyBankError, semantic_sha256


STRATEGY_BANK_CANDIDATE_KIND = "table4_strategy_bank_candidate_attestation"
STRATEGY_BANK_CANDIDATE_SCHEMA_VERSION = 1
_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "source_cohort",
        "source",
        "source_aggregate_manifest_sha256",
        "posthoc_evidence_manifest_sha256",
        "posthoc_evidence_payload_sha256",
        "bank_sha256",
        "provenance_sha256",
        "lock_field",
        "candidate_payload_sha256",
    }
)
_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class StrategyBankCandidateAttestation:
    payload: Mapping[str, Any]
    path: Path | None
    file_sha256: str

    @property
    def bank_sha256(self) -> str:
        return str(self.payload["bank_sha256"])


def write_strategy_bank_candidate_attestation(
    result: StrategyBankExport,
    cohort_audit: object,
    package: PosthocEvidencePackage,
    source_cohort: str,
    output_path: str | Path,
) -> StrategyBankCandidateAttestation:
    """Write the candidate hash needed to complete the protocol lock."""

    payload = _candidate_payload(result, cohort_audit, package, source_cohort)
    unsigned = dict(payload)
    payload["candidate_payload_sha256"] = semantic_sha256(unsigned)
    destination = Path(output_path)
    rendered = _render(payload)
    if destination.is_symlink():
        raise StrategyBankError("candidate attestation output must not be a symlink")
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise StrategyBankError(
                f"refusing to overwrite non-identical candidate attestation: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(rendered)
        temporary.replace(destination)
    return load_strategy_bank_candidate_attestation(destination)


def load_strategy_bank_candidate_attestation(
    value: str | Path | Mapping[str, Any],
) -> StrategyBankCandidateAttestation:
    path: Path | None = None
    file_sha256 = ""
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_symlink() or not path.is_file():
            raise StrategyBankError("candidate attestation must be a regular file")
        raw = path.read_bytes()
        file_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            value = strict_json_loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise StrategyBankError(f"invalid candidate attestation: {exc}") from exc
    if not isinstance(value, Mapping):
        raise StrategyBankError("candidate attestation root must be an object")
    payload = dict(value)
    if set(payload) != _FIELDS:
        raise StrategyBankError(
            "candidate attestation fields differ: "
            f"missing={sorted(_FIELDS - set(payload))}, "
            f"unknown={sorted(set(payload) - _FIELDS)}"
        )
    if payload.get("kind") != STRATEGY_BANK_CANDIDATE_KIND:
        raise StrategyBankError("candidate attestation kind is invalid")
    if payload.get("schema_version") != STRATEGY_BANK_CANDIDATE_SCHEMA_VERSION:
        raise StrategyBankError("candidate attestation schema is invalid")
    cohort = payload.get("source_cohort")
    if cohort not in {"gpt", "gemini"}:
        raise StrategyBankError("candidate source_cohort is invalid")
    if payload.get("lock_field") != f"{cohort}_strategy_bank_hash":
        raise StrategyBankError("candidate lock field does not match source cohort")
    source = payload.get("source")
    from ..contracts import Table1SourceCohort

    if not isinstance(source, Mapping):
        raise StrategyBankError("candidate source identity must be an object")
    expected_source_fields = {field.name for field in fields(Table1SourceCohort)}
    if set(source) != expected_source_fields:
        raise StrategyBankError("candidate source identity fields are invalid")
    source_values = dict(source)
    seeds = source_values.get("seeds")
    if not isinstance(seeds, list):
        raise StrategyBankError("candidate source seeds must be an array")
    source_values["seeds"] = tuple(seeds)
    try:
        Table1SourceCohort(**source_values)
    except (TypeError, ValueError) as exc:
        raise StrategyBankError(f"candidate source identity is invalid: {exc}") from exc
    for field in (
        "source_aggregate_manifest_sha256",
        "posthoc_evidence_manifest_sha256",
        "posthoc_evidence_payload_sha256",
        "bank_sha256",
        "provenance_sha256",
        "candidate_payload_sha256",
    ):
        if not _is_sha256(payload.get(field)):
            raise StrategyBankError(f"candidate {field} is not SHA-256")
    unsigned = dict(payload)
    declared = unsigned.pop("candidate_payload_sha256")
    if declared != semantic_sha256(unsigned):
        raise StrategyBankError("candidate attestation self-hash mismatch")
    return StrategyBankCandidateAttestation(payload, path, file_sha256)


def validate_locked_strategy_bank_candidate(
    result: StrategyBankExport,
    cohort_audit: object,
    package: PosthocEvidencePackage,
    source_cohort: str,
    candidate: str | Path | Mapping[str, Any],
    *,
    locked_bank_sha256: str,
) -> StrategyBankCandidateAttestation:
    """Require a locked re-export to reproduce the full candidate identity."""

    loaded = load_strategy_bank_candidate_attestation(candidate)
    expected = _candidate_payload(result, cohort_audit, package, source_cohort)
    for field, expected_value in expected.items():
        if loaded.payload.get(field) != expected_value:
            raise StrategyBankError(
                f"locked export differs from candidate attestation field {field}"
            )
    if loaded.bank_sha256 != locked_bank_sha256:
        raise StrategyBankError(
            "protocol lock bank hash differs from candidate attestation"
        )
    return loaded


def _candidate_payload(
    result: StrategyBankExport,
    cohort_audit: object,
    package: PosthocEvidencePackage,
    source_cohort: str,
) -> dict[str, Any]:
    from ..reporting.table1_source import Table1CohortAudit

    if not isinstance(cohort_audit, Table1CohortAudit) or not cohort_audit.protocol_valid:
        raise StrategyBankError("candidate export requires a complete Table1CohortAudit")
    if source_cohort not in {"gpt", "gemini"}:
        raise StrategyBankError("candidate source_cohort must be gpt or gemini")
    return {
        "kind": STRATEGY_BANK_CANDIDATE_KIND,
        "schema_version": STRATEGY_BANK_CANDIDATE_SCHEMA_VERSION,
        "source_cohort": source_cohort,
        "source": to_primitive(cohort_audit.source),
        "source_aggregate_manifest_sha256": cohort_audit.manifest_sha256,
        "posthoc_evidence_manifest_sha256": package.manifest_sha256,
        "posthoc_evidence_payload_sha256": package.payload_sha256,
        "bank_sha256": str(result.provenance.get("bank_sha256", "")),
        "provenance_sha256": semantic_sha256(result.provenance),
        "lock_field": f"{source_cohort}_strategy_bank_hash",
    }


def _render(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


__all__ = [
    "STRATEGY_BANK_CANDIDATE_KIND",
    "STRATEGY_BANK_CANDIDATE_SCHEMA_VERSION",
    "StrategyBankCandidateAttestation",
    "load_strategy_bank_candidate_attestation",
    "validate_locked_strategy_bank_candidate",
    "write_strategy_bank_candidate_attestation",
]
