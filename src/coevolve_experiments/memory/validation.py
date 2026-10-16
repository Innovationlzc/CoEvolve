"""Strict validation for the isolated Table 4 transfer schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .leak_check import scan_memory_leaks
from .schema import (
    FROZEN_STRATEGY_BANK_POLICY,
    STRATEGY_BANK_SCHEMA_VERSION,
    TRANSFER_RECORD_FIELDS,
    StrategyBankValidationError,
    semantic_sha256,
)
from ..serialization import strict_json_loads


@dataclass(frozen=True)
class StrategyBankValidation:
    valid: bool
    errors: tuple[str, ...]
    record_count: int
    rejected_count: int
    bank_sha256: str

    def __bool__(self) -> bool:
        return self.valid


def load_strategy_bank(value: object) -> dict[str, Any]:
    """Load a bank from a mapping, path, or an export result."""

    if hasattr(value, "bank"):
        value = getattr(value, "bank")
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            value = strict_json_loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StrategyBankValidationError(
                f"cannot read strategy bank {path}: {exc}"
            ) from exc
    if not isinstance(value, Mapping):
        raise StrategyBankValidationError("strategy bank root must be an object")
    return dict(value)


def validate_strategy_bank(
    bank: object,
    *,
    raise_on_error: bool = True,
) -> StrategyBankValidation:
    """Validate exact fields, caps, deterministic uniqueness, and all leaks."""

    errors: list[str] = []
    try:
        payload = load_strategy_bank(bank)
    except StrategyBankValidationError as exc:
        if raise_on_error:
            raise
        return StrategyBankValidation(False, (str(exc),), 0, 0, "")

    expected_root = {"schema_version", "records"}
    if set(payload) != expected_root:
        errors.append(
            "strategy bank root fields must be exactly "
            f"{sorted(expected_root)}; got {sorted(map(str, payload))}"
        )
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != STRATEGY_BANK_SCHEMA_VERSION
    ):
        errors.append(
            f"unsupported schema_version={payload.get('schema_version')!r}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        errors.append("strategy bank records must be a list")
        records = []

    if len(records) > FROZEN_STRATEGY_BANK_POLICY.maximum_records:
        errors.append(
            "strategy bank exceeds frozen record cap "
            f"{FROZEN_STRATEGY_BANK_POLICY.maximum_records}"
        )

    expected_fields = set(TRANSFER_RECORD_FIELDS)
    identities: set[str] = set()
    rejected_count = 0
    for index, record in enumerate(records):
        prefix = f"records[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        if set(record) != expected_fields:
            errors.append(
                f"{prefix} fields must be exactly {sorted(expected_fields)}; "
                f"got {sorted(map(str, record))}"
            )
        for field in TRANSFER_RECORD_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
            elif value != value.strip():
                errors.append(f"{prefix}.{field} has non-canonical outer whitespace")
        outcome = record.get("verified_outcome")
        if outcome not in {"accepted", "rejected"}:
            errors.append(
                f"{prefix}.verified_outcome must be accepted or rejected"
            )
        if outcome == "rejected":
            rejected_count += 1

        identity = semantic_sha256(dict(record))
        if identity in identities:
            errors.append(f"{prefix} duplicates an earlier six-field record")
        identities.add(identity)

        for finding in scan_memory_leaks(record):
            errors.append(
                f"{prefix}{finding.path[1:]} leaks {finding.category}: "
                f"{finding.detail}"
            )

    if rejected_count > FROZEN_STRATEGY_BANK_POLICY.maximum_rejected_counterexamples:
        errors.append(
            "strategy bank exceeds frozen rejected-counterexample cap "
            f"{FROZEN_STRATEGY_BANK_POLICY.maximum_rejected_counterexamples}"
        )

    digest = semantic_sha256(payload) if not errors else ""
    result = StrategyBankValidation(
        valid=not errors,
        errors=tuple(errors),
        record_count=len(records),
        rejected_count=rejected_count,
        bank_sha256=digest,
    )
    if errors and raise_on_error:
        raise StrategyBankValidationError("; ".join(errors))
    return result
