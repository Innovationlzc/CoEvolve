"""Independent foundation-model request and transport-usage reconciliation.

``FM calls`` counts logical requests issued by a method role. Provider retries
are transport attempts, not extra FM calls. A request may contain several
HTTP/transport attempts and may ultimately fail; both outcomes remain in the
ledger.

Billing is a separate completeness dimension. Providers do not always expose
per-request cost, and absence of that optional accounting must not invalidate
an otherwise valid physical run. Missing or inconsistent billing is reported
as unavailable/incomplete, never imputed as zero. Model identity and token
usage remain mandatory for successful responses, but are not manufactured for
failed requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence


FM_USAGE_LEDGER_FILENAME = "fm_usage_ledger.json"
FM_USAGE_LEDGER_KIND = "coevolve_fm_usage_ledger"
FM_USAGE_LEDGER_SCHEMA_VERSION = 2
FM_USAGE_LEDGER_LEGACY_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT_FIELDS = frozenset({
    "kind", "schema_version", "producer", "run_binding", "calls", "totals",
})
_PRODUCER_FIELDS = frozenset({"id", "source_sha256"})
_BINDING_FIELDS = frozenset({
    "run_id", "seed", "condition_fingerprint", "provider_route_id",
    "provider_route_fingerprint", "target_model",
})

# Schema v1 recorded successful responses only. Billing is optional while
# reading it so an otherwise valid legacy record is not lost merely because a
# provider did not expose a monetary amount.
_V1_CALL_REQUIRED_FIELDS = frozenset({
    "ordinal", "request_id", "role", "modality", "requested_model",
    "resolved_model", "response_model", "http_status", "input_tokens",
    "output_tokens",
})
_V1_CALL_ALLOWED_FIELDS = _V1_CALL_REQUIRED_FIELDS | {"billing_amount_usd"}
_V1_TOTAL_REQUIRED_FIELDS = frozenset({
    "llm_calls", "vlm_calls", "input_tokens", "output_tokens",
})
_V1_TOTAL_ALLOWED_FIELDS = _V1_TOTAL_REQUIRED_FIELDS | {"billing_amount_usd"}

# Schema v2 records logical requests. Nested attempts make logical-call and
# HTTP/transport-attempt totals independently recomputable.
_V2_CALL_REQUIRED_FIELDS = frozenset({
    "ordinal", "request_id", "role", "modality", "requested_model",
    "resolved_model", "outcome", "transport_attempts", "response_model",
    "input_tokens", "output_tokens",
})
_V2_CALL_ALLOWED_FIELDS = _V2_CALL_REQUIRED_FIELDS | {"billing_amount_usd"}
_V2_ATTEMPT_FIELDS = frozenset({
    "ordinal", "outcome", "http_status", "error_type",
})
_V2_TOTAL_REQUIRED_FIELDS = frozenset({
    "llm_calls", "vlm_calls", "successful_responses", "failed_requests",
    "transport_attempts", "input_tokens", "output_tokens",
})
_V2_TOTAL_BILLING_FIELDS = frozenset({
    "billing_records", "billing_amount_usd", "billing_completeness",
})
_V2_TOTAL_ALLOWED_FIELDS = _V2_TOTAL_REQUIRED_FIELDS | _V2_TOTAL_BILLING_FIELDS
_BILLING_STATUSES = frozenset({"complete", "partial", "unavailable"})


@dataclass(frozen=True)
class UsageEvidenceValidation:
    """Validation result with protocol and billing findings kept separate."""

    ledger_sha256: str = ""
    artifact_hashes: Mapping[str, str] | None = None
    request_ids: tuple[str, ...] = ()
    invalid_reasons: tuple[str, ...] = ()
    successful_request_ids: tuple[str, ...] = ()
    schema_version: int | None = None
    usage_totals: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    billing_amount_usd: float | None = None
    billing_completeness: str = "unavailable"
    billing_reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether logical-call/transport/token evidence is protocol-valid."""

        return not self.invalid_reasons

    @property
    def billing_complete(self) -> bool:
        """Whether the monetary total is fully reconciled and reportable."""

        return (
            self.billing_completeness == "complete"
            and self.billing_amount_usd is not None
            and not self.billing_reasons
        )


def validate_fm_usage_evidence(
    run_root: str | Path,
    *,
    manifest: Mapping[str, Any],
    expected_run_id: str,
    expected_seed: int,
    expected_condition_fingerprint: str,
    expected_provider_route_id: str,
    expected_provider_route_fingerprint: str,
    expected_target_model: str,
    expected_llm_calls: int | None,
    expected_vlm_calls: int | None,
    expected_input_tokens: int | None,
    expected_output_tokens: int | None,
    expected_billing_amount_usd: float | None,
    expected_billing_reconciliation_status: str | None = None,
    expected_schema_version: int | None = None,
    required_request_ids: Sequence[str] = (),
) -> UsageEvidenceValidation:
    """Recompute logical calls, attempts, tokens, and optional billing.

    Schema v2 is the production format. Schema v1 remains readable as a
    successful-response-only legacy format, where each record represents one
    logical request and one successful HTTP attempt.
    """

    root = Path(run_root)
    reasons: list[str] = []
    billing_reasons: list[str] = []
    result_hashes = manifest.get("result_artifact_hashes")
    result_hashes = result_hashes if isinstance(result_hashes, Mapping) else {}
    expected_hash = result_hashes.get("fm_usage_ledger_sha256")
    if not _sha256(expected_hash):
        return UsageEvidenceValidation(
            invalid_reasons=("FM_USAGE_LEDGER_PIN_MISSING",)
        )
    path = root / FM_USAGE_LEDGER_FILENAME
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        raw = path.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return UsageEvidenceValidation(
            invalid_reasons=("FM_USAGE_LEDGER_MISSING_OR_INVALID",)
        )
    artifact_hashes = {FM_USAGE_LEDGER_FILENAME: actual_hash}
    if actual_hash != expected_hash:
        reasons.append("FM_USAGE_LEDGER_HASH_MISMATCH")
    if not isinstance(value, Mapping):
        return UsageEvidenceValidation(
            ledger_sha256=actual_hash,
            artifact_hashes=artifact_hashes,
            invalid_reasons=("FM_USAGE_LEDGER_ROOT_INVALID",),
        )
    if set(value) != _ROOT_FIELDS:
        reasons.append("FM_USAGE_LEDGER_SCHEMA_FIELDS_INVALID")
    version = _integer(value.get("schema_version"))
    if (
        value.get("kind") != FM_USAGE_LEDGER_KIND
        or version not in {
            FM_USAGE_LEDGER_LEGACY_SCHEMA_VERSION,
            FM_USAGE_LEDGER_SCHEMA_VERSION,
        }
    ):
        reasons.append("FM_USAGE_LEDGER_IDENTITY_INVALID")
    if expected_schema_version is not None and version != expected_schema_version:
        reasons.append("FM_USAGE_LEDGER_SCHEMA_VERSION_MISMATCH")
    producer = value.get("producer")
    if (
        not isinstance(producer, Mapping)
        or set(producer) != _PRODUCER_FIELDS
        or not _nonempty(producer.get("id"))
        or not _sha256(producer.get("source_sha256"))
    ):
        reasons.append("FM_USAGE_LEDGER_PRODUCER_INVALID")
    expected_binding = {
        "run_id": expected_run_id,
        "seed": expected_seed,
        "condition_fingerprint": expected_condition_fingerprint,
        "provider_route_id": expected_provider_route_id,
        "provider_route_fingerprint": expected_provider_route_fingerprint,
        "target_model": expected_target_model,
    }
    binding = value.get("run_binding")
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_FIELDS:
        reasons.append("FM_USAGE_LEDGER_RUN_BINDING_INVALID")
    elif dict(binding) != expected_binding:
        reasons.append("FM_USAGE_LEDGER_RUN_BINDING_MISMATCH")

    calls = value.get("calls")
    if not isinstance(calls, list) or not calls:
        reasons.append("FM_USAGE_LEDGER_CALLS_INVALID")
        calls = []
    if version == FM_USAGE_LEDGER_SCHEMA_VERSION:
        reconciled = _validate_v2_calls(
            calls,
            expected_target_model=expected_target_model,
            reasons=reasons,
            billing_reasons=billing_reasons,
        )
        _validate_v2_totals(
            value.get("totals"),
            reconciled=reconciled,
            reasons=reasons,
            billing_reasons=billing_reasons,
        )
    else:
        reconciled = _validate_v1_calls(
            calls,
            expected_target_model=expected_target_model,
            reasons=reasons,
            billing_reasons=billing_reasons,
        )
        _validate_v1_totals(
            value.get("totals"),
            reconciled=reconciled,
            reasons=reasons,
            billing_reasons=billing_reasons,
        )

    expected_metrics = {
        "llm_calls": expected_llm_calls,
        "vlm_calls": expected_vlm_calls,
        "input_tokens": expected_input_tokens,
        "output_tokens": expected_output_tokens,
    }
    for field_name, wanted in expected_metrics.items():
        if wanted is not None and reconciled.usage_totals[field_name] != wanted:
            reasons.append(f"FM_USAGE_METRICS_{field_name.upper()}_MISMATCH")

    required = {
        str(item).strip() for item in required_request_ids if str(item).strip()
    }
    missing_requests = sorted(required - set(reconciled.successful_request_ids))
    if missing_requests:
        reasons.append("FM_USAGE_STALL_MODEL_OUTPUT_REQUEST_MISSING")

    billing_amount = reconciled.billing_amount_usd
    billing_status = reconciled.billing_completeness
    normalized_expected_status = str(
        expected_billing_reconciliation_status or ""
    ).strip().lower()
    accepted_expected_statuses = _BILLING_STATUSES | {
        "incomplete", "unknown", "not_available", "na",
    }
    if (
        normalized_expected_status
        and normalized_expected_status not in accepted_expected_statuses
    ):
        billing_reasons.append("FM_USAGE_METRICS_BILLING_STATUS_INVALID")
    if expected_billing_amount_usd is not None:
        if billing_amount is None or billing_status != "complete":
            billing_reasons.append("FM_USAGE_METRICS_BILLING_NOT_RECONCILABLE")
        elif not math.isclose(
            billing_amount,
            float(expected_billing_amount_usd),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            billing_reasons.append("FM_USAGE_METRICS_BILLING_AMOUNT_USD_MISMATCH")
    if normalized_expected_status == "complete" and billing_status != "complete":
        billing_reasons.append("FM_USAGE_METRICS_BILLING_COMPLETENESS_MISMATCH")

    deduped_billing_reasons = tuple(dict.fromkeys(billing_reasons))
    if deduped_billing_reasons:
        # A cost is reportable only when the complete ledger agrees with any
        # metrics-side billing claim.
        billing_amount = None
    return UsageEvidenceValidation(
        ledger_sha256=actual_hash,
        artifact_hashes=artifact_hashes,
        request_ids=reconciled.request_ids,
        invalid_reasons=tuple(dict.fromkeys(reasons)),
        successful_request_ids=reconciled.successful_request_ids,
        schema_version=version,
        usage_totals=MappingProxyType(dict(reconciled.usage_totals)),
        billing_amount_usd=billing_amount,
        billing_completeness=(
            billing_status if not deduped_billing_reasons else "incomplete"
        ),
        billing_reasons=deduped_billing_reasons,
    )


@dataclass(frozen=True)
class _ReconciledUsage:
    request_ids: tuple[str, ...]
    successful_request_ids: tuple[str, ...]
    usage_totals: Mapping[str, int]
    billing_records: int
    billing_amount_usd: float | None
    billing_completeness: str


def _validate_v1_calls(
    calls: Sequence[object],
    *,
    expected_target_model: str,
    reasons: list[str],
    billing_reasons: list[str],
) -> _ReconciledUsage:
    request_ids: list[str] = []
    usage = _empty_usage_totals()
    costs: list[float] = []
    for ordinal, call in enumerate(calls, 1):
        if not isinstance(call, Mapping):
            reasons.append("FM_USAGE_CALL_SCHEMA_INVALID")
            continue
        fields = set(call)
        if (
            not _V1_CALL_REQUIRED_FIELDS.issubset(fields)
            or not fields.issubset(_V1_CALL_ALLOWED_FIELDS)
        ):
            reasons.append("FM_USAGE_CALL_SCHEMA_INVALID")
            continue
        _validate_common_call(
            call,
            ordinal=ordinal,
            expected_target_model=expected_target_model,
            request_ids=request_ids,
            usage=usage,
            reasons=reasons,
        )
        for field_name in ("resolved_model", "response_model"):
            if call.get(field_name) != expected_target_model:
                reasons.append(f"FM_USAGE_CALL_{field_name.upper()}_MISMATCH")
        status = _integer(call.get("http_status"))
        if status is None or not 200 <= status < 300:
            reasons.append("FM_USAGE_CALL_HTTP_STATUS_INVALID")
        _accumulate_success_tokens(call, usage=usage, reasons=reasons)
        usage["successful_responses"] += 1
        usage["transport_attempts"] += 1
        _accumulate_optional_cost(
            call, costs=costs, billing_reasons=billing_reasons
        )
    return _finish_reconciliation(
        request_ids, request_ids, usage, costs, len(calls)
    )


def _validate_v2_calls(
    calls: Sequence[object],
    *,
    expected_target_model: str,
    reasons: list[str],
    billing_reasons: list[str],
) -> _ReconciledUsage:
    request_ids: list[str] = []
    successful_request_ids: list[str] = []
    usage = _empty_usage_totals()
    costs: list[float] = []
    for ordinal, call in enumerate(calls, 1):
        if not isinstance(call, Mapping):
            reasons.append("FM_USAGE_CALL_SCHEMA_INVALID")
            continue
        fields = set(call)
        if (
            not _V2_CALL_REQUIRED_FIELDS.issubset(fields)
            or not fields.issubset(_V2_CALL_ALLOWED_FIELDS)
        ):
            reasons.append("FM_USAGE_CALL_SCHEMA_INVALID")
            continue
        request_id = _validate_common_call(
            call,
            ordinal=ordinal,
            expected_target_model=expected_target_model,
            request_ids=request_ids,
            usage=usage,
            reasons=reasons,
        )
        resolved_model = call.get("resolved_model")
        if resolved_model is not None and resolved_model != expected_target_model:
            reasons.append("FM_USAGE_CALL_RESOLVED_MODEL_MISMATCH")
        attempt_outcome, attempt_count = _validate_transport_attempts(
            call.get("transport_attempts"), reasons=reasons
        )
        usage["transport_attempts"] += attempt_count
        outcome = call.get("outcome")
        if outcome not in {"success", "failure"}:
            reasons.append("FM_USAGE_CALL_OUTCOME_INVALID")
            outcome = "failure"
        expected_outcome = (
            "success" if attempt_outcome == "success" else "failure"
        )
        if outcome != expected_outcome:
            reasons.append("FM_USAGE_CALL_OUTCOME_ATTEMPT_MISMATCH")
        if outcome == "success":
            usage["successful_responses"] += 1
            if request_id:
                successful_request_ids.append(request_id)
            if resolved_model != expected_target_model:
                reasons.append("FM_USAGE_CALL_RESOLVED_MODEL_MISSING_OR_INVALID")
            if call.get("response_model") != expected_target_model:
                reasons.append("FM_USAGE_CALL_RESPONSE_MODEL_MISSING_OR_INVALID")
            _accumulate_success_tokens(call, usage=usage, reasons=reasons)
        else:
            usage["failed_requests"] += 1
            if call.get("response_model") is not None:
                reasons.append("FM_USAGE_FAILED_CALL_RESPONSE_MODEL_PRESENT")
            if (
                call.get("input_tokens") is not None
                or call.get("output_tokens") is not None
            ):
                reasons.append("FM_USAGE_FAILED_CALL_TOKEN_USAGE_PRESENT")
        _accumulate_optional_cost(
            call, costs=costs, billing_reasons=billing_reasons
        )
    return _finish_reconciliation(
        request_ids,
        successful_request_ids,
        usage,
        costs,
        len(calls),
    )


def _validate_common_call(
    call: Mapping[str, Any],
    *,
    ordinal: int,
    expected_target_model: str,
    request_ids: list[str],
    usage: dict[str, int],
    reasons: list[str],
) -> str:
    if call.get("ordinal") != ordinal:
        reasons.append("FM_USAGE_CALL_SEQUENCE_INVALID")
    request_id = str(call.get("request_id") or "").strip()
    if not request_id or request_id in request_ids:
        reasons.append("FM_USAGE_REQUEST_ID_INVALID_OR_DUPLICATE")
        request_id = ""
    else:
        request_ids.append(request_id)
    if not _nonempty(call.get("role")):
        reasons.append("FM_USAGE_CALL_ROLE_INVALID")
    modality = call.get("modality")
    if modality not in {"text", "vision"}:
        reasons.append("FM_USAGE_CALL_MODALITY_INVALID")
    else:
        usage["llm_calls" if modality == "text" else "vlm_calls"] += 1
    if call.get("requested_model") != expected_target_model:
        reasons.append("FM_USAGE_CALL_REQUESTED_MODEL_MISMATCH")
    return request_id


def _validate_transport_attempts(
    attempts: object,
    *,
    reasons: list[str],
) -> tuple[str, int]:
    if not isinstance(attempts, list) or not attempts:
        reasons.append("FM_USAGE_TRANSPORT_ATTEMPTS_INVALID")
        return "failure", 0
    final_outcome = "failure"
    for ordinal, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, Mapping) or set(attempt) != _V2_ATTEMPT_FIELDS:
            reasons.append("FM_USAGE_TRANSPORT_ATTEMPT_SCHEMA_INVALID")
            continue
        if attempt.get("ordinal") != ordinal:
            reasons.append("FM_USAGE_TRANSPORT_ATTEMPT_SEQUENCE_INVALID")
        outcome = attempt.get("outcome")
        status = _integer(attempt.get("http_status"))
        error_type = attempt.get("error_type")
        if outcome == "http_response":
            if (
                status is None
                or not 100 <= status <= 599
                or error_type is not None
            ):
                reasons.append("FM_USAGE_HTTP_ATTEMPT_INVALID")
            successful = status is not None and 200 <= status < 300
        elif outcome == "transport_error":
            if status is not None or not _nonempty(error_type):
                reasons.append("FM_USAGE_TRANSPORT_ERROR_ATTEMPT_INVALID")
            successful = False
        else:
            reasons.append("FM_USAGE_TRANSPORT_ATTEMPT_OUTCOME_INVALID")
            successful = False
        if successful and ordinal != len(attempts):
            reasons.append("FM_USAGE_TRANSPORT_RETRY_AFTER_SUCCESS")
        if ordinal == len(attempts):
            final_outcome = "success" if successful else "failure"
    return final_outcome, len(attempts)


def _accumulate_success_tokens(
    call: Mapping[str, Any],
    *,
    usage: dict[str, int],
    reasons: list[str],
) -> None:
    for field_name in ("input_tokens", "output_tokens"):
        count = _integer(call.get(field_name))
        if count is None or count < 0:
            reasons.append(f"FM_USAGE_CALL_{field_name.upper()}_INVALID")
        else:
            usage[field_name] += count


def _accumulate_optional_cost(
    call: Mapping[str, Any],
    *,
    costs: list[float],
    billing_reasons: list[str],
) -> None:
    if (
        "billing_amount_usd" not in call
        or call.get("billing_amount_usd") is None
    ):
        return
    cost = _number(call.get("billing_amount_usd"))
    if cost is None or cost < 0:
        billing_reasons.append("FM_USAGE_CALL_BILLING_INVALID")
    else:
        costs.append(cost)


def _finish_reconciliation(
    request_ids: Sequence[str],
    successful_request_ids: Sequence[str],
    usage: Mapping[str, int],
    costs: Sequence[float],
    call_count: int,
) -> _ReconciledUsage:
    billing_records = len(costs)
    if call_count > 0 and billing_records == call_count:
        billing_status = "complete"
        billing_amount: float | None = math.fsum(costs)
    elif billing_records:
        billing_status = "partial"
        billing_amount = None
    else:
        billing_status = "unavailable"
        billing_amount = None
    return _ReconciledUsage(
        request_ids=tuple(request_ids),
        successful_request_ids=tuple(successful_request_ids),
        usage_totals=MappingProxyType(dict(usage)),
        billing_records=billing_records,
        billing_amount_usd=billing_amount,
        billing_completeness=billing_status,
    )


def _validate_v1_totals(
    totals: object,
    *,
    reconciled: _ReconciledUsage,
    reasons: list[str],
    billing_reasons: list[str],
) -> None:
    if not isinstance(totals, Mapping):
        reasons.append("FM_USAGE_TOTALS_SCHEMA_INVALID")
        return
    fields = set(totals)
    if (
        not _V1_TOTAL_REQUIRED_FIELDS.issubset(fields)
        or not fields.issubset(_V1_TOTAL_ALLOWED_FIELDS)
    ):
        reasons.append("FM_USAGE_TOTALS_SCHEMA_INVALID")
    _compare_operational_totals(
        totals, reconciled.usage_totals, reasons, fields=_V1_TOTAL_REQUIRED_FIELDS
    )
    if "billing_amount_usd" in totals:
        _compare_declared_billing_amount(
            totals.get("billing_amount_usd"), reconciled, billing_reasons
        )


def _validate_v2_totals(
    totals: object,
    *,
    reconciled: _ReconciledUsage,
    reasons: list[str],
    billing_reasons: list[str],
) -> None:
    if not isinstance(totals, Mapping):
        reasons.append("FM_USAGE_TOTALS_SCHEMA_INVALID")
        return
    fields = set(totals)
    if (
        not _V2_TOTAL_REQUIRED_FIELDS.issubset(fields)
        or not fields.issubset(_V2_TOTAL_ALLOWED_FIELDS)
    ):
        reasons.append("FM_USAGE_TOTALS_SCHEMA_INVALID")
    _compare_operational_totals(
        totals, reconciled.usage_totals, reasons, fields=_V2_TOTAL_REQUIRED_FIELDS
    )
    declared_billing_fields = fields & _V2_TOTAL_BILLING_FIELDS
    if (
        declared_billing_fields
        and declared_billing_fields != _V2_TOTAL_BILLING_FIELDS
    ):
        billing_reasons.append("FM_USAGE_TOTAL_BILLING_FIELDS_INCOMPLETE")
        return
    if declared_billing_fields:
        if totals.get("billing_records") != reconciled.billing_records:
            billing_reasons.append("FM_USAGE_TOTAL_BILLING_RECORDS_MISMATCH")
        if totals.get("billing_completeness") != reconciled.billing_completeness:
            billing_reasons.append(
                "FM_USAGE_TOTAL_BILLING_COMPLETENESS_MISMATCH"
            )
        _compare_declared_billing_amount(
            totals.get("billing_amount_usd"), reconciled, billing_reasons
        )


def _compare_operational_totals(
    totals: Mapping[str, Any],
    computed: Mapping[str, int],
    reasons: list[str],
    *,
    fields: Sequence[str] | frozenset[str],
) -> None:
    for field_name in fields:
        wanted = computed[field_name]
        if field_name in totals and totals.get(field_name) != wanted:
            reasons.append(f"FM_USAGE_TOTAL_{field_name.upper()}_MISMATCH")


def _compare_declared_billing_amount(
    declared: object,
    reconciled: _ReconciledUsage,
    billing_reasons: list[str],
) -> None:
    if reconciled.billing_completeness == "complete":
        expected = reconciled.billing_amount_usd
        assert expected is not None
        if not _close(declared, expected):
            billing_reasons.append("FM_USAGE_TOTAL_BILLING_AMOUNT_USD_MISMATCH")
    elif declared is not None:
        # A partial sum is not the total cost of the run.
        billing_reasons.append(
            "FM_USAGE_TOTAL_INCOMPLETE_BILLING_MUST_BE_NULL"
        )


def _empty_usage_totals() -> dict[str, int]:
    return {
        "llm_calls": 0,
        "vlm_calls": 0,
        "successful_responses": 0,
        "failed_requests": 0,
        "transport_attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _close(value: object, expected: float) -> bool:
    number = _number(value)
    return number is not None and math.isclose(
        number, expected, rel_tol=0.0, abs_tol=1e-9
    )


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


__all__ = [
    "FM_USAGE_LEDGER_FILENAME",
    "FM_USAGE_LEDGER_KIND",
    "FM_USAGE_LEDGER_LEGACY_SCHEMA_VERSION",
    "FM_USAGE_LEDGER_SCHEMA_VERSION",
    "UsageEvidenceValidation",
    "validate_fm_usage_evidence",
]
