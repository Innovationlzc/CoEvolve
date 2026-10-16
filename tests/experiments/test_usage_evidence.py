from __future__ import annotations

import hashlib
import json
from pathlib import Path

from coevolve_experiments.reporting.usage_evidence import (
    FM_USAGE_LEDGER_FILENAME,
    FM_USAGE_LEDGER_KIND,
    FM_USAGE_LEDGER_LEGACY_SCHEMA_VERSION,
    FM_USAGE_LEDGER_SCHEMA_VERSION,
    validate_fm_usage_evidence,
)


RUN_ID = "table2_method_comparison__coevolve_full__seed_200"
CONDITION, ROUTE = "a" * 64, "b" * 64
MODEL, ROUTE_ID = "gpt-5.4-mini", "provider_a-openai-compatible-v1"


def _attempt(
    ordinal: int,
    *,
    status: int | None = 200,
    error_type: str | None = None,
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "outcome": "http_response" if status is not None else "transport_error",
        "http_status": status,
        "error_type": error_type,
    }


def _ledger() -> dict[str, object]:
    calls = [
        {
            "ordinal": 1,
            "request_id": "request-1",
            "role": "policy_generator",
            "modality": "text",
            "requested_model": MODEL,
            "resolved_model": MODEL,
            "outcome": "success",
            "transport_attempts": [
                _attempt(1, status=429),
                _attempt(2, status=200),
            ],
            "response_model": MODEL,
            "input_tokens": 10,
            "output_tokens": 2,
            "billing_amount_usd": 0.01,
        },
        {
            "ordinal": 2,
            "request_id": "request-2",
            "role": "reliable_feedback",
            "modality": "vision",
            "requested_model": MODEL,
            "resolved_model": MODEL,
            "outcome": "success",
            "transport_attempts": [_attempt(1, status=200)],
            "response_model": MODEL,
            "input_tokens": 20,
            "output_tokens": 3,
            "billing_amount_usd": 0.02,
        },
    ]
    return {
        "kind": FM_USAGE_LEDGER_KIND,
        "schema_version": FM_USAGE_LEDGER_SCHEMA_VERSION,
        "producer": {"id": "praxis_coop.fm_backend", "source_sha256": "c" * 64},
        "run_binding": {
            "run_id": RUN_ID,
            "seed": 200,
            "condition_fingerprint": CONDITION,
            "provider_route_id": ROUTE_ID,
            "provider_route_fingerprint": ROUTE,
            "target_model": MODEL,
        },
        "calls": calls,
        "totals": {
            "llm_calls": 1,
            "vlm_calls": 1,
            "successful_responses": 2,
            "failed_requests": 0,
            "transport_attempts": 3,
            "input_tokens": 30,
            "output_tokens": 5,
            "billing_records": 2,
            "billing_amount_usd": 0.03,
            "billing_completeness": "complete",
        },
    }


def _legacy_ledger() -> dict[str, object]:
    value = _ledger()
    value["schema_version"] = FM_USAGE_LEDGER_LEGACY_SCHEMA_VERSION
    value["calls"] = [
        {
            "ordinal": 1,
            "request_id": "request-1",
            "role": "policy_generator",
            "modality": "text",
            "requested_model": MODEL,
            "resolved_model": MODEL,
            "response_model": MODEL,
            "http_status": 200,
            "input_tokens": 10,
            "output_tokens": 2,
            "billing_amount_usd": 0.01,
        },
        {
            "ordinal": 2,
            "request_id": "request-2",
            "role": "reliable_feedback",
            "modality": "vision",
            "requested_model": MODEL,
            "resolved_model": MODEL,
            "response_model": MODEL,
            "http_status": 200,
            "input_tokens": 20,
            "output_tokens": 3,
            "billing_amount_usd": 0.02,
        },
    ]
    value["totals"] = {
        "llm_calls": 1,
        "vlm_calls": 1,
        "input_tokens": 30,
        "output_tokens": 5,
        "billing_amount_usd": 0.03,
    }
    return value


def _write(root: Path, ledger: dict[str, object]) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(ledger, sort_keys=True, indent=2) + "\n").encode()
    (root / FM_USAGE_LEDGER_FILENAME).write_bytes(raw)
    return {
        "result_artifact_hashes": {
            "fm_usage_ledger_sha256": hashlib.sha256(raw).hexdigest()
        }
    }


def _validate(
    root: Path,
    manifest: dict[str, object],
    *,
    required=(),
    llm_calls: int = 1,
    vlm_calls: int = 1,
    cost: float | None = 0.03,
    billing_status: str | None = "complete",
    schema_version: int | None = FM_USAGE_LEDGER_SCHEMA_VERSION,
):
    return validate_fm_usage_evidence(
        root,
        manifest=manifest,
        expected_run_id=RUN_ID,
        expected_seed=200,
        expected_condition_fingerprint=CONDITION,
        expected_provider_route_id=ROUTE_ID,
        expected_provider_route_fingerprint=ROUTE,
        expected_target_model=MODEL,
        expected_llm_calls=llm_calls,
        expected_vlm_calls=vlm_calls,
        expected_input_tokens=30,
        expected_output_tokens=5,
        expected_billing_amount_usd=cost,
        expected_billing_reconciliation_status=billing_status,
        expected_schema_version=schema_version,
        required_request_ids=required,
    )


def test_usage_recomputes_logical_calls_separately_from_transport_attempts(
    tmp_path: Path,
) -> None:
    manifest = _write(tmp_path, _ledger())
    result = _validate(tmp_path, manifest, required=("request-1",))

    assert result.valid
    assert result.request_ids == ("request-1", "request-2")
    assert result.usage_totals == {
        "llm_calls": 1,
        "vlm_calls": 1,
        "successful_responses": 2,
        "failed_requests": 0,
        "transport_attempts": 3,
        "input_tokens": 30,
        "output_tokens": 5,
    }
    assert result.billing_complete
    assert result.billing_amount_usd == 0.03


def test_failed_logical_request_is_recorded_without_fake_response_usage(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    calls = ledger["calls"]
    assert isinstance(calls, list)
    calls.append(
        {
            "ordinal": 3,
            "request_id": "request-3",
            "role": "policy_generator",
            "modality": "text",
            "requested_model": MODEL,
            "resolved_model": MODEL,
            "outcome": "failure",
            "transport_attempts": [
                _attempt(1, status=None, error_type="timeout"),
            ],
            "response_model": None,
            "input_tokens": None,
            "output_tokens": None,
            "billing_amount_usd": None,
        }
    )
    totals = ledger["totals"]
    assert isinstance(totals, dict)
    totals.update(
        {
            "llm_calls": 2,
            "failed_requests": 1,
            "transport_attempts": 4,
            "billing_records": 2,
            "billing_amount_usd": None,
            "billing_completeness": "partial",
        }
    )
    manifest = _write(tmp_path, ledger)

    result = _validate(
        tmp_path,
        manifest,
        llm_calls=2,
        cost=None,
        billing_status="partial",
    )

    assert result.valid
    assert result.usage_totals["failed_requests"] == 1
    assert result.usage_totals["transport_attempts"] == 4
    assert result.billing_amount_usd is None
    assert result.billing_completeness == "partial"
    assert not result.billing_complete


def test_missing_billing_is_not_protocol_invalid_or_zero(tmp_path: Path) -> None:
    ledger = _ledger()
    for call in ledger["calls"]:  # type: ignore[union-attr]
        call.pop("billing_amount_usd")
    totals = ledger["totals"]
    assert isinstance(totals, dict)
    totals.update(
        {
            "billing_records": 0,
            "billing_amount_usd": None,
            "billing_completeness": "unavailable",
        }
    )
    manifest = _write(tmp_path, ledger)

    result = _validate(
        tmp_path,
        manifest,
        cost=None,
        billing_status="unavailable",
    )

    assert result.valid
    assert result.billing_reasons == ()
    assert result.billing_amount_usd is None
    assert not result.billing_complete


def test_billing_mismatch_is_separate_from_protocol_validity(tmp_path: Path) -> None:
    ledger = _ledger()
    totals = ledger["totals"]
    assert isinstance(totals, dict)
    totals["billing_amount_usd"] = 99.0
    manifest = _write(tmp_path, ledger)

    result = _validate(tmp_path, manifest)

    assert result.valid
    assert "FM_USAGE_TOTAL_BILLING_AMOUNT_USD_MISMATCH" in result.billing_reasons
    assert result.billing_amount_usd is None
    assert not result.billing_complete


def test_caller_totals_cannot_override_per_call_usage(tmp_path: Path) -> None:
    ledger = _ledger()
    ledger["totals"]["input_tokens"] = 29  # type: ignore[index]
    manifest = _write(tmp_path, ledger)
    result = _validate(tmp_path, manifest)
    assert "FM_USAGE_TOTAL_INPUT_TOKENS_MISMATCH" in result.invalid_reasons


def test_usage_must_match_metrics_and_stall_model_requests(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _ledger())
    missing = _validate(tmp_path, manifest, required=("request-3",))
    assert "FM_USAGE_STALL_MODEL_OUTPUT_REQUEST_MISSING" in missing.invalid_reasons
    mismatched = _validate(tmp_path, manifest, llm_calls=2)
    assert "FM_USAGE_METRICS_LLM_CALLS_MISMATCH" in mismatched.invalid_reasons


def test_success_requires_model_and_tokens_but_failed_request_does_not(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    ledger["calls"][0]["response_model"] = None  # type: ignore[index]
    ledger["calls"][0]["input_tokens"] = None  # type: ignore[index]
    manifest = _write(tmp_path, ledger)

    result = _validate(tmp_path, manifest)

    assert "FM_USAGE_CALL_RESPONSE_MODEL_MISSING_OR_INVALID" in result.invalid_reasons
    assert "FM_USAGE_CALL_INPUT_TOKENS_INVALID" in result.invalid_reasons


def test_legacy_success_only_v1_ledger_remains_readable(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _legacy_ledger())
    result = _validate(tmp_path, manifest, schema_version=None)

    assert result.valid
    assert result.schema_version == FM_USAGE_LEDGER_LEGACY_SCHEMA_VERSION
    assert result.usage_totals["transport_attempts"] == 2
    assert result.billing_complete


def test_new_run_schema_pin_rejects_legacy_v1_ledger(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _legacy_ledger())

    result = _validate(tmp_path, manifest)

    assert "FM_USAGE_LEDGER_SCHEMA_VERSION_MISMATCH" in result.invalid_reasons


def test_usage_ledger_is_manifest_pinned(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _ledger())
    ledger = _ledger()
    ledger["calls"][0]["input_tokens"] = 999  # type: ignore[index]
    (tmp_path / FM_USAGE_LEDGER_FILENAME).write_text(
        json.dumps(ledger), encoding="utf-8"
    )
    assert "FM_USAGE_LEDGER_HASH_MISMATCH" in _validate(
        tmp_path, manifest
    ).invalid_reasons
