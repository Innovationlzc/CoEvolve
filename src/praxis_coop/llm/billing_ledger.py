"""Exact, secret-free per-call API billing ledgers for experiment artifacts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


def build_api_cost_ledger(
    model_usage: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deduplicated ledger without estimating unavailable charges."""

    records: list[dict[str, Any]] = []
    seen_objects: set[int] = set()
    for scope in ("text", "vision"):
        bucket = model_usage.get(scope, {})
        bucket = bucket if isinstance(bucket, Mapping) else {}
        calls = bucket.get("calls", ())
        calls = calls if isinstance(calls, list) else []
        for source in calls:
            if not isinstance(source, dict) or id(source) in seen_objects:
                continue
            seen_objects.add(id(source))
            records.append({"api_scope": scope, **source})
    records.sort(key=lambda item: float(item.get("started_at_unix", 0.0) or 0.0))

    calls: list[dict[str, Any]] = []
    exact_total_microusd = 0
    pending = 0
    completed = 0
    failed_transport_rounds = 0
    for index, record in enumerate(records, start=1):
        request_id = str(record.get("request_id", "")).strip()
        amount = record.get("billing_amount_microusd")
        if request_id and amount is not None:
            amount_microusd: int | None = int(amount)
            charge_status = "exact_provider_billing"
            exact_total_microusd += amount_microusd
            completed += 1
        elif request_id:
            amount_microusd = None
            charge_status = "pending_provider_billing"
            pending += 1
            completed += 1
        else:
            # An exhausted HTTP/transport retry round has no completed model
            # response and no provider request ID to bill or reconcile.
            amount_microusd = 0
            charge_status = "zero_no_completed_model_response"
            failed_transport_rounds += 1
        calls.append(
            {
                "call_index": index,
                "role": str(record.get("role", "")),
                "artifact_id": str(record.get("artifact_id", "")),
                "response_index": record.get("response_index"),
                "api_scope": str(record.get("api_scope", "")),
                "kind": str(record.get("kind", "")),
                "request_id": request_id,
                "requested_model": str(record.get("requested_model", "")),
                "resolved_model": str(record.get("resolved_model", "")),
                "started_at_unix": record.get("started_at_unix"),
                "latency_s": float(record.get("latency_s", 0.0) or 0.0),
                "http_status": record.get("http_status"),
                "provider_fatal": bool(record.get("provider_fatal", False)),
                "transient_provider_outage": bool(
                    record.get("transient_provider_outage", False)
                ),
                "provider_recovery_round": record.get("provider_recovery_round"),
                "input_tokens": int(record.get("input_tokens", 0) or 0),
                "output_tokens": int(record.get("output_tokens", 0) or 0),
                "cached_input_tokens": int(
                    record.get("cached_input_tokens", 0) or 0
                ),
                "image_input_tokens": int(
                    record.get("image_input_tokens", 0) or 0
                ),
                "reasoning_tokens": int(record.get("reasoning_tokens", 0) or 0),
                "charge_status": charge_status,
                "billing_amount_microusd": amount_microusd,
                "billing_amount_usd": (
                    amount_microusd / 1_000_000.0
                    if amount_microusd is not None
                    else None
                ),
                "billing_currency": str(record.get("billing_currency", "")),
                "billing_pricing_version": record.get("billing_pricing_version"),
                "billing_line_items": record.get("billing_line_items", []),
                "billing_missing_price_dimensions": record.get(
                    "billing_missing_price_dimensions", []
                ),
                "billing_lookup_status": str(
                    record.get("billing_lookup_status", "")
                ),
                "billing_lookup_error": str(record.get("billing_lookup_error", "")),
                "transport_failures": record.get("transport_failures", []),
            }
        )
    return {
        "kind": "coevolve_api_call_cost_ledger",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "currency": "USD",
        "experiment": dict(experiment or {}),
        "summary": {
            "recorded_api_rounds": len(calls),
            "completed_model_calls": completed,
            "failed_transport_rounds": failed_transport_rounds,
            "exact_billed_call_count": completed - pending,
            "pending_billing_call_count": pending,
            "billing_complete": pending == 0,
            "billing_amount_microusd": exact_total_microusd,
            "billing_amount_usd": exact_total_microusd / 1_000_000.0,
        },
        "calls": calls,
    }


def write_api_cost_ledger(
    output_dir: str | Path,
    model_usage: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write JSON plus a flat CSV suited to appendix/accounting analysis."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    ledger = build_api_cost_ledger(model_usage, experiment=experiment)
    (target / "api_call_costs.json").write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    columns = (
        "call_index",
        "role",
        "artifact_id",
        "response_index",
        "api_scope",
        "kind",
        "request_id",
        "requested_model",
        "resolved_model",
        "started_at_unix",
        "latency_s",
        "http_status",
        "provider_fatal",
        "transient_provider_outage",
        "provider_recovery_round",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "image_input_tokens",
        "reasoning_tokens",
        "charge_status",
        "billing_amount_microusd",
        "billing_amount_usd",
        "billing_currency",
        "billing_pricing_version",
        "billing_lookup_status",
        "billing_lookup_error",
    )
    with (target / "api_call_costs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ledger["calls"])
    return ledger


__all__ = ["build_api_cost_ledger", "write_api_cost_ledger"]
