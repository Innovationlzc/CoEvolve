#!/usr/bin/env python3
"""Post-terminal exact Provider billing reconciliation for one experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.llm.billing_ledger import (  # noqa: E402
    build_api_cost_ledger,
    write_api_cost_ledger,
)
from praxis_coop.llm.openai_compatible_client import (  # noqa: E402
    OpenAICompatibleTextClient,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="COEVOLVE_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("COEVOLVE_BASE_URL", ""))
    parser.add_argument("--usage-path-template", default="",
                        help="Optional same-origin accounting path containing {request_id}")
    parser.add_argument("--total-timeout-s", type=float, default=1800.0)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    usage_path = args.output_dir / "model_usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    calls = _unique_calls(usage)
    model_name = next(
        (
            str(item.get("requested_model", "")).strip()
            for item in calls
            if str(item.get("requested_model", "")).strip()
        ),
        "",
    )
    client = OpenAICompatibleTextClient(
        model_name=model_name,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        usage_path_template=args.usage_path_template,
    )
    client.call_records.extend(calls)
    provider_report = client.reconcile_billing(
        max_rounds=60,
        initial_backoff_s=5.0,
        per_request_delay_s=0.05,
        total_timeout_s=max(0.1, args.total_timeout_s),
        lookup_timeout_s=min(10.0, max(0.1, args.total_timeout_s)),
    )
    _refresh_usage_billing_summaries(usage)
    preview = build_api_cost_ledger(
        usage,
        experiment=_experiment_metadata(args.output_dir),
    )
    canonical_report = _canonical_reconciliation_report(
        provider_report,
        ledger_summary=preview["summary"],
        model_name=model_name,
        total_timeout_s=args.total_timeout_s,
    )
    usage["billing_reconciliation"] = canonical_report
    _write_json(args.output_dir / "billing_reconciliation.json", canonical_report)
    _write_json(usage_path, usage)

    experiment = _experiment_metadata(args.output_dir)
    ledger = write_api_cost_ledger(
        args.output_dir,
        usage,
        experiment=experiment,
    )
    post_report = {
        **canonical_report,
        "ledger_billing_complete": ledger["summary"]["billing_complete"],
        "billing_amount_microusd": ledger["summary"][
            "billing_amount_microusd"
        ],
        "billing_amount_usd": ledger["summary"]["billing_amount_usd"],
    }
    _write_json(
        args.output_dir / "billing_post_reconciliation.json",
        post_report,
    )
    _refresh_metrics(args.output_dir, usage, ledger)
    _reseal_manifest(args.output_dir, usage)
    print(json.dumps(post_report, ensure_ascii=False), flush=True)
    if args.require_complete and not ledger["summary"]["billing_complete"]:
        raise SystemExit(2)


def _unique_calls(usage: dict) -> list[dict]:
    calls: list[dict] = []
    seen_request_ids: set[str] = set()
    for scope in ("text", "vision"):
        bucket = usage.get(scope, {})
        bucket = bucket if isinstance(bucket, dict) else {}
        for record in bucket.get("calls", ()):
            if not isinstance(record, dict):
                continue
            request_id = str(record.get("request_id", "")).strip()
            if request_id and request_id in seen_request_ids:
                continue
            if request_id:
                seen_request_ids.add(request_id)
            calls.append(record)
    return calls


def _refresh_usage_billing_summaries(usage: dict) -> None:
    """Recompute serialized usage totals after call records are reconciled."""

    all_records: list[dict] = []
    for scope in ("text", "vision"):
        bucket = usage.get(scope, {})
        if not isinstance(bucket, dict):
            continue
        records = [
            item for item in bucket.get("calls", ()) if isinstance(item, dict)
        ]
        all_records.extend(records)
        completed = [
            item for item in records if str(item.get("request_id", "")).strip()
        ]
        billed = [
            item
            for item in completed
            if item.get("billing_amount_microusd") is not None
        ]
        amount_microusd = sum(
            int(item.get("billing_amount_microusd", 0) or 0) for item in billed
        )
        bucket.update(
            {
                "call_count": len(records),
                "billed_call_count": len(billed),
                "unbilled_call_count": len(completed) - len(billed),
                "failed_transport_round_count": len(records) - len(completed),
                "billing_amount_microusd": amount_microusd,
                "billing_amount_usd": amount_microusd / 1_000_000.0,
            }
        )

    role_usage = usage.get("roles", {})
    role_usage = role_usage if isinstance(role_usage, dict) else {}
    roles = role_usage.get("roles", {})
    roles = roles if isinstance(roles, dict) else {}
    for role, bucket in roles.items():
        if not isinstance(bucket, dict):
            continue
        records = [item for item in all_records if item.get("role") == role]
        completed = [
            item for item in records if str(item.get("request_id", "")).strip()
        ]
        billed = [
            item
            for item in completed
            if item.get("billing_amount_microusd") is not None
        ]
        amount_microusd = sum(
            int(item.get("billing_amount_microusd", 0) or 0) for item in billed
        )
        bucket.update(
            {
                "billed_call_count": len(billed),
                "unbilled_call_count": len(completed) - len(billed),
                "failed_transport_round_count": len(records) - len(completed),
                "billing_amount_microusd": amount_microusd,
                "billing_amount_usd": amount_microusd / 1_000_000.0,
            }
        )


def _canonical_reconciliation_report(
    provider_report: dict,
    *,
    ledger_summary: dict,
    model_name: str,
    total_timeout_s: float,
) -> dict:
    pending = int(ledger_summary.get("pending_billing_call_count", 0) or 0)
    complete = bool(ledger_summary.get("billing_complete", False))
    return {
        "status": "complete" if complete else "incomplete",
        "remaining": pending,
        "total_timeout_s": float(total_timeout_s),
        "source": "post_terminal_provider_request_reconciliation",
        "clients": [
            {
                "model_name": str(model_name),
                **dict(provider_report),
            }
        ],
        "ledger_billing_complete": complete,
        "billing_amount_microusd": int(
            ledger_summary.get("billing_amount_microusd", 0) or 0
        ),
        "billing_amount_usd": float(
            ledger_summary.get("billing_amount_usd", 0.0) or 0.0
        ),
    }


def _refresh_metrics(output_dir: Path, usage: dict, ledger: dict) -> None:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        return
    model = metrics.get("model", {})
    model = model if isinstance(model, dict) else {}
    summary = ledger.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    text_usage = usage.get("text", {})
    text_usage = text_usage if isinstance(text_usage, dict) else {}
    vision_usage = usage.get("vision", {})
    vision_usage = vision_usage if isinstance(vision_usage, dict) else {}
    text_calls = text_usage.get("calls", ())
    vision_calls = vision_usage.get("calls", ())
    model.update(
        {
            "llm_calls": len(text_calls) if isinstance(text_calls, list) else 0,
            "vlm_calls": len(vision_calls) if isinstance(vision_calls, list) else 0,
            "billed_call_count": int(
                summary.get("exact_billed_call_count", 0) or 0
            ),
            "unbilled_call_count": int(
                summary.get("pending_billing_call_count", 0) or 0
            ),
            "failed_transport_round_count": int(
                summary.get("failed_transport_rounds", 0) or 0
            ),
            "billing_amount_microusd": int(
                summary.get("billing_amount_microusd", 0) or 0
            ),
            "billing_amount_usd": float(
                summary.get("billing_amount_usd", 0.0) or 0.0
            ),
            "billing_reconciliation_status": str(
                usage.get("billing_reconciliation", {}).get("status", "")
            ),
        }
    )
    metrics["model"] = model
    _write_json(metrics_path, metrics)


def _reseal_manifest(output_dir: Path, usage: dict) -> None:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        return
    result_hashes = manifest.get("result_artifact_hashes", {})
    result_hashes = result_hashes if isinstance(result_hashes, dict) else {}
    for key, name in (
        ("billing_reconciliation_sha256", "billing_reconciliation.json"),
        ("billing_post_reconciliation_sha256", "billing_post_reconciliation.json"),
        ("api_call_costs_json_sha256", "api_call_costs.json"),
        ("api_call_costs_csv_sha256", "api_call_costs.csv"),
        ("model_usage_sha256", "model_usage.json"),
        ("metrics_sha256", "metrics.json"),
        ("final_summary_sha256", "final_summary.json"),
    ):
        result_hashes[key] = _sha256_file(output_dir / name)
    manifest["result_artifact_hashes"] = result_hashes
    manifest["model_usage"] = usage
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _json_sha256(manifest)
    _write_json(manifest_path, manifest, sort_keys=True)


def _write_json(path: Path, payload: dict, *, sort_keys: bool = False) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=sort_keys,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _experiment_metadata(output_dir: Path) -> dict:
    manifest_path = output_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    protocol = manifest.get("protocol", {})
    protocol = protocol if isinstance(protocol, dict) else {}
    model = manifest.get("model", {})
    model = model if isinstance(model, dict) else {}
    return {
        "protocol_id": str(protocol.get("protocol_id", "")),
        "run_index": manifest.get("run_index"),
        "seed": manifest.get("seed"),
        "model": str(model.get("requested", "")),
        "output_dir": str(output_dir),
        "terminal_status": str(manifest.get("status", "")),
    }


if __name__ == "__main__":
    if not os.environ.get("COEVOLVE_API_KEY", "").strip():
        # The client reports a structured missing-credential result; this warning
        # keeps the command's operational cause obvious without exposing secrets.
        print("[billing] COEVOLVE_API_KEY is not set", file=sys.stderr, flush=True)
    main()
