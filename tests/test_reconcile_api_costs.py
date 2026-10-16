from __future__ import annotations

import hashlib
import json
import sys

import scripts.reconcile_api_costs as reconciliation
from scripts.summarize_formal_evaluation import (
    _is_cost_complete,
    _manifest_hash_is_valid,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class _DelayedBillingClient:
    def __init__(self, **_kwargs):
        self.call_records = []

    def reconcile_billing(self, **_kwargs):
        for record in self.call_records:
            if record.get("request_id") == "req_delayed":
                record.update(
                    {
                        "billing_amount_microusd": 123,
                        "billing_amount_usd": 0.000123,
                        "billing_currency": "USD",
                        "billing_lookup_status": "ok",
                    }
                )
        return {"status": "complete", "remaining": 0, "resolved": 1}


def test_delayed_reconciliation_refreshes_every_formal_cost_artifact(
    tmp_path,
    monkeypatch,
):
    usage = {
        "text": {
            "call_count": 2,
            "billed_call_count": 0,
            "unbilled_call_count": 2,
            "billing_amount_microusd": 0,
            "billing_amount_usd": 0.0,
            "calls": [
                {
                    "kind": "remote_text_generation",
                    "role": "task_orchestrator",
                    "request_id": "req_delayed",
                    "requested_model": "model",
                    "resolved_model": "model",
                    "billing_amount_microusd": None,
                },
                {
                    "kind": "remote_text_generation",
                    "role": "task_orchestrator",
                    "request_id": "",
                    "http_status": 502,
                    "billing_amount_microusd": None,
                },
            ],
        },
        "vision": {
            "call_count": 1,
            "billed_call_count": 1,
            "unbilled_call_count": 0,
            "billing_amount_microusd": 50,
            "billing_amount_usd": 0.00005,
            "calls": [
                {
                    "kind": "remote_vision_generation",
                    "role": "vlm_failure_analyzer",
                    "request_id": "req_existing",
                    "requested_model": "model",
                    "resolved_model": "model",
                    "billing_amount_microusd": 50,
                }
            ],
        },
        "roles": {
            "model_name": "model",
            "roles": {
                "task_orchestrator": {
                    "call_count": 2,
                    "billed_call_count": 0,
                    "unbilled_call_count": 2,
                    "billing_amount_microusd": 0,
                },
                "vlm_failure_analyzer": {
                    "call_count": 1,
                    "billed_call_count": 1,
                    "unbilled_call_count": 0,
                    "billing_amount_microusd": 50,
                },
            },
        },
        "billing_reconciliation": {"status": "incomplete", "remaining": 1},
    }
    metrics = {
        "model": {
            "billed_call_count": 1,
            "unbilled_call_count": 2,
            "billing_amount_microusd": 50,
            "billing_amount_usd": 0.00005,
            "billing_reconciliation_status": "incomplete",
        }
    }
    manifest = {
        "evaluation_fingerprint_sha256": "frozen-fingerprint",
        "result_artifact_hashes": {},
        "model_usage": usage,
    }
    _write_json(tmp_path / "model_usage.json", usage)
    _write_json(tmp_path / "metrics.json", metrics)
    _write_json(tmp_path / "billing_reconciliation.json", {"status": "incomplete"})
    _write_json(tmp_path / "final_summary.json", [])
    _write_json(tmp_path / "run_manifest.json", manifest)

    monkeypatch.setattr(
        reconciliation,
        "OpenAICompatibleTextClient",
        _DelayedBillingClient,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_api_costs.py",
            "--output-dir",
            str(tmp_path),
            "--require-complete",
        ],
    )
    reconciliation.main()

    refreshed_usage = json.loads((tmp_path / "model_usage.json").read_text())
    refreshed_metrics = json.loads((tmp_path / "metrics.json").read_text())
    refreshed_manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    ledger = json.loads((tmp_path / "api_call_costs.json").read_text())

    assert refreshed_usage["text"]["billed_call_count"] == 1
    assert refreshed_usage["text"]["unbilled_call_count"] == 0
    assert refreshed_usage["text"]["failed_transport_round_count"] == 1
    assert refreshed_usage["billing_reconciliation"]["status"] == "complete"
    assert refreshed_usage["roles"]["roles"]["task_orchestrator"][
        "billing_amount_microusd"
    ] == 123
    assert refreshed_usage["roles"]["roles"]["task_orchestrator"][
        "unbilled_call_count"
    ] == 0
    assert refreshed_metrics["model"]["billed_call_count"] == 2
    assert refreshed_metrics["model"]["unbilled_call_count"] == 0
    assert refreshed_metrics["model"]["billing_amount_microusd"] == 173
    assert refreshed_metrics["model"]["billing_reconciliation_status"] == "complete"
    assert ledger["summary"]["failed_transport_rounds"] == 1
    assert ledger["summary"]["billing_amount_microusd"] == 173
    assert refreshed_manifest["evaluation_fingerprint_sha256"] == (
        "frozen-fingerprint"
    )
    assert refreshed_manifest["result_artifact_hashes"]["metrics_sha256"] == (
        hashlib.sha256((tmp_path / "metrics.json").read_bytes()).hexdigest()
    )
    assert _manifest_hash_is_valid(refreshed_manifest)
    assert _is_cost_complete(
        metrics=refreshed_metrics,
        manifest=refreshed_manifest,
        output_dir=tmp_path,
    )
