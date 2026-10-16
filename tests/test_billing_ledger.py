import json

from praxis_coop.llm.billing_ledger import (
    build_api_cost_ledger,
    write_api_cost_ledger,
)
from praxis_coop.llm.openai_compatible_client import OpenAICompatibleTextClient
from scripts.run_cap_box_carry_evolution import _usage_summary


def test_api_cost_ledger_records_exact_pending_and_failed_transport(tmp_path):
    usage = {
        "text": {
            "calls": [
                {
                    "role": "coder",
                    "artifact_id": "attempt_01_sync_team",
                    "response_index": 0,
                    "kind": "remote_text_generation",
                    "request_id": "req_exact",
                    "requested_model": "model",
                    "resolved_model": "model",
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "billing_amount_microusd": 321,
                    "billing_currency": "USD",
                },
                {
                    "role": "task_orchestrator",
                    "artifact_id": "attempt_02",
                    "response_index": 0,
                    "kind": "remote_text_generation",
                    "request_id": "req_pending",
                    "billing_amount_microusd": None,
                },
                {
                    "role": "task_orchestrator",
                    "artifact_id": "attempt_02",
                    "response_index": 0,
                    "kind": "remote_text_generation",
                    "request_id": "",
                    "http_status": 502,
                    "provider_fatal": False,
                    "transient_provider_outage": True,
                },
            ]
        },
        "vision": {"calls": []},
    }

    ledger = build_api_cost_ledger(usage, experiment={"seed": 0})

    assert ledger["experiment"]["seed"] == 0
    assert ledger["summary"] == {
        "recorded_api_rounds": 3,
        "completed_model_calls": 2,
        "failed_transport_rounds": 1,
        "exact_billed_call_count": 1,
        "pending_billing_call_count": 1,
        "billing_complete": False,
        "billing_amount_microusd": 321,
        "billing_amount_usd": 0.000321,
    }
    assert ledger["calls"][0]["role"] == "coder"
    assert ledger["calls"][1]["charge_status"] == "pending_provider_billing"
    assert ledger["calls"][2]["charge_status"] == (
        "zero_no_completed_model_response"
    )
    assert ledger["calls"][2]["billing_amount_usd"] == 0.0

    written = write_api_cost_ledger(tmp_path, usage, experiment={"seed": 0})
    assert written["summary"] == ledger["summary"]
    assert json.loads(
        (tmp_path / "api_call_costs.json").read_text(encoding="utf-8")
    )["calls"][0]["request_id"] == "req_exact"
    assert (tmp_path / "api_call_costs.csv").read_text(encoding="utf-8").splitlines()[
        0
    ].startswith("call_index,role,artifact_id")


def test_usage_summaries_do_not_count_failed_transport_as_unbilled_call():
    client = OpenAICompatibleTextClient(model_name="model")
    client.call_records.extend(
        [
            {
                "kind": "remote_text_generation",
                "request_id": "req_pending",
                "billing_amount_microusd": None,
            },
            {
                "kind": "remote_text_generation",
                "request_id": "",
                "http_status": 502,
                "provider_fatal": True,
                "billing_amount_microusd": None,
            },
        ]
    )

    client_summary = client.usage_summary()
    filtered_summary = _usage_summary(
        client,
        kinds={"remote_text_generation"},
    )

    for summary in (client_summary, filtered_summary):
        assert summary["billed_call_count"] == 0
        assert summary["unbilled_call_count"] == 1
        assert summary["failed_transport_round_count"] == 1
