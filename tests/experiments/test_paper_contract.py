import json
from pathlib import Path

import pytest

from coevolve_experiments.cli import main
from coevolve_experiments.contracts import SuiteTier, TableId
from coevolve_experiments.paper_contract import build_paper_contract


def test_core_contract_matches_registered_35_run_story() -> None:
    contract = build_paper_contract(SuiteTier.CORE)

    assert contract.seeds == (200, 201, 202, 203, 204)
    assert contract.physical_budget == 80
    assert contract.total_new_runs == 35
    assert contract.total_reused_rows == 3
    assert {table.table_id: table.new_run_count for table in contract.tables} == {
        TableId.TABLE_2: 15,
        TableId.TABLE_3: 10,
        TableId.TABLE_4: 10,
    }
    assert [
        row.condition_id
        for table in contract.tables
        for row in table.rows
        if row.execution == "reuse"
    ] == ["coevolve_full", "base", "empty"]


def test_full_contract_adds_only_coherent_and_altered_topology() -> None:
    core = build_paper_contract("core")
    full = build_paper_contract("full")

    core_rows = {
        (table.table_id, row.condition_id)
        for table in core.tables
        for row in table.rows
    }
    added = {
        (table.table_id, row.condition_id)
        for table in full.tables
        for row in table.rows
    } - core_rows
    assert added == {
        (TableId.TABLE_2, "coherent_adapted"),
        (TableId.TABLE_3, "altered_topology"),
    }
    assert full.total_new_runs == 45
    assert full.total_reused_rows == 3


def test_contract_detects_protocol_row_drift(tmp_path: Path) -> None:
    source = Path("src/coevolve_experiments/protocols")
    for name in ("common.yaml", "table2.yaml", "table3.yaml", "table4.yaml"):
        payload = json.loads((source / name).read_text(encoding="utf-8"))
        if name == "table2.yaml":
            payload["conditions"][0]["label"] = "Silently renamed"
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="registered paper contract"):
        build_paper_contract("core", protocol_root=tmp_path)


def test_main_table_metrics_are_fixed() -> None:
    contract = build_paper_contract("core")
    assert [metric.metric_id for metric in contract.metrics if metric.main_table] == [
        "success_at_80",
        "capped_physical_transactions_at_80",
        "verified_phase_completion",
        "rejected_physical_transactions",
        "foundation_model_calls",
    ]
    assert {
        metric.metric_id for metric in contract.metrics if not metric.main_table
    }.issuperset(
        {
            "accepted_setup_transactions",
            "accepted_progress_transactions",
            "input_tokens",
            "output_tokens",
            "wall_time_s",
            "api_cost_usd",
        }
    )
    assert {
        table.table_id: table.displayed_metrics for table in contract.tables
    } == {
        TableId.TABLE_2: (
            "success_at_80",
            "capped_physical_transactions_at_80",
            "verified_phase_completion",
            "rejected_physical_transactions",
            "foundation_model_calls",
        ),
        TableId.TABLE_3: (
            "success_at_80",
            "capped_physical_transactions_at_80",
            "verified_phase_completion",
            "rejected_physical_transactions",
        ),
        TableId.TABLE_4: (
            "success_at_80",
            "capped_physical_transactions_at_80",
            "rejected_physical_transactions",
            "foundation_model_calls",
        ),
    }
    assert [
        metric.metric_id
        for metric in contract.displayed_metrics_for(TableId.TABLE_3)
    ] == [
        "success_at_80",
        "capped_physical_transactions_at_80",
        "verified_phase_completion",
        "rejected_physical_transactions",
    ]


def test_paper_contract_cli_is_model_and_physics_free(capsys) -> None:
    assert main(["paper-contract", "--tier", "core"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "coevolve_paper_experiment_contract"
    assert payload["contract"]["total_new_runs"] == 35
    assert payload["contract"]["seeds"] == [200, 201, 202, 203, 204]
