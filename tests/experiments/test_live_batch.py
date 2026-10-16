from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coevolve_experiments import cli as cli_module
from coevolve_experiments.cli import main
from coevolve_experiments.contracts import MemoryCondition, TableId
from coevolve_experiments.live import batch as batch_module
from coevolve_experiments.live.batch import (
    LIVE_BATCH_MANIFEST_FILENAME,
    LIVE_BATCH_MANIFEST_KIND,
    LIVE_BATCH_MANIFEST_SCHEMA_VERSION,
    LiveBatchCell,
    LiveBatchError,
    VerifiedLiveBatch,
    execute_live_batch,
    inspect_live_batch,
    load_live_batch,
    prepare_live_batch,
)
from coevolve_experiments.live.bridge import (
    LIVE_BRIDGE_RECEIPT_FILENAME,
    live_run_reservation_path,
)
from coevolve_experiments.live.contracts import LiveRunReceipt
from coevolve_experiments.live.launch import FROM_SCRATCH_BOOTSTRAP_MISSING_CODE
from coevolve_experiments.live.preflight import LIVE_BRIDGE_UNAVAILABLE_CODE
from coevolve_experiments.serialization import sha256_json


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


_FULL_METHOD_COMPATIBILITY = _digest("full-method-compatibility")
_BASE_METHOD_COMMIT = "a" * 40
_RUNNER_INTEGRATION_COMMIT = "b" * 40


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _loader_fixture(tmp_path: Path, monkeypatch, *, run_count: int = 2):
    plan_path = (tmp_path / "plan.json").resolve()
    plan_path.write_text("{}", encoding="utf-8")
    capability_path = (tmp_path / "capability.json").resolve()
    capability_path.write_text("{}", encoding="utf-8")
    results_root = (tmp_path / "results").resolve()
    batch_root = (tmp_path / "batch").resolve()
    config_root = batch_root / "configs"
    config_root.mkdir(parents=True)
    runs = tuple(
        SimpleNamespace(
            run_id=f"run_{index}",
            output_relpath=f"table/condition/seed_{index:03d}",
            base_method_commit=_BASE_METHOD_COMMIT,
            full_method_compatibility_fingerprint=(
                _FULL_METHOD_COMPATIBILITY
            ),
        )
        for index in range(run_count)
    )
    plan = SimpleNamespace(
        path=plan_path,
        sha256=_digest("plan"),
        suite=SimpleNamespace(new_runs=runs),
    )
    capability = SimpleNamespace(
        manifest_path=capability_path,
        manifest_sha256=_digest("capability"),
        base_method_commit=_BASE_METHOD_COMMIT,
        runner_integration_commit=_RUNNER_INTEGRATION_COMMIT,
        full_method_compatibility=SimpleNamespace(
            fingerprint=_FULL_METHOD_COMPATIBILITY
        ),
    )
    monkeypatch.setattr(batch_module, "_load_execution_plan", lambda _path: plan)
    monkeypatch.setattr(
        batch_module,
        "load_runner_capability",
        lambda *_args, **_kwargs: capability,
    )

    cells = []
    verified_by_path = {}
    for ordinal, run in enumerate(runs):
        config_path = (config_root / f"{ordinal:03d}.json").resolve()
        config_path.write_text("{}", encoding="utf-8")
        output_dir = (results_root / run.output_relpath).resolve()
        verified = SimpleNamespace(
            sha256=_digest(f"config-{ordinal}"),
            config_fingerprint=_digest(f"config-fingerprint-{ordinal}"),
            launch=SimpleNamespace(
                output_dir=output_dir,
                launch_fingerprint=_digest(f"launch-{ordinal}"),
            ),
        )
        verified_by_path[config_path] = verified
        cells.append(
            {
                "ordinal": ordinal,
                "run_id": run.run_id,
                "output_dir": str(output_dir),
                "launch_config_path": str(config_path),
                "launch_config_sha256": verified.sha256,
                "config_fingerprint": verified.config_fingerprint,
                "launch_fingerprint": verified.launch.launch_fingerprint,
            }
        )
    monkeypatch.setattr(
        batch_module,
        "load_subprocess_launch_config",
        lambda path, **_kwargs: verified_by_path[Path(path)],
    )
    fingerprint_payload = {
        "kind": LIVE_BATCH_MANIFEST_KIND,
        "schema_version": LIVE_BATCH_MANIFEST_SCHEMA_VERSION,
        "plan_path": str(plan_path),
        "plan_sha256": plan.sha256,
        "results_root": str(results_root),
        "capability_manifest_path": str(capability_path),
        "capability_manifest_sha256": capability.manifest_sha256,
        "base_method_commit": _BASE_METHOD_COMMIT,
        "runner_integration_commit": _RUNNER_INTEGRATION_COMMIT,
        "full_method_compatibility_fingerprint": (
            _FULL_METHOD_COMPATIBILITY
        ),
        "partition": "full",
        "run_count": len(cells),
        "cells": cells,
    }
    document = {
        **fingerprint_payload,
        "batch_fingerprint": sha256_json(fingerprint_payload),
    }
    manifest_path = batch_root / LIVE_BATCH_MANIFEST_FILENAME
    _write(manifest_path, document)
    return manifest_path, document, plan


def test_batch_loader_rebuilds_every_cell_from_plan_and_config(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _document, plan = _loader_fixture(tmp_path, monkeypatch)

    loaded = load_live_batch(manifest_path)

    assert loaded.plan_path == plan.path
    assert tuple(cell.run_id for cell in loaded.cells) == ("run_0", "run_1")
    assert loaded.manifest_sha256 == sha256(manifest_path.read_bytes()).hexdigest()
    assert loaded.base_method_commit == _BASE_METHOD_COMMIT
    assert loaded.runner_integration_commit == _RUNNER_INTEGRATION_COMMIT


def test_batch_loader_rejects_legacy_schema_even_with_new_fingerprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, document, _plan = _loader_fixture(tmp_path, monkeypatch)
    document["schema_version"] = 4
    fingerprint_payload = dict(document)
    fingerprint_payload.pop("batch_fingerprint")
    document["batch_fingerprint"] = sha256_json(fingerprint_payload)
    _write(manifest_path, document)

    with pytest.raises(LiveBatchError, match="must equal integer 5"):
        load_live_batch(manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("base_method_commit", "0" * 40, "base method commit differs"),
        (
            "runner_integration_commit",
            "0" * 40,
            "runner integration commit differs",
        ),
    ),
)
def test_batch_loader_rejects_commit_identity_tamper_even_with_new_fingerprint(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: str,
    message: str,
) -> None:
    manifest_path, document, _plan = _loader_fixture(tmp_path, monkeypatch)
    document[field] = value
    fingerprint_payload = dict(document)
    fingerprint_payload.pop("batch_fingerprint")
    document["batch_fingerprint"] = sha256_json(fingerprint_payload)
    _write(manifest_path, document)

    with pytest.raises(LiveBatchError, match=message):
        load_live_batch(manifest_path)


def test_batch_loader_rejects_semantic_cell_tamper_even_with_new_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, document, _plan = _loader_fixture(tmp_path, monkeypatch)
    document["cells"][0]["run_id"] = "favorable_replacement"
    fingerprint_payload = dict(document)
    fingerprint_payload.pop("batch_fingerprint")
    document["batch_fingerprint"] = sha256_json(fingerprint_payload)
    _write(manifest_path, document)

    with pytest.raises(LiveBatchError, match="differs from plan run order"):
        load_live_batch(manifest_path)


def test_batch_loader_requires_complete_plan_not_a_selected_subset(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, document, _plan = _loader_fixture(tmp_path, monkeypatch)
    document["cells"].pop()
    document["run_count"] = 1
    fingerprint_payload = dict(document)
    fingerprint_payload.pop("batch_fingerprint")
    document["batch_fingerprint"] = sha256_json(fingerprint_payload)
    _write(manifest_path, document)

    with pytest.raises(LiveBatchError, match="every run in its registered partition"):
        load_live_batch(manifest_path)


def test_batch_partition_selects_complete_table_not_arbitrary_cells() -> None:
    runs = (
        SimpleNamespace(run_id="t2-a", table_id=TableId.TABLE_2),
        SimpleNamespace(run_id="t2-b", table_id=TableId.TABLE_2),
        SimpleNamespace(run_id="t3-a", table_id=TableId.TABLE_3),
        SimpleNamespace(run_id="t4-a", table_id=TableId.TABLE_4),
    )

    selected = batch_module._runs_for_partition(runs, "table2")

    assert tuple(run.run_id for run in selected) == ("t2-a", "t2-b")
    assert batch_module._runs_for_partition(runs, "full") == runs
    with pytest.raises(LiveBatchError, match="partition must be one of"):
        batch_module._runs_for_partition(runs, "seed_200_only")


def test_non_transfer_partition_needs_no_banks_and_rejects_extraneous_bank(
    tmp_path: Path,
) -> None:
    runs = (
        SimpleNamespace(
            strategy_bank=None,
            memory_condition=MemoryCondition.EMPTY,
        ),
    )
    kwargs = {
        "gpt_strategy_bank_path": None,
        "gpt_strategy_provenance_path": None,
        "gemini_strategy_bank_path": None,
        "gemini_strategy_provenance_path": None,
    }

    assert batch_module._transfer_paths_for_runs(runs, **kwargs) == {}

    kwargs["gpt_strategy_bank_path"] = tmp_path / "unused-bank.json"
    with pytest.raises(LiveBatchError, match="not permitted for this partition"):
        batch_module._transfer_paths_for_runs(runs, **kwargs)


def test_transfer_partition_requires_both_frozen_banks() -> None:
    runs = (
        SimpleNamespace(
            strategy_bank=object(),
            memory_condition=MemoryCondition.GPT_TO_GPT,
        ),
        SimpleNamespace(
            strategy_bank=object(),
            memory_condition=MemoryCondition.GEMINI_TO_GPT,
        ),
    )

    with pytest.raises(LiveBatchError, match="GPT strategy bank and provenance"):
        batch_module._transfer_paths_for_runs(
            runs,
            gpt_strategy_bank_path=None,
            gpt_strategy_provenance_path=None,
            gemini_strategy_bank_path=None,
            gemini_strategy_provenance_path=None,
        )


def test_prepare_batch_preflights_then_bootstraps_transfer_once_and_publishes(
    tmp_path: Path, monkeypatch
) -> None:
    plan_path = (tmp_path / "plan.json").resolve()
    plan_path.write_text("{}", encoding="utf-8")
    capability_path = (tmp_path / "capability.json").resolve()
    capability_path.write_text("{}", encoding="utf-8")
    gpt_bank = (tmp_path / "gpt-bank.json").resolve()
    gpt_provenance = (tmp_path / "gpt-provenance.json").resolve()
    gemini_bank = (tmp_path / "gemini-bank.json").resolve()
    gemini_provenance = (tmp_path / "gemini-provenance.json").resolve()
    for path in (gpt_bank, gpt_provenance, gemini_bank, gemini_provenance):
        path.write_text("{}", encoding="utf-8")
    results_root = (tmp_path / "results").resolve()
    batch_root = (tmp_path / "batch").resolve()
    plain = SimpleNamespace(
        run_id="plain",
        base_method_commit=_BASE_METHOD_COMMIT,
        strategy_bank=None,
        memory_condition=MemoryCondition.EMPTY,
        full_method_compatibility_fingerprint=_FULL_METHOD_COMPATIBILITY,
    )
    transfer = SimpleNamespace(
        run_id="transfer",
        base_method_commit=_BASE_METHOD_COMMIT,
        strategy_bank=object(),
        memory_condition=MemoryCondition.GPT_TO_GPT,
        full_method_compatibility_fingerprint=_FULL_METHOD_COMPATIBILITY,
    )
    plan = SimpleNamespace(
        path=plan_path,
        sha256=_digest("plan"),
        suite=SimpleNamespace(new_runs=(plain, transfer)),
    )
    capability = SimpleNamespace(
        manifest_path=capability_path,
        manifest_sha256=_digest("capability"),
        base_method_commit=_BASE_METHOD_COMMIT,
        runner_integration_commit=_RUNNER_INTEGRATION_COMMIT,
        full_method_compatibility=SimpleNamespace(
            fingerprint=_FULL_METHOD_COMPATIBILITY
        ),
    )
    monkeypatch.setattr(batch_module, "_load_execution_plan", lambda _path: plan)
    monkeypatch.setattr(
        batch_module,
        "load_runner_capability",
        lambda *_args, **_kwargs: capability,
    )

    compile_counts = {"plain": 0, "transfer": 0}

    def fake_compile(_plan_path, run, *, results_root, transfer_paths):
        compile_counts[run.run_id] += 1
        blockers = [LIVE_BRIDGE_UNAVAILABLE_CODE]
        if run is plain and compile_counts[run.run_id] == 1:
            blockers.append(FROM_SCRATCH_BOOTSTRAP_MISSING_CODE)
        if run is transfer and compile_counts[run.run_id] == 1:
            blockers.append("TRANSFER_MEMORY_BOOTSTRAP_NOT_PREPARED")
        return SimpleNamespace(
            run_spec=run,
            blockers=tuple(blockers),
            output_dir=(results_root / run.run_id).resolve(),
        )

    monkeypatch.setattr(batch_module, "_compile_for_run", fake_compile)
    prepared = []
    monkeypatch.setattr(
        batch_module,
        "prepare_transfer_from_plan",
        lambda **kwargs: prepared.append(kwargs),
    )
    prepared_from_scratch = []
    monkeypatch.setattr(
        batch_module,
        "_prepare_from_scratch_for_batch",
        lambda run, memory_dir, **kwargs: prepared_from_scratch.append(
            (run, memory_dir, kwargs)
        ),
    )

    def fake_write(launch, *, capability_manifest_path, destination):
        destination.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            sha256=_digest(f"config-{launch.run_spec.run_id}"),
            config_fingerprint=_digest(f"config-fp-{launch.run_spec.run_id}"),
            launch=SimpleNamespace(
                launch_fingerprint=_digest(f"launch-{launch.run_spec.run_id}")
            ),
        )

    monkeypatch.setattr(batch_module, "write_subprocess_launch_config", fake_write)
    sentinel = object()
    monkeypatch.setattr(batch_module, "load_live_batch", lambda _path: sentinel)

    result = prepare_live_batch(
        plan_path=plan_path,
        results_root=results_root,
        capability_manifest_path=capability_path,
        batch_root=batch_root,
        gpt_strategy_bank_path=gpt_bank,
        gpt_strategy_provenance_path=gpt_provenance,
    )

    assert result is sentinel
    assert len(prepared) == 1
    assert prepared[0]["run_id"] == "transfer"
    assert len(prepared_from_scratch) == 1
    assert prepared_from_scratch[0][0] is plain
    assert prepared_from_scratch[0][1] == results_root / "plain" / "memory"
    assert prepared_from_scratch[0][2] == {
        "plan_path": plan.path,
        "plan_sha256": plan.sha256,
    }
    assert compile_counts == {"plain": 2, "transfer": 2}
    assert (batch_root / LIVE_BATCH_MANIFEST_FILENAME).is_file()
    assert len(tuple((batch_root / "configs").glob("*.json"))) == 2


def _verified_batch(tmp_path: Path) -> VerifiedLiveBatch:
    config0 = (tmp_path / "configs" / "0.json").resolve()
    config1 = (tmp_path / "configs" / "1.json").resolve()
    for path in (config0, config1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    cells = tuple(
        LiveBatchCell(
            ordinal=index,
            run_id=f"run_{index}",
            output_dir=(tmp_path / "results" / f"run_{index}").resolve(),
            launch_config_path=config,
            launch_config_sha256=_digest(f"config-{index}"),
            config_fingerprint=_digest(f"config-fp-{index}"),
            launch_fingerprint=_digest(f"launch-{index}"),
        )
        for index, config in enumerate((config0, config1))
    )
    return VerifiedLiveBatch(
        manifest_path=(tmp_path / "batch_manifest.json").resolve(),
        manifest_sha256=_digest("manifest"),
        batch_fingerprint=_digest("batch"),
        plan_path=(tmp_path / "plan.json").resolve(),
        plan_sha256=_digest("plan"),
        results_root=(tmp_path / "results").resolve(),
        capability_manifest_path=(tmp_path / "capability.json").resolve(),
        capability_manifest_sha256=_digest("capability"),
        base_method_commit=_BASE_METHOD_COMMIT,
        runner_integration_commit=_RUNNER_INTEGRATION_COMMIT,
        full_method_compatibility_fingerprint=_FULL_METHOD_COMPATIBILITY,
        partition="full",
        cells=cells,
    )


def _receipt(cell: LiveBatchCell) -> LiveRunReceipt:
    return LiveRunReceipt(
        run_id=cell.run_id,
        output_dir=cell.output_dir,
        run_manifest_path=cell.output_dir / "run_manifest.json",
        metrics_path=cell.output_dir / "metrics.json",
        launch_config_path=cell.launch_config_path,
        bridge_receipt_path=cell.output_dir / LIVE_BRIDGE_RECEIPT_FILENAME,
        launch_config_sha256=cell.launch_config_sha256,
        run_manifest_sha256=_digest("manifest"),
        metrics_sha256=_digest("metrics"),
        bridge_receipt_sha256=_digest("receipt"),
        launch_fingerprint=cell.launch_fingerprint,
        base_method_commit=_BASE_METHOD_COMMIT,
        runner_integration_commit=_RUNNER_INTEGRATION_COMMIT,
        terminal_status="SUCCESS",
        task_success=True,
        protocol_valid=True,
        paper_evidence_eligible=True,
    )


def test_execute_batch_revalidates_completed_cell_and_runs_only_pending(
    tmp_path: Path, monkeypatch
) -> None:
    batch = _verified_batch(tmp_path)
    monkeypatch.setattr(batch_module, "load_live_batch", lambda _path: batch)
    first_receipt = batch.cells[0].output_dir / LIVE_BRIDGE_RECEIPT_FILENAME
    first_receipt.parent.mkdir(parents=True)
    first_receipt.write_text("{}", encoding="utf-8")
    validated = []
    monkeypatch.setattr(
        batch_module,
        "validate_live_bridge_receipt",
        lambda path, **kwargs: validated.append((path, kwargs)) or {"terminal_status": "SUCCESS"},
    )
    launched = []
    monkeypatch.setattr(
        batch_module,
        "launch_live_subprocess",
        lambda path, **_kwargs: launched.append(path) or _receipt(batch.cells[1]),
    )
    monkeypatch.setattr(
        batch_module,
        "inspect_live_batch",
        lambda _path: {"counts": {"completed": 2, "ready": 0, "blocked": 0}},
    )

    result = execute_live_batch(batch.manifest_path)

    assert validated == [
        (batch.cells[0].output_dir, {"expected_run_id": "run_0"})
    ]
    assert launched == [batch.cells[1].launch_config_path]
    assert result["completed_before"] == 1
    assert result["launched_count"] == 1


def test_execute_batch_stops_at_first_failure(tmp_path: Path, monkeypatch) -> None:
    batch = _verified_batch(tmp_path)
    monkeypatch.setattr(batch_module, "load_live_batch", lambda _path: batch)
    monkeypatch.setattr(
        batch_module,
        "launch_live_subprocess",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backend fatal")),
    )

    with pytest.raises(LiveBatchError, match="stopped at run_0"):
        execute_live_batch(batch.manifest_path)


def test_inspect_batch_marks_unattested_partial_output_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    batch = _verified_batch(tmp_path)
    monkeypatch.setattr(batch_module, "load_live_batch", lambda _path: batch)

    def fake_load(path, *, output_may_exist):
        assert output_may_exist is False
        if Path(path) == batch.cells[0].launch_config_path:
            return object()
        raise ValueError("PLANNED_OUTPUT_DIRECTORY_NOT_EMPTY")

    monkeypatch.setattr(batch_module, "load_subprocess_launch_config", fake_load)

    status = inspect_live_batch(batch.manifest_path)

    assert status["counts"] == {"completed": 0, "ready": 1, "blocked": 1}
    assert status["cells"][1]["status"] == "blocked"


def test_inspect_batch_marks_live_run_reservation_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    batch = _verified_batch(tmp_path)
    monkeypatch.setattr(batch_module, "load_live_batch", lambda _path: batch)
    monkeypatch.setattr(
        batch_module,
        "load_subprocess_launch_config",
        lambda *_args, **_kwargs: object(),
    )
    reservation_path = live_run_reservation_path(batch.cells[0].output_dir)
    reservation_path.parent.mkdir(parents=True, exist_ok=True)
    reservation_path.write_text("{}\n", encoding="utf-8")

    status = inspect_live_batch(batch.manifest_path)

    assert status["counts"] == {"completed": 0, "ready": 1, "blocked": 1}
    assert status["cells"][0]["status"] == "blocked"
    assert "reservation exists" in status["cells"][0]["detail"]


def test_prepare_live_batch_cli_forwards_only_fixed_paper_artifacts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    batch = _verified_batch(tmp_path)
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return batch

    monkeypatch.setattr(cli_module, "prepare_live_batch", fake_prepare)
    paths = {
        "plan": tmp_path / "plan.json",
        "results": tmp_path / "results",
        "capability": tmp_path / "capability.json",
        "batch": tmp_path / "batch",
        "gpt_bank": tmp_path / "gpt-bank.json",
        "gpt_provenance": tmp_path / "gpt-provenance.json",
        "gemini_bank": tmp_path / "gemini-bank.json",
        "gemini_provenance": tmp_path / "gemini-provenance.json",
    }

    assert main(
        [
            "prepare-live-batch",
            "--plan",
            str(paths["plan"]),
            "--results-root",
            str(paths["results"]),
            "--capability",
            str(paths["capability"]),
            "--batch-root",
            str(paths["batch"]),
            "--gpt-strategy-bank",
            str(paths["gpt_bank"]),
            "--gpt-strategy-provenance",
            str(paths["gpt_provenance"]),
            "--gemini-strategy-bank",
            str(paths["gemini_bank"]),
            "--gemini-strategy-provenance",
            str(paths["gemini_provenance"]),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["physics_dispatched"] is False
    assert output["run_count"] == 2
    assert captured == {
        "plan_path": paths["plan"],
        "results_root": paths["results"],
        "capability_manifest_path": paths["capability"],
        "batch_root": paths["batch"],
        "gpt_strategy_bank_path": paths["gpt_bank"],
        "gpt_strategy_provenance_path": paths["gpt_provenance"],
        "gemini_strategy_bank_path": paths["gemini_bank"],
        "gemini_strategy_provenance_path": paths["gemini_provenance"],
        "partition": "full",
    }
