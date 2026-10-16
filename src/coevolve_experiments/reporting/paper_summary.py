"""Plan-bound, reference-only paper summaries for Tables 2--4.

The generic :mod:`summarize` module is useful for reconciling arbitrary run
directories, but a paper table has a stronger contract: every registered row
must be present, every row has the frozen five-seed denominator, and reused
rows must resolve to their audited source records without copying artifacts.
This module implements that final join.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contracts import (
    ExperimentSuite,
    ReuseSpec,
    RunSpec,
    Table1SourceCohort,
    TableId,
)
from ..paper_contract import RegisteredRow, build_paper_contract
from ..scenes import load_scene
from ..serialization import to_primitive
from .collect import RunRecord
from .reuse import alias_reused_condition
from .summarize import build_appendix_tables, summarize_condition
from .table1_source import (
    Table1CohortAudit,
    audit_table1_source_cohort,
)


PAPER_SUMMARY_KIND = "coevolve_tables_2_4_paper_summary"
PAPER_SUMMARY_SCHEMA_VERSION = 1


class PaperSummaryError(RuntimeError):
    """The registered Tables 2--4 summary cannot be resolved safely."""


class IncompletePaperSummaryError(PaperSummaryError):
    """Raised when a caller requires all registered five-seed rows."""


def audit_external_reuse_source(
    suite: ExperimentSuite,
    source_manifest_path: str | Path,
    *,
    parent_manifest_path: str | Path | None = None,
) -> Table1CohortAudit:
    """Audit the one external source selected by the frozen suite.

    The current paper contract selects exact-V83 GPT for both external rows.
    The parent argument is nevertheless explicit for a provider-companion
    source: such a source is admissible only after its exact GPT parent has
    independently passed the source auditor.
    """

    source_path = _absolute_regular_path(
        source_manifest_path, label="Table-1 source aggregate manifest"
    )
    external = tuple(reuse for reuse in suite.reuses if reuse.source_selector.external)
    if not external:
        raise PaperSummaryError("suite has no external reuse source")
    selector_identities = {_selector_identity(reuse) for reuse in external}
    if len(selector_identities) != 1:
        raise PaperSummaryError(
            "paper summary supports exactly one frozen external source identity"
        )
    selector = external[0].source_selector
    source = _locked_source_for_selector(suite, selector)

    parent: Table1CohortAudit | None = None
    if source == suite.protocol_lock.table1_gemini_source:
        if parent_manifest_path is None:
            raise PaperSummaryError(
                "provider-companion Table-1 source requires "
                "--table1-parent-manifest"
            )
        parent_path = _absolute_regular_path(
            parent_manifest_path, label="Table-1 parent aggregate manifest"
        )
        parent = audit_table1_source_cohort(
            parent_path,
            suite.protocol_lock.table1_gpt_source,
        )
    elif parent_manifest_path is not None:
        raise PaperSummaryError(
            "literal exact-V83 GPT source must not declare a parent manifest"
        )

    audit = audit_table1_source_cohort(source_path, source, parent=parent)
    _require_audit_matches_external_reuses(audit, external)
    return audit


def summarize_paper_tables(
    suite: ExperimentSuite,
    new_records_by_run_id: Mapping[str, RunRecord],
    *,
    table1_source_audit: Table1CohortAudit,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Resolve all registered new and reused rows into one paper document.

    ``new_records_by_run_id`` must contain exactly one collector result for
    every planned *new* cell, including not-yet-started cells.  Missing or
    invalid cells therefore remain visible with their reason while all main
    table statistics stay null until the exact seeds 200--204 are valid.
    """

    contract = build_paper_contract(suite.tier)
    _require_exact_run_map(suite, new_records_by_run_id)
    external_reuses = tuple(
        reuse for reuse in suite.reuses if reuse.source_selector.external
    )
    _require_audit_matches_external_reuses(table1_source_audit, external_reuses)

    reuse_by_target = {
        (reuse.target_table, _reuse_condition_id(reuse)): reuse
        for reuse in suite.reuses
    }
    new_specs_by_row: dict[tuple[TableId, str], list[RunSpec]] = {}
    for run in suite.new_runs:
        new_specs_by_row.setdefault(
            (run.table_id, _run_condition_id(run)), []
        ).append(run)

    rows_by_table: dict[TableId, list[dict[str, Any]]] = {
        table.table_id: [] for table in contract.tables
    }
    table3_mirrored_records: tuple[RunRecord, ...] | None = None
    table3_mirrored_invalid_bindings: tuple[tuple[int, RunRecord], ...] = ()
    reuse_index: list[dict[str, Any]] = []

    for table in contract.tables:
        for registered in table.rows:
            key = (registered.table_id, registered.condition_id)
            if registered.execution == "new":
                specs = tuple(new_specs_by_row.get(key, ()))
                row_records = tuple(
                    new_records_by_run_id[spec.run_id] for spec in specs
                )
                row = _summarize_new_row(
                    registered,
                    specs,
                    row_records,
                    expected_seeds=contract.seeds,
                    physical_budget=contract.physical_budget,
                )
                if key == (TableId.TABLE_3, "mirrored"):
                    table3_mirrored_records = row_records
                    table3_mirrored_invalid_bindings = tuple(
                        (spec.seed, record)
                        for spec, record in zip(specs, row_records)
                        if not record.protocol_valid
                    )
            else:
                try:
                    reuse = reuse_by_target[key]
                except KeyError as exc:
                    raise PaperSummaryError(
                        f"registered reuse row has no ReuseSpec: {key}"
                    ) from exc
                if reuse.source_selector.external:
                    source_records = _table1_audit_records(
                        reuse, table1_source_audit
                    )
                    aliased = alias_reused_condition(
                        reuse,
                        source_records,
                        observed_hashes={
                            "result_manifest": table1_source_audit.manifest_sha256
                        },
                    )
                    invalid_source_records: tuple[RunRecord, ...] = ()
                    source_manifest = {
                        "path": table1_source_audit.manifest_path,
                        "sha256": table1_source_audit.manifest_sha256,
                        "internal_sha256": (
                            table1_source_audit.manifest_internal_sha256
                        ),
                    }
                else:
                    if table3_mirrored_records is None:
                        raise PaperSummaryError(
                            "Table-4 Empty must resolve after Table-3 Mirrored"
                        )
                    aliased = alias_reused_condition(
                        reuse,
                        table3_mirrored_records,
                        require_complete=False,
                    )
                    invalid_source_records = tuple(
                        record
                        for record in table3_mirrored_records
                        if not record.protocol_valid
                    )
                    invalid_seed_bindings = table3_mirrored_invalid_bindings
                    source_manifest = None
                if reuse.source_selector.external:
                    invalid_seed_bindings = ()
                row = _summarize_reused_row(
                    registered,
                    reuse,
                    aliased,
                    invalid_source_records=invalid_source_records,
                    invalid_seed_bindings=invalid_seed_bindings,
                    expected_seeds=contract.seeds,
                    physical_budget=contract.physical_budget,
                    source_manifest=source_manifest,
                )
                reuse_index.append(dict(row["source_reference"]))
            rows_by_table[table.table_id].append(row)

    tables: list[dict[str, Any]] = []
    incomplete_rows: list[str] = []
    for registered_table in contract.tables:
        rows = rows_by_table[registered_table.table_id]
        for row in rows:
            if not row["complete"]:
                incomplete_rows.append(
                    f"{registered_table.table_id.value}/{row['condition']}"
                )
        tables.append(
            {
                "table_id": registered_table.table_id.value,
                "research_question": registered_table.research_question,
                "complete": all(row["complete"] for row in rows),
                "rows": rows,
            }
        )

    if require_complete and incomplete_rows:
        raise IncompletePaperSummaryError(
            "paper summary is incomplete: " + ", ".join(incomplete_rows)
        )

    new_records = tuple(
        new_records_by_run_id[run.run_id] for run in suite.new_runs
    )
    appendix = build_appendix_tables(new_records)
    appendix["A4_reference_only_reuse_index"] = reuse_index
    return {
        "kind": PAPER_SUMMARY_KIND,
        "schema_version": PAPER_SUMMARY_SCHEMA_VERSION,
        "tier": suite.tier.value,
        "protocol_id": suite.protocol_lock.protocol_id,
        "expected_seeds": list(contract.seeds),
        "physical_budget": contract.physical_budget,
        "complete": not incomplete_rows,
        "incomplete_rows": incomplete_rows,
        "new_run_cells": {
            "expected": len(suite.new_runs),
            "protocol_valid": sum(
                record.protocol_valid for record in new_records
            ),
        },
        "reuse_policy": {
            "mode": "reference_only",
            "result_directories_copied": False,
            "registered_reuse_rows": len(suite.reuses),
        },
        "tables": tables,
        "appendix": appendix,
    }


def _summarize_new_row(
    registered: RegisteredRow,
    specs: tuple[RunSpec, ...],
    records: tuple[RunRecord, ...],
    *,
    expected_seeds: Iterable[int],
    physical_budget: int,
) -> dict[str, Any]:
    seeds = tuple(expected_seeds)
    if len(specs) != len(seeds) or {spec.seed for spec in specs} != set(seeds):
        raise PaperSummaryError(
            f"new row {registered.table_id.value}/{registered.condition_id} "
            "does not contain the exact registered five seeds"
        )
    for spec, record in zip(specs, records):
        _require_spec_matches_registered_row(spec, registered)
        if record.protocol_valid:
            _require_record_matches_spec(record, spec)
    valid = tuple(record for record in records if record.protocol_valid)
    summary = summarize_condition(
        valid,
        expected_runs=len(seeds),
        expected_seeds=seeds,
        physical_budget=physical_budget,
    )
    return _decorate_row(
        summary,
        registered,
        records=records,
        expected_seeds=seeds,
        execution="new",
        planned_identity=_planned_identity(specs[0]),
        source_reference=None,
        invalid_seed_bindings=tuple(
            (spec.seed, record)
            for spec, record in zip(specs, records)
            if not record.protocol_valid
        ),
    )


def _summarize_reused_row(
    registered: RegisteredRow,
    reuse: ReuseSpec,
    aliased: tuple[RunRecord, ...],
    *,
    invalid_source_records: tuple[RunRecord, ...],
    invalid_seed_bindings: tuple[tuple[int, RunRecord], ...],
    expected_seeds: Iterable[int],
    physical_budget: int,
    source_manifest: Mapping[str, str] | None,
) -> dict[str, Any]:
    seeds = tuple(expected_seeds)
    summary = summarize_condition(
        aliased,
        expected_runs=len(seeds),
        expected_seeds=seeds,
        physical_budget=physical_budget,
    )
    source_reference = {
        "reuse_id": reuse.reuse_id,
        "target_table": reuse.target_table.value,
        "target_condition": registered.condition_id,
        "mode": reuse.mode.value,
        "copies_results": reuse.copies_results,
        "source_selector": to_primitive(reuse.source_selector),
        "expected_hashes": {
            item.name: item.sha256 for item in reuse.expected_hashes
        },
        "source_manifest": dict(source_manifest) if source_manifest else None,
        "aliased_valid_seeds": sorted(
            record.seed for record in aliased if record.seed is not None
        ),
        "source_run_dirs": [record.run_dir for record in aliased],
    }
    return _decorate_row(
        summary,
        registered,
        records=(*aliased, *invalid_source_records),
        expected_seeds=seeds,
        execution="reference_only",
        planned_identity={
            "source_selector": to_primitive(reuse.source_selector),
            "expected_hashes": source_reference["expected_hashes"],
        },
        source_reference=source_reference,
        invalid_seed_bindings=invalid_seed_bindings,
    )


def _decorate_row(
    summary: Mapping[str, Any],
    registered: RegisteredRow,
    *,
    records: tuple[RunRecord, ...],
    expected_seeds: tuple[int, ...],
    execution: str,
    planned_identity: Mapping[str, Any],
    source_reference: Mapping[str, Any] | None,
    invalid_seed_bindings: tuple[tuple[int, RunRecord], ...],
) -> dict[str, Any]:
    result = dict(summary)
    valid_seeds = {
        record.seed
        for record in records
        if record.protocol_valid and record.seed is not None
    }
    invalid = tuple(record for record in records if not record.protocol_valid)
    if len(invalid_seed_bindings) != len(invalid):
        raise PaperSummaryError("invalid record bindings do not match invalid records")
    result.update(
        {
            "table": registered.table_id.value,
            "condition": registered.condition_id,
            "label": registered.label,
            "execution": execution,
            "planned_identity": dict(planned_identity),
            "expected_runs": len(expected_seeds),
            "expected_seeds": list(expected_seeds),
            "observed_valid_seeds": sorted(valid_seeds),
            "missing_valid_seeds": sorted(set(expected_seeds) - valid_seeds),
            "observed_attempts": sum(
                record.metrics_available or record.terminal_status != "MISSING"
                for record in records
            ),
            "infrastructure_invalid_attempts": sum(
                record.infrastructure_invalid for record in invalid
            ),
            "protocol_invalid_attempts": len(invalid),
            "invalid_seed_details": [
                {
                    "seed": expected_seed,
                    "observed_seed": record.seed,
                    "run_dir": record.run_dir,
                    "terminal_status": record.terminal_status,
                    "infrastructure_invalid": record.infrastructure_invalid,
                    "reasons": list(record.invalid_reasons),
                }
                for expected_seed, record in invalid_seed_bindings
            ],
            "source_reference": (
                dict(source_reference) if source_reference is not None else None
            ),
        }
    )
    # ``summarize_condition`` is the sole authority for aggregate metrics.
    # Assert its five-seed gate here so later presentation code cannot emit a
    # complete-looking row with a changed denominator.
    if result["complete"]:
        if result["observed_valid_seeds"] != list(expected_seeds):
            raise PaperSummaryError("complete paper row does not contain exactly five seeds")
        if result["missing_valid_seeds"]:
            raise PaperSummaryError("complete paper row reports missing seeds")
    return result


def _table1_audit_records(
    reuse: ReuseSpec,
    audit: Table1CohortAudit,
) -> tuple[RunRecord, ...]:
    """Project strictly audited legacy source rows into the generic record IR."""

    expected = {item.name: item.sha256 for item in reuse.expected_hashes}
    selector = reuse.source_selector
    if not selector.external:
        raise PaperSummaryError("Table-1 audit adapter requires an external reuse")
    if audit.manifest_sha256 != expected.get("result_manifest"):
        raise PaperSummaryError("Table-1 source aggregate manifest hash differs")
    condition = Path(selector.output_relpath).name
    rows: list[RunRecord] = []
    for run in audit.runs:
        kwargs: dict[str, Any] = {
            "run_dir": run.run_dir,
            "table": selector.source_table,
            "condition": condition,
            "method": selector.method_variant.value,
            "scene": selector.scene_id.value,
            "memory_condition": selector.memory_condition.value,
            "source_model": "",
            "target_model": run.backbone,
            "provider": run.provider,
            "provider_route_id": run.provider_route_id,
            "provider_route_fingerprint": run.provider_route_fingerprint,
            "strategy_bank_sha256": "",
            "seed": run.seed,
            "protocol_id": run.protocol_id,
            "base_method_commit": run.method_commit,
            "condition_fingerprint": run.condition_fingerprint,
            "invariant_fingerprint": expected["invariant"],
            "scene_fingerprint": expected["scene"],
            "terminal_fingerprint": expected["terminal"],
            "observation_contract_fingerprint": expected["observation_contract"],
            "task_instruction_fingerprint": expected["task_instruction"],
            "shared_prompt_contract_fingerprint": (
                run.shared_prompt_contract_fingerprint
            ),
            "decision_prompt_fingerprint": run.decision_prompt_fingerprint,
            "skill_api_fingerprint": expected["skill_api"],
            "full_method_compatibility_fingerprint": (
                run.full_method_compatibility_fingerprint
            ),
            "terminal_status": run.terminal_status,
            "metrics_available": True,
            "protocol_valid": True,
            "infrastructure_invalid": False,
            "invalid_reasons": (),
            "task_success": run.task_success,
            "physical_transactions": run.physical_transactions,
            "accepted_transactions": run.accepted_transactions,
            "rejected_transactions": run.rejected_transactions,
            "phase_completed": run.phase_completed,
            "phase_total": run.phase_total,
            "phase_completion": run.phase_completion,
            "fm_calls": run.fm_calls,
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
            "wall_time_s": run.wall_time_s,
            "cost_usd": run.cost_usd,
            "cost_complete": False,
            "artifact_hashes": {
                **dict(run.artifact_hashes),
                "source_aggregate_manifest": audit.manifest_sha256,
            },
        }
        rows.append(RunRecord(**kwargs))
    return tuple(rows)


def _require_audit_matches_external_reuses(
    audit: Table1CohortAudit,
    reuses: Iterable[ReuseSpec],
) -> None:
    selected = tuple(reuses)
    if not selected:
        raise PaperSummaryError("paper suite lacks its external reuse rows")
    if not audit.protocol_valid:
        raise PaperSummaryError("Table-1 source audit is not a complete five-run cohort")
    for reuse in selected:
        selector = reuse.source_selector
        expected = {item.name: item.sha256 for item in reuse.expected_hashes}
        if not selector.external:
            raise PaperSummaryError("external source check received an internal reuse")
        checks = {
            "backbone": (audit.source.backbone, selector.backbone),
            "protocol": (audit.source.protocol_id, selector.protocol_id),
            "method commit": (audit.source.method_commit, selector.method_commit),
            "seeds": (tuple(audit.source.seeds), tuple(selector.seeds)),
            "condition": (
                audit.source.condition_fingerprint,
                expected.get("condition"),
            ),
            "result manifest": (
                audit.manifest_sha256,
                expected.get("result_manifest"),
            ),
            "provider route ID": (
                audit.provider_route_id,
                selector.provider_route_id,
            ),
            "provider route fingerprint": (
                audit.provider_route_fingerprint,
                expected.get("provider_route"),
            ),
            "shared prompt contract": (
                audit.source.shared_prompt_contract_fingerprint,
                expected.get("shared_prompt_contract"),
            ),
            "Full decision prompt": (
                audit.source.decision_prompt_fingerprint,
                expected.get("decision_prompt"),
            ),
            "Full-method compatibility": (
                audit.full_method_compatibility.fingerprint,
                expected.get("full_method_compatibility"),
            ),
        }
        for label, (actual, wanted) in checks.items():
            if actual != wanted:
                raise PaperSummaryError(f"Table-1 reuse {label} differs from audit")
        if audit.source.provider_route_id != selector.provider_route_id:
            raise PaperSummaryError("Table-1 source provider route ID differs")
        if (
            audit.source.provider_route_fingerprint
            != selector.provider_route_fingerprint
        ):
            raise PaperSummaryError("Table-1 source provider route fingerprint differs")
        scene = load_scene(selector.scene_id)
        if scene.scene_hash != expected.get("scene"):
            raise PaperSummaryError("Table-1 reuse scene fingerprint differs")
        if (
            scene.asset.source_sha256
            != audit.shared_method_artifact_hashes.get("scene_sha256")
        ):
            raise PaperSummaryError(
                "Table-1 source scene asset differs from the registered base scene"
            )
        if (
            audit.shared_method_artifact_hashes.get("skill_api_sha256")
            != expected.get("skill_api")
        ):
            raise PaperSummaryError("Table-1 source skill API differs from reuse lock")


def _require_exact_run_map(
    suite: ExperimentSuite,
    records: Mapping[str, RunRecord],
) -> None:
    expected = {run.run_id for run in suite.new_runs}
    actual = set(records)
    if actual != expected:
        raise PaperSummaryError(
            "new-run record map differs from suite: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if not all(isinstance(value, RunRecord) for value in records.values()):
        raise PaperSummaryError("new-run record map contains a non-RunRecord value")


def _require_spec_matches_registered_row(
    spec: RunSpec,
    registered: RegisteredRow,
) -> None:
    if (
        spec.table_id != registered.table_id
        or _run_condition_id(spec) != registered.condition_id
        or spec.method_variant != registered.method_variant
        or spec.scene_id != registered.scene_id
        or spec.memory_condition != registered.memory_condition
    ):
        raise PaperSummaryError(
            f"RunSpec differs from registered row: {spec.run_id}"
        )


def _require_record_matches_spec(record: RunRecord, spec: RunSpec) -> None:
    if record.fm_calls is None or record.fm_calls <= 0:
        raise PaperSummaryError(
            f"protocol-valid record has no positive FM-call evidence: {spec.run_id}"
        )
    strategy = spec.strategy_bank
    expected: Mapping[str, Any] = {
        "table": spec.table_id.value,
        "condition": _run_condition_id(spec),
        "method": spec.method_variant.value,
        "scene": spec.scene_id.value,
        "memory_condition": spec.memory_condition.value,
        "source_model": strategy.source_backbone if strategy else "",
        "target_model": spec.target_backbone,
        "provider_route_id": spec.provider_route_id,
        "provider_route_fingerprint": spec.provider_route_fingerprint,
        "strategy_bank_sha256": strategy.sha256 if strategy else "",
        "seed": spec.seed,
        "protocol_id": spec.protocol_id,
        "base_method_commit": spec.base_method_commit,
        "condition_fingerprint": spec.condition_fingerprint,
        "invariant_fingerprint": spec.invariant_fingerprint,
        "scene_fingerprint": spec.scene_fingerprint,
        "terminal_fingerprint": spec.terminal_fingerprint,
        "observation_contract_fingerprint": (
            spec.observation_contract_fingerprint
        ),
        "task_instruction_fingerprint": spec.task_instruction_fingerprint,
        "shared_prompt_contract_fingerprint": (
            spec.shared_prompt_contract_fingerprint
        ),
        "decision_prompt_fingerprint": spec.decision_prompt_fingerprint,
        "skill_api_fingerprint": spec.skill_api_fingerprint,
        "full_method_compatibility_fingerprint": (
            spec.full_method_compatibility_fingerprint
        ),
    }
    differing = [
        name for name, value in expected.items() if getattr(record, name) != value
    ]
    if differing:
        raise PaperSummaryError(
            f"protocol-valid record differs from RunSpec {spec.run_id}: "
            f"{differing}"
        )


def _planned_identity(spec: RunSpec) -> dict[str, Any]:
    omitted = {"run_id", "seed", "output_relpath"}
    return {
        key: value
        for key, value in to_primitive(spec).items()
        if key not in omitted
    }


def _run_condition_id(run: RunSpec) -> str:
    parts = Path(run.output_relpath).parts
    if len(parts) != 3 or parts[0] != run.table_id.value:
        raise PaperSummaryError(f"RunSpec has malformed output_relpath: {run.run_id}")
    return parts[1]


def _reuse_condition_id(reuse: ReuseSpec) -> str:
    prefix = reuse.target_table.value + "__"
    if not reuse.reuse_id.startswith(prefix):
        raise PaperSummaryError(f"ReuseSpec has malformed reuse_id: {reuse.reuse_id}")
    condition = reuse.reuse_id[len(prefix) :]
    if not condition:
        raise PaperSummaryError(f"ReuseSpec has empty target condition: {reuse.reuse_id}")
    return condition


def _selector_identity(reuse: ReuseSpec) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(asdict(reuse.source_selector).items()))


def _locked_source_for_selector(
    suite: ExperimentSuite,
    selector: Any,
) -> Table1SourceCohort:
    for source in (
        suite.protocol_lock.table1_gpt_source,
        suite.protocol_lock.table1_gemini_source,
    ):
        if (
            source.backbone == selector.backbone
            and source.protocol_id == selector.protocol_id
            and source.method_commit == selector.method_commit
            and tuple(source.seeds) == tuple(selector.seeds)
        ):
            for name in (
                "provider_route_id",
                "provider_route_fingerprint",
                "full_method_compatibility_fingerprint",
            ):
                if hasattr(selector, name) or hasattr(source, name):
                    if getattr(selector, name, None) != getattr(source, name, None):
                        break
            else:
                return source
    raise PaperSummaryError("external reuse selector matches no locked Table-1 source")


def _absolute_regular_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise PaperSummaryError(f"{label} path must be absolute")
    if path.is_symlink():
        raise PaperSummaryError(f"{label} must not be a symlink")
    if not path.is_file():
        raise PaperSummaryError(f"{label} is not an existing regular file")
    return path.resolve()


__all__ = [
    "IncompletePaperSummaryError",
    "PAPER_SUMMARY_KIND",
    "PAPER_SUMMARY_SCHEMA_VERSION",
    "PaperSummaryError",
    "audit_external_reuse_source",
    "summarize_paper_tables",
]
