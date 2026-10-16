"""Small command surface for planning, transfer, validation, and reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from .contracts import SceneId
from .lockfile import (
    LOCK_SCHEMA_VERSION,
    load_protocol_lock,
    protocol_lock_document,
)
from .manuscript_audit import audit_manuscript_protocol
from .live import (
    compile_live_launch,
    execute_live_batch,
    inspect_live_batch,
    launch_live_subprocess,
    prepare_live_batch,
    validate_live_bridge_receipt,
    write_subprocess_launch_config,
)
from .memory import (
    export_audited_table1_strategy_bank,
    load_posthoc_source_evidence,
    prepare_transfer_from_plan,
    produce_posthoc_source_evidence,
    validate_locked_strategy_bank_candidate,
    validate_strategy_bank,
    write_strategy_bank_candidate_attestation,
    write_strategy_bank_export,
)
from .paper_contract import build_paper_contract
from .protocol_lock_builder import (
    build_canonical_protocol_lock,
    load_protocol_lock_build_inputs,
    write_protocol_lock_build,
)
from .registry import build_suite
from .reporting import (
    audit_frozen_table1_candidate,
    audit_external_reuse_source,
    audit_table1_source_cohort,
    collect_run,
    summarize_paper_tables,
    summarize_suite,
)
from .reporting.replay_evidence import (
    execute_strict_replay,
    load_strict_replay_manifest,
    prepare_strict_replay_manifest,
)
from .scenes import (
    EVIDENCE_CHECKS,
    load_scene,
    materialize_scene,
    produce_scene_evidence,
    validate_scene,
)
from .serialization import read_json, to_primitive, write_json
from .smoke import smoke_all_method_adapters, smoke_method_adapter
from .suite_plan import SUITE_PLAN_SCHEMA_VERSION, validate_suite_plan_document


def build_parser() -> argparse.ArgumentParser:
    """Backward-compatible CLI facade; parsing lives separately from handlers."""
    from .cli_parser import build_parser as register_arguments

    handlers = {name: value for name, value in globals().items() if name.startswith("_cmd_")}
    return register_arguments(handlers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    args._raw_invocation = tuple(
        [sys.executable, *sys.argv]
        if argv is None
        else [sys.executable, "-m", "coevolve_experiments", *arguments]
    )
    try:
        return int(args.handler(args) or 0)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "command": args.command, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


def _cmd_plan(args: argparse.Namespace) -> int:
    lock = load_protocol_lock(args.lock)
    suite = build_suite(args.tier, lock)
    execution_reports: list[dict[str, Any]] = []
    execution_failures: list[str] = []
    if args.require_execution_ready:
        scene_ids = sorted({run.scene_id for run in suite.new_runs}, key=lambda item: item.value)
        for scene_id in scene_ids:
            report = validate_scene(
                load_scene(scene_id), mode="execution", protocol_lock=lock
            )
            execution_reports.append(to_primitive(report))
            if not report.execution_ready:
                reasons = report.errors or (
                    "execution report lacks complete host-derived evidence",
                )
                execution_failures.append(
                    f"{report.scene_id.value}: {', '.join(reasons)}"
                )
        if execution_failures:
            raise ValueError(
                f"execution preflight failed: {'; '.join(execution_failures)}"
            )
    document = {
        "kind": "coevolve_tables_2_4_suite_plan",
        "schema_version": SUITE_PLAN_SCHEMA_VERSION,
        "planning_validated": True,
        "execution_preflight": "passed" if args.require_execution_ready else "not_requested",
        "protocol_lock": protocol_lock_document(lock),
        "suite": to_primitive(suite),
        "execution_reports": execution_reports,
    }
    _emit(document, args.output)
    return 0


def _cmd_paper_contract(args: argparse.Namespace) -> int:
    _emit(
        {
            "kind": "coevolve_paper_experiment_contract",
            "contract": build_paper_contract(args.tier),
        },
        args.output,
    )
    return 0


def _cmd_build_protocol_lock(args: argparse.Namespace) -> int:
    inputs = load_protocol_lock_build_inputs(args.input)
    build = build_canonical_protocol_lock(
        gpt_source=inputs.gpt_source,
        gemini_source=inputs.gemini_source,
        contract_artifacts=inputs.contract_artifacts,
    )
    persisted = write_protocol_lock_build(
        build,
        args.output,
        args.identity_output,
    )
    _emit(
        {
            "ok": True,
            "paper_evidence": False,
            "physics_dispatched": False,
            "execution_ready": False,
            "protocol_id": persisted.protocol_id,
            "protocol_lock_path": str(persisted.lock_path),
            "protocol_lock_file_sha256": persisted.lock_file_sha256,
            "protocol_identity_path": str(persisted.identity_path),
            "protocol_identity_file_sha256": persisted.identity_file_sha256,
            "new_run_count": build.receipt["new_run_count"],
            "reuse_count": build.receipt["reuse_count"],
            "next_step": (
                "review the identity receipt, perform both locked bank "
                "re-exports, then validate scenes and build the suite plan"
            ),
        },
        None,
    )
    return 0


def _cmd_audit_manuscript(args: argparse.Namespace) -> int:
    report = audit_manuscript_protocol(
        methodology_path=args.methodology,
        experiments_path=args.experiments,
        protocol_root=args.protocol_root,
    )
    report.require_ok()
    _emit(
        {
            "kind": "coevolve_manuscript_protocol_audit",
            "audit": report,
        },
        args.output,
    )
    return 0


def _cmd_compile_launch(args: argparse.Namespace) -> int:
    compiled = compile_live_launch(
        args.plan,
        args.run_id,
        results_root=args.results_root,
        strategy_bank_path=args.strategy_bank,
        strategy_bank_provenance_path=args.strategy_bank_provenance,
        require_launchable=args.require_launchable,
    )
    _emit(
        {
            "kind": "coevolve_compiled_live_launch",
            "launch": compiled,
        },
        args.output,
    )
    return 0


def _cmd_bind_live_launch(args: argparse.Namespace) -> int:
    compiled = compile_live_launch(
        args.plan,
        args.run_id,
        results_root=args.results_root,
        strategy_bank_path=args.strategy_bank,
        strategy_bank_provenance_path=args.strategy_bank_provenance,
        require_launchable=False,
    )
    verified = write_subprocess_launch_config(
        compiled,
        capability_manifest_path=args.capability,
        destination=args.launch_config,
    )
    _emit(
        {
            "kind": "coevolve_bound_live_launch",
            "paper_evidence": False,
            "launch_config_path": verified.path,
            "launch_config_sha256": verified.sha256,
            "config_fingerprint": verified.config_fingerprint,
            "run_id": verified.launch.run_spec.run_id,
            "runner_capability_sha256": verified.capability.manifest_sha256,
        },
        None,
    )
    return 0


def _cmd_run_live(args: argparse.Namespace) -> int:
    receipt = launch_live_subprocess(
        args.launch_config,
        timeout_seconds=args.timeout_seconds,
    )
    _emit(
        {
            "kind": "coevolve_completed_live_run",
            "receipt": receipt,
        },
        None,
    )
    return 0


def _cmd_prepare_live_batch(args: argparse.Namespace) -> int:
    batch = prepare_live_batch(
        plan_path=args.plan,
        results_root=args.results_root,
        capability_manifest_path=args.capability,
        batch_root=args.batch_root,
        gpt_strategy_bank_path=args.gpt_strategy_bank,
        gpt_strategy_provenance_path=args.gpt_strategy_provenance,
        gemini_strategy_bank_path=args.gemini_strategy_bank,
        gemini_strategy_provenance_path=args.gemini_strategy_provenance,
        partition=args.partition,
    )
    _emit(
        {
            "kind": "coevolve_prepared_live_batch",
            "paper_evidence": False,
            "physics_dispatched": False,
            "batch_manifest_path": batch.manifest_path,
            "batch_manifest_sha256": batch.manifest_sha256,
            "batch_fingerprint": batch.batch_fingerprint,
            "partition": batch.partition,
            "run_count": len(batch.cells),
        },
        None,
    )
    return 0


def _cmd_status_live_batch(args: argparse.Namespace) -> int:
    _emit(inspect_live_batch(args.manifest), args.output)
    return 0


def _cmd_run_live_batch(args: argparse.Namespace) -> int:
    _emit(
        execute_live_batch(
            args.manifest,
            timeout_seconds=args.timeout_seconds,
            max_runs=args.max_runs,
        ),
        args.output,
    )
    return 0


def _cmd_validate_scenes(args: argparse.Namespace) -> int:
    if args.mode == "execution" and args.lock is None:
        raise ValueError("execution scene validation requires --lock")
    lock = load_protocol_lock(args.lock) if args.lock is not None else None
    reports = [
        validate_scene(load_scene(scene_id), mode=args.mode, protocol_lock=lock,
                       repo_root=args.repo_root)
        for scene_id in SceneId
    ]
    reports_ok = all(
        report.execution_ready if args.mode == "execution" else report.ok
        for report in reports
    )
    document = {
        "kind": "coevolve_scene_validation",
        "mode": args.mode,
        "ok": reports_ok,
        "reports": [to_primitive(report) for report in reports],
    }
    _emit(document, args.output)
    return 0 if document["ok"] else 2


def _cmd_materialize_scene(args: argparse.Namespace) -> int:
    result = materialize_scene(
        load_scene(SceneId(args.scene)),
        args.repo_root,
        args.output,
    )
    _emit(
        {
            "kind": "coevolve_candidate_scene_materialization",
            "ok": True,
            "execution_ready": False,
            "paper_evidence": False,
            "result": result,
        },
        None,
    )
    return 0


def _cmd_produce_scene_evidence(args: argparse.Namespace) -> int:
    result = produce_scene_evidence(
        args.check,
        scene=load_scene(SceneId(args.scene)),
        protocol_lock=load_protocol_lock(args.lock),
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        gear_root=args.gear_root,
        invocation=args._raw_invocation,
    )
    _emit(
        {
            "kind": "coevolve_scene_evidence_production",
            "ok": True,
            "check": args.check,
            "execution_ready": False,
            "paper_evidence": False,
            "remaining_required_checks": [
                item for item in EVIDENCE_CHECKS if item != args.check
            ],
            "result": result,
        },
        None,
    )
    return 0


def _cmd_build_posthoc_evidence(args: argparse.Namespace) -> int:
    if len(args.source_run) != 5:
        raise ValueError(
            "build-posthoc-strategy-evidence requires exactly five --source-run values"
        )
    audit = audit_frozen_table1_candidate(
        args.source_manifest,
        args.source_cohort,
        parent_manifest_path=args.parent_manifest,
    )
    package = produce_posthoc_source_evidence(
        args.source_run,
        audit,
        args.output_dir,
    )
    _emit(
        {
            "ok": True,
            "paper_evidence": False,
            "source_cohort": args.source_cohort,
            "source_result_manifest_hash": audit.manifest_sha256,
            "package_path": str(package.root),
            "package_manifest_path": str(package.manifest_path),
            "package_manifest_sha256": package.manifest_sha256,
            "package_payload_sha256": package.payload_sha256,
            "run_count": len(package.runs),
            "next_step": "candidate-export-strategy-bank",
        },
        None,
    )
    return 0


def _cmd_candidate_export_bank(args: argparse.Namespace) -> int:
    if len(args.source_run) != 5:
        raise ValueError(
            "candidate-export-strategy-bank requires exactly five --source-run values"
        )
    audit = audit_frozen_table1_candidate(
        args.source_manifest,
        args.source_cohort,
        parent_manifest_path=args.parent_manifest,
    )
    package = load_posthoc_source_evidence(args.evidence_package, audit)
    result = export_audited_table1_strategy_bank(
        args.source_run,
        audit,
        posthoc_evidence_package=args.evidence_package,
    )
    persisted = write_strategy_bank_export(
        result,
        args.output,
        args.provenance_output,
    )
    candidate = write_strategy_bank_candidate_attestation(
        result,
        audit,
        package,
        args.source_cohort,
        args.candidate_output,
    )
    _emit(
        {
            "ok": True,
            "paper_evidence": False,
            "source_cohort": args.source_cohort,
            "bank_path": str(persisted.bank_path),
            "provenance_path": str(persisted.provenance_path),
            "candidate_attestation_path": str(candidate.path),
            "candidate_attestation_sha256": candidate.file_sha256,
            "bank_sha256": candidate.bank_sha256,
            "protocol_lock_field": candidate.payload["lock_field"],
            "protocol_lock_schema_version": LOCK_SCHEMA_VERSION,
            "next_step": (
                "freeze this hash in the protocol lock, then run "
                "export-strategy-bank"
            ),
        },
        None,
    )
    return 0


def _cmd_export_bank(args: argparse.Namespace) -> int:
    if len(args.source_run) != 5:
        raise ValueError("export-strategy-bank requires exactly five --source-run values")
    lock = load_protocol_lock(args.lock)
    if args.source_cohort == "gpt":
        if args.parent_manifest is not None:
            raise ValueError("the GPT source cohort must not declare --parent-manifest")
        source = lock.table1_gpt_source
        expected_bank_hash = lock.gpt_strategy_bank_hash
        cohort_audit = audit_table1_source_cohort(
            args.source_manifest,
            source,
        )
    else:
        if args.parent_manifest is None:
            raise ValueError(
                "the Gemini companion requires --parent-manifest for exact V83"
            )
        source = lock.table1_gemini_source
        expected_bank_hash = lock.gemini_strategy_bank_hash
        parent_audit = audit_table1_source_cohort(
            args.parent_manifest,
            lock.table1_gpt_source,
        )
        cohort_audit = audit_table1_source_cohort(
            args.source_manifest,
            source,
            parent=parent_audit,
        )
    package = load_posthoc_source_evidence(args.evidence_package, cohort_audit)
    result = export_audited_table1_strategy_bank(
        args.source_run,
        cohort_audit,
        expected_bank_sha256=expected_bank_hash,
        posthoc_evidence_package=args.evidence_package,
    )
    candidate = validate_locked_strategy_bank_candidate(
        result,
        cohort_audit,
        package,
        args.source_cohort,
        args.candidate_attestation,
        locked_bank_sha256=expected_bank_hash,
    )
    persisted = write_strategy_bank_export(
        result,
        args.output,
        args.provenance_output,
    )
    _emit(
        {
            "ok": True,
            "paper_evidence": False,
            "lock_reexport_verified": True,
            "bank_path": str(persisted.bank_path),
            "provenance_path": str(persisted.provenance_path),
            "bank_sha256": result.provenance["bank_sha256"],
            "record_count": len(result.bank["records"]),
            "candidate_attestation_sha256": candidate.file_sha256,
        },
        None,
    )
    return 0


def _cmd_validate_bank(args: argparse.Namespace) -> int:
    result = validate_strategy_bank(args.bank)
    if args.expected_sha256 and result.bank_sha256 != args.expected_sha256:
        raise ValueError(
            f"bank SHA-256 mismatch: expected {args.expected_sha256}, got {result.bank_sha256}"
        )
    _emit({"ok": True, **to_primitive(result)}, args.output)
    return 0


def _cmd_prepare_transfer(args: argparse.Namespace) -> int:
    prepared = prepare_transfer_from_plan(
        plan_path=args.plan,
        run_id=args.run_id,
        results_root=args.results_root,
        bank_path=args.bank,
        provenance_path=args.provenance,
    )
    _emit(
        {
            "ok": True,
            "run_id": prepared.run_spec.run_id,
            "plan_path": str(prepared.plan_path),
            "plan_sha256": prepared.plan_sha256,
            "run_dir": str(prepared.run_dir),
            "memory_dir": str(prepared.memory_dir),
            "seed_bank_sha256": prepared.bootstrap.seed_bank_sha256,
            "bootstrap_audit_path": str(prepared.bootstrap.audit_path),
            "bootstrap_snapshot_path": str(prepared.bootstrap.snapshot_path),
            "bootstrap_snapshot_sha256": prepared.bootstrap.snapshot_sha256,
        },
        None,
    )
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    if args.results_root is not None:
        if args.plan is None:
            raise ValueError("--results-root requires the external --plan")
        if args.source_aggregate_manifest is None:
            raise ValueError(
                "formal --results-root summary requires "
                "--table1-source-manifest"
            )
        payload = read_json(args.plan)
        if not isinstance(payload, Mapping):
            raise ValueError("suite plan root must be an object")
        verified = validate_suite_plan_document(payload)
        records = _collect_planned_runs(args.results_root, args.plan)
        if len(records) != len(verified.suite.new_runs):
            raise ValueError("planned collector returned the wrong number of run cells")
        by_run_id = {
            run.run_id: record
            for run, record in zip(verified.suite.new_runs, records)
        }
        source_audit = audit_external_reuse_source(
            verified.suite,
            args.source_aggregate_manifest,
            parent_manifest_path=args.source_parent_manifest,
        )
        document = summarize_paper_tables(
            verified.suite,
            by_run_id,
            table1_source_audit=source_audit,
            require_complete=args.require_complete,
        )
        resolved_plan = args.plan.resolve()
        document["suite_plan"] = {
            "path": str(resolved_plan),
            "sha256": hashlib.sha256(resolved_plan.read_bytes()).hexdigest(),
        }
        document["results_root"] = str(args.results_root.resolve())
    else:
        if args.plan is not None:
            raise ValueError("--plan is only valid with --results-root")
        if args.source_aggregate_manifest is not None:
            raise ValueError(
                "--table1-source-manifest is only valid with --results-root"
            )
        if args.source_parent_manifest is not None:
            raise ValueError(
                "--table1-parent-manifest is only valid with --results-root"
            )
        records = [collect_run(path) for path in args.run_dir]
        document = summarize_suite(records, require_complete=args.require_complete)
    _emit(document, args.output)
    return 0


def _cmd_adapter_smoke(args: argparse.Namespace) -> int:
    results = (
        smoke_all_method_adapters()
        if args.method == "all"
        else [smoke_method_adapter(args.method)]
    )
    document = {
        "kind": "coevolve_adapter_smoke",
        "ok": all(item["success"] for item in results),
        "physics_dispatched": False,
        "execution_capability": "contract_smoke_only",
        "paper_evidence": False,
        "results": results,
    }
    _emit(document, args.output)
    return 0 if document["ok"] else 2


def _cmd_prepare_strict_replay(args: argparse.Namespace) -> int:
    binding = prepare_strict_replay_manifest(
        args.run_dir,
        args.package_dir,
        mode=args.mode,
        destination=args.manifest_output,
    )
    _emit(
        {
            "kind": "coevolve_prepared_strict_replay",
            "paper_evidence": False,
            "physics_dispatched": False,
            "diagnostic_only": True,
            "manifest_path": str(binding.manifest_path),
            "manifest_sha256": binding.manifest_sha256,
            "history_mode": binding.mode.value,
            "action_count": binding.action_count,
            "action_sequence_sha256": binding.action_sequence_sha256,
        },
        None,
    )
    return 0


def _cmd_run_strict_replay(args: argparse.Namespace) -> int:
    binding = load_strict_replay_manifest(
        args.manifest,
        run_root=args.run_dir,
        package_root=args.package_dir,
    )
    result = execute_strict_replay(
        binding,
        python_executable=args.python_executable,
        timeout_s=args.timeout_seconds,
    )
    _emit(
        {
            "kind": "coevolve_completed_strict_replay",
            "paper_evidence": False,
            "physics_dispatched": True,
            "diagnostic_only": True,
            "result": result,
        },
        None,
    )
    return 0


def _collect_planned_runs(results_root: Path, plan_path: Path) -> list[Any]:
    resolved_plan_path = plan_path.resolve()
    try:
        plan_sha256 = hashlib.sha256(resolved_plan_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot hash external suite plan {resolved_plan_path}: {exc}") from exc
    payload = read_json(plan_path)
    if not isinstance(payload, Mapping):
        raise ValueError("suite plan root must be an object")
    suite = _verified_formal_suite(payload)
    run_plan = suite.get("run_plan", suite)
    if not isinstance(run_plan, Mapping):
        raise ValueError("suite plan is missing run_plan")
    runs = run_plan.get("new_runs")
    if not isinstance(runs, list):
        raise ValueError("suite plan new_runs must be an array")
    records = []
    execution_reports = payload.get("execution_reports")
    if not isinstance(execution_reports, list) or not all(
        isinstance(item, Mapping) for item in execution_reports
    ):
        raise ValueError("suite plan execution_reports must be an array of objects")
    from .reporting.scene_result_binding import resolve_scene_result_binding

    for value in runs:
        if not isinstance(value, Mapping):
            raise ValueError("suite plan contains a malformed run")
        relative = str(value.get("output_relpath", ""))
        parts = Path(relative).parts
        condition = parts[1] if len(parts) >= 2 else ""
        strategy_bank = value.get("strategy_bank", {})
        strategy_bank = strategy_bank if isinstance(strategy_bank, Mapping) else {}
        scene = load_scene(str(value.get("scene_id", "")))
        scene_result = resolve_scene_result_binding(
            execution_reports,
            scene_id=scene.scene_id.value,
            scene_fingerprint=str(value.get("scene_fingerprint", "")),
            scene_realization_sha256=str(
                scene.asset.execution_composite_sha256 or ""
            ),
        )
        expected = {
            "run_id": value.get("run_id", ""),
            "table": value.get("table_id", ""),
            "condition": condition,
            "method": value.get("method_variant", ""),
            "scene": value.get("scene_id", ""),
            "memory_condition": value.get("memory_condition", ""),
            "memory_policy": value.get("memory_policy", ""),
            "source_model": strategy_bank.get("source_backbone", ""),
            "source_provider_route_id": strategy_bank.get(
                "source_provider_route_id", ""
            ),
            "source_provider_route_fingerprint": strategy_bank.get(
                "source_provider_route_fingerprint", ""
            ),
            "strategy_bank_sha256": strategy_bank.get("sha256", ""),
            "strategy_bank": dict(strategy_bank),
            "target_model": value.get("target_backbone", ""),
            "provider_route_id": value.get("provider_route_id", ""),
            "provider_route_fingerprint": value.get(
                "provider_route_fingerprint", ""
            ),
            "seed": value.get("seed"),
            "protocol_id": value.get("protocol_id", ""),
            "base_method_commit": value.get("base_method_commit", ""),
            "condition_fingerprint": value.get("condition_fingerprint", ""),
            "invariant_fingerprint": value.get("invariant_fingerprint", ""),
            "scene_fingerprint": value.get("scene_fingerprint", ""),
            "scene_realization_sha256": scene_result.scene_realization_sha256,
            "scene_execution_evidence_fingerprint": (
                scene_result.scene_execution_evidence_fingerprint
            ),
            "compiled_model_sha256": scene_result.compiled_model_sha256,
            "target_center_xyz_m": list(scene.target.center.as_tuple()),
            "terminal_fingerprint": value.get("terminal_fingerprint", ""),
            "observation_contract_fingerprint": value.get(
                "observation_contract_fingerprint", ""
            ),
            "task_instruction_fingerprint": value.get(
                "task_instruction_fingerprint", ""
            ),
            "shared_prompt_contract_fingerprint": value.get(
                "shared_prompt_contract_fingerprint", ""
            ),
            "decision_prompt_fingerprint": value.get(
                "decision_prompt_fingerprint", ""
            ),
            "skill_api_fingerprint": value.get("skill_api_fingerprint", ""),
            "full_method_compatibility_fingerprint": value.get(
                "full_method_compatibility_fingerprint", ""
            ),
            "plan_path": str(resolved_plan_path),
            "plan_sha256": plan_sha256,
            "require_terminal_evidence": True,
            "require_usage_evidence": True,
            "require_native_memory_evidence": True,
            "require_transaction_accounting_evidence": True,
        }
        run_dir = results_root / relative
        if _has_formal_result_artifacts(run_dir):
            validate_live_bridge_receipt(
                run_dir,
                expected_run_id=str(value.get("run_id", "")),
            )
        records.append(
            collect_run(
                run_dir,
                expected=expected,
                physical_budget=int(value.get("physical_budget", 80)),
            )
        )
    return records


def _verified_formal_suite(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    verified = validate_suite_plan_document(payload)
    if verified.execution_preflight != "passed":
        raise ValueError(
            "formal summarize requires execution_preflight='passed'"
        )
    scene_ids = sorted(
        {run.scene_id for run in verified.suite.new_runs},
        key=lambda item: item.value,
    )
    current_report_objects = [
        validate_scene(
            load_scene(scene_id),
            mode="execution",
            protocol_lock=verified.protocol_lock,
        )
        for scene_id in scene_ids
    ]
    current_reports = [to_primitive(report) for report in current_report_objects]
    if payload.get("execution_reports") != current_reports or any(
        not report.execution_ready for report in current_report_objects
    ):
        raise ValueError(
            "suite plan execution reports do not match current hash-pinned preflight"
        )
    return to_primitive(verified.suite)


def _has_formal_result_artifacts(run_dir: Path) -> bool:
    """Distinguish an unstarted planned cell from an unattested result."""

    return any(
        (run_dir / filename).exists()
        for filename in (
            "run_manifest.json",
            "metrics.json",
            "live_bridge_receipt.json",
        )
    )


def _emit(value: Any, output: Path | None) -> None:
    if output is not None:
        write_json(output, value)
        return
    print(
        json.dumps(
            to_primitive(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


__all__ = ["build_parser", "main"]
