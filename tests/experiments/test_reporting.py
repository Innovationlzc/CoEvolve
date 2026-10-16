import json
from hashlib import sha256
import importlib

import pytest

from coevolve_experiments.contracts import (
    MethodDecisionPromptPin,
    MethodVariant,
    ProtocolLock,
    Table1SourceCohort,
)
from coevolve_experiments.provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE,
    SECONDARY_GEMINI_PROVIDER_ROUTE,
)
from coevolve_experiments.full_method_compatibility import (
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
)
from coevolve_experiments.memory.online_memory import (
    empty_action_exemplar_memory,
    empty_online_strategy_overlay,
)
from coevolve_experiments.memory.bootstrap import (
    TRANSFER_BOOTSTRAP_INITIAL_FILES,
    TRANSFER_BOOTSTRAP_MUTABLE_FILES,
    TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME,
    TRANSFER_BOOTSTRAP_SNAPSHOT_KIND,
    TRANSFER_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION,
)
from coevolve_experiments.memory.schema import (
    FROZEN_STRATEGY_BANK_POLICY,
    semantic_sha256,
)
from coevolve_experiments.memory.v83_native import (
    V83MemoryOperationGates,
    V83NativeMemoryBridge,
    V83_NATIVE_AUDIT_FILENAME,
)
from coevolve_experiments.reporting import (
    IncompleteConditionError,
    MixedConditionError,
    RunRecord,
    ReuseResolutionError,
    TraceSelectionError,
    alias_reused_condition,
    build_appendix_tables,
    build_figure2_trace,
    collect_run,
    summarize_condition,
    summarize_suite,
)
from coevolve_experiments.reporting.trace_projection import (
    build_host_mutation_delta,
)
from coevolve_experiments.registry import build_run_plan
from coevolve_experiments.serialization import sha256_json


def _reuse_lock(protocol_id: str) -> ProtocolLock:
    prompt = "9" * 64
    full_method = FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": "1" * 64,
            "prompt_sha256": prompt,
            "route_spec_sha256": "2" * 64,
            "scene_sha256": "3" * 64,
            "skill_api_sha256": "a" * 64,
            "static_memory_sha256": "4" * 64,
            "verifier_config_sha256": "5" * 64,
            "verifier_source_sha256": "6" * 64,
        },
        behavior_artifacts={
            path: sha256(f"fixture:{path}".encode("utf-8")).hexdigest()
            for path in V83_FULL_BEHAVIOR_RELPATHS
        },
    )
    return ProtocolLock(
        protocol_id=protocol_id,
        invariant_fingerprint="1" * 64,
        terminal_fingerprint="2" * 64,
        base_method_commit="a" * 40,
        table1_gpt_source=Table1SourceCohort(
            backbone="gpt-5.4-mini-2026-03-17",
            protocol_id="coevolve-eval-v83-v33-efficient",
            method_commit="b" * 40,
            seeds=(200, 201, 202, 203, 204),
            condition_fingerprint="3" * 64,
            result_manifest_hash="4" * 64,
            provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
            provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            shared_prompt_contract_fingerprint=prompt,
            decision_prompt_fingerprint=prompt,
            full_method_compatibility_fingerprint=full_method.fingerprint,
        ),
        table1_gemini_source=Table1SourceCohort(
            backbone="gemini-3.5-flash",
            protocol_id="coevolve-eval-v83-v33-efficient-gemini",
            method_commit="c" * 40,
            seeds=(200, 201, 202, 203, 204),
            condition_fingerprint="b" * 64,
            result_manifest_hash="c" * 64,
            provider_route_id=SECONDARY_GEMINI_PROVIDER_ROUTE.route_id,
            provider_route_fingerprint=SECONDARY_GEMINI_PROVIDER_ROUTE.fingerprint,
            shared_prompt_contract_fingerprint=prompt,
            decision_prompt_fingerprint=prompt,
            full_method_compatibility_fingerprint=full_method.fingerprint,
        ),
        gpt_strategy_bank_hash="5" * 64,
        gemini_strategy_bank_hash="6" * 64,
        target_provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        target_provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        observation_contract_fingerprint="7" * 64,
        task_instruction_fingerprint="8" * 64,
        shared_prompt_contract_fingerprint=prompt,
        method_decision_prompt_pins=tuple(
            MethodDecisionPromptPin(variant, prompt) for variant in MethodVariant
        ),
        skill_api_fingerprint="a" * 64,
        full_method_compatibility=full_method,
    )


def _write_run(root, *, seed, status="SUCCESS", success=True, physical=7):
    root.mkdir(parents=True)
    plan = {
        "table": "table2",
        "condition": "full",
        "method": "coevolve_full",
        "scene": "base",
        "memory_condition": "online_only",
        "target_model": "gpt-exact",
        "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        "seed": seed,
        "protocol_id": "non-table1-v1",
        "base_method_commit": "abc123",
        "condition_fingerprint": "condition-hash",
        "invariant_fingerprint": "invariant-hash",
        "scene_fingerprint": "scene-hash",
        "terminal_fingerprint": "terminal-hash",
        "observation_contract_fingerprint": "a" * 64,
        "task_instruction_fingerprint": "b" * 64,
        "shared_prompt_contract_fingerprint": "c" * 64,
        "decision_prompt_fingerprint": "e" * 64,
        "skill_api_fingerprint": "d" * 64,
        "full_method_compatibility_fingerprint": "f" * 64,
    }
    metrics = {
        "task_success": success,
        "physical_iterations": physical,
        "accepted_step_count": max(0, physical - 2),
        "rejected_step_count": min(2, physical),
        "accepted_setup_count": 1 if physical >= 3 else 0,
        "accepted_progress_count": max(0, physical - 3),
        "elapsed_s": 12.0,
        "phase_completion": {"completed": 6 if success else 3, "total": 6, "fraction": 1.0 if success else 0.5},
        "failure_counts": {"payload_drop": 1},
        "mutation_counts": {"strategy_mutation": 2},
        "model": {
            "llm_calls": 3,
            "vlm_calls": 2,
            "input_tokens": 10,
            "output_tokens": 4,
            "billing_amount_usd": 0.01,
            "billing_reconciliation_status": "complete",
        },
        "terminal": {"status": status},
    }
    manifest = {
        "status": status,
        "experiment": {
            "condition_fingerprint": "condition-hash",
            "invariant_fingerprint": "invariant-hash",
            "observation_contract_fingerprint": "a" * 64,
            "task_instruction_fingerprint": "b" * 64,
            "shared_prompt_contract_fingerprint": "c" * 64,
            "decision_prompt_fingerprint": "e" * 64,
            "skill_api_fingerprint": "d" * 64,
            "full_method_compatibility_fingerprint": "f" * 64,
        },
        "codebase": {"git_head": "abc123"},
        "model": {
            "provider": "provider_a",
            "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
            "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            "requested": "gpt-exact",
            "resolved": "gpt-exact",
        },
    }
    (root / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_table4_memory_evidence(root):
    memory = root / "memory"
    seed_path = memory / "seed_strategy_bank.json"
    overlay_path = memory / "online_strategy_overlay.json"
    action_path = memory / "action_exemplars.json"
    provenance_path = memory / "strategy_bank.provenance.json"
    bootstrap_path = memory / "memory_bootstrap_audit.json"
    snapshot_path = memory / TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME
    runtime_path = memory / "memory_runtime_audit.json"
    plan_path = (root.parent / "suite_plan.json").resolve()
    plan_path.write_text('{"kind":"test-suite-plan"}', encoding="utf-8")
    plan_hash = sha256(plan_path.read_bytes()).hexdigest()
    seed = {
        "schema_version": 1,
        "records": [
            {
                "phase_trigger": "approach",
                "failure_condition": "coordination regression",
                "coordination_pattern": "front-back push",
                "verified_outcome": "accepted",
                "avoid_rule": "avoid payload instability",
                "repair_direction": "reduce step length",
            }
        ],
    }
    overlay = empty_online_strategy_overlay()
    actions = empty_action_exemplar_memory()
    seed_hash = semantic_sha256(seed)
    overlay_hash = semantic_sha256(overlay)
    action_hash = semantic_sha256(actions)
    _write_json(seed_path, seed)
    seed_path.chmod(0o444)
    _write_json(overlay_path, overlay)
    _write_json(action_path, actions)
    _write_json(
        memory / "accepted_episode_manifest.json",
        {
            "kind": "accepted_checkpoint_episode",
            "schema_version": 1,
            "initial_checkpoint": "c0",
            "latest_checkpoint": "c0",
            "initial_state": {},
            "steps": [],
        },
    )
    _write_json(
        memory / "failure_cache.json",
        {
            "kind": "run_local_failure_cache",
            "schema_version": 1,
            "failures": [],
        },
    )
    provenance = {
        "kind": "strategy_transfer_provenance",
        "schema_version": 3,
        "bank_sha256": seed_hash,
        "source_model": "gpt-source",
        "source_provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "source_provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        "source_scene": "base",
        "source_protocol": "table1-gpt-v83",
        "source_method_commit": "a" * 40,
        "source_seeds": [200, 201, 202, 203, 204],
        "source_condition_fingerprint": "e" * 64,
        "source_result_manifest_hash": "f" * 64,
        "policy": FROZEN_STRATEGY_BANK_POLICY.as_dict(),
        "summary": {},
        "sources": [],
    }
    _write_json(provenance_path, provenance)
    bootstrap = {
        "kind": "table4_memory_bootstrap_audit",
        "schema_version": 1,
        "condition": "GPT→GPT",
        "source_model": "gpt-source",
        "source_model_family": "gpt",
        "target_model": "gpt-exact",
        "target_model_family": "gpt",
        "seed": {
            "path": seed_path.name,
            "sha256": seed_hash,
            "record_count": 1,
            "immutable": True,
            "mode": "0444",
            "provenance_path": str(provenance_path.resolve()),
            "provenance_sha256": semantic_sha256(provenance),
        },
        "online_overlay": {
            "path": overlay_path.name,
            "sha256_at_bootstrap": overlay_hash,
            "record_count_at_bootstrap": 0,
            "separate_from_seed": True,
            "target_run_local_updates_allowed": True,
        },
        "fresh_target_state": {
            "action_exemplars_empty": True,
            "accepted_manifest_steps_empty": True,
            "failure_cache_empty": True,
            "physical_state": "s0",
            "checkpoint": "c0",
        },
        "formal_binding": {
            "run_id": "table4__gpt_to_gpt__seed_000",
            "protocol_id": "non-table1-v1",
            "condition_fingerprint": "condition-hash",
            "strategy_bank_sha256": seed_hash,
            "source_model": "gpt-source",
            "source_provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
            "source_provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            "target_model": "gpt-exact",
            "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
            "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            "plan_path": str(plan_path),
            "plan_sha256": plan_hash,
            "observation_contract_fingerprint": "a" * 64,
            "task_instruction_fingerprint": "b" * 64,
            "shared_prompt_contract_fingerprint": "c" * 64,
            "decision_prompt_fingerprint": "e" * 64,
            "skill_api_fingerprint": "d" * 64,
            "full_method_compatibility_fingerprint": "f" * 64,
        },
    }
    _write_json(bootstrap_path, bootstrap)
    snapshot_payload = {
        "kind": TRANSFER_BOOTSTRAP_SNAPSHOT_KIND,
        "schema_version": TRANSFER_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION,
        "condition": "GPT→GPT",
        "target_model": "gpt-exact",
        "seed_bank_sha256": seed_hash,
        "formal_binding_sha256": sha256_json(bootstrap["formal_binding"]),
        "initial_files": [
            {
                "filename": name,
                "sha256": sha256((memory / name).read_bytes()).hexdigest(),
                "mutable_after_dispatch": name in TRANSFER_BOOTSTRAP_MUTABLE_FILES,
            }
            for name in TRANSFER_BOOTSTRAP_INITIAL_FILES
        ],
    }
    _write_json(
        snapshot_path,
        {
            **snapshot_payload,
            "snapshot_fingerprint": sha256_json(snapshot_payload),
        },
    )
    bootstrap_path.chmod(0o444)
    snapshot_path.chmod(0o444)
    bootstrap_hash = semantic_sha256(bootstrap)
    runtime = {
        "kind": "table4_memory_runtime_audit",
        "schema_version": 1,
        "lifecycle_state": "finalized",
        "condition": "GPT→GPT",
        "target_model": "gpt-exact",
        "run_id": "table4__gpt_to_gpt__seed_000",
        "protocol_id": "non-table1-v1",
        "condition_fingerprint": "condition-hash",
        "fairness_fingerprints": {
            "observation_contract_fingerprint": "a" * 64,
            "task_instruction_fingerprint": "b" * 64,
            "shared_prompt_contract_fingerprint": "c" * 64,
            "decision_prompt_fingerprint": "e" * 64,
            "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            "skill_api_fingerprint": "d" * 64,
            "full_method_compatibility_fingerprint": "f" * 64,
        },
        "bootstrap_audit_path": bootstrap_path.name,
        "bootstrap_audit_sha256": bootstrap_hash,
        "plan_path": str(plan_path),
        "plan_sha256": plan_hash,
        "seed": {
            "path": seed_path.name,
            "expected_sha256": seed_hash,
            "preflight_sha256": seed_hash,
            "current_sha256": seed_hash,
            "final_sha256": seed_hash,
            "immutable_preflight": True,
            "immutable_final": True,
        },
        "online_overlay": {
            "path": overlay_path.name,
            "initial_sha256": overlay_hash,
            "current_sha256": overlay_hash,
            "final_sha256": overlay_hash,
            "initial_record_count": 0,
            "current_record_count": 0,
            "final_record_count": 0,
        },
        "action_exemplars": {
            "path": action_path.name,
            "initial_sha256": action_hash,
            "current_sha256": action_hash,
            "final_sha256": action_hash,
            "initial_count": 0,
            "current_count": 0,
            "final_count": 0,
        },
        "operations": {
            "strategy_retrievals": 0,
            "exemplar_retrievals": 0,
            "strategy_promotions": 0,
            "exemplar_promotions": 0,
            "promotion_noops": 0,
        },
        "violations": [],
        "finalized": True,
    }
    _write_json(runtime_path, runtime)
    return (
        seed_hash,
        bootstrap_path,
        runtime_path,
        seed_path,
        plan_path,
        plan_hash,
    )


def _table4_expected(seed_hash, plan_path, plan_hash):
    return {
        "run_id": "table4__gpt_to_gpt__seed_000",
        "table": "table4_memory_transfer",
        "condition": "gpt_to_gpt",
        "method": "coevolve_full",
        "scene": "mirrored",
        "memory_condition": "gpt_to_gpt",
        "source_model": "gpt-source",
        "source_provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "source_provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        "target_model": "gpt-exact",
        "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        "seed": 0,
        "protocol_id": "non-table1-v1",
        "base_method_commit": "abc123",
        "condition_fingerprint": "condition-hash",
        "invariant_fingerprint": "invariant-hash",
        "scene_fingerprint": "scene-hash",
        "terminal_fingerprint": "terminal-hash",
        "strategy_bank_sha256": seed_hash,
        "strategy_bank": {
            "source_backbone": "gpt-source",
            "source_protocol_id": "table1-gpt-v83",
            "source_method_commit": "a" * 40,
            "source_seeds": [200, 201, 202, 203, 204],
            "source_condition_fingerprint": "e" * 64,
            "source_result_manifest_hash": "f" * 64,
            "source_provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
            "source_provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            "selector": "table1/gpt/bank",
            "sha256": seed_hash,
        },
        "observation_contract_fingerprint": "a" * 64,
        "task_instruction_fingerprint": "b" * 64,
        "shared_prompt_contract_fingerprint": "c" * 64,
        "decision_prompt_fingerprint": "e" * 64,
        "skill_api_fingerprint": "d" * 64,
        "full_method_compatibility_fingerprint": "f" * 64,
        "plan_path": str(plan_path),
        "plan_sha256": plan_hash,
    }


def test_collect_and_summarize_registered_five_run_condition(tmp_path):
    records = []
    for seed in range(5):
        run = tmp_path / f"seed_{seed:03d}"
        _write_run(run, seed=seed, success=seed < 4, status="SUCCESS" if seed < 4 else "SYSTEM_STALL", physical=seed + 5)
        records.append(collect_run(run))

    summary = summarize_condition(
        records,
        expected_seeds=range(5),
        require_complete=True,
    )

    assert summary["success_over_expected"] == "4/5"
    assert summary["cpt_at_80"] == pytest.approx((5 + 6 + 7 + 8 + 80) / 5)
    assert summary["cpt_at_80_std"] > 0
    assert summary["fm_calls"] == 5
    assert summary["fm_calls_std"] == 0
    assert summary["billing_complete"] is True
    assert summary["billing_complete_runs"] == 5
    assert summary["cost_usd_total"] == pytest.approx(0.05)


def test_condition_cost_is_na_not_zero_when_one_run_lacks_billing(tmp_path):
    records = []
    for seed in range(5):
        run = tmp_path / f"seed_{seed:03d}"
        _write_run(run, seed=seed)
        if seed == 4:
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            metrics["model"].pop("billing_amount_usd")
            metrics["model"]["billing_reconciliation_status"] = "unavailable"
            _write_json(run / "metrics.json", metrics)
        records.append(collect_run(run))

    summary = summarize_condition(
        records,
        expected_seeds=range(5),
        require_complete=True,
    )

    assert summary["complete"] is True
    assert summary["billing_complete"] is False
    assert summary["billing_complete_runs"] == 4
    assert summary["cost_usd_total"] is None


def test_collector_requires_both_fm_call_modalities(tmp_path):
    run = tmp_path / "seed_200"
    _write_run(run, seed=200)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["model"].pop("vlm_calls")
    _write_json(run / "metrics.json", metrics)

    record = collect_run(run)

    assert "FM_CALL_MODALITY_COUNT_MISSING" in record.invalid_reasons
    assert record.fm_calls is None


def test_collector_rejects_zero_foundation_model_calls(tmp_path):
    run = tmp_path / "seed_200"
    _write_run(run, seed=200)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["model"]["llm_calls"] = 0
    metrics["model"]["vlm_calls"] = 0
    _write_json(run / "metrics.json", metrics)

    record = collect_run(run)

    assert not record.protocol_valid
    assert "FM_CALL_COUNT_ZERO" in record.invalid_reasons


def test_collector_keeps_physical_run_valid_when_billing_is_unavailable(tmp_path):
    run = tmp_path / "seed_200"
    _write_run(run, seed=200)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["model"].pop("billing_amount_usd")
    metrics["model"]["billing_reconciliation_status"] = "unavailable"
    _write_json(run / "metrics.json", metrics)

    record = collect_run(run)

    assert record.protocol_valid
    assert record.cost_usd is None
    assert not record.cost_complete
    assert record.billing_reconciliation_status == "unavailable"
    assert "COST_MISSING_OR_INVALID" not in record.invalid_reasons


def test_collector_separates_invalid_billing_from_protocol_validity(tmp_path):
    run = tmp_path / "seed_200"
    _write_run(run, seed=200)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["model"]["billing_amount_usd"] = -1.0
    _write_json(run / "metrics.json", metrics)

    record = collect_run(run)

    assert record.protocol_valid
    assert record.cost_usd is None
    assert not record.cost_complete
    assert "NEGATIVE_COST" in record.billing_reasons
    assert "NEGATIVE_COST" not in record.invalid_reasons


def test_collector_rejects_inconsistent_phase_and_accepted_kind_accounting(tmp_path):
    run = tmp_path / "seed_200"
    _write_run(run, seed=200)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["accepted_progress_count"] -= 1
    metrics["phase_completion"]["fraction"] = 0.5
    _write_json(run / "metrics.json", metrics)

    record = collect_run(run)

    assert "ACCEPTED_KIND_ACCOUNTING_MISMATCH" in record.invalid_reasons
    assert "PHASE_COMPLETION_ACCOUNTING_MISMATCH" in record.invalid_reasons
    assert "SUCCESS_WITH_INCOMPLETE_PHASE_PLAN" in record.invalid_reasons


def test_collector_rejects_short_budget_terminal_and_nonfinite_usage(tmp_path):
    run = tmp_path / "seed_200"
    _write_run(
        run,
        seed=200,
        status="PHYSICAL_BUDGET_EXHAUSTED",
        success=False,
        physical=7,
    )
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["elapsed_s"] = "NaN"
    _write_json(run / "metrics.json", metrics)

    record = collect_run(run)

    assert "PHYSICAL_BUDGET_TERMINAL_COUNT_MISMATCH" in record.invalid_reasons
    assert "ELAPSED_TIME_MISSING_OR_INVALID" in record.invalid_reasons


def test_incomplete_condition_never_emits_formal_table_statistics(tmp_path):
    rows = []
    for seed in (200, 201):
        run = tmp_path / f"seed_{seed:03d}"
        _write_run(run, seed=seed)
        rows.append(collect_run(run))

    summary = summarize_condition(rows)

    assert summary["complete"] is False
    assert summary["successful_runs"] is None
    assert summary["success_rate"] is None
    assert summary["success_rate_wilson_95_low"] is None
    assert summary["cpt_at_80"] is None
    assert summary["phase_completion"] is None
    assert summary["observed_partial_diagnostics"]["successful_runs"] == 2


def test_formal_collection_requires_manifest_pinned_terminal_evidence(tmp_path):
    run = tmp_path / "seed_200"
    _write_run(run, seed=200)
    plan_path = run / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.update(
        {
            "run_id": "table2__full__seed_200",
            "condition_fingerprint": "a" * 64,
            "terminal_fingerprint": "b" * 64,
        }
    )
    _write_json(plan_path, plan)
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment"].update(
        {
            "condition_fingerprint": "a" * 64,
            "terminal_fingerprint": "b" * 64,
        }
    )
    _write_json(manifest_path, manifest)

    record = collect_run(
        run,
        expected={**plan, "require_terminal_evidence": True},
    )

    assert "TERMINAL_EVIDENCE_PIN_MISSING" in record.invalid_reasons


def test_formal_collection_requires_plan_bound_native_memory_receipt(tmp_path):
    run = tmp_path / "seed_200"
    _write_run(run, seed=200)
    plan_path = (tmp_path / "suite_plan.json").resolve()
    plan_path.write_text('{"kind":"test-suite-plan"}\n', encoding="utf-8")
    plan_sha256 = sha256(plan_path.read_bytes()).hexdigest()
    run_plan_path = run / "run_plan.json"
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    plan.update(
        {
            "run_id": "table2_method_comparison__coevolve_full__seed_200",
            "table": "table2_method_comparison",
            "memory_condition": "empty",
            "memory_policy": "online_from_empty",
            "base_method_commit": "a" * 40,
        }
    )
    _write_json(run_plan_path, plan)
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["codebase"]["git_head"] = "a" * 40
    _write_json(manifest_path, manifest)
    expected = {
        **plan,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "require_native_memory_evidence": True,
    }
    collect_module = importlib.import_module("coevolve_experiments.reporting.collect")
    native_expectation = collect_module._native_memory_expectation(expected)
    assert native_expectation is not None
    mode, gates, binding = native_expectation

    def no_store(_path):
        raise AssertionError("empty native finalization must not construct a store")

    native_root = run / "memory" / "v83_native"
    bridge = V83NativeMemoryBridge(
        strategy_path=(native_root / "strategy_specs.json").resolve(),
        action_exemplars_path=(native_root / "action_exemplars.json").resolve(),
        audit_path=(native_root / V83_NATIVE_AUDIT_FILENAME).resolve(),
        mode=mode,
        gates=V83MemoryOperationGates(**gates),
        strategy_store_factory=no_store,
        action_store_factory=no_store,
        run_binding=binding,
    )
    bridge.finalize(terminal_reason="SUCCESS")
    audit_sha256 = sha256(
        (native_root / V83_NATIVE_AUDIT_FILENAME).read_bytes()
    ).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result_artifact_hashes"] = {
        "v83_native_memory_runtime_audit_sha256": audit_sha256,
    }
    _write_json(manifest_path, manifest)

    record = collect_run(run, expected=expected)

    assert not any(
        reason.startswith("V83_NATIVE") for reason in record.invalid_reasons
    ), record.invalid_reasons
    assert record.artifact_hashes[
        "memory/v83_native/v83_native_memory_runtime_audit.json"
    ] == audit_sha256

    manifest["result_artifact_hashes"][
        "v83_native_memory_runtime_audit_sha256"
    ] = "0" * 64
    _write_json(manifest_path, manifest)
    tampered_pin = collect_run(run, expected=expected)
    assert "V83_NATIVE_MEMORY_AUDIT_PIN_MISMATCH" in tampered_pin.invalid_reasons


def test_post_table1_summary_defaults_to_registered_seeds_200_through_204(
    tmp_path,
):
    records = []
    for seed in range(200, 205):
        run = tmp_path / f"seed_{seed:03d}"
        _write_run(run, seed=seed)
        records.append(collect_run(run))

    summary = summarize_condition(records, require_complete=True)

    assert summary["complete"] is True
    assert summary["expected_seeds"] == [200, 201, 202, 203, 204]
    assert summary["success_over_expected"] == "5/5"


def test_infrastructure_invalid_is_disclosed_not_added_to_denominator(tmp_path):
    valid = tmp_path / "valid"
    invalid = tmp_path / "invalid"
    _write_run(valid, seed=0)
    _write_run(invalid, seed=0, status="PROVIDER_FATAL", success=False, physical=0)
    rows = [collect_run(valid), collect_run(invalid)]

    assert rows[1].infrastructure_invalid
    with pytest.raises(IncompleteConditionError):
        summarize_condition(rows, require_complete=True)
    appendix = build_appendix_tables(rows)
    assert len(appendix["A1_per_run_results"]) == 1
    assert appendix["A2_infrastructure_invalid_attempts"][0]["terminal_reason"] == "PROVIDER_FATAL"


def test_fingerprint_or_commit_mismatch_invalidates_a_row(tmp_path):
    run = tmp_path / "run"
    _write_run(run, seed=0)

    record = collect_run(
        run,
        expected={
            "condition_fingerprint": "another-condition",
            "invariant_fingerprint": "invariant-hash",
            "base_method_commit": "different-commit",
        },
    )

    assert not record.protocol_valid
    assert "CONDITION_FINGERPRINT_MISMATCH" in record.invalid_reasons
    assert "BASE_METHOD_COMMIT_MISMATCH" in record.invalid_reasons


def test_external_plan_metadata_is_compared_not_used_to_overwrite_actual(tmp_path):
    run = tmp_path / "run"
    _write_run(run, seed=1)
    plan_path = run / "run_plan.json"
    actual = json.loads(plan_path.read_text(encoding="utf-8"))
    actual.update(
        {
            "table": "wrong_table",
            "condition": "wrong_condition",
            "method": "wrong_method",
            "scene": "wrong_scene",
            "memory_condition": "wrong_memory",
        }
    )
    plan_path.write_text(json.dumps(actual), encoding="utf-8")

    record = collect_run(
        run,
        expected={
            "table": "table2",
            "condition": "full",
            "method": "coevolve_full",
            "scene": "base",
            "memory_condition": "online_only",
            "seed": 0,
        },
    )

    assert not record.protocol_valid
    assert record.table == "wrong_table"
    assert record.condition == "wrong_condition"
    assert {
        "TABLE_MISMATCH",
        "CONDITION_MISMATCH",
        "METHOD_MISMATCH",
        "SCENE_MISMATCH",
        "MEMORY_CONDITION_MISMATCH",
        "SEED_MISMATCH",
    }.issubset(record.invalid_reasons)


def test_external_plan_requires_all_live_contract_fingerprints(tmp_path):
    run = tmp_path / "run"
    _write_run(run, seed=0)
    plan_path = run / "run_plan.json"
    expected = json.loads(plan_path.read_text(encoding="utf-8"))

    valid = collect_run(run, expected=expected)
    assert valid.protocol_valid, valid.invalid_reasons

    missing_expectation = dict(expected)
    del missing_expectation["decision_prompt_fingerprint"]
    missing = collect_run(run, expected=missing_expectation)
    assert not missing.protocol_valid
    assert (
        "DECISION_PROMPT_FINGERPRINT_EXPECTATION_MISSING"
        in missing.invalid_reasons
    )

    manifest_path = run / "run_manifest.json"
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    del actual["experiment"]["skill_api_fingerprint"]
    _write_json(manifest_path, actual)
    unbound_live_api = collect_run(run, expected=expected)
    assert not unbound_live_api.protocol_valid
    assert (
        "SKILL_API_FINGERPRINT_MISMATCH"
        in unbound_live_api.invalid_reasons
    )


def test_task_success_requires_a_json_boolean(tmp_path):
    run = tmp_path / "run"
    _write_run(run, seed=0, status="SUCCESS", success="false")

    record = collect_run(run)

    assert not record.task_success
    assert not record.protocol_valid
    assert "TASK_SUCCESS_NOT_BOOLEAN" in record.invalid_reasons
    assert "SUCCESS_STATUS_WITHOUT_TASK_SUCCESS" in record.invalid_reasons


def test_manifest_and_metrics_terminal_status_conflict_invalidates_row(tmp_path):
    run = tmp_path / "run"
    _write_run(run, seed=0, status="SUCCESS", success=True)
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["terminal"]["status"] = "SYSTEM_STALL"
    _write_json(metrics_path, metrics)

    record = collect_run(run)

    assert not record.protocol_valid
    assert "TERMINAL_STATUS_CONFLICT" in record.invalid_reasons


def _write_table4_run(root):
    _write_run(root, seed=0)
    plan_path = root / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.update(
        {
            "table": "table4_memory_transfer",
            "condition": "gpt_to_gpt",
            "method": "coevolve_full",
            "scene": "mirrored",
            "memory_condition": "gpt_to_gpt",
            "strategy_bank": {
                "source_backbone": "gpt-source",
            },
        }
    )
    _write_json(plan_path, plan)
    return _write_table4_memory_evidence(root)


def test_table4_collection_requires_external_plan_and_finalized_memory_evidence(
    tmp_path,
):
    run = tmp_path / "run"
    (
        seed_hash,
        bootstrap_path,
        runtime_path,
        _,
        plan_path,
        plan_hash,
    ) = _write_table4_run(run)

    self_describing = collect_run(run)
    assert not self_describing.protocol_valid
    assert (
        "TABLE4_STRATEGY_BANK_EXPECTATION_MISSING"
        in self_describing.invalid_reasons
    )

    expected = _table4_expected(seed_hash, plan_path, plan_hash)
    valid = collect_run(run, expected=expected)
    assert valid.protocol_valid, valid.invalid_reasons
    assert valid.strategy_bank_sha256 == seed_hash
    assert {
        "memory_bootstrap_audit",
        "memory_bootstrap_snapshot",
        "memory_runtime_audit",
    }.issubset(valid.artifact_hashes)

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    del runtime["seed"]["preflight_sha256"]
    _write_json(runtime_path, runtime)
    missing_loaded_seed = collect_run(run, expected=expected)
    assert not missing_loaded_seed.protocol_valid
    assert any(
        "PREFLIGHT_SHA256_MISMATCH" in reason
        for reason in missing_loaded_seed.invalid_reasons
    )

    runtime["seed"]["preflight_sha256"] = seed_hash
    runtime["online_overlay"]["initial_record_count"] = 1
    _write_json(runtime_path, runtime)
    contaminated_overlay = collect_run(run, expected=expected)
    assert not contaminated_overlay.protocol_valid
    assert (
        "TABLE4_OVERLAY_INITIAL_COUNT_NOT_ZERO"
        in contaminated_overlay.invalid_reasons
    )

    runtime["online_overlay"]["initial_record_count"] = 0
    runtime["seed"]["immutable_final"] = False
    _write_json(runtime_path, runtime)
    mutable_seed = collect_run(run, expected=expected)
    assert not mutable_seed.protocol_valid
    assert (
        "TABLE4_RUNTIME_SEED_FINAL_IMMUTABILITY_UNPROVEN"
        in mutable_seed.invalid_reasons
    )

    runtime["seed"]["immutable_final"] = True
    _write_json(runtime_path, runtime)
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["source_model"] = "tampered-source"
    bootstrap_path.chmod(0o644)
    _write_json(bootstrap_path, bootstrap)
    tampered_bootstrap = collect_run(run, expected=expected)
    assert not tampered_bootstrap.protocol_valid
    assert {
        "TABLE4_BOOTSTRAP_SOURCE_MODEL_MISMATCH",
        "TABLE4_RUNTIME_BOOTSTRAP_HASH_MISMATCH",
    }.issubset(tampered_bootstrap.invalid_reasons)


def test_table4_collection_rejects_shared_symlinked_memory_root(tmp_path):
    run = tmp_path / "run"
    seed_hash, _, _, _, plan_path, plan_hash = _write_table4_run(run)
    shared = tmp_path / "shared-memory"
    (run / "memory").rename(shared)
    (run / "memory").symlink_to(shared, target_is_directory=True)

    record = collect_run(
        run,
        expected=_table4_expected(seed_hash, plan_path, plan_hash),
    )

    assert not record.protocol_valid
    assert "TABLE4_MEMORY_PATHS_NOT_ISOLATED" in record.invalid_reasons


@pytest.mark.parametrize("symlink_kind", ("run_dir", "ancestor"))
def test_table4_collection_rejects_symlinked_run_path(tmp_path, symlink_kind):
    real_parent = tmp_path / "real-parent"
    real_run = real_parent / "run"
    seed_hash, _, _, _, plan_path, plan_hash = _write_table4_run(real_run)

    if symlink_kind == "run_dir":
        inspected_run = tmp_path / "linked-run"
        inspected_run.symlink_to(real_run, target_is_directory=True)
    else:
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        inspected_run = linked_parent / "run"

    record = collect_run(
        inspected_run,
        expected=_table4_expected(seed_hash, plan_path, plan_hash),
    )

    assert not record.protocol_valid
    assert "TABLE4_MEMORY_PATHS_NOT_ISOLATED" in record.invalid_reasons


@pytest.mark.parametrize(
    ("field", "value"),
    (("physical_state", "s1"), ("checkpoint", "c1")),
)
def test_table4_collection_requires_exact_fresh_s0_c0(tmp_path, field, value):
    run = tmp_path / "run"
    seed_hash, bootstrap_path, _, _, plan_path, plan_hash = _write_table4_run(run)
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["fresh_target_state"][field] = value
    bootstrap_path.chmod(0o644)
    _write_json(bootstrap_path, bootstrap)

    record = collect_run(
        run,
        expected=_table4_expected(seed_hash, plan_path, plan_hash),
    )

    assert not record.protocol_valid
    assert "TABLE4_BOOTSTRAP_FRESH_STATE_UNPROVEN" in record.invalid_reasons


@pytest.mark.parametrize(
    ("section", "count_field", "reason"),
    (
        ("online_overlay", "current_record_count", "TABLE4_OVERLAY_CURRENT_COUNT_MISMATCH"),
        ("action_exemplars", "current_count", "TABLE4_ACTION_EXEMPLARS_CURRENT_COUNT_MISMATCH"),
    ),
)
def test_table4_collection_checks_current_counts_against_store_bytes(
    tmp_path,
    section,
    count_field,
    reason,
):
    run = tmp_path / "run"
    seed_hash, _, runtime_path, _, plan_path, plan_hash = _write_table4_run(run)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime[section][count_field] = 999
    _write_json(runtime_path, runtime)

    record = collect_run(
        run,
        expected=_table4_expected(seed_hash, plan_path, plan_hash),
    )

    assert not record.protocol_valid
    assert reason in record.invalid_reasons


@pytest.mark.parametrize(
    ("operation", "reason"),
    (
        ("strategy_promotions", "TABLE4_STRATEGY_PROMOTION_COUNT_MISMATCH"),
        ("exemplar_promotions", "TABLE4_EXEMPLAR_PROMOTION_COUNT_MISMATCH"),
    ),
)
def test_table4_collection_binds_promotion_counts_to_final_store_counts(
    tmp_path,
    operation,
    reason,
):
    run = tmp_path / "run"
    seed_hash, _, runtime_path, _, plan_path, plan_hash = _write_table4_run(run)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["operations"][operation] = 1
    _write_json(runtime_path, runtime)

    record = collect_run(
        run,
        expected=_table4_expected(seed_hash, plan_path, plan_hash),
    )

    assert not record.protocol_valid
    assert reason in record.invalid_reasons


def test_table4_collection_rejects_uppercase_provenance_sha256(tmp_path):
    run = tmp_path / "run"
    seed_hash, bootstrap_path, _, _, plan_path, plan_hash = _write_table4_run(run)
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["seed"]["provenance_sha256"] = bootstrap["seed"][
        "provenance_sha256"
    ].upper()
    bootstrap_path.chmod(0o644)
    _write_json(bootstrap_path, bootstrap)

    record = collect_run(
        run,
        expected=_table4_expected(seed_hash, plan_path, plan_hash),
    )

    assert not record.protocol_valid
    assert "TABLE4_BOOTSTRAP_PROVENANCE_HASH_INVALID" in record.invalid_reasons


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_protocol", "other-table1-protocol"),
        ("source_method_commit", "9" * 40),
        ("source_seeds", [200, 201, 202, 203, 999]),
        ("source_condition_fingerprint", "8" * 64),
        ("source_result_manifest_hash", "7" * 64),
    ),
)
def test_table4_collection_rejects_rehashed_source_provenance_relabelling(
    tmp_path,
    field,
    replacement,
):
    run = tmp_path / "run"
    (
        seed_hash,
        bootstrap_path,
        runtime_path,
        _,
        plan_path,
        plan_hash,
    ) = _write_table4_run(run)
    provenance_path = run / "memory" / "strategy_bank.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance[field] = replacement
    _write_json(provenance_path, provenance)

    # Rehash both self-reported audit layers to prove that consistency among
    # mutable result files is not enough; the external plan pin remains the
    # authority for the Table-1 source identity.
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["seed"]["provenance_sha256"] = semantic_sha256(provenance)
    bootstrap_path.chmod(0o644)
    _write_json(bootstrap_path, bootstrap)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["bootstrap_audit_sha256"] = semantic_sha256(bootstrap)
    _write_json(runtime_path, runtime)

    record = collect_run(
        run,
        expected=_table4_expected(seed_hash, plan_path, plan_hash),
    )

    assert not record.protocol_valid
    assert (
        "TABLE4_BOOTSTRAP_PROVENANCE_IDENTITY_MISMATCH"
        in record.invalid_reasons
    )


@pytest.mark.parametrize(
    ("field", "different"),
    (
        ("protocol_id", "other-protocol"),
        ("target_model", "other-model"),
        ("method", "other-method"),
        ("scene", "other-scene"),
        ("memory_condition", "other-memory"),
        ("condition_fingerprint", "other-fingerprint"),
        ("decision_prompt_fingerprint", "other-prompt-fingerprint"),
    ),
)
def test_same_named_conditions_with_different_frozen_identity_are_separate(
    field,
    different,
):
    base = RunRecord(
        run_dir="run-a",
        table="table2",
        condition="full",
        protocol_id="protocol",
        target_model="model",
        source_model="",
        method="full",
        scene="base",
        memory_condition="empty",
        base_method_commit="commit",
        condition_fingerprint="condition-fingerprint",
        invariant_fingerprint="invariant-fingerprint",
        scene_fingerprint="scene-fingerprint",
        terminal_fingerprint="terminal-fingerprint",
        seed=0,
        protocol_valid=True,
        task_success=True,
    )
    changed = RunRecord(**{**base.to_dict(), "run_dir": "run-b", field: different})

    summary = summarize_suite([base, changed], expected_runs=1)

    assert len(summary["conditions"]) == 2
    assert all(row["complete"] for row in summary["conditions"])
    with pytest.raises(MixedConditionError, match=field):
        summarize_condition([base, changed], expected_runs=1)


def _artifact_ref(root, path):
    raw = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _trace_policy(*, attempt, script_id, parent_script_id, primitive, semantic, roles):
    assignments = []
    robot_scripts = []
    for robot_id in ("back", "front"):
        role = roles[robot_id]
        shared = {
            "robot_id": robot_id,
            "coordination_mode": "async",
            "synchronization_group": script_id,
            "phase_id": "route_phase_1",
            "contact_intent": "maintain_payload_support",
        }
        assignments.append({**shared, "role": role})
        robot_scripts.append(
            {
                **shared,
                "script_id": f"{robot_id}_{script_id}",
                "code": f"def policy(api): api.{primitive}('{role}')",
            }
        )
    return {
        "attempt": attempt,
        "script_id": script_id,
        "parent_script_id": parent_script_id,
        "code": f"def policy(api): api.{primitive}()",
        "assignments": assignments,
        "robot_scripts": robot_scripts,
        "verification_contract": {
            "committed_subtask": {
                "phase_id": "route_phase_1",
                "subtask_id": script_id,
                "primitive": primitive,
                "semantic_objective": semantic,
                "coordination_mode": "async",
            }
        },
    }


def _model_strategy(policy):
    committed = policy["verification_contract"]["committed_subtask"]
    return {
        "primitive": committed["primitive"],
        "semantic_objective": committed["semantic_objective"],
        "coordination_mode": committed["coordination_mode"],
        "team_assignment": {
            item["robot_id"]: {"role": item["role"]}
            for item in policy["assignments"]
        },
    }


def _write_figure2_fixture(root):
    parent_dir = root / "attempt_01"
    child_dir = root / "attempt_02"
    roles_dir = root / "model_roles"
    memory_dir = root / "memory"
    for path in (parent_dir, child_dir, roles_dir, memory_dir):
        path.mkdir(parents=True, exist_ok=True)
    parent_policy = _trace_policy(
        attempt=1,
        script_id="parent_pivot",
        parent_script_id="accepted_parent",
        primitive="coordinated_pivot",
        semantic="counterclockwise",
        roles={"front": "inner", "back": "outer"},
    )
    child_policy = _trace_policy(
        attempt=2,
        script_id="child_alignment",
        parent_script_id="parent_pivot",
        primitive="coordinated_alignment",
        semantic="longitudinal",
        roles={"front": "corrector", "back": "reference"},
    )
    parent_payloads = {
        "policy.json": parent_policy,
        "evaluation.json": {
            "success": False,
            "failure_type": "formation_desync",
            "metrics": {"formation_error_m": 0.4},
            "evidence": {"source": "physics_verifier", "sample_count": 20},
        },
        "transition_decision.json": {"outcome": "REJECT_UNSAFE"},
        "evolution_record.json": {
            "checkpoint_before": "c0",
            "checkpoint_after": "",
            "rollback_performed": True,
            "controller_decision": {
                "repair_level": "strategy",
                "physical_iterations": 1,
            },
        },
        "rollback_receipt.json": {"checkpoint_id": "c0", "within_tolerance": True},
    }
    child_payloads = {
        "policy.json": child_policy,
        "evaluation.json": {
            "success": True,
            "failure_type": "none",
            "metrics": {"formation_error_m": 0.1},
            "evidence": {"source": "physics_verifier", "sample_count": 20},
        },
        "transition_decision.json": {"outcome": "ACCEPT_PROGRESS"},
        "evolution_record.json": {
            "checkpoint_before": "c0",
            "checkpoint_after": "c1",
            "rollback_performed": False,
            "controller_decision": {
                "repair_level": "none",
                "physical_iterations": 2,
            },
        },
        "pre_dispatch_restore.json": {"checkpoint_id": "c0", "within_tolerance": True},
        "execution_trace.json": {
            "attempt": 2,
            "script_id": "child_alignment",
            "status": "success",
        },
    }
    for name, payload in parent_payloads.items():
        _write_json(parent_dir / name, payload)
    for name, payload in child_payloads.items():
        _write_json(child_dir / name, payload)
    accepted_manifest = {
        "steps": [
            {
                "attempt": 2,
                "script_id": "child_alignment",
                "checkpoint_before": "c0",
                "checkpoint_after": "c1",
                "transition_outcome": "ACCEPT_PROGRESS",
            }
        ]
    }
    _write_json(memory_dir / "accepted_episode_manifest.json", accepted_manifest)

    parent_policy_path = parent_dir / "policy.json"
    child_policy_path = child_dir / "policy.json"
    delta = build_host_mutation_delta(
        parent_policy,
        child_policy,
        attempt=2,
        mutation_level="strategy_mutation",
        parent_policy_sha256=_artifact_ref(root, parent_policy_path)["sha256"],
        child_policy_sha256=_artifact_ref(root, child_policy_path)["sha256"],
    )
    delta_path = roles_dir / "mutation_effective_delta_attempt_02.json"
    _write_json(delta_path, delta)
    mutation_path = roles_dir / "mutation_operator_attempt_02.json"
    _write_json(
        mutation_path,
        {
            "role": "mutation_operator",
            "artifact_id": "attempt_02",
            "valid": True,
            "normalization_audit": {"raw_response_preserved": True},
            "parsed_response": {
                "mutation_level": "strategy_mutation",
                "parent_strategy": _model_strategy(parent_policy),
                "child_strategy": _model_strategy(child_policy),
                "host_effective_delta": delta,
            },
        },
    )
    proposal_path = roles_dir / "memory_update_attempt_02.json"
    _write_json(
        proposal_path,
        {
            "role": "memory_update",
            "artifact_id": "attempt_02",
            "valid": True,
            "parsed_response": {"promote": True, "avoid_rule": "change primitive"},
        },
    )
    promoted_record = {
        "id": "promoted_child",
        "evidence": {
            "rollout_refs": [
                {
                    "attempt": 2,
                    "script_id": "child_alignment",
                    "checkpoint_before": "c0",
                    "checkpoint_after": "c1",
                    "outcome": "accepted",
                }
            ]
        },
    }
    memory_path = memory_dir / "strategy_specs.json"
    _write_json(memory_path, {"specs": [promoted_record]})
    promotion_path = memory_dir / "promotion_attempt_02.json"
    _write_json(
        promotion_path,
        {
            "kind": "host_memory_promotion_receipt",
            "schema_version": 1,
            "producer_id": "praxis_coop.host.memory_promotion",
            "attempt": 2,
            "script_id": "child_alignment",
            "checkpoint_before": "c0",
            "checkpoint_after": "c1",
            "transition_outcome": "ACCEPT_PROGRESS",
            "memory_kind": "strategy_spec",
            "record_sha256": semantic_sha256(promoted_record),
            "policy_sha256": _artifact_ref(root, child_policy_path)["sha256"],
            "evaluation_sha256": _artifact_ref(root, child_dir / "evaluation.json")["sha256"],
            "execution_trace_sha256": _artifact_ref(root, child_dir / "execution_trace.json")["sha256"],
            "promoted": True,
        },
    )

    receipt = {
        "kind": "coevolve_figure2_trace_evidence",
        "schema_version": 1,
        "producer": {
            "producer_id": "praxis_coop.host.figure2_trace_evidence",
            "producer_version": 1,
        },
        "run_binding": {
            "protocol_id": "frozen",
            "condition": "coevolve_full",
            "target_model": "gpt-exact",
            "seed": 0,
            "base_method_commit": "a" * 40,
        },
        "parent": {
            "attempt": 1,
            "physical_iteration": 1,
            "script_id": "parent_pivot",
            "checkpoint_before": "c0",
            "artifacts": {
                "policy": _artifact_ref(root, parent_policy_path),
                "evaluation": _artifact_ref(root, parent_dir / "evaluation.json"),
                "transition_decision": _artifact_ref(root, parent_dir / "transition_decision.json"),
                "evolution_record": _artifact_ref(root, parent_dir / "evolution_record.json"),
                "rollback_receipt": _artifact_ref(root, parent_dir / "rollback_receipt.json"),
            },
        },
        "child": {
            "attempt": 2,
            "physical_iteration": 2,
            "script_id": "child_alignment",
            "parent_script_id": "parent_pivot",
            "checkpoint_before": "c0",
            "checkpoint_after": "c1",
            "artifacts": {
                "policy": _artifact_ref(root, child_policy_path),
                "evaluation": _artifact_ref(root, child_dir / "evaluation.json"),
                "transition_decision": _artifact_ref(root, child_dir / "transition_decision.json"),
                "evolution_record": _artifact_ref(root, child_dir / "evolution_record.json"),
                "pre_dispatch_restore": _artifact_ref(root, child_dir / "pre_dispatch_restore.json"),
                "execution_trace": _artifact_ref(root, child_dir / "execution_trace.json"),
                "accepted_manifest": _artifact_ref(root, memory_dir / "accepted_episode_manifest.json"),
            },
        },
        "mutation": {
            "mutation_level": "strategy_mutation",
            "model_artifact": _artifact_ref(root, mutation_path),
            "effective_delta_artifact": _artifact_ref(root, delta_path),
        },
        "promotion": {
            "proposal_artifact": _artifact_ref(root, proposal_path),
            "receipt": _artifact_ref(root, promotion_path),
            "memory_artifact": _artifact_ref(root, memory_path),
        },
    }
    receipt_path = root / "figure2_trace_evidence.json"
    _write_json(receipt_path, receipt)
    manifest = {
        "status": "SUCCESS",
        "experiment": {
            "protocol_id": "frozen",
            "condition": "coevolve_full",
            "target_model": "gpt-exact",
            "seed": 0,
            "base_method_commit": "a" * 40,
        },
        "protocol": {"protocol_id": "frozen"},
        "model": {"resolved": "gpt-exact"},
        "codebase": {"git_head": "a" * 40},
        "seed": 0,
        "result_artifact_hashes": {
            "figure2_trace_evidence_sha256": _artifact_ref(root, receipt_path)["sha256"]
        },
    }
    _write_json(root / "run_manifest.json", manifest)
    return RunRecord(
        run_dir=str(root),
        condition="coevolve_full",
        target_model="gpt-exact",
        seed=0,
        protocol_id="frozen",
        base_method_commit="a" * 40,
        protocol_valid=True,
        task_success=True,
    )


def _repin_trace_receipt(root):
    receipt_path = root / "figure2_trace_evidence.json"
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result_artifact_hashes"]["figure2_trace_evidence_sha256"] = (
        _artifact_ref(root, receipt_path)["sha256"]
    )
    _write_json(manifest_path, manifest)


def test_figure2_trace_requires_manifest_pinned_causal_lineage(tmp_path):
    root = tmp_path / "valid"
    record = _write_figure2_fixture(root)
    trace = build_figure2_trace(record)

    assert trace["causal_checkpoint"] == "c0"
    assert trace["events"][3]["mutation"]["effective"] is True
    assert trace["events"][4]["physical_iteration"] == 2
    assert trace["events"][5]["memory_update"]["proposal"]["avoid_rule"] == "change primitive"
    assert trace["events"][5]["memory_update"]["host_promotion"]["promoted"] is True
    assert len(trace["selection"]["trace_evidence_sha256"]) == 64

    missing = tmp_path / "missing-receipt"
    missing_record = _write_figure2_fixture(missing)
    (missing / "figure2_trace_evidence.json").unlink()
    with pytest.raises(TraceSelectionError, match="manifest-pinned"):
        build_figure2_trace(missing_record)

    gap = tmp_path / "iteration-gap"
    gap_record = _write_figure2_fixture(gap)
    receipt_path = gap / "figure2_trace_evidence.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["child"]["physical_iteration"] = 3
    evolution_path = gap / "attempt_02" / "evolution_record.json"
    evolution = json.loads(evolution_path.read_text(encoding="utf-8"))
    evolution["controller_decision"]["physical_iterations"] = 3
    _write_json(evolution_path, evolution)
    receipt["child"]["artifacts"]["evolution_record"] = _artifact_ref(gap, evolution_path)
    _write_json(receipt_path, receipt)
    _repin_trace_receipt(gap)
    with pytest.raises(TraceSelectionError, match="physical iterations must be consecutive"):
        build_figure2_trace(gap_record)

    unrelated = tmp_path / "unrelated-child"
    unrelated_record = _write_figure2_fixture(unrelated)
    receipt_path = unrelated / "figure2_trace_evidence.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["child"]["parent_script_id"] = "unrelated_policy"
    child_policy_path = unrelated / "attempt_02" / "policy.json"
    child_policy = json.loads(child_policy_path.read_text(encoding="utf-8"))
    child_policy["parent_script_id"] = "unrelated_policy"
    _write_json(child_policy_path, child_policy)
    receipt["child"]["artifacts"]["policy"] = _artifact_ref(unrelated, child_policy_path)
    _write_json(receipt_path, receipt)
    _repin_trace_receipt(unrelated)
    with pytest.raises(TraceSelectionError, match="does not name the rejected parent"):
        build_figure2_trace(unrelated_record)

    no_memory = tmp_path / "no-actual-promotion"
    no_memory_record = _write_figure2_fixture(no_memory)
    memory_path = no_memory / "memory" / "strategy_specs.json"
    _write_json(memory_path, {"specs": []})
    receipt_path = no_memory / "figure2_trace_evidence.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["promotion"]["memory_artifact"] = _artifact_ref(no_memory, memory_path)
    _write_json(receipt_path, receipt)
    _repin_trace_receipt(no_memory)
    with pytest.raises(TraceSelectionError, match="not uniquely present"):
        build_figure2_trace(no_memory_record)


def test_figure2_uses_host_promotion_and_derives_effective_delta(tmp_path):
    root = tmp_path / "proposal-is-not-authority"
    record = _write_figure2_fixture(root)
    proposal_path = root / "model_roles" / "memory_update_attempt_02.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["parsed_response"]["promote"] = False
    _write_json(proposal_path, proposal)
    receipt_path = root / "figure2_trace_evidence.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["promotion"]["proposal_artifact"] = _artifact_ref(root, proposal_path)
    _write_json(receipt_path, receipt)
    _repin_trace_receipt(root)
    assert build_figure2_trace(record)["events"][5]["memory_update"]["effective"]

    forged = tmp_path / "forged-delta"
    forged_record = _write_figure2_fixture(forged)
    delta_path = forged / "model_roles" / "mutation_effective_delta_attempt_02.json"
    delta = json.loads(delta_path.read_text(encoding="utf-8"))
    delta["changed_dimensions"] = ["invented.model.claim"]
    _write_json(delta_path, delta)
    mutation_path = forged / "model_roles" / "mutation_operator_attempt_02.json"
    mutation = json.loads(mutation_path.read_text(encoding="utf-8"))
    mutation["parsed_response"]["host_effective_delta"] = delta
    _write_json(mutation_path, mutation)
    receipt_path = forged / "figure2_trace_evidence.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["mutation"]["effective_delta_artifact"] = _artifact_ref(forged, delta_path)
    receipt["mutation"]["model_artifact"] = _artifact_ref(forged, mutation_path)
    _write_json(receipt_path, receipt)
    _repin_trace_receipt(forged)
    with pytest.raises(TraceSelectionError, match="changed_dimensions are not derived"):
        build_figure2_trace(forged_record)


def test_reused_rows_are_hash_checked_aliases_not_copies():
    lock = _reuse_lock("frozen")
    reuse = build_run_plan("core", lock).reuses[-1]
    expected = {item.name: item.sha256 for item in reuse.expected_hashes}
    source = [
        RunRecord(
            run_dir=f"source/seed_{seed:03d}",
            table="table3_scene_generalization",
            method="coevolve_full",
            scene="mirrored",
            memory_condition="empty",
            target_model="gpt-5.4-mini-2026-03-17",
            provider_route_id=reuse.source_selector.provider_route_id,
            provider_route_fingerprint=expected["provider_route"],
            seed=seed,
            protocol_id="frozen",
            base_method_commit="a" * 40,
            condition_fingerprint=expected["condition"],
            invariant_fingerprint=expected["invariant"],
            scene_fingerprint=expected["scene"],
            terminal_fingerprint=expected["terminal"],
            observation_contract_fingerprint=expected["observation_contract"],
            task_instruction_fingerprint=expected["task_instruction"],
            shared_prompt_contract_fingerprint=expected[
                "shared_prompt_contract"
            ],
            decision_prompt_fingerprint=expected["decision_prompt"],
            skill_api_fingerprint=expected["skill_api"],
            full_method_compatibility_fingerprint=expected[
                "full_method_compatibility"
            ],
            protocol_valid=True,
        )
        for seed in range(200, 205)
    ]

    aliased = alias_reused_condition(reuse, source)

    assert {row.table for row in aliased} == {"table4_memory_transfer"}
    assert {row.condition for row in aliased} == {"empty"}
    assert [row.run_dir for row in aliased] == [row.run_dir for row in source]

    wrong_commit = list(source)
    wrong_commit[0] = RunRecord(
        **{**wrong_commit[0].to_dict(), "base_method_commit": "d" * 40}
    )
    with pytest.raises(ReuseResolutionError, match="source method commit differs"):
        alias_reused_condition(reuse, wrong_commit)

    fairness_drift = list(source)
    fairness_drift[0] = RunRecord(
        **{
            **fairness_drift[0].to_dict(),
            "decision_prompt_fingerprint": "0" * 64,
        }
    )
    with pytest.raises(ReuseResolutionError, match="decision_prompt fingerprints differ"):
        alias_reused_condition(reuse, fairness_drift)

    source[0] = RunRecord(**{**source[0].to_dict(), "scene_fingerprint": "0" * 64})
    with pytest.raises(ReuseResolutionError, match="scene fingerprints differ"):
        alias_reused_condition(reuse, source)

    wrong_table = [
        RunRecord(**{**row.to_dict(), "table": "table2_method_comparison"})
        for row in source[1:]
    ]
    wrong_table.append(
        RunRecord(
            **{
                **source[0].to_dict(),
                "scene_fingerprint": expected["scene"],
                "table": "table2_method_comparison",
            }
        )
    )
    with pytest.raises(ReuseResolutionError, match="source table differs"):
        alias_reused_condition(reuse, wrong_table)


def test_external_table1_reuse_uses_its_own_protocol_commit_and_seed_identity():
    lock = _reuse_lock("post-table1")
    reuse = build_run_plan("core", lock).reuses[0]
    selector = reuse.source_selector
    expected = {item.name: item.sha256 for item in reuse.expected_hashes}
    rows = [
        RunRecord(
            run_dir=f"table1/seed_{seed:03d}",
            table=selector.source_table,
            method=selector.method_variant.value,
            scene=selector.scene_id.value,
            memory_condition=selector.memory_condition.value,
            target_model=selector.backbone,
            provider_route_id=selector.provider_route_id,
            provider_route_fingerprint=expected["provider_route"],
            seed=seed,
            protocol_id=selector.protocol_id,
            base_method_commit=selector.method_commit,
            condition_fingerprint=expected["condition"],
            invariant_fingerprint=expected["invariant"],
            scene_fingerprint=expected["scene"],
            terminal_fingerprint=expected["terminal"],
            observation_contract_fingerprint=expected["observation_contract"],
            task_instruction_fingerprint=expected["task_instruction"],
            shared_prompt_contract_fingerprint=expected[
                "shared_prompt_contract"
            ],
            decision_prompt_fingerprint=expected["decision_prompt"],
            skill_api_fingerprint=expected["skill_api"],
            full_method_compatibility_fingerprint=expected[
                "full_method_compatibility"
            ],
            protocol_valid=True,
        )
        for seed in selector.seeds
    ]

    aliased = alias_reused_condition(
        reuse,
        rows,
        observed_hashes={
            "result_manifest": lock.table1_gpt_source.result_manifest_hash
        },
    )

    assert len(aliased) == 5
    assert {row.seed for row in aliased} == set(lock.table1_gpt_source.seeds)
    assert selector.protocol_id != lock.protocol_id
    assert selector.method_commit != lock.base_method_commit
