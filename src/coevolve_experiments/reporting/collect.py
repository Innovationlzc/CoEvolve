"""Read one run without depending on a version-specific formal summarizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from ..contracts import StrategyBankPin
from ..provider_routes import get_provider_route
from .memory_audit import validate_table4_memory_audit
from .terminal_evidence import validate_terminal_evidence
from .terminal_aux_evidence import capability_bound_auxiliary_producer_allowlist
from .transaction_accounting import validate_transaction_accounting_evidence
from .usage_evidence import (
    FM_USAGE_LEDGER_SCHEMA_VERSION,
    validate_fm_usage_evidence,
)
from .v83_native_memory_audit import (
    V83_NATIVE_MEMORY_RELATIVE_ROOT,
    validate_v83_native_memory_audit,
)


INFRASTRUCTURE_INVALID_STATUSES = frozenset(
    {
        "PROVIDER_FATAL",
        "BACKEND_FATAL",
        "HOST_CONTRACT_ERROR",
        "OPERATOR_CANCELLED",
        "MANUAL_INTERRUPTION",
        "INFRASTRUCTURE_INVALID",
    }
)
VALID_TASK_TERMINALS = frozenset(
    {
        "SUCCESS",
        "PHYSICAL_BUDGET_EXHAUSTED",
        "SYSTEM_STALL",
        "CONSECUTIVE_PAYLOAD_DROP_LIMIT",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TABLE4_ID = "table4_memory_transfer"
_TRANSFER_MEMORY_CONDITIONS = frozenset({"gpt_to_gpt", "gemini_to_gpt"})
_V83_NATIVE_AUDIT_ARTIFACT = str(
    V83_NATIVE_MEMORY_RELATIVE_ROOT / "v83_native_memory_runtime_audit.json"
)
_V83_NATIVE_AUDIT_MANIFEST_PIN = "v83_native_memory_runtime_audit_sha256"


@dataclass(frozen=True)
class RunRecord:
    """One reconciled physical run or one transparently excluded attempt."""

    run_dir: str
    table: str = ""
    condition: str = ""
    method: str = ""
    scene: str = ""
    memory_condition: str = ""
    source_model: str = ""
    target_model: str = ""
    provider: str = ""
    provider_route_id: str = ""
    provider_route_fingerprint: str = ""
    strategy_bank_sha256: str = ""
    seed: int | None = None
    protocol_id: str = ""
    base_method_commit: str = ""
    condition_fingerprint: str = ""
    invariant_fingerprint: str = ""
    scene_fingerprint: str = ""
    terminal_fingerprint: str = ""
    observation_contract_fingerprint: str = ""
    task_instruction_fingerprint: str = ""
    shared_prompt_contract_fingerprint: str = ""
    decision_prompt_fingerprint: str = ""
    skill_api_fingerprint: str = ""
    full_method_compatibility_fingerprint: str = ""
    terminal_status: str = "MISSING"
    metrics_available: bool = False
    protocol_valid: bool = False
    infrastructure_invalid: bool = False
    invalid_reasons: tuple[str, ...] = ()
    task_success: bool = False
    physical_transactions: int | None = None
    accepted_transactions: int | None = None
    rejected_transactions: int | None = None
    accepted_setup_transactions: int | None = None
    accepted_progress_transactions: int | None = None
    phase_completed: int | None = None
    phase_total: int | None = None
    phase_completion: float | None = None
    fm_calls: int | None = None
    successful_fm_calls: int | None = None
    failed_fm_calls: int | None = None
    transport_attempts: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    wall_time_s: float | None = None
    cost_usd: float | None = None
    cost_complete: bool = False
    billing_reconciliation_status: str = "unavailable"
    billing_reasons: tuple[str, ...] = ()
    failure_counts: Mapping[str, int] = field(default_factory=dict)
    mutation_counts: Mapping[str, int] = field(default_factory=dict)
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_run(
    run_dir: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
    physical_budget: int = 80,
) -> RunRecord:
    """Collect a run and apply only protocol-neutral accounting checks.

    Version-specific evidence validation belongs to the frozen runtime that
    produced the run.  This reader additionally checks the experiment
    condition and invariant fingerprints so a v73 source cannot silently be
    mixed with v77/v78 artifacts.  Formal external-plan collection also
    requires manifest evidence for the provider route, observation, task,
    shared prompt, method decision prompt, and SONIC skill-API contracts; a
    run-local plan cannot stand in for those live pins.
    """

    root = Path(run_dir)
    has_external_expected = expected is not None
    metrics = _read_json(root / "metrics.json")
    manifest = _read_json(root / "run_manifest.json")
    plan = _read_json(root / "run_plan.json")
    expected_payload = dict(expected or plan)
    terminal = metrics.get("terminal", {})
    terminal = terminal if isinstance(terminal, dict) else {}
    manifest_status = manifest.get("status")
    metrics_status = terminal.get("status")
    status = str(manifest_status or metrics_status or "MISSING")
    infrastructure_invalid = status in INFRASTRUCTURE_INVALID_STATUSES
    physical = _integer(metrics.get("physical_iterations"))
    accepted = _integer(metrics.get("accepted_step_count"))
    rejected = _integer(metrics.get("rejected_step_count"))
    accepted_setup = _integer(metrics.get("accepted_setup_count"))
    accepted_progress = _integer(metrics.get("accepted_progress_count"))
    phase = metrics.get("phase_completion", {})
    phase = phase if isinstance(phase, dict) else {}
    model = metrics.get("model", {})
    model = model if isinstance(model, dict) else {}
    experiment = manifest.get("experiment", {})
    experiment = experiment if isinstance(experiment, dict) else {}
    protocol = manifest.get("protocol", {})
    protocol = protocol if isinstance(protocol, dict) else {}
    codebase = manifest.get("codebase", {})
    codebase = codebase if isinstance(codebase, dict) else {}
    reasons: list[str] = []
    billing_reasons: list[str] = []
    if not metrics:
        reasons.append("MISSING_OR_INVALID_METRICS")
    if not manifest:
        reasons.append("MISSING_OR_INVALID_RUN_MANIFEST")
    if (
        manifest_status not in (None, "")
        and metrics_status not in (None, "")
        and manifest_status != metrics_status
    ):
        reasons.append("TERMINAL_STATUS_CONFLICT")
    if metrics and physical is None:
        reasons.append("PHYSICAL_TRANSACTION_COUNT_MISSING")
    if metrics and accepted is None:
        reasons.append("ACCEPTED_TRANSACTION_COUNT_MISSING")
    if metrics and rejected is None:
        reasons.append("REJECTED_TRANSACTION_COUNT_MISSING")
    if physical is not None and accepted is not None and rejected is not None:
        if physical != accepted + rejected:
            reasons.append("PHYSICAL_ACCOUNTING_MISMATCH")
    if any(
        value is not None and value < 0
        for value in (
            physical,
            accepted,
            rejected,
            accepted_setup,
            accepted_progress,
        )
    ):
        reasons.append("NEGATIVE_PHYSICAL_ACCOUNTING")
    if metrics and (accepted_setup is None or accepted_progress is None):
        reasons.append("ACCEPTED_KIND_ACCOUNTING_MISSING")
    if (
        accepted is not None
        and accepted_setup is not None
        and accepted_progress is not None
        and accepted != accepted_setup + accepted_progress
    ):
        reasons.append("ACCEPTED_KIND_ACCOUNTING_MISMATCH")
    if physical is not None and physical > int(physical_budget):
        reasons.append("PHYSICAL_BUDGET_EXCEEDED")
    if status == "PHYSICAL_BUDGET_EXHAUSTED" and physical != int(physical_budget):
        reasons.append("PHYSICAL_BUDGET_TERMINAL_COUNT_MISMATCH")
    raw_task_success = metrics.get("task_success")
    task_success = raw_task_success is True
    if metrics and not isinstance(raw_task_success, bool):
        reasons.append("TASK_SUCCESS_NOT_BOOLEAN")
    if status == "SUCCESS" and not task_success:
        reasons.append("SUCCESS_STATUS_WITHOUT_TASK_SUCCESS")
    if task_success and status != "SUCCESS":
        reasons.append("TASK_SUCCESS_WITHOUT_SUCCESS_STATUS")
    if metrics and not infrastructure_invalid and status not in VALID_TASK_TERMINALS:
        reasons.append("UNKNOWN_TASK_TERMINAL_STATUS")
    phase_completed = _integer(phase.get("completed"))
    phase_total = _integer(phase.get("total"))
    phase_fraction = _number(phase.get("fraction"))
    if metrics and (
        phase_completed is None or phase_total is None or phase_fraction is None
    ):
        reasons.append("PHASE_COMPLETION_MISSING")
    if phase_fraction is not None and not 0.0 <= phase_fraction <= 1.0:
        reasons.append("PHASE_COMPLETION_OUT_OF_RANGE")
    if phase_completed is not None and phase_completed < 0:
        reasons.append("PHASE_COMPLETED_NEGATIVE")
    if phase_total is not None and phase_total <= 0:
        reasons.append("PHASE_TOTAL_NOT_POSITIVE")
    if (
        phase_completed is not None
        and phase_total is not None
        and phase_total > 0
    ):
        if phase_completed > phase_total:
            reasons.append("PHASE_COMPLETED_EXCEEDS_TOTAL")
        if phase_fraction is not None and not math.isclose(
            phase_fraction,
            phase_completed / phase_total,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            reasons.append("PHASE_COMPLETION_ACCOUNTING_MISMATCH")
    if status == "SUCCESS" and (
        phase_completed is None
        or phase_total is None
        or phase_completed != phase_total
        or phase_fraction is None
        or not math.isclose(phase_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        reasons.append("SUCCESS_WITH_INCOMPLETE_PHASE_PLAN")
    llm_calls = _integer(model.get("llm_calls"))
    vlm_calls = _integer(model.get("vlm_calls"))
    if metrics and (llm_calls is None or vlm_calls is None):
        # A total alone cannot prove that a missing text/vision ledger was
        # really zero.  Formal results must state both modalities explicitly.
        reasons.append("FM_CALL_MODALITY_COUNT_MISSING")
    if any(value is not None and value < 0 for value in (llm_calls, vlm_calls)):
        reasons.append("NEGATIVE_FM_CALL_COUNT")
    if (
        metrics
        and llm_calls is not None
        and vlm_calls is not None
        and llm_calls + vlm_calls <= 0
    ):
        reasons.append("FM_CALL_COUNT_ZERO")
    input_tokens = _integer(model.get("input_tokens"))
    output_tokens = _integer(model.get("output_tokens"))
    elapsed_s = _number(metrics.get("elapsed_s"))
    raw_cost_usd = _number(model.get("billing_amount_usd"))
    if metrics and (input_tokens is None or output_tokens is None):
        reasons.append("FM_TOKEN_COUNT_MISSING_OR_INVALID")
    if any(value is not None and value < 0 for value in (input_tokens, output_tokens)):
        reasons.append("NEGATIVE_FM_TOKEN_COUNT")
    if metrics and elapsed_s is None:
        reasons.append("ELAPSED_TIME_MISSING_OR_INVALID")
    elif elapsed_s is not None and elapsed_s < 0:
        reasons.append("NEGATIVE_ELAPSED_TIME")
    billing_status = str(
        model.get("billing_reconciliation_status", "unavailable")
    ).strip().lower()
    if billing_status not in {
        "complete",
        "partial",
        "unavailable",
        "incomplete",
        "unknown",
        "not_available",
        "na",
    }:
        billing_reasons.append("BILLING_RECONCILIATION_STATUS_INVALID")
        billing_status = "unavailable"
    if model.get("billing_amount_usd") is not None and raw_cost_usd is None:
        billing_reasons.append("COST_INVALID")
    elif raw_cost_usd is not None and raw_cost_usd < 0:
        billing_reasons.append("NEGATIVE_COST")
        raw_cost_usd = None
    if billing_status == "complete" and raw_cost_usd is None:
        billing_reasons.append("BILLING_COMPLETE_WITHOUT_COST")
    elif billing_status != "complete" and raw_cost_usd is not None:
        billing_reasons.append("UNRECONCILED_COST_IGNORED")
    cost_usd = raw_cost_usd if billing_status == "complete" else None
    cost_complete = billing_status == "complete" and cost_usd is not None

    actual_condition_fp = str(
        experiment.get("condition_fingerprint")
        or manifest.get("condition_fingerprint")
        or plan.get("condition_fingerprint")
        or ""
    )
    actual_invariant_fp = str(
        experiment.get("invariant_fingerprint")
        or manifest.get("invariant_fingerprint")
        or plan.get("invariant_fingerprint")
        or ""
    )
    _check_expected_fingerprint(
        reasons,
        "CONDITION_FINGERPRINT",
        expected_payload.get("condition_fingerprint"),
        actual_condition_fp,
    )
    _check_expected_fingerprint(
        reasons,
        "INVARIANT_FINGERPRINT",
        expected_payload.get("invariant_fingerprint"),
        actual_invariant_fp,
    )
    actual_scene_fp = str(
        experiment.get("scene_fingerprint")
        or manifest.get("scene_fingerprint")
        or plan.get("scene_fingerprint")
        or ""
    )
    actual_terminal_fp = str(
        experiment.get("terminal_fingerprint")
        or manifest.get("terminal_fingerprint")
        or plan.get("terminal_fingerprint")
        or ""
    )
    for label, actual in (
        ("CONDITION_FINGERPRINT", actual_condition_fp),
        ("INVARIANT_FINGERPRINT", actual_invariant_fp),
        ("SCENE_FINGERPRINT", actual_scene_fp),
        ("TERMINAL_FINGERPRINT", actual_terminal_fp),
    ):
        # A legacy source that predates these bindings may be admitted only by
        # a version-specific, hash-bound source-cohort auditor.  The generic
        # collector must not turn an absent fingerprint into a valid paper row
        # merely because no external expectation was supplied.
        if metrics and not actual:
            reasons.append(f"{label}_MISSING")
    _check_expected_fingerprint(
        reasons,
        "SCENE_FINGERPRINT",
        expected_payload.get("scene_fingerprint"),
        actual_scene_fp,
    )
    _check_expected_fingerprint(
        reasons,
        "TERMINAL_FINGERPRINT",
        expected_payload.get("terminal_fingerprint"),
        actual_terminal_fp,
    )
    actual_observation_contract_fp = _actual_fingerprint(
        experiment,
        manifest,
        {} if has_external_expected else plan,
        "observation_contract_fingerprint",
    )
    actual_task_instruction_fp = _actual_fingerprint(
        experiment,
        manifest,
        {} if has_external_expected else plan,
        "task_instruction_fingerprint",
    )
    actual_shared_prompt_contract_fp = _actual_fingerprint(
        experiment,
        manifest,
        {} if has_external_expected else plan,
        "shared_prompt_contract_fingerprint",
    )
    actual_decision_prompt_fp = _actual_fingerprint(
        experiment,
        manifest,
        {} if has_external_expected else plan,
        "decision_prompt_fingerprint",
    )
    actual_skill_api_fp = _actual_fingerprint(
        experiment,
        manifest,
        {} if has_external_expected else plan,
        "skill_api_fingerprint",
    )
    actual_full_method_compatibility_fp = _actual_fingerprint(
        experiment,
        manifest,
        {} if has_external_expected else plan,
        "full_method_compatibility_fingerprint",
    )
    for label, key, actual in (
        (
            "OBSERVATION_CONTRACT_FINGERPRINT",
            "observation_contract_fingerprint",
            actual_observation_contract_fp,
        ),
        (
            "TASK_INSTRUCTION_FINGERPRINT",
            "task_instruction_fingerprint",
            actual_task_instruction_fp,
        ),
        (
            "SHARED_PROMPT_CONTRACT_FINGERPRINT",
            "shared_prompt_contract_fingerprint",
            actual_shared_prompt_contract_fp,
        ),
        (
            "DECISION_PROMPT_FINGERPRINT",
            "decision_prompt_fingerprint",
            actual_decision_prompt_fp,
        ),
        (
            "SKILL_API_FINGERPRINT",
            "skill_api_fingerprint",
            actual_skill_api_fp,
        ),
        (
            "FULL_METHOD_COMPATIBILITY_FINGERPRINT",
            "full_method_compatibility_fingerprint",
            actual_full_method_compatibility_fp,
        ),
    ):
        if metrics and not actual:
            reasons.append(f"{label}_MISSING")
        expected_value = str(expected_payload.get(key) or "").strip()
        if has_external_expected and not _SHA256_RE.fullmatch(expected_value):
            reasons.append(f"{label}_EXPECTATION_MISSING")
        else:
            _check_expected_fingerprint(
                reasons,
                label,
                expected_value,
                actual,
            )
    expected_commit = str(expected_payload.get("base_method_commit", "")).strip()
    actual_commit = str(
        experiment.get("base_method_commit")
        or codebase.get("git_head")
        or plan.get("base_method_commit")
        or ""
    ).strip()
    if expected_commit and actual_commit != expected_commit:
        reasons.append("BASE_METHOD_COMMIT_MISMATCH")

    table = _actual_text(experiment, plan, "table", "table_id")
    condition = _actual_text(experiment, plan, "condition", "condition_id")
    if not condition:
        output_relpath = _actual_text(experiment, plan, "output_relpath")
        output_parts = Path(output_relpath).parts
        condition = output_parts[1] if len(output_parts) >= 2 else ""
    method = _actual_text(experiment, plan, "method", "method_variant")
    scene = _actual_text(experiment, plan, "scene", "scene_id")
    memory_condition = _actual_text(experiment, plan, "memory_condition")
    strategy_bank = plan.get("strategy_bank", {})
    strategy_bank = strategy_bank if isinstance(strategy_bank, dict) else {}
    source_model = _actual_text(experiment, plan, "source_model") or str(
        strategy_bank.get("source_backbone") or ""
    )
    _check_expected_text(reasons, "TABLE", expected_payload.get("table"), table)
    _check_expected_text(
        reasons, "CONDITION", expected_payload.get("condition"), condition
    )
    _check_expected_text(reasons, "METHOD", expected_payload.get("method"), method)
    _check_expected_text(reasons, "SCENE", expected_payload.get("scene"), scene)
    _check_expected_text(
        reasons,
        "MEMORY_CONDITION",
        expected_payload.get("memory_condition"),
        memory_condition,
    )
    _check_expected_text(
        reasons, "SOURCE_MODEL", expected_payload.get("source_model"), source_model
    )
    manifest_model = manifest.get("model", {})
    manifest_model = manifest_model if isinstance(manifest_model, dict) else {}
    expected_model = str(expected_payload.get("target_model", "")).strip()
    actual_model = str(
        manifest_model.get("resolved")
        or manifest_model.get("requested")
        or experiment.get("target_model")
        or plan.get("target_model")
        or ""
    ).strip()
    if expected_model and actual_model != expected_model:
        reasons.append("TARGET_MODEL_MISMATCH")
    target_model = actual_model
    provider = str(
        manifest_model.get("provider")
        or experiment.get("provider")
        or plan.get("provider")
        or ""
    ).strip()
    provider_route_id = str(
        manifest_model.get("provider_route_id")
        or experiment.get("provider_route_id")
        or ("" if has_external_expected else plan.get("provider_route_id"))
        or ""
    ).strip()
    provider_route_fingerprint = str(
        manifest_model.get("provider_route_fingerprint")
        or experiment.get("provider_route_fingerprint")
        or (
            ""
            if has_external_expected
            else plan.get("provider_route_fingerprint")
        )
        or ""
    ).strip()
    if metrics and not provider_route_id:
        reasons.append("PROVIDER_ROUTE_ID_MISSING")
    if metrics and not provider_route_fingerprint:
        reasons.append("PROVIDER_ROUTE_FINGERPRINT_MISSING")
    _check_expected_text(
        reasons,
        "PROVIDER_ROUTE_ID",
        expected_payload.get("provider_route_id"),
        provider_route_id,
    )
    expected_provider_route_fp = str(
        expected_payload.get("provider_route_fingerprint") or ""
    ).strip()
    if has_external_expected and not _SHA256_RE.fullmatch(
        expected_provider_route_fp
    ):
        reasons.append("PROVIDER_ROUTE_FINGERPRINT_EXPECTATION_MISSING")
    else:
        _check_expected_fingerprint(
            reasons,
            "PROVIDER_ROUTE_FINGERPRINT",
            expected_provider_route_fp,
            provider_route_fingerprint,
        )
    if provider_route_id and provider_route_fingerprint:
        try:
            route = get_provider_route(provider_route_id)
        except ValueError:
            reasons.append("PROVIDER_ROUTE_ID_UNREGISTERED")
        else:
            if route.fingerprint != provider_route_fingerprint:
                reasons.append("PROVIDER_ROUTE_FINGERPRINT_INVALID")
            if provider and route.provider != provider:
                reasons.append("PROVIDER_ROUTE_PROVIDER_MISMATCH")
    expected_protocol_id = str(expected_payload.get("protocol_id", "")).strip()
    actual_protocol_id = str(
        experiment.get("protocol_id")
        or protocol.get("protocol_id")
        or plan.get("protocol_id")
        or ""
    ).strip()
    if expected_protocol_id and actual_protocol_id != expected_protocol_id:
        reasons.append("PROTOCOL_ID_MISMATCH")
    protocol_id = actual_protocol_id
    effective_arguments = manifest.get("effective_arguments", {})
    effective_arguments = (
        effective_arguments if isinstance(effective_arguments, dict) else {}
    )
    seed = _integer(
        experiment.get("seed", plan.get("seed", effective_arguments.get("seed")))
    )
    expected_seed = _integer(expected_payload.get("seed"))
    if expected_payload.get("seed") is not None and expected_seed != seed:
        reasons.append("SEED_MISMATCH")
    if metrics and seed is None:
        reasons.append("SEED_MISSING")

    expected_table = str(
        expected_payload.get("table") or expected_payload.get("table_id") or ""
    ).strip()
    expected_memory_condition = str(
        expected_payload.get("memory_condition") or ""
    ).strip()
    transfer_run = (
        _TABLE4_ID in {table, expected_table}
        and (
            memory_condition in _TRANSFER_MEMORY_CONDITIONS
            or expected_memory_condition in _TRANSFER_MEMORY_CONDITIONS
        )
    )
    strategy_bank_sha256 = ""
    memory_audit_hashes: dict[str, str] = {}
    terminal_evidence_hashes: dict[str, str] = {}
    usage_evidence_hashes: dict[str, str] = {}
    accounting_evidence_hashes: dict[str, str] = {}
    failure_counts = _count_mapping(metrics.get("failure_counts"))
    mutation_counts = _mutation_counts(metrics)
    successful_fm_calls: int | None = None
    failed_fm_calls: int | None = None
    transport_attempts: int | None = None
    stall_model_request_ids: tuple[str, ...] = ()
    require_native_memory = (
        expected_payload.get("require_native_memory_evidence") is True
    )
    require_accounting = (
        expected_payload.get("require_transaction_accounting_evidence") is True
    )
    if require_accounting and metrics and manifest and not infrastructure_invalid:
        expected_run_id = str(expected_payload.get("run_id") or "").strip()
        live_launch = manifest.get("live_launch")
        live_launch = live_launch if isinstance(live_launch, Mapping) else {}
        capability_entrypoint_sha256 = str(
            live_launch.get("entrypoint_sha256") or ""
        ).strip()
        expected_compiled_model_sha256 = str(
            expected_payload.get("compiled_model_sha256") or ""
        ).strip()
        expected_scene_realization_sha256 = str(
            expected_payload.get("scene_realization_sha256") or ""
        ).strip()
        claimed_values = (
            physical,
            accepted,
            rejected,
            accepted_setup,
            accepted_progress,
            phase_completed,
            phase_total,
        )
        claimed_accounting = None
        if all(value is not None for value in claimed_values):
            claimed_accounting = {
                "physical_transactions": physical,
                "accepted_transactions": accepted,
                "rejected_transactions": rejected,
                "accepted_setup_transactions": accepted_setup,
                "accepted_progress_transactions": accepted_progress,
                "phase_completed": phase_completed,
                "phase_total": phase_total,
                "failure_counts": failure_counts,
                "mutation_counts": mutation_counts,
            }
        if (
            not expected_run_id
            or not _SHA256_RE.fullmatch(capability_entrypoint_sha256)
            or not _SHA256_RE.fullmatch(expected_compiled_model_sha256)
            or not _SHA256_RE.fullmatch(expected_scene_realization_sha256)
        ):
            reasons.append("TRANSACTION_ACCOUNTING_EXPECTATION_INVALID")
            accounting_validation = None
        else:
            accounting_validation = validate_transaction_accounting_evidence(
                root,
                expected_run_id=expected_run_id,
                physical_budget=physical_budget,
                expected_compiled_model_sha256=expected_compiled_model_sha256,
                expected_scene_realization_sha256=(
                    expected_scene_realization_sha256
                ),
                expected_producers=capability_bound_auxiliary_producer_allowlist(
                    capability_entrypoint_sha256
                ),
                claimed_accounting=claimed_accounting,
            )
            reasons.extend(accounting_validation.invalid_reasons)
            if accounting_validation.mismatched_fields:
                reasons.append("TRANSACTION_ACCOUNTING_LEDGER_FIELD_MISMATCH")
            accounting_evidence_hashes = {
                key: value
                for key, value in (accounting_validation.artifact_hashes or {}).items()
                if value
            }
        if accounting_validation is not None and accounting_validation.valid:
            ledger = accounting_validation.recomputed_ledger or {}
            phase_ledger = ledger.get("phase_completion")
            phase_ledger = (
                phase_ledger if isinstance(phase_ledger, Mapping) else {}
            )
            physical = _integer(ledger.get("physical_transactions"))
            accepted = _integer(ledger.get("accepted_transactions"))
            rejected = _integer(ledger.get("rejected_transactions"))
            accepted_setup = _integer(
                ledger.get("accepted_setup_transactions")
            )
            accepted_progress = _integer(
                ledger.get("accepted_progress_transactions")
            )
            phase_completed = _integer(phase_ledger.get("completed"))
            phase_total = _integer(phase_ledger.get("total"))
            phase_fraction = _number(phase_ledger.get("fraction"))
            failure_counts = _count_mapping(ledger.get("failure_counts"))
            mutation_counts = _count_mapping(ledger.get("mutation_counts"))
        else:
            # Formal new-table rows never expose caller-authored aggregate
            # counts when the receipt-derived ledger is unavailable.
            physical = accepted = rejected = None
            accepted_setup = accepted_progress = None
            phase_completed = phase_total = None
            phase_fraction = None
            failure_counts = {}
            mutation_counts = {}
        if (
            accounting_validation is not None
            and accounting_validation.valid
            and status == "SUCCESS"
            and (phase_completed is None or phase_completed != phase_total)
        ):
            reasons.append("SUCCESS_WITH_INCOMPLETE_DERIVED_PHASE_PLAN")
        if (
            accounting_validation is not None
            and accounting_validation.valid
            and status == "PHYSICAL_BUDGET_EXHAUSTED"
            and physical != int(physical_budget)
        ):
            reasons.append("DERIVED_BUDGET_TERMINAL_COUNT_MISMATCH")
    if transfer_run and not require_native_memory:
        expected_bank_sha256 = _expected_strategy_bank_sha256(expected_payload)
        expected_bank_pin = _expected_strategy_bank_pin(expected_payload)
        if not has_external_expected or not _SHA256_RE.fullmatch(expected_bank_sha256):
            reasons.append("TABLE4_STRATEGY_BANK_EXPECTATION_MISSING")
        elif expected_bank_pin is None:
            reasons.append("TABLE4_STRATEGY_BANK_SOURCE_EXPECTATION_MISSING")
        else:
            memory_validation = validate_table4_memory_audit(
                root,
                expected_strategy_bank_sha256=expected_bank_sha256,
                expected_condition=(
                    expected_memory_condition or memory_condition
                ),
                expected_target_model=(
                    str(expected_payload.get("target_model") or "").strip()
                    or target_model
                ),
                expected_source_model=(
                    str(expected_payload.get("source_model") or "").strip()
                    or source_model
                ),
                expected_strategy_bank_pin=expected_bank_pin,
                expected_formal_binding={
                    key: str(expected_payload.get(key) or "").strip()
                    for key in (
                        "run_id",
                        "protocol_id",
                        "condition_fingerprint",
                        "strategy_bank_sha256",
                        "source_model",
                        "source_provider_route_id",
                        "source_provider_route_fingerprint",
                        "target_model",
                        "provider_route_id",
                        "provider_route_fingerprint",
                        "plan_path",
                        "plan_sha256",
                        "observation_contract_fingerprint",
                        "task_instruction_fingerprint",
                        "shared_prompt_contract_fingerprint",
                        "decision_prompt_fingerprint",
                        "skill_api_fingerprint",
                        "full_method_compatibility_fingerprint",
                    )
                },
            )
            reasons.extend(memory_validation.invalid_reasons)
            strategy_bank_sha256 = memory_validation.loaded_seed_sha256
            memory_audit_hashes = {
                key: value
                for key, value in (memory_validation.artifact_hashes or {}).items()
                if value
            }
    if require_native_memory and metrics and manifest and not infrastructure_invalid:
        native_expectation = _native_memory_expectation(expected_payload)
        if native_expectation is None:
            reasons.append("V83_NATIVE_MEMORY_EXPECTATION_INVALID")
        else:
            native_mode, native_gates, native_binding = native_expectation
            expected_bank_sha256 = _expected_strategy_bank_sha256(expected_payload)
            transfer_bank_path = (
                root / "memory" / "seed_strategy_bank.json"
                if native_mode == "preloaded_seed"
                else None
            )
            native_validation = validate_v83_native_memory_audit(
                root,
                expected_run_binding=native_binding,
                expected_mode=native_mode,
                expected_operation_gates=native_gates,
                expected_terminal_reason=status,
                expected_transfer_bank=transfer_bank_path,
                expected_strategy_bank_sha256=expected_bank_sha256,
            )
            reasons.extend(native_validation.invalid_reasons)
            native_hashes = {
                key: value
                for key, value in native_validation.artifact_hashes.items()
                if value
            }
            memory_audit_hashes.update(native_hashes)
            if native_mode == "preloaded_seed":
                strategy_bank_sha256 = native_validation.loaded_seed_sha256
            result_hashes = manifest.get("result_artifact_hashes")
            result_hashes = (
                result_hashes if isinstance(result_hashes, Mapping) else {}
            )
            pinned_audit_sha256 = result_hashes.get(
                _V83_NATIVE_AUDIT_MANIFEST_PIN
            )
            actual_audit_sha256 = native_hashes.get(
                _V83_NATIVE_AUDIT_ARTIFACT, ""
            )
            if not isinstance(pinned_audit_sha256, str) or not _SHA256_RE.fullmatch(
                pinned_audit_sha256
            ):
                reasons.append("V83_NATIVE_MEMORY_AUDIT_PIN_MISSING")
            elif pinned_audit_sha256 != actual_audit_sha256:
                reasons.append("V83_NATIVE_MEMORY_AUDIT_PIN_MISMATCH")
    if expected_payload.get("require_terminal_evidence") is True and metrics and manifest:
        expected_run_id = str(expected_payload.get("run_id") or "").strip()
        if (
            not expected_run_id
            or seed is None
            or physical is None
            or not _SHA256_RE.fullmatch(actual_condition_fp)
            or not _SHA256_RE.fullmatch(actual_terminal_fp)
        ):
            reasons.append("TERMINAL_EVIDENCE_EXPECTATION_INVALID")
        else:
            terminal_validation = validate_terminal_evidence(
                root,
                manifest=manifest,
                expected_run_id=expected_run_id,
                expected_seed=seed,
                expected_condition_fingerprint=actual_condition_fp,
                expected_terminal_fingerprint=actual_terminal_fp,
                expected_terminal_status=status,
                expected_scene_fingerprint=actual_scene_fp,
                expected_scene_realization_sha256=str(
                    expected_payload.get("scene_realization_sha256") or ""
                ),
                expected_compiled_model_sha256=str(
                    expected_payload.get("compiled_model_sha256") or ""
                ),
                expected_scene_execution_evidence_fingerprint=str(
                    expected_payload.get("scene_execution_evidence_fingerprint")
                    or ""
                ),
                expected_target_center_xyz_m=expected_payload.get(
                    "target_center_xyz_m", ()
                ),
                physical_transactions=physical,
                physical_budget=physical_budget,
            )
            reasons.extend(terminal_validation.invalid_reasons)
            stall_model_request_ids = terminal_validation.stall_model_request_ids
            terminal_evidence_hashes = {
                key: value
                for key, value in (terminal_validation.artifact_hashes or {}).items()
                if value
            }
    if expected_payload.get("require_usage_evidence") is True and metrics and manifest:
        expected_run_id = str(expected_payload.get("run_id") or "").strip()
        if (
            not expected_run_id
            or seed is None
            or llm_calls is None
            or vlm_calls is None
            or input_tokens is None
            or output_tokens is None
            or not actual_condition_fp
            or not provider_route_id
            or not provider_route_fingerprint
            or not target_model
        ):
            reasons.append("FM_USAGE_EVIDENCE_EXPECTATION_INVALID")
        else:
            usage_validation = validate_fm_usage_evidence(
                root,
                manifest=manifest,
                expected_run_id=expected_run_id,
                expected_seed=seed,
                expected_condition_fingerprint=actual_condition_fp,
                expected_provider_route_id=provider_route_id,
                expected_provider_route_fingerprint=provider_route_fingerprint,
                expected_target_model=target_model,
                expected_llm_calls=llm_calls,
                expected_vlm_calls=vlm_calls,
                expected_input_tokens=input_tokens,
                expected_output_tokens=output_tokens,
                expected_billing_amount_usd=cost_usd,
                expected_billing_reconciliation_status=billing_status,
                expected_schema_version=FM_USAGE_LEDGER_SCHEMA_VERSION,
                required_request_ids=stall_model_request_ids,
            )
            reasons.extend(usage_validation.invalid_reasons)
            billing_reasons.extend(usage_validation.billing_reasons)
            successful_fm_calls = usage_validation.usage_totals.get(
                "successful_responses"
            )
            failed_fm_calls = usage_validation.usage_totals.get("failed_requests")
            transport_attempts = usage_validation.usage_totals.get(
                "transport_attempts"
            )
            cost_complete = usage_validation.billing_complete
            cost_usd = (
                usage_validation.billing_amount_usd if cost_complete else None
            )
            billing_status = usage_validation.billing_completeness
            usage_evidence_hashes = {
                key: value
                for key, value in (usage_validation.artifact_hashes or {}).items()
                if value
            }
    protocol_valid = bool(
        metrics
        and manifest
        and not infrastructure_invalid
        and not reasons
        and status in VALID_TASK_TERMINALS
    )
    return RunRecord(
        run_dir=str(root),
        table=table,
        condition=condition,
        method=method,
        scene=scene,
        memory_condition=memory_condition,
        source_model=source_model,
        target_model=target_model,
        provider=provider,
        provider_route_id=provider_route_id,
        provider_route_fingerprint=provider_route_fingerprint,
        strategy_bank_sha256=strategy_bank_sha256,
        seed=seed,
        protocol_id=protocol_id,
        base_method_commit=actual_commit,
        condition_fingerprint=actual_condition_fp,
        invariant_fingerprint=actual_invariant_fp,
        scene_fingerprint=actual_scene_fp,
        terminal_fingerprint=actual_terminal_fp,
        observation_contract_fingerprint=actual_observation_contract_fp,
        task_instruction_fingerprint=actual_task_instruction_fp,
        shared_prompt_contract_fingerprint=actual_shared_prompt_contract_fp,
        decision_prompt_fingerprint=actual_decision_prompt_fp,
        skill_api_fingerprint=actual_skill_api_fp,
        full_method_compatibility_fingerprint=(
            actual_full_method_compatibility_fp
        ),
        terminal_status=status,
        metrics_available=bool(metrics),
        protocol_valid=protocol_valid,
        infrastructure_invalid=infrastructure_invalid,
        invalid_reasons=tuple(reasons),
        task_success=task_success,
        physical_transactions=physical,
        accepted_transactions=accepted,
        rejected_transactions=rejected,
        accepted_setup_transactions=accepted_setup,
        accepted_progress_transactions=accepted_progress,
        phase_completed=phase_completed,
        phase_total=phase_total,
        phase_completion=phase_fraction,
        fm_calls=(llm_calls + vlm_calls) if llm_calls is not None and vlm_calls is not None else None,
        successful_fm_calls=successful_fm_calls,
        failed_fm_calls=failed_fm_calls,
        transport_attempts=transport_attempts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_time_s=elapsed_s,
        cost_usd=cost_usd,
        cost_complete=cost_complete,
        billing_reconciliation_status=billing_status,
        billing_reasons=tuple(dict.fromkeys(billing_reasons)),
        failure_counts=failure_counts,
        mutation_counts=mutation_counts,
        artifact_hashes={
            **_string_mapping(
                manifest.get(
                    "result_artifact_hashes",
                    manifest.get("artifact_hashes"),
                )
            ),
            **memory_audit_hashes,
            **terminal_evidence_hashes,
            **usage_evidence_hashes,
            **accounting_evidence_hashes,
        },
    )


def _expected_strategy_bank_sha256(expected: Mapping[str, Any]) -> str:
    direct = str(expected.get("strategy_bank_sha256") or "").strip()
    if direct:
        return direct
    nested = expected.get("strategy_bank")
    if isinstance(nested, Mapping):
        return str(nested.get("sha256") or "").strip()
    return ""


def _native_memory_expectation(
    expected: Mapping[str, Any],
) -> tuple[str, dict[str, bool], dict[str, Any]] | None:
    """Build the exact external identity expected by the native audit."""

    policy = str(expected.get("memory_policy") or "").strip()
    mode_by_policy = {
        "disabled": "disabled",
        "online_from_empty": "from_empty",
        "online_from_preloaded_strategies": "preloaded_seed",
    }
    mode = mode_by_policy.get(policy)
    seed = _integer(expected.get("seed"))
    table_id = str(expected.get("table") or expected.get("table_id") or "").strip()
    target_backbone = str(
        expected.get("target_model") or expected.get("target_backbone") or ""
    ).strip()
    strategy_bank_sha256 = _expected_strategy_bank_sha256(expected)
    binding: dict[str, Any] = {
        "run_id": str(expected.get("run_id") or "").strip(),
        "table_id": table_id,
        "seed": seed,
        "protocol_id": str(expected.get("protocol_id") or "").strip(),
        "base_method_commit": str(
            expected.get("base_method_commit") or ""
        ).strip(),
        "condition_fingerprint": str(
            expected.get("condition_fingerprint") or ""
        ).strip(),
        "invariant_fingerprint": str(
            expected.get("invariant_fingerprint") or ""
        ).strip(),
        "scene_fingerprint": str(
            expected.get("scene_fingerprint") or ""
        ).strip(),
        "terminal_fingerprint": str(
            expected.get("terminal_fingerprint") or ""
        ).strip(),
        "memory_condition": str(
            expected.get("memory_condition") or ""
        ).strip(),
        "memory_policy": policy,
        "target_backbone": target_backbone,
        "provider_route_id": str(
            expected.get("provider_route_id") or ""
        ).strip(),
        "provider_route_fingerprint": str(
            expected.get("provider_route_fingerprint") or ""
        ).strip(),
        "strategy_bank_sha256": strategy_bank_sha256,
        "plan_path": str(expected.get("plan_path") or "").strip(),
        "plan_sha256": str(expected.get("plan_sha256") or "").strip(),
        "observation_contract_fingerprint": str(
            expected.get("observation_contract_fingerprint") or ""
        ).strip(),
        "task_instruction_fingerprint": str(
            expected.get("task_instruction_fingerprint") or ""
        ).strip(),
        "shared_prompt_contract_fingerprint": str(
            expected.get("shared_prompt_contract_fingerprint") or ""
        ).strip(),
        "decision_prompt_fingerprint": str(
            expected.get("decision_prompt_fingerprint") or ""
        ).strip(),
        "skill_api_fingerprint": str(
            expected.get("skill_api_fingerprint") or ""
        ).strip(),
        "full_method_compatibility_fingerprint": str(
            expected.get("full_method_compatibility_fingerprint") or ""
        ).strip(),
    }
    if mode is None or seed is None:
        return None
    required_nonempty = {
        key: value
        for key, value in binding.items()
        if key not in {"seed", "strategy_bank_sha256"}
    }
    if any(not isinstance(value, str) or not value for value in required_nonempty.values()):
        return None
    if mode == "preloaded_seed":
        if not _SHA256_RE.fullmatch(strategy_bank_sha256):
            return None
    elif strategy_bank_sha256:
        return None
    gates = {
        "strategy_retrieval": mode != "disabled",
        "strategy_promotion": mode != "disabled",
        "action_exemplar_retrieval": mode != "disabled",
        "action_exemplar_promotion": mode != "disabled",
    }
    return mode, gates, binding


def _expected_strategy_bank_pin(
    expected: Mapping[str, Any],
) -> StrategyBankPin | None:
    nested = expected.get("strategy_bank")
    if not isinstance(nested, Mapping):
        return None
    try:
        payload = dict(nested)
        seeds = payload.get("source_seeds")
        if not isinstance(seeds, (list, tuple)):
            return None
        payload["source_seeds"] = tuple(seeds)
        return StrategyBankPin(**payload)
    except (TypeError, ValueError):
        return None


def _actual_fingerprint(
    experiment: Mapping[str, Any],
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    key: str,
) -> str:
    return str(
        experiment.get(key)
        or manifest.get(key)
        or plan.get(key)
        or ""
    ).strip()


def _check_expected_fingerprint(
    reasons: list[str], label: str, expected: object, actual: str
) -> None:
    value = str(expected or "").strip()
    if value and value != actual:
        reasons.append(f"{label}_MISMATCH")


def _check_expected_text(
    reasons: list[str], label: str, expected: object, actual: str
) -> None:
    if expected is None:
        return
    value = str(expected).strip()
    if value != actual:
        reasons.append(f"{label}_MISMATCH")


def _actual_text(
    experiment: Mapping[str, Any], plan: Mapping[str, Any], *keys: str
) -> str:
    for mapping in (experiment, plan):
        for key in keys:
            value = mapping.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _mutation_counts(metrics: Mapping[str, Any]) -> dict[str, int]:
    candidates = (
        metrics.get("mutation_counts"),
        metrics.get("strategy_improvement", {}).get("mutation_counts")
        if isinstance(metrics.get("strategy_improvement"), dict)
        else None,
    )
    for candidate in candidates:
        result = _count_mapping(candidate)
        if result:
            return result
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _integer(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sum_optional_ints(*values: object) -> int | None:
    parsed = [_integer(value) for value in values]
    available = [value for value in parsed if value is not None]
    return sum(available) if available else None


def _count_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): parsed
        for key, child in value.items()
        if (parsed := _integer(child)) is not None
    }


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(child)
        for key, child in value.items()
        if str(child).strip()
    }
