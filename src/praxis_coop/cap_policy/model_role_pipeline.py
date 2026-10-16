"""Auditable model-role pipeline for cooperative policy evolution.

Roles are separate model calls with separate prompts and artifacts, while
deterministic host code continues to own schema validation, physics safety,
rollback, and execution. All roles share one injected multimodal client per
run; the frozen benchmark pins that client to one exact backbone ID.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Literal, Protocol

from praxis_coop.configs import DEFAULT_MULTIMODAL_MODEL
from praxis_coop.llm.errors import ProviderFatalError

from .coordination_specs import (
    MODEL_COORDINATION_PRIMITIVE_METHODS,
    coordination_modes_for_method,
)
from .contracts import write_json
from .generation_recovery import (
    MAX_IDENTICAL_TYPED_ROLE_FAILURES,
    NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL,
    ROLE_GENERATION_STALL,
    TYPED_ROLE_FAILURES,
    classify_typed_generation_failure,
    consecutive_trailing_kind_count,
    editable_fields_for_mutation_level,
    typed_generation_feedback,
)
from .prompts import (
    CODE_REPAIR_SYSTEM_PROMPT,
    EVOLUTION_MANAGER_SYSTEM_PROMPT,
    MEMORY_UPDATE_SYSTEM_PROMPT,
    MUTATION_OPERATOR_SYSTEM_PROMPT,
    POLICY_CODER_SYSTEM_PROMPT,
    TASK_ORCHESTRATOR_SYSTEM_PROMPT,
    V50_LITE_FORMAL_SYSTEM_SUFFIX,
    VLM_FAILURE_ANALYZER_SYSTEM_PROMPT,
    build_role_user_prompt,
)
from .policy_executor import PAYLOAD_COOPERATIVE_METHODS
from .strategy_memory import ACTION_PARAMETER_KEYS
from .role_schemas import role_output_schema, validate_strict_json_schema


ModelRole = Literal[
    "task_orchestrator",
    "coder",
    "vlm_failure_analyzer",
    "evolution_manager",
    "mutation_operator",
    "code_repair",
    "memory_update",
]
MutationLevel = Literal[
    "strategy_mutation",
    "coordination_mutation",
    "parameter_tuning",
]

MUTATION_LEVELS = frozenset(
    {
        "strategy_mutation",
        "coordination_mutation",
        "parameter_tuning",
    }
)

ROLE_SYSTEM_PROMPTS: dict[ModelRole, str] = {
    "task_orchestrator": TASK_ORCHESTRATOR_SYSTEM_PROMPT,
    "coder": POLICY_CODER_SYSTEM_PROMPT,
    "vlm_failure_analyzer": VLM_FAILURE_ANALYZER_SYSTEM_PROMPT,
    "evolution_manager": EVOLUTION_MANAGER_SYSTEM_PROMPT,
    "mutation_operator": MUTATION_OPERATOR_SYSTEM_PROMPT,
    "code_repair": CODE_REPAIR_SYSTEM_PROMPT,
    "memory_update": MEMORY_UPDATE_SYSTEM_PROMPT,
}


class ModelRoleClient(Protocol):
    """Text and image methods supplied by the shared Qwen-VL client."""

    model_name: str

    def generate(self, *, system: str, user: str) -> str:
        ...

    def analyze_image(
        self,
        *,
        system: str,
        user: str,
        image_path: str | Path,
    ) -> str:
        ...


class ModelRoleError(ValueError):
    """Raised when a role response cannot satisfy its structured contract."""


class ModelRoleCandidateFeasibilityError(ModelRoleError):
    """A schema-valid mutation candidate fails its executable Host contract."""


class ModelRoleStallError(ModelRoleError):
    """Five invalid responses exhausted one role invocation without dispatch."""

    def __init__(
        self,
        message: str,
        *,
        role: str,
        artifact_id: str,
        response_count: int,
        validation_errors: tuple[str, ...],
        cycle_outcome: str = ROLE_GENERATION_STALL,
        selected_mutation_level: str = "",
        validation_error_kinds: tuple[str, ...] = (),
        typed_failure: str = "",
        attempted_change: tuple[str, ...] = (),
        editable_fields: tuple[str, ...] = (),
        result: str = "",
    ) -> None:
        super().__init__(message)
        self.role = role
        self.artifact_id = artifact_id
        self.response_count = response_count
        self.validation_errors = validation_errors
        self.cycle_outcome = str(cycle_outcome).strip() or ROLE_GENERATION_STALL
        self.selected_mutation_level = str(selected_mutation_level).strip()
        self.validation_error_kinds = validation_error_kinds
        self.typed_failure = str(typed_failure).strip()
        self.attempted_change = tuple(str(item) for item in attempted_change)
        self.editable_fields = tuple(str(item) for item in editable_fields)
        self.result = str(result).strip()


RoleValidator = Callable[[dict[str, Any]], None]


class _RoleBoundaryPropagationError(RuntimeError):
    """Carry an unexpected Host error through the role retry boundary."""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


def _is_exact_model_agent_error(exc: BaseException) -> bool:
    """Recognize only the production executable-policy validation error.

    ``model_agents`` imports this module, so importing its exception at module
    load time would introduce a cycle.  Async draft prevalidation happens only
    after both modules have loaded; a local import therefore gives us an exact
    type check without accepting arbitrary ``RuntimeError`` instances (or a
    different class that merely reuses the ``ModelAgentError`` name).
    """

    from .model_agents import ModelAgentError

    return type(exc) is ModelAgentError


def _is_model_candidate_validation_error(exc: BaseException) -> bool:
    """Whether a role retry may treat ``exc`` as model-candidate-local."""

    return isinstance(exc, ValueError) or _is_exact_model_agent_error(exc)


def _prevalidate_original_async_candidate(
    candidate: dict[str, Any],
    *,
    validator: RoleValidator | None,
) -> dict[str, Any] | None:
    """Return a Host-valid original async candidate, or ``None`` on conflict.

    Only response-contract errors and the production executable-policy error
    describe a candidate that Code Repair may resolve.  Provider failures,
    generic runtime errors, and unexpected Host exceptions must escape rather
    than being converted into a model repair attempt.
    """

    try:
        _validate_async_team_policy_payload(candidate)
        if validator is not None:
            validator(candidate)
    except Exception as exc:
        if _is_model_candidate_validation_error(exc):
            return None
        raise
    return candidate


@dataclass
class ModelRolePipeline:
    """Dispatch and audit independent roles on one shared multimodal client."""

    client: ModelRoleClient
    output_dir: Path
    model_name: str = DEFAULT_MULTIMODAL_MODEL
    response_retries: int = 4
    require_formal_model: bool = True
    _call_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _records: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _current_task_plan: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _status_sequence: int = field(default=0, init=False, repr=False)
    _last_high_level_status: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir) / "model_roles"
        configured_name = str(getattr(self.client, "model_name", self.model_name))
        if self.require_formal_model and configured_name != DEFAULT_MULTIMODAL_MODEL:
            raise ModelRoleError(
                "formal role pipeline requires one shared "
                f"{DEFAULT_MULTIMODAL_MODEL}; got {configured_name!r}"
            )
        if str(self.model_name) != configured_name:
            raise ModelRoleError(
                "role pipeline model_name must match the shared client: "
                f"{self.model_name!r} != {configured_name!r}"
            )
        call_index = self.output_dir / "role_call_index.json"
        if call_index.exists():
            try:
                call_history = json.loads(call_index.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                call_history = {}
            calls = (
                call_history.get("calls", ())
                if isinstance(call_history, dict)
                else ()
            )
            if isinstance(calls, list):
                for item in calls:
                    if not isinstance(item, dict):
                        continue
                    role = str(item.get("role", "")).strip()
                    artifact_id = str(item.get("artifact_id", "")).strip()
                    if not role or not artifact_id:
                        continue
                    self._records.append(dict(item))
                    key = f"{role}:{artifact_id}"
                    self._call_counts[key] = self._call_counts.get(key, 0) + 1
        active_plan = self.output_dir / "active_task_plan_latest.json"
        persisted_plan = (
            active_plan
            if active_plan.exists()
            else self.output_dir / "task_orchestrator_plan_latest.json"
        )
        if persisted_plan.exists():
            try:
                value = json.loads(persisted_plan.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = {}
            if isinstance(value, dict):
                try:
                    _validate_task_plan(value)
                except ModelRoleError:
                    value = {}
            self._current_task_plan = value
        live_status = self.output_dir / "live_high_level_status.json"
        if live_status.exists():
            try:
                value = json.loads(live_status.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = {}
            if isinstance(value, dict):
                self._last_high_level_status = value
                try:
                    self._status_sequence = max(
                        0,
                        int(value.get("event_sequence", 0)),
                    )
                except (TypeError, ValueError):
                    self._status_sequence = 0

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._records)

    @property
    def current_task_plan(self) -> dict[str, Any]:
        """Return the latest validated long-horizon Orchestrator plan."""

        return _json_safe_mapping(self._current_task_plan)

    @property
    def last_high_level_status(self) -> dict[str, Any]:
        """Return the latest human-facing experiment status."""

        return _json_safe_mapping(self._last_high_level_status)

    def publish_high_level_status(
        self,
        runtime_status: str,
        *,
        attempt: int | None = None,
        checkpoint_id: str = "",
        task_plan: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        last_outcome: dict[str, Any] | None = None,
        next_system_operation: str = "",
        active_model_role: str = "",
        artifact_dir: str | Path = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically publish one readable high-level evolution transition."""

        runtime_status = str(runtime_status).strip()
        if not runtime_status:
            raise ModelRoleError("runtime_status must be non-empty")
        previous = self._last_high_level_status
        effective_plan = (
            _json_safe_mapping(task_plan)
            if isinstance(task_plan, dict) and task_plan
            else self.current_task_plan
        )
        if effective_plan:
            try:
                _validate_task_plan(effective_plan)
            except ModelRoleError:
                effective_plan = self.current_task_plan
            else:
                self._current_task_plan = effective_plan
                _atomic_write_json(
                    self.output_dir / "active_task_plan_latest.json",
                    effective_plan,
                )
        if attempt is None:
            try:
                attempt = int(previous.get("attempt", 0))
            except (TypeError, ValueError):
                attempt = 0
        checkpoint_id = (
            str(checkpoint_id).strip()
            or str(previous.get("checkpoint_id", "")).strip()
        )
        phase = _phase_status_payload(effective_plan)
        subtask = _subtask_status_payload(
            effective_plan,
            runtime_status=runtime_status,
        )
        action = _next_action_payload(effective_plan)
        team_assignment = _team_assignment_payload(
            effective_plan,
            policy=policy,
        )
        policy_summary = _policy_status_payload(policy)
        if not policy_summary and isinstance(previous.get("active_policy"), dict):
            policy_summary = dict(previous["active_policy"])
        outcome = (
            _json_safe_mapping(last_outcome)
            if isinstance(last_outcome, dict)
            else dict(previous.get("last_outcome", {}))
            if isinstance(previous.get("last_outcome"), dict)
            else {}
        )
        self._status_sequence += 1
        timestamp = time.time()
        status = {
            "schema_version": 1,
            "event_sequence": self._status_sequence,
            "timestamp_unix": timestamp,
            "timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(timestamp),
            ),
            "model_name": self.model_name,
            "attempt": max(0, int(attempt)),
            "checkpoint_id": checkpoint_id,
            "runtime_status": runtime_status,
            "active_model_role": str(active_model_role).strip(),
            "current_phase": phase,
            "current_subtask": subtask,
            "next_action": action,
            "team_assignment": team_assignment,
            "active_policy": policy_summary,
            "last_outcome": outcome,
            "next_system_operation": (
                str(next_system_operation).strip()
                or _default_next_system_operation(runtime_status)
            ),
            "model_rationale": str(effective_plan.get("rationale", "")).strip(),
            "semantic_summary": _semantic_status_summary(
                runtime_status=runtime_status,
                phase=phase,
                subtask=subtask,
                action=action,
                team_assignment=team_assignment,
            ),
            "artifact_dir": str(artifact_dir),
            "details": (
                _json_safe_mapping(details)
                if isinstance(details, dict)
                else {}
            ),
        }
        _atomic_write_json(
            self.output_dir / "live_high_level_status.json",
            status,
        )
        _append_jsonl(
            self.output_dir / "high_level_status_history.jsonl",
            status,
        )
        self._last_high_level_status = status
        print(_status_console_line(status), flush=True)
        return _json_safe_mapping(status)

    def responses_for_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Return the latest valid response for each role at one attempt."""

        normalized = _safe_artifact_id(artifact_id)
        responses: dict[str, Any] = {}
        for record in self._records:
            if (
                record.get("artifact_id") == normalized
                and record.get("valid") is True
                and isinstance(record.get("parsed_response"), dict)
            ):
                responses[str(record["role"])] = dict(
                    record["parsed_response"]
                )
        return responses

    def usage_summary(self) -> dict[str, Any]:
        """Return role-level usage without double-counting the shared client."""

        current_client_usage = {
            str(item.get("request_id", "")).strip(): item
            for item in getattr(self.client, "call_records", ())
            if isinstance(item, dict) and str(item.get("request_id", "")).strip()
        }
        roles: dict[str, dict[str, Any]] = {}
        for record in self._records:
            role = str(record["role"])
            bucket = roles.setdefault(
                role,
                {
                    "call_count": 0,
                    "successful_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "image_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "latency_s": 0.0,
                    "billed_call_count": 0,
                    "unbilled_call_count": 0,
                    "billing_amount_microusd": 0,
                },
            )
            bucket["call_count"] += 1
            bucket["successful_calls"] += int(bool(record.get("valid", False)))
            usage = record.get("usage", {})
            usage = usage if isinstance(usage, dict) else {}
            request_id = str(usage.get("request_id", "")).strip()
            if request_id in current_client_usage:
                usage = {**usage, **current_client_usage[request_id]}
            bucket["input_tokens"] += int(usage.get("input_tokens", 0))
            bucket["output_tokens"] += int(usage.get("output_tokens", 0))
            bucket["cached_input_tokens"] += int(
                usage.get("cached_input_tokens", 0)
            )
            bucket["image_input_tokens"] += int(
                usage.get("image_input_tokens", 0)
            )
            bucket["reasoning_tokens"] += int(
                usage.get("reasoning_tokens", 0)
            )
            bucket["latency_s"] += float(
                usage.get("latency_s", record.get("latency_s", 0.0))
            )
            billed = usage.get("billing_amount_microusd") is not None
            bucket["billed_call_count"] += int(billed)
            bucket["unbilled_call_count"] += int(not billed)
            bucket["billing_amount_microusd"] += int(
                usage.get("billing_amount_microusd", 0) or 0
            )
        for bucket in roles.values():
            bucket["billing_amount_usd"] = (
                int(bucket["billing_amount_microusd"]) / 1_000_000.0
            )
        return {
            "model_name": self.model_name,
            "shared_model_instance": True,
            "total_role_calls": len(self._records),
            "roles": roles,
        }

    def plan_task(
        self,
        context: dict[str, Any],
        *,
        artifact_id: str,
        image_path: str | Path | None = None,
        normalizer: RoleValidator | None = None,
        validator: RoleValidator | None = None,
    ) -> dict[str, Any]:
        role_context = dict(context)
        previous_task_plan = role_context.get(
            "previous_task_plan",
            self.current_task_plan,
        )
        previous_task_plan = (
            previous_task_plan if isinstance(previous_task_plan, dict) else {}
        )
        previous_task_plan = _reconcile_task_plan_with_accepted_route_evidence(
            previous_task_plan,
            context=role_context,
        )
        role_context["previous_task_plan"] = previous_task_plan
        phase_evidence = _accepted_route_phase_evidence(
            previous_task_plan.get("global_plan", []),
            context=role_context,
        )
        if phase_evidence:
            role_context["host_phase_evidence"] = phase_evidence

        def validate_phase_transition(plan: dict[str, Any]) -> None:
            _validate_task_plan_phase_transition(
                plan,
                previous_task_plan=previous_task_plan,
                context=role_context,
            )

        def normalize_then_validate(plan: dict[str, Any]) -> None:
            if normalizer is not None:
                normalizer(plan)
            _validate_task_plan(plan)
            validate_phase_transition(plan)
            if validator is not None:
                validator(plan)

        attempt = _attempt_from_artifact_id(artifact_id)
        checkpoint_id = _checkpoint_id_from_context(role_context)
        try:
            plan = self.call_json_role(
                "task_orchestrator",
                role_context,
                artifact_id=artifact_id,
                image_path=image_path,
                validator=normalize_then_validate,
            )
        except ModelRoleError:
            raise
        self._current_task_plan = _json_safe_mapping(plan)
        write_json(
            self.output_dir / "task_orchestrator_plan_latest.json",
            self._current_task_plan,
        )
        write_json(
            self.output_dir
            / f"task_orchestrator_plan_{_safe_artifact_id(artifact_id)}.json",
            self._current_task_plan,
        )
        self.publish_high_level_status(
            "subtask_planned",
            attempt=attempt,
            checkpoint_id=checkpoint_id,
            task_plan=plan,
            active_model_role="task_orchestrator",
            next_system_operation="generate executable front/back policy",
        )
        return plan

    def code_policy(
        self,
        context: dict[str, Any],
        *,
        artifact_id: str,
        validator: RoleValidator | None = None,
    ) -> dict[str, Any]:
        policy = self.call_json_role(
            "coder",
            context,
            artifact_id=artifact_id,
            validator=_combine_validators(_validate_policy_payload, validator),
        )
        self.publish_high_level_status(
            "policy_ready",
            attempt=_attempt_from_artifact_id(artifact_id),
            checkpoint_id=_checkpoint_id_from_context(context),
            policy=policy,
            next_system_operation="run host policy validation and physical preflight",
        )
        return policy

    def generate_team_policy(
        self,
        context: dict[str, Any],
        *,
        artifact_id: str,
        coordination_mode: str,
        validator: RoleValidator | None = None,
    ) -> dict[str, Any]:
        """Generate one sync policy or two async drafts plus one team review."""

        mode = str(coordination_mode).strip()
        if mode not in {"sync", "async"}:
            raise ModelRoleError(
                "team policy generation requires coordination_mode sync or async"
            )
        base_context = {
            **context,
            "coordination_mode": mode,
            "call_contract": (
                "sync uses one shared Coder call; async uses one Coder call per "
                "robot followed by one Code Repair team-integration call"
            ),
        }
        turn_repair_instruction = ""
        turn_repair = base_context.get("turn_execution_repair_contract", {})
        if isinstance(turn_repair, dict) and turn_repair:
            turn_repair_instruction = (
                " This is a parameter-only turn repair: set angle_deg no greater "
                f"than {turn_repair.get('maximum_next_angle_deg')} and no lower "
                f"than {turn_repair.get('minimum_angle_deg')}; keep duration_s="
                f"{turn_repair.get('previous_duration_s')} and preserve direction, "
                "roles, travel direction, and radius. Override any draft outside "
                "that hard envelope."
            )
        wall_turn_instruction = ""
        wall_turn_envelope = base_context.get("host_wall_turn_angle_envelope", {})
        if isinstance(wall_turn_envelope, dict) and wall_turn_envelope:
            wall_turn_instruction = (
                " Authored-wall finite pivot sweep constrains only this maneuver's "
                "magnitude: preserve turn_direction="
                f"{wall_turn_envelope.get('turn_direction')!r} and the committed "
                "inner/outer roles, and use one shared angle_deg between "
                f"{wall_turn_envelope.get('minimum_angle_deg')} and "
                f"{wall_turn_envelope.get('maximum_angle_deg')}. Prefer one of "
                f"{wall_turn_envelope.get('preferred_model_angle_choices_deg')}. "
                "Do not switch signed direction to evade this angle cap."
            )
        alignment_sign_instruction = ""
        alignment_sign_repair = base_context.get(
            "alignment_sign_repair_contract",
            {},
        )
        if isinstance(alignment_sign_repair, dict) and alignment_sign_repair:
            alignment_sign_instruction = (
                " This is an exact alignment-sign repair: preserve axis="
                f"{alignment_sign_repair.get('semantic_objective')}, duration_s="
                f"{alignment_sign_repair.get('required_duration_s')}, and the "
                "committed reference/corrector roles. Use required_robot_deltas="
                f"{alignment_sign_repair.get('required_robot_deltas')}; do not "
                "swap roles or change magnitude."
            )
        task_plan = base_context.get("task_orchestrator_plan", {})
        if not isinstance(task_plan, dict) or not task_plan:
            task_plan = self.current_task_plan
        committed_phase_id = str(task_plan.get("current_phase_id", "")).strip()
        committed_subtask = task_plan.get("current_subtask", {})
        committed_subtask = (
            committed_subtask
            if isinstance(committed_subtask, dict)
            else {}
        )
        committed_synchronization_group = str(
            committed_subtask.get("subtask_id", "")
        ).strip()
        committed_strategy = base_context.get("committed_execution_strategy", {})
        committed_strategy = (
            committed_strategy if isinstance(committed_strategy, dict) else {}
        )
        committed_method = str(committed_strategy.get("primitive", "")).strip()
        attempt = _attempt_from_artifact_id(artifact_id)
        checkpoint_id = _checkpoint_id_from_context(base_context)
        self.publish_high_level_status(
            "coding",
            attempt=attempt,
            checkpoint_id=checkpoint_id,
            task_plan=task_plan,
            active_model_role="coder",
            next_system_operation=(
                "generate one shared team script"
                if mode == "sync"
                else "generate front/back scripts and integrate the team transaction"
            ),
        )
        safe_id = _safe_artifact_id(artifact_id)
        try:
            if mode == "sync":
                policy = self.call_json_role(
                    "coder",
                    {
                        **base_context,
                        "generation_scope": "sync_team",
                        "instruction": (
                            "Generate both robot scripts in one response. Their executable "
                            "code must be identical."
                            + turn_repair_instruction
                            + wall_turn_instruction
                            + alignment_sign_instruction
                        ),
                    },
                    artifact_id=f"{safe_id}_sync_team",
                    validator=_combine_validators(
                        _validate_sync_team_policy_payload,
                        validator,
                    ),
                )
                calls = ["coder:sync_team"]
            else:
                drafts: dict[str, dict[str, Any]] = {}
                for robot_id in ("front", "back"):
                    def validate_async_draft(
                        payload: dict[str, Any],
                        *,
                        expected: str = robot_id,
                    ) -> None:
                        _validate_async_robot_draft(
                            payload,
                            expected_robot=expected,
                            expected_method=committed_method,
                        )
                        _canonicalize_async_draft_phase_id(
                            payload,
                            expected_robot=expected,
                            committed_phase_id=committed_phase_id,
                        )
                        _canonicalize_async_draft_synchronization_group(
                            payload,
                            expected_robot=expected,
                            committed_synchronization_group=(
                                committed_synchronization_group
                            ),
                        )

                    drafts[robot_id] = self.call_json_role(
                        "coder",
                        {
                            **base_context,
                            "generation_scope": "async_robot",
                            "requested_robot": robot_id,
                            "other_robot_is_generated_separately": True,
                            "instruction": (
                                f"Generate only the {robot_id} robot draft for the committed "
                                "team subtask. Do not emit the other robot script."
                                + turn_repair_instruction
                                + wall_turn_instruction
                                + alignment_sign_instruction
                            ),
                        },
                        artifact_id=f"{safe_id}_{robot_id}",
                        validator=validate_async_draft,
                    )
                integration_context = {
                    key: base_context[key]
                    for key in (
                        "task_stage",
                        "route_discovery",
                        "current_checkpoint_id",
                        "checkpoint_id",
                        "current_subtask",
                        "committed_execution_strategy",
                        "evolution_manager",
                        "approved_strategy_mutation",
                        "mutation_contract",
                        "host_authored_wall_clearance",
                        "host_formation_gate",
                        "superseded_host_formation_gate",
                        "turn_execution_repair_contract",
                        "host_wall_turn_angle_envelope",
                        "alignment_sign_repair_contract",
                        "parameter_tuning_strategy_contract",
                        "host_contract",
                        "recent_rejected_team_transactions",
                        "allowed_sdk_methods",
                    )
                    if key in base_context
                }
                integration_context.update(
                    {
                        "coordination_mode": mode,
                        "call_contract": base_context["call_contract"],
                        "generation_scope": "async_team_review",
                        "draft_authority": (
                            "front_draft and back_draft are the only executable "
                            "inputs for this review. Approve them when they satisfy "
                            "one committed atomic subtask. Do not copy or restore a "
                            "previous team policy."
                        ),
                        "front_draft": drafts["front"],
                        "back_draft": drafts["back"],
                        "review_output_contract": {
                            "decision": "approve_drafts or repair_drafts",
                            "conflicts": ["short cross-robot conflict descriptions"],
                            "repaired_code": {
                                "front": (
                                    "empty when unchanged, otherwise one complete "
                                    "def policy(api) function"
                                ),
                                "back": (
                                    "empty when unchanged, otherwise one complete "
                                    "def policy(api) function"
                                ),
                            },
                            "rationale": "one concise review conclusion",
                        },
                        "instruction": (
                            "Review both drafts jointly. Return approve_drafts with "
                            "empty repaired_code values when they are coherent. Use "
                            "repair_drafts only for a concrete cross-robot conflict."
                            + turn_repair_instruction
                            + wall_turn_instruction
                            + alignment_sign_instruction
                        ),
                    }
                )
                reviewed_policy: dict[str, dict[str, Any]] = {}
                original_review = {
                    "decision": "approve_drafts",
                    "conflicts": [],
                    "repaired_code": {"front": "", "back": ""},
                    "rationale": (
                        "Host-valid front/back drafts preserved without an invalid repair"
                    ),
                }
                original_candidate = _build_async_team_policy_from_review(
                    drafts=drafts,
                    review=original_review,
                    task_plan=task_plan,
                )
                original_policy = _prevalidate_original_async_candidate(
                    original_candidate,
                    validator=validator,
                )

                def validate_team_review(review: dict[str, Any]) -> None:
                    try:
                        _validate_async_team_review(review)
                        candidate = _build_async_team_policy_from_review(
                            drafts=drafts,
                            review=review,
                            task_plan=task_plan,
                        )
                        _validate_async_team_policy_payload(candidate)
                        if validator is not None:
                            validator(candidate)
                    except Exception as exc:
                        if (
                            isinstance(exc, ProviderFatalError)
                            or _is_model_candidate_validation_error(exc)
                        ):
                            raise
                        raise _RoleBoundaryPropagationError(exc) from exc
                    reviewed_policy["policy"] = candidate

                try:
                    review = self.call_json_role(
                        "code_repair",
                        integration_context,
                        artifact_id=f"{safe_id}_team_review",
                        validator=validate_team_review,
                    )
                except ModelRoleStallError as exc:
                    if original_policy is None:
                        raise
                    policy = original_policy
                    review = {
                        **original_review,
                        "source": "host_validated_draft_fallback",
                        "rejected_repair_error": str(exc),
                    }
                else:
                    policy = reviewed_policy.get("policy", {})
                    if not policy:
                        raise ModelRoleError(
                            "async team review returned no validated integrated policy"
                        )
                policy["team_review"] = review
                calls = [
                    "coder:front",
                    "coder:back",
                    "code_repair:team_integration",
                ]
        except ModelRoleError:
            raise
        write_json(
            self.output_dir / f"team_policy_generation_{safe_id}.json",
            {
                "artifact_id": safe_id,
                "coordination_mode": mode,
                "logical_model_calls": calls,
                "policy": policy,
            },
        )
        self.publish_high_level_status(
            "policy_ready",
            attempt=attempt,
            checkpoint_id=checkpoint_id,
            task_plan=task_plan,
            policy=policy,
            next_system_operation="run host preflight, then atomically dispatch both robots",
        )
        return policy

    def analyze_rollout(
        self,
        context: dict[str, Any],
        *,
        artifact_id: str,
        image_path: str | Path,
    ) -> dict[str, Any]:
        analysis = self.call_json_role(
            "vlm_failure_analyzer",
            context,
            artifact_id=artifact_id,
            image_path=image_path,
            validator=_validate_vlm_failure_analysis,
        )
        self.publish_high_level_status(
            "failure_analysis_ready",
            attempt=_attempt_from_artifact_id(artifact_id),
            checkpoint_id=_checkpoint_id_from_context(context),
            active_model_role="vlm_failure_analyzer",
            next_system_operation=(
                "combine this visual diagnosis with authoritative physics evidence"
            ),
            details={
                "summary": analysis.get("summary", ""),
                "failure_type": analysis.get("failure_type", ""),
                "failure_analysis": analysis.get("failure_analysis", {}),
            },
        )
        return analysis

    def decide_mutation(
        self,
        context: dict[str, Any],
        *,
        artifact_id: str,
        image_path: str | Path | None = None,
        validator: RoleValidator | None = None,
    ) -> dict[str, Any]:
        decision = self.call_json_role(
            "evolution_manager",
            context,
            artifact_id=artifact_id,
            image_path=image_path,
            validator=_combine_validators(_validate_evolution_manager, validator),
        )
        self.publish_high_level_status(
            "mutation_level_selected",
            attempt=_attempt_from_artifact_id(artifact_id),
            checkpoint_id=_checkpoint_id_from_context(context),
            active_model_role="evolution_manager",
            next_system_operation=(
                "apply the selected mutation level to the current subtask strategy"
            ),
            details={
                "mutation_level": decision.get("mutation_level", ""),
                "failure_hypothesis": decision.get("failure_hypothesis", ""),
                "preserve": decision.get("preserve", []),
                "mutate": decision.get("mutate", []),
            },
        )
        return decision

    def mutate_strategy(
        self,
        context: dict[str, Any],
        *,
        artifact_id: str,
        validator: RoleValidator | None = None,
    ) -> dict[str, Any]:
        mutation = self.call_json_role(
            "mutation_operator",
            context,
            artifact_id=artifact_id,
            validator=_combine_validators(
                _validate_mutation_operator,
                _candidate_feasibility_validator(validator),
            ),
        )
        active_plan = _task_plan_with_status_strategy(
            self.current_task_plan,
            mutation.get("child_strategy"),
        )
        self.publish_high_level_status(
            "strategy_mutated",
            attempt=_attempt_from_artifact_id(artifact_id),
            checkpoint_id=_checkpoint_id_from_context(context),
            task_plan=active_plan,
            active_model_role="mutation_operator",
            next_system_operation=(
                "ask the Task Orchestrator to replace the complete current subtask"
                if mutation.get("mutation_level") == "strategy_mutation"
                else "generate executable scripts for the mutated strategy"
            ),
            details={
                "mutation_level": mutation.get("mutation_level", ""),
                "changed_dimensions": mutation.get("changed_dimensions", []),
                "host_effective_delta": mutation.get("host_effective_delta", {}),
                "mutation_rationale": mutation.get("rationale", ""),
            },
        )
        return mutation

    def repair_code(
        self,
        context: dict[str, Any],
        *,
        artifact_id: str,
        validator: RoleValidator | None = None,
    ) -> dict[str, Any]:
        policy = self.call_json_role(
            "code_repair",
            context,
            artifact_id=artifact_id,
            validator=_combine_validators(_validate_policy_payload, validator),
        )
        self.publish_high_level_status(
            "policy_ready",
            attempt=_attempt_from_artifact_id(artifact_id),
            checkpoint_id=_checkpoint_id_from_context(context),
            policy=policy,
            active_model_role="code_repair",
            next_system_operation="run final team-policy validation and physical preflight",
        )
        return policy

    def propose_memory_update(
        self,
        context: dict[str, Any],
        *,
        artifact_id: str,
    ) -> dict[str, Any]:
        proposal = self.call_json_role(
            "memory_update",
            context,
            artifact_id=artifact_id,
            validator=_validate_memory_update,
        )
        self.publish_high_level_status(
            "memory_update_ready",
            attempt=_attempt_from_artifact_id(artifact_id),
            checkpoint_id=_checkpoint_id_from_context(context),
            active_model_role="memory_update",
            next_system_operation=(
                "promote only if text claims remain consistent with verifier evidence"
            ),
            details={
                "promote": proposal.get("promote", False),
                "strategy_name": proposal.get("strategy_name", ""),
                "mutation_level": proposal.get("mutation_level", ""),
                "learned_rule": proposal.get("learned_rule", ""),
            },
        )
        return proposal

    def call_json_role(
        self,
        role: ModelRole,
        context: dict[str, Any],
        *,
        artifact_id: str,
        image_path: str | Path | None = None,
        validator: RoleValidator | None = None,
        response_retries: int | None = None,
    ) -> dict[str, Any]:
        """Call one role, validate JSON, and persist every response attempt."""

        if role not in ROLE_SYSTEM_PROMPTS:
            raise ModelRoleError(f"unknown model role: {role!r}")
        if image_path is not None:
            image_path = Path(image_path)
            if not image_path.exists():
                raise ModelRoleError(
                    f"{role} image does not exist: {image_path}"
                )
        retries = (
            self.response_retries
            if response_retries is None
            else max(0, int(response_retries))
        )
        base_context = _json_safe_mapping(context)
        self.publish_high_level_status(
            f"model_{role}_running",
            attempt=_attempt_from_artifact_id(artifact_id),
            checkpoint_id=_checkpoint_id_from_context(base_context),
            active_model_role=role,
            next_system_operation=_role_next_system_operation(role),
        )
        last_error = ""
        validation_error_history: list[str] = []
        validation_error_kinds: list[str] = []
        for response_index in range(retries + 1):
            call_context = dict(base_context)
            if last_error:
                cumulative_errors = "\n".join(validation_error_history)
                correction_instruction = (
                    "Return a materially corrected JSON object that satisfies the "
                    "same role contract. Every accumulated error below remains active "
                    "until corrected when it still applies to the same structured "
                    "contract; a primitive-specific enum error from a discarded "
                    "strategy primitive is obsolete. Do not regress a field fixed in "
                    "an earlier retry. "
                    "Each error names an exact invalid field. "
                    "When it requires a list, preserve the intended semantics but "
                    "use JSON syntax such as [\"item\"], never a boolean, object, "
                    "or scalar string. Do not change fields outside this role."
                )
                if (
                    role == "mutation_operator"
                    and "NO_EFFECTIVE_MUTATION" in cumulative_errors
                ):
                    correction_instruction += (
                        " The host executable projection found no mutation at the "
                        "required level. Change a real executable field: strategy "
                        "mutation changes primitive or symbolic objective; coordination "
                        "mutation changes mode or the front/back temporary-role mapping. "
                        "Changing prose, IDs, rationale, or labels does not count."
                    )
                if (
                    role == "task_orchestrator"
                    and "physical lifecycle forbids api." in last_error
                ):
                    correction_instruction += (
                        " The trusted payload lifecycle is authoritative: replace the "
                        "forbidden current_subtask.strategy.primitive, remove stale "
                        "deferred/inactive payload claims and stale dropped-payload "
                        "claims, and never select any method listed in "
                        "host_lifecycle_gate.forbidden_methods."
                    )
                if (
                    role == "task_orchestrator"
                    and "final lifecycle phase cannot activate" in cumulative_errors
                ):
                    correction_instruction += (
                        " The payload is not physically ready for release. Keep the "
                        "place/release/exit phase pending, restore the preceding final-"
                        "approach phase as current, and choose a transport, turn, or "
                        "alignment subtask from current geometry. Do not use backward "
                        "loaded transport as a substitute for post-release exit."
                    )
                if (
                    role == "task_orchestrator"
                    and "semantic_objective for api." in cumulative_errors
                    and "must be one of" in cumulative_errors
                ):
                    correction_instruction += (
                        " Replace current_subtask.strategy.semantic_objective with "
                        "exactly one token from the allowed values printed in the error. "
                        "Do not paraphrase that token or describe a desired effect."
                    )
                if (
                    role == "task_orchestrator"
                    and "coordinated_pivot role geometry requires" in last_error
                ):
                    correction_instruction += (
                        " This is a role-metadata repair, not a new strategy choice. "
                        "Preserve the coordinated_pivot turn direction from the latest "
                        "response and copy the exact front=inner/outer and back=inner/"
                        "outer assignment printed in the latest validation error. Do "
                        "not switch primitive or turn direction while repairing roles."
                    )
                if (
                    role == "task_orchestrator"
                    and "parameter_tuning strategy contract requires"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Copy primitive, semantic_objective, coordination_mode, and "
                        "the front/back role values exactly from "
                        "parameter_tuning_strategy_contract. Do not reselect the signed "
                        "turn from the older checkpoint episode; this repair preserves "
                        "the rejected transaction and changes only its later magnitude."
                    )
                if (
                    role == "task_orchestrator"
                    and "payload already completed this route turn"
                    in cumulative_errors
                    and "formation synchronization probe failed repeatedly"
                    not in cumulative_errors
                    and "requires coordinated_alignment at this checkpoint"
                    not in cumulative_errors
                    and "strategy_mutation has no alternative "
                    "coordinated_linear_transport objective"
                    not in cumulative_errors
                ):
                    correction_instruction += (
                        " End the pivot subtask. Set the current primitive to "
                        "coordinated_linear_transport with coordination_mode=sync, "
                        "then choose exactly one of forward, backward, left, or right "
                        "from the current global view, active stage goal, and visible "
                        "free space. Do not copy the accepted pivot merely because it "
                        "created the current checkpoint."
                    )
                if (
                    role == "task_orchestrator"
                    and (
                        "requires coordinated_alignment at this checkpoint"
                        in cumulative_errors
                        or (
                            "formation synchronization probe failed repeatedly"
                            in cumulative_errors
                            and "payload already completed this route turn"
                            in cumulative_errors
                        )
                        or (
                            "strategy_mutation has no alternative "
                            "coordinated_linear_transport objective"
                            in cumulative_errors
                            and "payload already completed this route turn"
                            in cumulative_errors
                        )
                    )
                ):
                    correction_instruction += (
                        " The completed route turn and unavailable alternative linear "
                        "motion leave exactly one executable repair "
                        "family at this checkpoint: set primitive="
                        "coordinated_alignment and coordination_mode=async. Choose "
                        "axis longitudinal, lateral, or yaw from measured relative "
                        "formation error, and assign one immutable robot ID as "
                        "reference and the other as corrector. Do not emit "
                        "coordinated_linear_transport or coordinated_pivot, and do "
                        "not repeat a checkpoint-local rejected alignment relation."
                    )
                if (
                    role == "task_orchestrator"
                    and "contradicts current_runtime_state" in cumulative_errors
                ):
                    correction_instruction += (
                        " Treat current_state_authority as the present checkpoint. "
                        "Delete every claim that the currently bilateral-supported box "
                        "is dropped, unsupported, or has zero support force; rejected "
                        "rollout damage is historical causal evidence only."
                    )
                if (
                    role == "task_orchestrator"
                    and "primitive must be a safe SDK method" in last_error
                ):
                    allowed = last_error.partition("allowed=")[2].strip()
                    correction_instruction += (
                        " Replace current_subtask.strategy.primitive with one exact "
                        "unchanged token from exact_sdk_primitive_names"
                        + (f" ({allowed})" if allowed else "")
                        + ". Do not invent, combine, rename, or paraphrase an SDK method."
                    )
                if (
                    role == "evolution_manager"
                    and "failure evidence requires" in last_error
                ):
                    required_level = last_error.partition(
                        "failure evidence requires "
                    )[2].partition(";")[0].strip()
                    correction_instruction += (
                        f" Set mutation_level exactly to {required_level} and keep "
                        "the diagnosis consistent with that level."
                    )
                if (
                    role == "evolution_manager"
                    and "strategy_mutation preserve may not freeze" in last_error
                ):
                    correction_instruction += (
                        " Keep only the global mission, current route phase, accepted "
                        "checkpoint progress, payload support, and physical safety in "
                        "preserve. Move the parent primitive, sync/async mode, formation, "
                        "and temporary roles out of preserve because the child SDK "
                        "contract may require them to change."
                    )
                if (
                    role == "evolution_manager"
                    and "may name one action parameter field but may not choose"
                    in last_error
                ):
                    correction_instruction += (
                        " Keep the single intended parameter field name in mutate, "
                        "but remove all old values, proposed values, equations, and "
                        "numeric ranges. The later parameter role chooses the value."
                    )
                if (
                    role in {"evolution_manager", "mutation_operator"}
                    and "may not output numerical action parameter fields"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Remove every API call expression, parameter name, and number "
                        "from the entire response, especially inheritance as well as "
                        "parent_strategy and child_strategy. inheritance must contain "
                        "only parameter-free semantic lineage text. Use symbolic values "
                        "such as forward, left, clockwise, or longitudinal for "
                        "semantic_objective; the Coder owns magnitudes."
                    )
                if (
                    role == "mutation_operator"
                    and "may name one field only in changed_dimensions"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Keep exactly the single intended parameter field name in "
                        "changed_dimensions. The reason may cite the rejected observed "
                        "value as failure evidence, but it must not prescribe a child "
                        "value or range. Remove all other parameter names and every "
                        "parameter from parent_strategy, child_strategy, and inheritance. "
                        "The later Coder chooses the bounded value."
                    )
                if (
                    role == "mutation_operator"
                    and (
                        "strategy_mutation must change the executable primitive"
                        in cumulative_errors
                        or (
                            "semantic_objective for api." in cumulative_errors
                            and "must be exactly" in cumulative_errors
                        )
                    )
                ):
                    correction_instruction += (
                        " A static motion primitive has one fixed semantic objective, "
                        "so renaming forward to stabilize or changing only roles/text "
                        "does not mutate executable behavior. For strategy_mutation, "
                        "select a genuinely different SDK primitive or a genuinely "
                        "different valid objective of a multi-objective coordination "
                        "primitive."
                    )
                if (
                    role == "mutation_operator"
                    and "payload already completed this route turn"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " The current corner is complete: coordinated_pivot is forbidden "
                        "for every remaining retry. If the rejected parent was "
                        "coordinated_linear_transport and the mutation level is strategy, "
                        "choose a different exact body-frame objective from backward, "
                        "left, or right, or choose coordinated_alignment with one exact "
                        "axis token longitudinal, lateral, or yaw and valid "
                        "reference/corrector roles. Do not return to pivot."
                    )
                if (
                    role == "mutation_operator"
                    and "child strategy mode" in last_error
                    and "is invalid for api." in last_error
                ):
                    method = (
                        last_error.rpartition("api.")[2]
                        .strip()
                        .split(maxsplit=1)[0]
                        .rstrip(".,;:")
                    )
                    legal_modes = sorted(coordination_modes_for_method(method))
                    if legal_modes:
                        correction_instruction += (
                            " The child primitive's hard SDK contract requires "
                            "coordination_mode="
                            f"{legal_modes[0]!r}. Strategy mutation must override an "
                            "incompatible parent or Evolution Manager preservation item. "
                            "Update formation_strategy, synchronization_intent, and both "
                            "temporary role assignments to match that mode."
                        )
                    if method == "coordinated_pivot":
                        correction_instruction += (
                            " For coordinated_pivot, assign exactly one robot role=inner "
                            "and the other role=outer in the same async transaction."
                        )
                if (
                    role == "mutation_operator"
                    and "child semantic_objective must be one of" in last_error
                ):
                    allowed = last_error.partition("must be one of")[2].strip()
                    correction_instruction += (
                        " Keep the latest child_strategy.primitive unchanged and replace "
                        "child_strategy.semantic_objective with exactly one "
                        f"unchanged token from {allowed}. Do not use descriptive prose "
                        "such as align_and_advance. An objective enum error produced by "
                        "a different child primitive on an earlier retry is obsolete. "
                        "Keep both robot objective strings consistent with the latest "
                        "child primitive and objective."
                    )
                if (
                    role == "task_orchestrator"
                    and "requires exactly one inner and one outer role"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Set current_subtask.team_assignment front/back roles to "
                        "the exact tokens inner and outer, one each. Do not use pivot, "
                        "swing, pivot_inner, swing_outer, or any decorated alias."
                    )
                if (
                    role == "task_orchestrator"
                    and "coordinated_pivot role geometry requires"
                    in cumulative_errors
                ):
                    required_mapping = last_error.partition(
                        "coordinated_pivot role geometry requires "
                    )[2].partition(";")[0].strip()
                    correction_instruction += (
                        " Keep the object keys front and back, then copy the "
                        "host-measured role mapping exactly"
                        + (
                            f": {required_mapping}."
                            if required_mapping
                            else "."
                        )
                        + " Turning never renames either robot."
                    )
                if (
                    role == "task_orchestrator"
                    and "current_subtask.team_assignment must contain exactly front and back"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Replace the team_assignment object keys with exactly front "
                        "and back. Put each selected role inside its robot object as "
                        "{\"front\": {\"role\": \"...\", \"objective\": \"...\"}, "
                        "\"back\": {\"role\": \"...\", \"objective\": \"...\"}}. "
                        "Never use role names such as inner and outer as object keys; "
                        "you must decide which physical robot receives each role."
                    )
                if (
                    role == "mutation_operator"
                    and "requires exactly one inner and one outer role"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Set child_strategy.team_assignment roles to the exact tokens "
                        "inner and outer, one each. Do not prefix or suffix those role "
                        "tokens."
                    )
                if (
                    role == "mutation_operator"
                    and "coordinated_pivot child role geometry requires"
                    in cumulative_errors
                ):
                    required_mapping = last_error.partition(
                        "coordinated_pivot child role geometry requires "
                    )[2].partition(";")[0].strip()
                    correction_instruction += (
                        " Keep team_assignment keyed by front and back and copy "
                        "the host-measured child role mapping exactly"
                        + (
                            f": {required_mapping}."
                            if required_mapping
                            else "."
                        )
                        + " Do not swap IDs when mutating temporary roles."
                    )
                if (
                    role == "task_orchestrator"
                    and "requires exactly one reference and one corrector role"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Set current_subtask.team_assignment front/back roles to "
                        "the exact tokens reference and corrector, one each. The "
                        "reference holds zero delta and the corrector applies the "
                        "single relative alignment correction."
                    )
                if (
                    role == "task_orchestrator"
                    and "requires one route-neutral formation repair"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " This checkpoint has a mandatory local physical precondition. "
                        "Copy host_formation_gate.primitive, semantic_objective, "
                        "coordination_mode, and the immutable front/back team_roles exactly "
                        "into current_subtask.strategy and current_subtask.team_assignment. "
                        "Set one reference and one corrector exactly as named by the gate. "
                        "Do not keep or combine the previous route-motion primitive in this "
                        "subtask. This one-shot alignment repair does not choose the later "
                        "route direction."
                    )
                if (
                    role == "task_orchestrator"
                    and "checkpoint-local exhausted coordination relation"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Your previous current_subtask repeated an exact physical "
                        "counterexample and is unavailable at this checkpoint. Do not "
                        "copy its primitive/objective/mode/front-back-role tuple again. "
                        "If evolution_manager selected strategy_mutation, change the "
                        "executable primitive or semantic objective using the current "
                        "global image and turn_episode. If it selected "
                        "coordination_mutation, change the temporary front/back role "
                        "mapping. Return a complete replacement current_subtask."
                    )
                if (
                    role == "task_orchestrator"
                    and "strategy-replan must" in cumulative_errors
                ):
                    correction_instruction += (
                        " This is the mandatory post-mutation Orchestrator replan. "
                        "Copy strategy_replan_contract.approved_execution_strategy "
                        "exactly, including primitive, semantic objective, mode, "
                        "formation strategy, synchronization intent, and immutable "
                        "front/back roles. Replace the complete current_subtask. "
                        "Rewrite objective, guidance, completion_condition, both "
                        "front/back role objectives, and rationale so each explicitly "
                        "describes the approved maneuver using the marker terms named "
                        "by the validation error. Do not retain prose from the rejected "
                        "parent subtask."
                    )
                if (
                    role == "mutation_operator"
                    and "checkpoint-local exhausted coordination relation"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " The rejected child_strategy is an exact checkpoint-local "
                        "physical counterexample. Read "
                        "checkpoint_exhausted_coordination_relations and replace it: "
                        "strategy_mutation must change primitive or semantic objective; "
                        "coordination_mutation must change the temporary front/back "
                        "roles. Do not preserve that tuple in inheritance."
                    )
                if (
                    role in {"task_orchestrator", "mutation_operator"}
                    and "superseded host formation gate relation"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " The exact primitive/objective/mode/front-back-role relation "
                        "inside superseded_host_formation_gate was physically executed "
                        "and regressed alignment. It is negative evidence, not a mandatory "
                        "gate. Preserve the global phase and residual formation-repair goal, "
                        "but select a different semantic objective, primitive, or temporary "
                        "role relation according to the Evolution Manager mutation level."
                    )
                if (
                    role == "evolution_manager"
                    and "strategy_mutation inheritance may not freeze"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Remove primitive, semantic objective, coordination mode, and "
                        "temporary role tokens from inheritance. For strategy_mutation, "
                        "inherit only the global mission, current route phase, accepted "
                        "checkpoint progress, payload support, and physical safety."
                    )
                if (
                    role == "evolution_manager"
                    and "strategy_mutation mutate may name abstract dimensions only"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Evolution Manager chooses the mutation level and dimension "
                        "names only; it does not select the replacement action. Replace "
                        "concrete mutate entries with abstract field names such as "
                        "\"maneuver family\", \"semantic objective\", or "
                        "\"formation strategy\". Remove API names, directions, modes, "
                        "roles, colons, and assigned values. Strategy Mutation will "
                        "choose the new executable contract."
                    )
                if (
                    role == "evolution_manager"
                    and "strategy_mutation forbidden may not prohibit changing"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Do not forbid changes to primitive, semantic objective, "
                        "coordination mode, or temporary roles during strategy_mutation. "
                        "Forbid only unsafe outcomes, exact counterexample replay, and "
                        "loss of accepted physical progress."
                    )
                if (
                    role == "mutation_operator"
                    and "requires exactly one reference and one corrector role"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Set child_strategy.team_assignment roles to the exact "
                        "tokens reference and corrector, one each. Do not use generic "
                        "payload_stabilizer roles for coordinated_alignment."
                    )
                if (
                    role == "mutation_operator"
                    and "no untried model-supported distance tier"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " The selected loaded-linear semantic objective has no legal "
                        "Coder parameter tier at this checkpoint: every in-clearance "
                        "tier was physically rejected and every larger tier exceeds "
                        "authored-wall clearance. Change child_strategy primitive or "
                        "semantic_objective using current visual and physical evidence; "
                        "do not retry that exhausted direction with another magnitude."
                    )
                if (
                    role == "code_repair"
                    and "coordinated_alignment requires a relative front/back delta"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Return two async coordinated_alignment scripts with the same "
                        "axis and duration. The committed reference robot must use "
                        "delta=0.0 and the committed corrector robot must use one "
                        "nonzero signed delta. Identical calls are invalid."
                    )
                if (
                    role in {"coder", "code_repair"}
                    and "repeats the exact verifier-rejected executable AST"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Read host_contract.forbidden_executable_transactions and do not "
                        "reproduce the blocked normalized transaction. Preserve the "
                        "committed primitive, direction, coordination mode, and robot roles, "
                        "then change one legal Coder-owned literal within the supplied "
                        "physics envelope. Strategy-level replacement remains outside this "
                        "role and must occur before Coder when no legal literal remains."
                    )
                if (
                    role == "code_repair"
                    and "async team policy scripts must use coordination_mode=async"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Set coordination_mode to the exact token async on both "
                        "assignments and both robot_scripts; do not omit the field."
                    )
                if (
                    role == "coder"
                    and "payload-cooperative skill and conflicts with contact_intent"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " Repair the draft metadata field contact_intent to exactly "
                        '"maintain_payload_support". Do not add contact_intent as an '
                        "api method argument and do not change the committed executable "
                        "primitive; contact intent belongs only in the Coder JSON envelope."
                    )
                if (
                    role == "code_repair"
                    and (
                        "api.coordinated_pivot requires explicit parameters"
                        in cumulative_errors
                        or "requires matching front/back travel_direction"
                        in cumulative_errors
                    )
                ):
                    correction_instruction += (
                        " In both pivot scripts, explicitly include identical "
                        "turn_direction, angle_deg, travel_direction=\"forward\", "
                        "duration_s, and swing_radius_m values. Preserve the already "
                        "selected values. Only role may differ: one inner and one outer. "
                        "Do not omit travel_direction or swing_radius_m from the inner "
                        "call even though its translation is stationary."
                    )
                if (
                    role == "code_repair"
                    and "angle_deg must be reduced by at least half"
                    in cumulative_errors
                ):
                    correction_instruction += (
                        " The front/back drafts are provisional and may both be "
                        "wrong. Read turn_execution_repair_contract and replace both "
                        "angle_deg values with one shared value no greater than "
                        "maximum_next_angle_deg. Keep previous_duration_s unchanged "
                        "and preserve direction, roles, travel direction, and radius."
                    )
                if (
                    role == "vlm_failure_analyzer"
                    and "missing keys: ['failure_analysis']" in last_error
                ):
                    correction_instruction += (
                        " Add this required top-level object without removing the visual "
                        "fields: \"failure_analysis\": {\"outcome\": \"accepted\" or "
                        "\"rejected\", \"primary_failure\": \"none\" or one concise "
                        "failure label, \"causal_chain\": [], \"physical_evidence\": [], "
                        "\"visual_evidence\": [], \"strategy_defect\": false, "
                        "\"coordination_defect\": false, \"magnitude_defect\": false, "
                        "\"reusable_context\": [], \"confidence\": 0.0}. Replace the "
                        "example values using the supplied physics and image evidence. "
                        "Do not return the legacy visual-only schema again."
                    )
                if role == "vlm_failure_analyzer" and any(
                    marker in cumulative_errors
                    for marker in (
                        "exact JSON boolean",
                        "failure_type is safety-only",
                        "team_heading_relative_to_target must be exactly",
                        "team_heading_relative_to_visible_free_space must be exactly",
                        "team_payload_posture_relative_to_visible_passage must be exactly",
                        "team_footprint_clearance must be exactly",
                        "route_direction_risk_basis must be exactly",
                    )
                ):
                    correction_instruction += (
                        " Repair the exact field named by the validation error. "
                        "All boolean fields, including immediate_route_direction_risk "
                        "and the three nested defect flags, must be literal JSON true "
                        "or false; strings such as \"High\", \"Low\", \"yes\", or "
                        "\"no failure\" are invalid. Top-level failure_type is "
                        "safety-only: use none, payload_drop, robot_fall, "
                        "payload_wall_contact, or manual_review_required. Put route "
                        "or planning diagnoses only in "
                        "failure_analysis.primary_failure. Use the exact route enum "
                        "tokens printed in the schema and validation error."
                    )
                if (
                    role == "task_orchestrator"
                    and "host turn-clearance replan required" in cumulative_errors
                ):
                    correction_instruction += (
                        " The current signed turn is wall-blocked, so its normal "
                        "turn_episode continuity lock is suspended for exactly one "
                        "turn-setup maneuver. Do not emit that blocked turn again. Read "
                        "host_authored_wall_clearance and the global view, then choose "
                        "either an unblocked signed coordinated_pivot with the geometry-"
                        "correct immutable front/back inner/outer roles, or one sync "
                        "coordinated_linear_transport whose semantic_objective is listed "
                        "in open_translation_directions. Preserve the active route phase."
                    )
                if (
                    role == "task_orchestrator"
                    and any(
                        marker in cumulative_errors
                        for marker in (
                            "authored-wall geometry marks the proposed body-frame translation",
                            "current_subtask.navigation.world_direction_xy contradicts",
                            "declared direction contradicts the active connected route segment",
                            "accepted straight-passage saturation requires a geometry-changing next subtask",
                            "shared team heading is already within the declared local navigation tolerance",
                        )
                    )
                ):
                    route_conflict = base_context.get(
                        "route_clearance_conflict_contract",
                        {},
                    )
                    route_conflict = (
                        route_conflict if isinstance(route_conflict, dict) else {}
                    )
                    if route_conflict.get("active", False):
                        correction_instruction += (
                            " The active route and local wall constraints form a "
                            "translation conflict, not a route reversal request. Keep "
                            "current_subtask.navigation.world_direction_xy exactly "
                            f"{route_conflict.get('preserve_navigation_world_direction_xy')}. "
                            "Choose a geometry repair from the measured feasible sets: "
                            "coordinated_pivot sign in "
                            f"{route_conflict.get('safe_heading_correction_turn_directions', [])}, "
                            "or sync coordinated_linear_transport direction in "
                            f"{route_conflict.get('route_reentry_translation_directions', [])}. "
                            "Do not emit the inverse route translation. The navigation "
                            "vector is the persistent route reference, not the local action "
                            "vector; keep it unchanged even when selecting a lateral re-entry. "
                            "Only turn signs in safe_heading_correction_turn_directions are "
                            "available; geometrically_clear_but_unexecutable_turns are not "
                            "valid SONIC actions."
                        )
                if (
                    role == "task_orchestrator"
                    and "points away from the physical target" in cumulative_errors
                ):
                    projection = base_context.get(
                        "trusted_body_frame_target_projection", {}
                    )
                    projection = (
                        projection if isinstance(projection, dict) else {}
                    )
                    required_direction = str(
                        projection.get("required_target_facing_direction", "")
                    ).strip()
                    required_alignment = projection.get(
                        "required_target_facing_alignment"
                    )
                    target_facing = projection.get(
                        "target_facing_translation_directions", []
                    )
                    target_facing = (
                        target_facing if isinstance(target_facing, list) else []
                    )
                    correction_instruction += (
                        " The deterministic body-frame target projection overrides "
                        "camera-space route prose. "
                    )
                    if required_direction:
                        correction_instruction += (
                            "If the repaired subtask remains "
                            "coordinated_linear_transport, set "
                            "semantic_objective to exactly "
                            f"{required_direction!r} (alignment="
                            f"{required_alignment}); do not emit the rejected "
                            "target-away direction again. "
                        )
                    correction_instruction += (
                        "The non-away body-frame directions are "
                        f"{target_facing}. Alternatively select a genuinely "
                        "geometry-changing pivot and preserve the active route phase."
                    )
                if (
                    role == "task_orchestrator"
                    and "world-point navigation target" in cumulative_errors
                    and "heading_alignment_required=true" in cumulative_errors
                ):
                    turn_clearance = base_context.get(
                        "turn_clearance_replan_contract", {}
                    )
                    turn_clearance = (
                        turn_clearance if isinstance(turn_clearance, dict) else {}
                    )
                    if turn_clearance.get("active", False):
                        correction_instruction += (
                            " The ordinary point-reference rule is suspended for one "
                            "checkpoint-local turn-clearance setup because the preserved "
                            f"{turn_clearance.get('preserve_turn_direction')} turn cannot "
                            "fit the calibrated pivot domain. Keep navigation.reference "
                            f"exactly {turn_clearance.get('preserve_navigation_reference')!r}, "
                            "select sync coordinated_linear_transport with semantic_objective "
                            "exactly in "
                            f"{turn_clearance.get('allowed_translation_directions', [])}, "
                            "and keep the same route phase active. This setup cannot prove "
                            "route or phase completion; resume the preserved signed turn "
                            "from the next accepted checkpoint."
                        )
                    else:
                        correction_instruction += (
                            " The point-reference heading contract currently permits no "
                            "linear translation. Keep navigation.reference unchanged and "
                            "select coordinated_pivot with semantic_objective exactly from "
                            "local_navigation_contract.allowed_turn_directions. Use the "
                            "geometry-correct immutable front/back inner/outer roles, make "
                            "one bounded turn, then let the next accepted checkpoint replan. "
                            "Do not emit forward, backward, left, or right while "
                            "heading_alignment_required remains true."
                        )
                if (
                    role == "task_orchestrator"
                    and "accepted route reversal guard prohibits" in cumulative_errors
                ):
                    reversal_guard = base_context.get(
                        "accepted_route_reversal_guard", {}
                    )
                    reversal_guard = (
                        reversal_guard
                        if isinstance(reversal_guard, dict)
                        else {}
                    )
                    blocked_return = str(
                        reversal_guard.get(
                            "blocked_immediate_return_direction", ""
                        )
                    ).strip()
                    latest_direction = str(
                        reversal_guard.get("latest_direction", "")
                    ).strip()
                    correction_instruction += (
                        f" Do not emit coordinated_linear_transport:{blocked_return}. "
                        "That one immediate return would complete an accepted A-B-A "
                        "route loop. The host does not require a pivot and does not "
                        "select the replacement action. Re-read the current global "
                        "view and trusted physics, then independently continue "
                        f"{latest_direction!r} or choose any other executable non-return "
                        "maneuver while preserving the current physical checkpoint."
                    )
                if (
                    role == "task_orchestrator"
                    and "exact prohibited inverse" in cumulative_errors
                ):
                    continuation = base_context.get(
                        "accepted_linear_continuation_contract", {}
                    )
                    continuation = (
                        continuation if isinstance(continuation, dict) else {}
                    )
                    clearance = base_context.get(
                        "host_authored_wall_clearance", {}
                    )
                    clearance = clearance if isinstance(clearance, dict) else {}
                    blocked_inverse = str(
                        continuation.get("blocked_immediate_inverse", "")
                    ).strip()
                    open_directions = clearance.get(
                        "open_translation_directions", []
                    )
                    open_directions = (
                        open_directions
                        if isinstance(open_directions, list)
                        else []
                    )
                    valid_linear = [
                        str(direction).strip()
                        for direction in open_directions
                        if str(direction).strip()
                        and str(direction).strip() != blocked_inverse
                    ]
                    correction_instruction += (
                        f" The exact linear direction {blocked_inverse!r} is "
                        "forbidden by the active continuation contract. It is "
                        "the inverse itself, not a clearance maneuver. Do not "
                        "emit it again or claim that it satisfies the contract. "
                        "The currently validator-compatible wall-clear linear "
                        f"directions are exactly {valid_linear}. Choose one of "
                        "those, or choose an unblocked coordinated_pivot with "
                        "geometry-correct inner/outer roles. Preserve the active "
                        "route phase."
                    )
                call_context["host_validation_feedback"] = {
                    "previous_response_rejected": True,
                    "error": last_error,
                    "accumulated_errors": list(validation_error_history),
                    "instruction": correction_instruction,
                }
                machine_feedback = _machine_readable_validation_feedback(last_error)
                if machine_feedback:
                    call_context["host_validation_feedback"]["machine_readable"] = (
                        machine_feedback
                    )
            user_prompt = build_role_user_prompt(role, call_context)
            system_prompt = ROLE_SYSTEM_PROMPTS[role]
            if _formal_v50_lite_context(base_context):
                system_prompt += V50_LITE_FORMAL_SYSTEM_SUFFIX
            started = time.time()
            before_usage_count = _client_record_count(self.client)
            raw = ""
            parsed: dict[str, Any] | None = None
            raw_parsed: dict[str, Any] | None = None
            raw_parsed_snapshot: dict[str, Any] | None = None
            schema_normalizations: list[str] = []
            valid = False
            boundary_error: Exception | None = None
            # Schema construction is Host-owned. A bad schema is a framework
            # defect and must never be converted into a model response retry.
            schema = role_output_schema(role, call_context)
            validate_strict_json_schema(schema)
            try:
                if image_path is None:
                    structured_generate = getattr(
                        self.client,
                        "generate_structured",
                        None,
                    )
                    if callable(structured_generate):
                        raw = structured_generate(
                            system=system_prompt,
                            user=user_prompt,
                            schema_name=f"{role}_output",
                            schema=schema,
                        )
                    else:
                        raw = self.client.generate(
                            system=system_prompt,
                            user=user_prompt,
                        )
                else:
                    structured_analyze = getattr(
                        self.client,
                        "analyze_image_structured",
                        None,
                    )
                    if callable(structured_analyze):
                        raw = structured_analyze(
                            system=system_prompt,
                            user=user_prompt,
                            image_path=image_path,
                            schema_name=f"{role}_output",
                            schema=schema,
                        )
                    else:
                        raw = self.client.analyze_image(
                            system=system_prompt,
                            user=user_prompt,
                            image_path=image_path,
                        )
                raw_parsed = _parse_json_object(raw)
                # Normalizers edit nested mappings in place. Preserve the provider
                # payload so the audit never attributes Host edits to the model.
                raw_parsed_snapshot = copy.deepcopy(raw_parsed)
                parsed, schema_normalizations = _normalize_role_payload(
                    role,
                    copy.deepcopy(raw_parsed_snapshot),
                )
                validator_normalizations_before = {
                    str(item).strip()
                    for item in (
                        parsed.get("host_schema_normalizations", ())
                        if isinstance(
                            parsed.get("host_schema_normalizations"),
                            list,
                        )
                        else ()
                    )
                    if str(item).strip()
                }
                if validator is not None:
                    validator(parsed)
                validator_normalizations = (
                    parsed.get("host_schema_normalizations", ())
                    if isinstance(
                        parsed.get("host_schema_normalizations"),
                        list,
                    )
                    else ()
                )
                for item in validator_normalizations:
                    normalized_item = str(item).strip()
                    if (
                        normalized_item
                        and normalized_item not in validator_normalizations_before
                        and normalized_item not in schema_normalizations
                    ):
                        schema_normalizations.append(normalized_item)
                valid = True
                error = ""
            except Exception as exc:  # noqa: BLE001 - normalize at role boundary
                audit_exc = (
                    exc.original
                    if isinstance(exc, _RoleBoundaryPropagationError)
                    else exc
                )
                error = f"{type(audit_exc).__name__}: {audit_exc}"
                last_error = error
                typed_failure = classify_typed_generation_failure(error)
                if typed_failure:
                    error_kind = typed_failure.lower()
                elif isinstance(
                    audit_exc,
                    ModelRoleCandidateFeasibilityError,
                ):
                    error_kind = "candidate_feasibility"
                else:
                    error_kind = "structured_validation"
                validation_error_kinds.append(error_kind)
                if (
                    isinstance(audit_exc, ProviderFatalError)
                    or isinstance(exc, _RoleBoundaryPropagationError)
                    or not _is_model_candidate_validation_error(audit_exc)
                ):
                    boundary_error = audit_exc
                if error not in validation_error_history:
                    validation_error_history.append(error)
            _annotate_client_records(
                self.client,
                before_count=before_usage_count,
                role=role,
                artifact_id=_safe_artifact_id(artifact_id),
                response_index=response_index,
            )
            record = {
                "role": role,
                "model_name": self.model_name,
                "shared_model_instance": True,
                "artifact_id": _safe_artifact_id(artifact_id),
                "response_index": response_index,
                "uses_image": image_path is not None,
                "image_path": "" if image_path is None else str(image_path),
                "context_sha256": _sha256_json(base_context),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": raw,
                "raw_parsed_response": raw_parsed_snapshot,
                "parsed_response": parsed,
                "host_schema_normalizations": schema_normalizations,
                "normalization_audit": _normalization_audit(
                    raw_parsed_snapshot,
                    parsed,
                    reasons=schema_normalizations,
                ),
                "valid": valid,
                "validation_error": error,
                "validation_error_kind": (
                    "" if valid else validation_error_kinds[-1]
                ),
                "latency_s": time.time() - started,
                "usage": _latest_client_usage(
                    self.client,
                    before_count=before_usage_count,
                ),
            }
            self._record_call(record)
            if boundary_error is not None:
                raise boundary_error
            if valid and parsed is not None:
                return parsed
            last_kind = validation_error_kinds[-1] if validation_error_kinds else ""
            if last_kind in {item.lower() for item in TYPED_ROLE_FAILURES}:
                if (
                    consecutive_trailing_kind_count(
                        validation_error_kinds,
                        last_kind,
                    )
                    >= MAX_IDENTICAL_TYPED_ROLE_FAILURES
                ):
                    break
        manager = base_context.get("evolution_manager", {})
        manager = manager if isinstance(manager, dict) else {}
        selected_mutation_level = str(manager.get("mutation_level", "")).strip()
        last_typed_failure = classify_typed_generation_failure(last_error)
        typed_feedback = (
            typed_generation_feedback(
                selected_mutation_level=selected_mutation_level,
                failure=last_typed_failure,
                error=last_error,
            )
            if last_typed_failure in TYPED_ROLE_FAILURES
            else {}
        )
        no_feasible_child = bool(
            last_typed_failure in TYPED_ROLE_FAILURES
            or (
                role == "mutation_operator"
                and validation_error_kinds
                and all(
                    kind == "candidate_feasibility"
                    for kind in validation_error_kinds
                )
            )
        )
        cycle_outcome = (
            NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL
            if no_feasible_child
            else ROLE_GENERATION_STALL
        )
        response_count = max(1, len(validation_error_kinds) or (retries + 1))
        self.publish_high_level_status(
            "model_role_failed",
            attempt=_attempt_from_artifact_id(artifact_id),
            checkpoint_id=_checkpoint_id_from_context(base_context),
            active_model_role=role,
            next_system_operation="retry or stop before physical dispatch",
            details={
                "failure_type": last_typed_failure or "structured_validation",
                "validation_error": last_error,
                "response_count": response_count,
                "cycle_outcome": cycle_outcome,
                "selected_mutation_level": selected_mutation_level,
                **typed_feedback,
            },
        )
        raise ModelRoleStallError(
            f"{role} failed structured validation after {response_count} "
            f"response(s): {last_error}",
            role=role,
            artifact_id=_safe_artifact_id(artifact_id),
            response_count=response_count,
            validation_errors=tuple(validation_error_history),
            cycle_outcome=cycle_outcome,
            selected_mutation_level=selected_mutation_level,
            validation_error_kinds=tuple(validation_error_kinds),
            typed_failure=last_typed_failure,
            attempted_change=tuple(typed_feedback.get("attempted_change", ())),
            editable_fields=tuple(
                typed_feedback.get(
                    "editable_fields",
                    editable_fields_for_mutation_level(selected_mutation_level),
                )
            ),
            result=str(typed_feedback.get("result", "")),
        )

    def _record_call(self, record: dict[str, Any]) -> None:
        role = str(record["role"])
        artifact_id = str(record["artifact_id"])
        key = f"{role}:{artifact_id}"
        sequence = self._call_counts.get(key, 0) + 1
        self._call_counts[key] = sequence
        suffix = "" if sequence == 1 else f"_call_{sequence:02d}"
        path = self.output_dir / f"{role}_{artifact_id}{suffix}.json"
        write_json(path, record)
        stored = {
            **record,
            "artifact_path": str(path),
        }
        self._records.append(stored)
        write_json(
            self.output_dir / "role_call_index.json",
            {
                "model_name": self.model_name,
                "shared_model_instance": True,
                "calls": self._records,
            },
        )


@dataclass
class BoundModelRoleClient:
    """Text client adapter for legacy validators with a named model role."""

    pipeline: ModelRolePipeline
    role: Literal["coder", "mutation_operator", "code_repair"]
    artifact_id: str
    context: dict[str, Any] = field(default_factory=dict)

    def generate(self, *, system: str, user: str) -> str:
        role_context = {
            **self.context,
            "role_specific_request": user,
            "legacy_system_contract": system,
        }
        if self.role == "coder":
            payload = self.pipeline.code_policy(
                role_context,
                artifact_id=self.artifact_id,
            )
        elif self.role == "mutation_operator":
            payload = self.pipeline.mutate_strategy(
                role_context,
                artifact_id=self.artifact_id,
            )
        else:
            payload = self.pipeline.repair_code(
                role_context,
                artifact_id=self.artifact_id,
            )
        return json.dumps(payload, ensure_ascii=False)


def _validate_task_plan(payload: dict[str, Any]) -> None:
    required = {
        "task_goal",
        "global_plan",
        "current_phase_id",
        "current_phase_status",
        "current_subtask",
        "constraints",
        "rationale",
    }
    _require_keys(payload, required, role="task_orchestrator")
    phases = payload.get("global_plan")
    if not isinstance(phases, list) or not phases:
        raise ModelRoleError("global_plan must be a non-empty list")
    phase_ids: list[str] = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ModelRoleError(f"global_plan[{index}] must be an object")
        _require_keys(
            phase,
            {"phase_id", "abstract_goal", "completion_evidence", "status"},
            role=f"task_orchestrator.global_plan[{index}]",
        )
        phase_id = str(phase.get("phase_id", "")).strip()
        if not phase_id:
            raise ModelRoleError(f"global_plan[{index}].phase_id must be non-empty")
        phase_ids.append(phase_id)
        if phase.get("status") not in {"pending", "active", "completed"}:
            raise ModelRoleError(
                f"global_plan[{index}].status must be pending, active, or completed"
            )
    if len(set(phase_ids)) != len(phase_ids):
        raise ModelRoleError("global_plan phase_id values must be unique")
    current_phase_id = str(payload.get("current_phase_id", "")).strip()
    if current_phase_id not in set(phase_ids):
        raise ModelRoleError("current_phase_id must reference one global_plan phase")
    if payload.get("current_phase_status") not in {
        "pending",
        "active",
        "completed",
    }:
        raise ModelRoleError(
            "current_phase_status must be pending, active, or completed"
        )
    subtask = payload.get("current_subtask")
    if not isinstance(subtask, dict):
        raise ModelRoleError("current_subtask must be an object")
    legacy_strategy = subtask.get("strategy", {})
    legacy_strategy = legacy_strategy if isinstance(legacy_strategy, dict) else {}
    if (
        "turn_comparison" not in subtask
        and str(legacy_strategy.get("semantic_objective", "")).strip()
        not in {"clockwise", "counterclockwise"}
    ):
        subtask["turn_comparison"] = {
            "active": False,
            "selected_direction": "none",
            "selected_heading": "none",
            "rejected_direction": "none",
            "rejected_heading": "none",
            "decision_basis": "not a signed turn",
        }
        normalizations = payload.setdefault("host_schema_normalizations", [])
        if isinstance(normalizations, list):
            normalizations.append(
                "current_subtask.turn_comparison: added inert legacy default for "
                "a non-turn strategy"
            )
    _require_keys(
        subtask,
        {
            "subtask_id",
            "objective",
            "guidance",
            "strategy",
            "navigation",
            "turn_comparison",
            "team_assignment",
            "completion_condition",
            "preserve",
        },
        role="task_orchestrator.current_subtask",
    )
    strategy = subtask.get("strategy")
    if not isinstance(strategy, dict):
        raise ModelRoleError("current_subtask.strategy must be an object")
    _require_keys(
        strategy,
        {
            "primitive",
            "coordination_mode",
            "semantic_objective",
            "formation_strategy",
            "synchronization_intent",
        },
        role="task_orchestrator.current_subtask.strategy",
    )
    if strategy["coordination_mode"] not in {"sync", "async"}:
        raise ModelRoleError(
            "current_subtask.strategy.coordination_mode must be sync or async"
        )
    navigation = subtask.get("navigation")
    if not isinstance(navigation, dict):
        navigation = _default_subtask_navigation(payload)
        subtask["navigation"] = navigation
        normalizations = payload.setdefault("host_schema_normalizations", [])
        if isinstance(normalizations, list):
            normalizations.append(
                "current_subtask.navigation: added generic semantic reference "
                "for a legacy task plan"
            )
    elif "world_direction_xy" not in navigation:
        navigation["world_direction_xy"] = [0.0, 0.0]
        normalizations = payload.setdefault("host_schema_normalizations", [])
        if isinstance(normalizations, list):
            normalizations.append(
                "current_subtask.navigation.world_direction_xy: added zero "
                "legacy default"
            )
    if (
        isinstance(navigation, dict)
        and "completion_world_direction_xy" not in navigation
        and isinstance(navigation.get("world_direction_xy"), list)
    ):
        navigation["completion_world_direction_xy"] = list(
            navigation["world_direction_xy"]
        )
        normalizations = payload.setdefault("host_schema_normalizations", [])
        if isinstance(normalizations, list):
            normalizations.append(
                "current_subtask.navigation.completion_world_direction_xy: copied "
                "the legacy declared direction"
            )
    _require_keys(
        navigation,
        {
            "reference",
            "arrival_condition",
            "world_direction_xy",
            "completion_world_direction_xy",
        },
        role="task_orchestrator.current_subtask.navigation",
    )
    if navigation["reference"] not in {
        "final_target",
        "visible_connected_free_space",
        "formation_only",
        "payload_lifecycle",
    }:
        raise ModelRoleError(
            "current_subtask.navigation.reference is not host-resolvable; use "
            "exactly one of final_target, visible_connected_free_space, "
            "formation_only, or payload_lifecycle"
        )
    expected_navigation = _default_subtask_navigation(payload)
    if (
        expected_navigation["reference"] == "final_target"
        and navigation["reference"] != "final_target"
    ):
        raise ModelRoleError(
            "the current route phase immediately precedes the payload placement/"
            "release phase, so current_subtask.navigation.reference must be "
            "'final_target'. This is a task-structure contract, not a scene-"
            "specific waypoint or turn instruction"
        )
    direction_x, direction_y = _navigation_direction_vector(
        navigation.get("world_direction_xy"),
        field="world_direction_xy",
    )
    completion_x, completion_y = _navigation_direction_vector(
        navigation.get("completion_world_direction_xy"),
        field="completion_world_direction_xy",
    )
    if navigation["reference"] == "visible_connected_free_space":
        if direction_x * direction_x + direction_y * direction_y <= 1.0e-8:
            raise ModelRoleError(
                "visible_connected_free_space requires a non-zero "
                "world_direction_xy selected from the current global view"
            )
        completion_norm = completion_x * completion_x + completion_y * completion_y
        if completion_norm <= 1.0e-8:
            raise ModelRoleError(
                "visible_connected_free_space requires a non-zero "
                "completion_world_direction_xy"
            )
        direction_norm = (direction_x * direction_x + direction_y * direction_y) ** 0.5
        completion_norm = completion_norm**0.5
        agreement = (
            direction_x * completion_x + direction_y * completion_y
        ) / (direction_norm * completion_norm)
        if agreement < 0.999:
            raise ModelRoleError(
                "STRUCTURED_CONTRACT_CONFLICT: navigation.world_direction_xy and "
                "completion_world_direction_xy declare different headings; make "
                "the structured fields internally consistent"
            )
    elif any(
        abs(value) > 1.0e-8
        for value in (direction_x, direction_y, completion_x, completion_y)
    ):
        raise ModelRoleError(
            "world_direction_xy and completion_world_direction_xy are model-authored "
            "only for visible_connected_free_space; use [0, 0] for target, "
            "formation, and lifecycle references"
        )
    _validate_turn_comparison(
        subtask.get("turn_comparison"),
        semantic_objective=str(strategy.get("semantic_objective", "")).strip(),
    )
    _validate_team_assignment(
        subtask.get("team_assignment"),
        role="task_orchestrator.current_subtask.team_assignment",
    )
    _require_string_list(subtask.get("preserve"), "current_subtask.preserve")
    _require_string_list(payload.get("constraints"), "constraints", allow_empty=False)


def _navigation_direction_vector(value: object, *, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ModelRoleError(
            f"current_subtask.navigation.{field} must contain exactly two numeric "
            "world-frame components"
        )
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ModelRoleError(
            f"current_subtask.navigation.{field} must contain exactly two numeric "
            "world-frame components"
        ) from exc
    if not isfinite(x) or not isfinite(y):
        raise ModelRoleError(
            f"current_subtask.navigation.{field} components must be finite"
        )
    return x, y


def _validate_turn_comparison(
    value: object,
    *,
    semantic_objective: str,
) -> None:
    if not isinstance(value, dict):
        raise ModelRoleError("current_subtask.turn_comparison must be an object")
    _require_keys(
        value,
        {
            "active",
            "selected_direction",
            "selected_heading",
            "rejected_direction",
            "rejected_heading",
            "decision_basis",
        },
        role="task_orchestrator.current_subtask.turn_comparison",
    )
    if not isinstance(value.get("active"), bool):
        raise ModelRoleError("current_subtask.turn_comparison.active must be boolean")
    selected = str(value.get("selected_direction", "")).strip().lower()
    rejected = str(value.get("rejected_direction", "")).strip().lower()
    selected_heading = str(value.get("selected_heading", "")).strip()
    rejected_heading = str(value.get("rejected_heading", "")).strip()
    decision_basis = str(value.get("decision_basis", "")).strip()
    signed_turn = semantic_objective in {"clockwise", "counterclockwise"}
    if signed_turn:
        expected_rejected = (
            "counterclockwise" if semantic_objective == "clockwise" else "clockwise"
        )
        if not value["active"]:
            raise ModelRoleError(
                "signed turn strategy requires turn_comparison.active=true"
            )
        if selected != semantic_objective or rejected != expected_rejected:
            raise ModelRoleError(
                "STRUCTURED_CONTRACT_CONFLICT: turn_comparison selected/rejected "
                "directions must match strategy.semantic_objective and its opposite"
            )
        if not selected_heading or not rejected_heading or not decision_basis:
            raise ModelRoleError(
                "active turn_comparison requires non-empty selected_heading, "
                "rejected_heading, and decision_basis"
            )
        return
    if value["active"] or selected != "none" or rejected != "none":
        raise ModelRoleError(
            "non-turn strategy requires turn_comparison.active=false and both "
            "direction fields equal to 'none'"
        )
    if not selected_heading or not rejected_heading or not decision_basis:
        raise ModelRoleError(
            "inactive turn_comparison still requires auditable non-empty heading "
            "placeholders and decision_basis"
        )


def _default_subtask_navigation(payload: dict[str, Any]) -> dict[str, Any]:
    """Infer only a semantic reference for legacy plans, never an action."""

    subtask = payload.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    strategy = subtask.get("strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    method = str(strategy.get("primitive", "")).strip()
    if method in {
        "place_payload_on_arms",
        "place_crouch",
        "release_payload",
        "exit_backward",
        "recover_stand",
        "recover_payload_inward",
        "hold_position",
    }:
        return {
            "reference": "payload_lifecycle",
            "arrival_condition": "satisfy the active physical lifecycle gate",
            "world_direction_xy": [0.0, 0.0],
            "completion_world_direction_xy": [0.0, 0.0],
        }
    if method == "coordinated_alignment":
        return {
            "reference": "formation_only",
            "arrival_condition": "reduce the declared front/back formation error",
            "world_direction_xy": [0.0, 0.0],
            "completion_world_direction_xy": [0.0, 0.0],
        }
    phases = payload.get("global_plan", ())
    current_id = str(payload.get("current_phase_id", "")).strip()
    if isinstance(phases, list):
        current_index = next(
            (
                index
                for index, phase in enumerate(phases)
                if isinstance(phase, dict)
                and str(phase.get("phase_id", "")).strip() == current_id
            ),
            -1,
        )
        if current_index >= 0 and current_index + 1 < len(phases):
            next_phase = phases[current_index + 1]
            next_text = (
                " ".join(
                    str(next_phase.get(key, "")).lower()
                    for key in ("phase_id", "abstract_goal", "completion_evidence")
                )
                if isinstance(next_phase, dict)
                else ""
            )
            payload_terms = any(
                term in next_text for term in ("payload", "box", "object")
            )
            placement_terms = any(
                term in next_text
                for term in ("place", "placement", "release", "deposit", "set down")
            )
            if payload_terms and placement_terms:
                return {
                    "reference": "final_target",
                    "arrival_condition": (
                        "reach the physical final target with a compatible team heading"
                    ),
                    "world_direction_xy": [0.0, 0.0],
                    "completion_world_direction_xy": [0.0, 0.0],
                }
    return {
        "reference": "visible_connected_free_space",
        "arrival_condition": (
            "reach the next visually verified connected free-space transition"
        ),
        # Legacy plans cannot reconstruct a route direction from text alone.
        # The validator rejects this zero vector for new route plans and asks
        # the Orchestrator to ground it in the current global frame.
        "world_direction_xy": [0.0, 0.0],
        "completion_world_direction_xy": [0.0, 0.0],
    }


_NON_ROUTE_COMPLETION_METHODS = frozenset(
    {
        "hold_position",
        "place_crouch",
        "place_payload_on_arms",
        "recover_payload_inward",
        "recover_stand",
    }
)

_LINEAR_ROUTE_METHODS = frozenset(
    {
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
        "coordinated_linear_transport",
    }
)
_TURN_ROUTE_METHODS = frozenset(
    {
        "carry_turn_in_place",
        "turn_clockwise",
        "turn_counterclockwise",
        "coordinated_arc_transport",
        "coordinated_pivot",
    }
)
_STANDARD_ROUTE_PHASE_REQUIREMENTS = {
    "initial_straight": "linear_transport",
    "first_corner": "turn",
    "cross_passage": "linear_transport",
    "second_corner": "turn",
    "final_straight": "linear_transport",
    "payload_placement": "finalized_payload_lifecycle",
}


def _validate_task_plan_phase_transition(
    payload: dict[str, Any],
    *,
    previous_task_plan: dict[str, Any],
    context: dict[str, Any],
) -> None:
    """Keep route phase state monotonic and grounded in the accepted action."""

    _validate_task_plan_matches_mission_topology(payload, context=context)
    previous_phases = previous_task_plan.get("global_plan", [])
    if isinstance(previous_phases, list) and previous_phases:
        _normalize_persistent_phase_ids(
            payload,
            previous_phases=previous_phases,
        )

    phases = payload.get("global_plan", [])
    _normalize_task_plan_to_host_phase_frontier(
        payload,
        phases=phases,
        context=context,
    )
    # Phase normalization can change the semantic navigation reference that is
    # valid for this subtask (for example, route discovery -> final target).
    # Re-run the schema contract against the normalized plan before Coder sees it.
    _validate_task_plan(payload)
    phases = payload.get("global_plan", [])
    current_phase_id = str(payload.get("current_phase_id", "")).strip()
    seen_incomplete_phase = False
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_status = str(phase.get("status", ""))
        if phase_status == "completed":
            if seen_incomplete_phase:
                raise ModelRoleError(
                    "completed route phases must form one contiguous prefix; "
                    "do not skip an incomplete phase"
                )
        else:
            seen_incomplete_phase = True
    first_incomplete = next(
        (
            str(phase.get("phase_id", "")).strip()
            for phase in phases
            if isinstance(phase, dict) and phase.get("status") != "completed"
        ),
        "",
    )
    if first_incomplete and current_phase_id != first_incomplete:
        raise ModelRoleError(
            "current_phase_id must reference the earliest non-completed route phase; "
            f"expected {first_incomplete!r}, got {current_phase_id!r}"
        )
    phase_status_by_id = {
        str(phase.get("phase_id", "")).strip(): str(phase.get("status", "")).strip()
        for phase in phases
        if isinstance(phase, dict)
    }
    current_phase_status = str(payload.get("current_phase_status", "")).strip()
    if phase_status_by_id.get(current_phase_id, "") != current_phase_status:
        raise ModelRoleError(
            "current_phase_status must match the referenced global_plan phase status"
        )
    active_phase_ids = [
        phase_id
        for phase_id, status in phase_status_by_id.items()
        if status == "active"
    ]
    if len(active_phase_ids) > 1:
        raise ModelRoleError("global_plan may contain at most one active phase")
    hard_lifecycle_gate = bool(
        isinstance(context.get("host_lifecycle_gate"), dict)
        and context["host_lifecycle_gate"].get("hard_required", False)
    )
    if first_incomplete and not hard_lifecycle_gate:
        if phase_status_by_id.get(first_incomplete) != "active":
            raise ModelRoleError(
                "the earliest non-completed route phase must be active while "
                "planning a route or coordination action"
            )
        if active_phase_ids != [first_incomplete]:
            raise ModelRoleError(
                "current_phase_id must be the only active route phase"
            )

    phase_evidence = _accepted_route_phase_evidence(phases, context=context)
    if phase_evidence:
        completed_prefix = _completed_phase_prefix_length(phases)
        maximum_prefix = int(phase_evidence["maximum_completed_prefix"])
        if completed_prefix > maximum_prefix:
            next_phase = str(phase_evidence.get("next_unproven_phase_id", "")).strip()
            next_requirement = str(
                phase_evidence.get("next_required_accepted_action", "")
            ).strip()
            raise ModelRoleError(
                "accepted route evidence permits at most "
                f"{maximum_prefix} completed route phase(s); "
                f"{next_phase!r} still requires an accepted {next_requirement} "
                "in chronological checkpoint history. A straight transport cannot "
                "prove a corner, and a rejected action cannot advance the phase."
            )

    if not isinstance(previous_phases, list) or not previous_phases:
        return
    previous_ids = [
        str(phase.get("phase_id", "")).strip()
        for phase in previous_phases
        if isinstance(phase, dict)
    ]
    current_ids = [
        str(phase.get("phase_id", "")).strip()
        for phase in phases
        if isinstance(phase, dict)
    ]
    if previous_ids != current_ids:
        raise ModelRoleError(
            "global_plan is a persistent route backbone; preserve phase IDs and order"
        )

    previous_by_id = {
        str(phase.get("phase_id", "")).strip(): phase
        for phase in previous_phases
        if isinstance(phase, dict)
    }
    newly_completed: list[str] = []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_id = str(phase.get("phase_id", "")).strip()
        previous_status = str(previous_by_id.get(phase_id, {}).get("status", ""))
        current_status = str(phase.get("status", ""))
        if previous_status == "completed" and current_status != "completed":
            raise ModelRoleError(
                f"completed route phase {phase_id!r} cannot be reopened"
            )
        if previous_status != "completed" and current_status == "completed":
            newly_completed.append(phase_id)

    if not newly_completed:
        return

    rollout = context.get("last_accepted_rollout", {})
    rollout = rollout if isinstance(rollout, dict) else {}
    if not bool(rollout.get("success", False)):
        raise ModelRoleError(
            "a route phase cannot advance without a successful accepted rollout"
        )
    accepted_methods = _policy_api_methods(context.get("last_accepted_policy"))
    blocked_methods = sorted(accepted_methods & _NON_ROUTE_COMPLETION_METHODS)
    if blocked_methods:
        raise ModelRoleError(
            "payload lifecycle, recovery, and support-only actions cannot complete "
            "a route phase; keep the previous phase active and plan its actual "
            f"transport or turn subtask. Last accepted methods={blocked_methods}"
        )


def _normalize_task_plan_to_host_phase_frontier(
    payload: dict[str, Any],
    *,
    phases: object,
    context: dict[str, Any],
) -> None:
    """Retire stale phase labels without selecting the next executable action."""

    if not isinstance(phases, list) or not phases:
        return
    evidence = _accepted_route_phase_evidence(phases, context=context)
    if not evidence:
        return
    maximum_prefix = int(evidence["maximum_completed_prefix"])
    candidate_prefix = _completed_phase_prefix_length(phases)
    has_noncontiguous_completion = any(
        isinstance(phase, dict) and phase.get("status") == "completed"
        for phase in phases[candidate_prefix:]
    )
    hard_lifecycle_gate = bool(
        isinstance(context.get("host_lifecycle_gate"), dict)
        and context["host_lifecycle_gate"].get("hard_required", False)
    )
    first_incomplete_status = (
        str(phases[candidate_prefix].get("status", "")).strip()
        if candidate_prefix < len(phases)
        and isinstance(phases[candidate_prefix], dict)
        else ""
    )
    needs_active_normalization = bool(
        not hard_lifecycle_gate
        and candidate_prefix < len(phases)
        and first_incomplete_status != "active"
    )
    minimum_prefix = int(evidence.get("persistent_plan_floor_prefix", 0))
    if (
        minimum_prefix <= candidate_prefix <= maximum_prefix
        and not has_noncontiguous_completion
        and not needs_active_normalization
    ):
        return

    effective_prefix = min(max(candidate_prefix, minimum_prefix), maximum_prefix)
    normalized_phases: list[Any] = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            normalized_phases.append(phase)
            continue
        normalized = dict(phase)
        if index < effective_prefix:
            normalized["status"] = "completed"
        elif index == effective_prefix and not hard_lifecycle_gate:
            normalized["status"] = "active"
        else:
            normalized["status"] = "pending"
        normalized_phases.append(normalized)
    payload["global_plan"] = normalized_phases
    if effective_prefix < len(normalized_phases):
        active_phase = normalized_phases[effective_prefix]
        if isinstance(active_phase, dict):
            payload["current_phase_id"] = str(
                active_phase.get("phase_id", "")
            ).strip()
            payload["current_phase_status"] = str(
                active_phase.get("status", "pending")
            ).strip()

    normalizations = payload.setdefault("host_schema_normalizations", [])
    if not isinstance(normalizations, list):
        normalizations = []
        payload["host_schema_normalizations"] = normalizations
    normalizations.append(
        "global_plan statuses: aligned the persistent completion prefix to the "
        f"trusted physical frontier {effective_prefix} and restored one current phase"
    )


def _reconcile_task_plan_with_accepted_route_evidence(
    task_plan: dict[str, Any],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Retire false phase completion claims after a trusted checkpoint resume."""

    if not task_plan or "accepted_route_action_history" not in context:
        return _json_safe_mapping(task_plan)
    plan = _json_safe_mapping(task_plan)
    phases = plan.get("global_plan", [])
    if not isinstance(phases, list) or not phases:
        return plan
    evidence = _accepted_route_phase_evidence(phases, context=context)
    if not evidence:
        return plan
    completed_prefix = _completed_phase_prefix_length(phases)
    maximum_prefix = int(evidence["maximum_completed_prefix"])
    minimum_prefix = int(evidence.get("persistent_plan_floor_prefix", 0))
    if minimum_prefix <= completed_prefix <= maximum_prefix:
        return plan

    effective_prefix = min(max(completed_prefix, minimum_prefix), maximum_prefix)

    normalized_phases: list[dict[str, Any]] = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            continue
        normalized = dict(phase)
        if index < effective_prefix:
            normalized["status"] = "completed"
        elif index == effective_prefix:
            normalized["status"] = "active"
        else:
            normalized["status"] = "pending"
        normalized_phases.append(normalized)
    plan["global_plan"] = normalized_phases
    if effective_prefix < len(normalized_phases):
        plan["current_phase_id"] = str(
            normalized_phases[effective_prefix].get("phase_id", "")
        ).strip()
        plan["current_phase_status"] = "active"
    plan.pop("current_subtask", None)
    plan.pop("constraints", None)
    plan.pop("rationale", None)
    plan["host_phase_reconciliation"] = {
        "reason": "persisted phase completion differed from trusted physical evidence",
        "previous_completed_prefix": completed_prefix,
        "reconciled_completed_prefix": effective_prefix,
        **evidence,
    }
    return plan


def _accepted_route_phase_evidence(
    phases: object,
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Return the maximum phase prefix supported by trusted physical evidence."""

    if "accepted_route_action_history" not in context:
        return {}
    if not isinstance(phases, list) or not phases:
        return {}
    mission_topology = _mission_topology_from_context(context)
    topology_bound = bool(
        mission_topology
        and len(phases) == len(mission_topology)
        and all(
            isinstance(phase, dict)
            and _normalized_phase_goal(phase.get("abstract_goal", ""))
            == _normalized_phase_goal(goal)
            for phase, goal in zip(phases, mission_topology)
        )
    )
    signatures = [
        _semantic_phase_signature(phase)
        for phase in phases
        if isinstance(phase, dict)
    ]
    legacy_standard_topology = bool(
        not mission_topology
        and signatures
        and all(
            signature in _STANDARD_ROUTE_PHASE_REQUIREMENTS
            for signature in signatures
        )
    )
    if not topology_bound and not legacy_standard_topology:
        return {}

    history = context.get("accepted_route_action_history", {})
    if isinstance(history, dict):
        raw_steps = history.get("steps", [])
    else:
        raw_steps = history
    steps = raw_steps if isinstance(raw_steps, list) else []
    accepted_actions: list[dict[str, Any]] = []
    action_kinds: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        transition_outcome = str(step.get("transition_outcome", "")).strip()
        if (
            transition_outcome not in {"ACCEPT_PROGRESS", "TASK_COMPLETE"}
            or step.get("positive_route_memory_eligible") is not True
        ):
            continue
        kind = _accepted_route_action_kind(step)
        if not kind:
            continue
        action_kinds.append(kind)
        accepted_actions.append(
            {
                "checkpoint_after": str(step.get("checkpoint_after", "")).strip(),
                "method": str(step.get("method", "")).strip(),
                "semantic_objective": str(
                    step.get("semantic_objective", "")
                ).strip(),
                "kind": kind,
            }
        )

    required_kinds = (
        [
            _STANDARD_ROUTE_PHASE_REQUIREMENTS[signature]
            for signature in signatures
        ]
        if legacy_standard_topology
        else ["trusted_physical_transition"] * len(phases)
    )
    action_evidence_maximum_prefix = 0
    for kind in action_kinds:
        if action_evidence_maximum_prefix >= len(required_kinds):
            break
        if kind == required_kinds[action_evidence_maximum_prefix]:
            action_evidence_maximum_prefix += 1

    if (
        action_evidence_maximum_prefix < len(required_kinds)
        and required_kinds[action_evidence_maximum_prefix]
        == "finalized_payload_lifecycle"
        and _trusted_final_lifecycle_complete(context)
    ):
        action_evidence_maximum_prefix += 1
    physical_guard = context.get("host_route_phase_guard", {})
    physical_guard = physical_guard if isinstance(physical_guard, dict) else {}
    if "maximum_completed_prefix" in physical_guard:
        try:
            physical_maximum_prefix = max(
                0,
                min(len(required_kinds), int(physical_guard["maximum_completed_prefix"])),
            )
        except (TypeError, ValueError):
            physical_maximum_prefix = 0
        maximum_prefix = physical_maximum_prefix
    else:
        physical_maximum_prefix = None
        maximum_prefix = 0

    trusted_checkpoint_matches_history = (
        _trusted_checkpoint_matches_accepted_history(
            context,
            history=history,
        )
    )
    # Action names remain useful diagnostics, but they never complete or retain
    # a phase. When the live checkpoint is exactly the manifest frontier, the
    # physical guard is both the lower and upper bound for persistent labels.
    persistent_plan_floor_prefix = (
        maximum_prefix
        if trusted_checkpoint_matches_history
        and physical_maximum_prefix is not None
        else 0
    )

    next_phase_id = ""
    next_requirement = ""
    if maximum_prefix < len(phases):
        phase = phases[maximum_prefix]
        if isinstance(phase, dict):
            next_phase_id = str(phase.get("phase_id", "")).strip()
        next_requirement = required_kinds[maximum_prefix]
    return {
        "source": (
            "mission_topology_and_trusted_physics"
            if topology_bound
            else "accepted_episode_manifest"
        ),
        "maximum_completed_prefix": maximum_prefix,
        "action_evidence_maximum_prefix": action_evidence_maximum_prefix,
        "physical_maximum_completed_prefix": physical_maximum_prefix,
        "persistent_plan_floor_prefix": persistent_plan_floor_prefix,
        "trusted_checkpoint_matches_history": (
            trusted_checkpoint_matches_history
        ),
        "trusted_physical_frontier": (
            {
                key: physical_guard[key]
                for key in (
                    "source",
                    "current_topological_phase",
                    "maximum_completed_prefix",
                    "route_geometry_contract_available",
                    "phase_anchors",
                    "rule",
                )
                if key in physical_guard
            }
            if physical_guard
            else {}
        ),
        "next_unproven_phase_id": next_phase_id,
        "next_required_accepted_action": next_requirement,
        "accepted_actions": accepted_actions,
        "rule": (
            "mission_topology defines phase order while trusted current physics "
            "and phase anchors exclusively decide completion; accepted action "
            "names are diagnostic only"
        ),
    }


def _mission_topology_from_context(context: dict[str, Any]) -> list[str]:
    raw = context.get("mission_topology", ())
    if not isinstance(raw, (list, tuple)):
        return []
    topology = [str(item).strip() for item in raw]
    if not topology or any(not item for item in topology):
        return []
    return topology


def _validate_task_plan_matches_mission_topology(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
) -> None:
    """Bind the model-owned plan to arbitrary scene-supplied phase data."""

    topology = _mission_topology_from_context(context)
    if not topology:
        return
    phases = payload.get("global_plan", [])
    if not isinstance(phases, list) or len(phases) != len(topology):
        raise ModelRoleError(
            "global_plan must contain exactly one phase per mission_topology "
            f"entry: expected {len(topology)} phases, got "
            f"{len(phases) if isinstance(phases, list) else 0}. Do not add "
            "payload initialization or other lifecycle prerequisites as extra "
            "route phases"
        )
    for index, (phase, expected_goal) in enumerate(zip(phases, topology)):
        if not isinstance(phase, dict):
            continue
        actual_goal = str(phase.get("abstract_goal", "")).strip()
        if _normalized_phase_goal(actual_goal) != _normalized_phase_goal(expected_goal):
            raise ModelRoleError(
                f"global_plan[{index}].abstract_goal must copy "
                f"mission_topology[{index}] verbatim; expected {expected_goal!r}, "
                f"got {actual_goal!r}. Scene topology is task data, not a route "
                "phase for the model to insert, remove, or rename"
            )


def _trusted_checkpoint_matches_accepted_history(
    context: dict[str, Any],
    *,
    history: object,
) -> bool:
    """Confirm that rollback returned to the latest accepted manifest state."""

    accepted_history = history if isinstance(history, dict) else {}
    latest_checkpoint = str(
        accepted_history.get("latest_checkpoint", "")
    ).strip()
    if not latest_checkpoint:
        return False
    runtime_candidates = (
        context.get("current_trusted_observation", {}),
        context.get("observation", {}),
    )
    for observation in runtime_candidates:
        observation = observation if isinstance(observation, dict) else {}
        runtime = observation.get("runtime_state", {})
        runtime = runtime if isinstance(runtime, dict) else {}
        checkpoint = runtime.get("checkpoint", {})
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        if str(checkpoint.get("checkpoint_id", "")).strip() == latest_checkpoint:
            return True
    return False


def _accepted_route_action_kind(step: dict[str, Any]) -> str:
    method = str(step.get("method", "")).strip()
    if method in _LINEAR_ROUTE_METHODS:
        return "linear_transport"
    if method in _TURN_ROUTE_METHODS:
        return "turn"
    if method == "release_payload":
        return "payload_release"
    if method == "exit_backward":
        return "crouched_retreat"
    if method == "recover_stand":
        return "stand_recovery"
    return ""


def _trusted_final_lifecycle_complete(context: dict[str, Any]) -> bool:
    trusted_observation = context.get("current_trusted_observation", {})
    trusted_observation = (
        trusted_observation if isinstance(trusted_observation, dict) else {}
    )
    observation = context.get("observation", {})
    observation = observation if isinstance(observation, dict) else {}
    runtime_candidates = (
        trusted_observation.get("runtime_state", {}),
        observation.get("runtime_state", {}),
        context.get("runtime_state", {}),
    )
    for runtime in runtime_candidates:
        if not isinstance(runtime, dict):
            continue
        lifecycle = runtime.get("post_release_lifecycle", {})
        if isinstance(lifecycle, dict) and bool(
            lifecycle.get("stand_recovered", False)
        ):
            return True
    return False


def _completed_phase_prefix_length(phases: object) -> int:
    if not isinstance(phases, list):
        return 0
    completed = 0
    for phase in phases:
        if not isinstance(phase, dict) or phase.get("status") != "completed":
            break
        completed += 1
    return completed


def _normalize_persistent_phase_ids(
    payload: dict[str, Any],
    *,
    previous_phases: list[Any],
) -> None:
    """Restore harmless model-renamed phase IDs without hiding plan changes."""

    phases = payload.get("global_plan", [])
    if not isinstance(phases, list):
        return
    previous_ids = [
        str(phase.get("phase_id", "")).strip()
        for phase in previous_phases
        if isinstance(phase, dict)
    ]
    current_ids = [
        str(phase.get("phase_id", "")).strip()
        for phase in phases
        if isinstance(phase, dict)
    ]
    if previous_ids == current_ids:
        return

    previous_signatures = [
        _semantic_phase_signature(phase)
        for phase in previous_phases
        if isinstance(phase, dict)
    ]
    current_signatures = [
        _semantic_phase_signature(phase)
        for phase in phases
        if isinstance(phase, dict)
    ]
    if (
        len(previous_ids) != len(previous_phases)
        or len(current_ids) != len(phases)
        or len(previous_ids) != len(current_ids)
        or not all(previous_signatures)
        or previous_signatures != current_signatures
    ):
        raise ModelRoleError(
            "global_plan is a persistent route backbone; preserve phase IDs and order"
        )

    current_phase_id = str(payload.get("current_phase_id", "")).strip()
    try:
        current_phase_index = current_ids.index(current_phase_id)
    except ValueError as exc:
        raise ModelRoleError(
            "current_phase_id must reference one global_plan phase"
        ) from exc

    normalized_phases: list[dict[str, Any]] = []
    for previous_id, previous_phase, phase in zip(
        previous_ids,
        previous_phases,
        phases,
        strict=True,
    ):
        normalized_phase = dict(phase)
        normalized_phase["phase_id"] = previous_id
        for field_name in ("abstract_goal", "completion_evidence"):
            if isinstance(previous_phase, dict) and field_name in previous_phase:
                normalized_phase[field_name] = previous_phase[field_name]
        normalized_phases.append(normalized_phase)
    payload["global_plan"] = normalized_phases
    payload["current_phase_id"] = previous_ids[current_phase_index]

    normalizations = payload.setdefault("host_schema_normalizations", [])
    if not isinstance(normalizations, list):
        normalizations = []
        payload["host_schema_normalizations"] = normalizations
    normalizations.append(
        "global_plan backbone: restored persistent IDs and descriptions by "
        "unchanged semantic route-phase order"
    )


def _semantic_phase_signature(phase: dict[str, Any]) -> str:
    """Classify stable route roles while allowing harmless prose rewrites."""

    text = _normalized_phase_goal(
        f"{phase.get('phase_id', '')} {phase.get('abstract_goal', '')}"
    )
    tokens = set(text.split())
    if (
        {"place", "payload"} <= tokens
        or {"release", "payload"} <= tokens
        or "placement" in tokens
    ):
        return "payload_placement"
    if "second" in tokens and "corner" in tokens:
        return "second_corner"
    if "first" in tokens and "corner" in tokens:
        return "first_corner"
    if "initial" in tokens and (
        "straight" in tokens or "passage" in tokens or "corridor" in tokens
    ):
        return "initial_straight"
    if "final" in tokens and "straight" in tokens and "corner" not in tokens:
        return "final_straight"
    if (
        "cross" in tokens
        and ("passage" in tokens or "corridor" in tokens)
        and "corner" not in tokens
    ):
        return "cross_passage"
    goal = _normalized_phase_goal(phase.get("abstract_goal", ""))
    return f"goal:{goal}" if goal else ""


def _normalized_phase_goal(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _policy_api_methods(value: object) -> set[str]:
    methods: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"method", "primitive", "skill", "skill_name"}:
                    token = str(child).strip()
                    if re.fullmatch(r"[a-z_][a-z0-9_]*", token):
                        methods.add(token)
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            methods.update(
                re.findall(r"\bapi\.([a-z_][a-z0-9_]*)\s*\(", item)
            )

    visit(value)
    methods.discard("stop_all")
    return methods


def _validate_policy_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("robot_scripts"), (dict, list, tuple)):
        raise ModelRoleError("policy payload requires explicit robot_scripts")
    if not str(payload.get("code", "")).strip() and not payload.get("robot_scripts"):
        raise ModelRoleError("policy payload requires executable code")


def _validate_sync_team_policy_payload(payload: dict[str, Any]) -> None:
    _validate_policy_payload(payload)
    scripts = _robot_script_payloads(payload)
    if set(scripts) != {"front", "back"}:
        raise ModelRoleError("sync team policy requires front and back robot_scripts")
    modes = {
        str(item.get("coordination_mode", "")).strip()
        for item in scripts.values()
    }
    if modes != {"sync"}:
        raise ModelRoleError("sync team policy scripts must use coordination_mode=sync")
    codes = {str(item.get("code", "")).strip() for item in scripts.values()}
    if "" in codes or len(codes) != 1:
        raise ModelRoleError(
            "sync team policy requires identical non-empty front/back executable code"
        )


def _validate_async_robot_draft(
    payload: dict[str, Any],
    *,
    expected_robot: str,
    expected_method: str = "",
) -> None:
    required = {
        "robot_id",
        "script_id",
        "role",
        "objective",
        "phase_id",
        "contact_intent",
        "synchronization_group",
        "coordination_mode",
        "code",
    }
    _require_keys(payload, required, role=f"coder.{expected_robot}")
    if str(payload.get("robot_id", "")).strip() != expected_robot:
        raise ModelRoleError(
            f"async robot draft must describe exactly {expected_robot}"
        )
    if payload.get("coordination_mode") != "async":
        raise ModelRoleError("async robot draft requires coordination_mode=async")
    if not str(payload.get("code", "")).strip():
        raise ModelRoleError("async robot draft requires executable code")
    if "robot_scripts" in payload:
        raise ModelRoleError(
            "async robot draft must not emit a full team robot_scripts object"
        )
    if expected_method in MODEL_COORDINATION_PRIMITIVE_METHODS:
        methods = _policy_api_methods(payload.get("code", ""))
        if methods != {expected_method}:
            raise ModelRoleError(
                f"async {expected_robot} draft for coordination primitive "
                f"api.{expected_method} must call that exact primitive; "
                f"got {sorted(methods)}"
            )
    if expected_method in PAYLOAD_COOPERATIVE_METHODS:
        intent = str(payload.get("contact_intent", "")).strip()
        if intent != "maintain_payload_support":
            raise ModelRoleError(
                f"api.{expected_method} is a payload-cooperative skill and "
                f"conflicts with contact_intent={intent!r}; async Coder drafts "
                "must declare contact_intent='maintain_payload_support' in "
                "metadata, not as an API argument"
            )


def _canonicalize_async_draft_phase_id(
    payload: dict[str, Any],
    *,
    expected_robot: str,
    committed_phase_id: str,
) -> None:
    """Keep Coder metadata under the committed Task Orchestrator phase."""

    canonical = str(committed_phase_id).strip()
    if not canonical:
        return
    generated = str(payload.get("phase_id", "")).strip()
    if generated == canonical:
        return
    payload["phase_id"] = canonical
    normalizations = payload.get("host_schema_normalizations")
    if not isinstance(normalizations, list):
        normalizations = []
        payload["host_schema_normalizations"] = normalizations
    audit = (
        f"{expected_robot}.phase_id: {generated or '<empty>'} -> {canonical} "
        "(Task Orchestrator authority)"
    )
    if audit not in normalizations:
        normalizations.append(audit)


def _canonicalize_async_draft_synchronization_group(
    payload: dict[str, Any],
    *,
    expected_robot: str,
    committed_synchronization_group: str,
) -> None:
    """Keep async transaction metadata under the committed team subtask."""

    canonical = str(committed_synchronization_group).strip()
    if not canonical:
        return
    generated = str(payload.get("synchronization_group", "")).strip()
    if generated == canonical:
        return
    payload["synchronization_group"] = canonical
    normalizations = payload.get("host_schema_normalizations")
    if not isinstance(normalizations, list):
        normalizations = []
        payload["host_schema_normalizations"] = normalizations
    audit = (
        f"{expected_robot}.synchronization_group: "
        f"{generated or '<empty>'} -> {canonical} "
        "(Task Orchestrator subtask authority)"
    )
    if audit not in normalizations:
        normalizations.append(audit)


def _validate_async_team_review(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {"decision", "conflicts", "repaired_code", "rationale"},
        role="code_repair.async_team_review",
    )
    decision = str(payload.get("decision", "")).strip()
    if decision not in {"approve_drafts", "repair_drafts"}:
        raise ModelRoleError(
            "async team review decision must be approve_drafts or repair_drafts"
        )
    _require_string_list(
        payload.get("conflicts"),
        "code_repair.async_team_review.conflicts",
    )
    repaired_code = payload.get("repaired_code")
    if not isinstance(repaired_code, dict) or set(repaired_code) != {
        "front",
        "back",
    }:
        raise ModelRoleError(
            "async team review repaired_code must contain exactly front and back"
        )
    for robot_id in ("front", "back"):
        if not isinstance(repaired_code.get(robot_id), str):
            raise ModelRoleError(
                f"async team review repaired_code.{robot_id} must be a string"
            )
    if not str(payload.get("rationale", "")).strip():
        raise ModelRoleError("async team review rationale must be non-empty")
    if decision == "approve_drafts":
        if any(str(repaired_code[robot_id]).strip() for robot_id in ("front", "back")):
            raise ModelRoleError(
                "approve_drafts requires empty front/back repaired_code values"
            )
    elif not any(
        str(repaired_code[robot_id]).strip() for robot_id in ("front", "back")
    ):
        raise ModelRoleError(
            "repair_drafts requires at least one non-empty repaired_code value"
        )


def _build_async_team_policy_from_review(
    *,
    drafts: dict[str, dict[str, Any]],
    review: dict[str, Any],
    task_plan: dict[str, Any],
) -> dict[str, Any]:
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    preserve = subtask.get("preserve", [])
    safety_constraints = [
        str(item).strip()
        for item in preserve
        if str(item).strip()
    ]
    repaired_code = review.get("repaired_code", {})
    repaired_code = repaired_code if isinstance(repaired_code, dict) else {}
    decision = str(review.get("decision", "")).strip()
    script_id = (
        str(subtask.get("subtask_id", "")).strip()
        or "async_team_transaction"
    )

    assignments: dict[str, dict[str, Any]] = {}
    robot_scripts: dict[str, dict[str, Any]] = {}
    for robot_id in ("front", "back"):
        draft = drafts[robot_id]
        code = str(draft.get("code", "")).strip()
        if decision == "repair_drafts":
            repaired = str(repaired_code.get(robot_id, "")).strip()
            if repaired:
                code = repaired
        shared = {
            "synchronization_group": str(
                draft.get("synchronization_group", "")
            ).strip(),
            "phase_id": str(draft.get("phase_id", "")).strip(),
            "contact_intent": str(
                draft.get("contact_intent", "unspecified")
            ).strip(),
            "coordination_mode": "async",
        }
        assignments[robot_id] = {
            "robot_id": robot_id,
            "role": str(draft.get("role", "")).strip(),
            "objective": str(draft.get("objective", "")).strip(),
            "safety_constraints": safety_constraints,
            **shared,
        }
        robot_scripts[robot_id] = {
            "robot_id": robot_id,
            "script_id": str(
                draft.get("script_id", f"{robot_id}_{script_id}")
            ).strip(),
            "code": code,
            **shared,
        }
    return {
        "script_id": script_id,
        "code": robot_scripts["front"]["code"],
        "assignments": assignments,
        "robot_scripts": robot_scripts,
        "rationale": str(review.get("rationale", "")).strip(),
        "plan": [str(subtask.get("objective", "")).strip()],
        "spec_candidate": {},
    }


def _validate_async_team_policy_payload(payload: dict[str, Any]) -> None:
    _validate_policy_payload(payload)
    scripts = _robot_script_payloads(payload)
    if set(scripts) != {"front", "back"}:
        raise ModelRoleError("async team policy requires front and back robot_scripts")
    modes = {
        str(item.get("coordination_mode", "")).strip()
        for item in scripts.values()
    }
    if modes != {"async"}:
        raise ModelRoleError("async team policy scripts must use coordination_mode=async")
    groups = {
        str(item.get("synchronization_group", "")).strip()
        for item in scripts.values()
    }
    if "" in groups or len(groups) != 1:
        raise ModelRoleError(
            "async team scripts must share one non-empty atomic synchronization_group"
        )


def _validate_vlm_failure_analysis(payload: dict[str, Any]) -> None:
    required = {
        "summary",
        "scene_visible",
        "payload_on_floor",
        "robots_standing_stably",
        "wall_collision_or_stuck",
        "payload_placed_on_target",
        "failure_type",
        "confidence",
        "failure_analysis",
        "visible_evidence",
        "target_visible",
        "target_relation_to_team",
        "team_heading_relative_to_target",
        "team_heading_relative_to_visible_free_space",
        "team_payload_posture_relative_to_visible_passage",
        "visible_free_space_connectivity",
        "passage_geometry_observations",
        "team_footprint_clearance",
        "relative_robot_layout",
        "payload_axis_relative_to_passage",
        "immediate_route_direction_risk",
        "route_direction_risk_basis",
        "route_direction_evidence",
    }
    errors: list[str] = []
    missing = sorted(required - set(payload))
    if missing:
        errors.append(
            f"vlm_failure_analyzer response missing keys: {missing}"
        )
    for key in (
        "scene_visible",
        "payload_on_floor",
        "robots_standing_stably",
        "wall_collision_or_stuck",
        "payload_placed_on_target",
        "target_visible",
        "immediate_route_direction_risk",
    ):
        if key in payload and not isinstance(payload.get(key), bool):
            errors.append(
                f"{key} must be an exact JSON boolean true or false"
            )
    failure_type = payload.get("failure_type")
    allowed_failure_types = {
        "none",
        "payload_drop",
        "robot_fall",
        "payload_wall_contact",
        "manual_review_required",
    }
    if "failure_type" in payload and failure_type not in allowed_failure_types:
        errors.append(
            "failure_type is safety-only and must be one of "
            f"{sorted(allowed_failure_types)}; planning labels belong in "
            "failure_analysis.primary_failure"
        )
    route_enums = {
        "team_heading_relative_to_target": {
            "toward",
            "away",
            "crosswise",
            "mixed",
            "uncertain",
        },
        "team_heading_relative_to_visible_free_space": {
            "aligned_with_open_passage",
            "toward_wall_or_blocked_space",
            "crosswise_to_passage",
            "mixed",
            "uncertain",
        },
        "team_payload_posture_relative_to_visible_passage": {
            "longitudinal_aligned",
            "lateral_aligned",
            "diagonal_or_misaligned",
            "mixed",
            "uncertain",
        },
        "team_footprint_clearance": {
            "clear",
            "tight",
            "blocked",
            "uncertain",
        },
        "route_direction_risk_basis": {
            "none",
            "heading_only",
            "executed_motion_into_blocked_space",
            "unexplained_retreat",
            "uncertain",
        },
    }
    for key, allowed in route_enums.items():
        if key in payload and payload.get(key) not in allowed:
            errors.append(
                f"{key} must be exactly one of {sorted(allowed)}"
            )
    for key in (
        "summary",
        "target_relation_to_team",
        "relative_robot_layout",
        "payload_axis_relative_to_passage",
    ):
        if key in payload and (
            not isinstance(payload.get(key), str) or not payload[key].strip()
        ):
            errors.append(f"{key} must be a non-empty string")
    if (
        "route_direction_evidence" in payload
        and not isinstance(payload.get("route_direction_evidence"), str)
    ):
        errors.append("route_direction_evidence must be a string")
    for key in (
        "visible_evidence",
        "visible_free_space_connectivity",
        "passage_geometry_observations",
    ):
        if key not in payload:
            continue
        try:
            _require_string_list(payload.get(key), key)
        except ModelRoleError as exc:
            errors.append(str(exc))
    analysis = payload.get("failure_analysis")
    if not isinstance(analysis, dict):
        if "failure_analysis" in payload:
            errors.append("failure_analysis must be an object")
    else:
        analysis_required = {
            "outcome",
            "primary_failure",
            "causal_chain",
            "physical_evidence",
            "visual_evidence",
            "strategy_defect",
            "coordination_defect",
            "magnitude_defect",
            "reusable_context",
            "confidence",
        }
        analysis_missing = sorted(analysis_required - set(analysis))
        if analysis_missing:
            errors.append(
                f"failure_analysis response missing keys: {analysis_missing}"
            )
        if (
            "outcome" in analysis
            and analysis.get("outcome") not in {"accepted", "rejected"}
        ):
            errors.append(
                "failure_analysis.outcome must be exactly accepted or rejected"
            )
        if "primary_failure" in analysis and (
            not isinstance(analysis.get("primary_failure"), str)
            or not analysis["primary_failure"].strip()
        ):
            errors.append(
                "failure_analysis.primary_failure must be a non-empty string"
            )
        for key in (
            "causal_chain",
            "physical_evidence",
            "visual_evidence",
            "reusable_context",
        ):
            if key not in analysis:
                continue
            try:
                _require_string_list(
                    analysis.get(key),
                    f"failure_analysis.{key}",
                )
            except ModelRoleError as exc:
                errors.append(str(exc))
        for key in (
            "strategy_defect",
            "coordination_defect",
            "magnitude_defect",
        ):
            if key in analysis and not isinstance(analysis.get(key), bool):
                errors.append(
                    f"failure_analysis.{key} must be an exact JSON boolean"
                )
        if "confidence" in analysis:
            try:
                _require_confidence(
                    analysis.get("confidence"),
                    "failure_analysis.confidence",
                )
            except ModelRoleError as exc:
                errors.append(str(exc))
    if "confidence" in payload:
        try:
            _require_confidence(payload.get("confidence"), "confidence")
        except ModelRoleError as exc:
            errors.append(str(exc))
    if errors:
        raise ModelRoleError(
            "vlm_failure_analyzer structured contract violations: "
            + "; ".join(dict.fromkeys(errors))
        )


def _normalize_role_payload(
    role: ModelRole,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Canonicalize harmless JSON-shape variants at the audited role boundary."""

    normalized = dict(payload)
    changes: list[str] = []

    if role in {
        "evolution_manager",
        "mutation_operator",
        "memory_update",
    }:
        raw_level = str(normalized.get("mutation_level", "")).strip().lower()
        level_aliases = {
            "strategy": "strategy_mutation",
            "coordination": "coordination_mutation",
            "parameter": "parameter_tuning",
            "execution": "parameter_tuning",
        }
        canonical_level = level_aliases.get(raw_level, raw_level)
        if canonical_level and canonical_level != normalized.get("mutation_level"):
            normalized["mutation_level"] = canonical_level
            changes.append(
                f"mutation_level: {raw_level} -> {canonical_level}"
            )

    def wrap_string(
        mapping: dict[str, Any],
        key: str,
        *,
        path: str,
    ) -> None:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            mapping[key] = [value.strip()]
            changes.append(f"{path}: string -> singleton list")

    if role == "task_orchestrator":
        status_aliases = {
            "not_started": "pending",
            "queued": "pending",
            "future": "pending",
            "in_progress": "active",
            "current": "active",
            # A rejected rollout does not fail the persistent route phase. The
            # trusted checkpoint remains inside that phase until its completion
            # evidence is accepted.
            "failed": "active",
            "retry": "active",
            "done": "completed",
            "complete": "completed",
            "succeeded": "completed",
            "success": "completed",
        }

        def normalize_status(mapping: dict[str, Any], key: str, *, path: str) -> None:
            raw = str(mapping.get(key, "")).strip().lower()
            canonical = status_aliases.get(raw, raw)
            if canonical and canonical != mapping.get(key):
                mapping[key] = canonical
                changes.append(f"{path}: {raw} -> {canonical}")

        raw_phases = normalized.get("global_plan")
        if isinstance(raw_phases, list):
            phases: list[Any] = []
            for index, raw_phase in enumerate(raw_phases):
                if not isinstance(raw_phase, dict):
                    phases.append(raw_phase)
                    continue
                phase = dict(raw_phase)
                normalize_status(
                    phase,
                    "status",
                    path=f"global_plan[{index}].status",
                )
                phases.append(phase)
            normalized["global_plan"] = phases
        normalize_status(
            normalized,
            "current_phase_status",
            path="current_phase_status",
        )
        raw_subtask = normalized.get("current_subtask")
        if isinstance(raw_subtask, dict):
            subtask = dict(raw_subtask)
            raw_strategy = subtask.get("strategy")
            if isinstance(raw_strategy, dict):
                strategy = dict(raw_strategy)
                raw_mode = str(strategy.get("coordination_mode", "")).strip().lower()
                mode_aliases = {
                    "sync_only": "sync",
                    "synchronous": "sync",
                    "synchronized": "sync",
                    "async_only": "async",
                    "asynchronous": "async",
                }
                canonical_mode = mode_aliases.get(raw_mode, raw_mode)
                mode_change_recorded = False
                method = str(strategy.get("primitive", "")).strip()
                legal_modes = coordination_modes_for_method(method)
                if len(legal_modes) == 1:
                    required_mode = next(iter(legal_modes))
                    if canonical_mode != required_mode:
                        changes.append(
                            "current_subtask.strategy.coordination_mode: "
                            f"{canonical_mode or '<empty>'} -> {required_mode} "
                            f"(api.{method} contract)"
                        )
                        mode_change_recorded = True
                        canonical_mode = required_mode
                if canonical_mode and canonical_mode != strategy.get(
                    "coordination_mode"
                ):
                    strategy["coordination_mode"] = canonical_mode
                    if not mode_change_recorded:
                        changes.append(
                            "current_subtask.strategy.coordination_mode: "
                            f"{raw_mode} -> {canonical_mode}"
                        )
                subtask["strategy"] = strategy
                if (
                    not isinstance(subtask.get("team_assignment"), dict)
                    and isinstance(strategy.get("team_assignment"), dict)
                ):
                    subtask["team_assignment"] = _json_safe_mapping(
                        strategy["team_assignment"]
                    )
                    changes.append(
                        "current_subtask.team_assignment: copied from strategy"
                    )
            if not isinstance(subtask.get("navigation"), dict):
                probe_plan = {**normalized, "current_subtask": subtask}
                subtask["navigation"] = _default_subtask_navigation(probe_plan)
                changes.append(
                    "current_subtask.navigation: added generic semantic reference"
                )
            wrap_string(
                subtask,
                "preserve",
                path="current_subtask.preserve",
            )
            normalized["current_subtask"] = subtask
        wrap_string(normalized, "constraints", path="constraints")
        return normalized, changes

    if role == "evolution_manager":
        for key in ("evidence", "preserve", "mutate", "forbidden"):
            wrap_string(normalized, key, path=key)
        for key in ("mutate", "inheritance"):
            value = normalized.get(key)
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                continue
            semantic_items: list[Any] = []
            for index, item in enumerate(value):
                semantic_item = _parameter_free_manager_item(item)
                if semantic_item != item:
                    changes.append(
                        f"{key}[{index}]: executable call -> symbolic strategy"
                    )
                if semantic_item:
                    semantic_items.append(semantic_item)
            normalized[key] = semantic_items
        if normalized.get("mutation_level") == "strategy_mutation":
            raw_mutate = normalized.get("mutate", ())
            abstract_mutate = _abstract_strategy_mutation_dimensions(raw_mutate)
            if abstract_mutate != raw_mutate:
                normalized["mutate"] = abstract_mutate
                changes.append(
                    "mutate: concrete strategy targets -> abstract mutation dimensions"
                )
            raw_forbidden = normalized.get("forbidden", ())
            safe_forbidden = _strategy_mutation_safe_forbidden(raw_forbidden)
            if safe_forbidden != raw_forbidden:
                normalized["forbidden"] = safe_forbidden
                changes.append(
                    "forbidden: removed constraints that froze mutable strategy fields"
                )
            raw_inheritance = normalized.get("inheritance", ())
            safe_inheritance = _strategy_mutation_safe_inheritance(
                raw_inheritance
            )
            if safe_inheritance != raw_inheritance:
                normalized["inheritance"] = safe_inheritance
                changes.append(
                    "inheritance: concrete parent contract -> safe mission invariants"
                )
        return normalized, changes

    if role == "mutation_operator":
        wrap_string(
            normalized,
            "changed_dimensions",
            path="changed_dimensions",
        )
        # Strategy prose and evidence may legitimately discuss height,
        # distance, or angle.  Parameter ownership is enforced later over
        # typed executable fields/effective backend arguments, not free text.
        return normalized, changes

    if role == "memory_update":
        for key in ("applicability", "avoid", "evidence_refs"):
            wrap_string(normalized, key, path=key)
        semantic_memory = {
            key: normalized.get(key)
            for key in (
                "strategy_name",
                "memory_kind",
                "applicability",
                "parent_strategy",
                "child_strategy",
                "learned_rule",
                "avoid",
            )
        }
        memory_text = json.dumps(semantic_memory, ensure_ascii=False)
        has_transferability_leak = bool(
            _find_parameter_keys(normalized)
            or _contains_numeric_value(semantic_memory)
            or re.search(
                r"\b(?:checkpoint|waypoint|world[_ -]?[xy]|target[_ -]?(?:x|y)|"
                r"distance_m|speed_mps|angle_deg|duration_s)\b",
                memory_text,
                flags=re.IGNORECASE,
            )
        )
        if normalized.get("promote") is True and has_transferability_leak:
            normalized["promote"] = False
            changes.append(
                "promote: true -> false (host memory-transferability gate)"
            )
        return normalized, changes

    if role != "vlm_failure_analyzer":
        return normalized, changes

    for key in (
        "visible_evidence",
        "visible_free_space_connectivity",
        "passage_geometry_observations",
    ):
        wrap_string(normalized, key, path=key)
    route_evidence = normalized.get("route_direction_evidence")
    if isinstance(route_evidence, list):
        normalized["route_direction_evidence"] = "; ".join(
            str(item).strip() for item in route_evidence if str(item).strip()
        )
        changes.append("route_direction_evidence: list -> joined string")

    raw_analysis = normalized.get("failure_analysis")
    if not isinstance(raw_analysis, dict):
        return normalized, changes
    analysis = dict(raw_analysis)
    for key in (
        "causal_chain",
        "physical_evidence",
        "visual_evidence",
        "reusable_context",
    ):
        wrap_string(analysis, key, path=f"failure_analysis.{key}")
    normalized["failure_analysis"] = analysis
    return normalized, changes


def _abstract_strategy_mutation_dimensions(value: object) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else ()
    dimensions: set[str] = set()
    for item in items:
        text = str(item).strip().lower()
        if not text:
            continue
        semantic_text = text.replace("_", " ").replace("-", " ")
        if (
            "primitive" in text
            or "maneuver" in text
            or "api." in text
            or any(
                token in text
                for token in (
                    "carry_",
                    "turn_",
                    "coordinated_",
                    "place_",
                    "recover_",
                    "release_",
                    "exit_",
                    "hold_position",
                )
            )
        ):
            dimensions.add("maneuver family")
        if (
            "semantic" in text
            or "objective" in text
            or any(
                re.search(rf"\b{token}\b", semantic_text)
                for token in (
                    "forward",
                    "backward",
                    "left",
                    "right",
                    "clockwise",
                    "counterclockwise",
                    "longitudinal",
                    "lateral",
                    "yaw",
                )
            )
        ):
            dimensions.add("semantic objective")
        if any(
            token in text
            for token in (
                "formation",
                "coordination",
                "sync",
                "async",
                "role",
                "reference",
                "corrector",
                "inner",
                "outer",
            )
        ):
            dimensions.add("formation strategy")
    if not dimensions:
        dimensions.add("maneuver family")
    return [
        dimension
        for dimension in (
            "maneuver family",
            "semantic objective",
            "formation strategy",
        )
        if dimension in dimensions
    ]


def _strategy_mutation_safe_forbidden(value: object) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else ()
    frozen_patterns = (
        "change primitive",
        "change semantic objective",
        "change coordination mode",
        "change coordination_mode",
        "change temporary role",
        "change role mapping",
        "change axis",
        "change direction",
    )
    safe = [
        str(item).strip()
        for item in items
        if str(item).strip()
        and not any(
            pattern in str(item).strip().lower()
            for pattern in frozen_patterns
        )
    ]
    if not safe:
        safe = [
            "repeat the exact checkpoint-local physical counterexample",
            "violate payload support or physical safety",
        ]
    return safe


def _strategy_mutation_safe_inheritance(value: object) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else ()
    safe_labels = (
        "global mission",
        "current route phase",
        "accepted checkpoint progress",
        "payload support",
        "physical safety",
    )
    safe = [
        label
        for label in safe_labels
        if any(label in str(item).strip().lower() for item in items)
    ]
    return safe or list(safe_labels)


def _validate_evolution_manager(payload: dict[str, Any]) -> None:
    required = {
        "mutation_level",
        "failure_hypothesis",
        "evidence",
        "preserve",
        "mutate",
        "forbidden",
        "success_test",
        "inheritance",
        "confidence",
    }
    _require_keys(payload, required, role="evolution_manager")
    if payload.get("mutation_level") not in MUTATION_LEVELS:
        raise ModelRoleError(
            f"mutation_level must be one of {sorted(MUTATION_LEVELS)}"
        )
    for key in ("evidence", "preserve", "mutate", "forbidden"):
        _require_string_list(payload.get(key), key, allow_empty=False)
    _require_confidence(payload.get("confidence"), "confidence")
    inheritance_leaks = _find_parameter_keys(payload.get("inheritance"))
    if inheritance_leaks:
        raise ModelRoleError(
            "evolution manager may not output numerical action parameter fields: "
            + ", ".join(sorted(inheritance_leaks))
        )
    mutation_leaks = _find_parameter_keys(payload.get("mutate"))
    if payload.get("mutation_level") == "parameter_tuning":
        if len(mutation_leaks) > 1:
            raise ModelRoleError(
                "evolution manager parameter_tuning may name exactly one action "
                "parameter field; got "
                + ", ".join(sorted(mutation_leaks))
            )
        if mutation_leaks and _contains_numeric_literal(payload.get("mutate")):
            raise ModelRoleError(
                "evolution manager parameter_tuning may name one action parameter "
                "field but may not choose its numeric value"
            )
    elif mutation_leaks:
        raise ModelRoleError(
            "evolution manager may not output numerical action parameter fields: "
            + ", ".join(sorted(mutation_leaks))
        )


def _validate_mutation_operator(payload: dict[str, Any]) -> None:
    required = {
        "mutation_level",
        "reason",
        "parent_strategy",
        "child_strategy",
        "changed_dimensions",
        "inheritance",
    }
    _require_keys(payload, required, role="mutation_operator")
    if payload.get("mutation_level") not in MUTATION_LEVELS:
        raise ModelRoleError(
            f"mutation_level must be one of {sorted(MUTATION_LEVELS)}"
        )
    strategy = payload.get("child_strategy")
    if not isinstance(strategy, dict):
        raise ModelRoleError("mutation_operator.child_strategy must be an object")
    _require_keys(
        strategy,
        {
            "primitive",
            "semantic_objective",
            "formation_strategy",
            "coordination_mode",
            "synchronization_intent",
            "team_assignment",
        },
        role="mutation_operator.child_strategy",
    )
    if strategy.get("coordination_mode") not in {"sync", "async"}:
        raise ModelRoleError(
            "child_strategy.coordination_mode must be sync or async"
        )
    _validate_team_assignment(
        strategy.get("team_assignment"),
        role="mutation_operator.child_strategy.team_assignment",
    )
    _require_string_list(
        payload.get("changed_dimensions"),
        "changed_dimensions",
        allow_empty=False,
    )
    level = str(payload.get("mutation_level", "")).strip()
    changed_parameter_keys = _find_parameter_keys(
        payload.get("changed_dimensions")
    )
    structured_parameter_keys = _find_structured_parameter_keys(
        {
            "parent_strategy": payload.get("parent_strategy"),
            "child_strategy": payload.get("child_strategy"),
        }
    )
    if level == "parameter_tuning":
        if len(changed_parameter_keys) != 1:
            raise ModelRoleError(
                "parameter_tuning mutation operator must name exactly one action "
                "parameter field in changed_dimensions"
            )
        if (
            _contains_numeric_literal(payload.get("changed_dimensions"))
            or structured_parameter_keys
        ):
            raise ModelRoleError(
                "parameter_tuning mutation operator may name one field only in "
                "changed_dimensions, but may not choose values or carry action "
                "parameters into strategy or inheritance"
            )
    elif changed_parameter_keys or structured_parameter_keys:
        raise ModelRoleError(
            "mutation operator may not output numerical action parameter fields"
        )


def _validate_team_assignment(value: object, *, role: str) -> None:
    if not isinstance(value, dict) or set(value) != {"front", "back"}:
        raise ModelRoleError(f"{role} must contain exactly front and back")
    for robot_id, assignment in value.items():
        if not isinstance(assignment, dict):
            raise ModelRoleError(f"{role}.{robot_id} must be an object")
        _require_keys(
            assignment,
            {"role", "objective"},
            role=f"{role}.{robot_id}",
        )
        if not str(assignment.get("role", "")).strip():
            raise ModelRoleError(f"{role}.{robot_id}.role must be non-empty")
        if not str(assignment.get("objective", "")).strip():
            raise ModelRoleError(f"{role}.{robot_id}.objective must be non-empty")


def _robot_script_payloads(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = payload.get("robot_scripts")
    scripts: dict[str, dict[str, Any]] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(item, dict):
                continue
            robot_id = str(item.get("robot_id") or key).strip()
            scripts[robot_id] = item
    elif isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, dict):
                continue
            robot_id = str(item.get("robot_id", "")).strip()
            if robot_id:
                scripts[robot_id] = item
    return scripts


def _validate_memory_update(payload: dict[str, Any]) -> None:
    required = {
        "promote",
        "strategy_name",
        "memory_kind",
        "mutation_level",
        "applicability",
        "parent_strategy",
        "child_strategy",
        "learned_rule",
        "avoid",
        "evidence_refs",
        "confidence",
    }
    _require_keys(payload, required, role="memory_update")
    if not isinstance(payload.get("promote"), bool):
        raise ModelRoleError("memory_update.promote must be boolean")
    if payload.get("mutation_level") not in {*MUTATION_LEVELS, "none"}:
        raise ModelRoleError(
            "memory_update.mutation_level must be a mutation level or none"
        )
    for key in ("applicability", "avoid", "evidence_refs"):
        _require_string_list(payload.get(key), key)
    _require_confidence(payload.get("confidence"), "confidence")
    if not payload.get("promote"):
        return
    leaked = _find_parameter_keys(payload)
    if leaked:
        raise ModelRoleError(
            "memory update leaks action parameters: " + ", ".join(sorted(leaked))
        )
    semantic_memory = {
        key: payload.get(key)
        for key in (
            "strategy_name",
            "memory_kind",
            "applicability",
            "parent_strategy",
            "child_strategy",
            "learned_rule",
            "avoid",
        )
    }
    if _contains_numeric_value(semantic_memory):
        raise ModelRoleError(
            "memory update contains a numerical action or scene value"
        )
    text = json.dumps(
        semantic_memory,
        ensure_ascii=False,
    )
    if re.search(
        r"\b(?:checkpoint|waypoint|world[_ -]?[xy]|target[_ -]?(?:x|y)|"
        r"distance_m|speed_mps|angle_deg|duration_s)\b",
        text,
        flags=re.IGNORECASE,
    ):
        raise ModelRoleError(
            "memory update contains a checkpoint, route, coordinate, or action magnitude"
        )


def _require_keys(payload: dict[str, Any], keys: set[str], *, role: str) -> None:
    missing = sorted(keys - set(payload))
    if missing:
        raise ModelRoleError(f"{role} response missing keys: {missing}")


def _combine_validators(
    first: RoleValidator,
    second: RoleValidator | None,
) -> RoleValidator:
    def validate(payload: dict[str, Any]) -> None:
        first(payload)
        if second is not None:
            second(payload)

    return validate


def _candidate_feasibility_validator(
    validator: RoleValidator | None,
) -> RoleValidator | None:
    """Tag post-schema mutation rejection as executable-candidate evidence."""

    if validator is None:
        return None

    def validate(payload: dict[str, Any]) -> None:
        try:
            validator(payload)
        except ModelRoleCandidateFeasibilityError:
            raise
        except (ModelRoleError, ValueError) as exc:
            raise ModelRoleCandidateFeasibilityError(str(exc)) from exc

    return validate


def _require_string_list(
    value: object,
    name: str,
    *,
    allow_empty: bool = True,
) -> None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ModelRoleError(f"{name} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ModelRoleError(f"{name} must not be empty")


def _require_confidence(value: object, name: str) -> None:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelRoleError(f"{name} must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ModelRoleError(f"{name} must be in [0, 1]")


def _find_parameter_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in ACTION_PARAMETER_KEYS:
                found.add(str(key))
            found.update(_find_parameter_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_parameter_keys(child))
    elif isinstance(value, str):
        for key in ACTION_PARAMETER_KEYS:
            # Parameter names embedded in a symbolic enum (for example,
            # ``corrector_delta_based``) are not executable parameter fields.
            # Match only a complete identifier token so semantic strategy labels
            # remain available while ``delta`` or ``distance_m=...`` still leak.
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])",
                value,
            ):
                found.add(key)
    return found


def _find_structured_parameter_keys(value: object) -> set[str]:
    """Find executable parameter *keys* without interpreting prose evidence."""

    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in ACTION_PARAMETER_KEYS:
                found.add(str(key))
            found.update(_find_structured_parameter_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_structured_parameter_keys(child))
    return found


def _symbolic_role_objective(
    *,
    primitive: str,
    semantic_objective: str,
    role: str,
) -> str:
    """Return parameter-free prose for a high-level temporary robot role."""

    role_objectives = {
        "reference": "serve as the temporary formation reference",
        "corrector": "apply the selected relative formation correction",
        "inner": "serve as the stationary inner support",
        "outer": "follow the selected outer support path",
    }
    if role in role_objectives:
        return role_objectives[role]
    action = " ".join(
        value for value in (semantic_objective, primitive) if value
    )
    return f"execute the shared {action} role".strip()


def _contains_numeric_literal(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_numeric_literal(key) or _contains_numeric_literal(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_numeric_literal(child) for child in value)
    if isinstance(value, str):
        return bool(
            re.search(
                r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z_])",
                value,
            )
        )
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parameter_free_manager_item(value: object) -> object:
    """Reduce one complete API call to the model-selected symbolic strategy."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return value
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Attribute)
        or not isinstance(expression.func.value, ast.Name)
        or expression.func.value.id != "api"
    ):
        return value
    method = expression.func.attr
    if method == "stop_all":
        return ""
    semantic_value = ""
    for keyword in expression.keywords:
        if keyword.arg not in {"direction", "turn_direction", "axis"}:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(
            keyword.value.value, str
        ):
            semantic_value = keyword.value.value.strip()
            break
    return f"{method}:{semantic_value}" if semantic_value else method


def _contains_numeric_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_contains_numeric_value(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_numeric_value(child) for child in value)
    return False


def _machine_readable_validation_feedback(error_text: str) -> dict[str, Any]:
    """Extract a validator JSON payload without weakening the human error text."""

    marker = "NO_EFFECTIVE_MUTATION "
    text = str(error_text)
    if marker not in text:
        return {}
    payload_text = text.partition(marker)[2].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return {"code": "NO_EFFECTIVE_MUTATION"}
    return payload if isinstance(payload, dict) else {"code": "NO_EFFECTIVE_MUTATION"}


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    fenced = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = [*fenced, text]
    decoder = json.JSONDecoder()
    saw_object_start = False
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
        else:
            if isinstance(payload, dict):
                return payload
        for start, character in enumerate(candidate):
            if character != "{":
                continue
            saw_object_start = True
            try:
                payload, _end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(payload, dict):
                return payload
    if not saw_object_start:
        raise ModelRoleError("response does not contain a JSON object")
    if last_error is not None:
        raise ModelRoleError(f"invalid JSON response: {last_error}") from last_error
    raise ModelRoleError("response root must be a JSON object")


def _normalization_audit(
    raw_payload: dict[str, Any] | None,
    normalized_payload: dict[str, Any] | None,
    *,
    reasons: list[str],
) -> dict[str, Any]:
    """Describe every Host-side structured-output change without aliasing raw data."""

    if raw_payload is None or normalized_payload is None:
        return {
            "raw_response_preserved": raw_payload is not None,
            "changed_fields": [],
            "changes": [],
            "reasons": list(reasons),
        }

    missing = object()
    changes: list[dict[str, Any]] = []

    def visit(before: object, after: object, path: str) -> None:
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(set(before) | set(after)):
                visit(
                    before.get(key, missing),
                    after.get(key, missing),
                    f"{path}.{key}" if path else str(key),
                )
            return
        if before == after:
            return
        changes.append(
            {
                "field": path,
                "before": (
                    {"missing": True}
                    if before is missing
                    else copy.deepcopy(before)
                ),
                "after": (
                    {"missing": True}
                    if after is missing
                    else copy.deepcopy(after)
                ),
            }
        )

    visit(raw_payload, normalized_payload, "")
    return {
        "raw_response_preserved": True,
        "changed_fields": [item["field"] for item in changes],
        "changes": changes,
        "reasons": list(reasons),
    }


def _json_safe_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace one JSON document atomically so live readers never see a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one complete status event to the experiment timeline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _attempt_from_artifact_id(artifact_id: str) -> int:
    matches = re.findall(r"(?:attempt[_-]?)?(\d+)", str(artifact_id))
    return int(matches[-1]) if matches else 0


def _checkpoint_id_from_context(context: dict[str, Any]) -> str:
    direct = str(context.get("checkpoint_id", "")).strip()
    if direct:
        return direct
    for key in (
        "current_trusted_observation",
        "observation",
        "runtime_state",
        "host_controller_guard",
    ):
        value = context.get(key)
        checkpoint_id = _checkpoint_id_from_value(value)
        if checkpoint_id:
            return checkpoint_id
    return ""


def _checkpoint_id_from_value(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    for key in ("checkpoint_id", "active_checkpoint_id", "latest_checkpoint"):
        checkpoint_id = str(value.get(key, "")).strip()
        if checkpoint_id:
            return checkpoint_id
    checkpoint = value.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint_id = str(checkpoint.get("checkpoint_id", "")).strip()
        if checkpoint_id:
            return checkpoint_id
    for key in ("runtime_state", "state", "current_state"):
        checkpoint_id = _checkpoint_id_from_value(value.get(key))
        if checkpoint_id:
            return checkpoint_id
    return ""


def _phase_status_payload(task_plan: dict[str, Any]) -> dict[str, Any]:
    phases = task_plan.get("global_plan", [])
    phases = phases if isinstance(phases, list) else []
    current_id = str(task_plan.get("current_phase_id", "")).strip()
    current_index = 0
    completed_count = 0
    abstract_goal = ""
    completion_evidence = ""
    plan_status = str(task_plan.get("current_phase_status", "")).strip()
    phase_summary: list[dict[str, str]] = []
    for index, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            continue
        phase_id = str(phase.get("phase_id", "")).strip()
        status = str(phase.get("status", "")).strip()
        phase_summary.append(
            {
                "phase_id": phase_id,
                "status": status,
                "abstract_goal": str(phase.get("abstract_goal", "")).strip(),
            }
        )
        completed_count += int(status == "completed")
        if phase_id == current_id:
            current_index = index
            abstract_goal = str(phase.get("abstract_goal", "")).strip()
            completion_evidence = str(
                phase.get("completion_evidence", "")
            ).strip()
            plan_status = plan_status or status
    return {
        "phase_id": current_id,
        "status": plan_status,
        "index": current_index,
        "total": len(phase_summary),
        "completed_count": completed_count,
        "abstract_goal": abstract_goal,
        "completion_evidence": completion_evidence,
        "global_plan": phase_summary,
    }


def _subtask_status_payload(
    task_plan: dict[str, Any],
    *,
    runtime_status: str,
) -> dict[str, Any]:
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    status_by_runtime = {
        "subtask_planned": "planned",
        "coding": "planned",
        "policy_ready": "ready",
        "preparing_execution": "ready",
        "preflight": "preflight",
        "executing": "executing",
        "evaluating": "evaluating",
        "checkpoint_validation": "evaluating",
        "rolling_back": "rejected",
        "rollback_complete": "retry_pending",
        "accepted": "accepted",
        "rejected": "rejected",
        "task_complete": "completed",
    }
    preserve = subtask.get("preserve", [])
    return {
        "subtask_id": str(subtask.get("subtask_id", "")).strip(),
        "runtime_status": status_by_runtime.get(runtime_status, runtime_status),
        "objective": str(subtask.get("objective", "")).strip(),
        "guidance": str(subtask.get("guidance", "")).strip(),
        "completion_condition": str(
            subtask.get("completion_condition", "")
        ).strip(),
        "preserve": (
            [str(item) for item in preserve]
            if isinstance(preserve, list)
            else []
        ),
    }


def _next_action_payload(task_plan: dict[str, Any]) -> dict[str, str]:
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    strategy = subtask.get("strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    return {
        "primitive": str(strategy.get("primitive", "")).strip(),
        "semantic_objective": str(
            strategy.get("semantic_objective", "")
        ).strip(),
        "formation_strategy": str(
            strategy.get("formation_strategy", "")
        ).strip(),
        "coordination_mode": str(
            strategy.get("coordination_mode", "")
        ).strip(),
        "synchronization_intent": str(
            strategy.get("synchronization_intent", "")
        ).strip(),
    }


def _team_assignment_payload(
    task_plan: dict[str, Any],
    *,
    policy: dict[str, Any] | None,
) -> dict[str, dict[str, str]]:
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    assignment = subtask.get("team_assignment", {})
    if not isinstance(assignment, dict):
        assignment = {}
    if not assignment and isinstance(policy, dict):
        raw_assignments = policy.get("assignments", {})
        if isinstance(raw_assignments, dict):
            assignment = raw_assignments
        elif isinstance(raw_assignments, (list, tuple)):
            assignment = {
                str(item.get("robot_id", "")).strip(): item
                for item in raw_assignments
                if isinstance(item, dict)
                and str(item.get("robot_id", "")).strip()
            }
    result: dict[str, dict[str, str]] = {}
    for robot_id in ("front", "back"):
        item = assignment.get(robot_id, {})
        item = item if isinstance(item, dict) else {}
        result[robot_id] = {
            "role": str(item.get("role", "")).strip(),
            "objective": str(item.get("objective", "")).strip(),
        }
    return result


def _policy_status_payload(policy: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(policy, dict) or not policy:
        return {}
    scripts = _robot_script_payloads(policy)
    return {
        "script_id": str(policy.get("script_id", "")).strip(),
        "parent_script_id": str(policy.get("parent_script_id", "")).strip(),
        "attempt": policy.get("attempt"),
        "rationale": str(policy.get("rationale", "")).strip(),
        "robot_script_ids": {
            robot_id: str(item.get("script_id", "")).strip()
            for robot_id, item in scripts.items()
        },
    }


def _task_plan_with_status_strategy(
    task_plan: dict[str, Any],
    strategy: object,
) -> dict[str, Any]:
    if not task_plan or not isinstance(strategy, dict):
        return task_plan
    updated = _json_safe_mapping(task_plan)
    subtask = updated.get("current_subtask", {})
    if not isinstance(subtask, dict):
        return task_plan
    subtask["strategy"] = _json_safe_mapping(strategy)
    assignment = strategy.get("team_assignment")
    if isinstance(assignment, dict):
        subtask["team_assignment"] = _json_safe_mapping(assignment)
    updated["current_subtask"] = subtask
    try:
        _validate_task_plan(updated)
    except ModelRoleError:
        return task_plan
    return updated


def _semantic_status_summary(
    *,
    runtime_status: str,
    phase: dict[str, Any],
    subtask: dict[str, Any],
    action: dict[str, str],
    team_assignment: dict[str, dict[str, str]],
) -> str:
    fields = [
        f"state={runtime_status}",
        f"phase={phase.get('phase_id') or 'unplanned'}",
        f"subtask={subtask.get('subtask_id') or 'unassigned'}",
    ]
    primitive = action.get("primitive", "")
    objective = action.get("semantic_objective", "")
    if primitive:
        fields.append(
            f"action={primitive}" + (f":{objective}" if objective else "")
        )
    mode = action.get("coordination_mode", "")
    if mode:
        fields.append(f"mode={mode}")
    for robot_id in ("front", "back"):
        role = team_assignment.get(robot_id, {}).get("role", "")
        if role:
            fields.append(f"{robot_id}={role}")
    return "; ".join(fields)


def _default_next_system_operation(runtime_status: str) -> str:
    return {
        "initializing": "capture the trusted global observation",
        "model_task_orchestrator_running": "produce the global phase and current subtask",
        "subtask_planned": "generate executable front/back policy",
        "coding": "validate generated scripts against the committed subtask",
        "policy_ready": "run preflight and dispatch the team transaction",
        "preparing_execution": "capture the pre-action global observation",
        "preflight": "validate the policy against host safety contracts",
        "executing": "wait for both robot scripts to finish",
        "evaluating": "combine physics and global-view evidence",
        "checkpoint_validation": "round-trip the candidate trusted checkpoint",
        "rolling_back": "restore the latest trusted checkpoint",
        "rollback_complete": "analyze the failure and generate a repaired policy",
        "accepted": "update the global phase and plan the next subtask",
        "rejected": "select mutation level and repair the current subtask",
        "task_complete": "stop the evolution loop",
    }.get(runtime_status, "continue the formal evolution pipeline")


def _role_next_system_operation(role: ModelRole) -> str:
    return {
        "task_orchestrator": "update the global phase and current team subtask",
        "coder": "generate executable robot Python",
        "vlm_failure_analyzer": "judge the rollout and identify the causal failure",
        "evolution_manager": "choose strategy, coordination, or parameter mutation",
        "mutation_operator": "mutate the current subtask strategy",
        "code_repair": "repair and integrate the executable team transaction",
        "memory_update": "propose a reusable strategy-memory update",
    }[role]


def _status_console_line(status: dict[str, Any]) -> str:
    summary = str(status.get("semantic_summary", "")).replace("\n", " ").strip()
    checkpoint = str(status.get("checkpoint_id", "")).strip() or "uncommitted"
    role = str(status.get("active_model_role", "")).strip()
    role_suffix = f" role={role}" if role else ""
    return (
        f"[cap-status] event={status.get('event_sequence', 0)} "
        f"attempt={status.get('attempt', 0)} checkpoint={checkpoint}"
        f"{role_suffix} {summary}"
    )


def _safe_artifact_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "call"


def _formal_v50_lite_context(context: dict[str, Any]) -> bool:
    """Return whether one role invocation uses the frozen V50-lite profile."""

    return bool(
        context.get("formal_v50_lite", False)
        or context.get("formal_model_owns_route", False)
    )


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _client_record_count(client: object) -> int:
    records = getattr(client, "call_records", ())
    return len(records) if isinstance(records, list) else 0


def _latest_client_usage(client: object, *, before_count: int) -> dict[str, Any]:
    records = getattr(client, "call_records", ())
    if not isinstance(records, list) or len(records) <= before_count:
        return {}
    latest = records[-1]
    return dict(latest) if isinstance(latest, dict) else {}


def _annotate_client_records(
    client: object,
    *,
    before_count: int,
    role: str,
    artifact_id: str,
    response_index: int,
) -> None:
    """Attach the generating pipeline stage to all transport/recovery rounds."""

    records = getattr(client, "call_records", ())
    if not isinstance(records, list):
        return
    for record in records[max(0, int(before_count)) :]:
        if not isinstance(record, dict):
            continue
        record.update(
            {
                "role": str(role),
                "artifact_id": str(artifact_id),
                "response_index": int(response_index),
            }
        )


__all__ = [
    "BoundModelRoleClient",
    "MUTATION_LEVELS",
    "ModelRole",
    "ModelRoleCandidateFeasibilityError",
    "ModelRoleError",
    "ModelRoleStallError",
    "ModelRolePipeline",
    "MutationLevel",
    "ROLE_SYSTEM_PROMPTS",
]
