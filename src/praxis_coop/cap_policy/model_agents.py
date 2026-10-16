"""Model-backed Code-as-Policy orchestration and evolution agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import ast
import hashlib
import json
from math import atan2, cos, degrees, hypot, isfinite, radians, sin
import re
from pathlib import Path
import time
from typing import Any, Callable, Protocol

from praxis_coop.llm import QwenTextClient
from praxis_coop.sonic.runtime_state import (
    payload_recovery_outward_evidence_m,
    payload_recovery_trigger,
    payload_support_center_error_m,
)
from praxis_coop.sonic.safety import (
    PAYLOAD_TOP_FACE_UP_LIMIT_DEG,
    TASK_CARRY_SAFETY,
)
from praxis_coop.sonic.upper_body import (
    DEFAULT_CARRY_READY_POSE_NAME,
    PAYLOAD_RECOVERY_POSE_NAME,
)

from .contracts import (
    BoxCarryTask,
    DEFAULT_MAX_TERMINAL_PAYLOAD_TILT_DEG,
    DEFAULT_ROBOT_EXIT_MIN_DISTANCE_M,
    DEFAULT_ROBOT_PRE_STAND_CLEARANCE_MARGIN_M,
    DEFAULT_TARGET_XY_TOLERANCE_M,
    EvaluationReport,
    ExecutionTrace,
    ObservationBundle,
    PolicyScript,
    RobotAssignment,
    RobotPolicyScript,
    write_json,
)
from .action_memory import (
    VerifiedActionMemoryStore,
    selected_subtask_objective,
)
from .failure_taxonomy import classify_failure, classify_report
from .release_preflight_evidence import release_preflight_evidence
from .generation_recovery import (
    HostContractError,
    MUTATION_SCOPE_INFEASIBLE,
)
from .model_role_pipeline import (
    ModelRoleError,
    ModelRolePipeline,
    ModelRoleStallError,
)
from .mutation_contract import (
    MUTATION_LEVEL_BY_RANK,
    MUTATION_LEVEL_RANK,
    MUTATION_SCOPE_INFEASIBLE as MUTATION_SCOPE_STATUS_INFEASIBLE,
    apply_minimum_mutation_rank,
    apply_infeasible_mutation_level,
    build_mutation_contract,
    check_selected_mutation_scope,
    host_effective_mutation_delta,
    no_effective_mutation_feedback,
    validate_manager_level,
    validate_mutation_transition,
)
from .local_navigation import (
    HEADING_ALIGNMENT_TOLERANCE_DEG,
    LOCAL_NAVIGATION_CONTRACT_KEY,
    MIN_HEADING_IMPROVEMENT_DEG,
    build_local_navigation_contract,
    refresh_local_navigation_contract,
    team_heading_deg,
    target_local_navigation_evidence,
    validate_strategy_against_local_navigation,
    wrap_degrees,
)
from .evolution_controller import (
    EvolutionController,
    EvolutionDecision,
    policy_motion_objective_signature,
    policy_role_signature,
    policy_strategy_signature,
    trace_has_physical_dispatch,
)
from .policy_executor import (
    CONTACT_INTENTS,
    validate_policy_code,
    validate_step_policy_code,
    validate_team_policy_scripts,
)
from .scene_clearance import (
    authored_scene_motion_clearance,
    candidate_is_blocked,
    maximum_safe_turn_angle_deg,
    open_translation_directions,
    pivot_role_assignment,
    unique_small_heading_correction,
)
from .route_spec import (
    route_guidance,
    route_spec_document,
    validate_locked_route_reference_unchanged,
    validate_task_plan_route_reference,
)
from .transition_verifier import (
    transition_is_route_progress,
    transition_outcome_from_report,
)
from .simulation_diagnostics import trusted_route_phase_guard
from .coordination_specs import (
    CALIBRATED_MODEL_PIVOT_DURATION_S,
    COORDINATED_ALIGNMENT,
    COORDINATED_ARC_TRANSPORT,
    COORDINATED_LINEAR_DISTANCE_CHOICES_M,
    COORDINATED_LINEAR_TRANSPORT,
    COORDINATED_PIVOT,
    COORDINATION_PRIMITIVE_METHODS,
    FINAL_APPROACH_LINEAR_DISTANCE_CHOICES_M,
    LOADED_TRANSLATION_DIRECTIONS,
    LONGITUDINAL_ALIGNMENT_TRIGGER_M,
    MAX_LONGITUDINAL_ALIGNMENT_DELTA_M,
    MODEL_ROUTE_TURN_ANGLE_DEG,
    MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG,
    MODEL_COORDINATION_PRIMITIVE_METHODS,
    MIN_LONGITUDINAL_ALIGNMENT_DELTA_M,
    VERIFIED_LINEAR_RAMP_S,
    VERIFIED_LINEAR_SPEED_MPS,
    VERIFIED_LOADED_LINEAR_MAX_M,
    coordination_modes_for_method,
    coordination_primitive_library_doc,
    coordination_primitive_method_spec,
    coordination_role_variant_limit,
    validate_coordination_primitive_parameters,
)
from .prompts import (
    COORDINATION_PARAMETER_SYSTEM_PROMPT,
    MODEL_AGENT_SYSTEM_PROMPT,
    SUBTASK_SELECTION_SYSTEM_PROMPT,
    SYNC_COMPENSATION_SYSTEM_PROMPT,
    TURN_EXECUTION_PARAMETER_SYSTEM_PROMPT,
    build_blocked_transactions_prompt,
    build_blocked_strategies_prompt,
    build_coordination_parameter_retry_prompt,
    build_coordination_parameter_user_prompt,
    build_evolution_user_prompt,
    build_initial_policy_user_prompt,
    build_model_replan_prompt,
    build_subtask_selection_retry_prompt,
    build_subtask_selection_user_prompt,
    build_sync_compensation_retry_prompt,
    build_sync_compensation_user_prompt,
    build_turn_execution_parameter_retry_prompt,
    build_turn_execution_parameter_user_prompt,
    manual_layout_scene_prior,
)
from .static_specs import (
    MODEL_ACTION_SKILL_METHODS,
    MODEL_SCRIPT_API_METHODS,
    load_static_skill_catalog,
    static_skill_library_doc,
)
from .turn_episode import (
    ALIGNED_PASSAGE_POSTURES,
    HOST_WALL_CLEARANCE_ESCAPE_MARKER,
    HOST_WALL_CLEARANCE_ESCAPE_PARAMETER_SOURCE,
    HOST_WALL_CLEARANCE_ESCAPE_SELECTION_SOURCE,
    HOST_WALL_HEADING_CORRECTION_MARKER,
    HOST_WALL_HEADING_CORRECTION_PARAMETER_SOURCE,
    HOST_WALL_HEADING_CORRECTION_SELECTION_SOURCE,
    HOST_WALL_TURN_ESCAPE_MARKER,
    HOST_WALL_TURN_ESCAPE_PARAMETER_SOURCE,
    HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE,
    PASSAGE_ALIGNED_STATUS,
    PASSAGE_ALIGNMENT_REQUIRED_STATUS,
    ROLE_BASED_TURN_METHODS,
    STATIC_TURN_METHODS,
    TURN_METHODS,
    TurnEpisodeError,
    policy_turn_intent,
    validate_selected_parameters_for_turn_episode,
)

MODEL_PLANNING_ACTION_METHODS = (
    MODEL_ACTION_SKILL_METHODS | MODEL_COORDINATION_PRIMITIVE_METHODS
)
MODEL_PLANNING_SCRIPT_API_METHODS = (
    MODEL_SCRIPT_API_METHODS | MODEL_COORDINATION_PRIMITIVE_METHODS
)
_COORDINATION_OBJECTIVE_PARAMETER = {
    COORDINATED_LINEAR_TRANSPORT: "direction",
    COORDINATED_ARC_TRANSPORT: "turn_direction",
    COORDINATED_PIVOT: "turn_direction",
    COORDINATED_ALIGNMENT: "axis",
}
_STATIC_MOTION_OBJECTIVES = {
    "carry_forward": "forward",
    "carry_backward": "backward",
    "carry_lateral_left": "left",
    "carry_lateral_right": "right",
    "turn_clockwise": "clockwise",
    "turn_counterclockwise": "counterclockwise",
}
_SIGNED_TURN_METHODS = frozenset(
    {
        COORDINATED_ARC_TRANSPORT,
        COORDINATED_PIVOT,
    }
)
_SIGNED_TURN_DIRECTIONS = frozenset({"clockwise", "counterclockwise"})
_GENERIC_TURN_EVIDENCE = (
    "default",
    "valid option",
    "either direction",
    "arbitrary",
)
_SUPPORT_ONLY_METHODS = frozenset(
    {
        "hold_position",
        "place_crouch",
        "recover_stand",
        "recover_payload_inward",
    }
)
_PRE_RELEASE_SETUP_METHODS = frozenset(
    {
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
        "turn_clockwise",
        "turn_counterclockwise",
        COORDINATED_LINEAR_TRANSPORT,
        COORDINATED_PIVOT,
        COORDINATED_ALIGNMENT,
        "hold_position",
    }
)
_POST_RELEASE_EXIT_METHODS = frozenset({"exit_backward"})
# ``release_payload`` performs the final guarded crouch itself. A level payload
# may enter that atomic transaction from the full release-height window; a
# tilted payload must first be lowered into the smaller setup window.
_MAX_RELEASE_ATOMIC_DROP_M = 0.45
_MAX_PRELOWERED_RELEASE_DROP_M = 0.30
_MAX_UNLOWERED_RELEASE_TILT_DEG = DEFAULT_MAX_TERMINAL_PAYLOAD_TILT_DEG
# Compatibility export. Runtime prompts use ``BoxCarryTask.mission_topology``;
# the algorithm does not own this benchmark-specific phase list.
MODEL_MISSION_TOPOLOGY = BoxCarryTask(task_id="", target="").mission_topology
CHECKPOINT_EXHAUSTED_COORDINATION_RELATIONS_FILE = (
    "checkpoint_exhausted_coordination_relations.json"
)
FORMATION_SYNC_REPAIR_GATE_FILE = "formation_sync_repair_gate.json"
FORMATION_SYNC_REPAIR_PROBE_DISTANCE_M = 1.0
FORMATION_SYNC_REPAIR_MAX_NORMALIZED_MISMATCH = 0.20
FORMATION_SYNC_REPAIR_MIN_PROGRESS_M = 0.05


class TextGenerationClient(Protocol):
    """Minimal model interface used by the Code-as-Policy agents."""

    def generate(self, *, system: str, user: str) -> str: ...


class ModelAgentError(RuntimeError):
    """Raised when a model response cannot be used as an executable policy."""


@dataclass
class ModelTaskOrchestrator:
    """Model-backed target -> executable policy script generator."""

    client: TextGenerationClient
    default_transport_skill: str = "carry_lateral_left"
    default_transport_distance_m: float = 1.0
    default_transport_speed_mps: float = 0.50
    default_transport_ramp_s: float = 1.5
    task_stage: str = "stage1_straight"
    route_discovery: bool = False
    require_team_plan: bool = True
    coordination_layer_enabled: bool = False
    action_memory: VerifiedActionMemoryStore | None = None
    role_pipeline: ModelRolePipeline | None = None
    accepted_episode_manifest_path: Path | None = None
    formal_eval: bool = False
    response_retries: int = 4

    def generate_initial_policy(
        self,
        task: BoxCarryTask,
        observation: ObservationBundle,
        *,
        attempt: int = 1,
    ) -> PolicyScript:
        route_discovery = bool(self.route_discovery or task.route_discovery)
        payload = {
            "task": model_task_context(task, route_discovery=route_discovery),
            "task_stage": self.task_stage,
            "route_discovery": route_discovery,
            "default_transport": {
                "skill": self.default_transport_skill,
                "distance_m": self.default_transport_distance_m,
                "speed_mps": self.default_transport_speed_mps,
                "ramp_s": self.default_transport_ramp_s,
            },
            "observation": _observation_payload(
                observation,
                route_discovery=route_discovery,
            ),
            "scene_prior": manual_layout_scene_prior(),
            "mission_topology": list(task.mission_topology),
            "static_skill_sdk": static_skill_library_doc(),
            "formal_model_owns_route": self.formal_eval,
        }
        if self.coordination_layer_enabled:
            payload["coordination_primitive_sdk"] = coordination_primitive_library_doc()
        accepted_route_history = _accepted_route_action_history(
            self.accepted_episode_manifest_path
        )
        completed_prefix = 0
        if accepted_route_history:
            payload["accepted_route_action_history"] = accepted_route_history
        route_phase_guard = trusted_route_phase_guard(
            task,
            observation.runtime_state,
            accepted_steps=accepted_route_history.get("steps", ()),
        )
        if route_phase_guard:
            payload["host_route_phase_guard"] = route_phase_guard
        completed_prefix = int(route_phase_guard.get("maximum_completed_prefix", 0))
        payload["route_spec"] = route_spec_document(task)
        active_route_guidance = route_guidance(
            task,
            observation.runtime_state,
            completed_prefix=completed_prefix,
        )
        if active_route_guidance:
            payload["active_route_guidance"] = active_route_guidance
        lifecycle_probe = {"observation": {"runtime_state": observation.runtime_state}}
        required_call, required_reason = _required_initial_call(lifecycle_probe)
        lifecycle_admissibility = _lifecycle_admissibility(lifecycle_probe)
        recommended_method = (
            _first_policy_api_method(f"api.{required_call}")
            if required_call
            else ""
        )
        hard_required_method = str(
            lifecycle_admissibility.get("hard_required_method", "")
        ).strip()
        hard_required = bool(
            hard_required_method and recommended_method == hard_required_method
        )
        lifecycle_forbidden_methods = _lifecycle_forbidden_methods(lifecycle_probe)
        _assert_formal_hard_contract_nonempty(
            formal_eval=self.formal_eval,
            allowed_methods=(
                MODEL_PLANNING_SCRIPT_API_METHODS
                if self.coordination_layer_enabled
                else MODEL_SCRIPT_API_METHODS
            ),
            forbidden_methods=lifecycle_forbidden_methods,
            required_method=(
                hard_required_method if hard_required else ""
            ),
        )
        if self.role_pipeline is not None:
            host_pivot_role_contract = _host_pivot_role_contract(
                observation.runtime_state
            )
            previous_task_plan = _lifecycle_consistent_previous_task_plan(
                self.role_pipeline.current_task_plan,
                forbidden_methods=lifecycle_forbidden_methods,
            )
            host_wall_clearance = (
                authored_scene_motion_clearance(observation.runtime_state)
                if route_discovery
                else {}
            )
            final_approach_transport_envelope = _final_approach_transport_envelope(
                observation.runtime_state,
                task_plan=previous_task_plan,
                host_wall_clearance=host_wall_clearance,
            )
            turn_clearance_replan_contract = (
                _host_wall_turn_clearance_replan_contract(
                    runtime_state=observation.runtime_state,
                    host_wall_clearance=host_wall_clearance,
                )
                if route_discovery
                else {}
            )
            route_clearance_conflict_contract = (
                _route_clearance_conflict_contract(
                    accepted_route_history,
                    host_wall_clearance=host_wall_clearance,
                    active_route_guidance=active_route_guidance,
                    runtime_state=observation.runtime_state,
                )
                if route_discovery
                else {}
            )
            trusted_final_target_projection = (
                _body_frame_final_target_projection(observation.runtime_state)
                if final_approach_transport_envelope.get("active", False)
                else {}
            )
            role_context = {
                **payload,
                "host_release_preflight_evidence": _release_preflight_planning_context(
                    observation.runtime_state,
                    route_discovery=route_discovery,
                    lifecycle=lifecycle_admissibility,
                    final_approach_envelope=final_approach_transport_envelope,
                ),
                "current_state_authority": _trusted_state_authority_summary(
                    payload["observation"].get("runtime_state", {})
                ),
                "previous_task_plan": previous_task_plan,
                "host_lifecycle_gate": {
                    **lifecycle_admissibility,
                    "hard_required": hard_required,
                    "required_call": required_call,
                    "recommended_method": recommended_method,
                    "reason": required_reason,
                    "forbidden_methods": sorted(lifecycle_forbidden_methods),
                },
                "role_boundary": (
                    "Maintain the abstract whole-task route plan, identify the current "
                    "phase, and assign one concrete team subtask. The Coder emits Python "
                    "from that guidance; the host validates but does not select the action."
                ),
            }
            if host_wall_clearance:
                role_context["host_authored_wall_clearance"] = host_wall_clearance
            if turn_clearance_replan_contract:
                role_context[
                    "advisory_turn_clearance"
                    if self.formal_eval
                    else "turn_clearance_replan_contract"
                ] = turn_clearance_replan_contract
            if route_clearance_conflict_contract:
                role_context[
                    "advisory_route_clearance"
                    if self.formal_eval
                    else "route_clearance_conflict_contract"
                ] = route_clearance_conflict_contract
            if final_approach_transport_envelope:
                role_context[
                    "advisory_final_approach_transport"
                    if self.formal_eval
                    else "final_approach_transport_envelope"
                ] = final_approach_transport_envelope
            if trusted_final_target_projection:
                role_context[
                    "advisory_target_projection"
                    if self.formal_eval
                    else "trusted_body_frame_target_projection"
                ] = trusted_final_target_projection
            if host_pivot_role_contract:
                role_context[
                    "advisory_pivot_role_geometry"
                    if self.formal_eval
                    else "host_pivot_role_contract"
                ] = host_pivot_role_contract
            role_context["formal_v50_lite"] = self.formal_eval

            def validate_initial_task_plan(candidate: dict[str, Any]) -> None:
                if not self.formal_eval:
                    _apply_host_pivot_roles_to_task_plan(
                        candidate,
                        runtime_state=observation.runtime_state,
                    )
                    _normalize_route_clearance_reentry_navigation(
                        candidate,
                        contract=route_clearance_conflict_contract,
                    )
                route_validation = validate_task_plan_route_reference(
                    task,
                    observation.runtime_state,
                    completed_prefix=completed_prefix,
                    task_plan=candidate,
                )
                if (
                    not self.formal_eval
                    and route_validation.get("active", False)
                    and not route_validation.get("valid", False)
                ):
                    raise ModelRoleError(
                        "current_subtask.navigation.world_direction_xy contradicts "
                        "the active connected route segment: "
                        f"{route_validation.get('reason', 'invalid route reference')}"
                    )
                local_navigation = build_local_navigation_contract(
                    candidate,
                    observation.runtime_state,
                )
                strategy = _execution_strategy_from_task_plan(candidate)
                candidate_final_approach_envelope = _final_approach_transport_envelope(
                    observation.runtime_state,
                    task_plan=candidate,
                    execution_strategy=strategy,
                    host_wall_clearance=host_wall_clearance,
                )
                turn_clearance_replan = bool(
                    _host_wall_turn_episode_replan_kind(
                        method=str(strategy.get("primitive", "")),
                        objective=str(strategy.get("semantic_objective", "")),
                        runtime_state=observation.runtime_state,
                        host_wall_clearance=host_wall_clearance,
                    )
                    == "translation"
                )
                if not self.formal_eval:
                    try:
                        validate_strategy_against_local_navigation(
                            method=str(strategy.get("primitive", "")),
                            objective=str(strategy.get("semantic_objective", "")),
                            contract=local_navigation,
                            allow_turn_clearance_replan=turn_clearance_replan,
                            route_clearance_conflict_contract=(
                                route_clearance_conflict_contract
                            ),
                        )
                    except ValueError as exc:
                        raise ModelRoleError(str(exc)) from exc
                _validate_orchestrator_plan_strategy(
                    candidate,
                    allowed_sdk_methods=(
                        MODEL_PLANNING_SCRIPT_API_METHODS
                        if self.coordination_layer_enabled
                        else MODEL_SCRIPT_API_METHODS
                    ),
                    hard_required_method=(
                        hard_required_method if hard_required else ""
                    ),
                    forbidden_methods=lifecycle_forbidden_methods,
                    runtime_state=observation.runtime_state,
                    host_wall_clearance=host_wall_clearance,
                    route_clearance_conflict_contract=(
                        route_clearance_conflict_contract
                    ),
                    turn_clearance_replan_contract=(
                        turn_clearance_replan_contract
                    ),
                    trusted_body_frame_target_projection=(
                        trusted_final_target_projection
                    ),
                    final_approach_transport_envelope=(
                        candidate_final_approach_envelope
                    ),
                    final_target_navigation_authorized=(
                        completed_prefix >= len(task.route_progress_gates)
                    ),
                    model_owns_route=self.formal_eval,
                )

            try:
                task_plan = self.role_pipeline.plan_task(
                    role_context,
                    artifact_id=f"attempt_{attempt:02d}",
                    image_path=_latest_observation_frame(observation),
                    validator=validate_initial_task_plan,
                )
            except ModelRoleError as exc:
                raise ModelAgentError(f"Task Orchestrator role failed: {exc}") from exc
            if self.formal_eval:
                shadow_strategy = _execution_strategy_from_task_plan(task_plan)
                shadow_route = validate_task_plan_route_reference(
                    task,
                    observation.runtime_state,
                    completed_prefix=completed_prefix,
                    task_plan=task_plan,
                )
                shadow_local_navigation = build_local_navigation_contract(
                    task_plan,
                    observation.runtime_state,
                )
                _write_formal_shadow_advisory(
                    self.role_pipeline.output_dir,
                    attempt=attempt,
                    checkpoint_id=_observation_checkpoint_id(observation),
                    stage="task_orchestrator_initial",
                    strategy=shadow_strategy,
                    checks=[
                        {
                            "name": "active_route_reference",
                            "legacy_would_reject": bool(
                                shadow_route.get("active", False)
                                and not shadow_route.get("valid", False)
                            ),
                            "reason": str(shadow_route.get("reason", "")),
                        },
                        _formal_shadow_check(
                            "local_navigation",
                            lambda: validate_strategy_against_local_navigation(
                                method=str(
                                    shadow_strategy.get("primitive", "")
                                ),
                                objective=str(
                                    shadow_strategy.get("semantic_objective", "")
                                ),
                                contract=shadow_local_navigation,
                                route_clearance_conflict_contract=(
                                    route_clearance_conflict_contract
                                ),
                            ),
                        ),
                        _formal_shadow_check(
                            "legacy_orchestrator_action_filters",
                            lambda: _validate_orchestrator_plan_strategy(
                                task_plan,
                                allowed_sdk_methods=(
                                    MODEL_PLANNING_SCRIPT_API_METHODS
                                    if self.coordination_layer_enabled
                                    else MODEL_SCRIPT_API_METHODS
                                ),
                                hard_required_method=(
                                    hard_required_method if hard_required else ""
                                ),
                                forbidden_methods=lifecycle_forbidden_methods,
                                runtime_state=observation.runtime_state,
                                host_wall_clearance=host_wall_clearance,
                                route_clearance_conflict_contract=(
                                    route_clearance_conflict_contract
                                ),
                                turn_clearance_replan_contract=(
                                    turn_clearance_replan_contract
                                ),
                                trusted_body_frame_target_projection=(
                                    trusted_final_target_projection
                                ),
                                final_approach_transport_envelope=(
                                    final_approach_transport_envelope
                                ),
                                model_owns_route=False,
                            ),
                        ),
                    ],
                )
            payload["task_orchestrator_plan"] = task_plan
            required_method = (
                hard_required_method if hard_required else ""
            )
            strategy = _execution_strategy_from_task_plan(
                task_plan,
                required_method=required_method,
                required_reason=required_reason,
            )
            local_navigation = build_local_navigation_contract(
                task_plan,
                observation.runtime_state,
            )
            final_approach_transport_envelope = _final_approach_transport_envelope(
                observation.runtime_state,
                task_plan=task_plan,
                execution_strategy=strategy,
                host_wall_clearance=host_wall_clearance,
            )
            host_wall_turn_angle_envelope = _host_wall_turn_angle_envelope(
                str(strategy.get("primitive", "")).strip(),
                str(strategy.get("semantic_objective", "")).strip(),
                host_wall_clearance,
                route_clearance_conflict_contract=(
                    route_clearance_conflict_contract
                ),
            )
            _parsed, policy = _generate_plan_guided_team_policy(
                role_pipeline=self.role_pipeline,
                task_plan=task_plan,
                execution_strategy=strategy,
                generation_context={
                    **role_context,
                    "allowed_sdk_methods": sorted(
                        MODEL_PLANNING_SCRIPT_API_METHODS
                        if self.coordination_layer_enabled
                        else MODEL_SCRIPT_API_METHODS
                    ),
                    "static_skill_sdk": static_skill_library_doc(),
                    "coordination_primitive_sdk": (
                        coordination_primitive_library_doc()
                        if self.coordination_layer_enabled
                        else {}
                    ),
                    "local_navigation_contract": local_navigation,
                    (
                        "advisory_final_approach_transport"
                        if self.formal_eval
                        else "final_approach_transport_envelope"
                    ): final_approach_transport_envelope,
                    (
                        "advisory_route_clearance"
                        if self.formal_eval
                        else "route_clearance_conflict_contract"
                    ): route_clearance_conflict_contract,
                    (
                        "advisory_turn_clearance"
                        if self.formal_eval
                        else "turn_clearance_replan_contract"
                    ): turn_clearance_replan_contract,
                    (
                        "advisory_wall_turn_angle_envelope"
                        if self.formal_eval
                        else "host_wall_turn_angle_envelope"
                    ): host_wall_turn_angle_envelope,
                },
                attempt=attempt,
                fallback_id=f"box_carry_attempt_{attempt:02d}",
                parent_script_id="",
                require_team_plan=self.require_team_plan,
                expected_method=required_method,
                forbidden_transaction_signatures=(),
                current_policy=None,
                repair_level="",
                repair_failure_type="",
            )
            write_json(
                self.role_pipeline.output_dir
                / f"subtask_assignment_attempt_{attempt:02d}.json",
                {
                    "task_plan": task_plan,
                    "execution_strategy": strategy,
                    "source": (
                        "host_lifecycle_gate" if hard_required else "task_orchestrator"
                    ),
                },
            )
            return policy
        if self.coordination_layer_enabled and not hard_required:
            selector_payload = {
                "task_stage": self.task_stage,
                "route_discovery": route_discovery,
                "formal_model_owns_route": self.formal_eval,
                "current_policy": {},
                "evaluation": {
                    "success": True,
                    "failure_type": "none",
                    "failure_level": "none",
                    "metrics": {"task_complete": False},
                },
                "observation": payload["observation"],
                "scene_prior": payload["scene_prior"],
                "static_skill_sdk": payload["static_skill_sdk"],
                "coordination_primitive_sdk": payload["coordination_primitive_sdk"],
                "evolution_controller": {
                    "action": "continue",
                    "repair_level": "none",
                },
                "recent_rejected_team_transactions": [],
                "recent_rejected_rollout_evidence": [],
            }
            forbidden_initial_objectives: set[str] = set()
            parameter_selection_errors: list[str] = []
            for _selection_attempt in range(
                len(MODEL_COORDINATION_PRIMITIVE_METHODS) * 3
            ):
                selected_subtask = _select_evolution_subtask(
                    client=self.client,
                    payload=selector_payload,
                    forbidden_methods=(),
                    forbidden_objectives=tuple(sorted(forbidden_initial_objectives)),
                )
                _attach_verified_action_exemplars(
                    selector_payload,
                    selected_subtask=selected_subtask,
                    action_memory=self.action_memory,
                )
                if (
                    selected_subtask.get("selected_method")
                    not in MODEL_COORDINATION_PRIMITIVE_METHODS
                ):
                    break
                selector_payload.pop("unverified_linear_direction_probe", None)
                initial_linear_probe: dict[str, Any] = {}
                checkpoint_id = _observation_checkpoint_id(observation)
                if checkpoint_id and checkpoint_id != "accepted_step_00":
                    initial_linear_probe = (
                        _unverified_linear_direction_probe_constraint(
                            PolicyScript(
                                script_id="checkpoint_resume_seed",
                                attempt=max(0, attempt - 1),
                                code="",
                            ),
                            selected_subtask=selected_subtask,
                            payload=selector_payload,
                        )
                    )
                if initial_linear_probe:
                    selector_payload["unverified_linear_direction_probe"] = (
                        initial_linear_probe
                    )

                def validate_initial_parameters(candidate: dict[str, Any]) -> None:
                    if initial_linear_probe:
                        _validate_unverified_linear_direction_probe_parameters(
                            candidate,
                            constraint=initial_linear_probe,
                        )

                try:
                    selected_parameters = _select_coordination_parameters(
                        client=self.client,
                        selected_subtask=selected_subtask,
                        payload=selector_payload,
                        candidate_validator=validate_initial_parameters,
                    )
                except ModelAgentError as exc:
                    method = str(selected_subtask.get("selected_method", "")).strip()
                    objective = str(
                        selected_subtask.get("motion_objective", "")
                    ).strip()
                    objective_signature = f"{method}:{objective}" if objective else ""
                    if (
                        not objective_signature
                        or objective_signature in forbidden_initial_objectives
                    ):
                        raise
                    forbidden_initial_objectives.add(objective_signature)
                    parameter_selection_errors.append(f"{objective_signature}: {exc}")
                    selector_payload["recent_rejected_team_transactions"] = [
                        {
                            "physically_dispatched": False,
                            "method": method,
                            "motion_objective": objective,
                            "failure": "structured_parameter_validation",
                            "error": str(exc),
                        }
                    ]
                    continue
                _compiled, policy = _compile_model_selected_coordination_policy(
                    current_policy=PolicyScript(
                        script_id="",
                        attempt=max(0, attempt - 1),
                        code="",
                    ),
                    attempt=attempt,
                    selected_subtask=selected_subtask,
                    selected_parameters=selected_parameters,
                    failure_level="none",
                    failure_type="none",
                    blocked_strategy_signatures=set(),
                    blocked_transaction_signatures=set(),
                )
                return policy
            else:
                raise ModelAgentError(
                    "initial coordination selection exhausted schema-valid objectives: "
                    + "; ".join(parameter_selection_errors)
                )
        return _generate_valid_policy(
            client=self.client,
            user_prompt=build_initial_policy_user_prompt(
                payload,
                attempt,
                required_call=required_call,
                required_reason=required_reason,
                hard_required=hard_required,
                exact_safe_transaction=_single_step_code_template(required_call),
                allowed_api_methods=tuple(
                    MODEL_PLANNING_SCRIPT_API_METHODS
                    if self.coordination_layer_enabled
                    else MODEL_SCRIPT_API_METHODS
                ),
            ),
            attempt=attempt,
            fallback_id=f"box_carry_attempt_{attempt:02d}",
            require_team_plan=self.require_team_plan,
            response_retries=self.response_retries,
            expected_method=(
                _first_policy_api_method(f"api.{required_call}")
                if hard_required
                else ""
            ),
        )

    def skill_library_prompt(self) -> str:
        if self.coordination_layer_enabled:
            return (
                static_skill_library_doc()
                + "\n\n"
                + coordination_primitive_library_doc()
            )
        return static_skill_library_doc()


@dataclass
class ModelCollaborativeEvolutionAgent:
    """Model-backed script repair and specification-memory summarizer."""

    output_dir: Path
    client: TextGenerationClient
    task: BoxCarryTask | None = None
    task_stage: str = "stage1_straight"
    route_discovery: bool = False
    require_team_plan: bool = True
    response_retries: int = 4
    rejected_transaction_limit: int = 8
    strategy_failure_budget: int = 0
    hierarchical_subtask_selection: bool = True
    coordination_layer_enabled: bool = False
    measured_sync_compensation_enabled: bool = False
    action_memory: VerifiedActionMemoryStore | None = None
    accepted_episode_manifest_path: Path | None = None
    history_dirs: tuple[Path, ...] = ()
    checkpoint_relation_history_dirs: tuple[Path, ...] = ()
    evolution_controller: EvolutionController | None = None
    branch_rewind_context: dict[str, Any] | None = None
    role_pipeline: ModelRolePipeline | None = None
    formal_eval: bool = False
    _manager_decisions: dict[int, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _generation_cycle_outcomes: dict[int, list[dict[str, Any]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _infeasible_mutation_levels: dict[str, dict[str, str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _minimum_mutation_rank_by_checkpoint: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _rejected_transactions: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _rejected_transaction_summaries: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _directionally_rejected_methods: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _directionally_rejected_objectives: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _nonrelaxable_motion_objectives: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _rejected_checkpoint_id: str = field(default="", init=False, repr=False)
    _rejected_checkpoint_source_attempt: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _strategy_failure_counts: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _strategy_failure_keys: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _blocked_strategy_signatures: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _blocked_motion_objectives: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _exhausted_coordination_relations: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _formation_sync_repair_gate: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.evolution_controller is None:
            self.evolution_controller = EvolutionController(
                max_physical_iterations=20,
                strategy_failure_budget=self.strategy_failure_budget,
                output_dir=self.output_dir,
            )
        gate_path = self.output_dir / FORMATION_SYNC_REPAIR_GATE_FILE
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            gate = {}
        if (
            not self.formal_eval
            and isinstance(gate, dict)
            and gate.get("schema_version") == 1
        ):
            self._formation_sync_repair_gate = gate

    @property
    def blocked_strategy_signatures(self) -> tuple[str, ...]:
        """Return model-generation exclusions, never physical failure-count bans."""

        blocked = set(self._blocked_strategy_signatures)
        return tuple(sorted(blocked))

    @property
    def blocked_motion_objectives(self) -> tuple[str, ...]:
        """Return explicit model/route exclusions, never count-based exhaustion."""

        blocked = set(self._blocked_motion_objectives)
        return tuple(sorted(blocked))

    @property
    def nonrelaxable_motion_objectives(self) -> tuple[str, ...]:
        """Return checkpoint-local route-risk exclusions that exhaustion cannot reopen."""

        return tuple(sorted(self._nonrelaxable_motion_objectives))

    @property
    def formation_sync_repair_gate(self) -> dict[str, Any]:
        """Return the persistent causal verification gate for loaded translation."""

        return dict(self._formation_sync_repair_gate)

    def decide_next_step(
        self,
        policy: PolicyScript,
        report: EvaluationReport,
        observation: ObservationBundle | None = None,
        trace: ExecutionTrace | None = None,
        *,
        checkpoint_id: str = "",
        physically_dispatched: bool | None = None,
        hard_iteration_limit_reached: bool = False,
    ) -> EvolutionDecision:
        """Apply verifier evidence to the external host controller before Qwen runs."""

        active_checkpoint_id = (
            checkpoint_id or _observation_checkpoint_id(observation)
        ).strip()
        if physically_dispatched:
            # A completed physical transaction supplies new evidence.  Recovery
            # scope is checkpoint-local and starts narrow again for that evidence.
            self.reset_after_physical_transaction(
                checkpoint_id=active_checkpoint_id,
                policy_attempt=policy.attempt,
            )
        self._prepare_evolution_evidence(policy, report, observation, trace)
        signature = _policy_strategy_budget_signature(policy)
        assert self.evolution_controller is not None
        host_decision = self.evolution_controller.record_and_decide(
            policy=policy,
            report=report,
            trace=trace,
            checkpoint_id=(checkpoint_id or _observation_checkpoint_id(observation)),
            physically_dispatched=physically_dispatched,
            hard_iteration_limit_reached=hard_iteration_limit_reached,
            strategy_failure_count=self._strategy_failure_counts.get(signature, 0),
        )
        if (
            self.role_pipeline is None
            or not host_decision.should_evolve
            or report.success
            or host_decision.repair_level in {"none", "system"}
            or _preflight_candidate_infeasible(report)
        ):
            return host_decision
        failure_analysis = _failure_analysis_from_report(report)
        if not self.formal_eval:
            failure_analysis = _apply_physics_authoritative_failure_analysis_override(
                failure_analysis,
                policy=policy,
                report=report,
                trace=trace,
            )
        rejected_rollout = _evaluation_payload(
            report,
            include_goal_metrics=not self.route_discovery,
        )
        current_trusted_observation = _observation_payload(
            observation,
            route_discovery=self.route_discovery,
        )
        current_trusted_runtime = current_trusted_observation.get(
            "runtime_state",
            {},
        )
        current_trusted_runtime = (
            current_trusted_runtime if isinstance(current_trusted_runtime, dict) else {}
        )
        raw_trusted_runtime = (
            observation.runtime_state
            if observation is not None and isinstance(observation.runtime_state, dict)
            else {}
        )
        host_wall_clearance = (
            authored_scene_motion_clearance(current_trusted_runtime)
            if self.route_discovery
            else {
                "available": False,
                "reason": "authored-wall clearance is active only in route-discovery mode",
            }
        )
        turn_clearance_replan_contract = _host_wall_turn_clearance_replan_contract(
            runtime_state=raw_trusted_runtime,
            host_wall_clearance=host_wall_clearance,
        )
        route_recovery_contract = _route_recovery_contract(
            policy,
            report,
            host_wall_clearance=host_wall_clearance,
            expose_clearance_candidates=not self.formal_eval,
            scope_checkpoint_id=_observation_checkpoint_id(observation),
        )
        parameter_envelope_exhausted = _minimum_effective_parameter_exhaustion(
            policy,
            report,
            trace=trace,
            forbidden_transaction_signatures=set(self._rejected_transactions),
        )
        alignment_sign_repair_contract = (
            {}
            if self.formal_eval
            else _alignment_sign_repair_contract(
                policy,
                report,
                trace,
                forbidden_transaction_signatures=set(self._rejected_transactions),
            )
        )
        active_sync_repair_gate = self._active_formation_sync_repair_gate()
        final_approach_transport_envelope = _final_approach_transport_envelope(
            raw_trusted_runtime,
            task_plan=self.role_pipeline.current_task_plan,
            policy=policy,
            host_wall_clearance=host_wall_clearance,
            forbidden_transaction_signatures=set(self._rejected_transactions),
            formation_sync_repair_gate=active_sync_repair_gate,
        )
        active_sync_repair_gate = _reconcile_formation_sync_gate_with_final_approach(
            active_sync_repair_gate,
            final_approach_transport_envelope=final_approach_transport_envelope,
            runtime_state=raw_trusted_runtime,
        )
        self._persist_reconciled_formation_sync_gate(active_sync_repair_gate)
        active_sync_probe_contract = (
            active_sync_repair_gate.get("probe_contract", {})
            if isinstance(active_sync_repair_gate, dict)
            else {}
        )
        active_sync_probe_contract = (
            active_sync_probe_contract
            if isinstance(active_sync_probe_contract, dict)
            else {}
        )
        if bool(
            active_sync_probe_contract.get(
                "magnitude_escalation_pending",
                False,
            )
        ):
            parameter_envelope_exhausted = {}
        raw_host_formation_gate = (
            {}
            if parameter_envelope_exhausted
            else _trusted_checkpoint_formation_gate(
                rejected_rollout,
                current_trusted_observation.get("runtime_state", {}),
            )
        )
        formation_gate_superseded = _rejected_alignment_invalidates_formation_gate(
            raw_host_formation_gate,
            policy=policy,
            report=report,
        )
        exhausted_relations = (
            ()
            if self.formal_eval
            else tuple(self._exhausted_coordination_relations.values())
        )
        host_formation_gate = (
            {}
            if (
                formation_gate_superseded
                or _coordination_relation_is_checkpoint_exhausted(
                    raw_host_formation_gate,
                    exhausted_relations,
                )
            )
            else raw_host_formation_gate
        )
        deferred_host_formation_gate = (
            host_formation_gate if active_sync_repair_gate else {}
        )
        if active_sync_repair_gate:
            host_formation_gate = {}
        mutation_parent_strategy = _policy_mutation_parent_strategy(policy)
        mutation_contract = build_mutation_contract(mutation_parent_strategy)
        self._apply_infeasible_mutation_levels(
            mutation_contract,
            checkpoint_id=active_checkpoint_id,
        )
        minimum_mutation_rank = self._minimum_mutation_rank_by_checkpoint.get(
            active_checkpoint_id,
            0,
        )
        apply_minimum_mutation_rank(mutation_contract, minimum_mutation_rank)
        _assert_formal_hard_contract_nonempty(
            formal_eval=self.formal_eval,
            allowed_methods=MODEL_PLANNING_SCRIPT_API_METHODS,
            mutation_contract=mutation_contract,
        )
        local_navigation_contract = _policy_local_navigation_contract(policy)
        if local_navigation_contract:
            local_navigation_contract = refresh_local_navigation_contract(
                local_navigation_contract,
                raw_trusted_runtime,
            )
        host_pivot_role_contract = _host_pivot_role_contract(
            raw_trusted_runtime
        )
        host_locked_coordination_reason = (
            ""
            if self.formal_eval
            else _host_locked_coordination_mutation_reason(
                mutation_parent_strategy,
                host_pivot_role_contract=host_pivot_role_contract,
            )
        )
        coordination_variants_exhausted = bool(
            not self.formal_eval
            and (
                _coordination_variants_exhausted(
                    mutation_parent_strategy,
                    exhausted_relations,
                )
                or host_locked_coordination_reason
            )
        )
        if coordination_variants_exhausted:
            coordination_level = mutation_contract["levels"][
                "coordination_mutation"
            ]
            coordination_level["available"] = False
            coordination_level["unavailable_reason"] = (
                host_locked_coordination_reason
                or (
                    "all executable temporary-role variants for this primitive, "
                    "semantic objective, and mode have verifier-grounded failures at "
                    "the current checkpoint"
                )
            )
            mutation_contract["available_levels"] = [
                level
                for level in mutation_contract["available_levels"]
                if level != "coordination_mutation"
            ]
        if not self.formal_eval:
            _reconcile_mutation_contract_with_local_navigation(
                mutation_contract,
                parent_strategy=mutation_parent_strategy,
                local_navigation_contract=local_navigation_contract,
                report=report,
                parameter_envelope_exhausted=parameter_envelope_exhausted,
            )
        manager_context = {
            "formal_v50_lite": self.formal_eval,
            "previous_subtask_outcome": "rejected",
            "rejected_endpoint_policy": _evolution_policy_payload(policy),
            "rejected_endpoint_rollout": rejected_rollout,
            "rejected_endpoint_failure_analysis": failure_analysis,
            "rejected_endpoint_execution_trace": _trace_payload(trace),
            "current_trusted_observation": current_trusted_observation,
            "current_state_authority": _trusted_state_authority_summary(
                current_trusted_runtime
            ),
            "host_authored_wall_clearance": host_wall_clearance,
            (
                "advisory_turn_clearance"
                if self.formal_eval
                else "turn_clearance_replan_contract"
            ): turn_clearance_replan_contract,
            "host_controller_guard": (
                _formal_host_controller_guard(host_decision)
                if self.formal_eval
                else host_decision.to_dict()
            ),
            "mutation_hierarchy": [
                "parameter_tuning",
                "coordination_mutation",
                "strategy_mutation",
            ],
            "mutation_contract": mutation_contract,
            "allowed_mutation_levels": list(
                mutation_contract.get("available_levels", ())
            ),
            "minimum_mutation_rank": minimum_mutation_rank,
            "minimum_mutation_level": MUTATION_LEVEL_BY_RANK[
                minimum_mutation_rank
            ],
            "host_authority": (
                "The host owns task completion, system failures, physical safety, "
                "rollback, exact-AST replay protection, and iteration accounting. "
                "The Evolution Manager owns the mutation level for this repair."
            ),
            "sdk_reference": (
                "Mutation levels and mutable fields are fully defined by "
                "mutation_contract; Task Orchestrator and Coder receive the expanded "
                "static/coordination SDK contracts."
            ),
        }
        if local_navigation_contract:
            manager_context["local_navigation_contract"] = (
                local_navigation_contract
            )
        if active_sync_repair_gate and not self.formal_eval:
            manager_context["formation_sync_repair_gate"] = active_sync_repair_gate
        if final_approach_transport_envelope and not self.formal_eval:
            manager_context["final_approach_transport_envelope"] = (
                final_approach_transport_envelope
            )
        if alignment_sign_repair_contract:
            manager_context["alignment_sign_repair_contract"] = (
                alignment_sign_repair_contract
            )
        if route_recovery_contract:
            manager_context["route_recovery_contract"] = route_recovery_contract
        if deferred_host_formation_gate:
            manager_context["deferred_host_formation_gate"] = {
                **deferred_host_formation_gate,
                "deferred_reason": (
                    "The active synchronization contract is already the route-neutral "
                    "physical precondition. "
                    + (
                        "Its linear probe is unavailable, so the required geometry "
                        "repair takes precedence."
                        if active_sync_repair_gate.get("probe_unavailable", False)
                        else "Run its loaded linear probe first."
                    )
                ),
            }
        if host_formation_gate:
            manager_context["host_formation_gate"] = host_formation_gate
        if self.formal_eval and raw_host_formation_gate:
            manager_context["advisory_formation_repair"] = {
                **raw_host_formation_gate,
                "enforcement": "shadow_only",
                "decision_owner": "evolution_manager_and_task_orchestrator",
            }
        if formation_gate_superseded:
            manager_context["superseded_host_formation_gate"] = (
                _superseded_formation_gate_context(raw_host_formation_gate)
            )
        if parameter_envelope_exhausted:
            manager_context["parameter_envelope_exhausted"] = (
                parameter_envelope_exhausted
            )
        if exhausted_relations:
            manager_context["checkpoint_exhausted_coordination_relations"] = list(
                exhausted_relations
            )
        if host_pivot_role_contract:
            manager_context[
                "advisory_pivot_role_geometry"
                if self.formal_eval
                else "host_pivot_role_contract"
            ] = host_pivot_role_contract
        active_operator_guidance = _active_branch_rewind_context(
            {"branch_rewind_context": self.branch_rewind_context or {}},
            (
                observation.runtime_state
                if observation is not None
                and isinstance(observation.runtime_state, dict)
                else {}
            ),
        )
        if active_operator_guidance:
            manager_context["checkpoint_local_operator_guidance"] = {
                "scope_checkpoint_id": active_operator_guidance.get(
                    "scope_checkpoint_id", ""
                ),
                "lesson": active_operator_guidance.get("lesson", ""),
                "expires": active_operator_guidance.get(
                    "persistence_mode", "until_next_accepted_checkpoint"
                ),
            }
        previous_cycle_outcomes = self._generation_cycle_outcomes.get(
            int(policy.attempt),
            [],
        )
        if previous_cycle_outcomes:
            manager_context["previous_generation_cycle_outcomes"] = [
                dict(item) for item in previous_cycle_outcomes
            ]
            manager_context["generation_cycle_recovery_rule"] = (
                "These are pre-dispatch generation outcomes, not physical failures. "
                "Reassess the mutation level from the current trusted state. Recovery "
                "scope expands monotonically: select only a level listed in "
                "allowed_mutation_levels and never select below minimum_mutation_rank. "
                "A prior "
                "NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL, SCOPE_MISMATCH, EXACT_BLOCKED, "
                "or NO_EFFECTIVE_DELTA with result NO_EXECUTABLE_CHILD_PRODUCED means "
                "the selected level produced no legal child in that cycle and narrower "
                "levels are no longer available. "
                "A prior MUTATION_SCOPE_INFEASIBLE means the selected mutation "
                "operator had no contract-valid effective child. The Host still does "
                "not select a replacement primitive, direction, mode, or roles."
            )
            manager_context["previous_generation_failure"] = dict(
                previous_cycle_outcomes[-1]
            )

        def validate_manager(candidate: dict[str, Any]) -> None:
            try:
                validate_manager_level(candidate, mutation_contract)
            except ValueError as exc:
                raise ModelRoleError(str(exc)) from exc
            _validate_manager_mutation_level(
                candidate,
                report=report,
                failure_analysis=failure_analysis,
                policy=policy,
                # Route discovery redacts target coordinates from model prompts,
                # but host validation needs the trusted target projection to
                # distinguish a required heading repair from a route mutation.
                runtime_state=raw_trusted_runtime,
                formation_gate=host_formation_gate,
                formation_sync_repair_gate=active_sync_repair_gate,
                parameter_envelope_exhausted=parameter_envelope_exhausted,
                alignment_sign_repair_contract=alignment_sign_repair_contract,
                model_owns_mutation_level=self.formal_eval,
                minimum_structural_mutation=(
                    host_decision.semantic_failure_count >= 2
                ),
                coordination_variants_exhausted=coordination_variants_exhausted,
                mutation_contract=mutation_contract,
            )

        selection_retry = 0
        while True:
            if not mutation_contract.get("available_levels"):
                raise HostContractError(
                    "no mutation level remains after checkpoint-local minimum-rank "
                    f"and infeasible-scope filtering at {active_checkpoint_id}"
                )
            try:
                manager = self.role_pipeline.decide_mutation(
                    manager_context,
                    artifact_id=f"attempt_{policy.attempt:02d}",
                    image_path=_latest_observation_frame(observation),
                    validator=validate_manager,
                )
            except ModelRoleStallError:
                raise
            except ModelRoleError as exc:
                raise ModelAgentError(
                    f"Evolution Manager role failed: {exc}"
                ) from exc
            try:
                self._reject_infeasible_selected_mutation_scope(
                    manager,
                    parent_strategy=mutation_parent_strategy,
                    # Formal pivot-role identity is model-owned.  The advisory
                    # Host geometry must not prove that role mutation is empty.
                    host_pivot_role_contract=(
                        {} if self.formal_eval else host_pivot_role_contract
                    ),
                    checkpoint_id=active_checkpoint_id,
                    artifact_id=f"attempt_{policy.attempt:02d}",
                )
            except ModelRoleStallError as exc:
                if exc.cycle_outcome != MUTATION_SCOPE_INFEASIBLE:
                    raise
                selection_retry += 1
                self._apply_infeasible_mutation_levels(
                    mutation_contract,
                    checkpoint_id=active_checkpoint_id,
                )
                apply_minimum_mutation_rank(
                    mutation_contract,
                    minimum_mutation_rank,
                )
                manager_context["mutation_contract"] = mutation_contract
                manager_context["allowed_mutation_levels"] = list(
                    mutation_contract.get("available_levels", ())
                )
                manager_context["manager_selection_retry"] = {
                    "retry_index": selection_retry,
                    "rejected_level": str(
                        manager.get("mutation_level", "")
                    ).strip(),
                    "reason": str(exc),
                    "generation_cycle_consumed": False,
                }
                continue
            break

        decision = _evolution_decision_from_manager(host_decision, manager)
        self._manager_decisions[policy.attempt] = dict(manager)
        write_json(
            self.output_dir / f"model_evolution_manager_{policy.attempt:02d}.json",
            {
                "host_guard": host_decision.to_dict(),
                "failure_analysis": failure_analysis,
                "manager": manager,
                "effective_decision": decision.to_dict(),
            },
        )
        return decision

    def rewind_to_checkpoint(
        self,
        checkpoint_id: str,
        *,
        branch_rewind_context: dict[str, Any],
    ) -> int:
        """Retire child-branch exclusions and reload evidence for one ancestor state."""

        checkpoint_id = str(checkpoint_id).strip()
        if not checkpoint_id:
            raise ValueError("checkpoint rewind requires a non-empty checkpoint id")
        self.branch_rewind_context = dict(branch_rewind_context)
        retired = 0
        if self.evolution_controller is not None:
            retired = self.evolution_controller.rewind_strategy_evidence_to_checkpoint(
                checkpoint_id
            )
        self._refresh_rejected_transactions(checkpoint_id=checkpoint_id)
        return retired

    def reset_generation_cycle(
        self,
        policy_attempt: int,
        *,
        stall_event: dict[str, Any] | None = None,
    ) -> None:
        """Force a fresh Manager decision and retain bounded cycle evidence."""

        policy_attempt = int(policy_attempt)
        manager_decision = self._manager_decisions.pop(policy_attempt, {})
        if not isinstance(stall_event, dict) or not stall_event:
            return
        selected_level = str(
            stall_event.get("selected_mutation_level", "")
            or manager_decision.get("mutation_level", "")
        ).strip()
        typed_failure = str(stall_event.get("typed_failure", "")).strip()
        cycle_outcome = str(stall_event.get("cycle_outcome", "")).strip()
        checkpoint_id = str(stall_event.get("checkpoint_id", "")).strip()
        escalation_failure = bool(
            cycle_outcome == "NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL"
            or typed_failure
            in {"SCOPE_MISMATCH", "NO_EFFECTIVE_DELTA", "EXACT_BLOCKED"}
        )
        if escalation_failure and selected_level in MUTATION_LEVEL_RANK:
            current_rank = self._minimum_mutation_rank_by_checkpoint.get(
                checkpoint_id,
                0,
            )
            next_rank = min(2, MUTATION_LEVEL_RANK[selected_level] + 1)
            self._minimum_mutation_rank_by_checkpoint[checkpoint_id] = max(
                current_rank,
                next_rank,
            )
        outcome = {
            key: stall_event.get(key)
            for key in (
                "cycle_outcome",
                "checkpoint_id",
                "generation_cycle",
                "role",
                "artifact_id",
                "selected_mutation_level",
                "response_count",
                "physical_dispatch",
                "validation_errors",
                "validation_error_kinds",
                "typed_failure",
                "attempted_change",
                "editable_fields",
                "result",
            )
        }
        outcome["selected_mutation_level"] = selected_level
        outcome["minimum_mutation_rank_after"] = (
            self._minimum_mutation_rank_by_checkpoint.get(checkpoint_id, 0)
        )
        outcome["minimum_mutation_level_after"] = MUTATION_LEVEL_BY_RANK[
            outcome["minimum_mutation_rank_after"]
        ]
        history = self._generation_cycle_outcomes.setdefault(policy_attempt, [])
        history.append(outcome)
        del history[:-3]
        write_json(
            self.output_dir
            / f"generation_cycle_feedback_attempt_{policy_attempt:02d}.json",
            {
                "policy_attempt": policy_attempt,
                "outcomes": history,
            },
        )

    def reset_after_physical_transaction(
        self,
        *,
        checkpoint_id: str,
        policy_attempt: int,
    ) -> None:
        """Reset checkpoint-local generation escalation after real execution."""

        checkpoint_id = str(checkpoint_id).strip()
        if checkpoint_id:
            self._minimum_mutation_rank_by_checkpoint.pop(checkpoint_id, None)
        self._generation_cycle_outcomes.pop(int(policy_attempt), None)
        self._manager_decisions.pop(int(policy_attempt), None)

    def _apply_infeasible_mutation_levels(
        self,
        contract: dict[str, Any],
        *,
        checkpoint_id: str,
    ) -> None:
        marked = self._infeasible_mutation_levels.get(str(checkpoint_id).strip(), {})
        for level, reason in marked.items():
            apply_infeasible_mutation_level(contract, level, reason=reason)

    def _reject_infeasible_selected_mutation_scope(
        self,
        manager: dict[str, Any],
        *,
        parent_strategy: dict[str, Any],
        host_pivot_role_contract: dict[str, Any] | None,
        checkpoint_id: str,
        artifact_id: str,
    ) -> None:
        level = str(manager.get("mutation_level", "")).strip()
        scope = check_selected_mutation_scope(
            level,
            parent_strategy=parent_strategy,
            host_pivot_role_contract=host_pivot_role_contract,
        )
        if scope.get("status") != MUTATION_SCOPE_STATUS_INFEASIBLE:
            return
        checkpoint_id = str(checkpoint_id).strip()
        reason = str(scope.get("reason", "")).strip()
        if checkpoint_id:
            self._infeasible_mutation_levels.setdefault(checkpoint_id, {})[level] = (
                reason
            )
        raise ModelRoleStallError(
            reason,
            role="evolution_manager",
            artifact_id=artifact_id,
            response_count=1,
            validation_errors=(reason,),
            cycle_outcome=MUTATION_SCOPE_INFEASIBLE,
            selected_mutation_level=level,
            validation_error_kinds=("mutation_scope_infeasible",),
        )

    def _prepare_evolution_evidence(
        self,
        policy: PolicyScript,
        report: EvaluationReport,
        observation: ObservationBundle | None,
        trace: ExecutionTrace | None,
    ) -> None:
        checkpoint_id = _observation_checkpoint_id(observation)
        if not self.formal_eval:
            self._update_formation_sync_repair_gate(
                policy,
                report,
                trace,
                checkpoint_id=checkpoint_id,
                runtime_state=(
                    observation.runtime_state
                    if observation is not None
                    and isinstance(observation.runtime_state, dict)
                    else {}
                ),
            )
        if self.evolution_controller is not None:
            self.evolution_controller.set_active_checkpoint(checkpoint_id)
        self._refresh_rejected_transactions(
            checkpoint_id=checkpoint_id,
            runtime_state=(
                observation.runtime_state
                if observation is not None
                and isinstance(observation.runtime_state, dict)
                else {}
            ),
        )
        if not self.formal_eval:
            for relation in _load_checkpoint_alignment_parameter_exhaustions(
                (
                    *self.checkpoint_relation_history_dirs,
                    *self.history_dirs,
                    self.output_dir,
                ),
                checkpoint_id=checkpoint_id,
                minimum_attempt_index=self._rejected_checkpoint_source_attempt,
            ):
                self._remember_exhausted_coordination_relation(relation)
        # A branch rewind evaluates the failed child transaction before restoring
        # an ancestor.  The next model turn observes that ancestor, so recording
        # the child policy again here would incorrectly make its rejection look
        # checkpoint-local to the restored state.
        retired_child_evidence = _report_restored_ancestor_checkpoint(
            report,
            checkpoint_id=checkpoint_id,
        )
        if not report.success and not retired_child_evidence:
            self._remember_rejected_transaction(
                policy,
                failure_type=str(report.failure_type or ""),
            )
            self._remember_failed_strategy(policy, report=report, trace=trace)
            parameter_exhaustion = (
                {}
                if self.formal_eval
                else _minimum_effective_alignment_parameter_exhaustion(policy, report)
            )
            if parameter_exhaustion:
                self._remember_exhausted_coordination_relation(
                    {
                        **parameter_exhaustion,
                        "scope_checkpoint_id": checkpoint_id,
                    }
                )
        if (
            not self.formal_eval
            and not retired_child_evidence
            and _has_measured_goal_regression(
                report.failure_type,
                report.metrics,
                evidence=report.evidence,
            )
        ):
            self._remember_directionally_rejected_methods(policy)
            if report.failure_type == "route_regression":
                _methods, objectives = _policy_directional_rejections(policy)
                self._nonrelaxable_motion_objectives.update(objectives)

    def _active_formation_sync_repair_gate(self) -> dict[str, Any]:
        if not bool(self._formation_sync_repair_gate.get("active", False)):
            return {}
        return {
            key: value
            for key, value in self._formation_sync_repair_gate.items()
            if key != "last_processed_event_key"
        }

    def _persist_reconciled_formation_sync_gate(
        self,
        gate: dict[str, Any] | None,
    ) -> None:
        """Keep a checkpoint-local unavailable probe closed across model turns."""

        contract = gate if isinstance(gate, dict) else {}
        if not (
            contract.get("active", False) and contract.get("probe_unavailable", False)
        ):
            return
        last_processed_event_key = self._formation_sync_repair_gate.get(
            "last_processed_event_key",
            "",
        )
        self._formation_sync_repair_gate = dict(contract)
        if last_processed_event_key:
            self._formation_sync_repair_gate["last_processed_event_key"] = (
                last_processed_event_key
            )
        write_json(
            self.output_dir / FORMATION_SYNC_REPAIR_GATE_FILE,
            self._formation_sync_repair_gate,
        )

    def _update_formation_sync_repair_gate(
        self,
        policy: PolicyScript,
        report: EvaluationReport,
        trace: ExecutionTrace | None,
        *,
        checkpoint_id: str,
        runtime_state: dict[str, Any] | None = None,
    ) -> None:
        """Keep dynamic gait-repair evidence alive across accepted posture steps."""

        trace_contract = (
            trace.verification_contract
            if trace is not None and isinstance(trace.verification_contract, dict)
            else {}
        )
        policy_contract = (
            policy.verification_contract
            if isinstance(policy.verification_contract, dict)
            else {}
        )
        verification_contract = trace_contract or policy_contract
        event_key = "|".join(
            (
                str(policy.attempt),
                policy.script_id,
                str(bool(report.success)),
                str(report.failure_type),
                checkpoint_id,
                str(verification_contract.get("kind", "")),
            )
        )
        if (
            event_key
            and event_key
            == str(
                self._formation_sync_repair_gate.get(
                    "last_processed_event_key",
                    "",
                )
            )
            and not (
                _checkpoint_roundtrip_geometry_repair_failed(report)
                and not self._formation_sync_repair_gate.get(
                    "probe_unavailable",
                    False,
                )
            )
        ):
            return

        is_linear = _policy_uses_linear_transport(policy)
        probe_result = (
            report.evidence.get("formation_sync_repair_probe", {})
            if isinstance(report.evidence, dict)
            else {}
        )
        probe_result = probe_result if isinstance(probe_result, dict) else {}
        probe_contract_active = bool(
            verification_contract.get("kind") == "formation_sync_repair_probe"
        )
        route_obstruction = _rejected_route_obstruction_evidence(report)
        route_level_regression = bool(
            not report.success
            and is_linear
            and report.failure_type in {"goal_regression", "route_regression"}
        )

        if not report.success and is_linear and route_obstruction.get("active"):
            previous = (
                dict(self._formation_sync_repair_gate)
                if bool(self._formation_sync_repair_gate)
                else {}
            )
            self._formation_sync_repair_gate = {
                "schema_version": 1,
                "kind": "formation_sync_repair",
                "active": False,
                "status": "superseded_by_route_obstruction",
                "current_checkpoint_id": checkpoint_id,
                "superseded_at": {
                    "attempt": policy.attempt,
                    "script_id": policy.script_id,
                    "failure_type": str(report.failure_type),
                },
                "route_obstruction": route_obstruction,
                "previous_gate": {
                    key: value
                    for key, value in previous.items()
                    if key != "last_processed_event_key"
                },
                "last_processed_event_key": event_key,
                "note": (
                    "A wall collision or wall-confirmed single-sided support is a "
                    "route/strategy defect. It supersedes same-direction synchronization "
                    "probing after rollback."
                ),
            }
            write_json(
                self.output_dir / FORMATION_SYNC_REPAIR_GATE_FILE,
                self._formation_sync_repair_gate,
            )
            return

        if route_level_regression and bool(
            self._formation_sync_repair_gate.get("active", False)
        ):
            previous = dict(self._formation_sync_repair_gate)
            self._formation_sync_repair_gate = {
                "schema_version": 1,
                "kind": "formation_sync_repair",
                "active": False,
                "status": "superseded_by_route_regression",
                "current_checkpoint_id": checkpoint_id,
                "superseded_at": {
                    "attempt": policy.attempt,
                    "script_id": policy.script_id,
                    "failure_type": str(report.failure_type),
                },
                "probe_contract_active": probe_contract_active,
                "probe_result": dict(probe_result),
                "previous_gate": {
                    key: value
                    for key, value in previous.items()
                    if key != "last_processed_event_key"
                },
                "last_processed_event_key": event_key,
                "note": (
                    "The loaded synchronization probe cannot override measured "
                    "route-level regression. Preserve any useful synchronization "
                    "measurement, retire the probe gate, and require a strategy change."
                ),
            }
            write_json(
                self.output_dir / FORMATION_SYNC_REPAIR_GATE_FILE,
                self._formation_sync_repair_gate,
            )
            return

        active_probe_physics_failure = bool(
            self._formation_sync_repair_gate.get("active", False)
            and probe_contract_active
            and report.failure_type
            in {
                "payload_not_supported",
                "payload_slip",
                "payload_unstable",
            }
        )
        if (
            not report.success
            and is_linear
            and (
                report.failure_type == "formation_desync"
                or active_probe_physics_failure
            )
        ):
            previous = (
                self._formation_sync_repair_gate
                if bool(self._formation_sync_repair_gate.get("active", False))
                else {}
            )
            settle_hard_failure = bool(
                trace is not None
                and "formation_sync_settle_hard_limit" in str(trace.error)
            )
            failure_count = int(previous.get("failure_count", 0)) + 1
            # A hard settle abort on the original route transaction proves that
            # the two independent backends diverged during that rollout. It
            # does not prove that the restored trusted checkpoint has a static
            # formation defect. First require a shorter, direction-matched
            # synchronization probe. Escalate to a geometry-changing repair
            # only when that explicit probe (or payload support) also fails.
            if (
                active_probe_physics_failure
                or (settle_hard_failure and probe_contract_active)
            ):
                failure_count = max(2, failure_count)
            required_distance_m = _formation_sync_probe_distance_m(
                policy,
                force_short=settle_hard_failure,
            )
            probe_magnitude_escalated = False
            if (
                not settle_hard_failure
                and probe_contract_active
                and bool(probe_result.get("active", False))
                and bool(probe_result.get("distance_contract_satisfied", False))
                and bool(probe_result.get("meaningful_actual_progress", False))
                and bool(probe_result.get("no_wall_contact", False))
            ):
                try:
                    current_distance_m = float(
                        probe_result.get(
                            "required_distance_m",
                            required_distance_m,
                        )
                    )
                except (TypeError, ValueError):
                    current_distance_m = required_distance_m
                larger_distances = sorted(
                    distance_m
                    for distance_m in FINAL_APPROACH_LINEAR_DISTANCE_CHOICES_M
                    if distance_m > current_distance_m + 1.0e-9
                )
                if larger_distances:
                    required_distance_m = larger_distances[0]
                    probe_magnitude_escalated = True
                    failure_count = 0
            trigger = previous.get("trigger")
            if not isinstance(trigger, dict) or not trigger:
                trigger = {
                    "checkpoint_id": checkpoint_id,
                    "attempt": policy.attempt,
                    "script_id": policy.script_id,
                    "transaction": _policy_transaction_for_prompt(policy),
                }
            self._formation_sync_repair_gate = {
                "schema_version": 1,
                "kind": "formation_sync_repair",
                "active": True,
                "status": (
                    "probe_failed_settle_hard_limit"
                    if settle_hard_failure
                    else "probe_failed_payload_support"
                    if active_probe_physics_failure
                    else "probe_magnitude_escalated_after_short_window_desync"
                    if probe_magnitude_escalated
                    else "probe_required"
                ),
                "trigger": trigger,
                "current_checkpoint_id": checkpoint_id,
                "failure_count": failure_count,
                "cumulative_failure_count": int(
                    previous.get(
                        "cumulative_failure_count",
                        previous.get("failure_count", 0),
                    )
                )
                + 1,
                "probe_contract": {
                    "primitive": COORDINATED_LINEAR_TRANSPORT,
                    "semantic_objective": _policy_linear_objective(policy),
                    "required_distance_m": required_distance_m,
                    "magnitude_escalation_pending": probe_magnitude_escalated,
                    "magnitude_escalation_reason": (
                        "the shorter probe produced meaningful loaded motion but "
                        "its window ended before front/back response converged"
                        if probe_magnitude_escalated
                        else ""
                    ),
                    "maximum_normalized_displacement_mismatch": (
                        FORMATION_SYNC_REPAIR_MAX_NORMALIZED_MISMATCH
                    ),
                    "minimum_actual_progress_m": (FORMATION_SYNC_REPAIR_MIN_PROGRESS_M),
                    "clearance_rule": (
                        "Only a physically accepted loaded linear probe clears "
                        "this gate; alignment or recovery acceptance does not."
                    ),
                },
                "last_failure": {
                    "attempt": policy.attempt,
                    "script_id": policy.script_id,
                    "failure_type": str(report.failure_type),
                    "metrics": {
                        key: report.metrics.get(key)
                        for key in (
                            "front_step_displacement_m",
                            "back_step_displacement_m",
                            "front_back_step_vector_desync_m",
                        )
                        if key in report.metrics
                    },
                },
                "last_processed_event_key": event_key,
            }
        elif bool(self._formation_sync_repair_gate.get("active", False)):
            self._formation_sync_repair_gate["current_checkpoint_id"] = checkpoint_id
            self._formation_sync_repair_gate["last_processed_event_key"] = event_key
            if _checkpoint_roundtrip_geometry_repair_failed(report):
                probe_contract = self._formation_sync_repair_gate.get(
                    "probe_contract",
                    {},
                )
                probe_contract = (
                    dict(probe_contract) if isinstance(probe_contract, dict) else {}
                )
                probe_contract.update(
                    {
                        "available": False,
                        "unavailable_reason": (
                            "final_approach_transport_envelope_exhausted"
                        ),
                    }
                )
                self._formation_sync_repair_gate.update(
                    {
                        "status": "geometry_repair_required",
                        "probe_unavailable": True,
                        "required_repair": COORDINATED_ALIGNMENT,
                        "repair_precedence": (
                            "The last non-linear repair did not survive checkpoint "
                            "round-trip semantics. Change the alignment axis or temporary "
                            "role relation; do not reopen linear transport or tune delta only."
                        ),
                        "probe_contract": probe_contract,
                    }
                )
                self._formation_sync_repair_gate["last_observation"] = {
                    "checkpoint_id": checkpoint_id,
                    "attempt": policy.attempt,
                    "script_id": policy.script_id,
                    "success": False,
                    "failure_type": str(report.failure_type),
                    "probe_contract_active": False,
                    "probe_result": {},
                    "note": (
                        "Checkpoint round-trip verification disproved this geometry "
                        "repair structure; keep the exhausted linear probe closed."
                    ),
                }
            elif (
                not report.success
                and probe_contract_active
                and report.failure_type == "no_payload_progress"
                and bool(probe_result.get("active", False))
                and bool(probe_result.get("distance_contract_satisfied", False))
                and bool(probe_result.get("no_wall_contact", False))
            ):
                probe_contract = self._formation_sync_repair_gate.get(
                    "probe_contract",
                    {},
                )
                probe_contract = (
                    dict(probe_contract) if isinstance(probe_contract, dict) else {}
                )
                try:
                    current_distance_m = float(
                        probe_contract.get(
                            "required_distance_m",
                            min(FINAL_APPROACH_LINEAR_DISTANCE_CHOICES_M),
                        )
                    )
                except (TypeError, ValueError):
                    current_distance_m = min(FINAL_APPROACH_LINEAR_DISTANCE_CHOICES_M)
                larger_distances = sorted(
                    distance_m
                    for distance_m in FINAL_APPROACH_LINEAR_DISTANCE_CHOICES_M
                    if distance_m > current_distance_m + 1.0e-9
                )
                if larger_distances:
                    next_distance_m = larger_distances[0]
                    prior_failure_count = int(
                        self._formation_sync_repair_gate.get(
                            "cumulative_failure_count",
                            self._formation_sync_repair_gate.get(
                                "failure_count",
                                0,
                            ),
                        )
                    )
                    probe_contract.update(
                        {
                            "required_distance_m": next_distance_m,
                            "previous_required_distance_m": current_distance_m,
                            "magnitude_escalation_pending": True,
                            "magnitude_escalation_reason": (
                                "the smaller direction-matched probe remained "
                                "physically safe but did not cross SONIC's "
                                "loaded lateral locomotion onset"
                            ),
                        }
                    )
                    self._formation_sync_repair_gate.update(
                        {
                            "status": "probe_magnitude_escalated",
                            "failure_count": 0,
                            "cumulative_failure_count": prior_failure_count + 1,
                            "probe_contract": probe_contract,
                        }
                    )
                else:
                    self._formation_sync_repair_gate.update(
                        {
                            "status": "probe_magnitude_exhausted",
                            "failure_count": max(
                                2,
                                int(
                                    self._formation_sync_repair_gate.get(
                                        "failure_count",
                                        0,
                                    )
                                )
                                + 1,
                            ),
                        }
                    )
                self._formation_sync_repair_gate["last_observation"] = {
                    "checkpoint_id": checkpoint_id,
                    "attempt": policy.attempt,
                    "script_id": policy.script_id,
                    "success": False,
                    "failure_type": str(report.failure_type),
                    "probe_contract_active": True,
                    "probe_result": dict(probe_result),
                    "note": (
                        "The bounded probe remained physically safe but did not "
                        "produce meaningful loaded motion. Advance at most to "
                        "the next final-approach distance tier."
                    ),
                }
            elif (
                report.success
                and probe_contract_active
                and bool(probe_result.get("passed", False))
            ):
                self._formation_sync_repair_gate.update(
                    {
                        "active": False,
                        "status": "verified",
                        "cleared_at": {
                            "checkpoint_id": checkpoint_id,
                            "attempt": policy.attempt,
                            "script_id": policy.script_id,
                        },
                        "successful_probe": dict(probe_result),
                    }
                )
            elif (
                report.success
                and _first_policy_api_method(policy.code)
                in {COORDINATED_ALIGNMENT, COORDINATED_PIVOT}
                and policy.attempt
                > int(
                    (
                        self._formation_sync_repair_gate.get("last_failure", {})
                        if isinstance(
                            self._formation_sync_repair_gate.get("last_failure"),
                            dict,
                        )
                        else {}
                    ).get("attempt", -1)
                )
            ):
                projection = _body_frame_final_target_projection(
                    runtime_state if isinstance(runtime_state, dict) else {}
                )
                remaining_target_turn = _declared_local_reorientation_turn_direction(
                    runtime_state if isinstance(runtime_state, dict) else {},
                    contract=_policy_local_navigation_contract(policy),
                )
                target_direction = str(
                    projection.get("required_target_facing_direction", "")
                ).strip()
                probe_contract = self._formation_sync_repair_gate.get(
                    "probe_contract",
                    {},
                )
                probe_contract = (
                    dict(probe_contract) if isinstance(probe_contract, dict) else {}
                )
                for key in ("available", "unavailable_reason"):
                    probe_contract.pop(key, None)
                for key in (
                    "probe_unavailable",
                    "required_repair",
                    "repair_precedence",
                ):
                    self._formation_sync_repair_gate.pop(key, None)
                if target_direction in LOADED_TRANSLATION_DIRECTIONS:
                    probe_contract["semantic_objective"] = target_direction
                reorientation_complete = not remaining_target_turn
                self._formation_sync_repair_gate.update(
                    {
                        "status": (
                            "probe_required_after_geometry_repair"
                            if reorientation_complete
                            else "target_reorientation_incomplete"
                        ),
                        "failure_count": (
                            0
                            if reorientation_complete
                            else max(
                                2,
                                int(
                                    self._formation_sync_repair_gate.get(
                                        "failure_count",
                                        0,
                                    )
                                ),
                            )
                        ),
                        "probe_contract": probe_contract,
                        "last_observation": {
                            "checkpoint_id": checkpoint_id,
                            "attempt": policy.attempt,
                            "script_id": policy.script_id,
                            "success": True,
                            "failure_type": "",
                            "probe_contract_active": False,
                            "probe_result": {},
                            "target_facing_direction": target_direction,
                            "remaining_target_turn": remaining_target_turn,
                            "note": (
                                "The target still projects strongly to the body side; "
                                "continue the bounded signed reorientation before "
                                "reopening loaded linear transport."
                                if remaining_target_turn
                                else (
                                    "An accepted geometry repair creates a new trusted "
                                    "checkpoint. Re-open one bounded linear probe in the "
                                    "new body-frame target direction."
                                )
                            ),
                        },
                    }
                )
            else:
                self._formation_sync_repair_gate["status"] = (
                    "geometry_repair_required"
                    if self._formation_sync_repair_gate.get(
                        "probe_unavailable",
                        False,
                    )
                    else "probe_required"
                )
                self._formation_sync_repair_gate["last_observation"] = {
                    "checkpoint_id": checkpoint_id,
                    "attempt": policy.attempt,
                    "script_id": policy.script_id,
                    "success": bool(report.success),
                    "failure_type": str(report.failure_type),
                    "probe_contract_active": probe_contract_active,
                    "probe_result": dict(probe_result),
                    "note": (
                        "Accepted non-probe posture/recovery steps do not verify "
                        "front/back locomotion synchronization."
                    ),
                }
        else:
            return
        write_json(
            self.output_dir / FORMATION_SYNC_REPAIR_GATE_FILE,
            self._formation_sync_repair_gate,
        )

    def evolve(
        self,
        policy: PolicyScript,
        report: EvaluationReport,
        observation: ObservationBundle | None = None,
        trace: ExecutionTrace | None = None,
        *,
        controller_decision: EvolutionDecision | None = None,
        _generation_retry_depth: int = 0,
    ) -> tuple[PolicyScript, str]:
        if report.success and bool(report.metrics.get("task_complete", 0.0)):
            return policy, ""
        if controller_decision is None:
            controller_decision = self.decide_next_step(
                policy,
                report,
                observation,
                trace,
            )
        else:
            self._prepare_evolution_evidence(policy, report, observation, trace)
        if not controller_decision.should_evolve:
            return policy, ""
        if self.role_pipeline is not None:
            return self._evolve_plan_guided(
                policy,
                report,
                observation,
                trace,
                controller_decision=controller_decision,
            )
        blocked_strategy_set = set(self.blocked_strategy_signatures)
        blocked_objective_set = set(self.blocked_motion_objectives)
        if self.formal_eval:
            # Formal runs retain rejected executable ASTs as hard evidence but
            # do not turn historical semantic choices into a host action menu.
            blocked_strategy_set = set()
            blocked_objective_set = set()
        assert self.evolution_controller is not None
        current_strategy = policy_strategy_signature(policy)
        current_objective = policy_motion_objective_signature(policy)
        ancestor_rewind_active = bool(
            self.branch_rewind_context
            and _active_branch_rewind_context(
                {"branch_rewind_context": self.branch_rewind_context},
                (
                    observation.runtime_state
                    if observation is not None
                    and isinstance(observation.runtime_state, dict)
                    else {}
                ),
            )
        )
        structured_candidate_preblocked = bool(
            controller_decision.repair_level in {"coordination", "script", "execution"}
            and (
                (current_objective and current_objective in blocked_objective_set)
                or (current_strategy and current_strategy in blocked_strategy_set)
            )
        )
        if structured_candidate_preblocked:
            if _generation_retry_depth >= 6:
                raise ModelAgentError(
                    "checkpoint-local coordination candidate exclusions exhausted the "
                    "bounded generation search"
                )
            escalated_decision = self.evolution_controller.escalate_candidate_space_exhausted(
                controller_decision,
                error=(
                    "the current coordination primitive and semantic objective already "
                    "exhausted host-valid generated candidates at this checkpoint"
                ),
            )
            write_json(
                self.output_dir
                / f"model_structured_preblocked_escalation_{policy.attempt:02d}.json",
                {
                    "checkpoint_id": self._rejected_checkpoint_id,
                    "strategy_signature": current_strategy,
                    "motion_objective_signature": current_objective,
                    "blocked_strategy_signatures": sorted(blocked_strategy_set),
                    "blocked_motion_objectives": sorted(blocked_objective_set),
                    "escalated_controller_decision": escalated_decision.to_dict(),
                },
            )
            return self.evolve(
                policy,
                report,
                observation,
                trace,
                controller_decision=escalated_decision,
                _generation_retry_depth=_generation_retry_depth + 1,
            )
        if (
            not self.formal_eval
            and controller_decision.repair_level == "strategy"
            and not ancestor_rewind_active
        ):
            if current_objective:
                blocked_objective_set.add(current_objective)
            elif current_strategy:
                blocked_strategy_set.add(current_strategy)
        blocked_strategies = tuple(sorted(blocked_strategy_set))
        strategy_failure_counts = dict(
            self.evolution_controller.strategy_failure_counts
        )
        for signature, count in self._strategy_failure_counts.items():
            strategy_failure_counts[signature] = max(
                count,
                strategy_failure_counts.get(signature, 0),
            )
        rejected_transactions = tuple(self._blocked_transaction_context_entries())
        evidence_history_dirs = tuple(
            dict.fromkeys((*self.history_dirs, self.output_dir))
        )
        recent_rejected_rollouts: list[dict[str, Any]] = []
        for history_dir in evidence_history_dirs:
            recent_rejected_rollouts.extend(
                _load_rejected_rollout_evidence(
                    history_dir,
                    checkpoint_id=self._rejected_checkpoint_id,
                    limit=self.rejected_transaction_limit,
                )
            )
        deduplicated_rejected_rollouts = {
            json.dumps(item, ensure_ascii=True, sort_keys=True): item
            for item in recent_rejected_rollouts
        }
        recent_rejected_rollouts = list(deduplicated_rejected_rollouts.values())[
            -max(1, int(self.rejected_transaction_limit)) :
        ]
        trusted_checkpoint_linear_motion: dict[str, Any] = {}
        for history_dir in evidence_history_dirs:
            candidate = _load_trusted_checkpoint_linear_motion(
                history_dir,
                checkpoint_id=self._rejected_checkpoint_id,
            )
            if candidate:
                trusted_checkpoint_linear_motion = candidate
        payload = {
            "task_stage": self.task_stage,
            "route_discovery": self.route_discovery,
            "current_policy": _evolution_policy_payload(policy),
            "evaluation": _evaluation_payload(
                report,
                include_goal_metrics=not self.route_discovery,
            ),
            "execution_trace": _trace_payload(trace),
            "observation": _observation_payload(
                observation,
                route_discovery=self.route_discovery,
            ),
            "scene_prior": manual_layout_scene_prior(),
            "allowed_api_methods": sorted(
                MODEL_PLANNING_SCRIPT_API_METHODS
                if self.coordination_layer_enabled
                else MODEL_SCRIPT_API_METHODS
            ),
            "static_skill_sdk": static_skill_library_doc(),
            "evolution_controller": controller_decision.to_dict(),
            "physical_failure_diagnostics": {
                "count_based_strategy_blocking": False,
                "failure_counts": dict(sorted(strategy_failure_counts.items())),
                "model_generation_exclusions": list(blocked_strategies),
                "explicit_motion_objective_exclusions": sorted(blocked_objective_set),
                "signature_definition": (
                    "coordination_mode:mover_method|method:semantic_objective for "
                    "parameterized coordination primitives"
                ),
                "exact_rejected_ast_replay_blocked": True,
            },
            "recent_rejected_team_transactions": list(rejected_transactions),
            "recent_rejected_rollout_evidence": recent_rejected_rollouts,
            "formal_model_owns_route": self.formal_eval,
        }
        manager_decision = self._manager_decisions.get(policy.attempt, {})
        if manager_decision:
            payload["evolution_manager"] = dict(manager_decision)
        if trusted_checkpoint_linear_motion:
            payload["trusted_checkpoint_linear_motion"] = (
                trusted_checkpoint_linear_motion
            )
        if self.branch_rewind_context:
            payload["branch_rewind_context"] = dict(self.branch_rewind_context)
        candidate_repair_level = _effective_candidate_repair_level(
            payload,
            requested_level=controller_decision.repair_level,
        )
        if candidate_repair_level != controller_decision.repair_level:
            payload["evolution_controller"] = {
                **payload["evolution_controller"],
                "action": "switch_strategy",
                "repair_level": candidate_repair_level,
                "reason": (
                    "The failed child branch was retired; select a distinct strategy "
                    "from the restored ancestor checkpoint."
                ),
                "required_change": (
                    "change the geometry-changing coordination primitive without "
                    "reattributing child failures to this ancestor"
                ),
            }
        if self.coordination_layer_enabled:
            payload["coordination_primitive_sdk"] = coordination_primitive_library_doc()
        if self.role_pipeline is not None and report.success:
            try:
                payload["task_orchestrator_plan"] = self.role_pipeline.plan_task(
                    {
                        **payload,
                        "role_boundary": (
                            "The previous bounded step is physically accepted. Select "
                            "the next high-level subtask without emitting code."
                        ),
                    },
                    artifact_id=f"attempt_{policy.attempt + 1:02d}",
                    image_path=_latest_observation_frame(observation),
                )
            except ModelRoleError as exc:
                raise ModelAgentError(f"Task Orchestrator role failed: {exc}") from exc
        required_method, required_example, required_reason = _required_evolution_call(
            payload,
            previous_policy=policy,
            previous_report=report,
        )
        lifecycle_admissibility = _lifecycle_admissibility(
            payload,
            previous_policy=policy,
            previous_report=report,
        )
        hard_required = bool(
            required_method
            and required_method
            == str(lifecycle_admissibility.get("hard_required_method", "")).strip()
        )
        lifecycle_forbidden_methods = _lifecycle_forbidden_methods(
            payload,
            previous_policy=policy,
            previous_report=report,
        )
        forbidden_methods = tuple(
            sorted(
                lifecycle_forbidden_methods
                if self.formal_eval
                else set(self._directionally_rejected_methods)
                | lifecycle_forbidden_methods
                | _branch_rewind_forbidden_methods(self.branch_rewind_context)
            )
        )
        payload["allowed_api_methods"] = sorted(
            set(payload["allowed_api_methods"]) - lifecycle_forbidden_methods
        )
        forbidden_objectives = (
            ()
            if self.formal_eval
            else tuple(
                sorted(
                    set(self._directionally_rejected_objectives) | blocked_objective_set
                )
            )
        )
        candidate_rejection_path = (
            self.output_dir
            / f"attempt_{policy.attempt + 1:02d}_candidate_rejections.jsonl"
        )
        selected_subtask: dict[str, str] = {}
        observation_payload = payload.get("observation", {})
        finalization_open = _payload_is_released_on_target(
            observation_payload.get("runtime_state", {})
            if isinstance(observation_payload, dict)
            else {}
        )
        operator_fast_start = (
            self.coordination_layer_enabled
            and not self.formal_eval
            and report.success
            and not bool(report.metrics.get("task_complete", 0.0))
            and _first_policy_api_method(policy.code) == "place_payload_on_arms"
        )
        if operator_fast_start:
            selected_subtask = {
                "selected_method": COORDINATED_LINEAR_TRANSPORT,
                "coordination_mode": "sync",
                "motion_objective_parameter": "direction",
                "motion_objective": "forward",
                "reason": (
                    "operator-provided fast opening seed after verified payload placement"
                ),
            }
            payload["operator_fast_start_seed"] = {
                "method": COORDINATED_LINEAR_TRANSPORT,
                "coordination_mode": "sync",
                "direction": "forward",
                "distance_m": 2.0,
                "speed_mps": VERIFIED_LINEAR_SPEED_MPS,
                "ramp_s": VERIFIED_LINEAR_RAMP_S,
            }
        if (
            not selected_subtask
            and not hard_required
            and self.hierarchical_subtask_selection
            and (
                bool(payload.get("coordination_primitive_sdk"))
                or finalization_open
                or bool(self._directionally_rejected_methods)
                or bool(blocked_strategies)
                or controller_decision.repair_level == "strategy"
                or (
                    controller_decision.repair_level == "coordination"
                    and report.failure_type == "formation_desync"
                    and _has_measured_executor_mismatch(report.metrics)
                )
                or (
                    controller_decision.repair_level == "execution"
                    and report.failure_type
                    in {
                        "formation_desync",
                        "payload_not_supported",
                        "payload_slip",
                        "payload_unstable",
                    }
                )
                or (
                    bool(report.success)
                    and _first_policy_api_method(policy.code) == "hold_position"
                    and bool(rejected_transactions)
                )
                or _candidate_rejection_count(candidate_rejection_path) >= 3
            )
        ):
            selected_subtask = _select_evolution_subtask(
                client=self.client,
                payload=payload,
                forbidden_methods=forbidden_methods,
                forbidden_objectives=forbidden_objectives,
                nonrelaxable_forbidden_objectives=(self.nonrelaxable_motion_objectives),
                blocked_strategy_signatures=blocked_strategies,
                response_audit_path=(
                    self.output_dir
                    / f"model_subtask_selection_{policy.attempt:02d}_rejections.jsonl"
                ),
            )
            write_json(
                self.output_dir / f"model_subtask_selection_{policy.attempt:02d}.json",
                selected_subtask,
            )
        if (
            self.role_pipeline is not None
            and not report.success
            and selected_subtask
            and not hard_required
        ):
            approved_candidate = dict(selected_subtask)

            def validate_mutation(candidate: dict[str, Any]) -> None:
                _validate_mutation_operator_choice(
                    candidate,
                    approved_candidate=approved_candidate,
                    manager_decision=manager_decision,
                    current_policy=policy,
                )

            try:
                mutation = self.role_pipeline.mutate_strategy(
                    {
                        "evolution_manager": manager_decision,
                        "failure_analysis": _failure_analysis_from_report(report),
                        "parent_policy": _evolution_policy_payload(policy),
                        "host_allowlisted_candidate": approved_candidate,
                        "instruction": (
                            "Express this host-allowlisted candidate as a reusable "
                            "strategy mutation at exactly the approved mutation level. "
                            "The executable parameters are selected later."
                        ),
                    },
                    artifact_id=f"attempt_{policy.attempt:02d}",
                    validator=validate_mutation,
                )
            except ModelRoleError as exc:
                raise ModelAgentError(f"Mutation Operator role failed: {exc}") from exc
            selected_subtask = {
                **selected_subtask,
                "model_mutation": mutation,
            }
            payload["approved_mutation"] = mutation
        selected_method = str(selected_subtask.get("selected_method", "")).strip()
        selected_mode = str(selected_subtask.get("coordination_mode", "")).strip()
        current_method = _first_policy_api_method(policy.code)
        current_modes = {
            str(script.coordination_mode).strip()
            for script in policy.robot_scripts
            if str(script.coordination_mode).strip()
        }
        current_mode = next(iter(current_modes)) if len(current_modes) == 1 else ""
        normalized_repair_level = _normalize_repair_level_for_selected_strategy(
            repair_level=candidate_repair_level,
            current_method=current_method,
            current_mode=current_mode,
            selected_method=selected_method,
            selected_mode=selected_mode,
        )
        if normalized_repair_level != candidate_repair_level:
            candidate_repair_level = normalized_repair_level
            payload["evolution_controller"] = {
                **payload["evolution_controller"],
                "action": "switch_strategy",
                "repair_level": candidate_repair_level,
                "reason": (
                    "Current trusted geometry selected a different executable "
                    "coordination strategy than the failed transaction."
                ),
                "required_change": (
                    "compile the geometry-selected method and mode while keeping "
                    "the exact rejected AST blocked"
                ),
            }
            payload["geometry_selected_strategy_transition"] = {
                "from": {
                    "method": current_method,
                    "coordination_mode": current_mode,
                },
                "to": {
                    "method": selected_method,
                    "coordination_mode": selected_mode,
                },
            }
        _attach_verified_action_exemplars(
            payload,
            selected_subtask=selected_subtask,
            action_memory=self.action_memory,
        )
        unverified_linear_probe = _unverified_linear_direction_probe_constraint(
            policy,
            selected_subtask=selected_subtask,
            payload=payload,
        )
        if unverified_linear_probe:
            payload["unverified_linear_direction_probe"] = unverified_linear_probe
        measured_sync_compensation = (
            _measured_sync_compensation_constraint(
                payload,
                selected_subtask=selected_subtask,
            )
            if self.measured_sync_compensation_enabled
            else None
        )
        selected_sync_parameters: dict[str, Any] = {}
        compiled_sync_policy: tuple[dict[str, Any], PolicyScript] | None = None
        if measured_sync_compensation:

            def validate_sync_candidate(candidate: dict[str, Any]) -> None:
                nonlocal compiled_sync_policy
                compiled_sync_policy = _compile_model_selected_sync_policy(
                    current_policy=policy,
                    attempt=policy.attempt + 1,
                    selected_subtask=selected_subtask,
                    selected_parameters=candidate,
                    measured_constraint=measured_sync_compensation,
                    failure_level=candidate_repair_level,
                    failure_type=report.failure_type,
                    blocked_strategy_signatures=set(blocked_strategies),
                    blocked_transaction_signatures=set(self._rejected_transactions),
                )

            try:
                selected_sync_parameters = _select_sync_compensation_parameters(
                    client=self.client,
                    constraint=measured_sync_compensation,
                    candidate_validator=validate_sync_candidate,
                )
            except ModelAgentError as exc:
                if _generation_retry_depth >= 6:
                    raise ModelAgentError(
                        "structured measured-sync candidate search exhausted seven distinct "
                        f"generation rounds: {exc}"
                    ) from exc
                assert self.evolution_controller is not None
                self._remember_candidate_space_exhaustion(selected_subtask)
                escalated_decision = (
                    self.evolution_controller.escalate_candidate_space_exhausted(
                        controller_decision,
                        error=str(exc),
                    )
                )
                write_json(
                    self.output_dir
                    / f"model_sync_candidate_exhaustion_{policy.attempt:02d}.json",
                    {
                        "checkpoint_id": self._rejected_checkpoint_id,
                        "selected_subtask": selected_subtask,
                        "error": str(exc),
                        "previous_controller_decision": controller_decision.to_dict(),
                        "escalated_controller_decision": escalated_decision.to_dict(),
                    },
                )
                return self.evolve(
                    policy,
                    report,
                    observation,
                    trace,
                    controller_decision=escalated_decision,
                    _generation_retry_depth=_generation_retry_depth + 1,
                )
            write_json(
                self.output_dir / f"model_sync_compensation_{policy.attempt:02d}.json",
                selected_sync_parameters,
            )
        turn_execution_constraint = _turn_execution_repair_constraint(
            policy,
            repair_level=candidate_repair_level,
            failure_type=report.failure_type,
        )
        linear_execution_constraint = _linear_execution_repair_constraint(
            policy,
            repair_level=candidate_repair_level,
            failure_type=report.failure_type,
            selected_subtask=selected_subtask,
        )
        linear_continuation_constraint = _linear_success_continuation_constraint(
            policy,
            report=report,
            selected_subtask=selected_subtask,
        )
        if not linear_continuation_constraint:
            linear_continuation_constraint = (
                _trusted_checkpoint_linear_continuation_constraint(
                    payload,
                    selected_subtask=selected_subtask,
                )
            )
        if not linear_continuation_constraint and operator_fast_start:
            linear_continuation_constraint = {
                "kind": "accepted_envelope_continuation",
                "method": COORDINATED_LINEAR_TRANSPORT,
                "coordination_mode": "sync",
                "direction": "forward",
                "previous_distance_m": 2.0,
                "maximum_distance_m": 2.0,
                "allowed_distance_choices_m": [2.0],
                "required_speed_mps": VERIFIED_LINEAR_SPEED_MPS,
                "required_ramp_s": VERIFIED_LINEAR_RAMP_S,
                "source": "operator_fast_start_seed",
            }
        selected_turn_parameters: dict[str, Any] = {}
        if turn_execution_constraint:
            selected_turn_parameters = _select_turn_execution_parameters(
                client=self.client,
                constraint=turn_execution_constraint,
            )
            write_json(
                self.output_dir
                / f"model_turn_execution_parameters_{policy.attempt:02d}.json",
                selected_turn_parameters,
            )
        selected_coordination_parameters: dict[str, Any] = {}
        compiled_coordination_policy: tuple[dict[str, Any], PolicyScript] | None = None
        selected_coordination_method = str(selected_subtask.get("selected_method", ""))
        if (
            selected_coordination_method in COORDINATION_PRIMITIVE_METHODS
            and not measured_sync_compensation
            and not turn_execution_constraint
        ):

            def validate_coordination_candidate(candidate: dict[str, Any]) -> None:
                nonlocal compiled_coordination_policy
                _validate_alignment_spacing_repair_candidate(
                    candidate,
                    runtime_state=observation.runtime_state,
                    report=report,
                )
                compiled_coordination_policy = (
                    _compile_model_selected_coordination_policy(
                        current_policy=policy,
                        attempt=policy.attempt + 1,
                        selected_subtask=selected_subtask,
                        selected_parameters=candidate,
                        failure_level=candidate_repair_level,
                        failure_type=report.failure_type,
                        blocked_strategy_signatures=set(blocked_strategies),
                        blocked_transaction_signatures=set(self._rejected_transactions),
                        explicit_branch_rewind=ancestor_rewind_active,
                    )
                )
                if linear_continuation_constraint:
                    _validate_linear_success_continuation_parameters(
                        candidate,
                        constraint=linear_continuation_constraint,
                    )
                if unverified_linear_probe:
                    _validate_unverified_linear_direction_probe_parameters(
                        candidate,
                        constraint=unverified_linear_probe,
                    )

            try:
                selected_coordination_parameters = _select_coordination_parameters(
                    client=self.client,
                    selected_subtask=selected_subtask,
                    payload=payload,
                    candidate_validator=validate_coordination_candidate,
                )
            except ModelAgentError as exc:
                parameter_fallback: dict[str, Any] = {}
                fallback_constraint = (
                    linear_execution_constraint or linear_continuation_constraint
                )
                fallback_error = ""
                if fallback_constraint:
                    parameter_fallback = _conservative_linear_parameters(
                        fallback_constraint,
                    )
                    if parameter_fallback:
                        parameter_fallback["model_selection_error"] = str(exc)
                        try:
                            validate_coordination_candidate(parameter_fallback)
                        except ModelAgentError as fallback_exc:
                            fallback_error = str(fallback_exc)
                            parameter_fallback = {}
                if not parameter_fallback:
                    parameter_fallback = _heading_correction_parameter_fallback(
                        selected_subtask,
                        observation.runtime_state.get("turn_episode", {}),
                    )
                    if parameter_fallback:
                        parameter_fallback["model_selection_error"] = str(exc)
                        try:
                            validate_coordination_candidate(parameter_fallback)
                        except ModelAgentError as fallback_exc:
                            fallback_error = str(fallback_exc)
                            parameter_fallback = {}
                if (
                    not parameter_fallback
                    and selected_coordination_method != COORDINATED_ALIGNMENT
                ):
                    for candidate in _conservative_coordination_parameter_candidates(
                        selected_subtask
                    ):
                        candidate["model_selection_error"] = str(exc)
                        try:
                            validate_coordination_candidate(candidate)
                        except ModelAgentError as fallback_exc:
                            fallback_error = str(fallback_exc)
                            continue
                        parameter_fallback = candidate
                        break
                if parameter_fallback:
                    selected_coordination_parameters = parameter_fallback
                    write_json(
                        self.output_dir
                        / f"model_coordination_parameter_fallback_{policy.attempt:02d}.json",
                        {
                            "checkpoint_id": self._rejected_checkpoint_id,
                            "constraint": fallback_constraint,
                            "model_error": str(exc),
                            "fallback": parameter_fallback,
                        },
                    )
                elif fallback_error:
                    exc = ModelAgentError(
                        f"{exc}; conservative compiler fallbacks were rejected: "
                        f"{fallback_error}"
                    )
                if not parameter_fallback:
                    if controller_decision.repair_level not in {
                        "none",
                        "coordination",
                        "script",
                        "execution",
                        "strategy",
                    }:
                        raise
                    if _generation_retry_depth >= 6:
                        raise ModelAgentError(
                            "structured coordination candidate search exhausted seven distinct "
                            f"generation rounds: {exc}"
                        ) from exc
                    assert self.evolution_controller is not None
                    self._remember_candidate_space_exhaustion(selected_subtask)
                    escalated_decision = (
                        self.evolution_controller.escalate_candidate_space_exhausted(
                            controller_decision,
                            error=str(exc),
                        )
                    )
                    write_json(
                        self.output_dir
                        / (
                            f"model_structured_candidate_exhaustion_{policy.attempt:02d}.json"
                            if _generation_retry_depth == 0
                            else (
                                "model_structured_candidate_exhaustion_"
                                f"{policy.attempt:02d}_retry_{_generation_retry_depth:02d}.json"
                            )
                        ),
                        {
                            "checkpoint_id": self._rejected_checkpoint_id,
                            "selected_subtask": selected_subtask,
                            "error": str(exc),
                            "previous_controller_decision": controller_decision.to_dict(),
                            "escalated_controller_decision": escalated_decision.to_dict(),
                        },
                    )
                    return self.evolve(
                        policy,
                        report,
                        observation,
                        trace,
                        controller_decision=escalated_decision,
                        _generation_retry_depth=_generation_retry_depth + 1,
                    )
            write_json(
                self.output_dir
                / f"model_coordination_parameters_{policy.attempt:02d}.json",
                selected_coordination_parameters,
            )
        if measured_sync_compensation and selected_sync_parameters:
            if compiled_sync_policy is None:
                raise ModelAgentError(
                    "structured measured-sync candidate was not compiler-validated"
                )
            parsed, repaired = compiled_sync_policy
        elif turn_execution_constraint and selected_turn_parameters:
            parsed, repaired = _compile_model_selected_turn_policy(
                current_policy=policy,
                attempt=policy.attempt + 1,
                selected_parameters=selected_turn_parameters,
                constraint=turn_execution_constraint,
                failure_level=controller_decision.repair_level,
                failure_type=report.failure_type,
                blocked_strategy_signatures=set(blocked_strategies),
            )
        elif selected_coordination_parameters:
            if compiled_coordination_policy is None:
                raise ModelAgentError(
                    "structured coordination candidate was not compiler-validated"
                )
            parsed, repaired = compiled_coordination_policy
        else:
            parsed, repaired = _generate_valid_policy_with_payload(
                client=self.client,
                user_prompt=build_evolution_user_prompt(
                    payload,
                    required_example=required_example,
                    required_reason=required_reason,
                    hard_required=hard_required,
                    forbidden_methods=forbidden_methods,
                    selected_subtask=selected_subtask,
                    measured_sync_compensation=measured_sync_compensation,
                    selected_sync_parameters=selected_sync_parameters,
                    blocked_strategy_signatures=blocked_strategies,
                ),
                attempt=policy.attempt + 1,
                fallback_id=f"box_carry_attempt_{policy.attempt + 1:02d}",
                parent_script_id=policy.script_id,
                require_team_plan=self.require_team_plan,
                response_retries=self.response_retries,
                expected_method=required_method if hard_required else "",
                forbidden_task_call_signatures=tuple(self._rejected_transactions),
                forbidden_task_transactions=tuple(
                    self._blocked_transaction_context_entries()
                ),
                forbidden_task_methods=forbidden_methods,
                forbidden_strategy_signatures=blocked_strategies,
                current_policy=policy,
                repair_level=("" if hard_required else candidate_repair_level),
                repair_failure_type=("" if hard_required else report.failure_type),
                selected_subtask_method=selected_subtask.get("selected_method", ""),
                selected_subtask_coordination=selected_subtask.get(
                    "coordination_mode", ""
                ),
                selected_subtask_motion_parameter=selected_subtask.get(
                    "motion_objective_parameter", ""
                ),
                selected_subtask_motion_objective=selected_subtask.get(
                    "motion_objective", ""
                ),
                measured_sync_compensation=measured_sync_compensation,
                selected_sync_parameters=selected_sync_parameters,
                retry_transport_limits=_unsafe_retry_transport_limits(
                    policy,
                    failure_type=report.failure_type,
                ),
                required_transport_ramp_s=(
                    1.5 if report.failure_type in _FIXED_RAMP_RETRY_FAILURES else None
                ),
                candidate_rejection_path=candidate_rejection_path,
                correction_code_template=(
                    _single_step_code_template(required_example)
                    if hard_required
                    else ""
                ),
            )
        if self.role_pipeline is not None:
            compiled_candidate = repaired
            role_policy: PolicyScript | None = None

            def validate_compiled_role_payload(
                candidate_payload: dict[str, Any],
            ) -> None:
                nonlocal role_policy
                role_policy = _validated_role_compiled_policy(
                    candidate_payload,
                    compiled_candidate=compiled_candidate,
                    current_policy=policy,
                    selected_subtask=selected_subtask,
                    require_team_plan=self.require_team_plan,
                )

            role_context = {
                "approved_high_level_plan": payload.get("task_orchestrator_plan", {}),
                "evolution_manager": manager_decision,
                "approved_mutation": payload.get("approved_mutation", {}),
                "host_compiled_candidate": asdict(compiled_candidate),
                "host_contract": {
                    "transaction_signature_must_match": (
                        _policy_transaction_signature(compiled_candidate)
                    ),
                    "exact_rejected_ast_replay_blocked": True,
                    "one_task_changing_call_per_robot": True,
                    "stop_all_required": True,
                },
            }
            try:
                if report.success:
                    role_payload = self.role_pipeline.code_policy(
                        role_context,
                        artifact_id=f"attempt_{policy.attempt + 1:02d}",
                        validator=validate_compiled_role_payload,
                    )
                else:
                    role_payload = self.role_pipeline.repair_code(
                        role_context,
                        artifact_id=f"attempt_{policy.attempt:02d}",
                        validator=validate_compiled_role_payload,
                    )
            except ModelRoleError as exc:
                role_name = "Coder" if report.success else "Code Repair"
                raise ModelAgentError(f"{role_name} role failed: {exc}") from exc
            if role_policy is None:
                raise ModelAgentError(
                    "model role policy passed without producing a validated transaction"
                )
            parsed = role_payload
            repaired = role_policy
        spec = str(parsed.get("spec_candidate", "")).strip()
        write_json(
            self.output_dir / f"model_evolution_decision_{policy.attempt:02d}.json",
            parsed,
        )

        write_json(self.output_dir / "evolved_policy.json", repaired)
        return repaired, spec

    def _evolve_plan_guided(
        self,
        policy: PolicyScript,
        report: EvaluationReport,
        observation: ObservationBundle | None,
        trace: ExecutionTrace | None,
        *,
        controller_decision: EvolutionDecision,
    ) -> tuple[PolicyScript, str]:
        """Generate the next policy from the persistent Orchestrator plan.

        The formal role pipeline deliberately bypasses the legacy
        `_select_evolution_subtask` candidate-enumeration path. Models own the
        plan, current subtask, mutation, and executable scripts. Host code owns
        only lifecycle gates, SDK/schema validation, exact-AST replay
        protection, physics safety, rollback, and completion.
        """

        assert self.role_pipeline is not None
        next_attempt = policy.attempt + 1
        preflight_reorchestration = _preflight_candidate_infeasible(report)
        observation_payload = _observation_payload(
            observation,
            route_discovery=self.route_discovery,
        )
        raw_trusted_runtime = (
            observation.runtime_state
            if observation is not None and isinstance(observation.runtime_state, dict)
            else {}
        )
        current_trusted_runtime = observation_payload.get("runtime_state", {})
        current_trusted_runtime = (
            current_trusted_runtime if isinstance(current_trusted_runtime, dict) else {}
        )
        host_wall_clearance = (
            authored_scene_motion_clearance(current_trusted_runtime)
            if self.route_discovery
            else {
                "available": False,
                "reason": "authored-wall clearance is active only in route-discovery mode",
            }
        )
        turn_clearance_replan_contract = _host_wall_turn_clearance_replan_contract(
            runtime_state=raw_trusted_runtime,
            host_wall_clearance=host_wall_clearance,
        )
        failure_analysis = _failure_analysis_from_report(report)
        if not self.formal_eval:
            failure_analysis = _apply_physics_authoritative_failure_analysis_override(
                failure_analysis,
                policy=policy,
                report=report,
                trace=trace,
            )
        prior_rollout = _evaluation_payload(
            report,
            include_goal_metrics=not self.route_discovery,
        )
        route_recovery_contract = _route_recovery_contract(
            policy,
            report,
            host_wall_clearance=host_wall_clearance,
            expose_clearance_candidates=not self.formal_eval,
            scope_checkpoint_id=_observation_checkpoint_id(observation),
        )
        manager_decision = dict(self._manager_decisions.get(policy.attempt, {}))
        if preflight_reorchestration:
            manager_decision = {}
        elif not manager_decision and not report.success:
            manager_decision = _manager_fallback_from_controller(controller_decision)
        # Route discovery intentionally redacts target coordinates from the
        # model-visible compact state. Lifecycle gates are host-owned physics
        # decisions and must read the unredacted trusted runtime instead.
        lifecycle_probe = {"observation": {"runtime_state": raw_trusted_runtime}}
        required_method, required_example, required_reason = _required_evolution_call(
            lifecycle_probe,
            previous_policy=policy,
            previous_report=report,
        )
        lifecycle_admissibility = _lifecycle_admissibility(
            lifecycle_probe,
            previous_policy=policy,
            previous_report=report,
        )
        hard_required = bool(
            required_method
            and required_method
            == str(lifecycle_admissibility.get("hard_required_method", "")).strip()
        )
        lifecycle_forbidden_methods = _lifecycle_forbidden_methods(
            lifecycle_probe,
            previous_policy=policy,
            previous_report=report,
        )
        _assert_formal_hard_contract_nonempty(
            formal_eval=self.formal_eval,
            allowed_methods=MODEL_PLANNING_SCRIPT_API_METHODS,
            forbidden_methods=lifecycle_forbidden_methods,
            required_method=required_method if hard_required else "",
        )
        current_state_authority = _trusted_state_authority_summary(
            current_trusted_runtime
        )
        parameter_envelope_exhausted = (
            _minimum_effective_parameter_exhaustion(
                policy,
                report,
                trace=trace,
                forbidden_transaction_signatures=set(self._rejected_transactions),
            )
            if not report.success
            else {}
        )
        alignment_sign_repair_contract = (
            _alignment_sign_repair_contract(
                policy,
                report,
                trace,
                forbidden_transaction_signatures=set(self._rejected_transactions),
            )
            if not report.success
            else {}
        )
        if self.formal_eval:
            alignment_sign_repair_contract = {}
        active_sync_repair_gate = _formation_sync_gate_for_planning(
            self._active_formation_sync_repair_gate(),
            lifecycle_hard_required=hard_required,
        )
        formal_sync_diagnostics = (
            dict(active_sync_repair_gate) if self.formal_eval else {}
        )
        if self.formal_eval:
            # Formal evaluation exposes measured coordination failure to the
            # model but never compiles it into a host-selected sync/async repair.
            active_sync_repair_gate = {}
        rejected_host_wall_heading_correction = _rejected_host_wall_heading_correction(
            policy, report
        ) or _load_rejected_host_wall_heading_correction(
            tuple(dict.fromkeys((*self.history_dirs, self.output_dir))),
            checkpoint_id=_observation_checkpoint_id(observation),
            minimum_attempt_index=self._rejected_checkpoint_source_attempt,
            limit=self.rejected_transaction_limit,
        )
        host_wall_heading_correction_contract = (
            {}
            if rejected_host_wall_heading_correction or self.formal_eval
            else _host_wall_heading_correction_contract(
                formation_sync_repair_gate=active_sync_repair_gate,
                host_wall_clearance=host_wall_clearance,
            )
        )
        manager_decision = _manager_decision_for_host_wall_heading_correction(
            manager_decision,
            contract=host_wall_heading_correction_contract,
        )
        active_sync_probe_contract = (
            active_sync_repair_gate.get("probe_contract", {})
            if isinstance(active_sync_repair_gate, dict)
            else {}
        )
        active_sync_probe_contract = (
            active_sync_probe_contract
            if isinstance(active_sync_probe_contract, dict)
            else {}
        )
        if bool(
            active_sync_probe_contract.get(
                "magnitude_escalation_pending",
                False,
            )
        ):
            # The new bounded probe tier is untried. A minimum-distance
            # exhaustion derived from the rejected shorter transaction is
            # stale evidence and must not force a strategy mutation.
            parameter_envelope_exhausted = {}
        raw_host_formation_gate = (
            {}
            if hard_required or parameter_envelope_exhausted
            else _trusted_checkpoint_formation_gate(
                prior_rollout, observation_payload.get("runtime_state", {})
            )
        )
        formation_gate_superseded = (
            not report.success
            and _rejected_alignment_invalidates_formation_gate(
                raw_host_formation_gate,
                policy=policy,
                report=report,
            )
        )
        if formation_gate_superseded:
            self._remember_exhausted_coordination_relation(
                {
                    **raw_host_formation_gate,
                    "status": "superseded_by_physical_rejection",
                    "reason": (
                        "the matching checkpoint-local formation relation was "
                        "physically executed and rejected; it cannot become mandatory "
                        "again at the same checkpoint"
                    ),
                    "source_artifact_dir": str(
                        self.output_dir / f"attempt_{policy.attempt:02d}"
                    ),
                }
            )
        exhausted_relations = tuple(self._exhausted_coordination_relations.values())
        if self.formal_eval:
            exhausted_relations = ()
        host_formation_gate = (
            {}
            if self.formal_eval
            or (
                formation_gate_superseded
                or _coordination_relation_is_checkpoint_exhausted(
                    raw_host_formation_gate,
                    exhausted_relations,
                )
            )
            else raw_host_formation_gate
        )
        deferred_host_formation_gate = (
            host_formation_gate if active_sync_repair_gate else {}
        )
        if active_sync_repair_gate:
            host_formation_gate = {}
        previous_task_plan = _lifecycle_consistent_previous_task_plan(
            self.role_pipeline.current_task_plan,
            forbidden_methods=lifecycle_forbidden_methods,
        )
        previous_task_plan = _retire_checkpoint_exhausted_task_plan_strategy(
            previous_task_plan,
            exhausted_relations=exhausted_relations,
        )
        if formation_gate_superseded:
            previous_task_plan = _retire_superseded_formation_subtask(
                previous_task_plan,
                formation_gate=raw_host_formation_gate,
            )
        route_progress_accepted = transition_is_route_progress(report)
        completed_prefix = 0
        if route_progress_accepted:
            previous_task_plan = (
                _supersede_accepted_current_subtask(
                    previous_task_plan,
                    formation_gate=host_formation_gate,
                )
                if host_formation_gate
                else _retire_accepted_current_subtask_for_replan(
                    previous_task_plan,
                )
            )
        final_approach_transport_envelope = _final_approach_transport_envelope(
            raw_trusted_runtime,
            task_plan=previous_task_plan,
            policy=policy,
            host_wall_clearance=host_wall_clearance,
            forbidden_transaction_signatures=set(self._rejected_transactions),
            formation_sync_repair_gate=active_sync_repair_gate,
        )
        active_sync_repair_gate = _reconcile_formation_sync_gate_with_final_approach(
            active_sync_repair_gate,
            final_approach_transport_envelope=final_approach_transport_envelope,
            runtime_state=raw_trusted_runtime,
        )
        self._persist_reconciled_formation_sync_gate(active_sync_repair_gate)
        if active_sync_repair_gate.get("probe_unavailable", False):
            host_wall_heading_correction_contract = {}
        trusted_final_target_projection = (
            _body_frame_final_target_projection(raw_trusted_runtime)
            if (
                _task_plan_is_final_approach(previous_task_plan)
                or final_approach_transport_envelope.get("active", False)
            )
            else {}
        )
        if trusted_final_target_projection and final_approach_transport_envelope.get(
            "active", False
        ):
            trusted_final_target_projection = {
                **trusted_final_target_projection,
                "target_capture_active": True,
                "target_xy_error_m": final_approach_transport_envelope.get(
                    "target_xy_error_m"
                ),
            }
        planning_context = {
            "formal_v50_lite": self.formal_eval,
            "host_release_preflight_evidence": _release_preflight_planning_context(
                raw_trusted_runtime,
                route_discovery=self.route_discovery,
                lifecycle=lifecycle_admissibility,
                final_approach_envelope=final_approach_transport_envelope,
            ),
            "task_stage": self.task_stage,
            "route_discovery": self.route_discovery,
            "current_trusted_observation": observation_payload,
            "current_state_authority": current_state_authority,
            "host_authored_wall_clearance": host_wall_clearance,
            (
                "advisory_turn_clearance"
                if self.formal_eval
                else "turn_clearance_replan_contract"
            ): turn_clearance_replan_contract,
            "evolution_manager": manager_decision,
            "previous_task_plan": previous_task_plan,
            "retrieved_strategy_memory": (
                list(observation.retrieved_specs) if observation is not None else []
            ),
            "scene_prior": manual_layout_scene_prior(),
            "mission_topology": list(
                self.task.mission_topology
                if self.task is not None
                else MODEL_MISSION_TOPOLOGY
            ),
            "phase_update_rule": (
                "Use only the current trusted post-rollback state and accepted history. "
                "A rejected endpoint never advances global_plan."
            ),
            "host_route_progress_guard": {
                "previous_transition_outcome": transition_outcome_from_report(
                    report
                ).value,
                "route_stall_iterations": controller_decision.route_stall_iterations,
                "setup_does_not_complete_subtask": True,
                "progress_required_to_advance_phase": True,
            },
            "host_lifecycle_gate": {
                **lifecycle_admissibility,
                "hard_required": hard_required,
                "required_method": required_method,
                "required_call": required_example,
                "recommended_method": required_method,
                "reason": required_reason,
                "forbidden_methods": sorted(lifecycle_forbidden_methods),
            },
            "exact_sdk_primitive_names": sorted(MODEL_PLANNING_SCRIPT_API_METHODS),
            "static_skill_sdk": static_skill_library_doc(),
            "coordination_primitive_sdk": coordination_primitive_library_doc(),
        }
        previous_cycle_outcomes = self._generation_cycle_outcomes.get(
            int(policy.attempt),
            [],
        )
        if previous_cycle_outcomes:
            planning_context["previous_generation_cycle_outcomes"] = [
                dict(item) for item in previous_cycle_outcomes
            ]
        host_pivot_role_contract = _host_pivot_role_contract(raw_trusted_runtime)
        if host_pivot_role_contract:
            planning_context[
                "advisory_pivot_role_geometry"
                if self.formal_eval
                else "host_pivot_role_contract"
            ] = host_pivot_role_contract
        if host_wall_heading_correction_contract:
            planning_context["host_wall_heading_correction_contract"] = (
                host_wall_heading_correction_contract
            )
        if rejected_host_wall_heading_correction:
            planning_context["rejected_host_wall_heading_correction"] = (
                rejected_host_wall_heading_correction
            )
        accepted_route_history = _accepted_route_action_history(
            self.accepted_episode_manifest_path
        )
        raw_accepted_straight_saturation_contract = (
            _accepted_straight_saturation_contract(
                accepted_route_history,
                host_wall_clearance=host_wall_clearance,
            )
        )
        accepted_straight_saturation_contract = (
            raw_accepted_straight_saturation_contract
        )
        accepted_linear_continuation_contract = (
            {}
            if accepted_straight_saturation_contract
            else _accepted_linear_continuation_contract(policy, report)
        )
        if accepted_linear_continuation_contract:
            planning_context[
                "advisory_linear_continuation"
                if self.formal_eval
                else "accepted_linear_continuation_contract"
            ] = accepted_linear_continuation_contract
        if trusted_final_target_projection:
            planning_context[
                "advisory_target_projection"
                if self.formal_eval
                else "trusted_body_frame_target_projection"
            ] = trusted_final_target_projection
        if final_approach_transport_envelope:
            planning_context[
                "advisory_final_approach_transport"
                if self.formal_eval
                else "final_approach_transport_envelope"
            ] = final_approach_transport_envelope
        if self.nonrelaxable_motion_objectives and not self.formal_eval:
            planning_context["checkpoint_local_route_exclusions"] = list(
                self.nonrelaxable_motion_objectives
            )
        accepted_route_reversal_guard = _accepted_route_reversal_guard(
            accepted_route_history
        )
        if accepted_route_history:
            planning_context["accepted_route_action_history"] = accepted_route_history
        active_route_guidance: dict[str, Any] = {}
        if self.task is not None:
            route_phase_guard = trusted_route_phase_guard(
                self.task,
                current_trusted_runtime,
                accepted_steps=accepted_route_history.get("steps", ()),
            )
            if route_phase_guard:
                planning_context["host_route_phase_guard"] = route_phase_guard
                completed_prefix = int(
                    route_phase_guard.get("maximum_completed_prefix", 0)
                )
            planning_context["route_spec"] = route_spec_document(self.task)
            active_route_guidance = route_guidance(
                self.task,
                current_trusted_runtime,
                completed_prefix=completed_prefix,
            )
            if active_route_guidance:
                planning_context["active_route_guidance"] = active_route_guidance
        route_clearance_conflict_contract = _route_clearance_conflict_contract(
            accepted_route_history,
            host_wall_clearance=host_wall_clearance,
            active_route_guidance=active_route_guidance,
            runtime_state=raw_trusted_runtime,
            saturation_contract=raw_accepted_straight_saturation_contract,
        )
        if route_clearance_conflict_contract:
            planning_context[
                "advisory_route_clearance"
                if self.formal_eval
                else "route_clearance_conflict_contract"
            ] = route_clearance_conflict_contract
        if accepted_route_reversal_guard:
            planning_context[
                "advisory_route_reversal"
                if self.formal_eval
                else "accepted_route_reversal_guard"
            ] = accepted_route_reversal_guard
        if accepted_straight_saturation_contract:
            planning_context[
                "advisory_route_cadence"
                if self.formal_eval
                else "accepted_straight_saturation_contract"
            ] = accepted_straight_saturation_contract
        if host_formation_gate:
            planning_context["host_formation_gate"] = host_formation_gate
        if self.formal_eval and raw_host_formation_gate:
            planning_context["advisory_formation_repair"] = {
                **raw_host_formation_gate,
                "enforcement": "shadow_only",
                "decision_owner": "task_orchestrator",
            }
        if deferred_host_formation_gate:
            planning_context["deferred_host_formation_gate"] = {
                **deferred_host_formation_gate,
                "deferred_reason": (
                    "The active synchronization contract has priority. Perform its "
                    + (
                        "required geometry-changing repair before reconsidering "
                        "this static formation correction."
                        if active_sync_repair_gate.get("probe_unavailable", False)
                        else "loaded linear probe before reconsidering this static "
                        "formation correction."
                    )
                ),
            }
        if formation_gate_superseded:
            planning_context["superseded_host_formation_gate"] = (
                _superseded_formation_gate_context(raw_host_formation_gate)
            )
        if parameter_envelope_exhausted:
            planning_context["parameter_envelope_exhausted"] = (
                parameter_envelope_exhausted
            )
        if exhausted_relations:
            planning_context["checkpoint_exhausted_coordination_relations"] = list(
                exhausted_relations
            )
        if active_sync_repair_gate:
            planning_context["formation_sync_repair_gate"] = active_sync_repair_gate
        if formal_sync_diagnostics:
            planning_context["measured_formation_sync_diagnostics"] = (
                formal_sync_diagnostics
            )
        if alignment_sign_repair_contract:
            planning_context["alignment_sign_repair_contract"] = (
                alignment_sign_repair_contract
            )
        if route_recovery_contract:
            planning_context["route_recovery_contract"] = route_recovery_contract
        parent_strategy = (
            _policy_mutation_parent_strategy(policy)
            if not route_progress_accepted
            else {}
        )
        mutation_contract = build_mutation_contract(parent_strategy)
        self._apply_infeasible_mutation_levels(
            mutation_contract,
            checkpoint_id=_observation_checkpoint_id(observation),
        )
        apply_minimum_mutation_rank(
            mutation_contract,
            self._minimum_mutation_rank_by_checkpoint.get(
                _observation_checkpoint_id(observation),
                0,
            ),
        )
        _assert_formal_hard_contract_nonempty(
            formal_eval=bool(self.formal_eval and not report.success),
            allowed_methods=MODEL_PLANNING_SCRIPT_API_METHODS,
            forbidden_methods=lifecycle_forbidden_methods,
            mutation_contract=mutation_contract,
        )
        prior_local_navigation = _policy_local_navigation_contract(policy)
        if prior_local_navigation:
            prior_local_navigation = refresh_local_navigation_contract(
                prior_local_navigation,
                raw_trusted_runtime,
            )
        if not report.success and not self.formal_eval:
            _reconcile_mutation_contract_with_local_navigation(
                mutation_contract,
                parent_strategy=parent_strategy,
                local_navigation_contract=prior_local_navigation,
                report=report,
                parameter_envelope_exhausted=parameter_envelope_exhausted,
            )
        parameter_tuning_contract = bool(
            not report.success
            and manager_decision.get("mutation_level") == "parameter_tuning"
        )
        if parameter_tuning_contract:
            planning_context["parameter_tuning_strategy_contract"] = (
                _parameter_free_mutation_strategy(parent_strategy)
            )
        if not report.success and not preflight_reorchestration:
            planning_context["mutation_contract"] = mutation_contract
        if report.success:
            planning_context.update(
                {
                    "previous_subtask_outcome": (
                        "accepted_progress"
                        if route_progress_accepted
                        else "accepted_setup"
                    ),
                    "last_accepted_policy": _evolution_policy_payload(policy),
                    "last_accepted_rollout": prior_rollout,
                    "accepted_subtask_transition_rule": (
                        "The prior bounded action passed the physical checkpoint gate. "
                        "Read last_accepted_rollout.evidence.checkpoint_outcome: an "
                        "accepted restorable state does not by itself prove route "
                        "correctness or route-phase completion. ACCEPT_SETUP keeps the "
                        "same active subtask and route reference; only ACCEPT_PROGRESS "
                        "or TASK_COMPLETE permits route-phase advancement."
                    ),
                }
            )
        elif preflight_reorchestration:
            planning_context.update(
                {
                    "previous_subtask_outcome": "preflight_candidate_infeasible",
                    "last_rejected_policy": _evolution_policy_payload(policy),
                    "last_rejected_rollout": prior_rollout,
                    "preflight_candidate_rejection": {
                        "failure_type": report.failure_type,
                        "physically_dispatched": False,
                        "blocked_exact_candidate": True,
                        "note": (
                            "This candidate was never physically dispatched. Treat it "
                            "as checkpoint-local candidate rejection, not a physical "
                            "strategy, coordination, or execution failure. Select a "
                            "different bounded subtask. The Host does not choose the "
                            "replacement primitive, direction, or roles."
                        ),
                    },
                }
            )
        else:
            planning_context.update(
                {
                    "previous_subtask_outcome": "rejected",
                    "last_rejected_policy": _evolution_policy_payload(policy),
                    "last_rejected_rollout": prior_rollout,
                    "last_rejected_failure_analysis": failure_analysis,
                }
            )
        active_rewind_guidance = _active_branch_rewind_context(
            {"branch_rewind_context": self.branch_rewind_context or {}},
            observation_payload.get("runtime_state", {}),
        )
        if active_rewind_guidance:
            planning_context["checkpoint_local_operator_guidance"] = {
                "scope_checkpoint_id": active_rewind_guidance.get(
                    "scope_checkpoint_id", ""
                ),
                "lesson": active_rewind_guidance.get("lesson", ""),
                "expires": active_rewind_guidance.get(
                    "persistence_mode", "until_next_accepted_checkpoint"
                ),
            }

        # Reject recovery has exactly one executable-strategy writer.
        # Mutation Operator produces and Host-validates the child before Task
        # Orchestrator is allowed to expand it into a subtask plan.
        physical_reject_evolution = bool(
            not report.success
            and not preflight_reorchestration
            and not hard_required
        )
        mutation: dict[str, Any] = {}
        locked_execution_strategy: dict[str, Any] = {}
        mutation_base_plan: dict[str, Any] = {}

        if physical_reject_evolution:
            mutation_base_plan = _task_plan_with_execution_strategy(
                previous_task_plan,
                parent_strategy,
            )
            mutation_lock_navigation = _mutation_lock_local_navigation_contract(
                mutation_base_plan,
                raw_trusted_runtime,
            )

            def validate_single_writer_mutation(candidate: dict[str, Any]) -> None:
                child_strategy = candidate.get("child_strategy", {})
                child_strategy = (
                    child_strategy if isinstance(child_strategy, dict) else {}
                )
                if not self.formal_eval:
                    _apply_host_pivot_roles_to_strategy(
                        child_strategy,
                        runtime_state=raw_trusted_runtime,
                        audit_container=candidate,
                    )
                candidate_plan = _task_plan_with_execution_strategy(
                    mutation_base_plan,
                    child_strategy,
                )
                locked_route = validate_locked_route_reference_unchanged(
                    mutation_base_plan,
                    candidate_plan,
                )
                if (
                    not self.formal_eval
                    and locked_route.get("active", False)
                    and not locked_route.get("valid", False)
                ):
                    raise ModelRoleError(
                        str(locked_route.get("reason", "route lock changed"))
                    )
                candidate_final_approach_envelope = (
                    _final_approach_transport_envelope(
                        raw_trusted_runtime,
                        task_plan=candidate_plan,
                        policy=policy,
                        execution_strategy=child_strategy,
                        host_wall_clearance=host_wall_clearance,
                        forbidden_transaction_signatures=set(
                            self._rejected_transactions
                        ),
                        formation_sync_repair_gate=active_sync_repair_gate,
                    )
                )
                _validate_plan_guided_mutation(
                    candidate,
                    manager_decision=manager_decision,
                    parent_strategy=parent_strategy,
                    allowed_sdk_methods=MODEL_PLANNING_SCRIPT_API_METHODS,
                    runtime_state=raw_trusted_runtime,
                    formation_gate=host_formation_gate,
                    exhausted_relations=exhausted_relations,
                    host_wall_clearance=host_wall_clearance,
                    route_clearance_conflict_contract=(
                        route_clearance_conflict_contract
                    ),
                    turn_clearance_replan_contract=(
                        turn_clearance_replan_contract
                    ),
                    nonrelaxable_motion_objectives=(
                        () if self.formal_eval else self.nonrelaxable_motion_objectives
                    ),
                    trusted_body_frame_target_projection=(
                        trusted_final_target_projection
                    ),
                    final_approach_transport_envelope=(
                        candidate_final_approach_envelope
                    ),
                    formation_sync_repair_gate=active_sync_repair_gate,
                    host_wall_heading_correction_contract=(
                        host_wall_heading_correction_contract
                    ),
                    rejected_host_wall_heading_correction=(
                        rejected_host_wall_heading_correction
                    ),
                    forbidden_transaction_signatures=set(
                        self._rejected_transactions
                    ),
                    local_navigation_contract=mutation_lock_navigation,
                    route_recovery_contract=route_recovery_contract,
                    model_owns_route=self.formal_eval,
                    model_owned_setup=False,
                    mutation_contract=mutation_contract,
                )
                if formation_gate_superseded and not self.formal_eval:
                    _validate_mutation_does_not_repeat_superseded_formation_gate(
                        candidate,
                        formation_gate=raw_host_formation_gate,
                    )

            try:
                mutation = self.role_pipeline.mutate_strategy(
                    {
                        "formal_v50_lite": self.formal_eval,
                        "task_orchestrator_plan": mutation_base_plan,
                        "current_subtask": mutation_base_plan.get(
                            "current_subtask", {}
                        ),
                        "parent_policy": {
                            "script_id": policy.script_id,
                            "attempt": policy.attempt,
                            "strategy": parent_strategy,
                        },
                        "current_trusted_observation": observation_payload,
                        "current_state_authority": current_state_authority,
                        "checkpoint_local_operator_guidance": planning_context.get(
                            "checkpoint_local_operator_guidance", {}
                        ),
                        "host_authored_wall_clearance": host_wall_clearance,
                        (
                            "advisory_turn_clearance"
                            if self.formal_eval
                            else "turn_clearance_replan_contract"
                        ): turn_clearance_replan_contract,
                        (
                            "advisory_route_clearance"
                            if self.formal_eval
                            else "route_clearance_conflict_contract"
                        ): route_clearance_conflict_contract,
                        (
                            "advisory_pivot_role_geometry"
                            if self.formal_eval
                            else "host_pivot_role_contract"
                        ): host_pivot_role_contract,
                        (
                            "advisory_target_projection"
                            if self.formal_eval
                            else "trusted_body_frame_target_projection"
                        ): trusted_final_target_projection,
                        (
                            "advisory_route_recovery"
                            if self.formal_eval
                            else "route_recovery_contract"
                        ): route_recovery_contract,
                        (
                            "advisory_formation_repair"
                            if self.formal_eval
                            else "host_formation_gate"
                        ): (
                            raw_host_formation_gate
                            if self.formal_eval
                            else host_formation_gate
                        ),
                        "formation_sync_repair_gate": active_sync_repair_gate,
                        "local_navigation_contract": mutation_lock_navigation,
                        "rejected_host_wall_heading_correction": (
                            rejected_host_wall_heading_correction
                        ),
                        "superseded_host_formation_gate": planning_context.get(
                            "superseded_host_formation_gate", {}
                        ),
                        "rejected_endpoint_policy": _evolution_policy_payload(policy),
                        "rejected_endpoint_rollout": prior_rollout,
                        "failure_analysis": failure_analysis,
                        "evolution_manager": manager_decision,
                        "mutation_contract": mutation_contract,
                        "checkpoint_exhausted_coordination_relations": list(
                            exhausted_relations
                        ),
                        "checkpoint_local_route_exclusions": list(
                            ()
                            if self.formal_eval
                            else self.nonrelaxable_motion_objectives
                        ),
                        "static_skill_sdk": static_skill_library_doc(),
                        "coordination_primitive_sdk": (
                            coordination_primitive_library_doc()
                        ),
                        "rejected_transaction_signatures": list(
                            self._blocked_transaction_context_entries()
                        ),
                        "instruction": (
                            "You are the sole writer of executable child strategy "
                            "fields for this rejected physical transaction. Produce "
                            "one child at exactly Evolution Manager's selected level "
                            "from trusted checkpoint evidence. No Task Orchestrator "
                            "strategy proposal exists yet. The Host selects no "
                            "replacement primitive, direction, mode, role, or literal."
                        ),
                    },
                    artifact_id=f"attempt_{next_attempt:02d}",
                    validator=validate_single_writer_mutation,
                )
            except ModelRoleError as exc:
                raise ModelAgentError(
                    f"Strategy Mutation role failed: {exc}"
                ) from exc
            locked_execution_strategy = json.loads(
                json.dumps(
                    mutation["child_strategy"],
                    ensure_ascii=False,
                    default=str,
                )
            )
            write_json(
                self.role_pipeline.output_dir
                / f"locked_mutation_child_attempt_{next_attempt:02d}.json",
                {
                    "schema_version": 1,
                    "source": "host_validated_mutation_operator_child",
                    "immutable_executable_fields_downstream": True,
                    "mutation_level": mutation.get("mutation_level", ""),
                    "child_strategy": locked_execution_strategy,
                    "host_effective_delta": mutation.get(
                        "host_effective_delta", {}
                    ),
                    "host_lock_navigation_contract": mutation_lock_navigation,
                },
            )
            planning_context.update(
                {
                    "approved_strategy_mutation": mutation,
                    "locked_child_strategy": locked_execution_strategy,
                    "single_writer_contract": {
                        "writer": "mutation_operator",
                        "locked_executable_fields": [
                            "primitive",
                            "semantic_objective",
                            "coordination_mode",
                            "temporary_front_back_roles",
                        ],
                        "orchestrator_may_change_executable_fields": False,
                    },
                    "instruction": (
                        "Expand locked_child_strategy into the complete current "
                        "subtask and persistent plan. Preserve its primitive, semantic "
                        "objective, coordination mode, and temporary front/back roles "
                        "exactly. You may write plan phases, rationale, role objectives, "
                        "synchronization description, guidance, and termination text."
                    ),
                }
            )

        def validate_task_plan(candidate: dict[str, Any]) -> None:
            if not self.formal_eval:
                _apply_host_pivot_roles_to_task_plan(
                    candidate,
                    runtime_state=raw_trusted_runtime,
                )
                _normalize_route_clearance_reentry_navigation(
                    candidate,
                    contract=route_clearance_conflict_contract,
                )
            if self.task is not None:
                route_validation = validate_task_plan_route_reference(
                    self.task,
                    raw_trusted_runtime,
                    completed_prefix=completed_prefix,
                    task_plan=candidate,
                )
                if (
                    not self.formal_eval
                    and route_validation.get("active", False)
                    and not route_validation.get("valid", False)
                ):
                    raise ModelRoleError(
                        "current_subtask.navigation.world_direction_xy contradicts "
                        "the active connected route segment: "
                        f"{route_validation.get('reason', 'invalid route reference')}"
                    )
            locked_route = validate_locked_route_reference_unchanged(
                previous_task_plan,
                candidate,
            )
            if (
                not self.formal_eval
                and locked_route.get("active", False)
                and not locked_route.get("valid", False)
            ):
                raise ModelRoleError(
                    str(locked_route.get("reason", "route lock changed"))
                )
            candidate_strategy = _execution_strategy_from_task_plan(candidate)
            if locked_execution_strategy:
                _validate_orchestrator_preserves_locked_child(
                    candidate,
                    locked_child=locked_execution_strategy,
                )
            turn_clearance_replan = bool(
                _host_wall_turn_episode_replan_kind(
                    method=str(candidate_strategy.get("primitive", "")),
                    objective=str(candidate_strategy.get("semantic_objective", "")),
                    runtime_state=raw_trusted_runtime,
                    host_wall_clearance=host_wall_clearance,
                )
                == "translation"
            )
            candidate_local_navigation = build_local_navigation_contract(
                candidate,
                raw_trusted_runtime,
            )
            if not self.formal_eval:
                try:
                    validate_strategy_against_local_navigation(
                        method=str(candidate_strategy.get("primitive", "")),
                        objective=str(
                            candidate_strategy.get("semantic_objective", "")
                        ),
                        contract=candidate_local_navigation,
                        allow_formation_repair=bool(
                            host_formation_gate or active_sync_repair_gate
                        ),
                        allow_route_recovery=bool(
                            route_recovery_contract.get("active", False)
                        ),
                        allow_turn_clearance_replan=turn_clearance_replan,
                        route_clearance_conflict_contract=(
                            route_clearance_conflict_contract
                        ),
                    )
                except ValueError as exc:
                    raise ModelRoleError(str(exc)) from exc
            candidate_final_approach_envelope = (
                _final_approach_transport_envelope(
                    raw_trusted_runtime,
                    task_plan=candidate,
                    policy=policy,
                    execution_strategy=candidate_strategy,
                    host_wall_clearance=host_wall_clearance,
                    forbidden_transaction_signatures=set(
                        self._rejected_transactions
                    ),
                    formation_sync_repair_gate=active_sync_repair_gate,
                )
            )
            if parameter_tuning_contract:
                _validate_parameter_tuning_task_plan(
                    candidate,
                    parent_strategy=parent_strategy,
                )
            elif (
                not report.success
                and manager_decision.get("mutation_level")
                == "coordination_mutation"
            ):
                _validate_coordination_mutation_task_plan(
                    candidate,
                    parent_strategy=parent_strategy,
                )
            elif (
                not report.success
                and manager_decision.get("mutation_level") == "strategy_mutation"
                and not turn_clearance_replan
                and not self.formal_eval
            ):
                _validate_strategy_mutation_linear_feasibility(
                    parent_strategy=parent_strategy,
                    method=str(candidate_strategy.get("primitive", "")).strip(),
                    objective=str(
                        candidate_strategy.get("semantic_objective", "")
                    ).strip(),
                    host_wall_clearance=host_wall_clearance,
                    trusted_body_frame_target_projection=(
                        trusted_final_target_projection
                    ),
                )
            # Return the checkpoint-local repair contract before broader
            # availability feedback. Otherwise a model can keep treating an
            # older accepted pivot as if it happened after the probe failure.
            if not self.formal_eval:
                _validate_task_plan_against_formation_sync_repair_gate(
                    candidate,
                    gate=active_sync_repair_gate,
                    runtime_state=raw_trusted_runtime,
                    host_wall_clearance=host_wall_clearance,
                    rejected_host_wall_heading_correction=(
                        rejected_host_wall_heading_correction
                    ),
                )
            _validate_orchestrator_plan_strategy(
                candidate,
                allowed_sdk_methods=MODEL_PLANNING_SCRIPT_API_METHODS,
                hard_required_method=required_method if hard_required else "",
                forbidden_methods=lifecycle_forbidden_methods,
                runtime_state=raw_trusted_runtime,
                formation_gate=host_formation_gate,
                exhausted_relations=exhausted_relations,
                host_wall_clearance=host_wall_clearance,
                nonrelaxable_motion_objectives=(
                    () if self.formal_eval else self.nonrelaxable_motion_objectives
                ),
                trusted_body_frame_target_projection=(trusted_final_target_projection),
                accepted_linear_continuation_contract=(
                    accepted_linear_continuation_contract
                ),
                accepted_route_reversal_guard=accepted_route_reversal_guard,
                accepted_straight_saturation_contract=(
                    accepted_straight_saturation_contract
                ),
                final_approach_transport_envelope=(
                    candidate_final_approach_envelope
                ),
                host_wall_heading_correction_contract=(
                    host_wall_heading_correction_contract
                ),
                route_clearance_conflict_contract=(
                    route_clearance_conflict_contract
                ),
                turn_clearance_replan_contract=(
                    turn_clearance_replan_contract
                ),
                route_recovery_contract=route_recovery_contract,
                final_target_navigation_authorized=(
                    completed_prefix >= len(self.task.route_progress_gates)
                    if self.task is not None
                    else None
                ),
                model_owns_route=self.formal_eval,
            )
            if formation_gate_superseded:
                _validate_plan_does_not_repeat_superseded_formation_gate(
                    candidate,
                    formation_gate=raw_host_formation_gate,
                )
            if (
                locked_execution_strategy
                and manager_decision.get("mutation_level") == "strategy_mutation"
            ):
                semantic_strategy = {
                    **locked_execution_strategy,
                    "formation_strategy": str(
                        candidate_strategy.get("formation_strategy", "")
                    ).strip(),
                    "synchronization_intent": str(
                        candidate_strategy.get("synchronization_intent", "")
                    ).strip(),
                }
                _validate_strategy_mutation_orchestrator_replan(
                    candidate,
                    previous_task_plan=previous_task_plan,
                    approved_strategy=semantic_strategy,
                )

        task_orchestrator_replan_delegated = False

        def normalize_task_plan(candidate: dict[str, Any]) -> None:
            if (
                locked_execution_strategy
                and manager_decision.get("mutation_level") == "strategy_mutation"
            ):
                _inherit_strategy_replan_navigation_contract(
                    candidate,
                    parent_plan=previous_task_plan,
                )

        try:
            task_plan = self.role_pipeline.plan_task(
                planning_context,
                artifact_id=f"attempt_{next_attempt:02d}",
                image_path=_latest_observation_frame(observation),
                normalizer=(
                    normalize_task_plan if locked_execution_strategy else None
                ),
                validator=validate_task_plan,
            )
        except ModelRoleError as exc:
            if (
                exhausted_relations
                and not hard_required
                and not parameter_tuning_contract
                and not locked_execution_strategy
                and "checkpoint-local exhausted coordination relation" in str(exc)
            ):
                task_plan = _task_plan_with_execution_strategy(
                    previous_task_plan,
                    parent_strategy,
                )
                current_subtask = task_plan.get("current_subtask", {})
                current_subtask = (
                    current_subtask if isinstance(current_subtask, dict) else {}
                )
                current_subtask["checkpoint_local_replan_required"] = True
                current_subtask["guidance"] = (
                    "Task Orchestrator could not replace the exact exhausted relation. "
                    "Strategy Mutation must choose a materially different executable "
                    "strategy from current trusted physics and global visual evidence."
                )
                task_plan["current_subtask"] = current_subtask
                task_orchestrator_replan_delegated = True
                write_json(
                    self.role_pipeline.output_dir
                    / f"task_orchestrator_replan_delegation_attempt_{next_attempt:02d}.json",
                    {
                        "reason": str(exc),
                        "global_plan_preserved": task_plan.get("global_plan", []),
                        "current_phase_id": task_plan.get("current_phase_id", ""),
                        "rejected_parent_strategy": parent_strategy,
                        "checkpoint_exhausted_coordination_relations": list(
                            exhausted_relations
                        ),
                        "delegated_role": "mutation_operator",
                        "contract": (
                            "The host selected no replacement action. Strategy Mutation "
                            "must propose one and pass the existing structured validators."
                        ),
                    },
                )
            else:
                raise ModelAgentError(f"Task Orchestrator role failed: {exc}") from exc

        if self.formal_eval:
            shadow_strategy = _execution_strategy_from_task_plan(task_plan)
            shadow_route = (
                validate_task_plan_route_reference(
                    self.task,
                    raw_trusted_runtime,
                    completed_prefix=completed_prefix,
                    task_plan=task_plan,
                )
                if self.task is not None
                else {}
            )
            shadow_locked_route = validate_locked_route_reference_unchanged(
                previous_task_plan,
                task_plan,
            )
            shadow_local_navigation = build_local_navigation_contract(
                task_plan,
                raw_trusted_runtime,
            )
            _write_formal_shadow_advisory(
                self.role_pipeline.output_dir,
                attempt=next_attempt,
                checkpoint_id=_observation_checkpoint_id(observation),
                stage="task_orchestrator_replan",
                strategy=shadow_strategy,
                checks=[
                    {
                        "name": "active_route_reference",
                        "legacy_would_reject": bool(
                            shadow_route.get("active", False)
                            and not shadow_route.get("valid", False)
                        ),
                        "reason": str(shadow_route.get("reason", "")),
                    },
                    {
                        "name": "parent_route_lock",
                        "legacy_would_reject": bool(
                            shadow_locked_route.get("active", False)
                            and not shadow_locked_route.get("valid", False)
                        ),
                        "reason": str(shadow_locked_route.get("reason", "")),
                    },
                    _formal_shadow_check(
                        "local_navigation",
                        lambda: validate_strategy_against_local_navigation(
                            method=str(shadow_strategy.get("primitive", "")),
                            objective=str(
                                shadow_strategy.get("semantic_objective", "")
                            ),
                            contract=shadow_local_navigation,
                            allow_formation_repair=bool(raw_host_formation_gate),
                            allow_route_recovery=bool(
                                route_recovery_contract.get("active", False)
                            ),
                            route_clearance_conflict_contract=(
                                route_clearance_conflict_contract
                            ),
                        ),
                    ),
                    _formal_shadow_check(
                        "legacy_orchestrator_action_filters",
                        lambda: _validate_orchestrator_plan_strategy(
                            task_plan,
                            allowed_sdk_methods=MODEL_PLANNING_SCRIPT_API_METHODS,
                            hard_required_method=(
                                required_method if hard_required else ""
                            ),
                            forbidden_methods=lifecycle_forbidden_methods,
                            runtime_state=raw_trusted_runtime,
                            formation_gate=raw_host_formation_gate,
                            host_wall_clearance=host_wall_clearance,
                            nonrelaxable_motion_objectives=(
                                self.nonrelaxable_motion_objectives
                            ),
                            trusted_body_frame_target_projection=(
                                trusted_final_target_projection
                            ),
                            accepted_linear_continuation_contract=(
                                accepted_linear_continuation_contract
                            ),
                            accepted_route_reversal_guard=(
                                accepted_route_reversal_guard
                            ),
                            accepted_straight_saturation_contract=(
                                accepted_straight_saturation_contract
                            ),
                            final_approach_transport_envelope=(
                                final_approach_transport_envelope
                            ),
                            route_clearance_conflict_contract=(
                                route_clearance_conflict_contract
                            ),
                            turn_clearance_replan_contract=(
                                turn_clearance_replan_contract
                            ),
                            model_owns_route=False,
                        ),
                    ),
                ],
            )

        generation_payload = {
            "formal_v50_lite": self.formal_eval,
            "host_release_preflight_evidence": planning_context.get(
                "host_release_preflight_evidence", {}
            ),
            "task_stage": self.task_stage,
            "route_discovery": self.route_discovery,
            "current_policy": _evolution_policy_payload(policy),
            "evaluation": _evaluation_payload(
                report,
                include_goal_metrics=not self.route_discovery,
            ),
            "execution_trace": _trace_payload(trace),
            "observation": observation_payload,
            "task_orchestrator_plan": task_plan,
            "failure_analysis": failure_analysis,
            "evolution_manager": manager_decision,
            "host_route_progress_guard": planning_context[
                "host_route_progress_guard"
            ],
            "mutation_contract": mutation_contract,
            "scene_prior": manual_layout_scene_prior(),
            "static_skill_sdk": static_skill_library_doc(),
            "coordination_primitive_sdk": coordination_primitive_library_doc(),
            "host_authored_wall_clearance": host_wall_clearance,
            (
                "advisory_turn_clearance"
                if self.formal_eval
                else "turn_clearance_replan_contract"
            ): turn_clearance_replan_contract,
            (
                "advisory_route_clearance"
                if self.formal_eval
                else "route_clearance_conflict_contract"
            ): route_clearance_conflict_contract,
            "recent_rejected_team_transactions": list(
                self._blocked_transaction_context_entries()
            ),
            "recent_rejected_executable_transactions": list(
                self._blocked_transaction_context_entries()
            ),
        }
        local_navigation = build_local_navigation_contract(
            task_plan,
            raw_trusted_runtime,
        )
        if local_navigation:
            planning_context["local_navigation_contract"] = local_navigation
            generation_payload["local_navigation_contract"] = local_navigation
        if host_pivot_role_contract:
            generation_payload[
                "advisory_pivot_role_geometry"
                if self.formal_eval
                else "host_pivot_role_contract"
            ] = host_pivot_role_contract
        if self.formal_eval and raw_host_formation_gate:
            generation_payload["advisory_formation_repair"] = {
                **raw_host_formation_gate,
                "enforcement": "shadow_only",
            }
        if final_approach_transport_envelope:
            generation_payload[
                "advisory_final_approach_transport"
                if self.formal_eval
                else "final_approach_transport_envelope"
            ] = final_approach_transport_envelope
        if host_formation_gate:
            generation_payload["host_formation_gate"] = host_formation_gate
        if deferred_host_formation_gate:
            generation_payload["deferred_host_formation_gate"] = planning_context[
                "deferred_host_formation_gate"
            ]
        if formation_gate_superseded:
            generation_payload["superseded_host_formation_gate"] = (
                _superseded_formation_gate_context(raw_host_formation_gate)
            )
        if parameter_envelope_exhausted:
            generation_payload["parameter_envelope_exhausted"] = (
                parameter_envelope_exhausted
            )
        if exhausted_relations:
            generation_payload["checkpoint_exhausted_coordination_relations"] = list(
                exhausted_relations
            )
        if active_sync_repair_gate:
            generation_payload["formation_sync_repair_gate"] = active_sync_repair_gate
        if host_wall_heading_correction_contract:
            generation_payload["host_wall_heading_correction_contract"] = (
                host_wall_heading_correction_contract
            )
        if rejected_host_wall_heading_correction:
            generation_payload["rejected_host_wall_heading_correction"] = (
                rejected_host_wall_heading_correction
            )
        if alignment_sign_repair_contract:
            generation_payload["alignment_sign_repair_contract"] = (
                alignment_sign_repair_contract
            )
        if route_recovery_contract:
            generation_payload["route_recovery_contract"] = route_recovery_contract
        if self.nonrelaxable_motion_objectives and not self.formal_eval:
            generation_payload["checkpoint_local_route_exclusions"] = list(
                self.nonrelaxable_motion_objectives
            )
        if parameter_tuning_contract:
            turn_repair_constraint = _turn_execution_repair_constraint(
                policy,
                repair_level=controller_decision.repair_level,
                failure_type=report.failure_type,
            )
            if turn_repair_constraint:
                generation_payload["turn_execution_repair_contract"] = (
                    turn_repair_constraint
                )
            linear_repair_constraint = _linear_execution_repair_constraint(
                policy,
                repair_level=controller_decision.repair_level,
                failure_type=report.failure_type,
                selected_subtask={
                    "selected_method": str(
                        parent_strategy.get("primitive", "")
                    ).strip(),
                    "motion_objective": str(
                        parent_strategy.get("semantic_objective", "")
                    ).strip(),
                },
                forbidden_transaction_signatures=set(self._rejected_transactions),
            )
            if linear_repair_constraint:
                generation_payload["linear_execution_repair_contract"] = (
                    linear_repair_constraint
                )
        if hard_required:
            execution_strategy = _execution_strategy_from_task_plan(
                task_plan,
                required_method=required_method,
                required_reason=required_reason,
            )
            strategy_source = "host_lifecycle_gate"
        elif report.success or preflight_reorchestration:
            execution_strategy = _execution_strategy_from_task_plan(task_plan)
            strategy_source = (
                "task_orchestrator_preflight_reorchestration"
                if preflight_reorchestration
                else "task_orchestrator"
            )
        elif physical_reject_evolution:
            if not mutation or not locked_execution_strategy:
                raise ModelAgentError(
                    "single-writer recovery reached planning without a Host-locked "
                    "Mutation Operator child"
                )
            execution_strategy = dict(locked_execution_strategy)
            strategy_source = "host_locked_mutation_operator_child"

        effective_plan = _task_plan_with_execution_strategy(
            task_plan,
            execution_strategy,
        )
        local_navigation = build_local_navigation_contract(
            effective_plan,
            raw_trusted_runtime,
        )
        final_approach_transport_envelope = _final_approach_transport_envelope(
            raw_trusted_runtime,
            task_plan=effective_plan,
            policy=policy,
            execution_strategy=execution_strategy,
            host_wall_clearance=host_wall_clearance,
            forbidden_transaction_signatures=set(self._rejected_transactions),
            formation_sync_repair_gate=active_sync_repair_gate,
        )
        host_wall_turn_angle_envelope = _host_wall_turn_angle_envelope(
            str(execution_strategy.get("primitive", "")).strip(),
            str(execution_strategy.get("semantic_objective", "")).strip(),
            host_wall_clearance,
            route_clearance_conflict_contract=route_clearance_conflict_contract,
        )
        write_json(
            self.role_pipeline.output_dir
            / f"subtask_assignment_attempt_{next_attempt:02d}.json",
            {
                "global_plan": effective_plan.get("global_plan", []),
                "current_phase_id": effective_plan.get("current_phase_id", ""),
                "current_subtask": effective_plan.get("current_subtask", {}),
                "execution_strategy": execution_strategy,
                "strategy_source": strategy_source,
                "task_orchestrator_replan_delegated": (
                    task_orchestrator_replan_delegated
                ),
                "evolution_manager": manager_decision,
                "mutation": mutation,
            },
        )
        parsed, repaired = _generate_plan_guided_team_policy(
            role_pipeline=self.role_pipeline,
            task_plan=effective_plan,
            execution_strategy=execution_strategy,
            generation_context={
                **generation_payload,
                "task_orchestrator_plan": effective_plan,
                "approved_strategy_mutation": mutation,
                "host_lifecycle_gate": {
                    **lifecycle_admissibility,
                    "hard_required": hard_required,
                    "required_method": required_method,
                    "required_call": required_example,
                    "recommended_method": required_method,
                    "reason": required_reason,
                },
                "allowed_sdk_methods": sorted(MODEL_PLANNING_SCRIPT_API_METHODS),
                "verified_action_memory": (
                    _plan_guided_action_exemplars(
                        self.action_memory,
                        execution_strategy=execution_strategy,
                    )
                ),
                (
                    "advisory_final_approach_transport"
                    if self.formal_eval
                    else "final_approach_transport_envelope"
                ): final_approach_transport_envelope,
                (
                    "advisory_wall_turn_angle_envelope"
                    if self.formal_eval
                    else "host_wall_turn_angle_envelope"
                ): host_wall_turn_angle_envelope,
                "local_navigation_contract": local_navigation,
            },
            attempt=next_attempt,
            fallback_id=f"box_carry_attempt_{next_attempt:02d}",
            parent_script_id=policy.script_id,
            require_team_plan=self.require_team_plan,
            expected_method=required_method if hard_required else "",
            forbidden_transaction_signatures=tuple(self._rejected_transactions),
            current_policy=policy,
            # A branch rewind may carry a strategy-level controller label,
            # while the supplied parent report is verifier-accepted. That is
            # replanning from a trusted state, not repair of a physical failure.
            # The Orchestrator owns the new commitment; Coder must not be told
            # both to preserve it and to change its method/direction/mode.
            repair_level=(
                ""
                if report.success or hard_required or preflight_reorchestration
                else controller_decision.repair_level
            ),
            repair_failure_type=(
                ""
                if report.success or hard_required or preflight_reorchestration
                else report.failure_type
            ),
        )
        if mutation:
            mutation_delta = mutation.get("host_effective_delta", {})
            write_json(
                self.role_pipeline.output_dir
                / f"mutation_effective_delta_attempt_{next_attempt:02d}.json",
                mutation_delta,
            )
            self.role_pipeline.publish_high_level_status(
                "executable_mutation_validated",
                attempt=next_attempt,
                checkpoint_id=_observation_checkpoint_id(observation),
                task_plan=effective_plan,
                policy=repaired,
                active_model_role="host_validator",
                next_system_operation="run physical preflight and dispatch",
                details={"host_effective_delta": mutation_delta},
            )
        spec = str(parsed.get("spec_candidate", "")).strip()
        write_json(
            self.output_dir / f"model_evolution_decision_{policy.attempt:02d}.json",
            {
                "decision_path": "plan_guided_model_roles",
                "task_plan": effective_plan,
                "evolution_manager": manager_decision,
                "mutation": mutation,
                "policy": parsed,
            },
        )
        write_json(self.output_dir / "evolved_policy.json", repaired)
        return repaired, spec

    def _remember_rejected_transaction(
        self,
        policy: PolicyScript,
        *,
        failure_type: str = "",
    ) -> None:
        signature = _policy_transaction_signature(policy)
        if not signature:
            return
        self._rejected_transactions[signature] = _policy_transaction_for_prompt(policy)
        self._rejected_transaction_summaries[signature] = (
            _compact_blocked_transaction_signature(
                policy,
                failure_type=failure_type,
            )
        )

    def _blocked_transaction_context_entries(self) -> list[str]:
        """Readable checkpoint-local blockers; exact AST identity remains hashed internally."""

        entries: list[str] = []
        for signature in self._rejected_transactions:
            summary = str(
                self._rejected_transaction_summaries.get(signature, "")
            ).strip()
            entries.append(summary or _blocked_transaction_fingerprint(signature))
        return entries

    def _remember_candidate_space_exhaustion(
        self,
        selected_subtask: dict[str, str],
    ) -> None:
        """Exclude a model-only dead end before recursively selecting another candidate."""

        if str(selected_subtask.get("selection_source", "")).strip() in {
            HOST_WALL_CLEARANCE_ESCAPE_SELECTION_SOURCE,
            HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE,
        }:
            # The semantic action space came from current authored-wall geometry.
            # A pre-dispatch generation failure is not evidence that the physical
            # escape direction is invalid.
            return
        method = str(selected_subtask.get("selected_method", "")).strip()
        objective = str(selected_subtask.get("motion_objective", "")).strip()
        mode = str(selected_subtask.get("coordination_mode", "")).strip()
        if method and objective:
            self._blocked_motion_objectives.add(f"{method}:{objective}")
        elif method and mode:
            self._blocked_strategy_signatures.add(f"{mode}:{method}")

    def _remember_exhausted_coordination_relation(
        self,
        relation: dict[str, Any],
    ) -> None:
        signature = _exhausted_coordination_relation_signature(relation)
        if signature:
            normalized = dict(relation)
            self._exhausted_coordination_relations[signature] = normalized
            if str(normalized.get("scope_checkpoint_id", "")).strip():
                _persist_checkpoint_exhausted_coordination_relation(
                    self.output_dir / CHECKPOINT_EXHAUSTED_COORDINATION_RELATIONS_FILE,
                    normalized,
                )

    def _refresh_rejected_transactions(
        self,
        *,
        checkpoint_id: str,
        runtime_state: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_source_attempt = _authoritative_checkpoint_source_attempt(
            self.accepted_episode_manifest_path,
            checkpoint_id=checkpoint_id,
        )
        if (
            checkpoint_id != self._rejected_checkpoint_id
            or checkpoint_source_attempt != self._rejected_checkpoint_source_attempt
        ):
            self._rejected_transactions.clear()
            self._rejected_transaction_summaries.clear()
            self._directionally_rejected_methods.clear()
            self._directionally_rejected_objectives.clear()
            self._nonrelaxable_motion_objectives.clear()
            self._strategy_failure_counts.clear()
            self._strategy_failure_keys.clear()
            self._blocked_strategy_signatures.clear()
            self._blocked_motion_objectives.clear()
            self._exhausted_coordination_relations.clear()
            self._rejected_checkpoint_id = checkpoint_id
            self._rejected_checkpoint_source_attempt = checkpoint_source_attempt
        rewind_runtime_state = dict(runtime_state or {})
        rewind_checkpoint = rewind_runtime_state.get("checkpoint", {})
        rewind_checkpoint = (
            dict(rewind_checkpoint) if isinstance(rewind_checkpoint, dict) else {}
        )
        rewind_checkpoint.setdefault("checkpoint_id", checkpoint_id)
        rewind_runtime_state["checkpoint"] = rewind_checkpoint
        active_branch_rewind = bool(
            self.branch_rewind_context
            and _active_branch_rewind_context(
                {"branch_rewind_context": self.branch_rewind_context},
                rewind_runtime_state,
            )
        )
        release_parent_evidence = bool(
            active_branch_rewind
            and str(self.branch_rewind_context.get("source", "")).strip()
            in {"", "operator_branch_rewind_note"}
        )
        if release_parent_evidence:
            # Only an operator-supplied branch note can intentionally reopen
            # parent candidates after a controller or primitive revision.
            # Automatic child-dead-end rewinds must reload failures already
            # observed at the restored parent checkpoint. The exact retired
            # child transaction remains non-replayable in either case.
            retired_signature = str(
                self.branch_rewind_context.get(
                    "retired_child_transaction_signature",
                    "",
                )
            ).strip()
            retired_transaction = str(
                self.branch_rewind_context.get(
                    "retired_child_transaction",
                    "",
                )
            ).strip()
            if retired_signature and retired_transaction:
                self._rejected_transactions[retired_signature] = retired_transaction
            route_exclusion = str(
                self.branch_rewind_context.get("blocked_motion_objective", "")
            ).strip()
            if (
                self.branch_rewind_context.get(
                    "enforce_blocked_motion_objective", False
                )
                and route_exclusion
            ):
                self._nonrelaxable_motion_objectives.add(route_exclusion)
            return
        history_dirs = tuple(dict.fromkeys((*self.history_dirs, self.output_dir)))
        route_phase_guard = (
            trusted_route_phase_guard(self.task, runtime_state)
            if self.task is not None
            else {}
        )
        for history_dir in history_dirs:
            # A JSON/code-generation exhaustion is not a physics experiment. Keep
            # such exclusions process-local so a restarted run can use repaired
            # validators or a conservative compiler fallback. Only verifier-grounded
            # rollout evidence may persistently block a semantic motion objective.
            for policy in _load_rejected_policy_history(
                history_dir,
                checkpoint_id=checkpoint_id,
                limit=0,
                minimum_attempt_index=checkpoint_source_attempt,
            ):
                self._remember_rejected_transaction(policy)
            self._directionally_rejected_methods.update(
                _load_goal_regression_method_history(
                    history_dir,
                    checkpoint_id=checkpoint_id,
                    limit=self.rejected_transaction_limit,
                    minimum_attempt_index=checkpoint_source_attempt,
                )
            )
            self._directionally_rejected_objectives.update(
                _load_goal_regression_objective_history(
                    history_dir,
                    checkpoint_id=checkpoint_id,
                    limit=self.rejected_transaction_limit,
                    minimum_attempt_index=checkpoint_source_attempt,
                )
            )
            self._nonrelaxable_motion_objectives.update(
                _load_route_regression_objective_history(
                    history_dir,
                    checkpoint_id=checkpoint_id,
                    limit=self.rejected_transaction_limit,
                    route_phase_guard=route_phase_guard,
                    minimum_attempt_index=checkpoint_source_attempt,
                )
            )
            for evidence_key, historical_policy in _load_failed_strategy_history(
                history_dir,
                checkpoint_id=checkpoint_id,
                limit=self.rejected_transaction_limit,
                minimum_attempt_index=checkpoint_source_attempt,
            ):
                self._remember_strategy_signature(
                    historical_policy,
                    evidence_key=evidence_key,
                )
        if active_branch_rewind:
            retired_signature = str(
                self.branch_rewind_context.get(
                    "retired_child_transaction_signature",
                    "",
                )
            ).strip()
            retired_transaction = str(
                self.branch_rewind_context.get(
                    "retired_child_transaction",
                    "",
                )
            ).strip()
            if retired_signature and retired_transaction:
                self._rejected_transactions[retired_signature] = retired_transaction

    def _remember_directionally_rejected_methods(self, policy: PolicyScript) -> None:
        methods, objectives = _policy_directional_rejections(policy)
        self._directionally_rejected_methods.update(methods)
        self._directionally_rejected_objectives.update(objectives)

    def _remember_failed_strategy(
        self,
        policy: PolicyScript,
        *,
        report: EvaluationReport,
        trace: ExecutionTrace | None,
    ) -> None:
        classification = classify_report(report)
        physical_dispatch = (
            trace_has_physical_dispatch(trace)
            if trace is not None
            else bool(report.metrics) and classification.level != "system"
        )
        if report.success or classification.level == "system" or not physical_dispatch:
            return
        self._remember_strategy_signature(
            policy,
            evidence_key=f"attempt_{policy.attempt:02d}",
        )

    def _remember_strategy_signature(
        self,
        policy: PolicyScript,
        *,
        evidence_key: str,
    ) -> None:
        signature = _policy_strategy_budget_signature(policy)
        if not signature or evidence_key in self._strategy_failure_keys:
            return
        self._strategy_failure_keys.add(evidence_key)
        count = self._strategy_failure_counts.get(signature, 0) + 1
        self._strategy_failure_counts[signature] = count
        # Counts remain useful diagnostics, but never block a strategy or semantic
        # objective. Exact rejected AST transactions are protected separately.


def make_qwen_client(
    *,
    model_name: str,
    device_map: str,
    max_gpu_memory: str | None,
    max_cpu_memory: str | None,
    temperature: float,
    max_new_tokens: int,
) -> QwenTextClient:
    """Create the default local model client used by live experiments."""

    return QwenTextClient(
        model_name=model_name,
        device_map=device_map,
        max_gpu_memory=max_gpu_memory,
        max_cpu_memory=max_cpu_memory,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def _attach_verified_action_exemplars(
    payload: dict[str, Any],
    *,
    selected_subtask: dict[str, Any],
    action_memory: VerifiedActionMemoryStore | None,
) -> None:
    """Attach a tiny same-action reference set after semantic selection."""

    if action_memory is None or not selected_subtask:
        payload.pop("verified_action_exemplars", None)
        return
    method = str(selected_subtask.get("selected_method", "")).strip()
    mode = str(selected_subtask.get("coordination_mode", "")).strip()
    phase = (
        "route_discovery_transport"
        if bool(payload.get("route_discovery", False))
        else str(payload.get("task_stage", ""))
    )
    payload["verified_action_exemplars"] = action_memory.retrieve(
        method=method,
        semantic_objective=selected_subtask_objective(selected_subtask),
        coordination_mode=mode,
        phase=phase,
        limit=2,
    )


def _verifier_straight_yaw_alignment(
    evaluation: object,
) -> dict[str, float]:
    """Return one verifier-owned relative-yaw correction, if present."""

    report = evaluation if isinstance(evaluation, dict) else {}
    if str(report.get("failure_type", "")) not in {
        "straightness_regression",
        "payload_slip",
    }:
        return {}
    evidence = report.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    straight = evidence.get("straight_transport", {})
    straight = straight if isinstance(straight, dict) else {}
    if (
        str(straight.get("correction_method", "")) != COORDINATED_ALIGNMENT
        or str(straight.get("correction_alignment_axis", "")) != "yaw"
    ):
        return {}
    raw_deltas = straight.get("correction_yaw_deltas_deg", {})
    if not isinstance(raw_deltas, dict) or set(raw_deltas) != {"front", "back"}:
        return {}
    try:
        deltas = {
            robot_id: float(raw_deltas[robot_id]) for robot_id in ("front", "back")
        }
    except (TypeError, ValueError):
        return {}
    if not all(isfinite(value) and abs(value) <= 30.0 for value in deltas.values()):
        return {}
    reference_robots = [
        robot_id for robot_id, value in deltas.items() if abs(value) <= 1.0e-9
    ]
    moving_deltas = [abs(value) for value in deltas.values() if abs(value) > 1.0e-9]
    if len(reference_robots) != 1 or len(moving_deltas) != 1 or moving_deltas[0] < 2.0:
        return {}
    return deltas


def _verifier_endpoint_longitudinal_alignment(
    evaluation: object,
) -> dict[str, float]:
    """Return one verifier-owned forward-only lagger correction, if required."""

    report = evaluation if isinstance(evaluation, dict) else {}
    if not bool(report.get("success", False)):
        return {}
    evidence = report.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    alignment = evidence.get("endpoint_longitudinal_alignment", {})
    alignment = alignment if isinstance(alignment, dict) else {}
    if not alignment:
        metrics = report.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        try:
            signed_stagger = float(metrics["robot_pair_longitudinal_stagger_m"])
        except (KeyError, TypeError, ValueError):
            return {}
        if (
            not isfinite(signed_stagger)
            or abs(signed_stagger) <= LONGITUDINAL_ALIGNMENT_TRIGGER_M
        ):
            return {}
        lagging_robot = "front" if signed_stagger > 0.0 else "back"
        correction_delta = min(
            MAX_LONGITUDINAL_ALIGNMENT_DELTA_M,
            max(MIN_LONGITUDINAL_ALIGNMENT_DELTA_M, abs(signed_stagger)),
        )
        alignment = {
            "required": True,
            "axis": "longitudinal",
            "robot_deltas_m": {
                robot_id: correction_delta if robot_id == lagging_robot else 0.0
                for robot_id in ("front", "back")
            },
            "compatibility_source": "persisted_pre_alignment-evidence_metrics",
        }
    if (
        not bool(alignment.get("required", False))
        or str(alignment.get("axis", "")) != "longitudinal"
    ):
        return {}
    raw_deltas = alignment.get("robot_deltas_m", {})
    if not isinstance(raw_deltas, dict) or set(raw_deltas) != {"front", "back"}:
        return {}
    try:
        deltas = {
            robot_id: float(raw_deltas[robot_id]) for robot_id in ("front", "back")
        }
    except (TypeError, ValueError):
        return {}
    if not all(
        isfinite(value) and 0.0 <= value <= MAX_LONGITUDINAL_ALIGNMENT_DELTA_M
        for value in deltas.values()
    ):
        return {}
    reference_robots = [
        robot_id for robot_id, value in deltas.items() if abs(value) <= 1.0e-9
    ]
    moving_deltas = [value for value in deltas.values() if value > 1.0e-9]
    if (
        len(reference_robots) != 1
        or len(moving_deltas) != 1
        or moving_deltas[0] < MIN_LONGITUDINAL_ALIGNMENT_DELTA_M
    ):
        return {}
    return deltas


def _verifier_pair_spacing_alignment(
    evaluation: object,
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    """Derive one local-body correction after a measured too-close failure."""

    report = evaluation if isinstance(evaluation, dict) else {}
    if str(report.get("failure_type", "")) != "formation_desync":
        return {}
    evidence = report.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    spacing = evidence.get("robot_pair_spacing", {})
    spacing = spacing if isinstance(spacing, dict) else {}
    try:
        failed_separation = float(spacing["measured_xy_separation_m"])
        minimum_separation = float(spacing["minimum_xy_separation_m"])
    except (KeyError, TypeError, ValueError):
        return {}
    if (
        not all(isfinite(value) for value in (failed_separation, minimum_separation))
        or failed_separation >= minimum_separation
    ):
        return {}

    geometry = _alignment_body_frame_geometry(runtime_state)
    if not geometry:
        return {}
    try:
        current_separation = float(geometry["robot_pair_xy_separation_m"])
        effects = geometry["positive_delta_separation_effect_m_per_m"]
    except (KeyError, TypeError, ValueError):
        return {}

    candidates: list[dict[str, Any]] = []
    for axis in ("lateral", "longitudinal"):
        for robot_id in ("front", "back"):
            try:
                effect = float(effects[robot_id][axis])
            except (KeyError, TypeError, ValueError):
                continue
            if not isfinite(effect):
                continue
            if axis == "longitudinal":
                if effect <= 1.0e-6:
                    continue
                delta_sign = 1.0
            else:
                if abs(effect) <= 1.0e-6:
                    continue
                delta_sign = 1.0 if effect > 0.0 else -1.0
            candidates.append(
                {
                    "axis": axis,
                    "moving_robot": robot_id,
                    "effect": effect,
                    "delta_sign": delta_sign,
                }
            )
    if not candidates:
        return {}

    # This is a coordinate conversion, not a route-direction decision.
    selected = max(candidates, key=lambda item: abs(float(item["effect"])))
    effect_magnitude = abs(float(selected["effect"]))
    minimum_delta_m = (
        MIN_LONGITUDINAL_ALIGNMENT_DELTA_M
        if selected["axis"] == "longitudinal"
        else 0.05
    )
    desired_gain = max(
        minimum_delta_m,
        minimum_separation + minimum_delta_m - current_separation,
    )
    delta_magnitude = min(
        MAX_LONGITUDINAL_ALIGNMENT_DELTA_M,
        max(
            minimum_delta_m,
            desired_gain / effect_magnitude,
        ),
    )
    moving_robot = str(selected["moving_robot"])
    reference_robot = "back" if moving_robot == "front" else "front"
    signed_delta = float(selected["delta_sign"]) * delta_magnitude
    return {
        "axis": str(selected["axis"]),
        "moving_robot": moving_robot,
        "reference_robot": reference_robot,
        "robot_deltas_m": {
            moving_robot: signed_delta,
            reference_robot: 0.0,
        },
        "failed_separation_m": failed_separation,
        "minimum_separation_m": minimum_separation,
        "trusted_separation_m": current_separation,
        "predicted_separation_gain_m": effect_magnitude * delta_magnitude,
    }


def _select_evolution_subtask(
    *,
    client: TextGenerationClient,
    payload: dict[str, Any],
    forbidden_methods: tuple[str, ...],
    forbidden_objectives: tuple[str, ...] = (),
    nonrelaxable_forbidden_objectives: tuple[str, ...] = (),
    blocked_strategy_signatures: tuple[str, ...] = (),
    response_retries: int = 2,
    response_audit_path: Path | None = None,
) -> dict[str, str]:
    """Ask the same model for a compact method choice before script synthesis."""

    formal_model_owns_route = bool(payload.get("formal_model_owns_route", False))
    evaluation = payload.get("evaluation", {})
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    evaluation_metrics = evaluation.get("metrics", {})
    evaluation_metrics = (
        evaluation_metrics if isinstance(evaluation_metrics, dict) else {}
    )
    current_policy = payload.get("current_policy", {})
    current_policy = current_policy if isinstance(current_policy, dict) else {}
    current_method = _first_policy_api_method(str(current_policy.get("code", "")))
    observation = payload.get("observation", {})
    observation = observation if isinstance(observation, dict) else {}
    runtime_state = observation.get("runtime_state", {})
    runtime_state = runtime_state if isinstance(runtime_state, dict) else {}
    turn_episode = runtime_state.get("turn_episode", {})
    turn_episode = turn_episode if isinstance(turn_episode, dict) else {}
    route_discovery = bool(payload.get("route_discovery", False))
    host_wall_clearance = (
        authored_scene_motion_clearance(runtime_state)
        if route_discovery
        else {
            "available": False,
            "reason": "host wall clearance is active only in route-discovery mode",
        }
    )
    wall_clear_loaded_directions = (
        _model_executable_loaded_translation_directions(host_wall_clearance)
    )
    all_loaded_translations_blocked = bool(
        host_wall_clearance.get("available", False) and not wall_clear_loaded_directions
    )
    verifier_spacing_alignment = _verifier_pair_spacing_alignment(
        evaluation,
        runtime_state,
    )
    if verifier_spacing_alignment and not formal_model_owns_route:
        return {
            "selected_method": COORDINATED_ALIGNMENT,
            "coordination_mode": "async",
            "motion_objective_parameter": "axis",
            "motion_objective": str(verifier_spacing_alignment["axis"]),
            "reason": (
                "The rejected rollout crossed the minimum robot-pair spacing. "
                f"Hold {verifier_spacing_alignment['reference_robot']} and move only "
                f"{verifier_spacing_alignment['moving_robot']} along the locally projected "
                "body axis that increases separation before attempting route motion again."
            ),
            "selection_source": "verifier_pair_spacing_alignment",
        }
    verifier_longitudinal_deltas = _verifier_endpoint_longitudinal_alignment(evaluation)
    if (
        not formal_model_owns_route
        and verifier_longitudinal_deltas
        and current_method != COORDINATED_ALIGNMENT
    ):
        moving_robot = max(
            verifier_longitudinal_deltas,
            key=verifier_longitudinal_deltas.get,
        )
        return {
            "selected_method": COORDINATED_ALIGNMENT,
            "coordination_mode": "async",
            "motion_objective_parameter": "axis",
            "motion_objective": "longitudinal",
            "reason": (
                "The accepted checkpoint has a recoverable front/back stagger. "
                f"Hold the reference robot and move only lagging {moving_robot} "
                "a short body-forward distance before route motion."
            ),
            "selection_source": "verifier_endpoint_longitudinal_alignment",
        }
    verifier_alignment_deltas = _verifier_straight_yaw_alignment(evaluation)
    if verifier_alignment_deltas and not formal_model_owns_route:
        moving_robot = max(
            verifier_alignment_deltas,
            key=lambda robot_id: abs(verifier_alignment_deltas[robot_id]),
        )
        # The failed translation already supplied the required correction from
        # measured robot roots. This is a verifier-owned recovery transaction,
        # so it must run before a route episode can replay its linear probe.
        return {
            "selected_method": COORDINATED_ALIGNMENT,
            "coordination_mode": "async",
            "motion_objective_parameter": "axis",
            "motion_objective": "yaw",
            "reason": (
                "Execute the verifier-owned relative-yaw correction before "
                f"another translation; hold the reference robot and align {moving_robot}."
            ),
            "selection_source": "verifier_straight_heading_alignment",
        }
    host_heading_correction = unique_small_heading_correction(host_wall_clearance)
    if (
        not formal_model_owns_route
        and host_heading_correction
        and str(evaluation.get("failure_type", ""))
        in {
            "no_payload_progress",
            "route_regression",
            "alignment_regression",
        }
    ):
        return {
            "selected_method": COORDINATED_PIVOT,
            "coordination_mode": "async",
            "motion_objective_parameter": "turn_direction",
            "motion_objective": str(host_heading_correction["turn_direction"]),
            "reason": (
                "Execute one host-authored-wall clearance correction before "
                f"retrying body-frame "
                f"{host_heading_correction['desired_translation_direction']}."
            ),
            "selection_source": HOST_WALL_HEADING_CORRECTION_SELECTION_SOURCE,
            **host_heading_correction,
        }
    if (
        not formal_model_owns_route
        and bool(turn_episode.get("active", False))
        and str(turn_episode.get("purpose", "")) == "heading_correction"
    ):
        primitive = str(turn_episode.get("primitive", ""))
        direction = str(turn_episode.get("direction", ""))
        roles = turn_episode.get("roles", {})
        roles = roles if isinstance(roles, dict) else {}
        try:
            correction_angle_deg = float(turn_episode["correction_angle_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelAgentError(
                "verifier heading correction is missing a valid correction_angle_deg"
            ) from exc
        if (
            primitive != COORDINATED_PIVOT
            or direction not in _SIGNED_TURN_DIRECTIONS
            or set(roles) != {"front", "back"}
            or set(roles.values()) != {"inner", "outer"}
            or correction_angle_deg <= 0.0
        ):
            raise ModelAgentError(
                "verifier heading correction has an invalid pivot, direction, roles, or angle"
            )
        # This is a verifier-owned recovery transaction, not a strategy search.
        # Return before model-generation exclusion filtering.
        return {
            "selected_method": COORDINATED_PIVOT,
            "coordination_mode": "async",
            "motion_objective_parameter": "turn_direction",
            "motion_objective": direction,
            "reason": (
                "Execute the verifier-owned one-shot heading correction with "
                f"angle_deg={correction_angle_deg:g}."
            ),
            "selection_source": "verifier_heading_correction",
        }
    route_turn_direction = str(turn_episode.get("direction", ""))
    route_turn_clearance_override = bool(
        candidate_is_blocked(
            host_wall_clearance,
            family="turn",
            objective=route_turn_direction,
        )
        and _model_executable_loaded_translation_directions(host_wall_clearance)
    )
    if (
        not formal_model_owns_route
        and bool(turn_episode.get("active", False))
        and str(turn_episode.get("purpose", "")) == "route_turn"
        and str(turn_episode.get("status", "turning")) == "turning"
        and not bool(turn_episode.get("switch_allowed", False))
        and not route_turn_clearance_override
    ):
        primitive = str(turn_episode.get("primitive", ""))
        direction = str(turn_episode.get("direction", ""))
        roles = turn_episode.get("roles", {})
        roles = roles if isinstance(roles, dict) else {}
        role_contract_valid = bool(
            (primitive in STATIC_TURN_METHODS and not roles)
            or (
                primitive in ROLE_BASED_TURN_METHODS
                and set(roles) == {"front", "back"}
                and set(roles.values()) == {"inner", "outer"}
            )
        )
        if (
            primitive not in TURN_METHODS
            or direction not in _SIGNED_TURN_DIRECTIONS
            or not role_contract_valid
        ):
            raise ModelAgentError(
                "active route-turn episode has an invalid primitive, direction, or roles"
            )
        return {
            "selected_method": primitive,
            "coordination_mode": (
                "sync" if primitive in STATIC_TURN_METHODS else "async"
            ),
            "motion_objective_parameter": (
                "" if primitive in STATIC_TURN_METHODS else "turn_direction"
            ),
            "motion_objective": direction,
            "reason": (
                "Continue the verifier-accepted signed route-turn episode until "
                "the global visual verifier reports passage alignment."
            ),
            "selection_source": "verifier_route_turn_continuation",
        }
    if route_turn_clearance_override:
        turn_episode = {
            **turn_episode,
            "active": False,
            "status": "host_clearance_replan",
            "switch_allowed": True,
            "host_clearance_replan_reason": (
                f"{route_turn_direction} finite coordinated-pivot sweep is locally "
                "blocked before the minimum executable angle while a body-frame "
                "translation remains open"
            ),
        }
    released_on_target = _payload_is_released_on_target(runtime_state)
    coordination_layer_enabled = bool(payload.get("coordination_primitive_sdk"))
    controller = payload.get("evolution_controller", {})
    controller = controller if isinstance(controller, dict) else {}
    robot_scripts = current_policy.get("robot_scripts", ())
    robot_scripts = robot_scripts if isinstance(robot_scripts, (list, tuple)) else ()
    current_modes = {
        str(item.get("coordination_mode", "")).strip()
        for item in robot_scripts
        if isinstance(item, dict) and str(item.get("coordination_mode", "")).strip()
    }
    current_coordination_mode = (
        next(iter(current_modes)) if len(current_modes) == 1 else ""
    )
    candidate_methods = (
        {
            "exit_forward",
            "exit_backward",
            "exit_lateral_left",
            "exit_lateral_right",
        }
        if released_on_target
        else (
            set(MODEL_COORDINATION_PRIMITIVE_METHODS)
            | {"place_crouch", "recover_stand", "hold_position"}
            if coordination_layer_enabled
            else set(MODEL_PLANNING_ACTION_METHODS) | {"hold_position"}
        )
    )
    if evaluation.get("failure_type") == "no_payload_progress":
        # The current checkpoint is already physically stable. These methods
        # can preserve or change posture, but none can satisfy the later
        # strategy-repair validator's requirement for a new motion objective.
        # Excluding them here keeps subtask selection and code validation from
        # creating an impossible generate/reject loop.
        candidate_methods.difference_update(_SUPPORT_ONLY_METHODS)
    elif bool(evaluation.get("success", False)) and not bool(
        evaluation_metrics.get("task_complete", False)
    ):
        # A settled hold is useful after instability, but cannot repair an
        # accepted action into a progress loop.
        candidate_methods.discard("hold_position")
    if (
        bool(evaluation.get("success", False))
        and current_method == COORDINATED_LINEAR_TRANSPORT
    ):
        # Verifier-owned longitudinal/yaw corrections return before this generic
        # selector. A free lateral alignment cannot turn the team at a corner and
        # otherwise becomes a micro-action loop after each accepted translation.
        candidate_methods.discard(COORDINATED_ALIGNMENT)
    if (
        controller.get("action") == "switch_strategy"
        or controller.get("repair_level") == "strategy"
    ):
        # Checkpoint recovery already established a stable physical state. A
        # support-only posture action cannot satisfy a host-required strategy switch.
        candidate_methods.intersection_update(
            {
                "carry_forward",
                "carry_backward",
                "carry_lateral_left",
                "carry_lateral_right",
                "turn_clockwise",
                "turn_counterclockwise",
                *MODEL_COORDINATION_PRIMITIVE_METHODS,
            }
        )
    allowed_coordination_modes = {"sync", "async"}
    repair_level = str(controller.get("repair_level", "")).strip()
    failure_type = str(evaluation.get("failure_type", "")).strip()
    evaluation_evidence = evaluation.get("evidence", {})
    evaluation_evidence = (
        evaluation_evidence if isinstance(evaluation_evidence, dict) else {}
    )
    straight_transport_evidence = evaluation_evidence.get("straight_transport", {})
    straight_transport_evidence = (
        straight_transport_evidence
        if isinstance(straight_transport_evidence, dict)
        else {}
    )
    required_straight_correction_direction = str(
        straight_transport_evidence.get("correction_turn_direction", "")
    ).strip()
    required_straight_correction_method = str(
        straight_transport_evidence.get(
            "correction_method",
            COORDINATED_PIVOT,
        )
    ).strip()
    if failure_type == "straightness_regression":
        if required_straight_correction_method == COORDINATED_ALIGNMENT:
            candidate_methods.intersection_update({COORDINATED_ALIGNMENT})
            allowed_coordination_modes = {"async"}
        else:
            candidate_methods.intersection_update({COORDINATED_PIVOT})
            allowed_coordination_modes = {"async"}
    measured_sync_retry = bool(
        current_coordination_mode == "sync"
        and current_method in _TRANSPORT_METHODS
        and _measured_sync_compensation_constraint(
            payload,
            selected_subtask={
                "selected_method": current_method,
                "coordination_mode": "sync",
            },
        )
    )
    if route_discovery:
        target_progress = 0.0
        body_frame_goal_alignment: dict[str, Any] = {}
    else:
        try:
            target_progress = float(
                evaluation_metrics.get("payload_target_progress_m", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            target_progress = 0.0
        body_frame_goal_alignment = _body_frame_goal_alignment(
            runtime_state,
            evaluation_metrics,
        )
    branch_rewind_context = _active_branch_rewind_context(
        payload,
        runtime_state,
        body_frame_goal_alignment,
    )
    if branch_rewind_context and route_discovery:
        # The report still describes the retired child transaction, while the
        # camera and physics state now belong to its parent.  Child-local
        # execution/coordination locks must not narrow the parent's search to
        # the very primitive family that the rewind is trying to leave.
        candidate_methods = set(MODEL_COORDINATION_PRIMITIVE_METHODS)
        allowed_coordination_modes = {"sync", "async"}
        repair_level = "strategy"
    if (
        bool(turn_episode.get("translation_pending", False))
        and not all_loaded_translations_blocked
    ):
        # Payload yaw, not robot-root yaw, has already completed the current
        # corner. Keep route choice open across all body-frame linear
        # directions and bounded alignment, but do not let a new pivot/arc
        # over-rotate the same corner before the team translates away.
        candidate_methods.difference_update(TURN_METHODS)
        if (
            bool(evaluation.get("success", False))
            and current_method == COORDINATED_ALIGNMENT
        ):
            # One accepted alignment is a setup action, not a route-progress
            # primitive. Repeating the same accepted yaw/lateral adjustment at
            # every new checkpoint creates false evolution while the payload
            # remains at the corner.
            candidate_methods.intersection_update({COORDINATED_LINEAR_TRANSPORT})
            allowed_coordination_modes = {"sync"}
    current_linear_objective = _linear_objective_from_code(
        str(current_policy.get("code", ""))
    )
    trusted_checkpoint_linear_motion = payload.get(
        "trusted_checkpoint_linear_motion",
        {},
    )
    trusted_checkpoint_linear_motion = (
        trusted_checkpoint_linear_motion
        if isinstance(trusted_checkpoint_linear_motion, dict)
        else {}
    )
    trusted_checkpoint_linear_direction = str(
        trusted_checkpoint_linear_motion.get("direction", "")
    ).strip()
    current_goal_scores = body_frame_goal_alignment.get(
        "candidate_goal_alignment",
        {},
    )
    current_goal_scores = (
        current_goal_scores if isinstance(current_goal_scores, dict) else {}
    )
    try:
        current_linear_goal_alignment = float(
            current_goal_scores[current_linear_objective]
        )
    except (KeyError, TypeError, ValueError):
        current_linear_goal_alignment = None
    successful_linear_continuation = bool(
        evaluation.get("success", False)
        and current_method == COORDINATED_LINEAR_TRANSPORT
        and not branch_rewind_context
        and target_progress > 0.0
        and not bool(evaluation_metrics.get("active_stage_goal_reached", False))
        and not bool(evaluation_metrics.get("task_complete", False))
        and (
            current_linear_goal_alignment is None
            or current_linear_goal_alignment >= 0.35
        )
    )
    accepted_linear_direction_continuation = bool(
        evaluation.get("success", False)
        and current_method == COORDINATED_LINEAR_TRANSPORT
        and current_linear_objective
        and not branch_rewind_context
        and not bool(evaluation_metrics.get("task_complete", False))
    )
    if successful_linear_continuation:
        # The verifier just established both the semantic direction and the
        # loaded coordination relation. Preserve that result for the next
        # bounded segment; a model-only preference is not failure evidence for
        # switching to alignment, arc, or pivot.
        candidate_methods.intersection_update({current_method})
        allowed_coordination_modes = {"sync"}
    if (
        repair_level == "execution"
        and current_method in candidate_methods
        and not branch_rewind_context
    ):
        # The host controller has already classified this as an execution-envelope
        # retry. Keep the model inside the previously verified strategy while it
        # chooses the concrete bounded parameter repair.
        candidate_methods.intersection_update({current_method})
        if current_coordination_mode:
            allowed_coordination_modes = {current_coordination_mode}
    elif repair_level == "coordination" and current_method in candidate_methods:
        # A coordination repair keeps the physical motion objective. The code
        # synthesis turn remains responsible for choosing the temporary roles.
        candidate_methods.intersection_update({current_method})
        if measured_sync_retry:
            # Equal sync commands produced unequal measured displacement. Keep
            # the atomic sync barrier and calibrate the executor distances.
            allowed_coordination_modes = {"sync"}
        elif current_coordination_mode == "sync":
            allowed_coordination_modes = {"async"}
    elif (
        repair_level == "strategy"
        and not branch_rewind_context
        and failure_type
        in {
            "formation_desync",
            "payload_not_supported",
            "payload_slip",
            "payload_unstable",
        }
        and current_method in _TRANSPORT_METHODS
        and current_method in candidate_methods
        and target_progress >= -0.005
    ):
        # These failures invalidate the current collaboration relation, not the
        # body-frame motion objective. Even when that relation exhausts its
        # budget, change sync/async first instead of inventing an unrelated
        # turn or direction. A measured goal regression remains free to change
        # the mover objective through the normal strategy path.
        blocked = set(blocked_strategy_signatures)
        current_method_has_untried_mode = any(
            f"{mode}:{current_method}" not in blocked
            for mode in allowed_coordination_modes
        )
        if current_method_has_untried_mode:
            candidate_methods.intersection_update({current_method})
    nonrelaxable_objectives = set(nonrelaxable_forbidden_objectives)
    effective_forbidden_objectives = (
        set()
        if formal_model_owns_route
        else set(forbidden_objectives) | nonrelaxable_objectives
    )
    branch_blocked_objective = ""
    if branch_rewind_context:
        scores = body_frame_goal_alignment.get("candidate_goal_alignment", {})
        scores = scores if isinstance(scores, dict) else {}
        finite_scores = {
            str(direction): float(score)
            for direction, score in scores.items()
            if isinstance(score, (int, float)) and isfinite(float(score))
        }
        configured_objective = str(
            branch_rewind_context.get("blocked_motion_objective", "")
        ).strip()
        configured_method, separator, configured_semantic_objective = (
            configured_objective.partition(":")
        )
        if (
            separator
            and configured_method in COORDINATION_PRIMITIVE_METHODS
            and configured_semantic_objective
        ):
            branch_blocked_objective = configured_objective
            effective_forbidden_objectives.add(branch_blocked_objective)
        elif finite_scores:
            best_direction, best_score = max(
                finite_scores.items(),
                key=lambda item: item[1],
            )
            if best_score >= 0.35:
                branch_blocked_objective = (
                    f"{COORDINATED_LINEAR_TRANSPORT}:{best_direction}"
                )
                effective_forbidden_objectives.add(branch_blocked_objective)
    linear_geometry_gate = _linear_transport_geometry_gate(
        runtime_state,
        body_frame_goal_alignment,
        repair_level=repair_level,
        forbidden_objectives=tuple(sorted(effective_forbidden_objectives)),
    )
    linear_geometry_gate["branch_rewind_override"] = bool(branch_rewind_context)
    linear_geometry_gate["branch_rewind_blocked_objective"] = branch_blocked_objective
    linear_geometry_gate["linear_method_excluded"] = False
    if linear_geometry_gate["active"]:
        # Once the supported payload is already aligned with the active-goal axis,
        # another turn/alignment search can only destroy useful geometry. The host
        # selects the method family here; the model still chooses an executable
        # body-frame direction from measured projections below.
        candidate_methods.intersection_update({COORDINATED_LINEAR_TRANSPORT})
    elif linear_geometry_gate["best_body_frame_objective_blocked"]:
        # Every alternative body-frame direction is weaker than the verifier-blocked
        # best direction and will be rejected by _validated_linear_goal_direction.
        # Do not expose impossible enum values to the model. It must first choose a
        # general coordination primitive that changes the heading/formation geometry;
        # the signed turn direction remains a model decision.
        candidate_methods.discard(COORDINATED_LINEAR_TRANSPORT)
        linear_geometry_gate["constraint"] = "change_geometry_before_linear_transport"
        linear_geometry_gate["linear_method_excluded"] = True
    if branch_rewind_context and route_discovery:
        # Route discovery intentionally has no host goal vector with which to
        # rank sibling linear directions. A rewind lesson that requires a
        # geometry change must therefore expose only primitives that can change
        # heading or formation. A support hold/crouch cannot satisfy the rewind,
        # while blocking one model-chosen linear direction merely creates a loop.
        candidate_methods.intersection_update(
            {
                COORDINATED_ALIGNMENT,
                COORDINATED_PIVOT,
            }
        )
        linear_geometry_gate["constraint"] = (
            "branch_rewind_requires_geometry_change_before_linear_transport"
        )
        linear_geometry_gate["linear_method_excluded"] = True
    if current_method == "hold_position" and bool(evaluation.get("success", False)):
        candidate_methods.discard("hold_position")
    passage_alignment_gate = _route_passage_alignment_gate(
        runtime_state,
        turn_episode=turn_episode,
        route_discovery=route_discovery,
    )
    wall_clearance_escape_directions = _host_wall_clearance_escape_directions(
        passage_alignment_gate=passage_alignment_gate,
        host_wall_clearance=host_wall_clearance,
    )
    wall_turn_escape_active = bool(
        passage_alignment_gate["aligned"]
        and bool(turn_episode.get("translation_pending", False))
        and all_loaded_translations_blocked
    )
    passage_alignment_gate["host_wall_clearance_escape"] = {
        "active": bool(wall_clearance_escape_directions),
        "open_body_frame_directions": list(wall_clearance_escape_directions),
        "basis": (
            "both signed 30-degree turn headings are blocked by authored-wall "
            "geometry while one or more body-frame translations remain open"
            if wall_clearance_escape_directions
            else "inactive"
        ),
    }
    passage_alignment_gate["host_wall_turn_escape"] = {
        "active": wall_turn_escape_active,
        "blocked_body_frame_directions": (
            list(LOADED_TRANSLATION_DIRECTIONS) if wall_turn_escape_active else []
        ),
        "basis": (
            "The global VLM called the formation passage-aligned, but current "
            "authored-wall geometry leaves no executable loaded body-frame "
            "translation. Treat alignment as provisional and let the Agent choose "
            "a signed pivot and temporary inner/outer roles."
            if wall_turn_escape_active
            else "inactive"
        ),
    }
    if wall_clearance_escape_directions:
        # A diagonal visual posture normally requires a heading change. When
        # authored geometry proves that both signed turn headings are blocked,
        # forcing a turn creates an impossible action space. Let the model choose
        # only among locally open body-frame translations to create turn clearance.
        candidate_methods = {COORDINATED_LINEAR_TRANSPORT}
        allowed_coordination_modes = {"sync"}
        repair_level = "strategy"
        linear_geometry_gate["constraint"] = (
            "host_wall_clearance_escape_linear_transport"
        )
        linear_geometry_gate["linear_method_excluded"] = False
    elif wall_turn_escape_active:
        # Visual axis alignment is insufficient when every executable loaded
        # translation intersects nearby authored geometry. Reopen one bounded
        # geometry-changing pivot instead of presenting the model with a linear
        # method whose direction enum is empty.
        candidate_methods = {COORDINATED_PIVOT}
        allowed_coordination_modes = {"async"}
        repair_level = "strategy"
        linear_geometry_gate["constraint"] = (
            "passage_alignment_has_no_executable_linear_direction"
        )
        linear_geometry_gate["linear_method_excluded"] = True
    elif passage_alignment_gate["requires_heading_change"]:
        # This is a categorical current-view safety contract, not a route hint.
        # Override stale retry-level narrowing so a rejected translation cannot
        # force another linear attempt from the same diagonal checkpoint.
        candidate_methods = {COORDINATED_PIVOT}
        allowed_coordination_modes = {"async"}
        repair_level = "strategy"
        linear_geometry_gate["constraint"] = (
            "global_passage_posture_requires_shared_heading_change"
        )
        linear_geometry_gate["linear_method_excluded"] = True
    elif passage_alignment_gate["aligned"] and bool(
        turn_episode.get("translation_pending", False)
    ):
        # A turn episode ends only after the global view verifies one principal
        # formation axis against open passage. The model still owns the
        # forward/left/right body-frame choice.
        candidate_methods = {COORDINATED_LINEAR_TRANSPORT}
        allowed_coordination_modes = {"sync"}
    if bool(turn_episode.get("active", False)) and not bool(
        turn_episode.get("switch_allowed", False)
    ):
        episode_status = str(turn_episode.get("status", "turning"))
        support_methods = {"hold_position", "recover_stand"}
        if episode_status == "probe_ready":
            candidate_methods.intersection_update(
                {COORDINATED_LINEAR_TRANSPORT, *support_methods}
            )
            allowed_coordination_modes = {"sync"}
        else:
            candidate_methods.intersection_update(set(TURN_METHODS) | support_methods)
            allowed_coordination_modes = {"async", "sync"}
    if formal_model_owns_route:
        # Rebuild the formal capability space after collecting diagnostics.
        # Lifecycle exclusions and immediate authored-wall collisions remain
        # hard constraints; host route preferences, continuation rules, and
        # historical semantic exclusions do not choose an action for the model.
        candidate_methods = (
            {
                "exit_forward",
                "exit_backward",
                "exit_lateral_left",
                "exit_lateral_right",
            }
            if released_on_target
            else set(MODEL_COORDINATION_PRIMITIVE_METHODS)
            | {"place_crouch", "recover_stand", "hold_position"}
        )
        if evaluation.get("failure_type") == "no_payload_progress":
            candidate_methods.difference_update(_SUPPORT_ONLY_METHODS)
        elif bool(evaluation.get("success", False)) and not bool(
            evaluation_metrics.get("task_complete", False)
        ):
            candidate_methods.discard("hold_position")
        allowed_coordination_modes = {"sync", "async"}
        wall_clearance_escape_directions = ()
        wall_turn_escape_active = False
        route_turn_clearance_override = False
    allowed_methods = (
        [COORDINATED_LINEAR_TRANSPORT]
        if wall_clearance_escape_directions
        else sorted(candidate_methods - set(forbidden_methods))
    )
    if not allowed_methods:
        raise ModelAgentError(
            "no model-selectable SDK methods remain after verifier exclusions"
        )
    if len(allowed_methods) == 1 and not wall_clearance_escape_directions:
        only_method = allowed_methods[0]
        allowed_coordination_modes = {
            mode
            for mode in allowed_coordination_modes
            if f"{mode}:{only_method}" not in set(blocked_strategy_signatures)
        }
        if not allowed_coordination_modes:
            raise ModelAgentError(
                f"all coordination modes for api.{only_method} were excluded by the "
                "current model-generation search"
            )
    blocked_signatures = (
        set()
        if wall_clearance_escape_directions or formal_model_owns_route
        else set(blocked_strategy_signatures)
    )
    allowed_strategy_pairs = sorted(
        f"{mode}:{method}"
        for method in allowed_methods
        for mode in allowed_coordination_modes
        if f"{mode}:{method}" not in blocked_signatures
        and mode in coordination_modes_for_method(method)
        and not (method == "hold_position" and mode != "sync")
        and not (method in {"place_crouch", "recover_stand"} and mode != "sync")
        and not (method.startswith("exit_") and mode != "sync")
    )
    if not allowed_strategy_pairs:
        raise ModelAgentError(
            "no legal coordination_mode:method pair remains after current generation exclusions"
        )
    # Do not expose a method or mode that cannot form at least one executable
    # pair. The model now chooses one enum-like pair instead of independently
    # combining two locally valid values into a globally blocked strategy.
    allowed_methods = sorted({pair.split(":", 1)[1] for pair in allowed_strategy_pairs})
    allowed_coordination_modes = {
        pair.split(":", 1)[0] for pair in allowed_strategy_pairs
    }
    method_specs = {
        str(method["name"]): str(method.get("description", ""))
        for family in load_static_skill_catalog()
        for method in family["methods"]
        if method["name"] in allowed_methods
    }
    method_specs.update(
        {
            method_name: str(
                coordination_primitive_method_spec(method_name)["description"]
            )
            for method_name in allowed_methods
            if method_name in COORDINATION_PRIMITIVE_METHODS
        }
    )
    if "hold_position" in allowed_methods:
        method_specs["hold_position"] = (
            "stabilize loaded support without target progress; use only when evidence requires it"
        )
    for method in allowed_methods:
        if method.startswith("exit_"):
            method_specs[method] = (
                "unloaded body-relative SONIC exit after verified target support; "
                "select from robot yaw and visible clearance"
            )
    motion_objectives: dict[str, dict[str, object]] = {}
    trusted_route_observation = runtime_state.get(
        "trusted_route_observation",
        {},
    )
    trusted_route_filtered_objectives: list[str] = []
    current_call_arguments = _first_task_call_literal_arguments(
        str(current_policy.get("code", ""))
    )
    for method in allowed_methods:
        parameter_name = _COORDINATION_OBJECTIVE_PARAMETER.get(method, "")
        if not parameter_name:
            continue
        parameter_spec = coordination_primitive_method_spec(method)["parameters"][
            parameter_name
        ]
        choices = list(parameter_spec.get("choices", ()))
        choices = [
            choice
            for choice in choices
            if not _motion_objective_is_forbidden(
                method,
                str(choice),
                tuple(sorted(effective_forbidden_objectives)),
            )
        ]
        if method == COORDINATED_LINEAR_TRANSPORT:
            choices = [
                choice
                for choice in choices
                if _loaded_translation_has_supported_tier(
                    host_wall_clearance,
                    str(choice),
                )
            ]
        elif method in _SIGNED_TURN_METHODS and (
            wall_clear_loaded_directions or formal_model_owns_route
        ):
            choices = [
                choice
                for choice in choices
                if not candidate_is_blocked(
                    host_wall_clearance,
                    family="turn",
                    objective=str(choice),
                )
            ]
        if (
            not formal_model_owns_route
            and failure_type == "straightness_regression"
            and method == COORDINATED_PIVOT
            and required_straight_correction_direction in choices
        ):
            choices = [required_straight_correction_direction]
        if (
            not formal_model_owns_route
            and failure_type == "straightness_regression"
            and method == COORDINATED_ALIGNMENT
            and "yaw" in choices
        ):
            choices = ["yaw"]
        if (
            not formal_model_owns_route
            and method == COORDINATED_LINEAR_TRANSPORT
            and linear_geometry_gate["active"]
        ):
            # A waypoint transition can make the previous successful body-frame
            # direction invalid even though the primitive family remains correct.
            # Expose the unique measured direction to the model instead of asking
            # it to echo an enum that the host validator will reject afterwards.
            measured_direction = str(
                linear_geometry_gate.get("best_body_frame_direction", "")
            )
            if measured_direction in choices:
                choices = [measured_direction]
        current_motion_objective = str(current_call_arguments.get(parameter_name, ""))
        if (
            not formal_model_owns_route
            and method == current_method
            and method == COORDINATED_LINEAR_TRANSPORT
            and accepted_linear_direction_continuation
            and current_motion_objective in choices
        ):
            # The just-accepted policy is newer than any persisted artifact
            # that may reuse an old checkpoint id after a branch rewind.
            choices = [current_motion_objective]
        elif (
            not formal_model_owns_route
            and method == COORDINATED_LINEAR_TRANSPORT
            and trusted_checkpoint_linear_direction in choices
        ):
            # Failed descendants are rolled back to this checkpoint, so its
            # accepted body-frame direction remains the authoritative linear
            # direction until another maneuver earns a new checkpoint.
            choices = [trusted_checkpoint_linear_direction]
        elif (
            not formal_model_owns_route
            and method == current_method
            and (
                repair_level in {"coordination", "script", "execution"}
                or successful_linear_continuation
            )
            and current_motion_objective in choices
        ):
            # A freshly accepted loaded linear segment is direct physical
            # evidence for its body-frame direction. Do not let the next
            # linear proposal reverse that direction merely from model text;
            # the model remains free to select a geometry-changing primitive.
            choices = [current_motion_objective]
        if (
            not formal_model_owns_route
            and failure_type in {"goal_regression", "route_regression"}
            and repair_level == "strategy"
        ):
            if (
                method == COORDINATED_LINEAR_TRANSPORT
                and current_linear_objective in choices
            ):
                choices.remove(current_linear_objective)
            elif method == current_method and current_motion_objective in choices:
                choices.remove(current_motion_objective)
        if (
            not formal_model_owns_route
            and method == COORDINATED_LINEAR_TRANSPORT
            and wall_clearance_escape_directions
        ):
            # Override stale accepted-direction continuation, but retain explicit
            # verifier-grounded objective exclusions. The model still chooses the
            # semantic direction from current locally open geometry.
            choices = [
                choice
                for choice in parameter_spec.get("choices", ())
                if str(choice) in set(wall_clearance_escape_directions)
                and not _motion_objective_is_forbidden(
                    method,
                    str(choice),
                    tuple(sorted(effective_forbidden_objectives)),
                )
            ]
        if (
            not formal_model_owns_route
            and bool(turn_episode.get("active", False))
            and method in TURN_METHODS
            and not bool(turn_episode.get("switch_allowed", False))
        ):
            committed_direction = str(turn_episode.get("direction", ""))
            choices = [choice for choice in choices if choice == committed_direction]
        route_safe_choices: list[object] = []
        for choice in choices:
            if not formal_model_owns_route and _trusted_route_blocks_motion_objective(
                selected_method=method,
                motion_objective=str(choice),
                trusted_route_observation=trusted_route_observation,
            ):
                trusted_route_filtered_objectives.append(f"{method}:{choice}")
                continue
            if (
                not formal_model_owns_route
                and method in _SIGNED_TURN_METHODS
                and _trusted_route_blocks_signed_turn(
                    motion_objective=str(choice),
                    trusted_route_observation=trusted_route_observation,
                    host_wall_clearance=host_wall_clearance,
                )
            ):
                trusted_route_filtered_objectives.append(f"{method}:{choice}")
                continue
            route_safe_choices.append(choice)
        choices = route_safe_choices
        if not choices:
            continue
        motion_objectives[method] = {
            "parameter": parameter_name,
            "choices": choices,
        }
    objective_methods_without_choices = {
        method
        for method in allowed_methods
        if method in _COORDINATION_OBJECTIVE_PARAMETER
        and method not in motion_objectives
    }
    relaxed_cross_primitive_turn_aliases: set[str] = set()
    remaining_objective_pairs = [
        pair
        for pair in allowed_strategy_pairs
        if pair.split(":", 1)[1] not in objective_methods_without_choices
    ]
    if (
        objective_methods_without_choices
        and not remaining_objective_pairs
        and route_discovery
    ):
        # A failed signed turn records both its exact primitive objective and a
        # shared turn-direction alias. The alias accelerates ordinary search,
        # but after both aliases have accumulated it can hide untried primitive
        # families and leave the selector with an empty action space. Recover
        # only those cross-primitive aliases; exact verifier-grounded failures
        # and strategy budgets remain blocked.
        shared_turn_aliases = {
            f"turn:{direction}"
            for direction in _SIGNED_TURN_DIRECTIONS
            if f"turn:{direction}" in effective_forbidden_objectives
        }
        relaxable_shared_turn_aliases = shared_turn_aliases - nonrelaxable_objectives
        exact_forbidden_objectives = tuple(
            sorted(effective_forbidden_objectives - relaxable_shared_turn_aliases)
        )
        for method in sorted(objective_methods_without_choices & _SIGNED_TURN_METHODS):
            parameter_name = _COORDINATION_OBJECTIVE_PARAMETER[method]
            parameter_spec = coordination_primitive_method_spec(method)["parameters"][
                parameter_name
            ]
            choices = [
                str(choice)
                for choice in parameter_spec.get("choices", ())
                if not _motion_objective_is_forbidden(
                    method,
                    str(choice),
                    exact_forbidden_objectives,
                )
            ]
            if choices:
                motion_objectives[method] = {
                    "parameter": parameter_name,
                    "choices": choices,
                }
        if any(method in motion_objectives for method in _SIGNED_TURN_METHODS):
            relaxed_cross_primitive_turn_aliases = relaxable_shared_turn_aliases
            effective_forbidden_objectives.difference_update(
                relaxable_shared_turn_aliases
            )
            objective_methods_without_choices = {
                method
                for method in allowed_methods
                if method in _COORDINATION_OBJECTIVE_PARAMETER
                and method not in motion_objectives
            }
    if objective_methods_without_choices:
        allowed_strategy_pairs = [
            pair
            for pair in allowed_strategy_pairs
            if pair.split(":", 1)[1] not in objective_methods_without_choices
        ]
        allowed_methods = sorted(
            {pair.split(":", 1)[1] for pair in allowed_strategy_pairs}
        )
        method_specs = {
            method: description
            for method, description in method_specs.items()
            if method in allowed_methods
        }
        if not allowed_strategy_pairs:
            raise ModelAgentError(
                "no legal coordination primitive objective remains after physical budget exclusions"
            )
    current_execution_state = _selection_current_execution_state(runtime_state)
    current_execution_state["host_authored_wall_clearance"] = host_wall_clearance
    selection_context = {
        "current_execution_state": current_execution_state,
        "state_timeline": {
            "current_execution_state": (
                "AUTHORITATIVE: the next physical action starts from exactly this trusted state"
            ),
            "historical_rollout_evaluation": (
                "NEGATIVE HISTORY ONLY: its endpoint damage was rolled back and is not current"
            ),
        },
        "task_stage": payload.get("task_stage"),
        "task_orchestrator_plan": payload.get("task_orchestrator_plan", {}),
        "route_discovery": route_discovery,
        "route_discovery_contract": (
            "Use active_route_guidance as the validated connected-segment geometry. It does "
            "not prescribe a primitive, body-frame direction, signed turn, roles, mode, or "
            "parameters; choose those from the global camera, current physical formation, "
            "collision evidence, and prior verified rollouts."
            if route_discovery
            else "legacy host waypoint guidance"
        ),
        "global_camera_frame_calibration": (
            {
                "screen_right": "world -X",
                "screen_down": "world +Y",
                "scope": (
                    "fixed camera-frame calibration only; this is not a route, waypoint, "
                    "or prescribed robot action"
                ),
            }
            if route_discovery
            else {}
        ),
        "current_policy": payload.get("current_policy"),
        "historical_rollout_evaluation": _selection_historical_rollout_summary(
            payload.get("evaluation", {})
        ),
        "visual_observations": observation.get("visual_observations", []),
        "retrieved_specs": observation.get("retrieved_specs", []),
        "scene_prior": payload.get("scene_prior", {}),
        "coordination_primitive_sdk": payload.get(
            "coordination_primitive_sdk",
            coordination_primitive_library_doc(),
        ),
        "blocked_methods_at_this_checkpoint": list(forbidden_methods),
        "blocked_motion_objectives_at_this_checkpoint": list(
            sorted(effective_forbidden_objectives)
        ),
        "nonrelaxable_route_risk_objectives_at_this_checkpoint": list(
            sorted(nonrelaxable_objectives)
        ),
        "cross_primitive_turn_alias_recovery": {
            "active": bool(relaxed_cross_primitive_turn_aliases),
            "relaxed_aliases": sorted(relaxed_cross_primitive_turn_aliases),
            "preserved_rule": (
                "Only shared direction aliases were relaxed after action-space "
                "exhaustion; exact primitive objectives and failed transactions remain blocked."
            ),
        },
        "blocked_strategy_signatures_at_this_checkpoint": list(
            blocked_strategy_signatures
        ),
        "recent_rejected_team_transactions": payload.get(
            "recent_rejected_team_transactions", []
        ),
        "recent_rejected_rollout_evidence": payload.get(
            "recent_rejected_rollout_evidence", []
        ),
        "trusted_checkpoint_linear_motion": trusted_checkpoint_linear_motion,
        "trusted_route_filtered_motion_objectives": sorted(
            set(trusted_route_filtered_objectives)
        ),
        "allowed_methods": method_specs,
        "allowed_strategy_pairs": allowed_strategy_pairs,
        "required_motion_objective_by_method": motion_objectives,
        "body_frame_goal_alignment": body_frame_goal_alignment,
        "alignment_body_frame_geometry": _alignment_body_frame_geometry(runtime_state),
        "current_trusted_route_observation": trusted_route_observation,
        "pre_transport_passage_alignment_gate": passage_alignment_gate,
        "host_geometry_method_constraint": linear_geometry_gate,
        "branch_rewind_context": branch_rewind_context,
        "turn_episode": turn_episode,
        "turn_direction_decision_contract": {
            "decision_owner": "model",
            "available_signed_directions": sorted(
                {
                    str(choice)
                    for method, objective in motion_objectives.items()
                    if method in _SIGNED_TURN_METHODS
                    for choice in objective.get("choices", ())
                    if str(choice) in _SIGNED_TURN_DIRECTIONS
                }
            ),
            "physics_projected_headings": current_execution_state.get(
                "fixed_global_camera_heading_projection",
                {},
            ),
            "evidence_sources": [
                "current trusted payload and robot headings",
                "latest global-camera passage and free-space observation",
                "recent measured world-frame motion from this checkpoint",
            ],
            "required_output_fields_for_arc_or_pivot": [
                "direction_evidence",
                "alternative_direction",
                "alternative_rejection_reason",
            ],
            "required_comparison": (
                "Compare the host-projected clockwise and counterclockwise screen headings "
                "against the immediately connected passage in the current global frame. "
                "State both projected headings in direction_evidence."
            ),
            "prohibited_basis": (
                "default, merely valid, arbitrary, route-hardcoded, or direct conversion "
                "of an image-left/right/up/down word into a body-frame turn sign"
            ),
        },
        "host_authored_wall_clearance": host_wall_clearance,
        "coordination_modes": {
            "sync": (
                "same phase objective, same mover method, coordinated timing; bounded "
                "front/back parameter compensation is allowed"
            ),
            "async": (
                "different temporary roles or objectives: one may hold while the other moves, "
                "or the same mover may use role-specific parameters outside the bounded sync "
                "compensation envelope; identical or bounded-compensation calls are sync, and two "
                "different movers are invalid"
            ),
        },
    }
    selection_context["coordination_modes"] = {
        mode: description
        for mode, description in selection_context["coordination_modes"].items()
        if mode in allowed_coordination_modes
    }
    if formal_model_owns_route and response_audit_path is not None:
        audit_path = response_audit_path.with_name(
            response_audit_path.stem.replace("_rejections", "")
            + "_candidate_space_audit.json"
        )
        write_json(
            audit_path,
            {
                "schema_version": 1,
                "decision_owner": "model",
                "allowed_methods": allowed_methods,
                "allowed_strategy_pairs": allowed_strategy_pairs,
                "allowed_motion_objectives": motion_objectives,
                "hard_exclusions": {
                    "lifecycle_methods": list(forbidden_methods),
                    "immediate_authored_wall_clearance": host_wall_clearance,
                    "exact_rejected_ast_replay": True,
                },
                "soft_exclusions_applied": [],
                "host_semantic_selection_disabled": True,
            },
        )
    base_prompt = build_subtask_selection_user_prompt(selection_context)
    prompt = base_prompt
    last_error = ""
    for response_index in range(max(0, int(response_retries)) + 1):
        raw = client.generate(system=SUBTASK_SELECTION_SYSTEM_PROMPT, user=prompt)
        try:
            parsed = _parse_model_json(raw)
            selected_method = str(parsed.get("selected_method", "")).strip()
            coordination_mode = str(parsed.get("coordination_mode", "")).strip()
            selected_pair = str(parsed.get("strategy_pair", "")).strip()
            if selected_pair:
                if selected_pair not in allowed_strategy_pairs:
                    raise ModelAgentError(
                        "strategy_pair must be one of "
                        f"{allowed_strategy_pairs}; got {selected_pair!r}"
                    )
                coordination_mode, selected_method = selected_pair.split(":", 1)
            elif selected_method in allowed_strategy_pairs:
                # Qwen occasionally places the enum pair in selected_method.
                # It is unambiguous because the exact value is host-allowlisted.
                coordination_mode, selected_method = selected_method.split(":", 1)
            reason = str(parsed.get("reason", "")).strip()
            if selected_method not in allowed_methods:
                raise ModelAgentError(
                    f"selected_method must be one of {allowed_methods}; got {selected_method!r}"
                )
            if coordination_mode not in allowed_coordination_modes:
                raise ModelAgentError(
                    "coordination_mode must be one of "
                    f"{sorted(allowed_coordination_modes)}; got {coordination_mode!r}"
                )
            if selected_method == "hold_position" and coordination_mode == "async":
                raise ModelAgentError(
                    "hold_position as the selected subtask requires sync; async must select its mover method"
                )
            if selected_method.startswith("exit_") and coordination_mode != "sync":
                raise ModelAgentError(
                    "unloaded exit methods require sync with identical front/back parameters"
                )
            strategy_signature = f"{coordination_mode}:{selected_method}"
            if strategy_signature not in allowed_strategy_pairs:
                raise ModelAgentError(
                    f"strategy pair {strategy_signature!r} is unavailable; choose exactly one of "
                    f"{allowed_strategy_pairs}"
                )
            selection = {
                "selected_method": selected_method,
                "coordination_mode": coordination_mode,
                "reason": reason,
            }
            if wall_clearance_escape_directions:
                selection["selection_source"] = (
                    HOST_WALL_CLEARANCE_ESCAPE_SELECTION_SOURCE
                )
            elif wall_turn_escape_active:
                selection["selection_source"] = HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE
            elif route_turn_clearance_override:
                if selected_method in _SIGNED_TURN_METHODS:
                    selection["selection_source"] = (
                        HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE
                    )
                elif selected_method == COORDINATED_LINEAR_TRANSPORT:
                    selection["selection_source"] = (
                        HOST_WALL_CLEARANCE_ESCAPE_SELECTION_SOURCE
                    )
            _validate_selection_current_state_claims(
                selected_method=selected_method,
                reason=reason,
                runtime_state=runtime_state,
            )
            objective_spec = motion_objectives.get(selected_method)
            if objective_spec is not None:
                motion_objective = str(parsed.get("motion_objective", "")).strip()
                choices = list(objective_spec["choices"])
                if motion_objective not in choices:
                    raise ModelAgentError(
                        f"motion_objective for api.{selected_method} must be one of "
                        f"{choices}; got {motion_objective!r}"
                    )
                selection["motion_objective_parameter"] = str(
                    objective_spec["parameter"]
                )
                selection["motion_objective"] = motion_objective
                if not formal_model_owns_route:
                    _validate_selection_against_trusted_route(
                        selected_method=selected_method,
                        motion_objective=motion_objective,
                        trusted_route_observation=runtime_state.get(
                            "trusted_route_observation",
                            {},
                        ),
                        host_wall_clearance=host_wall_clearance,
                    )
                if (
                    not formal_model_owns_route
                    and selected_method == COORDINATED_LINEAR_TRANSPORT
                ):
                    selection.update(
                        _validated_linear_goal_direction(
                            motion_objective,
                            choices=choices,
                            geometry=selection_context["body_frame_goal_alignment"],
                        )
                    )
                if selected_method in _SIGNED_TURN_METHODS:
                    preserve_selected_turn = bool(
                        repair_level in {"coordination", "execution"}
                        and selected_method == current_method
                        and motion_objective == current_motion_objective
                    )
                    if preserve_selected_turn:
                        selection["direction_evidence_source"] = (
                            "preserved_model_selected_turn_during_bounded_repair"
                        )
                    else:
                        selection.update(
                            _validated_turn_direction_evidence(
                                parsed,
                                selected_direction=motion_objective,
                                available_directions=tuple(
                                    str(choice) for choice in choices
                                ),
                                heading_projection=current_execution_state.get(
                                    "fixed_global_camera_heading_projection",
                                    {},
                                )
                                if route_discovery
                                else {},
                            )
                        )
            return selection
        except ModelAgentError as exc:
            last_error = str(exc)
            if response_audit_path is not None:
                _append_candidate_rejection(
                    response_audit_path,
                    response_index=response_index,
                    error=last_error,
                    raw_response=raw,
                )
            if response_index >= max(0, int(response_retries)):
                break
            prompt = build_subtask_selection_retry_prompt(
                base_prompt,
                last_error,
                allowed_strategy_pairs=allowed_strategy_pairs,
                required_motion_objective_by_method=motion_objectives,
                current_execution_state=current_execution_state,
            )
    raise ModelAgentError(
        "model failed compact subtask selection after "
        f"{max(0, int(response_retries)) + 1} response(s): {last_error}"
    )


def _select_coordination_parameters(
    *,
    client: TextGenerationClient,
    selected_subtask: dict[str, str],
    payload: dict[str, Any],
    response_retries: int = 3,
    candidate_validator: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Select bounded geometry/roles after the semantic primitive is fixed."""

    method = str(selected_subtask.get("selected_method", "")).strip()
    mode = str(selected_subtask.get("coordination_mode", "")).strip()
    if method not in COORDINATION_PRIMITIVE_METHODS:
        raise ModelAgentError(
            f"structured coordination parameters require a coordination primitive, got {method!r}"
        )
    if mode not in coordination_modes_for_method(method):
        raise ModelAgentError(
            f"api.{method} cannot execute with coordination_mode={mode!r}"
        )
    observation = payload.get("observation", {})
    observation = observation if isinstance(observation, dict) else {}
    runtime_state = observation.get("runtime_state", {})
    runtime_state = runtime_state if isinstance(runtime_state, dict) else {}
    host_wall_clearance = (
        authored_scene_motion_clearance(runtime_state)
        if bool(payload.get("route_discovery", False))
        else {}
    )
    evaluation = payload.get("evaluation", {})
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    pair_spacing_alignment = _pair_spacing_alignment_parameter_fallback(
        selected_subtask,
        evaluation,
        runtime_state,
    )
    straight_heading_alignment = _straight_heading_alignment_parameter_fallback(
        selected_subtask,
        evaluation,
    )
    endpoint_longitudinal_alignment = (
        _endpoint_longitudinal_alignment_parameter_fallback(
            selected_subtask,
            evaluation,
        )
    )
    heading_correction = _heading_correction_parameter_fallback(
        selected_subtask,
        runtime_state.get("turn_episode", {}),
    )
    host_wall_heading_correction = _host_wall_heading_correction_parameter_fallback(
        selected_subtask
    )
    host_wall_clearance_escape = _host_wall_clearance_escape_parameter_fallback(
        selected_subtask
    )
    route_turn_continuation = _route_turn_continuation_parameter_fallback(
        selected_subtask,
        runtime_state.get("turn_episode", {}),
        host_wall_clearance=host_wall_clearance,
    )
    route_turn_probe = _route_turn_probe_parameter_fallback(
        selected_subtask,
        runtime_state.get("turn_episode", {}),
    )
    verifier_parameters = (
        pair_spacing_alignment
        or endpoint_longitudinal_alignment
        or straight_heading_alignment
        or heading_correction
        or host_wall_heading_correction
        or host_wall_clearance_escape
        or route_turn_continuation
        or route_turn_probe
    )
    if verifier_parameters:
        if candidate_validator is not None:
            candidate_validator(verifier_parameters)
        return verifier_parameters
    evaluation_evidence = evaluation.get("evidence", {})
    evaluation_evidence = (
        evaluation_evidence if isinstance(evaluation_evidence, dict) else {}
    )
    visual_facts = evaluation_evidence.get("visual_verifier", {})
    visual_facts = visual_facts if isinstance(visual_facts, dict) else {}
    target_relation = str(visual_facts.get("target_relation_to_team", "")).strip()
    target_distance_evidence: dict[str, Any] = {
        "source": "latest_global_visual_evaluator",
        "qualitative_relation": target_relation or "unavailable",
        "contract": (
            "Use this only to choose a distance tier after the semantic direction has "
            "already been selected; it is not a route or direction instruction."
        ),
    }
    target_state = runtime_state.get("target", {})
    payload_state = runtime_state.get("payload", {})
    if isinstance(target_state, dict) and isinstance(payload_state, dict):
        try:
            target_distance_evidence["physics_xy_distance_m"] = hypot(
                float(target_state["center_x_m"]) - float(payload_state["x"]),
                float(target_state["center_y_m"]) - float(payload_state["y"]),
            )
            target_distance_evidence["source"] = "current_mujoco_state"
        except (KeyError, TypeError, ValueError):
            pass
    context = {
        "selected_subtask": dict(selected_subtask),
        "parameter_schema": coordination_primitive_method_spec(method)["parameters"],
        "required_output": _coordination_parameter_output_schema(method),
        "current_runtime_state": compact_runtime_state(runtime_state),
        "current_visual_verifier_facts": visual_facts,
        "current_visual_observations": observation.get("visual_observations", []),
        "scene_scale_prior": payload.get("scene_prior", manual_layout_scene_prior()),
        "final_target_distance_evidence": target_distance_evidence,
        "current_state_semantics": (
            "trusted checkpoint after rollback; the next compiled action starts here"
        ),
        "turn_episode": runtime_state.get("turn_episode", {}),
        "failed_rollout_evaluation": payload.get("evaluation", {}),
        "recent_rejected_rollout_evidence": payload.get(
            "recent_rejected_rollout_evidence",
            [],
        ),
        "verified_same_action_exemplars": payload.get(
            "verified_action_exemplars",
            [],
        ),
        "operator_fast_start_seed": payload.get("operator_fast_start_seed", {}),
        "unverified_linear_direction_probe": payload.get(
            "unverified_linear_direction_probe",
            {},
        ),
        "exemplar_use_contract": (
            "Examples are verifier-accepted execution references only. They are already "
            "filtered to this primitive, semantic objective, and coordination mode. Reuse "
            "their executable envelope only after current camera/physics justified the selected "
            "objective; never treat an old success as a current route instruction."
        ),
    }
    turn_angle_envelope = _host_wall_turn_angle_envelope(
        method,
        str(selected_subtask.get("motion_objective", "")),
        host_wall_clearance,
        route_clearance_conflict_contract=payload.get(
            "route_clearance_conflict_contract",
            {},
        ),
    )
    if turn_angle_envelope:
        context["host_wall_turn_angle_envelope"] = turn_angle_envelope
    if bool(payload.get("route_discovery", False)):
        context["global_camera_frame_calibration"] = {
            "screen_right": "world -X",
            "screen_down": "world +Y",
        }
    if method == COORDINATED_ALIGNMENT:
        context["alignment_body_frame_geometry"] = _alignment_body_frame_geometry(
            runtime_state
        )
    if method in {COORDINATED_ARC_TRANSPORT, COORDINATED_PIVOT}:
        context["arc_travel_direction_contract"] = {
            "decision_owner": "model",
            "choices": list(LOADED_TRANSLATION_DIRECTIONS),
            "rule": (
                "Select travel_direction independently from turn_direction by comparing "
                "the robots' live body yaw with the visible passage. Do not default to "
                "body-forward merely because the primitive is an arc. Body-backward loaded "
                "travel is unavailable because the payload is not grasped."
            ),
        }
        if (
            str(selected_subtask.get("selection_source", ""))
            == HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE
        ):
            context["host_wall_turn_escape_contract"] = {
                "reason": (
                    "The current trusted formation has no executable loaded "
                    "forward/left/right translation despite a provisional visual "
                    "passage-aligned label."
                ),
                "decision_owner": "model",
                "required_decisions": [
                    "clockwise or counterclockwise",
                    "front or back as the temporary inner robot",
                ],
                "rule": (
                    "Use current global-view geometry and robot poses. Do not blindly "
                    "continue the previous turn direction or role assignment."
                ),
            }
    base_prompt = build_coordination_parameter_user_prompt(context)
    prompt = base_prompt
    last_error = ""
    for response_index in range(max(0, int(response_retries)) + 1):
        raw = client.generate(
            system=COORDINATION_PARAMETER_SYSTEM_PROMPT,
            user=prompt,
        )
        try:
            parsed = _parse_model_json(raw)
            selected = _normalize_coordination_parameter_payload(
                method=method,
                mode=mode,
                selected_subtask=selected_subtask,
                payload=parsed,
            )
            _validate_selected_coordination_parameters(
                selected,
                selected_subtask=selected_subtask,
                host_wall_turn_angle_envelope=turn_angle_envelope,
            )
            validate_selected_parameters_for_turn_episode(
                selected,
                runtime_state.get("turn_episode", {}),
            )
            if candidate_validator is not None:
                candidate_validator(selected)
            return selected
        except (KeyError, TypeError, ValueError, ModelAgentError) as exc:
            last_error = str(exc)
            if response_index >= max(0, int(response_retries)):
                break
            prompt = build_coordination_parameter_retry_prompt(base_prompt, last_error)
    raise ModelAgentError(
        "model failed structured coordination-parameter selection after "
        f"{max(0, int(response_retries)) + 1} response(s): {last_error}"
    )


def _coordination_parameter_output_schema(method: str) -> dict[str, str]:
    schemas = {
        COORDINATED_LINEAR_TRANSPORT: {
            "distance_m": "number",
            "speed_mps": "number",
            "ramp_s": "number",
            "reason": "string",
        },
        COORDINATED_ARC_TRANSPORT: {
            "travel_direction": "forward, left, or right",
            "angle_deg": "number",
            "duration_s": "number",
            "front_radius_m": "number",
            "back_radius_m": "number",
            "reason": "string",
        },
        COORDINATED_PIVOT: {
            "travel_direction": "forward",
            "angle_deg": "number",
            "duration_s": "number",
            "inner_robot": "front or back",
            "swing_radius_m": "number",
            "reason": "string",
        },
        COORDINATED_ALIGNMENT: {
            "duration_s": "number",
            "front_delta": "number",
            "back_delta": "number",
            "reason": "string",
        },
    }
    return schemas[method]


def _finite_model_number(payload: dict[str, Any], name: str) -> float:
    value = float(payload[name])
    if not isfinite(value):
        raise ModelAgentError(f"{name} must be a finite number")
    return value


def _normalize_coordination_parameter_payload(
    *,
    method: str,
    mode: str,
    selected_subtask: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    objective_parameter = str(
        selected_subtask.get("motion_objective_parameter", "")
    ).strip()
    objective = str(selected_subtask.get("motion_objective", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    selection_source = str(selected_subtask.get("selection_source", "")).strip()
    if selection_source == HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE:
        reason = f"{HOST_WALL_TURN_ESCAPE_MARKER} {reason}".strip()
    if method == COORDINATED_LINEAR_TRANSPORT:
        shared = {
            "direction": objective,
            "distance_m": _finite_model_number(payload, "distance_m"),
            "speed_mps": _finite_model_number(payload, "speed_mps"),
            "ramp_s": _finite_model_number(payload, "ramp_s"),
        }
        robot_parameters = {robot_id: dict(shared) for robot_id in ("front", "back")}
    elif method == COORDINATED_ARC_TRANSPORT:
        shared = {
            "turn_direction": objective,
            "travel_direction": str(payload.get("travel_direction", "forward")).strip(),
            "angle_deg": _finite_model_number(payload, "angle_deg"),
            "duration_s": _finite_model_number(payload, "duration_s"),
        }
        robot_parameters = {
            robot_id: {
                **shared,
                "radius_m": _finite_model_number(payload, f"{robot_id}_radius_m"),
            }
            for robot_id in ("front", "back")
        }
    elif method == COORDINATED_PIVOT:
        inner_robot = str(payload.get("inner_robot", "")).strip()
        if inner_robot not in {"front", "back"}:
            raise ModelAgentError("inner_robot must be exactly 'front' or 'back'")
        shared = {
            "turn_direction": objective,
            "travel_direction": str(payload.get("travel_direction", "forward")).strip(),
            "angle_deg": _finite_model_number(payload, "angle_deg"),
            "duration_s": _finite_model_number(payload, "duration_s"),
            "swing_radius_m": _finite_model_number(payload, "swing_radius_m"),
        }
        robot_parameters = {
            robot_id: {
                **shared,
                "role": "inner" if robot_id == inner_robot else "outer",
            }
            for robot_id in ("front", "back")
        }
    elif method == COORDINATED_ALIGNMENT:
        shared = {
            "axis": objective,
            "duration_s": _finite_model_number(payload, "duration_s"),
        }
        robot_parameters = {
            robot_id: {
                **shared,
                "delta": _finite_model_number(payload, f"{robot_id}_delta"),
            }
            for robot_id in ("front", "back")
        }
    else:  # pragma: no cover - guarded above
        raise ModelAgentError(f"unsupported coordination primitive {method!r}")
    if not objective_parameter or not objective:
        raise ModelAgentError(
            f"api.{method} requires a model-selected semantic objective before parameters"
        )
    return {
        "method": method,
        "coordination_mode": mode,
        "motion_objective_parameter": objective_parameter,
        "motion_objective": objective,
        "robot_parameters": robot_parameters,
        "reason": reason,
        "selection_source": (
            HOST_WALL_TURN_ESCAPE_PARAMETER_SOURCE
            if selection_source == HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE
            else "model"
        ),
    }


def _validate_selected_coordination_parameters(
    selected: dict[str, Any],
    *,
    selected_subtask: dict[str, str],
    host_wall_turn_angle_envelope: dict[str, Any] | None = None,
) -> None:
    method = str(selected.get("method", ""))
    mode = str(selected.get("coordination_mode", ""))
    selection_source = str(selected_subtask.get("selection_source", ""))
    verifier_heading_correction = selection_source in {
        "verifier_heading_correction",
        "host_wall_heading_correction",
    }
    verifier_route_turn_probe = selection_source == "verifier_route_turn_probe"
    if method != str(selected_subtask.get("selected_method", "")):
        raise ModelAgentError(
            "parameter stage changed the selected coordination primitive"
        )
    if mode != str(selected_subtask.get("coordination_mode", "")):
        raise ModelAgentError("parameter stage changed the selected coordination mode")
    robot_parameters = selected.get("robot_parameters", {})
    if not isinstance(robot_parameters, dict) or set(robot_parameters) != {
        "front",
        "back",
    }:
        raise ModelAgentError(
            "coordination parameters require exactly front/back profiles"
        )
    objective_parameter = str(selected.get("motion_objective_parameter", ""))
    objective = str(selected.get("motion_objective", ""))
    for robot_id in ("front", "back"):
        parameters = robot_parameters[robot_id]
        if not isinstance(parameters, dict):
            raise ModelAgentError(
                f"{robot_id} coordination parameters must be an object"
            )
        if str(parameters.get(objective_parameter, "")) != objective:
            raise ModelAgentError(
                f"{robot_id} changed selected {objective_parameter}={objective!r}"
            )
        try:
            validate_coordination_primitive_parameters(method, parameters)
        except ValueError as exc:
            raise ModelAgentError(str(exc)) from exc
    front = robot_parameters["front"]
    back = robot_parameters["back"]
    if method == COORDINATED_LINEAR_TRANSPORT and front != back:
        raise ModelAgentError(
            "initial coordinated linear transport requires one shared sync profile"
        )
    if method == COORDINATED_LINEAR_TRANSPORT and not verifier_route_turn_probe:
        distance_m = float(front["distance_m"])
        if not any(
            abs(distance_m - choice) <= 1.0e-9
            for choice in COORDINATED_LINEAR_DISTANCE_CHOICES_M
        ):
            choices = ", ".join(
                f"{choice:g}" for choice in COORDINATED_LINEAR_DISTANCE_CHOICES_M
            )
            raise ModelAgentError(
                f"model-selected linear distance_m must be exactly one of {choices}m"
            )
        required_speed_mps = VERIFIED_LINEAR_SPEED_MPS
        if abs(float(front["speed_mps"]) - required_speed_mps) > 1.0e-9:
            raise ModelAgentError(
                f"model-selected linear speed_mps must remain {required_speed_mps:.2f}"
            )
        if abs(float(front["ramp_s"]) - VERIFIED_LINEAR_RAMP_S) > 1.0e-9:
            raise ModelAgentError(
                f"model-selected linear ramp_s must remain {VERIFIED_LINEAR_RAMP_S:g}"
            )
    wall_turn_envelope = (
        host_wall_turn_angle_envelope
        if isinstance(host_wall_turn_angle_envelope, dict)
        else {}
    )
    if (
        method in {COORDINATED_ARC_TRANSPORT, COORDINATED_PIVOT}
        and not verifier_heading_correction
    ):
        route_angles = {
            float(front["angle_deg"]),
            float(back["angle_deg"]),
        }
        selected_angle = next(iter(route_angles)) if len(route_angles) == 1 else -1.0
        if wall_turn_envelope:
            try:
                minimum_angle = float(wall_turn_envelope["minimum_angle_deg"])
                maximum_angle = float(wall_turn_envelope["maximum_angle_deg"])
            except (KeyError, TypeError, ValueError):
                raise ModelAgentError(
                    "authored-wall turn-angle envelope is malformed"
                ) from None
            if len(route_angles) != 1 or not (
                minimum_angle <= selected_angle <= maximum_angle + 1.0e-9
            ):
                raise ModelAgentError(
                    "model-selected route turn must obey the supplied authored-wall "
                    f"turn-angle envelope [{minimum_angle:g}, {maximum_angle:g}] "
                    f"degrees; got angle_deg={selected_angle:g}. Preserve the signed "
                    "direction and reduce only the shared angle"
                )
        elif len(route_angles) != 1 or not any(
                abs(selected_angle - choice) <= 1.0e-9
                for choice in MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG
        ):
            choices = ", ".join(
                f"{choice:g}" for choice in MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG
            )
            raise ModelAgentError(
                "model-selected route turns must use one shared angle from "
                f"{choices} degrees or the supplied authored-wall turn-angle "
                "envelope; only a verifier-owned heading correction may otherwise "
                "use another angle"
            )
    if method == COORDINATED_PIVOT and not verifier_heading_correction:
        route_durations = {
            float(front["duration_s"]),
            float(back["duration_s"]),
        }
        if route_durations != {CALIBRATED_MODEL_PIVOT_DURATION_S}:
            raise ModelAgentError(
                "model-selected coordinated_pivot front/back duration_s must "
                f"both remain the calibrated {CALIBRATED_MODEL_PIVOT_DURATION_S:g}s; "
                "only a verifier-owned heading correction may use another duration"
            )
    if method == COORDINATED_ARC_TRANSPORT:
        if str(front["travel_direction"]) != str(back["travel_direction"]):
            raise ModelAgentError(
                "coordinated arc requires one shared body-frame travel_direction"
            )
        if abs(float(front["radius_m"]) - float(back["radius_m"])) < 0.05:
            raise ModelAgentError(
                "coordinated arc front/back radii must differ by at least 0.05m"
            )
    if method == COORDINATED_PIVOT and {
        str(front.get("role")),
        str(back.get("role")),
    } != {"inner", "outer"}:
        raise ModelAgentError(
            "coordinated pivot requires one inner and one outer robot"
        )
    if method == COORDINATED_PIVOT and str(front["travel_direction"]) != str(
        back["travel_direction"]
    ):
        raise ModelAgentError(
            "coordinated pivot requires one shared outer-role travel_direction"
        )
    if method == COORDINATED_PIVOT and str(front["travel_direction"]) != "forward":
        raise ModelAgentError(
            "coordinated pivot requires travel_direction='forward'; the inner robot "
            "must remain translation-stationary and neither robot may back up"
        )
    if (
        method == COORDINATED_ALIGNMENT
        and abs(float(front["delta"]) - float(back["delta"])) <= 1.0e-9
    ):
        raise ModelAgentError("coordinated alignment front/back deltas must differ")
    if (
        method == COORDINATED_ALIGNMENT
        and str(front["axis"]) == "lateral"
        and float(front["delta"]) * float(back["delta"]) < 0.0
    ):
        raise ModelAgentError(
            "loaded lateral alignment cannot command divergent front/back motion; "
            "use one reference robot or coordinated_linear_transport for lateral travel"
        )


def _linear_execution_repair_constraint(
    policy: PolicyScript,
    *,
    repair_level: str,
    failure_type: str,
    selected_subtask: dict[str, str],
    forbidden_transaction_signatures: set[str] | None = None,
) -> dict[str, Any]:
    """Describe one safer linear retry without reopening its semantics."""

    if repair_level != "execution" or failure_type not in _UNSAFE_MAGNITUDE_FAILURES:
        return {}
    if str(selected_subtask.get("selected_method", "")) != COORDINATED_LINEAR_TRANSPORT:
        return {}
    if len(policy.robot_scripts) != 2:
        return {}
    methods = {_first_policy_api_method(script.code) for script in policy.robot_scripts}
    modes = {script.coordination_mode for script in policy.robot_scripts}
    if methods != {COORDINATED_LINEAR_TRANSPORT} or modes != {"sync"}:
        return {}
    arguments = [
        _first_task_call_literal_arguments(script.code)
        for script in policy.robot_scripts
    ]
    try:
        directions = {str(item["direction"]) for item in arguments}
        distances = {float(item["distance_m"]) for item in arguments}
        speeds = {float(item["speed_mps"]) for item in arguments}
        ramps = {float(item["ramp_s"]) for item in arguments}
    except (KeyError, TypeError, ValueError):
        return {}
    if not all(len(values) == 1 for values in (directions, distances, speeds, ramps)):
        return {}
    direction = next(iter(directions))
    if direction != str(selected_subtask.get("motion_objective", "")):
        return {}
    distance_m = next(iter(distances))
    speed_mps = next(iter(speeds))
    ramp_s = next(iter(ramps))
    next_lower_distance_m = _next_lower_unblocked_linear_distance_tier(
        policy,
        distance_m=distance_m,
        forbidden_transaction_signatures=(forbidden_transaction_signatures or set()),
    )
    if next_lower_distance_m is None:
        return {}
    return {
        "kind": "unsafe_execution_repair",
        "method": COORDINATED_LINEAR_TRANSPORT,
        "coordination_mode": "sync",
        "direction": direction,
        "previous_distance_m": distance_m,
        "previous_speed_mps": speed_mps,
        "previous_ramp_s": ramp_s,
        "next_lower_distance_m": next_lower_distance_m,
        "required_speed_mps": VERIFIED_LINEAR_SPEED_MPS,
        "required_ramp_s": VERIFIED_LINEAR_RAMP_S,
        "allowed_repairs": ["reduce_distance_to_next_unblocked_tier"],
    }


def _linear_success_continuation_constraint(
    policy: PolicyScript,
    *,
    report: EvaluationReport,
    selected_subtask: dict[str, str],
) -> dict[str, Any]:
    """Expose the full verified loaded distance set after an accepted action."""

    if not report.success or bool(report.metrics.get("task_complete", 0.0)):
        return {}
    if str(selected_subtask.get("selected_method", "")) != COORDINATED_LINEAR_TRANSPORT:
        return {}
    if str(selected_subtask.get("coordination_mode", "")) != "sync":
        return {}
    if len(policy.robot_scripts) != 2:
        return {}
    methods = {_first_policy_api_method(script.code) for script in policy.robot_scripts}
    modes = {script.coordination_mode for script in policy.robot_scripts}
    if methods != {COORDINATED_LINEAR_TRANSPORT} or modes != {"sync"}:
        return {}
    arguments = [
        _first_task_call_literal_arguments(script.code)
        for script in policy.robot_scripts
    ]
    try:
        directions = {str(item["direction"]) for item in arguments}
        distances = {float(item["distance_m"]) for item in arguments}
        speeds = {float(item["speed_mps"]) for item in arguments}
        ramps = {float(item["ramp_s"]) for item in arguments}
    except (KeyError, TypeError, ValueError):
        return {}
    if not all(len(values) == 1 for values in (directions, distances, speeds, ramps)):
        return {}
    direction = next(iter(directions))
    if direction != str(selected_subtask.get("motion_objective", "")):
        return {}
    previous_distance_m = next(iter(distances))
    maximum_distance_m = VERIFIED_LOADED_LINEAR_MAX_M
    return {
        "kind": "accepted_envelope_continuation",
        "method": COORDINATED_LINEAR_TRANSPORT,
        "coordination_mode": "sync",
        "direction": direction,
        "previous_distance_m": previous_distance_m,
        "maximum_distance_m": maximum_distance_m,
        "allowed_distance_choices_m": [
            choice
            for choice in COORDINATED_LINEAR_DISTANCE_CHOICES_M
            if choice <= maximum_distance_m + 1.0e-9
        ],
        "required_speed_mps": VERIFIED_LINEAR_SPEED_MPS,
        "required_ramp_s": VERIFIED_LINEAR_RAMP_S,
    }


def _trusted_checkpoint_linear_continuation_constraint(
    payload: dict[str, Any],
    *,
    selected_subtask: dict[str, str],
) -> dict[str, Any]:
    """Reuse a loaded linear envelope that created the restored checkpoint."""

    motion = payload.get("trusted_checkpoint_linear_motion", {})
    if not isinstance(motion, dict) or not motion.get("shared_envelope", False):
        return {}
    if (
        str(selected_subtask.get("selected_method", "")) != COORDINATED_LINEAR_TRANSPORT
        or str(selected_subtask.get("coordination_mode", "")) != "sync"
        or str(selected_subtask.get("motion_objective", ""))
        != str(motion.get("direction", ""))
    ):
        return {}
    try:
        distance_m = float(motion["distance_m"])
        float(motion["speed_mps"])
        float(motion["ramp_s"])
    except (KeyError, TypeError, ValueError):
        return {}
    maximum_distance_m = VERIFIED_LOADED_LINEAR_MAX_M
    return {
        "kind": "accepted_envelope_continuation",
        "method": COORDINATED_LINEAR_TRANSPORT,
        "coordination_mode": "sync",
        "direction": str(motion["direction"]),
        "previous_distance_m": distance_m,
        "maximum_distance_m": maximum_distance_m,
        "allowed_distance_choices_m": [
            choice
            for choice in COORDINATED_LINEAR_DISTANCE_CHOICES_M
            if choice <= maximum_distance_m + 1.0e-9
        ],
        "required_speed_mps": VERIFIED_LINEAR_SPEED_MPS,
        "required_ramp_s": VERIFIED_LINEAR_RAMP_S,
        "source_checkpoint_id": str(motion.get("checkpoint_id", "")),
        "source_script_id": str(motion.get("source_script_id", "")),
    }


def _validate_linear_success_continuation_parameters(
    selected: dict[str, Any],
    *,
    constraint: dict[str, Any],
) -> None:
    """Keep loaded linear choices inside the verified 0.25-1.0m envelope."""

    if constraint.get("kind") != "accepted_envelope_continuation":
        return
    robot_parameters = selected.get("robot_parameters", {})
    violations: list[str] = []
    for robot_id in ("front", "back"):
        parameters = robot_parameters.get(robot_id, {})
        try:
            distance_m = float(parameters["distance_m"])
            speed_mps = float(parameters["speed_mps"])
            ramp_s = float(parameters["ramp_s"])
        except (KeyError, TypeError, ValueError):
            violations.append(f"{robot_id} must provide literal linear parameters")
            continue
        if str(parameters.get("direction", "")) != str(constraint["direction"]):
            violations.append(f"{robot_id} changed the accepted direction")
        allowed_distances = tuple(
            float(value) for value in constraint.get("allowed_distance_choices_m", ())
        )
        if not any(abs(distance_m - value) <= 1.0e-9 for value in allowed_distances):
            violations.append(
                f"{robot_id} distance_m is outside the accepted tier continuation"
            )
        if abs(speed_mps - float(constraint["required_speed_mps"])) > 1.0e-9:
            violations.append(f"{robot_id} changed the accepted speed_mps")
        if abs(ramp_s - float(constraint["required_ramp_s"])) > 1.0e-9:
            violations.append(f"{robot_id} changed the accepted ramp_s")
    if violations:
        raise ModelAgentError(
            "same-direction loaded linear continuation must use an allowed 0.25-1.0m "
            "distance while preserving speed_mps=0.50 and ramp_s=1.5; "
            + "; ".join(dict.fromkeys(violations))
        )


def _conservative_linear_parameters(
    constraint: dict[str, Any],
) -> dict[str, Any]:
    """Return an audited linear fallback after model literal validation fails."""

    kind = str(constraint.get("kind", ""))
    if kind == "unsafe_execution_repair":
        distance_m = float(constraint["next_lower_distance_m"])
        speed_mps = VERIFIED_LINEAR_SPEED_MPS
        ramp_s = VERIFIED_LINEAR_RAMP_S
        reason = (
            "Conservative controller fallback after invalid model literals: reduce the "
            "failed linear segment by one supported tier while preserving the verified gait."
        )
    elif kind == "accepted_envelope_continuation":
        distance_m = _linear_distance_tier_at_or_below(
            float(
                constraint.get("previous_distance_m", constraint["maximum_distance_m"])
            )
        )
        speed_mps = VERIFIED_LINEAR_SPEED_MPS
        ramp_s = VERIFIED_LINEAR_RAMP_S
        reason = (
            "Conservative controller fallback after invalid model literals: repeat the "
            "latest physics-accepted loaded linear envelope."
        )
    else:
        return {}
    shared = {
        "direction": str(constraint["direction"]),
        "distance_m": distance_m,
        "speed_mps": speed_mps,
        "ramp_s": ramp_s,
    }
    return {
        "method": COORDINATED_LINEAR_TRANSPORT,
        "coordination_mode": "sync",
        "motion_objective_parameter": "direction",
        "motion_objective": str(constraint["direction"]),
        "robot_parameters": {robot_id: dict(shared) for robot_id in ("front", "back")},
        "reason": reason,
        "selection_source": "controller_fallback_after_model_validation",
    }


def _linear_distance_tier_at_or_below(distance_m: float) -> float:
    eligible = [
        choice
        for choice in COORDINATED_LINEAR_DISTANCE_CHOICES_M
        if choice <= float(distance_m) + 1.0e-9
    ]
    return max(eligible, default=COORDINATED_LINEAR_DISTANCE_CHOICES_M[0])


def _unverified_linear_direction_probe_constraint(
    policy: PolicyScript,
    *,
    selected_subtask: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Bound the first loaded rollout on a newly selected body-frame axis."""

    if (
        selected_subtask.get("selected_method") != COORDINATED_LINEAR_TRANSPORT
        or payload.get("operator_fast_start_seed")
        or selected_subtask.get("selection_source") == "verifier_route_turn_probe"
    ):
        return {}
    selected_direction = str(selected_subtask.get("motion_objective", "")).strip()
    if selected_direction not in LOADED_TRANSLATION_DIRECTIONS:
        return {}
    current_direction = _policy_linear_objective(policy)
    if current_direction == selected_direction:
        return {}
    exemplars = payload.get("verified_same_action_exemplars", ())
    if isinstance(exemplars, (list, tuple)) and exemplars:
        return {}
    return {
        "active": True,
        "method": COORDINATED_LINEAR_TRANSPORT,
        "direction": selected_direction,
        # Keep the first probe useful enough to reveal SONIC motion. This is
        # deliberately independent of tuple ordering so adding a smaller
        # correction tier cannot silently turn exploration into micro-actions.
        "required_distance_m": VERIFIED_LOADED_LINEAR_MAX_M,
        "required_speed_mps": VERIFIED_LINEAR_SPEED_MPS,
        "required_ramp_s": VERIFIED_LINEAR_RAMP_S,
        "reason": (
            "first loaded rollout on a new body-frame direction without a "
            "verifier-accepted same-action exemplar"
        ),
    }


def _validate_unverified_linear_direction_probe_parameters(
    candidate: dict[str, Any],
    *,
    constraint: dict[str, Any],
) -> None:
    if candidate.get("method") != COORDINATED_LINEAR_TRANSPORT:
        return
    robot_parameters = candidate.get("robot_parameters", {})
    required_direction = str(constraint["direction"])
    required_distance = float(constraint["required_distance_m"])
    required_speed = float(constraint["required_speed_mps"])
    required_ramp = float(constraint["required_ramp_s"])
    for robot_id in ("front", "back"):
        parameters = robot_parameters.get(robot_id, {})
        if str(parameters.get("direction", "")) != required_direction:
            raise ModelAgentError(
                f"new-direction probe must keep direction={required_direction!r}"
            )
        if abs(float(parameters.get("distance_m", 0.0)) - required_distance) > 1.0e-9:
            raise ModelAgentError(
                f"first unverified {required_direction} rollout requires "
                f"distance_m={required_distance:g}"
            )
        if abs(float(parameters.get("speed_mps", 0.0)) - required_speed) > 1.0e-9:
            raise ModelAgentError(
                f"new-direction probe requires speed_mps={required_speed:g}"
            )
        if abs(float(parameters.get("ramp_s", 0.0)) - required_ramp) > 1.0e-9:
            raise ModelAgentError(
                f"new-direction probe requires ramp_s={required_ramp:g}"
            )


def _next_lower_linear_distance_tier(distance_m: float) -> float | None:
    eligible = [
        choice
        for choice in COORDINATED_LINEAR_DISTANCE_CHOICES_M
        if choice < float(distance_m) - 1.0e-9
    ]
    return max(eligible) if eligible else None


def _next_lower_unblocked_linear_distance_tier(
    policy: PolicyScript,
    *,
    distance_m: float,
    forbidden_transaction_signatures: set[str],
) -> float | None:
    """Skip lower distance tiers already rejected at the same checkpoint."""

    eligible = sorted(
        (
            choice
            for choice in _linear_distance_choices_for_policy(policy)
            if choice < float(distance_m) - 1.0e-9
        ),
        reverse=True,
    )
    for choice in eligible:
        candidate = _policy_with_linear_distance(policy, distance_m=choice)
        signature = _policy_transaction_signature(candidate)
        if not signature or signature not in forbidden_transaction_signatures:
            return choice
    return None


def _linear_distance_choices_for_policy(
    policy: PolicyScript,
) -> tuple[float, ...]:
    """Return the bounded distance tiers available to this loaded policy."""

    if not _policy_is_final_approach(policy):
        return COORDINATED_LINEAR_DISTANCE_CHOICES_M
    return tuple(
        sorted(
            {
                *COORDINATED_LINEAR_DISTANCE_CHOICES_M,
                *FINAL_APPROACH_LINEAR_DISTANCE_CHOICES_M,
            }
        )
    )


def _policy_with_linear_distance(
    policy: PolicyScript,
    *,
    distance_m: float,
) -> PolicyScript:
    scripts = tuple(
        replace(
            script,
            code=_replace_first_task_call_keyword_literal(
                script.code,
                keyword_name="distance_m",
                value=float(distance_m),
            ),
        )
        for script in policy.robot_scripts
    )
    return replace(
        policy,
        code=(
            _replace_first_task_call_keyword_literal(
                policy.code,
                keyword_name="distance_m",
                value=float(distance_m),
            )
            if policy.code
            else policy.code
        ),
        robot_scripts=scripts,
    )


def _replace_first_task_call_keyword_literal(
    code: str,
    *,
    keyword_name: str,
    value: object,
) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            not isinstance(owner, ast.Name)
            or owner.id != "api"
            or node.func.attr == "stop_all"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == keyword_name:
                keyword.value = ast.copy_location(
                    ast.Constant(value=value), keyword.value
                )
                ast.fix_missing_locations(tree)
                return ast.unparse(tree)
        return code
    return code


def _next_higher_linear_distance_tier(distance_m: float) -> float | None:
    for choice in COORDINATED_LINEAR_DISTANCE_CHOICES_M:
        if choice > float(distance_m) + 1.0e-9:
            return choice
    return None


def _conservative_coordination_parameter_candidates(
    selected_subtask: dict[str, str],
) -> tuple[dict[str, Any], ...]:
    """Compile bounded SDK defaults without treating malformed model JSON as physics."""

    method = str(selected_subtask.get("selected_method", "")).strip()
    mode = str(selected_subtask.get("coordination_mode", "")).strip()
    objective = str(selected_subtask.get("motion_objective", "")).strip()
    objective_parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(method, "")
    if (
        method not in COORDINATION_PRIMITIVE_METHODS
        or not objective
        or not objective_parameter
    ):
        return ()

    def payload(
        robot_parameters: dict[str, dict[str, Any]], reason: str
    ) -> dict[str, Any]:
        return {
            "method": method,
            "coordination_mode": mode,
            "motion_objective_parameter": objective_parameter,
            "motion_objective": objective,
            "robot_parameters": robot_parameters,
            "reason": reason,
            "selection_source": "controller_fallback_after_model_validation",
        }

    reason = (
        "Host compiler fallback: preserve the model-selected primitive and semantic "
        "objective while using one bounded SDK parameter profile."
    )
    candidates: list[dict[str, Any]] = []
    if method == COORDINATED_LINEAR_TRANSPORT:
        for distance_m in (2.0, 1.5, 1.0, 0.5):
            shared = {
                "direction": objective,
                "distance_m": distance_m,
                "speed_mps": VERIFIED_LINEAR_SPEED_MPS,
                "ramp_s": VERIFIED_LINEAR_RAMP_S,
            }
            candidates.append(
                payload(
                    {robot_id: dict(shared) for robot_id in ("front", "back")},
                    reason,
                )
            )
    elif method == COORDINATED_ARC_TRANSPORT:
        for duration_s, inner_radius_m, outer_radius_m in (
            (4.0, 0.4, 0.8),
            (6.0, 0.4, 0.8),
            (4.0, 0.4, 1.2),
        ):
            shared = {
                "turn_direction": objective,
                "travel_direction": "forward",
                "angle_deg": 5.0,
                "duration_s": duration_s,
            }
            for inner_robot in ("front", "back"):
                outer_robot = "back" if inner_robot == "front" else "front"
                candidates.append(
                    payload(
                        {
                            inner_robot: {**shared, "radius_m": inner_radius_m},
                            outer_robot: {**shared, "radius_m": outer_radius_m},
                        },
                        reason,
                    )
                )
    elif method == COORDINATED_PIVOT:
        shared = {
            "turn_direction": objective,
            "travel_direction": "forward",
            "angle_deg": MODEL_ROUTE_TURN_ANGLE_DEG,
            "duration_s": CALIBRATED_MODEL_PIVOT_DURATION_S,
            "swing_radius_m": 0.8,
        }
        for inner_robot in ("front", "back"):
            outer_robot = "back" if inner_robot == "front" else "front"
            candidates.append(
                payload(
                    {
                        inner_robot: {**shared, "role": "inner"},
                        outer_robot: {**shared, "role": "outer"},
                    },
                    reason,
                )
            )
    elif method == COORDINATED_ALIGNMENT:
        magnitude = 5.0 if objective == "yaw" else 0.05
        for moving_robot in ("front", "back"):
            reference_robot = "back" if moving_robot == "front" else "front"
            for signed_delta in (magnitude, -magnitude):
                shared = {"axis": objective, "duration_s": 2.0}
                candidates.append(
                    payload(
                        {
                            moving_robot: {**shared, "delta": signed_delta},
                            reference_robot: {**shared, "delta": 0.0},
                        },
                        reason,
                    )
                )
    return tuple(candidates)


def _heading_correction_parameter_fallback(
    selected_subtask: dict[str, str],
    turn_episode: object,
) -> dict[str, Any]:
    """Compile the verifier-owned one-shot drift correction without model tuning."""

    episode = turn_episode if isinstance(turn_episode, dict) else {}
    if (
        not bool(episode.get("active", False))
        or str(episode.get("purpose", "")) != "heading_correction"
        or str(selected_subtask.get("selected_method", "")) != COORDINATED_PIVOT
        or str(selected_subtask.get("coordination_mode", "")) != "async"
    ):
        return {}
    direction = str(episode.get("direction", ""))
    roles = episode.get("roles", {})
    roles = roles if isinstance(roles, dict) else {}
    try:
        angle_deg = float(episode["correction_angle_deg"])
    except (KeyError, TypeError, ValueError):
        return {}
    if (
        direction not in _SIGNED_TURN_DIRECTIONS
        or set(roles) != {"front", "back"}
        or set(roles.values()) != {"inner", "outer"}
        or angle_deg <= 0.0
    ):
        return {}
    shared = {
        "turn_direction": direction,
        "travel_direction": "forward",
        "angle_deg": angle_deg,
        "duration_s": 4.0,
        "swing_radius_m": 0.8,
    }
    return {
        "method": COORDINATED_PIVOT,
        "coordination_mode": "async",
        "motion_objective_parameter": "turn_direction",
        "motion_objective": direction,
        "robot_parameters": {
            robot_id: {**shared, "role": str(roles[robot_id])}
            for robot_id in ("front", "back")
        },
        "reason": (
            "Host fallback: execute the verifier-owned one-shot heading correction "
            "without promoting it to a route turn."
        ),
        "selection_source": "verifier_heading_correction_fallback",
    }


def _host_wall_heading_correction_parameter_fallback(
    selected_subtask: dict[str, Any],
) -> dict[str, Any]:
    """Compile one geometry-owned small yaw correction before wall-clear travel."""

    if (
        str(selected_subtask.get("selection_source", ""))
        != HOST_WALL_HEADING_CORRECTION_SELECTION_SOURCE
        or str(selected_subtask.get("selected_method", "")) != COORDINATED_PIVOT
        or str(selected_subtask.get("coordination_mode", "")) != "async"
    ):
        return {}
    direction = str(selected_subtask.get("turn_direction", ""))
    if not direction:
        direction = str(selected_subtask.get("motion_objective", ""))
    roles = {
        str(selected_subtask.get("inner_robot", "")): "inner",
        str(selected_subtask.get("outer_robot", "")): "outer",
    }
    try:
        angle_deg = float(selected_subtask["angle_deg"])
    except (KeyError, TypeError, ValueError):
        return {}
    if (
        direction not in _SIGNED_TURN_DIRECTIONS
        or set(roles) != {"front", "back"}
        or set(roles.values()) != {"inner", "outer"}
        or not 2.0 <= angle_deg <= 15.0
    ):
        return {}
    shared = {
        "turn_direction": direction,
        "travel_direction": "forward",
        "angle_deg": angle_deg,
        "duration_s": 4.0,
        "swing_radius_m": 0.8,
    }
    return {
        "method": COORDINATED_PIVOT,
        "coordination_mode": "async",
        "motion_objective_parameter": "turn_direction",
        "motion_objective": direction,
        "robot_parameters": {
            robot_id: {**shared, "role": role} for robot_id, role in roles.items()
        },
        "reason": (
            f"{HOST_WALL_HEADING_CORRECTION_MARKER} Execute the smallest authored-wall "
            "clearance correction before resuming the blocked translation."
        ),
        "selection_source": HOST_WALL_HEADING_CORRECTION_PARAMETER_SOURCE,
    }


def _host_wall_clearance_escape_parameter_fallback(
    selected_subtask: dict[str, Any],
) -> dict[str, Any]:
    """Compile one geometry-owned translation that creates room to turn."""

    if (
        str(selected_subtask.get("selection_source", ""))
        != HOST_WALL_CLEARANCE_ESCAPE_SELECTION_SOURCE
        or str(selected_subtask.get("selected_method", ""))
        != COORDINATED_LINEAR_TRANSPORT
        or str(selected_subtask.get("coordination_mode", "")) != "sync"
    ):
        return {}
    direction = str(selected_subtask.get("motion_objective", "")).strip()
    if direction not in LOADED_TRANSLATION_DIRECTIONS:
        return {}
    shared = {
        "direction": direction,
        "distance_m": 1.0,
        "speed_mps": 0.5,
        "ramp_s": 1.5,
    }
    return {
        "method": COORDINATED_LINEAR_TRANSPORT,
        "coordination_mode": "sync",
        "motion_objective_parameter": "direction",
        "motion_objective": direction,
        "robot_parameters": {robot_id: dict(shared) for robot_id in ("front", "back")},
        "reason": (
            f"{HOST_WALL_CLEARANCE_ESCAPE_MARKER} Both signed turn headings are "
            "locally wall-blocked; execute one synchronized open-direction "
            "translation to create turning clearance."
        ),
        "selection_source": HOST_WALL_CLEARANCE_ESCAPE_PARAMETER_SOURCE,
    }


def _straight_heading_alignment_parameter_fallback(
    selected_subtask: dict[str, str],
    evaluation: object,
) -> dict[str, Any]:
    """Compile a measured front/back yaw split without another model call."""

    if (
        str(selected_subtask.get("selection_source", ""))
        != "verifier_straight_heading_alignment"
        or str(selected_subtask.get("selected_method", "")) != COORDINATED_ALIGNMENT
        or str(selected_subtask.get("coordination_mode", "")) != "async"
        or str(selected_subtask.get("motion_objective", "")) != "yaw"
    ):
        return {}
    deltas = _verifier_straight_yaw_alignment(evaluation)
    if not deltas:
        return {}
    shared = {"axis": "yaw", "duration_s": 2.0}
    return {
        "method": COORDINATED_ALIGNMENT,
        "coordination_mode": "async",
        "motion_objective_parameter": "axis",
        "motion_objective": "yaw",
        "robot_parameters": {
            robot_id: {**shared, "delta": deltas[robot_id]}
            for robot_id in ("front", "back")
        },
        "reason": (
            "Host fallback: execute exactly the verifier-measured relative-yaw "
            "correction before retrying translation."
        ),
        "selection_source": "verifier_straight_heading_alignment_fallback",
    }


def _pair_spacing_alignment_parameter_fallback(
    selected_subtask: dict[str, str],
    evaluation: object,
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    """Compile one verifier-owned spacing repair without another model call."""

    if (
        str(selected_subtask.get("selection_source", ""))
        != "verifier_pair_spacing_alignment"
        or str(selected_subtask.get("selected_method", "")) != COORDINATED_ALIGNMENT
        or str(selected_subtask.get("coordination_mode", "")) != "async"
    ):
        return {}
    repair = _verifier_pair_spacing_alignment(evaluation, runtime_state)
    if not repair or str(selected_subtask.get("motion_objective", "")) != str(
        repair["axis"]
    ):
        return {}
    shared = {"axis": str(repair["axis"]), "duration_s": 1.0}
    deltas = repair["robot_deltas_m"]
    return {
        "method": COORDINATED_ALIGNMENT,
        "coordination_mode": "async",
        "motion_objective_parameter": "axis",
        "motion_objective": str(repair["axis"]),
        "robot_parameters": {
            robot_id: {**shared, "delta": float(deltas[robot_id])}
            for robot_id in ("front", "back")
        },
        "reason": (
            "Host fallback: keep one robot stationary and apply exactly one local-body "
            "correction whose world projection increases robot-pair separation."
        ),
        "selection_source": "verifier_pair_spacing_alignment_fallback",
    }


def _endpoint_longitudinal_alignment_parameter_fallback(
    selected_subtask: dict[str, str],
    evaluation: object,
) -> dict[str, Any]:
    """Compile the verifier-measured lagger catch-up without another model call."""

    if (
        str(selected_subtask.get("selection_source", ""))
        != "verifier_endpoint_longitudinal_alignment"
        or str(selected_subtask.get("selected_method", "")) != COORDINATED_ALIGNMENT
        or str(selected_subtask.get("coordination_mode", "")) != "async"
        or str(selected_subtask.get("motion_objective", "")) != "longitudinal"
    ):
        return {}
    deltas = _verifier_endpoint_longitudinal_alignment(evaluation)
    if not deltas:
        return {}
    shared = {"axis": "longitudinal", "duration_s": 1.0}
    return {
        "method": COORDINATED_ALIGNMENT,
        "coordination_mode": "async",
        "motion_objective_parameter": "axis",
        "motion_objective": "longitudinal",
        "robot_parameters": {
            robot_id: {**shared, "delta": deltas[robot_id]}
            for robot_id in ("front", "back")
        },
        "reason": (
            "Host fallback: keep the accepted checkpoint and execute exactly one "
            "forward-only lagging-robot catch-up."
        ),
        "selection_source": "verifier_endpoint_longitudinal_alignment_fallback",
    }


def _route_turn_continuation_parameter_fallback(
    selected_subtask: dict[str, str],
    turn_episode: object,
    *,
    host_wall_clearance: object = None,
) -> dict[str, Any]:
    """Continue one accepted signed route-turn relation without model role drift."""

    episode = turn_episode if isinstance(turn_episode, dict) else {}
    if (
        not bool(episode.get("active", False))
        or str(episode.get("purpose", "")) != "route_turn"
        or str(episode.get("status", "turning")) != "turning"
        or bool(episode.get("switch_allowed", False))
        or str(selected_subtask.get("selection_source", ""))
        != "verifier_route_turn_continuation"
    ):
        return {}
    method = str(episode.get("primitive", ""))
    direction = str(episode.get("direction", ""))
    roles = episode.get("roles", {})
    roles = roles if isinstance(roles, dict) else {}
    try:
        angle_deg = float(episode.get("last_angle_deg", 30.0))
    except (TypeError, ValueError):
        return {}
    if (
        method != COORDINATED_PIVOT
        or direction not in _SIGNED_TURN_DIRECTIONS
        or set(roles) != {"front", "back"}
        or set(roles.values()) != {"inner", "outer"}
        or not 2.0 <= angle_deg <= 30.0
    ):
        return {}
    maximum_safe_angle = maximum_safe_turn_angle_deg(
        host_wall_clearance,
        direction,
    )
    if maximum_safe_angle is not None:
        if maximum_safe_angle < 2.0:
            return {}
        angle_deg = min(angle_deg, maximum_safe_angle)
    shared = {
        "turn_direction": direction,
        "travel_direction": "forward",
        "angle_deg": angle_deg,
        "duration_s": CALIBRATED_MODEL_PIVOT_DURATION_S,
        "swing_radius_m": 0.8,
    }
    return {
        "method": method,
        "coordination_mode": "async",
        "motion_objective_parameter": "turn_direction",
        "motion_objective": direction,
        "robot_parameters": {
            robot_id: {**shared, "role": str(roles[robot_id])}
            for robot_id in ("front", "back")
        },
        "reason": (
            "Continue the verifier-accepted route-turn direction and inner/outer "
            "roles; cap only angle_deg when the current finite authored-wall pivot "
            "sweep is tighter. Only a physical or visual rejection may switch direction."
        ),
        "selection_source": "verifier_route_turn_continuation_fallback",
    }


def _route_turn_probe_parameter_fallback(
    selected_subtask: dict[str, str],
    turn_episode: object,
) -> dict[str, Any]:
    """Compile the bounded post-turn translation without model direction drift."""

    episode = turn_episode if isinstance(turn_episode, dict) else {}
    if (
        not (
            bool(episode.get("translation_pending", False))
            or (
                bool(episode.get("active", False))
                and str(episode.get("status", "")) == "probe_ready"
            )
        )
        or str(episode.get("purpose", "")) != "route_turn"
        or str(episode.get("status", "")) not in {"probe_ready", "completed"}
        or bool(episode.get("switch_allowed", False))
        or str(selected_subtask.get("selection_source", ""))
        != "verifier_route_turn_probe"
    ):
        return {}
    direction = str(episode.get("probe_direction", "forward"))
    try:
        maximum_distance_m = float(episode.get("probe_max_distance_m", 1.0))
    except (TypeError, ValueError):
        return {}
    if direction != "forward" or maximum_distance_m <= 0.0:
        return {}
    distance_m = min(1.0, maximum_distance_m)
    shared = {
        "direction": direction,
        "distance_m": distance_m,
        # Both frozen SONIC policies reliably enter gait at the verified 0.50
        # command. A 0.20 probe repeatedly left one backend below its gait-start
        # threshold while the other walked, so nominal distance remains only a
        # bounded planner envelope and physics monitors the resulting rollout.
        "speed_mps": VERIFIED_LINEAR_SPEED_MPS,
        "ramp_s": 1.5,
    }
    return {
        "method": COORDINATED_LINEAR_TRANSPORT,
        "coordination_mode": "sync",
        "motion_objective_parameter": "direction",
        "motion_objective": direction,
        "robot_parameters": {robot_id: dict(shared) for robot_id in ("front", "back")},
        "reason": (
            "Probe the verified post-turn heading with one bounded synchronized "
            "translation before reopening route search."
        ),
        "selection_source": "verifier_route_turn_probe_fallback",
    }


def _select_sync_compensation_parameters(
    *,
    client: TextGenerationClient,
    constraint: dict[str, Any],
    candidate_validator: Callable[[dict[str, Any]], None] | None = None,
    response_retries: int = 3,
) -> dict[str, Any]:
    """Ask the same model to solve only the measured two-executor calibration."""

    context = {
        **constraint,
        "required_relation": (
            f"{constraint['farther_robot']}_speed_mps < "
            f"{constraint['nearer_robot']}_speed_mps == "
            f"{constraint['required_speed_mps']} and both distance_m == "
            f"{constraint['required_distance_m']} and abs(front_speed_mps - "
            f"back_speed_mps) <= {constraint['maximum_pair_speed_delta_mps']}"
        ),
        "measured_distance_ratio": round(
            float(constraint["nearer_displacement_m"])
            / float(constraint["farther_displacement_m"]),
            3,
        ),
        "ramp_s": 1.5,
    }
    base_prompt = build_sync_compensation_user_prompt(context)
    prompt = base_prompt
    last_error = ""
    last_candidate: dict[str, Any] = {}
    for response_index in range(max(0, int(response_retries)) + 1):
        raw = client.generate(system=SYNC_COMPENSATION_SYSTEM_PROMPT, user=prompt)
        try:
            parsed = _parse_model_json(raw)
            last_candidate = parsed
            selected = {
                "front_distance_m": float(parsed["front_distance_m"]),
                "back_distance_m": float(parsed["back_distance_m"]),
                "front_speed_mps": float(parsed["front_speed_mps"]),
                "back_speed_mps": float(parsed["back_speed_mps"]),
                "reason": str(parsed.get("reason", "")).strip(),
            }
            _validate_selected_sync_parameter_values(
                selected,
                constraint=constraint,
            )
            if candidate_validator is not None:
                candidate_validator(selected)
            return selected
        except (KeyError, TypeError, ValueError, ModelAgentError) as exc:
            last_error = str(exc)
            if response_index >= max(0, int(response_retries)):
                break
            prompt = build_sync_compensation_retry_prompt(base_prompt, last_error)
    normalized = _normalize_sync_parameter_candidate(
        last_candidate,
        constraint=constraint,
    )
    if normalized:
        try:
            if candidate_validator is not None:
                candidate_validator(normalized)
            return normalized
        except ModelAgentError as exc:
            last_error = str(exc)
    raise ModelAgentError(
        "model failed measured sync-parameter selection after "
        f"{max(0, int(response_retries)) + 1} response(s): {last_error}"
    )


def _turn_execution_repair_constraint(
    policy: PolicyScript,
    *,
    repair_level: str,
    failure_type: str,
) -> dict[str, Any]:
    """Describe one execution-only arc/pivot repair without reopening strategy."""

    if repair_level != "execution" or failure_type not in _UNSAFE_MAGNITUDE_FAILURES:
        return {}
    if len(policy.robot_scripts) != 2:
        return {}
    methods = {_first_policy_api_method(script.code) for script in policy.robot_scripts}
    if len(methods) != 1:
        return {}
    method = next(iter(methods))
    if method not in {
        COORDINATED_ARC_TRANSPORT,
        COORDINATED_PIVOT,
    }:
        return {}
    arguments = [
        _first_task_call_literal_arguments(script.code)
        for script in policy.robot_scripts
    ]
    try:
        angles = {float(item["angle_deg"]) for item in arguments}
        durations = {float(item["duration_s"]) for item in arguments}
    except (KeyError, TypeError, ValueError):
        return {}
    if len(angles) != 1 or len(durations) != 1:
        return {}
    angle_deg = next(iter(angles))
    duration_s = next(iter(durations))
    spec = coordination_primitive_method_spec(method)["parameters"]
    minimum_angle_deg = float(spec["angle_deg"]["min"])
    maximum_angle_deg = float(spec["angle_deg"]["max"])
    minimum_duration_s = float(spec["duration_s"]["min"])
    can_reduce_angle = angle_deg > minimum_angle_deg + 1.0e-9
    if not can_reduce_angle:
        return {}
    return {
        "method": method,
        "previous_angle_deg": angle_deg,
        "previous_duration_s": duration_s,
        "minimum_angle_deg": minimum_angle_deg,
        "maximum_angle_deg": maximum_angle_deg,
        "maximum_next_angle_deg": max(minimum_angle_deg, 0.5 * angle_deg),
        "minimum_duration_s": minimum_duration_s,
        "maximum_duration_s": float(spec["duration_s"]["max"]),
        "allowed_repairs": ["reduce_angle"],
        "hard_rule": (
            "reduce angle_deg to at most maximum_next_angle_deg with duration_s fixed; "
            "at minimum angle the host must switch coordination"
        ),
    }


def _select_turn_execution_parameters(
    *,
    client: TextGenerationClient,
    constraint: dict[str, Any],
    response_retries: int = 3,
) -> dict[str, Any]:
    """Ask the model for only two literals, then normalize inside the safe envelope."""

    base_prompt = build_turn_execution_parameter_user_prompt(constraint)
    prompt = base_prompt
    last_error = ""
    for response_index in range(max(0, int(response_retries)) + 1):
        raw = client.generate(
            system=TURN_EXECUTION_PARAMETER_SYSTEM_PROMPT,
            user=prompt,
        )
        try:
            parsed = _parse_model_json(raw)
            selected = {
                "repair": str(parsed.get("repair", "")).strip(),
                "angle_deg": float(parsed["angle_deg"]),
                "duration_s": float(parsed["duration_s"]),
                "reason": str(parsed.get("reason", "")).strip(),
                "selection_source": "model",
            }
            _validate_selected_turn_execution_parameters(
                selected,
                constraint=constraint,
            )
            return selected
        except (KeyError, TypeError, ValueError, ModelAgentError) as exc:
            last_error = str(exc)
            if response_index >= max(0, int(response_retries)):
                break
            prompt = build_turn_execution_parameter_retry_prompt(
                base_prompt,
                last_error,
            )
    fallback = _conservative_turn_execution_parameters(constraint)
    if fallback:
        fallback["model_selection_error"] = last_error
        return fallback
    raise ModelAgentError(
        "model failed bounded turn-parameter selection after "
        f"{max(0, int(response_retries)) + 1} response(s): {last_error}"
    )


def _validate_selected_turn_execution_parameters(
    selected: dict[str, Any],
    *,
    constraint: dict[str, Any],
) -> None:
    previous_angle = float(constraint["previous_angle_deg"])
    previous_duration = float(constraint["previous_duration_s"])
    angle = float(selected["angle_deg"])
    duration = float(selected["duration_s"])
    minimum_angle = float(constraint["minimum_angle_deg"])
    maximum_angle = float(constraint["maximum_angle_deg"])
    minimum_duration = float(constraint["minimum_duration_s"])
    maximum_duration = float(constraint["maximum_duration_s"])
    if not minimum_angle <= angle <= maximum_angle:
        raise ModelAgentError(
            f"angle_deg={angle:g} must be within [{minimum_angle:g}, {maximum_angle:g}]"
        )
    if not minimum_duration <= duration <= maximum_duration:
        raise ModelAgentError(
            f"duration_s={duration:g} must be within "
            f"[{minimum_duration:g}, {maximum_duration:g}]"
        )
    angle_changed = abs(angle - previous_angle) > 1.0e-9
    duration_changed = abs(duration - previous_duration) > 1.0e-9
    if not angle_changed or duration_changed:
        raise ModelAgentError(
            f"reduce angle_deg and keep duration_s={previous_duration:g} unchanged "
            f"from angle_deg={previous_angle:g}"
        )
    if angle >= previous_angle:
        raise ModelAgentError("angle_deg must decrease after payload instability")
    maximum_next_angle = float(constraint["maximum_next_angle_deg"])
    if angle > maximum_next_angle + 1.0e-9:
        raise ModelAgentError(
            f"angle_deg must be <= {maximum_next_angle:g} after payload instability"
        )
    expected_repair = "reduce_angle"
    repair = str(selected.get("repair", "")).strip()
    if expected_repair not in constraint["allowed_repairs"]:
        raise ModelAgentError(
            f"{expected_repair} is unavailable for the current turn envelope"
        )
    if repair and repair != expected_repair:
        raise ModelAgentError(
            f"repair={repair!r} conflicts with selected {expected_repair} values"
        )


def _conservative_turn_execution_parameters(
    constraint: dict[str, Any],
) -> dict[str, Any]:
    """Keep a real run moving if a small model cannot mutate its code literals."""

    previous_duration = float(constraint["previous_duration_s"])
    if "reduce_angle" in constraint["allowed_repairs"]:
        angle = float(constraint["maximum_next_angle_deg"])
        return {
            "repair": "reduce_angle",
            "angle_deg": angle,
            "duration_s": previous_duration,
            "reason": (
                "Conservative controller fallback after invalid model literals: reduce "
                "the bounded turn while preserving command duration."
            ),
            "selection_source": "controller_fallback_after_model_validation",
        }
    return {}


def _normalize_sync_parameter_candidate(
    candidate: dict[str, Any],
    *,
    constraint: dict[str, Any],
) -> dict[str, Any]:
    """Project a near-bound model speed into the safe synchronized envelope."""

    try:
        selected = {
            "front_distance_m": float(candidate["front_distance_m"]),
            "back_distance_m": float(candidate["back_distance_m"]),
            "front_speed_mps": float(candidate["front_speed_mps"]),
            "back_speed_mps": float(candidate["back_speed_mps"]),
            "reason": str(candidate.get("reason", "")).strip(),
        }
        required_speed = float(constraint["required_speed_mps"])
        required_distance = float(constraint["required_distance_m"])
        min_speed = float(constraint["minimum_speed_mps"])
        max_pair_speed_delta = float(constraint["maximum_pair_speed_delta_mps"])
        farther_robot = str(constraint["farther_robot"])
        nearer_robot = str(constraint["nearer_robot"])
    except (KeyError, TypeError, ValueError):
        return {}
    if farther_robot not in {"front", "back"} or nearer_robot not in {"front", "back"}:
        return {}
    original = {
        key: selected[key]
        for key in (
            "front_distance_m",
            "back_distance_m",
            "front_speed_mps",
            "back_speed_mps",
        )
    }
    selected["front_distance_m"] = required_distance
    selected["back_distance_m"] = required_distance
    selected[f"{nearer_robot}_speed_mps"] = required_speed
    minimum_effective_speed = max(min_speed, required_speed - max_pair_speed_delta)
    selected[f"{farther_robot}_speed_mps"] = max(
        minimum_effective_speed,
        min(required_speed, selected[f"{farther_robot}_speed_mps"]),
    )
    if selected[f"{farther_robot}_speed_mps"] >= required_speed - 1.0e-9:
        return {}
    try:
        _validate_selected_sync_parameter_values(selected, constraint=constraint)
    except ModelAgentError:
        return {}
    selected["host_schema_normalization"] = (
        "projected model-selected sync calibration into the effective speed envelope: "
        f"front ({original['front_distance_m']:g}, {original['front_speed_mps']:g}) -> "
        f"({selected['front_distance_m']:g}, {selected['front_speed_mps']:g}), back "
        f"({original['back_distance_m']:g}, {original['back_speed_mps']:g}) -> "
        f"({selected['back_distance_m']:g}, {selected['back_speed_mps']:g})"
    )
    return selected


def _validate_selected_sync_parameter_values(
    selected: dict[str, Any],
    *,
    constraint: dict[str, Any],
) -> None:
    values = {
        key: float(selected[key])
        for key in (
            "front_distance_m",
            "back_distance_m",
            "front_speed_mps",
            "back_speed_mps",
        )
    }
    min_distance = float(constraint["minimum_distance_m"])
    max_distance = float(constraint["maximum_distance_m"])
    min_speed = float(constraint["minimum_speed_mps"])
    max_speed = float(constraint["maximum_speed_mps"])
    required_speed = float(constraint.get("required_speed_mps", max_speed))
    required_distance = float(constraint.get("required_distance_m", max_distance))
    maximum_pair_speed_delta = float(
        constraint.get("maximum_pair_speed_delta_mps", 0.05)
    )
    for robot_id in ("front", "back"):
        distance = values[f"{robot_id}_distance_m"]
        speed = values[f"{robot_id}_speed_mps"]
        if not min_distance <= distance <= max_distance:
            raise ModelAgentError(
                f"{robot_id}_distance_m={distance:g} must be within "
                f"[{min_distance:g}, {max_distance:g}]"
            )
        if not min_speed <= speed <= max_speed:
            raise ModelAgentError(
                f"{robot_id}_speed_mps={speed:g} must be within "
                f"[{min_speed:g}, {max_speed:g}]"
            )
        if abs(distance - required_distance) > 1.0e-9:
            raise ModelAgentError(
                f"{robot_id}_distance_m={distance:g} must remain at the shared nominal "
                f"distance {required_distance:g}; tune speed only for this measured retry"
            )
    farther_robot = str(constraint["farther_robot"])
    nearer_robot = str(constraint["nearer_robot"])
    pair_speed_delta = abs(values["front_speed_mps"] - values["back_speed_mps"])
    if pair_speed_delta > maximum_pair_speed_delta + 1.0e-9:
        raise ModelAgentError(
            f"sync speed compensation delta {pair_speed_delta:g} exceeds the bounded "
            f"team limit {maximum_pair_speed_delta:g}"
        )
    if not (
        values[f"{farther_robot}_speed_mps"]
        < values[f"{nearer_robot}_speed_mps"] - 1.0e-9
        and abs(values[f"{nearer_robot}_speed_mps"] - required_speed) <= 1.0e-9
    ):
        raise ModelAgentError(
            f"{farther_robot} moved farther, so require "
            f"{farther_robot}_speed_mps < {nearer_robot}_speed_mps={required_speed:g} "
            f"while both distance_m={required_distance:g}; got "
            f"front=({values['front_distance_m']:g}, {values['front_speed_mps']:g}) "
            f"back=({values['back_distance_m']:g}, {values['back_speed_mps']:g})"
        )


def _compile_model_selected_sync_policy(
    *,
    current_policy: PolicyScript,
    attempt: int,
    selected_subtask: dict[str, str],
    selected_parameters: dict[str, Any],
    measured_constraint: dict[str, Any],
    failure_level: str,
    failure_type: str,
    blocked_strategy_signatures: set[str],
    blocked_transaction_signatures: set[str],
) -> tuple[dict[str, Any], PolicyScript]:
    """Compile the model's structured calibration decisions into safe SDK code."""

    method = str(selected_subtask.get("selected_method", "")).strip()
    mode = str(selected_subtask.get("coordination_mode", "")).strip()
    if method not in _TRANSPORT_METHODS or mode != "sync":
        raise ModelAgentError(
            "structured sync-policy compilation requires a selected sync transport method"
        )
    _validate_selected_sync_parameter_values(
        selected_parameters,
        constraint=measured_constraint,
    )
    ramp_s = float(measured_constraint.get("required_ramp_s", 1.5))
    script_id = f"box_carry_attempt_{attempt:02d}"
    synchronization_group = f"sync_compensation_{attempt:02d}"
    phase_id = "measured_sync_compensation"

    def code_for(robot_id: str) -> str:
        distance = float(selected_parameters[f"{robot_id}_distance_m"])
        speed = float(selected_parameters[f"{robot_id}_speed_mps"])
        direction_argument = ""
        if method == COORDINATED_LINEAR_TRANSPORT:
            direction = str(selected_subtask.get("motion_objective", "")).strip()
            if direction not in LOADED_TRANSLATION_DIRECTIONS:
                raise ModelAgentError(
                    "coordinated linear sync compilation requires a selected body-frame direction"
                )
            direction_argument = f'direction="{direction}", '
        return (
            "def policy(api):\n"
            f"    api.{method}({direction_argument}distance_m={distance:g}, "
            f"speed_mps={speed:g}, ramp_s={ramp_s:g})\n"
            "    api.stop_all()"
        )

    robot_codes = {robot_id: code_for(robot_id) for robot_id in ("front", "back")}
    payload = {
        "assignments": [
            {
                "robot_id": robot_id,
                "role": "synchronized_mover",
                "objective": "compensate measured executor displacement mismatch",
                "synchronization_group": synchronization_group,
                "phase_id": phase_id,
                "contact_intent": "maintain_payload_support",
                "coordination_mode": "sync",
                "safety_constraints": [],
            }
            for robot_id in ("front", "back")
        ],
        "robot_scripts": [
            {
                "robot_id": robot_id,
                "script_id": f"{robot_id}_{script_id}",
                "code": robot_codes[robot_id],
                "synchronization_group": synchronization_group,
                "phase_id": phase_id,
                "contact_intent": "maintain_payload_support",
                "coordination_mode": "sync",
            }
            for robot_id in ("front", "back")
        ],
        "script_id": script_id,
        "rationale": (
            str(selected_parameters.get("reason", "")).strip()
            or "Apply the model-selected measured sync compensation."
        ),
        "plan": ["execute one bounded measured sync calibration"],
        "spec_candidate": (
            "Measured executor mismatch: preserve the mover and sync barrier, use bounded "
            "per-robot compensation, and require verifier acceptance before reuse."
        ),
        "compiled_from_model_decisions": {
            "subtask": dict(selected_subtask),
            "parameters": dict(selected_parameters),
        },
    }
    policy = _policy_from_model_payload(
        payload,
        attempt=attempt,
        fallback_id=script_id,
        parent_script_id=current_policy.script_id,
        require_team_plan=True,
    )
    _validate_model_selected_subtask(
        policy,
        selected_method=method,
        coordination_mode=mode,
        motion_objective_parameter=str(
            selected_subtask.get("motion_objective_parameter", "")
        ),
        motion_objective=str(selected_subtask.get("motion_objective", "")),
    )
    _validate_measured_sync_compensation(policy, constraint=measured_constraint)
    _validate_selected_sync_parameters(
        policy,
        method=method,
        selected=selected_parameters,
    )
    strategy_signature = policy_strategy_signature(policy)
    if strategy_signature in blocked_strategy_signatures:
        raise ModelAgentError(
            f"strategy {strategy_signature} is excluded from the current model-generation search"
        )
    transaction_signature = _policy_transaction_signature(policy)
    if transaction_signature in blocked_transaction_signatures:
        raise ModelAgentError(
            "structured measured-sync parameters repeat a verifier-rejected executable "
            "transaction at AST level"
        )
    _validate_failure_level_repair(
        current_policy,
        policy,
        failure_level=failure_level,
        failure_type=failure_type,
        blocked_strategy_signatures=blocked_strategy_signatures,
        allow_measured_sync_compensation=True,
    )
    return payload, policy


def _compile_model_selected_coordination_policy(
    *,
    current_policy: PolicyScript,
    attempt: int,
    selected_subtask: dict[str, str],
    selected_parameters: dict[str, Any],
    failure_level: str,
    failure_type: str,
    blocked_strategy_signatures: set[str],
    blocked_transaction_signatures: set[str],
    explicit_branch_rewind: bool = False,
) -> tuple[dict[str, Any], PolicyScript]:
    """Compile model-selected coordination semantics and profiles into two scripts."""

    _validate_selected_coordination_parameters(
        selected_parameters,
        selected_subtask=selected_subtask,
    )
    method = str(selected_subtask["selected_method"])
    mode = str(selected_subtask["coordination_mode"])
    robot_parameters = selected_parameters["robot_parameters"]
    script_id = f"box_carry_attempt_{attempt:02d}"
    synchronization_group = f"coordination_group_{attempt:02d}"
    phase_id = f"coordination_{attempt:02d}"

    def code_for(robot_id: str) -> str:
        parameters = robot_parameters[robot_id]
        ordered_names = coordination_primitive_method_spec(method)["parameters"].keys()
        arguments = ", ".join(
            f"{name}={_python_literal(parameters[name])}" for name in ordered_names
        )
        return f"def policy(api):\n    api.{method}({arguments})\n    api.stop_all()"

    def role_for(robot_id: str) -> str:
        parameters = robot_parameters[robot_id]
        if method == COORDINATED_LINEAR_TRANSPORT:
            return "synchronized_mover"
        if method == COORDINATED_ARC_TRANSPORT:
            other_id = "back" if robot_id == "front" else "front"
            return (
                "outer"
                if float(parameters["radius_m"])
                > float(robot_parameters[other_id]["radius_m"])
                else "inner"
            )
        if method == COORDINATED_PIVOT:
            return str(parameters["role"])
        delta = float(parameters["delta"])
        if abs(delta) <= 1.0e-9:
            return "reference"
        return "positive_adjuster" if delta > 0.0 else "negative_adjuster"

    robot_codes = {robot_id: code_for(robot_id) for robot_id in ("front", "back")}
    objective = str(selected_subtask.get("motion_objective", ""))
    payload = {
        "assignments": [
            {
                "robot_id": robot_id,
                "role": role_for(robot_id),
                "objective": f"{method}:{objective}",
                "synchronization_group": synchronization_group,
                "phase_id": phase_id,
                "contact_intent": "maintain_payload_support",
                "coordination_mode": mode,
                "safety_constraints": [],
            }
            for robot_id in ("front", "back")
        ],
        "robot_scripts": [
            {
                "robot_id": robot_id,
                "script_id": f"{robot_id}_{script_id}",
                "code": robot_codes[robot_id],
                "synchronization_group": synchronization_group,
                "phase_id": phase_id,
                "contact_intent": "maintain_payload_support",
                "coordination_mode": mode,
            }
            for robot_id in ("front", "back")
        ],
        "script_id": script_id,
        "rationale": (
            str(selected_parameters.get("reason", "")).strip()
            or str(selected_subtask.get("reason", "")).strip()
            or "Execute the selected bounded coordination primitive."
        ),
        "plan": [f"execute one bounded {method} transaction"],
        "spec_candidate": (
            "Use the selected coordination primitive and temporary role relation only when "
            "the physics verifier confirms comparable support and geometry."
        ),
        "compiled_from_model_decisions": {
            "subtask": dict(selected_subtask),
            "parameters": dict(selected_parameters),
        },
    }
    policy = _policy_from_model_payload(
        payload,
        attempt=attempt,
        fallback_id=script_id,
        parent_script_id=current_policy.script_id,
        require_team_plan=True,
    )
    _validate_model_selected_subtask(
        policy,
        selected_method=method,
        coordination_mode=mode,
        motion_objective_parameter=str(
            selected_subtask.get("motion_objective_parameter", "")
        ),
        motion_objective=objective,
    )
    verifier_owned_heading_correction = bool(
        selected_subtask.get("selection_source") == "verifier_heading_correction"
        and selected_parameters.get("selection_source")
        == "verifier_heading_correction_fallback"
    )
    host_owned_heading_correction = bool(
        selected_subtask.get("selection_source")
        == HOST_WALL_HEADING_CORRECTION_SELECTION_SOURCE
        and selected_parameters.get("selection_source")
        == HOST_WALL_HEADING_CORRECTION_PARAMETER_SOURCE
    )
    host_owned_wall_clearance_escape = bool(
        selected_subtask.get("selection_source")
        == HOST_WALL_CLEARANCE_ESCAPE_SELECTION_SOURCE
        and selected_parameters.get("selection_source")
        == HOST_WALL_CLEARANCE_ESCAPE_PARAMETER_SOURCE
    )
    host_owned_wall_turn_escape = bool(
        selected_subtask.get("selection_source")
        == HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE
        and selected_parameters.get("selection_source")
        == HOST_WALL_TURN_ESCAPE_PARAMETER_SOURCE
    )
    verifier_owned_straight_alignment = bool(
        selected_subtask.get("selection_source")
        == "verifier_straight_heading_alignment"
        and selected_parameters.get("selection_source")
        == "verifier_straight_heading_alignment_fallback"
    )
    verifier_owned_longitudinal_alignment = bool(
        selected_subtask.get("selection_source")
        == "verifier_endpoint_longitudinal_alignment"
        and selected_parameters.get("selection_source")
        == "verifier_endpoint_longitudinal_alignment_fallback"
    )
    verifier_owned_spacing_alignment = bool(
        selected_subtask.get("selection_source") == "verifier_pair_spacing_alignment"
        and selected_parameters.get("selection_source")
        == "verifier_pair_spacing_alignment_fallback"
    )
    verifier_owned_route_turn = bool(
        selected_subtask.get("selection_source") == "verifier_route_turn_continuation"
        and selected_parameters.get("selection_source")
        == "verifier_route_turn_continuation_fallback"
    )
    verifier_owned_route_probe = bool(
        selected_subtask.get("selection_source") == "verifier_route_turn_probe"
        and selected_parameters.get("selection_source")
        == "verifier_route_turn_probe_fallback"
    )
    verifier_owned_transaction = bool(
        verifier_owned_straight_alignment
        or verifier_owned_longitudinal_alignment
        or verifier_owned_spacing_alignment
        or verifier_owned_heading_correction
        or host_owned_heading_correction
        or host_owned_wall_clearance_escape
        or host_owned_wall_turn_escape
        or verifier_owned_route_turn
        or verifier_owned_route_probe
    )
    strategy_signature = policy_strategy_signature(policy)
    if (
        strategy_signature in blocked_strategy_signatures
        and not verifier_owned_transaction
    ):
        raise ModelAgentError(
            f"strategy {strategy_signature} is excluded from the current model-generation search"
        )
    transaction_signature = _policy_transaction_signature(policy)
    if transaction_signature in blocked_transaction_signatures:
        raise ModelAgentError(
            "structured coordination candidate repeats a verifier-rejected executable transaction"
        )
    if not verifier_owned_transaction and not explicit_branch_rewind:
        _validate_failure_level_repair(
            current_policy,
            policy,
            failure_level=failure_level,
            failure_type=failure_type,
            blocked_strategy_signatures=blocked_strategy_signatures,
        )
        retry_limits = _unsafe_retry_transport_limits(
            current_policy,
            failure_type=failure_type,
        )
        if retry_limits is not None:
            _validate_unsafe_retry_magnitude(
                policy,
                max_distance_m=retry_limits[0],
                max_speed_mps=retry_limits[1],
            )
    return payload, policy


def _python_literal(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not isfinite(numeric):
            raise ModelAgentError("compiled policy parameters must be finite")
        return repr(numeric)
    raise ModelAgentError(f"unsupported compiled policy literal {value!r}")


def _compile_model_selected_turn_policy(
    *,
    current_policy: PolicyScript,
    attempt: int,
    selected_parameters: dict[str, Any],
    constraint: dict[str, Any],
    failure_level: str,
    failure_type: str,
    blocked_strategy_signatures: set[str],
) -> tuple[dict[str, Any], PolicyScript]:
    """Compile a structured model angle/duration decision into both robot scripts."""

    _validate_selected_turn_execution_parameters(
        selected_parameters,
        constraint=constraint,
    )
    method = str(constraint["method"])
    script_id = f"box_carry_attempt_{attempt:02d}"
    updates = {
        "angle_deg": float(selected_parameters["angle_deg"]),
        "duration_s": float(selected_parameters["duration_s"]),
    }
    rewritten_scripts = tuple(
        replace(
            script,
            script_id=f"{script.robot_id}_{script_id}",
            code=_rewrite_task_call_keywords(
                script.code,
                method=method,
                updates=updates,
            ),
        )
        for script in current_policy.robot_scripts
    )
    payload = {
        "assignments": [asdict(item) for item in current_policy.assignments],
        "robot_scripts": [asdict(item) for item in rewritten_scripts],
        "script_id": script_id,
        "rationale": str(selected_parameters.get("reason", "")).strip(),
        "plan": ["execute one bounded model-selected turn-envelope repair"],
        "spec_candidate": (
            "After loaded-turn instability, preserve the selected geometry and roles while "
            "testing one lower-load execution envelope before changing strategy."
        ),
        "compiled_from_model_decisions": {
            "constraint": dict(constraint),
            "parameters": dict(selected_parameters),
        },
    }
    policy = _policy_from_model_payload(
        payload,
        attempt=attempt,
        fallback_id=script_id,
        parent_script_id=current_policy.script_id,
        require_team_plan=True,
    )
    strategy_signature = policy_strategy_signature(policy)
    if strategy_signature in blocked_strategy_signatures:
        raise ModelAgentError(
            f"strategy {strategy_signature} is excluded from the current model-generation search"
        )
    _validate_failure_level_repair(
        current_policy,
        policy,
        failure_level=failure_level,
        failure_type=failure_type,
        blocked_strategy_signatures=blocked_strategy_signatures,
    )
    return payload, policy


def _rewrite_task_call_keywords(
    code: str,
    *,
    method: str,
    updates: dict[str, float],
) -> str:
    """Rewrite literal keyword values in one allowlisted API call through Python AST."""

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ModelAgentError(f"cannot rewrite invalid policy code: {exc}") from exc
    matching_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            isinstance(owner, ast.Name)
            and owner.id == "api"
            and node.func.attr == method
        ):
            matching_calls.append(node)
    if len(matching_calls) != 1:
        raise ModelAgentError(
            f"expected exactly one api.{method}(...) call while compiling turn repair"
        )
    call = matching_calls[0]
    keywords = {item.arg: item for item in call.keywords if item.arg is not None}
    missing = sorted(set(updates) - set(keywords))
    if missing:
        raise ModelAgentError(
            f"api.{method}(...) is missing required literal keywords {missing}"
        )
    for name, value in updates.items():
        keywords[name].value = ast.Constant(value=float(value))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _validate_strategy_against_host_formation_gate(
    *,
    method: str,
    objective: str,
    mode: str,
    assignment: object,
    formation_gate: dict[str, Any],
) -> None:
    """Keep a route-neutral formation repair bound to measured robot IDs."""

    if not formation_gate:
        return
    expected_roles = formation_gate.get("team_roles", {})
    expected_roles = expected_roles if isinstance(expected_roles, dict) else {}
    expected_method = str(
        formation_gate.get("primitive", COORDINATED_ALIGNMENT)
    ).strip()
    expected_objective = str(
        formation_gate.get("semantic_objective", "longitudinal")
    ).strip()
    expected_mode = str(formation_gate.get("coordination_mode", "async")).strip()
    assignment = assignment if isinstance(assignment, dict) else {}
    roles_by_robot = {
        robot_id: str(
            assignment.get(robot_id, {}).get("role", "")
            if isinstance(assignment.get(robot_id), dict)
            else ""
        ).strip()
        for robot_id in ("front", "back")
    }
    if (
        method != expected_method
        or objective != expected_objective
        or mode != expected_mode
        or roles_by_robot != expected_roles
    ):
        raise ModelRoleError(
            "the current trusted checkpoint requires one route-neutral formation "
            "repair before further route motion: "
            f"primitive={expected_method!r}, "
            f"semantic_objective={expected_objective!r}, "
            f"coordination_mode={expected_mode!r}, "
            f"roles={expected_roles}. This physical lagger/reference gate does not "
            "select the later route direction"
        )


def _validate_strategy_not_superseded_formation_gate(
    *,
    method: str,
    objective: str,
    mode: str,
    assignment: object,
    formation_gate: object,
) -> None:
    """Reject the exact high-level relation that just regressed physically."""

    gate = formation_gate if isinstance(formation_gate, dict) else {}
    assignment = assignment if isinstance(assignment, dict) else {}
    roles = {
        robot_id: str(
            assignment.get(robot_id, {}).get("role", "")
            if isinstance(assignment.get(robot_id), dict)
            else ""
        ).strip()
        for robot_id in ("front", "back")
    }
    candidate = _exhausted_coordination_relation_signature(
        {
            "primitive": method,
            "semantic_objective": objective,
            "coordination_mode": mode,
            "team_roles": roles,
        }
    )
    if candidate and candidate == _exhausted_coordination_relation_signature(gate):
        raise ModelRoleError(
            "superseded host formation gate relation cannot be selected again: "
            "the matching primitive, semantic objective, mode, and immutable-ID "
            "roles physically moved without improving alignment. Choose a materially "
            "different strategy or coordination relation"
        )


def _validate_plan_does_not_repeat_superseded_formation_gate(
    task_plan: dict[str, Any],
    *,
    formation_gate: dict[str, Any],
) -> None:
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    strategy = subtask.get("strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    _validate_strategy_not_superseded_formation_gate(
        method=str(strategy.get("primitive", "")).strip(),
        objective=str(strategy.get("semantic_objective", "")).strip(),
        mode=str(strategy.get("coordination_mode", "")).strip(),
        assignment=subtask.get(
            "team_assignment",
            strategy.get("team_assignment", {}),
        ),
        formation_gate=formation_gate,
    )


def _validate_mutation_does_not_repeat_superseded_formation_gate(
    mutation: dict[str, Any],
    *,
    formation_gate: dict[str, Any],
) -> None:
    strategy = mutation.get("child_strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    _validate_strategy_not_superseded_formation_gate(
        method=str(strategy.get("primitive", "")).strip(),
        objective=str(strategy.get("semantic_objective", "")).strip(),
        mode=str(strategy.get("coordination_mode", "")).strip(),
        assignment=strategy.get("team_assignment", {}),
        formation_gate=formation_gate,
    )


def _loaded_translation_clearance_m(
    clearance: object,
    objective: str,
) -> float | None:
    report = clearance if isinstance(clearance, dict) else {}
    candidates = report.get("translation_candidates", {})
    candidates = candidates if isinstance(candidates, dict) else {}
    evidence = candidates.get(str(objective).strip(), {})
    evidence = evidence if isinstance(evidence, dict) else {}
    try:
        value = float(evidence["clearance_m"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if isfinite(value) and value >= 0.0 else None


def _loaded_translation_has_supported_tier(
    clearance: object,
    objective: str,
) -> bool:
    """Distinguish a blocked room-scale horizon from an executable short tier."""

    if not candidate_is_blocked(
        clearance,
        family="translation",
        objective=objective,
    ):
        return True
    clearance_m = _loaded_translation_clearance_m(clearance, objective)
    return bool(
        clearance_m is not None
        and clearance_m + 1.0e-9 >= min(COORDINATED_LINEAR_DISTANCE_CHOICES_M)
    )


def _model_executable_loaded_translation_directions(
    clearance: object,
) -> tuple[str, ...]:
    return tuple(
        direction
        for direction in LOADED_TRANSLATION_DIRECTIONS
        if _loaded_translation_has_supported_tier(clearance, direction)
    )


def _validate_strategy_against_host_wall_clearance(
    *,
    method: str,
    objective: str,
    host_wall_clearance: dict[str, Any] | None,
) -> None:
    """Reject only immediate authored-wall conflicts, without planning the route."""

    clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    if not clearance.get("available", False):
        return
    translation_methods = {
        COORDINATED_LINEAR_TRANSPORT,
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
    }
    turn_methods = {
        COORDINATED_PIVOT,
        COORDINATED_ARC_TRANSPORT,
        "turn_clockwise",
        "turn_counterclockwise",
    }
    if method in translation_methods and objective in LOADED_TRANSLATION_DIRECTIONS:
        if not _loaded_translation_has_supported_tier(clearance, objective):
            clearance_m = _loaded_translation_clearance_m(clearance, objective)
            executable_directions = (
                _model_executable_loaded_translation_directions(clearance)
            )
            raise ModelRoleError(
                "predictive authored-wall geometry leaves no model-supported "
                f"distance tier for body-frame translation {objective!r}: "
                f"clearance_m={float(clearance_m or 0.0):g}, "
                "smallest_supported_distance_m="
                f"{min(COORDINATED_LINEAR_DISTANCE_CHOICES_M):g}. This does not "
                "claim that the current checkpoint is already touching a wall"
                + (
                    "; translations with at least one supported distance tier="
                    f"{list(executable_directions)}"
                    if executable_directions
                    else ""
                )
                + ". Replan the semantic direction or maneuver before code generation."
            )
        return
    if method in turn_methods and objective in _SIGNED_TURN_DIRECTIONS:
        if method == COORDINATED_PIVOT:
            maximum = maximum_safe_turn_angle_deg(clearance, objective)
            minimum_calibrated_angle = min(MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG)
            if (
                maximum is not None
                and maximum + 1.0e-9 < minimum_calibrated_angle
            ):
                open_translations = list(
                    _model_executable_loaded_translation_directions(clearance)
                )
                raise ModelRoleError(
                    "authoritative authored-wall geometry leaves only "
                    f"{maximum:g} degrees for turn_direction={objective!r}, below "
                    "the loaded coordinated_pivot calibration domain of "
                    f"{minimum_calibrated_angle:g} degrees. A geometrically clear "
                    "micro-sweep is not an executable SONIC pivot. Preserve the "
                    "route intent and first choose an open translation that increases "
                    "turn clearance"
                    + (
                        f"; currently open translations={open_translations}"
                        if open_translations
                        else ""
                    )
                    + "."
                )
        if candidate_is_blocked(
            clearance,
            family="turn",
            objective=objective,
        ) and _model_executable_loaded_translation_directions(clearance):
            turn_candidates = clearance.get("turn_candidates", {})
            turn_candidates = (
                turn_candidates if isinstance(turn_candidates, dict) else {}
            )
            open_turns = [
                direction
                for direction in sorted(_SIGNED_TURN_DIRECTIONS)
                if isinstance(turn_candidates.get(direction), dict)
                and turn_candidates[direction].get("relation") != "blocked"
            ]
            open_translations = list(
                _model_executable_loaded_translation_directions(clearance)
            )
            raise ModelRoleError(
                "host turn-clearance replan required: authoritative authored-wall "
                f"geometry marks the proposed {objective} finite coordinated-pivot "
                "sweep as blocked before the minimum executable angle. Do not repeat "
                "that blocked turn. Choose the next "
                "turn-setup maneuver from current global route evidence using either "
                f"an unblocked signed turn {open_turns} or one synchronized "
                "coordinated_linear_transport in an open body-frame direction "
                f"{open_translations}."
            )


def _validate_strategy_mutation_linear_feasibility(
    *,
    parent_strategy: dict[str, Any],
    method: str,
    objective: str,
    host_wall_clearance: dict[str, Any] | None,
    trusted_body_frame_target_projection: dict[str, Any] | None,
) -> None:
    """Report an empty linear mutation space as one coherent constraint."""

    if method != COORDINATED_LINEAR_TRANSPORT:
        return
    clearance = (
        host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    )
    projection = (
        trusted_body_frame_target_projection
        if isinstance(trusted_body_frame_target_projection, dict)
        else {}
    )
    if not clearance.get("available", False) or not projection:
        return

    open_directions = set(
        _model_executable_loaded_translation_directions(clearance)
    )
    scores = projection.get("candidate_target_alignment", {})
    scores = scores if isinstance(scores, dict) else {}
    finite_scores = {
        str(direction): float(score)
        for direction, score in scores.items()
        if str(direction) in LOADED_TRANSLATION_DIRECTIONS
        and isinstance(score, (int, float))
        and isfinite(float(score))
    }
    if not open_directions or not finite_scores:
        return

    if bool(projection.get("target_away_setup_authorized", False)):
        target_compatible = set(finite_scores)
    else:
        best_score = max(finite_scores.values())
        target_compatible = {
            direction
            for direction, score in finite_scores.items()
            if score >= -0.05 and not (best_score >= 0.85 and score < 0.35)
        }
    parent_is_linear = (
        str(parent_strategy.get("primitive", "")).strip()
        == COORDINATED_LINEAR_TRANSPORT
    )
    parent_objective = (
        str(parent_strategy.get("semantic_objective", "")).strip()
        if parent_is_linear
        else ""
    )
    alternative_linear = sorted(
        (open_directions & target_compatible)
        - ({parent_objective} if parent_objective else set())
    )
    if alternative_linear:
        if objective in alternative_linear:
            return
        raise ModelRoleError(
            "the proposed linear objective does not satisfy the combined trusted "
            f"constraints: wall-open translations={sorted(open_directions)}, "
            f"target-compatible translations={sorted(target_compatible)}, and the "
            "remaining strategy-mutation-compatible linear objectives are "
            f"{alternative_linear}. Choose one combined-valid objective or a "
            "geometry-changing maneuver; do not alternate between constraints "
            "reported by individual validators."
        )

    turn_candidates = clearance.get("turn_candidates", {})
    turn_candidates = turn_candidates if isinstance(turn_candidates, dict) else {}
    open_turns = sorted(
        direction
        for direction in _SIGNED_TURN_DIRECTIONS
        if isinstance(turn_candidates.get(direction), dict)
        and turn_candidates[direction].get("relation") != "blocked"
    )
    raise ModelRoleError(
        "strategy_mutation has no alternative coordinated_linear_transport "
        "objective satisfying the combined trusted constraints: wall-open "
        f"translations={sorted(open_directions)}, target-compatible translations="
        f"{sorted(target_compatible)}, and the rejected parent objective "
        f"{parent_objective!r} must change at this mutation level. Do not alternate "
        "between individually invalid linear directions. Select a geometry-changing "
        "maneuver primitive and infer its direction from the global view; locally "
        f"wall-open signed turns={open_turns}. The host does not choose the signed turn."
    )


def _accepted_linear_continuation_contract(
    policy: PolicyScript,
    report: EvaluationReport,
) -> dict[str, Any]:
    """Prevent one accepted route segment from being immediately undone."""

    if not report.success:
        return {}
    try:
        strategy = _policy_mutation_parent_strategy(policy)
    except ModelAgentError:
        return {}
    if strategy.get("primitive") != COORDINATED_LINEAR_TRANSPORT:
        return {}
    objective = str(strategy.get("semantic_objective", "")).strip()
    opposite = {
        "forward": "backward",
        "backward": "forward",
        "left": "right",
        "right": "left",
    }.get(objective, "")
    if not opposite:
        return {}
    try:
        progress_m = float(report.metrics.get("payload_target_progress_m", 0.0))
        displacement_m = float(
            report.metrics.get("payload_step_displacement_xy_m", 0.0)
        )
        required_m = max(
            0.05,
            float(report.metrics.get("required_step_displacement_m", 0.0)),
        )
        wall_contacts = float(report.metrics.get("wall_contact_count", 0.0))
    except (TypeError, ValueError):
        return {}
    visual = report.evidence.get("visual_verifier", {})
    visual = visual if isinstance(visual, dict) else {}
    visual_evidence = visual.get("evidence", {})
    visual_evidence = visual_evidence if isinstance(visual_evidence, dict) else {}
    if not (
        progress_m >= required_m
        and displacement_m >= required_m
        and wall_contacts <= 0.0
        and visual.get("status") == "ok"
        and not bool(visual_evidence.get("route_direction_risk_confirmed", False))
        and str(
            visual_evidence.get(
                "team_heading_relative_to_visible_free_space",
                "",
            )
        ).strip()
        == "aligned_with_open_passage"
    ):
        return {}
    phase_ids = {
        str(script.phase_id).strip()
        for script in policy.robot_scripts
        if str(script.phase_id).strip()
    }
    return {
        "active": True,
        "source": "latest_verifier_accepted_checkpoint",
        "primitive": COORDINATED_LINEAR_TRANSPORT,
        "semantic_objective": objective,
        "blocked_immediate_inverse": opposite,
        "phase_id": next(iter(phase_ids)) if len(phase_ids) == 1 else "",
        "measured_target_progress_m": round(progress_m, 6),
        "measured_displacement_m": round(displacement_m, 6),
        "rule": (
            "continue the accepted direction or choose a geometry-changing "
            "maneuver; do not erase the newest trusted segment by immediately "
            "selecting its exact inverse"
        ),
        "expires_after": (
            "rejected continuation, verified phase transition, or accepted "
            "geometry-changing maneuver"
        ),
    }


def _host_pivot_role_contract(
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Expose deterministic ID-role geometry without selecting a turn sign."""

    state = runtime_state if isinstance(runtime_state, dict) else {}
    assignments = {
        direction: pivot_role_assignment(state, direction)
        for direction in ("clockwise", "counterclockwise")
    }
    assignments = {
        direction: mapping
        for direction, mapping in assignments.items()
        if set(mapping) == {"front", "back"}
        and set(mapping.values()) == {"inner", "outer"}
    }
    if not assignments:
        return {}
    return {
        "turn_direction_decision_owner": "task_orchestrator",
        "immutable_id_role_mapping_owner": "host_geometry",
        "role_assignment_by_direction": assignments,
        "rule": (
            "Choose the signed turn from visual route evidence, then copy the "
            "corresponding front/back role mapping exactly. This mapping does not "
            "choose the route direction."
        ),
    }


def _host_wall_heading_correction_contract(
    *,
    formation_sync_repair_gate: dict[str, Any] | None,
    host_wall_clearance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind one blocked synchronization probe to an exact local yaw repair."""

    gate = (
        formation_sync_repair_gate
        if isinstance(formation_sync_repair_gate, dict)
        else {}
    )
    clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    if not gate.get("active", False) or not clearance.get("available", False):
        return {}
    probe = gate.get("probe_contract", {})
    probe = probe if isinstance(probe, dict) else {}
    desired_translation = str(probe.get("semantic_objective", "")).strip()
    if desired_translation not in LOADED_TRANSLATION_DIRECTIONS:
        return {}
    translations = clearance.get("translation_candidates", {})
    translations = translations if isinstance(translations, dict) else {}
    translation_evidence = translations.get(desired_translation, {})
    if not (
        isinstance(translation_evidence, dict)
        and translation_evidence.get("relation") == "blocked"
    ):
        return {}
    corrections = clearance.get(
        "small_heading_corrections_for_blocked_translation",
        {},
    )
    corrections = corrections if isinstance(corrections, dict) else {}
    correction = corrections.get(desired_translation, {})
    correction = correction if isinstance(correction, dict) else {}
    turn_direction = str(correction.get("turn_direction", "")).strip()
    inner_robot = str(correction.get("inner_robot", "")).strip()
    outer_robot = str(correction.get("outer_robot", "")).strip()
    try:
        angle_deg = float(correction["angle_deg"])
    except (KeyError, TypeError, ValueError):
        return {}
    if (
        turn_direction not in _SIGNED_TURN_DIRECTIONS
        or {inner_robot, outer_robot} != {"front", "back"}
        or inner_robot == outer_robot
        or not 2.0 <= angle_deg <= 15.0
    ):
        return {}
    return {
        "active": True,
        "source": "authored_wall_geometry_for_blocked_formation_sync_probe",
        "selection_source": HOST_WALL_HEADING_CORRECTION_SELECTION_SOURCE,
        "parameter_source": HOST_WALL_HEADING_CORRECTION_PARAMETER_SOURCE,
        "marker": HOST_WALL_HEADING_CORRECTION_MARKER,
        "primitive": COORDINATED_PIVOT,
        "coordination_mode": "async",
        "desired_translation_direction": desired_translation,
        "turn_direction": turn_direction,
        "angle_deg": angle_deg,
        "duration_s": 4.0,
        "swing_radius_m": 0.8,
        "travel_direction": "forward",
        "robot_roles": {
            inner_robot: "inner",
            outer_robot: "outer",
        },
        "scope": (
            "one geometry-owned heading correction before re-evaluating the "
            "blocked formation synchronization translation probe"
        ),
    }


def _rejected_host_wall_heading_correction(
    policy: PolicyScript,
    report: EvaluationReport,
) -> dict[str, Any]:
    """Return checkpoint-local evidence that the one-shot wall repair was unsafe."""

    if (
        report.success
        or HOST_WALL_HEADING_CORRECTION_MARKER not in policy.rationale
        or report.failure_type
        not in {
            "formation_desync",
            "no_payload_progress",
            "payload_not_supported",
            "payload_slip",
            "payload_unstable",
            "robot_pair_too_close",
        }
    ):
        return {}
    method, objective = _policy_motion_objective(policy)
    if method != COORDINATED_PIVOT or objective not in _SIGNED_TURN_DIRECTIONS:
        return {}
    return {
        "active": True,
        "source": "physics_rejected_host_wall_heading_correction",
        "primitive": method,
        "semantic_objective": objective,
        "failure_type": report.failure_type,
        "failed_transaction": _policy_transaction_for_prompt(policy),
        "required_next_repair": COORDINATED_ALIGNMENT,
        "rule": (
            "Do not tune or replay this geometry-owned pivot at the same checkpoint. "
            "First repair the loaded support/formation relation with one bounded "
            "coordinated_alignment, then recompute wall clearance from fresh physics."
        ),
    }


def _load_rejected_host_wall_heading_correction(
    history_dirs: tuple[Path, ...],
    *,
    checkpoint_id: str,
    minimum_attempt_index: int = 0,
    limit: int = 80,
) -> dict[str, Any]:
    """Reload a physical wall-correction failure for the same checkpoint lineage."""

    for history_dir in reversed(tuple(dict.fromkeys(history_dirs))):
        for policy in reversed(
            _load_rejected_policy_history(
                history_dir,
                checkpoint_id=checkpoint_id,
                limit=limit,
                minimum_attempt_index=minimum_attempt_index,
            )
        ):
            if HOST_WALL_HEADING_CORRECTION_MARKER not in policy.rationale:
                continue
            method, objective = _policy_motion_objective(policy)
            if method != COORDINATED_PIVOT or objective not in _SIGNED_TURN_DIRECTIONS:
                continue
            return {
                "active": True,
                "source": "checkpoint_lineage_physical_rejection_history",
                "scope_checkpoint_id": checkpoint_id,
                "source_attempt": policy.attempt,
                "primitive": method,
                "semantic_objective": objective,
                "failure_type": "historical_physical_rejection",
                "failed_transaction": _policy_transaction_for_prompt(policy),
                "required_next_repair": COORDINATED_ALIGNMENT,
                "rule": (
                    "Do not tune or replay this geometry-owned pivot at the same "
                    "checkpoint. Continue bounded coordinated_alignment repair until "
                    "one repair is accepted and creates a new checkpoint, then recompute "
                    "wall clearance from fresh physics."
                ),
            }
    return {}


def _strategy_matches_host_wall_heading_correction(
    *,
    method: str,
    objective: str,
    assignment: object,
    contract: dict[str, Any] | None,
) -> bool:
    correction = contract if isinstance(contract, dict) else {}
    if not correction.get("active", False):
        return False
    assignment_map = assignment if isinstance(assignment, dict) else {}
    roles = {
        robot_id: str(
            assignment_map.get(robot_id, {}).get("role", "")
            if isinstance(assignment_map.get(robot_id), dict)
            else ""
        ).strip()
        for robot_id in ("front", "back")
    }
    return bool(
        method == correction.get("primitive")
        and objective == correction.get("turn_direction")
        and roles == correction.get("robot_roles")
    )


def _strategy_matches_turn_clearance_replan_contract(
    *,
    method: str,
    objective: str,
    mode: str,
    contract: dict[str, Any] | None,
) -> bool:
    clearance_setup = contract if isinstance(contract, dict) else {}
    allowed = {
        str(item).strip()
        for item in clearance_setup.get("allowed_translation_directions", ())
        if str(item).strip()
    }
    return bool(
        clearance_setup.get("active", False)
        and method == clearance_setup.get(
            "required_primitive", COORDINATED_LINEAR_TRANSPORT
        )
        and mode == clearance_setup.get("required_coordination_mode", "sync")
        and objective in allowed
    )


def _validate_host_wall_heading_correction_policy(
    policy: PolicyScript,
    *,
    contract: dict[str, Any] | None,
) -> bool:
    """Validate and identify the exact executable local wall-clear correction."""

    correction = contract if isinstance(contract, dict) else {}
    if not correction.get("active", False):
        return False
    intent = policy_turn_intent(policy)
    expected_roles = correction.get("robot_roles", {})
    if intent is None:
        raise ModelAgentError(
            "host wall-heading correction requires one auditable coordinated_pivot "
            "call for each robot"
        )
    if (
        intent.method != correction.get("primitive")
        or intent.direction != correction.get("turn_direction")
        or dict(intent.roles) != expected_roles
        or abs(intent.angle_deg - float(correction["angle_deg"])) > 1.0e-9
    ):
        raise ModelAgentError(
            "host wall-heading correction must preserve the authored turn direction, "
            "angle, and host-derived inner/outer robot roles exactly"
        )
    scripts = {script.robot_id: script for script in policy.robot_scripts}
    if set(scripts) != {"front", "back"}:
        raise ModelAgentError(
            "host wall-heading correction requires explicit front and back scripts"
        )
    expected_literals = {
        "travel_direction": str(correction["travel_direction"]),
        "duration_s": float(correction["duration_s"]),
        "swing_radius_m": float(correction["swing_radius_m"]),
    }
    for robot_id, script in scripts.items():
        arguments = _first_task_call_literal_arguments(script.code)
        if (
            str(arguments.get("travel_direction", ""))
            != expected_literals["travel_direction"]
        ):
            raise ModelAgentError(
                "host wall-heading correction requires travel_direction='forward'"
            )
        for name in ("duration_s", "swing_radius_m"):
            try:
                actual = float(arguments[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelAgentError(
                    f"host wall-heading correction requires numeric {name}"
                ) from exc
            if abs(actual - float(expected_literals[name])) > 1.0e-9:
                raise ModelAgentError(
                    "host wall-heading correction must preserve the bounded authored "
                    f"{name}={expected_literals[name]:g}; got {actual:g} for {robot_id}"
                )
    return True


def _apply_host_pivot_roles_to_strategy(
    strategy: object,
    *,
    runtime_state: dict[str, Any] | None,
    audit_container: dict[str, Any],
) -> None:
    """Compile a model-selected pivot sign into geometry-correct robot roles."""

    if not isinstance(strategy, dict):
        return
    if str(strategy.get("primitive", "")).strip() != COORDINATED_PIVOT:
        return
    direction = str(strategy.get("semantic_objective", "")).strip()
    expected_roles = pivot_role_assignment(runtime_state or {}, direction)
    if set(expected_roles) != {"front", "back"} or set(expected_roles.values()) != {
        "inner",
        "outer",
    }:
        return
    assignment = strategy.get("team_assignment", {})
    if not isinstance(assignment, dict):
        assignment = {}
        strategy["team_assignment"] = assignment
    previous_roles: dict[str, str] = {}
    for robot_id in ("front", "back"):
        robot_assignment = assignment.get(robot_id, {})
        if not isinstance(robot_assignment, dict):
            robot_assignment = {}
            assignment[robot_id] = robot_assignment
        previous_roles[robot_id] = str(robot_assignment.get("role", "")).strip()
        role = expected_roles[robot_id]
        robot_assignment["role"] = role
        robot_assignment["objective"] = (
            "remain stationary as the inner pivot point"
            if role == "inner"
            else "perform the forward outer arc during the pivot"
        )
    if previous_roles == expected_roles:
        return
    normalizations = audit_container.get("host_schema_normalizations", [])
    normalizations = normalizations if isinstance(normalizations, list) else []
    normalizations.append(
        "compiled model-selected coordinated_pivot "
        f"direction={direction} to host-measured immutable-ID roles "
        f"front={expected_roles['front']}, back={expected_roles['back']}"
    )
    audit_container["host_schema_normalizations"] = normalizations


def _apply_host_pivot_roles_to_task_plan(
    task_plan: dict[str, Any],
    *,
    runtime_state: dict[str, Any] | None,
) -> None:
    subtask = task_plan.get("current_subtask", {})
    if not isinstance(subtask, dict):
        return
    strategy = subtask.get("strategy", {})
    if not isinstance(strategy, dict):
        return
    strategy_with_assignment = {
        **strategy,
        "team_assignment": subtask.get("team_assignment", {}),
    }
    _apply_host_pivot_roles_to_strategy(
        strategy_with_assignment,
        runtime_state=runtime_state,
        audit_container=task_plan,
    )
    subtask["team_assignment"] = strategy_with_assignment["team_assignment"]


def _normalize_route_clearance_reentry_navigation(
    task_plan: dict[str, Any],
    *,
    contract: dict[str, Any] | None,
) -> None:
    """Keep a local re-entry action separate from the locked route reference.

    The model owns the re-entry primitive and body-frame direction.  The
    navigation vector, however, names the persistent connected route segment;
    it must not be replaced with the transient translation vector.
    """

    conflict = contract if isinstance(contract, dict) else {}
    if not conflict.get("active", False):
        return
    subtask = task_plan.get("current_subtask", {})
    if not isinstance(subtask, dict):
        return
    strategy = subtask.get("strategy", {})
    if not isinstance(strategy, dict):
        return
    method = str(strategy.get("primitive", "")).strip()
    objective = str(strategy.get("semantic_objective", "")).strip()
    setup_directions = {
        str(item).strip()
        for key in (
            "route_reentry_translation_directions",
            "clearance_setup_translation_directions",
        )
        for item in conflict.get(key, ())
        if str(item).strip()
    }
    if not (
        method == COORDINATED_LINEAR_TRANSPORT
        and objective in setup_directions
    ):
        return
    preserved_direction = conflict.get("preserve_navigation_world_direction_xy")
    if not (
        isinstance(preserved_direction, (list, tuple))
        and len(preserved_direction) == 2
    ):
        return
    navigation = subtask.get("navigation", {})
    navigation = navigation if isinstance(navigation, dict) else {}
    previous_direction = navigation.get("world_direction_xy")
    navigation["world_direction_xy"] = list(preserved_direction)
    navigation["local_geometry_repair"] = {
        "primitive": method,
        "body_frame_direction": objective,
        "scope_checkpoint_id": str(
            conflict.get("scope_checkpoint_id", "")
        ).strip(),
    }
    subtask["navigation"] = navigation
    task_plan["current_subtask"] = subtask
    if previous_direction == list(preserved_direction):
        return
    normalizations = task_plan.get("host_schema_normalizations", [])
    normalizations = normalizations if isinstance(normalizations, list) else []
    normalizations.append(
        "preserved the validated connected-route world_direction_xy while "
        f"retaining model-selected local clearance setup action {method}:{objective}"
    )
    task_plan["host_schema_normalizations"] = normalizations


def _local_contract_checkpoint_id(
    *,
    runtime_state: dict[str, Any] | None,
    local_navigation_contract: dict[str, Any] | None,
) -> str:
    """Resolve one trusted local scope; disagreement fails closed."""

    state = runtime_state if isinstance(runtime_state, dict) else {}
    checkpoint = state.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    local = (
        local_navigation_contract
        if isinstance(local_navigation_contract, dict)
        else {}
    )
    observed = {
        value
        for value in (
            str(checkpoint.get("checkpoint_id", "")).strip(),
            str(local.get("scope_checkpoint_id", "")).strip(),
        )
        if value
    }
    return next(iter(observed)) if len(observed) == 1 else ""


def _validate_model_owned_navigation_reference(
    task_plan: dict[str, Any],
    *,
    method: str,
    final_target_navigation_authorized: bool | None,
) -> None:
    """Reject internally impossible references without choosing a route action."""

    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    navigation = subtask.get("navigation", {})
    navigation = navigation if isinstance(navigation, dict) else {}
    reference = str(navigation.get("reference", "")).strip()
    if reference == "formation_only" and method in (
        set(TURN_METHODS) | set(_TRANSPORT_METHODS)
    ):
        raise ModelRoleError(
            "current_subtask.navigation.reference='formation_only' is a pure "
            "team-repair declaration and cannot authorize linear, arc, or pivot "
            "route motion"
        )
    if reference == "final_target" and not bool(
        final_target_navigation_authorized
    ):
        raise ModelRoleError(
            "current_subtask.navigation.reference='final_target' is premature: "
            "trusted topology has not reached the target-approach phase"
        )


def _validate_orchestrator_plan_strategy(
    task_plan: dict[str, Any],
    *,
    allowed_sdk_methods: set[str] | frozenset[str],
    hard_required_method: str = "",
    forbidden_methods: set[str] | frozenset[str] = frozenset(),
    runtime_state: dict[str, Any] | None = None,
    formation_gate: dict[str, Any] | None = None,
    exhausted_relations: tuple[dict[str, Any], ...] = (),
    host_wall_clearance: dict[str, Any] | None = None,
    nonrelaxable_motion_objectives: tuple[str, ...] = (),
    trusted_body_frame_target_projection: dict[str, Any] | None = None,
    accepted_linear_continuation_contract: dict[str, Any] | None = None,
    accepted_route_reversal_guard: dict[str, Any] | None = None,
    accepted_straight_saturation_contract: dict[str, Any] | None = None,
    final_approach_transport_envelope: dict[str, Any] | None = None,
    host_wall_heading_correction_contract: dict[str, Any] | None = None,
    route_clearance_conflict_contract: dict[str, Any] | None = None,
    turn_clearance_replan_contract: dict[str, Any] | None = None,
    route_recovery_contract: dict[str, Any] | None = None,
    final_target_navigation_authorized: bool | None = None,
    model_owns_route: bool = False,
) -> None:
    """Validate one committed subtask without constructing a candidate menu."""

    advisory_only = bool(model_owns_route)
    _validate_final_lifecycle_phase_state(
        task_plan,
        runtime_state=runtime_state,
    )
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    model_owned_setup = bool(model_owns_route and subtask.get("setup_only", False))
    strategy = subtask.get("strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    method = str(strategy.get("primitive", "")).strip()
    mode = str(strategy.get("coordination_mode", "")).strip()
    objective = str(strategy.get("semantic_objective", "")).strip()
    if model_owns_route:
        _validate_model_owned_navigation_reference(
            task_plan,
            method=method,
            final_target_navigation_authorized=(
                final_target_navigation_authorized
            ),
        )
        local_navigation = build_local_navigation_contract(
            task_plan,
            runtime_state,
        )
        checkpoint_id = _local_contract_checkpoint_id(
            runtime_state=runtime_state,
            local_navigation_contract=local_navigation,
        )
        scoped_route_recovery = bool(
            _contract_matches_checkpoint(
                route_recovery_contract,
                checkpoint_id=checkpoint_id,
            )
        )
        scoped_route_clearance = (
            route_clearance_conflict_contract
            if _contract_matches_checkpoint(
                route_clearance_conflict_contract,
                checkpoint_id=checkpoint_id,
            )
            else {}
        )
        scoped_turn_clearance = (
            turn_clearance_replan_contract
            if _contract_matches_checkpoint(
                turn_clearance_replan_contract,
                checkpoint_id=checkpoint_id,
            )
            else {}
        )
        if not model_owned_setup:
            try:
                validate_strategy_against_local_navigation(
                    method=method,
                    objective=objective,
                    contract=local_navigation,
                    allow_formation_repair=bool(formation_gate),
                    allow_route_recovery=scoped_route_recovery,
                    allow_turn_clearance_replan=(
                        _strategy_matches_turn_clearance_replan_contract(
                            method=method,
                            objective=objective,
                            mode=mode,
                            contract=scoped_turn_clearance,
                        )
                    ),
                    route_clearance_conflict_contract=scoped_route_clearance,
                )
            except ValueError as exc:
                raise ModelRoleError(str(exc)) from exc
    turn_clearance_replan = _strategy_matches_turn_clearance_replan_contract(
        method=method,
        objective=objective,
        mode=mode,
        contract=turn_clearance_replan_contract,
    )
    if method not in set(allowed_sdk_methods):
        raise ModelRoleError(
            "current_subtask.strategy.primitive must be a safe SDK method; "
            f"got {method!r}; allowed={sorted(allowed_sdk_methods)}"
        )
    if not advisory_only:
        _validate_final_approach_strategy_availability(
            method=method,
            envelope=final_approach_transport_envelope,
            allow_turn_clearance_replan=turn_clearance_replan,
        )
    if hard_required_method and method != hard_required_method:
        raise ModelRoleError(
            "the current physical lifecycle gate requires "
            f"api.{hard_required_method}; got api.{method}"
        )
    if method in set(forbidden_methods):
        raise ModelRoleError(
            "the current trusted physical lifecycle forbids "
            f"api.{method}; choose a subtask consistent with the active payload state"
        )
    if mode not in coordination_modes_for_method(method):
        raise ModelRoleError(
            f"coordination_mode={mode!r} is not executable for api.{method}"
        )
    objective_parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(method, "")
    if objective_parameter:
        choices = set(
            coordination_primitive_method_spec(method)["parameters"][
                objective_parameter
            ].get("choices", ())
        )
        if objective not in choices:
            raise ModelRoleError(
                f"semantic_objective for api.{method} must be one of "
                f"{sorted(choices)}; got {objective!r}"
            )
    elif method in _STATIC_MOTION_OBJECTIVES:
        expected_objective = _STATIC_MOTION_OBJECTIVES[method]
        if objective != expected_objective:
            raise ModelRoleError(
                f"semantic_objective for api.{method} must be exactly "
                f"{expected_objective!r}; got {objective!r}"
            )
    elif not objective:
        raise ModelRoleError(
            "current_subtask.strategy.semantic_objective must be non-empty"
        )
    host_wall_heading_correction = _strategy_matches_host_wall_heading_correction(
        method=method,
        objective=objective,
        assignment=subtask.get("team_assignment", {}),
        contract=host_wall_heading_correction_contract,
    )
    exact_motion_objective = f"{method}:{objective}"
    if (
        not advisory_only
        and
        exact_motion_objective in set(nonrelaxable_motion_objectives)
        and not host_wall_heading_correction
    ):
        raise ModelRoleError(
            "the current trusted checkpoint has physics-authoritative route "
            f"regression evidence against {exact_motion_objective!r}; choose a "
            "different semantic objective or maneuver primitive"
        )
    continuation = (
        accepted_linear_continuation_contract
        if isinstance(accepted_linear_continuation_contract, dict)
        else {}
    )
    if (
        not advisory_only
        and
        continuation.get("active", False)
        and method == continuation.get("primitive")
        and objective == continuation.get("blocked_immediate_inverse")
        and not turn_clearance_replan
    ):
        blocked_inverse = str(continuation.get("blocked_immediate_inverse", "")).strip()
        accepted_direction = str(continuation.get("semantic_objective", "")).strip()
        open_directions = list(
            _model_executable_loaded_translation_directions(
                host_wall_clearance or {}
            )
        )
        valid_linear_alternatives = [
            direction for direction in open_directions if direction != blocked_inverse
        ]
        raise ModelRoleError(
            f"linear direction {blocked_inverse!r} is the exact prohibited "
            "inverse of the latest verifier-accepted same-phase direction "
            f"{accepted_direction!r}; {blocked_inverse!r} is not a geometry-"
            "changing maneuver and must not be emitted again. Current wall-clear "
            "linear alternatives after removing that inverse are "
            f"{valid_linear_alternatives}; select one of those exact directions "
            "or an unblocked geometry-changing pivot. If the accepted direction "
            "is now wall-blocked, do not continue it."
        )
    reversal_guard = (
        accepted_route_reversal_guard
        if isinstance(accepted_route_reversal_guard, dict)
        else {}
    )
    formation_required = bool(
        isinstance(formation_gate, dict) and formation_gate.get("required", False)
    )
    saturation = (
        accepted_straight_saturation_contract
        if isinstance(accepted_straight_saturation_contract, dict)
        else {}
    )
    route_clearance_conflict = (
        route_clearance_conflict_contract
        if isinstance(route_clearance_conflict_contract, dict)
        else {}
    )
    route_reentry_translations = set(
        route_clearance_conflict.get("route_reentry_translation_directions", ())
    )
    clearance_setup_translations = set(
        route_clearance_conflict.get(
            "clearance_setup_translation_directions",
            (),
        )
    )
    conflict_translations = route_reentry_translations | clearance_setup_translations
    conflict_turns = set(
        route_clearance_conflict.get(
            "safe_heading_correction_turn_directions",
            (),
        )
    )
    conflict_repair_allowed = bool(
        route_clearance_conflict.get("active", False)
        and (
            (method == COORDINATED_PIVOT and objective in conflict_turns)
            or (
                method == COORDINATED_LINEAR_TRANSPORT
                and objective in conflict_translations
            )
        )
    )
    saturation_repair_allowed = bool(
        conflict_repair_allowed
        or (
            not route_clearance_conflict.get("active", False)
            and method == COORDINATED_PIVOT
        )
    )
    if (
        not advisory_only
        and
        route_clearance_conflict.get("active", False)
        and not hard_required_method
        and not formation_required
        and not conflict_repair_allowed
    ):
        raise ModelRoleError(
            "the active route direction has no model-supported translation tier "
            "at this checkpoint. Preserve its navigation reference and choose "
            "one measured clearance setup: coordinated_pivot with semantic_objective "
            f"in {sorted(conflict_turns)}, or synchronized "
            "coordinated_linear_transport with semantic_objective in "
            f"{sorted(conflict_translations)}. These actions are setup only and "
            "cannot complete the route phase"
        )
    if (
        not advisory_only
        and
        saturation.get("active", False)
        and not hard_required_method
        and not formation_required
        and not saturation_repair_allowed
    ):
        saturated_direction = str(saturation.get("saturated_direction", "")).strip()
        open_turns = (
            list(
                route_clearance_conflict.get(
                    "safe_heading_correction_turn_directions",
                    (),
                )
            )
            if route_clearance_conflict.get("active", False)
            else list(saturation.get("unblocked_turn_directions", ()))
        )
        reentry_directions = sorted(conflict_translations)
        raise ModelRoleError(
            "accepted straight-passage saturation requires a geometry-changing "
            "next subtask: select a measured-safe coordinated_pivot or route-reentry "
            "linear translation and infer its roles from the global view. Do not issue "
            f"another room-scale {saturated_direction!r} linear segment; unblocked "
            f"signed turn options are {open_turns}; route-reentry translations are "
            f"{reentry_directions}"
        )
    if (
        not advisory_only
        and
        reversal_guard.get("active", False)
        and method == COORDINATED_LINEAR_TRANSPORT
        and objective
        == reversal_guard.get("blocked_immediate_return_direction")
        and not turn_clearance_replan
    ):
        earlier_direction = str(
            reversal_guard.get("earlier_direction", "")
        ).strip()
        latest_direction = str(
            reversal_guard.get("latest_direction", "")
        ).strip()
        raise ModelRoleError(
            "accepted route reversal guard prohibits an immediate return to "
            f"{earlier_direction!r} after adjacent accepted linear motions "
            f"{earlier_direction!r} -> {latest_direction!r}. The safe checkpoint "
            "is retained, but another inverse translation would create a route "
            "oscillation. Independently choose any other executable maneuver from "
            "the current global view; the host does not require a pivot or select "
            "a replacement direction."
        )
    if not advisory_only and not host_wall_heading_correction:
        _validate_strategy_against_host_wall_clearance(
            method=method,
            objective=objective,
            host_wall_clearance=host_wall_clearance,
        )
    if not advisory_only and not turn_clearance_replan:
        _validate_final_approach_target_direction(
            task_plan,
            method=method,
            objective=objective,
            projection=trusted_body_frame_target_projection,
        )
    if method == COORDINATED_PIVOT:
        assignment = subtask.get("team_assignment", {})
        assignment = assignment if isinstance(assignment, dict) else {}
        roles_by_robot = {
            robot_id: str(
                assignment.get(robot_id, {}).get("role", "")
                if isinstance(assignment.get(robot_id), dict)
                else ""
            ).strip()
            for robot_id in ("front", "back")
        }
        roles = sorted(roles_by_robot.values())
        if roles != ["inner", "outer"]:
            raise ModelRoleError(
                "coordinated_pivot task assignment requires exactly one inner "
                "and one outer role"
            )
        expected_roles = pivot_role_assignment(runtime_state or {}, objective)
        if not model_owns_route and expected_roles and roles_by_robot != expected_roles:
            inner_robot = next(
                robot_id for robot_id, role in expected_roles.items() if role == "inner"
            )
            outer_robot = next(
                robot_id for robot_id, role in expected_roles.items() if role == "outer"
            )
            raise ModelRoleError(
                "coordinated_pivot role geometry requires "
                f"{inner_robot}=inner and {outer_robot}=outer for "
                f"turn_direction={objective}; front/back are immutable robot IDs"
            )
    _validate_orchestrator_turn_projection_evidence(
        task_plan,
        method=method,
        objective=objective,
        runtime_state=runtime_state,
        required=bool(model_owns_route),
    )
    if method == COORDINATED_ALIGNMENT:
        assignment = subtask.get("team_assignment", {})
        assignment = assignment if isinstance(assignment, dict) else {}
        roles = sorted(
            str(
                assignment.get(robot_id, {}).get("role", "")
                if isinstance(assignment.get(robot_id), dict)
                else ""
            ).strip()
            for robot_id in ("front", "back")
        )
        if roles != ["corrector", "reference"]:
            raise ModelRoleError(
                "coordinated_alignment task assignment requires exactly one "
                "reference and one corrector role"
            )
    if not advisory_only:
        _validate_strategy_not_checkpoint_exhausted(
            method=method,
            objective=objective,
            mode=mode,
            assignment=subtask.get("team_assignment", {}),
            exhausted_relations=exhausted_relations,
        )
        _validate_strategy_against_host_formation_gate(
            method=method,
            objective=objective,
            mode=mode,
            assignment=subtask.get("team_assignment", {}),
            formation_gate=(
                _trusted_checkpoint_longitudinal_alignment(runtime_state or {})
                if formation_gate is None
                else formation_gate
            ),
        )
    _validate_model_strategy_turn_episode(
        method=method,
        objective=objective,
        assignment=subtask.get("team_assignment", {}),
        runtime_state=runtime_state,
        local_navigation_contract=build_local_navigation_contract(
            task_plan,
            runtime_state,
        ),
        host_wall_clearance=host_wall_clearance,
        host_wall_heading_correction_contract=(
            host_wall_heading_correction_contract
        ),
        route_clearance_conflict_contract=route_clearance_conflict_contract,
        active_only=model_owns_route,
        allow_model_owned_setup=model_owned_setup,
    )
    if runtime_state and not advisory_only:
        current_state_claims = ". ".join(
            str(value)
            for value in (
                task_plan.get("rationale", ""),
                subtask.get("objective", ""),
                subtask.get("guidance", ""),
                subtask.get("completion_condition", ""),
            )
            if value
        )
        try:
            _validate_selection_current_state_claims(
                selected_method=method,
                reason=current_state_claims,
                runtime_state=runtime_state,
            )
        except ModelAgentError as exc:
            raise ModelRoleError(str(exc)) from exc


def _formal_shadow_check(
    name: str,
    validator: Callable[[], None],
) -> dict[str, Any]:
    """Evaluate one retired Host heuristic without changing admissibility."""

    try:
        validator()
    except (ModelAgentError, ModelRoleError, ValueError) as exc:
        return {
            "name": str(name),
            "legacy_would_reject": True,
            "reason": str(exc),
        }
    return {
        "name": str(name),
        "legacy_would_reject": False,
        "reason": "",
    }


def _assert_formal_hard_contract_nonempty(
    *,
    formal_eval: bool,
    allowed_methods: set[str] | frozenset[str],
    forbidden_methods: set[str] | frozenset[str] = frozenset(),
    required_method: str = "",
    mutation_contract: dict[str, Any] | None = None,
) -> None:
    """Fail closed only for a provably contradictory Host-owned hard contract."""

    if not formal_eval:
        return
    allowed = {str(item).strip() for item in allowed_methods if str(item).strip()}
    forbidden = {
        str(item).strip() for item in forbidden_methods if str(item).strip()
    }
    executable = allowed - forbidden
    reasons: list[str] = []
    required = str(required_method).strip()
    if required and required not in executable:
        reasons.append(
            f"required lifecycle method {required!r} is not executable"
        )
    if not required and not executable:
        reasons.append("all SDK methods are forbidden by Host-owned lifecycle gates")
    contract = mutation_contract if isinstance(mutation_contract, dict) else {}
    if contract and not [
        str(item).strip()
        for item in contract.get("available_levels", ())
        if str(item).strip()
    ]:
        reasons.append("mutation contract exposes no available mutation level")
    if reasons:
        raise HostContractError(
            "HOST_CONTRACT_ERROR: "
            + "; ".join(reasons)
            + f"; allowed={sorted(allowed)} forbidden={sorted(forbidden)}"
        )


def _write_formal_shadow_advisory(
    output_dir: Path,
    *,
    attempt: int,
    checkpoint_id: str,
    stage: str,
    strategy: dict[str, Any],
    checks: list[dict[str, Any]],
) -> Path:
    """Persist model action, retired-heuristic verdicts, and later physics result."""

    path = Path(output_dir) / f"host_advisory_attempt_{int(attempt):02d}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    stages = payload.get("stages", [])
    stages = [item for item in stages if isinstance(item, dict)]
    stage_record = {
        "stage": str(stage),
        "checkpoint_id": str(checkpoint_id).strip(),
        "proposed_action": {
            "primitive": str(strategy.get("primitive", "")).strip(),
            "semantic_objective": str(
                strategy.get("semantic_objective", "")
            ).strip(),
            "coordination_mode": str(
                strategy.get("coordination_mode", "")
            ).strip(),
            "team_assignment": strategy.get("team_assignment", {}),
        },
        "shadow_checks": checks,
        "legacy_would_reject": any(
            bool(item.get("legacy_would_reject", False)) for item in checks
        ),
        "enforcement": "advisory_only",
    }
    stages = [item for item in stages if item.get("stage") != stage]
    stages.append(stage_record)
    payload.update(
        {
            "schema_version": 1,
            "kind": "v50_lite_host_advisory",
            "attempt": int(attempt),
            "checkpoint_id": str(checkpoint_id).strip(),
            "stages": stages,
            "actual_physical_verifier_result": payload.get(
                "actual_physical_verifier_result",
                {"status": "pending_physical_dispatch"},
            ),
        }
    )
    write_json(path, payload)
    return path


def _validate_orchestrator_turn_projection_evidence(
    task_plan: dict[str, Any],
    *,
    method: str,
    objective: str,
    runtime_state: dict[str, Any] | None,
    required: bool,
) -> None:
    """Validate structured signed-turn fields without parsing model prose."""

    if method not in _SIGNED_TURN_METHODS:
        return
    state = runtime_state if isinstance(runtime_state, dict) else {}
    projection = _fixed_global_camera_heading_projection(state)
    if not projection:
        return
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    comparison = subtask.get("turn_comparison", {})
    comparison = comparison if isinstance(comparison, dict) else {}
    if not comparison and not required:
        return
    if not comparison:
        raise ModelRoleError(
            "signed turn requires current_subtask.turn_comparison; natural-language "
            "rationale is audit text and cannot satisfy this structured contract"
        )
    alternative = (
        "counterclockwise" if objective == "clockwise" else "clockwise"
    )
    selected_direction = str(
        comparison.get("selected_direction", "")
    ).strip().lower()
    rejected_direction = str(
        comparison.get("rejected_direction", "")
    ).strip().lower()
    if (
        not bool(comparison.get("active", False))
        or selected_direction != objective
        or rejected_direction != alternative
    ):
        raise ModelRoleError(
            "STRUCTURED_TURN_COMPARISON_CONFLICT: selected_direction must match "
            "strategy.semantic_objective and rejected_direction must be its opposite"
        )
    selected_projection = projection.get(objective, {})
    rejected_projection = projection.get(alternative, {})
    selected_projection = (
        selected_projection if isinstance(selected_projection, dict) else {}
    )
    rejected_projection = (
        rejected_projection if isinstance(rejected_projection, dict) else {}
    )
    expected_selected = _normalize_screen_heading_label(
        selected_projection.get("screen_heading", "")
    )
    expected_rejected = _normalize_screen_heading_label(
        rejected_projection.get("screen_heading", "")
    )
    actual_selected = _normalize_screen_heading_label(
        comparison.get("selected_heading", "")
    )
    actual_rejected = _normalize_screen_heading_label(
        comparison.get("rejected_heading", "")
    )
    if (
        not expected_selected
        or not expected_rejected
        or actual_selected != expected_selected
        or actual_rejected != expected_rejected
    ):
        raise ModelRoleError(
            "STRUCTURED_TURN_COMPARISON_CONFLICT: selected_heading and "
            "rejected_heading must match their own host-projected screen headings; "
            f"expected {objective}={expected_selected!r}, "
            f"{alternative}={expected_rejected!r}"
        )


def _normalize_screen_heading_label(value: object) -> str:
    """Canonicalize lexical heading variants without changing turn semantics."""

    normalized = " ".join(str(value).strip().lower().replace("_", " ").split())
    normalized = re.sub(
        r"^screen\s*[- ]?\s*heading\s*(?::|=|-)?\s*",
        "",
        normalized,
    )
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")


def _validate_final_approach_target_direction(
    task_plan: dict[str, Any],
    *,
    method: str,
    objective: str,
    projection: dict[str, Any] | None,
) -> None:
    """Keep final-approach body-frame translations pointed at the real target."""

    geometry = projection if isinstance(projection, dict) else {}
    if method != COORDINATED_LINEAR_TRANSPORT or not (
        _task_plan_is_final_approach(task_plan)
        or bool(geometry.get("target_capture_active", False))
    ):
        return
    scores = geometry.get("candidate_target_alignment", {})
    scores = scores if isinstance(scores, dict) else {}
    try:
        selected_score = float(scores[objective])
    except (KeyError, TypeError, ValueError):
        return
    finite_scores = {
        str(direction): float(score)
        for direction, score in scores.items()
        if str(direction) in LOADED_TRANSLATION_DIRECTIONS
        and isinstance(score, (int, float))
        and isfinite(float(score))
    }
    if not finite_scores:
        return
    best_direction, best_score = max(
        finite_scores.items(),
        key=lambda item: item[1],
    )
    target_away_setup_authorized = bool(
        geometry.get("target_away_setup_authorized", False)
    )
    if selected_score < -0.05 and not target_away_setup_authorized:
        raise ModelRoleError(
            "final-approach SDK directions are robot-body-frame commands: "
            f"{objective!r} points away from the physical target "
            f"(alignment={selected_score:.3f}). The strongest target-facing "
            f"body-frame direction is {best_direction!r} "
            f"(alignment={best_score:.3f}); choose {best_direction!r} exactly "
            "for linear transport or choose a geometry-changing maneuver. "
            "Camera-space ahead/left/right labels are advisory only"
        )
    if (
        best_score >= 0.85
        and selected_score < 0.35
        and not target_away_setup_authorized
    ):
        raise ModelRoleError(
            "the trusted final-target projection has one unambiguous body-frame "
            f"translation: {best_direction!r} (alignment={best_score:.3f}); "
            f"{objective!r} has alignment={selected_score:.3f}. Preserve the "
            "high-level phase, but replace the body-frame semantic objective"
        )


def _validate_parameter_tuning_task_plan(
    task_plan: dict[str, Any],
    *,
    parent_strategy: dict[str, Any],
) -> None:
    """Prevent route replanning after Manager selects parameter-only repair."""

    child = _execution_strategy_from_task_plan(task_plan)
    expected = {
        key: str(parent_strategy.get(key, "")).strip()
        for key in ("primitive", "semantic_objective", "coordination_mode")
    }
    actual = {
        key: str(child.get(key, "")).strip()
        for key in ("primitive", "semantic_objective", "coordination_mode")
    }

    def roles(strategy: dict[str, Any]) -> dict[str, str]:
        assignment = strategy.get("team_assignment", {})
        assignment = assignment if isinstance(assignment, dict) else {}
        return {
            robot_id: str(
                assignment.get(robot_id, {}).get("role", "")
                if isinstance(assignment.get(robot_id), dict)
                else ""
            ).strip()
            for robot_id in ("front", "back")
        }

    expected_roles = roles(parent_strategy)
    actual_roles = roles(child)
    if actual != expected or actual_roles != expected_roles:
        raise ModelRoleError(
            "parameter_tuning strategy contract requires the Task Orchestrator "
            "to preserve the rejected transaction's executable strategy exactly: "
            f"primitive={expected['primitive']!r}, "
            f"semantic_objective={expected['semantic_objective']!r}, "
            f"coordination_mode={expected['coordination_mode']!r}, "
            f"roles={expected_roles}; got strategy={actual}, roles={actual_roles}"
        )


def _validate_coordination_mutation_task_plan(
    task_plan: dict[str, Any],
    *,
    parent_strategy: dict[str, Any],
) -> None:
    """Keep Orchestrator semantics inside a coordination-only repair mask."""

    child = _execution_strategy_from_task_plan(task_plan)
    immutable_fields = ("primitive", "semantic_objective")
    mismatches = {
        field: {
            "parent": str(parent_strategy.get(field, "")).strip(),
            "orchestrator": str(child.get(field, "")).strip(),
        }
        for field in immutable_fields
        if str(parent_strategy.get(field, "")).strip()
        != str(child.get(field, "")).strip()
    }
    if mismatches:
        raise ModelRoleError(
            "coordination_mutation task-plan contract preserves the rejected "
            "primitive and semantic objective; Task Orchestrator may change only "
            "coordination_mode and temporary "
            f"front/back roles. mismatches={mismatches}"
        )
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    strategy = subtask.get("strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    parent_formation = str(parent_strategy.get("formation_strategy", "")).strip()
    if parent_formation:
        # formation_strategy is descriptive metadata. The executable relation is
        # carried by coordination_mode and team_assignment, so canonicalize model
        # paraphrases instead of turning equivalent plans into generation stalls.
        strategy["formation_strategy"] = parent_formation
        subtask["strategy"] = strategy
        task_plan["current_subtask"] = subtask


def _manager_decision_for_host_wall_heading_correction(
    manager_decision: dict[str, Any],
    *,
    contract: dict[str, Any] | None,
) -> dict[str, Any]:
    """Promote a wall-required maneuver change above stale parameter tuning."""

    decision = dict(manager_decision)
    correction = contract if isinstance(contract, dict) else {}
    if (
        not correction.get("active", False)
        or decision.get("mutation_level") == "strategy_mutation"
    ):
        return decision

    previous_level = str(decision.get("mutation_level", "")).strip()
    turn_direction = str(correction.get("turn_direction", "")).strip()
    desired_translation = str(
        correction.get("desired_translation_direction", "")
    ).strip()
    return {
        "mutation_level": "strategy_mutation",
        "failure_hypothesis": (
            f"The requested {desired_translation or 'linear'} probe is blocked by "
            "current authored-wall geometry, so tuning its magnitude cannot make "
            "the executable transaction physically available."
        ),
        "evidence": [
            "current authored-wall clearance blocks the pending linear probe",
            (
                "the host geometry contract requires one bounded "
                f"coordinated_pivot:{turn_direction or 'signed'} heading correction"
            ),
        ],
        "preserve": [
            "global mission and current route phase",
            "accepted checkpoint progress",
            "bilateral payload support and physical safety",
        ],
        "mutate": [
            "maneuver family",
            "semantic motion objective",
            "temporary coordination roles",
        ],
        "forbidden": [
            "replay the exact rejected executable transaction",
            "dispatch a wall-blocked translation",
            "lose bilateral payload support or contact a wall or floor",
        ],
        "success_test": [
            "the exact host wall-heading correction passes the physical verifier",
            "the pending linear probe is re-evaluated from the new checkpoint",
        ],
        "inheritance": [
            "global mission",
            "current route phase",
            "accepted checkpoint progress",
            "payload support and physical safety",
        ],
        "confidence": float(decision.get("confidence", 1.0) or 1.0),
        "host_mutation_level_override": {
            "from": previous_level,
            "to": "strategy_mutation",
            "reason": "authored_wall_geometry_requires_maneuver_family_change",
        },
    }


def _lifecycle_consistent_previous_task_plan(
    task_plan: dict[str, Any],
    *,
    forbidden_methods: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Drop only a persisted plan whose active subtask contradicts live physics."""

    plan = task_plan if isinstance(task_plan, dict) else {}
    subtask = plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    strategy = subtask.get("strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    method = str(strategy.get("primitive", "")).strip()
    if method and method in set(forbidden_methods):
        return {}
    return json.loads(json.dumps(plan, ensure_ascii=False, default=str))


def _retire_superseded_formation_subtask(
    task_plan: dict[str, Any],
    *,
    formation_gate: dict[str, Any],
) -> dict[str, Any]:
    """Keep the global plan without anchoring on a rejected formation repair."""

    plan = json.loads(
        json.dumps(
            task_plan if isinstance(task_plan, dict) else {},
            ensure_ascii=False,
            default=str,
        )
    )
    if not plan or not formation_gate:
        return plan
    previous_subtask = plan.pop("current_subtask", {})
    previous_subtask = previous_subtask if isinstance(previous_subtask, dict) else {}
    plan.pop("constraints", None)
    plan.pop("rationale", None)
    plan.pop("checkpoint_transition", None)
    plan["rejected_subtask_history"] = {
        "subtask_id": str(previous_subtask.get("subtask_id", "")).strip(),
        "outcome": "measured_alignment_regression",
        "strategy_is_not_a_next_step_constraint": True,
    }
    plan["checkpoint_transition"] = {
        "status": "formation_repair_strategy_replan_required",
        "instruction": (
            "The residual formation error remains observable, but the matching "
            "formation-gate relation regressed when physically executed. Preserve "
            "the global route phase and plan a materially different repair."
        ),
    }
    return plan


def _supersede_accepted_current_subtask(
    task_plan: dict[str, Any],
    *,
    formation_gate: dict[str, Any],
) -> dict[str, Any]:
    """Keep route history while removing a completed subtask's stale strategy."""

    plan = json.loads(
        json.dumps(
            task_plan if isinstance(task_plan, dict) else {},
            ensure_ascii=False,
            default=str,
        )
    )
    if not plan or not isinstance(formation_gate, dict) or not formation_gate:
        return plan
    previous_subtask = plan.pop("current_subtask", {})
    previous_subtask = previous_subtask if isinstance(previous_subtask, dict) else {}
    plan.pop("constraints", None)
    plan.pop("rationale", None)
    plan["accepted_subtask_history"] = {
        "subtask_id": str(previous_subtask.get("subtask_id", "")).strip(),
        "outcome": "accepted_checkpoint_committed",
        "strategy_is_not_a_next_step_constraint": True,
    }
    plan["checkpoint_transition"] = {
        "status": "formation_repair_required_before_route_motion",
        "primitive": str(formation_gate.get("primitive", "")).strip(),
        "semantic_objective": str(formation_gate.get("semantic_objective", "")).strip(),
        "coordination_mode": str(formation_gate.get("coordination_mode", "")).strip(),
        "instruction": (
            "Plan the next subtask from host_formation_gate. The accepted prior "
            "subtask is history and none of its preserve/parameter constraints remain active."
        ),
    }
    return plan


def _retire_accepted_current_subtask_for_replan(
    task_plan: dict[str, Any],
) -> dict[str, Any]:
    """Make every accepted bounded action history before the next replan."""

    plan = json.loads(
        json.dumps(
            task_plan if isinstance(task_plan, dict) else {},
            ensure_ascii=False,
            default=str,
        )
    )
    if not plan:
        return plan
    previous_subtask = plan.pop("current_subtask", {})
    previous_subtask = previous_subtask if isinstance(previous_subtask, dict) else {}
    plan.pop("constraints", None)
    plan.pop("rationale", None)
    plan["accepted_subtask_history"] = {
        "subtask_id": str(previous_subtask.get("subtask_id", "")).strip(),
        "outcome": "accepted_checkpoint_committed",
        "strategy_is_not_a_next_step_constraint": True,
    }
    plan["checkpoint_transition"] = {
        "status": "accepted_checkpoint_replan_required",
        "instruction": (
            "Reconfirm the active mission phase from host_phase_evidence and "
            "plan a fresh bounded subtask from the current trusted physics and "
            "global image. The accepted primitive, direction, navigation "
            "reference, and parameters are history rather than defaults."
        ),
    }
    return plan


def _execution_strategy_from_task_plan(
    task_plan: dict[str, Any],
    *,
    required_method: str = "",
    required_reason: str = "",
) -> dict[str, Any]:
    """Return the Orchestrator commitment or one mandatory lifecycle strategy."""

    subtask = task_plan.get("current_subtask", {})
    if not isinstance(subtask, dict):
        raise ModelAgentError("Task Orchestrator plan has no current_subtask")
    strategy = subtask.get("strategy", {})
    if not isinstance(strategy, dict):
        raise ModelAgentError("Task Orchestrator current_subtask has no strategy")
    result = dict(strategy)
    assignment = subtask.get("team_assignment")
    if not isinstance(assignment, dict):
        assignment = result.get("team_assignment")
    result["team_assignment"] = json.loads(
        json.dumps(assignment, ensure_ascii=False, default=str)
    )
    if required_method:
        # The host owns the payload lifecycle gate, but the Orchestrator remains
        # the single source of truth for semantic front/back role assignment.
        result["primitive"] = required_method
        result["semantic_objective"] = (
            required_reason
            or str(result.get("semantic_objective", "")).strip()
            or required_method
        )
        result["coordination_mode"] = "sync"
    return result


def _task_plan_with_execution_strategy(
    task_plan: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    """Attach the evolved team strategy while preserving the global plan."""

    result = json.loads(json.dumps(task_plan, ensure_ascii=False, default=str))
    subtask = result.get("current_subtask", {})
    if not isinstance(subtask, dict):
        raise ModelAgentError("Task Orchestrator plan has no current_subtask")
    subtask["strategy"] = json.loads(
        json.dumps(strategy, ensure_ascii=False, default=str)
    )
    subtask["team_assignment"] = json.loads(
        json.dumps(
            strategy.get("team_assignment", subtask.get("team_assignment", {})),
            ensure_ascii=False,
            default=str,
        )
    )
    result["current_subtask"] = subtask
    return result


def _validate_orchestrator_preserves_locked_child(
    task_plan: dict[str, Any],
    *,
    locked_child: dict[str, Any],
) -> None:
    """Keep Task Orchestrator downstream of the sole mutation writer."""

    proposed = _execution_strategy_from_task_plan(task_plan)
    executable_fields = (
        "primitive",
        "semantic_objective",
        "coordination_mode",
    )
    field_mismatches = {
        field: {
            "locked": str(locked_child.get(field, "")).strip(),
            "orchestrator": str(proposed.get(field, "")).strip(),
        }
        for field in executable_fields
        if str(locked_child.get(field, "")).strip()
        != str(proposed.get(field, "")).strip()
    }

    def role_projection(strategy: dict[str, Any]) -> dict[str, str]:
        assignment = strategy.get("team_assignment", {})
        assignment = assignment if isinstance(assignment, dict) else {}
        return {
            robot_id: str(
                assignment.get(robot_id, {}).get("role", "")
                if isinstance(assignment.get(robot_id), dict)
                else ""
            ).strip()
            for robot_id in ("front", "back")
        }

    locked_roles = role_projection(locked_child)
    proposed_roles = role_projection(proposed)
    if field_mismatches or proposed_roles != locked_roles:
        raise ModelRoleError(
            "ORCHESTRATOR_CHILD_MISMATCH: Task Orchestrator is downstream of the "
            "Host-locked Mutation Operator child and may not rewrite executable "
            f"fields; field_mismatches={field_mismatches}, "
            f"locked_roles={locked_roles}, orchestrator_roles={proposed_roles}"
        )


def _strategy_mutation_replan_contract(
    strategy: dict[str, Any],
) -> dict[str, Any]:
    """Project the structural child contract without role-objective prose."""

    assignment = strategy.get("team_assignment", {})
    assignment = assignment if isinstance(assignment, dict) else {}
    return {
        key: str(strategy.get(key, "")).strip()
        for key in (
            "primitive",
            "semantic_objective",
            "coordination_mode",
            "formation_strategy",
            "synchronization_intent",
        )
    } | {
        "team_assignment": {
            robot_id: {
                "role": str(
                    assignment.get(robot_id, {}).get("role", "")
                    if isinstance(assignment.get(robot_id), dict)
                    else ""
                ).strip()
            }
            for robot_id in ("front", "back")
        }
    }


_STRATEGY_REPLAN_NAVIGATION_IDENTITY_FIELDS = (
    "validated_route_segment_id",
    "route_reference_locked",
    "route_observation_captured_at_unix",
)


def _inherit_strategy_replan_navigation_contract(
    task_plan: dict[str, Any],
    *,
    parent_plan: dict[str, Any],
) -> None:
    """Inherit route identity without rewriting child navigation semantics.

    Strategy mutation owns the navigation reference, direction vectors, local
    geometry repair, primitive, mode, and temporary roles. The Host may carry
    checkpoint-validated route identity metadata forward, but it must not turn a
    schema-valid child into a semantically different plan.
    """

    parent_subtask = parent_plan.get("current_subtask", {})
    parent_subtask = parent_subtask if isinstance(parent_subtask, dict) else {}
    parent_navigation = parent_subtask.get("navigation", {})
    if not isinstance(parent_navigation, dict) or not parent_navigation:
        return

    subtask = task_plan.get("current_subtask", {})
    if not isinstance(subtask, dict):
        return
    navigation = subtask.get("navigation", {})
    navigation = navigation if isinstance(navigation, dict) else {}
    changed_fields: list[str] = []
    for navigation_field in _STRATEGY_REPLAN_NAVIGATION_IDENTITY_FIELDS:
        if navigation_field not in parent_navigation:
            continue
        inherited = json.loads(
            json.dumps(
                parent_navigation[navigation_field],
                ensure_ascii=False,
                default=str,
            )
        )
        if navigation.get(navigation_field) != inherited:
            navigation[navigation_field] = inherited
            changed_fields.append(navigation_field)
    subtask["navigation"] = navigation
    task_plan["current_subtask"] = subtask
    if not changed_fields:
        return

    normalizations = task_plan.get("host_schema_normalizations", [])
    normalizations = normalizations if isinstance(normalizations, list) else []
    normalizations.append(
        "strategy replan navigation: inherited checkpoint route identity fields "
        + ", ".join(changed_fields)
        + "; child reference, direction vectors, local geometry repair, primitive, "
        "mode, and roles were preserved"
    )
    task_plan["host_schema_normalizations"] = normalizations


def _mutation_lock_local_navigation_contract(
    task_plan: dict[str, Any],
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recover an authoritative active-turn contract before locking a child.

    A formation-repair parent can temporarily carry ``formation_only`` even
    while trusted runtime still owns an unfinished route/local-navigation turn.
    In single-writer recovery the downstream Orchestrator cannot repair an
    already locked executable child, so that persistent runtime reference must
    be checked at the Mutation Operator boundary.  This helper restores only
    the existing reference and geometry; it does not select an action.
    """

    state = runtime_state if isinstance(runtime_state, dict) else {}
    fallback = build_local_navigation_contract(task_plan, state)
    episode = state.get("turn_episode", {})
    episode = episode if isinstance(episode, dict) else {}
    purpose = str(episode.get("purpose", "")).strip()
    if not (
        bool(episode.get("active", False))
        and purpose in {"route_turn", "local_navigation_turn"}
    ):
        return fallback

    def mark_required(contract: dict[str, Any]) -> dict[str, Any]:
        marked = dict(contract)
        marked["mutation_lock_required"] = True
        marked["mutation_lock_source"] = "trusted_runtime_turn_episode"
        marked["turn_episode_status"] = str(episode.get("status", "")).strip()
        return marked

    # Legacy route-turn episodes own a signed turn and roles but no local world
    # reference. Marking the lock still makes the turn-episode validator run
    # before Task Orchestrator, so loaded linear motion cannot be locked early.
    if purpose == "route_turn":
        return mark_required(fallback)

    reference = str(episode.get("reference", "")).strip()
    if reference not in {
        "final_target",
        "runtime_local_goal",
        "visible_connected_free_space",
    }:
        return mark_required(fallback)

    source_subtask = task_plan.get("current_subtask", {})
    source_subtask = source_subtask if isinstance(source_subtask, dict) else {}
    source_navigation = source_subtask.get("navigation", {})
    source_navigation = (
        source_navigation if isinstance(source_navigation, dict) else {}
    )
    navigation = {
        key: json.loads(json.dumps(source_navigation[key]))
        for key in _STRATEGY_REPLAN_NAVIGATION_IDENTITY_FIELDS
        if key in source_navigation
    }
    navigation.update(
        {
            "reference": reference,
            "route_reference_locked": True,
            "route_observation_captured_at_unix": episode.get(
                "route_observation_captured_at_unix",
                navigation.get("route_observation_captured_at_unix", 0.0),
            ),
        }
    )
    world_direction = episode.get("world_direction_xy")
    if isinstance(world_direction, (list, tuple)) and len(world_direction) >= 2:
        navigation["world_direction_xy"] = [
            world_direction[0],
            world_direction[1],
        ]

    synthetic_plan = {
        "current_subtask": {
            **json.loads(json.dumps(source_subtask)),
            "navigation": navigation,
        }
    }
    contract = build_local_navigation_contract(synthetic_plan, state)
    if not contract:
        return mark_required(fallback)
    return mark_required(contract)


def _validate_strategy_mutation_orchestrator_replan(
    task_plan: dict[str, Any],
    *,
    previous_task_plan: dict[str, Any],
    approved_strategy: dict[str, Any],
) -> None:
    """Require a strategy mutation to replace the complete subtask semantics."""

    child = _execution_strategy_from_task_plan(task_plan)
    exact_fields = (
        "primitive",
        "semantic_objective",
        "coordination_mode",
        "formation_strategy",
        "synchronization_intent",
    )
    mismatches = {
        key: {
            "approved": str(approved_strategy.get(key, "")).strip(),
            "orchestrator": str(child.get(key, "")).strip(),
        }
        for key in exact_fields
        if str(child.get(key, "")).strip()
        != str(approved_strategy.get(key, "")).strip()
    }
    approved_assignment = approved_strategy.get("team_assignment", {})
    child_assignment = child.get("team_assignment", {})
    approved_assignment = (
        approved_assignment if isinstance(approved_assignment, dict) else {}
    )
    child_assignment = child_assignment if isinstance(child_assignment, dict) else {}
    approved_roles = {
        robot_id: str(
            approved_assignment.get(robot_id, {}).get("role", "")
            if isinstance(approved_assignment.get(robot_id), dict)
            else ""
        ).strip()
        for robot_id in ("front", "back")
    }
    child_roles = {
        robot_id: str(
            child_assignment.get(robot_id, {}).get("role", "")
            if isinstance(child_assignment.get(robot_id), dict)
            else ""
        ).strip()
        for robot_id in ("front", "back")
    }
    if mismatches or child_roles != approved_roles:
        raise ModelRoleError(
            "strategy-replan must preserve the approved mutation contract exactly: "
            f"field_mismatches={mismatches}, approved_roles={approved_roles}, "
            f"orchestrator_roles={child_roles}"
        )

    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    previous_subtask = previous_task_plan.get("current_subtask", {})
    previous_subtask = previous_subtask if isinstance(previous_subtask, dict) else {}
    semantic_fields = {
        "current_subtask.objective": str(subtask.get("objective", "")).strip(),
        "current_subtask.guidance": str(subtask.get("guidance", "")).strip(),
        "current_subtask.completion_condition": str(
            subtask.get("completion_condition", "")
        ).strip(),
        "rationale": str(task_plan.get("rationale", "")).strip(),
    }
    for robot_id in ("front", "back"):
        assignment = child_assignment.get(robot_id, {})
        assignment = assignment if isinstance(assignment, dict) else {}
        semantic_fields[f"current_subtask.team_assignment.{robot_id}.objective"] = str(
            assignment.get("objective", "")
        ).strip()

    markers = _strategy_replan_semantic_markers(approved_strategy)
    stale_fields: list[str] = []
    unrelated_fields: list[str] = []
    previous_values = {
        "current_subtask.objective": str(previous_subtask.get("objective", "")).strip(),
        "current_subtask.guidance": str(previous_subtask.get("guidance", "")).strip(),
        "current_subtask.completion_condition": str(
            previous_subtask.get("completion_condition", "")
        ).strip(),
        "rationale": str(previous_task_plan.get("rationale", "")).strip(),
    }
    previous_assignment = previous_subtask.get("team_assignment", {})
    previous_assignment = (
        previous_assignment if isinstance(previous_assignment, dict) else {}
    )
    for robot_id in ("front", "back"):
        assignment = previous_assignment.get(robot_id, {})
        assignment = assignment if isinstance(assignment, dict) else {}
        previous_values[f"current_subtask.team_assignment.{robot_id}.objective"] = str(
            assignment.get("objective", "")
        ).strip()

    for path, text in semantic_fields.items():
        previous_text = previous_values.get(path, "")
        if previous_text and text.casefold() == previous_text.casefold():
            stale_fields.append(path)
        lowered = text.casefold()
        required_markers = markers
        if path.startswith("current_subtask.team_assignment."):
            robot_id = path.split(".")[2]
            required_markers = _strategy_replan_role_markers(
                method=str(approved_strategy.get("primitive", "")),
                role=approved_roles.get(robot_id, ""),
                fallback=markers,
            )
        if not text or not any(marker in lowered for marker in required_markers):
            unrelated_fields.append(path)
    # A model-only retry may begin from a partially persisted, already-correct
    # replan. Equality with that draft is not evidence that rejected physical
    # semantics survived. The executable contract and semantic relevance are
    # the authoritative checks; only text unrelated to the approved maneuver
    # should block code generation.
    if unrelated_fields:
        raise ModelRoleError(
            "strategy-replan must describe "
            "the approved maneuver in every semantic field: "
            f"unchanged_fields={stale_fields}, "
            f"unrelated_fields={unrelated_fields}, "
            f"required_maneuver_markers={sorted(markers)}"
        )


def _strategy_replan_semantic_markers(
    strategy: dict[str, Any],
) -> frozenset[str]:
    method = str(strategy.get("primitive", "")).strip().casefold()
    objective = str(strategy.get("semantic_objective", "")).strip().casefold()
    markers = {
        objective,
        objective.replace("_", " "),
    }
    markers.update(
        {
            COORDINATED_LINEAR_TRANSPORT: {
                "linear",
                "transport",
                "translate",
                "advance",
                "travel",
                "straight",
                "progress",
            },
            COORDINATED_ARC_TRANSPORT: {
                "arc",
                "curve",
                "turn",
                "rotate",
                "heading",
                "yaw",
                "swing",
            },
            COORDINATED_PIVOT: {
                "pivot",
                "turn",
                "rotate",
                "rotation",
                "heading",
                "yaw",
                "swing",
            },
            COORDINATED_ALIGNMENT: {
                "align",
                "alignment",
                "correct",
                "correction",
                "formation",
                "reference",
                "separation",
                "spacing",
                "stagger",
                "hold",
                "catch up",
                "lagger",
            },
        }.get(method, set())
    )
    method_tokens = {
        token
        for token in method.replace("-", "_").split("_")
        if token and token not in {"coordinated", "carry"}
    }
    markers.update(method_tokens)
    if objective == "backward":
        markers.update({"retreat", "reverse", "back away"})
    return frozenset(marker for marker in markers if marker)


def _strategy_replan_role_markers(
    *,
    method: str,
    role: str,
    fallback: frozenset[str],
) -> frozenset[str]:
    """Return role-local language without forcing plan-level wording."""

    role = str(role).strip().casefold()
    role_markers = {
        "reference": {
            "reference",
            "hold",
            "remain",
            "stationary",
            "fixed",
            "maintain",
            "zero",
            "anchor",
            "stable",
        },
        "corrector": {
            "correct",
            "align",
            "adjust",
            "reduce",
            "offset",
            "move",
            "catch up",
        },
        "mover": {
            "move",
            "transport",
            "translate",
            "advance",
            "travel",
            "retreat",
        },
        "inner": {"inner", "pivot", "hold", "remain", "stationary"},
        "outer": {"outer", "arc", "move", "travel", "swing"},
    }.get(role, set())
    method = str(method).strip()
    if method == "place_crouch" and role == "mover":
        # Placement is already authorized by the structured primitive,
        # lifecycle gate, and immutable role mapping.  Accept its natural
        # lowering vocabulary instead of requiring transport-only mover words.
        role_markers.update({"place", "crouch", "lower", "support", "payload"})
    if method == COORDINATED_ALIGNMENT and role == "corrector":
        role_markers.update({"separation", "spacing", "stagger", "formation"})
    return frozenset(role_markers) or fallback


def _manager_fallback_from_controller(
    decision: EvolutionDecision,
) -> dict[str, Any]:
    level = {
        "strategy": "strategy_mutation",
        "coordination": "coordination_mutation",
        "script": "strategy_mutation",
        "execution": "parameter_tuning",
    }.get(decision.repair_level, "strategy_mutation")
    return {
        "mutation_level": level,
        "failure_hypothesis": decision.reason,
        "evidence": [decision.reason or "host controller repair evidence"],
        "preserve": ["current Orchestrator global plan and accepted checkpoints"],
        "mutate": [decision.required_change or level],
        "forbidden": ["repeat the exact verifier-rejected executable AST"],
        "success_test": "the next bounded rollout passes the physics verifier",
        "inheritance": "retain accepted task progress",
        "confidence": 1.0,
        "source": "host_controller_fallback",
    }


def _formal_host_controller_guard(
    decision: EvolutionDecision,
) -> dict[str, Any]:
    """Expose lifecycle, budget, and soft failure evidence without choosing a level."""

    semantic_failure_count = int(getattr(decision, "semantic_failure_count", 0))
    return {
        "should_evolve": decision.should_evolve,
        "failure_type": decision.failure_type,
        "physical_iterations": decision.physical_iterations,
        "max_physical_iterations": decision.max_physical_iterations,
        "budget_remaining": decision.budget_remaining,
        "budget_exhausted": decision.budget_exhausted,
        "semantic_failure_count": semantic_failure_count,
        "route_stall_iterations": int(
            getattr(decision, "route_stall_iterations", 0)
        ),
        "minimum_mutation_level": "model_selected",
        "parameter_tuning_allowed": True,
        "repeated_semantic_failure_guidance": (
            "Repeated semantic failure is evidence to reconsider a structural "
            "mutation, but it does not make an otherwise available mutation level "
            "illegal."
            if semantic_failure_count >= 2
            else "No repeated-failure escalation guidance is active."
        ),
        "semantic_repair_fields_withheld": [
            "action",
            "repair_level",
            "reason",
            "required_change",
        ],
    }


def _validate_plan_guided_mutation(
    payload: dict[str, Any],
    *,
    manager_decision: dict[str, Any],
    parent_strategy: dict[str, Any],
    allowed_sdk_methods: set[str] | frozenset[str],
    runtime_state: dict[str, Any] | None = None,
    formation_gate: dict[str, Any] | None = None,
    exhausted_relations: tuple[dict[str, Any], ...] = (),
    host_wall_clearance: dict[str, Any] | None = None,
    nonrelaxable_motion_objectives: tuple[str, ...] = (),
    trusted_body_frame_target_projection: dict[str, Any] | None = None,
    final_approach_transport_envelope: dict[str, Any] | None = None,
    formation_sync_repair_gate: dict[str, Any] | None = None,
    host_wall_heading_correction_contract: dict[str, Any] | None = None,
    route_clearance_conflict_contract: dict[str, Any] | None = None,
    turn_clearance_replan_contract: dict[str, Any] | None = None,
    rejected_host_wall_heading_correction: dict[str, Any] | None = None,
    forbidden_transaction_signatures: set[str] | None = None,
    local_navigation_contract: dict[str, Any] | None = None,
    route_recovery_contract: dict[str, Any] | None = None,
    model_owns_route: bool = False,
    model_owned_setup: bool = False,
    mutation_contract: dict[str, Any] | None = None,
) -> None:
    """Enforce mutation level and static SDK membership after model reasoning."""

    advisory_only = bool(model_owns_route)
    expected_level = str(manager_decision.get("mutation_level", "")).strip()
    actual_level = str(payload.get("mutation_level", "")).strip()
    if actual_level != expected_level:
        raise ModelRoleError(
            f"mutation_level must remain {expected_level!r}; got {actual_level!r}"
        )
    child = payload.get("child_strategy", {})
    if not isinstance(child, dict):
        raise ModelRoleError("child_strategy must be an object")
    method = str(child.get("primitive", "")).strip()
    mode = str(child.get("coordination_mode", "")).strip()
    objective = str(child.get("semantic_objective", "")).strip()
    if method not in set(allowed_sdk_methods):
        raise ModelRoleError(f"child strategy uses unknown SDK method {method!r}")
    turn_clearance_replan = _strategy_matches_turn_clearance_replan_contract(
        method=method,
        objective=objective,
        mode=mode,
        contract=turn_clearance_replan_contract,
    )
    if not advisory_only:
        _validate_final_approach_strategy_availability(
            method=method,
            envelope=final_approach_transport_envelope,
            allow_turn_clearance_replan=turn_clearance_replan,
        )
    if mode not in coordination_modes_for_method(method):
        raise ModelRoleError(
            f"child strategy mode {mode!r} is invalid for api.{method}"
        )
    objective_parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(method, "")
    if objective_parameter:
        choices = set(
            coordination_primitive_method_spec(method)["parameters"][
                objective_parameter
            ].get("choices", ())
        )
        if objective not in choices:
            raise ModelRoleError(
                f"child semantic_objective must be one of {sorted(choices)}"
            )
    elif method in _STATIC_MOTION_OBJECTIVES:
        expected_objective = _STATIC_MOTION_OBJECTIVES[method]
        if objective != expected_objective:
            raise ModelRoleError(
                f"semantic_objective for api.{method} must be exactly "
                f"{expected_objective!r}; got {objective!r}"
            )
    effective_mutation_contract = (
        mutation_contract
        if isinstance(mutation_contract, dict) and mutation_contract
        else build_mutation_contract(parent_strategy)
    )
    try:
        validate_mutation_transition(
            payload,
            parent_strategy=parent_strategy,
            contract=effective_mutation_contract,
        )
    except ValueError as exc:
        raise ModelRoleError(str(exc)) from exc
    host_wall_heading_correction = _strategy_matches_host_wall_heading_correction(
        method=method,
        objective=objective,
        assignment=child.get("team_assignment", {}),
        contract=host_wall_heading_correction_contract,
    )
    exact_motion_objective = f"{method}:{objective}"
    if (
        not advisory_only
        and
        exact_motion_objective in set(nonrelaxable_motion_objectives)
        and not host_wall_heading_correction
    ):
        raise ModelRoleError(
            "the current trusted checkpoint has physics-authoritative route "
            f"regression evidence against {exact_motion_objective!r}; mutate to "
            "a different semantic objective or maneuver primitive"
        )
    if (
        expected_level == "strategy_mutation"
        and not turn_clearance_replan
        and not model_owns_route
    ):
        _validate_strategy_mutation_linear_feasibility(
            parent_strategy=parent_strategy,
            method=method,
            objective=objective,
            host_wall_clearance=host_wall_clearance,
            trusted_body_frame_target_projection=(
                trusted_body_frame_target_projection
            ),
        )
    if not advisory_only and not host_wall_heading_correction:
        _validate_strategy_against_host_wall_clearance(
            method=method,
            objective=objective,
            host_wall_clearance=host_wall_clearance,
        )
        _validate_loaded_linear_strategy_has_untried_tier(
            method=method,
            objective=objective,
            mode=mode,
            host_wall_clearance=host_wall_clearance,
            forbidden_transaction_signatures=(
                forbidden_transaction_signatures or set()
            ),
        )
    if not advisory_only and not turn_clearance_replan:
        _validate_final_approach_target_direction(
            {},
            method=method,
            objective=objective,
            projection=trusted_body_frame_target_projection,
        )
    if not advisory_only:
        _validate_task_plan_against_formation_sync_repair_gate(
            _task_plan_with_execution_strategy(
                {
                    "current_phase_id": "mutation_validation",
                    "global_plan": [],
                    "current_subtask": {},
                },
                child,
            ),
            gate=formation_sync_repair_gate,
            runtime_state=runtime_state,
            local_navigation_contract=local_navigation_contract,
            host_wall_clearance=host_wall_clearance,
            rejected_host_wall_heading_correction=(
                rejected_host_wall_heading_correction
            ),
        )
    parent_method = str(parent_strategy.get("primitive", "")).strip()
    parent_objective = str(parent_strategy.get("semantic_objective", "")).strip()
    child_assignment = child.get("team_assignment", {})
    if not advisory_only:
        _validate_strategy_not_checkpoint_exhausted(
            method=method,
            objective=objective,
            mode=mode,
            assignment=child_assignment,
            exhausted_relations=exhausted_relations,
        )
    turn_episode = (
        runtime_state.get("turn_episode", {}) if isinstance(runtime_state, dict) else {}
    )
    turn_episode = turn_episode if isinstance(turn_episode, dict) else {}
    if (
        not model_owns_route
        and expected_level == "strategy_mutation"
        and bool(turn_episode.get("active", False))
        and str(turn_episode.get("purpose", "")).strip() == "route_turn"
        and bool(turn_episode.get("switch_allowed", False))
        and parent_method == COORDINATED_PIVOT
        and parent_objective in _SIGNED_TURN_DIRECTIONS
    ):
        required_objective = (
            "clockwise"
            if parent_objective == "counterclockwise"
            else "counterclockwise"
        )
        if method != COORDINATED_PIVOT or objective != required_objective:
            expected_roles = pivot_role_assignment(
                runtime_state or {},
                required_objective,
            )
            role_instruction = (
                f" with immutable-ID roles {expected_roles}" if expected_roles else ""
            )
            raise ModelRoleError(
                "the verifier rejected the parent signed pivot and the live turn "
                "episode permits a direction switch. Because this repair level is "
                "strategy_mutation and unfinished turns cannot become linear motion, "
                "child_strategy must use primitive='coordinated_pivot' and "
                f"semantic_objective={required_objective!r}{role_instruction}"
            )
    if method == COORDINATED_PIVOT:
        roles_by_robot = {
            robot_id: str(
                child_assignment.get(robot_id, {}).get("role", "")
                if isinstance(child_assignment, dict)
                and isinstance(child_assignment.get(robot_id), dict)
                else ""
            ).strip()
            for robot_id in ("front", "back")
        }
        roles = {role for role in roles_by_robot.values() if role}
        if roles != {"inner", "outer"}:
            raise ModelRoleError(
                "coordinated_pivot child strategy requires exactly one inner and one "
                "outer role"
            )
        expected_roles = pivot_role_assignment(runtime_state or {}, objective)
        if not model_owns_route and expected_roles and roles_by_robot != expected_roles:
            inner_robot = next(
                robot_id for robot_id, role in expected_roles.items() if role == "inner"
            )
            outer_robot = next(
                robot_id for robot_id, role in expected_roles.items() if role == "outer"
            )
            raise ModelRoleError(
                "coordinated_pivot child role geometry requires "
                f"{inner_robot}=inner and {outer_robot}=outer for "
                f"turn_direction={objective}; front/back are immutable robot IDs"
            )
    if method == COORDINATED_ALIGNMENT:
        roles = {
            str(item.get("role", "")).strip()
            for item in (
                child_assignment.values() if isinstance(child_assignment, dict) else ()
            )
            if isinstance(item, dict)
        }
        if roles != {"reference", "corrector"}:
            raise ModelRoleError(
                "coordinated_alignment child strategy requires exactly one "
                "reference and one corrector role"
            )
    local = (
        local_navigation_contract
        if isinstance(local_navigation_contract, dict)
        else {}
    )
    if (
        model_owns_route
        and str(local.get("reference", "")).strip() == "formation_only"
        and method in (set(TURN_METHODS) | set(_TRANSPORT_METHODS))
    ):
        raise ModelRoleError(
            "navigation.reference='formation_only' cannot be advanced by "
            "linear, arc, or pivot route motion"
        )
    checkpoint_id = _local_contract_checkpoint_id(
        runtime_state=runtime_state,
        local_navigation_contract=local,
    )
    scoped_route_recovery = bool(
        _contract_matches_checkpoint(
            route_recovery_contract,
            checkpoint_id=checkpoint_id,
        )
    )
    scoped_route_clearance = (
        route_clearance_conflict_contract
        if _contract_matches_checkpoint(
            route_clearance_conflict_contract,
            checkpoint_id=checkpoint_id,
        )
        else {}
    )
    scoped_turn_clearance = (
        turn_clearance_replan_contract
        if _contract_matches_checkpoint(
            turn_clearance_replan_contract,
            checkpoint_id=checkpoint_id,
        )
        else {}
    )
    authoritative_navigation_lock = bool(
        model_owns_route
        and expected_level == "strategy_mutation"
        and local.get("mutation_lock_required", False)
    )
    navigation_deferred_to_replan = bool(
        model_owns_route
        and expected_level == "strategy_mutation"
        and not authoritative_navigation_lock
    )
    if not (model_owned_setup or navigation_deferred_to_replan):
        try:
            validate_strategy_against_local_navigation(
                method=method,
                objective=objective,
                contract=local,
                allow_formation_repair=bool(
                    formation_gate or formation_sync_repair_gate
                ),
                allow_route_recovery=scoped_route_recovery,
                allow_turn_clearance_replan=(
                    turn_clearance_replan and bool(scoped_turn_clearance)
                ),
                route_clearance_conflict_contract=scoped_route_clearance,
            )
        except ValueError as exc:
            raise ModelRoleError(str(exc)) from exc
    if not advisory_only:
        _validate_strategy_against_host_formation_gate(
            method=method,
            objective=objective,
            mode=mode,
            assignment=child_assignment,
            formation_gate=(
                _trusted_checkpoint_longitudinal_alignment(runtime_state or {})
                if formation_gate is None
                else formation_gate
            ),
        )
    _validate_model_strategy_turn_episode(
        method=method,
        objective=objective,
        assignment=child_assignment,
        runtime_state=runtime_state,
        local_navigation_contract=local_navigation_contract,
        host_wall_clearance=host_wall_clearance,
        host_wall_heading_correction_contract=(
            host_wall_heading_correction_contract
        ),
        route_clearance_conflict_contract=route_clearance_conflict_contract,
        active_only=model_owns_route,
        allow_model_owned_setup=bool(
            model_owned_setup or navigation_deferred_to_replan
        ),
    )


def _host_wall_turn_episode_replan_kind(
    *,
    method: str,
    objective: str,
    runtime_state: dict[str, Any] | None,
    host_wall_clearance: dict[str, Any] | None,
) -> str:
    """Classify a model-selected escape from a wall-blocked signed turn."""

    contract = _host_wall_turn_clearance_replan_contract(
        runtime_state=runtime_state,
        host_wall_clearance=host_wall_clearance,
    )
    if not contract.get("active", False):
        return ""
    if (
        method == COORDINATED_LINEAR_TRANSPORT
        and objective in set(contract.get("allowed_translation_directions", ()))
    ):
        return "translation"
    return ""


def _host_wall_turn_clearance_replan_contract(
    *,
    runtime_state: dict[str, Any] | None,
    host_wall_clearance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Authorize one setup translation when the committed turn cannot fit.

    This contract is deliberately checkpoint-local. It preserves the active
    signed turn and merely gives the model room to execute the next calibrated
    pivot; it does not advance the route phase or replace the target reference.
    """

    state = runtime_state if isinstance(runtime_state, dict) else {}
    episode = state.get("turn_episode", {})
    episode = episode if isinstance(episode, dict) else {}
    clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    direction = str(episode.get("direction", "")).strip()
    purpose = str(episode.get("purpose", "")).strip()
    status = str(episode.get("status", "")).strip()
    maximum = maximum_safe_turn_angle_deg(clearance, direction)
    minimum_calibrated = min(MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG)
    turn_unexecutable = bool(
        direction in _SIGNED_TURN_DIRECTIONS
        and (
            candidate_is_blocked(
                clearance,
                family="turn",
                objective=direction,
            )
            or (
                maximum is not None
                and maximum + 1.0e-9 < minimum_calibrated
            )
        )
    )
    open_directions = list(
        _model_executable_loaded_translation_directions(clearance)
    )
    if not (
        clearance.get("available", False)
        and bool(episode.get("active", False))
        and purpose in {"route_turn", "local_navigation_turn"}
        and status not in {"probe_ready", "passage_aligned", "completed"}
        and turn_unexecutable
        and open_directions
    ):
        return {}
    checkpoint = state.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    return {
        "active": True,
        "source": "unfinished_signed_turn_plus_authored_wall_sweep",
        "scope_checkpoint_id": str(checkpoint.get("checkpoint_id", "")).strip(),
        "preserve_turn_direction": direction,
        "preserve_turn_purpose": purpose,
        "preserve_navigation_reference": str(
            episode.get("reference", "")
        ).strip(),
        "blocked_turn_max_safe_angle_deg": maximum,
        "minimum_calibrated_pivot_angle_deg": minimum_calibrated,
        "allowed_translation_directions": open_directions,
        "required_primitive": COORDINATED_LINEAR_TRANSPORT,
        "required_coordination_mode": "sync",
        "setup_only": True,
        "resume_signed_turn_after_setup": True,
        "phase_completion_authorized": False,
    }


def _validate_model_strategy_turn_episode(
    *,
    method: str,
    objective: str,
    assignment: object,
    runtime_state: dict[str, Any] | None,
    local_navigation_contract: dict[str, Any] | None = None,
    host_wall_clearance: dict[str, Any] | None = None,
    host_wall_heading_correction_contract: dict[str, Any] | None = None,
    route_clearance_conflict_contract: dict[str, Any] | None = None,
    active_only: bool = False,
    allow_model_owned_setup: bool = False,
) -> None:
    """Move the live turn-continuity contract ahead of script generation."""

    state = runtime_state if isinstance(runtime_state, dict) else {}
    episode = state.get("turn_episode", {})
    episode = episode if isinstance(episode, dict) else {}
    purpose = str(episode.get("purpose", "")).strip()
    status = str(episode.get("status", "")).strip()
    active = bool(episode.get("active", False))
    if purpose not in {"route_turn", "local_navigation_turn"}:
        return
    if active_only and not (
        active or status == PASSAGE_ALIGNMENT_REQUIRED_STATUS
    ):
        return
    if _strategy_matches_host_wall_heading_correction(
        method=method,
        objective=objective,
        assignment=assignment,
        contract=host_wall_heading_correction_contract,
    ):
        # This is not a second route turn. It is one exact local correction
        # authorized by current authored-wall geometry for a blocked probe.
        return
    route_clearance_conflict = (
        route_clearance_conflict_contract
        if isinstance(route_clearance_conflict_contract, dict)
        else {}
    )
    if (
        route_clearance_conflict.get("active", False)
        and method == COORDINATED_PIVOT
        and objective
        in set(
            route_clearance_conflict.get(
                "safe_heading_correction_turn_directions",
                (),
            )
        )
    ):
        # A completed room-scale turn does not forbid a later bounded heading
        # correction when the next accepted straight segment reaches a wall.
        # The finite wall sweep still caps the executable angle.
        return
    if _host_wall_turn_episode_replan_kind(
        method=method,
        objective=objective,
        runtime_state=state,
        host_wall_clearance=host_wall_clearance,
    ):
        return
    turn_required = bool(
        (active and status != "probe_ready")
        or status == PASSAGE_ALIGNMENT_REQUIRED_STATUS
    )
    if (
        turn_required
        and method in _TRANSPORT_METHODS
        and not allow_model_owned_setup
    ):
        direction_rule = (
            "either clockwise or counterclockwise is selectable from current "
            "geometry because switch_allowed=true"
            if bool(episode.get("switch_allowed", False))
            else (
                f"preserve turn_direction={str(episode.get('direction', '')).strip()!r}"
            )
        )
        raise ModelRoleError(
            "the current live route-turn episode is unfinished: choose "
            f"a model-exposed shared-heading turn and {direction_rule}; loaded linear transport "
            "is unavailable until global passage alignment is verified"
        )

    if method not in TURN_METHODS:
        return
    target_reorientation_direction = _declared_local_reorientation_turn_direction(
        state,
        contract=local_navigation_contract,
    )
    turn_episode_complete = bool(
        not active
        and status
        in {"passage_aligned", "completed", "completed_with_translation"}
    )
    if (
        turn_episode_complete
        and method == COORDINATED_PIVOT
        and objective == target_reorientation_direction
    ):
        # The completed route turn and this target-local heading adjustment are
        # different maneuvers. Near the final target, a strong body-lateral
        # projection may be rotated into the more reliable forward gait.
        return
    assignment_map = assignment if isinstance(assignment, dict) else {}
    robot_parameters = {
        robot_id: {
            "turn_direction": objective,
            "role": str(
                assignment_map.get(robot_id, {}).get("role", "")
                if isinstance(assignment_map.get(robot_id), dict)
                else ""
            ).strip(),
        }
        for robot_id in ("front", "back")
    }
    selected = {
        "method": method,
        "robot_parameters": robot_parameters,
    }
    try:
        validate_selected_parameters_for_turn_episode(selected, episode)
    except TurnEpisodeError as exc:
        raise ModelRoleError(
            f"the proposed strategy violates the current live turn episode: {exc}"
        ) from exc


def _validate_task_plan_against_formation_sync_repair_gate(
    task_plan: dict[str, Any],
    *,
    gate: dict[str, Any] | None,
    runtime_state: dict[str, Any] | None,
    local_navigation_contract: dict[str, Any] | None = None,
    host_wall_clearance: dict[str, Any] | None = None,
    rejected_host_wall_heading_correction: dict[str, Any] | None = None,
) -> None:
    """Keep Orchestrator output consistent with the active physical probe state."""

    contract = gate if isinstance(gate, dict) else {}
    if not contract.get("active", False):
        return
    strategy = _execution_strategy_from_task_plan(task_plan)
    method = str(strategy.get("primitive", "")).strip()
    objective = str(strategy.get("semantic_objective", "")).strip()
    if contract.get("probe_unavailable", False):
        required_repair = str(contract.get("required_repair", "")).strip()
        if required_repair == COORDINATED_ALIGNMENT:
            if method == COORDINATED_ALIGNMENT:
                return
            raise ModelRoleError(
                "the final-approach transport envelope exhausted the active linear "
                "synchronization probe after the route turn was passage-aligned; "
                "the repair requires coordinated_alignment at this checkpoint"
            )
        if method in {COORDINATED_ALIGNMENT, COORDINATED_PIVOT}:
            return
        raise ModelRoleError(
            "the final-approach transport envelope exhausted the active linear "
            "synchronization probe; choose one geometry-changing "
            "coordinated_alignment or coordinated_pivot repair"
        )
    failure_count = int(contract.get("failure_count", 0))
    if failure_count < 2:
        probe = contract.get("probe_contract", {})
        probe = probe if isinstance(probe, dict) else {}
        required_objective = str(probe.get("semantic_objective", "")).strip()
        clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
        translations = clearance.get("translation_candidates", {})
        translations = translations if isinstance(translations, dict) else {}
        probe_clearance = translations.get(required_objective, {})
        probe_clearance = probe_clearance if isinstance(probe_clearance, dict) else {}
        if str(probe_clearance.get("relation", "")).strip() == "blocked":
            rejected_correction = (
                rejected_host_wall_heading_correction
                if isinstance(rejected_host_wall_heading_correction, dict)
                else {}
            )
            if rejected_correction.get("active", False):
                if method == COORDINATED_ALIGNMENT:
                    return
                raise ModelRoleError(
                    "the authored-wall heading correction was physically rejected at "
                    "this checkpoint; choose one bounded coordinated_alignment support/"
                    "formation repair before recomputing the blocked translation"
                )
            corrections = clearance.get(
                "small_heading_corrections_for_blocked_translation",
                {},
            )
            corrections = corrections if isinstance(corrections, dict) else {}
            correction = corrections.get(required_objective, {})
            correction = correction if isinstance(correction, dict) else {}
            required_turn = str(correction.get("turn_direction", "")).strip()
            if (
                required_turn
                and method == COORDINATED_PIVOT
                and objective == required_turn
            ):
                return
            raise ModelRoleError(
                "the active formation synchronization probe is temporarily "
                f"deferred because authored-wall geometry blocks {required_objective!r}; "
                + (
                    "first schedule a bounded coordinated_pivot:"
                    f"{required_turn} heading correction, then re-evaluate the probe"
                    if required_turn
                    else "select a wall-clear geometry-changing maneuver before probing"
                )
            )
        if method != COORDINATED_LINEAR_TRANSPORT or (
            required_objective and objective != required_objective
        ):
            raise ModelRoleError(
                "the active formation synchronization gate requires its "
                "direction-matched coordinated_linear_transport probe before "
                "another strategy is scheduled"
            )
        return

    state = runtime_state if isinstance(runtime_state, dict) else {}
    target_turn = _declared_local_reorientation_turn_direction(
        state,
        contract=(
            local_navigation_contract
            if isinstance(local_navigation_contract, dict)
            else build_local_navigation_contract(task_plan, state)
        ),
    )
    if target_turn:
        if method != COORDINATED_PIVOT or objective != target_turn:
            raise ModelRoleError(
                "the exhausted lateral synchronization probe requires one "
                f"target-local coordinated_pivot:{target_turn} reorientation; "
                "do not repeat linear transport or substitute formation alignment"
            )
        return
    turn_episode = state.get("turn_episode", {})
    turn_episode = turn_episode if isinstance(turn_episode, dict) else {}
    route_turn_complete = (
        not bool(turn_episode.get("active", False))
        and str(turn_episode.get("status", "")).strip() == "passage_aligned"
    )
    if route_turn_complete:
        if method != COORDINATED_ALIGNMENT:
            raise ModelRoleError(
                "the formation synchronization probe failed repeatedly and "
                "the payload already completed this route turn; the repair "
                "requires coordinated_alignment at this checkpoint, not linear "
                "transport or another pivot"
            )
        return
    if method not in {COORDINATED_ALIGNMENT, COORDINATED_PIVOT}:
        raise ModelRoleError(
            "the formation synchronization probe failed repeatedly; choose one "
            "geometry-changing coordinated_alignment or coordinated_pivot repair"
        )


def _formation_sync_gate_for_planning(
    gate: dict[str, Any] | None,
    *,
    lifecycle_hard_required: bool,
) -> dict[str, Any]:
    """Retire locomotion repair once a physical lifecycle transition is ready."""

    if lifecycle_hard_required:
        return {}
    return dict(gate) if isinstance(gate, dict) else {}


def _reconcile_formation_sync_gate_with_final_approach(
    gate: dict[str, Any] | None,
    *,
    final_approach_transport_envelope: dict[str, Any] | None,
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve a linear-probe contract that the final envelope has exhausted."""

    contract = dict(gate) if isinstance(gate, dict) else {}
    envelope = (
        final_approach_transport_envelope
        if isinstance(final_approach_transport_envelope, dict)
        else {}
    )
    if not (
        contract.get("active", False)
        and envelope.get("active", False)
        and envelope.get("exhausted", False)
    ):
        return contract

    state = runtime_state if isinstance(runtime_state, dict) else {}
    turn_episode = state.get("turn_episode", {})
    turn_episode = turn_episode if isinstance(turn_episode, dict) else {}
    route_turn_complete = bool(
        not turn_episode.get("active", False)
        and str(turn_episode.get("status", "")).strip() == "passage_aligned"
    )
    probe_contract = contract.get("probe_contract", {})
    probe_contract = dict(probe_contract) if isinstance(probe_contract, dict) else {}
    probe_contract.update(
        {
            "available": False,
            "unavailable_reason": "final_approach_transport_envelope_exhausted",
        }
    )
    contract.update(
        {
            "probe_unavailable": True,
            "required_repair": (
                COORDINATED_ALIGNMENT if route_turn_complete else "geometry_changing"
            ),
            "repair_precedence": (
                "The exhausted final-approach envelope overrides the lower-failure-"
                "count linear-probe requirement. Do not retry linear transport until "
                "a verifier-accepted geometry repair creates a new checkpoint."
            ),
            "probe_contract": probe_contract,
        }
    )
    return contract


def _validate_final_lifecycle_phase_state(
    task_plan: dict[str, Any],
    *,
    runtime_state: dict[str, Any] | None,
) -> None:
    """Keep place-and-exit planning behind physical target readiness."""

    current_phase_id = str(task_plan.get("current_phase_id", "")).strip()
    current_phase: dict[str, Any] = {}
    for item in task_plan.get("global_plan", ()):
        if (
            isinstance(item, dict)
            and str(item.get("phase_id", "")).strip() == current_phase_id
        ):
            current_phase = item
            break
    phase_text = " ".join(
        (
            current_phase_id,
            str(current_phase.get("abstract_goal", "")),
        )
    ).lower()
    place_marker = any(token in phase_text for token in ("place", "release"))
    exit_marker = any(token in phase_text for token in ("exit", "retreat", "withdraw"))
    if not place_marker:
        return
    runtime = runtime_state if isinstance(runtime_state, dict) else {}
    payload_released = _payload_is_released_on_target(runtime)
    if payload_released:
        return
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    strategy = subtask.get("strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    method = str(strategy.get("primitive", "")).strip()
    current_phase_status = str(task_plan.get("current_phase_status", "")).strip()
    phase_status = str(current_phase.get("status", "")).strip()
    if (
        current_phase_status != "completed"
        and phase_status != "completed"
        and (
            not exit_marker
            or method
            not in {
                "exit_forward",
                "exit_backward",
                "exit_lateral_left",
                "exit_lateral_right",
                "recover_stand",
            }
        )
    ):
        # Placement setup belongs to the final lifecycle phase. Entering that
        # phase is not evidence that release or exit has completed.
        return
    payload = runtime.get("payload", {})
    target = runtime.get("target", {})
    payload = payload if isinstance(payload, dict) else {}
    target = target if isinstance(target, dict) else {}
    try:
        target_error_m = hypot(
            float(payload.get("x")) - float(target.get("center_x_m")),
            float(payload.get("y")) - float(target.get("center_y_m")),
        )
    except (TypeError, ValueError):
        target_error_m = float("inf")
    raise ModelRoleError(
        "final lifecycle completion or post-release exit cannot activate before "
        "the payload is physically released on the target; keep the placement "
        "phase active and choose a readiness-improving setup or guarded release. "
        f"Current payload-target xy error={target_error_m:.3f}m, "
        f"target_contact={bool(payload.get('target_contact', False))}, "
        f"released={bool(payload.get('released', False))}"
    )


def _parameter_free_mutation_strategy(
    strategy: dict[str, Any],
) -> dict[str, Any]:
    """Expose behavior semantics to Mutation without leaking Coder magnitudes."""

    source = strategy if isinstance(strategy, dict) else {}
    result = {
        key: json.loads(json.dumps(source.get(key), ensure_ascii=False, default=str))
        for key in (
            "primitive",
            "semantic_objective",
            "formation_strategy",
            "coordination_mode",
            "synchronization_intent",
            "team_assignment",
        )
        if key in source
    }
    method = str(result.get("primitive", "")).strip()
    objective = str(result.get("semantic_objective", "")).strip()
    if "(" not in objective:
        return result
    try:
        expression = ast.parse(objective, mode="eval").body
    except SyntaxError:
        expression = None
    if not isinstance(expression, ast.Call):
        return result
    objective_parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(method, "")
    if objective_parameter:
        for keyword in expression.keywords:
            if keyword.arg != objective_parameter:
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                break
            if isinstance(value, str) and value.strip():
                result["semantic_objective"] = value.strip()
                return result
    result["semantic_objective"] = _STATIC_MOTION_OBJECTIVES.get(method, method)
    return result


def _policy_effective_parameter_projection(
    policy: PolicyScript,
    *,
    parameter_fields: set[str],
) -> dict[str, Any]:
    """Project literal backend parameters by field and immutable robot ID."""

    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="front",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="team",
            coordination_mode="sync",
        ),
        RobotPolicyScript(
            robot_id="back",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="team",
            coordination_mode="sync",
        ),
    )
    values: dict[str, dict[str, Any]] = {
        field: {} for field in sorted(parameter_fields)
    }
    for script in scripts:
        arguments = _first_task_call_literal_arguments(script.code)
        for parameter_name in values:
            if parameter_name in arguments:
                values[parameter_name][script.robot_id] = arguments[parameter_name]
    return {
        field: robot_values
        for field, robot_values in values.items()
        if robot_values
    }


def _finalize_host_effective_mutation_delta(
    mutation: dict[str, Any],
    *,
    parent_policy: PolicyScript,
    child_policy: PolicyScript,
) -> dict[str, Any]:
    """Complete Mutation's audit with the Coder's actual executable literals."""

    parent_strategy = _policy_mutation_parent_strategy(parent_policy)
    contract = build_mutation_contract(parent_strategy)
    level = str(mutation.get("mutation_level", "")).strip()
    allowed_parameter_fields = {
        str(item)
        for item in contract.get("levels", {})
        .get(level, {})
        .get("allowed_parameter_fields", ())
        if str(item)
    }
    parent_parameters = _policy_effective_parameter_projection(
        parent_policy,
        parameter_fields=allowed_parameter_fields,
    )
    child_parameters = _policy_effective_parameter_projection(
        child_policy,
        parameter_fields=allowed_parameter_fields,
    )
    delta = host_effective_mutation_delta(
        mutation,
        parent_strategy=parent_strategy,
        parent_effective_parameters=parent_parameters,
        child_effective_parameters=child_parameters,
    )
    mutation["host_effective_delta"] = delta
    if level == "parameter_tuning":
        changed = set(delta["changed_parameter_fields"])
        if len(changed) != 1 or not changed <= allowed_parameter_fields:
            feedback = no_effective_mutation_feedback(
                mutation_level=level,
                delta=delta,
            )
            feedback["allowed_parameter_fields"] = sorted(allowed_parameter_fields)
            raise ModelAgentError(
                "NO_EFFECTIVE_MUTATION "
                + json.dumps(feedback, ensure_ascii=True, sort_keys=True)
            )
    elif not bool(delta.get("effective", False)):
        feedback = no_effective_mutation_feedback(
            mutation_level=level,
            delta=delta,
        )
        raise ModelAgentError(
            "NO_EFFECTIVE_MUTATION "
            + json.dumps(feedback, ensure_ascii=True, sort_keys=True)
        )
    return delta


def _preflight_candidate_infeasible(report: EvaluationReport) -> bool:
    """Return whether the latest candidate was blocked before physical dispatch."""

    if str(getattr(report, "failure_type", "") or "").strip() == "turn_not_ready":
        return True
    evidence = getattr(report, "evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    preflight = evidence.get("turn_episode_preflight", {})
    preflight = preflight if isinstance(preflight, dict) else {}
    if preflight.get("physically_dispatched") is False:
        error = str(preflight.get("error", ""))
        if "PREFLIGHT_TURN_CLEARANCE_INSUFFICIENT" in error:
            return True
    return False


def _policy_mutation_parent_strategy(policy: PolicyScript) -> dict[str, Any]:
    """Reconstruct Mutation's parent from the transaction that actually failed."""

    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="front",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="team",
            coordination_mode="sync",
        ),
        RobotPolicyScript(
            robot_id="back",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="team",
            coordination_mode="sync",
        ),
    )
    methods = {_first_policy_api_method(script.code) for script in scripts}
    methods.discard("")
    if len(methods) != 1:
        raise ModelAgentError(
            "cannot reconstruct one mutation parent from a team transaction with "
            f"methods={sorted(methods)}"
        )
    method = next(iter(methods))
    modes = {
        str(script.coordination_mode).strip()
        for script in scripts
        if str(script.coordination_mode).strip()
    }
    mode = next(iter(modes)) if len(modes) == 1 else "async"
    first_code = scripts[0].code
    objective_parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(method, "")
    if objective_parameter:
        objective = str(
            _first_task_call_literal_arguments(first_code).get(objective_parameter, "")
        ).strip()
    else:
        objective = {
            **_STATIC_MOTION_OBJECTIVES,
            "place_payload_on_arms": "establish_payload_support",
            "place_crouch": "lower_payload",
            "recover_stand": "recover_upright",
            "recover_payload_inward": "recover_payload_inward",
            "release_payload": "release_payload",
        }.get(method, method)
    assignments_by_robot = {
        assignment.robot_id: {
            "role": assignment.role,
            "objective": assignment.objective,
        }
        for assignment in policy.assignments
        if assignment.robot_id in {"front", "back"}
    }
    for robot_id in ("front", "back"):
        assignments_by_robot.setdefault(
            robot_id,
            {
                "role": "payload_support",
                "objective": f"execute {method} with {robot_id} robot",
            },
        )
    return {
        "primitive": method,
        "semantic_objective": objective or method,
        "formation_strategy": f"{mode} {method}",
        "coordination_mode": mode,
        "synchronization_intent": (
            "shared atomic payload-support phase"
            if mode == "sync"
            else "continuous temporary front/back role coordination"
        ),
        "team_assignment": assignments_by_robot,
    }


def _plan_guided_action_exemplars(
    store: VerifiedActionMemoryStore | None,
    *,
    execution_strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    if store is None:
        return []
    method = str(execution_strategy.get("primitive", "")).strip()
    mode = str(execution_strategy.get("coordination_mode", "")).strip()
    parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(method, "")
    value = execution_strategy.get("semantic_objective")
    objective = {parameter: value} if parameter and value not in {None, ""} else {}
    return store.retrieve(
        method=method,
        semantic_objective=objective,
        coordination_mode=mode,
        limit=2,
    )


def _validate_policy_matches_execution_strategy(
    policy: PolicyScript,
    strategy: dict[str, Any],
) -> None:
    method = str(strategy.get("primitive", "")).strip()
    mode = str(strategy.get("coordination_mode", "")).strip()
    objective_parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(method, "")
    objective = str(strategy.get("semantic_objective", "")).strip()
    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="front",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="team",
            coordination_mode=mode or "sync",
        ),
    )
    methods = {_first_policy_api_method(item.code) for item in scripts}
    if methods != {method}:
        raise ModelAgentError(
            f"generated policy must execute committed api.{method}; got {sorted(methods)}"
        )
    modes = {str(item.coordination_mode).strip() for item in scripts}
    if modes != {mode}:
        raise ModelAgentError(
            f"generated policy must use committed coordination_mode={mode!r}"
        )
    if objective_parameter:
        _validate_model_selected_subtask(
            policy,
            selected_method=method,
            coordination_mode=mode,
            motion_objective_parameter=objective_parameter,
            motion_objective=objective,
        )
    expected_assignments = strategy.get("team_assignment", {})
    if isinstance(expected_assignments, dict) and policy.assignments:
        actual_roles = {item.robot_id: item.role for item in policy.assignments}
        expected_roles = {
            robot_id: str(value.get("role", "")).strip()
            for robot_id, value in expected_assignments.items()
            if isinstance(value, dict)
        }
        if set(expected_roles) == {"front", "back"} and actual_roles != expected_roles:
            raise ModelAgentError(
                "generated assignment roles do not match the committed team subtask"
            )


def _loaded_linear_team_transaction_signature(
    *,
    direction: str,
    distance_m: float,
) -> str:
    """Build the canonical sync transaction identity for one loaded linear tier."""

    code = (
        "def policy(api):\n"
        "    api.coordinated_linear_transport("
        f"direction={direction!r}, distance_m={float(distance_m)!r}, "
        f"speed_mps={VERIFIED_LINEAR_SPEED_MPS!r}, "
        f"ramp_s={VERIFIED_LINEAR_RAMP_S!r})\n"
        "    api.stop_all()\n"
    )
    policy = PolicyScript(
        script_id="loaded_linear_tier_probe",
        attempt=0,
        code=code,
        robot_scripts=tuple(
            RobotPolicyScript(
                robot_id=robot_id,
                script_id=f"{robot_id}_loaded_linear_tier_probe",
                code=code,
                synchronization_group="loaded_linear_tier_probe",
                contact_intent="maintain_payload_support",
                coordination_mode="sync",
            )
            for robot_id in ("front", "back")
        ),
    )
    return _policy_transaction_signature(policy)


def _validate_loaded_linear_strategy_has_untried_tier(
    *,
    method: str,
    objective: str,
    mode: str,
    host_wall_clearance: dict[str, Any] | None,
    forbidden_transaction_signatures: set[str],
) -> None:
    """Reject a committed linear strategy that Coder cannot legally instantiate."""

    if method != COORDINATED_LINEAR_TRANSPORT or mode != "sync":
        return
    clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    candidates = clearance.get("translation_candidates", {})
    candidates = candidates if isinstance(candidates, dict) else {}
    direction_clearance = candidates.get(objective, {})
    direction_clearance = (
        direction_clearance if isinstance(direction_clearance, dict) else {}
    )
    try:
        clearance_m = float(direction_clearance["clearance_m"])
    except (KeyError, TypeError, ValueError):
        return

    within_clearance = [
        float(distance_m)
        for distance_m in COORDINATED_LINEAR_DISTANCE_CHOICES_M
        if float(distance_m) <= clearance_m + 1.0e-9
    ]
    rejected_tiers = [
        distance_m
        for distance_m in within_clearance
        if _loaded_linear_team_transaction_signature(
            direction=objective,
            distance_m=distance_m,
        )
        in forbidden_transaction_signatures
    ]
    untried_tiers = [
        distance_m for distance_m in within_clearance if distance_m not in rejected_tiers
    ]
    if untried_tiers:
        return

    over_clearance = [
        float(distance_m)
        for distance_m in COORDINATED_LINEAR_DISTANCE_CHOICES_M
        if float(distance_m) > clearance_m + 1.0e-9
    ]
    raise ModelRoleError(
        "committed coordinated_linear_transport has no untried model-supported "
        f"distance tier for semantic_objective={objective!r}: "
        f"rejected_within_clearance={rejected_tiers}, "
        f"over_authored_wall_clearance={over_clearance}, "
        f"clearance_m={clearance_m:g}. Strategy Mutation must change the maneuver "
        "primitive or semantic objective before Coder; parameter guessing cannot "
        "repair this contract"
    )


def _validate_policy_within_host_wall_clearance(
    policy: PolicyScript,
    *,
    clearance: object,
) -> None:
    """Reject generated linear or pivot magnitudes outside measured free space."""

    report = clearance if isinstance(clearance, dict) else {}
    translation_candidates = report.get("translation_candidates", {})
    translation_candidates = (
        translation_candidates if isinstance(translation_candidates, dict) else {}
    )
    turn_candidates = report.get("turn_candidates", {})
    turn_candidates = turn_candidates if isinstance(turn_candidates, dict) else {}
    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="front",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="team",
            coordination_mode="sync",
        ),
    )
    for script in scripts:
        method = _first_policy_api_method(script.code)
        arguments = _first_task_call_literal_arguments(script.code)
        if method == COORDINATED_LINEAR_TRANSPORT:
            direction = str(arguments.get("direction", "")).strip()
            evidence = translation_candidates.get(direction, {})
            evidence = evidence if isinstance(evidence, dict) else {}
            try:
                distance_m = float(arguments["distance_m"])
                clearance_m = float(evidence["clearance_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if distance_m <= clearance_m + 1.0e-9:
                continue
            raise ModelAgentError(
                "generated room-scale linear transaction exceeds deterministic "
                f"authored-wall clearance for {direction!r}: distance_m={distance_m:g}, "
                f"clearance_m={clearance_m:g}. The Coder cannot make an already "
                "committed route strategy physically fit by emitting an overlong script"
            )
        if method != COORDINATED_PIVOT:
            continue
        direction = str(arguments.get("turn_direction", "")).strip()
        evidence = turn_candidates.get(direction, {})
        evidence = evidence if isinstance(evidence, dict) else {}
        try:
            angle_deg = abs(float(arguments["angle_deg"]))
            max_safe_angle_deg = float(evidence["max_safe_angle_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if angle_deg <= max_safe_angle_deg + 1.0e-9:
            continue
        raise ModelAgentError(
            "generated coordinated_pivot exceeds the deterministic finite-sweep "
            f"authored-wall angle envelope for {direction!r}: "
            f"angle_deg={angle_deg:g}, max_safe_angle_deg={max_safe_angle_deg:g}. "
            "Preserve the committed signed direction and inner/outer roles, and "
            "reduce only angle_deg to a legal same-sign value"
        )


def _host_wall_turn_angle_envelope(
    method: str,
    objective: str,
    clearance: object,
    *,
    route_clearance_conflict_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose a Coder-owned angle cap without changing model-owned route intent."""

    if method != COORDINATED_PIVOT or objective not in _SIGNED_TURN_DIRECTIONS:
        return {}
    maximum = maximum_safe_turn_angle_deg(clearance, objective)
    if maximum is None or maximum < 2.0:
        return {}
    conflict = (
        route_clearance_conflict_contract
        if isinstance(route_clearance_conflict_contract, dict)
        else {}
    )
    conflict_envelopes = conflict.get("safe_turn_angle_envelopes", {})
    conflict_envelopes = (
        conflict_envelopes if isinstance(conflict_envelopes, dict) else {}
    )
    conflict_envelope = conflict_envelopes.get(objective, {})
    conflict_envelope = (
        conflict_envelope if isinstance(conflict_envelope, dict) else {}
    )
    if conflict.get("active", False) and conflict_envelope:
        try:
            maximum = min(maximum, float(conflict_envelope["maximum_angle_deg"]))
        except (KeyError, TypeError, ValueError):
            return {}
    supported = [
        float(angle)
        for angle in MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG
        if float(angle) <= maximum + 1.0e-9
    ]
    if not supported:
        # Geometry alone does not make a command executable. Loaded-pivot
        # calibration established the model-owned SONIC angle catalogue;
        # smaller verifier-owned corrections use a separate contract.
        return {}
    return {
        "source": "finite_coordinated_pivot_sweep_against_authored_walls",
        "turn_direction": objective,
        "minimum_angle_deg": min(supported),
        "maximum_angle_deg": round(maximum, 3),
        "preferred_model_angle_choices_deg": supported,
        "preserve_signed_direction": True,
        "rule": (
            "This is an executable magnitude envelope, not a route-direction choice. "
            "Keep the Orchestrator-selected turn direction and geometry-derived roles; "
            "choose one shared angle_deg no larger than maximum_angle_deg."
        ),
    }


def _validate_policy_matches_host_formation_gate(
    policy: PolicyScript,
    formation_gate: object,
    *,
    allow_corrector_magnitude_tuning: bool = False,
) -> None:
    """Preserve the measured lagger/reference relation and bounded correction."""

    gate = formation_gate if isinstance(formation_gate, dict) else {}
    if not gate:
        return
    expected_deltas = gate.get("robot_deltas_m", {})
    expected_deltas = expected_deltas if isinstance(expected_deltas, dict) else {}
    expected_axis = str(gate.get("semantic_objective", "longitudinal")).strip()
    scripts = {item.robot_id: item for item in policy.robot_scripts}
    if set(scripts) != {"front", "back"} or set(expected_deltas) != {
        "front",
        "back",
    }:
        raise ModelAgentError(
            "host_formation_gate requires explicit front/back robot scripts and deltas"
        )
    durations: set[float] = set()
    actual_deltas: dict[str, float] = {}
    for robot_id in ("front", "back"):
        arguments = _first_task_call_literal_arguments(scripts[robot_id].code)
        try:
            axis = str(arguments["axis"])
            delta = float(arguments["delta"])
            duration_s = float(arguments["duration_s"])
        except (KeyError, TypeError, ValueError):
            raise ModelAgentError(
                "host_formation_gate requires literal axis, delta, and duration_s "
                f"for robot {robot_id}"
            ) from None
        if axis != expected_axis:
            raise ModelAgentError(
                f"host_formation_gate requires axis={expected_axis!r} for both robots"
            )
        actual_deltas[robot_id] = delta
        durations.add(duration_s)
    if len(durations) != 1:
        raise ModelAgentError(
            "host_formation_gate requires one shared front/back duration_s"
        )
    normalized_expected = {
        robot_id: float(expected_deltas[robot_id]) for robot_id in ("front", "back")
    }
    if allow_corrector_magnitude_tuning:
        reference_robots = [
            robot_id
            for robot_id, delta in normalized_expected.items()
            if abs(delta) <= 1.0e-9
        ]
        corrector_robots = [
            robot_id
            for robot_id, delta in normalized_expected.items()
            if abs(delta) > 1.0e-9
        ]
        if len(reference_robots) != 1 or len(corrector_robots) != 1:
            raise ModelAgentError(
                "host_formation_gate magnitude tuning requires exactly one "
                "reference and one corrector delta"
            )
        reference_robot = reference_robots[0]
        corrector_robot = corrector_robots[0]
        expected_corrector_delta = normalized_expected[corrector_robot]
        actual_corrector_delta = actual_deltas[corrector_robot]
        minimum_magnitude = 2.0 if expected_axis == "yaw" else 0.05
        if abs(actual_deltas[reference_robot]) > 1.0e-9:
            raise ModelAgentError(
                "host_formation_gate magnitude tuning must keep "
                f"{reference_robot} at delta=0"
            )
        if (
            actual_corrector_delta * expected_corrector_delta <= 0.0
            or abs(actual_corrector_delta) < minimum_magnitude - 1.0e-9
            or abs(actual_corrector_delta) > abs(expected_corrector_delta) + 1.0e-9
        ):
            raise ModelAgentError(
                "host_formation_gate magnitude tuning must preserve the measured "
                f"{corrector_robot} correction sign and choose a magnitude in "
                f"[{minimum_magnitude:g}, {abs(expected_corrector_delta):g}]; "
                f"got {actual_corrector_delta:g}"
            )
        return
    if any(
        abs(actual_deltas[robot_id] - normalized_expected[robot_id]) > 1.0e-9
        for robot_id in ("front", "back")
    ):
        raise ModelAgentError(
            "host_formation_gate requires the measured forward-only deltas "
            f"{normalized_expected}; got {actual_deltas}. Keep the reference at "
            "delta=0 and move only the lagging corrector"
        )


def _generation_context_mapping(
    generation_context: dict[str, Any],
    *keys: str,
) -> dict[str, Any]:
    """Return the first non-empty mapping stored under any of the given keys."""

    context = generation_context if isinstance(generation_context, dict) else {}
    for key in keys:
        value = context.get(key, {})
        if isinstance(value, dict) and value:
            return value
    return {}


def _generation_context_checkpoint_id(
    generation_context: dict[str, Any],
) -> str:
    """Resolve one Host-observed checkpoint without repairing a contract scope."""

    context = generation_context if isinstance(generation_context, dict) else {}
    candidates: list[str] = []
    for observation_key in ("observation", "current_trusted_observation"):
        observation = context.get(observation_key, {})
        observation = observation if isinstance(observation, dict) else {}
        runtime_state = observation.get("runtime_state", {})
        runtime_state = runtime_state if isinstance(runtime_state, dict) else {}
        checkpoint = runtime_state.get("checkpoint", {})
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        candidates.append(str(checkpoint.get("checkpoint_id", "")).strip())
    local_navigation = context.get("local_navigation_contract", {})
    local_navigation = (
        local_navigation if isinstance(local_navigation, dict) else {}
    )
    candidates.extend(
        (
            str(local_navigation.get("scope_checkpoint_id", "")).strip(),
            str(context.get("current_checkpoint_id", "")).strip(),
            str(context.get("checkpoint_id", "")).strip(),
        )
    )
    observed = {candidate for candidate in candidates if candidate}
    return next(iter(observed)) if len(observed) == 1 else ""


def _contract_matches_checkpoint(
    contract: dict[str, Any] | None,
    *,
    checkpoint_id: str,
) -> bool:
    """Require an active Host contract to carry its own exact checkpoint scope."""

    candidate = contract if isinstance(contract, dict) else {}
    scope = str(candidate.get("scope_checkpoint_id", "")).strip()
    return bool(
        checkpoint_id
        and candidate.get("active", False)
        and scope
        and scope == checkpoint_id
    )


def _committed_route_recovery_contract(
    *,
    execution_strategy: dict[str, Any],
    generation_context: dict[str, Any],
    committed_subtask: dict[str, Any],
) -> dict[str, Any]:
    """Bind verifier recovery intent without Host-selecting a direction.

    Formal advisory keys remain family-ban shadows. The returned contract is
    the verification interpretation for an already committed linear setup.
    """

    strategy = execution_strategy if isinstance(execution_strategy, dict) else {}
    # ``committed_subtask.setup_only`` is model-authored audit metadata.  It
    # cannot create a verifier privilege.  Keep the argument in this boundary
    # so callers make that distinction explicit, but never read it as an
    # authorization source.
    _ = committed_subtask
    method = str(strategy.get("primitive", "")).strip()
    objective = str(strategy.get("semantic_objective", "")).strip()
    mode = str(strategy.get("coordination_mode", "")).strip()
    if (
        method != COORDINATED_LINEAR_TRANSPORT
        or objective not in LOADED_TRANSLATION_DIRECTIONS
    ):
        return {}
    route_recovery = _generation_context_mapping(
        generation_context,
        "route_recovery_contract",
        "advisory_route_recovery",
    )
    route_clearance = _generation_context_mapping(
        generation_context,
        "route_clearance_conflict_contract",
        "advisory_route_clearance",
    )
    turn_clearance = _generation_context_mapping(
        generation_context,
        "turn_clearance_replan_contract",
        "advisory_turn_clearance",
    )
    checkpoint_id = _generation_context_checkpoint_id(generation_context)
    if not checkpoint_id:
        return {}

    selected_contract: dict[str, Any] = {}
    if _contract_matches_checkpoint(
        turn_clearance,
        checkpoint_id=checkpoint_id,
    ) and bool(
        turn_clearance.get("setup_only", False)
        and not turn_clearance.get("phase_completion_authorized", True)
        and _strategy_matches_turn_clearance_replan_contract(
            method=method,
            objective=objective,
            mode=mode,
            contract=turn_clearance,
        )
    ):
        selected_contract = turn_clearance
    else:
        route_aligned = {
            str(item).strip()
            for item in route_clearance.get(
                "route_aligned_translation_directions",
                (),
            )
            if str(item).strip()
        }
        listed_setup = {
            str(item).strip()
            for key in (
                "clearance_setup_translation_directions",
                "route_reentry_translation_directions",
            )
            for item in route_clearance.get(key, ())
            if str(item).strip()
        }
        if _contract_matches_checkpoint(
            route_clearance,
            checkpoint_id=checkpoint_id,
        ) and bool(
            route_clearance.get("setup_only", False)
            and not route_clearance.get("phase_completion_authorized", True)
            and objective in listed_setup
            and objective not in route_aligned
        ):
            selected_contract = route_clearance
        elif _contract_matches_checkpoint(
            route_recovery,
            checkpoint_id=checkpoint_id,
        ):
            selected_contract = route_recovery

    if not selected_contract:
        return {}
    source = str(selected_contract.get("source", "")).strip() or (
        "host_checkpoint_scoped_route_recovery"
    )
    rejected_motion = selected_contract.get("rejected_motion", {})
    rejected_motion = rejected_motion if isinstance(rejected_motion, dict) else {}
    return {
        "active": True,
        "host_authoritative": True,
        "source": source,
        "scope_checkpoint_id": str(
            selected_contract.get("scope_checkpoint_id", "")
        ).strip(),
        "primitive": method,
        "semantic_objective": objective,
        "coordination_mode": mode,
        "setup_only": True,
        "phase_completion_authorized": False,
        "positive_route_memory_eligible": False,
        "rejected_motion": dict(rejected_motion),
    }


def _generate_plan_guided_team_policy(
    *,
    role_pipeline: ModelRolePipeline,
    task_plan: dict[str, Any],
    execution_strategy: dict[str, Any],
    generation_context: dict[str, Any],
    attempt: int,
    fallback_id: str,
    parent_script_id: str,
    require_team_plan: bool,
    expected_method: str,
    forbidden_transaction_signatures: tuple[str, ...],
    current_policy: PolicyScript | None,
    repair_level: str,
    repair_failure_type: str,
) -> tuple[dict[str, Any], PolicyScript]:
    """Generate scripts from one committed subtask, then host-validate them."""

    mode = str(execution_strategy.get("coordination_mode", "")).strip()
    formal_v50_lite = bool(generation_context.get("formal_v50_lite", False))
    policy_holder: dict[str, PolicyScript] = {}
    forbidden_signatures = set(forbidden_transaction_signatures)

    def validate_payload(payload: dict[str, Any]) -> None:
        normalizations = _canonicalize_robot_script_shorthand(payload)
        normalizations.extend(
            _canonicalize_team_coordination_mode(payload, selected_mode=mode)
        )
        normalizations.extend(
            _canonicalize_alignment_delta_roles(
                payload,
                execution_strategy=execution_strategy,
            )
        )
        if normalizations:
            payload["host_schema_normalizations"] = normalizations
        policy = _policy_from_model_payload(
            payload,
            attempt=attempt,
            fallback_id=fallback_id,
            parent_script_id=parent_script_id,
            require_team_plan=require_team_plan,
            expected_method=expected_method,
        )
        _validate_policy_matches_execution_strategy(policy, execution_strategy)
        host_wall_heading_correction = _strategy_matches_host_wall_heading_correction(
            method=str(execution_strategy.get("primitive", "")).strip(),
            objective=str(execution_strategy.get("semantic_objective", "")).strip(),
            assignment=execution_strategy.get("team_assignment", {}),
            contract=generation_context.get(
                "host_wall_heading_correction_contract",
                {},
            ),
        )
        if host_wall_heading_correction and not formal_v50_lite:
            _validate_host_wall_heading_correction_policy(
                policy,
                contract=generation_context.get(
                    "host_wall_heading_correction_contract",
                    {},
                ),
            )
        rejected_wall_alignment_repair = bool(
            isinstance(
                generation_context.get("rejected_host_wall_heading_correction"),
                dict,
            )
            and generation_context.get(
                "rejected_host_wall_heading_correction",
                {},
            ).get("active", False)
            and str(execution_strategy.get("primitive", "")).strip()
            == COORDINATED_ALIGNMENT
        )
        _validate_policy_matches_host_lifecycle_gate(
            policy,
            generation_context.get("host_lifecycle_gate", {}),
        )
        turn_clearance_context = generation_context.get(
            "turn_clearance_replan_contract",
            generation_context.get("advisory_turn_clearance", {}),
        )
        final_approach_context = generation_context.get(
            "final_approach_transport_envelope",
            generation_context.get("advisory_final_approach_transport", {}),
        )
        allow_turn_clearance_replan = _strategy_matches_turn_clearance_replan_contract(
            method=str(execution_strategy.get("primitive", "")).strip(),
            objective=str(execution_strategy.get("semantic_objective", "")).strip(),
            mode=str(execution_strategy.get("coordination_mode", "")).strip(),
            contract=turn_clearance_context,
        )
        allow_corrector_magnitude_tuning = bool(
            str(
                generation_context.get("evolution_manager", {}).get(
                    "mutation_level",
                    "",
                )
                if isinstance(generation_context.get("evolution_manager"), dict)
                else ""
            ).strip()
            == "parameter_tuning"
        )
        alignment_sign_repair_contract = generation_context.get(
            "alignment_sign_repair_contract",
            {},
        )
        formation_sync_repair_gate = generation_context.get(
            "formation_sync_repair_gate",
            {},
        )
        enforce_formation_sync_probe = bool(
            isinstance(formation_sync_repair_gate, dict)
            and formation_sync_repair_gate.get("active", False)
            and not alignment_sign_repair_contract
            and not host_wall_heading_correction
            and not rejected_wall_alignment_repair
        )
        sync_probe_distance_m = None
        if isinstance(
            formation_sync_repair_gate, dict
        ) and formation_sync_repair_gate.get("active", False):
            probe_contract = formation_sync_repair_gate.get("probe_contract", {})
            probe_contract = (
                probe_contract if isinstance(probe_contract, dict) else {}
            )
            sync_probe_distance_m = float(
                probe_contract.get(
                    "required_distance_m",
                    FORMATION_SYNC_REPAIR_PROBE_DISTANCE_M,
                )
            )

        def validate_legacy_failure_level() -> None:
            if current_policy is None or not repair_level:
                return
            _validate_failure_level_repair(
                current_policy,
                policy,
                failure_level=repair_level,
                failure_type=repair_failure_type,
                blocked_strategy_signatures=set(),
                formation_sync_probe_distance_m=sync_probe_distance_m,
                forbidden_transaction_signatures=forbidden_signatures,
            )

        if formal_v50_lite:
            shadow_checks = [
                _formal_shadow_check(
                    "final_approach_transport",
                    lambda: _validate_final_approach_transport_policy(
                        policy,
                        envelope=final_approach_context,
                        allow_turn_clearance_replan=allow_turn_clearance_replan,
                    ),
                ),
                _formal_shadow_check(
                    "legacy_authored_wall_margin",
                    lambda: _validate_policy_within_host_wall_clearance(
                        policy,
                        clearance=generation_context.get(
                            "host_authored_wall_clearance", {}
                        ),
                    ),
                ),
                _formal_shadow_check(
                    "host_formation_gate",
                    lambda: _validate_policy_matches_host_formation_gate(
                        policy,
                        generation_context.get("host_formation_gate", {}),
                        allow_corrector_magnitude_tuning=(
                            allow_corrector_magnitude_tuning
                        ),
                    ),
                ),
                _formal_shadow_check(
                    "alignment_sign_repair",
                    lambda: _validate_alignment_sign_repair_policy(
                        policy,
                        contract=alignment_sign_repair_contract,
                    ),
                ),
                _formal_shadow_check(
                    "failure_level_repair",
                    validate_legacy_failure_level,
                ),
            ]
            if enforce_formation_sync_probe:
                shadow_checks.append(
                    _formal_shadow_check(
                        "formation_sync_repair_probe",
                        lambda: _validate_formation_sync_repair_probe_policy(
                            policy,
                            gate=formation_sync_repair_gate,
                        ),
                    )
                )
            if host_wall_heading_correction:
                shadow_checks.append(
                    _formal_shadow_check(
                        "host_wall_heading_correction",
                        lambda: _validate_host_wall_heading_correction_policy(
                            policy,
                            contract=generation_context.get(
                                "host_wall_heading_correction_contract", {}
                            ),
                        ),
                    )
                )
            _write_formal_shadow_advisory(
                role_pipeline.output_dir,
                attempt=attempt,
                checkpoint_id=str(
                    generation_context.get("current_checkpoint_id", "")
                    or generation_context.get("checkpoint_id", "")
                ),
                stage="coder",
                strategy=execution_strategy,
                checks=shadow_checks,
            )
        else:
            _validate_final_approach_transport_policy(
                policy,
                envelope=final_approach_context,
                allow_turn_clearance_replan=allow_turn_clearance_replan,
            )
            _validate_policy_within_host_wall_clearance(
                policy,
                clearance=generation_context.get(
                    "host_authored_wall_clearance", {}
                ),
            )
            _validate_policy_matches_host_formation_gate(
                policy,
                generation_context.get("host_formation_gate", {}),
                allow_corrector_magnitude_tuning=allow_corrector_magnitude_tuning,
            )
            _validate_alignment_sign_repair_policy(
                policy,
                contract=alignment_sign_repair_contract,
            )
            if enforce_formation_sync_probe:
                _validate_formation_sync_repair_probe_policy(
                    policy,
                    gate=formation_sync_repair_gate,
                )
            validate_legacy_failure_level()
        signature = _policy_transaction_signature(policy)
        if signature and signature in forbidden_signatures:
            raise ModelAgentError(
                "generated policy repeats the exact verifier-rejected executable AST. "
                "Read host_contract.forbidden_executable_transactions and change a legal "
                "Coder-owned literal while preserving the committed strategy. The host "
                "supplies no replacement primitive, direction, coordination mode, or robot "
                "role. A strategy with no remaining legal literal must return to Strategy "
                "Mutation before Coder. blocked_fingerprint="
                f"{_blocked_transaction_fingerprint(signature)}"
            )
        lifecycle_forbidden = _generation_lifecycle_forbidden_methods(
            generation_context
        )
        if not expected_method:
            used_methods = {
                _first_policy_api_method(item.code) for item in policy.robot_scripts
            }
            invalid_lifecycle = sorted(used_methods & lifecycle_forbidden)
            if invalid_lifecycle:
                raise ModelAgentError(
                    "active payload lifecycle forbids reinitialization methods: "
                    + ", ".join(invalid_lifecycle)
                )
        approved_mutation = generation_context.get("approved_strategy_mutation", {})
        if current_policy is not None and isinstance(approved_mutation, dict) and approved_mutation:
            _finalize_host_effective_mutation_delta(
                approved_mutation,
                parent_policy=current_policy,
                child_policy=policy,
            )
        policy_holder["policy"] = policy

    try:
        parsed = role_pipeline.generate_team_policy(
            {
                **generation_context,
                "task_orchestrator_plan": task_plan,
                "current_subtask": task_plan.get("current_subtask", {}),
                "committed_execution_strategy": execution_strategy,
                "host_contract": {
                    "safe_sdk_only": True,
                    "one_task_changing_call_per_robot": True,
                    "stop_all_required": True,
                    "exact_rejected_ast_replay_blocked": True,
                    "forbidden_executable_fingerprints": list(
                        generation_context.get(
                            "recent_rejected_team_transactions",
                            (),
                        )
                    ),
                    "forbidden_executable_transactions": list(
                        generation_context.get(
                            "recent_rejected_executable_transactions",
                            (),
                        )
                    ),
                    "physics_and_rollback_owned_by_host": True,
                    "host_does_not_select_route_action": True,
                    "formation_sync_repair_probe_required": bool(
                        not formal_v50_lite
                        and isinstance(
                            generation_context.get("formation_sync_repair_gate"),
                            dict,
                        )
                        and generation_context.get(
                            "formation_sync_repair_gate",
                            {},
                        ).get("active", False)
                        and not generation_context.get(
                            "alignment_sign_repair_contract",
                            {},
                        )
                        and str(execution_strategy.get("primitive", "")).strip()
                        == COORDINATED_LINEAR_TRANSPORT
                    ),
                    "formation_sync_strategy_repair_allowed": bool(
                        not formal_v50_lite
                        and isinstance(
                            generation_context.get("formation_sync_repair_gate"),
                            dict,
                        )
                        and (
                            int(
                                generation_context.get(
                                    "formation_sync_repair_gate",
                                    {},
                                ).get("failure_count", 0)
                            )
                            >= 2
                            or generation_context.get(
                                "formation_sync_repair_gate",
                                {},
                            ).get("probe_unavailable", False)
                        )
                        and not generation_context.get(
                            "alignment_sign_repair_contract",
                            {},
                        )
                    ),
                    "alignment_sign_repair_required": bool(
                        not formal_v50_lite
                        and generation_context.get(
                            "alignment_sign_repair_contract",
                            {},
                        )
                    ),
                },
            },
            artifact_id=f"attempt_{attempt:02d}",
            coordination_mode=mode,
            validator=validate_payload,
        )
    except ModelRoleError as exc:
        raise ModelAgentError(f"plan-guided Coder pipeline failed: {exc}") from exc
    policy = policy_holder.get("policy")
    if policy is None:
        raise ModelAgentError(
            "plan-guided Coder pipeline returned no validated team policy"
        )
    host_wall_heading_correction = _strategy_matches_host_wall_heading_correction(
        method=str(execution_strategy.get("primitive", "")).strip(),
        objective=str(execution_strategy.get("semantic_objective", "")).strip(),
        assignment=execution_strategy.get("team_assignment", {}),
        contract=generation_context.get(
            "host_wall_heading_correction_contract",
            {},
        ),
    )
    if (
        not formal_v50_lite
        and host_wall_heading_correction
        and HOST_WALL_HEADING_CORRECTION_MARKER not in policy.rationale
    ):
        policy = replace(
            policy,
            rationale=(
                f"{HOST_WALL_HEADING_CORRECTION_MARKER} {policy.rationale}"
            ).strip(),
        )
        parsed["rationale"] = policy.rationale
    observation = generation_context.get(
        "observation",
        generation_context.get("current_trusted_observation", {}),
    )
    observation = observation if isinstance(observation, dict) else {}
    replan_kind = (
        ""
        if formal_v50_lite
        else _host_wall_turn_episode_replan_kind(
            method=str(execution_strategy.get("primitive", "")).strip(),
            objective=str(execution_strategy.get("semantic_objective", "")).strip(),
            runtime_state=observation.get("runtime_state", {}),
            host_wall_clearance=generation_context.get(
                "host_authored_wall_clearance",
                {},
            ),
        )
    )
    if replan_kind:
        marker = (
            HOST_WALL_TURN_ESCAPE_MARKER
            if replan_kind == "turn"
            else HOST_WALL_CLEARANCE_ESCAPE_MARKER
        )
        if marker not in policy.rationale:
            policy = replace(policy, rationale=f"{marker} {policy.rationale}".strip())
            parsed["rationale"] = policy.rationale
    turn_clearance_replan = generation_context.get(
        "turn_clearance_replan_contract",
        {},
    )
    committed_subtask = task_plan.get("current_subtask", {})
    committed_subtask = (
        committed_subtask if isinstance(committed_subtask, dict) else {}
    )
    policy = replace(
        policy,
        verification_contract={
            "committed_subtask": {
                "schema_version": 1,
                "source": "host_binding_from_validated_task_orchestrator_plan",
                "immutable_by_coder": True,
                "checkpoint_id": _generation_context_checkpoint_id(
                    generation_context
                ),
                "phase_id": str(task_plan.get("current_phase_id", "")).strip(),
                "subtask_id": str(committed_subtask.get("subtask_id", "")).strip(),
                "primitive": str(execution_strategy.get("primitive", "")).strip(),
                "semantic_objective": str(
                    execution_strategy.get("semantic_objective", "")
                ).strip(),
                "coordination_mode": str(
                    execution_strategy.get("coordination_mode", "")
                ).strip(),
                "navigation_reference": str(
                    (
                        committed_subtask.get("navigation", {})
                        if isinstance(committed_subtask.get("navigation"), dict)
                        else {}
                    ).get("reference", "")
                ).strip(),
                "setup_only": bool(committed_subtask.get("setup_only", False)),
                "phase_completion_authorized": not bool(
                    committed_subtask.get("setup_only", False)
                ),
                "positive_route_memory_eligible": not bool(
                    committed_subtask.get("setup_only", False)
                ),
            }
        },
    )
    lifecycle_gate = generation_context.get("host_lifecycle_gate", {})
    lifecycle_gate = lifecycle_gate if isinstance(lifecycle_gate, dict) else {}
    lifecycle_state = str(lifecycle_gate.get("lifecycle_state", "")).strip()
    lifecycle_method = str(execution_strategy.get("primitive", "")).strip()
    if lifecycle_state in {"place", "release"} and lifecycle_method != "release_payload":
        policy = replace(
            policy,
            verification_contract={
                **dict(policy.verification_contract),
                "lifecycle_setup": {
                    "active": True,
                    "lifecycle_state": lifecycle_state,
                    "objective": str(lifecycle_gate.get("objective", "")).strip(),
                    "setup_only": True,
                    "phase_completion_authorized": False,
                    "positive_route_memory_eligible": False,
                },
            },
        )
    formation_gate = generation_context.get("host_formation_gate", {})
    if (
        isinstance(formation_gate, dict)
        and formation_gate.get("source") == "verifier_pair_spacing_repair"
    ):
        policy = replace(
            policy,
            verification_contract={
                **dict(policy.verification_contract),
                "kind": "formation_spacing_reserve",
                **formation_gate,
            },
        )
    formation_sync_repair_gate = generation_context.get(
        "formation_sync_repair_gate",
        {},
    )
    if (
        isinstance(formation_sync_repair_gate, dict)
        and formation_sync_repair_gate.get("active", False)
        and _policy_uses_linear_transport(policy)
    ):
        policy = replace(
            policy,
            verification_contract={
                **dict(policy.verification_contract),
                **_formation_sync_repair_verification_contract(
                    formation_sync_repair_gate,
                    policy=policy,
                ),
            },
        )
    local_navigation = generation_context.get("local_navigation_contract", {})
    navigation = committed_subtask.get("navigation", {})
    navigation = navigation if isinstance(navigation, dict) else {}
    contract_runtime_state = observation.get("runtime_state", {})
    contract_runtime_state = (
        contract_runtime_state if isinstance(contract_runtime_state, dict) else {}
    )
    if (
        not formal_v50_lite
        and bool(generation_context.get("route_discovery", False))
        and str(navigation.get("reference", "")).strip()
        in {"final_target", "visible_connected_free_space"}
        and team_heading_deg(contract_runtime_state) is not None
        and not (
            isinstance(local_navigation, dict)
            and local_navigation.get("active", False)
        )
    ):
        raise ModelAgentError(
            "validated route subtask produced no host-resolvable local navigation "
            "contract; reject before physical dispatch instead of executing with "
            "verification_contract={}"
        )
    if isinstance(local_navigation, dict) and local_navigation.get("active", False):
        policy = replace(
            policy,
            verification_contract={
                **dict(policy.verification_contract),
                LOCAL_NAVIGATION_CONTRACT_KEY: local_navigation,
            },
        )
    route_clearance_conflict = _generation_context_mapping(
        generation_context,
        "route_clearance_conflict_contract",
        "advisory_route_clearance",
    )
    if (
        not formal_v50_lite
        and route_clearance_conflict.get("active", False)
    ):
        policy = replace(
            policy,
            verification_contract={
                **dict(policy.verification_contract),
                "route_clearance_conflict": route_clearance_conflict,
            },
        )
    route_recovery = _committed_route_recovery_contract(
        execution_strategy=execution_strategy,
        generation_context=generation_context,
        committed_subtask=committed_subtask,
    )
    if route_recovery.get("active", False):
        policy = replace(
            policy,
            verification_contract={
                **dict(policy.verification_contract),
                "route_recovery": route_recovery,
            },
        )
    # Merge the checkpoint-scoped setup authorization last. Some formation
    # contracts intentionally replace the base contract rather than extending
    # it, so attaching this earlier could silently remove the authorization
    # before pre-dispatch validation.
    if replan_kind == "translation":
        if not (
            isinstance(turn_clearance_replan, dict)
            and turn_clearance_replan.get("active", False)
            and _strategy_matches_turn_clearance_replan_contract(
                method=str(execution_strategy.get("primitive", "")).strip(),
                objective=str(
                    execution_strategy.get("semantic_objective", "")
                ).strip(),
                mode=str(execution_strategy.get("coordination_mode", "")).strip(),
                contract=turn_clearance_replan,
            )
        ):
            raise ModelAgentError(
                "turn-clearance setup policy lost its checkpoint-scoped host "
                "authorization before pre-dispatch; this is a system contract "
                "propagation error, not an Agent strategy failure"
            )
        policy = replace(
            policy,
            verification_contract={
                **dict(policy.verification_contract),
                "turn_clearance_replan": dict(turn_clearance_replan),
            },
        )
    _validate_host_bound_verification_contract(
        policy,
        task_plan=task_plan,
        execution_strategy=execution_strategy,
    )
    return parsed, policy


def _validate_host_bound_verification_contract(
    policy: PolicyScript,
    *,
    task_plan: dict[str, Any],
    execution_strategy: dict[str, Any],
) -> None:
    """Require a non-empty host-owned contract matching the committed subtask."""

    contract = (
        policy.verification_contract
        if isinstance(policy.verification_contract, dict)
        else {}
    )
    committed = contract.get("committed_subtask", {})
    committed = committed if isinstance(committed, dict) else {}
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    expected = {
        "phase_id": str(task_plan.get("current_phase_id", "")).strip(),
        "subtask_id": str(subtask.get("subtask_id", "")).strip(),
        "primitive": str(execution_strategy.get("primitive", "")).strip(),
        "semantic_objective": str(
            execution_strategy.get("semantic_objective", "")
        ).strip(),
        "coordination_mode": str(
            execution_strategy.get("coordination_mode", "")
        ).strip(),
    }
    mismatches = {
        key: {"expected": value, "actual": str(committed.get(key, "")).strip()}
        for key, value in expected.items()
        if not value or str(committed.get(key, "")).strip() != value
    }
    if (
        not contract
        or committed.get("source")
        != "host_binding_from_validated_task_orchestrator_plan"
        or not bool(committed.get("immutable_by_coder", False))
        or mismatches
    ):
        raise ModelAgentError(
            "SYSTEM_INVALID verification contract: policy must retain the non-empty "
            "host binding for the committed Task Orchestrator subtask; "
            f"mismatches={mismatches}"
        )


def _validate_policy_matches_host_lifecycle_gate(
    policy: PolicyScript,
    gate: object,
) -> None:
    """Keep host-owned final lifecycle transactions exact across both robots."""

    if not isinstance(gate, dict) or not bool(gate.get("hard_required", False)):
        return
    required_call = str(gate.get("required_call", "")).strip()
    if not required_call:
        return
    expected_code = _single_step_code_template(required_call)
    expected_method = _first_policy_api_method(expected_code)
    expected_arguments = _first_task_call_literal_arguments(expected_code)
    if not expected_method:
        raise ModelAgentError(
            f"host lifecycle gate contains an invalid required_call: {required_call!r}"
        )

    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="team",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="team",
        ),
    )
    for script in scripts:
        actual_method = _first_policy_api_method(script.code)
        actual_arguments = _first_task_call_literal_arguments(script.code)
        actual_call = _first_task_call_source(script.code)
        if actual_method != expected_method or actual_arguments != expected_arguments:
            raise ModelAgentError(
                f"robot {script.robot_id} must preserve the exact host lifecycle "
                f"transaction api.{required_call}; got "
                f"{actual_call or f'api.{actual_method}(...)'}"
            )


def _policy_from_model_payload(
    payload: dict[str, Any],
    *,
    attempt: int,
    fallback_id: str,
    parent_script_id: str = "",
    require_team_plan: bool = False,
    expected_method: str = "",
) -> PolicyScript:
    supplied_code = str(payload.get("code", "")).strip()
    assignments = _parse_assignments(
        payload.get("assignments"), require=require_team_plan
    )
    robot_scripts = _parse_robot_scripts(
        payload.get("robot_scripts"), require=require_team_plan
    )
    required_contact_intent = _required_method_contact_intent(expected_method)
    if required_contact_intent:
        assignments = tuple(
            replace(item, contact_intent=required_contact_intent)
            for item in assignments
        )
        robot_scripts = tuple(
            replace(item, contact_intent=required_contact_intent)
            for item in robot_scripts
        )
    if require_team_plan:
        # `code` is a redundant display field for team plans. Smaller models
        # sometimes concatenate the front/back scripts there even though each
        # robot script is a valid one-action transaction. The executable
        # canonical policy is therefore the already-validated front script.
        code = next(
            (item.code for item in robot_scripts if item.robot_id == "front"),
            "",
        )
    else:
        code = supplied_code
    if not code:
        raise ModelAgentError("model response did not include executable policy code")
    try:
        if require_team_plan:
            validate_step_policy_code(code)
        else:
            validate_policy_code(code)
    except Exception as exc:  # noqa: BLE001 - preserve model-output validation context
        raise ModelAgentError(f"model policy code failed validation: {exc}") from exc
    script_id = str(payload.get("script_id") or fallback_id).strip() or fallback_id
    plan = tuple(
        str(item).strip() for item in _as_list(payload.get("plan")) if str(item).strip()
    )
    rationale = str(payload.get("rationale", "")).strip()
    if assignments and robot_scripts:
        scripts_by_robot = {item.robot_id: item for item in robot_scripts}
        normalized_assignments: list[RobotAssignment] = []
        assignment_groups = {
            item.robot_id: item.synchronization_group for item in assignments
        }
        for script in robot_scripts:
            if assignment_groups.get(script.robot_id) != script.synchronization_group:
                raise ModelAgentError(
                    f"robot {script.robot_id} assignment/script synchronization_group mismatch"
                )
        for assignment in assignments:
            script = scripts_by_robot[assignment.robot_id]
            phase_id = assignment.phase_id or script.phase_id
            contact_intent = (
                script.contact_intent
                if assignment.contact_intent == "unspecified"
                else assignment.contact_intent
            )
            if phase_id != script.phase_id:
                raise ModelAgentError(
                    f"robot {assignment.robot_id} assignment/script phase_id mismatch"
                )
            if contact_intent != script.contact_intent:
                raise ModelAgentError(
                    f"robot {assignment.robot_id} assignment/script contact_intent mismatch"
                )
            if assignment.coordination_mode != script.coordination_mode:
                raise ModelAgentError(
                    f"robot {assignment.robot_id} assignment/script coordination_mode mismatch"
                )
            normalized_assignments.append(
                replace(
                    assignment,
                    phase_id=phase_id,
                    contact_intent=contact_intent,
                )
            )
        assignments = tuple(normalized_assignments)
    if require_team_plan:
        try:
            validate_team_policy_scripts(robot_scripts)
        except Exception as exc:  # noqa: BLE001 - retain model correction context
            raise ModelAgentError(
                f"model team transaction failed validation: {exc}"
            ) from exc
        assignments = _normalize_coordination_assignment_roles(
            assignments,
            robot_scripts,
        )
    if expected_method:
        for script in robot_scripts or (
            RobotPolicyScript(
                robot_id="team",
                script_id=script_id,
                code=code,
                synchronization_group="team",
            ),
        ):
            actual_method, _stop = validate_step_policy_code(script.code)
            if actual_method != expected_method:
                raise ModelAgentError(
                    f"robot {script.robot_id} must execute api.{expected_method}(...) at this physical checkpoint; "
                    f"got api.{actual_method}(...)"
                )
    return PolicyScript(
        script_id=script_id,
        attempt=attempt,
        code=code,
        plan=plan,
        assignments=assignments,
        robot_scripts=robot_scripts,
        parent_script_id=parent_script_id,
        rationale=rationale,
    )


def _normalize_coordination_assignment_roles(
    assignments: tuple[RobotAssignment, ...],
    robot_scripts: tuple[RobotPolicyScript, ...],
) -> tuple[RobotAssignment, ...]:
    """Reject assignment/code role conflicts without rewriting model decisions."""

    if {item.robot_id for item in assignments} != {"front", "back"}:
        return assignments
    if {item.robot_id for item in robot_scripts} != {"front", "back"}:
        return assignments
    methods = {
        script.robot_id: _first_policy_api_method(script.code)
        for script in robot_scripts
    }
    if len(set(methods.values())) != 1:
        return assignments
    method = next(iter(methods.values()))
    expected_roles: dict[str, str] = {}
    if method == COORDINATED_ARC_TRANSPORT:
        radii = {
            script.robot_id: float(
                _first_task_call_literal_arguments(script.code)["radius_m"]
            )
            for script in robot_scripts
        }
        if abs(radii["front"] - radii["back"]) < 1.0e-9:
            return assignments
        expected_roles = {
            robot_id: "outer" if radius == max(radii.values()) else "inner"
            for robot_id, radius in radii.items()
        }
    elif method == COORDINATED_PIVOT:
        expected_roles = {
            script.robot_id: str(
                _first_task_call_literal_arguments(script.code).get("role", "")
            )
            for script in robot_scripts
        }
        if set(expected_roles.values()) != {"inner", "outer"}:
            return assignments
    elif method == COORDINATED_ALIGNMENT:
        deltas = {
            script.robot_id: float(
                _first_task_call_literal_arguments(script.code).get("delta", 0.0)
            )
            for script in robot_scripts
        }
        actual_roles = {
            assignment.robot_id: assignment.role for assignment in assignments
        }
        role_conflicts = {
            robot_id: role
            for robot_id, role in actual_roles.items()
            if role
            not in (
                {"reference"}
                if abs(deltas[robot_id]) <= 1.0e-9
                else {
                    "corrector",
                    (
                        "positive_adjuster"
                        if deltas[robot_id] > 0.0
                        else "negative_adjuster"
                    ),
                }
            )
        }
        if role_conflicts:
            raise ModelAgentError(
                "assignment roles conflict with the executable front/back relation: "
                f"declared={actual_roles}, executable_deltas={deltas}. The Host will "
                "not rewrite robot roles; repair the model response"
            )
        return assignments
    else:
        return assignments

    actual_roles = {
        assignment.robot_id: assignment.role for assignment in assignments
    }
    if actual_roles != expected_roles:
        raise ModelAgentError(
            "assignment roles conflict with the executable front/back relation: "
            f"declared={actual_roles}, executable={expected_roles}. The Host will "
            "not rewrite robot roles; repair the model response"
        )
    return assignments


def _parse_assignments(value: Any, *, require: bool) -> tuple[RobotAssignment, ...]:
    if isinstance(value, dict):
        items = []
        for robot_id, item in value.items():
            if isinstance(item, dict):
                items.append({**item, "robot_id": robot_id})
            else:
                raise ModelAgentError(
                    f"assignment mapping entry {robot_id!r} must be a JSON object"
                )
    else:
        items = _as_list(value)
    if require and not items:
        raise ModelAgentError(
            "model response did not include explicit front/back assignments"
        )
    assignments: list[RobotAssignment] = []
    for item in items:
        if not isinstance(item, dict):
            raise ModelAgentError("each assignment must be a JSON object")
        assignments.append(
            RobotAssignment(
                robot_id=str(item.get("robot_id", "")).strip(),
                role=str(item.get("role", "")).strip(),
                objective=str(item.get("objective", "")).strip(),
                synchronization_group=str(
                    item.get("synchronization_group", "")
                ).strip(),
                safety_constraints=tuple(
                    str(entry).strip()
                    for entry in _as_list(item.get("safety_constraints"))
                    if str(entry).strip()
                ),
                phase_id=str(item.get("phase_id", "")).strip(),
                contact_intent=_validate_contact_intent(
                    str(item.get("contact_intent", "unspecified")).strip()
                    or "unspecified"
                ),
                coordination_mode=_validate_coordination_mode(
                    str(item.get("coordination_mode", "sync"))
                ),
            )
        )
    if assignments and {item.robot_id for item in assignments} != {"front", "back"}:
        raise ModelAgentError(
            "assignments must contain exactly robot_id front and back"
        )
    if any(
        not item.role or not item.objective or not item.synchronization_group
        for item in assignments
    ):
        raise ModelAgentError(
            "each assignment requires role, objective, and synchronization_group"
        )
    return tuple(assignments)


def _parse_robot_scripts(value: Any, *, require: bool) -> tuple[RobotPolicyScript, ...]:
    if isinstance(value, dict):
        items = []
        for robot_id, item in value.items():
            if isinstance(item, dict):
                items.append({**item, "robot_id": robot_id})
            else:
                items.append({"robot_id": robot_id, "code": item})
    else:
        items = _as_list(value)
    if require and not items:
        raise ModelAgentError(
            "model response did not include explicit front/back robot_scripts"
        )
    scripts: list[RobotPolicyScript] = []
    for item in items:
        if not isinstance(item, dict):
            raise ModelAgentError("each robot_scripts entry must be a JSON object")
        robot_id = str(item.get("robot_id", "")).strip()
        script_code = str(item.get("code", "")).strip()
        script_id = str(item.get("script_id") or f"{robot_id}_policy").strip()
        synchronization_group = str(item.get("synchronization_group", "")).strip()
        if not script_code or not synchronization_group:
            raise ModelAgentError(
                f"robot script {robot_id or '<missing>'} requires code and synchronization_group"
            )
        try:
            validate_step_policy_code(script_code)
        except Exception as exc:  # noqa: BLE001
            raise ModelAgentError(
                f"robot {robot_id} policy code failed validation: {exc}"
            ) from exc
        phase_id = str(item.get("phase_id", "")).strip() or synchronization_group
        contact_intent = str(item.get("contact_intent", "")).strip().lower()
        if not contact_intent or contact_intent in {"unspecified", "support"}:
            contact_intent = _inferred_contact_intent(script_code)
        contact_intent = _validate_contact_intent(contact_intent)
        coordination_mode = _validate_coordination_mode(
            str(item.get("coordination_mode", "sync"))
        )
        scripts.append(
            RobotPolicyScript(
                robot_id=robot_id,
                script_id=script_id,
                code=script_code,
                synchronization_group=synchronization_group,
                phase_id=phase_id,
                contact_intent=contact_intent,
                coordination_mode=coordination_mode,
            )
        )
    if scripts and {item.robot_id for item in scripts} != {"front", "back"}:
        raise ModelAgentError(
            "robot_scripts must contain exactly robot_id front and back"
        )
    if len({item.script_id for item in scripts}) != len(scripts):
        scripts = [
            RobotPolicyScript(
                robot_id=item.robot_id,
                script_id=f"{item.robot_id}_{item.script_id}",
                code=item.code,
                synchronization_group=item.synchronization_group,
                phase_id=item.phase_id,
                contact_intent=item.contact_intent,
                coordination_mode=item.coordination_mode,
            )
            for item in scripts
        ]
    return tuple(scripts)


def _validate_contact_intent(value: str) -> str:
    intent = str(value).strip().lower() or "unspecified"
    # ``support`` is natural model shorthand but lacks payload lifecycle
    # semantics. Assignment metadata leaves it unresolved so reconciliation
    # can inherit the intent inferred from the executable robot script.
    if intent == "support":
        intent = "unspecified"
    if intent not in CONTACT_INTENTS:
        raise ModelAgentError(
            f"unknown contact_intent {intent!r}; expected one of {sorted(CONTACT_INTENTS)}"
        )
    return intent


def _validate_coordination_mode(value: str) -> str:
    mode = str(value).strip().lower() or "sync"
    if mode not in {"sync", "async"}:
        raise ModelAgentError(
            f"unknown coordination_mode {mode!r}; expected sync or async"
        )
    return mode


def _inferred_contact_intent(code: str) -> str:
    method = _first_policy_api_method(code)
    if method == "place_payload_on_arms":
        return "establish_payload_support"
    if method == "recover_stand":
        return "reposition_without_payload"
    if method in {
        "exit_forward",
        "exit_backward",
        "exit_lateral_left",
        "exit_lateral_right",
    }:
        return "exit_task_area"
    if (
        method
        in {
            "carry_forward",
            "carry_backward",
            "carry_lateral_left",
            "carry_lateral_right",
            "turn_clockwise",
            "turn_counterclockwise",
            "place_crouch",
            "recover_payload_inward",
        }
        | COORDINATION_PRIMITIVE_METHODS
    ):
        return "maintain_payload_support"
    if method == "release_payload":
        return "release_payload"
    if method == "hold_position":
        return "hold"
    return "none"


def _required_method_contact_intent(method: str) -> str:
    """Normalize metadata for host-mandated payload lifecycle actions."""

    return {
        "place_payload_on_arms": "establish_payload_support",
        "recover_payload_inward": "maintain_payload_support",
        "release_payload": "release_payload",
        "exit_backward": "exit_task_area",
        "recover_stand": "reposition_without_payload",
    }.get(str(method).strip(), "")


def _generate_valid_policy(
    *,
    client: TextGenerationClient,
    user_prompt: str,
    attempt: int,
    fallback_id: str,
    require_team_plan: bool,
    response_retries: int,
    expected_method: str = "",
) -> PolicyScript:
    _payload, policy = _generate_valid_policy_with_payload(
        client=client,
        user_prompt=user_prompt,
        attempt=attempt,
        fallback_id=fallback_id,
        require_team_plan=require_team_plan,
        response_retries=response_retries,
        expected_method=expected_method,
    )
    return policy


def _generate_valid_policy_with_payload(
    *,
    client: TextGenerationClient,
    user_prompt: str,
    attempt: int,
    fallback_id: str,
    require_team_plan: bool,
    response_retries: int,
    parent_script_id: str = "",
    expected_method: str = "",
    forbidden_task_call_signatures: tuple[str, ...] = (),
    forbidden_task_transactions: tuple[str, ...] = (),
    forbidden_task_methods: tuple[str, ...] = (),
    forbidden_strategy_signatures: tuple[str, ...] = (),
    current_policy: PolicyScript | None = None,
    repair_level: str = "",
    repair_failure_type: str = "",
    selected_subtask_method: str = "",
    selected_subtask_coordination: str = "",
    selected_subtask_motion_parameter: str = "",
    selected_subtask_motion_objective: str = "",
    measured_sync_compensation: dict[str, Any] | None = None,
    selected_sync_parameters: dict[str, Any] | None = None,
    retry_transport_limits: tuple[float, float] | None = None,
    required_transport_ramp_s: float | None = None,
    candidate_rejection_path: Path | None = None,
    correction_code_template: str = "",
) -> tuple[dict[str, Any], PolicyScript]:
    blocked_signatures = set(forbidden_task_call_signatures)
    blocked_strategies = set(forbidden_strategy_signatures)
    blocked_transactions = list(dict.fromkeys(forbidden_task_transactions))
    coordination_repair_hint = (
        _coordination_repair_hint(current_policy)
        if repair_level == "coordination"
        else ""
    )
    base_prompt = (
        user_prompt
        + build_blocked_transactions_prompt(blocked_transactions)
        + build_blocked_strategies_prompt(tuple(sorted(blocked_strategies)))
    )
    if coordination_repair_hint:
        base_prompt += (
            "\n\nHOST REQUIRED COORDINATION RELATION (HARD):\n"
            f"- {coordination_repair_hint}\n"
            "- Apply this executable relationship in the first response. Metadata-only changes "
            "and numerical tuning that preserves the rejected role assignment are invalid."
        )
    prompt = base_prompt
    last_error: Exception | None = None
    rejected_transactions: list[str] = []
    for response_index in range(max(0, int(response_retries)) + 1):
        candidate_policy: PolicyScript | None = None
        try:
            raw = client.generate(system=MODEL_AGENT_SYSTEM_PROMPT, user=prompt)
        except Exception as exc:  # noqa: BLE001 - normalize model-runtime failures at the agent boundary
            raise ModelAgentError(
                f"model generation failed while producing executable team plan: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            payload = _parse_model_json(raw)
            normalizations = _canonicalize_robot_script_shorthand(payload)
            if selected_subtask_method:
                _validate_raw_model_selected_subtask(
                    payload,
                    selected_method=selected_subtask_method,
                    motion_objective_parameter=selected_subtask_motion_parameter,
                    motion_objective=selected_subtask_motion_objective,
                )
            normalizations.extend(
                _canonicalize_team_coordination_mode(
                    payload,
                    selected_mode=selected_subtask_coordination,
                )
            )
            if normalizations:
                payload["host_schema_normalizations"] = normalizations
            policy = _policy_from_model_payload(
                payload,
                attempt=attempt,
                fallback_id=fallback_id,
                parent_script_id=parent_script_id,
                require_team_plan=require_team_plan,
                expected_method=expected_method,
            )
            candidate_policy = policy
            if selected_subtask_method:
                _validate_model_selected_subtask(
                    policy,
                    selected_method=selected_subtask_method,
                    coordination_mode=selected_subtask_coordination,
                    motion_objective_parameter=selected_subtask_motion_parameter,
                    motion_objective=selected_subtask_motion_objective,
                )
            if measured_sync_compensation:
                _validate_measured_sync_compensation(
                    policy,
                    constraint=measured_sync_compensation,
                )
            if selected_sync_parameters:
                _validate_selected_sync_parameters(
                    policy,
                    method=selected_subtask_method,
                    selected=selected_sync_parameters,
                )
            if forbidden_task_methods:
                methods = {
                    _first_policy_api_method(script.code)
                    for script in policy.robot_scripts
                } or {_first_policy_api_method(policy.code)}
                repeated_methods = sorted(methods.intersection(forbidden_task_methods))
                if repeated_methods:
                    blocked_calls = ", ".join(
                        f"api.{method}" for method in repeated_methods
                    )
                    raise ModelAgentError(
                        "the previous body-frame movement increased active-goal distance; "
                        f"do not repeat {blocked_calls} with different numerical parameters"
                    )
            strategy_signature = policy_strategy_signature(policy)
            if strategy_signature and strategy_signature in blocked_strategies:
                raise ModelAgentError(
                    f"strategy {strategy_signature} is excluded from the current "
                    "model-generation search; choose another host-valid candidate"
                )
            if current_policy is not None and repair_level:
                _validate_failure_level_repair(
                    current_policy,
                    policy,
                    failure_level=repair_level,
                    failure_type=repair_failure_type,
                    blocked_strategy_signatures=blocked_strategies,
                    allow_measured_sync_compensation=bool(
                        measured_sync_compensation and selected_sync_parameters
                    ),
                )
            transaction_signature = _policy_transaction_signature(policy)
            if transaction_signature and transaction_signature in blocked_signatures:
                transaction = _policy_transaction_for_prompt(policy)
                rejected_transactions.append(transaction)
                raise ModelAgentError(
                    "repaired policy repeats a verifier-rejected executable transaction at AST level; "
                    "renaming script_id, phase_id, roles, or rationale is not an executable change"
                )
            if retry_transport_limits is not None:
                _validate_unsafe_retry_magnitude(
                    policy,
                    max_distance_m=retry_transport_limits[0],
                    max_speed_mps=retry_transport_limits[1],
                )
            if required_transport_ramp_s is not None:
                _validate_transport_ramp(
                    policy,
                    required_ramp_s=required_transport_ramp_s,
                )
            return payload, policy
        except ModelAgentError as exc:
            last_error = exc
            if candidate_policy is not None:
                candidate_signature = _policy_transaction_signature(candidate_policy)
                candidate_transaction = _policy_transaction_for_prompt(candidate_policy)
                if candidate_signature:
                    blocked_signatures.add(candidate_signature)
                if candidate_transaction not in rejected_transactions:
                    rejected_transactions.append(candidate_transaction)
            if candidate_rejection_path is not None:
                _append_candidate_rejection(
                    candidate_rejection_path,
                    response_index=response_index,
                    error=str(exc),
                    raw_response=raw,
                )
            if response_index >= max(0, int(response_retries)):
                break
            prompt = build_model_replan_prompt(
                base_prompt,
                response_index=response_index + 1,
                error=str(exc),
                rejected_transactions=rejected_transactions,
                coordination_repair_hint=_coordination_repair_hint_for_rejection(
                    str(exc),
                    candidate_policy=(
                        candidate_policy if repair_level == "coordination" else None
                    ),
                    fallback=(
                        coordination_repair_hint
                        if repair_level == "coordination"
                        else ""
                    ),
                ),
                correction_code_template=correction_code_template,
            )
    raise ModelAgentError(
        f"model failed executable team-plan validation after {max(0, int(response_retries)) + 1} response(s): "
        f"{last_error}"
    )


def _coordination_repair_hint(policy: PolicyScript | None) -> str:
    """Render executable alternatives that differ from the failed relationship."""

    if policy is None or len(policy.robot_scripts) != 2:
        return ""
    methods = {
        script.robot_id: _first_policy_api_method(script.code)
        for script in policy.robot_scripts
    }
    if set(methods) != {"front", "back"}:
        return ""
    holders = [
        robot_id for robot_id, method in methods.items() if method == "hold_position"
    ]
    movers = [
        robot_id
        for robot_id, method in methods.items()
        if method in MODEL_PLANNING_ACTION_METHODS
    ]
    if len(holders) == 1 and len(movers) == 1:
        holder_id = holders[0]
        mover_id = movers[0]
        mover_method = methods[mover_id]
        return (
            f"robot={mover_id} must call api.hold_position(...); "
            f"robot={holder_id} must call api.{mover_method}(...); "
            "all assignment/script coordination_mode fields must be async"
        )
    unique_methods = set(methods.values())
    if len(unique_methods) == 1:
        mover_method = next(iter(unique_methods))
        if mover_method in COORDINATION_PRIMITIVE_METHODS:
            if mover_method == COORDINATED_ARC_TRANSPORT:
                return (
                    "keep both robots on api.coordinated_arc_transport(...) in async mode; "
                    "either select a different shared body-frame travel_direction from camera/yaw "
                    "evidence while keeping the role assignment, or swap which robot receives the "
                    "larger versus smaller radius_m while preserving travel_direction; keep shared "
                    "turn_direction, angle_deg, and duration_s; changing only the two radius magnitudes "
                    "without swapping their robot assignment is not a coordination repair"
                )
            if mover_method == COORDINATED_PIVOT:
                return (
                    "keep both robots on api.coordinated_pivot(...) in async mode, preserve "
                    "travel_direction='forward', and swap the one-inner/one-outer role relation; "
                    "the inner role remains translation-stationary and neither role may back up"
                )
            if mover_method == COORDINATED_ALIGNMENT:
                scripts = {script.robot_id: script for script in policy.robot_scripts}
                arguments = {
                    robot_id: _first_task_call_literal_arguments(script.code)
                    for robot_id, script in scripts.items()
                }
                front_delta = arguments["front"].get("delta")
                back_delta = arguments["back"].get("delta")
                axis = arguments["front"].get("axis")
                duration_s = arguments["front"].get("duration_s")
                return (
                    "keep both robots on api.coordinated_alignment(...) with coordination_mode=async; "
                    f"preserve axis={axis!r}, duration_s={duration_s!r}, and both delta magnitudes; "
                    f"front must use delta={back_delta!r} and back must use delta={front_delta!r}, "
                    "which exactly swaps the rejected executable positive/negative/reference roles; "
                    "do not merely change delta magnitude or assignment prose"
                )
            return (
                "keep both robots on api.coordinated_linear_transport(...) in sync mode and use "
                "bounded measured front/back speed compensation"
            )
        if mover_method in MODEL_ACTION_SKILL_METHODS:
            return (
                "the failed relation had both robots call "
                f"api.{mover_method}(...) in sync mode; set all assignment/script "
                "coordination_mode fields to async and choose exactly one unseen relation: "
                f"front=api.{mover_method}(...) with back=api.hold_position(...), or "
                f"front=api.hold_position(...) with back=api.{mover_method}(...); "
                "do not let both robots call the mover method"
            )
    return ""


def _coordination_repair_hint_for_rejection(
    error: str,
    *,
    candidate_policy: PolicyScript | None,
    fallback: str = "",
) -> str:
    """Preserve primitive geometry constraints when parsing fails before PolicyScript creation."""

    if "coordinated_arc_transport" in error and "radius_m must be >= 0.05" in error:
        return (
            "keep both robots on api.coordinated_arc_transport(...) with coordination_mode=async; "
            "preserve shared turn_direction, angle_deg, and duration_s; relative to the original "
            "rejected transaction, swap which robot receives the larger versus smaller radius_m; "
            "assign both robots a positive radius_m of at least 0.05 m, keep the two values at "
            "least 0.05 m apart, and do not restore the original assignment after replacing zero"
        )
    if "coordination-level arc repair must either" in error:
        return (
            "keep both robots on api.coordinated_arc_transport(...) with coordination_mode=async; "
            "either change the shared body-frame travel_direction using camera/yaw evidence while "
            "keeping the current radius roles, or swap which robot receives the larger versus "
            "smaller radius_m while preserving travel_direction; both radii must be at least "
            "0.05 m and differ by at least 0.05 m; preserve shared turn_direction, angle_deg, "
            "and duration_s"
        )
    if (
        "coordinated_arc_transport requires distinct inner/outer radius_m values"
        in error
    ):
        return (
            "keep both robots on api.coordinated_arc_transport(...) with coordination_mode=async; "
            "preserve shared turn_direction, angle_deg, and duration_s; choose two positive "
            "radius_m values that differ by at least 0.05 m so exactly one profile is inner and "
            "one is outer; swap which robot receives the larger versus smaller radius relative "
            "to the original rejected transaction and do not copy one radius to both robots"
        )
    if (
        "coordination-level failure cannot be repaired by parameter tuning alone"
        in error
        and fallback
    ):
        return fallback
    if candidate_policy is not None:
        candidate_hint = _coordination_repair_hint(candidate_policy)
        if candidate_hint:
            return candidate_hint
    return fallback


def _validate_raw_model_selected_subtask(
    payload: dict[str, Any],
    *,
    selected_method: str,
    motion_objective_parameter: str = "",
    motion_objective: str = "",
) -> None:
    """Reject stale current-policy calls before selected mode normalization."""

    raw_scripts = payload.get("robot_scripts", {})
    if isinstance(raw_scripts, dict):
        scripts = raw_scripts
    elif isinstance(raw_scripts, list):
        scripts = {
            str(item.get("robot_id", "")): item
            for item in raw_scripts
            if isinstance(item, dict)
        }
    else:
        scripts = {}
    if set(scripts) != {"front", "back"}:
        return
    actual_methods: dict[str, str] = {}
    actual_objectives: dict[str, object] = {}
    for robot_id in ("front", "back"):
        item = scripts[robot_id]
        code = str(item.get("code", "")) if isinstance(item, dict) else ""
        actual_methods[robot_id] = _first_policy_api_method(code)
        if motion_objective_parameter:
            actual_objectives[robot_id] = _first_task_call_literal_arguments(code).get(
                motion_objective_parameter
            )
    if set(actual_methods.values()) != {selected_method}:
        raise ModelAgentError(
            "model-selected subtask requires both robot scripts to call "
            f"api.{selected_method}(...) before team-mode normalization; got "
            f"{actual_methods}. Do not reuse the rejected current_policy method."
        )
    encoded_objective = _STATIC_MOTION_OBJECTIVES.get(selected_method, "")
    if (
        motion_objective
        and encoded_objective
        and motion_objective != encoded_objective
    ):
        raise ModelAgentError(
            f"api.{selected_method} encodes semantic objective "
            f"{encoded_objective!r}; got {motion_objective!r}"
        )
    if motion_objective_parameter and (
        set(actual_objectives.values()) != {motion_objective}
    ):
        raise ModelAgentError(
            "model-selected subtask requires both robot scripts to use "
            f"{motion_objective_parameter}={motion_objective!r}; got "
            f"{actual_objectives}. Do not reuse the rejected current_policy objective."
        )


def _validate_model_selected_subtask(
    policy: PolicyScript,
    *,
    selected_method: str,
    coordination_mode: str,
    motion_objective_parameter: str = "",
    motion_objective: str = "",
) -> None:
    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="team",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="team",
            coordination_mode="sync",
        ),
    )
    actual_modes = {script.coordination_mode for script in scripts}
    if actual_modes != {coordination_mode}:
        raise ModelAgentError(
            "generated scripts must implement the model-selected coordination_mode="
            f"{coordination_mode}; got {sorted(actual_modes)}"
        )
    methods = {_first_policy_api_method(script.code) for script in scripts}
    if coordination_mode == "sync":
        valid = methods == {selected_method}
    elif selected_method in COORDINATION_PRIMITIVE_METHODS:
        valid = methods == {selected_method}
    else:
        valid = selected_method in methods and methods.issubset(
            {selected_method, "hold_position"}
        )
    if not valid:
        raise ModelAgentError(
            "generated scripts must implement the model-selected SDK method "
            f"api.{selected_method} with coordination_mode={coordination_mode}; "
            f"got {sorted(methods)}"
        )
    encoded_objective = _STATIC_MOTION_OBJECTIVES.get(selected_method, "")
    if encoded_objective:
        if motion_objective_parameter or (
            motion_objective and motion_objective != encoded_objective
        ):
            raise ModelAgentError(
                f"api.{selected_method} encodes semantic objective "
                f"{encoded_objective!r} and requires no objective parameter; got "
                f"parameter={motion_objective_parameter!r}, "
                f"objective={motion_objective!r}"
            )
        return
    if motion_objective_parameter or motion_objective:
        expected_parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(selected_method, "")
        if motion_objective_parameter != expected_parameter or not motion_objective:
            raise ModelAgentError(
                "model-selected coordination objective metadata is incomplete or inconsistent"
            )
        actual_objectives = {
            _first_task_call_literal_arguments(script.code).get(
                motion_objective_parameter
            )
            for script in scripts
        }
        if actual_objectives != {motion_objective}:
            raise ModelAgentError(
                f"generated api.{selected_method} scripts must implement model-selected "
                f"{motion_objective_parameter}={motion_objective!r}; got "
                f"{sorted(repr(item) for item in actual_objectives)}"
            )


def _canonicalize_team_coordination_mode(
    payload: dict[str, Any],
    *,
    selected_mode: str = "",
) -> list[str]:
    """Validate redundant mode metadata without changing controller choreography.

    One mover plus one hold is executable async, while two identical static
    calls are executable sync. Coordination primitives retain the mode selected
    by the model-owned strategy. Any explicit disagreement is returned to the
    model; changing this field would change runtime controller semantics.
    """

    assignment_entries = _model_team_entries(payload.get("assignments"))
    script_entries = _model_team_entries(payload.get("robot_scripts"))
    if set(assignment_entries) != {"front", "back"} or set(script_entries) != {
        "front",
        "back",
    }:
        return []

    declared_modes = {
        str(entry.get("coordination_mode", "sync")).strip().lower() or "sync"
        for entry in (*assignment_entries.values(), *script_entries.values())
    }
    methods = {
        robot_id: _first_policy_api_method(str(entry.get("code", "")))
        for robot_id, entry in script_entries.items()
    }
    call_signatures = {
        robot_id: _task_call_signature(str(entry.get("code", "")))
        for robot_id, entry in script_entries.items()
    }
    common_method = (
        next(iter(set(methods.values())))
        if len(set(methods.values())) == 1 and all(methods.values())
        else ""
    )
    is_coordination_primitive = common_method in COORDINATION_PRIMITIVE_METHODS
    executable_mode = ""
    executable_reason = ""
    if is_coordination_primitive:
        if selected_mode in {"sync", "async"}:
            executable_mode = selected_mode
            executable_reason = (
                "preserved the model-selected mode for a coordination primitive; "
                "cross-robot parameters determine its executable relationship"
            )
    elif sorted(methods.values()).count("hold_position") == 1 and all(methods.values()):
        executable_mode = "async"
        executable_reason = (
            "one mover plus one hold_position is an atomic async team transaction"
        )
    elif len(set(methods.values())) == 1 and all(methods.values()):
        if len(set(call_signatures.values())) == 1 and all(call_signatures.values()):
            executable_mode = "sync"
            executable_reason = (
                "identical executable front/back calls form one sync team transaction"
            )

    if executable_mode:
        canonical_mode = executable_mode
        reason = executable_reason
    elif selected_mode in {"sync", "async"}:
        canonical_mode = selected_mode
        reason = "aligned redundant metadata with the model-selected team mode"
    elif len(declared_modes) == 1:
        return []
    else:
        return []

    if declared_modes != {canonical_mode}:
        raise ModelAgentError(
            "declared coordination_mode conflicts with the executable team "
            f"transaction: declared={sorted(declared_modes)}, "
            f"required={canonical_mode!r} ({reason}). The Host will not rewrite "
            "sync/async controller choreography"
        )
    return []


def _canonicalize_alignment_delta_roles(
    payload: dict[str, Any],
    *,
    execution_strategy: dict[str, Any],
) -> list[str]:
    """Reject a delta attached to the committed reference robot."""

    if str(execution_strategy.get("primitive", "")).strip() != COORDINATED_ALIGNMENT:
        return []
    assignments = execution_strategy.get("team_assignment", {})
    assignments = assignments if isinstance(assignments, dict) else {}
    roles = {
        robot_id: str(
            assignments.get(robot_id, {}).get("role", "")
            if isinstance(assignments.get(robot_id), dict)
            else ""
        ).strip()
        for robot_id in ("front", "back")
    }
    correctors = [robot_id for robot_id, role in roles.items() if role == "corrector"]
    references = [robot_id for robot_id, role in roles.items() if role == "reference"]
    if len(correctors) != 1 or len(references) != 1:
        return []

    scripts = _model_team_entries(payload.get("robot_scripts"))
    if set(scripts) != {"front", "back"}:
        return []
    if {
        _first_policy_api_method(str(entry.get("code", "")))
        for entry in scripts.values()
    } != {COORDINATED_ALIGNMENT}:
        return []

    deltas: dict[str, float] = {}
    for robot_id, entry in scripts.items():
        arguments = _first_task_call_literal_arguments(str(entry.get("code", "")))
        try:
            delta = float(arguments["delta"])
        except (KeyError, TypeError, ValueError):
            return []
        if not isfinite(delta):
            return []
        deltas[robot_id] = delta

    moving_robots = [
        robot_id for robot_id, delta in deltas.items() if abs(delta) > 1.0e-9
    ]
    if len(moving_robots) != 1:
        return []
    source_robot = moving_robots[0]
    corrector_robot = correctors[0]
    reference_robot = references[0]
    if source_robot == corrector_robot:
        return []
    raise ModelAgentError(
        "coordinated_alignment executable delta is attached to the committed "
        f"reference robot {reference_robot!r}; the nonzero delta must be authored "
        f"on corrector {corrector_robot!r}. The Host will not move executable "
        "literals between robots"
    )


def _canonicalize_robot_script_shorthand(payload: dict[str, Any]) -> list[str]:
    """Normalize redundant script envelopes without changing executable calls."""

    before_signatures = _model_team_task_call_signatures(
        payload.get("robot_scripts")
    )
    assignments = _model_team_entries(payload.get("assignments"))
    scripts = payload.get("robot_scripts")
    if set(assignments) != {"front", "back"} or not isinstance(scripts, dict):
        return []
    normalized: list[str] = []
    for robot_id in ("front", "back"):
        raw_script = scripts.get(robot_id)
        if isinstance(raw_script, str):
            script_entry: dict[str, Any] = {"code": raw_script}
            scripts[robot_id] = script_entry
            normalized.append(
                f"robot_scripts.{robot_id}: expanded code-string shorthand"
            )
        elif isinstance(raw_script, dict):
            script_entry = raw_script
        else:
            continue
        assignment = assignments[robot_id]

        code = str(script_entry.get("code", "")).strip()
        code_lines = [line.strip() for line in code.splitlines() if line.strip()]
        if (
            code_lines
            and not any(line.startswith("def policy") for line in code_lines)
            and all(line.startswith("api.") for line in code_lines)
        ):
            code = "def policy(api):\n" + "\n".join(
                f"    {line}" for line in code_lines
            )
            script_entry["code"] = code
            normalized.append(
                f"robot_scripts.{robot_id}.code: wrapped bare safe API body "
                "in def policy(api)"
            )

        inherited: list[str] = []
        for field_name in (
            "synchronization_group",
            "phase_id",
            "contact_intent",
            "coordination_mode",
        ):
            if (
                not str(script_entry.get(field_name, "")).strip()
                and str(assignment.get(field_name, "")).strip()
            ):
                script_entry[field_name] = assignment[field_name]
                inherited.append(field_name)
        if inherited:
            normalized.append(
                f"robot_scripts.{robot_id}: inherited redundant assignment metadata "
                + ",".join(inherited)
            )

        inherited_assignment: list[str] = []
        for field_name in (
            "synchronization_group",
            "phase_id",
            "contact_intent",
            "coordination_mode",
        ):
            if (
                not str(assignment.get(field_name, "")).strip()
                and str(script_entry.get(field_name, "")).strip()
            ):
                assignment[field_name] = script_entry[field_name]
                inherited_assignment.append(field_name)
        if inherited_assignment:
            normalized.append(
                f"assignments.{robot_id}: inherited redundant script metadata "
                + ",".join(inherited_assignment)
            )

        inferred_intent = _inferred_contact_intent(str(script_entry.get("code", "")))
        for entry_name, entry in (
            (f"assignments.{robot_id}", assignment),
            (f"robot_scripts.{robot_id}", script_entry),
        ):
            declared_intent = str(entry.get("contact_intent", "")).strip().lower()
            if declared_intent in {
                "payload_support",
                "support",
            } and inferred_intent not in {"", "none"}:
                entry["contact_intent"] = inferred_intent
                normalized.append(
                    f"{entry_name}.contact_intent: {declared_intent} -> "
                    f"{inferred_intent} from executable API method"
                )
    after_signatures = _model_team_task_call_signatures(
        payload.get("robot_scripts")
    )
    if before_signatures != after_signatures:
        raise ModelAgentError(
            "schema shorthand normalization changed a robot's executable task-call "
            "AST; only envelope and lexical aliases are permitted"
        )
    return normalized


def _model_team_task_call_signatures(value: Any) -> dict[str, str]:
    """Project per-robot executable ASTs across dict and code-string envelopes."""

    if not isinstance(value, dict):
        return {}
    signatures: dict[str, str] = {}
    for robot_id, entry in value.items():
        code = entry if isinstance(entry, str) else (
            entry.get("code", "") if isinstance(entry, dict) else ""
        )
        signatures[str(robot_id)] = _task_call_signature(str(code))
    return signatures


def _model_team_entries(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {
            str(robot_id): entry
            for robot_id, entry in value.items()
            if isinstance(entry, dict)
        }
    entries: dict[str, dict[str, Any]] = {}
    for entry in _as_list(value):
        if isinstance(entry, dict):
            robot_id = str(entry.get("robot_id", "")).strip()
            if robot_id:
                entries[robot_id] = entry
    return entries


def _policy_transaction_for_prompt(policy: PolicyScript) -> str:
    """Render blocked code without exposing internal AST serialization."""

    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="front",
            script_id=f"front_{policy.script_id}",
            code=policy.code,
            synchronization_group="legacy_team",
        ),
        RobotPolicyScript(
            robot_id="back",
            script_id=f"back_{policy.script_id}",
            code=policy.code,
            synchronization_group="legacy_team",
        ),
    )
    return "\n".join(
        f"robot={script.robot_id} sync_group={script.synchronization_group} "
        f"coordination_mode={script.coordination_mode}\n{script.code.strip()}"
        for script in scripts
    )


def _observation_checkpoint_id(observation: ObservationBundle | None) -> str:
    if observation is None or not isinstance(observation.runtime_state, dict):
        return ""
    checkpoint = observation.runtime_state.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        return ""
    return str(checkpoint.get("checkpoint_id", "")).strip()


def _report_restored_ancestor_checkpoint(
    report: EvaluationReport,
    *,
    checkpoint_id: str,
) -> bool:
    """Return whether a failed child report is now viewed from its restored parent."""

    if not checkpoint_id or not isinstance(report.evidence, dict):
        return False
    rewind = report.evidence.get("automatic_branch_rewind", {})
    if not isinstance(rewind, dict):
        return False
    restored_parent = str(rewind.get("restored_parent_checkpoint_id", "")).strip()
    retired_child = str(rewind.get("retired_checkpoint_id", "")).strip()
    return bool(
        restored_parent == checkpoint_id
        and retired_child
        and retired_child != restored_parent
    )


def _load_generation_candidate_exhaustions(
    output_dir: Path,
    *,
    checkpoint_id: str,
    limit: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Recover checkpoint-local semantic candidates exhausted before dispatch."""

    candidates: list[tuple[int, str, str]] = []
    resume_source = _read_json_object(output_dir / "resume_source.json")
    paths = {
        *output_dir.glob("model_coordination_candidate_exhaustion_*.json"),
        *output_dir.glob("model_structured_candidate_exhaustion_*.json"),
    }
    for path in sorted(paths):
        match = re.search(r"exhaustion_(\d+)", path.stem)
        if match is None:
            continue
        attempt_index = int(match.group(1))
        payload = _read_json_object(path)
        recorded_checkpoint = str(payload.get("checkpoint_id", "")).strip()
        if not recorded_checkpoint:
            record = _read_json_object(
                output_dir / f"attempt_{attempt_index:02d}" / "evolution_record.json"
            )
            recorded_checkpoint = str(record.get("checkpoint_before", "")).strip()
        if not recorded_checkpoint:
            try:
                source_attempt = int(resume_source.get("next_attempt", 0)) - 1
            except (TypeError, ValueError):
                source_attempt = -1
            if source_attempt == attempt_index:
                recorded_checkpoint = str(
                    resume_source.get("checkpoint_id", "")
                ).strip()
        if checkpoint_id and recorded_checkpoint != checkpoint_id:
            continue
        selected = payload.get("selected_subtask", {})
        selected = selected if isinstance(selected, dict) else {}
        if str(selected.get("selection_source", "")).strip() in {
            "host_wall_heading_correction",
            HOST_WALL_CLEARANCE_ESCAPE_SELECTION_SOURCE,
            HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE,
        }:
            # This correction is synthesized from live authored-wall geometry.
            # Exhaustion here means a host-valid transaction was rejected before
            # dispatch by an obsolete generation rule, not that physics failed.
            continue
        method = str(selected.get("selected_method", "")).strip()
        mode = str(selected.get("coordination_mode", "")).strip()
        objective = str(selected.get("motion_objective", "")).strip()
        if method and objective:
            candidates.append((attempt_index, "objective", f"{method}:{objective}"))
        elif method and mode:
            candidates.append((attempt_index, "strategy", f"{mode}:{method}"))
    candidates.sort(key=lambda item: item[0])
    retained = candidates[-max(1, int(limit)) :]
    strategies = {value for _index, kind, value in retained if kind == "strategy"}
    objectives = {value for _index, kind, value in retained if kind == "objective"}
    return tuple(sorted(strategies)), tuple(sorted(objectives))


def _attempt_index_from_artifact_path(value: object) -> int | None:
    """Read an attempt index from an attempt artifact directory path."""

    match = re.fullmatch(r"attempt_(\d+)", Path(str(value)).name)
    return int(match.group(1)) if match else None


def _authoritative_checkpoint_source_attempt(
    manifest_path: Path | None,
    *,
    checkpoint_id: str,
) -> int:
    """Return the accepted attempt that created the active checkpoint lineage."""

    checkpoint_id = str(checkpoint_id).strip()
    if manifest_path is None or not checkpoint_id:
        return 0
    manifest = _read_json_object(Path(manifest_path))
    source_attempts: list[int] = []
    for step in manifest.get("steps", ()):
        if not isinstance(step, dict):
            continue
        if str(step.get("checkpoint_after", "")).strip() != checkpoint_id:
            continue
        try:
            source_attempts.append(int(step.get("attempt", 0)))
        except (TypeError, ValueError):
            continue
    return max(source_attempts, default=0)


def _attempt_checkpoint_before(
    attempt_dir: Path,
    evaluation: dict[str, Any] | None = None,
) -> str:
    """Return the trusted checkpoint from which one attempt was dispatched."""

    payload = evaluation if isinstance(evaluation, dict) else {}
    evidence = payload.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    checkpoint_outcome = evidence.get("checkpoint_outcome", {})
    checkpoint_outcome = (
        checkpoint_outcome if isinstance(checkpoint_outcome, dict) else {}
    )
    checkpoint_id = str(checkpoint_outcome.get("checkpoint_before", "")).strip()
    if checkpoint_id:
        return checkpoint_id
    record = _read_json_object(attempt_dir / "evolution_record.json")
    return str(record.get("checkpoint_before", "")).strip()


def _attempt_evaluation_is_system_failure(evaluation: dict[str, Any]) -> bool:
    """Keep provider/evidence failures out of persistent action counterexamples."""

    if str(evaluation.get("failure_level", "")).strip() == "system":
        return True
    classification = classify_failure(
        str(evaluation.get("failure_type", "")),
        metrics=(
            evaluation.get("metrics", {})
            if isinstance(evaluation.get("metrics"), dict)
            else {}
        ),
        evidence=(
            evaluation.get("evidence", {})
            if isinstance(evaluation.get("evidence"), dict)
            else {}
        ),
    )
    return classification.level == "system"


def _legacy_coarse_recurrence_without_departure(
    evaluation: dict[str, Any],
) -> bool:
    """Identify pre-V2 recurrence evidence that did not prove a real cycle."""

    evidence = evaluation.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    transition = evidence.get("transition_decision", {})
    transition = transition if isinstance(transition, dict) else {}
    transition_evidence = transition.get("evidence", {})
    transition_evidence = (
        transition_evidence if isinstance(transition_evidence, dict) else {}
    )
    return bool(
        str(transition_evidence.get("source", "")).strip()
        == "state_progress_cycle_detector"
        and str(transition_evidence.get("kind", "")).strip()
        == "coarse_state_recurrence"
        and not bool(transition_evidence.get("departed_before_return", False))
    )


def _load_rejected_policy_history(
    output_dir: Path,
    *,
    checkpoint_id: str,
    limit: int,
    minimum_attempt_index: int = 0,
) -> tuple[PolicyScript, ...]:
    """Recover recent rejected transactions after an interrupted evolution run."""

    candidates: list[tuple[int, PolicyScript]] = []
    for attempt_dir in output_dir.glob("attempt_*"):
        try:
            attempt_index = int(attempt_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if attempt_index < minimum_attempt_index:
            continue
        evaluation = _read_attempt_evaluation(attempt_dir)
        if (
            not evaluation
            or bool(evaluation.get("success", False))
            or _attempt_evaluation_is_system_failure(evaluation)
            or _legacy_coarse_recurrence_without_departure(evaluation)
        ):
            continue
        recorded_checkpoint = _attempt_checkpoint_before(attempt_dir, evaluation)
        if checkpoint_id and recorded_checkpoint != checkpoint_id:
            continue
        trace = _read_json_object(attempt_dir / "execution_trace.json")
        if not _trace_payload_has_physical_dispatch(trace):
            # Code/preflight failures remain useful within their live generation
            # process, but cannot become persistent physics bans after validators
            # or episode contracts are repaired.
            continue
        policy_payload = _read_json_object(attempt_dir / "policy.json")
        if not policy_payload:
            policy_payload = _read_json_object(attempt_dir / "team_plan.json")
        policy = _policy_from_history_payload(policy_payload)
        if policy is not None and not _obsolete_small_heading_preflight_rejection(
            evaluation,
            policy,
        ):
            candidates.append((attempt_index, policy))
    candidates.sort(key=lambda item: item[0])
    selected = candidates if int(limit) <= 0 else candidates[-max(1, int(limit)) :]
    return tuple(policy for _index, policy in selected)


def _obsolete_small_heading_preflight_rejection(
    evaluation: dict[str, Any],
    policy: PolicyScript,
) -> bool:
    """Ignore one pre-fix rejection now accepted by the turn continuity gate."""

    evidence = evaluation.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    preflight = evidence.get("turn_episode_preflight", {})
    preflight = preflight if isinstance(preflight, dict) else {}
    if bool(preflight.get("physically_dispatched", True)):
        return False
    error_text = str(preflight.get("error", ""))
    if (
        "global view already verifies a longitudinal or lateral passage-aligned"
        not in error_text
    ):
        return False
    if len(policy.robot_scripts) != 2:
        return False
    arguments = [
        _first_task_call_literal_arguments(script.code)
        for script in policy.robot_scripts
    ]
    methods = {_first_policy_api_method(script.code) for script in policy.robot_scripts}
    try:
        angles = {float(item["angle_deg"]) for item in arguments}
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        methods == {COORDINATED_PIVOT}
        and len(angles) == 1
        and 0.0 < next(iter(angles)) < MODEL_ROUTE_TURN_ANGLE_DEG
    )


def _load_trusted_checkpoint_linear_motion(
    output_dir: Path,
    *,
    checkpoint_id: str,
) -> dict[str, Any]:
    """Find the accepted linear rollout that created one live checkpoint."""

    candidates: list[tuple[int, dict[str, Any]]] = []
    for attempt_dir in output_dir.glob("attempt_*"):
        try:
            attempt_index = int(attempt_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        evaluation = _read_attempt_evaluation(attempt_dir)
        if not evaluation or not bool(evaluation.get("success", False)):
            continue
        record = _read_json_object(attempt_dir / "evolution_record.json")
        if str(record.get("checkpoint_after", "")).strip() != checkpoint_id:
            continue
        policy_payload = _read_json_object(attempt_dir / "policy.json")
        if not policy_payload:
            policy_payload = _read_json_object(attempt_dir / "team_plan.json")
        policy = _policy_from_history_payload(policy_payload)
        if policy is None or len(policy.robot_scripts) != 2:
            continue
        methods = {
            _first_policy_api_method(script.code) for script in policy.robot_scripts
        }
        modes = {script.coordination_mode for script in policy.robot_scripts}
        if methods != {COORDINATED_LINEAR_TRANSPORT} or modes != {"sync"}:
            continue
        arguments = [
            _first_task_call_literal_arguments(script.code)
            for script in policy.robot_scripts
        ]
        try:
            directions = {str(item["direction"]) for item in arguments}
            distances = {float(item["distance_m"]) for item in arguments}
            speeds = {float(item["speed_mps"]) for item in arguments}
            ramps = {float(item["ramp_s"]) for item in arguments}
        except (KeyError, TypeError, ValueError):
            continue
        if len(directions) != 1:
            continue
        shared_envelope = all(len(values) == 1 for values in (distances, speeds, ramps))
        candidate: dict[str, Any] = {
            "checkpoint_id": checkpoint_id,
            "source_attempt": attempt_index,
            "source_script_id": policy.script_id,
            "method": COORDINATED_LINEAR_TRANSPORT,
            "coordination_mode": "sync",
            "direction": next(iter(directions)),
            "shared_envelope": shared_envelope,
            "evidence": "physics-accepted rollout that created the restored checkpoint",
        }
        if shared_envelope:
            candidate.update(
                {
                    "distance_m": next(iter(distances)),
                    "speed_mps": next(iter(speeds)),
                    "ramp_s": next(iter(ramps)),
                }
            )
        candidates.append((attempt_index, candidate))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


_REJECTED_ROLLOUT_METRICS = (
    "payload_target_progress_m",
    "payload_step_displacement_xy_m",
    "front_step_displacement_m",
    "back_step_displacement_m",
    "front_back_step_vector_desync_m",
    "payload_formation_slip_m",
    "payload_support_center_error_m",
    "payload_support_force_front_n",
    "payload_support_force_back_n",
    "payload_support_force_balance_ratio",
    "payload_tilt_deg",
)


def _load_rejected_rollout_evidence(
    output_dir: Path,
    *,
    checkpoint_id: str,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    """Load compact verifier metrics plus executable calls for immediate replanning."""

    candidates: list[tuple[int, dict[str, Any]]] = []
    for attempt_dir in output_dir.glob("attempt_*"):
        try:
            attempt_index = int(attempt_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        evaluation = _read_attempt_evaluation(attempt_dir)
        if (
            not evaluation
            or bool(evaluation.get("success", False))
            or _attempt_evaluation_is_system_failure(evaluation)
            or _legacy_coarse_recurrence_without_departure(evaluation)
        ):
            continue
        recorded_checkpoint = _attempt_checkpoint_before(attempt_dir, evaluation)
        if checkpoint_id and recorded_checkpoint != checkpoint_id:
            continue
        trace = _read_json_object(attempt_dir / "execution_trace.json")
        if not _trace_payload_has_physical_dispatch(trace):
            # Preflight/code failures describe a stale generation contract, not
            # observed robot motion.  Persisting them as rollout evidence can
            # incorrectly steer a resumed checkpoint after that contract is fixed.
            continue
        policy_payload = _read_json_object(attempt_dir / "policy.json")
        if not policy_payload:
            policy_payload = _read_json_object(attempt_dir / "team_plan.json")
        policy = _policy_from_history_payload(policy_payload)
        if policy is None:
            continue
        metrics = evaluation.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        scripts = policy.robot_scripts or (
            RobotPolicyScript(
                robot_id="front",
                script_id=f"front_{policy.script_id}",
                code=policy.code,
                synchronization_group="legacy_team",
                coordination_mode="sync",
            ),
            RobotPolicyScript(
                robot_id="back",
                script_id=f"back_{policy.script_id}",
                code=policy.code,
                synchronization_group="legacy_team",
                coordination_mode="sync",
            ),
        )
        candidates.append(
            (
                attempt_index,
                {
                    "attempt": attempt_index,
                    "script_id": policy.script_id,
                    "checkpoint_before": recorded_checkpoint or checkpoint_id,
                    "failure_type": str(evaluation.get("failure_type", "")),
                    "failed_subtask": str(evaluation.get("failed_subtask", "")),
                    "strategy_signature": policy_strategy_signature(policy),
                    "robot_calls": [
                        {
                            "robot_id": script.robot_id,
                            "coordination_mode": script.coordination_mode,
                            "call": _first_task_call_source(script.code),
                        }
                        for script in sorted(scripts, key=lambda item: item.robot_id)
                    ],
                    "physics_metrics": {
                        key: metrics[key]
                        for key in _REJECTED_ROLLOUT_METRICS
                        if key in metrics
                    },
                },
            )
        )
    candidates.sort(key=lambda item: item[0])
    return tuple(item for _index, item in candidates[-max(1, int(limit)) :])


def _measured_sync_compensation_constraint(
    payload: dict[str, Any],
    *,
    selected_subtask: dict[str, str],
) -> dict[str, Any]:
    """Turn a measured same-checkpoint sync mismatch into an ordering constraint."""

    method = str(selected_subtask.get("selected_method", "")).strip()
    coordination_mode = str(selected_subtask.get("coordination_mode", "")).strip()
    if method not in _TRANSPORT_METHODS or coordination_mode != "sync":
        return {}
    selected_motion_objective = str(
        selected_subtask.get("motion_objective", "")
    ).strip()
    if method == COORDINATED_LINEAR_TRANSPORT and selected_motion_objective not in {
        "forward",
        "backward",
        "left",
        "right",
    }:
        return {}
    evaluation = payload.get("evaluation", {})
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    current_failure = str(evaluation.get("failure_type", "")).strip()
    if current_failure not in {"", "none", "formation_desync"}:
        return {}
    evidence_items = payload.get("recent_rejected_rollout_evidence", ())
    if not isinstance(evidence_items, (list, tuple)):
        return {}
    expected_signature = f"sync:{method}"
    for evidence in reversed(evidence_items):
        if not isinstance(evidence, dict):
            continue
        # Unequal final displacement is only diagnostic when the verifier
        # explicitly reports formation desynchronization.  After a payload
        # loss, one robot often travels farther because support has already
        # failed; treating that consequence as executor calibration produced
        # asymmetric speed commands and made the next support failure worse.
        if str(evidence.get("failure_type", "")) != "formation_desync":
            continue
        if str(evidence.get("strategy_signature", "")) != expected_signature:
            continue
        robot_calls = evidence.get("robot_calls", ())
        if not isinstance(robot_calls, (list, tuple)):
            continue
        call_methods = {
            str(item.get("robot_id", "")): _first_policy_api_method(
                str(item.get("call", ""))
            )
            for item in robot_calls
            if isinstance(item, dict)
        }
        if call_methods != {"front": method, "back": method}:
            continue
        if method == COORDINATED_LINEAR_TRANSPORT:
            evidence_objectives = {
                str(item.get("robot_id", "")): _linear_objective_from_code(
                    str(item.get("call", ""))
                )
                for item in robot_calls
                if isinstance(item, dict)
            }
            if evidence_objectives != {
                "front": selected_motion_objective,
                "back": selected_motion_objective,
            }:
                # SONIC's executor response is body-direction dependent. A
                # lateral mismatch cannot calibrate a sagittal command (or the
                # opposite lateral direction), even when the primitive name is
                # identical.
                continue
        call_parameters = {
            str(item.get("robot_id", "")): _transport_parameters_from_source(
                str(item.get("call", "")),
                method=method,
            )
            for item in robot_calls
            if isinstance(item, dict)
        }
        if any(call_parameters.get(robot_id) is None for robot_id in ("front", "back")):
            continue
        call_ramps = {
            str(item.get("robot_id", "")): _first_task_call_literal_arguments(
                str(item.get("call", ""))
            ).get("ramp_s")
            for item in robot_calls
            if isinstance(item, dict)
        }
        try:
            ramp_values = {
                float(call_ramps[robot_id]) for robot_id in ("front", "back")
            }
        except (KeyError, TypeError, ValueError):
            continue
        if len(ramp_values) != 1:
            continue
        metrics = evidence.get("physics_metrics", {})
        if not isinstance(metrics, dict):
            continue
        try:
            front_displacement = float(metrics["front_step_displacement_m"])
            back_displacement = float(metrics["back_step_displacement_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if min(front_displacement, back_displacement) < 0.05:
            continue
        if abs(front_displacement - back_displacement) <= 0.01:
            continue
        farther_robot = "front" if front_displacement > back_displacement else "back"
        nearer_robot = "back" if farther_robot == "front" else "front"
        displacement_by_robot = {
            "front": front_displacement,
            "back": back_displacement,
        }
        return {
            "method": method,
            "motion_objective": (
                selected_motion_objective or _LEGACY_LINEAR_DIRECTIONS.get(method, "")
            ),
            "farther_robot": farther_robot,
            "nearer_robot": nearer_robot,
            "farther_displacement_m": round(displacement_by_robot[farther_robot], 6),
            "nearer_displacement_m": round(displacement_by_robot[nearer_robot], 6),
            "source_attempt": int(evidence.get("attempt", 0) or 0),
            "minimum_distance_m": 0.05,
            "maximum_distance_m": max(
                call_parameters["front"][0],
                call_parameters["back"][0],
            ),
            "minimum_speed_mps": 0.15,
            "maximum_speed_mps": max(
                call_parameters["front"][1],
                call_parameters["back"][1],
            ),
            "required_speed_mps": max(
                call_parameters["front"][1],
                call_parameters["back"][1],
            ),
            "required_distance_m": max(
                call_parameters["front"][0],
                call_parameters["back"][0],
            ),
            "required_ramp_s": ramp_values.pop(),
            "maximum_pair_speed_delta_mps": 0.05,
        }
    return {}


def _load_goal_regression_method_history(
    output_dir: Path,
    *,
    checkpoint_id: str,
    limit: int,
    minimum_attempt_index: int = 0,
) -> tuple[str, ...]:
    """Recover body-frame methods proven to move away at one checkpoint."""

    methods: set[str] = set()
    for policy in _load_goal_regression_policy_history(
        output_dir,
        checkpoint_id=checkpoint_id,
        limit=limit,
        minimum_attempt_index=minimum_attempt_index,
    ):
        policy_methods, _objectives = _policy_directional_rejections(policy)
        methods.update(policy_methods)
    return tuple(sorted(methods))


def _load_goal_regression_objective_history(
    output_dir: Path,
    *,
    checkpoint_id: str,
    limit: int,
    minimum_attempt_index: int = 0,
) -> tuple[str, ...]:
    """Recover rejected objectives for parameterized coordination primitives."""

    objectives: set[str] = set()
    for policy in _load_goal_regression_policy_history(
        output_dir,
        checkpoint_id=checkpoint_id,
        limit=limit,
        minimum_attempt_index=minimum_attempt_index,
    ):
        _methods, policy_objectives = _policy_directional_rejections(policy)
        objectives.update(policy_objectives)
    return tuple(sorted(objectives))


def _load_route_regression_objective_history(
    output_dir: Path,
    *,
    checkpoint_id: str,
    limit: int,
    route_phase_guard: dict[str, Any] | None = None,
    minimum_attempt_index: int = 0,
) -> tuple[str, ...]:
    """Recover visual-route exclusions that action-space recovery must not relax."""

    objectives: set[str] = set()
    attempt_dirs = sorted(Path(output_dir).glob("attempt_*"))[-max(1, int(limit)) :]
    for attempt_dir in attempt_dirs:
        try:
            attempt_index = int(attempt_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if attempt_index < minimum_attempt_index:
            continue
        evaluation = _read_attempt_evaluation(attempt_dir)
        if str(evaluation.get("failure_type", "")) != "route_regression":
            continue
        if _route_regression_scope_is_stale(
            evaluation,
            route_phase_guard=route_phase_guard,
        ):
            continue
        recorded_checkpoint = _attempt_checkpoint_before(attempt_dir, evaluation)
        if checkpoint_id and recorded_checkpoint != checkpoint_id:
            continue
        policy_payload = _read_json_object(attempt_dir / "policy.json")
        if not policy_payload:
            policy_payload = _read_json_object(attempt_dir / "team_plan.json")
        policy = _policy_from_history_payload(policy_payload)
        if policy is None:
            continue
        _methods, policy_objectives = _policy_directional_rejections(policy)
        objectives.update(policy_objectives)
    return tuple(sorted(objectives))


def _route_regression_scope_is_stale(
    evaluation: dict[str, Any],
    *,
    route_phase_guard: dict[str, Any] | None,
) -> bool:
    """Retire a semantic-phase ban when trusted physics disproves its scope."""

    guard = route_phase_guard if isinstance(route_phase_guard, dict) else {}
    try:
        physical_prefix = int(guard["maximum_completed_prefix"])
    except (KeyError, TypeError, ValueError):
        return False
    if _legacy_coarse_recurrence_without_departure(evaluation):
        # V1 treated several consecutive checkpoints in one coarse grid cell as
        # a cycle. Only V2 evidence proves that the trajectory first departed
        # from the cell and then returned.
        return True
    evidence = evaluation.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    verification = evidence.get("route_direction_verification", {})
    verification = verification if isinstance(verification, dict) else {}
    return bool(
        str(verification.get("decision", "")).strip()
        == "reject_final_approach_target_regression"
        and physical_prefix < int(guard.get("route_gate_count", 4))
    )


def _load_goal_regression_policy_history(
    output_dir: Path,
    *,
    checkpoint_id: str,
    limit: int,
    minimum_attempt_index: int = 0,
) -> tuple[PolicyScript, ...]:
    policies: list[PolicyScript] = []
    attempt_dirs = sorted(output_dir.glob("attempt_*"))[-max(1, int(limit)) :]
    for attempt_dir in attempt_dirs:
        try:
            attempt_index = int(attempt_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if attempt_index < minimum_attempt_index:
            continue
        evaluation = _read_attempt_evaluation(attempt_dir)
        if _attempt_evaluation_is_system_failure(evaluation):
            continue
        metrics = evaluation.get("metrics", {})
        if not isinstance(metrics, dict) or not _has_measured_goal_regression(
            str(evaluation.get("failure_type", "")),
            metrics,
            evidence=(
                evaluation.get("evidence", {})
                if isinstance(evaluation.get("evidence"), dict)
                else {}
            ),
        ):
            continue
        recorded_checkpoint = _attempt_checkpoint_before(attempt_dir, evaluation)
        if checkpoint_id and recorded_checkpoint != checkpoint_id:
            continue
        policy_payload = _read_json_object(attempt_dir / "policy.json")
        if not policy_payload:
            policy_payload = _read_json_object(attempt_dir / "team_plan.json")
        policy = _policy_from_history_payload(policy_payload)
        if policy is None:
            continue
        policies.append(policy)
    return tuple(policies)


def _policy_directional_rejections(
    policy: PolicyScript,
) -> tuple[set[str], set[str]]:
    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="team",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="legacy_team",
        ),
    )
    methods: set[str] = set()
    objectives: set[str] = set()
    for script in scripts:
        method = _first_policy_api_method(script.code)
        if method not in MODEL_PLANNING_ACTION_METHODS:
            continue
        objective_parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(method)
        if objective_parameter is None:
            methods.add(method)
            continue
        arguments = _first_task_call_literal_arguments(script.code)
        objective = str(arguments.get(objective_parameter, "")).strip()
        if objective:
            objectives.add(f"{method}:{objective}")
            if method in _SIGNED_TURN_METHODS and objective in _SIGNED_TURN_DIRECTIONS:
                objectives.add(f"turn:{objective}")
    return methods, objectives


def _branch_rewind_forbidden_methods(context: object) -> set[str]:
    """Keep verifier-rejected primitive families blocked across ancestor rewinds."""

    payload = context if isinstance(context, dict) else {}
    methods = payload.get("blocked_primitive_families", ())
    if not isinstance(methods, (list, tuple, set)):
        return set()
    return {
        str(method)
        for method in methods
        if str(method) in COORDINATION_PRIMITIVE_METHODS
    }


def _motion_objective_is_forbidden(
    method: str,
    objective: str,
    forbidden_objectives: tuple[str, ...],
) -> bool:
    """Share signed-turn failure evidence across arc and pivot primitives."""

    forbidden = set(forbidden_objectives)
    if f"{method}:{objective}" in forbidden:
        return True
    return bool(
        method in _SIGNED_TURN_METHODS
        and objective in _SIGNED_TURN_DIRECTIONS
        and f"turn:{objective}" in forbidden
    )


def _normalize_repair_level_for_selected_strategy(
    *,
    repair_level: str,
    current_method: str,
    current_mode: str,
    selected_method: str,
    selected_mode: str,
) -> str:
    """Let a geometry-authorized strategy change pass the matching compiler gate."""

    if repair_level not in {"execution", "coordination", "script"}:
        return repair_level
    if current_method and selected_method and selected_method != current_method:
        return "strategy"
    if (
        repair_level == "execution"
        and current_mode
        and selected_mode
        and selected_mode != current_mode
    ):
        return "strategy"
    return repair_level


def _body_frame_goal_alignment(
    runtime_state: dict[str, Any],
    evaluation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Project the current stage goal into the team's measured body frame."""

    payload = runtime_state.get("payload", {})
    robots = runtime_state.get("robots", {})
    if not isinstance(payload, dict) or not isinstance(robots, dict):
        return {}
    try:
        payload_x = float(payload["x"])
        payload_y = float(payload["y"])
        goal_x = float(evaluation_metrics["active_stage_goal_x_m"])
        goal_y = float(evaluation_metrics["active_stage_goal_y_m"])
    except (KeyError, TypeError, ValueError):
        return {}
    yaws = []
    for robot in robots.values():
        if not isinstance(robot, dict):
            continue
        try:
            yaws.append(radians(float(robot["yaw_deg"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not yaws:
        return {}
    mean_yaw = atan2(
        sum(sin(value) for value in yaws),
        sum(cos(value) for value in yaws),
    )
    goal_dx = goal_x - payload_x
    goal_dy = goal_y - payload_y
    goal_norm = hypot(goal_dx, goal_dy)
    if goal_norm <= 1.0e-9:
        return {}
    goal_unit = (goal_dx / goal_norm, goal_dy / goal_norm)
    forward = (cos(mean_yaw), sin(mean_yaw))
    left = (-sin(mean_yaw), cos(mean_yaw))
    vectors = {
        "forward": forward,
        "backward": (-forward[0], -forward[1]),
        "left": left,
        "right": (-left[0], -left[1]),
    }
    scores = {
        direction: round(
            vector[0] * goal_unit[0] + vector[1] * goal_unit[1],
            4,
        )
        for direction, vector in vectors.items()
    }
    payload_axis_alignment_abs = None
    try:
        payload_yaw = radians(float(payload["yaw_deg"]))
    except (KeyError, TypeError, ValueError):
        try:
            payload_yaw = float(payload["yaw_rad"])
        except (KeyError, TypeError, ValueError):
            payload_yaw = None
    if payload_yaw is not None:
        payload_long_axis = (cos(payload_yaw), sin(payload_yaw))
        payload_axis_alignment_abs = round(
            abs(
                payload_long_axis[0] * goal_unit[0]
                + payload_long_axis[1] * goal_unit[1]
            ),
            4,
        )
    return {
        "source": "host_projection_of_current_trusted_state",
        "payload_xy_m": [round(payload_x, 4), round(payload_y, 4)],
        "active_goal_xy_m": [round(goal_x, 4), round(goal_y, 4)],
        "active_goal_delta_xy_m": [round(goal_dx, 4), round(goal_dy, 4)],
        "mean_robot_yaw_deg": round(mean_yaw * 180.0 / 3.141592653589793, 3),
        "payload_long_axis_goal_alignment_abs": payload_axis_alignment_abs,
        "candidate_goal_alignment": scores,
        "interpretation": (
            "+1 points directly toward the current stage goal, 0 is orthogonal, and -1 "
            "points directly away; choose from the remaining allowlisted objectives and "
            "measured rollout evidence"
        ),
    }


def _body_frame_final_target_projection(
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    """Project the live final target into the team's measured body frame."""

    target = runtime_state.get("target", {})
    target = target if isinstance(target, dict) else {}
    if not bool(target.get("available", False)):
        return {}
    try:
        target_x = float(target["center_x_m"])
        target_y = float(target["center_y_m"])
    except (KeyError, TypeError, ValueError):
        return {}
    projection = _body_frame_goal_alignment(
        runtime_state,
        {
            "active_stage_goal_x_m": target_x,
            "active_stage_goal_y_m": target_y,
        },
    )
    if not projection:
        return {}
    scores = projection.pop("candidate_goal_alignment", {})
    ranked_scores = sorted(
        (
            (str(direction), float(score))
            for direction, score in scores.items()
            if str(direction) in LOADED_TRANSLATION_DIRECTIONS
            and isinstance(score, (int, float))
            and isfinite(float(score))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    projection["source"] = "host_projection_of_live_final_target"
    projection["candidate_target_alignment"] = scores
    projection["target_facing_translation_directions"] = [
        direction for direction, score in ranked_scores if score >= -0.05
    ]
    projection["target_away_translation_directions"] = [
        direction for direction, score in ranked_scores if score < -0.05
    ]
    if ranked_scores:
        best_direction, best_score = ranked_scores[0]
        projection["required_target_facing_direction"] = best_direction
        projection["required_target_facing_alignment"] = round(best_score, 4)
    projection["target_away_setup_authorized"] = False
    projection["direction_authority"] = (
        "SDK directions are robot-body-frame commands. VLM ahead/left/right "
        "descriptions are camera-space advisory and cannot override this projection."
    )
    projection["interpretation"] = (
        "+1 points directly toward the final target, 0 is orthogonal, and -1 "
        "points directly away. A target-away final-approach translation requires "
        "an explicit host-authorized obstacle-clearing setup."
    )
    local_navigation = target_local_navigation_evidence(runtime_state)
    if local_navigation:
        projection["local_heading"] = local_navigation
    return projection


def _policy_local_navigation_contract(
    policy: PolicyScript | None,
) -> dict[str, Any]:
    if policy is None or not isinstance(policy.verification_contract, dict):
        return {}
    contract = policy.verification_contract.get(LOCAL_NAVIGATION_CONTRACT_KEY, {})
    return contract if isinstance(contract, dict) else {}


def _declared_local_reorientation_turn_direction(
    runtime_state: dict[str, Any],
    *,
    contract: dict[str, Any] | None,
) -> str:
    """Return the shortest turn toward an explicitly declared local reference.

    Merely having a final target in runtime state never activates this rule.
    The Task Orchestrator must declare the current navigation reference and the
    declaration must be compiled into the executable policy.
    """

    local = contract if isinstance(contract, dict) else {}
    if (
        not local.get("active", False)
        or local.get("reference") not in {"final_target", "runtime_local_goal"}
    ):
        return ""
    refreshed = refresh_local_navigation_contract(local, runtime_state)
    if not refreshed or not refreshed.get("heading_alignment_required", False):
        return ""
    allowed = tuple(refreshed.get("allowed_turn_directions", ()))
    return str(allowed[0]).strip() if len(allowed) == 1 else ""


def _final_approach_transport_envelope(
    runtime_state: dict[str, Any],
    *,
    task_plan: dict[str, Any],
    policy: PolicyScript | None = None,
    execution_strategy: dict[str, Any] | None = None,
    host_wall_clearance: dict[str, Any] | None = None,
    forbidden_transaction_signatures: set[str] | None = None,
    formation_sync_repair_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Intersect target-local tiers with clearance without changing direction."""

    payload = runtime_state.get("payload", {})
    target = runtime_state.get("target", {})
    payload = payload if isinstance(payload, dict) else {}
    target = target if isinstance(target, dict) else {}
    try:
        target_xy_error_m = hypot(
            float(target["center_x_m"]) - float(payload["x"]),
            float(target["center_y_m"]) - float(payload["y"]),
        )
    except (KeyError, TypeError, ValueError):
        return {}
    plan_is_final_approach = _task_plan_is_final_approach(task_plan)
    if not plan_is_final_approach and target_xy_error_m > 1.0:
        return {}
    if target_xy_error_m > 1.0:
        return {}

    sync_gate = (
        formation_sync_repair_gate
        if isinstance(formation_sync_repair_gate, dict)
        else {}
    )
    sync_probe_distance_m: float | None = None
    if sync_gate.get("active", False):
        probe_contract = sync_gate.get("probe_contract", {})
        probe_contract = probe_contract if isinstance(probe_contract, dict) else {}
        try:
            candidate_distance_m = float(
                probe_contract.get(
                    "required_distance_m",
                    FORMATION_SYNC_REPAIR_PROBE_DISTANCE_M,
                )
            )
        except (TypeError, ValueError):
            candidate_distance_m = 0.0
        if isfinite(candidate_distance_m) and candidate_distance_m > 0.0:
            sync_probe_distance_m = candidate_distance_m

    preferred = (
        (sync_probe_distance_m,)
        if sync_probe_distance_m is not None
        else ((0.5, 0.25) if target_xy_error_m > 0.40 else (0.25,))
    )
    blocked = forbidden_transaction_signatures or set()
    current_subtask = task_plan.get("current_subtask", {})
    current_subtask = current_subtask if isinstance(current_subtask, dict) else {}
    plan_strategy = current_subtask.get("strategy", {})
    strategy = (
        execution_strategy
        if isinstance(execution_strategy, dict)
        else (plan_strategy if isinstance(plan_strategy, dict) else {})
    )
    strategy_method = str(strategy.get("primitive", "")).strip()
    strategy_direction = str(strategy.get("semantic_objective", "")).strip()
    strategy_mode = str(strategy.get("coordination_mode", "sync")).strip() or "sync"
    candidate_template = policy
    if (
        strategy_method == COORDINATED_LINEAR_TRANSPORT
        and strategy_direction in LOADED_TRANSLATION_DIRECTIONS
    ):
        code = (
            "def policy(api):\n"
            "    api.coordinated_linear_transport("
            f"direction={strategy_direction!r}, distance_m=0.5, "
            f"speed_mps={VERIFIED_LINEAR_SPEED_MPS:g}, "
            f"ramp_s={VERIFIED_LINEAR_RAMP_S:g})\n"
            "    api.stop_all()\n"
        )
        candidate_template = PolicyScript(
            script_id="final_approach_envelope_candidate",
            attempt=0,
            code=code,
            robot_scripts=tuple(
                RobotPolicyScript(
                    robot_id=robot_id,
                    script_id=f"{robot_id}_final_approach_envelope_candidate",
                    code=code,
                    synchronization_group="final_approach_envelope_candidate",
                    contact_intent="maintain_payload_support",
                    coordination_mode=strategy_mode,
                )
                for robot_id in ("front", "back")
            ),
        )
    direction = (
        strategy_direction
        if strategy_method == COORDINATED_LINEAR_TRANSPORT
        else (
            _policy_linear_objective(candidate_template)
            if not strategy_method and candidate_template is not None
            else ""
        )
    )
    clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    translation_candidates = clearance.get("translation_candidates", {})
    translation_candidates = (
        translation_candidates if isinstance(translation_candidates, dict) else {}
    )
    direction_clearance = translation_candidates.get(direction, {})
    direction_clearance = (
        direction_clearance if isinstance(direction_clearance, dict) else {}
    )
    clearance_m = _safe_float(direction_clearance.get("clearance_m"))
    # The 0.50 m target tier is not permission to violate a 0.44 m wall cap.
    # Choose only an existing calibrated tier in the intersection. An active
    # synchronization probe remains exact: if it cannot fit, it is unavailable,
    # not silently replaced by a shorter, unverified probe.
    allowed_distances = tuple(
        distance_m
        for distance_m in preferred
        if clearance_m is None or distance_m <= clearance_m + 1.0e-9
    )
    clearance_evidence = (
        {
            "semantic_objective": direction,
            "authored_wall_clearance_m": clearance_m,
            "clearance_excluded_distance_m": [
                distance_m for distance_m in preferred if distance_m not in allowed_distances
            ],
        }
        if clearance_m is not None
        else {}
    )
    required_distance_m: float | None = None
    for distance_m in allowed_distances:
        if (
            candidate_template is None
            or not blocked
            or not _policy_uses_linear_transport(candidate_template)
        ):
            required_distance_m = distance_m
            break
        candidate = _policy_with_linear_distance(
            candidate_template,
            distance_m=distance_m,
        )
        signature = _policy_transaction_signature(candidate)
        if not signature or signature not in blocked:
            required_distance_m = distance_m
            break
    if required_distance_m is None:
        return {
            "active": True,
            "exhausted": True,
            "target_xy_error_m": target_xy_error_m,
            "allowed_distance_m": list(allowed_distances),
            **clearance_evidence,
            "reason": (
                "no target-local transport tier is both within authored-wall "
                "clearance and unblocked at this checkpoint; replan before Coder"
            ),
        }
    envelope = {
        "active": True,
        "scope": "final_target_approach_only",
        "activation_source": (
            "persistent_final_approach_plan"
            if plan_is_final_approach
            else "live_final_target_capture"
        ),
        "target_xy_error_m": target_xy_error_m,
        "release_alignment_tolerance_m": DEFAULT_TARGET_XY_TOLERANCE_M,
        "allowed_distance_m": list(allowed_distances),
        **clearance_evidence,
        "required_distance_m": required_distance_m,
        "required_speed_mps": VERIFIED_LINEAR_SPEED_MPS,
        "required_ramp_s": VERIFIED_LINEAR_RAMP_S,
        "rule": (
            "This envelope limits magnitude only. The Task Orchestrator still "
            "owns the body-frame direction from live target projection."
        ),
    }
    if sync_probe_distance_m is not None:
        envelope["distance_override_source"] = "formation_sync_repair_gate"
        envelope["rule"] = (
            "The active direction-matched formation synchronization probe is "
            "the more specific safety contract for this one transaction. It "
            "overrides the ordinary target-local distance only; direction, "
            "speed, ramp, wall clearance, and payload safety remain unchanged."
        )
    return envelope


def _task_plan_is_final_approach(task_plan: dict[str, Any]) -> bool:
    """Return whether the persistent plan is in the final loaded route phase."""

    if not isinstance(task_plan, dict):
        return False
    current_phase_id = str(task_plan.get("current_phase_id", "")).strip().lower()
    if "final_straight" in current_phase_id or "final_approach" in current_phase_id:
        return True
    for phase in task_plan.get("global_plan", ()):
        if not isinstance(phase, dict):
            continue
        if str(phase.get("phase_id", "")).strip().lower() != current_phase_id:
            continue
        phase_id = str(phase.get("phase_id", "")).strip().lower()
        return "final_straight" in phase_id or "final_approach" in phase_id
    return False


def _alignment_body_frame_geometry(runtime_state: dict[str, Any]) -> dict[str, Any]:
    """Project local alignment signs onto the current robot-pair separation."""

    robots = runtime_state.get("robots", {})
    if not isinstance(robots, dict):
        return {}
    parsed: dict[str, tuple[float, float, float]] = {}
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            return {}
        try:
            x = float(robot["x"])
            y = float(robot["y"])
            yaw = (
                float(robot["yaw_rad"])
                if robot.get("yaw_rad") is not None
                else radians(float(robot["yaw_deg"]))
            )
        except (KeyError, TypeError, ValueError):
            return {}
        if not all(isfinite(value) for value in (x, y, yaw)):
            return {}
        parsed[robot_id] = (x, y, yaw)

    front_x, front_y, _ = parsed["front"]
    back_x, back_y, _ = parsed["back"]
    pair_dx = back_x - front_x
    pair_dy = back_y - front_y
    separation = hypot(pair_dx, pair_dy)
    if separation <= 1.0e-9:
        return {}
    pair_unit = (pair_dx / separation, pair_dy / separation)
    effects: dict[str, dict[str, float]] = {}
    for robot_id, (_x, _y, yaw) in parsed.items():
        forward = (cos(yaw), sin(yaw))
        body_left = (-sin(yaw), cos(yaw))
        # d|back-front|/dq has opposite signs for the two endpoints.
        endpoint_sign = -1.0 if robot_id == "front" else 1.0
        effects[robot_id] = {
            "longitudinal": round(
                endpoint_sign * (pair_unit[0] * forward[0] + pair_unit[1] * forward[1]),
                4,
            ),
            "lateral": round(
                endpoint_sign
                * (pair_unit[0] * body_left[0] + pair_unit[1] * body_left[1]),
                4,
            ),
        }
    return {
        "source": "host_projection_of_current_trusted_state",
        "robot_pair_xy_separation_m": round(separation, 4),
        "front_to_back_world_unit_xy": [round(pair_unit[0], 4), round(pair_unit[1], 4)],
        "positive_delta_separation_effect_m_per_m": effects,
        "sign_contract": {
            "longitudinal_positive": "that robot's body-forward direction",
            "lateral_positive": "that robot's body-left direction",
            "effect_positive": "increases current front/back XY separation",
            "effect_negative": "decreases current front/back XY separation",
        },
        "interpretation": (
            "Multiply each robot's signed delta by its listed coefficient and sum. "
            "This is a local coordinate conversion, not a route or skill recommendation."
        ),
    }


def _trusted_checkpoint_formation_gate(
    rejected_rollout: object,
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    """Prioritize a measured spacing repair before residual stagger repair."""

    spacing_repair = _verifier_pair_spacing_alignment(
        rejected_rollout,
        runtime_state,
    )
    if spacing_repair:
        moving_robot = str(spacing_repair["moving_robot"])
        reference_robot = str(spacing_repair["reference_robot"])
        deltas = spacing_repair["robot_deltas_m"]
        checkpoint = runtime_state.get("checkpoint", {})
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        return {
            "required": True,
            "source": "verifier_pair_spacing_repair",
            "scope_checkpoint_id": str(checkpoint.get("checkpoint_id", "")).strip(),
            "primitive": COORDINATED_ALIGNMENT,
            "semantic_objective": str(spacing_repair["axis"]),
            "coordination_mode": "async",
            "lagging_robot": "",
            "moving_robot": moving_robot,
            "reference_robot": reference_robot,
            "team_roles": {
                moving_robot: "corrector",
                reference_robot: "reference",
            },
            "robot_deltas_m": {
                robot_id: round(float(deltas[robot_id]), 4)
                for robot_id in ("front", "back")
            },
            "failed_separation_m": round(
                float(spacing_repair["failed_separation_m"]),
                4,
            ),
            "trusted_separation_m": round(
                float(spacing_repair["trusted_separation_m"]),
                4,
            ),
            "minimum_separation_m": round(
                float(spacing_repair["minimum_separation_m"]),
                4,
            ),
            "contract": (
                "The rejected transaction crossed the minimum robot-pair spacing. "
                "Before residual stagger or route motion, hold the reference and "
                "apply exactly one local-body correction whose measured projection "
                "increases separation. This does not choose the later route direction."
            ),
        }
    return _trusted_checkpoint_longitudinal_alignment(runtime_state)


def _rejected_alignment_invalidates_formation_gate(
    formation_gate: object,
    *,
    policy: PolicyScript,
    report: EvaluationReport,
) -> bool:
    """Do not make a physically rejected alignment relation mandatory again."""

    gate = formation_gate if isinstance(formation_gate, dict) else {}
    if (
        not gate
        or report.success
        or not report.metrics
        or classify_report(report).level == "system"
    ):
        return False
    checkpoint_outcome = report.evidence.get("checkpoint_outcome", {})
    checkpoint_outcome = (
        checkpoint_outcome if isinstance(checkpoint_outcome, dict) else {}
    )
    if checkpoint_outcome.get("physically_dispatched") is False:
        return False
    try:
        parent = _policy_mutation_parent_strategy(policy)
    except ModelAgentError:
        return False
    parent_assignment = parent.get("team_assignment", {})
    parent_assignment = parent_assignment if isinstance(parent_assignment, dict) else {}
    parent_relation = {
        "primitive": str(parent.get("primitive", "")).strip(),
        "semantic_objective": str(parent.get("semantic_objective", "")).strip(),
        "coordination_mode": str(parent.get("coordination_mode", "")).strip(),
        "team_roles": {
            robot_id: str(
                parent_assignment.get(robot_id, {}).get("role", "")
                if isinstance(parent_assignment.get(robot_id), dict)
                else ""
            ).strip()
            for robot_id in ("front", "back")
        },
    }
    return (
        _exhausted_coordination_relation_signature(parent_relation)
        == _exhausted_coordination_relation_signature(gate)
        != ""
    )


def _superseded_formation_gate_context(
    formation_gate: dict[str, Any],
) -> dict[str, Any]:
    """Explain why a current-state measurement is advisory for this repair."""

    return {
        **formation_gate,
        "status": "superseded_by_measured_alignment_regression",
        "instruction": (
            "The current checkpoint still has a measured formation error, but the "
            "matching primitive/objective/mode/role relation just moved physically "
            "without improving alignment. Do not copy or magnitude-tune that relation. "
            "Choose a materially different strategy or coordination relation and let "
            "the physics verifier test both formation improvement and route progress."
        ),
    }


def _minimum_effective_parameter_exhaustion(
    policy: PolicyScript,
    report: EvaluationReport,
    *,
    trace: ExecutionTrace | None = None,
    forbidden_transaction_signatures: set[str] | None = None,
) -> dict[str, Any]:
    """Describe a physically rejected primitive with no smaller legal setting."""

    if _alignment_sign_repair_contract(
        policy,
        report,
        trace,
        forbidden_transaction_signatures=(forbidden_transaction_signatures or set()),
    ):
        # The SONIC magnitude floor does not exhaust an untried opposite sign.
        # Treat the sign as one bounded executable parameter before escalating
        # the alignment relation or maneuver family.
        return {}
    alignment = _minimum_effective_alignment_parameter_exhaustion(policy, report)
    if alignment:
        return alignment
    return _minimum_effective_linear_parameter_exhaustion(
        policy,
        report,
        forbidden_transaction_signatures=(forbidden_transaction_signatures or set()),
    )


def _alignment_sign_repair_contract(
    policy: PolicyScript,
    report: EvaluationReport,
    trace: ExecutionTrace | None,
    *,
    forbidden_transaction_signatures: set[str] | None = None,
) -> dict[str, Any]:
    """Return one exact opposite-sign repair after a spacing safety abort."""

    if report.success or report.failure_type != "formation_desync":
        return {}
    if trace is None or "robot_pair_too_close" not in str(trace.error):
        return {}
    scripts = {item.robot_id: item for item in policy.robot_scripts}
    if set(scripts) != {"front", "back"} or {
        _first_policy_api_method(item.code) for item in scripts.values()
    } != {COORDINATED_ALIGNMENT}:
        return {}
    arguments = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in scripts.items()
    }
    axes = {str(item.get("axis", "")).strip() for item in arguments.values()}
    if len(axes) != 1 or next(iter(axes)) not in {"lateral", "longitudinal"}:
        return {}
    try:
        deltas = {
            robot_id: float(item["delta"]) for robot_id, item in arguments.items()
        }
        durations = {float(item["duration_s"]) for item in arguments.values()}
    except (KeyError, TypeError, ValueError):
        return {}
    correctors = [robot_id for robot_id, delta in deltas.items() if abs(delta) > 1.0e-9]
    references = [
        robot_id for robot_id, delta in deltas.items() if abs(delta) <= 1.0e-9
    ]
    if len(correctors) != 1 or len(references) != 1 or len(durations) != 1:
        return {}
    corrector = correctors[0]
    reference = references[0]
    required_deltas = {
        corrector: -deltas[corrector],
        reference: 0.0,
    }
    flipped_scripts = tuple(
        replace(
            script,
            code=_rewrite_task_call_keywords(
                script.code,
                method=COORDINATED_ALIGNMENT,
                updates={"delta": required_deltas[script.robot_id]},
            ),
        )
        for script in policy.robot_scripts
    )
    flipped_policy = replace(policy, robot_scripts=flipped_scripts)
    flipped_signature = _policy_transaction_signature(flipped_policy)
    if flipped_signature in (forbidden_transaction_signatures or set()):
        return {}
    roles = {
        item.robot_id: str(item.role).strip()
        for item in policy.assignments
        if item.robot_id in {"front", "back"}
    }
    if set(roles) != {"front", "back"}:
        roles = {corrector: "corrector", reference: "reference"}
    return {
        "kind": "alignment_sign_repair",
        "mutation_level": "parameter_tuning",
        "primitive": COORDINATED_ALIGNMENT,
        "semantic_objective": next(iter(axes)),
        "coordination_mode": "async",
        "preserve_roles": roles,
        "corrector_robot": corrector,
        "reference_robot": reference,
        "rejected_robot_deltas": deltas,
        "required_robot_deltas": required_deltas,
        "required_duration_s": next(iter(durations)),
        "tune_only": "delta_sign",
        "reason": (
            "The rejected minimum-magnitude correction crossed the too-close "
            "spacing guard. Preserve the measured roles and axis, and test the "
            "previously untried opposite local-body sign."
        ),
    }


def _minimum_effective_linear_parameter_exhaustion(
    policy: PolicyScript,
    report: EvaluationReport,
    *,
    forbidden_transaction_signatures: set[str] | None = None,
) -> dict[str, Any]:
    """Detect loaded-linear rejection with no lower untried distance tier."""

    if report.success or report.failure_type not in _UNSAFE_MAGNITUDE_FAILURES:
        return {}
    if len(policy.robot_scripts) != 2:
        return {}
    scripts = {item.robot_id: item for item in policy.robot_scripts}
    if set(scripts) != {"front", "back"}:
        return {}
    if {_first_policy_api_method(item.code) for item in scripts.values()} != {
        COORDINATED_LINEAR_TRANSPORT
    }:
        return {}
    modes = {str(item.coordination_mode).strip().lower() for item in scripts.values()}
    if modes != {"sync"}:
        return {}
    arguments = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in scripts.items()
    }
    try:
        directions = {str(item["direction"]).strip() for item in arguments.values()}
        distances = {float(item["distance_m"]) for item in arguments.values()}
    except (KeyError, TypeError, ValueError):
        return {}
    supported_distances = _linear_distance_choices_for_policy(policy)
    minimum_distance_m = min(supported_distances)
    if len(directions) != 1 or len(distances) != 1:
        return {}
    direction = next(iter(directions))
    distance_m = next(iter(distances))
    next_unblocked_distance_m = _next_lower_unblocked_linear_distance_tier(
        policy,
        distance_m=distance_m,
        forbidden_transaction_signatures=(forbidden_transaction_signatures or set()),
    )
    if next_unblocked_distance_m is not None:
        return {}
    return {
        "kind": "loaded_linear_parameter_envelope_exhausted",
        "primitive": COORDINATED_LINEAR_TRANSPORT,
        "semantic_objective": direction,
        "coordination_mode": "sync",
        "failed_distance_m": distance_m,
        "minimum_supported_distance_m": minimum_distance_m,
        "lower_supported_tiers_already_rejected": [
            choice for choice in supported_distances if choice < distance_m - 1.0e-9
        ],
        "physical_failure_type": report.failure_type,
        "required_escalation": (
            "strategy_mutation; coordinated_linear_transport has no lower "
            "untried supported distance tier or alternate loaded coordination relation"
        ),
    }


def _minimum_effective_alignment_parameter_exhaustion(
    policy: PolicyScript,
    report: EvaluationReport,
) -> dict[str, Any]:
    """Detect an alignment failure with no legal parameter-only repair."""

    safety_failures = {
        "formation_desync",
        "payload_drop",
        "payload_not_supported",
        "payload_slip",
        "payload_unstable",
        "robot_fall",
    }
    progress_failures = {
        "alignment_regression",
        "no_payload_progress",
    }
    if report.success or report.failure_type not in safety_failures | progress_failures:
        return {}
    if len(policy.robot_scripts) != 2:
        return {}
    scripts = {item.robot_id: item for item in policy.robot_scripts}
    if set(scripts) != {"front", "back"} or {
        _first_policy_api_method(item.code) for item in scripts.values()
    } != {COORDINATED_ALIGNMENT}:
        return {}
    arguments = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in scripts.items()
    }
    axes = {str(item.get("axis", "")).strip() for item in arguments.values()}
    if len(axes) != 1:
        return {}
    axis = next(iter(axes))
    try:
        deltas = {
            robot_id: float(item["delta"]) for robot_id, item in arguments.items()
        }
    except (KeyError, TypeError, ValueError):
        return {}
    reference_robots = [
        robot_id for robot_id, delta in deltas.items() if abs(delta) <= 1.0e-9
    ]
    corrector_robots = [
        robot_id for robot_id, delta in deltas.items() if abs(delta) > 1.0e-9
    ]
    minimum_delta = (
        2.0
        if axis == "yaw"
        else MIN_LONGITUDINAL_ALIGNMENT_DELTA_M
        if axis == "longitudinal"
        else 0.05
    )
    maximum_delta = (
        30.0
        if axis == "yaw"
        else MAX_LONGITUDINAL_ALIGNMENT_DELTA_M
        if axis == "longitudinal"
        else 0.50
    )
    if len(reference_robots) != 1 or len(corrector_robots) != 1:
        return {}
    failed_delta = deltas[corrector_robots[0]]
    at_lower_safety_bound = (
        report.failure_type in safety_failures
        and abs(failed_delta) <= minimum_delta + 1.0e-9
    )
    at_upper_progress_bound = (
        report.failure_type in progress_failures
        and abs(failed_delta) >= maximum_delta - 1.0e-9
    )
    if not at_lower_safety_bound and not at_upper_progress_bound:
        return {}
    roles = {
        item.robot_id: str(item.role).strip()
        for item in policy.assignments
        if item.robot_id in {"front", "back"} and str(item.role).strip()
    }
    if set(roles) != {"front", "back"}:
        roles = {
            corrector_robots[0]: "corrector",
            reference_robots[0]: "reference",
        }
    modes = {
        str(item.coordination_mode).strip()
        for item in policy.robot_scripts
        if str(item.coordination_mode).strip()
    }
    exhausted = {
        "kind": (
            "alignment_parameter_ceiling_exhausted"
            if at_upper_progress_bound
            else "alignment_parameter_floor_exhausted"
        ),
        "primitive": COORDINATED_ALIGNMENT,
        "semantic_objective": axis,
        "coordination_mode": next(iter(modes)) if len(modes) == 1 else "async",
        "team_roles": roles,
        "failed_corrector_robot": corrector_robots[0],
        "reference_robot": reference_robots[0],
        "failed_delta": failed_delta,
        "minimum_effective_abs_delta": minimum_delta,
        "physical_failure_type": report.failure_type,
    }
    if at_upper_progress_bound:
        exhausted.update(
            {
                "maximum_allowed_abs_delta": maximum_delta,
                "required_escalation": (
                    "coordination_mutation_or_strategy_mutation; parameter_tuning "
                    "cannot increase delta beyond the trusted formation gate"
                ),
            }
        )
    else:
        exhausted["required_escalation"] = (
            "coordination_mutation_or_strategy_mutation; parameter_tuning is "
            "not executable below the SONIC observability floor"
        )
    return exhausted


def _exhausted_coordination_relation_signature(
    relation: object,
) -> str:
    item = relation if isinstance(relation, dict) else {}
    roles = item.get("team_roles", {})
    roles = roles if isinstance(roles, dict) else {}
    payload = {
        "primitive": str(item.get("primitive", "")).strip(),
        "semantic_objective": str(item.get("semantic_objective", "")).strip(),
        "coordination_mode": str(item.get("coordination_mode", "")).strip(),
        "team_roles": {
            robot_id: str(roles.get(robot_id, "")).strip()
            for robot_id in ("front", "back")
        },
    }
    if (
        not payload["primitive"]
        or not payload["semantic_objective"]
        or not payload["coordination_mode"]
        or not all(payload["team_roles"].values())
    ):
        return ""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _validate_strategy_not_checkpoint_exhausted(
    *,
    method: str,
    objective: str,
    mode: str,
    assignment: object,
    exhausted_relations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> None:
    assignment = assignment if isinstance(assignment, dict) else {}
    roles = {
        robot_id: str(
            assignment.get(robot_id, {}).get("role", "")
            if isinstance(assignment.get(robot_id), dict)
            else ""
        ).strip()
        for robot_id in ("front", "back")
    }
    candidate = _exhausted_coordination_relation_signature(
        {
            "primitive": method,
            "semantic_objective": objective,
            "coordination_mode": mode,
            "team_roles": roles,
        }
    )
    if not candidate:
        return
    for relation in exhausted_relations:
        if candidate == _exhausted_coordination_relation_signature(relation):
            raise ModelRoleError(
                "checkpoint-local exhausted coordination relation cannot be selected "
                "again: the same primitive, semantic objective, mode, and immutable-ID "
                "roles already failed physically at SONIC's minimum effective magnitude. "
                "Change the coordination roles or the strategy"
            )


def _coordination_relation_is_checkpoint_exhausted(
    candidate: object,
    exhausted_relations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> bool:
    """Return whether one exact primitive/objective/mode/role relation is exhausted."""

    candidate_signature = _exhausted_coordination_relation_signature(candidate)
    if not candidate_signature:
        return False
    return any(
        candidate_signature == _exhausted_coordination_relation_signature(relation)
        for relation in exhausted_relations
    )


def _coordination_variants_exhausted(
    parent_strategy: dict[str, Any],
    exhausted_relations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> bool:
    """Return whether every finite temporary-role variant failed at this checkpoint."""

    method = str(parent_strategy.get("primitive", "")).strip()
    objective = str(parent_strategy.get("semantic_objective", "")).strip()
    mode = str(parent_strategy.get("coordination_mode", "")).strip()
    variant_limit = coordination_role_variant_limit(method)
    if not variant_limit or not method or not objective or not mode:
        return False
    role_variants: set[tuple[str, str]] = set()
    for relation in exhausted_relations:
        if (
            str(relation.get("primitive", "")).strip() != method
            or str(relation.get("semantic_objective", "")).strip() != objective
            or str(relation.get("coordination_mode", "")).strip() != mode
        ):
            continue
        roles = relation.get("team_roles", {})
        roles = roles if isinstance(roles, dict) else {}
        signature = tuple(
            str(roles.get(robot_id, "")).strip()
            for robot_id in ("front", "back")
        )
        if all(signature):
            role_variants.add(signature)
    return len(role_variants) >= variant_limit


def _host_locked_coordination_mutation_reason(
    parent_strategy: dict[str, Any],
    *,
    host_pivot_role_contract: dict[str, Any] | None,
) -> str:
    """Explain when a signed pivot has no coordination-only child strategy."""

    method = str(parent_strategy.get("primitive", "")).strip()
    objective = str(parent_strategy.get("semantic_objective", "")).strip()
    mode = str(parent_strategy.get("coordination_mode", "")).strip()
    contract = (
        host_pivot_role_contract
        if isinstance(host_pivot_role_contract, dict)
        else {}
    )
    assignments = contract.get("role_assignment_by_direction", {})
    assignments = assignments if isinstance(assignments, dict) else {}
    fixed_roles = assignments.get(objective, {})
    fixed_roles = fixed_roles if isinstance(fixed_roles, dict) else {}
    if (
        method != COORDINATED_PIVOT
        or objective not in _SIGNED_TURN_DIRECTIONS
        or mode != "async"
        or set(coordination_modes_for_method(method)) != {"async"}
        or set(fixed_roles) != {"front", "back"}
        or set(fixed_roles.values()) != {"inner", "outer"}
    ):
        return ""
    return (
        "api.coordinated_pivot exposes only async mode, and host geometry fixes "
        f"the immutable-ID roles for {objective}: front={fixed_roles['front']}, "
        f"back={fixed_roles['back']}. No coordination-only child exists; change "
        "the maneuver primitive or semantic objective with strategy_mutation"
    )


def _reconcile_mutation_contract_with_local_navigation(
    contract: dict[str, Any],
    *,
    parent_strategy: dict[str, Any],
    local_navigation_contract: dict[str, Any] | None,
    report: EvaluationReport,
    parameter_envelope_exhausted: dict[str, Any] | None = None,
) -> None:
    """Keep model-owned mutation levels consistent with executable route space.

    A physical support failure during the correct signed local turn does not
    prove that the route direction is wrong.  Preserve the world direction and
    sign while leaving all physically meaningful repair layers available:
    parameter tuning, temporary-role/coordination repair, and a strategy switch
    between shared-heading implementations such as static turn and pivot.
    """

    local = (
        local_navigation_contract
        if isinstance(local_navigation_contract, dict)
        else {}
    )
    parent_method = str(parent_strategy.get("primitive", "")).strip()
    parent_objective = str(
        parent_strategy.get("semantic_objective", "")
    ).strip()
    allowed_turns = {
        str(item).strip()
        for item in local.get("allowed_turn_directions", ())
        if str(item).strip()
    }
    local_turn_still_required = bool(
        local.get("active", False)
        and local.get("heading_alignment_required", False)
        and parent_method in TURN_METHODS
        and allowed_turns == {parent_objective}
    )
    execution_failure = report.failure_type in {
        "formation_desync",
        "payload_drop",
        "payload_not_supported",
        "payload_slip",
        "payload_unstable",
        "robot_pair_too_close",
    }
    if not (local_turn_still_required and execution_failure):
        return

    levels = contract.get("levels", {})
    levels = levels if isinstance(levels, dict) else {}
    available_levels = [
        level
        for level in contract.get("available_levels", ())
        if isinstance(levels.get(level), dict)
        and levels[level].get("available", False)
    ]
    exhausted = (
        parameter_envelope_exhausted
        if isinstance(parameter_envelope_exhausted, dict)
        else {}
    )
    if exhausted and "parameter_tuning" in available_levels:
        available_levels.remove("parameter_tuning")
        parameter = levels.get("parameter_tuning", {})
        if isinstance(parameter, dict):
            parameter["available"] = False
            parameter["unavailable_reason"] = (
                "the bounded parameter envelope for this rejected executable is exhausted"
            )
    contract["available_levels"] = available_levels
    exposed_turn_implementations = _model_exposed_shared_heading_implementations(
        parent_objective
    )
    contract["local_navigation_reconciliation"] = {
        "reason": "correct_signed_turn_execution_failure",
        "preserve_world_direction": list(
            local.get("goal_world_direction_xy", ())
        ),
        "required_semantic_objective": parent_objective,
        "allowed_shared_heading_implementations": list(
            exposed_turn_implementations
        ),
        "allowed_repairs": [
            "change one bounded action magnitude",
            "change temporary inner/outer coordination roles when supported",
            "switch shared-heading implementation while preserving turn sign",
        ],
    }


def _model_exposed_shared_heading_implementations(
    direction: str,
) -> tuple[str, ...]:
    """Return callable shared-heading methods that preserve one signed turn.

    Coordination internals such as ``coordinated_arc_transport`` may still be
    recognized by evaluators for old artifacts, but they must not be advertised
    as mutation choices when they are absent from the model-facing SDK. Static
    turn methods encode their sign in the method name, so exposing both signs
    would contradict the locked world-direction contract.
    """

    direction = str(direction).strip().lower()
    static_method = {
        "clockwise": "turn_clockwise",
        "counterclockwise": "turn_counterclockwise",
    }.get(direction, "")
    exposed_methods = MODEL_PLANNING_ACTION_METHODS
    candidates = (COORDINATED_PIVOT, static_method)
    return tuple(
        method
        for method in candidates
        if method and method in exposed_methods
    )


def _retire_checkpoint_exhausted_task_plan_strategy(
    task_plan: object,
    *,
    exhausted_relations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep the global plan but remove an exact checkpoint-local dead subtask."""

    plan = task_plan if isinstance(task_plan, dict) else {}
    if not plan or not exhausted_relations:
        return dict(plan)
    retired = json.loads(json.dumps(plan))
    current_subtask = retired.get("current_subtask", {})
    current_subtask = current_subtask if isinstance(current_subtask, dict) else {}
    strategy = current_subtask.get("strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    assignment = current_subtask.get(
        "team_assignment",
        strategy.get("team_assignment", {}),
    )
    assignment = assignment if isinstance(assignment, dict) else {}
    roles = {
        robot_id: str(
            assignment.get(robot_id, {}).get("role", "")
            if isinstance(assignment.get(robot_id), dict)
            else ""
        ).strip()
        for robot_id in ("front", "back")
    }
    candidate = {
        "primitive": str(strategy.get("primitive", "")).strip(),
        "semantic_objective": str(strategy.get("semantic_objective", "")).strip(),
        "coordination_mode": str(strategy.get("coordination_mode", "")).strip(),
        "team_roles": roles,
    }
    if not _coordination_relation_is_checkpoint_exhausted(
        candidate,
        exhausted_relations,
    ):
        return retired
    current_subtask["strategy"] = {}
    current_subtask["team_assignment"] = {}
    current_subtask["checkpoint_local_replan_required"] = True
    current_subtask["guidance"] = (
        "The prior exact coordination relation is retired at this checkpoint. "
        "Infer a materially different executable strategy or temporary role mapping "
        "from the current trusted physics state and global image."
    )
    retired["current_subtask"] = current_subtask
    retired["checkpoint_local_strategy_retired"] = True
    return retired


def _persist_checkpoint_exhausted_coordination_relation(
    path: Path,
    relation: dict[str, Any],
) -> None:
    """Merge one verifier-grounded relation into the run-local checkpoint ledger."""

    scope_checkpoint_id = str(relation.get("scope_checkpoint_id", "")).strip()
    signature = _exhausted_coordination_relation_signature(relation)
    if not scope_checkpoint_id or not signature:
        return
    payload = _read_json_object(path)
    stored_relations = payload.get("relations", ())
    stored_relations = (
        stored_relations if isinstance(stored_relations, (list, tuple)) else ()
    )
    merged: dict[str, dict[str, Any]] = {}
    for item in (*stored_relations, relation):
        if not isinstance(item, dict):
            continue
        item_scope = str(item.get("scope_checkpoint_id", "")).strip()
        item_signature = _exhausted_coordination_relation_signature(item)
        if item_scope and item_signature:
            merged[f"{item_scope}\0{item_signature}"] = dict(item)
    write_json(
        path,
        {
            "schema_version": 1,
            "relations": [merged[key] for key in sorted(merged)],
        },
    )


def _load_checkpoint_alignment_parameter_exhaustions(
    history_dirs: tuple[Path, ...],
    *,
    checkpoint_id: str,
    minimum_attempt_index: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Load physically proven minimum-delta failures for one checkpoint."""

    if not checkpoint_id:
        return ()
    relations: dict[str, dict[str, Any]] = {}
    seen_dirs: set[Path] = set()
    for directory in history_dirs:
        resolved = Path(directory).resolve(strict=False)
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        persisted = _read_json_object(
            resolved / CHECKPOINT_EXHAUSTED_COORDINATION_RELATIONS_FILE
        )
        persisted_relations = persisted.get("relations", ())
        if isinstance(persisted_relations, (list, tuple)):
            for relation in persisted_relations:
                if not isinstance(relation, dict):
                    continue
                if (
                    str(relation.get("scope_checkpoint_id", "")).strip()
                    != checkpoint_id
                ):
                    continue
                source_attempt = _attempt_index_from_artifact_path(
                    relation.get("source_artifact_dir", "")
                )
                if (
                    source_attempt is not None
                    and source_attempt < minimum_attempt_index
                ):
                    continue
                signature = _exhausted_coordination_relation_signature(relation)
                if signature:
                    relations[signature] = dict(relation)
        for attempt_dir in resolved.glob("attempt_*"):
            try:
                attempt_index = int(attempt_dir.name.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if attempt_index < minimum_attempt_index:
                continue
            evaluation = _read_attempt_evaluation(attempt_dir)
            if (
                not evaluation
                or bool(evaluation.get("success", False))
                or _attempt_evaluation_is_system_failure(evaluation)
            ):
                continue
            outcome = evaluation.get("evidence", {})
            outcome = outcome if isinstance(outcome, dict) else {}
            checkpoint_outcome = outcome.get("checkpoint_outcome", {})
            checkpoint_outcome = (
                checkpoint_outcome if isinstance(checkpoint_outcome, dict) else {}
            )
            record = _read_json_object(attempt_dir / "evolution_record.json")
            source_checkpoint = str(
                checkpoint_outcome.get("checkpoint_before")
                or record.get("checkpoint_before")
                or ""
            ).strip()
            if source_checkpoint != checkpoint_id:
                continue
            policy_payload = _read_json_object(attempt_dir / "policy.json")
            if not policy_payload:
                policy_payload = _read_json_object(attempt_dir / "team_plan.json")
            policy = _policy_from_history_payload(policy_payload)
            if policy is None:
                continue
            report = EvaluationReport(
                success=False,
                failure_type=str(evaluation.get("failure_type", "")),
                metrics=(
                    dict(evaluation.get("metrics", {}))
                    if isinstance(evaluation.get("metrics"), dict)
                    else {}
                ),
                evidence=outcome,
            )
            relation = _minimum_effective_alignment_parameter_exhaustion(
                policy,
                report,
            )
            if not relation:
                continue
            relation = {
                **relation,
                "scope_checkpoint_id": checkpoint_id,
                "source_artifact_dir": str(attempt_dir),
            }
            signature = _exhausted_coordination_relation_signature(relation)
            if signature:
                relations[signature] = relation
    return tuple(relations.values())


def _trusted_checkpoint_longitudinal_alignment(
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    """Derive one forward-only lagger repair from the current trusted state."""

    if not isinstance(runtime_state, dict):
        return {}
    payload = runtime_state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    support_robots = {
        str(robot_id)
        for robot_id in (
            contacts.get(
                "support_robots",
                contacts.get("robots", ()),
            )
            if contacts
            else payload.get("supporting_robots", ())
        )
    }
    floor_contact = bool(
        contacts.get("floor", False)
        if contacts
        else payload.get("floor_contact", False)
    )
    if (
        not bool(payload.get("active", False))
        or support_robots != {"front", "back"}
        or floor_contact
    ):
        return {}

    turn_episode = runtime_state.get("turn_episode", {})
    turn_episode = turn_episode if isinstance(turn_episode, dict) else {}
    if (
        bool(turn_episode.get("active", False))
        and str(turn_episode.get("purpose", "")).strip() == "route_turn"
        and str(turn_episode.get("status", "")).strip() != PASSAGE_ALIGNED_STATUS
    ):
        return {}

    robots = runtime_state.get("robots", {})
    if not isinstance(robots, dict):
        return {}
    parsed: dict[str, tuple[float, float, float]] = {}
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            return {}
        root_pose = robot.get("root_pose", {})
        pose = root_pose if isinstance(root_pose, dict) and root_pose else robot
        try:
            x = float(pose["x"])
            y = float(pose["y"])
            yaw = (
                float(pose["yaw_rad"])
                if pose.get("yaw_rad") is not None
                else radians(float(pose["yaw_deg"]))
            )
        except (KeyError, TypeError, ValueError):
            return {}
        if not all(isfinite(value) for value in (x, y, yaw)):
            return {}
        parsed[robot_id] = (x, y, yaw)

    front_x, front_y, front_yaw = parsed["front"]
    back_x, back_y, back_yaw = parsed["back"]
    mean_heading = atan2(
        sin(front_yaw) + sin(back_yaw),
        cos(front_yaw) + cos(back_yaw),
    )
    signed_stagger = (back_x - front_x) * cos(mean_heading) + (back_y - front_y) * sin(
        mean_heading
    )
    if (
        not isfinite(signed_stagger)
        or abs(signed_stagger) <= LONGITUDINAL_ALIGNMENT_TRIGGER_M
    ):
        return {}

    lagging_robot = "front" if signed_stagger > 0.0 else "back"
    reference_robot = "back" if lagging_robot == "front" else "front"
    correction_delta = min(
        MAX_LONGITUDINAL_ALIGNMENT_DELTA_M,
        max(MIN_LONGITUDINAL_ALIGNMENT_DELTA_M, abs(signed_stagger)),
    )
    checkpoint = runtime_state.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    return {
        "required": True,
        "scope_checkpoint_id": str(checkpoint.get("checkpoint_id", "")).strip(),
        "primitive": COORDINATED_ALIGNMENT,
        "semantic_objective": "longitudinal",
        "coordination_mode": "async",
        "signed_back_minus_front_stagger_m": round(signed_stagger, 4),
        "lagging_robot": lagging_robot,
        "reference_robot": reference_robot,
        "team_roles": {
            lagging_robot: "corrector",
            reference_robot: "reference",
        },
        "robot_deltas_m": {
            lagging_robot: round(correction_delta, 4),
            reference_robot: 0.0,
        },
        "contract": (
            "Before another route transaction, hold the measured reference robot "
            "and move only the measured lagger body-forward. This is a current-"
            "checkpoint formation safety gate, not a route-direction decision."
        ),
    }


def _validate_alignment_spacing_repair_candidate(
    candidate: dict[str, Any],
    *,
    runtime_state: dict[str, Any],
    report: EvaluationReport,
) -> None:
    """Prevent a spacing repair from repeating the measured unsafe sign relation."""

    if (
        candidate.get("method") != COORDINATED_ALIGNMENT
        or candidate.get("motion_objective") not in {"longitudinal", "lateral"}
        or report.failure_type != "formation_desync"
    ):
        return
    geometry = _alignment_body_frame_geometry(runtime_state)
    if not geometry:
        return
    try:
        current_separation = float(geometry["robot_pair_xy_separation_m"])
        failed_separation = float(report.metrics["robot_pair_xy_separation_m"])
        axis = str(candidate["motion_objective"])
        robot_parameters = candidate["robot_parameters"]
        effects = geometry["positive_delta_separation_effect_m_per_m"]
        predicted_change = sum(
            float(effects[robot_id][axis]) * float(robot_parameters[robot_id]["delta"])
            for robot_id in ("front", "back")
        )
    except (KeyError, TypeError, ValueError):
        return
    measured_change = failed_separation - current_separation
    if measured_change < -1.0e-6 and predicted_change <= 1.0e-6:
        raise ModelAgentError(
            "the rejected formation became too close, but this alignment sign pair is "
            f"predicted to change separation by {predicted_change:.4f}m; choose local-body "
            "delta signs whose world-frame projection increases robot-pair separation"
        )
    if measured_change > 1.0e-6 and predicted_change >= -1.0e-6:
        raise ModelAgentError(
            "the rejected formation became too far apart, but this alignment sign pair is "
            f"predicted to change separation by {predicted_change:.4f}m; choose local-body "
            "delta signs whose world-frame projection decreases robot-pair separation"
        )


def _linear_transport_geometry_gate(
    runtime_state: dict[str, Any],
    geometry: dict[str, Any],
    *,
    repair_level: str,
    forbidden_objectives: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Select the linear method family when a body-frame axis points at the goal."""

    payload = (
        runtime_state.get("payload", {}) if isinstance(runtime_state, dict) else {}
    )
    payload = payload if isinstance(payload, dict) else {}
    supporters = set(payload.get("supporting_robots", ()))
    scores = geometry.get("candidate_goal_alignment", {})
    scores = scores if isinstance(scores, dict) else {}
    finite_scores: dict[str, float] = {}
    for direction, raw_score in scores.items():
        if str(direction) not in LOADED_TRANSLATION_DIRECTIONS:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if isfinite(score):
            finite_scores[str(direction)] = score
    best_direction = ""
    best_score = -1.0
    if finite_scores:
        best_direction, best_score = max(
            finite_scores.items(),
            key=lambda item: item[1],
        )
    try:
        payload_axis_alignment = float(
            geometry.get("payload_long_axis_goal_alignment_abs")
        )
    except (TypeError, ValueError):
        payload_axis_alignment = -1.0
    best_objective = (
        f"{COORDINATED_LINEAR_TRANSPORT}:{best_direction}" if best_direction else ""
    )
    best_objective_blocked = best_objective in set(forbidden_objectives)
    active = bool(
        repair_level in {"none", "strategy"}
        and supporters == {"front", "back"}
        and best_score >= 0.85
        and not best_objective_blocked
    )
    return {
        "active": active,
        "constraint": ("coordinated_linear_transport_only" if active else "none"),
        "basis": "current_trusted_state_geometry",
        "dual_payload_support": supporters == {"front", "back"},
        # The long payload may be carried perpendicular to the travel direction.
        # Keep this value as model-visible geometry, but do not use it to decide
        # whether a body-frame linear motion can reduce the active-goal error.
        "payload_long_axis_goal_alignment_abs": round(payload_axis_alignment, 4),
        "best_body_frame_direction": best_direction,
        "best_body_frame_goal_alignment": round(best_score, 4),
        "best_body_frame_objective": best_objective,
        "best_body_frame_objective_blocked": best_objective_blocked,
        "repair_level": repair_level,
    }


def _route_passage_alignment_gate(
    runtime_state: dict[str, Any],
    *,
    turn_episode: dict[str, Any],
    route_discovery: bool,
) -> dict[str, Any]:
    """Require one principal team-payload axis before loaded straight travel."""

    route_observation = runtime_state.get("trusted_route_observation", {})
    route_observation = route_observation if isinstance(route_observation, dict) else {}
    posture = str(
        route_observation.get(
            "team_payload_posture_relative_to_visible_passage",
            "",
        )
    ).strip()
    heading_relation = str(
        route_observation.get(
            "team_heading_relative_to_visible_free_space",
            "",
        )
    ).strip()
    episode_status = str(turn_episode.get("status", "")).strip()
    has_fresh_visual_posture = bool(posture)
    if has_fresh_visual_posture:
        aligned = posture in ALIGNED_PASSAGE_POSTURES
        evidence_source = "current_trusted_checkpoint_global_vlm_posture"
    elif route_observation:
        aligned = heading_relation == "aligned_with_open_passage"
        evidence_source = "legacy_current_global_vlm_heading"
    else:
        aligned = episode_status == PASSAGE_ALIGNED_STATUS
        evidence_source = "persisted_turn_episode"

    payload = runtime_state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    dual_support = set(payload.get("supporting_robots", ())) == {"front", "back"}
    released = bool(payload.get("target_contact", False)) and not payload.get(
        "supporting_robots", ()
    )
    active = bool(route_discovery and dual_support and not released)
    explicit_alignment_required = (
        episode_status == PASSAGE_ALIGNMENT_REQUIRED_STATUS
        or posture in {"diagonal_to_passage", "mixed", "uncertain"}
        or (
            not has_fresh_visual_posture
            and heading_relation
            in {
                "toward_wall_or_blocked_space",
                "crosswise_to_passage",
                "mixed",
                "uncertain",
            }
        )
    )
    requires_heading_change = bool(active and explicit_alignment_required)
    return {
        "active": active,
        "aligned": bool(active and aligned),
        "requires_heading_change": requires_heading_change,
        "accepted_postures": sorted(ALIGNED_PASSAGE_POSTURES),
        "observed_posture": posture or "legacy_or_unavailable",
        "observed_team_heading_relation": heading_relation or "unavailable",
        "evidence_source": evidence_source,
        "constraint": (
            "shared_heading_change_before_loaded_linear_transport"
            if requires_heading_change
            else "loaded_linear_transport_may_choose_body_frame_axis"
            if active and aligned
            else "inactive"
        ),
        "reason": (
            "The full team-payload formation is diagonal, mixed, or not yet "
            "visually aligned with a connected open passage."
            if requires_heading_change
            else "A principal team-payload axis is visually aligned with open passage."
            if active and aligned
            else "The loaded route-alignment gate is not active."
        ),
    }


def _host_wall_clearance_escape_directions(
    *,
    passage_alignment_gate: object,
    host_wall_clearance: object,
) -> tuple[str, ...]:
    """Return open translations only when every signed turn is wall-blocked."""

    gate = passage_alignment_gate if isinstance(passage_alignment_gate, dict) else {}
    if not bool(gate.get("requires_heading_change", False)):
        return ()
    if not all(
        candidate_is_blocked(
            host_wall_clearance,
            family="turn",
            objective=direction,
        )
        for direction in _SIGNED_TURN_DIRECTIONS
    ):
        return ()
    return _model_executable_loaded_translation_directions(host_wall_clearance)


def _active_branch_rewind_context(
    payload: dict[str, Any],
    runtime_state: dict[str, Any],
    geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a rewind lesson until one alternate geometry action is accepted."""

    raw = payload.get("branch_rewind_context", {})
    if not isinstance(raw, dict) or not bool(raw.get("active", False)):
        return {}
    scope_checkpoint_id = str(raw.get("scope_checkpoint_id", "")).strip()
    checkpoint = runtime_state.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    current_checkpoint_id = str(checkpoint.get("checkpoint_id", "")).strip()
    if not scope_checkpoint_id or current_checkpoint_id == scope_checkpoint_id:
        return dict(raw)
    if str(raw.get("persistence_mode", "")) == "until_route_objective_reoriented":
        scope_heading = raw.get("scope_team_heading_deg")
        current_heading = team_heading_deg(runtime_state)
        try:
            heading_change_deg = abs(
                wrap_degrees(float(current_heading) - float(scope_heading))
            )
        except (TypeError, ValueError):
            heading_change_deg = None
        if heading_change_deg is not None:
            if heading_change_deg < HEADING_ALIGNMENT_TOLERANCE_DEG:
                return dict(raw)
            return {}
        scope_goal = raw.get("scope_active_goal_xy_m", ())
        scope_goal = scope_goal if isinstance(scope_goal, (list, tuple)) else ()
        if len(scope_goal) != 2:
            return {}
        current_geometry = geometry
        if not isinstance(current_geometry, dict):
            current_geometry = _body_frame_goal_alignment(
                runtime_state,
                {
                    "active_stage_goal_x_m": scope_goal[0],
                    "active_stage_goal_y_m": scope_goal[1],
                },
            )
        objective = str(raw.get("blocked_motion_objective", "")).strip()
        method, separator, direction = objective.partition(":")
        scores = current_geometry.get("candidate_goal_alignment", {})
        scores = scores if isinstance(scores, dict) else {}
        try:
            alignment = float(scores[direction])
        except (KeyError, TypeError, ValueError):
            return {}
        if (
            separator
            and method == COORDINATED_LINEAR_TRANSPORT
            and alignment < 0.0
        ):
            return dict(raw)
        return {}
    if str(raw.get("persistence_mode", "")) == "until_route_turn_probe_success":
        turn_episode = runtime_state.get("turn_episode", {})
        turn_episode = turn_episode if isinstance(turn_episode, dict) else {}
        if not (
            str(turn_episode.get("purpose", "")) == "route_turn"
            and str(turn_episode.get("status", "")) == "completed"
            and not bool(turn_episode.get("active", False))
        ):
            return dict(raw)
    return {}


def _effective_candidate_repair_level(
    payload: dict[str, Any],
    *,
    requested_level: str,
) -> str:
    """Let an active route rewind change geometry despite a local retry decision."""

    if not bool(payload.get("route_discovery", False)):
        return requested_level
    observation = payload.get("observation", {})
    observation = observation if isinstance(observation, dict) else {}
    runtime_state = observation.get("runtime_state", {})
    runtime_state = runtime_state if isinstance(runtime_state, dict) else {}
    if _active_branch_rewind_context(payload, runtime_state):
        return "strategy"
    return requested_level


def _validated_linear_goal_direction(
    selected_direction: str,
    *,
    choices: list[str],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Reject body-frame guesses that contradict a strong current-goal projection."""

    scores = geometry.get("candidate_goal_alignment", {})
    if not isinstance(scores, dict) or selected_direction not in scores:
        return {}
    available_scores = {
        direction: float(scores[direction])
        for direction in choices
        if direction in scores
    }
    if not available_scores:
        return {}
    best_direction, best_score = max(
        available_scores.items(),
        key=lambda item: item[1],
    )
    selected_score = float(available_scores[selected_direction])
    all_scores = {
        direction: float(score)
        for direction, score in scores.items()
        if isinstance(score, (int, float)) and isfinite(float(score))
    }
    strongest_direction, strongest_score = max(
        all_scores.items(),
        key=lambda item: item[1],
        default=(best_direction, best_score),
    )
    if strongest_score >= 0.35 and selected_score <= 0.0:
        raise ModelAgentError(
            "selected body-frame direction contradicts the current trusted-state goal "
            "projection because it has no positive score: "
            f"{selected_direction}={selected_score:.3f}, while measured "
            f"{strongest_direction}={strongest_score:.3f}; do not dispatch a direction "
            "that is orthogonal to or away from the active goal"
        )
    if best_score >= 0.35 and selected_score < best_score - 0.25:
        raise ModelAgentError(
            "selected body-frame direction contradicts the current trusted-state goal "
            f"projection: {selected_direction}={selected_score:.3f}, while remaining "
            f"{best_direction}={best_score:.3f}; use measured geometry instead of enumerating directions"
        )
    return {
        "goal_alignment_score": selected_score,
        "best_available_goal_direction": best_direction,
    }


def _validated_turn_direction_evidence(
    payload: dict[str, Any],
    *,
    selected_direction: str,
    available_directions: tuple[str, ...] | None = None,
    heading_projection: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Require an auditable model comparison before a signed turn is dispatched."""

    direction_evidence = str(payload.get("direction_evidence", "")).strip()
    reported_alternative = str(payload.get("alternative_direction", "")).strip()
    alternative_rejection_reason = str(
        payload.get("alternative_rejection_reason", "")
    ).strip()
    if selected_direction not in _SIGNED_TURN_DIRECTIONS:
        raise ModelAgentError(
            f"turn direction must be one of {sorted(_SIGNED_TURN_DIRECTIONS)}"
        )
    expected_alternative = (
        "counterclockwise" if selected_direction == "clockwise" else "clockwise"
    )
    available = tuple(
        dict.fromkeys(
            direction
            for direction in (
                available_directions or tuple(sorted(_SIGNED_TURN_DIRECTIONS))
            )
            if direction in _SIGNED_TURN_DIRECTIONS
        )
    )
    if selected_direction not in available:
        raise ModelAgentError(
            f"selected turn direction {selected_direction!r} is not available in {list(available)}"
        )
    single_direction_remaining = available == (selected_direction,)
    # Once the model selects a signed direction, its opposite enum is
    # deterministic bookkeeping. Normalize schema typos without changing the
    # selected motion; evidence for both directions remains model-owned below.
    alternative_direction = expected_alternative
    direction_evidence_source = "model_direction_evidence"
    if not _is_specific_turn_direction_evidence(direction_evidence):
        reason = str(payload.get("reason", "")).strip()
        if single_direction_remaining and _is_specific_turn_direction_evidence(reason):
            # The model has already made the geometry argument in its required
            # rationale. When the verifier exposes one signed enum, duplicating
            # that text into another JSON field is not a physical safety gate.
            direction_evidence = reason
            direction_evidence_source = (
                "model_reason_single_checkpoint_allowed_direction"
            )
        else:
            raise ModelAgentError(
                "turn direction_evidence must cite current camera/heading/free-space geometry; "
                "a default or merely valid direction is not executable evidence"
            )
    if not single_direction_remaining and not _turn_evidence_cites_projection(
        direction_evidence,
        heading_projection=heading_projection,
    ):
        raise ModelAgentError(
            "turn direction_evidence must compare both host-projected candidates from "
            "turn_direction_decision_contract, including clockwise, counterclockwise, "
            "their screen headings, and the connected passage; screen direction words "
            "alone do not determine a robot-body turn sign"
        )
    alternative_rejection_source = "model_alternative_rejection_reason"
    if len(alternative_rejection_reason) < 16:
        if single_direction_remaining:
            alternative_rejection_reason = (
                f"{expected_alternative} was removed by checkpoint-local verifier "
                "exclusions, leaving one executable signed direction."
            )
            alternative_rejection_source = "checkpoint_local_verifier_exclusion"
        else:
            raise ModelAgentError(
                "alternative_rejection_reason must state the observed regression or clearance risk"
            )
    return {
        "direction_evidence": direction_evidence,
        "direction_evidence_source": direction_evidence_source,
        "alternative_direction": alternative_direction,
        "reported_alternative_direction": reported_alternative,
        "alternative_rejection_reason": alternative_rejection_reason,
        "alternative_rejection_source": alternative_rejection_source,
    }


def _is_specific_turn_direction_evidence(value: str) -> bool:
    """Return whether model text contains a non-generic signed-turn rationale."""

    normalized = value.lower()
    return bool(
        len(value) >= 24
        and not any(phrase in normalized for phrase in _GENERIC_TURN_EVIDENCE)
    )


def _turn_evidence_cites_projection(
    value: str,
    *,
    heading_projection: dict[str, Any] | None,
    destination_terms: tuple[str, ...] = (
        "passage",
        "corridor",
        "opening",
        "free space",
    ),
) -> bool:
    """Require the signed choice to use exact host-projected camera headings."""

    projection = heading_projection if isinstance(heading_projection, dict) else {}
    clockwise = projection.get("clockwise", {})
    counterclockwise = projection.get("counterclockwise", {})
    if not isinstance(clockwise, dict) or not isinstance(counterclockwise, dict):
        return True
    clockwise_heading = str(clockwise.get("screen_heading", "")).strip().lower()
    counterclockwise_heading = (
        str(counterclockwise.get("screen_heading", "")).strip().lower()
    )
    if not clockwise_heading or not counterclockwise_heading:
        return True
    normalized = value.lower().replace("_", "-")
    required = {
        "clockwise",
        "counterclockwise",
        clockwise_heading.replace("_", "-"),
        counterclockwise_heading.replace("_", "-"),
    }
    return (
        all(token in normalized for token in required)
        and any(token in normalized for token in destination_terms)
    )


def _turn_rationale_commits_selected_projection(
    value: str,
    *,
    selected_direction: str,
    heading_projection: dict[str, Any] | None,
) -> bool:
    """Bind the selected enum to its own projected heading in model prose."""

    if selected_direction not in _SIGNED_TURN_DIRECTIONS:
        return False
    alternative = (
        "counterclockwise" if selected_direction == "clockwise" else "clockwise"
    )
    projection = heading_projection if isinstance(heading_projection, dict) else {}
    selected = projection.get(selected_direction, {})
    rejected = projection.get(alternative, {})
    if not isinstance(selected, dict) or not isinstance(rejected, dict):
        return True
    selected_heading = str(selected.get("screen_heading", "")).strip().lower()
    rejected_heading = str(rejected.get("screen_heading", "")).strip().lower()
    if not selected_heading or not rejected_heading:
        return True
    normalized = " ".join(value.lower().replace("_", "-").split())
    return bool(
        f"selected {selected_direction} -> {selected_heading}" in normalized
        and f"rejected {alternative} -> {rejected_heading}" in normalized
    )


def _validate_selection_current_state_claims(
    *,
    selected_method: str,
    reason: str,
    runtime_state: dict[str, Any],
) -> None:
    """Reject repair choices justified by damage absent from the restored state."""

    payload = (
        runtime_state.get("payload", {}) if isinstance(runtime_state, dict) else {}
    )
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    supporters = set(
        payload.get("supporting_robots") or contacts.get("support_robots") or ()
    )
    normalized = " ".join(str(reason).lower().replace("-", " ").split())
    false_current_failure_claims = (
        "not supported by either",
        "not supported by both",
        "unsupported by both",
        "lost support from both",
        "support is lost",
        "payload has dropped",
        "payload is dropped",
        "box has dropped",
        "box is dropped",
        "no robot support",
        "no support from either",
        "neither robot is supporting",
        "robots are not supporting",
        "zero support force",
        "no support force",
        "support force = 0",
        "support force is 0",
    )
    contradictions: list[str] = []
    if supporters == {"front", "back"} and any(
        claim in normalized for claim in false_current_failure_claims
    ):
        contradictions.append("both robots currently support the payload")

    walls = payload.get("wall_contacts")
    if walls is None:
        walls = contacts.get("walls", ())
    false_current_wall_claims = (
        "touching a wall",
        "touching the wall",
        "wall contact",
        "colliding with a wall",
        "colliding with the wall",
        "in contact with a wall",
        "in contact with the wall",
    )
    explicit_no_wall_claims = (
        "no wall contact",
        "no floor or wall contact",
        "not touching a wall",
        "not touching the wall",
        "without wall contact",
        "without floor or wall contact",
        "without causing wall contact",
        "without causing payload tilt or wall contact",
        "free of wall contact",
        "free of floor or wall contact",
        "avoid wall contact",
        "avoiding wall contact",
        "avoiding floor or wall contact",
        "avoids wall contact",
        "prevent wall contact",
        "prevents wall contact",
        "risk of wall contact",
    )
    if (
        not walls
        and any(claim in normalized for claim in false_current_wall_claims)
        and not any(claim in normalized for claim in explicit_no_wall_claims)
    ):
        contradictions.append("the current payload has no wall contact")

    if not contradictions:
        return
    raise ModelAgentError(
        "selection rationale contradicts current_runtime_state: "
        + "; ".join(contradictions)
        + ". Treat rejected endpoint damage as historical evidence and choose "
        f"api.{selected_method} from the restored trusted checkpoint"
    )


def _trusted_state_authority_summary(runtime_state: dict[str, Any]) -> dict[str, Any]:
    """Expose a short, unambiguous present-state boundary to model roles."""

    runtime = runtime_state if isinstance(runtime_state, dict) else {}
    payload = runtime.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    supporters = sorted(str(item) for item in payload.get("supporting_robots", ()))
    recovery = runtime.get("payload_inward_recovery", {})
    recovery = recovery if isinstance(recovery, dict) else {}
    bilateral_support = set(supporters) == {"front", "back"}
    floor_contact = bool(payload.get("floor_contact", False))
    wall_contacts = list(payload.get("wall_contacts", ()))
    payload_tilt_deg = _safe_float(payload.get("tilt_deg"))
    payload_top_face_up = bool(
        payload_tilt_deg is not None
        and payload_tilt_deg < PAYLOAD_TOP_FACE_UP_LIMIT_DEG
    )
    robots = runtime.get("robots", {})
    robots = robots if isinstance(robots, dict) else {}
    robot_xy: dict[str, tuple[float, float]] = {}
    robot_yaw: dict[str, float] = {}
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        robot = robot if isinstance(robot, dict) else {}
        x = _safe_float(robot.get("x"))
        y = _safe_float(robot.get("y"))
        yaw = _safe_float(robot.get("yaw_deg"))
        if x is not None and y is not None:
            robot_xy[robot_id] = (x, y)
        if yaw is not None:
            robot_yaw[robot_id] = yaw
    pair_separation_m = (
        hypot(
            robot_xy["front"][0] - robot_xy["back"][0],
            robot_xy["front"][1] - robot_xy["back"][1],
        )
        if set(robot_xy) == {"front", "back"}
        else None
    )
    heading_disagreement_deg = (
        abs(wrap_degrees(robot_yaw["front"] - robot_yaw["back"]))
        if set(robot_yaw) == {"front", "back"}
        else None
    )
    return {
        "temporal_scope": "current restored trusted checkpoint only",
        "payload_active": bool(payload.get("active", False)),
        "supporting_robots": supporters,
        "bilateral_payload_support": bilateral_support,
        "payload_floor_contact": floor_contact,
        "payload_wall_contacts": wall_contacts,
        "payload_top_face_up": payload_top_face_up,
        "carry_posture_physically_valid": bool(
            bilateral_support and not floor_contact and not wall_contacts
        ),
        "payload_orientation_interpretation": (
            "payload.tilt_deg measures the box-top normal from world up. "
            "Tilt, including a top-face crossing at 90 degrees, is carry-quality "
            "evidence in the task profile. It is not a drop while legal bilateral "
            "upper-limb support remains, the payload stays off the floor, and both "
            "robots remain upright"
        ),
        "payload_inward_recovery_available": bool(recovery.get("available", False)),
        "formation_measurements": {
            "robot_pair_xy_separation_m": pair_separation_m,
            "front_yaw_deg": robot_yaw.get("front"),
            "back_yaw_deg": robot_yaw.get("back"),
            "heading_disagreement_deg": heading_disagreement_deg,
            "payload_support_center_error_m": _safe_float(
                payload.get("support_center_xy_error_m")
            ),
            "payload_longitudinal_offset_m": _safe_float(
                payload.get("support_center_signed_longitudinal_offset_m")
            ),
            "payload_lateral_offset_m": _safe_float(
                payload.get("support_center_signed_lateral_offset_m")
            ),
        },
        "formation_guidance_policy": (
            "Formation measurements are current physical facts. Any repair need, "
            "primitive, direction, temporary roles, and coordination mode remain "
            "model decisions unless a lifecycle or explicit collision gate applies."
        ),
        "instruction": (
            "Historical rejected rollout damage is not present state. Plan from "
            "these current checkpoint facts."
        ),
    }


def _trusted_route_blocks_motion_objective(
    *,
    selected_method: str,
    motion_objective: str,
    trusted_route_observation: object,
) -> bool:
    """Return whether current trusted geometry blocks one semantic objective."""

    route = (
        trusted_route_observation if isinstance(trusted_route_observation, dict) else {}
    )
    return bool(
        selected_method == COORDINATED_LINEAR_TRANSPORT
        and motion_objective == "forward"
        and str(route.get("team_heading_relative_to_visible_free_space", ""))
        == "toward_wall_or_blocked_space"
    )


def _trusted_route_blocks_signed_turn(
    *,
    motion_objective: str,
    trusted_route_observation: object,
    host_wall_clearance: object = None,
) -> bool:
    """Filter a visually blocked turn unless exact wall geometry overrides it."""

    if motion_objective not in _SIGNED_TURN_DIRECTIONS:
        return False
    route = (
        trusted_route_observation if isinstance(trusted_route_observation, dict) else {}
    )
    alternative = "counterclockwise" if motion_objective == "clockwise" else "clockwise"
    selected_relation = str(
        route.get(
            f"{motion_objective}_candidate_relative_to_connected_passage",
            "uncertain",
        )
    ).strip()
    alternative_relation = str(
        route.get(
            f"{alternative}_candidate_relative_to_connected_passage",
            "uncertain",
        )
    ).strip()
    clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    turn_candidates = clearance.get("turn_candidates", {})
    turn_candidates = turn_candidates if isinstance(turn_candidates, dict) else {}
    selected_clearance = turn_candidates.get(motion_objective, {})
    selected_clearance = (
        selected_clearance if isinstance(selected_clearance, dict) else {}
    )
    alternative_clearance = turn_candidates.get(alternative, {})
    alternative_clearance = (
        alternative_clearance if isinstance(alternative_clearance, dict) else {}
    )
    host_geometry_resolves_visual_conflict = bool(
        clearance.get("available", False)
        and selected_clearance.get("relation") in {"open", "constrained", "limited"}
        and alternative_clearance.get("relation") == "blocked"
    )
    return bool(
        selected_relation == "toward_blocked_space"
        and alternative_relation in {"aligned", "toward_open_space"}
        and not host_geometry_resolves_visual_conflict
    )


def _validate_selection_against_trusted_route(
    *,
    selected_method: str,
    motion_objective: str,
    trusted_route_observation: object,
    host_wall_clearance: object = None,
) -> None:
    """Prevent dispatch if a stale candidate bypasses trusted route filtering."""

    if _trusted_route_blocks_motion_objective(
        selected_method=selected_method,
        motion_objective=motion_objective,
        trusted_route_observation=trusted_route_observation,
    ):
        raise ModelAgentError(
            "current_trusted_route_observation says body-forward points toward a wall "
            "or blocked region; coordinated_linear_transport(direction='forward') "
            "cannot be physically dispatched from this checkpoint. Select a remaining "
            "lateral direction or a bounded turn/alignment strategy from current geometry."
        )
    open_linear = _model_executable_loaded_translation_directions(
        host_wall_clearance
    )
    if (
        selected_method == COORDINATED_LINEAR_TRANSPORT
        and not _loaded_translation_has_supported_tier(
            host_wall_clearance,
            motion_objective,
        )
    ):
        raise ModelAgentError(
            "predictive authored-wall clearance leaves no model-supported "
            f"distance tier for body-frame {motion_objective}"
            + (
                f" while these body-frame translations remain open: {list(open_linear)}"
                if open_linear
                else ""
            )
            + ". Choose from current physics geometry; do not follow a contradictory "
            "visual free-space label."
        )

    route = (
        trusted_route_observation if isinstance(trusted_route_observation, dict) else {}
    )
    if selected_method in _SIGNED_TURN_METHODS:
        if (
            candidate_is_blocked(
                host_wall_clearance,
                family="turn",
                objective=motion_objective,
            )
            and open_linear
        ):
            raise ModelAgentError(
                "authoritative authored-wall clearance marks the selected "
                f"{motion_objective} finite coordinated-pivot sweep as blocked before "
                "the minimum executable angle, "
                f"while body-frame translations {list(open_linear)} remain open. "
                "Do not pivot toward a wall merely because the VLM arrow label says open."
            )
        alternative = (
            "counterclockwise" if motion_objective == "clockwise" else "clockwise"
        )
        alternative_relation = str(
            route.get(
                f"{alternative}_candidate_relative_to_connected_passage",
                "uncertain",
            )
        ).strip()
        if _trusted_route_blocks_signed_turn(
            motion_objective=motion_objective,
            trusted_route_observation=route,
            host_wall_clearance=host_wall_clearance,
        ):
            raise ModelAgentError(
                f"current physics-arrow visual evidence marks {motion_objective} as "
                f"toward_blocked_space while {alternative} is {alternative_relation}; "
                "select the open signed candidate or provide a newly observed route state"
            )


_SELECTION_TRANSACTION_METRICS: tuple[str, ...] = ()


def _selection_historical_rollout_summary(evaluation: Any) -> dict[str, Any]:
    """Keep transaction deltas without presenting a rejected endpoint as live state."""

    if not isinstance(evaluation, dict):
        return {}
    metrics = evaluation.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    evidence = evaluation.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "semantics": (
            "historical verifier result; endpoint pose, contacts, tilt, and support were "
            "rolled back and must not be described as current"
        ),
        "success": bool(evaluation.get("success", False)),
        "failure_type": str(evaluation.get("failure_type", "")),
        "failure_level": str(evaluation.get("failure_level", "")),
        "diagnosis": str(evaluation.get("diagnosis", "")),
        "allowed_repairs": list(evaluation.get("allowed_repairs", ())),
        "forbidden_repairs": list(evaluation.get("forbidden_repairs", ())),
        "failed_subtask": str(evaluation.get("failed_subtask", "")),
        "transaction_deltas": {
            key: metrics[key]
            for key in _SELECTION_TRANSACTION_METRICS
            if key in metrics
        },
        "route_direction_verification": evidence.get(
            "route_direction_verification", {}
        ),
    }


def _selection_current_execution_state(runtime_state: Any) -> dict[str, Any]:
    """Put authoritative checkpoint facts ahead of historical failure evidence."""

    state = runtime_state if isinstance(runtime_state, dict) else {}
    payload = state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    checkpoint = state.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    robots = state.get("robots", {})
    robots = robots if isinstance(robots, dict) else {}
    supporters = payload.get("supporting_robots")
    if supporters is None:
        supporters = contacts.get("support_robots", ())
    walls = payload.get("wall_contacts")
    if walls is None:
        walls = contacts.get("walls", ())
    floor_contact = payload.get("floor_contact")
    if floor_contact is None:
        floor_contact = contacts.get("floor", False)
    return {
        "authoritative_for_next_action": True,
        "coordinate_convention": _route_neutral_coordinate_convention(),
        "robot_identity_contract": _immutable_robot_identity_contract(),
        "checkpoint_id": checkpoint.get("checkpoint_id", ""),
        "control_healthy": bool(state.get("control_healthy", False)),
        "fixed_global_camera_heading_projection": (
            _fixed_global_camera_heading_projection(state)
        ),
        "payload": {
            "active": bool(payload.get("active", False)),
            "x": payload.get("x"),
            "y": payload.get("y"),
            "z": payload.get("z"),
            "yaw_deg": payload.get("yaw_deg"),
            "heading_vector_world_xy": _heading_vector_world_xy(
                payload.get("yaw_deg")
            ),
            "tilt_deg": payload.get("tilt_deg"),
            "top_surface_corner_z_range_m": payload.get("top_surface_corner_z_range_m"),
            "support_polygon_inside": bool(
                payload.get("support_polygon", {}).get("inside", False)
                if isinstance(payload.get("support_polygon"), dict)
                else False
            ),
            "supporting_robots": list(supporters or ()),
            "wall_contacts": list(walls or ()),
            "floor_contact": bool(floor_contact),
        },
        "robots": {
            robot_id: {
                **{
                    key: robot.get(key)
                    for key in ("x", "y", "yaw_deg", "pelvis_height_m")
                    if key in robot
                },
                "heading_vector_world_xy": _heading_vector_world_xy(
                    robot.get("yaw_deg")
                ),
            }
            for robot_id in ("front", "back")
            if isinstance((robot := robots.get(robot_id)), dict)
        },
    }


def _route_neutral_coordinate_convention() -> dict[str, Any]:
    """Describe frames without revealing which route action should be selected."""

    return {
        "world_axes": {
            "+X": "initial authored robot-forward direction",
            "+Y": "initial authored robot-left direction",
            "+Z": "up",
        },
        "yaw": {
            "zero": "heading world +X",
            "positive": "counterclockwise around world +Z when viewed top-down",
            "negative": "clockwise around world +Z when viewed top-down",
        },
        "fixed_global_camera": {
            "screen_right": "world -X",
            "screen_down": "world +Y",
            "view_direction": "toward world -Z",
        },
        "body_translation": {
            "forward": "along each robot's current heading vector",
            "backward": "opposite each robot's current heading vector",
            "left": "90 degrees counterclockwise from each robot's current heading",
            "right": "90 degrees clockwise from each robot's current heading",
        },
        "route_neutral": True,
    }


def _immutable_robot_identity_contract() -> dict[str, str]:
    return {
        "front": "immutable simulator robot id front; never relabel by screen position",
        "back": "immutable simulator robot id back; never relabel by screen position",
    }


def _heading_vector_world_xy(yaw_deg: object) -> list[float] | None:
    try:
        yaw_rad = radians(float(yaw_deg))
    except (TypeError, ValueError):
        return None
    if not isfinite(yaw_rad):
        return None
    return [round(cos(yaw_rad), 6), round(sin(yaw_rad), 6)]


def _fixed_global_camera_heading_projection(
    runtime_state: dict[str, Any],
    *,
    candidate_angle_deg: float = MODEL_ROUTE_TURN_ANGLE_DEG,
) -> dict[str, Any]:
    """Project exact robot yaw and both signed turn candidates into screen axes."""

    robots = runtime_state.get("robots", {})
    if not isinstance(robots, dict):
        return {}
    yaw_samples: list[float] = []
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            continue
        raw_yaw = robot.get("yaw_deg")
        if raw_yaw is None and robot.get("yaw_rad") is not None:
            try:
                raw_yaw = degrees(float(robot["yaw_rad"]))
            except (TypeError, ValueError):
                raw_yaw = None
        try:
            yaw_deg = float(raw_yaw)
        except (TypeError, ValueError):
            continue
        if isfinite(yaw_deg):
            yaw_samples.append(radians(yaw_deg))
    if not yaw_samples:
        return {}
    mean_yaw_rad = atan2(
        sum(sin(value) for value in yaw_samples),
        sum(cos(value) for value in yaw_samples),
    )
    angle_deg = abs(float(candidate_angle_deg))
    if not isfinite(angle_deg) or angle_deg <= 0.0:
        angle_deg = MODEL_ROUTE_TURN_ANGLE_DEG

    def projected(yaw_rad: float) -> dict[str, Any]:
        screen_x = -cos(yaw_rad)
        screen_y = sin(yaw_rad)
        return {
            "world_yaw_deg": round(_wrap_degrees(degrees(yaw_rad)), 3),
            "screen_heading": _screen_heading_label(screen_x, screen_y),
            "screen_vector_xy": [round(screen_x, 4), round(screen_y, 4)],
        }

    return {
        "source": "fresh MuJoCo robot-root yaw projected through fixed global camera",
        "world_yaw_convention": (
            "+yaw=counterclockwise, -yaw=clockwise, zero=world +X"
        ),
        "screen_axis_calibration": {
            "screen_right": "world -X",
            "screen_down": "world +Y",
        },
        "candidate_angle_deg": angle_deg,
        "current": projected(mean_yaw_rad),
        "clockwise": projected(mean_yaw_rad - radians(angle_deg)),
        "counterclockwise": projected(mean_yaw_rad + radians(angle_deg)),
        "body_frame_translation_candidates": {
            "forward": projected(mean_yaw_rad),
            "backward": projected(mean_yaw_rad + radians(180.0)),
            "left": projected(mean_yaw_rad + radians(90.0)),
            "right": projected(mean_yaw_rad - radians(90.0)),
        },
        "contract": (
            "These are exact kinematic projections, not a route recommendation. "
            "Compare both turn candidates or all four body-frame translation "
            "candidates with the connected passage seen in the current global frame; "
            "never infer a turn sign or translation objective from screen left/right "
            "words alone."
        ),
    }


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _screen_heading_label(screen_x: float, screen_y: float) -> str:
    """Return one of eight fixed-camera directions for a projected heading."""

    angle = (degrees(atan2(screen_y, screen_x)) + 360.0) % 360.0
    labels = (
        "right",
        "down-right",
        "down",
        "down-left",
        "left",
        "up-left",
        "up",
        "up-right",
    )
    return labels[int((angle + 22.5) // 45.0) % len(labels)]


def _load_failed_strategy_history(
    output_dir: Path,
    *,
    checkpoint_id: str,
    limit: int,
    minimum_attempt_index: int = 0,
) -> tuple[tuple[str, PolicyScript], ...]:
    """Recover physically executed failures for the per-checkpoint budget."""

    candidates: list[tuple[int, str, PolicyScript]] = []
    for attempt_dir in output_dir.glob("attempt_*"):
        try:
            attempt_index = int(attempt_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if attempt_index < minimum_attempt_index:
            continue
        evaluation = _read_attempt_evaluation(attempt_dir)
        if not evaluation or bool(evaluation.get("success", False)):
            continue
        classification = classify_failure(
            str(evaluation.get("failure_type", "")),
            metrics=(
                evaluation.get("metrics", {})
                if isinstance(evaluation.get("metrics"), dict)
                else {}
            ),
            evidence=(
                evaluation.get("evidence", {})
                if isinstance(evaluation.get("evidence"), dict)
                else {}
            ),
        )
        if classification.level == "system":
            continue
        recorded_checkpoint = _attempt_checkpoint_before(attempt_dir, evaluation)
        if checkpoint_id and recorded_checkpoint != checkpoint_id:
            continue
        trace = _read_json_object(attempt_dir / "execution_trace.json")
        if not _trace_payload_has_physical_dispatch(trace):
            continue
        policy_payload = _read_json_object(attempt_dir / "policy.json")
        if not policy_payload:
            policy_payload = _read_json_object(attempt_dir / "team_plan.json")
        policy = _policy_from_history_payload(policy_payload)
        if policy is not None:
            candidates.append((attempt_index, attempt_dir.name, policy))
    candidates.sort(key=lambda item: item[0])
    return tuple(
        (evidence_key, policy)
        for _index, evidence_key, policy in candidates[-max(1, int(limit)) :]
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_attempt_evaluation(attempt_dir: Path) -> dict[str, Any]:
    """Read the newest verifier audit before the original rollout verdict."""

    for name in (
        "corrected_evaluation.json",
        "evaluation.json",
        "preliminary_physics_evaluation.json",
    ):
        payload = _read_json_object(Path(attempt_dir) / name)
        if payload:
            return payload
    return {}


def _trace_payload_has_physical_dispatch(payload: dict[str, Any]) -> bool:
    calls = payload.get("skill_calls", ()) if isinstance(payload, dict) else ()
    if not isinstance(calls, list):
        return False
    for call in calls:
        if not isinstance(call, dict):
            continue
        status = str(call.get("status", "")).strip().lower()
        if status and status not in {"dry_run", "dry_run_validated", "planned"}:
            return True
    return False


def _policy_from_history_payload(payload: dict[str, Any]) -> PolicyScript | None:
    if not payload:
        return None
    scripts: list[RobotPolicyScript] = []
    for item in _as_list(payload.get("robot_scripts")):
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip()
        robot_id = str(item.get("robot_id", "")).strip()
        if not code or robot_id not in {"front", "back"}:
            continue
        scripts.append(
            RobotPolicyScript(
                robot_id=robot_id,
                script_id=str(item.get("script_id", "")),
                code=code,
                synchronization_group=str(item.get("synchronization_group", "")),
                phase_id=str(item.get("phase_id", "")),
                contact_intent=str(item.get("contact_intent", "unspecified")),
                coordination_mode=_validate_coordination_mode(
                    str(item.get("coordination_mode", "sync"))
                ),
            )
        )
    if {item.robot_id for item in scripts} != {"front", "back"}:
        return None
    return PolicyScript(
        script_id=str(payload.get("script_id", "historical_rejected_policy")),
        attempt=int(payload.get("attempt", 1)),
        code=str(payload.get("code", scripts[0].code)),
        robot_scripts=tuple(scripts),
        rationale=str(payload.get("rationale", "")),
        verification_contract=dict(payload.get("verification_contract", {})),
    )


def _parse_model_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _decode_first_json_object(text)
    if not isinstance(payload, dict):
        raise ModelAgentError("model response must be a JSON object")
    return payload


def _latest_observation_frame(
    observation: ObservationBundle | None,
) -> Path | None:
    if observation is None:
        return None
    for visual in reversed(observation.visual_observations):
        path = Path(str(visual.frame_path))
        if visual.frame_path and path.exists():
            return path
    return None


def _failure_analysis_from_report(
    report: EvaluationReport,
) -> dict[str, Any]:
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    visual = evidence.get("visual_verifier", {})
    visual = visual if isinstance(visual, dict) else {}
    visual_evidence = visual.get("evidence", {})
    visual_evidence = visual_evidence if isinstance(visual_evidence, dict) else {}
    analysis = visual_evidence.get("failure_analysis", {})
    analysis_outcome = (
        str(analysis.get("outcome", "")).strip().casefold()
        if isinstance(analysis, dict)
        else ""
    )
    expected_outcome = "accepted" if report.success else "rejected"
    if (
        isinstance(analysis, dict)
        and analysis
        and analysis_outcome in {"", expected_outcome}
    ):
        return dict(analysis)
    if _checkpoint_roundtrip_geometry_repair_failed(report):
        return {
            "outcome": "rejected",
            "primary_failure": "checkpoint-round-trip geometry repair had no durable effect",
            "causal_chain": [
                "the non-linear action was physically safe before checkpoint commit",
                "planner reset removed the intended alignment effect",
                "the same alignment axis/role structure cannot repair this checkpoint",
            ],
            "physical_evidence": [
                f"physics verifier failure_type={report.failure_type}",
                str(report.diagnosis),
            ],
            "visual_evidence": [],
            "strategy_defect": True,
            "coordination_defect": False,
            "magnitude_defect": False,
            "reusable_context": [
                "preserve the trusted parent checkpoint and bilateral support",
                "change alignment axis or select a different coordination primitive",
            ],
            "confidence": 1.0,
            "source": "physics_authoritative_checkpoint_roundtrip_semantic_override",
        }
    classification = classify_report(report)
    metrics = report.metrics if isinstance(report.metrics, dict) else {}
    try:
        step_vector_desync_m = float(
            metrics.get("front_back_step_vector_desync_m", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        step_vector_desync_m = 0.0
    coordination_defect = bool(
        classification.level == "coordination"
        or (
            not report.success
            and report.failure_type
            in {
                "payload_drop",
                "payload_not_supported",
                "payload_slip",
                "formation_desync",
            }
            and step_vector_desync_m >= TASK_CARRY_SAFETY.warning_robot_step_desync_m
        )
    )
    causal_chain = [classification.diagnosis]
    physical_evidence = [
        f"physics verifier failure_type={report.failure_type}",
        f"host failure_level={classification.level}",
    ]
    if coordination_defect and step_vector_desync_m > 0.0:
        causal_chain.append(
            "front/back realized displacement diverged before payload support was lost"
        )
        physical_evidence.append(
            f"front_back_step_vector_desync_m={step_vector_desync_m:.3f}"
        )
    return {
        "outcome": "accepted" if report.success else "rejected",
        "primary_failure": report.failure_type,
        "causal_chain": causal_chain,
        "physical_evidence": physical_evidence,
        "visual_evidence": [],
        "strategy_defect": classification.level == "strategy",
        "coordination_defect": coordination_defect,
        "magnitude_defect": bool(
            classification.level == "execution" and not coordination_defect
        ),
        "reusable_context": [],
        "confidence": 1.0,
        "source": "deterministic_physics_fallback",
    }


def _checkpoint_roundtrip_geometry_repair_failed(
    report: EvaluationReport,
) -> bool:
    """Identify a safe non-linear action whose semantic effect did not persist."""

    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    semantic_report = evidence.get("checkpoint_roundtrip_semantic_evaluation", {})
    semantic_report = semantic_report if isinstance(semantic_report, dict) else {}
    allowed_repairs = set(report.allowed_repairs) | {
        str(item) for item in semantic_report.get("allowed_repairs", ())
    }
    forbidden_repairs = set(report.forbidden_repairs) | {
        str(item) for item in semantic_report.get("forbidden_repairs", ())
    }
    semantic_failure_type = str(
        semantic_report.get("failure_type", report.failure_type)
    ).strip()
    return bool(
        not report.success
        and semantic_failure_type in {"no_payload_progress", "goal_regression"}
        and "change_alignment_delta_only" in forbidden_repairs
        and (
            "rewrite_alignment_axis_or_moving_reference_relationship" in allowed_repairs
        )
    )


def _rejected_route_obstruction_evidence(
    report: EvaluationReport,
) -> dict[str, Any]:
    """Extract endpoint wall evidence even when an earlier monitor owns the label."""

    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    payload = evidence.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    walls = contacts.get("walls")
    if walls is None:
        walls = payload.get("wall_contacts")
    if walls is None:
        walls = evidence.get("wall_contacts", ())
    wall_contacts = tuple(str(item) for item in (walls or ()))

    supporters = payload.get("supporting_robots")
    if supporters is None:
        supporters = contacts.get("support_robots", ())
    supporting_robots = tuple(sorted(str(item) for item in (supporters or ())))
    single_sided_support = bool(len(set(supporting_robots) & {"front", "back"}) == 1)

    visual = evidence.get("visual_verifier", {})
    visual = visual if isinstance(visual, dict) else {}
    visual_evidence = visual.get("evidence", visual)
    visual_evidence = visual_evidence if isinstance(visual_evidence, dict) else {}
    visual_wall = bool(
        visual_evidence.get("wall_collision_or_stuck", False)
        or visual_evidence.get("payload_wall_contact_or_high_risk", False)
    )
    direct_wall_failure = report.failure_type == "payload_wall_contact"
    route_obstruction = bool(
        wall_contacts
        or direct_wall_failure
        or (
            visual_wall
            and (
                single_sided_support
                or report.failure_type
                in {
                    "formation_desync",
                    "payload_not_supported",
                    "payload_slip",
                }
            )
        )
    )
    if not route_obstruction:
        return {"active": False}
    return {
        "active": True,
        "source": "rejected_endpoint_physics_and_global_view",
        "wall_contacts": list(wall_contacts),
        "visual_wall_collision_or_stuck": visual_wall,
        "supporting_robots": list(supporting_robots),
        "single_sided_support": single_sided_support,
        "reported_failure_type": str(report.failure_type),
        "interpretation": (
            "The rejected rollout entered blocked space. Formation divergence remains "
            "causal evidence, but another same-direction linear probe is not a valid "
            "route repair."
        ),
    }


def _policy_motion_objective(policy: PolicyScript) -> tuple[str, str]:
    method = ""
    arguments: dict[str, Any] = {}
    scripts = policy.robot_scripts or ()
    if scripts:
        method = _first_policy_api_method(scripts[0].code)
        arguments = _first_task_call_literal_arguments(scripts[0].code)
    if not method:
        method = _first_policy_api_method(policy.code)
        arguments = _first_task_call_literal_arguments(policy.code)
    parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(method, "")
    objective = (
        str(arguments.get(parameter, "")).strip()
        if parameter
        else _STATIC_MOTION_OBJECTIVES.get(method, "")
    )
    return method, objective


def _route_recovery_contract(
    policy: PolicyScript,
    report: EvaluationReport,
    *,
    host_wall_clearance: dict[str, Any],
    expose_clearance_candidates: bool = True,
    scope_checkpoint_id: str = "",
) -> dict[str, Any]:
    obstruction = _rejected_route_obstruction_evidence(report)
    target_bearing_coupling = _target_bearing_coupled_turn_recovery(
        policy,
        report,
    )
    continued_recovery = _continued_route_recovery_after_execution_failure(
        policy,
        report,
    )
    if not obstruction.get("active") and not target_bearing_coupling.get(
        "active",
        False,
    ) and not continued_recovery.get("active", False):
        return {}
    method, objective = _policy_motion_objective(policy)
    clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    candidates = clearance.get("translation_candidates", {})
    candidates = candidates if isinstance(candidates, dict) else {}
    nonblocked = list(
        _model_executable_loaded_translation_directions(clearance)
    )
    if target_bearing_coupling.get("active", False):
        contract = {
            "active": True,
            "source": "target_bearing_coupled_turn_recovery",
            "scope_checkpoint_id": str(
                target_bearing_coupling.get("scope_checkpoint_id", "")
            ).strip(),
            "rejected_motion": {
                "primitive": method,
                "semantic_objective": objective,
            },
            "target_bearing_coupling": target_bearing_coupling,
            "repair_rule": (
                "The requested shared-heading turn executed with the correct sign, "
                "but its arc translated the payload so the world-point goal bearing "
                "rotated with the team. The model may choose one bounded, wall-clear "
                "loaded translation to change that local geometry, then recompute "
                "heading from the next trusted checkpoint."
            ),
            "direction_authority": (
                "The model selects the body-frame translation from the fresh global "
                "view and current clearance. The host does not prescribe a scene "
                "direction; this contract only prevents a correct-but-ineffective "
                "pivot from deadlocking all translations."
            ),
        }
        if expose_clearance_candidates:
            contract.update(
                {
                    "required_mutation_level": "strategy_mutation",
                    "current_trusted_checkpoint_clearance": clearance,
                    "locally_nonblocked_body_frame_translations": nonblocked,
                    "open_body_frame_translations": list(
                        open_translation_directions(clearance)
                    ),
                    "candidate_measurements": candidates,
                }
            )
        return contract

    if continued_recovery.get("active", False):
        contract = dict(continued_recovery)
        if expose_clearance_candidates:
            contract.update(
                {
                    "required_mutation_level": "parameter_tuning",
                    "current_trusted_checkpoint_clearance": clearance,
                    "locally_nonblocked_body_frame_translations": nonblocked,
                    "open_body_frame_translations": list(
                        open_translation_directions(clearance)
                    ),
                    "candidate_measurements": candidates,
                }
            )
        return contract

    contract = {
        "active": True,
        "source": "route_obstruction_recovery",
        "scope_checkpoint_id": str(scope_checkpoint_id).strip(),
        "rejected_motion": {
            "primitive": method,
            "semantic_objective": objective,
        },
        "rejected_endpoint_obstruction": obstruction,
        "repair_rule": (
            "Plan from the restored trusted checkpoint. Change the executable "
            "maneuver family or semantic direction, and do not repeat the rejected "
            "motion into the obstruction."
        ),
        "loaded_backward_contract": (
            "If backward is selected as the escape direction, both robots must call "
            "coordinated_linear_transport(direction='backward') with identical "
            "parameters in one sync transaction."
        ),
        "direction_authority": (
            "The model selects the recovery primitive and direction from the fresh "
            "global view. This contract authorizes one route-neutral recovery but "
            "does not choose backward or any other motion."
        ),
    }
    if expose_clearance_candidates:
        contract.update(
            {
                "required_mutation_level": "strategy_mutation",
                "current_trusted_checkpoint_clearance": clearance,
                "locally_nonblocked_body_frame_translations": nonblocked,
                "open_body_frame_translations": list(
                    open_translation_directions(clearance)
                ),
                "candidate_measurements": candidates,
            }
        )
    return contract


def _continued_route_recovery_after_execution_failure(
    policy: PolicyScript,
    report: EvaluationReport,
) -> dict[str, Any]:
    """Keep checkpoint-local geometry recovery through execution repair."""

    verification = (
        policy.verification_contract
        if isinstance(policy.verification_contract, dict)
        else {}
    )
    inherited = verification.get("route_recovery", {})
    inherited = inherited if isinstance(inherited, dict) else {}
    method, objective = _policy_motion_objective(policy)
    if (
        not inherited.get("active", False)
        or method != COORDINATED_LINEAR_TRANSPORT
        or report.success
        or report.failure_type
        not in {
            "formation_desync",
            "payload_drop",
            "payload_not_supported",
            "payload_slip",
            "payload_unstable",
            "robot_fall",
            "timeout",
        }
    ):
        return {}
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    outcome = evidence.get("checkpoint_outcome", {})
    outcome = outcome if isinstance(outcome, dict) else {}
    if not bool(outcome.get("physically_dispatched", False)):
        return {}
    local = verification.get(LOCAL_NAVIGATION_CONTRACT_KEY, {})
    local = local if isinstance(local, dict) else {}
    scope_checkpoint_id = str(
        inherited.get("scope_checkpoint_id", "")
        or local.get("scope_checkpoint_id", "")
    ).strip()
    if not scope_checkpoint_id:
        return {}
    return {
        **inherited,
        "active": True,
        "source": str(inherited.get("source", "")).strip()
        or "continued_checkpoint_scoped_route_recovery",
        "scope_checkpoint_id": scope_checkpoint_id,
        "recovery_attempt_failure": {
            "primitive": method,
            "semantic_objective": objective,
            "failure_type": str(report.failure_type),
        },
        "repair_rule": (
            "The checkpoint-local geometry recovery remains necessary. Repair the "
            "execution or coordination defect without losing that authorization; "
            "it expires after an accepted child checkpoint or new route-obstruction "
            "evidence."
        ),
        "direction_authority": (
            "The model still selects a wall-clear body-frame translation. Physical "
            "safety and wall-clearance checks remain authoritative."
        ),
    }


def _target_bearing_coupled_turn_recovery(
    policy: PolicyScript,
    report: EvaluationReport,
) -> dict[str, Any]:
    """Detect a correct signed pivot whose arc rotates the goal bearing too."""

    method, objective = _policy_motion_objective(policy)
    if (
        report.success
        or report.failure_type != "route_regression"
        or method not in TURN_METHODS
        or objective not in _SIGNED_TURN_DIRECTIONS
    ):
        return {}
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    local = evidence.get("local_navigation_contract", {})
    local = local if isinstance(local, dict) else {}
    transition = evidence.get("local_navigation_transition", {})
    transition = transition if isinstance(transition, dict) else {}
    if not (
        local.get("active", False)
        and str(local.get("reference_geometry", "")).strip() == "world_point"
        and bool(transition.get("available", False))
        and bool(transition.get("requested_turn_matches_local_error", False))
        and bool(transition.get("measured_heading_motion_matches_request", False))
        and bool(transition.get("meaningful_team_heading_motion", False))
        and not bool(transition.get("semantic_heading_improved", False))
    ):
        return {}
    try:
        wall_contacts = int(float(report.metrics.get("wall_contact_count", 0)))
        heading_change = float(transition.get("team_heading_change_deg", 0.0))
        before_error = float(
            transition.get("signed_heading_error_before_deg", 0.0)
        )
        after_error = float(
            transition.get("signed_heading_error_after_deg", 0.0)
        )
    except (TypeError, ValueError):
        return {}
    if wall_contacts > 0:
        return {}
    inferred_goal_bearing_change = heading_change + after_error - before_error
    return {
        "active": True,
        "reference": str(local.get("reference", "")).strip(),
        "scope_checkpoint_id": str(local.get("scope_checkpoint_id", "")).strip(),
        "requested_turn_direction": objective,
        "team_heading_change_deg": round(heading_change, 6),
        "heading_error_before_deg": round(before_error, 6),
        "heading_error_after_deg": round(after_error, 6),
        "inferred_goal_bearing_change_deg": round(
            inferred_goal_bearing_change,
            6,
        ),
        "wall_contact_count": wall_contacts,
        "interpretation": (
            "The turn sign and measured yaw were correct, but payload arc "
            "translation changed the world-point bearing enough that local heading "
            "error did not improve. A bounded translation is an evidence-grounded "
            "geometry reset, not a relaxation of turn-direction semantics."
        ),
    }


def _apply_physics_authoritative_failure_analysis_override(
    analysis: dict[str, Any],
    *,
    policy: PolicyScript,
    report: EvaluationReport,
    trace: ExecutionTrace | None,
) -> dict[str, Any]:
    """Apply deterministic route-obstruction and loaded-turn repair precedence."""

    route_obstruction = _rejected_route_obstruction_evidence(report)
    if route_obstruction.get("active") and _policy_uses_linear_transport(policy):
        overridden = dict(analysis)
        physical_evidence = [
            str(item)
            for item in overridden.get("physical_evidence", ())
            if str(item).strip()
        ]
        physical_evidence.extend(
            [
                (
                    "rejected endpoint wall contacts="
                    f"{route_obstruction.get('wall_contacts', [])}"
                ),
                (
                    "rejected endpoint supporting_robots="
                    f"{route_obstruction.get('supporting_robots', [])}"
                ),
            ]
        )
        overridden.update(
            {
                "outcome": "rejected",
                "primary_failure": "route_obstruction",
                "causal_chain": [
                    "the loaded linear rollout entered wall-blocked space",
                    "front/back response divergence degraded or removed one payload support",
                    "rollback restored the last trusted checkpoint before route recovery",
                ],
                "physical_evidence": list(dict.fromkeys(physical_evidence)),
                "strategy_defect": True,
                "coordination_defect": bool(
                    route_obstruction.get("single_sided_support", False)
                    or analysis.get("coordination_defect", False)
                ),
                "magnitude_defect": False,
                "reusable_context": [
                    "wall obstruction supersedes a same-direction synchronization probe",
                    "choose escape motion from current trusted wall clearance",
                    "loaded backward escape is a synchronized two-robot transaction",
                ],
                "source": "physics_authoritative_route_obstruction_override",
                "authoritative_repair_contract": {
                    "mutation_level": "strategy_mutation",
                    "forbid_same_motion_objective": True,
                    "loaded_backward_requires_sync": True,
                },
            }
        )
        return overridden

    if (
        report.failure_type == "formation_desync"
        and trace is not None
        and "formation_sync_hard_limit" in str(trace.error)
        and _policy_is_final_approach(policy)
        and _policy_uses_linear_transport(policy)
    ):
        metrics = report.metrics if isinstance(report.metrics, dict) else {}
        try:
            target_progress_m = float(
                metrics.get("payload_final_target_progress_m", 0.0)
            )
        except (TypeError, ValueError):
            target_progress_m = 0.0
        method, objective = _policy_motion_objective(policy)
        if (
            target_progress_m > 0.02
            and method == COORDINATED_LINEAR_TRANSPORT
            and objective in LOADED_TRANSLATION_DIRECTIONS
        ):
            overridden = dict(analysis)
            overridden.update(
                {
                    "outcome": "rejected",
                    "primary_failure": "execution_magnitude",
                    "causal_chain": [
                        "the selected final-approach direction made positive target progress",
                        "the command window outlasted the two backends' synchronized response",
                        "the formation guard rolled back before accepting the endpoint",
                    ],
                    "physical_evidence": [
                        "runtime safety abort=formation_sync_hard_limit",
                        f"payload_final_target_progress_m={target_progress_m:.3f}",
                        "rollback restored the last trusted checkpoint",
                    ],
                    "strategy_defect": False,
                    "coordination_defect": False,
                    "magnitude_defect": True,
                    "reusable_context": [
                        f"preserve {method}:{objective} in sync mode",
                        "reduce only the final-approach loaded distance window",
                    ],
                    "source": "physics_authoritative_linear_execution_override",
                    "authoritative_repair_contract": {
                        "mutation_level": "parameter_tuning",
                        "preserve_method": method,
                        "preserve_direction": objective,
                        "preserve_coordination_mode": "sync",
                        "tune_only": "distance_m",
                    },
                }
            )
            return overridden

    if (
        report.failure_type == "payload_unstable"
        and trace is not None
        and "payload_top_face_inverted" in str(trace.error)
        and _policy_uses_linear_transport(policy)
    ):
        evidence = report.evidence if isinstance(report.evidence, dict) else {}
        payload = evidence.get("payload", {})
        payload = payload if isinstance(payload, dict) else {}
        contacts = payload.get("contacts", {})
        contacts = contacts if isinstance(contacts, dict) else {}
        supporters = payload.get(
            "supporting_robots",
            contacts.get("support_robots", ()),
        )
        walls = contacts.get("walls", payload.get("wall_contacts", ()))
        floor_contact = bool(contacts.get("floor", False))
        method, objective = _policy_motion_objective(policy)
        if (
            set(str(item) for item in (supporters or ())) == {"front", "back"}
            and not floor_contact
            and not tuple(walls or ())
            and method == COORDINATED_LINEAR_TRANSPORT
            and objective in LOADED_TRANSLATION_DIRECTIONS
        ):
            metrics = report.metrics if isinstance(report.metrics, dict) else {}
            overridden = dict(analysis)
            overridden.update(
                {
                    "outcome": "rejected",
                    "primary_failure": "execution_magnitude",
                    "causal_chain": [
                        "the synchronized loaded translation retained bilateral support",
                        "the payload top face crossed the physical inversion boundary",
                        "rollback restored the passage-aligned parent checkpoint",
                    ],
                    "physical_evidence": [
                        "runtime safety abort=payload_top_face_inverted",
                        "supporting_robots=['back', 'front']",
                        "payload floor contact=false and wall contacts=[]",
                        (
                            "payload_step_displacement_m="
                            f"{float(metrics.get('payload_step_displacement_m', 0.0)):.3f}"
                        ),
                    ],
                    "strategy_defect": False,
                    "coordination_defect": False,
                    "magnitude_defect": True,
                    "reusable_context": [
                        f"preserve {method}:{objective} and synchronized team motion",
                        "reduce only the loaded linear distance tier",
                    ],
                    "source": "physics_authoritative_linear_execution_override",
                    "authoritative_repair_contract": {
                        "mutation_level": "parameter_tuning",
                        "preserve_method": method,
                        "preserve_direction": objective,
                        "preserve_coordination_mode": "sync",
                        "tune_only": "distance_m",
                    },
                }
            )
            return overridden

    spacing_abort = ""
    if trace is not None:
        trace_error = str(trace.error)
        if "robot_pair_too_close" in trace_error:
            spacing_abort = "robot_pair_too_close"
        elif "robot_pair_too_far" in trace_error:
            spacing_abort = "robot_pair_too_far"
    if report.failure_type != "formation_desync" or not spacing_abort:
        return analysis

    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    payload = evidence.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    support_forces = contacts.get("support_forces", {})
    support_forces = support_forces if isinstance(support_forces, dict) else {}
    force_by_robot = support_forces.get("normal_force_n_by_robot", {})
    force_by_robot = force_by_robot if isinstance(force_by_robot, dict) else {}
    try:
        bilateral_support = all(
            float(force_by_robot.get(robot_id, 0.0)) > 0.0
            for robot_id in ("front", "back")
        )
    except (TypeError, ValueError):
        bilateral_support = False
    if (
        not bilateral_support
        or bool(contacts.get("floor", False))
        or bool(contacts.get("walls", ()))
    ):
        return analysis

    methods = {_first_policy_api_method(script.code) for script in policy.robot_scripts}
    if len(methods) != 1:
        return analysis
    method = next(iter(methods))
    arguments = {
        script.robot_id: _first_task_call_literal_arguments(script.code)
        for script in policy.robot_scripts
    }
    metrics = report.metrics if isinstance(report.metrics, dict) else {}
    separation = float(metrics.get("robot_pair_xy_separation_m", 0.0) or 0.0)
    front_force = float(force_by_robot["front"])
    back_force = float(force_by_robot["back"])
    if method == COORDINATED_ALIGNMENT and spacing_abort == "robot_pair_too_close":
        axes = {str(item.get("axis", "")).strip() for item in arguments.values()}
        deltas = {
            robot_id: float(item.get("delta", 0.0))
            for robot_id, item in arguments.items()
        }
        roles = {
            assignment.robot_id: assignment.role for assignment in policy.assignments
        }
        if len(axes) != 1 or set(roles.values()) != {"reference", "corrector"}:
            return analysis
        axis = next(iter(axes))
        overridden = dict(analysis)
        overridden.update(
            {
                "outcome": "rejected",
                "primary_failure": "formation_desync",
                "causal_chain": [
                    "the selected alignment relation moved the corrector toward the reference",
                    "robot separation crossed the minimum spacing guard",
                    "rollback restored the same trusted checkpoint before an opposite-sign repair",
                ],
                "physical_evidence": [
                    (
                        "trace safety abort=robot_pair_too_close at separation "
                        f"{separation:.3f}m"
                    ),
                    f"alignment axis={axis}, deltas={deltas}, roles={roles}",
                    (
                        "bilateral support remained positive "
                        f"(front={front_force:.3f}N, back={back_force:.3f}N)"
                    ),
                    "payload had no floor or wall contact at the rejected endpoint",
                ],
                "strategy_defect": False,
                "coordination_defect": False,
                "magnitude_defect": True,
                "reusable_context": [
                    f"preserve {method}:{axis}",
                    "preserve the temporary reference/corrector relation and reverse only the corrector delta sign",
                ],
                "source": "physics_authoritative_alignment_sign_override",
                "authoritative_repair_contract": {
                    "mutation_level": "parameter_tuning",
                    "preserve_method": method,
                    "preserve_axis": axis,
                    "preserve_roles": roles,
                    "rejected_deltas": deltas,
                    "change_only": "corrector delta sign",
                },
            }
        )
        return overridden

    if method not in {COORDINATED_ARC_TRANSPORT, COORDINATED_PIVOT}:
        return analysis
    directions = {
        str(item.get("turn_direction", "")).strip() for item in arguments.values()
    }
    roles = {
        robot_id: str(item.get("role", "")).strip()
        for robot_id, item in arguments.items()
    }
    if len(directions) != 1 or set(roles.values()) != {"inner", "outer"}:
        return analysis
    direction = next(iter(directions))
    overridden = dict(analysis)
    overridden.update(
        {
            "outcome": "rejected",
            "primary_failure": "formation_desync",
            "causal_chain": [
                "the physical spacing guard stopped an otherwise supported loaded turn",
                "the selected signed pivot and inner/outer relation remain viable",
                "reduce only the bounded turn angle before retrying",
            ],
            "physical_evidence": [
                (f"trace safety abort={spacing_abort} at separation {separation:.3f}m"),
                (
                    "bilateral support remained positive "
                    f"(front={front_force:.3f}N, back={back_force:.3f}N)"
                ),
                "payload had no floor or wall contact at the rejected endpoint",
            ],
            "strategy_defect": False,
            "coordination_defect": False,
            "magnitude_defect": True,
            "reusable_context": [
                (
                    f"preserve {method}:{direction} with "
                    f"front={roles.get('front', '')}, back={roles.get('back', '')}"
                ),
                "tune angle_deg only; preserve duration and radius",
            ],
            "source": "physics_authoritative_robot_pair_spacing_override",
            "authoritative_repair_contract": {
                "mutation_level": "parameter_tuning",
                "preserve_method": method,
                "preserve_turn_direction": direction,
                "preserve_roles": roles,
                "tune_only": "angle_deg",
            },
        }
    )
    return overridden


def _validate_manager_mutation_level(
    candidate: dict[str, Any],
    *,
    report: EvaluationReport,
    failure_analysis: dict[str, Any],
    policy: PolicyScript | None = None,
    runtime_state: dict[str, Any] | None = None,
    formation_gate: dict[str, Any] | None = None,
    formation_sync_repair_gate: dict[str, Any] | None = None,
    parameter_envelope_exhausted: dict[str, Any] | None = None,
    alignment_sign_repair_contract: dict[str, Any] | None = None,
    model_owns_mutation_level: bool = False,
    minimum_structural_mutation: bool = False,
    coordination_variants_exhausted: bool = False,
    mutation_contract: dict[str, Any] | None = None,
) -> None:
    level = str(candidate.get("mutation_level", ""))
    parent_strategy = (
        _policy_mutation_parent_strategy(policy) if policy is not None else {}
    )
    if policy is not None:
        effective_mutation_contract = (
            mutation_contract
            if isinstance(mutation_contract, dict) and mutation_contract
            else build_mutation_contract(parent_strategy)
        )
        try:
            validate_manager_level(candidate, effective_mutation_contract)
        except ValueError as exc:
            raise ModelRoleError(str(exc)) from exc
    sync_repair_gate = (
        formation_sync_repair_gate
        if isinstance(formation_sync_repair_gate, dict)
        else {}
    )
    sync_probe_contract = sync_repair_gate.get("probe_contract", {})
    sync_probe_contract = (
        sync_probe_contract if isinstance(sync_probe_contract, dict) else {}
    )
    sync_magnitude_escalation_pending = bool(
        sync_repair_gate.get("active", False)
        and sync_probe_contract.get("magnitude_escalation_pending", False)
    )
    exhausted = (
        {}
        if sync_magnitude_escalation_pending
        else parameter_envelope_exhausted
        if isinstance(parameter_envelope_exhausted, dict)
        else {}
    )
    if not model_owns_mutation_level and exhausted and level == "parameter_tuning":
        raise ModelRoleError(
            "parameter_envelope_exhausted proves that the rejected primitive "
            "has no legal parameter-only mutation remaining; choose the required "
            "coordination_mutation or strategy_mutation escalation"
        )
    if (
        not model_owns_mutation_level
        and minimum_structural_mutation
        and level == "parameter_tuning"
    ):
        available_levels = {
            str(item).strip()
            for item in (
                mutation_contract.get("available_levels", ())
                if isinstance(mutation_contract, dict)
                else ()
            )
            if str(item).strip()
        }
        if available_levels != {"parameter_tuning"}:
            raise ModelRoleError(
                "the same checkpoint-local semantic strategy failed twice; "
                "parameter_tuning is no longer legal and the Evolution Manager must "
                "choose coordination_mutation or strategy_mutation"
            )
    expected = ""
    host_level = classify_report(report).level
    if (
        failure_analysis.get("source")
        == "physics_authoritative_route_obstruction_override"
    ):
        expected = "strategy_mutation"
    elif (
        failure_analysis.get("source")
        == "physics_authoritative_checkpoint_roundtrip_semantic_override"
    ):
        expected = "strategy_mutation"
    elif (
        failure_analysis.get("source")
        == "physics_authoritative_alignment_sign_override"
    ):
        expected = "parameter_tuning"
    elif (
        failure_analysis.get("source")
        == "physics_authoritative_linear_execution_override"
    ):
        expected = "parameter_tuning"
    elif host_level == "script":
        # "Script" identifies the rejected executable transaction. Its repair
        # can live at any of the three model-owned mutation levels, depending
        # on whether geometry, coordination, or magnitude caused the failure.
        expected = ""
    elif host_level == "strategy":
        expected = "strategy_mutation"
    elif host_level == "coordination":
        expected = "coordination_mutation"
    elif bool(failure_analysis.get("strategy_defect", False)):
        expected = "strategy_mutation"
    elif bool(failure_analysis.get("coordination_defect", False)):
        expected = "coordination_mutation"
    elif bool(failure_analysis.get("magnitude_defect", False)):
        expected = "parameter_tuning"
    else:
        expected = {
            "execution": "parameter_tuning",
        }.get(host_level, "")
    route_level_strategy_failure = bool(
        host_level == "strategy"
        or report.failure_type in {"goal_regression", "route_regression"}
        or failure_analysis.get("source")
        == "physics_authoritative_route_obstruction_override"
    )
    alignment_sign_repair = bool(
        failure_analysis.get("source")
        == "physics_authoritative_alignment_sign_override"
        or (
            isinstance(alignment_sign_repair_contract, dict)
            and alignment_sign_repair_contract
        )
    )
    parent_method = str(parent_strategy.get("primitive", "")).strip()
    parent_objective = str(parent_strategy.get("semantic_objective", "")).strip()
    rejected_wall_heading_correction = bool(
        policy is not None
        and HOST_WALL_HEADING_CORRECTION_MARKER in policy.rationale
        and report.failure_type
        in {
            "formation_desync",
            "no_payload_progress",
            "payload_not_supported",
            "payload_slip",
            "payload_unstable",
            "robot_pair_too_close",
        }
    )
    target_reorientation_direction = _declared_local_reorientation_turn_direction(
        runtime_state if isinstance(runtime_state, dict) else {},
        contract=_policy_local_navigation_contract(policy),
    )
    target_reorientation_execution_repair = bool(
        sync_repair_gate.get("active", False)
        and int(sync_repair_gate.get("failure_count", 0)) >= 2
        and parent_method == COORDINATED_PIVOT
        and parent_objective == target_reorientation_direction
        and report.failure_type
        in {
            "formation_desync",
            "payload_not_supported",
            "payload_slip",
            "payload_unstable",
        }
    )
    if (
        sync_repair_gate.get("active", False)
        and not route_level_strategy_failure
        and not alignment_sign_repair
    ):
        if sync_repair_gate.get("probe_unavailable", False):
            expected = "strategy_mutation"
        elif rejected_wall_heading_correction:
            expected = "strategy_mutation"
        elif target_reorientation_execution_repair:
            expected = "parameter_tuning"
        elif (
            int(sync_repair_gate.get("failure_count", 0)) >= 2
            and parent_method == COORDINATED_ALIGNMENT
            and bool(failure_analysis.get("coordination_defect", False))
            and report.failure_type
            in {
                "alignment_regression",
                "formation_desync",
                "payload_not_supported",
                "payload_slip",
                "payload_unstable",
            }
        ):
            # The exhausted linear probe has already selected the geometry-
            # repair family. If one alignment then fails through asymmetric
            # support, changing axes while preserving the same temporary
            # corrector relation is not a strategy repair.
            expected = "coordination_mutation"
        else:
            expected = (
                "strategy_mutation"
                if int(sync_repair_gate.get("failure_count", 0)) >= 2
                else "parameter_tuning"
            )
    gate = formation_gate if isinstance(formation_gate, dict) else {}
    if gate and not sync_repair_gate.get("active", False):
        parent_assignment = parent_strategy.get("team_assignment", {})
        parent_assignment = (
            parent_assignment if isinstance(parent_assignment, dict) else {}
        )
        parent_roles = {
            robot_id: str(
                parent_assignment.get(robot_id, {}).get("role", "")
                if isinstance(parent_assignment.get(robot_id), dict)
                else ""
            ).strip()
            for robot_id in ("front", "back")
        }
        gate_roles = gate.get("team_roles", {})
        gate_roles = gate_roles if isinstance(gate_roles, dict) else {}
        if (
            str(parent_strategy.get("primitive", "")).strip()
            != str(gate.get("primitive", "")).strip()
            or str(parent_strategy.get("semantic_objective", "")).strip()
            != str(gate.get("semantic_objective", "")).strip()
        ):
            expected = "strategy_mutation"
        elif (
            str(parent_strategy.get("coordination_mode", "")).strip()
            != str(gate.get("coordination_mode", "")).strip()
            or parent_roles != gate_roles
        ):
            expected = "coordination_mutation"
        elif not expected:
            expected = "parameter_tuning"
    if str(exhausted.get("primitive", "")).strip() == COORDINATED_LINEAR_TRANSPORT:
        expected = "strategy_mutation"
    if coordination_variants_exhausted:
        expected = "strategy_mutation"
    if model_owns_mutation_level:
        # Formal evaluation treats host classifications and repair gates as
        # measured evidence. They cannot select strategy, coordination, or
        # parameter mutation on the Evolution Manager's behalf.
        expected = ""
    if expected and level != expected:
        raise ModelRoleError(f"failure evidence requires {expected}; got {level}")
    if level == "parameter_tuning":
        allowed_parameter_fields = {
            COORDINATED_LINEAR_TRANSPORT: {"distance_m"},
            COORDINATED_ARC_TRANSPORT: {"angle_deg"},
            COORDINATED_PIVOT: {"angle_deg"},
            COORDINATED_ALIGNMENT: {"delta"},
        }.get(parent_method, set())
        mutate = candidate.get("mutate", ())
        mutate_items = mutate if isinstance(mutate, list) else ()
        if allowed_parameter_fields and mutate_items:
            named_fields = {
                field_name
                for item in mutate_items
                for field_name in (
                    "distance_m",
                    "speed_mps",
                    "ramp_s",
                    "angle_deg",
                    "duration_s",
                    "radius_m",
                    "swing_radius_m",
                    "delta",
                )
                if re.search(
                    rf"\b{re.escape(field_name)}\b",
                    str(item).strip().lower(),
                )
            }
            if named_fields != allowed_parameter_fields:
                raise ModelRoleError(
                    f"parameter_tuning for api.{parent_method} must name exactly "
                    f"{sorted(allowed_parameter_fields)} in mutate; got "
                    f"{sorted(named_fields) or [str(item) for item in mutate_items]}"
                )
    state = runtime_state if isinstance(runtime_state, dict) else {}
    turn_episode = state.get("turn_episode", {})
    turn_episode = turn_episode if isinstance(turn_episode, dict) else {}
    completed_turn_pending_translation = bool(
        str(turn_episode.get("status", "")).strip() in {"passage_aligned", "completed"}
        and bool(turn_episode.get("translation_pending", False))
    )
    proposed_mutation_text = json.dumps(
        {
            "mutate": candidate.get("mutate", ()),
            "inheritance": candidate.get("inheritance", ()),
        },
        ensure_ascii=False,
        default=str,
    ).lower()
    if (
        not model_owns_mutation_level
        and completed_turn_pending_translation
        and "coordinated_pivot" in proposed_mutation_text
    ):
        raise ModelRoleError(
            "the live turn episode already has passage_aligned with "
            "translation_pending=true; do not reopen coordinated_pivot. Repair "
            "the linear body-frame objective, use bounded coordinated_alignment, "
            "or tune one linear execution magnitude"
        )
    if level == "strategy_mutation":
        preserve = candidate.get("preserve", ())
        preserve_items = preserve if isinstance(preserve, list) else ()
        frozen_contract_terms = (
            "coordination_mode",
            "coordination mode",
            "sync mode",
            "async mode",
            "role assignment",
            "formation strategy",
            "parent primitive",
        )
        incompatible = [
            str(item)
            for item in preserve_items
            if any(term in str(item).strip().lower() for term in frozen_contract_terms)
        ]
        if incompatible:
            raise ModelRoleError(
                "strategy_mutation preserve may not freeze the parent primitive, "
                "coordination mode, formation strategy, or temporary role assignment; "
                f"incompatible preserve entries={incompatible}"
            )
        mutate = candidate.get("mutate", ())
        mutate_items = mutate if isinstance(mutate, list) else ()
        concrete_strategy_tokens = (
            *tuple(sorted(MODEL_PLANNING_SCRIPT_API_METHODS)),
            "forward",
            "backward",
            "clockwise",
            "counterclockwise",
            "longitudinal",
            "lateral",
            "yaw",
            "sync",
            "async",
            "reference",
            "corrector",
            "inner",
            "outer",
        )
        concrete_mutations = [
            str(item)
            for item in mutate_items
            if ":" in str(item)
            or "=" in str(item)
            or any(
                re.search(rf"\b{re.escape(token)}\b", str(item).strip().lower())
                for token in concrete_strategy_tokens
            )
        ]
        if concrete_mutations:
            raise ModelRoleError(
                "strategy_mutation mutate may name abstract dimensions only; the "
                "Evolution Manager must not select concrete primitive, semantic "
                "objective, coordination mode, direction, or temporary role values. "
                "Strategy Mutation owns the replacement values. "
                f"concrete mutate entries={concrete_mutations}"
            )
        forbidden = candidate.get("forbidden", ())
        forbidden_items = forbidden if isinstance(forbidden, list) else ()
        frozen_forbidden_patterns = (
            "change primitive",
            "change semantic objective",
            "change coordination mode",
            "change coordination_mode",
            "change temporary role",
            "change role mapping",
        )
        frozen_forbidden = [
            str(item)
            for item in forbidden_items
            if any(
                pattern in str(item).strip().lower()
                for pattern in frozen_forbidden_patterns
            )
        ]
        if frozen_forbidden:
            raise ModelRoleError(
                "strategy_mutation forbidden may not prohibit changing the parent "
                "primitive, semantic objective, coordination mode, or temporary roles. "
                "Forbid unsafe outcomes and exact counterexample replay instead. "
                f"incompatible forbidden entries={frozen_forbidden}"
            )
        inheritance = candidate.get("inheritance", ())
        inheritance_items = inheritance if isinstance(inheritance, list) else ()
        frozen_inheritance_terms = (
            "semantic objective",
            "semantic_objective",
            "coordination mode",
            "coordination_mode",
            "role assignment",
            "team_roles",
            "team roles",
            "reference",
            "corrector",
            "inner",
            "outer",
            *tuple(sorted(MODEL_PLANNING_SCRIPT_API_METHODS)),
        )
        frozen_inheritance = [
            str(item)
            for item in inheritance_items
            if any(
                term in str(item).strip().lower() for term in frozen_inheritance_terms
            )
        ]
        if frozen_inheritance:
            raise ModelRoleError(
                "strategy_mutation inheritance may not freeze the parent primitive, "
                "semantic objective, coordination mode, or temporary roles; inherit "
                "only the global mission, current phase, accepted checkpoint progress, "
                "payload support, and physical safety. "
                f"incompatible inheritance entries={frozen_inheritance}"
            )


def _evolution_decision_from_manager(
    host_decision: EvolutionDecision,
    manager: dict[str, Any],
) -> EvolutionDecision:
    level = str(manager["mutation_level"])
    action, repair_level = {
        "strategy_mutation": ("switch_strategy", "strategy"),
        "coordination_mutation": ("change_coordination", "coordination"),
        "parameter_tuning": ("tune_parameter", "execution"),
    }[level]
    mutate = manager.get("mutate", ())
    mutate = mutate if isinstance(mutate, list) else ()
    return replace(
        host_decision,
        action=action,
        repair_level=repair_level,
        reason=str(manager.get("failure_hypothesis", "")).strip()
        or host_decision.reason,
        required_change="; ".join(str(item) for item in mutate)
        or host_decision.required_change,
    )


def _validate_mutation_operator_choice(
    candidate: dict[str, Any],
    *,
    approved_candidate: dict[str, Any],
    manager_decision: dict[str, Any],
    current_policy: PolicyScript,
) -> None:
    selected_method = str(candidate.get("selected_method", "")).strip()
    selected_mode = str(candidate.get("coordination_mode", "")).strip()
    selected_objective = str(candidate.get("motion_objective", "")).strip()
    approved_method = str(approved_candidate.get("selected_method", "")).strip()
    approved_mode = str(approved_candidate.get("coordination_mode", "")).strip()
    approved_objective = str(approved_candidate.get("motion_objective", "")).strip()
    if (
        selected_method != approved_method
        or selected_mode != approved_mode
        or selected_objective != approved_objective
    ):
        raise ModelRoleError(
            "mutation operator must preserve the host-allowlisted method, mode, "
            "and semantic objective"
        )

    current_method = _first_policy_api_method(current_policy.code)
    current_modes = {
        str(script.coordination_mode).strip()
        for script in current_policy.robot_scripts
        if str(script.coordination_mode).strip()
    }
    current_mode = next(iter(current_modes)) if len(current_modes) == 1 else ""
    objective_parameter = _COORDINATION_OBJECTIVE_PARAMETER.get(current_method, "")
    current_arguments = _first_task_call_literal_arguments(current_policy.code)
    current_objective = (
        str(current_arguments.get(objective_parameter, "")).strip()
        if objective_parameter
        else ""
    )
    method_changed = bool(
        current_method and approved_method and current_method != approved_method
    )
    objective_changed = bool(
        (current_objective or approved_objective)
        and current_objective != approved_objective
    )
    mode_changed = bool(
        current_mode and approved_mode and current_mode != approved_mode
    )
    dimensions = {
        str(item).strip().lower()
        for item in candidate.get("changed_dimensions", ())
        if str(item).strip()
    }
    role_change = any(
        token in dimension
        for dimension in dimensions
        for token in ("role", "coordination", "sync", "async")
    )
    level = str(manager_decision.get("mutation_level", "")).strip()
    if level == "strategy_mutation" and not (method_changed or objective_changed):
        raise ModelRoleError(
            "strategy_mutation must change the maneuver family or semantic objective"
        )
    if level == "coordination_mutation":
        if method_changed or objective_changed:
            raise ModelRoleError(
                "coordination_mutation must preserve method and semantic objective"
            )
        if not (mode_changed or role_change):
            raise ModelRoleError(
                "coordination_mutation must change coordination mode or temporary roles"
            )
    if level == "parameter_tuning" and (
        method_changed or objective_changed or mode_changed or role_change
    ):
        raise ModelRoleError(
            "parameter_tuning must preserve strategy, objective, mode, and roles"
        )


def _validated_role_compiled_policy(
    payload: dict[str, Any],
    *,
    compiled_candidate: PolicyScript,
    current_policy: PolicyScript,
    selected_subtask: dict[str, Any],
    require_team_plan: bool,
) -> PolicyScript:
    normalizations = _canonicalize_robot_script_shorthand(payload)
    normalizations.extend(
        _canonicalize_team_coordination_mode(
            payload,
            selected_mode=str(selected_subtask.get("coordination_mode", "")).strip(),
        )
    )
    if normalizations:
        payload["host_schema_normalizations"] = normalizations
    policy = _policy_from_model_payload(
        payload,
        attempt=compiled_candidate.attempt,
        fallback_id=compiled_candidate.script_id,
        parent_script_id=current_policy.script_id,
        require_team_plan=require_team_plan,
    )
    selected_method = str(selected_subtask.get("selected_method", "")).strip()
    if selected_method:
        _validate_model_selected_subtask(
            policy,
            selected_method=selected_method,
            coordination_mode=str(
                selected_subtask.get("coordination_mode", "")
            ).strip(),
            motion_objective_parameter=str(
                selected_subtask.get("motion_objective_parameter", "")
            ).strip(),
            motion_objective=str(selected_subtask.get("motion_objective", "")).strip(),
        )
    expected_signature = _policy_transaction_signature(compiled_candidate)
    actual_signature = _policy_transaction_signature(policy)
    if not expected_signature or actual_signature != expected_signature:
        raise ModelRoleError(
            "Coder/Code Repair changed the approved executable transaction; "
            "method, roles, and parameters must match the host-validated candidate"
        )
    return policy


def _decode_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ModelAgentError(f"could not parse model JSON response: {text[:500]}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _evolution_policy_payload(policy: PolicyScript) -> dict[str, Any]:
    """Expose executable history without recycling model-authored claims."""

    return {
        "script_id": policy.script_id,
        "attempt": policy.attempt,
        "code": policy.code,
        "assignments": [asdict(item) for item in policy.assignments],
        "robot_scripts": [asdict(item) for item in policy.robot_scripts],
        "parent_script_id": policy.parent_script_id,
    }


_EVOLUTION_METRICS = (
    "payload_z_m",
    "payload_x_m",
    "payload_y_m",
    "payload_target_progress_m",
    "payload_final_target_progress_m",
    "payload_step_displacement_m",
    "payload_step_displacement_xy_m",
    "front_step_displacement_m",
    "back_step_displacement_m",
    "front_back_forward_displacement_desync_m",
    "front_back_step_vector_desync_m",
    "payload_step_yaw_drift_deg",
    "payload_formation_slip_m",
    "robot_payload_translation_tracking_ratio",
    "payload_support_center_error_m",
    "payload_support_force_front_n",
    "payload_support_force_back_n",
    "payload_support_force_balance_ratio",
    "payload_tilt_deg",
    "wall_contact_count",
    "payload_robot_contact_count",
    "robot_pair_xy_separation_m",
    "robot_pair_separation_along_active_goal_m",
    "robot_pair_separation_cross_active_goal_m",
    "robot_pair_axis_alignment_active_goal",
    "payload_long_axis_alignment_active_goal",
    "payload_long_axis_alignment_active_goal_improvement",
    "team_forward_alignment_active_goal",
    "team_forward_alignment_active_goal_improvement",
    "active_stage_goal_delta_x_m",
    "active_stage_goal_delta_y_m",
    "active_stage_goal_x_m",
    "active_stage_goal_y_m",
    "active_stage_goal_distance_m",
    "active_stage_goal_reached",
    "task_complete",
    "required_step_displacement_m",
)

_ROUTE_DISCOVERY_HIDDEN_METRICS = frozenset(
    {
        "payload_target_progress_m",
        "payload_final_target_progress_m",
        "robot_pair_separation_along_active_goal_m",
        "robot_pair_separation_cross_active_goal_m",
        "robot_pair_axis_alignment_active_goal",
        "payload_long_axis_alignment_active_goal",
        "payload_long_axis_alignment_active_goal_improvement",
        "team_forward_alignment_active_goal",
        "team_forward_alignment_active_goal_improvement",
        "active_stage_goal_delta_x_m",
        "active_stage_goal_delta_y_m",
        "active_stage_goal_x_m",
        "active_stage_goal_y_m",
        "active_stage_goal_distance_m",
        "active_stage_goal_reached",
    }
)


def _evaluation_payload(
    report: EvaluationReport,
    *,
    include_goal_metrics: bool = True,
) -> dict[str, Any]:
    """Project bounded physics and visual decisions without raw state dumps."""

    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    classification = classify_report(report)
    selected_evidence: dict[str, Any] = {}
    evidence_keys = [
        "checkpoint_outcome",
        "decision_policy",
        "script_status",
        "skill_call_statuses",
        "formation_synchronization",
        "endpoint_longitudinal_alignment",
        "straight_transport",
        "payload_support_center",
        "payload_support_force",
        "robot_pair_spacing",
        "post_action_stability_gate",
        "precommit_stability_gate",
        "rollback_verification",
        "pre_dispatch_recovery",
        "route_direction_verification",
        "transition_decision",
        "physical_quality_warnings",
    ]
    if include_goal_metrics:
        evidence_keys.append("active_stage_goal")
    for key in evidence_keys:
        value = evidence.get(key)
        if value not in (None, "", {}, []):
            selected_evidence[key] = value

    visual = evidence.get("visual_verifier", {})
    if isinstance(visual, dict) and visual:
        visual_evidence = visual.get("evidence", {})
        visual_evidence = visual_evidence if isinstance(visual_evidence, dict) else {}
        selected_evidence["visual_verifier"] = {
            "status": visual.get("status"),
            "model_name": visual.get("model_name"),
            "failure_type": visual.get("failure_type"),
            "confidence": visual.get("confidence"),
            "summary": visual.get("summary"),
            "scene_visible": visual_evidence.get("scene_visible"),
            "payload_on_floor": visual_evidence.get("payload_on_floor"),
            "robots_standing_stably": visual_evidence.get("robots_standing_stably"),
            "wall_collision_or_stuck": visual_evidence.get("wall_collision_or_stuck"),
            "payload_placed_on_target": visual_evidence.get("payload_placed_on_target"),
            "visible_evidence": visual_evidence.get("visible_evidence", []),
            "target_visible": visual_evidence.get("target_visible"),
            "target_relation_to_team": visual_evidence.get(
                "target_relation_to_team", ""
            ),
            "team_heading_relative_to_target": visual_evidence.get(
                "team_heading_relative_to_target", "uncertain"
            ),
            "team_heading_relative_to_visible_free_space": visual_evidence.get(
                "team_heading_relative_to_visible_free_space", "uncertain"
            ),
            "team_payload_posture_relative_to_visible_passage": visual_evidence.get(
                "team_payload_posture_relative_to_visible_passage", "uncertain"
            ),
            "immediate_route_direction_risk": bool(
                visual_evidence.get("immediate_route_direction_risk", False)
            ),
            "route_direction_risk_basis": visual_evidence.get(
                "route_direction_risk_basis", "uncertain"
            ),
            "route_direction_risk_confirmed": bool(
                visual_evidence.get("route_direction_risk_confirmed", False)
            ),
            "route_direction_evidence": visual_evidence.get(
                "route_direction_evidence", ""
            ),
            "rollout_motion_hint": visual_evidence.get("rollout_motion_hint", {}),
            "visible_free_space_connectivity": visual_evidence.get(
                "visible_free_space_connectivity", []
            ),
            "passage_geometry_observations": visual_evidence.get(
                "passage_geometry_observations", []
            ),
            "team_footprint_clearance": visual_evidence.get(
                "team_footprint_clearance", ""
            ),
            "relative_robot_layout": visual_evidence.get("relative_robot_layout", ""),
            "payload_axis_relative_to_passage": visual_evidence.get(
                "payload_axis_relative_to_passage", ""
            ),
            "visible_issues": visual_evidence.get("visible_issues", []),
            "observations": visual_evidence.get("observations", []),
            "risks": visual_evidence.get("risks", []),
            "possible_issues": visual_evidence.get("possible_issues", []),
            "failure_analysis": visual_evidence.get("failure_analysis", {}),
        }

    return {
        "success": report.success,
        "failure_type": report.failure_type,
        "failure_level": classification.level,
        "diagnosis": classification.diagnosis,
        "allowed_repairs": list(classification.allowed_repairs),
        "forbidden_repairs": list(classification.forbidden_repairs),
        "failed_subtask": report.failed_subtask,
        "metrics": {
            name: report.metrics[name]
            for name in _EVOLUTION_METRICS
            if name in report.metrics
            and (include_goal_metrics or name not in _ROUTE_DISCOVERY_HIDDEN_METRICS)
        },
        "evidence": selected_evidence,
        "visual_summary": report.visual_summary,
    }


def _append_candidate_rejection(
    path: Path,
    *,
    response_index: int,
    error: str,
    raw_response: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "response_index": int(response_index),
        "error": str(error),
        "raw_response": str(raw_response),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _candidate_rejection_count(path: Path) -> int:
    try:
        return sum(
            1
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError:
        return 0


def _observation_payload(
    observation: ObservationBundle | None,
    *,
    route_discovery: bool = False,
) -> dict[str, Any]:
    if observation is None:
        return {}
    return {
        "runtime_state_path": observation.runtime_state_path,
        "runtime_state": compact_runtime_state(
            observation.runtime_state,
            include_target_coordinates=not route_discovery,
        ),
        "visual_observations": [
            asdict(item) for item in observation.visual_observations
        ],
        "retrieved_specs": list(observation.retrieved_specs),
    }


def compact_runtime_state(
    state: dict[str, Any],
    *,
    include_target_coordinates: bool = True,
) -> dict[str, Any]:
    """Project live state to the fields a code agent can act on."""

    if not isinstance(state, dict):
        return {}
    payload = state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    support_forces = contacts.get("support_forces", {})
    support_forces = support_forces if isinstance(support_forces, dict) else {}
    support_target = payload.get("live_support_target", {})
    support_target = support_target if isinstance(support_target, dict) else {}
    support_center_error = payload_support_center_error_m(state)
    robots = state.get("robots", {})
    robot_summary: dict[str, dict[str, Any]] = {}
    if isinstance(robots, dict):
        for robot_id in ("front", "back"):
            robot = robots.get(robot_id, {})
            if not isinstance(robot, dict):
                continue
            robot_summary[robot_id] = {
                key: robot.get(key)
                for key in (
                    "x",
                    "y",
                    "yaw_deg",
                    "pelvis_height_m",
                    "initial_x",
                    "initial_y",
                )
                if key in robot
            }
            robot_summary[robot_id]["heading_vector_world_xy"] = (
                _heading_vector_world_xy(robot.get("yaw_deg"))
            )
            try:
                robot_summary[robot_id]["distance_from_authored_start_m"] = hypot(
                    float(robot["x"]) - float(robot["initial_x"]),
                    float(robot["y"]) - float(robot["initial_y"]),
                )
            except (KeyError, TypeError, ValueError):
                pass
            upper_body_lock = robot.get("upper_body_lock", {})
            if isinstance(upper_body_lock, dict) and upper_body_lock.get("enabled"):
                robot_summary[robot_id]["upper_body_lock_max_abs_error_rad"] = (
                    upper_body_lock.get("max_abs_error_rad")
                )
            place_crouch = robot.get("place_crouch", {})
            if isinstance(place_crouch, dict) and place_crouch:
                robot_summary[robot_id]["place_crouch"] = {
                    key: place_crouch.get(key)
                    for key in (
                        "crouch",
                        "pelvis_height_m",
                        "post_release_latched",
                    )
                    if key in place_crouch
                }
    robot_pair_xy_separation_m = None
    if {"front", "back"}.issubset(robot_summary):
        front = robot_summary["front"]
        back = robot_summary["back"]
        try:
            robot_pair_xy_separation_m = hypot(
                float(front["x"]) - float(back["x"]),
                float(front["y"]) - float(back["y"]),
            )
        except (KeyError, TypeError, ValueError):
            robot_pair_xy_separation_m = None
    state_time = _safe_float(state.get("time_unix"))
    checkpoint = state.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    turn_episode = state.get("turn_episode", {})
    turn_episode = turn_episode if isinstance(turn_episode, dict) else {}
    trusted_route_observation = state.get("trusted_route_observation", {})
    trusted_route_observation = (
        trusted_route_observation if isinstance(trusted_route_observation, dict) else {}
    )
    upper_body_lock = state.get("upper_body_pose_lock", {})
    upper_body_lock = upper_body_lock if isinstance(upper_body_lock, dict) else {}
    current_upper_body_pose = str(upper_body_lock.get("pose_name", ""))
    target_upper_body_pose = str(
        upper_body_lock.get("target_pose_name", current_upper_body_pose)
    )
    outward_payload_offset_m = _safe_float(
        payload.get("support_center_signed_longitudinal_offset_m")
    )
    recovery_outward_evidence_m = payload_recovery_outward_evidence_m(state)
    lateral_payload_offset_m = _safe_float(
        payload.get("support_center_signed_lateral_offset_m")
    )
    supporting_robots = frozenset(
        str(robot_id) for robot_id in contacts.get("support_robots", ())
    )
    control_healthy = bool(
        state.get("control_health", {}).get("healthy", False)
        if isinstance(state.get("control_health"), dict)
        else False
    )
    checkpoint_resync_active = bool(checkpoint.get("resync_active", False))
    recovery_threshold_m = TASK_CARRY_SAFETY.warning_payload_formation_slip_m
    recovery_trigger = payload_recovery_trigger(
        state,
        minimum_outward_offset_m=recovery_threshold_m,
    )
    payload_recovery_available = bool(
        control_healthy
        and not checkpoint_resync_active
        and bool(payload.get("active", False))
        and not bool(contacts.get("floor", False))
        and {"front", "back"}.issubset(supporting_robots)
        and bool(upper_body_lock.get("configured", False))
        and not bool(upper_body_lock.get("transition_active", False))
        and current_upper_body_pose == DEFAULT_CARRY_READY_POSE_NAME
        and target_upper_body_pose == DEFAULT_CARRY_READY_POSE_NAME
        and bool(recovery_trigger["active"])
        and support_center_error is not None
        and support_center_error <= TASK_CARRY_SAFETY.max_support_center_error_m
    )
    return {
        "coordinate_convention": _route_neutral_coordinate_convention(),
        "robot_identity_contract": _immutable_robot_identity_contract(),
        "runtime_status": state.get("runtime_status", ""),
        "runtime_fault": state.get("runtime_fault", {}),
        "runtime_age_s": None
        if state_time is None
        else max(0.0, time.time() - state_time),
        "control_healthy": control_healthy,
        "checkpoint_resync_active": checkpoint_resync_active,
        # Scoped rollback lessons are valid only at one accepted transaction.
        # Keep its identity in the compact model state so those lessons cannot
        # silently become global or silently disappear.
        "checkpoint": {
            key: checkpoint.get(key)
            for key in ("checkpoint_id", "action")
            if key in checkpoint
        },
        # This verifier-owned transaction contract must survive model-state
        # compaction. Otherwise the selector guesses a turn that preflight
        # already knows is illegal, producing a no-motion rejection loop.
        "turn_episode": dict(turn_episode),
        "trusted_route_observation": dict(trusted_route_observation),
        "fixed_global_camera_heading_projection": (
            _fixed_global_camera_heading_projection(state)
        ),
        "elastic_band_dropped": bool(state.get("elastic_band_dropped", False)),
        "payload": {
            "present": bool(payload.get("present", False)),
            "active": bool(payload.get("active", False)),
            "x": payload.get("x"),
            "y": payload.get("y"),
            "z": payload.get("z"),
            "yaw_deg": payload.get("yaw_deg"),
            "heading_vector_world_xy": _heading_vector_world_xy(
                payload.get("yaw_deg")
            ),
            "tilt_deg": payload.get("tilt_deg"),
            "linear_speed_mps": payload.get("linear_speed_mps"),
            "angular_speed_radps": payload.get("angular_speed_radps"),
            "released": bool(payload.get("released", False)),
            "support_target_ready": bool(support_target.get("ready", False)),
            "support_tracking_available": bool(
                support_target.get(
                    "tracking_available", support_target.get("ready", False)
                )
            ),
            "support_wait_reasons": list(support_target.get("wait_reasons", ())),
            "support_target_x": support_target.get("x"),
            "support_target_y": support_target.get("y"),
            "support_center_xy_error_m": support_center_error,
            "support_center_signed_longitudinal_offset_m": (outward_payload_offset_m),
            "support_center_signed_lateral_offset_m": lateral_payload_offset_m,
            "support_offset_convention": payload.get("support_offset_convention", ""),
            # Only approved palm-surface contacts are carrying support. Generic
            # torso/arm contacts remain available as diagnostics but must not
            # unlock a transport action.
            "supporting_robots": list(contacts.get("support_robots", ())),
            "support_geometries": list(contacts.get("support_geometries", ())),
            "support_force_available": bool(support_forces.get("available", False)),
            "support_normal_force_n_by_robot": support_forces.get(
                "normal_force_n_by_robot", {}
            ),
            "support_force_balance_ratio": support_forces.get("balance_ratio"),
            "all_robot_contacts": list(contacts.get("robots", ())),
            "floor_contact": bool(contacts.get("floor", False)),
            "wall_contacts": list(contacts.get("walls", ())),
            "target_contact": bool(contacts.get("target", False)),
        },
        "upper_body_pose_lock": {
            "configured": bool(upper_body_lock.get("configured", False)),
            "pose_name": current_upper_body_pose,
            "target_pose_name": target_upper_body_pose,
            "transition_active": bool(upper_body_lock.get("transition_active", False)),
            "last_event": (
                upper_body_lock.get("last_event", {})
                if isinstance(upper_body_lock.get("last_event"), dict)
                else {}
            ),
        },
        "post_release_lifecycle": (
            {
                key: state.get("post_release_lifecycle", {}).get(key)
                for key in (
                    "payload_released",
                    "crouch_latched",
                    "crouched_retreat_completed",
                    "stand_recovered",
                )
                if key in state.get("post_release_lifecycle", {})
            }
            if isinstance(state.get("post_release_lifecycle"), dict)
            else {}
        ),
        "payload_inward_recovery": {
            "available": payload_recovery_available,
            "method": "recover_payload_inward",
            "coordination_mode": "sync_only",
            "required_start_pose": DEFAULT_CARRY_READY_POSE_NAME,
            "target_pose": PAYLOAD_RECOVERY_POSE_NAME,
            "trigger_reasons": list(recovery_trigger["reasons"]),
            "measured_body_forward_offset_m": recovery_outward_evidence_m,
            "current_body_forward_offset_m": outward_payload_offset_m,
            "minimum_body_forward_offset_m": recovery_threshold_m,
            "measured_tilt_deg": recovery_trigger["tilt_deg"],
            "minimum_tilt_deg": recovery_trigger["minimum_tilt_deg"],
            "support_center_error_m": support_center_error,
            "maximum_support_center_error_m": (
                TASK_CARRY_SAFETY.max_support_center_error_m
            ),
        },
        "target": {
            key: state.get("target", {}).get(key)
            for key in (
                (
                    "available",
                    "center_x_m",
                    "center_y_m",
                    "center_z_m",
                    "top_z_m",
                    "payload_center_target_z_m",
                )
                if include_target_coordinates
                else ("available",)
            )
            if isinstance(state.get("target"), dict) and key in state.get("target", {})
        },
        "robots": robot_summary,
        "robot_pair_xy_separation_m": robot_pair_xy_separation_m,
    }


def _accepted_route_action_history(
    manifest_path: Path | None,
) -> dict[str, Any]:
    """Load a compact, chronological route-evidence view of accepted steps."""

    if manifest_path is None:
        return {}
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(manifest, dict):
        return {}
    raw_steps = manifest.get("steps", [])
    if not isinstance(raw_steps, list):
        return {}

    steps: list[dict[str, Any]] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        robot_calls = raw_step.get("robot_calls", {})
        robot_calls = robot_calls if isinstance(robot_calls, dict) else {}
        representative = next(
            (
                robot_calls.get(robot_id)
                for robot_id in ("front", "back")
                if isinstance(robot_calls.get(robot_id), dict)
            ),
            {},
        )
        representative = representative if isinstance(representative, dict) else {}
        parameters = representative.get("parameters", {})
        parameters = parameters if isinstance(parameters, dict) else {}
        robot_scripts = raw_step.get("robot_scripts", {})
        robot_scripts = robot_scripts if isinstance(robot_scripts, dict) else {}
        representative_script = next(
            (
                robot_scripts.get(robot_id)
                for robot_id in ("front", "back")
                if isinstance(robot_scripts.get(robot_id), dict)
            ),
            {},
        )
        representative_script = (
            representative_script if isinstance(representative_script, dict) else {}
        )
        measured = raw_step.get("measured_outcome", {})
        measured = measured if isinstance(measured, dict) else {}
        semantic_objective = next(
            (
                str(parameters[key]).strip()
                for key in ("direction", "turn_direction", "axis")
                if key in parameters and str(parameters[key]).strip()
            ),
            "",
        )
        transition_outcome = str(raw_step.get("transition_outcome", "")).strip()
        positive_route_memory_eligible = bool(
            raw_step.get("positive_route_memory_eligible", False)
        )
        steps.append(
            {
                "index": int(raw_step.get("index", len(steps) + 1)),
                "checkpoint_after": str(raw_step.get("checkpoint_after", "")).strip(),
                "method": str(representative.get("method", "")).strip(),
                "semantic_objective": semantic_objective,
                "phase_id": str(representative_script.get("phase_id", "")).strip(),
                "coordination_mode": str(raw_step.get("coordination_mode", "")).strip(),
                "transition_outcome": transition_outcome,
                "positive_route_memory_eligible": positive_route_memory_eligible,
                "payload_step_displacement_xy_m": measured.get(
                    "payload_step_displacement_xy_m"
                ),
                "payload_step_signed_yaw_change_deg": measured.get(
                    "payload_step_signed_yaw_change_deg"
                ),
                "payload_target_progress_m": measured.get("payload_target_progress_m"),
                "measured_outcome": {
                    key: measured.get(key)
                    for key in ("payload_x_m", "payload_y_m", "payload_z_m")
                    if measured.get(key) is not None
                },
            }
        )
    return {
        "source": str(path),
        "latest_checkpoint": str(manifest.get("latest_checkpoint", "")).strip(),
        "accepted_step_count": len(steps),
        "route_progress_step_count": sum(
            step["transition_outcome"] in {"ACCEPT_PROGRESS", "TASK_COMPLETE"}
            and step["positive_route_memory_eligible"]
            for step in steps
        ),
        "steps": steps,
        "authority": (
            "accepted checkpoint chronology limits phase completion; rejected "
            "rollouts and model-written phase labels are not route evidence"
        ),
    }


def _accepted_route_reversal_guard(
    accepted_route_history: dict[str, Any] | None,
) -> dict[str, Any]:
    """Block only the next action that would complete an accepted A-B-A loop."""

    history = accepted_route_history if isinstance(accepted_route_history, dict) else {}
    raw_steps = history.get("steps", [])
    raw_steps = raw_steps if isinstance(raw_steps, list) else []
    route_steps = [
        step
        for step in raw_steps
        if isinstance(step, dict)
        and str(step.get("method", "")).strip()
        and str(step.get("transition_outcome", "")).strip()
        in {"ACCEPT_PROGRESS", "TASK_COMPLETE"}
        and bool(step.get("positive_route_memory_eligible", False))
    ]
    if len(route_steps) < 2:
        return {}
    earlier = route_steps[-2]
    latest = route_steps[-1]
    if any(
        str(step.get("method", "")).strip() != COORDINATED_LINEAR_TRANSPORT
        or str(step.get("semantic_objective", "")).strip()
        not in LOADED_TRANSLATION_DIRECTIONS
        for step in (earlier, latest)
    ):
        return {}
    earlier_direction = str(earlier.get("semantic_objective", "")).strip()
    latest_direction = str(latest.get("semantic_objective", "")).strip()
    opposite = {
        "forward": "backward",
        "backward": "forward",
        "left": "right",
        "right": "left",
    }.get(earlier_direction, "")
    if not opposite or latest_direction != opposite:
        return {}
    return {
        "active": True,
        "source": "accepted_checkpoint_chronology",
        "earlier_direction": earlier_direction,
        "latest_direction": latest_direction,
        "blocked_immediate_return_direction": earlier_direction,
        "latest_checkpoint": str(latest.get("checkpoint_after", "")).strip(),
        "model_retains_route_ownership": True,
        "rule": (
            "retain both physically safe checkpoints, but do not immediately undo "
            "the latest accepted inverse translation; the model remains free to "
            "continue the latest direction or choose any non-return maneuver"
        ),
    }


def _accepted_straight_saturation_contract(
    accepted_route_history: dict[str, Any] | None,
    *,
    host_wall_clearance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Detect when another room-scale straight segment no longer fits."""

    history = accepted_route_history if isinstance(accepted_route_history, dict) else {}
    steps = [
        step
        for step in history.get("steps", ())
        if isinstance(step, dict)
        and str(step.get("method", "")).strip()
        and str(step.get("transition_outcome", "")).strip()
        in {"ACCEPT_PROGRESS", "TASK_COMPLETE"}
        and bool(step.get("positive_route_memory_eligible", False))
    ]
    if len(steps) < 2:
        return {}
    previous, latest = steps[-2:]
    method = str(latest.get("method", "")).strip()
    direction = str(latest.get("semantic_objective", "")).strip()
    phase_id = str(latest.get("phase_id", "")).strip()
    if not (
        method == COORDINATED_LINEAR_TRANSPORT
        and str(previous.get("method", "")).strip() == method
        and str(previous.get("semantic_objective", "")).strip() == direction
        and direction in LOADED_TRANSLATION_DIRECTIONS
        and phase_id
        and str(previous.get("phase_id", "")).strip() == phase_id
    ):
        return {}
    try:
        measured_displacement_m = float(
            latest.get("payload_step_displacement_xy_m", 0.0)
        )
    except (TypeError, ValueError):
        return {}
    if measured_displacement_m < min(COORDINATED_LINEAR_DISTANCE_CHOICES_M):
        return {}
    clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    translations = clearance.get("translation_candidates", {})
    translations = translations if isinstance(translations, dict) else {}
    direction_clearance = translations.get(direction, {})
    direction_clearance = (
        direction_clearance if isinstance(direction_clearance, dict) else {}
    )
    try:
        remaining_clearance_m = float(direction_clearance["clearance_m"])
    except (KeyError, TypeError, ValueError):
        return {}
    turns = clearance.get("turn_candidates", {})
    turns = turns if isinstance(turns, dict) else {}
    unblocked_turns = [
        turn_direction
        for turn_direction in sorted(_SIGNED_TURN_DIRECTIONS)
        if isinstance(turns.get(turn_direction), dict)
        and turns[turn_direction].get("relation") != "blocked"
    ]
    if remaining_clearance_m >= measured_displacement_m or not unblocked_turns:
        return {}
    return {
        "active": True,
        "source": "accepted_checkpoint_chronology_plus_authored_wall_geometry",
        "phase_id": phase_id,
        "saturated_direction": direction,
        "latest_checkpoint": str(latest.get("checkpoint_after", "")).strip(),
        "latest_measured_displacement_m": round(measured_displacement_m, 6),
        "remaining_same_direction_clearance_m": round(remaining_clearance_m, 6),
        "unblocked_turn_directions": unblocked_turns,
        "required_next_primitive": COORDINATED_PIVOT,
        "rule": (
            "two consecutive accepted same-phase straight segments reached a "
            "corner where another segment of the verified room-scale magnitude "
            "no longer fits; change passage geometry instead of micro-tuning "
            "linear distance"
        ),
    }


def _route_clearance_conflict_contract(
    accepted_route_history: dict[str, Any] | None,
    *,
    host_wall_clearance: dict[str, Any] | None,
    active_route_guidance: dict[str, Any] | None,
    runtime_state: dict[str, Any] | None,
    saturation_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose feasible geometry repairs when route and local clearance conflict.

    The contract preserves the authored route direction. It activates whenever
    every route-aligned body-frame translation lacks a calibrated distance tier,
    even if the team has not first accumulated two same-direction route steps.
    The Task Orchestrator still chooses among measured setup capabilities.
    """

    clearance = host_wall_clearance if isinstance(host_wall_clearance, dict) else {}
    guidance = active_route_guidance if isinstance(active_route_guidance, dict) else {}
    state = runtime_state if isinstance(runtime_state, dict) else {}
    saturation = (
        saturation_contract
        if isinstance(saturation_contract, dict)
        else _accepted_straight_saturation_contract(
            accepted_route_history,
            host_wall_clearance=clearance,
        )
    )
    if not (clearance.get("available", False) and guidance.get("active", False)):
        return {}

    try:
        route_x, route_y = (
            float(guidance["world_direction_xy"][0]),
            float(guidance["world_direction_xy"][1]),
        )
        route_norm = hypot(route_x, route_y)
    except (KeyError, IndexError, TypeError, ValueError):
        return {}
    if not isfinite(route_norm) or route_norm <= 1.0e-9:
        return {}
    route_x /= route_norm
    route_y /= route_norm

    translations = clearance.get("translation_candidates", {})
    translations = translations if isinstance(translations, dict) else {}
    minimum_translation_m = min(COORDINATED_LINEAR_DISTANCE_CHOICES_M)
    minimum_route_alignment = cos(radians(HEADING_ALIGNMENT_TOLERANCE_DEG))
    route_aligned_translations: list[str] = []
    route_translation_clearance: dict[str, dict[str, Any]] = {}
    for direction in LOADED_TRANSLATION_DIRECTIONS:
        evidence = translations.get(direction, {})
        evidence = evidence if isinstance(evidence, dict) else {}
        try:
            direction_x = float(evidence["world_direction_xy"][0])
            direction_y = float(evidence["world_direction_xy"][1])
            direction_norm = hypot(direction_x, direction_y)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not isfinite(direction_norm) or direction_norm <= 1.0e-9:
            continue
        direction_x /= direction_norm
        direction_y /= direction_norm
        alignment = direction_x * route_x + direction_y * route_y
        if alignment + 1.0e-9 < minimum_route_alignment:
            continue
        route_aligned_translations.append(direction)
        clearance_m = _loaded_translation_clearance_m(clearance, direction)
        route_translation_clearance[direction] = {
            "route_alignment": round(alignment, 6),
            "clearance_m": (
                round(clearance_m, 6) if clearance_m is not None else None
            ),
            "has_supported_distance_tier": (
                _loaded_translation_has_supported_tier(clearance, direction)
            ),
        }
    route_translation_dead_end = bool(route_aligned_translations) and not any(
        _loaded_translation_has_supported_tier(clearance, direction)
        for direction in route_aligned_translations
    )
    if not saturation.get("active", False) and not route_translation_dead_end:
        return {}

    heading = team_heading_deg(state)
    if heading is None:
        return {}
    route_bearing = degrees(atan2(route_y, route_x))
    heading_error = wrap_degrees(route_bearing - heading)
    improving_turn = (
        "counterclockwise"
        if heading_error > 1.0
        else "clockwise"
        if heading_error < -1.0
        else ""
    )
    turn_candidates = clearance.get("turn_candidates", {})
    turn_candidates = turn_candidates if isinstance(turn_candidates, dict) else {}
    safe_turns: list[str] = []
    safe_turn_envelopes: dict[str, dict[str, float]] = {}
    unexecutable_turns: dict[str, dict[str, Any]] = {}
    minimum_calibrated_pivot_deg = min(MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG)
    for direction in ("clockwise", "counterclockwise"):
        evidence = turn_candidates.get(direction, {})
        evidence = evidence if isinstance(evidence, dict) else {}
        try:
            maximum = float(evidence["max_safe_angle_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if evidence.get("relation") == "blocked" or maximum < 2.0:
            continue
        heading_improvement_cap = max(
            0.0,
            2.0 * abs(heading_error) - MIN_HEADING_IMPROVEMENT_DEG,
        )
        effective_maximum = min(maximum, heading_improvement_cap)
        supported_angles = [
            float(angle)
            for angle in MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG
            if float(angle) <= effective_maximum + 1.0e-9
        ]
        if not supported_angles:
            unexecutable_turns[direction] = {
                "geometric_max_safe_angle_deg": round(maximum, 3),
                "heading_improving_max_angle_deg": round(
                    heading_improvement_cap,
                    3,
                ),
                "minimum_calibrated_loaded_pivot_deg": (
                    minimum_calibrated_pivot_deg
                ),
                "reason": (
                    "no calibrated loaded pivot tier is both inside the finite "
                    "wall sweep and guaranteed to reduce the signed heading error"
                ),
            }
            continue
        envelope = {
            "minimum_angle_deg": min(supported_angles),
            "maximum_angle_deg": round(effective_maximum, 3),
            "geometric_maximum_angle_deg": round(maximum, 3),
            "calibrated_angle_choices_deg": supported_angles,
        }
        if not improving_turn or direction == improving_turn:
            safe_turns.append(direction)
            safe_turn_envelopes[direction] = envelope

    corridor = guidance.get("corridor", {})
    corridor = corridor if isinstance(corridor, dict) else {}
    measured = guidance.get("measured", {})
    measured = measured if isinstance(measured, dict) else {}
    try:
        payload_x, payload_y = (
            float(measured["payload_xy_m"][0]),
            float(measured["payload_xy_m"][1]),
        )
        center_x, center_y = (
            float(corridor["center_xy_m"][0]),
            float(corridor["center_xy_m"][1]),
        )
    except (KeyError, IndexError, TypeError, ValueError):
        payload_x = payload_y = center_x = center_y = 0.0
        cross_track_available = False
    else:
        cross_track_available = True

    # Left normal of the active route. A translation is a route-reentry option
    # only when it is non-regressive and measurably reduces centerline error.
    normal_x, normal_y = -route_y, route_x
    signed_cross_track = (
        (payload_x - center_x) * normal_x
        + (payload_y - center_y) * normal_y
        if cross_track_available
        else 0.0
    )
    reentry_translations: list[str] = []
    reentry_evidence: dict[str, dict[str, float]] = {}
    saturated_direction = str(saturation.get("saturated_direction", "")).strip()
    inverse_direction = {
        "forward": "backward",
        "backward": "forward",
        "left": "right",
        "right": "left",
    }.get(saturated_direction, "")
    for direction in LOADED_TRANSLATION_DIRECTIONS:
        if direction in {saturated_direction, inverse_direction}:
            continue
        evidence = translations.get(direction, {})
        evidence = evidence if isinstance(evidence, dict) else {}
        if evidence.get("relation") == "blocked":
            continue
        try:
            direction_x, direction_y = (
                float(evidence["world_direction_xy"][0]),
                float(evidence["world_direction_xy"][1]),
            )
            clearance_m = float(evidence["clearance_m"])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if clearance_m < minimum_translation_m:
            continue
        route_alignment = direction_x * route_x + direction_y * route_y
        if route_alignment < -0.05 or not cross_track_available:
            continue
        probe_distance = min(0.5, clearance_m)
        projected_cross_track = signed_cross_track + probe_distance * (
            direction_x * normal_x + direction_y * normal_y
        )
        improvement = abs(signed_cross_track) - abs(projected_cross_track)
        if improvement < 0.05:
            continue
        reentry_translations.append(direction)
        reentry_evidence[direction] = {
            "available_clearance_m": round(clearance_m, 3),
            "route_alignment": round(route_alignment, 4),
            "projected_cross_track_improvement_m": round(improvement, 4),
        }

    executable_translations = list(
        _model_executable_loaded_translation_directions(clearance)
    )
    clearance_setup_translations = [
        direction
        for direction in executable_translations
        if direction not in route_aligned_translations
        and direction not in reentry_translations
    ]

    if (
        not safe_turns
        and not reentry_translations
        and not clearance_setup_translations
    ):
        return {}
    activation_reason = (
        "route_aligned_translation_has_no_supported_distance_tier"
        if route_translation_dead_end
        else "accepted_straight_saturation"
    )
    return {
        "active": True,
        "source": "active_route_plus_authored_wall_geometry",
        "activation_reason": activation_reason,
        "scope_checkpoint_id": str(
            state.get("checkpoint", {}).get("checkpoint_id", "")
            if isinstance(state.get("checkpoint"), dict)
            else ""
        ),
        "active_segment_id": str(guidance.get("segment_id", "")).strip(),
        "preserve_navigation_world_direction_xy": [
            round(route_x, 6),
            round(route_y, 6),
        ],
        "saturated_body_frame_direction": saturated_direction,
        "route_aligned_translation_directions": route_aligned_translations,
        "route_aligned_translation_clearance": route_translation_clearance,
        "minimum_supported_translation_distance_m": minimum_translation_m,
        "all_route_aligned_translation_tiers_blocked": route_translation_dead_end,
        "signed_heading_error_to_active_segment_deg": round(heading_error, 6),
        "safe_heading_correction_turn_directions": safe_turns,
        "safe_turn_angle_envelopes": safe_turn_envelopes,
        "geometrically_clear_but_unexecutable_turns": unexecutable_turns,
        "route_reentry_translation_directions": reentry_translations,
        "route_reentry_evidence": reentry_evidence,
        "clearance_setup_translation_directions": clearance_setup_translations,
        "inverse_route_translation_setup_only": (
            inverse_direction
            if inverse_direction in clearance_setup_translations
            else ""
        ),
        "setup_only": True,
        "phase_completion_authorized": False,
        "decision_owner": "task_orchestrator",
        "parameter_owner": "coder_within_skill_and_wall_envelopes",
        "rule": (
            "preserve the active route reference and choose one physically executable "
            "geometry setup; a wall-clear micro-sweep below the calibrated SONIC "
            "pivot domain is not an available pivot. A lateral or inverse translation "
            "may create clearance, but remains setup-only and is never route progress"
        ),
    }


def model_task_context(
    task: BoxCarryTask,
    *,
    route_discovery: bool | None = None,
) -> dict[str, Any]:
    """Expose the semantic task without leaking host-authored route waypoints."""

    route_discovery = (
        task.route_discovery if route_discovery is None else route_discovery
    )
    if not route_discovery:
        return asdict(task)
    return {
        "task_id": task.task_id,
        "target": task.target,
        "scene_id": task.scene_id,
        "payload_id": task.payload_id,
        "robot_ids": list(task.robot_ids),
        "target_region": task.target_region,
        "task_scope": "full_scene_route_discovery",
        "route_discovery": True,
        "final_success_contract": {
            "payload_physically_supported_by_target": task.require_payload_on_target,
            "robots_exit_after_release": task.require_robots_exit,
            "robot_exit_min_distance_from_target_m": (
                task.robot_exit_min_distance_from_target_m
            ),
            "robot_pre_stand_clearance_margin_m": (
                task.robot_pre_stand_clearance_margin_m
            ),
            "robot_pre_stand_min_distance_from_target_m": (
                task.robot_exit_min_distance_from_target_m
                + task.robot_pre_stand_clearance_margin_m
            ),
            "required_post_release_sequence": (
                [
                    "release_payload",
                    "synchronized_crouched_exit_backward",
                    "recover_stand",
                    "finish_upright_and_clear_of_payload",
                ]
                if task.require_robots_exit
                else []
            ),
        },
        "physical_safety_contract": {
            "minimum_robot_xy_separation_m": task.min_robot_xy_separation_m,
            "payload_must_remain_supported_by_both_robots": (
                task.require_each_robot_payload_contact
            ),
        },
    }


def _required_initial_call(payload: dict[str, Any]) -> tuple[str, str]:
    runtime = payload.get("observation", {}).get("runtime_state", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    payload_state = runtime.get("payload", {})
    payload_state = payload_state if isinstance(payload_state, dict) else {}
    supporters = _payload_supporting_robots(payload_state)
    if _payload_ready_for_release(runtime):
        return (
            "release_payload(ramp_s=4, hold_s=2)",
            "the loaded payload COM projection is inside the physical target support region within the guarded release height",
        )
    if _payload_is_released_on_target(runtime):
        lifecycle = _post_release_lifecycle(runtime)
        robots_clear = _robots_clear_of_released_payload(payload, runtime)
        robots_ready_to_stand = _robots_clear_of_released_payload(
            payload,
            runtime,
            include_pre_stand_reserve=True,
        )
        if (
            lifecycle["crouched_retreat_completed"]
            and robots_ready_to_stand
            and not lifecycle["stand_recovered"]
        ):
            return (
                "recover_stand(height=0.80, ramp_s=6, hold_s=2)",
                "the synchronized crouched retreat is complete; recover both robots upright",
            )
        if lifecycle["stand_recovered"] and not robots_clear:
            return (
                "",
                "the robots recovered upright before reaching the required target clearance; restore the latest crouched checkpoint and continue synchronized backward retreat",
            )
        if lifecycle["crouched_retreat_completed"] and not robots_ready_to_stand:
            return (
                "exit_backward(distance_m=0.5, speed_mps=0.50, ramp_s=1.5)",
                "the crouched retreat is physically valid but lacks the pre-stand clearance reserve; continue backward together while crouched",
            )
        return (
            "exit_backward(distance_m=0.5, speed_mps=0.20, ramp_s=1.5)",
            "the target supports the released payload; both robots must retreat backward together while crouched",
        )
    if _payload_aligned_for_place_crouch(runtime):
        return (
            "place_crouch(place_height=0.55, ramp_s=2, hold_s=3)",
            "the carried payload footprint overlaps the target support region but remains too high for guarded release",
        )
    if not bool(payload_state.get("active", False)):
        return "place_payload_on_arms(hold_s=4)", "the deferred payload is not active"
    if not {"front", "back"}.issubset(supporters):
        return (
            "place_payload_on_arms(hold_s=4, force=True)",
            "the active payload lacks continuous dual support",
        )
    recovery = runtime.get("payload_inward_recovery", {})
    if isinstance(recovery, dict) and bool(recovery.get("available", False)):
        return (
            "recover_payload_inward(ramp_s=3, hold_s=2)",
            "the carried payload reached a preventive inward-recovery gate",
        )
    default = payload.get("default_transport", {})
    skill = str(default.get("skill", "carry_forward"))
    distance = float(default.get("distance_m", 2.0))
    speed = float(default.get("speed_mps", 0.50))
    ramp = float(default.get("ramp_s", 1.5))
    return (
        f"{skill}(distance_m={distance:.3f}, speed_mps={speed:.3f}, ramp_s={ramp:.3f})",
        (
            "the payload is active and supported; this is only a nominal seed from the "
            "configured static-skill trial, not a required route or formation"
        ),
    )


def _required_evolution_call(
    payload: dict[str, Any],
    *,
    previous_policy: PolicyScript | None = None,
    previous_report: EvaluationReport | None = None,
) -> tuple[str, str, str]:
    runtime = payload.get("observation", {}).get("runtime_state", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    payload_state = runtime.get("payload", {})
    payload_state = payload_state if isinstance(payload_state, dict) else {}
    supporters = _payload_supporting_robots(payload_state)
    if _payload_ready_for_release(runtime):
        return (
            "release_payload",
            "release_payload(ramp_s=4, hold_s=2)",
            "the payload COM projection is inside the physical target support region within the release height; execute the guarded crouch-and-release atomic transaction",
        )
    if _payload_is_released_on_target(runtime):
        lifecycle = _post_release_lifecycle(runtime)
        if _recover_stand_clearance_retry_required(
            previous_policy,
            previous_report,
        ):
            return (
                "exit_backward",
                "exit_backward(distance_m=0.5, speed_mps=0.50, ramp_s=1.5)",
                "the previous stand recovery physically crossed back inside the target-clearance boundary; restore the crouched checkpoint and add another synchronized backward segment before retrying stand recovery",
            )
        robots_clear = _robots_clear_of_released_payload(payload, runtime)
        robots_ready_to_stand = _robots_clear_of_released_payload(
            payload,
            runtime,
            include_pre_stand_reserve=True,
        )
        if (
            lifecycle["crouched_retreat_completed"]
            and robots_ready_to_stand
            and not lifecycle["stand_recovered"]
        ):
            return (
                "recover_stand",
                "recover_stand(height=0.80, ramp_s=6, hold_s=2)",
                "the accepted crouched backward retreat is complete; recover both robots upright before completion",
            )
        if lifecycle["stand_recovered"]:
            if not robots_clear:
                return (
                    "",
                    "",
                    "the robots recovered upright before reaching the required target clearance; restore the latest crouched checkpoint and continue synchronized backward retreat",
                )
            return (
                "",
                "",
                "the target supports the released payload and the mandatory crouched retreat plus upright recovery are complete",
            )
        if lifecycle["crouched_retreat_completed"] and not robots_ready_to_stand:
            return (
                "exit_backward",
                "exit_backward(distance_m=0.5, speed_mps=0.50, ramp_s=1.5)",
                "the accepted crouched retreat lacks enough reserve for stand-recovery drift; continue one synchronized backward segment before standing",
            )
        return (
            "exit_backward",
            "exit_backward(distance_m=0.5, speed_mps=0.20, ramp_s=1.5)",
            "the target supports the released payload; execute one identical synchronized backward retreat while both robots remain crouched",
        )
    if _payload_aligned_for_place_crouch(runtime):
        return (
            "place_crouch",
            "place_crouch(place_height=0.55, ramp_s=2, hold_s=3)",
            "the payload footprint overlaps the target support region; lower both robots together before guarded release",
        )
    if not bool(payload_state.get("active", False)):
        return (
            "place_payload_on_arms",
            "place_payload_on_arms(hold_s=4)",
            "the restored physical checkpoint contains a deferred payload",
        )
    if not {"front", "back"}.issubset(supporters):
        return (
            "",
            "",
            "the rejected endpoint lost support, but the active payload must be recovered "
            "through checkpoint rollback rather than reinitialized or teleported",
        )
    recovery = runtime.get("payload_inward_recovery", {})
    if isinstance(recovery, dict) and bool(recovery.get("available", False)):
        return (
            "recover_payload_inward",
            "recover_payload_inward(ramp_s=3, hold_s=2)",
            "the carried payload reached a preventive slip-or-tilt warning; "
            "perform the stationary synchronized arm recovery before more locomotion",
        )

    return (
        "",
        "",
        "no host-selected repair action; the model must infer the next experiment from evidence",
    )


def _lifecycle_forbidden_methods(
    payload: dict[str, Any],
    *,
    previous_policy: PolicyScript | None = None,
    previous_report: EvaluationReport | None = None,
) -> set[str]:
    """Keep lifecycle methods consistent with the current trusted physics."""

    admissibility = _lifecycle_admissibility(
        payload,
        previous_policy=previous_policy,
        previous_report=previous_report,
    )
    return set(admissibility["forbidden_methods"])


def _generation_lifecycle_forbidden_methods(
    generation_context: dict[str, Any],
) -> set[str]:
    """Use the authoritative lifecycle gate already frozen for this generation.

    The model-facing observation may omit task geometry that the Host used to
    build ``host_lifecycle_gate``. Recomputing admissibility from that reduced
    observation can therefore contradict the gate, for example by rejecting a
    gate-allowed placement crouch as payload reinitialization.
    """

    gate = generation_context.get("host_lifecycle_gate")
    if isinstance(gate, dict) and "forbidden_methods" in gate:
        methods = gate.get("forbidden_methods")
        if not isinstance(methods, (list, tuple, set, frozenset)):
            raise ModelAgentError(
                "host_lifecycle_gate.forbidden_methods must be a sequence"
            )
        return {
            method
            for item in methods
            if (method := str(item).strip())
        }
    return _lifecycle_forbidden_methods(generation_context)


def _lifecycle_admissibility(
    payload: dict[str, Any],
    *,
    previous_policy: PolicyScript | None = None,
    previous_report: EvaluationReport | None = None,
) -> dict[str, Any]:
    """Return the readiness-driven lifecycle action contract.

    Carry remains a method family. Once the payload COM projects inside the
    physical target support and only height/release remain, the Host owns the
    next primitive: ``place_crouch`` then ``release_payload``. Exact rejected
    ASTs stay checkpoint-local blockers; they do not reopen locomotion over
    the target disk.
    """

    runtime = payload.get("observation", {}).get("runtime_state", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    payload_state = runtime.get("payload", {})
    payload_state = payload_state if isinstance(payload_state, dict) else {}
    all_methods = set(MODEL_PLANNING_SCRIPT_API_METHODS)
    if "payload" not in runtime:
        return {
            "lifecycle_state": "unknown",
            "placement_ready": False,
            "release_completed": False,
            "retreat_completed": False,
            "upright_recovered": False,
            "allowed_action_families": sorted(all_methods),
            "forbidden_methods": [],
            "hard_required_method": "",
            "objective": "await an authoritative payload lifecycle observation",
            "setup_phase_completion_eligible": False,
            "setup_positive_route_memory_eligible": False,
        }
    active = bool(payload_state.get("active", False))
    supporters = _payload_supporting_robots(payload_state)
    placement_ready = _payload_ready_for_release(runtime)
    released = _payload_is_released_on_target(runtime)
    recovery = runtime.get("payload_inward_recovery", {})
    recovery_available = bool(
        isinstance(recovery, dict) and recovery.get("available", False)
    )
    lifecycle = _post_release_lifecycle(runtime)
    retreat_completed = bool(lifecycle["crouched_retreat_completed"])
    upright_recovered = bool(lifecycle["stand_recovered"])
    force_additional_retreat = _recover_stand_clearance_retry_required(
        previous_policy,
        previous_report,
    )
    robots_clear = _robots_clear_of_released_payload(payload, runtime)
    robots_ready_to_stand = _robots_clear_of_released_payload(
        payload,
        runtime,
        include_pre_stand_reserve=True,
    )

    hard_required_method = ""
    objective = ""
    if not active:
        state = "initialize"
        allowed = {"place_payload_on_arms"}
        hard_required_method = "place_payload_on_arms"
        objective = "activate the deferred payload on the live bilateral support"
    elif not {"front", "back"}.issubset(supporters) and not released:
        state = "rollback_required"
        allowed = set()
        objective = "restore the latest trusted checkpoint with bilateral support"
    elif released:
        if upright_recovered and robots_clear:
            state = "complete"
            allowed = {"stop_all"}
            objective = "seal the physically completed lifecycle"
        elif upright_recovered:
            state = "rollback_required"
            allowed = set()
            objective = "restore the latest crouched checkpoint before further retreat"
        elif (
            force_additional_retreat
            or not retreat_completed
            or not robots_ready_to_stand
        ):
            state = "retreat"
            allowed = set(_POST_RELEASE_EXIT_METHODS)
            objective = "increase robot clearance while preserving the released payload"
        else:
            state = "recover"
            # A further bounded crouched retreat remains a legal setup if the
            # exact stand-recovery transaction was rejected at this checkpoint.
            allowed = set(_POST_RELEASE_EXIT_METHODS) | {"recover_stand"}
            objective = "recover upright after establishing sufficient target clearance"
    elif recovery_available:
        state = "support_recovery"
        allowed = {"recover_payload_inward"}
        hard_required_method = "recover_payload_inward"
        objective = "restore the preventive bilateral payload support margin"
    else:
        aligned_for_place_crouch = _payload_aligned_for_place_crouch(runtime)
        if placement_ready:
            state = "release"
            allowed = {"release_payload"}
            hard_required_method = "release_payload"
            objective = "complete guarded release while maintaining support"
        elif aligned_for_place_crouch:
            state = "place"
            allowed = {"place_crouch"}
            hard_required_method = "place_crouch"
            objective = "lower both robots onto the target before guarded release"
        else:
            state = "carry"
            allowed = set(_PRE_RELEASE_SETUP_METHODS)
            objective = "advance the current route while maintaining bilateral support"

    allowed &= all_methods
    return {
        "lifecycle_state": state,
        "placement_ready": placement_ready,
        "release_completed": released,
        "retreat_completed": retreat_completed,
        "upright_recovered": upright_recovered,
        "allowed_action_families": sorted(allowed),
        "forbidden_methods": sorted(all_methods - allowed),
        "hard_required_method": hard_required_method,
        "objective": objective,
        "setup_phase_completion_eligible": False,
        "setup_positive_route_memory_eligible": False,
    }


def _recover_stand_clearance_retry_required(
    policy: PolicyScript | None,
    report: EvaluationReport | None,
) -> bool:
    """Keep retreating when measured stand drift invalidates nominal clearance."""

    if (
        policy is None
        or report is None
        or report.success
        or report.failure_type != "robots_not_clear"
    ):
        return False
    scripts = policy.robot_scripts or ()
    methods = {
        _first_policy_api_method(script.code)
        for script in scripts
        if _first_policy_api_method(script.code)
    }
    if not methods:
        method = _first_policy_api_method(policy.code)
        methods = {method} if method else set()
    return methods == {"recover_stand"}


def _release_preflight_planning_context(
    runtime: dict[str, Any],
    *,
    route_discovery: bool,
    lifecycle: dict[str, Any],
    final_approach_envelope: dict[str, Any],
) -> dict[str, Any]:
    """Expose release evidence only where target-local sensing is authorized."""

    if (
        route_discovery
        and str(lifecycle.get("lifecycle_state", "")) not in {"place", "release"}
        and not final_approach_envelope.get("active", False)
    ):
        # Early route discovery deliberately redacts the goal coordinates.
        # Adding a target correction vector here would bypass that boundary.
        return {}
    return release_preflight_evidence(runtime)


def _payload_ready_for_release(runtime: dict[str, Any]) -> bool:
    payload = runtime.get("payload", {})
    target = runtime.get("target", {})
    if not isinstance(payload, dict) or not isinstance(target, dict):
        return False
    if bool(payload.get("released", False)):
        return False
    supporters = _payload_supporting_robots(payload)
    if not bool(payload.get("active", False)) or not {"front", "back"}.issubset(
        supporters
    ):
        return False
    if _payload_contact_flag(
        payload, "floor_contact", "floor"
    ) or _payload_contact_values(
        payload,
        "wall_contacts",
        "walls",
    ):
        return False
    control = runtime.get("control_health", {})
    if isinstance(control, dict) and control and not bool(control.get("healthy", False)):
        return False
    support_polygon = payload.get("support_polygon", {})
    if (
        isinstance(support_polygon, dict)
        and support_polygon.get("available", False)
        and not support_polygon.get("inside", False)
    ):
        return False
    try:
        tilt_deg = float(payload.get("tilt_deg", 0.0))
    except (TypeError, ValueError):
        return False
    if not isfinite(tilt_deg) or tilt_deg >= PAYLOAD_TOP_FACE_UP_LIMIT_DEG:
        return False
    try:
        xy_error = hypot(
            float(payload.get("x")) - float(target.get("center_x_m")),
            float(payload.get("y")) - float(target.get("center_y_m")),
        )
        drop = float(payload.get("z")) - float(target.get("payload_center_target_z_m"))
        target_support_radius_m = max(
            0.0,
            float(target.get("radius_m", DEFAULT_TARGET_XY_TOLERANCE_M)),
        )
    except (TypeError, ValueError):
        return False
    # Keep the model-side lifecycle decision identical to the physical gate in
    # wait_for_payload_release_ready().  A fixed planning tolerance here used
    # to route physically releasable states through an unnecessary crouch,
    # which could shed one side of support before the guarded release ran.
    allowed_xy_error_m = (
        target_support_radius_m
        if target_support_radius_m > 0.0
        else DEFAULT_TARGET_XY_TOLERANCE_M
    )
    return (
        xy_error <= allowed_xy_error_m
        and -0.03 <= drop <= _MAX_RELEASE_ATOMIC_DROP_M
        and (
            drop <= _MAX_PRELOWERED_RELEASE_DROP_M
            or tilt_deg <= _MAX_UNLOWERED_RELEASE_TILT_DEG
        )
    )


def _payload_aligned_for_place_crouch(runtime: dict[str, Any]) -> bool:
    """Return whether only vertical lowering remains before guarded release."""

    payload = runtime.get("payload", {})
    target = runtime.get("target", {})
    if not isinstance(payload, dict) or not isinstance(target, dict):
        return False
    supporters = _payload_supporting_robots(payload)
    if (
        not bool(payload.get("active", False))
        or not {"front", "back"}.issubset(supporters)
        or _payload_contact_flag(payload, "target_contact", "target")
        or _payload_contact_flag(payload, "floor_contact", "floor")
        or bool(_payload_contact_values(payload, "wall_contacts", "walls"))
    ):
        return False
    try:
        center_error_m = hypot(
            float(payload.get("x")) - float(target.get("center_x_m")),
            float(payload.get("y")) - float(target.get("center_y_m")),
        )
        drop = float(payload.get("z")) - float(target.get("payload_center_target_z_m"))
        target_support_radius_m = max(
            0.0,
            float(target.get("radius_m", DEFAULT_TARGET_XY_TOLERANCE_M)),
        )
    except (TypeError, ValueError):
        return False
    allowed_xy_error_m = (
        target_support_radius_m
        if target_support_radius_m > 0.0
        else DEFAULT_TARGET_XY_TOLERANCE_M
    )
    # Crouching changes only height. An edge intersection between a long box
    # footprint and the cylinder cannot repair a horizontally unsupported COM
    # and therefore must not force the placement lifecycle to begin.
    return center_error_m <= allowed_xy_error_m and drop > 0.30


def _payload_target_footprint_gap_m(
    runtime: dict[str, Any],
) -> float | None:
    """Measure the planar gap between the oriented box and target cylinder."""

    payload = runtime.get("payload", {})
    target = runtime.get("target", {})
    if not isinstance(payload, dict) or not isinstance(target, dict):
        return None
    try:
        delta_x = float(target.get("center_x_m")) - float(payload.get("x"))
        delta_y = float(target.get("center_y_m")) - float(payload.get("y"))
        yaw = (
            float(payload.get("yaw_rad"))
            if payload.get("yaw_rad") is not None
            else radians(float(payload.get("yaw_deg")))
        )
        radius_m = float(target.get("radius_m", 0.30))
    except (TypeError, ValueError):
        return None
    half_extents = payload.get("half_extents_xyz_m", (0.90, 0.05, 0.05))
    if not isinstance(half_extents, (list, tuple)) or len(half_extents) < 2:
        half_extents = (0.90, 0.05, 0.05)
    try:
        half_length_m = abs(float(half_extents[0]))
        half_width_m = abs(float(half_extents[1]))
    except (TypeError, ValueError):
        return None
    if not all(
        isfinite(value)
        for value in (
            delta_x,
            delta_y,
            yaw,
            radius_m,
            half_length_m,
            half_width_m,
        )
    ):
        return None
    along_m = abs(delta_x * cos(yaw) + delta_y * sin(yaw))
    cross_m = abs(-delta_x * sin(yaw) + delta_y * cos(yaw))
    distance_to_box_m = hypot(
        max(0.0, along_m - half_length_m),
        max(0.0, cross_m - half_width_m),
    )
    return max(0.0, distance_to_box_m - max(0.0, radius_m))


def _payload_is_released_on_target(runtime: dict[str, Any]) -> bool:
    payload = runtime.get("payload", {})
    if not isinstance(payload, dict):
        return False
    lifecycle = _post_release_lifecycle(runtime)
    return bool(
        payload.get("active", False)
        and (bool(payload.get("released", False)) or lifecycle["payload_released"])
        and _payload_contact_flag(payload, "target_contact", "target")
        and not _payload_contact_flag(payload, "floor_contact", "floor")
        and not _payload_contact_values(payload, "wall_contacts", "walls")
    )


def _payload_contact_values(
    payload: dict[str, Any],
    compact_key: str,
    contact_key: str,
) -> tuple[Any, ...]:
    """Read one payload contact field from compact or raw runtime state."""

    value = payload.get(compact_key)
    if value is None:
        contacts = payload.get("contacts", {})
        contacts = contacts if isinstance(contacts, dict) else {}
        value = contacts.get(contact_key, ())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,) if value else ()


def _payload_contact_flag(
    payload: dict[str, Any],
    compact_key: str,
    contact_key: str,
) -> bool:
    value = payload.get(compact_key)
    if value is not None:
        return bool(value)
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    return bool(contacts.get(contact_key, False))


def _payload_supporting_robots(payload: dict[str, Any]) -> set[str]:
    return {
        str(robot_id)
        for robot_id in _payload_contact_values(
            payload,
            "supporting_robots",
            "support_robots",
        )
    }


def _post_release_lifecycle(runtime: dict[str, Any]) -> dict[str, bool]:
    value = runtime.get("post_release_lifecycle", {})
    lifecycle = value if isinstance(value, dict) else {}
    return {
        "payload_released": bool(lifecycle.get("payload_released", False)),
        "crouch_latched": bool(lifecycle.get("crouch_latched", False)),
        "crouched_retreat_completed": bool(
            lifecycle.get("crouched_retreat_completed", False)
        ),
        "stand_recovered": bool(lifecycle.get("stand_recovered", False)),
    }


def _robots_clear_of_released_payload(
    payload: dict[str, Any],
    runtime: dict[str, Any],
    *,
    include_pre_stand_reserve: bool = False,
) -> bool:
    """Return true only when both live robot roots clear the final target."""

    target = runtime.get("target", {})
    robots = runtime.get("robots", {})
    if not isinstance(target, dict) or not isinstance(robots, dict):
        return True
    try:
        target_x = float(target["center_x_m"])
        target_y = float(target["center_y_m"])
    except (KeyError, TypeError, ValueError):
        return True

    task_context = payload.get("task", {})
    task_context = task_context if isinstance(task_context, dict) else {}
    success_contract = task_context.get("final_success_contract", {})
    success_contract = success_contract if isinstance(success_contract, dict) else {}
    raw_clearance = task_context.get(
        "robot_exit_min_distance_from_target_m",
        success_contract.get(
            "robot_exit_min_distance_from_target_m",
            DEFAULT_ROBOT_EXIT_MIN_DISTANCE_M,
        ),
    )
    try:
        required_clearance = max(0.0, float(raw_clearance))
    except (TypeError, ValueError):
        required_clearance = DEFAULT_ROBOT_EXIT_MIN_DISTANCE_M
    if include_pre_stand_reserve:
        raw_margin = task_context.get(
            "robot_pre_stand_clearance_margin_m",
            success_contract.get(
                "robot_pre_stand_clearance_margin_m",
                DEFAULT_ROBOT_PRE_STAND_CLEARANCE_MARGIN_M,
            ),
        )
        try:
            required_clearance += max(0.0, float(raw_margin))
        except (TypeError, ValueError):
            required_clearance += DEFAULT_ROBOT_PRE_STAND_CLEARANCE_MARGIN_M

    distances: list[float] = []
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            return True
        try:
            distances.append(
                hypot(
                    float(robot["x"]) - target_x,
                    float(robot["y"]) - target_y,
                )
            )
        except (KeyError, TypeError, ValueError):
            return True
    return bool(distances) and min(distances) + 1.0e-9 >= required_clearance


def _single_step_code_template(call_expression: str) -> str:
    return f"def policy(api):\n    api.{call_expression}\n    api.stop_all()"


def _first_policy_api_method(code: str) -> str:
    match = re.search(r"api\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
    if not match or match.group(1) == "stop_all":
        return ""
    return match.group(1)


def _first_task_call_source(code: str) -> str:
    """Render the executable non-stop API call without model-authored prose."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if not isinstance(owner, ast.Name) or owner.id != "api":
            continue
        if node.func.attr == "stop_all":
            continue
        return ast.unparse(node)
    return ""


def _transport_parameters_from_source(
    source: str,
    *,
    method: str,
) -> tuple[float, float] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            not isinstance(owner, ast.Name)
            or owner.id != "api"
            or node.func.attr != method
        ):
            continue
        keyword_values = {item.arg: item.value for item in node.keywords}
        try:
            return (
                float(ast.literal_eval(keyword_values["distance_m"])),
                float(ast.literal_eval(keyword_values["speed_mps"])),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _task_call_signature(code: str) -> str:
    """Return a semantic AST signature for the one non-stop SDK call."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            not isinstance(owner, ast.Name)
            or owner.id != "api"
            or node.func.attr == "stop_all"
        ):
            continue
        return ast.dump(node, annotate_fields=True, include_attributes=False)
    return ""


def _first_task_call_literal_arguments(code: str) -> dict[str, object]:
    """Return literal keyword arguments for the single model-visible skill call."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            not isinstance(owner, ast.Name)
            or owner.id != "api"
            or node.func.attr == "stop_all"
        ):
            continue
        values: dict[str, object] = {}
        try:
            for keyword in node.keywords:
                if keyword.arg is None:
                    return {}
                values[keyword.arg] = ast.literal_eval(keyword.value)
        except (TypeError, ValueError):
            return {}
        return values
    return {}


_LEGACY_LINEAR_DIRECTIONS = {
    "carry_forward": "forward",
    "carry_backward": "backward",
    "carry_lateral_left": "left",
    "carry_lateral_right": "right",
}


def _linear_objective_from_code(code: str) -> str:
    method = _first_policy_api_method(code)
    if method in _LEGACY_LINEAR_DIRECTIONS:
        return _LEGACY_LINEAR_DIRECTIONS[method]
    if method == COORDINATED_LINEAR_TRANSPORT:
        direction = str(_first_task_call_literal_arguments(code).get("direction", ""))
        return (
            direction if direction in {"forward", "backward", "left", "right"} else ""
        )
    return ""


def _policy_linear_objective(policy: PolicyScript) -> str:
    """Normalize legacy and coordination-layer linear calls to one objective."""

    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="team",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="team",
        ),
    )
    objectives: set[str] = set()
    for script in scripts:
        direction = _linear_objective_from_code(script.code)
        if not direction:
            return ""
        objectives.add(direction)
    return next(iter(objectives)) if len(objectives) == 1 else ""


def _policy_strategy_budget_signature(policy: PolicyScript) -> str:
    strategy = policy_strategy_signature(policy)
    if not strategy:
        return ""
    objective = policy_motion_objective_signature(policy)
    return f"{strategy}|{objective}" if objective else strategy


_TRANSPORT_METHODS = frozenset(
    {
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
    }
) | frozenset({COORDINATED_LINEAR_TRANSPORT})


def _policy_uses_linear_transport(policy: PolicyScript) -> bool:
    return any(
        method in _TRANSPORT_METHODS
        for method, _distance_m, _speed_mps in _policy_transport_magnitudes(policy)
    )


def _policy_is_final_approach(policy: PolicyScript) -> bool:
    phase_ids = [
        str(script.phase_id).strip().lower()
        for script in policy.robot_scripts
        if str(script.phase_id).strip()
    ]
    return any(
        "final_straight" in phase_id or "final_approach" in phase_id
        for phase_id in phase_ids
    )


def _formation_sync_probe_distance_m(
    policy: PolicyScript,
    *,
    force_short: bool = False,
) -> float:
    """Use a short direction-matched probe near a target or settle abort."""

    if not force_short and not _policy_is_final_approach(policy):
        return FORMATION_SYNC_REPAIR_PROBE_DISTANCE_M
    arguments = [
        _first_task_call_literal_arguments(script.code)
        for script in policy.robot_scripts
    ]
    try:
        failed_distances = {float(item["distance_m"]) for item in arguments}
    except (KeyError, TypeError, ValueError):
        failed_distances = set()
    failed_distance_m = (
        next(iter(failed_distances))
        if len(failed_distances) == 1
        else FORMATION_SYNC_REPAIR_PROBE_DISTANCE_M
    )
    lower = [
        distance_m
        for distance_m in FINAL_APPROACH_LINEAR_DISTANCE_CHOICES_M
        if distance_m < failed_distance_m - 1.0e-9
    ]
    return max(
        lower,
        default=min(FINAL_APPROACH_LINEAR_DISTANCE_CHOICES_M),
    )


def _validate_formation_sync_repair_probe_policy(
    policy: PolicyScript,
    *,
    gate: dict[str, Any],
) -> None:
    """Require one bounded physical probe before long translation is reopened."""

    if gate.get("probe_unavailable", False):
        scripts = policy.robot_scripts or (
            RobotPolicyScript(
                robot_id="team",
                script_id=policy.script_id,
                code=policy.code,
                synchronization_group="legacy_team",
            ),
        )
        methods = {_first_policy_api_method(script.code) for script in scripts}
        required_repair = str(gate.get("required_repair", "")).strip()
        allowed_methods = (
            {COORDINATED_ALIGNMENT}
            if required_repair == COORDINATED_ALIGNMENT
            else {COORDINATED_ALIGNMENT, COORDINATED_PIVOT}
        )
        if methods and methods <= allowed_methods:
            return
        raise ModelAgentError(
            "formation_sync_repair linear probe is unavailable under the exhausted "
            "final-approach envelope; executable policy must use the required "
            f"geometry repair {required_repair or 'alignment_or_pivot'}"
        )

    if not _policy_uses_linear_transport(policy):
        scripts = policy.robot_scripts or (
            RobotPolicyScript(
                robot_id="team",
                script_id=policy.script_id,
                code=policy.code,
                synchronization_group="legacy_team",
            ),
        )
        methods = {_first_policy_api_method(script.code) for script in scripts}
        if int(gate.get("failure_count", 0)) >= 2 and methods <= {
            COORDINATED_ALIGNMENT,
            COORDINATED_PIVOT,
        }:
            return
        raise ModelAgentError(
            "formation_sync_repair requires one auditable linear probe call for "
            "each robot; a non-linear coordination repair is allowed only after "
            "two same-checkpoint probe failures"
        )

    probe_contract = gate.get("probe_contract", {})
    probe_contract = probe_contract if isinstance(probe_contract, dict) else {}
    required_distance_m = float(
        probe_contract.get(
            "required_distance_m",
            FORMATION_SYNC_REPAIR_PROBE_DISTANCE_M,
        )
    )
    required_objective = str(probe_contract.get("semantic_objective", "")).strip()
    candidate_objective = _policy_linear_objective(policy)
    if required_objective and candidate_objective != required_objective:
        raise ModelAgentError(
            "formation_sync_repair is direction-specific: the failed "
            f"{required_objective!r} response cannot be verified by "
            f"{candidate_objective!r} motion"
        )
    magnitudes = _policy_transport_magnitudes(policy)
    expected_count = 2 if policy.robot_scripts else 1
    if len(magnitudes) != expected_count:
        raise ModelAgentError(
            "formation_sync_repair requires one auditable linear probe call for "
            "each robot"
        )
    invalid_methods = sorted(
        {
            method
            for method, _distance_m, _speed_mps in magnitudes
            if method != COORDINATED_LINEAR_TRANSPORT
        }
    )
    if invalid_methods:
        raise ModelAgentError(
            "formation_sync_repair must use coordinated_linear_transport, not "
            + ", ".join(invalid_methods)
        )
    invalid_distances = sorted(
        {
            distance_m
            for _method, distance_m, _speed_mps in magnitudes
            if abs(distance_m - required_distance_m) > 1.0e-9
        }
    )
    if invalid_distances:
        rendered = ", ".join(f"{value:g}" for value in invalid_distances)
        raise ModelAgentError(
            "formation_sync_repair has not been physically verified, so loaded "
            f"linear transport must be exactly the {required_distance_m:g}m "
            f"verification probe; generated distance_m={rendered}"
        )


def _formation_sync_repair_verification_contract(
    gate: dict[str, Any],
    *,
    policy: PolicyScript,
) -> dict[str, Any]:
    probe_contract = gate.get("probe_contract", {})
    probe_contract = probe_contract if isinstance(probe_contract, dict) else {}
    return {
        "kind": "formation_sync_repair_probe",
        "source": "linear_formation_desync",
        "scope_checkpoint_id": str(gate.get("current_checkpoint_id", "")),
        "primitive": COORDINATED_LINEAR_TRANSPORT,
        "semantic_objective": str(
            probe_contract.get(
                "semantic_objective",
                _policy_linear_objective(policy),
            )
        ).strip(),
        "required_distance_m": float(
            probe_contract.get(
                "required_distance_m",
                FORMATION_SYNC_REPAIR_PROBE_DISTANCE_M,
            )
        ),
        "minimum_actual_progress_m": float(
            probe_contract.get(
                "minimum_actual_progress_m",
                FORMATION_SYNC_REPAIR_MIN_PROGRESS_M,
            )
        ),
        "maximum_normalized_displacement_mismatch": float(
            probe_contract.get(
                "maximum_normalized_displacement_mismatch",
                FORMATION_SYNC_REPAIR_MAX_NORMALIZED_MISMATCH,
            )
        ),
        "trigger": (
            dict(gate.get("trigger", {}))
            if isinstance(gate.get("trigger"), dict)
            else {}
        ),
    }


def _validate_final_approach_strategy_availability(
    *,
    method: str,
    envelope: dict[str, Any] | None,
    allow_turn_clearance_replan: bool = False,
) -> None:
    """Reject exhausted target-local linear motion before Coder invocation."""

    contract = envelope if isinstance(envelope, dict) else {}
    if allow_turn_clearance_replan or not (
        contract.get("active", False)
        and contract.get("exhausted", False)
        and method == COORDINATED_LINEAR_TRANSPORT
    ):
        return
    raise ModelRoleError(
        "final-approach short transport envelope is exhausted at this "
        "checkpoint; coordinated_linear_transport is unavailable until a "
        "verifier-accepted geometry-changing pivot or alignment creates a new "
        "trusted checkpoint"
    )


def _validate_final_approach_transport_policy(
    policy: PolicyScript,
    *,
    envelope: dict[str, Any] | None,
    allow_turn_clearance_replan: bool = False,
) -> None:
    """Enforce the target-local Coder magnitude contract."""

    contract = envelope if isinstance(envelope, dict) else {}
    if allow_turn_clearance_replan or not contract.get("active", False):
        return
    methods = {_first_policy_api_method(script.code) for script in policy.robot_scripts}
    if methods != {COORDINATED_LINEAR_TRANSPORT}:
        return
    if contract.get("exhausted", False):
        raise ModelAgentError(
            "final-approach short transport envelope is exhausted at this "
            "checkpoint; choose a geometry-changing strategy"
        )
    required_distance_m = float(contract["required_distance_m"])
    required_speed_mps = float(
        contract.get("required_speed_mps", VERIFIED_LINEAR_SPEED_MPS)
    )
    required_ramp_s = float(contract.get("required_ramp_s", VERIFIED_LINEAR_RAMP_S))
    violations: list[str] = []
    for script in policy.robot_scripts:
        arguments = _first_task_call_literal_arguments(script.code)
        try:
            distance_m = float(arguments["distance_m"])
            speed_mps = float(arguments["speed_mps"])
            ramp_s = float(arguments["ramp_s"])
        except (KeyError, TypeError, ValueError):
            violations.append(f"{script.robot_id} requires literal distance/speed/ramp")
            continue
        if abs(distance_m - required_distance_m) > 1.0e-9:
            violations.append(
                f"{script.robot_id} distance_m must be {required_distance_m:g}"
            )
        if abs(speed_mps - required_speed_mps) > 1.0e-9:
            violations.append(
                f"{script.robot_id} speed_mps must be {required_speed_mps:g}"
            )
        if abs(ramp_s - required_ramp_s) > 1.0e-9:
            violations.append(f"{script.robot_id} ramp_s must be {required_ramp_s:g}")
    if violations:
        raise ModelAgentError(
            "final-approach transport must use the target-local physical "
            "envelope; " + "; ".join(violations)
        )


_UNSAFE_MAGNITUDE_FAILURES = frozenset(
    {
        "robot_fall",
        "payload_drop",
        "payload_not_supported",
        "payload_unstable",
        "payload_slip",
        "formation_desync",
        "payload_wall_contact",
    }
)

_STRUCTURAL_SCRIPT_FAILURES = frozenset(
    {
        "formation_desync",
        "payload_drop",
        "payload_not_supported",
        "payload_slip",
        "payload_wall_contact",
        "robot_fall",
    }
)

_FIXED_RAMP_RETRY_FAILURES = frozenset(
    {"formation_desync", "no_payload_progress", "payload_not_supported"}
)


def _unsafe_retry_transport_limits(
    policy: PolicyScript,
    *,
    failure_type: str,
) -> tuple[float, float] | None:
    """Bound the next model draft after a physically unsafe transport failure."""

    if failure_type not in _UNSAFE_MAGNITUDE_FAILURES:
        return None
    magnitudes = _policy_transport_magnitudes(policy)
    if not magnitudes:
        return None
    return (
        max(item[1] for item in magnitudes),
        max(item[2] for item in magnitudes),
    )


def _validate_unsafe_retry_magnitude(
    policy: PolicyScript,
    *,
    max_distance_m: float,
    max_speed_mps: float,
) -> None:
    violations: list[str] = []
    for method, distance_m, speed_mps in _policy_transport_magnitudes(policy):
        if distance_m > max_distance_m + 1.0e-9:
            violations.append(
                f"api.{method} increases distance_m from the unsafe bound "
                f"{max_distance_m:g} to {distance_m:g}"
            )
        if speed_mps > max_speed_mps + 1.0e-9:
            violations.append(
                f"api.{method} increases speed_mps from the unsafe bound "
                f"{max_speed_mps:g} to {speed_mps:g}"
            )
    if violations:
        raise ModelAgentError(
            "the previous physical rollout was unsafe, so its immediate retry may not increase "
            "transport magnitude; "
            + "; ".join(violations)
            + ". Select a safer bounded experiment from the evidence. The host has not selected "
            "a skill, direction, route, formation, or coordination pattern."
        )


def _policy_transport_magnitudes(
    policy: PolicyScript,
) -> tuple[tuple[str, float, float], ...]:
    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="front",
            script_id=f"front_{policy.script_id}",
            code=policy.code,
            synchronization_group="legacy_team",
        ),
    )
    values: list[tuple[str, float, float]] = []
    for script in scripts:
        try:
            tree = ast.parse(script.code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            owner = node.func.value
            method = node.func.attr
            if (
                not isinstance(owner, ast.Name)
                or owner.id != "api"
                or method not in _TRANSPORT_METHODS
            ):
                continue
            keyword_values = {
                item.arg: item.value
                for item in node.keywords
                if item.arg in {"distance_m", "speed_mps"}
            }
            try:
                distance_m = float(ast.literal_eval(keyword_values["distance_m"]))
                speed_mps = float(ast.literal_eval(keyword_values["speed_mps"]))
            except (KeyError, TypeError, ValueError):
                continue
            values.append((method, distance_m, speed_mps))
    return tuple(values)


def _validate_measured_sync_compensation(
    policy: PolicyScript,
    *,
    constraint: dict[str, Any],
) -> None:
    """Require model-chosen commands to respect the observed executor ordering."""

    method = str(constraint.get("method", ""))
    source_motion_objective = str(constraint.get("motion_objective", "")).strip()
    candidate_motion_objective = _policy_linear_objective(policy)
    if (
        source_motion_objective
        and candidate_motion_objective != source_motion_objective
    ):
        raise ModelAgentError(
            "measured sync compensation is direction-specific: source objective "
            f"{source_motion_objective!r}, candidate objective "
            f"{candidate_motion_objective!r}"
        )
    farther_robot = str(constraint.get("farther_robot", ""))
    nearer_robot = str(constraint.get("nearer_robot", ""))
    parameters = _transport_parameters_by_robot(policy, method=method)
    if set(parameters) != {"front", "back"}:
        raise ModelAgentError(
            f"measured sync compensation requires front and back to both call api.{method} "
            "with literal distance_m and speed_mps"
        )
    farther_distance, farther_speed = parameters[farther_robot]
    nearer_distance, nearer_speed = parameters[nearer_robot]
    required_speed = float(
        constraint.get("required_speed_mps", max(farther_speed, nearer_speed))
    )
    required_distance = float(
        constraint.get("required_distance_m", max(farther_distance, nearer_distance))
    )
    required_ramp = float(constraint.get("required_ramp_s", 1.5))
    _validate_transport_ramp(policy, required_ramp_s=required_ramp)
    if (
        farther_speed < nearer_speed - 1.0e-9
        and abs(nearer_speed - required_speed) <= 1.0e-9
        and abs(farther_distance - required_distance) <= 1.0e-9
        and abs(nearer_distance - required_distance) <= 1.0e-9
    ):
        return
    raise ModelAgentError(
        "measured sync compensation was not implemented in executable code: "
        f"{farther_robot} moved {constraint.get('farther_displacement_m')} m while "
        f"{nearer_robot} moved {constraint.get('nearer_displacement_m')} m in the rejected "
        f"api.{method} rollout, so {farther_robot} speed_mps must be strictly smaller than "
        f"{nearer_robot} speed_mps={required_speed:g} while both distance_m values remain "
        f"at {required_distance:g}; "
        f"candidate values were {farther_robot}=({farther_distance:g}, {farther_speed:g}) and "
        f"{nearer_robot}=({nearer_distance:g}, {nearer_speed:g}). Choose the exact bounded "
        f"compensation values yourself and keep ramp_s={required_ramp:g}."
    )


def _validate_selected_sync_parameters(
    policy: PolicyScript,
    *,
    method: str,
    selected: dict[str, Any],
) -> None:
    parameters = _transport_parameters_by_robot(policy, method=method)
    if set(parameters) != {"front", "back"}:
        raise ModelAgentError(
            "the executable scripts do not contain both model-selected sync parameter calls"
        )
    mismatches: list[str] = []
    for robot_id in ("front", "back"):
        actual_distance, actual_speed = parameters[robot_id]
        expected_distance = float(selected[f"{robot_id}_distance_m"])
        expected_speed = float(selected[f"{robot_id}_speed_mps"])
        if abs(actual_distance - expected_distance) > 1.0e-9:
            mismatches.append(
                f"{robot_id} distance_m={actual_distance:g}, model selected {expected_distance:g}"
            )
        if abs(actual_speed - expected_speed) > 1.0e-9:
            mismatches.append(
                f"{robot_id} speed_mps={actual_speed:g}, model selected {expected_speed:g}"
            )
    if mismatches:
        raise ModelAgentError(
            "code synthesis must preserve the same model's separately selected front/back "
            "calibration values exactly; "
            + "; ".join(mismatches)
            + ". Write separate front and back code strings without averaging or copying."
        )


def _transport_parameters_by_robot(
    policy: PolicyScript,
    *,
    method: str,
) -> dict[str, tuple[float, float]]:
    values: dict[str, tuple[float, float]] = {}
    for script in policy.robot_scripts:
        try:
            tree = ast.parse(script.code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            owner = node.func.value
            if (
                not isinstance(owner, ast.Name)
                or owner.id != "api"
                or node.func.attr != method
            ):
                continue
            keyword_values = {item.arg: item.value for item in node.keywords}
            try:
                values[script.robot_id] = (
                    float(ast.literal_eval(keyword_values["distance_m"])),
                    float(ast.literal_eval(keyword_values["speed_mps"])),
                )
            except (KeyError, TypeError, ValueError):
                pass
            break
    return values


def _validate_transport_ramp(
    policy: PolicyScript,
    *,
    required_ramp_s: float,
) -> None:
    violations: list[str] = []
    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="front",
            script_id=f"front_{policy.script_id}",
            code=policy.code,
            synchronization_group="legacy_team",
        ),
    )
    for script in scripts:
        try:
            tree = ast.parse(script.code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            owner = node.func.value
            method = node.func.attr
            if (
                not isinstance(owner, ast.Name)
                or owner.id != "api"
                or method not in _TRANSPORT_METHODS
            ):
                continue
            ramp_node = next(
                (item.value for item in node.keywords if item.arg == "ramp_s"),
                None,
            )
            try:
                ramp_s = float(ast.literal_eval(ramp_node))
            except (TypeError, ValueError):
                violations.append(f"api.{method} must provide a literal ramp_s")
                continue
            if abs(ramp_s - required_ramp_s) > 1.0e-9:
                violations.append(
                    f"api.{method} uses ramp_s={ramp_s:g}, expected {required_ramp_s:g}"
                )
    if violations:
        raise ModelAgentError(
            "ramp_s is a zero-speed planner hold, not an acceleration profile, and changing it "
            "cannot repair gait onset, team desynchronization, or missing progress; "
            + "; ".join(violations)
            + ". Keep the verified ramp and change an evidence-grounded executable decision instead."
        )


def _policy_transaction_signature(policy: PolicyScript) -> str:
    """Capture executable calls plus phase/contact/synchronization decisions."""

    if not policy.robot_scripts:
        inferred_intent = _inferred_contact_intent(policy.code)
        payload = [
            {
                "robot_id": robot_id,
                "dispatch_group": 0,
                "coordination_mode": "sync",
                "contact_intent": inferred_intent,
                "task_call": _task_call_signature(policy.code),
            }
            for robot_id in ("front", "back")
        ]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    group_indices: dict[str, int] = {}
    payload = [
        {
            "robot_id": script.robot_id,
            "dispatch_group": group_indices.setdefault(
                script.synchronization_group,
                len(group_indices),
            ),
            "contact_intent": (
                _inferred_contact_intent(script.code)
                if script.contact_intent == "unspecified"
                else script.contact_intent
            ),
            "coordination_mode": script.coordination_mode,
            "task_call": _task_call_signature(script.code),
        }
        for script in policy.robot_scripts
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


_COMPACT_BLOCKER_ARG_ORDER = (
    "direction",
    "distance_m",
    "angle_deg",
    "delta",
    "axis",
    "place_height",
    "yaw_deg",
)


def _render_compact_blocker_arg(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        rendered = f"{value:g}"
        return rendered
    return str(value)


def _compact_blocked_transaction_signature(
    policy: PolicyScript,
    *,
    failure_type: str = "",
) -> str:
    """Readable exact-blocker identity: primitive, objective, mode, args, failure."""

    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="front",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="legacy_team",
            coordination_mode="sync",
        ),
        RobotPolicyScript(
            robot_id="back",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="legacy_team",
            coordination_mode="sync",
        ),
    )
    rendered_calls: list[str] = []
    for script in sorted(scripts, key=lambda item: item.robot_id):
        method = _first_policy_api_method(script.code)
        arguments = _first_task_call_literal_arguments(script.code)
        ordered: list[str] = []
        seen: set[str] = set()
        for name in _COMPACT_BLOCKER_ARG_ORDER:
            if name in arguments:
                ordered.append(
                    f"{name}={_render_compact_blocker_arg(arguments[name])}"
                )
                seen.add(name)
        for name, value in arguments.items():
            if name in seen or name in {"speed_mps", "ramp_s", "duration_s"}:
                continue
            ordered.append(f"{name}={_render_compact_blocker_arg(value)}")
        call = method or "unknown"
        if ordered:
            call = f"{call}({', '.join(ordered)})"
        else:
            call = f"{call}()"
        mode = str(script.coordination_mode or "sync").strip() or "sync"
        rendered_calls.append(f"{call}, mode={mode}")
    unique_calls = list(dict.fromkeys(rendered_calls))
    signature = unique_calls[0] if len(unique_calls) == 1 else "; ".join(
        f"{script.robot_id}:{call}"
        for script, call in zip(
            sorted(scripts, key=lambda item: item.robot_id),
            rendered_calls,
        )
    )
    failure = str(failure_type).strip()
    if failure and failure not in {"", "none"}:
        return f"{signature} → rejected: {failure}"
    return signature


def _blocked_transaction_fingerprint(signature: str) -> str:
    """Expose a stable blocked-AST identity without leaking a replacement action."""

    digest = hashlib.sha256(str(signature).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:20]}"


def _policy_alignment_structure_signature(policy: PolicyScript) -> str:
    """Describe an alignment transaction without its numeric magnitudes."""

    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="front",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="legacy_team",
            coordination_mode="sync",
        ),
        RobotPolicyScript(
            robot_id="back",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="legacy_team",
            coordination_mode="sync",
        ),
    )
    structure: list[dict[str, str]] = []
    for script in scripts:
        if _first_policy_api_method(script.code) != COORDINATED_ALIGNMENT:
            return ""
        arguments = _first_task_call_literal_arguments(script.code)
        axis = str(arguments.get("axis", "")).strip()
        try:
            delta = float(arguments["delta"])
        except (KeyError, TypeError, ValueError):
            return ""
        if not axis or not isfinite(delta):
            return ""
        motion_role = (
            "hold"
            if abs(delta) <= 1.0e-9
            else ("positive" if delta > 0.0 else "negative")
        )
        structure.append(
            {
                "robot_id": script.robot_id,
                "coordination_mode": script.coordination_mode,
                "axis": axis,
                "motion_role": motion_role,
            }
        )
    return json.dumps(
        sorted(structure, key=lambda item: item["robot_id"]),
        ensure_ascii=False,
        sort_keys=True,
    )


def _policy_alignment_reference_corrector_pattern(
    policy: PolicyScript,
) -> tuple[str, str] | None:
    """Return immutable reference/corrector IDs without encoding delta sign."""

    roles: dict[str, str] = {}
    for script in policy.robot_scripts:
        if _first_policy_api_method(script.code) != COORDINATED_ALIGNMENT:
            return None
        delta = _safe_float(
            _first_task_call_literal_arguments(script.code).get("delta")
        )
        if delta is None:
            return None
        roles[script.robot_id] = "reference" if abs(delta) <= 1.0e-9 else "corrector"
    if set(roles) != {"front", "back"} or set(roles.values()) != {
        "reference",
        "corrector",
    }:
        return None
    reference = next(
        robot_id for robot_id, role in roles.items() if role == "reference"
    )
    corrector = next(
        robot_id for robot_id, role in roles.items() if role == "corrector"
    )
    return reference, corrector


def _policy_arc_travel_direction(policy: PolicyScript) -> str:
    """Return one shared body-frame travel axis for an arc or pivot policy."""

    scripts = policy.robot_scripts or (
        RobotPolicyScript(
            robot_id="team",
            script_id=policy.script_id,
            code=policy.code,
            synchronization_group="legacy_team",
        ),
    )
    directions: set[str] = set()
    for script in scripts:
        method = _first_policy_api_method(script.code)
        if method not in {COORDINATED_ARC_TRANSPORT, COORDINATED_PIVOT}:
            return ""
        arguments = _first_task_call_literal_arguments(script.code)
        directions.add(str(arguments.get("travel_direction", "forward")))
    return next(iter(directions)) if len(directions) == 1 else ""


def _validate_failure_level_repair(
    previous: PolicyScript,
    candidate: PolicyScript,
    *,
    failure_level: str,
    failure_type: str,
    blocked_strategy_signatures: set[str],
    allow_measured_sync_compensation: bool = False,
    formation_sync_probe_distance_m: float | None = None,
    forbidden_transaction_signatures: set[str] | None = None,
) -> None:
    previous_strategy = policy_strategy_signature(previous)
    candidate_strategy = policy_strategy_signature(candidate)
    previous_arc_travel = _policy_arc_travel_direction(previous)
    candidate_arc_travel = _policy_arc_travel_direction(candidate)
    changed_arc_travel = bool(
        previous_arc_travel
        and candidate_arc_travel
        and previous_arc_travel != candidate_arc_travel
    )
    if not previous_strategy:
        return
    previous_linear_objective = _policy_linear_objective(previous)
    equivalent_linear_rename = bool(
        previous_linear_objective
        and _policy_linear_objective(candidate) == previous_linear_objective
        and candidate_strategy != previous_strategy
    )
    if equivalent_linear_rename and (
        failure_level == "strategy" or previous_strategy in blocked_strategy_signatures
    ):
        raise ModelAgentError(
            "strategy-level failure cannot be repaired by renaming the same body-frame "
            f"linear objective ({previous_linear_objective}); select a genuinely different "
            "motion objective or coordination primitive"
        )
    if previous_strategy in blocked_strategy_signatures:
        if not candidate_strategy or candidate_strategy == previous_strategy:
            raise ModelAgentError(
                "the current strategy is excluded from this model-generation search; select "
                "a different host-valid coordination mode and/or mover method"
            )
        return
    if failure_level == "strategy":
        if not candidate_strategy:
            raise ModelAgentError(
                "strategy-level failure requires a different mover objective and/or coordination "
                "mode; a support-only hold is not a strategy repair"
            )
        previous_objective = policy_motion_objective_signature(previous)
        candidate_objective = policy_motion_objective_signature(candidate)
        if (
            candidate_strategy == previous_strategy
            and previous_objective
            and candidate_objective
            and candidate_objective != previous_objective
        ):
            return
        if candidate_strategy == previous_strategy and changed_arc_travel:
            return
        if candidate_strategy == previous_strategy:
            raise ModelAgentError(
                "strategy-level failure requires changing semantic motion objective, "
                "coordination_mode, and/or mover method; changing only numerical parameters "
                "is forbidden"
            )
        return
    if failure_level == "script":
        previous_transaction = _policy_transaction_signature(previous)
        candidate_transaction = _policy_transaction_signature(candidate)
        previous_alignment_structure = _policy_alignment_structure_signature(previous)
        candidate_alignment_structure = _policy_alignment_structure_signature(candidate)
        if (
            previous_alignment_structure
            and candidate_alignment_structure == previous_alignment_structure
        ):
            raise ModelAgentError(
                "script-level physical failure in coordinated_alignment requires "
                "a structural rewrite; keeping the same axis and moving/reference "
                "robot pattern while changing only numeric magnitudes is invalid"
            )
        if (
            failure_type in _STRUCTURAL_SCRIPT_FAILURES
            and candidate_strategy == previous_strategy
            and policy_motion_objective_signature(candidate)
            == policy_motion_objective_signature(previous)
            and policy_role_signature(candidate) == policy_role_signature(previous)
            and not changed_arc_travel
        ):
            raise ModelAgentError(
                "script-level support/coordination failure requires a structural rewrite: "
                "change the body-frame objective, coordination primitive/mode, or front/back "
                "mover-role relation; a same-objective numerical rewrite is invalid"
            )
        if previous_transaction and candidate_transaction == previous_transaction:
            raise ModelAgentError(
                "script-level physical failure requires a materially different executable "
                "transaction; changing only identifiers, roles, phase labels, or rationale "
                "does not rewrite the rejected AST"
            )
        return
    if failure_level == "coordination":
        if not candidate_strategy:
            return
        previous_method = previous_strategy.split(":", 1)[1]
        candidate_method = candidate_strategy.split(":", 1)[1]
        if (
            previous_method not in COORDINATION_PRIMITIVE_METHODS
            and candidate_method in COORDINATION_PRIMITIVE_METHODS
        ):
            # One-time migration from a legacy single-mover relation into the
            # continuous coordination layer is itself a coordination repair.
            return
        if candidate_method != previous_method:
            raise ModelAgentError(
                "coordination-level failure must first keep the motion objective and change "
                "coordination mode or temporary roles"
            )
        if allow_measured_sync_compensation:
            if not candidate_strategy.startswith("sync:"):
                raise ModelAgentError(
                    "measured executor compensation must use the model-selected sync mover strategy"
                )
            return
        if (
            previous_method in {COORDINATED_ARC_TRANSPORT, COORDINATED_PIVOT}
            and failure_type in _UNSAFE_MAGNITUDE_FAILURES
        ):
            _validate_arc_coordination_envelope(previous, candidate)
        if policy_role_signature(candidate) == policy_role_signature(previous):
            if (
                previous_method in {COORDINATED_ARC_TRANSPORT, COORDINATED_PIVOT}
                and changed_arc_travel
            ):
                return
            if previous_method == COORDINATED_ARC_TRANSPORT:
                raise ModelAgentError(
                    "coordination-level arc repair must either change the shared body-frame "
                    "travel_direction or swap which robot receives the larger versus smaller "
                    "radius_m relative to the rejected transaction; both radii must remain "
                    "positive and changing only their magnitudes is invalid"
                )
            raise ModelAgentError(
                "coordination-level failure cannot be repaired by parameter tuning alone; "
                "change coordination mode or the support/mover role pattern"
            )
        if previous_method == COORDINATED_ALIGNMENT:
            _validate_alignment_coordination_role_swap(previous, candidate)
        return
    if (
        failure_level == "execution"
        and previous_strategy not in blocked_strategy_signatures
    ):
        if not candidate_strategy:
            if failure_type == "robot_fall":
                return
            raise ModelAgentError(
                "execution-level motion failure must tune the current bounded mover; "
                "a support-only hold does not test execution progress"
            )
        if candidate_strategy != previous_strategy:
            raise ModelAgentError(
                "execution-level failure must keep the validated mode and mover method while "
                "tuning one bounded execution parameter; strategy changes are allowed only "
                "after the exploration budget is exhausted"
            )
        if allow_measured_sync_compensation:
            if failure_type != "formation_desync":
                raise ModelAgentError(
                    "measured sync compensation is only valid for verifier-grounded "
                    "formation_desync"
                )
            return
        if (
            previous.robot_scripts
            and policy_role_signature(candidate) != policy_role_signature(previous)
            and not (
                previous_strategy.split(":", 1)[-1] == COORDINATED_ALIGNMENT
                and _policy_alignment_reference_corrector_pattern(candidate)
                == _policy_alignment_reference_corrector_pattern(previous)
                and _policy_alignment_reference_corrector_pattern(previous) is not None
            )
        ):
            raise ModelAgentError(
                "execution-level failure must keep the current role pattern and tune execution only"
            )
        if failure_type == "no_payload_progress":
            _validate_no_progress_execution_magnitude(previous, candidate)
        if (
            previous_strategy.split(":", 1)[-1]
            in {
                COORDINATED_ARC_TRANSPORT,
                COORDINATED_PIVOT,
            }
            and failure_type in _UNSAFE_MAGNITUDE_FAILURES
        ):
            _validate_arc_execution_envelope(previous, candidate)
        if (
            previous_strategy.split(":", 1)[-1] == COORDINATED_LINEAR_TRANSPORT
            and failure_type in _UNSAFE_MAGNITUDE_FAILURES
        ):
            _validate_linear_execution_envelope(
                previous,
                candidate,
                required_distance_m=formation_sync_probe_distance_m,
                forbidden_transaction_signatures=(
                    forbidden_transaction_signatures or set()
                ),
            )


def _validate_arc_coordination_envelope(
    previous: PolicyScript,
    candidate: PolicyScript,
) -> None:
    """Keep the minimum-load turn window fixed while changing coordination."""

    previous_scripts = {script.robot_id: script for script in previous.robot_scripts}
    candidate_scripts = {script.robot_id: script for script in candidate.robot_scripts}
    if set(previous_scripts) != {"front", "back"} or set(candidate_scripts) != {
        "front",
        "back",
    }:
        return
    before = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in previous_scripts.items()
    }
    after = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in candidate_scripts.items()
    }
    violations: list[str] = []
    for parameter_name in ("angle_deg", "duration_s"):
        try:
            previous_value = float(before["front"][parameter_name])
            candidate_value = float(after["front"][parameter_name])
        except (KeyError, TypeError, ValueError):
            violations.append(f"shared {parameter_name} must remain a literal number")
            continue
        if abs(candidate_value - previous_value) > 1.0e-9:
            violations.append(
                f"{parameter_name} must remain unchanged at {previous_value:g}"
            )
        for robot_id in ("front", "back"):
            try:
                robot_value = float(after[robot_id][parameter_name])
            except (KeyError, TypeError, ValueError):
                violations.append(
                    f"{robot_id} {parameter_name} must remain a literal number"
                )
                continue
            if abs(robot_value - candidate_value) > 1.0e-9:
                violations.append(f"front/back {parameter_name} must remain shared")
    if violations:
        raise ModelAgentError(
            "coordination-level arc/pivot repair must preserve the bounded angle and "
            "duration while changing travel direction or temporary roles; "
            + "; ".join(dict.fromkeys(violations))
        )


def _validate_arc_execution_envelope(
    previous: PolicyScript,
    candidate: PolicyScript,
) -> None:
    """Keep turn geometry fixed while reducing one continuous execution load."""

    previous_scripts = {script.robot_id: script for script in previous.robot_scripts}
    candidate_scripts = {script.robot_id: script for script in candidate.robot_scripts}
    if set(previous_scripts) != {"front", "back"} or set(candidate_scripts) != {
        "front",
        "back",
    }:
        return
    before = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in previous_scripts.items()
    }
    after = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in candidate_scripts.items()
    }
    method = _first_policy_api_method(previous_scripts["front"].code)
    fixed_names = (
        ("turn_direction", "travel_direction", "role", "swing_radius_m")
        if method == COORDINATED_PIVOT
        else ("turn_direction", "travel_direction", "radius_m")
        if method == COORDINATED_ARC_TRANSPORT
        else ("turn_direction",)
    )
    violations: list[str] = []
    parameter_context = ""
    for robot_id in ("front", "back"):
        for name in fixed_names:
            if before[robot_id].get(name) != after[robot_id].get(name):
                violations.append(f"{robot_id} changed fixed {name}")
    try:
        before_angle = float(before["front"]["angle_deg"])
        after_angle = float(after["front"]["angle_deg"])
        before_duration = float(before["front"]["duration_s"])
        after_duration = float(after["front"]["duration_s"])
    except (KeyError, TypeError, ValueError):
        violations.append("angle_deg and duration_s must remain literal numbers")
    else:
        parameter_context = (
            f"previous shared angle_deg={before_angle:g}, duration_s={before_duration:g}; "
            f"candidate shared angle_deg={after_angle:g}, duration_s={after_duration:g}"
        )
        if abs(after_duration - before_duration) > 1.0e-9:
            violations.append(
                "duration_s must remain unchanged after payload instability"
            )
        if after_angle >= before_angle:
            violations.append("angle_deg must decrease after payload instability")
        elif after_angle > max(5.0, 0.5 * before_angle) + 1.0e-9:
            violations.append("angle_deg must be reduced by at least half")
        for robot_id in ("front", "back"):
            if after[robot_id].get("angle_deg") != after["front"].get("angle_deg"):
                violations.append("front/back angle_deg must remain shared")
            if after[robot_id].get("duration_s") != after["front"].get("duration_s"):
                violations.append("front/back duration_s must remain shared")
    if violations:
        raise ModelAgentError(
            "unsafe arc/pivot execution repair must preserve direction, roles, and radii "
            "while reducing one continuous load; "
            + "; ".join(dict.fromkeys(violations))
            + (f"; {parameter_context}" if parameter_context else "")
        )


def _validate_alignment_coordination_role_swap(
    previous: PolicyScript,
    candidate: PolicyScript,
) -> None:
    """Require a real alignment-role swap without freezing rejected deltas."""

    previous_scripts = {script.robot_id: script for script in previous.robot_scripts}
    candidate_scripts = {script.robot_id: script for script in candidate.robot_scripts}
    if set(previous_scripts) != {"front", "back"} or set(candidate_scripts) != {
        "front",
        "back",
    }:
        return
    before = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in previous_scripts.items()
    }
    after = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in candidate_scripts.items()
    }
    violations: list[str] = []
    for robot_id in ("front", "back"):
        for fixed_name in ("axis", "duration_s"):
            if before[robot_id].get(fixed_name) != after[robot_id].get(fixed_name):
                violations.append(f"{robot_id} changed fixed {fixed_name}")
    for robot_id, peer_id in (("front", "back"), ("back", "front")):
        peer_delta = _safe_float(before[peer_id].get("delta"))
        candidate_delta = _safe_float(after[robot_id].get("delta"))
        if peer_delta is None or candidate_delta is None:
            violations.append(f"{robot_id} delta must remain a literal number")
        elif abs(peer_delta) <= 1.0e-9 and abs(candidate_delta) > 1.0e-9:
            violations.append(f"{robot_id} must become the zero-delta reference")
        elif abs(peer_delta) > 1.0e-9 and abs(candidate_delta) <= 1.0e-9:
            violations.append(f"{robot_id} must become the nonzero-delta corrector")
    if violations:
        raise ModelAgentError(
            "coordination-level alignment repair must swap the front/back delta assignments "
            "while preserving axis and duration_s; the Coder must select a new legal "
            "nonzero correction literal rather than inherit a rejected magnitude or sign; "
            + "; ".join(dict.fromkeys(violations))
        )


def _validate_alignment_sign_repair_policy(
    policy: PolicyScript,
    *,
    contract: object,
) -> None:
    """Require the exact previously untried alignment-sign transaction."""

    if not isinstance(contract, dict) or not contract:
        return
    scripts = {item.robot_id: item for item in policy.robot_scripts}
    if set(scripts) != {"front", "back"}:
        raise ModelAgentError(
            "alignment_sign_repair_contract requires front and back scripts"
        )
    required_deltas = contract.get("required_robot_deltas", {})
    required_deltas = required_deltas if isinstance(required_deltas, dict) else {}
    expected_axis = str(contract.get("semantic_objective", "")).strip()
    expected_duration = _safe_float(contract.get("required_duration_s"))
    expected_roles = contract.get("preserve_roles", {})
    expected_roles = expected_roles if isinstance(expected_roles, dict) else {}
    actual_roles = {
        item.robot_id: str(item.role).strip()
        for item in policy.assignments
        if item.robot_id in {"front", "back"}
    }
    violations: list[str] = []
    if actual_roles != expected_roles:
        violations.append(
            f"front/back roles must remain {expected_roles}; got {actual_roles}"
        )
    for robot_id in ("front", "back"):
        method = _first_policy_api_method(scripts[robot_id].code)
        arguments = _first_task_call_literal_arguments(scripts[robot_id].code)
        if method != COORDINATED_ALIGNMENT:
            violations.append(f"{robot_id} must keep api.{COORDINATED_ALIGNMENT}")
            continue
        if str(arguments.get("axis", "")).strip() != expected_axis:
            violations.append(f"{robot_id} must keep axis={expected_axis!r}")
        actual_delta = _safe_float(arguments.get("delta"))
        required_delta = _safe_float(required_deltas.get(robot_id))
        if (
            actual_delta is None
            or required_delta is None
            or abs(actual_delta - required_delta) > 1.0e-9
        ):
            violations.append(
                f"{robot_id} delta must be {required_delta}; got {actual_delta}"
            )
        actual_duration = _safe_float(arguments.get("duration_s"))
        if (
            actual_duration is None
            or expected_duration is None
            or abs(actual_duration - expected_duration) > 1.0e-9
        ):
            violations.append(
                f"{robot_id} duration_s must remain {expected_duration}; "
                f"got {actual_duration}"
            )
    if violations:
        raise ModelAgentError(
            "alignment sign repair must preserve primitive, lateral/longitudinal "
            "axis, duration, and temporary roles while reversing only the "
            "corrector delta sign; " + "; ".join(dict.fromkeys(violations))
        )


def _validate_linear_execution_envelope(
    previous: PolicyScript,
    candidate: PolicyScript,
    *,
    required_distance_m: float | None = None,
    forbidden_transaction_signatures: set[str] | None = None,
) -> None:
    """Require the selected lower distance at the verified gait command."""

    previous_scripts = {script.robot_id: script for script in previous.robot_scripts}
    candidate_scripts = {script.robot_id: script for script in candidate.robot_scripts}
    if set(previous_scripts) != {"front", "back"} or set(candidate_scripts) != {
        "front",
        "back",
    }:
        return
    before = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in previous_scripts.items()
    }
    after = {
        robot_id: _first_task_call_literal_arguments(script.code)
        for robot_id, script in candidate_scripts.items()
    }
    violations: list[str] = []
    for robot_id in ("front", "back"):
        for fixed_name in ("direction", "speed_mps", "ramp_s"):
            if before[robot_id].get(fixed_name) != after[robot_id].get(fixed_name):
                violations.append(f"{robot_id} changed fixed {fixed_name}")
    distance_changed = any(
        before[robot_id].get("distance_m") != after[robot_id].get("distance_m")
        for robot_id in ("front", "back")
    )
    if not distance_changed:
        violations.append("distance_m must decrease by exactly one supported tier")
    for robot_id in ("front", "back"):
        try:
            before_distance = float(before[robot_id]["distance_m"])
            after_distance = float(after[robot_id]["distance_m"])
            before_speed = float(before[robot_id]["speed_mps"])
            after_speed = float(after[robot_id]["speed_mps"])
        except (KeyError, TypeError, ValueError):
            violations.append(
                f"{robot_id} distance_m and speed_mps must be literal numbers"
            )
            continue
        expected_distance = (
            float(required_distance_m)
            if required_distance_m is not None
            else _next_lower_unblocked_linear_distance_tier(
                previous,
                distance_m=before_distance,
                forbidden_transaction_signatures=(
                    forbidden_transaction_signatures or set()
                ),
            )
        )
        if expected_distance is None:
            violations.append(f"{robot_id} has no lower supported distance tier")
        elif abs(after_distance - expected_distance) > 1.0e-9:
            qualifier = (
                "active synchronization probe distance"
                if required_distance_m is not None
                else "next unblocked tier"
            )
            violations.append(
                f"{robot_id} distance_m must decrease to the {qualifier} "
                f"{expected_distance:g}"
            )
        if abs(after_speed - before_speed) > 1.0e-9:
            violations.append(f"{robot_id} changed fixed speed_mps")
        if abs(after_speed - VERIFIED_LINEAR_SPEED_MPS) > 1.0e-9:
            violations.append(
                f"{robot_id} speed_mps must remain {VERIFIED_LINEAR_SPEED_MPS:.2f}"
            )
    if violations:
        distance_contract = (
            f"to the active synchronization probe distance {required_distance_m:g}"
            if required_distance_m is not None
            else "to the next unblocked tier"
        )
        raise ModelAgentError(
            "unsafe coordinated-linear execution repair must preserve direction, speed_mps=0.50, "
            "ramp_s=1.5, mode, and the front/back relation while reducing distance "
            f"{distance_contract}; " + "; ".join(dict.fromkeys(violations))
        )


def _validate_no_progress_execution_magnitude(
    previous: PolicyScript,
    candidate: PolicyScript,
) -> None:
    """Require a real, one-parameter magnitude increase for supported primitives."""

    previous_scripts = {
        script.robot_id: script
        for script in (
            previous.robot_scripts
            or (
                RobotPolicyScript(
                    robot_id="team",
                    script_id=previous.script_id,
                    code=previous.code,
                    synchronization_group="team",
                ),
            )
        )
    }
    candidate_scripts = {
        script.robot_id: script
        for script in (
            candidate.robot_scripts
            or (
                RobotPolicyScript(
                    robot_id="team",
                    script_id=candidate.script_id,
                    code=candidate.code,
                    synchronization_group="team",
                ),
            )
        )
    }
    if set(previous_scripts) != set(candidate_scripts):
        return
    methods = {
        _first_policy_api_method(script.code) for script in previous_scripts.values()
    }
    if methods != {COORDINATED_ALIGNMENT}:
        return

    violations: list[str] = []
    for robot_id in sorted(previous_scripts):
        before = _first_task_call_literal_arguments(previous_scripts[robot_id].code)
        after = _first_task_call_literal_arguments(candidate_scripts[robot_id].code)
        for fixed_name in ("axis", "duration_s"):
            if before.get(fixed_name) != after.get(fixed_name):
                violations.append(f"{robot_id} changed fixed {fixed_name}")
        try:
            before_delta = float(before["delta"])
            after_delta = float(after["delta"])
        except (KeyError, TypeError, ValueError):
            violations.append(f"{robot_id} delta must remain a literal number")
            continue
        if before_delta == 0.0:
            if after_delta != 0.0:
                violations.append(f"{robot_id} changed the reference role")
            continue
        if before_delta * after_delta <= 0.0:
            violations.append(f"{robot_id} changed delta sign")
        elif abs(after_delta) <= abs(before_delta) + 1.0e-9:
            violations.append(
                f"{robot_id} abs(delta) did not increase from {abs(before_delta):g}"
            )
    if violations:
        raise ModelAgentError(
            "no-progress coordinated_alignment repair must keep axis, duration, and role signs "
            "while increasing abs(delta) for each moving profile; "
            + "; ".join(violations)
        )


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_measured_executor_mismatch(metrics: dict[str, Any]) -> bool:
    front = _safe_float(metrics.get("front_step_displacement_m"))
    back = _safe_float(metrics.get("back_step_displacement_m"))
    return front is not None and back is not None and abs(front - back) > 0.01


def _has_measured_goal_regression(
    failure_type: str,
    metrics: dict[str, Any],
    *,
    evidence: dict[str, Any] | None = None,
) -> bool:
    """Keep directional evidence even when a higher-priority safety failure wins."""

    evidence = evidence if isinstance(evidence, dict) else {}
    if str(evidence.get("monitored_physical_failure", "")).strip() == failure_type:
        # Motion after an online safety abort is an uncontrolled endpoint, not
        # evidence that the selected body-frame objective was wrong.
        return False
    if failure_type in {"goal_regression", "route_regression"}:
        return True
    if failure_type in {"payload_drop", "robot_fall", "payload_wall_contact"}:
        # Once the physical state is invalid, its endpoint displacement does
        # not identify whether the body-frame mover direction was useful. In
        # particular, a one-mover async drop must not ban the same method for
        # a still-untried synchronized team transaction.
        return False
    displacement = _safe_float(metrics.get("payload_step_displacement_xy_m"))
    progress = _safe_float(metrics.get("payload_target_progress_m"))
    required = _safe_float(metrics.get("required_step_displacement_m"))
    threshold = max(0.01, min(0.05, required if required is not None else 0.01))
    return bool(
        displacement is not None
        and progress is not None
        and displacement >= threshold
        and progress <= -threshold
    )


def _trace_payload(trace: ExecutionTrace | None) -> dict[str, Any]:
    if trace is None:
        return {}
    return {
        "script_id": trace.script_id,
        "attempt": trace.attempt,
        "status": trace.status,
        "error": trace.error,
        "skill_calls": [
            {
                "index": item.index,
                "skill_name": item.skill_name,
                "parameters": item.parameters,
                "effective_parameters": item.effective_parameters,
                "status": item.status,
                "artifact_path": item.artifact_path,
                "stderr_tail": item.stderr[-1000:],
            }
            for item in trace.skill_calls
        ],
        "robot_executions": [asdict(item) for item in trace.robot_executions],
    }
