"""Restricted Python policy executor for generated robot scripts."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
import traceback

from .contracts import (
    ExecutionTrace,
    PlannedSkillCall,
    PolicyScript,
    RobotExecutionTrace,
    RobotPolicyScript,
    write_json,
)
from .coordination_specs import (
    COORDINATION_PRIMITIVE_METHODS,
    validate_coordination_primitive_parameters,
)
from .skill_api import (
    ALLOWED_G1_SKILL_API_METHODS,
    G1SkillAPI,
    MAX_TRANSPORT_RAMP_S,
    MAX_TRANSPORT_SEGMENT_M,
    MAX_TRANSPORT_SPEED_MPS,
    MAX_EXIT_SEGMENT_M,
)
from .static_specs import validate_catalog_parameters
from praxis_coop.sonic.team_coordination import (
    TeamAction,
    validate_asynchronous_actions,
    validate_synchronized_actions,
)
from praxis_coop.sonic.coordination_primitives import (
    validate_coordination_primitive_team_actions,
)


FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.Lambda,
    ast.ClassDef,
)
FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "locals",
    "open",
    "input",
    "breakpoint",
}

CONTACT_INTENTS = frozenset(
    {
        "unspecified",
        "maintain_payload_support",
        "establish_payload_support",
        "reestablish_payload_contact",
        "release_payload",
        "reposition_without_payload",
        "exit_task_area",
        "hold",
        "none",
    }
)
PAYLOAD_COOPERATIVE_METHODS = frozenset(
    {
        "place_payload_on_arms",
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
        "turn_clockwise",
        "turn_counterclockwise",
        "place_crouch",
        "recover_stand",
        "recover_payload_inward",
    }
) | COORDINATION_PRIMITIVE_METHODS
PAYLOAD_RELEASE_METHODS = frozenset({"release_payload"})
EXIT_METHODS = frozenset(
    {"exit_forward", "exit_backward", "exit_lateral_left", "exit_lateral_right"}
)
COORDINATION_MODES = frozenset({"sync", "async"})
ASYNCHRONOUS_TEAM_METHODS = frozenset(
    {
        "hold_position",
        "carry_forward",
        "carry_lateral_left",
        "carry_lateral_right",
        "turn_clockwise",
        "turn_counterclockwise",
    }
) | COORDINATION_PRIMITIVE_METHODS
POSTURE_LATCHING_TEAM_METHODS = frozenset(
    {
        "place_crouch",
        "release_payload",
        "exit_backward",
    }
)


@dataclass
class PolicyExecutor:
    """Execute a policy script against a G1SkillAPI instance."""

    artifact_dir: Path

    def execute(self, policy: PolicyScript, api: G1SkillAPI) -> ExecutionTrace:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.artifact_dir / f"{policy.script_id}.py"
        script_path.write_text(policy.code, encoding="utf-8")
        robot_executions: tuple[RobotExecutionTrace, ...] = ()
        try:
            validate_policy_code(policy.code)
            if policy.robot_scripts:
                robot_executions = self._execute_robot_scripts(policy.robot_scripts, api)
            else:
                self._run_policy_function(policy.code, script_path, api)
            trace = ExecutionTrace(
                script_id=policy.script_id,
                attempt=policy.attempt,
                status="success",
                skill_calls=api.records,
                robot_executions=robot_executions,
                artifact_dir=str(self.artifact_dir),
                verification_contract=dict(policy.verification_contract),
            )
        except Exception as exc:  # noqa: BLE001 - execution trace should preserve any policy error
            robot_executions = getattr(exc, "robot_executions", robot_executions)
            trace = ExecutionTrace(
                script_id=policy.script_id,
                attempt=policy.attempt,
                status="script_error",
                skill_calls=api.records,
                robot_executions=robot_executions,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                artifact_dir=str(self.artifact_dir),
                verification_contract=dict(policy.verification_contract),
            )
        write_json(self.artifact_dir / "execution_trace.json", trace)
        return trace

    def _execute_robot_scripts(
        self,
        scripts: tuple[RobotPolicyScript, ...],
        api: G1SkillAPI,
    ) -> tuple[RobotExecutionTrace, ...]:
        by_robot = {script.robot_id: script for script in scripts}
        if set(by_robot) != {"front", "back"}:
            raise ValueError("team policy must contain exactly one front and one back robot script")
        if any(not script.synchronization_group.strip() for script in scripts):
            raise ValueError("each robot script must declare a non-empty synchronization_group")

        executions: list[RobotExecutionTrace] = []
        calls_by_robot: dict[str, tuple[PlannedSkillCall, ...]] = {}
        normalized_metadata: dict[str, tuple[str, str, str]] = {}
        for script in scripts:
            robot_id = script.robot_id
            path = self.artifact_dir / f"{script.script_id}.py"
            path.write_text(script.code, encoding="utf-8")
            recorder = _RecordingSkillAPI()
            try:
                validate_policy_code(script.code)
                self._run_policy_function(script.code, path, recorder)
                planned = recorder.calls
                _validate_robot_transaction(planned, robot_id=robot_id)
                task_method = next(call.method for call in planned if call.method != "stop_all")
                phase_id = script.phase_id.strip() or script.synchronization_group
                contact_intent = _normalized_contact_intent(
                    script.contact_intent,
                    task_method=task_method,
                )
                coordination_mode = _normalized_coordination_mode(script.coordination_mode)
                execution = RobotExecutionTrace(
                    robot_id=robot_id,
                    script_id=script.script_id,
                    status="planned",
                    synchronization_group=script.synchronization_group,
                    planned_calls=planned,
                    phase_id=phase_id,
                    contact_intent=contact_intent,
                    coordination_mode=coordination_mode,
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-robot planning failure
                execution = RobotExecutionTrace(
                    robot_id=robot_id,
                    script_id=script.script_id,
                    status="script_error",
                    synchronization_group=script.synchronization_group,
                    planned_calls=recorder.calls,
                    error=f"{type(exc).__name__}: {exc}",
                    phase_id=script.phase_id,
                    contact_intent=script.contact_intent,
                    coordination_mode=script.coordination_mode,
                )
                executions.append(execution)
                raise
            executions.append(execution)
            calls_by_robot[robot_id] = planned
            normalized_metadata[robot_id] = (
                phase_id,
                contact_intent,
                coordination_mode,
            )

        groups: dict[str, list[str]] = {}
        for script in scripts:
            groups.setdefault(script.synchronization_group, []).append(script.robot_id)
        try:
            _validate_team_transaction_groups(
                groups=groups,
                calls_by_robot=calls_by_robot,
                normalized_metadata=normalized_metadata,
            )
            for sync_group, robot_ids in groups.items():
                scoped_robots = tuple(robot_ids)
                reference_robot = robot_ids[0]
                reference_calls = calls_by_robot[reference_robot]
                coordination_mode = normalized_metadata[reference_robot][2]
                coordinated_phase_completed_idle = False
                posture_latching_phase_completed = False
                for call_index, call in enumerate(reference_calls):
                    row_calls = {
                        robot_id: calls_by_robot[robot_id][call_index]
                        for robot_id in robot_ids
                    }
                    if (
                        len(robot_ids) > 1
                        and any(row_call.method != "stop_all" for row_call in row_calls.values())
                        and all(
                            row_call.method in ASYNCHRONOUS_TEAM_METHODS
                            for row_call in row_calls.values()
                        )
                    ):
                        api.dispatch_coordinated_planned_calls(
                            row_calls,
                            phase_id=normalized_metadata[reference_robot][0],
                            synchronization_group=sync_group,
                            coordination_mode=coordination_mode,
                        )
                        coordinated_phase_completed_idle = True
                    elif (
                        (
                            coordinated_phase_completed_idle
                            or posture_latching_phase_completed
                        )
                        and all(
                            row_call.method == "stop_all"
                            for row_call in row_calls.values()
                        )
                    ):
                        reason = (
                            "the completed lifecycle phase intentionally latches its "
                            "lower-body posture until the next lifecycle action"
                            if posture_latching_phase_completed
                            else (
                                "the atomic coordinated phase already published its "
                                "brake and post-action IDLE settle"
                            )
                        )
                        api.record_coordinated_stop_boundary(
                            robot_ids=scoped_robots,
                            reason=reason,
                        )
                    else:
                        api.dispatch_planned_call(
                            call.method,
                            call.parameters,
                            robot_ids=scoped_robots,
                        )
                        posture_latching_phase_completed = all(
                            row_call.method in POSTURE_LATCHING_TEAM_METHODS
                            for row_call in row_calls.values()
                        )
        except Exception as exc:  # noqa: BLE001 - attach per-robot dispatch context
            failed = tuple(replace(item, status="dispatch_failed", error=str(exc)) for item in executions)
            raise TeamDispatchError(str(exc), robot_executions=failed) from exc

        status = (
            "command_publish_completed"
            if bool(getattr(api.backend, "execute", True))
            else "dry_run_validated"
        )
        return tuple(replace(item, status=status) for item in executions)

    def _run_policy_function(self, code: str, path: Path, api: object) -> None:
        namespace: dict[str, object] = {"__builtins__": self._safe_builtins()}
        exec(compile(code, str(path), "exec"), namespace, namespace)  # noqa: S102
        fn = namespace.get("policy")
        if not callable(fn):
            raise ValueError("policy script must define callable policy(api)")
        fn(api)

    @staticmethod
    def _safe_builtins() -> dict[str, object]:
        allowed = {
            "abs": abs,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "str": str,
            "sum": sum,
            "tuple": tuple,
        }
        return allowed


def validate_policy_code(code: str) -> None:
    """Validate generated policy code before it can touch the robot SDK."""

    tree = ast.parse(code)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "policy" for node in tree.body):
        raise ValueError("policy script must define def policy(api)")
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"forbidden Python construct: {type(node).__name__}")
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FORBIDDEN_CALLS:
                raise ValueError(f"forbidden call in policy script: {name}")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "api" and node.func.attr not in ALLOWED_G1_SKILL_API_METHODS:
                    allowed = ", ".join(sorted(ALLOWED_G1_SKILL_API_METHODS))
                    raise ValueError(f"unknown G1SkillAPI method: api.{node.func.attr}; allowed: {allowed}")


def validate_step_policy_code(
    code: str,
    *,
    validate_parameters: bool = True,
) -> tuple[str, str]:
    """Validate the one-action transaction required by checkpointed execution."""

    planned_calls = _validated_step_policy_calls(
        code,
        validate_parameters=validate_parameters,
    )
    return planned_calls[0].method, planned_calls[1].method


def _validated_step_policy_calls(
    code: str,
    *,
    validate_parameters: bool = True,
) -> tuple[PlannedSkillCall, ...]:
    """Return the literal calls from one validated checkpoint transaction."""

    validate_policy_code(code)
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != "policy":
        raise ValueError("checkpointed code must contain exactly one top-level def policy(api)")
    function = functions[0]
    if (
        len(function.args.args) != 1
        or function.args.args[0].arg != "api"
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.defaults
        or function.args.kw_defaults
    ):
        raise ValueError("checkpointed policy must have the exact signature def policy(api)")
    if len(function.body) != 2 or not all(isinstance(statement, ast.Expr) for statement in function.body):
        raise ValueError(
            "checkpointed policy(api) must contain exactly two expression statements: "
            "one api skill call followed by api.stop_all()"
        )
    for statement in function.body:
        call = statement.value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "api"
        ):
            raise ValueError("checkpointed policy statements may only be direct api.<skill>(...) calls")
    recorder = _RecordingSkillAPI()
    namespace: dict[str, object] = {"__builtins__": PolicyExecutor._safe_builtins()}
    exec(compile(code, "<generated-policy-validation>", "exec"), namespace, namespace)  # noqa: S102
    policy = namespace["policy"]
    policy(recorder)
    planned_calls = recorder.calls
    calls = [item.method for item in planned_calls]
    if len(planned_calls) != 2 or calls[0] == "stop_all" or calls[1] != "stop_all":
        raise ValueError(
            "checkpointed step must contain exactly one task-changing api call "
            "followed by api.stop_all(); got " + repr(calls)
        )
    if validate_parameters:
        _validate_planned_skill_parameters(planned_calls[0])
    return planned_calls


def validate_team_policy_scripts(scripts: tuple[RobotPolicyScript, ...]) -> None:
    """Validate cross-robot transaction semantics without dispatching a command."""

    by_robot = {script.robot_id: script for script in scripts}
    if set(by_robot) != {"front", "back"} or len(scripts) != 2:
        raise ValueError("team policy must contain exactly one front and one back robot script")
    if any(not script.synchronization_group.strip() for script in scripts):
        raise ValueError("each robot script must declare a non-empty synchronization_group")

    calls_by_robot: dict[str, tuple[PlannedSkillCall, ...]] = {}
    normalized_metadata: dict[str, tuple[str, str, str]] = {}
    for script in scripts:
        planned = _validated_step_policy_calls(script.code)
        _validate_robot_transaction(planned, robot_id=script.robot_id)
        task_method = planned[0].method
        calls_by_robot[script.robot_id] = planned
        normalized_metadata[script.robot_id] = (
            script.phase_id.strip() or script.synchronization_group,
            _normalized_contact_intent(script.contact_intent, task_method=task_method),
            _normalized_coordination_mode(script.coordination_mode),
        )

    groups: dict[str, list[str]] = {}
    for script in scripts:
        groups.setdefault(script.synchronization_group, []).append(script.robot_id)
    _validate_team_transaction_groups(
        groups=groups,
        calls_by_robot=calls_by_robot,
        normalized_metadata=normalized_metadata,
    )


def _validate_planned_skill_parameters(call: PlannedSkillCall) -> None:
    if call.method in COORDINATION_PRIMITIVE_METHODS:
        validate_coordination_primitive_parameters(call.method, call.parameters)
        return
    transport_methods = {
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
    }
    if call.method == "release_payload":
        unknown = set(call.parameters) - {"ramp_s", "hold_s"}
        if unknown:
            raise ValueError(f"release_payload received unsupported parameters: {sorted(unknown)}")
        ramp = float(call.parameters.get("ramp_s", 4.0))
        hold = float(call.parameters.get("hold_s", 2.0))
        if ramp < 1.0 or ramp > 8.0:
            raise ValueError(f"release_payload ramp_s must be in [1.0, 8.0], got {ramp}")
        if hold < 0.5 or hold > 10.0:
            raise ValueError(f"release_payload hold_s must be in [0.5, 10.0], got {hold}")
        return
    if call.method == "recover_payload_inward":
        validate_catalog_parameters(call.method, call.parameters)
        return
    if call.method in EXIT_METHODS:
        unknown = set(call.parameters) - {"distance_m", "speed_mps", "ramp_s"}
        if unknown:
            raise ValueError(f"{call.method} received unsupported parameters: {sorted(unknown)}")
        distance = abs(float(call.parameters.get("distance_m", 0.50)))
        speed = abs(float(call.parameters.get("speed_mps", 0.20)))
        ramp = float(call.parameters.get("ramp_s", 1.5))
        if distance <= 0.0 or distance > MAX_EXIT_SEGMENT_M:
            raise ValueError(
                f"{call.method} distance_m must be in (0, {MAX_EXIT_SEGMENT_M:.2f}], got {distance}"
            )
        if speed <= 0.0 or speed > MAX_TRANSPORT_SPEED_MPS:
            raise ValueError(
                f"{call.method} speed_mps must be in (0, {MAX_TRANSPORT_SPEED_MPS:.2f}], got {speed}"
            )
        if ramp < 0.0 or ramp > MAX_TRANSPORT_RAMP_S:
            raise ValueError(
                f"{call.method} ramp_s must be in [0, {MAX_TRANSPORT_RAMP_S:.1f}], got {ramp}"
            )
        return
    if call.method not in transport_methods:
        return
    distance = abs(float(call.parameters.get("distance_m", 0.30)))
    speed = abs(float(call.parameters.get("speed_mps", 0.10)))
    ramp = float(call.parameters.get("ramp_s", 1.0))
    if distance <= 0.0 or distance > MAX_TRANSPORT_SEGMENT_M:
        raise ValueError(
            f"{call.method} distance_m must be in (0, {MAX_TRANSPORT_SEGMENT_M:.2f}], got {distance}"
        )
    if speed <= 0.0 or speed > MAX_TRANSPORT_SPEED_MPS:
        raise ValueError(
            f"{call.method} speed_mps must be in (0, {MAX_TRANSPORT_SPEED_MPS:.2f}], got {speed}"
        )
    if ramp < 0.0 or ramp > MAX_TRANSPORT_RAMP_S:
        raise ValueError(
            f"{call.method} ramp_s must be in [0, {MAX_TRANSPORT_RAMP_S:.1f}], got {ramp}"
        )


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


class _RecordingSkillAPI:
    """Capture one robot script's allowlisted SDK calls before dispatch."""

    def __init__(self) -> None:
        self._calls: list[PlannedSkillCall] = []

    @property
    def calls(self) -> tuple[PlannedSkillCall, ...]:
        return tuple(self._calls)

    def __getattr__(self, name: str):
        if name not in ALLOWED_G1_SKILL_API_METHODS:
            raise AttributeError(name)

        def record(**parameters):
            call = PlannedSkillCall(
                index=len(self._calls) + 1,
                method=name,
                parameters=dict(parameters),
            )
            self._calls.append(call)
            return call

        return record


class TeamDispatchError(RuntimeError):
    """Preserve front/back script state when a synchronized dispatch fails."""

    def __init__(self, message: str, *, robot_executions: tuple[RobotExecutionTrace, ...]) -> None:
        super().__init__(message)
        self.robot_executions = robot_executions


def _validate_robot_transaction(
    calls: tuple[PlannedSkillCall, ...],
    *,
    robot_id: str,
) -> None:
    task_calls = [call for call in calls if call.method != "stop_all"]
    if len(task_calls) != 1 or len(calls) != 2 or calls[-1].method != "stop_all":
        raise ValueError(
            f"robot {robot_id} must emit exactly one task-changing call followed by stop_all"
        )
    _validate_planned_skill_parameters(task_calls[0])


def _validate_synchronized_transactions(
    *,
    sync_group: str,
    calls_by_robot: dict[str, tuple[PlannedSkillCall, ...]],
) -> None:
    """Require one shared objective while allowing bounded compensation."""

    call_counts = {len(calls) for calls in calls_by_robot.values()}
    if len(call_counts) != 1:
        raise ValueError(
            f"synchronization_group {sync_group!r} has different transaction lengths: "
            f"{ {robot_id: len(calls) for robot_id, calls in calls_by_robot.items()} }"
        )
    for index in range(next(iter(call_counts))):
        row = {robot_id: calls[index] for robot_id, calls in calls_by_robot.items()}
        methods = {call.method for call in row.values()}
        if len(methods) != 1:
            raise ValueError(
                f"synchronization_group {sync_group!r} requires the same atomic skill at "
                f"call {index + 1}; got { {robot_id: call.method for robot_id, call in row.items()} }"
            )
        if _row_parameters_differ(row):
            method = next(iter(methods))
            if method not in ASYNCHRONOUS_TEAM_METHODS:
                raise ValueError(
                    f"sync coordination_group {sync_group!r} cannot compensate parameters "
                    f"for api.{method}; this skill requires identical parameters"
                )
            validate_synchronized_actions(
                {
                    robot_id: TeamAction(
                        method=call.method,
                        parameters=dict(call.parameters),
                    )
                    for robot_id, call in row.items()
                }
            )


def _validate_team_transaction_groups(
    *,
    groups: dict[str, list[str]],
    calls_by_robot: dict[str, tuple[PlannedSkillCall, ...]],
    normalized_metadata: dict[str, tuple[str, str, str]],
) -> None:
    """Apply the shared sync/async contract before any physical dispatch."""

    for sync_group, robot_ids in groups.items():
        reference_robot = robot_ids[0]
        if len(robot_ids) == 1:
            _validate_independent_dispatch(
                robot_id=reference_robot,
                task_method=calls_by_robot[reference_robot][0].method,
            )
            continue
        phases = {normalized_metadata[robot_id][0] for robot_id in robot_ids}
        if len(phases) != 1:
            raise ValueError(
                f"synchronization_group {sync_group!r} spans different phase_id values: "
                f"{sorted(phases)}"
            )
        modes = {normalized_metadata[robot_id][2] for robot_id in robot_ids}
        if len(modes) != 1:
            raise ValueError(
                f"synchronization_group {sync_group!r} spans different "
                f"coordination_mode values: {sorted(modes)}"
            )
        coordination_mode = next(iter(modes))
        intents = {normalized_metadata[robot_id][1] for robot_id in robot_ids}
        if coordination_mode == "sync" and len(intents) != 1:
            raise ValueError(
                f"sync coordination_group {sync_group!r} requires matching "
                f"contact_intent values; got {sorted(intents)}"
            )
        group_calls = {
            robot_id: calls_by_robot[robot_id]
            for robot_id in robot_ids
        }
        validate_coordination_primitive_team_actions(
            {
                robot_id: TeamAction(
                    method=calls[0].method,
                    parameters=dict(calls[0].parameters),
                )
                for robot_id, calls in group_calls.items()
            },
            coordination_mode=coordination_mode,
        )
        if coordination_mode == "sync":
            _validate_synchronized_transactions(
                sync_group=sync_group,
                calls_by_robot=group_calls,
            )
        else:
            _validate_asynchronous_transactions(
                sync_group=sync_group,
                calls_by_robot=group_calls,
            )


def _row_parameters_differ(row: dict[str, PlannedSkillCall]) -> bool:
    parameter_sets = [call.parameters for call in row.values()]
    return any(parameters != parameter_sets[0] for parameters in parameter_sets[1:])


def _validate_asynchronous_transactions(
    *,
    sync_group: str,
    calls_by_robot: dict[str, tuple[PlannedSkillCall, ...]],
) -> None:
    """Validate one atomically published but intentionally asymmetric team phase."""

    call_counts = {len(calls) for calls in calls_by_robot.values()}
    if call_counts != {2}:
        raise ValueError(
            f"async coordination_group {sync_group!r} requires one action plus stop_all "
            f"per robot; got { {robot_id: len(calls) for robot_id, calls in calls_by_robot.items()} }"
        )
    for robot_id, calls in calls_by_robot.items():
        task_call = calls[0]
        if calls[1].method != "stop_all":
            raise ValueError(
                f"async robot {robot_id} must finish its bounded action with stop_all"
            )
        if task_call.method not in ASYNCHRONOUS_TEAM_METHODS:
            raise ValueError(
                f"api.{task_call.method} is not supported in async loaded coordination; "
                f"allowed={sorted(ASYNCHRONOUS_TEAM_METHODS)}"
            )
    try:
        validate_asynchronous_actions(
            {
                robot_id: TeamAction(
                    method=calls[0].method,
                    parameters=dict(calls[0].parameters),
                )
                for robot_id, calls in calls_by_robot.items()
            }
        )
    except ValueError as exc:
        raise ValueError(f"async coordination_group {sync_group!r}: {exc}") from exc


def _normalized_contact_intent(value: str, *, task_method: str) -> str:
    intent = str(value).strip() or "unspecified"
    if intent not in CONTACT_INTENTS:
        raise ValueError(
            f"unknown contact_intent {intent!r}; expected one of {sorted(CONTACT_INTENTS)}"
        )
    if intent == "unspecified":
        if task_method == "place_payload_on_arms":
            return "establish_payload_support"
        if task_method in PAYLOAD_COOPERATIVE_METHODS:
            return "maintain_payload_support"
        if task_method in PAYLOAD_RELEASE_METHODS:
            return "release_payload"
        if task_method in EXIT_METHODS:
            return "exit_task_area"
        if task_method == "hold_position":
            return "hold"
        return "none"
    if task_method == "recover_stand" and intent in {
        "maintain_payload_support",
        "reposition_without_payload",
    }:
        return intent
    if task_method in PAYLOAD_COOPERATIVE_METHODS and intent not in {
        "maintain_payload_support",
        "establish_payload_support",
    }:
        raise ValueError(
            f"api.{task_method} is a payload-cooperative skill and conflicts with "
            f"contact_intent={intent!r}"
        )
    if task_method in PAYLOAD_RELEASE_METHODS and intent != "release_payload":
        raise ValueError(
            f"api.{task_method} requires contact_intent='release_payload', got {intent!r}"
        )
    if task_method in EXIT_METHODS and intent not in {
        "exit_task_area",
        "reposition_without_payload",
    }:
        raise ValueError(
            f"api.{task_method} requires an unloaded exit contact intent, got {intent!r}"
        )
    return intent


def _normalized_coordination_mode(value: str) -> str:
    mode = str(value).strip().lower() or "sync"
    if mode not in COORDINATION_MODES:
        raise ValueError(
            f"unknown coordination_mode {mode!r}; expected one of {sorted(COORDINATION_MODES)}"
        )
    return mode


def _validate_independent_dispatch(
    *,
    robot_id: str,
    task_method: str,
) -> None:
    if task_method in PAYLOAD_COOPERATIVE_METHODS | PAYLOAD_RELEASE_METHODS:
        raise ValueError(
            f"robot {robot_id} selected payload-cooperative api.{task_method} in an independent "
            "synchronization_group; loaded support actions require a shared dispatch barrier"
        )
