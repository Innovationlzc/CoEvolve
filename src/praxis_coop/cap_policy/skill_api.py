"""Safe multi-robot Skill SDK exposed to generated policy scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

from .backends import SkillBackend, StaticSonicSkillBackend
from .coordination_specs import (
    COORDINATION_PRIMITIVE_METHODS,
    validate_coordination_primitive_parameters,
)
from .contracts import PlannedSkillCall, SkillCallRecord
from .static_specs import MODEL_SCRIPT_API_METHODS, validate_catalog_parameters
from praxis_coop.sonic.runtime_state import (
    wait_for_payload_on_target,
    wait_for_payload_recovery_ready,
    wait_for_payload_release_ready,
    wait_for_payload_supported,
)
from praxis_coop.sonic.safety import get_carry_safety_profile
from praxis_coop.sonic.team_coordination import (
    TeamAction,
    validate_asynchronous_actions,
    validate_synchronized_actions,
)
from praxis_coop.sonic.coordination_primitives import (
    validate_coordination_primitive_team_actions,
)


ALLOWED_G1_SKILL_API_METHODS = MODEL_SCRIPT_API_METHODS | COORDINATION_PRIMITIVE_METHODS

MAX_TRANSPORT_SEGMENT_M = 2.0
MAX_TRANSPORT_SPEED_MPS = 0.50
MAX_TRANSPORT_RAMP_S = 4.0
MAX_SKILL_CALLS_PER_POLICY = 48
MAX_EXIT_SEGMENT_M = 0.50
DEFAULT_MAX_PAYLOAD_SUPPORT_CENTER_ERROR_M = 0.05


@dataclass
class G1SkillAPI:
    """Allowlisted static motor SDK for two-G1 box-carry policies."""

    backend: SkillBackend = field(default_factory=StaticSonicSkillBackend)
    output_dir: Path = Path("results/deployment/cap_policy_live")
    heartbeat_file: Path = Path("results/deployment/cap_policy_live/planner_heartbeat.json")
    runtime_state_file: Path | None = None
    _call_index: int = 0
    _records: list[SkillCallRecord] = field(default_factory=list)

    @property
    def records(self) -> tuple[SkillCallRecord, ...]:
        return tuple(self._records)

    def place_payload_on_arms(
        self,
        *,
        payload_z: float | None = None,
        hold_s: float = 4.0,
        force: bool = False,
    ) -> SkillCallRecord:
        """Wait for stable arms, place at the live support pose, and verify support."""

        record = self._call(
            "place_payload_on_arms",
            payload_z=payload_z,
            place_hold_seconds=hold_s,
            payload_force=force,
        )
        self._require_payload_supported()
        return record

    def hold_position(self, *, hold_s: float = 2.0) -> SkillCallRecord:
        """Keep the selected robot scope in planner IDLE for a bounded interval."""

        hold = float(hold_s)
        if hold < 0.5 or hold > 10.0:
            raise ValueError(f"hold_position hold_s must be in [0.5, 10.0], got {hold_s}")
        return self._call("hold_position", hold_seconds=hold)

    def coordinated_linear_transport(
        self,
        *,
        direction: str,
        distance_m: float,
        speed_mps: float,
        ramp_s: float,
    ) -> SkillCallRecord:
        """Describe one robot profile in a paired rigid-payload linear phase."""

        return self._paired_coordination_profile(
            "coordinated_linear_transport",
            direction=direction,
            distance_m=distance_m,
            speed_mps=speed_mps,
            ramp_s=ramp_s,
        )

    def coordinated_arc_transport(
        self,
        *,
        turn_direction: str,
        angle_deg: float,
        duration_s: float,
        radius_m: float,
        travel_direction: str = "forward",
    ) -> SkillCallRecord:
        """Describe one robot's continuous radius in a paired arc phase."""

        return self._paired_coordination_profile(
            "coordinated_arc_transport",
            turn_direction=turn_direction,
            travel_direction=travel_direction,
            angle_deg=angle_deg,
            duration_s=duration_s,
            radius_m=radius_m,
        )

    def coordinated_pivot(
        self,
        *,
        turn_direction: str,
        angle_deg: float,
        duration_s: float,
        role: str,
        swing_radius_m: float,
        travel_direction: str = "forward",
    ) -> SkillCallRecord:
        """Describe an inner or outer robot profile in one paired pivot phase."""

        return self._paired_coordination_profile(
            "coordinated_pivot",
            turn_direction=turn_direction,
            travel_direction=travel_direction,
            angle_deg=angle_deg,
            duration_s=duration_s,
            role=role,
            swing_radius_m=swing_radius_m,
        )

    def coordinated_alignment(
        self,
        *,
        axis: str,
        delta: float,
        duration_s: float,
    ) -> SkillCallRecord:
        """Describe one robot's bounded relative correction in a paired phase."""

        return self._paired_coordination_profile(
            "coordinated_alignment",
            axis=axis,
            delta=delta,
            duration_s=duration_s,
        )

    def carry_forward(
        self,
        *,
        distance_m: float = 1.0,
        speed_mps: float = 0.50,
        ramp_s: float = 1.5,
    ) -> SkillCallRecord:
        """Move both robots forward while keeping the tuned carry arm pose."""

        return self._transport_call(
            "carry_forward", distance_m=distance_m, speed_mps=speed_mps, ramp_s=ramp_s
        )

    def carry_backward(
        self,
        *,
        distance_m: float = 1.0,
        speed_mps: float = 0.50,
        ramp_s: float = 1.5,
    ) -> SkillCallRecord:
        """Move both robots backward while keeping the tuned carry arm pose."""

        return self._transport_call(
            "carry_backward", distance_m=distance_m, speed_mps=speed_mps, ramp_s=ramp_s
        )

    def carry_lateral_left(
        self,
        *,
        distance_m: float = 1.0,
        speed_mps: float = 0.50,
        ramp_s: float = 1.5,
    ) -> SkillCallRecord:
        """Move left relative to the robots' body frame with the carry pose locked."""

        return self._transport_call(
            "carry_lateral_left", distance_m=distance_m, speed_mps=speed_mps, ramp_s=ramp_s
        )

    def carry_lateral_right(
        self,
        *,
        distance_m: float = 1.0,
        speed_mps: float = 0.50,
        ramp_s: float = 1.5,
    ) -> SkillCallRecord:
        """Move right relative to the robots' body frame with the carry pose locked."""

        return self._transport_call(
            "carry_lateral_right", distance_m=distance_m, speed_mps=speed_mps, ramp_s=ramp_s
        )

    def turn_clockwise(
        self,
        *,
        yaw_deg: float = 90.0,
        yaw_rate_degps: float = 5.0,
        turn_segment_deg: float = 3.0,
    ) -> SkillCallRecord:
        """Turn clockwise by a bounded yaw angle. Clockwise is negative yaw."""

        validate_catalog_parameters(
            "turn_clockwise",
            {
                "yaw_deg": yaw_deg,
                "yaw_rate_degps": yaw_rate_degps,
                "turn_segment_deg": turn_segment_deg,
            },
        )
        return self._call(
            "carry_turn_in_place",
            yaw_deg=-abs(yaw_deg),
            yaw_rate_degps=yaw_rate_degps,
            turn_segment_deg=turn_segment_deg,
        )

    def turn_counterclockwise(
        self,
        *,
        yaw_deg: float = 90.0,
        yaw_rate_degps: float = 5.0,
        turn_segment_deg: float = 3.0,
    ) -> SkillCallRecord:
        """Turn counterclockwise by a bounded yaw angle. Counterclockwise is positive yaw."""

        validate_catalog_parameters(
            "turn_counterclockwise",
            {
                "yaw_deg": yaw_deg,
                "yaw_rate_degps": yaw_rate_degps,
                "turn_segment_deg": turn_segment_deg,
            },
        )
        return self._call(
            "carry_turn_in_place",
            yaw_deg=abs(yaw_deg),
            yaw_rate_degps=yaw_rate_degps,
            turn_segment_deg=turn_segment_deg,
        )

    def place_crouch(
        self,
        *,
        place_height: float = 0.55,
        ramp_s: float = 2.0,
        hold_s: float = 10.0,
    ) -> SkillCallRecord:
        """Lower both robots in place while preserving the carry arm pose."""

        validate_catalog_parameters(
            "place_crouch",
            {"place_height": place_height, "ramp_s": ramp_s, "hold_s": hold_s},
        )
        return self._call(
            "carry_place_crouch",
            place_start_height=0.80,
            place_height=place_height,
            place_ramp_seconds=ramp_s,
            place_hold_seconds=hold_s,
        )

    def recover_stand(
        self,
        *,
        height: float = 0.80,
        ramp_s: float = 6.0,
        hold_s: float = 5.0,
    ) -> SkillCallRecord:
        """Recover from a crouch/place posture to upright idle."""

        validate_catalog_parameters(
            "recover_stand",
            {"height": height, "ramp_s": ramp_s, "hold_s": hold_s},
        )
        return self._call(
            "carry_recover_stand",
            height=height,
            place_ramp_seconds=ramp_s,
            place_hold_seconds=hold_s,
        )

    def recover_payload_inward(
        self,
        *,
        ramp_s: float = 3.0,
        hold_s: float = 2.0,
    ) -> SkillCallRecord:
        """Synchronously retract both loaded arm supports from the 70 to 60 degree pose."""

        ramp = float(ramp_s)
        hold = float(hold_s)
        validate_catalog_parameters(
            "recover_payload_inward",
            {"ramp_s": ramp, "hold_s": hold},
        )
        self._require_payload_supported(timeout_s=3.0, stable_s=0.3)
        self._require_payload_recovery_ready()
        record = self._call(
            "carry_recover_payload_inward",
            recovery_ramp_seconds=ramp,
            recovery_hold_seconds=hold,
        )
        self._require_payload_supported(timeout_s=5.0, stable_s=0.5)
        return record

    def release_payload(
        self,
        *,
        ramp_s: float = 4.0,
        hold_s: float = 2.0,
    ) -> SkillCallRecord:
        """Crouch and withdraw both palms only when the target can take the load."""

        ramp = float(ramp_s)
        hold = float(hold_s)
        if ramp < 1.0 or ramp > 8.0:
            raise ValueError(f"release_payload ramp_s must be in [1.0, 8.0], got {ramp_s}")
        if hold < 0.5 or hold > 10.0:
            raise ValueError(f"release_payload hold_s must be in [0.5, 10.0], got {hold_s}")
        self._require_payload_release_ready()
        record = self._call(
            "release_payload",
            height=0.55,
            release_ramp_seconds=ramp,
            release_hold_seconds=hold,
        )
        self._require_payload_on_target(allow_robot_contact=True)
        return record

    def exit_forward(
        self,
        *,
        distance_m: float = 0.50,
        speed_mps: float = 0.20,
        ramp_s: float = 1.5,
    ) -> SkillCallRecord:
        """Walk forward without the payload after verified physical release."""

        return self._exit_call(
            "exit_forward", distance_m=distance_m, speed_mps=speed_mps, ramp_s=ramp_s
        )

    def exit_backward(
        self,
        *,
        distance_m: float = 0.50,
        speed_mps: float = 0.20,
        ramp_s: float = 1.5,
    ) -> SkillCallRecord:
        """Walk backward without the payload after verified physical release."""

        return self._exit_call(
            "exit_backward", distance_m=distance_m, speed_mps=speed_mps, ramp_s=ramp_s
        )

    def exit_lateral_left(
        self,
        *,
        distance_m: float = 0.50,
        speed_mps: float = 0.20,
        ramp_s: float = 1.5,
    ) -> SkillCallRecord:
        """Walk body-left without the payload after verified physical release."""

        return self._exit_call(
            "exit_lateral_left", distance_m=distance_m, speed_mps=speed_mps, ramp_s=ramp_s
        )

    def exit_lateral_right(
        self,
        *,
        distance_m: float = 0.50,
        speed_mps: float = 0.20,
        ramp_s: float = 1.5,
    ) -> SkillCallRecord:
        """Walk body-right without the payload after verified physical release."""

        return self._exit_call(
            "exit_lateral_right", distance_m=distance_m, speed_mps=speed_mps, ramp_s=ramp_s
        )

    def stop_all(self) -> SkillCallRecord:
        """Hold both robots in current idle/carry-ready state."""

        return self._call("stop_all", hold_seconds=1.0)

    def record_coordinated_stop_boundary(
        self,
        *,
        robot_ids: tuple[str, ...],
        reason: str,
    ) -> SkillCallRecord:
        """Audit a script stop already completed by the atomic team publisher."""

        if self._call_index >= MAX_SKILL_CALLS_PER_POLICY:
            raise RuntimeError(
                f"policy exceeded the {MAX_SKILL_CALLS_PER_POLICY}-call physical safety budget"
            )
        robots = tuple(dict.fromkeys(str(item) for item in robot_ids))
        if not robots or not set(robots).issubset({"front", "back"}):
            raise ValueError(f"invalid robot scope: {robots}")
        self._call_index += 1
        now = time.time()
        record = SkillCallRecord(
            index=self._call_index,
            skill_name="stop_all",
            parameters={"hold_seconds": 1.0},
            status="protocol_noop_after_coordinated_idle",
            started_at_unix=now,
            finished_at_unix=now,
            robot_ids=robots,
            effective_parameters={
                "physical_publish": False,
                "reason": str(reason),
            },
        )
        self._records.append(record)
        return record

    def dispatch_planned_call(
        self,
        method: str,
        parameters: dict[str, Any],
        *,
        robot_ids: tuple[str, ...],
    ) -> SkillCallRecord:
        """Dispatch a validated call through a robot-scoped backend."""

        if method not in ALLOWED_G1_SKILL_API_METHODS:
            raise ValueError(f"unknown planned G1SkillAPI method: {method}")
        robots = tuple(dict.fromkeys(str(item) for item in robot_ids))
        if not robots or not set(robots).issubset({"front", "back"}):
            raise ValueError(f"invalid robot scope: {robots}")
        with_robots = getattr(self.backend, "with_robots", None)
        if not callable(with_robots):
            configured = tuple(getattr(self.backend, "robots", ()))
            if configured != robots:
                raise RuntimeError(
                    f"backend cannot scope dispatch from {configured} to {robots}"
                )
            return getattr(self, method)(**parameters)
        original_backend = self.backend
        self.backend = with_robots(robots)
        try:
            return getattr(self, method)(**parameters)
        finally:
            self.backend = original_backend

    def dispatch_coordinated_planned_calls(
        self,
        calls_by_robot: dict[str, PlannedSkillCall],
        *,
        phase_id: str,
        synchronization_group: str,
        coordination_mode: str,
    ) -> SkillCallRecord:
        """Atomically publish a loaded phase with per-robot command payloads."""

        if set(calls_by_robot) != {"front", "back"}:
            raise ValueError("coordinated dispatch requires exactly front and back actions")
        mode = str(coordination_mode).strip().lower()
        if mode not in {"sync", "async"}:
            raise ValueError("coordinated dispatch mode must be sync or async")
        execute_team_actions = getattr(self.backend, "execute_team_actions", None)
        if not callable(execute_team_actions):
            raise RuntimeError("configured SONIC backend does not support coordinated team dispatch")
        if self._call_index >= MAX_SKILL_CALLS_PER_POLICY:
            raise RuntimeError(
                f"policy exceeded the {MAX_SKILL_CALLS_PER_POLICY}-call physical safety budget"
            )
        actions = {
            robot_id: {
                "method": call.method,
                "parameters": dict(call.parameters),
            }
            for robot_id, call in calls_by_robot.items()
        }
        _validate_coordinated_actions(actions, coordination_mode=mode)
        self._require_payload_supported(timeout_s=3.0, stable_s=0.3)
        self._call_index += 1
        call_dir = self.output_dir / (
            f"{self._call_index:02d}_async_team_phase"
            if mode == "async"
            else f"{self._call_index:02d}_sync_compensated_team_phase"
        )
        record = execute_team_actions(
            actions_by_robot=actions,
            phase_id=phase_id,
            synchronization_group=synchronization_group,
            coordination_mode=mode,
            output_dir=call_dir,
            heartbeat_file=self.heartbeat_file,
            runtime_state_file=self.runtime_state_file,
            call_index=self._call_index,
        )
        self._records.append(record)
        if record.status.startswith("failed_rc_"):
            detail = record.stderr.strip() or record.stdout.strip() or "no backend diagnostic"
            raise RuntimeError(
                f"SONIC {mode} team phase failed ({record.status}); subsequent skills are blocked: "
                f"{detail}"
            )
        self._require_payload_supported(timeout_s=5.0, stable_s=0.5)
        return record

    def _paired_coordination_profile(self, method: str, **parameters: Any) -> SkillCallRecord:
        """Reject unpaired direct use while keeping the public SDK contract explicit."""

        validate_coordination_primitive_parameters(method, parameters)
        raise RuntimeError(
            f"api.{method} is a paired coordination primitive; execute it through front/back "
            "robot scripts sharing one synchronization_group"
        )

    def dispatch_asynchronous_planned_calls(
        self,
        calls_by_robot: dict[str, PlannedSkillCall],
        *,
        phase_id: str,
        synchronization_group: str,
    ) -> SkillCallRecord:
        """Compatibility wrapper for the original async-only dispatcher."""

        return self.dispatch_coordinated_planned_calls(
            calls_by_robot,
            phase_id=phase_id,
            synchronization_group=synchronization_group,
            coordination_mode="async",
        )

    def _transport_call(
        self,
        skill_name: str,
        *,
        distance_m: float,
        speed_mps: float,
        ramp_s: float,
    ) -> SkillCallRecord:
        distance = abs(float(distance_m))
        speed = abs(float(speed_mps))
        validate_catalog_parameters(
            skill_name,
            {"distance_m": distance, "speed_mps": speed, "ramp_s": ramp_s},
        )
        if distance <= 0.0 or distance > MAX_TRANSPORT_SEGMENT_M:
            raise ValueError(
                f"{skill_name} distance_m must be in (0, {MAX_TRANSPORT_SEGMENT_M:.2f}], got {distance_m}"
            )
        if speed <= 0.0 or speed > MAX_TRANSPORT_SPEED_MPS:
            raise ValueError(
                f"{skill_name} speed_mps must be in (0, {MAX_TRANSPORT_SPEED_MPS:.2f}], got {speed_mps}"
            )
        ramp = float(ramp_s)
        if ramp < 0.0 or ramp > MAX_TRANSPORT_RAMP_S:
            raise ValueError(
                f"{skill_name} ramp_s must be in [0, {MAX_TRANSPORT_RAMP_S:.1f}], got {ramp_s}"
            )
        self._require_payload_supported(timeout_s=3.0, stable_s=0.3)
        record = self._call(
            skill_name,
            distance_m=distance,
            speed_mps=speed,
            transport_ramp_seconds=ramp,
        )
        self._require_payload_supported(timeout_s=5.0, stable_s=0.5)
        return record

    def _exit_call(
        self,
        skill_name: str,
        *,
        distance_m: float,
        speed_mps: float,
        ramp_s: float,
    ) -> SkillCallRecord:
        distance = abs(float(distance_m))
        speed = abs(float(speed_mps))
        ramp = float(ramp_s)
        if distance <= 0.0 or distance > MAX_EXIT_SEGMENT_M:
            raise ValueError(
                f"{skill_name} distance_m must be in (0, {MAX_EXIT_SEGMENT_M:.2f}], got {distance_m}"
            )
        if speed <= 0.0 or speed > MAX_TRANSPORT_SPEED_MPS:
            raise ValueError(
                f"{skill_name} speed_mps must be in (0, {MAX_TRANSPORT_SPEED_MPS:.2f}], got {speed_mps}"
            )
        if ramp < 0.0 or ramp > MAX_TRANSPORT_RAMP_S:
            raise ValueError(
                f"{skill_name} ramp_s must be in [0, {MAX_TRANSPORT_RAMP_S:.1f}], got {ramp_s}"
            )
        self._require_payload_on_target(
            timeout_s=4.0,
            stable_s=0.3,
            allow_robot_contact=True,
        )
        record = self._call(
            skill_name,
            distance_m=distance,
            speed_mps=speed,
            transport_ramp_seconds=ramp,
        )
        self._require_payload_on_target(
            timeout_s=5.0,
            stable_s=0.5,
            allow_robot_contact=True,
        )
        return record

    def _call(self, skill_name: str, **parameters: Any) -> SkillCallRecord:
        if self._call_index >= MAX_SKILL_CALLS_PER_POLICY:
            raise RuntimeError(
                f"policy exceeded the {MAX_SKILL_CALLS_PER_POLICY}-call physical safety budget"
            )
        self._call_index += 1
        call_dir = self.output_dir / f"{self._call_index:02d}_{skill_name}"
        record = self.backend.execute_skill(
            skill_name=skill_name,
            parameters=dict(parameters),
            output_dir=call_dir,
            heartbeat_file=self.heartbeat_file,
            runtime_state_file=self.runtime_state_file,
            call_index=self._call_index,
        )
        self._records.append(record)
        if record.status.startswith("failed_rc_"):
            detail = record.stderr.strip() or record.stdout.strip() or "no backend diagnostic"
            raise RuntimeError(
                f"SONIC skill {skill_name} failed ({record.status}); subsequent skills are blocked: {detail}"
            )
        return record

    def _require_payload_supported(self, *, timeout_s: float = 8.0, stable_s: float = 1.0) -> None:
        if self.runtime_state_file is None or not bool(getattr(self.backend, "execute", True)):
            return
        max_support_center_xy_error_m = float(
            getattr(
                self.backend,
                "effective_max_payload_support_center_error_m",
                DEFAULT_MAX_PAYLOAD_SUPPORT_CENTER_ERROR_M,
            )
        )
        if max_support_center_xy_error_m < 0.0:
            raise ValueError(
                "backend max_payload_support_center_error_m must be non-negative, "
                f"got {max_support_center_xy_error_m}"
            )
        wait_for_payload_supported(
            self.runtime_state_file,
            timeout_s=timeout_s,
            stable_s=stable_s,
            max_age_s=2.0,
            min_z_m=0.80,
            max_support_center_xy_error_m=max_support_center_xy_error_m,
            allow_non_support_robot_contacts=(
                getattr(self.backend, "payload_safety_profile", "task") == "task"
            ),
            safety_profile=getattr(self.backend, "payload_safety_profile", "task"),
        )

    def _require_payload_release_ready(
        self,
        *,
        timeout_s: float = 8.0,
        stable_s: float = 0.5,
    ) -> None:
        if self.runtime_state_file is None or not bool(getattr(self.backend, "execute", True)):
            return
        wait_for_payload_release_ready(
            self.runtime_state_file,
            timeout_s=timeout_s,
            stable_s=stable_s,
            max_age_s=2.0,
        )

    def _require_payload_recovery_ready(self) -> None:
        """Open the one-way 70-to-50 recovery only for measured outward payload slip."""

        if self.runtime_state_file is None or not bool(getattr(self.backend, "execute", True)):
            return
        robots = tuple(getattr(self.backend, "robots", ()))
        if len(robots) != 2 or set(robots) != {"front", "back"}:
            raise RuntimeError(
                "recover_payload_inward requires one synchronized front/back backend scope"
            )
        profile = get_carry_safety_profile(
            getattr(self.backend, "payload_safety_profile", "task")
        )
        wait_for_payload_recovery_ready(
            Path(self.runtime_state_file),
            timeout_s=3.0,
            stable_s=0.3,
            max_age_s=2.0,
            minimum_outward_offset_m=(
                profile.warning_payload_formation_slip_m
            ),
        )

    def _require_payload_on_target(
        self,
        *,
        timeout_s: float = 12.0,
        stable_s: float = 1.0,
        allow_robot_contact: bool = False,
    ) -> None:
        if self.runtime_state_file is None or not bool(getattr(self.backend, "execute", True)):
            return
        wait_for_payload_on_target(
            self.runtime_state_file,
            timeout_s=timeout_s,
            stable_s=stable_s,
            max_age_s=2.0,
            require_released=True,
            allow_robot_contact=allow_robot_contact,
        )


def _validate_coordinated_actions(
    actions: dict[str, dict[str, Any]],
    *,
    coordination_mode: str,
) -> None:
    for robot_id, action in actions.items():
        method = str(action.get("method", ""))
        parameters = action.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"coordinated action parameters for {robot_id} must be a mapping")
        if method == "hold_position":
            hold = float(parameters.get("hold_s", 2.0))
            if hold < 0.5 or hold > 10.0:
                raise ValueError(
                    f"hold_position hold_s must be in [0.5, 10.0], got {hold}"
                )
            continue
        if method in COORDINATION_PRIMITIVE_METHODS:
            validate_coordination_primitive_parameters(method, parameters)
        else:
            validate_catalog_parameters(method, parameters)
    team_actions = {
        robot_id: TeamAction(
            method=str(action.get("method", "")),
            parameters=dict(action.get("parameters", {})),
        )
        for robot_id, action in actions.items()
    }
    validate_coordination_primitive_team_actions(
        team_actions,
        coordination_mode=coordination_mode,
    )
    if coordination_mode == "sync":
        validate_synchronized_actions(team_actions)
    else:
        validate_asynchronous_actions(team_actions)


def _validate_async_actions(actions: dict[str, dict[str, Any]]) -> None:
    _validate_coordinated_actions(actions, coordination_mode="async")
