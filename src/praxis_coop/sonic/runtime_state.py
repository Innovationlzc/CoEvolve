"""Read and verify the live MuJoCo payload initialization state."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import cos, hypot, isfinite, radians, sin
from pathlib import Path
import time

from .safety import (
    PAYLOAD_RECOVERY_TILT_TRIGGER_DEG,
    PAYLOAD_TOP_FACE_UP_LIMIT_DEG,
    TASK_CARRY_SAFETY,
    CarrySafetyProfile,
    get_carry_safety_profile,
)
from .upper_body import (
    DEFAULT_CARRY_READY_POSE_NAME,
    PAYLOAD_RECOVERY_POSE_NAME,
)


REQUIRED_PAYLOAD_SUPPORTERS = frozenset({"front", "back"})
REQUIRED_PAYLOAD_SUPPORT_GEOMETRIES = frozenset(
    {
        "front_left_carry_support_plate",
        "front_right_carry_support_plate",
        "back_left_carry_support_plate",
        "back_right_carry_support_plate",
    }
)
MIN_INITIAL_PAYLOAD_SUPPORT_MARGIN_M = 0.02


@dataclass(frozen=True)
class PayloadRuntimeStatus:
    """Small, typed view of Term1's continuously written runtime state."""

    available: bool = False
    fresh: bool = False
    age_s: float = float("inf")
    elastic_band_dropped: bool = False
    present: bool = False
    active: bool = False
    support_target_ready: bool = False
    support_tracking_available: bool = False
    support_wait_reasons: tuple[str, ...] = ()
    support_center_xy_error_available: bool = False
    support_center_xy_error_m: float = float("inf")
    z: float = 0.0
    tilt_deg: float = 180.0
    top_surface_corner_z_range_m: float = float("inf")
    linear_speed_mps: float = float("inf")
    angular_speed_radps: float = float("inf")
    supporting_robots: frozenset[str] = frozenset()
    supporting_geometries: frozenset[str] = frozenset()
    non_support_robot_contacts: tuple[str, ...] = ()
    support_polygon_available: bool = False
    support_polygon_com_margin_m: float = float("-inf")
    support_force_available: bool = False
    support_normal_force_n_by_robot: dict[str, float] = field(default_factory=dict)
    support_force_total_n: float = 0.0
    support_force_balance_ratio: float = 0.0
    floor_contact: bool = False
    wall_contacts: tuple[str, ...] = ()
    min_robot_pelvis_height_m: float = 0.0
    robot_xy_separation_available: bool = False
    robot_xy_separation_m: float = 0.0
    control_healthy: bool = False
    runtime_status: str = ""
    runtime_fault: str = ""
    checkpoint_resync_active: bool = False
    error: str = ""

    def support_force_balanced(
        self,
        *,
        min_force_per_robot_n: float = 0.5,
        min_balance_ratio: float = 0.10,
    ) -> bool:
        """Require both robots to carry load when force telemetry is available."""

        if not self.support_force_available:
            return True
        return (
            all(
                self.support_normal_force_n_by_robot.get(robot_id, 0.0)
                >= max(0.0, float(min_force_per_robot_n))
                for robot_id in REQUIRED_PAYLOAD_SUPPORTERS
            )
            and self.support_force_balance_ratio >= max(0.0, float(min_balance_ratio))
        )

    def ready_to_place(self) -> bool:
        return (
            self.available
            and self.fresh
            and self.control_healthy
            and not self.checkpoint_resync_active
            and self.elastic_band_dropped
            and self.present
            and self.support_target_ready
        )

    def all_support_pads_contacting(self) -> bool:
        """Return whether all four authored palm proxy pads contact the box."""

        return REQUIRED_PAYLOAD_SUPPORT_GEOMETRIES.issubset(self.supporting_geometries)

    def payload_support_contacts_healthy(
        self,
        *,
        require_all_support_pads: bool = False,
        min_support_force_per_robot_n: float = 0.0,
        min_support_force_balance_ratio: float = 0.0,
    ) -> bool:
        """Check physical load paths without conflating them with settling speed."""

        return (
            self.available
            and self.fresh
            and self.control_healthy
            and not self.checkpoint_resync_active
            and self.active
            and REQUIRED_PAYLOAD_SUPPORTERS.issubset(self.supporting_robots)
            and (not require_all_support_pads or self.all_support_pads_contacting())
            and self.support_force_balanced(
                min_force_per_robot_n=min_support_force_per_robot_n,
                min_balance_ratio=min_support_force_balance_ratio,
            )
        )

    def payload_supported(
        self,
        *,
        min_z_m: float | None = None,
        max_tilt_deg: float | None = None,
        max_top_surface_z_range_m: float | None = None,
        max_linear_speed_mps: float = 0.05,
        max_angular_speed_radps: float = 0.12,
        max_support_center_xy_error_m: float | None = None,
        require_all_support_pads: bool | None = None,
        min_support_force_per_robot_n: float | None = None,
        min_support_force_balance_ratio: float | None = None,
        allow_wall_contact: bool | None = None,
        support_center_is_hard_failure: bool | None = None,
        min_robot_xy_separation_m: float | None = None,
        min_support_polygon_com_margin_m: float | None = None,
        allow_non_support_robot_contacts: bool = False,
        safety_profile: str | CarrySafetyProfile = "task",
    ) -> bool:
        profile = get_carry_safety_profile(safety_profile)
        min_z_m = profile.min_payload_z_m if min_z_m is None else float(min_z_m)
        max_tilt_deg = (
            profile.settled_max_tilt_deg if max_tilt_deg is None else float(max_tilt_deg)
        )
        max_top_surface_z_range_m = (
            profile.max_top_surface_z_range_m
            if max_top_surface_z_range_m is None
            else float(max_top_surface_z_range_m)
        )
        max_support_center_xy_error_m = (
            profile.max_support_center_error_m
            if max_support_center_xy_error_m is None
            else float(max_support_center_xy_error_m)
        )
        require_all_support_pads = (
            profile.require_all_support_pads_when_settled
            if require_all_support_pads is None
            else bool(require_all_support_pads)
        )
        min_support_force_per_robot_n = (
            profile.min_support_force_per_robot_n
            if min_support_force_per_robot_n is None
            else float(min_support_force_per_robot_n)
        )
        min_support_force_balance_ratio = (
            profile.min_support_force_balance_ratio
            if min_support_force_balance_ratio is None
            else float(min_support_force_balance_ratio)
        )
        allow_wall_contact = (
            not profile.wall_contact_is_hard_failure
            if allow_wall_contact is None
            else bool(allow_wall_contact)
        )
        support_center_is_hard_failure = (
            profile.name == "strict"
            if support_center_is_hard_failure is None
            else bool(support_center_is_hard_failure)
        )
        min_robot_xy_separation_m = (
            profile.min_robot_xy_separation_m
            if min_robot_xy_separation_m is None
            else max(0.0, float(min_robot_xy_separation_m))
        )
        task_contact_state = (
            self.available
            and self.fresh
            and self.control_healthy
            and not self.checkpoint_resync_active
            and self.active
            and self.z > min_z_m
            and self.linear_speed_mps <= max_linear_speed_mps
            and self.angular_speed_radps <= max_angular_speed_radps
            and not self.floor_contact
            and (allow_wall_contact or not self.wall_contacts)
            and REQUIRED_PAYLOAD_SUPPORTERS.issubset(self.supporting_robots)
            and (
                bool(allow_non_support_robot_contacts)
                or not self.non_support_robot_contacts
            )
            and self.robot_xy_separation_available
            and self.robot_xy_separation_m >= min_robot_xy_separation_m
            and (
                min_support_polygon_com_margin_m is None
                or (
                    self.support_polygon_available
                    and self.support_polygon_com_margin_m
                    >= float(min_support_polygon_com_margin_m)
                )
            )
        )
        if profile.name == "task":
            return (
                task_contact_state
                and (
                    not support_center_is_hard_failure
                    or (
                        self.support_center_xy_error_available
                        and self.support_center_xy_error_m
                        <= max_support_center_xy_error_m
                    )
                )
                and (not require_all_support_pads or self.all_support_pads_contacting())
                and self.support_force_balanced(
                    min_force_per_robot_n=min_support_force_per_robot_n,
                    min_balance_ratio=min_support_force_balance_ratio,
                )
            )
        return (
            task_contact_state
            and (
                not support_center_is_hard_failure
                or (
                    self.support_center_xy_error_available
                    and self.support_center_xy_error_m <= max_support_center_xy_error_m
                )
            )
            and self.tilt_deg <= max_tilt_deg
            and self.top_surface_corner_z_range_m <= max_top_surface_z_range_m
            and (not require_all_support_pads or self.all_support_pads_contacting())
            and self.support_force_balanced(
                min_force_per_robot_n=min_support_force_per_robot_n,
                min_balance_ratio=min_support_force_balance_ratio,
            )
        )

    def describe(self) -> str:
        if self.error:
            return self.error
        support_error = (
            f"{self.support_center_xy_error_m:.3f}m"
            if self.support_center_xy_error_available
            else "<unavailable>"
        )
        return (
            f"fresh={self.fresh} age={self.age_s:.2f}s dropped={self.elastic_band_dropped} "
            f"control_healthy={self.control_healthy} runtime_status={self.runtime_status or '<unknown>'} "
            f"checkpoint_resync={self.checkpoint_resync_active} "
            f"present={self.present} active={self.active} target_ready={self.support_target_ready} "
            f"tracking_available={self.support_tracking_available} "
            f"support_xy_error={support_error} "
            f"z={self.z:.3f} tilt={self.tilt_deg:.2f}deg "
            f"top_dz={self.top_surface_corner_z_range_m:.3f}m "
            f"speed={self.linear_speed_mps:.3f}m/s angular={self.angular_speed_radps:.3f}rad/s "
            f"floor={self.floor_contact} walls={list(self.wall_contacts)} "
            f"min_pelvis={self.min_robot_pelvis_height_m:.3f} "
            f"robot_xy_separation={self.robot_xy_separation_m:.3f}m "
            f"supporting={sorted(self.supporting_robots)} "
            f"support_geometries={sorted(self.supporting_geometries)} "
            f"non_support_robot_contacts={list(self.non_support_robot_contacts)} "
            f"support_com_margin={self.support_polygon_com_margin_m:.3f}m "
            f"support_force_available={self.support_force_available} "
            f"support_force_n={self.support_normal_force_n_by_robot} "
            f"support_force_balance={self.support_force_balance_ratio:.3f} "
            f"wait_reasons={list(self.support_wait_reasons)}"
        )


def read_payload_runtime_status(
    path: Path,
    *,
    max_age_s: float = 2.0,
) -> PayloadRuntimeStatus:
    """Read one atomic Term1 snapshot without treating stale files as live state."""

    path = Path(path)
    try:
        stat = path.stat()
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return PayloadRuntimeStatus(error=f"runtime state unavailable at {path}: {exc}")
    if not isinstance(payload, dict):
        return PayloadRuntimeStatus(error=f"runtime state at {path} is not a JSON object")

    age_s = max(0.0, time.time() - stat.st_mtime)
    payload_state = payload.get("payload", {})
    if not isinstance(payload_state, dict):
        payload_state = {}
    contacts = payload_state.get("contacts", {})
    if not isinstance(contacts, dict):
        contacts = {}
    support_polygon = payload_state.get("support_polygon", {})
    if not isinstance(support_polygon, dict):
        support_polygon = {}
    support_forces = contacts.get("support_forces", {})
    if not isinstance(support_forces, dict):
        support_forces = {}
    raw_force_by_robot = support_forces.get("normal_force_n_by_robot", {})
    if not isinstance(raw_force_by_robot, dict):
        raw_force_by_robot = {}
    support_force_by_robot = {
        str(robot_id): max(0.0, _safe_float(force_n))
        for robot_id, force_n in raw_force_by_robot.items()
    }
    target = payload_state.get("live_support_target", {})
    if not isinstance(target, dict):
        target = {}
    payload_x = _optional_float(payload_state.get("x"))
    payload_y = _optional_float(payload_state.get("y"))
    target_x = _optional_float(target.get("x"))
    target_y = _optional_float(target.get("y"))
    support_center_xy_error_available = all(
        value is not None for value in (payload_x, payload_y, target_x, target_y)
    )
    support_center_xy_error_m = (
        hypot(payload_x - target_x, payload_y - target_y)
        if support_center_xy_error_available
        else float("inf")
    )
    control_health = payload.get("control_health", {})
    if not isinstance(control_health, dict):
        control_health = {}
    checkpoint = payload.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    runtime_fault = payload.get("runtime_fault", {})
    if not isinstance(runtime_fault, dict):
        runtime_fault = {}
    robots = payload.get("robots", {})
    robot_heights: list[float] = []
    robot_xy: dict[str, tuple[float, float]] = {}
    if isinstance(robots, dict):
        for robot_id, robot in robots.items():
            if isinstance(robot, dict):
                robot_heights.append(_safe_float(robot.get("pelvis_height_m", 0.0)))
                x = _optional_float(robot.get("x"))
                y = _optional_float(robot.get("y"))
                if x is not None and y is not None:
                    robot_xy[str(robot_id)] = (x, y)
    robot_xy_separation_available = {"front", "back"}.issubset(robot_xy)
    robot_xy_separation_m = (
        hypot(
            robot_xy["front"][0] - robot_xy["back"][0],
            robot_xy["front"][1] - robot_xy["back"][1],
        )
        if robot_xy_separation_available
        else 0.0
    )

    return PayloadRuntimeStatus(
        available=True,
        fresh=age_s <= max(0.0, float(max_age_s)),
        age_s=age_s,
        elastic_band_dropped=bool(payload.get("elastic_band_dropped", False)),
        present=bool(payload_state.get("present", False)),
        active=bool(payload_state.get("active", False)),
        support_target_ready=bool(target.get("ready", False)),
        support_tracking_available=bool(
            target.get("tracking_available", target.get("ready", False))
        ),
        support_wait_reasons=tuple(str(item) for item in target.get("wait_reasons", ())),
        support_center_xy_error_available=support_center_xy_error_available,
        support_center_xy_error_m=support_center_xy_error_m,
        z=_safe_float(payload_state.get("z", 0.0)),
        tilt_deg=_safe_float(payload_state.get("tilt_deg", 180.0)),
        top_surface_corner_z_range_m=_safe_float(
            payload_state.get("top_surface_corner_z_range_m", float("inf"))
        ),
        linear_speed_mps=_safe_float(payload_state.get("linear_speed_mps", float("inf"))),
        angular_speed_radps=_safe_float(payload_state.get("angular_speed_radps", float("inf"))),
        # Only palm-aligned support-pad contacts count as carrying support.
        # Torso, side, or other incidental contacts must not pass this gate.
        supporting_robots=frozenset(str(item) for item in contacts.get("support_robots", ())),
        supporting_geometries=frozenset(
            str(item) for item in contacts.get("support_geometries", ())
        ),
        non_support_robot_contacts=tuple(
            str(item) for item in contacts.get("non_support_robot_contacts", ())
        ),
        support_polygon_available=bool(support_polygon.get("available", False)),
        support_polygon_com_margin_m=_safe_float(
            support_polygon.get("com_margin_m", float("-inf"))
        ),
        support_force_available=bool(support_forces.get("available", False)),
        support_normal_force_n_by_robot=support_force_by_robot,
        support_force_total_n=max(
            0.0, _safe_float(support_forces.get("total_normal_force_n", 0.0))
        ),
        support_force_balance_ratio=max(
            0.0, _safe_float(support_forces.get("balance_ratio", 0.0))
        ),
        floor_contact=bool(contacts.get("floor", False)),
        wall_contacts=tuple(str(item) for item in contacts.get("walls", ())),
        min_robot_pelvis_height_m=min(robot_heights) if robot_heights else 0.0,
        robot_xy_separation_available=robot_xy_separation_available,
        robot_xy_separation_m=robot_xy_separation_m,
        control_healthy=bool(control_health.get("healthy", False)),
        runtime_status=str(payload.get("runtime_status", "")),
        runtime_fault=str(runtime_fault.get("type", "")),
        checkpoint_resync_active=bool(checkpoint.get("resync_active", False)),
    )


def read_runtime_motion_snapshot(path: Path) -> dict[str, object]:
    """Read the compact live state needed to measure one physical skill.

    The snapshot intentionally comes from Term1's MuJoCo state, not from the
    planner command envelope. Missing positions are treated as an experiment
    error instead of being replaced with zero and producing fake progress.
    """

    path = Path(path)
    try:
        stat = path.stat()
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"runtime motion state unavailable at {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"runtime motion state at {path} is not a JSON object")

    payload = state.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("runtime motion state is missing payload")
    target = payload.get("live_support_target")
    if not isinstance(target, dict):
        raise RuntimeError("runtime motion state is missing payload.live_support_target")
    contacts = payload.get("contacts", {})
    if not isinstance(contacts, dict):
        contacts = {}

    robots = state.get("robots")
    if not isinstance(robots, dict) or not robots:
        raise RuntimeError("runtime motion state is missing robots")
    robot_snapshots: dict[str, dict[str, float]] = {}
    for robot_id, robot in robots.items():
        if not isinstance(robot, dict):
            continue
        robot_snapshots[str(robot_id)] = {
            "x": _required_float(robot, "x", f"robots.{robot_id}"),
            "y": _required_float(robot, "y", f"robots.{robot_id}"),
            "yaw_rad": _required_float(robot, "yaw_rad", f"robots.{robot_id}"),
            "pelvis_height_m": _required_float(
                robot,
                "pelvis_height_m",
                f"robots.{robot_id}",
            ),
        }
    if not robot_snapshots:
        raise RuntimeError("runtime motion state has no valid robot positions")

    return {
        "source": "mujoco_live_state",
        "source_path": str(path),
        "captured_at_unix": time.time(),
        "state_file_age_s": max(0.0, time.time() - stat.st_mtime),
        "time_unix": _required_float(state, "time_unix", "runtime"),
        "sim_time_s": _required_float(state, "sim_time_s", "runtime"),
        "mean_heading_yaw_rad": _required_float(
            state,
            "mean_heading_yaw_rad",
            "runtime",
        ),
        "payload": {
            "x": _required_float(payload, "x", "payload"),
            "y": _required_float(payload, "y", "payload"),
            "z": _required_float(payload, "z", "payload"),
            "yaw_rad": _required_float(payload, "yaw_rad", "payload"),
            "tilt_deg": _required_float(payload, "tilt_deg", "payload"),
            "support_target_x": _required_float(
                target,
                "x",
                "payload.live_support_target",
            ),
            "support_target_y": _required_float(
                target,
                "y",
                "payload.live_support_target",
            ),
            "supporting_robots": [
                str(item) for item in contacts.get("support_robots", ())
            ],
            "supporting_geometries": [
                str(item) for item in contacts.get("support_geometries", ())
            ],
            "floor_contact": bool(contacts.get("floor", False)),
            "wall_contacts": [str(item) for item in contacts.get("walls", ())],
        },
        "robots": robot_snapshots,
    }


def compute_runtime_motion_metrics(
    before: dict[str, object],
    after: dict[str, object],
    *,
    target_direction_xy: tuple[float, float],
    reference_frame: str = "body",
) -> dict[str, object]:
    """Measure real payload/team displacement between two MuJoCo snapshots."""

    direction = _unit_xy(target_direction_xy)
    heading = _required_float(before, "mean_heading_yaw_rad", "before")
    if reference_frame == "body":
        direction = (
            cos(heading) * direction[0] - sin(heading) * direction[1],
            sin(heading) * direction[0] + cos(heading) * direction[1],
        )
    elif reference_frame != "world":
        raise ValueError(f"unsupported motion reference frame: {reference_frame}")

    before_payload = _required_mapping(before, "payload", "before")
    after_payload = _required_mapping(after, "payload", "after")
    payload_delta = _point_delta(before_payload, after_payload, "payload")
    payload_metrics = _projected_delta(payload_delta, direction)

    before_robots = _required_mapping(before, "robots", "before")
    after_robots = _required_mapping(after, "robots", "after")
    robot_ids = sorted(set(before_robots) & set(after_robots))
    if not robot_ids:
        raise RuntimeError("motion snapshots have no common robots")
    robot_metrics: dict[str, dict[str, object]] = {}
    robot_deltas: dict[str, tuple[float, float]] = {}
    for robot_id in robot_ids:
        robot_before = _required_mapping(before_robots, robot_id, "before.robots")
        robot_after = _required_mapping(after_robots, robot_id, "after.robots")
        delta = _point_delta(robot_before, robot_after, f"robots.{robot_id}")
        robot_deltas[robot_id] = delta
        robot_metrics[robot_id] = _projected_delta(delta, direction)

    team_delta = (
        sum(delta[0] for delta in robot_deltas.values()) / len(robot_deltas),
        sum(delta[1] for delta in robot_deltas.values()) / len(robot_deltas),
    )
    team_metrics = _projected_delta(team_delta, direction)
    progress_values = {
        robot_id: float(metrics["target_direction_progress_m"])
        for robot_id, metrics in robot_metrics.items()
    }
    progress_gap = max(progress_values.values()) - min(progress_values.values())

    payload_team_slip = (
        payload_delta[0] - team_delta[0],
        payload_delta[1] - team_delta[1],
    )
    support_offset_before = (
        _required_float(before_payload, "x", "before.payload")
        - _required_float(before_payload, "support_target_x", "before.payload"),
        _required_float(before_payload, "y", "before.payload")
        - _required_float(before_payload, "support_target_y", "before.payload"),
    )
    support_offset_after = (
        _required_float(after_payload, "x", "after.payload")
        - _required_float(after_payload, "support_target_x", "after.payload"),
        _required_float(after_payload, "y", "after.payload")
        - _required_float(after_payload, "support_target_y", "after.payload"),
    )
    support_slip_delta = (
        support_offset_after[0] - support_offset_before[0],
        support_offset_after[1] - support_offset_before[1],
    )

    formation_change_m = None
    if {"front", "back"}.issubset(robot_deltas):
        before_front = _required_mapping(before_robots, "front", "before.robots")
        before_back = _required_mapping(before_robots, "back", "before.robots")
        after_front = _required_mapping(after_robots, "front", "after.robots")
        after_back = _required_mapping(after_robots, "back", "after.robots")
        before_relative = (
            _required_float(before_front, "x", "before.robots.front")
            - _required_float(before_back, "x", "before.robots.back"),
            _required_float(before_front, "y", "before.robots.front")
            - _required_float(before_back, "y", "before.robots.back"),
        )
        after_relative = (
            _required_float(after_front, "x", "after.robots.front")
            - _required_float(after_back, "x", "after.robots.back"),
            _required_float(after_front, "y", "after.robots.front")
            - _required_float(after_back, "y", "after.robots.back"),
        )
        formation_change_m = hypot(
            after_relative[0] - before_relative[0],
            after_relative[1] - before_relative[1],
        )

    return {
        "source": "mujoco_before_after_state",
        "elapsed_sim_s": _required_float(after, "sim_time_s", "after")
        - _required_float(before, "sim_time_s", "before"),
        "reference_frame": reference_frame,
        "target_direction_world_xy": [direction[0], direction[1]],
        "payload": payload_metrics,
        "robots": robot_metrics,
        "team_mean": team_metrics,
        "robot_target_progress_gap_m": progress_gap,
        "formation_change_m": formation_change_m,
        "payload_relative_to_team": _projected_delta(payload_team_slip, direction),
        "payload_relative_to_support_target": {
            "before_offset_xy_m": list(support_offset_before),
            "after_offset_xy_m": list(support_offset_after),
            **_projected_delta(support_slip_delta, direction),
        },
    }


def wait_for_payload_supported(
    path: Path,
    *,
    timeout_s: float = 8.0,
    stable_s: float = 1.0,
    poll_s: float = 0.1,
    max_age_s: float = 2.0,
    min_z_m: float | None = None,
    max_support_center_xy_error_m: float | None = None,
    require_all_support_pads: bool | None = None,
    min_support_polygon_com_margin_m: float | None = None,
    allow_non_support_robot_contacts: bool = False,
    safety_profile: str | CarrySafetyProfile = "task",
) -> PayloadRuntimeStatus:
    """Require continuous dual-robot support, not one lucky contact sample."""

    profile = get_carry_safety_profile(safety_profile)
    required_stable_s = max(0.0, float(stable_s))
    min_z_m = profile.min_payload_z_m if min_z_m is None else float(min_z_m)
    max_support_center_xy_error_m = (
        profile.max_support_center_error_m
        if max_support_center_xy_error_m is None
        else float(max_support_center_xy_error_m)
    )
    require_all_support_pads = (
        profile.require_all_support_pads_when_settled
        if require_all_support_pads is None
        else bool(require_all_support_pads)
    )
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    stable_since: float | None = None
    last_status = PayloadRuntimeStatus(error=f"runtime state unavailable at {path}")
    while True:
        now = time.monotonic()
        last_status = read_payload_runtime_status(path, max_age_s=max_age_s)
        if last_status.payload_supported(
            min_z_m=min_z_m,
            max_tilt_deg=profile.settled_max_tilt_deg,
            max_top_surface_z_range_m=profile.max_top_surface_z_range_m,
            max_support_center_xy_error_m=max_support_center_xy_error_m,
            require_all_support_pads=require_all_support_pads,
            min_support_force_per_robot_n=profile.min_support_force_per_robot_n,
            min_support_force_balance_ratio=profile.min_support_force_balance_ratio,
            allow_wall_contact=not profile.wall_contact_is_hard_failure,
            min_support_polygon_com_margin_m=min_support_polygon_com_margin_m,
            allow_non_support_robot_contacts=allow_non_support_robot_contacts,
            safety_profile=profile,
        ):
            if stable_since is None:
                stable_since = now
            if now - stable_since >= required_stable_s:
                return last_status
        else:
            stable_since = None
        if now >= deadline:
            raise RuntimeError(
                "payload initialization failed continuous physical support check: "
                + last_status.describe()
            )
        time.sleep(max(0.01, min(float(poll_s), deadline - now)))


def observe_payload_carry_stability(
    path: Path,
    *,
    timeout_s: float = 8.0,
    stable_s: float = 2.0,
    poll_s: float = 0.1,
    max_age_s: float = 2.0,
    safety_profile: str | CarrySafetyProfile = "task",
    contact_loss_grace_s: float | None = None,
    min_z_m: float | None = None,
    min_robot_pelvis_height_m: float | None = None,
    min_robot_xy_separation_m: float | None = None,
    max_tilt_deg: float | None = None,
    max_top_surface_z_range_m: float | None = None,
    max_support_center_xy_error_m: float | None = None,
    min_support_force_per_robot_n: float | None = None,
    min_support_force_balance_ratio: float | None = None,
) -> dict[str, object]:
    """Latch task failures and require a continuous settled carry window.

    A single low-force gait-transfer sample, four-pad mismatch, formation
    offset, or support-center offset does not reject an otherwise carried box.
    The selected profile decides which physical events are hard failures.
    """

    profile = get_carry_safety_profile(safety_profile)
    required_stable_s = max(0.0, float(stable_s))
    contact_loss_grace_s = (
        profile.support_loss_grace_s
        if contact_loss_grace_s is None
        else float(contact_loss_grace_s)
    )
    min_z_m = profile.min_payload_z_m if min_z_m is None else float(min_z_m)
    min_robot_pelvis_height_m = (
        profile.min_robot_pelvis_height_m
        if min_robot_pelvis_height_m is None
        else float(min_robot_pelvis_height_m)
    )
    min_robot_xy_separation_m = (
        profile.min_robot_xy_separation_m
        if min_robot_xy_separation_m is None
        else float(min_robot_xy_separation_m)
    )
    max_tilt_deg = (
        profile.settled_max_tilt_deg if max_tilt_deg is None else float(max_tilt_deg)
    )
    max_top_surface_z_range_m = (
        profile.max_top_surface_z_range_m
        if max_top_surface_z_range_m is None
        else float(max_top_surface_z_range_m)
    )
    max_support_center_xy_error_m = (
        profile.max_support_center_error_m
        if max_support_center_xy_error_m is None
        else float(max_support_center_xy_error_m)
    )
    min_support_force_per_robot_n = (
        profile.min_support_force_per_robot_n
        if min_support_force_per_robot_n is None
        else float(min_support_force_per_robot_n)
    )
    min_support_force_balance_ratio = (
        profile.min_support_force_balance_ratio
        if min_support_force_balance_ratio is None
        else float(min_support_force_balance_ratio)
    )

    started = time.monotonic()
    deadline = started + max(0.0, float(timeout_s))
    stable_since: float | None = None
    unsupported_since: float | None = None
    sample_count = 0
    worst = {
        "min_payload_z_m": float("inf"),
        "max_payload_tilt_deg": 0.0,
        "max_top_surface_z_range_m": 0.0,
        "max_support_center_xy_error_m": 0.0,
        "min_robot_pelvis_height_m": float("inf"),
        "min_robot_xy_separation_m": float("inf"),
        "min_support_force_per_robot_n": float("inf"),
        "min_support_force_balance_ratio": float("inf"),
        "floor_contact_seen": False,
        "wall_contacts_seen": [],
    }
    last = PayloadRuntimeStatus(error=f"runtime state unavailable at {path}")

    while True:
        now = time.monotonic()
        last = read_payload_runtime_status(path, max_age_s=max_age_s)
        sample_count += 1
        _update_stability_extrema(worst, last)
        failure_type, reason = _hard_carry_failure(
            last,
            min_z_m=min_z_m,
            min_robot_pelvis_height_m=min_robot_pelvis_height_m,
            min_robot_xy_separation_m=min_robot_xy_separation_m,
            max_tilt_deg=profile.hard_max_tilt_deg,
            max_top_surface_z_range_m=max_top_surface_z_range_m,
            wall_contact_is_hard_failure=profile.wall_contact_is_hard_failure,
            enforce_geometry_limits=profile.name == "strict",
        )
        if failure_type:
            return _stability_result(
                success=False,
                failure_type=failure_type,
                reason=reason,
                started=started,
                stable_since=stable_since,
                sample_count=sample_count,
                worst=worst,
                status=last,
            )

        support_contacts_healthy = last.payload_support_contacts_healthy(
            require_all_support_pads=profile.require_all_support_pads_when_settled,
            min_support_force_per_robot_n=min_support_force_per_robot_n,
            min_support_force_balance_ratio=min_support_force_balance_ratio,
        )
        supported = last.payload_supported(
            min_z_m=min_z_m,
            max_tilt_deg=max_tilt_deg,
            max_top_surface_z_range_m=max_top_surface_z_range_m,
            max_support_center_xy_error_m=max_support_center_xy_error_m,
            require_all_support_pads=profile.require_all_support_pads_when_settled,
            min_support_force_per_robot_n=min_support_force_per_robot_n,
            min_support_force_balance_ratio=min_support_force_balance_ratio,
            allow_wall_contact=not profile.wall_contact_is_hard_failure,
            allow_non_support_robot_contacts=profile.name == "task",
            safety_profile=profile,
        )
        if supported:
            unsupported_since = None
            if stable_since is None:
                stable_since = now
            if now - stable_since >= required_stable_s:
                return _stability_result(
                    success=True,
                    failure_type="none",
                    reason=(
                        "continuous settled dual-robot hand/forearm carry window passed"
                    ),
                    started=started,
                    stable_since=stable_since,
                    sample_count=sample_count,
                    worst=worst,
                    status=last,
                )
        else:
            stable_since = None
            if support_contacts_healthy:
                # The payload still has both robot load paths. Velocity settling
                # gets the full timeout instead of being mislabeled as contact loss.
                unsupported_since = None
            else:
                if unsupported_since is None:
                    unsupported_since = now
                if (
                    last.available
                    and last.fresh
                    and last.control_healthy
                    and now - unsupported_since >= max(0.0, float(contact_loss_grace_s))
                ):
                    return _stability_result(
                        success=False,
                        failure_type="payload_not_supported",
                        reason=(
                            "continuous dual-robot hand/forearm support did not recover within "
                            f"{contact_loss_grace_s:.2f}s: {last.describe()}"
                        ),
                        started=started,
                        stable_since=stable_since,
                        sample_count=sample_count,
                        worst=worst,
                        status=last,
                    )

        if now >= deadline:
            if not last.available or not last.fresh or not last.control_healthy:
                failure_type = "runtime_state_missing"
            elif support_contacts_healthy and profile.name == "task":
                result = _stability_result(
                    success=True,
                    failure_type="none",
                    reason=(
                        "task-profile carry remained bilaterally supported and free "
                        "of hard physical failures, but did not fully settle before "
                        "the observation timeout"
                    ),
                    started=started,
                    stable_since=stable_since,
                    sample_count=sample_count,
                    worst=worst,
                    status=last,
                )
                result.update(
                    {
                        "settled": False,
                        "quality_warning": {
                            "type": "residual_motion_at_stability_timeout",
                            "reason": (
                                "dual upper-limb support remained healthy; residual "
                                "payload motion is retained as quality evidence instead "
                                "of being mislabeled as an unsafe carry failure"
                            ),
                            "linear_speed_mps": last.linear_speed_mps,
                            "angular_speed_radps": last.angular_speed_radps,
                        },
                    }
                )
                return result
            elif support_contacts_healthy:
                failure_type = "payload_unstable"
            else:
                failure_type = "payload_not_supported"
            return _stability_result(
                success=False,
                failure_type=failure_type,
                reason="post-action stability window timed out: " + last.describe(),
                started=started,
                stable_since=stable_since,
                sample_count=sample_count,
                worst=worst,
                status=last,
            )
        time.sleep(max(0.01, min(float(poll_s), deadline - now)))


def _hard_carry_failure(
    status: PayloadRuntimeStatus,
    *,
    min_z_m: float,
    min_robot_pelvis_height_m: float,
    min_robot_xy_separation_m: float = 0.0,
    max_tilt_deg: float,
    max_top_surface_z_range_m: float,
    wall_contact_is_hard_failure: bool,
    enforce_geometry_limits: bool = False,
) -> tuple[str, str]:
    if status.available and status.active and (
        status.floor_contact or status.z <= min_z_m - 0.05
    ):
        return "payload_drop", status.describe()
    if status.wall_contacts and wall_contact_is_hard_failure:
        return "payload_wall_contact", status.describe()
    if status.available and status.min_robot_pelvis_height_m < min_robot_pelvis_height_m:
        return "robot_fall", status.describe()
    if (
        status.available
        and status.robot_xy_separation_available
        and status.robot_xy_separation_m < min_robot_xy_separation_m
    ):
        return "formation_desync", status.describe()
    # The top-face boundary is a strict-profile geometry check. In task mode,
    # legal bilateral upper-limb support and floor clearance are authoritative;
    # orientation remains diagnostic so a supported transient is not confused
    # with a dropped payload.
    if (
        enforce_geometry_limits
        and
        status.available
        and status.active
        and status.tilt_deg >= PAYLOAD_TOP_FACE_UP_LIMIT_DEG
    ):
        return "payload_unstable", status.describe()
    if (
        enforce_geometry_limits
        and status.available
        and (
            status.tilt_deg > max_tilt_deg
            or status.top_surface_corner_z_range_m > max_top_surface_z_range_m
        )
    ):
        return "payload_unstable", status.describe()
    if status.available and not status.active:
        return "payload_not_active", status.describe()
    return "", ""


def _update_stability_extrema(
    worst: dict[str, object],
    status: PayloadRuntimeStatus,
) -> None:
    worst["min_payload_z_m"] = min(float(worst["min_payload_z_m"]), status.z)
    worst["max_payload_tilt_deg"] = max(
        float(worst["max_payload_tilt_deg"]), status.tilt_deg
    )
    worst["max_top_surface_z_range_m"] = max(
        float(worst["max_top_surface_z_range_m"]),
        status.top_surface_corner_z_range_m,
    )
    if status.support_center_xy_error_available:
        worst["max_support_center_xy_error_m"] = max(
            float(worst["max_support_center_xy_error_m"]),
            status.support_center_xy_error_m,
        )
    worst["min_robot_pelvis_height_m"] = min(
        float(worst["min_robot_pelvis_height_m"]),
        status.min_robot_pelvis_height_m,
    )
    if status.robot_xy_separation_available:
        worst["min_robot_xy_separation_m"] = min(
            float(worst["min_robot_xy_separation_m"]),
            status.robot_xy_separation_m,
        )
    if status.support_force_available:
        min_force = min(
            status.support_normal_force_n_by_robot.get("front", 0.0),
            status.support_normal_force_n_by_robot.get("back", 0.0),
        )
        worst["min_support_force_per_robot_n"] = min(
            float(worst["min_support_force_per_robot_n"]), min_force
        )
        worst["min_support_force_balance_ratio"] = min(
            float(worst["min_support_force_balance_ratio"]),
            status.support_force_balance_ratio,
        )
    worst["floor_contact_seen"] = bool(worst["floor_contact_seen"]) or status.floor_contact
    walls = set(str(item) for item in worst["wall_contacts_seen"])
    walls.update(status.wall_contacts)
    worst["wall_contacts_seen"] = sorted(walls)


def _stability_result(
    *,
    success: bool,
    failure_type: str,
    reason: str,
    started: float,
    stable_since: float | None,
    sample_count: int,
    worst: dict[str, object],
    status: PayloadRuntimeStatus,
) -> dict[str, object]:
    now = time.monotonic()
    serializable_worst = {
        key: (None if isinstance(value, float) and value == float("inf") else value)
        for key, value in worst.items()
    }
    return {
        "success": bool(success),
        "failure_type": str(failure_type),
        "reason": str(reason),
        "elapsed_s": now - started,
        "stable_duration_s": 0.0 if stable_since is None else now - stable_since,
        "sample_count": int(sample_count),
        "worst_observed": serializable_worst,
        "final_status": {
            "description": status.describe(),
            "z_m": status.z,
            "tilt_deg": status.tilt_deg,
            "support_center_xy_error_m": (
                status.support_center_xy_error_m
                if status.support_center_xy_error_available
                else None
            ),
            "supporting_robots": sorted(status.supporting_robots),
            "supporting_geometries": sorted(status.supporting_geometries),
            "support_force_n_by_robot": status.support_normal_force_n_by_robot,
            "support_force_balance_ratio": status.support_force_balance_ratio,
            "floor_contact": status.floor_contact,
            "wall_contacts": list(status.wall_contacts),
            "min_robot_pelvis_height_m": status.min_robot_pelvis_height_m,
            "robot_xy_separation_available": status.robot_xy_separation_available,
            "robot_xy_separation_m": status.robot_xy_separation_m,
        },
    }


def wait_for_payload_initialization_ready(
    path: Path,
    *,
    timeout_s: float = 45.0,
    stable_s: float = 1.0,
    poll_s: float = 0.1,
    max_age_s: float = 2.0,
    min_z_m: float | None = None,
    max_support_center_xy_error_m: float | None = None,
    require_all_support_pads: bool | None = None,
    allow_non_support_robot_contacts: bool = False,
    safety_profile: str | CarrySafetyProfile = "task",
) -> PayloadRuntimeStatus:
    """Wait for either a deferred placement gate or an already-supported payload."""

    profile = get_carry_safety_profile(safety_profile)
    min_z_m = profile.min_payload_z_m if min_z_m is None else float(min_z_m)
    max_support_center_xy_error_m = (
        profile.max_support_center_error_m
        if max_support_center_xy_error_m is None
        else float(max_support_center_xy_error_m)
    )
    require_all_support_pads = (
        profile.require_all_support_pads_when_settled
        if require_all_support_pads is None
        else bool(require_all_support_pads)
    )
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    stable_since: float | None = None
    last_status = PayloadRuntimeStatus(error=f"runtime state unavailable at {path}")
    while True:
        now = time.monotonic()
        last_status = read_payload_runtime_status(path, max_age_s=max_age_s)
        ready = (
            not last_status.active and last_status.ready_to_place()
        ) or last_status.payload_supported(
            min_z_m=min_z_m,
            max_tilt_deg=profile.settled_max_tilt_deg,
            max_top_surface_z_range_m=profile.max_top_surface_z_range_m,
            max_support_center_xy_error_m=max_support_center_xy_error_m,
            require_all_support_pads=require_all_support_pads,
            min_support_force_per_robot_n=profile.min_support_force_per_robot_n,
            min_support_force_balance_ratio=profile.min_support_force_balance_ratio,
            allow_wall_contact=not profile.wall_contact_is_hard_failure,
            allow_non_support_robot_contacts=allow_non_support_robot_contacts,
            safety_profile=profile,
        )
        if ready:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= max(0.0, float(stable_s)):
                return last_status
        else:
            stable_since = None
        if now >= deadline:
            raise RuntimeError(
                "payload initialization gate did not become ready: " + last_status.describe()
            )
        time.sleep(max(0.01, min(float(poll_s), deadline - now)))


def wait_for_payload_release_ready(
    path: Path,
    *,
    timeout_s: float = 8.0,
    stable_s: float = 0.5,
    poll_s: float = 0.1,
    max_age_s: float = 2.0,
    target_xy_tolerance_m: float | None = None,
    max_drop_to_target_m: float = 0.45,
) -> dict:
    """Require a loaded, crouched payload directly above the physical target."""

    return _wait_for_target_predicate(
        path,
        timeout_s=timeout_s,
        stable_s=stable_s,
        poll_s=poll_s,
        max_age_s=max_age_s,
        label="payload release gate",
        predicate=lambda state: _payload_release_ready(
            state,
            target_xy_tolerance_m=target_xy_tolerance_m,
            max_drop_to_target_m=max_drop_to_target_m,
        ),
    )


def wait_for_payload_on_target(
    path: Path,
    *,
    timeout_s: float = 12.0,
    stable_s: float = 1.0,
    poll_s: float = 0.1,
    max_age_s: float = 2.0,
    target_xy_tolerance_m: float | None = None,
    target_z_tolerance_m: float | None = None,
    require_released: bool = True,
    allow_robot_contact: bool = False,
) -> dict:
    """Wait until the released box is physically supported by the target.

    ``allow_robot_contact`` is reserved for the release/retreat transition.  Final
    task completion keeps the default strict zero-robot-contact requirement.
    """

    return _wait_for_target_predicate(
        path,
        timeout_s=timeout_s,
        stable_s=stable_s,
        poll_s=poll_s,
        max_age_s=max_age_s,
        label="payload-on-target gate",
        predicate=lambda state: _payload_on_target(
            state,
            target_xy_tolerance_m=target_xy_tolerance_m,
            target_z_tolerance_m=target_z_tolerance_m,
            require_released=require_released,
            allow_robot_contact=allow_robot_contact,
        ),
    )


def wait_for_upper_body_pose_transition(
    path: Path,
    *,
    pose_name: str,
    event_id: str = "",
    timeout_s: float = 8.0,
    stable_s: float = 0.2,
    poll_s: float = 0.1,
    max_age_s: float = 2.0,
) -> dict:
    """Wait until Term1 confirms a synchronized motor-side pose event completed."""

    expected_pose = str(pose_name).strip()
    expected_event_id = str(event_id).strip()
    if not expected_pose:
        raise ValueError("pose_name must be non-empty")
    return _wait_for_target_predicate(
        path,
        timeout_s=timeout_s,
        stable_s=stable_s,
        poll_s=poll_s,
        max_age_s=max_age_s,
        label=f"upper-body pose transition to {expected_pose}",
        predicate=lambda state: _upper_body_pose_transition_complete(
            state,
            pose_name=expected_pose,
            event_id=expected_event_id,
        ),
    )


def wait_for_payload_recovery_ready(
    path: Path,
    *,
    timeout_s: float = 3.0,
    stable_s: float = 0.3,
    poll_s: float = 0.1,
    max_age_s: float = 2.0,
    minimum_outward_offset_m: float | None = None,
) -> dict:
    """Require the one legal elbow60-to-elbow50 loaded recovery state."""

    threshold_m = (
        TASK_CARRY_SAFETY.warning_payload_formation_slip_m
        if minimum_outward_offset_m is None
        else max(0.0, float(minimum_outward_offset_m))
    )
    return _wait_for_target_predicate(
        path,
        timeout_s=timeout_s,
        stable_s=stable_s,
        poll_s=poll_s,
        max_age_s=max_age_s,
        label="payload inward-recovery gate",
        predicate=lambda state: _payload_recovery_ready(
            state,
            minimum_outward_offset_m=threshold_m,
        ),
    )


def payload_recovery_outward_evidence_m(state: dict) -> float | None:
    """Return stable outward-slip evidence for one accepted checkpoint.

    The live payload can settle a few millimeters between model observation
    and skill dispatch.  A successful restore identifies the same trusted
    physical transaction, so retain its saved offset as evidence instead of
    turning that settling motion into a recovery-gate race.
    """

    candidates: list[float] = []
    payload = state.get("payload", {}) if isinstance(state, dict) else {}
    if isinstance(payload, dict):
        current = _optional_float(
            payload.get("support_center_signed_longitudinal_offset_m")
        )
        if current is not None and isfinite(current):
            candidates.append(current)

    checkpoint = state.get("checkpoint", {}) if isinstance(state, dict) else {}
    if (
        isinstance(checkpoint, dict)
        and str(checkpoint.get("action", "")) == "restore"
        and bool(checkpoint.get("success", False))
        and bool(str(checkpoint.get("checkpoint_id", "")).strip())
    ):
        reference = checkpoint.get("reference_pose", {})
        reference_payload = (
            reference.get("payload", {}) if isinstance(reference, dict) else {}
        )
        if isinstance(reference_payload, dict):
            saved = _optional_float(
                reference_payload.get(
                    "support_center_signed_longitudinal_offset_m"
                )
            )
            if saved is not None and isfinite(saved):
                candidates.append(saved)

    return max(candidates) if candidates else None


def payload_recovery_tilt_evidence_deg(state: dict) -> float | None:
    """Return the worst trusted current/checkpoint payload tilt."""

    candidates: list[float] = []
    payload = state.get("payload", {}) if isinstance(state, dict) else {}
    if isinstance(payload, dict):
        current = _optional_float(payload.get("tilt_deg"))
        if current is not None and isfinite(current):
            candidates.append(current)

    checkpoint = state.get("checkpoint", {}) if isinstance(state, dict) else {}
    if (
        isinstance(checkpoint, dict)
        and str(checkpoint.get("action", "")) == "restore"
        and bool(checkpoint.get("success", False))
        and bool(str(checkpoint.get("checkpoint_id", "")).strip())
    ):
        reference = checkpoint.get("reference_pose", {})
        reference_payload = (
            reference.get("payload", {}) if isinstance(reference, dict) else {}
        )
        if isinstance(reference_payload, dict):
            saved = _optional_float(reference_payload.get("tilt_deg"))
            if saved is not None and isfinite(saved):
                candidates.append(saved)

    return max(candidates) if candidates else None


def payload_recovery_trigger(
    state: dict,
    *,
    minimum_outward_offset_m: float | None = None,
) -> dict[str, object]:
    """Describe the single preventive arm-recovery trigger."""

    threshold_m = (
        TASK_CARRY_SAFETY.warning_payload_formation_slip_m
        if minimum_outward_offset_m is None
        else max(0.0, float(minimum_outward_offset_m))
    )
    outward_offset_m = payload_recovery_outward_evidence_m(state)
    tilt_deg = payload_recovery_tilt_evidence_deg(state)
    outward_slip = bool(
        outward_offset_m is not None and outward_offset_m >= threshold_m
    )
    high_tilt = bool(
        tilt_deg is not None
        and PAYLOAD_RECOVERY_TILT_TRIGGER_DEG
        <= tilt_deg
        < PAYLOAD_TOP_FACE_UP_LIMIT_DEG
    )
    reasons = [
        reason
        for reason, active in (
            ("body_forward_slip", outward_slip),
            ("high_tilt_before_overturn", high_tilt),
        )
        if active
    ]
    return {
        "active": bool(reasons),
        "reasons": reasons,
        "body_forward_offset_m": outward_offset_m,
        "minimum_body_forward_offset_m": threshold_m,
        "tilt_deg": tilt_deg,
        "minimum_tilt_deg": PAYLOAD_RECOVERY_TILT_TRIGGER_DEG,
        "top_face_up_limit_deg": PAYLOAD_TOP_FACE_UP_LIMIT_DEG,
    }


def payload_support_center_error_m(state: dict) -> float | None:
    """Measure payload COM displacement from the live dual-arm support target."""

    payload = state.get("payload", {}) if isinstance(state, dict) else {}
    target = payload.get("live_support_target", {}) if isinstance(payload, dict) else {}
    if not isinstance(target, dict):
        return None
    payload_x = _optional_float(payload.get("x"))
    payload_y = _optional_float(payload.get("y"))
    target_x = _optional_float(target.get("x"))
    target_y = _optional_float(target.get("y"))
    if any(
        value is None
        for value in (payload_x, payload_y, target_x, target_y)
    ):
        return None
    return hypot(payload_x - target_x, payload_y - target_y)


def _wait_for_target_predicate(
    path: Path,
    *,
    timeout_s: float,
    stable_s: float,
    poll_s: float,
    max_age_s: float,
    label: str,
    predicate,
) -> dict:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    stable_since: float | None = None
    last_state: dict = {}
    last_reason = "runtime state unavailable"
    while True:
        now = time.monotonic()
        last_state, state_reason = _read_fresh_runtime_state(path, max_age_s=max_age_s)
        ready, predicate_reason = predicate(last_state) if last_state else (False, state_reason)
        last_reason = predicate_reason or state_reason
        if ready:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= max(0.0, float(stable_s)):
                return last_state
        else:
            stable_since = None
        if now >= deadline:
            raise RuntimeError(f"{label} did not pass: {last_reason}")
        time.sleep(max(0.01, min(float(poll_s), deadline - now)))


def _read_fresh_runtime_state(path: Path, *, max_age_s: float) -> tuple[dict, str]:
    path = Path(path)
    try:
        stat = path.stat()
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"runtime state unavailable at {path}: {exc}"
    if not isinstance(state, dict):
        return {}, "runtime state is not an object"
    age_s = max(0.0, time.time() - stat.st_mtime)
    if age_s > max(0.0, float(max_age_s)):
        return {}, f"runtime state is stale: age={age_s:.2f}s"
    return state, ""


def _upper_body_pose_transition_complete(
    state: dict,
    *,
    pose_name: str,
    event_id: str,
) -> tuple[bool, str]:
    lock = state.get("upper_body_pose_lock", {})
    if not isinstance(lock, dict) or not bool(lock.get("configured", False)):
        return False, "motor-side upper-body pose lock is unavailable"
    last_event = lock.get("last_event", {})
    if not isinstance(last_event, dict):
        return False, "upper-body runtime event status is unavailable"
    if str(last_event.get("event", "")) != "recover_payload_inward":
        return False, "Term1 has not accepted recover_payload_inward"
    if event_id and str(last_event.get("event_id", "")) != event_id:
        return False, "Term1 is reporting a different upper-body event id"
    if str(lock.get("target_pose_name", "")) != pose_name:
        return False, f"upper-body target is not {pose_name}"
    if bool(lock.get("transition_active", False)):
        return False, "upper-body pose transition is still active"
    if str(lock.get("pose_name", "")) != pose_name:
        return False, f"completed upper-body pose is not {pose_name}"
    if not bool(last_event.get("success", False)) or not bool(
        last_event.get("completed", False)
    ):
        error = str(last_event.get("error", "")).strip()
        return False, error or "upper-body runtime event has not completed"
    return True, ""


def _payload_recovery_ready(
    state: dict,
    *,
    minimum_outward_offset_m: float,
) -> tuple[bool, str]:
    """Validate a stationary, dual-supported, one-way arm-recovery precondition."""

    if not isinstance(state, dict):
        return False, "runtime state is not an object"
    if str(state.get("runtime_status", "running")) != "running":
        return False, "Term1 runtime is not running"
    control = state.get("control_health", {})
    if isinstance(control, dict) and not bool(control.get("healthy", False)):
        return False, "dual SONIC LowCmd streams are not healthy"
    checkpoint = state.get("checkpoint", {})
    if isinstance(checkpoint, dict) and bool(checkpoint.get("resync_active", False)):
        return False, "checkpoint controller resynchronization is still active"

    lock = state.get("upper_body_pose_lock", {})
    if not isinstance(lock, dict) or not bool(lock.get("configured", False)):
        return False, "motor-side upper-body pose lock is unavailable"
    current_pose = str(lock.get("pose_name", ""))
    target_pose = str(lock.get("target_pose_name", current_pose))
    if bool(lock.get("transition_active", False)):
        return False, "another upper-body pose transition is active"
    if current_pose == PAYLOAD_RECOVERY_POSE_NAME or target_pose == PAYLOAD_RECOVERY_POSE_NAME:
        return False, f"payload recovery is already complete at {PAYLOAD_RECOVERY_POSE_NAME}"
    if current_pose != DEFAULT_CARRY_READY_POSE_NAME or target_pose != current_pose:
        return False, (
            "payload recovery requires the settled starting pose "
            f"{DEFAULT_CARRY_READY_POSE_NAME}"
        )

    payload = state.get("payload", {})
    if not isinstance(payload, dict) or not bool(payload.get("active", False)):
        return False, "payload is not active"
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    if bool(contacts.get("floor", False)):
        return False, "payload is touching the floor"
    supporting = frozenset(str(item) for item in contacts.get("support_robots", ()))
    if not REQUIRED_PAYLOAD_SUPPORTERS.issubset(supporting):
        return False, "continuous front/back upper-limb support is required"
    support_error_m = payload_support_center_error_m(state)
    if support_error_m is None:
        return False, "live dual-arm support-center tracking is unavailable"
    if support_error_m > TASK_CARRY_SAFETY.max_support_center_error_m:
        return False, (
            "payload is outside the arm-recovery support envelope: "
            f"support_center_error={support_error_m:.3f}m, "
            f"maximum={TASK_CARRY_SAFETY.max_support_center_error_m:.3f}m"
        )
    trigger = payload_recovery_trigger(
        state,
        minimum_outward_offset_m=minimum_outward_offset_m,
    )
    if not bool(trigger["active"]):
        return False, (
            "payload has not reached either preventive arm-recovery gate: "
            f"body_forward_offset={trigger['body_forward_offset_m']!r}m "
            f"(required={trigger['minimum_body_forward_offset_m']:.3f}m), "
            f"tilt={trigger['tilt_deg']!r}deg "
            f"(required={trigger['minimum_tilt_deg']:.1f}deg while top face is up)"
        )
    return True, ""


def _payload_release_ready(
    state: dict,
    *,
    target_xy_tolerance_m: float | None,
    max_drop_to_target_m: float,
) -> tuple[bool, str]:
    payload = state.get("payload", {})
    target = state.get("target", {})
    contacts = payload.get("contacts", {}) if isinstance(payload, dict) else {}
    robots = state.get("robots", {})
    control = state.get("control_health", {})
    if not isinstance(payload, dict) or not isinstance(target, dict):
        return False, "payload or target state is missing"
    if not bool(control.get("healthy", False)):
        return False, "SONIC control is not healthy"
    if not bool(payload.get("active", False)):
        return False, "payload is not active"
    if not {"front", "back"}.issubset(set(contacts.get("support_robots", ()))):
        return False, "both robots are not carrying the payload before release"
    if bool(contacts.get("floor", False)) or bool(contacts.get("walls", ())):
        return False, "payload has floor or wall contact before release"
    tilt_deg = _safe_float(payload.get("tilt_deg"))
    if tilt_deg >= PAYLOAD_TOP_FACE_UP_LIMIT_DEG:
        return False, f"payload top face is not upward (tilt={tilt_deg:.2f}deg)"
    support_polygon = payload.get("support_polygon", {})
    if (
        isinstance(support_polygon, dict)
        and support_polygon.get("available", False)
        and not support_polygon.get("inside", False)
    ):
        return False, "payload COM is outside the measured bilateral support polygon"
    if not bool(target.get("available", False)):
        return False, "physical target geometry is unavailable"
    xy_error = hypot(
        _safe_float(payload.get("x")) - _safe_float(target.get("center_x_m")),
        _safe_float(payload.get("y")) - _safe_float(target.get("center_y_m")),
    )
    target_support_radius_m = max(0.0, _safe_float(target.get("radius_m")))
    allowed_xy_error_m = (
        target_support_radius_m
        if target_xy_tolerance_m is None
        else max(0.0, float(target_xy_tolerance_m))
    )
    if xy_error > allowed_xy_error_m:
        return (
            False,
            "payload COM projection is outside the target support radius "
            f"(center_error={xy_error:.3f}m, limit={allowed_xy_error_m:.3f}m)",
        )
    target_payload_z = _safe_float(target.get("payload_center_target_z_m"))
    drop = _safe_float(payload.get("z")) - target_payload_z
    if drop < -0.03 or drop > max(0.0, float(max_drop_to_target_m)):
        return False, f"payload release height offset is {drop:.3f}m"
    if drop > 0.30 and tilt_deg > 20.0:
        return False, (
            "payload must be level or pre-lowered before release "
            f"(height_offset={drop:.3f}m, tilt={tilt_deg:.2f}deg)"
        )
    pelvis_heights = [
        _safe_float(robot.get("pelvis_height_m"))
        for robot in robots.values()
        if isinstance(robot, dict)
    ] if isinstance(robots, dict) else []
    if len(pelvis_heights) != 2 or min(pelvis_heights) < 0.35:
        return False, "one or both robots are not upright enough to release"
    return True, "ready"


def payload_target_footprint_gap_m(state: dict) -> float | None:
    """Return the planar gap between an oriented box and the target cylinder."""

    payload = state.get("payload", {})
    target = state.get("target", {})
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
    values = (
        delta_x,
        delta_y,
        yaw,
        radius_m,
        half_length_m,
        half_width_m,
    )
    if not all(isfinite(value) for value in values):
        return None
    along_m = abs(delta_x * cos(yaw) + delta_y * sin(yaw))
    cross_m = abs(-delta_x * sin(yaw) + delta_y * cos(yaw))
    distance_to_box_m = hypot(
        max(0.0, along_m - half_length_m),
        max(0.0, cross_m - half_width_m),
    )
    return max(0.0, distance_to_box_m - max(0.0, radius_m))


def _payload_on_target(
    state: dict,
    *,
    target_xy_tolerance_m: float | None,
    target_z_tolerance_m: float | None,
    require_released: bool,
    allow_robot_contact: bool = False,
) -> tuple[bool, str]:
    payload = state.get("payload", {})
    target = state.get("target", {})
    contacts = payload.get("contacts", {}) if isinstance(payload, dict) else {}
    robots = state.get("robots", {})
    control = state.get("control_health", {})
    if not isinstance(payload, dict) or not isinstance(target, dict):
        return False, "payload or target state is missing"
    if not bool(control.get("healthy", False)):
        return False, "SONIC control is not healthy"
    if not bool(payload.get("active", False)) or not bool(target.get("available", False)):
        return False, "payload or target is inactive"
    xy_error = hypot(
        _safe_float(payload.get("x")) - _safe_float(target.get("center_x_m")),
        _safe_float(payload.get("y")) - _safe_float(target.get("center_y_m")),
    )
    z_error = abs(
        _safe_float(payload.get("z"))
        - _safe_float(target.get("payload_center_target_z_m"))
    )
    if (
        target_xy_tolerance_m is not None
        and xy_error > max(0.0, float(target_xy_tolerance_m))
    ):
        return False, f"payload-target xy error is {xy_error:.3f}m"
    if (
        target_z_tolerance_m is not None
        and z_error > max(0.0, float(target_z_tolerance_m))
    ):
        return False, f"payload-target z error is {z_error:.3f}m"
    if not bool(contacts.get("target", False)):
        return False, "target geometry is not physically contacting the payload"
    if bool(contacts.get("floor", False)) or bool(contacts.get("walls", ())):
        return False, "payload has floor or wall contact"
    if require_released and not bool(payload.get("released", False)):
        return False, "payload release event has not completed"
    if _safe_float(payload.get("linear_speed_mps")) > 0.03:
        return False, "payload is still translating"
    if _safe_float(payload.get("angular_speed_radps")) > 0.08:
        return False, "payload is still rotating"
    if require_released and not allow_robot_contact and (
        contacts.get("support_robots") or contacts.get("robots")
    ):
        return False, "robot contact remains after payload release"
    pelvis_heights = [
        _safe_float(robot.get("pelvis_height_m"))
        for robot in robots.values()
        if isinstance(robot, dict)
    ] if isinstance(robots, dict) else []
    if len(pelvis_heights) != 2 or min(pelvis_heights) < 0.35:
        return False, "one or both robots have fallen"
    return True, "supported by target"


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _required_float(mapping: dict, key: str, context: str) -> float:
    value = mapping.get(key)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context}.{key} is missing or non-numeric") from exc


def _required_mapping(mapping: dict, key: str, context: str) -> dict:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"{context}.{key} is missing or not an object")
    return value


def _unit_xy(vector: tuple[float, float]) -> tuple[float, float]:
    norm = hypot(float(vector[0]), float(vector[1]))
    if norm <= 1.0e-9:
        raise ValueError("target motion direction must be non-zero")
    return float(vector[0]) / norm, float(vector[1]) / norm


def _point_delta(before: dict, after: dict, context: str) -> tuple[float, float]:
    return (
        _required_float(after, "x", f"after.{context}")
        - _required_float(before, "x", f"before.{context}"),
        _required_float(after, "y", f"after.{context}")
        - _required_float(before, "y", f"before.{context}"),
    )


def _projected_delta(
    delta: tuple[float, float],
    direction: tuple[float, float],
) -> dict[str, object]:
    progress = delta[0] * direction[0] + delta[1] * direction[1]
    lateral = -delta[0] * direction[1] + delta[1] * direction[0]
    return {
        "delta_xy_m": [delta[0], delta[1]],
        "net_displacement_m": hypot(delta[0], delta[1]),
        "target_direction_progress_m": progress,
        "cross_track_displacement_m": lateral,
    }
