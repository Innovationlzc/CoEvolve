from math import pi
from pathlib import Path

from praxis_coop.sonic.commands import SonicLocomotionMode
from praxis_coop.sonic.skill_presets import (
    SonicStaticSkillBuildConfig,
    build_static_sonic_skill_stream,
)
from praxis_coop.sonic.stationary_hold import (
    PLACE_CROUCH_STATIONARY_LOCK_S,
    PLACE_CROUCH_STATIONARY_SETTLE_S,
    PLACE_CROUCH_STATIONARY_SPEED,
    StationaryPlanarHold,
)
from praxis_coop.sonic.team_sync import TeamMotionSnapshot


def _idle_crouch_command(robot_id: str):
    stream = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name="carry_place_crouch",
            place_start_height=0.80,
            place_height=0.55,
            place_ramp_seconds=2.0,
            place_hold_seconds=3.0,
        )
    )
    return next(command for command in stream.commands if command.robot_id == robot_id)


def _snapshot(
    *,
    front: tuple[float, float],
    back: tuple[float, float],
    front_yaw: float = 0.0,
    back_yaw: float = 0.0,
) -> TeamMotionSnapshot:
    return TeamMotionSnapshot(
        available=True,
        fresh=True,
        age_s=0.01,
        sim_time_s=1.0,
        robot_xy={"front": front, "back": back},
        robot_yaw_rad={"front": front_yaw, "back": back_yaw},
    )


def test_place_crouch_stream_locks_then_settles_with_zero_planar_command():
    stream = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name="carry_place_crouch",
            place_start_height=0.80,
            place_height=0.55,
            place_ramp_seconds=2.0,
            place_hold_seconds=3.0,
        )
    )
    front = [command for command in stream.commands if command.robot_id == "front"]
    assert [command.phase_id.rsplit(".", 1)[-1] for command in front] == [
        "lock",
        "lower",
        "hold",
        "settle",
    ]
    assert [command.duration_s for command in front] == [
        PLACE_CROUCH_STATIONARY_LOCK_S,
        2.0,
        3.0,
        PLACE_CROUCH_STATIONARY_SETTLE_S,
    ]
    assert [command.height for command in front] == [0.80, 0.55, 0.55, 0.55]
    assert stream.duration_s == (
        PLACE_CROUCH_STATIONARY_LOCK_S + 2.0 + 3.0 + PLACE_CROUCH_STATIONARY_SETTLE_S
    )
    for command in stream.commands:
        assert command.mode is SonicLocomotionMode.IDLE
        assert command.movement == (0.0, 0.0, 0.0)
        assert command.speed == PLACE_CROUCH_STATIONARY_SPEED
        assert command.motion_axis == "height"


def test_place_crouch_rise_skips_settle_after_returning_to_stand():
    stream = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name="carry_place_crouch",
            place_ramp_seconds=2.0,
            place_hold_seconds=1.0,
            place_rise=True,
        )
    )
    front = [command for command in stream.commands if command.robot_id == "front"]
    assert [command.phase_id.rsplit(".", 1)[-1] for command in front] == [
        "lock",
        "lower",
        "hold",
        "rise",
    ]
    assert front[-1].height == 0.80


def test_place_crouch_planar_hold_stays_idle_inside_deadband():
    hold = StationaryPlanarHold()
    assert hold.capture(_snapshot(front=(0.0, 0.0), back=(0.0, 1.0)))
    commands = {
        "front": _idle_crouch_command("front"),
        "back": _idle_crouch_command("back"),
    }
    adjusted = hold.adjust(
        commands,
        _snapshot(front=(0.005, 0.0), back=(0.0, 1.004)),
    )
    for command in adjusted.values():
        assert command.mode is SonicLocomotionMode.IDLE
        assert command.movement == (0.0, 0.0, 0.0)
        assert command.speed == PLACE_CROUCH_STATIONARY_SPEED


def test_place_crouch_planar_hold_uses_deploy_frame_slow_walk():
    hold = StationaryPlanarHold()
    assert hold.capture(_snapshot(front=(0.0, 0.0), back=(0.0, 1.0)))
    commands = {
        "front": _idle_crouch_command("front"),
        "back": _idle_crouch_command("back"),
    }
    # Front drifted body-backward by 9 cm while facing +X.
    adjusted = hold.adjust(
        commands,
        _snapshot(front=(-0.09, 0.0), back=(0.0, 1.0)),
    )
    front = adjusted["front"]
    assert front.mode is SonicLocomotionMode.SLOW_WALK
    assert front.movement[0] > 0.99
    assert abs(front.movement[1]) < 1.0e-6
    assert 0.0 < front.speed <= 0.05
    assert adjusted["back"].mode is SonicLocomotionMode.IDLE


def test_place_crouch_planar_hold_maps_world_error_through_body_yaw():
    hold = StationaryPlanarHold()
    assert hold.capture(
        _snapshot(front=(0.0, 0.0), back=(0.0, 0.0), front_yaw=pi / 2.0)
    )
    commands = {"front": _idle_crouch_command("front")}
    # Facing +Y; world +X error is body right, so deploy left is negative.
    adjusted = hold.adjust(
        commands,
        _snapshot(front=(-0.09, 0.0), back=(0.0, 0.0), front_yaw=pi / 2.0),
    )
    front = adjusted["front"]
    assert front.mode is SonicLocomotionMode.SLOW_WALK
    assert abs(front.movement[0]) < 1.0e-6
    assert front.movement[1] < -0.99


def test_place_crouch_motor_lock_does_not_teleport_root_xy():
    source = (
        Path(str((Path(__file__).resolve().parents[1] / 'src/praxis_coop/sonic_mujoco/dual_runtime.py').resolve()))
        .read_text(encoding="utf-8")
    )
    assert "max_hip_pitch_delta_rad: float = -0.26" in source
    assert "never writes root XY" in source or "does not teleport root XY" in source
