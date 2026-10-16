from dataclasses import replace
import json
import time

import pytest

from praxis_coop.cap_policy.backends import StaticSonicSkillBackend
from praxis_coop.cap_policy.contracts import ExecutionTrace
from praxis_coop.cap_policy.evaluator import _monitored_physical_failure
from praxis_coop.sonic.commands import SonicCommand, SonicLocomotionMode
from praxis_coop.sonic.planner_keepalive import DualSonicIdleKeepalive
from praxis_coop.sonic.runtime_state import (
    compute_runtime_motion_metrics,
    read_payload_runtime_status,
    read_runtime_motion_snapshot,
)
from praxis_coop.sonic.team_sync import (
    FormationSyncConfig,
    FormationSyncController,
    TeamMotionSnapshot,
)
from praxis_coop.sonic.zmq_publisher import (
    HEADER_SIZE,
    build_command_message,
    build_planner_message_for_command,
    send_synchronized_control_command,
    send_synchronized_planner_commands,
)
from scripts.run_sonic_static_skill import _PayloadSafetyMonitor


def _translation_commands(
    speed: float = 0.20,
    *,
    front_speed: float | None = None,
    back_speed: float | None = None,
) -> dict[str, SonicCommand]:
    return {
        robot_id: SonicCommand(
            t=1.5,
            robot_id=robot_id,
            skill_node="carry_forward",
            mode=SonicLocomotionMode.SLOW_WALK,
            movement=(1.0, 0.0, 0.0),
            facing=(1.0, 0.0, 0.0),
            speed=(
                front_speed
                if robot_id == "front" and front_speed is not None
                else back_speed
                if robot_id == "back" and back_speed is not None
                else speed
            ),
            duration_s=3.0,
            phase="carry_forward.forward.move",
            phase_id="carry_forward.forward.move",
            motion_axis="longitudinal",
            sync_group="static_sonic_skill",
        )
        for robot_id in ("front", "back")
    }


def _snapshot(
    *,
    front: tuple[float, float] = (0.0, 0.5),
    back: tuple[float, float] = (0.0, -0.5),
    payload: tuple[float, float] = (0.0, 0.0),
    support: tuple[float, float] = (0.0, 0.0),
    sim_time_s: float = 1.0,
) -> TeamMotionSnapshot:
    return TeamMotionSnapshot(
        available=True,
        fresh=True,
        age_s=0.01,
        sim_time_s=sim_time_s,
        robot_xy={"front": front, "back": back},
        payload_xy=payload,
        support_target_xy=support,
    )


def _controller(tmp_path, *, active_catchup: bool = True) -> FormationSyncController:
    return FormationSyncController(
        tmp_path / "runtime_state.json",
        FormationSyncConfig(
            active_catchup=active_catchup,
            error_filter_alpha=1.0,
        ),
    )


def test_dual_idle_keepalive_is_pauseable_and_keeps_heartbeat_fresh(tmp_path):
    publishers = []

    class FakePublisher:
        def __init__(self, config):
            self.config = config
            self.commands = []
            self.closed = False
            publishers.append(self)

        def send_planner_command(self, command):
            self.commands.append(command)

        def close(self):
            self.closed = True

    heartbeat = tmp_path / "planner_heartbeat.json"
    keepalive = DualSonicIdleKeepalive(
        heartbeat_file=heartbeat,
        publish_hz=100.0,
        startup_delay_s=0.0,
        publisher_factory=FakePublisher,
    )

    keepalive.start()
    time.sleep(0.03)
    assert keepalive.running
    keepalive.stop()

    assert not keepalive.running
    assert {publisher.config.port for publisher in publishers} == {5556, 5558}
    assert all(publisher.closed for publisher in publishers)
    assert all(publisher.commands for publisher in publishers)
    assert all(
        command.mode == SonicLocomotionMode.IDLE
        for publisher in publishers
        for command in publisher.commands
    )
    heartbeat_payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert heartbeat_payload["phase"] == "cap_agent_idle_keepalive"
    assert heartbeat_payload["count"] >= 1

    keepalive.start()
    keepalive.stop()
    assert len(publishers) == 4


def test_dual_idle_keepalive_leases_heartbeat_without_stopping_planner_output(tmp_path):
    publishers = []

    class FakePublisher:
        def __init__(self, config):
            self.config = config
            self.commands = []
            publishers.append(self)

        def send_planner_command(self, command):
            self.commands.append(command)

        def close(self):
            pass

    heartbeat = tmp_path / "planner_heartbeat.json"
    keepalive = DualSonicIdleKeepalive(
        heartbeat_file=heartbeat,
        publish_hz=200.0,
        startup_delay_s=0.0,
        publisher_factory=FakePublisher,
    )

    keepalive.start()
    with keepalive.suspend_heartbeat_writes():
        counts_before = [len(publisher.commands) for publisher in publishers]
        heartbeat.write_text(
            json.dumps({"phase": "sim_checkpoint_restore", "checkpoint_event_id": "event"}),
            encoding="utf-8",
        )
        time.sleep(0.04)
        assert keepalive.running
        assert json.loads(heartbeat.read_text(encoding="utf-8"))["checkpoint_event_id"] == "event"
        assert all(
            len(publisher.commands) > count
            for publisher, count in zip(publishers, counts_before, strict=True)
        )

    time.sleep(0.02)
    keepalive.stop()
    assert json.loads(heartbeat.read_text(encoding="utf-8"))["phase"] == (
        "cap_agent_idle_keepalive"
    )


def test_formation_sync_default_preserves_equal_commands(tmp_path):
    controller = _controller(tmp_path, active_catchup=False)
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(front=(0.010, 0.5), sim_time_s=1.1),
    )

    assert failure == ""
    assert adjusted == commands
    assert controller.report()["controller"] == "fixed_equal_command_monitor"
    assert controller.report()["last"]["longitudinal_gap_m"] == pytest.approx(0.01)


def test_formation_sync_records_payload_offset_without_replanning_equal_commands(
    tmp_path,
):
    controller = _controller(tmp_path, active_catchup=False)
    commands = _translation_commands()
    displaced = _snapshot(support=(0.010, 0.0))
    controller.adjust_snapshot(commands, displaced)

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(support=(0.010, 0.0), sim_time_s=1.1),
    )

    assert failure == ""
    assert adjusted == commands
    diagnostics = controller.report()["last"]
    assert diagnostics["payload_support_offset_xy_m"] == pytest.approx(
        (0.010, 0.0)
    )
    assert diagnostics["differential_correction_mps"] == pytest.approx(0.0)
    assert diagnostics["payload_speed_correction_mps"] == pytest.approx(0.0)


def test_formation_sync_slows_front_leader_and_boosts_back_lagger(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(front=(0.010, 0.5), sim_time_s=1.1),
    )

    assert failure == ""
    assert adjusted["front"].speed == pytest.approx(0.186)
    assert adjusted["back"].speed == pytest.approx(0.214)
    assert adjusted["front"].movement == commands["front"].movement
    assert adjusted["back"].movement == commands["back"].movement
    assert controller.report()["last"]["catchup_leader"] == "front"


def test_formation_sync_preserves_stagger_present_before_new_transaction(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands()
    staggered = _snapshot(
        front=(0.10, 0.5),
        back=(0.0, -0.5),
        sim_time_s=1.0,
    )
    controller.adjust_snapshot(commands, staggered)

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(
            front=(0.10, 0.5),
            back=(0.0, -0.5),
            sim_time_s=1.1,
        ),
    )

    assert failure == ""
    assert adjusted == commands
    diagnostics = controller.report()["last"]
    assert diagnostics["phase_progress_gap_m"] == pytest.approx(0.0)
    assert diagnostics["absolute_longitudinal_stagger_m"] == pytest.approx(0.10)
    assert diagnostics["catchup_leader"] == ""


def test_formation_sync_corrects_new_divergence_from_staggered_checkpoint(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands()
    controller.adjust_snapshot(
        commands,
        _snapshot(
            front=(-0.80, 0.5),
            back=(0.0, -0.5),
            sim_time_s=1.0,
        ),
    )

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(
            front=(-0.78, 0.5),
            back=(0.0, -0.5),
            sim_time_s=1.1,
        ),
    )

    assert failure == ""
    assert adjusted["front"].speed < commands["front"].speed
    assert adjusted["back"].speed > commands["back"].speed
    diagnostics = controller.report()["last"]
    assert diagnostics["phase_progress_gap_m"] == pytest.approx(0.02)
    assert diagnostics["absolute_longitudinal_stagger_m"] == pytest.approx(-0.78)
    assert diagnostics["catchup_leader"] == "front"


def test_formation_sync_clears_leader_before_reversing_catchup(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())
    controller.adjust_snapshot(
        commands,
        _snapshot(front=(0.020, 0.5), sim_time_s=1.1),
    )

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(back=(0.020, -0.5), sim_time_s=1.2),
    )

    assert failure == ""
    assert adjusted["front"].speed == pytest.approx(commands["front"].speed)
    assert adjusted["back"].speed == pytest.approx(commands["back"].speed)
    assert controller.report()["last"]["catchup_leader"] == ""

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(back=(0.020, -0.5), sim_time_s=1.3),
    )

    assert failure == ""
    assert adjusted["front"].speed > commands["front"].speed
    assert adjusted["back"].speed < commands["back"].speed
    assert controller.report()["last"]["catchup_leader"] == "back"


def test_formation_sync_slows_back_leader_and_boosts_front_lagger(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(back=(0.010, -0.5), sim_time_s=1.1),
    )

    assert failure == ""
    assert adjusted["front"].speed == pytest.approx(0.214)
    assert adjusted["back"].speed == pytest.approx(0.186)
    assert controller.report()["last"]["catchup_leader"] == "back"


def test_formation_sync_bounds_large_backend_response_mismatch(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands()
    front_x = 0.0
    back_x = 0.0
    peak_gap = 0.0
    dt_s = 0.05

    for index in range(200):
        adjusted, failure = controller.adjust_snapshot(
            commands,
            _snapshot(
                front=(front_x, 0.5),
                back=(back_x, -0.5),
                sim_time_s=1.0 + index * dt_s,
            ),
        )
        assert failure == ""
        # Model the observed deployment asymmetry: back produces only half the
        # root progress per commanded metre/second.
        front_x += adjusted["front"].speed * dt_s
        back_x += 0.5 * adjusted["back"].speed * dt_s
        peak_gap = max(peak_gap, abs(front_x - back_x))

    assert peak_gap < 0.08
    assert abs(front_x - back_x) < 0.06
    diagnostics = controller.report()["last"]
    assert diagnostics["leader_slowdown_mps"] > diagnostics["lagger_speed_boost_mps"]
    assert controller.report()["catchup_activation_count"] >= 1


def test_formation_sync_preserves_asymmetric_backend_calibration(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands(front_speed=0.16, back_speed=0.24)
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(front=(0.010, 0.5), sim_time_s=1.1),
    )

    assert failure == ""
    assert adjusted["front"].speed == pytest.approx(0.146)
    assert adjusted["back"].speed == pytest.approx(0.254)
    diagnostics = controller.report()["last"]
    assert diagnostics["calibrated_base_speed_mps"] == {
        "front": pytest.approx(0.16),
        "back": pytest.approx(0.24),
    }
    assert diagnostics["catchup_lagger"] == "back"


def test_formation_sync_asymmetric_calibration_handles_back_leader(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands(front_speed=0.16, back_speed=0.24)
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(back=(0.010, -0.5), sim_time_s=1.1),
    )

    assert failure == ""
    assert adjusted["front"].speed == pytest.approx(0.174)
    assert adjusted["back"].speed == pytest.approx(0.226)
    assert controller.report()["last"]["catchup_lagger"] == "front"


def test_formation_sync_keeps_calibrated_commands_when_state_is_stale(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands(front_speed=0.16, back_speed=0.24)

    adjusted, failure = controller.adjust_snapshot(
        commands,
        TeamMotionSnapshot(available=True, fresh=False, age_s=2.0),
    )

    assert failure == ""
    assert adjusted == commands


def test_formation_sync_applies_common_payload_recentering_correction(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(support=(0.010, 0.0), sim_time_s=1.1),
    )

    assert failure == ""
    assert adjusted["front"].speed == pytest.approx(0.1904)
    assert adjusted["back"].speed == pytest.approx(0.1904)
    assert adjusted["front"].movement == pytest.approx((1.0, 0.0, 0.0))


def test_formation_sync_prioritizes_catchup_over_common_payload_acceleration(
    tmp_path,
):
    controller = _controller(tmp_path)
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(
            front=(0.010, 0.5),
            support=(-0.050, 0.0),
            sim_time_s=1.1,
        ),
    )

    assert failure == ""
    assert adjusted["front"].speed < commands["front"].speed
    assert adjusted["back"].speed > commands["back"].speed
    diagnostics = controller.report()["last"]
    assert diagnostics["payload_speed_correction_mps"] == pytest.approx(0.0)
    assert diagnostics["payload_correction_suppressed_for_catchup"]


def test_loaded_formation_sync_never_injects_divergent_cross_track_motion(tmp_path):
    controller = _controller(tmp_path)
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(
            front=(0.0, 0.52),
            back=(0.0, -0.5),
            sim_time_s=1.1,
        ),
    )

    assert failure == ""
    assert adjusted["front"].movement == commands["front"].movement
    assert adjusted["back"].movement == commands["back"].movement
    assert adjusted["front"].movement == adjusted["back"].movement
    diagnostics = controller.report()["last"]
    assert diagnostics["cross_track_error_m"] == pytest.approx(0.02)
    assert diagnostics["cross_track_correction_mps"] == pytest.approx(0.0)


def test_formation_sync_hard_stops_before_large_formation_split(tmp_path):
    controller = FormationSyncController(
        tmp_path / "runtime_state.json",
        FormationSyncConfig(
            active_catchup=True,
            error_filter_alpha=1.0,
            hard_desync_m=0.04,
        ),
    )
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(front=(0.041, 0.5), sim_time_s=1.1),
    )

    assert adjusted == commands
    assert failure.startswith("formation_sync_hard_limit")


def test_formation_sync_does_not_abort_on_one_raw_gait_phase_gap(tmp_path):
    controller = FormationSyncController(
        tmp_path / "runtime_state.json",
        FormationSyncConfig(
            active_catchup=False,
            error_filter_alpha=0.10,
            hard_desync_m=0.04,
        ),
    )
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())

    adjusted, failure = controller.adjust_snapshot(
        commands,
        _snapshot(front=(0.045, 0.5), sim_time_s=1.1),
    )

    assert adjusted == commands
    assert failure == ""
    assert controller.report()["last"]["longitudinal_gap_m"] == pytest.approx(0.045)


def test_formation_sync_hard_stops_after_filtered_formation_diverges(tmp_path):
    controller = FormationSyncController(
        tmp_path / "runtime_state.json",
        FormationSyncConfig(
            active_catchup=False,
            error_filter_alpha=0.50,
            hard_desync_m=0.04,
        ),
    )
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())

    failure = ""
    for index in range(1, 6):
        _adjusted, failure = controller.adjust_snapshot(
            commands,
            _snapshot(front=(0.060, 0.5), sim_time_s=1.0 + 0.1 * index),
        )
        if failure:
            break

    assert failure.startswith("formation_sync_hard_limit")
    assert "progress_gap=0.0600m" in failure


def test_formation_sync_observes_residual_motion_during_idle_settle(tmp_path):
    controller = FormationSyncController(
        tmp_path / "runtime_state.json",
        FormationSyncConfig(
            active_catchup=False,
            error_filter_alpha=1.0,
            hard_desync_m=0.04,
        ),
    )
    commands = _translation_commands()
    controller.adjust_snapshot(commands, _snapshot())
    controller.adjust_snapshot(
        commands,
        _snapshot(
            front=(0.020, 0.5),
            back=(0.020, -0.5),
            sim_time_s=1.1,
        ),
    )
    idle = {
        robot_id: replace(
            command,
            mode=SonicLocomotionMode.IDLE,
            movement=(0.0, 0.0, 0.0),
            speed=-1.0,
            phase="carry_forward.forward.brake",
            phase_id="carry_forward.forward.brake",
        )
        for robot_id, command in commands.items()
    }

    unchanged, failure = controller.adjust_snapshot(
        idle,
        _snapshot(
            front=(0.030, 0.5),
            back=(0.020, -0.5),
            sim_time_s=1.2,
        ),
    )

    assert unchanged == idle
    assert failure == ""
    report = controller.report()
    assert report["settle_sample_count"] == 1
    assert report["last"]["settle_observation"]
    assert report["last"]["residual_motion_m"]["front"] == pytest.approx(0.01)
    assert report["last"]["residual_motion_m"]["back"] == pytest.approx(0.0)

    _unchanged, failure = controller.adjust_snapshot(
        idle,
        _snapshot(
            front=(0.080, 0.5),
            back=(0.020, -0.5),
            sim_time_s=1.3,
        ),
    )

    assert failure.startswith("formation_sync_settle_hard_limit")


def test_active_formation_sync_corrects_lagger_during_idle_settle(tmp_path):
    controller = FormationSyncController(
        tmp_path / "runtime_state.json",
        FormationSyncConfig(
            active_catchup=True,
            error_filter_alpha=1.0,
            hard_desync_m=0.50,
        ),
    )
    commands = _translation_commands(speed=0.50)
    controller.adjust_snapshot(commands, _snapshot())
    controller.adjust_snapshot(
        commands,
        _snapshot(
            front=(0.020, 0.5),
            back=(0.020, -0.5),
            sim_time_s=1.1,
        ),
    )
    idle = {
        robot_id: replace(
            command,
            mode=SonicLocomotionMode.IDLE,
            movement=(0.0, 0.0, 0.0),
            speed=-1.0,
            phase="carry_forward.forward.brake",
            phase_id="carry_forward.forward.brake",
        )
        for robot_id, command in commands.items()
    }

    adjusted, failure = controller.adjust_snapshot(
        idle,
        _snapshot(
            front=(0.220, 0.5),
            back=(0.020, -0.5),
            support=(-0.49, 0.0),
            sim_time_s=1.2,
        ),
    )

    assert failure == ""
    assert adjusted == idle

    for sim_time_s in (1.3, 1.4):
        adjusted, failure = controller.adjust_snapshot(
            idle,
            _snapshot(
                front=(0.220, 0.5),
                back=(0.020, -0.5),
                support=(-0.49, 0.0),
                sim_time_s=sim_time_s,
            ),
        )
        assert failure == ""
        assert adjusted == idle

    adjusted, failure = controller.adjust_snapshot(
        idle,
        _snapshot(
            front=(0.220, 0.5),
            back=(0.020, -0.5),
            support=(-0.49, 0.0),
            sim_time_s=1.5,
        ),
    )

    assert failure == ""
    assert adjusted["front"].mode == SonicLocomotionMode.IDLE
    assert adjusted["back"].mode == SonicLocomotionMode.SLOW_WALK
    assert adjusted["back"].movement == commands["back"].movement
    assert adjusted["back"].speed == pytest.approx(0.03)
    report = controller.report()
    assert report["last"]["settle_catchup_leader"] == "front"
    assert report["last"]["settle_catchup_lagger"] == "back"
    assert not report["last"]["commands_remain_idle"]

    adjusted, failure = controller.adjust_snapshot(
        idle,
        _snapshot(
            front=(0.220, 0.5),
            back=(0.219, -0.5),
            support=(-0.49, 0.0),
            sim_time_s=1.6,
        ),
    )

    assert failure == ""
    assert adjusted == idle
    assert controller.report()["last"]["commands_remain_idle"]


def test_settle_catchup_regression_is_quality_warning_not_abort(tmp_path):
    controller = FormationSyncController(
        tmp_path / "runtime_state.json",
        FormationSyncConfig(
            active_catchup=True,
            error_filter_alpha=1.0,
            hard_desync_m=0.50,
        ),
    )
    commands = _translation_commands(speed=0.50)
    controller.adjust_snapshot(commands, _snapshot())
    controller.adjust_snapshot(
        commands,
        _snapshot(front=(0.020, 0.5), back=(0.020, -0.5), sim_time_s=1.1),
    )
    idle = {
        robot_id: replace(
            command,
            mode=SonicLocomotionMode.IDLE,
            movement=(0.0, 0.0, 0.0),
            speed=-1.0,
            phase="carry_forward.forward.brake",
            phase_id="carry_forward.forward.brake",
        )
        for robot_id, command in commands.items()
    }

    for sim_time_s in (1.2, 1.3, 1.4, 1.5):
        _adjusted, failure = controller.adjust_snapshot(
            idle,
            _snapshot(
                front=(0.220, 0.5),
                back=(0.020, -0.5),
                sim_time_s=sim_time_s,
            ),
        )
        assert failure == ""

    _adjusted, failure = controller.adjust_snapshot(
        idle,
        _snapshot(
            front=(0.220, 0.5),
            back=(0.1694, -0.5),
            sim_time_s=1.6,
        ),
    )
    assert failure == ""

    _adjusted, failure = controller.adjust_snapshot(
        idle,
        _snapshot(
            front=(0.220, 0.5),
            back=(0.1540, -0.5),
            sim_time_s=1.7,
        ),
    )

    assert failure == ""
    report = controller.report()["last"]
    assert report["settle_catchup_regression_warning"]
    assert report["quality_warning"].startswith(
        "formation_sync_settle_catchup_regression"
    )
    assert "failure_reason" not in report


def test_linear_endpoint_lock_keeps_every_settle_command_idle(tmp_path):
    controller = FormationSyncController(
        tmp_path / "runtime_state.json",
        FormationSyncConfig(active_catchup=True, error_filter_alpha=1.0),
    )
    commands = _translation_commands(speed=0.50)
    controller.adjust_snapshot(commands, _snapshot())
    controller.adjust_snapshot(
        commands,
        _snapshot(front=(0.02, 0.5), back=(0.02, -0.5), sim_time_s=1.1),
    )
    idle = {
        robot_id: replace(
            command,
            mode=SonicLocomotionMode.IDLE,
            movement=(0.0, 0.0, 0.0),
            speed=-1.0,
            phase="carry_forward.forward.brake",
            phase_id="carry_forward.forward.brake",
        )
        for robot_id, command in commands.items()
    }

    controller.lock_translation_endpoint()
    for sim_time_s in (1.2, 1.3, 1.4, 1.5, 1.6):
        adjusted, failure = controller.adjust_snapshot(
            idle,
            _snapshot(
                front=(0.22, 0.5),
                back=(0.02, -0.5),
                sim_time_s=sim_time_s,
            ),
        )
        assert failure == ""
        assert adjusted == idle
    assert controller.report()["endpoint_idle_locked"] is True
    assert controller.report()["last"]["commands_remain_idle"] is True


def test_active_settle_catchup_never_bypasses_payload_offset_hard_limit(tmp_path):
    controller = FormationSyncController(
        tmp_path / "runtime_state.json",
        FormationSyncConfig(
            active_catchup=True,
            error_filter_alpha=1.0,
            hard_desync_m=0.50,
        ),
    )
    commands = _translation_commands(speed=0.50)
    controller.adjust_snapshot(commands, _snapshot())
    controller.adjust_snapshot(
        commands,
        _snapshot(front=(0.020, 0.5), back=(0.020, -0.5), sim_time_s=1.1),
    )
    idle = {
        robot_id: replace(
            command,
            mode=SonicLocomotionMode.IDLE,
            movement=(0.0, 0.0, 0.0),
            speed=-1.0,
            phase="carry_forward.forward.brake",
            phase_id="carry_forward.forward.brake",
        )
        for robot_id, command in commands.items()
    }

    _adjusted, failure = controller.adjust_snapshot(
        idle,
        _snapshot(
            front=(0.220, 0.5),
            back=(0.020, -0.5),
            support=(-0.51, 0.0),
            sim_time_s=1.2,
        ),
    )

    assert failure.startswith("formation_sync_settle_hard_limit")


class _FakeSocket:
    def __init__(self) -> None:
        self.messages: list[bytes] = []

    def send(self, message: bytes) -> None:
        self.messages.append(message)


class _FakePublisher:
    def __init__(self) -> None:
        self._socket = _FakeSocket()
        self.logged: list[SonicCommand] = []

    def _log_planner_command(self, command: SonicCommand) -> None:
        self.logged.append(command)


def test_synchronized_publish_prebuilds_and_sends_one_message_per_robot():
    commands = _translation_commands()
    publishers = {robot_id: _FakePublisher() for robot_id in commands}

    skew_s = send_synchronized_planner_commands(publishers, commands)

    assert skew_s >= 0.0
    for robot_id, publisher in publishers.items():
        assert publisher._socket.messages == [build_planner_message_for_command(commands[robot_id])]
        assert publisher.logged == [commands[robot_id]]


def test_synchronized_control_command_sends_same_prebuilt_message():
    publishers = {robot_id: _FakePublisher() for robot_id in ("front", "back")}

    skew_s = send_synchronized_control_command(
        publishers,
        start=False,
        stop=False,
        planner=False,
    )

    expected = build_command_message(start=False, stop=False, planner=False)
    assert skew_s >= 0.0
    assert publishers["front"]._socket.messages == [expected]
    assert publishers["back"]._socket.messages == [expected]


def test_synchronized_control_command_can_request_planner_reset():
    publishers = {robot_id: _FakePublisher() for robot_id in ("front", "back")}

    send_synchronized_control_command(
        publishers,
        start=True,
        stop=False,
        planner=True,
        reset_planner=True,
    )

    expected = build_command_message(
        start=True,
        stop=False,
        planner=True,
        reset_planner=True,
    )
    assert b"reset_planner" in expected[:HEADER_SIZE]
    assert publishers["front"]._socket.messages == [expected]
    assert publishers["back"]._socket.messages == [expected]


def test_synchronized_control_command_can_clear_heading_offset():
    publishers = {robot_id: _FakePublisher() for robot_id in ("front", "back")}

    send_synchronized_control_command(
        publishers,
        start=False,
        stop=False,
        planner=True,
        delta_heading=0.0,
    )

    expected = build_command_message(
        start=False,
        stop=False,
        planner=True,
        delta_heading=0.0,
    )
    assert publishers["front"]._socket.messages == [expected]
    assert publishers["back"]._socket.messages == [expected]


def test_checkpoint_idle_resync_reinitializes_both_planners(tmp_path):
    backend = StaticSonicSkillBackend(execute=False)

    result = backend.resynchronize_idle(
        output_dir=tmp_path / "idle",
        heartbeat_file=tmp_path / "planner_heartbeat.json",
        runtime_state_file=tmp_path / "runtime_state.json",
        duration_s=4.0,
    )

    command = result["command"]
    assert "--planner-soft-reset" in command
    assert command[command.index("--planner-soft-reset-streamed-seconds") + 1] == "0.5"
    assert command[command.index("--planner-soft-reset-idle-seconds") + 1] == "2.0"
    assert command[command.index("--stand-phase-id") + 1] == "dual_motor_preflight_stand"


def test_fresh_bootstrap_can_emit_the_guarded_drop_phase(tmp_path):
    backend = StaticSonicSkillBackend(execute=False)

    result = backend.resynchronize_idle(
        output_dir=tmp_path / "bootstrap",
        heartbeat_file=tmp_path / "planner_heartbeat.json",
        runtime_state_file=tmp_path / "runtime_state.json",
        stand_phase_id="static_skill_pre_stand",
    )

    command = result["command"]
    assert command[command.index("--stand-phase-id") + 1] == "static_skill_pre_stand"


def _supported_runtime_state(*, include_force: bool = True) -> dict:
    contacts = {
        "support_robots": ["front", "back"],
        "support_geometries": [
            "front_left_carry_support_plate",
            "back_left_carry_support_plate",
        ],
        "floor": False,
        "walls": [],
    }
    if include_force:
        contacts["support_forces"] = {
            "available": True,
            "normal_force_n_by_robot": {"front": 8.0, "back": 6.0},
            "total_normal_force_n": 14.0,
            "balance_ratio": 0.75,
        }
    return {
        "runtime_status": "running",
        "control_health": {"healthy": True},
        "checkpoint": {"resync_active": False},
        "elastic_band_dropped": True,
        "robots": {
            "front": {"x": 0.0, "y": 0.0, "pelvis_height_m": 0.80},
            "back": {"x": 0.0, "y": 0.84, "pelvis_height_m": 0.80},
        },
        "payload": {
            "present": True,
            "active": True,
            "x": 0.0,
            "y": 0.0,
            "z": 1.0,
            "tilt_deg": 1.0,
            "top_surface_corner_z_range_m": 0.01,
            "linear_speed_mps": 0.0,
            "angular_speed_radps": 0.0,
            "live_support_target": {"ready": True, "x": 0.0, "y": 0.0},
            "contacts": contacts,
        },
    }


def _payload_safety_monitor(runtime_state, *, contact_loss_grace_s: float = 0.4):
    return _PayloadSafetyMonitor(
        runtime_state,
        max_age_s=1.0,
        min_payload_z_m=0.65,
        min_robot_pelvis_height_m=0.35,
        min_robot_xy_separation_m=0.61,
        max_tilt_deg=15.0,
        hard_tilt_deg=25.0,
        tilt_grace_s=0.4,
        contact_loss_grace_s=contact_loss_grace_s,
        max_support_center_xy_error_m=0.25,
        min_support_force_per_robot_n=0.0,
        min_support_force_balance_ratio=0.0,
        force_imbalance_grace_s=0.4,
    )


def test_payload_monitor_prioritizes_sustained_support_loss_over_spacing(
    tmp_path,
    monkeypatch,
):
    runtime_state = tmp_path / "runtime_state.json"
    state = _supported_runtime_state()
    state["robots"]["back"]["y"] = 0.50
    state["payload"]["contacts"]["support_robots"] = ["front"]
    runtime_state.write_text(json.dumps(state), encoding="utf-8")
    clock = [10.0]
    monkeypatch.setattr(
        "scripts.run_sonic_static_skill.time.monotonic",
        lambda: clock[0],
    )
    monitor = _payload_safety_monitor(runtime_state)

    assert monitor.check() == ""
    clock[0] = 10.5
    failure = monitor.check()

    assert failure.startswith("dual_payload_support_lost")
    assert "robot_xy_separation=0.500m" in failure


def test_payload_monitor_ignores_one_transient_contact_sample(
    tmp_path,
    monkeypatch,
):
    runtime_state = tmp_path / "runtime_state.json"
    state = _supported_runtime_state()
    state["payload"]["contacts"]["support_robots"] = ["front"]
    runtime_state.write_text(json.dumps(state), encoding="utf-8")
    clock = [20.0]
    monkeypatch.setattr(
        "scripts.run_sonic_static_skill.time.monotonic",
        lambda: clock[0],
    )
    monitor = _payload_safety_monitor(runtime_state)

    assert monitor.check() == ""
    state["payload"]["contacts"]["support_robots"] = ["front", "back"]
    runtime_state.write_text(json.dumps(state), encoding="utf-8")
    clock[0] = 20.2

    assert monitor.check() == ""
    assert monitor.contact_loss_started is None


def test_evaluator_prefers_payload_support_failure_over_spacing_marker():
    trace = ExecutionTrace(
        script_id="attempt",
        attempt=1,
        status="script_error",
        error=(
            "physical safety abort: robot_pair_too_close; "
            "dual_payload_support_lost"
        ),
    )

    assert _monitored_physical_failure(trace) == "payload_not_supported"


def test_payload_status_requires_balanced_force_when_telemetry_exists(tmp_path):
    runtime_state = tmp_path / "runtime_state.json"
    state = _supported_runtime_state()
    runtime_state.write_text(json.dumps(state), encoding="utf-8")
    status = read_payload_runtime_status(runtime_state)

    assert status.payload_supported()
    assert status.support_force_balanced()
    assert status.support_normal_force_n_by_robot == {"front": 8.0, "back": 6.0}

    state["payload"]["contacts"]["support_forces"].update(
        {
            "normal_force_n_by_robot": {"front": 13.9, "back": 0.1},
            "balance_ratio": 0.1 / 13.9,
        }
    )
    runtime_state.write_text(json.dumps(state), encoding="utf-8")
    imbalanced = read_payload_runtime_status(runtime_state)
    assert not imbalanced.support_force_balanced()
    assert imbalanced.payload_supported()
    assert not imbalanced.payload_supported(
        min_support_force_per_robot_n=0.5,
        min_support_force_balance_ratio=0.10,
    )


def test_payload_initialization_can_require_all_four_support_pads(tmp_path):
    runtime_state = tmp_path / "runtime_state.json"
    state = _supported_runtime_state()
    runtime_state.write_text(json.dumps(state), encoding="utf-8")

    partial = read_payload_runtime_status(runtime_state)
    assert partial.payload_supported()
    assert not partial.payload_supported(require_all_support_pads=True)

    state["payload"]["contacts"]["support_geometries"] = [
        "front_left_carry_support_plate",
        "front_right_carry_support_plate",
        "back_left_carry_support_plate",
        "back_right_carry_support_plate",
    ]
    runtime_state.write_text(json.dumps(state), encoding="utf-8")
    complete = read_payload_runtime_status(runtime_state)
    assert complete.all_support_pads_contacting()
    assert complete.payload_supported(require_all_support_pads=True)


def test_payload_status_keeps_old_runtime_artifacts_compatible(tmp_path):
    runtime_state = tmp_path / "runtime_state.json"
    runtime_state.write_text(
        json.dumps(_supported_runtime_state(include_force=False)),
        encoding="utf-8",
    )

    status = read_payload_runtime_status(runtime_state)

    assert not status.support_force_available
    assert status.payload_supported()


def test_transport_tracking_survives_temporary_placement_readiness_loss(tmp_path):
    runtime_state = tmp_path / "runtime_state.json"
    state = _supported_runtime_state()
    state["payload"]["x"] = 0.01
    state["payload"]["live_support_target"].update(
        {
            "ready": False,
            "tracking_available": True,
            "wait_reasons": ["back is rotating (0.465rad/s)"],
        }
    )
    runtime_state.write_text(json.dumps(state), encoding="utf-8")

    status = read_payload_runtime_status(runtime_state)

    assert not status.support_target_ready
    assert status.support_tracking_available
    assert status.support_center_xy_error_available
    assert status.support_center_xy_error_m == pytest.approx(0.01)


def test_cap_backend_records_live_team_sync_and_force_limits(tmp_path):
    backend = StaticSonicSkillBackend(execute=False)
    command = backend._command(
        "carry_forward",
        {"distance_m": 0.05, "speed_mps": 0.20},
        tmp_path / "skill",
        tmp_path / "planner_heartbeat.json",
        tmp_path / "runtime_state.json",
    )

    assert command[command.index("--team-sync-pause-leader-error-m") + 1] == "0.008"
    assert command[command.index("--payload-safety-profile") + 1] == "task"
    assert "--team-sync-hard-desync-m" not in command
    assert backend.carry_safety.team_hard_desync_m == pytest.approx(0.50)
    assert command[command.index("--team-sync-min-active-speed-ratio") + 1] == "0.0"
    assert "--payload-support-min-force-n" not in command
    assert "--payload-support-min-balance-ratio" not in command
    assert "--no-team-sync" not in command


def test_evaluator_maps_live_formation_abort_to_formation_desync():
    trace = ExecutionTrace(
        script_id="attempt",
        attempt=1,
        status="script_error",
        error="physical safety abort: formation_sync_hard_limit: error=0.041m",
    )

    assert _monitored_physical_failure(trace) == "formation_desync"


def test_evaluator_maps_inverted_payload_abort_to_payload_slip():
    trace = ExecutionTrace(
        script_id="attempt",
        attempt=1,
        status="script_error",
        error="physical safety abort during carry: payload_slip: tilt=102deg",
    )

    assert _monitored_physical_failure(trace) == "payload_slip"


def _motion_state(*, payload_x: float, front_x: float, back_x: float) -> dict:
    support_x = (front_x + back_x) / 2.0
    return {
        "runtime_status": "running",
        "time_unix": 10.0,
        "sim_time_s": 2.0,
        "mean_heading_yaw_rad": 0.0,
        "robots": {
            "front": {
                "x": front_x,
                "y": 0.5,
                "yaw_rad": 0.0,
                "pelvis_height_m": 0.8,
            },
            "back": {
                "x": back_x,
                "y": -0.5,
                "yaw_rad": 0.0,
                "pelvis_height_m": 0.8,
            },
        },
        "payload": {
            "x": payload_x,
            "y": 0.0,
            "z": 1.0,
            "yaw_rad": 0.0,
            "tilt_deg": 1.0,
            "live_support_target": {"x": support_x, "y": 0.0},
            "contacts": {
                "support_robots": ["front", "back"],
                "support_geometries": sorted(
                    {
                        "front_left_carry_support_plate",
                        "front_right_carry_support_plate",
                        "back_left_carry_support_plate",
                        "back_right_carry_support_plate",
                    }
                ),
                "floor": False,
                "walls": [],
            },
        },
    }


def test_runtime_motion_metrics_separate_team_progress_and_payload_slip(tmp_path):
    runtime_state = tmp_path / "runtime_state.json"
    before_state = _motion_state(payload_x=0.0, front_x=0.0, back_x=0.0)
    runtime_state.write_text(json.dumps(before_state), encoding="utf-8")
    before = read_runtime_motion_snapshot(runtime_state)

    after_state = _motion_state(payload_x=0.04, front_x=0.05, back_x=0.03)
    after_state["sim_time_s"] = 5.0
    runtime_state.write_text(json.dumps(after_state), encoding="utf-8")
    after = read_runtime_motion_snapshot(runtime_state)
    metrics = compute_runtime_motion_metrics(
        before,
        after,
        target_direction_xy=(1.0, 0.0),
        reference_frame="body",
    )

    assert metrics["payload"]["target_direction_progress_m"] == pytest.approx(0.04)
    assert metrics["team_mean"]["target_direction_progress_m"] == pytest.approx(0.04)
    assert metrics["robot_target_progress_gap_m"] == pytest.approx(0.02)
    assert metrics["formation_change_m"] == pytest.approx(0.02)
    assert metrics["payload_relative_to_team"]["net_displacement_m"] == pytest.approx(0.0)
    assert metrics["payload_relative_to_support_target"]["net_displacement_m"] == pytest.approx(0.0)
