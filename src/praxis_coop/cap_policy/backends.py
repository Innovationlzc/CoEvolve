"""Execution backends for Code-as-Policy Skill SDK calls."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Protocol

from .contracts import SkillCallRecord
from praxis_coop.sonic.safety import CarrySafetyProfile, get_carry_safety_profile


REPO_ROOT = Path(__file__).resolve().parents[3]
SONIC_PYTHON = REPO_ROOT / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
VALIDATE_STATIC_SKILL = REPO_ROOT / "scripts" / "validate_static_sonic_skill.py"
COORDINATED_STATIC_SKILL = REPO_ROOT / "scripts" / "run_sonic_coordinated_skill.py"
DUAL_PREFLIGHT = REPO_ROOT / "scripts" / "run_dual_official_sonic_motor_preflight.py"
TRANSLATION_SKILLS = frozenset(
    {"carry_forward", "carry_backward", "carry_lateral_left", "carry_lateral_right"}
)
UNLOADED_TRANSLATION_SKILLS = frozenset(
    {
        "parallel_forward",
        "parallel_backward",
        "parallel_lateral_left",
        "parallel_lateral_right",
    }
)
EXIT_SKILL_TO_UNLOADED_SKILL = {
    "exit_forward": "parallel_forward",
    "exit_backward": "parallel_backward",
    "exit_lateral_left": "parallel_lateral_left",
    "exit_lateral_right": "parallel_lateral_right",
}
EXIT_TRANSLATION_SKILLS = frozenset(EXIT_SKILL_TO_UNLOADED_SKILL)
ALL_TRANSLATION_SKILLS = (
    TRANSLATION_SKILLS | UNLOADED_TRANSLATION_SKILLS | EXIT_TRANSLATION_SKILLS
)
PAYLOAD_SAFETY_SKILLS = TRANSLATION_SKILLS | frozenset(
    {
        "carry_turn_in_place",
        "carry_place_crouch",
        "carry_recover_stand",
        "carry_recover_payload_inward",
        "place_payload_on_arms",
        "release_payload",
    }
)


class SkillBackend(Protocol):
    """Backend used by G1SkillAPI to execute one safe skill call."""

    def execute_skill(
        self,
        *,
        skill_name: str,
        parameters: dict[str, object],
        output_dir: Path,
        heartbeat_file: Path,
        runtime_state_file: Path | None,
        call_index: int,
    ) -> SkillCallRecord:
        ...


@dataclass
class StaticSonicSkillBackend:
    """Publish tuned static SONIC skills through validate_static_sonic_skill.py."""

    execute: bool = False
    robots: tuple[str, ...] = ("front", "back")
    front_port: int = 5556
    back_port: int = 5558
    publish_hz: float = 20.0
    default_start_pulse_seconds: float = 0.0
    default_stand_seconds: float = 1.0
    minimum_transport_window_s: float = 1.0
    minimum_lateral_transport_window_s: float = 0.4
    nominal_distance_duration_gain: float = 1.0
    # The manual success envelope is 2 m nominal at 0.5 command m/s. Keeping
    # that distance as the neutral baseline lets distance/speed compile to the
    # same four-second active window without adding a second hidden duration.
    nominal_distance_duration_baseline_m: float = 2.0
    minimum_transport_ramp_s: float = 1.5
    post_transport_idle_s: float = 1.0
    payload_safety_profile: str = "task"
    # Optional legacy overrides are retained for strict debug comparisons. Production
    # experiments select one profile instead of tuning each verifier threshold.
    max_payload_support_center_error_m: float | None = None
    payload_support_min_force_n: float | None = None
    payload_support_min_balance_ratio: float | None = None
    payload_force_imbalance_grace_s: float | None = None
    team_sync_enabled: bool = True
    team_sync_active_catchup: bool = False
    team_sync_longitudinal_kp: float = 2.0
    team_sync_cross_track_kp: float = 0.6
    team_sync_payload_kp: float = 1.2
    team_sync_error_filter_alpha: float = 0.35
    team_sync_pause_leader_error_m: float = 0.008
    team_sync_resume_error_m: float = 0.003
    team_sync_lagging_boost_mps: float = 0.03
    team_sync_max_correction_mps: float = 0.06
    team_sync_min_active_speed_ratio: float = 0.0
    team_sync_hard_desync_m: float | None = None
    linear_response_calibration_file: Path | None = None
    stream_output: bool = True

    @property
    def carry_safety(self) -> CarrySafetyProfile:
        return get_carry_safety_profile(self.payload_safety_profile)

    @property
    def effective_max_payload_support_center_error_m(self) -> float:
        if self.max_payload_support_center_error_m is not None:
            return float(self.max_payload_support_center_error_m)
        return self.carry_safety.max_support_center_error_m

    def with_robots(self, robots: tuple[str, ...]) -> "StaticSonicSkillBackend":
        """Return a backend view targeting exactly the requested deploy instances."""

        normalized = tuple(dict.fromkeys(str(robot) for robot in robots))
        if not normalized or not set(normalized).issubset({"front", "back"}):
            raise ValueError(f"invalid SONIC robot scope: {normalized}")
        return replace(self, robots=normalized)

    def resynchronize_idle(
        self,
        *,
        output_dir: Path,
        heartbeat_file: Path,
        runtime_state_file: Path | None,
        duration_s: float = 4.5,
        start_pulse_seconds: float = 0.0,
        stand_phase_id: str = "dual_motor_preflight_stand",
    ) -> dict[str, object]:
        """Keep both deploy planners in IDLE while a restored state settles."""

        output_dir.mkdir(parents=True, exist_ok=True)
        python_bin = str(SONIC_PYTHON) if SONIC_PYTHON.exists() else sys.executable
        command = [
            python_bin,
            str(DUAL_PREFLIGHT),
            "--front-port",
            str(self.front_port),
            "--back-port",
            str(self.back_port),
            "--start-pulse-seconds",
            str(max(0.0, float(start_pulse_seconds))),
            "--startup-delay",
            "0.3",
            "--planner-soft-reset",
            "--planner-soft-reset-streamed-seconds",
            "0.5",
            "--planner-soft-reset-idle-seconds",
            "2.0",
            "--stabilize-seconds",
            "0",
            "--stand-seconds",
            str(max(0.5, float(duration_s))),
            "--stand-phase-id",
            str(stand_phase_id),
            "--walk-meters",
            "0",
            "--upper-body-pose",
            "none",
            "--hand-pose",
            "none",
            "--final-idle-seconds",
            "0",
            "--no-stop",
            "--heartbeat-file",
            str(heartbeat_file),
            "--output-dir",
            str(output_dir),
        ]
        (output_dir / "idle_resync_command.sh").write_text(_shell_join(command) + "\n", encoding="utf-8")
        started = time.time()
        if not self.execute:
            return {"status": "dry_run", "command": command}
        returncode, stdout, stderr = _run_command(
            command,
            cwd=REPO_ROOT,
            stream_output=self.stream_output,
        )
        (output_dir / "idle_resync_stdout.log").write_text(stdout, encoding="utf-8")
        (output_dir / "idle_resync_stderr.log").write_text(stderr, encoding="utf-8")
        result = {
            "status": "success" if returncode == 0 else f"failed_rc_{returncode}",
            "returncode": returncode,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "runtime_state_file": "" if runtime_state_file is None else str(runtime_state_file),
            "command": command,
        }
        if returncode != 0:
            detail = stderr.strip() or stdout.strip()
            raise RuntimeError(f"SONIC IDLE resynchronization failed: {detail}")
        return result

    def execute_skill(
        self,
        *,
        skill_name: str,
        parameters: dict[str, object],
        output_dir: Path,
        heartbeat_file: Path,
        runtime_state_file: Path | None,
        call_index: int,
    ) -> SkillCallRecord:
        started = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        requested_parameters = dict(parameters)
        effective_parameters = dict(parameters)
        command = self._command(
            skill_name,
            effective_parameters,
            output_dir,
            heartbeat_file,
            runtime_state_file,
        )
        command_file = output_dir / f"{call_index:02d}_{skill_name}_command.sh"
        command_file.write_text(_shell_join(command) + "\n", encoding="utf-8")

        status = "dry_run"
        stdout = ""
        stderr = ""
        if self.execute:
            returncode, stdout, stderr = _run_command(
                command,
                cwd=REPO_ROOT,
                stream_output=self.stream_output,
            )
            status = "success" if returncode == 0 else f"failed_rc_{returncode}"
            (output_dir / f"{call_index:02d}_{skill_name}_stdout.log").write_text(stdout, encoding="utf-8")
            (output_dir / f"{call_index:02d}_{skill_name}_stderr.log").write_text(stderr, encoding="utf-8")

        return SkillCallRecord(
            index=call_index,
            skill_name=skill_name,
            parameters=requested_parameters,
            status=status,
            started_at_unix=started,
            finished_at_unix=time.time(),
            artifact_path=str(output_dir),
            stdout=stdout,
            stderr=stderr,
            robot_ids=tuple(self.robots),
            effective_parameters=effective_parameters,
        )

    def execute_team_actions(
        self,
        *,
        actions_by_robot: dict[str, dict[str, object]],
        phase_id: str,
        synchronization_group: str,
        coordination_mode: str,
        output_dir: Path,
        heartbeat_file: Path,
        runtime_state_file: Path | None,
        call_index: int,
    ) -> SkillCallRecord:
        """Run one atomic publish loop with explicit per-robot actions."""

        if set(actions_by_robot) != {"front", "back"}:
            raise ValueError("coordinated SONIC dispatch requires front and back actions")
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = str(coordination_mode).strip().lower()
        if mode not in {"sync", "async"}:
            raise ValueError("coordinated SONIC dispatch mode must be sync or async")
        plan = {
            "coordination_mode": mode,
            "phase_id": str(phase_id),
            "synchronization_group": str(synchronization_group),
            "actions": actions_by_robot,
        }
        plan_path = output_dir / (
            "async_team_plan.json" if mode == "async" else "sync_compensated_team_plan.json"
        )
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        python_bin = str(SONIC_PYTHON) if SONIC_PYTHON.exists() else sys.executable
        command = [
            python_bin,
            str(COORDINATED_STATIC_SKILL),
            "--plan-file",
            str(plan_path),
            "--front-port",
            str(self.front_port),
            "--back-port",
            str(self.back_port),
            "--publish-hz",
            str(self.publish_hz),
            "--start-pulse-seconds",
            str(self.default_start_pulse_seconds),
            "--stand-seconds",
            str(self.default_stand_seconds),
            "--minimum-transport-window-s",
            str(self.minimum_transport_window_s),
            "--minimum-lateral-transport-window-s",
            str(self.minimum_lateral_transport_window_s),
            "--transport-duration-gain",
            str(self.nominal_distance_duration_gain),
            "--transport-duration-baseline-m",
            str(self.nominal_distance_duration_baseline_m),
            "--minimum-transport-ramp-s",
            str(self.minimum_transport_ramp_s),
            "--post-transport-idle-s",
            str(self.post_transport_idle_s),
            "--payload-safety-profile",
            str(self.payload_safety_profile),
            "--team-sync-longitudinal-kp",
            str(self.team_sync_longitudinal_kp),
            "--team-sync-cross-track-kp",
            str(self.team_sync_cross_track_kp),
            "--team-sync-payload-kp",
            str(self.team_sync_payload_kp),
            "--team-sync-error-filter-alpha",
            str(self.team_sync_error_filter_alpha),
            "--team-sync-pause-leader-error-m",
            str(self.team_sync_pause_leader_error_m),
            "--team-sync-resume-error-m",
            str(self.team_sync_resume_error_m),
            "--team-sync-lagging-boost-mps",
            str(self.team_sync_lagging_boost_mps),
            "--team-sync-max-correction-mps",
            str(self.team_sync_max_correction_mps),
            "--team-sync-min-active-speed-ratio",
            str(self.team_sync_min_active_speed_ratio),
            "--heartbeat-file",
            str(heartbeat_file),
            "--output-dir",
            str(output_dir),
        ]
        if self.linear_response_calibration_file is not None:
            command.extend(
                [
                    "--linear-response-calibration-file",
                    str(self.linear_response_calibration_file),
                ]
            )
        safety_overrides = {
            "--payload-safety-max-support-center-error-m": (
                self.max_payload_support_center_error_m
            ),
            "--payload-support-min-force-n": self.payload_support_min_force_n,
            "--payload-support-min-balance-ratio": (
                self.payload_support_min_balance_ratio
            ),
            "--payload-force-imbalance-grace-s": (
                self.payload_force_imbalance_grace_s
            ),
            "--team-sync-hard-desync-m": self.team_sync_hard_desync_m,
        }
        for option, value in safety_overrides.items():
            if value is not None:
                command.extend([option, str(value)])
        if runtime_state_file is not None:
            command.extend(["--runtime-state-file", str(runtime_state_file)])
        if not self.team_sync_enabled:
            command.append("--no-team-sync")
        if self.team_sync_active_catchup:
            command.append("--team-sync-active-catchup")
        command_file = output_dir / f"{call_index:02d}_{mode}_team_phase_command.sh"
        command_file.write_text(_shell_join(command) + "\n", encoding="utf-8")

        started = time.time()
        status = "dry_run"
        stdout = ""
        stderr = ""
        if self.execute:
            returncode, stdout, stderr = _run_command(
                command,
                cwd=REPO_ROOT,
                stream_output=self.stream_output,
            )
            status = "success" if returncode == 0 else f"failed_rc_{returncode}"
            (output_dir / f"{call_index:02d}_{mode}_team_phase_stdout.log").write_text(
                stdout,
                encoding="utf-8",
            )
            (output_dir / f"{call_index:02d}_{mode}_team_phase_stderr.log").write_text(
                stderr,
                encoding="utf-8",
            )
        coordinated_result: dict[str, object] = {}
        result_path = output_dir / "coordinated_team_result.json"
        if result_path.exists():
            try:
                loaded_result = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(loaded_result, dict):
                    coordinated_result = loaded_result
            except (OSError, json.JSONDecodeError):
                coordinated_result = {
                    "status": "artifact_unavailable",
                    "path": str(result_path),
                }
        moving_methods = [
            str(action.get("method", ""))
            for action in actions_by_robot.values()
            if str(action.get("method", "")) != "hold_position"
        ]
        skill_name = moving_methods[0] if moving_methods else "hold_position"
        return SkillCallRecord(
            index=call_index,
            skill_name=skill_name,
            parameters={
                "coordination_mode": mode,
                "phase_id": phase_id,
                "synchronization_group": synchronization_group,
                "actions": actions_by_robot,
            },
            status=status,
            started_at_unix=started,
            finished_at_unix=time.time(),
            artifact_path=str(output_dir),
            stdout=stdout,
            stderr=stderr,
            robot_ids=("front", "back"),
            effective_parameters={
                "actions": actions_by_robot,
                "distance_semantics": {
                    "distance_m": "mujoco_projected_endpoint_target",
                    "speed_mps": "sonic_nominal_command",
                    "feedback": "payload_projected_displacement",
                    "calibration_file": (
                        ""
                        if self.linear_response_calibration_file is None
                        else str(self.linear_response_calibration_file)
                    ),
                },
                "coordinated_team_result": coordinated_result,
            },
        )

    def _command(
        self,
        skill_name: str,
        parameters: dict[str, object],
        output_dir: Path,
        heartbeat_file: Path,
        runtime_state_file: Path | None,
    ) -> list[str]:
        if not SONIC_PYTHON.exists():
            python_bin = sys.executable
        else:
            python_bin = str(SONIC_PYTHON)

        if skill_name in {"stop_all", "hold_position"}:
            return self._idle_command(
                python_bin=python_bin,
                parameters=parameters,
                output_dir=output_dir,
                heartbeat_file=heartbeat_file,
            )

        runner_skill_name = EXIT_SKILL_TO_UNLOADED_SKILL.get(skill_name, skill_name)
        if skill_name in ALL_TRANSLATION_SKILLS:
            requested_distance = abs(float(parameters.get("distance_m", 0.0)))
            requested_speed = abs(float(parameters.get("speed_mps", 0.0)))
            distance_above_baseline = max(
                0.0,
                requested_distance - self.nominal_distance_duration_baseline_m,
            )
            distance_window_s = (
                self.nominal_distance_duration_gain
                * distance_above_baseline
                / max(requested_speed, 1.0e-6)
            )
            minimum_window_s = (
                self.minimum_lateral_transport_window_s
                if skill_name
                in {
                    "carry_lateral_left",
                    "carry_lateral_right",
                    "exit_lateral_left",
                    "exit_lateral_right",
                }
                else self.minimum_transport_window_s
            )
            parameters.setdefault(
                "minimum_motion_duration_s",
                max(
                    0.4,
                    float(minimum_window_s) + distance_window_s,
                ),
            )
            parameters["transport_ramp_seconds"] = max(
                float(parameters.get("transport_ramp_seconds", 0.0)),
                float(self.minimum_transport_ramp_s),
            )
        if skill_name in {"carry_forward", "exit_forward"} and "speed_mps" in parameters:
            requested_speed = abs(float(parameters["speed_mps"]))
            parameters.setdefault(
                "front_speed_mps",
                requested_speed,
            )
            parameters.setdefault(
                "back_speed_mps",
                requested_speed,
            )
        if skill_name in {
            "carry_lateral_left",
            "carry_lateral_right",
            "exit_lateral_left",
            "exit_lateral_right",
        } and "speed_mps" in parameters:
            requested_speed = abs(float(parameters["speed_mps"]))
            parameters.setdefault(
                "front_speed_mps",
                requested_speed,
            )
            parameters.setdefault(
                "back_speed_mps",
                requested_speed,
            )

        command = [
            python_bin,
            str(VALIDATE_STATIC_SKILL),
            "--skill",
            runner_skill_name,
            "--robots",
            *self.robots,
            "--front-port",
            str(self.front_port),
            "--back-port",
            str(self.back_port),
            "--publish-hz",
            str(self.publish_hz),
            "--heartbeat-file",
            str(heartbeat_file),
            "--output-root",
            str(output_dir),
        ]
        if runtime_state_file is not None:
            command.extend(["--runtime-state-file", str(runtime_state_file)])
        if skill_name in PAYLOAD_SAFETY_SKILLS:
            command.extend(["--payload-safety-profile", str(self.payload_safety_profile)])
            safety_overrides = {
                "--payload-safety-max-support-center-error-m": (
                    self.max_payload_support_center_error_m
                ),
                "--payload-support-min-force-n": self.payload_support_min_force_n,
                "--payload-support-min-balance-ratio": (
                    self.payload_support_min_balance_ratio
                ),
                "--payload-force-imbalance-grace-s": (
                    self.payload_force_imbalance_grace_s
                ),
            }
            for option, value in safety_overrides.items():
                if value is not None:
                    command.extend([option, str(value)])
        if skill_name in TRANSLATION_SKILLS:
            command.extend(
                [
                    "--post-skill-idle-seconds",
                    str(self.post_transport_idle_s),
                    "--team-sync-longitudinal-kp",
                    str(self.team_sync_longitudinal_kp),
                    "--team-sync-cross-track-kp",
                    str(self.team_sync_cross_track_kp),
                    "--team-sync-payload-kp",
                    str(self.team_sync_payload_kp),
                    "--team-sync-error-filter-alpha",
                    str(self.team_sync_error_filter_alpha),
                    "--team-sync-pause-leader-error-m",
                    str(self.team_sync_pause_leader_error_m),
                    "--team-sync-resume-error-m",
                    str(self.team_sync_resume_error_m),
                    "--team-sync-lagging-boost-mps",
                    str(self.team_sync_lagging_boost_mps),
                    "--team-sync-max-correction-mps",
                    str(self.team_sync_max_correction_mps),
                    "--team-sync-min-active-speed-ratio",
                    str(self.team_sync_min_active_speed_ratio),
                ]
            )
            if self.team_sync_hard_desync_m is not None:
                command.extend(
                    ["--team-sync-hard-desync-m", str(self.team_sync_hard_desync_m)]
                )
            if not self.team_sync_enabled:
                command.append("--no-team-sync")
            if self.team_sync_active_catchup:
                command.append("--team-sync-active-catchup")
        start_pulse = parameters.pop("start_pulse_seconds", self.default_start_pulse_seconds)
        stand = parameters.pop("stand_seconds", self.default_stand_seconds)
        final_idle = parameters.pop("final_idle_seconds", 0.0)
        command.extend(
            [
                "--start-pulse-seconds",
                str(start_pulse),
                "--stand-seconds",
                str(stand),
                "--final-idle-seconds",
                str(final_idle),
            ]
        )
        option_map = {
            "distance_m": "--distance-m",
            "speed_mps": "--speed-mps",
            "front_speed_mps": "--front-speed-mps",
            "back_speed_mps": "--back-speed-mps",
            "transport_ramp_seconds": "--transport-ramp-seconds",
            "minimum_motion_duration_s": "--minimum-motion-duration-s",
            "height": "--height",
            "yaw_deg": "--yaw-deg",
            "yaw_rate_degps": "--yaw-rate-degps",
            "turn_segment_deg": "--turn-segment-deg",
            "turn_command_scale": "--turn-command-scale",
            "place_start_height": "--place-start-height",
            "place_height": "--place-height",
            "place_ramp_seconds": "--place-ramp-seconds",
            "place_hold_seconds": "--place-hold-seconds",
            "release_ramp_seconds": "--release-ramp-seconds",
            "release_hold_seconds": "--release-hold-seconds",
            "recovery_ramp_seconds": "--recovery-ramp-seconds",
            "recovery_hold_seconds": "--recovery-hold-seconds",
            "upper_body_event_id": "--upper-body-event-id",
            "payload_x": "--payload-x",
            "payload_y": "--payload-y",
            "payload_z": "--payload-z",
            "payload_yaw": "--payload-yaw",
            "payload_event_id": "--payload-event-id",
            "mode": "--mode",
            "current_yaw_deg": "--current-yaw-deg",
        }
        for key, option in option_map.items():
            if key not in parameters:
                continue
            value = parameters[key]
            if value is None:
                continue
            command.extend([option, str(value)])
        if bool(parameters.get("payload_force", False)):
            command.append("--payload-force")
        if bool(parameters.get("reset_heading_state", False)):
            command.append("--reset-heading-state")
        return command

    def _idle_command(
        self,
        *,
        python_bin: str,
        parameters: dict[str, object],
        output_dir: Path,
        heartbeat_file: Path,
    ) -> list[str]:
        """Build a real planner IDLE transaction instead of a zero-distance gait."""

        duration_s = max(
            0.5,
            float(parameters.get("hold_seconds", self.default_stand_seconds)),
        )
        return [
            python_bin,
            str(DUAL_PREFLIGHT),
            "--robots",
            *self.robots,
            "--front-port",
            str(self.front_port),
            "--back-port",
            str(self.back_port),
            "--publish-hz",
            str(self.publish_hz),
            "--start-pulse-seconds",
            "0",
            "--stabilize-seconds",
            "0",
            "--stand-seconds",
            str(duration_s),
            "--walk-meters",
            "0",
            "--upper-body-pose",
            "none",
            "--hand-pose",
            "none",
            "--final-idle-seconds",
            "0",
            "--no-stop",
            "--heartbeat-file",
            str(heartbeat_file),
            "--output-dir",
            str(output_dir),
        ]


def _shell_join(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    stream_output: bool,
) -> tuple[int, str, str]:
    """Run a backend command while preserving artifacts and optional live output."""

    if not stream_output:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr

    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output.append(line)
        print(line, end="", flush=True)
    return process.wait(), "".join(output), ""
