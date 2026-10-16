#!/usr/bin/env python3
"""Validate one tuned static SONIC skill through the live Term1/2/3 setup.

This is a Terminal 4 convenience wrapper. It assumes:

- Terminal 1 is running `run_dual_sonic_manual_layout_sim.py`;
- Terminal 2/3 are running official GEAR-SONIC deploy backends;
- the heartbeat file path matches Terminal 1.

The script does not claim automatic physical success. It publishes the selected
skill through the official deploy path and writes the command artifact for
manual visual validation.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.sonic.runtime_state import wait_for_payload_supported  # noqa: E402
from praxis_coop.sonic.upper_body import PAYLOAD_RECOVERY_OBSERVATION_S  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SONIC_PYTHON = REPO_ROOT / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
RUN_STATIC_SKILL = REPO_ROOT / "scripts" / "run_sonic_static_skill.py"
DEFAULT_OUT = REPO_ROOT / "results" / "deployment" / "static_skill_live"
DEFAULT_HEADING_STATE = "static_skill_heading_state.json"
DEFAULT_RUNTIME_STATE = "runtime_state.json"
VALUE_OPTIONS = {
    "--distance-m",
    "--speed-mps",
    "--front-speed-mps",
    "--back-speed-mps",
    "--transport-ramp-seconds",
    "--minimum-motion-duration-s",
    "--mode",
    "--height",
    "--facing-x",
    "--facing-y",
    "--movement-x",
    "--movement-y",
    "--yaw-deg",
    "--yaw-rate-degps",
    "--turn-segment-deg",
    "--turn-command-scale",
    "--turn-heading-mode",
    "--place-start-height",
    "--place-height",
    "--place-ramp-seconds",
    "--place-hold-seconds",
    "--release-ramp-seconds",
    "--release-hold-seconds",
    "--recovery-ramp-seconds",
    "--recovery-hold-seconds",
    "--payload-x",
    "--payload-y",
    "--payload-z",
    "--payload-yaw",
    "--payload-rearward-offset-m",
    "--payload-event-id",
    "--upper-body-event-id",
}
DIRECT_VALUE_ARGS = (
    ("distance_m", "--distance-m"),
    ("speed_mps", "--speed-mps"),
    ("front_speed_mps", "--front-speed-mps"),
    ("back_speed_mps", "--back-speed-mps"),
    ("transport_ramp_seconds", "--transport-ramp-seconds"),
    ("minimum_motion_duration_s", "--minimum-motion-duration-s"),
    ("mode", "--mode"),
    ("height", "--height"),
    ("facing_x", "--facing-x"),
    ("facing_y", "--facing-y"),
    ("movement_x", "--movement-x"),
    ("movement_y", "--movement-y"),
    ("yaw_deg", "--yaw-deg"),
    ("yaw_rate_degps", "--yaw-rate-degps"),
    ("turn_segment_deg", "--turn-segment-deg"),
    ("turn_command_scale", "--turn-command-scale"),
    ("turn_heading_mode", "--turn-heading-mode"),
    ("place_start_height", "--place-start-height"),
    ("place_height", "--place-height"),
    ("place_ramp_seconds", "--place-ramp-seconds"),
    ("place_hold_seconds", "--place-hold-seconds"),
    ("release_ramp_seconds", "--release-ramp-seconds"),
    ("release_hold_seconds", "--release-hold-seconds"),
    ("recovery_ramp_seconds", "--recovery-ramp-seconds"),
    ("recovery_hold_seconds", "--recovery-hold-seconds"),
    ("payload_x", "--payload-x"),
    ("payload_y", "--payload-y"),
    ("payload_z", "--payload-z"),
    ("payload_yaw", "--payload-yaw"),
    ("payload_rearward_offset_m", "--payload-rearward-offset-m"),
    ("payload_event_id", "--payload-event-id"),
    ("upper_body_event_id", "--upper-body-event-id"),
)
MODE_CHOICES = (
    "skill_default",
    "idle",
    "idle_squat",
    "squat",
    "slow_walk",
    "walk",
    "run",
    "object_carrying",
)
DEPLOY_HEADING_IDLE_SKILLS = frozenset(
    {
        "place_payload_on_arms",
        "carry_place_crouch",
        "carry_recover_stand",
        "carry_recover_payload_inward",
        "release_payload",
    }
)


SKILL_DEFAULTS: dict[str, list[str]] = {
    "parallel_forward": [
        "--distance-m",
        "2.0",
        "--speed-mps",
        "0.20",
    ],
    "parallel_backward": [
        "--distance-m",
        "1.5",
        "--speed-mps",
        "0.12",
    ],
    "carry_forward": [
        "--distance-m",
        "2.0",
        "--speed-mps",
        "0.50",
    ],
    "carry_backward": [
        "--distance-m",
        "2.0",
        "--speed-mps",
        "0.50",
    ],
    "carry_lateral_left": [
        "--distance-m",
        "2.0",
        "--speed-mps",
        "0.50",
    ],
    "carry_lateral_right": [
        "--distance-m",
        "2.0",
        "--speed-mps",
        "0.50",
    ],
    "parallel_lateral_left": [
        "--distance-m",
        "1.5",
        "--speed-mps",
        "0.45",
    ],
    "parallel_lateral_right": [
        "--distance-m",
        "1.5",
        "--speed-mps",
        "0.45",
    ],
    "carry_turn_in_place": [
        "--yaw-deg",
        "90",
        "--yaw-rate-degps",
        "5",
        "--turn-segment-deg",
        "3",
        "--turn-command-scale",
        "1.0",
        "--turn-heading-mode",
        "incremental",
    ],
    "carry_turn_clockwise_90": [
        # SONIC/GEAR-SONIC convention used in this wrapper:
        # +yaw is counterclockwise, -yaw is clockwise.
        "--yaw-deg",
        "-90",
        "--yaw-rate-degps",
        "5",
        "--turn-segment-deg",
        "3",
        "--turn-command-scale",
        "1.0",
        "--turn-heading-mode",
        "incremental",
    ],
    "carry_turn_counterclockwise_90": [
        "--yaw-deg",
        "90",
        "--yaw-rate-degps",
        "5",
        "--turn-segment-deg",
        "3",
        "--turn-command-scale",
        "1.0",
        "--turn-heading-mode",
        "incremental",
    ],
    "carry_place_crouch": [
        "--place-start-height",
        "0.80",
        "--place-height",
        "0.55",
        "--place-ramp-seconds",
        "2",
        "--place-hold-seconds",
        "10",
    ],
    "place_payload_on_arms": [
        "--height",
        "0.80",
        "--place-hold-seconds",
        "10",
    ],
    "carry_recover_stand": [
        "--height",
        "0.80",
        "--place-ramp-seconds",
        "3",
        "--place-hold-seconds",
        "5",
    ],
    "carry_recover_payload_inward": [
        "--height",
        "0.80",
        "--recovery-ramp-seconds",
        "6",
        "--recovery-hold-seconds",
        "2",
    ],
    "release_payload": [
        "--height",
        "0.55",
        "--release-ramp-seconds",
        "4",
        "--release-hold-seconds",
        "2",
    ],
}


UNDERLYING_SKILL = {
    "carry_turn_clockwise_90": "carry_turn_in_place",
    "carry_turn_counterclockwise_90": "carry_turn_in_place",
}

def main() -> None:
    args = parse_args()
    if not SONIC_PYTHON.exists():
        raise SystemExit(f"missing SONIC python: {SONIC_PYTHON}")
    if args.reset_heading_state:
        _write_heading_state(args.heading_state_file, 0.0, "reset")

    if args.skill == "all":
        skills = (
            "parallel_forward",
            "carry_forward",
            "carry_backward",
            "carry_lateral_left",
            "carry_lateral_right",
            "carry_turn_clockwise_90",
            "carry_turn_counterclockwise_90",
            "place_payload_on_arms",
            "carry_place_crouch",
            "carry_recover_stand",
            "carry_recover_payload_inward",
        )
    else:
        skills = (args.skill,)

    heartbeat_file = args.heartbeat_file or args.output_root / "planner_heartbeat.json"
    for skill in skills:
        run_single_skill(args, skill, heartbeat_file)


def parse_args() -> argparse.Namespace:
    raw_args = sys.argv[1:]
    force_dry_run = "--dry-run" in raw_args
    if force_dry_run:
        raw_args = [arg for arg in raw_args if arg != "--dry-run"]
    extra_args: list[str] = []
    if "--extra" in raw_args:
        extra_index = raw_args.index("--extra")
        extra_args = raw_args[extra_index + 1 :]
        raw_args = raw_args[:extra_index]
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skill",
        choices=(*sorted(SKILL_DEFAULTS), "all"),
        required=True,
        help="Static skill to validate from Terminal 4.",
    )
    parser.add_argument("--robots", nargs="+", choices=("front", "back"), default=["front", "back"])
    parser.add_argument("--front-port", type=int, default=5556)
    parser.add_argument("--back-port", type=int, default=5558)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--heartbeat-file", type=Path, default=None)
    parser.add_argument("--heading-state-file", type=Path, default=None)
    parser.add_argument("--runtime-state-file", type=Path, default=None)
    parser.add_argument("--runtime-state-max-age-s", type=float, default=1.0)
    parser.add_argument("--payload-ready-timeout-s", type=float, default=45.0)
    parser.add_argument("--payload-ready-stable-s", type=float, default=1.0)
    parser.add_argument("--payload-verify-timeout-s", type=float, default=15.0)
    parser.add_argument("--payload-support-stable-s", type=float, default=1.0)
    parser.add_argument("--payload-min-z", type=float, default=0.80)
    parser.add_argument("--payload-safety-profile", choices=("task", "strict"), default="task")
    parser.add_argument("--payload-safety-min-z", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-safety-min-pelvis-height", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-safety-max-tilt-deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-safety-hard-tilt-deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-tilt-grace-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-safety-max-support-center-error-m", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-contact-loss-grace-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-support-min-force-n", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-support-min-balance-ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-force-imbalance-grace-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-team-sync", action="store_true")
    parser.add_argument(
        "--team-sync-active-catchup",
        action="store_true",
        help="Explicit ablation: allow different leader/lagger velocity commands.",
    )
    parser.add_argument("--team-sync-longitudinal-kp", type=float, default=2.0)
    parser.add_argument("--team-sync-cross-track-kp", type=float, default=0.6)
    parser.add_argument("--team-sync-payload-kp", type=float, default=1.2)
    parser.add_argument("--team-sync-error-filter-alpha", type=float, default=0.35)
    parser.add_argument("--team-sync-pause-leader-error-m", type=float, default=0.008)
    parser.add_argument("--team-sync-resume-error-m", type=float, default=0.003)
    parser.add_argument("--team-sync-lagging-boost-mps", type=float, default=0.03)
    parser.add_argument("--team-sync-max-correction-mps", type=float, default=0.06)
    parser.add_argument("--team-sync-min-active-speed-ratio", type=float, default=0.0)
    parser.add_argument("--team-sync-hard-desync-m", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--post-skill-idle-seconds", type=float, default=1.0)
    parser.add_argument("--no-payload-safety-monitor", action="store_true")
    parser.add_argument(
        "--no-payload-auto-reset",
        action="store_true",
        help="Fail instead of re-placing an active payload that has already fallen or lost support.",
    )
    parser.add_argument(
        "--no-live-runtime-heading",
        action="store_true",
        help="Do not read live MuJoCo root yaw from runtime_state.json for body-relative lateral skills.",
    )
    parser.add_argument("--reset-heading-state", action="store_true")
    parser.add_argument(
        "--current-yaw-deg",
        type=float,
        default=None,
        help="Override the heading state for this wrapper call. 0 means facing +X; +90 is counterclockwise.",
    )
    parser.add_argument(
        "--start-pulse-seconds",
        type=float,
        default=3.0,
        help="Startup command/heartbeat window. Use 0 only after Term1 is already dropped and stable.",
    )
    parser.add_argument(
        "--stand-seconds",
        type=float,
        default=3.0,
        help="Pre-skill stabilize time. Use 1 or 0 for repeated commands after the robots are already stable.",
    )
    parser.add_argument("--final-idle-seconds", type=float, default=0.0)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--distance-m", type=float, default=None, help="Translation distance for forward/back/lateral skills.")
    parser.add_argument("--speed-mps", type=float, default=None, help="Translation speed for forward/back/lateral skills.")
    parser.add_argument(
        "--front-speed-mps",
        type=float,
        default=None,
        help="Override only the front SONIC backend speed while keeping the shared phase window.",
    )
    parser.add_argument(
        "--back-speed-mps",
        type=float,
        default=None,
        help="Override only the back SONIC backend speed while keeping the shared phase window.",
    )
    parser.add_argument(
        "--transport-ramp-seconds",
        type=float,
        default=None,
        help="Planner-aware zero-speed engage/brake hold around a translation skill.",
    )
    parser.add_argument(
        "--minimum-motion-duration-s",
        type=float,
        default=None,
        help="Minimum active gait command window; used by the cooperative Agent backend.",
    )
    parser.add_argument("--mode", choices=MODE_CHOICES, default=None, help="Override the SONIC locomotion mode.")
    parser.add_argument("--height", type=float, default=None, help="Planner height command, mainly for crouch/place tuning.")
    parser.add_argument("--facing-x", type=float, default=None, help="Override planner facing x.")
    parser.add_argument("--facing-y", type=float, default=None, help="Override planner facing y.")
    parser.add_argument("--movement-x", type=float, default=None, help="Override planner movement x.")
    parser.add_argument("--movement-y", type=float, default=None, help="Override planner movement y.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Turn angle. + is counterclockwise, - is clockwise.")
    parser.add_argument("--yaw-rate-degps", type=float, default=None, help="Turn rate in degrees/s.")
    parser.add_argument("--turn-segment-deg", type=float, default=None, help="Increment size for segmented turn commands.")
    parser.add_argument("--turn-command-scale", type=float, default=None, help="Scale applied to --yaw-deg before publishing.")
    parser.add_argument(
        "--turn-heading-mode",
        choices=("incremental", "absolute", "absolute_state"),
        default=None,
        help="Turn command mode. Keep incremental unless deliberately debugging heading state.",
    )
    parser.add_argument("--place-start-height", type=float, default=None, help="Starting height for crouch/place.")
    parser.add_argument("--place-height", type=float, default=None, help="Target height for crouch/place.")
    parser.add_argument("--place-ramp-seconds", type=float, default=None, help="Seconds to lower from start height to place height.")
    parser.add_argument("--place-hold-seconds", type=float, default=None, help="Seconds to hold the place height.")
    parser.add_argument("--release-ramp-seconds", type=float, default=None, help="Seconds to withdraw both carry arms during final payload release.")
    parser.add_argument("--release-hold-seconds", type=float, default=None, help="Seconds to hold the released arm pose.")
    parser.add_argument(
        "--recovery-ramp-seconds",
        type=float,
        default=None,
        help="Seconds for the synchronized elbow60-to-elbow50 arm recovery transition.",
    )
    parser.add_argument(
        "--recovery-hold-seconds",
        type=float,
        default=None,
        help="Stationary dual-support hold after the payload recovery transition.",
    )
    parser.add_argument("--payload-x", type=float, default=None, help="Optional payload placement x override.")
    parser.add_argument("--payload-y", type=float, default=None, help="Optional payload placement y override.")
    parser.add_argument("--payload-z", type=float, default=None, help="Optional payload placement z override.")
    parser.add_argument("--payload-yaw", type=float, default=None, help="Optional payload placement yaw override.")
    parser.add_argument(
        "--payload-rearward-offset-m",
        type=float,
        default=None,
        help="Place the payload this far behind the live four-palm center in the mean body frame.",
    )
    parser.add_argument("--payload-event-id", default=None, help="Optional stable payload event id.")
    parser.add_argument(
        "--upper-body-event-id",
        default=None,
        help="Optional stable synchronized upper-body event id.",
    )
    parser.add_argument("--payload-force", action="store_true", help="Force payload re-placement if already active.")
    parser.add_argument(
        "--lateral-facing-mode",
        choices=("deploy_hold", "body_heading"),
        default="deploy_hold",
        help=(
            "For lateral skills, deploy_hold keeps planner facing at +X and only changes movement; "
            "body_heading sends facing equal to the live body yaw and may trigger SONIC to turn."
        ),
    )
    parser.add_argument("--deploy-facing-x", type=float, default=1.0)
    parser.add_argument("--deploy-facing-y", type=float, default=0.0)
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Additional arguments passed to run_sonic_static_skill.py after '--'.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(raw_args)
    args.extra = [*_direct_value_args(args), *extra_args]
    args.dry_run = args.dry_run or force_dry_run
    args.start_pulse_seconds_explicit = "--start-pulse-seconds" in raw_args
    args.stand_seconds_explicit = "--stand-seconds" in raw_args
    if args.skill == "carry_recover_payload_inward" and (
        len(args.robots) != 2 or set(args.robots) != {"front", "back"}
    ):
        parser.error(
            "carry_recover_payload_inward requires exactly --robots front back"
        )
    if args.heading_state_file is None:
        args.heading_state_file = args.output_root / DEFAULT_HEADING_STATE
    if args.runtime_state_file is None:
        args.runtime_state_file = args.output_root / DEFAULT_RUNTIME_STATE
    return args


def _direct_value_args(args: argparse.Namespace) -> list[str]:
    if (args.movement_x is None) != (args.movement_y is None):
        raise SystemExit("--movement-x and --movement-y must be provided together")
    if (args.facing_x is None) != (args.facing_y is None):
        raise SystemExit("--facing-x and --facing-y must be provided together")

    result: list[str] = []
    for attr, option in DIRECT_VALUE_ARGS:
        value = getattr(args, attr)
        if value is None:
            continue
        result.extend([option, str(value)])
    if getattr(args, "payload_force", False):
        result.append("--payload-force")
    return result


def run_single_skill(
    args: argparse.Namespace,
    skill: str,
    heartbeat_file: Path,
    *,
    output_dir: Path | None = None,
) -> None:
    command = build_command(args, skill, heartbeat_file, output_dir=output_dir)
    context = describe_command(args, skill, command)
    if context:
        print(f"[static-skill-validate] {context}", flush=True)
    print("[static-skill-validate] " + " ".join(str(item) for item in command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    if skill in {"place_payload_on_arms", "carry_recover_payload_inward"} and args.runtime_state_file is not None:
        _require_payload_supported(args, skill=skill)
    _update_heading_state_after_success(args, skill, command)


def _require_payload_supported(args: argparse.Namespace, *, skill: str) -> None:
    runtime_state_file = Path(args.runtime_state_file)
    status = wait_for_payload_supported(
        runtime_state_file,
        timeout_s=args.payload_verify_timeout_s,
        # The child command continuously monitors recovery for 15 seconds. A
        # fresh final sample here closes the subprocess handoff race.
        stable_s=(0.0 if skill == "carry_recover_payload_inward" else args.payload_support_stable_s),
        max_age_s=args.runtime_state_max_age_s,
        min_z_m=args.payload_min_z,
        allow_non_support_robot_contacts=(
            args.payload_safety_profile == "task"
        ),
        safety_profile=args.payload_safety_profile,
    )
    print(
        f"[static-skill-validate] {skill} final payload support verified: "
        f"z={status.z:.3f} supporting={sorted(status.supporting_robots)} "
        f"support_geometries={sorted(status.supporting_geometries)} "
        f"floor={status.floor_contact} walls={list(status.wall_contacts)}",
        flush=True,
    )


def build_command(
    args: argparse.Namespace,
    skill: str,
    heartbeat_file: Path,
    *,
    output_dir: Path | None = None,
) -> list[str]:
    output_dir = output_dir or args.output_root / f"{skill}_validation"
    underlying_skill = UNDERLYING_SKILL.get(skill, skill)
    extra = _extra_with_body_relative_heading(args, skill)
    skill_args = _merge_value_option_overrides(SKILL_DEFAULTS[skill], extra)
    command = [
        str(SONIC_PYTHON),
        str(RUN_STATIC_SKILL),
        "--skill",
        underlying_skill,
        "--robots",
        *args.robots,
        "--front-port",
        str(args.front_port),
        "--back-port",
        str(args.back_port),
        "--start-pulse-seconds",
        str(_effective_start_pulse_seconds(args, skill)),
        "--stand-seconds",
        str(_effective_stand_seconds(args, skill)),
        "--final-idle-seconds",
        str(args.final_idle_seconds),
        "--publish-hz",
        str(args.publish_hz),
        "--heartbeat-file",
        str(heartbeat_file),
        "--output-dir",
        str(output_dir),
        "--no-stop",
        *skill_args,
    ]
    if args.runtime_state_file is not None:
        command.extend(
            [
                "--runtime-state-file",
                str(args.runtime_state_file),
                "--runtime-state-max-age-s",
                str(args.runtime_state_max_age_s),
            ]
        )
    if skill == "place_payload_on_arms" and args.runtime_state_file is not None:
        command.extend(
            [
                "--payload-ready-timeout-s",
                str(args.payload_ready_timeout_s),
                "--payload-ready-stable-s",
                str(args.payload_ready_stable_s),
                "--payload-verify-timeout-s",
                str(args.payload_verify_timeout_s),
                "--payload-support-stable-s",
                str(args.payload_support_stable_s),
                "--payload-min-z",
                str(args.payload_min_z),
                "--payload-safety-profile",
                str(args.payload_safety_profile),
            ]
        )
        if args.no_payload_auto_reset:
            command.append("--no-payload-auto-reset")
    if skill == "release_payload" and args.runtime_state_file is not None:
        command.extend(
            [
                "--payload-ready-timeout-s",
                str(args.payload_ready_timeout_s),
                "--payload-ready-stable-s",
                str(args.payload_ready_stable_s),
                "--payload-verify-timeout-s",
                str(args.payload_verify_timeout_s),
                "--payload-support-stable-s",
                str(args.payload_support_stable_s),
                "--payload-safety-profile",
                str(args.payload_safety_profile),
            ]
        )
    if skill.startswith("carry_") and args.runtime_state_file is not None:
        command.extend(
            [
                "--payload-safety-profile",
                str(args.payload_safety_profile),
                "--team-sync-longitudinal-kp",
                str(args.team_sync_longitudinal_kp),
                "--team-sync-cross-track-kp",
                str(args.team_sync_cross_track_kp),
                "--team-sync-payload-kp",
                str(args.team_sync_payload_kp),
                "--team-sync-error-filter-alpha",
                str(args.team_sync_error_filter_alpha),
                "--team-sync-pause-leader-error-m",
                str(args.team_sync_pause_leader_error_m),
                "--team-sync-resume-error-m",
                str(args.team_sync_resume_error_m),
                "--team-sync-lagging-boost-mps",
                str(args.team_sync_lagging_boost_mps),
                "--team-sync-max-correction-mps",
                str(args.team_sync_max_correction_mps),
                "--team-sync-min-active-speed-ratio",
                str(args.team_sync_min_active_speed_ratio),
                "--post-skill-idle-seconds",
                str(_effective_post_skill_idle_seconds(args, skill)),
            ]
        )
        legacy_safety_overrides = {
            "--payload-safety-min-z": args.payload_safety_min_z,
            "--payload-safety-min-pelvis-height": args.payload_safety_min_pelvis_height,
            "--payload-safety-max-tilt-deg": args.payload_safety_max_tilt_deg,
            "--payload-safety-hard-tilt-deg": args.payload_safety_hard_tilt_deg,
            "--payload-tilt-grace-s": args.payload_tilt_grace_s,
            "--payload-safety-max-support-center-error-m": (
                args.payload_safety_max_support_center_error_m
            ),
            "--payload-contact-loss-grace-s": args.payload_contact_loss_grace_s,
            "--payload-support-min-force-n": args.payload_support_min_force_n,
            "--payload-support-min-balance-ratio": (
                args.payload_support_min_balance_ratio
            ),
            "--payload-force-imbalance-grace-s": (
                args.payload_force_imbalance_grace_s
            ),
            "--team-sync-hard-desync-m": args.team_sync_hard_desync_m,
        }
        for option, value in legacy_safety_overrides.items():
            if value is not None:
                command.extend([option, str(value)])
        if args.no_team_sync:
            command.append("--no-team-sync")
        if args.team_sync_active_catchup:
            command.append("--team-sync-active-catchup")
        if args.no_payload_safety_monitor:
            command.append("--no-payload-safety-monitor")
    return command


def _effective_start_pulse_seconds(args: argparse.Namespace, skill: str) -> float:
    if skill in {
        "carry_place_crouch",
        "carry_recover_stand",
        "carry_recover_payload_inward",
        "release_payload",
    } and not getattr(
        args, "start_pulse_seconds_explicit", False
    ):
        return 0.0
    return float(args.start_pulse_seconds)


def _effective_post_skill_idle_seconds(args: argparse.Namespace, skill: str) -> float:
    requested = max(0.0, float(args.post_skill_idle_seconds))
    if skill == "carry_recover_payload_inward":
        return max(requested, PAYLOAD_RECOVERY_OBSERVATION_S)
    return requested


def _merge_value_option_overrides(default_args: list[str], extra_args: list[str]) -> list[str]:
    overridden = {
        value
        for index, value in enumerate(extra_args[:-1])
        if value in VALUE_OPTIONS
    }
    merged_defaults: list[str] = []
    index = 0
    while index < len(default_args):
        value = default_args[index]
        if value in overridden and value in VALUE_OPTIONS and index + 1 < len(default_args):
            index += 2
            continue
        merged_defaults.append(value)
        index += 1
    return [*merged_defaults, *extra_args]


def _effective_stand_seconds(args: argparse.Namespace, skill: str) -> float:
    if skill in {
        "carry_place_crouch",
        "carry_recover_stand",
        "carry_recover_payload_inward",
        "release_payload",
    } and not getattr(
        args, "stand_seconds_explicit", False
    ):
        return 1.0
    if skill not in {"carry_lateral_left", "carry_lateral_right", "parallel_lateral_left", "parallel_lateral_right"}:
        return float(args.stand_seconds)
    if getattr(args, "stand_seconds_explicit", False):
        return float(args.stand_seconds)
    if _live_runtime_is_dropped(args):
        return 0.0
    return float(args.stand_seconds)


def _live_runtime_is_dropped(args: argparse.Namespace) -> bool:
    path = getattr(args, "runtime_state_file", None)
    if path is None:
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("elastic_band_dropped", False))


def _extra_with_body_relative_heading(args: argparse.Namespace, skill: str) -> list[str]:
    extra = list(args.extra)
    if skill in DEPLOY_HEADING_IDLE_SKILLS:
        if "--facing-x" in extra or "--facing-y" in extra:
            return extra
        # In SONIC's deploy frame, +X preserves the current body heading.
        # Re-encoding the live world yaw here would apply that yaw a second time.
        facing = _unit_xy(
            getattr(args, "deploy_facing_x", 1.0),
            getattr(args, "deploy_facing_y", 0.0),
        )
        return [
            "--facing-x",
            f"{facing[0]:.9f}",
            "--facing-y",
            f"{facing[1]:.9f}",
            *extra,
        ]
    if skill not in {"carry_lateral_left", "carry_lateral_right", "parallel_lateral_left", "parallel_lateral_right"}:
        return extra
    if "--movement-x" in extra or "--movement-y" in extra:
        return extra
    heading, _source = _current_heading_info(args)
    side_sign = 1.0 if skill.endswith("_left") else -1.0
    lateral_facing_mode = getattr(args, "lateral_facing_mode", "deploy_hold")
    if lateral_facing_mode == "body_heading":
        facing = (math.cos(heading), math.sin(heading))
        movement = (-math.sin(heading) * side_sign, math.cos(heading) * side_sign)
    else:
        facing = _unit_xy(getattr(args, "deploy_facing_x", 1.0), getattr(args, "deploy_facing_y", 0.0))
        movement = (-facing[1] * side_sign, facing[0] * side_sign)
    return [
        "--movement-x",
        f"{movement[0]:.9f}",
        "--movement-y",
        f"{movement[1]:.9f}",
        "--facing-x",
        f"{facing[0]:.9f}",
        "--facing-y",
        f"{facing[1]:.9f}",
        *extra,
    ]


def describe_command(args: argparse.Namespace, skill: str, command: list[str]) -> str:
    underlying_skill = UNDERLYING_SKILL.get(skill, skill)
    if underlying_skill == "carry_turn_in_place":
        yaw_deg = _last_float_option(command, "--yaw-deg", 0.0)
        scale = _last_float_option(command, "--turn-command-scale", 1.0)
        signed_yaw_deg = yaw_deg * scale
        direction = "clockwise" if signed_yaw_deg < 0.0 else "counterclockwise" if signed_yaw_deg > 0.0 else "no_yaw"
        return (
            f"turn_convention: +yaw=counterclockwise, -yaw=clockwise; "
            f"skill={skill} yaw_deg={yaw_deg:.1f} scale={scale:.3f} => {direction} {abs(signed_yaw_deg):.1f}deg"
        )
    if skill in {"carry_lateral_left", "carry_lateral_right", "parallel_lateral_left", "parallel_lateral_right"}:
        heading, source = _current_heading_info(args)
        facing_x = _last_float_option(command, "--facing-x", 1.0)
        facing_y = _last_float_option(command, "--facing-y", 0.0)
        facing_norm = math.hypot(facing_x, facing_y)
        if facing_norm <= 1.0e-9:
            facing = (1.0, 0.0)
        else:
            facing = (facing_x / facing_norm, facing_y / facing_norm)
        movement_x = _last_float_option(command, "--movement-x", 0.0)
        movement_y = _last_float_option(command, "--movement-y", 0.0)
        movement_norm = math.hypot(movement_x, movement_y)
        if movement_norm <= 1.0e-9:
            side_sign = 1.0 if skill.endswith("_left") else -1.0
            movement = (-facing[1] * side_sign, facing[0] * side_sign)
        else:
            movement = (movement_x / movement_norm, movement_y / movement_norm)
        heading_deg = math.degrees(math.atan2(facing[1], facing[0]))
        body_heading_deg = math.degrees(heading)
        distance = _last_float_option(command, "--distance-m", 0.0)
        speed = _last_float_option(command, "--speed-mps", 0.0)
        ramp = _last_float_option(command, "--transport-ramp-seconds", 0.0)
        mode = _last_option(command, "--mode", "skill_default")
        stand_seconds = _last_float_option(command, "--stand-seconds", 0.0)
        return (
            f"body_relative_lateral: body_heading={body_heading_deg:.1f}deg source={source} "
            f"planner_facing=({facing[0]:.3f},{facing[1]:.3f}) heading={heading_deg:.1f}deg "
            f"lateral_facing_mode={getattr(args, 'lateral_facing_mode', 'deploy_hold')}; "
            f"{skill} planner_movement=({movement[0]:.3f},{movement[1]:.3f}); "
            f"distance={distance:.2f}m speed={speed:.2f}m/s ramp={ramp:.2f}s "
            f"mode={mode} pre_stand={stand_seconds:.1f}s"
        )
    if skill == "carry_place_crouch":
        start_pulse_seconds = _last_float_option(command, "--start-pulse-seconds", 0.0)
        stand_seconds = _last_float_option(command, "--stand-seconds", 0.0)
        start_height = _last_float_option(command, "--place-start-height", 0.80)
        place_height = _last_float_option(command, "--place-height", 0.68)
        ramp_seconds = _last_float_option(command, "--place-ramp-seconds", 0.0)
        hold_seconds = _last_float_option(command, "--place-hold-seconds", 0.0)
        return (
            "place_crouch: both robots use one shared heartbeat height target; "
            f"start_pulse={start_pulse_seconds:.1f}s pre_stand={stand_seconds:.1f}s "
            f"height {start_height:.2f}->{place_height:.2f} "
            f"ramp={ramp_seconds:.1f}s hold={hold_seconds:.1f}s"
        )
    if skill == "carry_recover_stand":
        start_pulse_seconds = _last_float_option(command, "--start-pulse-seconds", 0.0)
        stand_seconds = _last_float_option(command, "--stand-seconds", 0.0)
        height = _last_float_option(command, "--height", 0.80)
        ramp_seconds = _last_float_option(command, "--place-ramp-seconds", 0.0)
        hold_seconds = _last_float_option(command, "--place-hold-seconds", 0.0)
        return (
            "recover_stand: releases carry_place_crouch lower-body assist and holds IDLE; "
            f"start_pulse={start_pulse_seconds:.1f}s pre_stand={stand_seconds:.1f}s "
            f"height={height:.2f} recover={ramp_seconds:.1f}s hold={hold_seconds:.1f}s"
        )
    if skill == "place_payload_on_arms":
        hold_seconds = _last_float_option(command, "--place-hold-seconds", 0.0)
        return (
            "place_payload_on_arms: wait for stable carry-ready arms, then place the deferred "
            f"long_box at the live four-palm support pose; hold={hold_seconds:.1f}s"
        )
    return ""


def _current_heading_rad(args: argparse.Namespace) -> float:
    return _current_heading_info(args)[0]


def _current_heading_info(args: argparse.Namespace) -> tuple[float, str]:
    if args.current_yaw_deg is not None:
        return math.radians(float(args.current_yaw_deg)), "manual_current_yaw_deg"
    live = _read_live_runtime_heading(args)
    if live is not None:
        return live
    return 0.0, "default_zero_no_live_runtime_heading"


def _read_live_runtime_heading(args: argparse.Namespace) -> tuple[float, str] | None:
    if getattr(args, "no_live_runtime_heading", False):
        return None
    path = getattr(args, "runtime_state_file", None)
    if path is None:
        return None
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return None
    max_age = max(0.0, float(getattr(args, "runtime_state_max_age_s", 1.0)))
    age = time.time() - stat.st_mtime
    if age > max_age:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    yaw = _robot_mean_yaw_from_runtime_payload(payload, tuple(getattr(args, "robots", ())))
    if yaw is None:
        try:
            yaw = float(payload["mean_heading_yaw_rad"])
        except (KeyError, TypeError, ValueError):
            return None
    return _wrap_angle(yaw), f"live_runtime_state:{path}"


def _robot_mean_yaw_from_runtime_payload(payload: dict, robot_ids: tuple[str, ...]) -> float | None:
    robots = payload.get("robots")
    if not isinstance(robots, dict):
        return None
    yaws = []
    for robot_id in robot_ids:
        robot_payload = robots.get(robot_id)
        if not isinstance(robot_payload, dict):
            continue
        try:
            yaws.append(float(robot_payload["yaw_rad"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not yaws:
        return None
    return math.atan2(
        sum(math.sin(yaw) for yaw in yaws) / len(yaws),
        sum(math.cos(yaw) for yaw in yaws) / len(yaws),
    )


def _update_heading_state_after_success(args: argparse.Namespace, skill: str, command: list[str]) -> None:
    underlying_skill = UNDERLYING_SKILL.get(skill, skill)
    if underlying_skill != "carry_turn_in_place":
        return
    yaw_deg = _last_float_option(command, "--yaw-deg", 0.0)
    scale = _last_float_option(command, "--turn-command-scale", 1.0)
    delta = math.radians(yaw_deg * scale)
    current = _read_heading_state(args.heading_state_file)
    _write_heading_state(args.heading_state_file, _wrap_angle(current + delta), skill)


def _read_heading_state(path: Path) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _wrap_angle(float(payload.get("heading_yaw_rad", 0.0)))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def _write_heading_state(path: Path, yaw_rad: float, source_skill: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaw = _wrap_angle(yaw_rad)
    path.write_text(
        json.dumps(
            {
                "heading_yaw_rad": yaw,
                "heading_yaw_deg": math.degrees(yaw),
                "source_skill": source_skill,
                "updated_unix": time.time(),
                "convention": "+yaw=counterclockwise, -yaw=clockwise, zero=world +X",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _last_float_option(values: list[str], name: str, default: float) -> float:
    result = default
    for index, value in enumerate(values[:-1]):
        if value == name:
            try:
                result = float(values[index + 1])
            except ValueError:
                result = default
    return result


def _last_option(values: list[str], name: str, default: str) -> str:
    result = default
    for index, value in enumerate(values[:-1]):
        if value == name:
            result = values[index + 1]
    return result


def _unit_xy(x: float, y: float) -> tuple[float, float]:
    norm = math.hypot(float(x), float(y))
    if norm <= 1.0e-9:
        return (1.0, 0.0)
    return (float(x) / norm, float(y) / norm)


def _wrap_angle(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi + 1.0e-12:
        return math.pi
    return wrapped


if __name__ == "__main__":
    main()
