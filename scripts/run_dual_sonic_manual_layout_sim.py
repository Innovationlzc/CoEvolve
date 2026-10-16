#!/usr/bin/env python3
"""Launch a two-G1 official SONIC MuJoCo scene from the user's manual layout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
GEAR_ROOT = REPO_ROOT / "third_party" / "GR00T-WholeBodyControl"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(GEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(GEAR_ROOT))

from praxis_coop.sonic_mujoco.manual_layout_scene import (  # noqa: E402
    DEFAULT_DUAL_NO_PAYLOAD_SCENE_REL_PATH,
    DEFAULT_DUAL_SCENE_REL_PATH,
    DEFAULT_PAYLOAD_INITIAL_STATE,
    DEFAULT_PAYLOAD_MASS_KG,
    DEFAULT_PAYLOAD_PLACEMENT,
    DEFAULT_PAYLOAD_REARWARD_OFFSET_M,
    DEFAULT_USDA_SCENE,
    generate_dual_manual_layout_scene,
)
from praxis_coop.sonic.upper_body import UPPER_BODY_POSES, resolve_upper_body_pose  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usda", type=Path, default=DEFAULT_USDA_SCENE)
    parser.add_argument(
        "--interface",
        default=os.environ.get("SONIC_DDS_INTERFACE", "auto"),
        help="DDS network interface. Use the same value for both deploy terminals.",
    )
    parser.add_argument(
        "--with-box",
        action="store_true",
        help="Include long_box. Leave off for the first dual no-box motor validation.",
    )
    parser.add_argument(
        "--box-mass-kg",
        type=float,
        default=DEFAULT_PAYLOAD_MASS_KG,
        help="Mass for the included long_box when --with-box is set.",
    )
    parser.add_argument(
        "--payload-hold",
        action="store_true",
        help="Keep long_box fixed at its initial pose until the configured release condition.",
    )
    parser.add_argument(
        "--payload-release-after-drop-s",
        type=float,
        default=None,
        help="Release --payload-hold this many seconds after ElasticBand drop.",
    )
    parser.add_argument(
        "--payload-release-at-s",
        type=float,
        default=None,
        help="Release --payload-hold at this MuJoCo sim time, independent of drop.",
    )
    parser.add_argument(
        "--box-placement",
        choices=("authored", "support_center", "robot_midpoint"),
        default=DEFAULT_PAYLOAD_PLACEMENT,
        help=(
            "Initial long_box pose when --with-box is set. 'support_center' keeps the "
            "authored support height/forward offset and centers the payload between the two G1s."
        ),
    )
    parser.add_argument(
        "--box-initial-state",
        choices=("deferred", "active"),
        default=DEFAULT_PAYLOAD_INITIAL_STATE,
        help=(
            "When --with-box is set, 'deferred' loads long_box hidden at a far staging pose "
            "until Term4 sends place_payload_on_arms. 'active' places it physically at startup."
        ),
    )
    parser.add_argument(
        "--box-center-z",
        type=float,
        default=None,
        help="Optional override for long_box COM z. Default keeps the authored Isaac layout z.",
    )
    parser.add_argument(
        "--box-forward-offset-m",
        type=float,
        default=None,
        help=(
            "Optional forward offset from the two-robot midpoint along robot heading. "
            "Default reuses the authored Isaac layout offset."
        ),
    )
    parser.add_argument(
        "--payload-max-support-surface-tilt-deg",
        type=float,
        default=8.0,
        help=(
            "Maximum live palm collision-surface tilt allowed before the one-shot payload "
            "placement event. Post-placement tilt is diagnostic in the task profile."
        ),
    )
    parser.add_argument(
        "--payload-rearward-offset-m",
        type=float,
        default=DEFAULT_PAYLOAD_REARWARD_OFFSET_M,
        help=(
            "Rearward offset from the four-palm live support center, measured in the "
            "mean robot heading frame. The verified default is 0.10m: it keeps the "
            "COM over the horizontal support plates with additional inward margin."
        ),
    )
    parser.add_argument("--no-onscreen", action="store_true")
    parser.add_argument(
        "--viewer-camera-preset",
        choices=("workcell", "global"),
        default="workcell",
        help=(
            "Use 'global' as the interactive viewer's initial view. The viewer remains "
            "mouse-adjustable; a separate fixed offscreen camera supplies VLM evidence."
        ),
    )
    parser.add_argument(
        "--vlm-global-frame-file",
        type=Path,
        default=None,
        help=(
            "Fixed offscreen global frame written by Term1. With the global preset and a "
            "runtime-state file, the default is vlm_global_frame.png beside that state file."
        ),
    )
    parser.add_argument(
        "--vlm-global-frame-interval-s",
        type=float,
        default=0.25,
        help="Wall-clock interval for the independent fixed VLM frame.",
    )
    parser.add_argument("--disable-elastic-band", action="store_true")
    parser.add_argument("--reset-on-fall", action="store_true")
    parser.add_argument(
        "--stop-on-fall",
        dest="stop_on_fall",
        action="store_true",
        help="Stop the runtime after the first detected fall (default).",
    )
    parser.add_argument(
        "--continue-after-fall",
        dest="stop_on_fall",
        action="store_false",
        help=(
            "Keep the simulator and checkpoint service alive after a fall so an "
            "external verifier can restore a trusted state. A fall remains a hard "
            "physical failure and is never accepted."
        ),
    )
    parser.set_defaults(stop_on_fall=True)
    parser.add_argument(
        "--lowcmd-watchdog-timeout-s",
        type=float,
        default=1.0,
        help="Suspend and fail the rollout if either deploy stops refreshing LowCmd for this long.",
    )
    parser.add_argument(
        "--no-stop-on-control-loss",
        dest="stop_on_control_loss",
        action="store_false",
        help="Suspend the robots but keep Term1 open after LowCmd loss for manual diagnosis.",
    )
    parser.set_defaults(stop_on_control_loss=True)
    parser.add_argument(
        "--no-require-lowcmd-before-drop",
        action="store_true",
        help="Allow pressing 9 before both deploy backends have sent body lowcmd. Not recommended.",
    )
    parser.add_argument(
        "--lowcmd-warmup-s",
        type=float,
        default=2.0,
        help="Required continuous time after each robot first receives body lowcmd before drop is allowed.",
    )
    parser.add_argument(
        "--auto-drop-when-ready",
        action="store_true",
        help="Automatically disable ElasticBand after both robots have body lowcmd for --lowcmd-warmup-s.",
    )
    parser.add_argument(
        "--min-drop-time-s",
        type=float,
        default=0.0,
        help="Earliest sim time at which manual or automatic drop is allowed.",
    )
    parser.add_argument(
        "--planner-heartbeat-file",
        type=Path,
        default=None,
        help="Require this file to be freshly touched by the planner publisher before drop is allowed.",
    )
    parser.add_argument(
        "--planner-heartbeat-max-age-s",
        type=float,
        default=2.0,
        help="Maximum wall-clock age for --planner-heartbeat-file before drop is blocked.",
    )
    parser.add_argument(
        "--planner-heartbeat-drop-phase-keyword",
        default="",
        help=(
            "If set, auto-drop is blocked until the planner heartbeat phase contains this keyword. "
            "The check also matches skill_node and sync_group. Use 'static_sonic_skill' for static skill tests."
        ),
    )
    parser.add_argument(
        "--upper-body-lock-pose",
        choices=sorted(("none", "custom", *UPPER_BODY_POSES)),
        default="none",
        help="Apply a symmetric motor-side arm pose lock in MuJoCo; leaves SONIC in charge of lower-body control.",
    )
    parser.add_argument(
        "--upper-body-lock-values",
        nargs=17,
        metavar="RAD",
        help="Custom 17-float motor-side arm pose lock target; requires --upper-body-lock-pose custom.",
    )
    parser.add_argument(
        "--upper-body-lock-kp",
        type=float,
        default=18.0,
        help="PD Kp for the motor-side arm pose lock.",
    )
    parser.add_argument(
        "--upper-body-lock-kd",
        type=float,
        default=1.2,
        help="PD Kd for the motor-side arm pose lock.",
    )
    parser.add_argument(
        "--upper-body-lock-ramp-s",
        type=float,
        default=8.0,
        help="Smooth ramp duration from the scene initial arm pose to the locked pose.",
    )
    parser.add_argument(
        "--upper-body-lock-start-after-drop-s",
        type=float,
        default=0.0,
        help="Wait this many seconds after pressing 9 before enabling the arm pose lock.",
    )
    parser.add_argument(
        "--upper-body-lock-before-drop",
        action="store_true",
        help="Apply the arm pose lock before ElasticBand is disabled. Off by default for stability.",
    )
    parser.add_argument(
        "--upper-body-lock-include-waist",
        action="store_true",
        help="Also lock waist joints. Off by default because waist motion helps SONIC balance.",
    )
    parser.add_argument(
        "--upper-body-lock-observed-by-policy",
        action="store_true",
        help=(
            "Expose the physical locked arm pose to the SONIC policy. By default the locked "
            "upper-body joints are hidden from policy observation so lower-body balance does "
            "not crouch in response to a motor-side carry pose lock."
        ),
    )
    parser.add_argument(
        "--upper-body-lock-world-level",
        action="store_true",
        help=(
            "Compensate torso pitch by adjusting shoulder pitch so the carry forearms stay "
            "closer to their world-frame level during squat/place skills."
        ),
    )
    parser.add_argument(
        "--upper-body-lock-level-gain",
        type=float,
        default=1.0,
        help="Gain for world-level shoulder-pitch compensation when --upper-body-lock-world-level is set.",
    )
    parser.add_argument(
        "--upper-body-lock-level-limit-rad",
        type=float,
        default=0.45,
        help="Absolute radian limit for world-level shoulder-pitch compensation.",
    )
    parser.add_argument(
        "--upper-body-lock-world-level-phase-keyword",
        default="carry_place_crouch",
        help=(
            "Only enable world-level compensation while the planner heartbeat phase contains this keyword. "
            "Use an empty string to enable whenever the lock is active."
        ),
    )
    parser.add_argument(
        "--upper-body-lock-palm-level",
        action="store_true",
        help="Use the live palm collision-surface normals to keep both wrist supports world-level.",
    )
    parser.add_argument(
        "--upper-body-lock-palm-level-gain",
        type=float,
        default=1.0,
        help="Wrist-pitch correction gain for --upper-body-lock-palm-level.",
    )
    parser.add_argument(
        "--upper-body-lock-palm-level-limit-rad",
        type=float,
        default=0.70,
        help="Maximum per-step palm-level wrist-pitch correction in radians.",
    )
    parser.add_argument(
        "--upper-body-lock-palm-level-phase-keyword",
        default="",
        help=(
            "Only enable live wrist pitch/roll palm leveling while the planner phase "
            "contains this keyword. Empty enables it for every phase."
        ),
    )
    parser.add_argument(
        "--upper-body-lock-palm-tilt-deg",
        type=float,
        default=None,
        help=(
            "Drive both live palm collision surfaces to this world-frame tilt in "
            "degrees. Use 60 for raised physical hand backstops; mutually exclusive "
            "with --upper-body-lock-palm-level."
        ),
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run N physics steps and exit; useful for headless mapping checks.",
    )
    parser.add_argument(
        "--max-sim-time-s",
        type=float,
        default=None,
        help="Stop automatically after this many MuJoCo seconds. Useful for no-onscreen regression runs.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Write a JSON runtime summary on exit.",
    )
    parser.add_argument(
        "--runtime-state-file",
        type=Path,
        default=None,
        help=(
            "Continuously write live MuJoCo root poses/yaws for Terminal 4 body-relative skills. "
            "Defaults to <planner-heartbeat-dir>/runtime_state.json when a heartbeat file is set."
        ),
    )
    parser.add_argument(
        "--runtime-state-write-interval-s",
        type=float,
        default=0.1,
        help="Minimum interval for --runtime-state-file updates.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Store full MuJoCo integration checkpoints here. Defaults to "
            "<runtime-state-dir>/checkpoints when live runtime state is enabled."
        ),
    )
    parser.add_argument(
        "--mujoco-solver",
        choices=("newton", "cg", "pgs"),
        default="newton",
        help=(
            "MuJoCo constraint solver. Keep newton for normal runs; use cg for "
            "contact-rich checkpoint recovery when sparse Hessian factorization "
            "becomes singular."
        ),
    )
    parser.add_argument(
        "--disable-place-crouch-lock",
        action="store_true",
        help="Disable the runtime-side stationary crouch assist for carry_place_crouch.",
    )
    parser.add_argument("--place-crouch-kp", type=float, default=40.0)
    parser.add_argument("--place-crouch-kd", type=float, default=7.0)
    parser.add_argument("--place-crouch-torque-blend", type=float, default=0.55)
    parser.add_argument("--place-crouch-torque-limit-fraction", type=float, default=0.45)
    args = parser.parse_args()
    if (
        args.upper_body_lock_palm_level
        and args.upper_body_lock_palm_tilt_deg is not None
    ):
        parser.error(
            "--upper-body-lock-palm-level and "
            "--upper-body-lock-palm-tilt-deg are mutually exclusive"
        )
    if (
        args.upper_body_lock_palm_tilt_deg is not None
        and not 0.0 <= args.upper_body_lock_palm_tilt_deg <= 80.0
    ):
        parser.error("--upper-body-lock-palm-tilt-deg must be in [0, 80]")

    # Load native simulation dependencies only after offline CLI validation.
    from praxis_coop.sonic_mujoco.dual_runtime import (
        DualG1SonicMujocoRuntime, DualRobotSpec, PayloadHold,
        PlaceCrouchMotorLock, UpperBodyPoseLock,
    )

    _require_multi_robot_dds_patch()

    interface = _resolve_interface(args.interface)
    scene_rel_path = DEFAULT_DUAL_SCENE_REL_PATH if args.with_box else DEFAULT_DUAL_NO_PAYLOAD_SCENE_REL_PATH
    generated = generate_dual_manual_layout_scene(
        source_usda=args.source_usda,
        gear_root=GEAR_ROOT,
        scene_rel_path=scene_rel_path,
        include_payload=args.with_box,
        payload_mass_kg=args.box_mass_kg,
        payload_placement=args.box_placement,
        payload_initial_state=args.box_initial_state,
        payload_center_z=args.box_center_z,
        payload_forward_offset_m=args.box_forward_offset_m,
    )
    config = _load_official_wbc_config()
    config["ROBOT_SCENE"] = str(generated.scene_rel_path)
    config["INTERFACE"] = interface
    config["ENABLE_ONSCREEN"] = not args.no_onscreen
    config["ENABLE_ELASTIC_BAND"] = not args.disable_elastic_band
    config["USE_JOYSTICK"] = 0

    print(f"[praxis-dual-sonic] source_usda={generated.source_usda}", flush=True)
    print(f"[praxis-dual-sonic] robot_scene={generated.scene_rel_path}", flush=True)
    if args.with_box:
        print(f"[praxis-dual-sonic] long_box_mass_kg={args.box_mass_kg:.3g}", flush=True)
        print(f"[praxis-dual-sonic] long_box_placement={generated.payload_placement}", flush=True)
        print(f"[praxis-dual-sonic] long_box_initial_state={generated.payload_initial_state}", flush=True)
        print(
            f"[praxis-dual-sonic] payload_rearward_offset_m={args.payload_rearward_offset_m:.3f}",
            flush=True,
        )
        if generated.payload_pose_xyzyaw is not None:
            x, y, z, yaw = generated.payload_pose_xyzyaw
            print(
                "[praxis-dual-sonic] long_box_pose_xyzyaw="
                f"({x:.3f}, {y:.3f}, {z:.3f}, {yaw:.3f})",
                flush=True,
            )
    print(f"[praxis-dual-sonic] using DDS interface: {interface}", flush=True)
    print("[praxis-dual-sonic] deploy terminals:", flush=True)
    print(
        "  front: scripts/run_official_sonic_deploy_zmq_manager.sh "
        f"{interface} --robot-id front --dds-topic-prefix front --zmq-port 5556 --zmq-out-port 5557",
        flush=True,
    )
    print(
        "  back : scripts/run_official_sonic_deploy_zmq_manager.sh "
        f"{interface} --robot-id back --dds-topic-prefix back --zmq-port 5558 --zmq-out-port 5559",
        flush=True,
    )
    print(
        "  sync : PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python "
        "scripts/run_dual_official_sonic_motor_preflight.py",
        flush=True,
    )
    if config.get("ENABLE_ELASTIC_BAND", False):
        print(
            "[praxis-dual-sonic] press 9 only after both deploys are in CONTROL "
            "and the sync/planner command publisher is already running",
            flush=True,
        )
    upper_body_lock = None
    upper_body_lock_target = resolve_upper_body_pose(args.upper_body_lock_pose, args.upper_body_lock_values)
    if upper_body_lock_target is not None:
        upper_body_lock = UpperBodyPoseLock(
            target=upper_body_lock_target,
            pose_name=args.upper_body_lock_pose,
            kp=args.upper_body_lock_kp,
            kd=args.upper_body_lock_kd,
            ramp_s=args.upper_body_lock_ramp_s,
            include_waist=args.upper_body_lock_include_waist,
            start_after_drop_s=args.upper_body_lock_start_after_drop_s,
            enabled_before_drop=args.upper_body_lock_before_drop,
            hide_from_policy=not args.upper_body_lock_observed_by_policy,
            world_level=args.upper_body_lock_world_level,
            level_gain=args.upper_body_lock_level_gain,
            level_limit_rad=args.upper_body_lock_level_limit_rad,
            world_level_phase_keyword=args.upper_body_lock_world_level_phase_keyword,
            palm_level=args.upper_body_lock_palm_level,
            palm_level_gain=args.upper_body_lock_palm_level_gain,
            palm_level_limit_rad=args.upper_body_lock_palm_level_limit_rad,
            palm_level_phase_keyword=args.upper_body_lock_palm_level_phase_keyword,
            palm_tilt_deg=args.upper_body_lock_palm_tilt_deg,
        )
        print(
            f"[praxis-dual-sonic] upper_body_lock_pose={args.upper_body_lock_pose} "
            f"kp={args.upper_body_lock_kp:.2f} kd={args.upper_body_lock_kd:.2f} "
            f"ramp_s={args.upper_body_lock_ramp_s:.2f} "
            f"start_after_drop_s={args.upper_body_lock_start_after_drop_s:.2f} "
            f"include_waist={args.upper_body_lock_include_waist} "
            f"before_drop={args.upper_body_lock_before_drop} "
            f"hide_from_policy={not args.upper_body_lock_observed_by_policy} "
            f"world_level={args.upper_body_lock_world_level} "
            f"palm_level={args.upper_body_lock_palm_level} "
            f"palm_tilt_deg={args.upper_body_lock_palm_tilt_deg} "
            f"palm_level_phase_keyword={args.upper_body_lock_palm_level_phase_keyword!r}",
            flush=True,
        )

    runtime_state_file = args.runtime_state_file
    if runtime_state_file is None and args.planner_heartbeat_file is not None:
        runtime_state_file = args.planner_heartbeat_file.parent / "runtime_state.json"
    if runtime_state_file is not None:
        print(f"[praxis-dual-sonic] runtime_state_file={runtime_state_file}", flush=True)
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and runtime_state_file is not None:
        checkpoint_dir = runtime_state_file.parent / "checkpoints"
    if checkpoint_dir is not None:
        print(f"[praxis-dual-sonic] checkpoint_dir={checkpoint_dir}", flush=True)
    print(f"[praxis-dual-sonic] mujoco_solver={args.mujoco_solver}", flush=True)

    if args.payload_hold and not args.with_box:
        raise ValueError("--payload-hold requires --with-box")
    payload_hold = None
    if args.payload_hold:
        payload_hold = PayloadHold(
            release_after_drop_s=args.payload_release_after_drop_s,
            release_at_sim_time_s=args.payload_release_at_s,
        )
        print(
            "[praxis-dual-sonic] payload_hold requested "
            f"release_after_drop_s={args.payload_release_after_drop_s} "
            f"release_at_sim_time_s={args.payload_release_at_s}",
            flush=True,
        )

    runtime = DualG1SonicMujocoRuntime(
        config=config,
        robot_specs=(
            DualRobotSpec("front", "front_", "front"),
            DualRobotSpec("back", "back_", "back"),
        ),
        onscreen=not args.no_onscreen,
        reset_on_fall=args.reset_on_fall,
        upper_body_pose_lock=upper_body_lock,
        require_lowcmd_before_drop=not args.no_require_lowcmd_before_drop,
        lowcmd_warmup_s=args.lowcmd_warmup_s,
        auto_drop_when_ready=args.auto_drop_when_ready,
        min_drop_time_s=args.min_drop_time_s,
        planner_heartbeat_file=args.planner_heartbeat_file,
        planner_heartbeat_max_age_s=args.planner_heartbeat_max_age_s,
        planner_heartbeat_drop_phase_keyword=args.planner_heartbeat_drop_phase_keyword,
        stop_on_fall=args.stop_on_fall,
        max_sim_time_s=args.max_sim_time_s,
        summary_file=args.summary_file,
        runtime_state_file=runtime_state_file,
        runtime_state_write_interval_s=args.runtime_state_write_interval_s,
        payload_hold=payload_hold,
        checkpoint_dir=checkpoint_dir,
        place_crouch_motor_lock=None
        if args.disable_place_crouch_lock
        else PlaceCrouchMotorLock(
            kp=args.place_crouch_kp,
            kd=args.place_crouch_kd,
            torque_blend=args.place_crouch_torque_blend,
            torque_limit_fraction=args.place_crouch_torque_limit_fraction,
        ),
        lowcmd_watchdog_timeout_s=args.lowcmd_watchdog_timeout_s,
        stop_on_control_loss=args.stop_on_control_loss,
        viewer_camera_preset=args.viewer_camera_preset,
        global_observation_frame_file=args.vlm_global_frame_file,
        global_observation_write_interval_s=args.vlm_global_frame_interval_s,
        payload_max_support_surface_tilt_deg=args.payload_max_support_surface_tilt_deg,
        payload_rearward_offset_m=args.payload_rearward_offset_m,
        mujoco_solver=args.mujoco_solver,
    )
    if args.smoke_steps > 0:
        for _ in range(args.smoke_steps):
            runtime.step()
        runtime.close()
        print(f"[praxis-dual-sonic] smoke_steps={args.smoke_steps} ok", flush=True)
        return
    runtime.start()


def _load_official_wbc_config() -> dict:
    # YAML is a simulation extra; inspecting CLI options must remain offline.
    import yaml

    config_path = (
        GEAR_ROOT
        / "gear_sonic"
        / "utils"
        / "mujoco_sim"
        / "wbc_configs"
        / "g1_29dof_sonic_model12.yaml"
    )
    with config_path.open() as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def _require_multi_robot_dds_patch() -> None:
    bridge_path = GEAR_ROOT / "gear_sonic" / "utils" / "mujoco_sim" / "unitree_sdk2py_bridge.py"
    try:
        bridge_source = bridge_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"[praxis-dual-sonic] missing SONIC bridge source: {bridge_path}") from exc
    required_markers = ("DDS_TOPIC_PREFIX", "def _topic", "rt/lowstate")
    if not all(marker in bridge_source for marker in required_markers):
        raise SystemExit(
            "[praxis-dual-sonic] ERROR: SONIC Python bridge does not support per-robot DDS topic prefixes.\n"
            "Apply the CoEvolve SONIC patches first:\n"
            f"  cd {GEAR_ROOT}\n"
            f"  git apply {REPO_ROOT / 'patches' / 'gear_sonic_multi_robot_dds_prefix.patch'}\n"
            f"  git apply {REPO_ROOT / 'patches' / 'gear_sonic_heading_increment.patch'}\n"
            "Then rebuild deploy:\n"
            "  cd gear_sonic_deploy && source scripts/setup_env.sh && just --justfile .justfile build"
        )


def _resolve_interface(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        output = subprocess.check_output(
            ["ip", "-o", "-4", "addr", "show", "up", "scope", "global"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return "lo"
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        iface = parts[1]
        if iface == "lo" or iface.startswith(("tailscale", "docker", "br-", "veth")) or iface == "Meta":
            continue
        if _interface_has_broadcast_multicast(iface):
            return iface
    return "lo"


def _interface_has_broadcast_multicast(iface: str) -> bool:
    try:
        output = subprocess.check_output(
            ["ip", "-o", "link", "show", iface],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "BROADCAST" in output and "MULTICAST" in output


if __name__ == "__main__":
    main()
