#!/usr/bin/env python3
"""Launch a two-G1 official SONIC MuJoCo scene from the user's manual layout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
GEAR_ROOT = REPO_ROOT / "third_party" / "GR00T-WholeBodyControl"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(GEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(GEAR_ROOT))

from praxis_coop.sonic_mujoco.dual_runtime import (  # noqa: E402
    DualG1SonicMujocoRuntime,
    DualRobotSpec,
    UpperBodyPoseLock,
)
from praxis_coop.sonic_mujoco.manual_layout_scene import (  # noqa: E402
    DEFAULT_DUAL_NO_PAYLOAD_SCENE_REL_PATH,
    DEFAULT_DUAL_SCENE_REL_PATH,
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
    parser.add_argument("--no-onscreen", action="store_true")
    parser.add_argument("--disable-elastic-band", action="store_true")
    parser.add_argument("--reset-on-fall", action="store_true")
    parser.add_argument(
        "--stop-on-fall",
        action="store_true",
        help="Stop the runtime after the first detected fall. Useful for automated self-tests.",
    )
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
        "--upper-body-lock-pose",
        choices=sorted(("none", *UPPER_BODY_POSES)),
        default="none",
        help="Apply a symmetric motor-side arm pose lock in MuJoCo; leaves SONIC in charge of lower-body control.",
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
        "--smoke-steps",
        type=int,
        default=0,
        help="Run N physics steps and exit; useful for headless mapping checks.",
    )
    args = parser.parse_args()

    interface = _resolve_interface(args.interface)
    scene_rel_path = DEFAULT_DUAL_SCENE_REL_PATH if args.with_box else DEFAULT_DUAL_NO_PAYLOAD_SCENE_REL_PATH
    generated = generate_dual_manual_layout_scene(
        source_usda=args.source_usda,
        gear_root=GEAR_ROOT,
        scene_rel_path=scene_rel_path,
        include_payload=args.with_box,
    )
    config = _load_official_wbc_config()
    config["ROBOT_SCENE"] = str(generated.scene_rel_path)
    config["INTERFACE"] = interface
    config["ENABLE_ONSCREEN"] = not args.no_onscreen
    config["ENABLE_ELASTIC_BAND"] = not args.disable_elastic_band
    config["USE_JOYSTICK"] = 0

    print(f"[praxis-dual-sonic] source_usda={generated.source_usda}", flush=True)
    print(f"[praxis-dual-sonic] robot_scene={generated.scene_rel_path}", flush=True)
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
    upper_body_lock_target = resolve_upper_body_pose(args.upper_body_lock_pose)
    if upper_body_lock_target is not None:
        upper_body_lock = UpperBodyPoseLock(
            target=upper_body_lock_target,
            kp=args.upper_body_lock_kp,
            kd=args.upper_body_lock_kd,
            ramp_s=args.upper_body_lock_ramp_s,
            include_waist=args.upper_body_lock_include_waist,
            start_after_drop_s=args.upper_body_lock_start_after_drop_s,
            enabled_before_drop=args.upper_body_lock_before_drop,
        )
        print(
            f"[praxis-dual-sonic] upper_body_lock_pose={args.upper_body_lock_pose} "
            f"kp={args.upper_body_lock_kp:.2f} kd={args.upper_body_lock_kd:.2f} "
            f"ramp_s={args.upper_body_lock_ramp_s:.2f} "
            f"start_after_drop_s={args.upper_body_lock_start_after_drop_s:.2f} "
            f"include_waist={args.upper_body_lock_include_waist} "
            f"before_drop={args.upper_body_lock_before_drop}",
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
        stop_on_fall=args.stop_on_fall,
    )
    if args.smoke_steps > 0:
        for _ in range(args.smoke_steps):
            runtime.step()
        runtime.close()
        print(f"[praxis-dual-sonic] smoke_steps={args.smoke_steps} ok", flush=True)
        return
    runtime.start()


def _load_official_wbc_config() -> dict:
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
