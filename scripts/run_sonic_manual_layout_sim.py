#!/usr/bin/env python3
"""Launch the official SONIC MuJoCo sim with the generated manual layout scene."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
GEAR_ROOT = REPO_ROOT / "third_party" / "GR00T-WholeBodyControl"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(GEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(GEAR_ROOT))

from gear_sonic.scripts.run_sim_loop import SimWrapper  # noqa: E402
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model  # noqa: E402
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig, override_wbc_config  # noqa: E402
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory  # noqa: E402
from praxis_coop.sonic_mujoco.manual_layout_scene import (  # noqa: E402
    DEFAULT_SCENE_REL_PATH,
    generate_manual_layout_scene,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-rel-path", type=Path, default=DEFAULT_SCENE_REL_PATH)
    parser.add_argument(
        "--interface",
        default=os.environ.get("SONIC_DDS_INTERFACE", "auto"),
        help="DDS network interface for Unitree SDK channels. Use the same value for deploy.",
    )
    parser.add_argument("--no-onscreen", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument(
        "--disable-elastic-band",
        action="store_true",
        help="Start MuJoCo without the virtual support band. Use only when deploy is already ready.",
    )
    parser.add_argument(
        "--auto-drop-after",
        type=float,
        default=None,
        help="Automatically disable the virtual support band after N seconds, equivalent to pressing 9.",
    )
    args = parser.parse_args()

    generated = generate_manual_layout_scene(gear_root=GEAR_ROOT, scene_rel_path=args.scene_rel_path)
    config = _load_official_wbc_config()
    interface = _resolve_interface(args.interface)
    sim_cfg = SimLoopConfig(
        interface=interface,
        enable_onscreen=not args.no_onscreen,
        enable_offscreen=args.offscreen,
        body_control_device="dummy",
        hand_control_device="dummy",
        enable_real_device=False,
        data_collection=False,
        initial_mode="idle",
        verbose=False,
    )
    config = override_wbc_config(config, sim_cfg)
    config["ROBOT_SCENE"] = str(generated.scene_rel_path)
    config["ENV_NAME"] = sim_cfg.env_name
    config["INTERFACE"] = interface
    if args.disable_elastic_band:
        config["ENABLE_ELASTIC_BAND"] = False

    print(f"[praxis-sonic] launching official SONIC sim with ROBOT_SCENE={config['ROBOT_SCENE']}", flush=True)
    print(f"[praxis-sonic] using DDS interface: {interface}", flush=True)
    if config.get("ENABLE_ELASTIC_BAND", False):
        print(
            "[praxis-sonic] ElasticBand is ON initially. Press 9 in the MuJoCo window to drop the robot, "
            "or use --auto-drop-after N.",
            flush=True,
        )
    print(
        "[praxis-sonic] start deploy in another terminal with: "
        f"scripts/run_official_sonic_deploy_zmq_manager.sh {interface}",
        flush=True,
    )

    robot_model = instantiate_g1_robot_model()
    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=sim_cfg.env_name,
        config=config,
        onscreen=config.get("ENABLE_ONSCREEN", True),
        offscreen=config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=False,
    )
    if args.auto_drop_after is not None:
        _schedule_elastic_band_drop(sim_wrapper.sim, args.auto_drop_after)
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=False,
        mp_start_method=sim_cfg.mp_start_method,
        camera_port=args.camera_port,
    )


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


def _schedule_elastic_band_drop(simulator, delay_s: float) -> None:
    if delay_s < 0.0:
        raise ValueError("--auto-drop-after must be non-negative")

    def worker() -> None:
        time.sleep(delay_s)
        sim_env = getattr(simulator, "sim_env", None)
        elastic_band = getattr(sim_env, "elastic_band", None)
        if elastic_band is None:
            print("[praxis-sonic] auto-drop skipped: ElasticBand is not configured", flush=True)
            return
        elastic_band.enable = False
        print(f"[praxis-sonic] auto-drop: ElasticBand disabled after {delay_s:.1f}s", flush=True)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


if __name__ == "__main__":
    main()
