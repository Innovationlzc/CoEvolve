#!/usr/bin/env python3
"""Build a SONIC/MuJoCo XML scene from the manual two-G1 USD layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.sonic_mujoco.manual_layout_scene import (  # noqa: E402
    DEFAULT_GEAR_ROOT,
    DEFAULT_DUAL_NO_PAYLOAD_SCENE_REL_PATH,
    DEFAULT_DUAL_SCENE_REL_PATH,
    DEFAULT_SCENE_REL_PATH,
    DEFAULT_USDA_SCENE,
    generate_dual_manual_layout_scene,
    generate_manual_layout_scene,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usda", type=Path, default=DEFAULT_USDA_SCENE)
    parser.add_argument("--gear-root", type=Path, default=DEFAULT_GEAR_ROOT)
    parser.add_argument("--scene-rel-path", type=Path, default=DEFAULT_SCENE_REL_PATH)
    parser.add_argument(
        "--dual-g1",
        action="store_true",
        help="Generate the namespaced two-G1 scene instead of the single-G1 motor preflight scene.",
    )
    parser.add_argument(
        "--use-manual-robot-z",
        action="store_true",
        help="Use USDA robot z instead of the official stable standing pelvis z in the dual-G1 scene.",
    )
    parser.add_argument(
        "--no-payload",
        action="store_true",
        help="Omit long_box from the dual-G1 scene for motor-only synchronization validation.",
    )
    parser.add_argument(
        "--box-mass-kg",
        type=float,
        default=5.0,
        help="Mass for long_box when payload is included.",
    )
    parser.add_argument(
        "--compile-check",
        action="store_true",
        help="After writing the scene, compile it with mujoco.MjModel.from_xml_path.",
    )
    args = parser.parse_args()

    if args.dual_g1:
        scene_rel_path = args.scene_rel_path
        if scene_rel_path == DEFAULT_SCENE_REL_PATH:
            scene_rel_path = (
                DEFAULT_DUAL_NO_PAYLOAD_SCENE_REL_PATH
                if args.no_payload
                else DEFAULT_DUAL_SCENE_REL_PATH
            )
        result = generate_dual_manual_layout_scene(
            source_usda=args.source_usda,
            gear_root=args.gear_root,
            scene_rel_path=scene_rel_path,
            use_official_standing_z=not args.use_manual_robot_z,
            include_payload=not args.no_payload,
            payload_mass_kg=args.box_mass_kg,
        )
        print(f"[praxis-sonic] source_usda={result.source_usda}")
        print(f"[praxis-sonic] wrote scene={result.scene_path}")
        print(f"[praxis-sonic] robot_scene={result.scene_rel_path}")
        for robot_xml in result.robot_xmls:
            print(
                "[praxis-sonic] "
                f"robot_id={robot_xml.robot_id} prefix={robot_xml.prefix} "
                f"xml={robot_xml.output_path} root_xy={robot_xml.root_xy}"
            )
        print(f"[praxis-sonic] prim_count={result.prim_count}")
        print(f"[praxis-sonic] note={result.robot_start_note}")
        if not args.no_payload:
            print(f"[praxis-sonic] long_box_mass_kg={args.box_mass_kg:.3g}")
        if args.compile_check:
            _compile_check(result.scene_path)
        return

    result = generate_manual_layout_scene(
        source_usda=args.source_usda,
        gear_root=args.gear_root,
        scene_rel_path=args.scene_rel_path,
        payload_mass_kg=args.box_mass_kg,
    )
    print(f"[praxis-sonic] source_usda={result.source_usda}")
    print(f"[praxis-sonic] wrote scene={result.scene_path}")
    print(f"[praxis-sonic] robot_scene={result.scene_rel_path}")
    print(f"[praxis-sonic] prim_count={result.prim_count}")
    print(f"[praxis-sonic] note={result.robot_start_note}")
    print(f"[praxis-sonic] long_box_mass_kg={args.box_mass_kg:.3g}")
    if args.compile_check:
        _compile_check(result.scene_path)


def _compile_check(scene_path: Path) -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "mujoco is required for --compile-check. Run this script with "
            "third_party/GR00T-WholeBodyControl/.venv_sim/bin/python."
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    print(
        "[praxis-sonic] compile_check=ok "
        f"nbody={model.nbody} njnt={model.njnt} nu={model.nu} ngeom={model.ngeom}"
    )


if __name__ == "__main__":
    main()
