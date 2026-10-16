"""Generate SONIC/MuJoCo scenes from the user's hand-authored USD layout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
from math import atan2, cos, sin
from pathlib import Path
import xml.etree.ElementTree as ET

from praxis_coop.utils.usda_layout import UsdaPrim2D, parse_manual_layout_prims


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_USDA_SCENE = REPO_ROOT / "assets" / "manual_scenes" / "two_g1_box_push_layout.usda"
DEFAULT_GEAR_ROOT = Path(str((Path(__file__).resolve().parents[3] / 'third_party/GR00T-WholeBodyControl').resolve()))
DEFAULT_SCENE_REL_PATH = Path("gear_sonic/data/robot_model/model_data/g1/praxis_manual_layout_scene_43dof.xml")
DEFAULT_DUAL_SCENE_REL_PATH = Path(
    "gear_sonic/data/robot_model/model_data/g1/praxis_manual_layout_two_g1_scene_43dof.xml"
)
DEFAULT_DUAL_NO_PAYLOAD_SCENE_REL_PATH = Path(
    "gear_sonic/data/robot_model/model_data/g1/praxis_manual_layout_two_g1_no_payload_scene_43dof.xml"
)
DEFAULT_G1_SOURCE_XML = Path("gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml")
DEFAULT_FRONT_G1_XML = Path("gear_sonic/data/robot_model/model_data/g1/praxis_g1_front_29dof_with_hand.xml")
DEFAULT_BACK_G1_XML = Path("gear_sonic/data/robot_model/model_data/g1/praxis_g1_back_29dof_with_hand.xml")
DEFAULT_PAYLOAD_MASS_KG = 0.5
DEFAULT_PAYLOAD_PLACEMENT = "support_center"
DEFAULT_PAYLOAD_INITIAL_STATE = "deferred"
DEFAULT_PAYLOAD_STAGING_XYZ = (100.0, 100.0, 0.20)
DEFAULT_PAYLOAD_REARWARD_OFFSET_M = 0.10


def payload_rearward_offset_xy(
    robot_yaws: tuple[float, ...] | list[float],
    distance_m: float = DEFAULT_PAYLOAD_REARWARD_OFFSET_M,
) -> tuple[float, float]:
    """Express a rearward payload offset in the live mean robot frame."""

    if not robot_yaws:
        mean_yaw = 0.0
    else:
        mean_yaw = atan2(
            sum(sin(yaw) for yaw in robot_yaws) / len(robot_yaws),
            sum(cos(yaw) for yaw in robot_yaws) / len(robot_yaws),
        )
    distance_m = max(0.0, float(distance_m))
    return -distance_m * cos(mean_yaw), -distance_m * sin(mean_yaw)


def support_surface_pitch_correction(
    normal_xyz: tuple[float, float, float] | list[float],
    heading_yaw: float,
) -> float:
    """Return wrist-pitch correction that rotates a palm normal toward world +Z."""

    normal_x, normal_y, normal_z = (float(value) for value in normal_xyz)
    forward_component = normal_x * cos(heading_yaw) + normal_y * sin(heading_yaw)
    return atan2(-forward_component, normal_z)


def support_surface_roll_correction(
    normal_xyz: tuple[float, float, float] | list[float],
    heading_yaw: float,
) -> float:
    """Return wrist-roll correction that rotates a palm normal toward world +Z."""

    normal_x, normal_y, normal_z = (float(value) for value in normal_xyz)
    left_component = -normal_x * sin(heading_yaw) + normal_y * cos(heading_yaw)
    return atan2(left_component, normal_z)


def payload_runtime_geom_properties(active: bool) -> tuple[tuple[float, ...], int, int]:
    """Return visibility and always-live collision masks for deferred/active payloads."""

    rgba = (0.1, 0.45, 0.9, 1.0 if active else 0.0)
    return rgba, 1, 3


@dataclass(frozen=True)
class SonicMujocoSceneResult:
    source_usda: Path
    gear_root: Path
    scene_rel_path: Path
    scene_path: Path
    prim_count: int
    robot_start_note: str


@dataclass(frozen=True)
class NamespacedG1Xml:
    robot_id: str
    prefix: str
    source_path: Path
    output_rel_path: Path
    output_path: Path
    root_xy: tuple[float, float]
    root_yaw: float


@dataclass(frozen=True)
class SonicMujocoDualSceneResult:
    source_usda: Path
    gear_root: Path
    scene_rel_path: Path
    scene_path: Path
    robot_xmls: tuple[NamespacedG1Xml, NamespacedG1Xml]
    prim_count: int
    robot_start_note: str
    payload_mass_kg: float | None = None
    payload_placement: str = "omitted"
    payload_initial_state: str = "omitted"
    payload_pose_xyzyaw: tuple[float, float, float, float] | None = None


def generate_manual_layout_scene(
    *,
    source_usda: Path = DEFAULT_USDA_SCENE,
    gear_root: Path = DEFAULT_GEAR_ROOT,
    scene_rel_path: Path = DEFAULT_SCENE_REL_PATH,
    payload_mass_kg: float = DEFAULT_PAYLOAD_MASS_KG,
) -> SonicMujocoSceneResult:
    """Write a SONIC-compatible MJCF scene containing the manual layout.

    The output file is intentionally placed next to SONIC's stock G1 MJCF files
    so its relative include of ``g1_29dof_with_hand.xml`` and mesh paths stay
    identical to the official simulator setup.
    """

    prims = parse_manual_layout_prims(source_usda)
    if not prims:
        raise FileNotFoundError(f"Could not parse manual layout prims from {source_usda}")

    scene_path = gear_root / scene_rel_path
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(_build_scene_xml(prims, payload_mass_kg=payload_mass_kg), encoding="utf-8")
    return SonicMujocoSceneResult(
        source_usda=source_usda,
        gear_root=gear_root,
        scene_rel_path=scene_rel_path,
        scene_path=scene_path,
        prim_count=len(prims),
        robot_start_note="single official SONIC G1 starts at the stock MJCF pelvis pose near world origin",
    )


def generate_dual_manual_layout_scene(
    *,
    source_usda: Path = DEFAULT_USDA_SCENE,
    gear_root: Path = DEFAULT_GEAR_ROOT,
    scene_rel_path: Path = DEFAULT_DUAL_SCENE_REL_PATH,
    source_g1_rel_path: Path = DEFAULT_G1_SOURCE_XML,
    front_g1_rel_path: Path = DEFAULT_FRONT_G1_XML,
    back_g1_rel_path: Path = DEFAULT_BACK_G1_XML,
    use_official_standing_z: bool = True,
    include_payload: bool = True,
    payload_mass_kg: float = DEFAULT_PAYLOAD_MASS_KG,
    payload_placement: str = DEFAULT_PAYLOAD_PLACEMENT,
    payload_initial_state: str = DEFAULT_PAYLOAD_INITIAL_STATE,
    payload_center_z: float | None = None,
    payload_forward_offset_m: float | None = None,
) -> SonicMujocoDualSceneResult:
    """Write a dual-G1 SONIC-native MJCF scene for the manual layout.

    This scene is a compile-time bridge artifact. It proves that the user's
    manual layout can host two distinguishable G1 models in one MuJoCo model.
    Runtime control still needs a dual Unitree bridge because the upstream sim
    and deploy currently assume one set of ``rt/lowstate``/``rt/lowcmd`` topics.
    """

    prims = parse_manual_layout_prims(source_usda)
    if not prims:
        raise FileNotFoundError(f"Could not parse manual layout prims from {source_usda}")
    front_prim = _require_prim(prims, "g1")
    back_prim = _require_prim(prims, "g1_01")
    payload_prim = (
        _resolve_dual_payload_prim(
            prims,
            placement=payload_placement,
            center_z=payload_center_z,
            forward_offset_m=payload_forward_offset_m,
        )
        if include_payload
        else None
    )
    if payload_initial_state not in {"active", "deferred"}:
        raise ValueError(
            "payload_initial_state must be one of: active, deferred; "
            f"got {payload_initial_state!r}"
        )

    source_g1_path = gear_root / source_g1_rel_path
    front_xml = _write_namespaced_g1_xml(
        source_path=source_g1_path,
        output_path=gear_root / front_g1_rel_path,
        output_rel_path=front_g1_rel_path,
        robot_id="g1",
        prefix="front_",
        robot_prim=front_prim,
        use_official_standing_z=use_official_standing_z,
    )
    back_xml = _write_namespaced_g1_xml(
        source_path=source_g1_path,
        output_path=gear_root / back_g1_rel_path,
        output_rel_path=back_g1_rel_path,
        robot_id="g1_01",
        prefix="back_",
        robot_prim=back_prim,
        use_official_standing_z=use_official_standing_z,
    )

    scene_path = gear_root / scene_rel_path
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(
        _build_dual_scene_xml(
            prims,
            front_g1_rel_path.name,
            back_g1_rel_path.name,
            include_payload=include_payload,
            payload_mass_kg=payload_mass_kg,
            payload_prim=payload_prim,
            payload_active_at_start=payload_initial_state == "active",
        ),
        encoding="utf-8",
    )
    placement_note = ""
    if payload_prim is not None:
        placement_note = (
            f", payload_placement={payload_placement}, "
            f"payload_pose=({payload_prim.x:.3f}, {payload_prim.y:.3f}, "
            f"{payload_prim.z:.3f}, yaw={payload_prim.yaw:.3f})"
        )
    return SonicMujocoDualSceneResult(
        source_usda=source_usda,
        gear_root=gear_root,
        scene_rel_path=scene_rel_path,
        scene_path=scene_path,
        robot_xmls=(front_xml, back_xml),
        prim_count=len(prims),
        robot_start_note=(
            "two namespaced G1 MJCF copies start at the hand-authored USDA x/y/yaw; "
            "z uses the official standing pelvis height unless disabled; "
            f"payload={'included' if include_payload else 'omitted'}"
            + (f", payload_mass_kg={payload_mass_kg:.3g}" if include_payload else "")
            + (f", payload_initial_state={payload_initial_state}" if include_payload else "")
            + placement_note
        ),
        payload_mass_kg=payload_mass_kg if include_payload else None,
        payload_placement=payload_placement if include_payload else "omitted",
        payload_initial_state=payload_initial_state if include_payload else "omitted",
        payload_pose_xyzyaw=None
        if payload_prim is None
        else (payload_prim.x, payload_prim.y, payload_prim.z, payload_prim.yaw),
    )


def _build_scene_xml(prims: list[UsdaPrim2D], *, payload_mass_kg: float = DEFAULT_PAYLOAD_MASS_KG) -> str:
    walls = [prim for prim in prims if prim.kind == "wall"]
    long_box = _require_prim(prims, "long_box")
    target = _require_prim(prims, "target")

    wall_xml = "\n".join(_wall_geom_xml(prim) for prim in walls)
    box_xml = _free_box_xml(long_box, mass_kg=payload_mass_kg)
    target_xml = _target_xml(target)
    global_camera_xml = _global_topdown_camera_xml(prims)

    return f"""<mujoco model="praxis_manual_layout_g1_43dof_scene">
  <include file="g1_29dof_with_hand.xml"/>

  <statistic center="-0.2 4.8 0.5" extent="9.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-120" elevation="-25" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="12 12" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
{global_camera_xml}
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.02 0.001"/>
{wall_xml}
{box_xml}
{target_xml}
    <site name="manual_layout_origin" pos="0 0 0" size="0.05" rgba="1 1 1 1" type="sphere"/>
  </worldbody>

  <default>
    <geom friction="1.0 0.02 0.001" condim="3"/>
  </default>
</mujoco>
"""


def _build_dual_scene_xml(
    prims: list[UsdaPrim2D],
    front_file: str,
    back_file: str,
    *,
    include_payload: bool = True,
    payload_mass_kg: float = DEFAULT_PAYLOAD_MASS_KG,
    payload_prim: UsdaPrim2D | None = None,
    payload_active_at_start: bool = True,
) -> str:
    walls = [prim for prim in prims if prim.kind == "wall"]
    long_box = _require_prim(prims, "long_box")
    target = _require_prim(prims, "target")

    wall_xml = "\n".join(_wall_geom_xml(prim) for prim in walls)
    box_source = payload_prim or long_box
    box_xml = (
        _free_box_xml(box_source, mass_kg=payload_mass_kg, active=payload_active_at_start)
        if include_payload
        else ""
    )
    support_site_xml = _payload_support_site_xml(box_source) if include_payload else ""
    target_xml = _target_xml(target)
    global_camera_xml = _global_topdown_camera_xml(prims)

    return f"""<mujoco model="praxis_manual_layout_two_g1_43dof_scene">
  <include file="{escape(front_file)}"/>
  <include file="{escape(back_file)}"/>

  <statistic center="-0.2 4.8 0.5" extent="9.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-120" elevation="-25" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="manual_groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="manual_groundplane" texture="manual_groundplane" texuniform="true" texrepeat="12 12" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
{global_camera_xml}
    <geom name="manual_floor" size="0 0 0.05" type="plane" material="manual_groundplane" friction="1.0 0.02 0.001"/>
{wall_xml}
{box_xml}
{support_site_xml}
{target_xml}
    <site name="manual_layout_origin" pos="0 0 0" size="0.05" rgba="1 1 1 1" type="sphere"/>
  </worldbody>

  <default>
    <geom friction="1.0 0.02 0.001" condim="3"/>
  </default>
</mujoco>
"""


def _global_topdown_camera_xml(prims: list[UsdaPrim2D]) -> str:
    """Return a fixed camera whose image axes match the VLM frame calibration."""

    scene_prims = [prim for prim in prims if prim.kind in {"wall", "target"}]
    if not scene_prims:
        center_x, center_y = 0.5, 4.9
    else:
        x_bounds: list[float] = []
        y_bounds: list[float] = []
        for prim in scene_prims:
            half_x = 0.5 * (
                abs(cos(prim.yaw)) * abs(prim.sx)
                + abs(sin(prim.yaw)) * abs(prim.sy)
            )
            half_y = 0.5 * (
                abs(sin(prim.yaw)) * abs(prim.sx)
                + abs(cos(prim.yaw)) * abs(prim.sy)
            )
            x_bounds.extend((prim.x - half_x, prim.x + half_x))
            y_bounds.extend((prim.y - half_y, prim.y + half_y))
        center_x = 0.5 * (min(x_bounds) + max(x_bounds))
        center_y = 0.5 * (min(y_bounds) + max(y_bounds))
    return (
        f'    <camera name="vlm_global_topdown" pos="{center_x:.6g} {center_y:.6g} 13" '
        'xyaxes="-1 0 0 0 -1 0" fovy="45"/>'
    )


def _write_namespaced_g1_xml(
    *,
    source_path: Path,
    output_path: Path,
    output_rel_path: Path,
    robot_id: str,
    prefix: str,
    robot_prim: UsdaPrim2D,
    use_official_standing_z: bool,
) -> NamespacedG1Xml:
    tree = ET.parse(source_path)
    root = tree.getroot()
    root.set("model", f"{prefix}g1_29dof_with_hand")

    pelvis = _find_named(root, "body", "pelvis")
    if pelvis is None:
        raise ValueError(f"Could not find root pelvis body in {source_path}")
    official_z = _numbers_from_xml_attr(pelvis.get("pos", "0 0 0"))[2]
    pelvis.set("pos", _vec3(robot_prim.x, robot_prim.y, official_z if use_official_standing_z else robot_prim.z))
    pelvis.set("quat", _yaw_quat(robot_prim.yaw))

    _prefix_xml_names_and_refs(root, prefix)
    _add_payload_support_collision_geoms(root, prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(output_path, encoding="unicode", xml_declaration=False)
    return NamespacedG1Xml(
        robot_id=robot_id,
        prefix=prefix,
        source_path=source_path,
        output_rel_path=output_rel_path,
        output_path=output_path,
        root_xy=(robot_prim.x, robot_prim.y),
        root_yaw=robot_prim.yaw,
    )


def _prefix_xml_names_and_refs(root: ET.Element, prefix: str) -> None:
    name_ref_attrs = {
        "body",
        "class",
        "childclass",
        "geom",
        "geom1",
        "geom2",
        "joint",
        "joint1",
        "joint2",
        "mesh",
        "objname",
        "site",
        "tendon",
    }
    for elem in root.iter():
        if "name" in elem.attrib:
            elem.set("name", _prefixed(prefix, elem.attrib["name"]))
        for attr in name_ref_attrs:
            if attr in elem.attrib:
                elem.set(attr, _prefixed(prefix, elem.attrib[attr]))


def _prefixed(prefix: str, value: str) -> str:
    if not value or value.startswith(prefix):
        return value
    return f"{prefix}{value}"


def _add_payload_support_collision_geoms(root: ET.Element, prefix: str) -> None:
    """Add palm and forearm collision approximations for payload contact.

    The upstream palm mesh is sloped and irregular. A thin collision pad is
    aligned with its visible upper surface, as is customary for simulation
    collision meshes. It follows the SONIC-controlled wrist rigidly, but does
    not form an invisible tray or constrain the payload pose. A second pad
    follows the visible forearm from elbow to wrist so the physical support
    model matches the intended palm-and-forearm carry posture.
    """

    side_configs = {
        "left": {
            "quat": "0.99356511 -0.06945599 -0.05921871 0.06706255",
            "plate_pos": "0.08058909 0.00765705 0.03615214",
            # Calibrated in the settled elbow60 level-forearm pose. This spans
            # the real elbow-to-wrist link and meets the palm pad at the wrist.
            "forearm_quat": "0.98984472 -0.05632365 0.12987652 0.01292910",
            "forearm_pos": "0.08845502 0.00257667 -0.00923497",
        },
        "right": {
            "quat": "0.99684220 0.04106122 -0.05108420 -0.04483317",
            "plate_pos": "0.08118773 -0.00614675 0.03632788",
            "forearm_quat": "0.99108244 0.03514656 0.12819469 -0.00929697",
            "forearm_pos": "0.08899118 -0.00188759 -0.00831201",
        },
    }
    for side, config in side_configs.items():
        palm_body = _find_named(root, "body", f"{prefix}{side}_wrist_yaw_link")
        if palm_body is None:
            raise ValueError(f"Could not find {prefix}{side}_wrist_yaw_link for payload support geom")
        ET.SubElement(
            palm_body,
            "geom",
            {
                "name": f"{prefix}{side}_carry_support_plate",
                "type": "box",
                # In the elbow110 carry pose this sits just under the box
                # bottom and is close to world-level rather than wrist-tilted.
                "pos": config["plate_pos"],
                "quat": config["quat"],
                "size": "0.06 0.035 0.006",
                # Collision-only approximation coincident with the visible
                # palm. Alpha zero avoids drawing duplicate hand geometry.
                "rgba": "0 0 0 0",
                "friction": "3.0 0.05 0.002",
                # Dedicated payload-support bit. These collision proxies must
                # not collide with the robot itself while the arms are down.
                "contype": "2",
                "conaffinity": "0",
            },
        )
        forearm_body = _find_named(root, "body", f"{prefix}{side}_elbow_link")
        if forearm_body is None:
            raise ValueError(f"Could not find {prefix}{side}_elbow_link for payload support geom")
        ET.SubElement(
            forearm_body,
            "geom",
            {
                "name": f"{prefix}{side}_carry_forearm_support_plate",
                "type": "box",
                "pos": config["forearm_pos"],
                "quat": config["forearm_quat"],
                # The 0.22 m surface spans the physical forearm. It overlaps
                # the palm proxy slightly so a free box cannot fall through an
                # artificial collision gap at the wrist.
                "size": "0.11 0.032 0.006",
                "rgba": "0 0 0 0",
                "friction": "3.0 0.05 0.002",
                "contype": "2",
                "conaffinity": "0",
            },
        )


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element | None:
    for elem in root.iter(tag):
        if elem.get("name") == name:
            return elem
    return None


def _numbers_from_xml_attr(payload: str) -> tuple[float, float, float]:
    values = [float(item) for item in payload.replace(",", " ").split()]
    if len(values) < 3:
        raise ValueError(f"Expected at least three numeric values, got: {payload}")
    return values[0], values[1], values[2]


def _resolve_dual_payload_prim(
    prims: list[UsdaPrim2D],
    *,
    placement: str,
    center_z: float | None,
    forward_offset_m: float | None,
) -> UsdaPrim2D:
    front = _require_prim(prims, "g1")
    back = _require_prim(prims, "g1_01")
    box = _require_prim(prims, "long_box")
    if placement == "authored":
        return replace(box, z=box.z if center_z is None else float(center_z))

    midpoint_x = 0.5 * (front.x + back.x)
    midpoint_y = 0.5 * (front.y + back.y)
    heading = _mean_yaw((front.yaw, back.yaw))
    forward = (cos(heading), sin(heading))

    if placement == "robot_midpoint":
        offset = 0.0 if forward_offset_m is None else float(forward_offset_m)
    elif placement == "support_center":
        if forward_offset_m is None:
            authored_delta = (box.x - midpoint_x, box.y - midpoint_y)
            offset = authored_delta[0] * forward[0] + authored_delta[1] * forward[1]
        else:
            offset = float(forward_offset_m)
    else:
        raise ValueError(
            "payload_placement must be one of: authored, support_center, robot_midpoint; "
            f"got {placement!r}"
        )

    return replace(
        box,
        x=midpoint_x + offset * forward[0],
        y=midpoint_y + offset * forward[1],
        z=box.z if center_z is None else float(center_z),
    )


def _mean_yaw(yaws: tuple[float, ...]) -> float:
    return atan2(sum(sin(yaw) for yaw in yaws), sum(cos(yaw) for yaw in yaws))


def _wall_geom_xml(prim: UsdaPrim2D) -> str:
    size = _half_extents(prim)
    return (
        f'    <geom name="{escape(prim.name)}" type="box" pos="{_vec3(prim.x, prim.y, prim.z)}" '
        f'quat="{_yaw_quat(prim.yaw)}" size="{_vec3(*size)}" rgba="0.42 0.44 0.46 1" '
        'friction="1.0 0.02 0.001"/>\n'
    )


def _free_box_xml(prim: UsdaPrim2D, *, mass_kg: float = DEFAULT_PAYLOAD_MASS_KG, active: bool = True) -> str:
    if mass_kg <= 0.0:
        raise ValueError(f"payload mass must be positive, got {mass_kg}")
    size = _half_extents(prim)
    x, y, z = (prim.x, prim.y, prim.z) if active else DEFAULT_PAYLOAD_STAGING_XYZ
    rgba = "0.1 0.45 0.9 1" if active else "0.1 0.45 0.9 0"
    # Keep collision masks compiled in even for deferred payloads. MuJoCo does
    # not reliably rebuild contact candidates when a geom is compiled with
    # contype/conaffinity=0 and those masks are flipped at runtime.
    # Bit 1 keeps normal world/robot contact; bit 2 accepts contact from the
    # wrist support proxies without making those proxies self-collide.
    contact = 'contype="1" conaffinity="3"'
    return f"""    <body name="long_box" pos="{_vec3(x, y, z)}" quat="{_yaw_quat(prim.yaw)}">
      <freejoint name="long_box_freejoint"/>
      <geom name="long_box_geom" type="box" size="{_vec3(*size)}" mass="{mass_kg:.9g}"
        rgba="{rgba}" friction="2.0 0.05 0.002" {contact}/>
    </body>"""


def _payload_support_site_xml(prim: UsdaPrim2D) -> str:
    return (
        f'    <site name="long_box_support_pose" pos="{_vec3(prim.x, prim.y, prim.z)}" '
        f'quat="{_yaw_quat(prim.yaw)}" type="box" size="0.04 0.04 0.04" '
        'rgba="0.1 0.45 0.9 0"/>\n'
    )


def _target_xml(prim: UsdaPrim2D) -> str:
    radius = max(0.06, 0.5 * min(abs(prim.sx), abs(prim.sy)))
    height = max(0.02, abs(prim.sz))
    return (
        f'    <geom name="target_marker" type="cylinder" pos="{_vec3(prim.x, prim.y, prim.z)}" '
        f'size="{radius:.6g} {height / 2.0:.6g}" rgba="0.95 0.18 0.10 0.55" '
        'friction="1.4 0.03 0.001" contype="1" conaffinity="1"/>'
    )


def _require_prim(prims: list[UsdaPrim2D], name: str) -> UsdaPrim2D:
    for prim in prims:
        if prim.name == name:
            return prim
    raise ValueError(f"Missing required prim in manual layout: {name}")


def _half_extents(prim: UsdaPrim2D) -> tuple[float, float, float]:
    return (abs(prim.sx) / 2.0, abs(prim.sy) / 2.0, abs(prim.sz) / 2.0)


def _vec3(x: float, y: float, z: float) -> str:
    return f"{x:.9g} {y:.9g} {z:.9g}"


def _yaw_quat(yaw: float) -> str:
    return f"{cos(yaw / 2.0):.9g} 0 0 {sin(yaw / 2.0):.9g}"
