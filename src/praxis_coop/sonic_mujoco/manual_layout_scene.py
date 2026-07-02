"""Generate SONIC/MuJoCo scenes from the user's hand-authored USD layout."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import cos, sin
from pathlib import Path
import xml.etree.ElementTree as ET

from praxis_coop.utils.usda_layout import UsdaPrim2D, parse_manual_layout_prims


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_USDA_SCENE = REPO_ROOT / "assets" / "manual_scenes" / "two_g1_box_push_layout.usda"
DEFAULT_GEAR_ROOT = Path("/home/admin1/Project/CoEvolve/third_party/GR00T-WholeBodyControl")
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


def generate_manual_layout_scene(
    *,
    source_usda: Path = DEFAULT_USDA_SCENE,
    gear_root: Path = DEFAULT_GEAR_ROOT,
    scene_rel_path: Path = DEFAULT_SCENE_REL_PATH,
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
    scene_path.write_text(_build_scene_xml(prims), encoding="utf-8")
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
        ),
        encoding="utf-8",
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
        ),
    )


def _build_scene_xml(prims: list[UsdaPrim2D]) -> str:
    walls = [prim for prim in prims if prim.kind == "wall"]
    long_box = _require_prim(prims, "long_box")
    target = _require_prim(prims, "target")

    wall_xml = "\n".join(_wall_geom_xml(prim) for prim in walls)
    box_xml = _free_box_xml(long_box)
    target_xml = _target_xml(target)

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
) -> str:
    walls = [prim for prim in prims if prim.kind == "wall"]
    long_box = _require_prim(prims, "long_box")
    target = _require_prim(prims, "target")

    wall_xml = "\n".join(_wall_geom_xml(prim) for prim in walls)
    box_xml = _free_box_xml(long_box) if include_payload else ""
    target_xml = _target_xml(target)

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
    <geom name="manual_floor" size="0 0 0.05" type="plane" material="manual_groundplane" friction="1.0 0.02 0.001"/>
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


def _wall_geom_xml(prim: UsdaPrim2D) -> str:
    size = _half_extents(prim)
    return (
        f'    <geom name="{escape(prim.name)}" type="box" pos="{_vec3(prim.x, prim.y, prim.z)}" '
        f'quat="{_yaw_quat(prim.yaw)}" size="{_vec3(*size)}" rgba="0.42 0.44 0.46 1" '
        'friction="1.0 0.02 0.001"/>\n'
    )


def _free_box_xml(prim: UsdaPrim2D) -> str:
    size = _half_extents(prim)
    return f"""    <body name="long_box" pos="{_vec3(prim.x, prim.y, prim.z)}" quat="{_yaw_quat(prim.yaw)}">
      <freejoint name="long_box_freejoint"/>
      <geom name="long_box_geom" type="box" size="{_vec3(*size)}" mass="10.0"
        rgba="0.1 0.45 0.9 1" friction="1.2 0.03 0.001"/>
    </body>"""


def _target_xml(prim: UsdaPrim2D) -> str:
    radius = max(0.06, 0.5 * min(abs(prim.sx), abs(prim.sy)))
    height = max(0.02, abs(prim.sz))
    return (
        f'    <geom name="target_marker" type="cylinder" pos="{_vec3(prim.x, prim.y, prim.z)}" '
        f'size="{radius:.6g} {height / 2.0:.6g}" rgba="0.95 0.18 0.10 0.55" '
        'contype="0" conaffinity="0"/>'
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
