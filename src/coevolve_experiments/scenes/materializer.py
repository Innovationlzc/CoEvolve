"""Deterministic, hash-bound materialization of candidate USDA scenes.

This module deliberately stops before simulator validation.  It turns one
already validated :class:`SceneSpec` into candidate USDA bytes and a stable
receipt; the resulting files are not paper evidence until the independent
execution preflight has collected its simulator receipts.

The source stage is repository-relative, SHA-256 pinned, and never followed
through a symlink.  Outputs must also remain below the supplied repository
root.  Existing, different bytes are refused instead of being silently
overwritten so that re-running the generator cannot erase an audited artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fnmatch import fnmatchcase
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Iterable, Mapping, Sequence

from .integrity import validate_scene
from .loader import compute_asset_composite_hash, compute_scene_hash
from .schema import RealizationParameter, SceneAsset, SceneSpec


MATERIALIZATION_RECEIPT_KIND = "coevolve_scene_materialization"
MATERIALIZATION_RECEIPT_SCHEMA_VERSION = 1
MATERIALIZER_ID = "coevolve.scenes.deterministic-usda"
MATERIALIZER_VERSION = "1.0.0"

_WORLD_DECLARATION = re.compile(r'(?m)^def\s+Xform\s+"World"\s*$')
_DIRECT_PRIM_DECLARATION = re.compile(
    r'(?m)^    def(?:\s+(?P<type>[A-Za-z_][A-Za-z0-9_]*))?'
    r'\s+"(?P<name>[^"\r\n]+)"'
)
_DIRECT_TRANSLATE = re.compile(
    r"(?m)^        (?:double3|float3) xformOp:translate = "
    r"\((?P<value>[^\r\n()]*)\)$"
)
_DIRECT_SCALE = re.compile(
    r"(?m)^        (?:double3|float3) xformOp:scale = "
    r"\((?P<value>[^\r\n()]*)\)$"
)
_DIRECT_ORIENT = re.compile(
    r"(?m)^        quat[fd] xformOp:orient = "
    r"\((?P<value>[^\r\n()]*)\)$"
)


class SceneMaterializationError(ValueError):
    """Raised when candidate generation cannot preserve its audit contract."""


@dataclass(frozen=True)
class MaterializedTransform:
    """A direct ``/World`` prim transform, primarily useful for auditing."""

    translation: tuple[float, float, float]
    scale: tuple[float, float, float]
    yaw_rad: float


@dataclass(frozen=True)
class MaterializationResult:
    """Paths and immutable hashes produced by one materialization."""

    execution_path: Path
    receipt_path: Path
    execution_sha256: str
    execution_composite_sha256: str
    input_scene_spec_sha256: str
    materialized_scene_spec_sha256: str
    receipt_sha256: str
    pinned_asset: SceneAsset
    execution_written: bool
    receipt_written: bool


@dataclass(frozen=True)
class _PrimSpan:
    name: str
    type_name: str | None
    start: int
    open_brace: int
    close_brace: int


def materialize_scene(
    scene: SceneSpec,
    repo_root: str | Path,
    output_path: str | Path,
) -> MaterializationResult:
    """Materialize ``scene`` below ``repo_root`` and write a sibling receipt.

    ``output_path`` may be a normalized POSIX path relative to ``repo_root`` or
    an absolute path already below it.  The receipt is named
    ``<output>.receipt.json``.  Both files are deterministic; a second call is
    a no-op, while pre-existing different bytes fail closed.
    """

    root = _safe_repository_root(repo_root)
    execution, execution_relpath = _safe_output_path(root, output_path)
    if execution.suffix != ".usda":
        raise SceneMaterializationError("scene output path must end in .usda")
    receipt = execution.with_name(execution.name + ".receipt.json")
    receipt_relpath = execution_relpath + ".receipt.json"
    _require_safe_descendant(root, receipt, allow_missing=True, label="receipt")

    computed_scene_hash = compute_scene_hash(scene)
    if computed_scene_hash != scene.scene_hash:
        raise SceneMaterializationError(
            "in-memory SceneSpec does not match its declared scene_hash"
        )

    source = _safe_source_path(root, scene.asset.source_path)
    if execution == source:
        raise SceneMaterializationError(
            "materialized output must not overwrite the pinned source asset"
        )
    source_bytes = _read_regular_file(source, "source scene asset")
    source_sha256 = _sha256(source_bytes)
    if source_sha256 != scene.asset.source_sha256:
        raise SceneMaterializationError(
            "source scene asset SHA-256 mismatch: "
            f"expected {scene.asset.source_sha256}, got {source_sha256}"
        )

    planning = validate_scene(scene, mode="planning", repo_root=root)
    if not planning.ok:
        raise SceneMaterializationError(
            "SceneSpec planning validation failed: " + "; ".join(planning.errors)
        )

    execution_bytes = _realize(scene, source_bytes)
    execution_sha256 = _sha256(execution_bytes)
    provisional_asset = replace(
        scene.asset,
        execution_path=execution_relpath,
        execution_sha256=execution_sha256,
        execution_composite_sha256="0" * 64,
    )
    composite_sha256 = compute_asset_composite_hash(provisional_asset)
    pinned_asset = replace(
        provisional_asset,
        execution_composite_sha256=composite_sha256,
    )
    materialized_scene_sha256 = compute_scene_hash(
        replace(scene, asset=pinned_asset)
    )
    receipt_payload = {
        "kind": MATERIALIZATION_RECEIPT_KIND,
        "schema_version": MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        "materializer": {
            "id": MATERIALIZER_ID,
            "version": MATERIALIZER_VERSION,
        },
        "scene_id": scene.scene_id.value,
        "scene_spec": {
            "input_sha256": scene.scene_hash,
            "materialized_sha256": materialized_scene_sha256,
        },
        "source": {
            "relative_path": scene.asset.source_path,
            "sha256": source_sha256,
            "size_bytes": len(source_bytes),
        },
        "realization": {
            "kind": scene.asset.realization,
            "parameters_sha256": _canonical_sha256(
                [asdict(item) for item in scene.asset.realization_parameters]
            ),
        },
        "execution": {
            "relative_path": execution_relpath,
            "sha256": execution_sha256,
            "composite_sha256": composite_sha256,
            "size_bytes": len(execution_bytes),
        },
        "receipt_relative_path": receipt_relpath,
    }
    receipt_bytes = _canonical_json_bytes(receipt_payload)

    _prepare_output_parent(root, execution.parent)
    _preflight_existing(execution, execution_bytes, "scene output")
    _preflight_existing(receipt, receipt_bytes, "materialization receipt")
    execution_written = _atomic_create_or_confirm(execution, execution_bytes)
    receipt_written = _atomic_create_or_confirm(receipt, receipt_bytes)
    return MaterializationResult(
        execution_path=execution,
        receipt_path=receipt,
        execution_sha256=execution_sha256,
        execution_composite_sha256=composite_sha256,
        input_scene_spec_sha256=scene.scene_hash,
        materialized_scene_spec_sha256=materialized_scene_sha256,
        receipt_sha256=_sha256(receipt_bytes),
        pinned_asset=pinned_asset,
        execution_written=execution_written,
        receipt_written=receipt_written,
    )


def inspect_materialized_transform(
    stage: str | bytes,
    prim_path: str,
) -> MaterializedTransform:
    """Read a direct ``/World`` transform from generated USDA text.

    This is an audit helper, not a USD replacement.  It accepts only direct
    prim paths and the exact transform representation emitted/consumed by this
    materializer.
    """

    if isinstance(stage, bytes):
        try:
            text = stage.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SceneMaterializationError("USDA stage is not UTF-8") from exc
    elif isinstance(stage, str):
        text = stage
    else:
        raise TypeError("stage must be str or bytes")
    name = _direct_name(prim_path)
    prim = _require_prim(text, name)
    return _read_transform(text, prim)


def _realize(scene: SceneSpec, source_bytes: bytes) -> bytes:
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SceneMaterializationError("source USDA asset is not UTF-8") from exc
    _top_level_prims(source_text)

    realization = scene.asset.realization
    if realization == "authored":
        output_text = source_text
    elif realization == "mirror_x":
        output_text = _realize_mirror(source_text, scene.asset.realization_parameters)
    elif realization == "narrow_corridor":
        output_text = _realize_narrow(source_text, scene.asset.realization_parameters)
    elif realization == "altered_topology":
        output_text = _realize_altered_topology(
            source_text,
            scene,
            scene.asset.realization_parameters,
        )
    else:  # SceneAsset normally prevents this, but keep the dispatcher closed.
        raise SceneMaterializationError(f"unsupported realization: {realization}")

    _verify_realization(source_text, output_text, scene)
    _verify_semantic_bindings(output_text, scene)
    output = output_text.encode("utf-8")
    if realization == "authored" and output != source_bytes:
        raise SceneMaterializationError("authored realization changed pinned source bytes")
    return output


def _realize_mirror(
    text: str,
    parameters: Sequence[RealizationParameter],
) -> str:
    if len(parameters) != 1:
        raise SceneMaterializationError("mirror_x requires exactly one parameter")
    parameter = parameters[0]
    if parameter.operation != "apply_xy_matrix_and_pi_minus_yaw":
        raise SceneMaterializationError("mirror_x has the wrong operation")
    names = tuple(_top_level_prims(text))
    targets = _expand_targets(parameter.targets, names, allow_wildcard=True)
    if not targets:
        raise SceneMaterializationError("mirror_x selected no prims")
    for name in targets:
        transform = inspect_materialized_transform(text, f"/World/{name}")
        x, y, z = transform.translation
        reflected_yaw = _normalize_angle(math.pi - transform.yaw_rad)
        text = _replace_transform(
            text,
            name,
            translation=(-x, y, z),
            yaw_rad=reflected_yaw,
        )
    return text


def _realize_narrow(
    text: str,
    parameters: Sequence[RealizationParameter],
) -> str:
    seen: set[str] = set()
    names = tuple(_top_level_prims(text))
    for parameter in parameters:
        if parameter.operation != "set_box_transform_xyz_scale_xyz_yaw":
            raise SceneMaterializationError("narrow corridor contains a non-wall operation")
        targets = _expand_targets(parameter.targets, names, allow_wildcard=False)
        for name in targets:
            if not name.startswith("wall_"):
                raise SceneMaterializationError(
                    f"narrow transform target is not an authored wall: /World/{name}"
                )
            if name in seen:
                raise SceneMaterializationError(
                    f"narrow wall is transformed more than once: /World/{name}"
                )
            seen.add(name)
            x, y, z, sx, sy, sz, yaw = parameter.values
            text = _replace_transform(
                text,
                name,
                translation=(x, y, z),
                scale=(sx, sy, sz),
                yaw_rad=yaw,
            )
    if not seen:
        raise SceneMaterializationError("narrow corridor selected no walls")
    return text


def _realize_altered_topology(
    text: str,
    scene: SceneSpec,
    parameters: Sequence[RealizationParameter],
) -> str:
    if not parameters or parameters[0].operation != "remove_matching_prims":
        raise SceneMaterializationError(
            "altered topology must begin by removing authored walls"
        )

    # The held-out topology changes the target center as part of the SceneSpec,
    # while its realization-parameter list is intentionally wall-only.  Bind
    # the target directly to the hash-checked SceneSpec before changing walls.
    target = scene.target.center
    text = _replace_transform(
        text,
        "target",
        translation=(target.x, target.y, target.z),
    )

    names = tuple(_top_level_prims(text))
    removed = _expand_targets(parameters[0].targets, names, allow_wildcard=True)
    if not removed or any(not name.startswith("wall_") for name in removed):
        raise SceneMaterializationError(
            "altered topology removal must select one or more authored wall_* prims"
        )
    text = _remove_prims(text, removed)

    added_names: set[str] = set()
    blocks: list[str] = []
    remaining_names = set(_top_level_prims(text))
    for parameter in parameters[1:]:
        if parameter.operation != "add_collision_box_xyz_scale_xyz_yaw":
            raise SceneMaterializationError(
                "altered topology contains an unsupported add operation"
            )
        if len(parameter.targets) != 1:
            raise SceneMaterializationError(
                "each altered-topology collision box must have exactly one target"
            )
        name = _direct_name(parameter.targets[0])
        if not name.startswith("topology_wall_"):
            raise SceneMaterializationError(
                f"added collision box has an unsafe name: /World/{name}"
            )
        if name in remaining_names or name in added_names:
            raise SceneMaterializationError(
                f"added collision box path is duplicated: /World/{name}"
            )
        added_names.add(name)
        blocks.append(_collision_box_block(name, parameter.values))
    if not blocks:
        raise SceneMaterializationError("altered topology adds no collision boxes")
    return _insert_world_children(text, blocks)


def _verify_realization(source: str, output: str, scene: SceneSpec) -> None:
    realization = scene.asset.realization
    parameters = scene.asset.realization_parameters
    if realization == "authored":
        if source != output:
            raise SceneMaterializationError("authored realization is not a byte copy")
        return

    source_names = tuple(_top_level_prims(source))
    output_prims = _top_level_prims(output)
    if realization == "mirror_x":
        targets = _expand_targets(parameters[0].targets, source_names, allow_wildcard=True)
        for name in targets:
            before = inspect_materialized_transform(source, f"/World/{name}")
            after = inspect_materialized_transform(output, f"/World/{name}")
            expected_translation = (
                -before.translation[0],
                before.translation[1],
                before.translation[2],
            )
            _require_vec_close(
                after.translation,
                expected_translation,
                f"mirrored {name} translation",
            )
            _require_vec_close(after.scale, before.scale, f"mirrored {name} scale")
            _require_angle_close(
                after.yaw_rad,
                math.pi - before.yaw_rad,
                f"mirrored {name} yaw",
            )
    elif realization == "narrow_corridor":
        for parameter in parameters:
            expected = parameter.values
            for target in parameter.targets:
                name = _direct_name(target)
                actual = inspect_materialized_transform(output, target)
                _require_vec_close(actual.translation, expected[:3], f"narrow {name} translation")
                _require_vec_close(actual.scale, expected[3:6], f"narrow {name} scale")
                _require_angle_close(actual.yaw_rad, expected[6], f"narrow {name} yaw")
    elif realization == "altered_topology":
        removed = _expand_targets(parameters[0].targets, source_names, allow_wildcard=True)
        lingering = sorted(set(removed).intersection(output_prims))
        if lingering:
            raise SceneMaterializationError(
                f"altered topology retained authored walls: {lingering}"
            )
        for parameter in parameters[1:]:
            target = parameter.targets[0]
            name = _direct_name(target)
            prim = output_prims.get(name)
            if prim is None or prim.type_name != "Cube":
                raise SceneMaterializationError(
                    f"altered topology did not add Cube {target}"
                )
            block = output[prim.start : prim.close_brace + 1]
            if 'prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]' not in block:
                raise SceneMaterializationError(f"{target} lacks collision APIs")
            if "bool physics:collisionEnabled = 1" not in block:
                raise SceneMaterializationError(f"{target} does not enable collision")
            actual = _read_transform(output, prim)
            expected = parameter.values
            _require_vec_close(actual.translation, expected[:3], f"altered {name} translation")
            _require_vec_close(actual.scale, expected[3:6], f"altered {name} scale")
            _require_angle_close(actual.yaw_rad, expected[6], f"altered {name} yaw")


def _verify_semantic_bindings(text: str, scene: SceneSpec) -> None:
    robot_prims = {"front": "g1", "back": "g1_01"}
    for robot in scene.initial_state.robots:
        prim_name = robot_prims.get(robot.robot_id)
        if prim_name is None:
            raise SceneMaterializationError(
                f"no authored prim binding for robot ID {robot.robot_id!r}"
            )
        actual = inspect_materialized_transform(text, f"/World/{prim_name}")
        expected_position = robot.pose.position.as_tuple()
        _require_vec_close(
            actual.translation,
            expected_position,
            f"{robot.robot_id} robot position",
        )
        _require_angle_close(actual.yaw_rad, robot.pose.yaw_rad, f"{robot.robot_id} robot yaw")

    payload = inspect_materialized_transform(text, "/World/long_box")
    _require_vec_close(
        payload.translation,
        scene.initial_state.payload.position.as_tuple(),
        "payload position",
    )
    _require_angle_close(payload.yaw_rad, scene.initial_state.payload.yaw_rad, "payload yaw")

    target = inspect_materialized_transform(text, "/World/target")
    _require_vec_close(target.translation, scene.target.center.as_tuple(), "target center")
    expected_diameter = 2.0 * scene.target.radius_m
    if not (
        math.isclose(target.scale[0], expected_diameter, abs_tol=1e-9)
        and math.isclose(target.scale[1], expected_diameter, abs_tol=1e-9)
    ):
        raise SceneMaterializationError(
            "target visual XY scale does not match the SceneSpec radius"
        )


def _top_level_prims(text: str) -> dict[str, _PrimSpan]:
    world_match = _WORLD_DECLARATION.search(text)
    if world_match is None:
        raise SceneMaterializationError('source USDA has no top-level Xform "World"')
    world_open = _next_unquoted_brace(text, world_match.end())
    world_close = _matching_brace(text, world_open)
    prims: dict[str, _PrimSpan] = {}
    for match in _DIRECT_PRIM_DECLARATION.finditer(text, world_open + 1, world_close):
        if _brace_depth(text, world_open + 1, match.start()) != 0:
            continue
        name = match.group("name")
        if name in prims:
            raise SceneMaterializationError(f"duplicate direct /World prim: {name}")
        open_brace = _next_unquoted_brace(text, match.end(), stop=world_close)
        close_brace = _matching_brace(text, open_brace)
        if close_brace > world_close:
            raise SceneMaterializationError(f"prim /World/{name} escapes World block")
        prims[name] = _PrimSpan(
            name=name,
            type_name=match.group("type"),
            start=match.start(),
            open_brace=open_brace,
            close_brace=close_brace,
        )
    if not prims:
        raise SceneMaterializationError("source USDA has no direct /World prims")
    return prims


def _require_prim(text: str, name: str) -> _PrimSpan:
    prim = _top_level_prims(text).get(name)
    if prim is None:
        raise SceneMaterializationError(f"missing required prim: /World/{name}")
    return prim


def _read_transform(text: str, prim: _PrimSpan) -> MaterializedTransform:
    block = text[prim.start : prim.close_brace + 1]
    translation = _single_vec(_DIRECT_TRANSLATE, block, prim.name, "translate", 3)
    scale = _single_vec(_DIRECT_SCALE, block, prim.name, "scale", 3)
    quaternion = _single_vec(_DIRECT_ORIENT, block, prim.name, "orient", 4)
    w, x, y, z = quaternion
    if not math.isclose(x, 0.0, abs_tol=1e-10) or not math.isclose(y, 0.0, abs_tol=1e-10):
        raise SceneMaterializationError(
            f"/World/{prim.name} orientation is not a yaw-only quaternion"
        )
    norm = math.sqrt(w * w + z * z)
    if not math.isclose(norm, 1.0, abs_tol=1e-9):
        raise SceneMaterializationError(
            f"/World/{prim.name} orientation quaternion is not normalized"
        )
    yaw = _normalize_angle(2.0 * math.atan2(z, w))
    return MaterializedTransform(translation=translation, scale=scale, yaw_rad=yaw)


def _replace_transform(
    text: str,
    name: str,
    *,
    translation: Sequence[float] | None = None,
    scale: Sequence[float] | None = None,
    yaw_rad: float | None = None,
) -> str:
    prim = _require_prim(text, name)
    block = text[prim.start : prim.close_brace + 1]
    if translation is not None:
        block = _replace_single_property(
            block,
            _DIRECT_TRANSLATE,
            "        double3 xformOp:translate = " + _tuple_text(translation),
            name,
            "translate",
        )
    if scale is not None:
        block = _replace_single_property(
            block,
            _DIRECT_SCALE,
            "        double3 xformOp:scale = " + _tuple_text(scale),
            name,
            "scale",
        )
    if yaw_rad is not None:
        half = _normalize_angle(yaw_rad) / 2.0
        quaternion = (
            _zero_small(math.cos(half)),
            0.0,
            0.0,
            _zero_small(math.sin(half)),
        )
        block = _replace_single_property(
            block,
            _DIRECT_ORIENT,
            "        quatd xformOp:orient = " + _tuple_text(quaternion),
            name,
            "orient",
        )
    return text[: prim.start] + block + text[prim.close_brace + 1 :]


def _remove_prims(text: str, names: Iterable[str]) -> str:
    prims = _top_level_prims(text)
    spans: list[tuple[int, int]] = []
    for name in names:
        prim = prims.get(name)
        if prim is None:
            raise SceneMaterializationError(f"cannot remove missing prim /World/{name}")
        end = prim.close_brace + 1
        if end < len(text) and text[end] == "\r":
            end += 1
        if end < len(text) and text[end] == "\n":
            end += 1
        spans.append((prim.start, end))
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]
    return text


def _insert_world_children(text: str, blocks: Sequence[str]) -> str:
    world_match = _WORLD_DECLARATION.search(text)
    if world_match is None:
        raise SceneMaterializationError('source USDA has no top-level Xform "World"')
    world_open = _next_unquoted_brace(text, world_match.end())
    world_close = _matching_brace(text, world_open)
    insertion = "\n" + "\n".join(block.rstrip("\n") for block in blocks) + "\n"
    return text[:world_close] + insertion + text[world_close:]


def _collision_box_block(name: str, values: Sequence[float]) -> str:
    if len(values) != 7:
        raise SceneMaterializationError("collision box requires seven transform values")
    x, y, z, sx, sy, sz, yaw = values
    half = _normalize_angle(yaw) / 2.0
    quaternion = (
        _zero_small(math.cos(half)),
        0.0,
        0.0,
        _zero_small(math.sin(half)),
    )
    return (
        f'    def Cube "{name}" (\n'
        '        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]\n'
        "    )\n"
        "    {\n"
        "        bool physics:collisionEnabled = 1\n"
        "        double size = 1\n"
        f"        quatd xformOp:orient = {_tuple_text(quaternion)}\n"
        f"        double3 xformOp:scale = {_tuple_text((sx, sy, sz))}\n"
        f"        double3 xformOp:translate = {_tuple_text((x, y, z))}\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate", '
        '"xformOp:orient", "xformOp:scale"]\n'
        "    }\n"
    )


def _expand_targets(
    targets: Sequence[str],
    available_names: Sequence[str],
    *,
    allow_wildcard: bool,
) -> tuple[str, ...]:
    selected: list[str] = []
    available = tuple(available_names)
    for target in targets:
        pattern = _direct_name(target, allow_wildcard=allow_wildcard)
        has_wildcard = "*" in pattern
        if has_wildcard:
            matches = tuple(name for name in available if fnmatchcase(name, pattern))
        else:
            matches = (pattern,) if pattern in available else ()
        if not matches:
            raise SceneMaterializationError(f"realization target matched no prim: {target}")
        for name in matches:
            if name in selected:
                raise SceneMaterializationError(
                    f"realization target selects a prim more than once: /World/{name}"
                )
            selected.append(name)
    return tuple(selected)


def _direct_name(path: str, *, allow_wildcard: bool = False) -> str:
    if not isinstance(path, str) or not path.startswith("/World/"):
        raise SceneMaterializationError(f"prim target must be below /World: {path!r}")
    name = path[len("/World/") :]
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise SceneMaterializationError(f"prim target must be a direct /World child: {path!r}")
    if any(character in name for character in "?[]"):
        raise SceneMaterializationError(f"unsupported prim target pattern: {path!r}")
    if "*" in name and not allow_wildcard:
        raise SceneMaterializationError(f"wildcard prim target is forbidden here: {path!r}")
    if name.count("*") > 1 or ("*" in name and not name.endswith("*")):
        raise SceneMaterializationError(f"unsupported prim wildcard pattern: {path!r}")
    return name


def _replace_single_property(
    block: str,
    pattern: re.Pattern[str],
    replacement: str,
    prim_name: str,
    property_name: str,
) -> str:
    matches = tuple(pattern.finditer(block))
    if len(matches) != 1:
        raise SceneMaterializationError(
            f"/World/{prim_name} must have exactly one direct {property_name} property"
        )
    match = matches[0]
    return block[: match.start()] + replacement + block[match.end() :]


def _single_vec(
    pattern: re.Pattern[str],
    block: str,
    prim_name: str,
    property_name: str,
    length: int,
) -> tuple[float, ...]:
    matches = tuple(pattern.finditer(block))
    if len(matches) != 1:
        raise SceneMaterializationError(
            f"/World/{prim_name} must have exactly one direct {property_name} property"
        )
    fields = [field.strip() for field in matches[0].group("value").split(",")]
    if len(fields) != length:
        raise SceneMaterializationError(
            f"/World/{prim_name} {property_name} has the wrong arity"
        )
    try:
        values = tuple(float(field) for field in fields)
    except ValueError as exc:
        raise SceneMaterializationError(
            f"/World/{prim_name} {property_name} is not numeric"
        ) from exc
    if any(not math.isfinite(value) for value in values):
        raise SceneMaterializationError(
            f"/World/{prim_name} {property_name} contains a non-finite value"
        )
    return values


def _next_unquoted_brace(text: str, start: int, *, stop: int | None = None) -> int:
    limit = len(text) if stop is None else stop
    in_string = False
    escaped = False
    in_comment = False
    in_asset = False
    for index in range(start, limit):
        character = text[index]
        if in_comment:
            if character == "\n":
                in_comment = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if in_asset:
            if character == "@":
                in_asset = False
            continue
        if character == "#":
            in_comment = True
        elif character == '"':
            in_string = True
        elif character == "@":
            in_asset = True
        elif character == "{":
            return index
    raise SceneMaterializationError("USDA declaration has no opening brace")


def _matching_brace(text: str, open_brace: int) -> int:
    if open_brace >= len(text) or text[open_brace] != "{":
        raise SceneMaterializationError("invalid USDA brace cursor")
    depth = 0
    in_string = False
    escaped = False
    in_comment = False
    in_asset = False
    for index in range(open_brace, len(text)):
        character = text[index]
        if in_comment:
            if character == "\n":
                in_comment = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if in_asset:
            if character == "@":
                in_asset = False
            continue
        if character == "#":
            in_comment = True
        elif character == '"':
            in_string = True
        elif character == "@":
            in_asset = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                break
    raise SceneMaterializationError("USDA stage has unbalanced braces")


def _brace_depth(text: str, start: int, stop: int) -> int:
    depth = 0
    index = start
    while index < stop:
        character = text[index]
        if character == "#":
            newline = text.find("\n", index + 1, stop)
            index = stop if newline < 0 else newline + 1
            continue
        if character == '"':
            index = _skip_delimited(text, index, '"', stop, escapes=True)
            continue
        if character == "@":
            index = _skip_delimited(text, index, "@", stop, escapes=False)
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                raise SceneMaterializationError("USDA stage has unbalanced braces")
        index += 1
    return depth


def _skip_delimited(
    text: str,
    start: int,
    delimiter: str,
    stop: int,
    *,
    escapes: bool,
) -> int:
    index = start + 1
    while index < stop:
        if escapes and text[index] == "\\":
            index += 2
            continue
        if text[index] == delimiter:
            return index + 1
        index += 1
    raise SceneMaterializationError("unterminated USDA string or asset literal")


def _safe_repository_root(value: str | Path) -> Path:
    root = Path(value)
    if root.is_symlink():
        raise SceneMaterializationError("repository root must not be a symlink")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise SceneMaterializationError("repository root does not exist") from exc
    if not resolved.is_dir():
        raise SceneMaterializationError("repository root must be a directory")
    return resolved


def _safe_source_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative_path
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise SceneMaterializationError("source path must be normalized and repository-relative")
    source = root.joinpath(*pure.parts)
    _require_safe_descendant(root, source, allow_missing=False, label="source")
    return source


def _safe_output_path(root: Path, value: str | Path) -> tuple[Path, str]:
    raw = os.fspath(value)
    if not raw or "\\" in raw:
        raise SceneMaterializationError("output path must be a normalized POSIX path")
    path = Path(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SceneMaterializationError("output path must not contain dot segments")
    if path.is_absolute():
        candidate = path
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise SceneMaterializationError("output path escapes repository root") from exc
    else:
        pure = PurePosixPath(raw)
        if pure.is_absolute() or pure.as_posix() != raw:
            raise SceneMaterializationError("output path must be normalized POSIX")
        relative = Path(*pure.parts)
        candidate = root / relative
    if not relative.parts:
        raise SceneMaterializationError("output path must name a file")
    _require_safe_descendant(root, candidate, allow_missing=True, label="output")
    return candidate, PurePosixPath(*relative.parts).as_posix()


def _require_safe_descendant(
    root: Path,
    candidate: Path,
    *,
    allow_missing: bool,
    label: str,
) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise SceneMaterializationError(f"{label} path escapes repository root") from exc
    cursor = root
    for index, part in enumerate(relative.parts):
        cursor = cursor / part
        if cursor.is_symlink():
            raise SceneMaterializationError(f"{label} path contains a symlink")
        if cursor.exists() and index < len(relative.parts) - 1 and not cursor.is_dir():
            raise SceneMaterializationError(f"{label} parent is not a directory")
        if not cursor.exists() and not allow_missing:
            raise SceneMaterializationError(f"{label} path is missing: {candidate}")
    existing = candidate if candidate.exists() else _nearest_existing_parent(candidate)
    try:
        existing.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise SceneMaterializationError(f"{label} path escapes repository root") from exc


def _nearest_existing_parent(path: Path) -> Path:
    cursor = path
    while not cursor.exists():
        parent = cursor.parent
        if parent == cursor:
            raise SceneMaterializationError("cannot resolve output parent")
        cursor = parent
    return cursor


def _prepare_output_parent(root: Path, parent: Path) -> None:
    relative = parent.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SceneMaterializationError("output parent contains a symlink")
        if cursor.exists():
            if not cursor.is_dir():
                raise SceneMaterializationError("output parent component is not a directory")
            continue
        try:
            cursor.mkdir(mode=0o755)
        except OSError as exc:
            raise SceneMaterializationError(f"cannot create output directory: {cursor}") from exc
        if cursor.is_symlink() or not cursor.is_dir():
            raise SceneMaterializationError("output directory creation was unsafe")


def _preflight_existing(path: Path, expected: bytes, label: str) -> None:
    if path.is_symlink():
        raise SceneMaterializationError(f"{label} must not be a symlink")
    if not path.exists():
        return
    actual = _read_regular_file(path, label)
    if actual != expected:
        raise SceneMaterializationError(
            f"existing {label} differs from deterministic bytes; refusing overwrite"
        )


def _atomic_create_or_confirm(path: Path, payload: bytes) -> bool:
    if path.exists():
        # The complete preflight ran immediately before this check.  Re-read to
        # fail if another writer changed the file in the meantime.
        if _read_regular_file(path, path.name) != payload:
            raise SceneMaterializationError(
                f"existing output changed during materialization: {path}"
            )
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            # A hard-link commit is atomic and, unlike os.replace(), cannot
            # overwrite a file that another writer placed after preflight.
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or _read_regular_file(path, path.name) != payload:
                raise SceneMaterializationError(
                    f"different output appeared during materialization: {path}"
                )
            temporary.unlink()
            return False
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return True


def _read_regular_file(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SceneMaterializationError(f"cannot read {label}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SceneMaterializationError(f"{label} is not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256(encoded)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tuple_text(values: Sequence[float]) -> str:
    return "(" + ", ".join(_number_text(value) for value in values) + ")"


def _number_text(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SceneMaterializationError("cannot emit a non-finite USDA number")
    numeric = _zero_small(numeric)
    if numeric.is_integer():
        return str(int(numeric))
    return repr(numeric)


def _zero_small(value: float) -> float:
    return 0.0 if abs(value) < 1e-15 else value


def _normalize_angle(value: float) -> float:
    normalized = float(value) % (2.0 * math.pi)
    if math.isclose(normalized, 2.0 * math.pi, abs_tol=1e-12):
        return 0.0
    return _zero_small(normalized)


def _require_vec_close(
    actual: Sequence[float],
    expected: Sequence[float],
    label: str,
) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(float(left), float(right), abs_tol=1e-9)
        for left, right in zip(actual, expected)
    ):
        raise SceneMaterializationError(
            f"{label} does not match SceneSpec: {tuple(actual)} != {tuple(expected)}"
        )


def _require_angle_close(actual: float, expected: float, label: str) -> None:
    difference = (actual - expected + math.pi) % (2.0 * math.pi) - math.pi
    if not math.isclose(difference, 0.0, abs_tol=1e-9):
        raise SceneMaterializationError(
            f"{label} does not match SceneSpec: {actual} != {expected}"
        )


__all__ = [
    "MATERIALIZATION_RECEIPT_KIND",
    "MATERIALIZATION_RECEIPT_SCHEMA_VERSION",
    "MATERIALIZER_ID",
    "MATERIALIZER_VERSION",
    "MaterializationResult",
    "MaterializedTransform",
    "SceneMaterializationError",
    "inspect_materialized_transform",
    "materialize_scene",
]
