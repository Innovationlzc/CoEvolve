"""Small parser for the hand-authored USDA layout used in demos.

This is intentionally narrow: it extracts the transform fields we need for the
manual two-G1 box-push scene without requiring USD Python bindings in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from pathlib import Path
import re


@dataclass(frozen=True)
class UsdaPrim2D:
    name: str
    kind: str
    x: float
    y: float
    z: float
    sx: float
    sy: float
    sz: float
    yaw: float

    def to_json(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "sx": self.sx,
            "sy": self.sy,
            "sz": self.sz,
            "yaw": self.yaw,
        }


def parse_manual_layout_prims(path: Path) -> list[UsdaPrim2D]:
    """Extract top-level wall/object/target transforms from a USDA file."""

    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    prims: list[UsdaPrim2D] = []
    authored_names = dict.fromkeys(
        re.findall(r'def(?:\s+\w+)?\s+"([^"]+)"', text)
    )
    selected_names = [
        name
        for name in authored_names
        if name in {"g1", "g1_01", "long_box", "target"}
        or re.fullmatch(r"wall_\d+", name)
    ]
    for name in selected_names:
        block = _extract_prim_block(text, name)
        if not block:
            continue
        translate = _triple(block, r"xformOp:translate = \(([^)]+)\)", default=(0.0, 0.0, 0.0))
        scale = _triple(block, r"xformOp:scale = \(([^)]+)\)", default=(1.0, 1.0, 1.0))
        orient = _quad(block, r"xformOp:orient = \(([^)]+)\)", default=(1.0, 0.0, 0.0, 0.0))
        kind = "robot" if name.startswith("g1") else "wall" if name.startswith("wall_") else name
        prims.append(
            UsdaPrim2D(
                name=name,
                kind=kind,
                x=translate[0],
                y=translate[1],
                z=translate[2],
                sx=scale[0],
                sy=scale[1],
                sz=scale[2],
                yaw=_yaw_from_quat_wxyz(orient),
            )
        )
    return prims


def _extract_prim_block(text: str, name: str) -> str:
    match = re.search(rf'def(?:\s+\w+)?\s+"{re.escape(name)}"', text)
    if not match:
        return ""
    open_idx = text.find("{", match.end())
    if open_idx < 0:
        return ""
    start = open_idx + 1
    depth = 1
    idx = start
    while idx < len(text) and depth > 0:
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        idx += 1
    return text[start : idx - 1]


def _triple(block: str, pattern: str, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    match = re.search(pattern, block)
    if not match:
        return default
    values = _numbers(match.group(1))
    if len(values) < 3:
        return default
    return (values[0], values[1], values[2])


def _quad(block: str, pattern: str, *, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    match = re.search(pattern, block)
    if not match:
        return default
    values = _numbers(match.group(1))
    if len(values) < 4:
        return default
    return (values[0], values[1], values[2], values[3])


def _numbers(payload: str) -> list[float]:
    return [float(item.strip()) for item in payload.split(",") if item.strip()]


def _yaw_from_quat_wxyz(quat: tuple[float, float, float, float]) -> float:
    w, x, y, z = quat
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
