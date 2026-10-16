"""Load JSON-compatible YAML protocol files with strict conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..contracts import MemoryCondition, MethodVariant, SceneId, TableId
from ..serialization import strict_json_loads
from .schema import CommonProtocol, ProtocolBundle, ProtocolCondition, TableProtocol


_PROTOCOL_DIR = Path(__file__).resolve().parent


def load_protocol_bundle(root: Optional[Path] = None) -> ProtocolBundle:
    protocol_root = Path(root) if root is not None else _PROTOCOL_DIR
    common_raw = _read_mapping(protocol_root / "common.yaml")
    _require_keys(
        common_raw,
        {
            "schema_version",
            "seeds",
            "physical_budget",
            "target_backbone",
            "target_provider_route_id",
            "source_backbones",
            "source_provider_route_ids",
            "protocol_invariant_ref",
            "terminal_predicate_ref",
        },
        "common protocol",
    )
    common = CommonProtocol(
        schema_version=_integer(common_raw, "schema_version"),
        seeds=tuple(_integer_value(value, "seeds") for value in _list(common_raw, "seeds")),
        physical_budget=_integer(common_raw, "physical_budget"),
        target_backbone=_string(common_raw, "target_backbone"),
        target_provider_route_id=_string(common_raw, "target_provider_route_id"),
        source_backbones=tuple(
            _string_value(value, "source_backbones")
            for value in _list(common_raw, "source_backbones")
        ),
        source_provider_route_ids=tuple(
            _string_value(value, "source_provider_route_ids")
            for value in _list(common_raw, "source_provider_route_ids")
        ),
        protocol_invariant_ref=_string(common_raw, "protocol_invariant_ref"),
        terminal_predicate_ref=_string(common_raw, "terminal_predicate_ref"),
    )
    tables = tuple(
        _load_table(protocol_root / filename)
        for filename in ("table2.yaml", "table3.yaml", "table4.yaml")
    )
    return ProtocolBundle(common=common, tables=tables)


def _load_table(path: Path) -> TableProtocol:
    raw = _read_mapping(path)
    _require_keys(raw, {"schema_version", "table_id", "conditions"}, str(path))
    conditions = []
    for index, value in enumerate(_list(raw, "conditions")):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: conditions[{index}] must be an object")
        required = {
            "condition_id",
            "label",
            "core",
            "execution",
            "method_variant",
            "scene_id",
            "memory_condition",
        }
        optional = {"source"}
        _require_keys(
            value,
            required,
            f"{path}: conditions[{index}]",
            optional=optional,
        )
        source = value.get("source")
        if source is not None and not isinstance(source, str):
            raise ValueError(f"{path}: conditions[{index}].source must be a string or null")
        conditions.append(
            ProtocolCondition(
                condition_id=_string(value, "condition_id"),
                label=_string(value, "label"),
                core=_boolean(value, "core"),
                execution=_string(value, "execution"),
                method_variant=MethodVariant(_string(value, "method_variant")),
                scene_id=SceneId(_string(value, "scene_id")),
                memory_condition=MemoryCondition(_string(value, "memory_condition")),
                source=source,
            )
        )
    return TableProtocol(
        schema_version=_integer(raw, "schema_version"),
        table_id=TableId(_string(raw, "table_id")),
        conditions=tuple(conditions),
    )


def _read_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing protocol file: {path}") from exc
    except ValueError as exc:
        raise ValueError(
            f"{path} must use the repository's JSON-compatible YAML subset: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level object")
    return payload


def _string(mapping: Mapping[str, Any], key: str) -> str:
    return _string_value(_required(mapping, key), key)


def _string_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    return _integer_value(_required(mapping, key), key)


def _integer_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(mapping: Mapping[str, Any], key: str) -> bool:
    value = _required(mapping, key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _list(mapping: Mapping[str, Any], key: str) -> list[Any]:
    value = _required(mapping, key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required key: {key}")
    return mapping[key]


def _require_keys(
    mapping: Mapping[str, Any],
    required: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    actual = set(mapping)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        raise ValueError(f"{label} fields differ: missing={missing}, unknown={unknown}")


__all__ = ["load_protocol_bundle"]
