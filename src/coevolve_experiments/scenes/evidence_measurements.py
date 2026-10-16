"""Frozen, check-specific acceptance rules for MuJoCo scene evidence."""

from __future__ import annotations

from math import isfinite
import re
from typing import Any, Callable, Mapping

from .schema import SceneSpec


MIN_INITIAL_SAMPLES = 30
MIN_STABLE_WINDOW_S = 1.0
MAX_INITIAL_PENETRATION_M = 0.005
MAX_INITIAL_ROBOT_DRIFT_M = 0.02
MAX_INITIAL_PAYLOAD_DRIFT_M = 0.02
MAX_INITIAL_LINEAR_SPEED_MPS = 0.05
MAX_INITIAL_ANGULAR_SPEED_RADPS = 0.10
MAX_SCRIPTED_PHYSICAL_ITERATIONS = 80

MUJOCO_MODEL_INVENTORY_KIND = "coevolve_mujoco_model_inventory"
MUJOCO_MODEL_INVENTORY_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_OBJECT_TYPES = ("body", "joint", "actuator", "geom")
_MODEL_LIST_FIELDS = {
    "body": "bodies",
    "joint": "joints",
    "actuator": "actuators",
    "geom": "geoms",
}
_MINIMUM_MODEL_COUNTS = {
    "body": 4,  # world, two robots, and payload
    "joint": 2,
    "actuator": 2,
    "geom": 3,
}


def measurement_errors(
    evidence_name: str,
    value: Mapping[str, Any],
    scene: SceneSpec,
) -> list[str]:
    """Apply the frozen acceptance rule for one evidence type."""

    return _VALIDATORS[evidence_name](value, scene, evidence_name)


def mujoco_model_inventory_errors(
    value: Any,
    measurements: Mapping[str, Any],
    evidence_name: str = "simulator_integrity",
) -> list[str]:
    """Validate a complete raw ``mjModel`` inventory against its receipt.

    The inventory is loaded from a separately hash-pinned artifact. Acceptance
    is derived here from complete object IDs, counts, role bindings, and the
    compiled-model digest; there is no producer-supplied pass flag.
    """

    label = f"{evidence_name} MuJoCo model inventory"
    if not isinstance(value, dict):
        return [f"{label} must be a JSON object"]
    fields = frozenset(
        {
            "kind",
            "schema_version",
            "compiled_model_sha256",
            "bodies",
            "joints",
            "actuators",
            "geoms",
            "role_bindings",
        }
    )
    errors = _field_errors(value, fields, label)
    if value.get("kind") != MUJOCO_MODEL_INVENTORY_KIND:
        errors.append(f"{label} has wrong kind")
    if value.get("schema_version") != MUJOCO_MODEL_INVENTORY_SCHEMA_VERSION:
        errors.append(f"{label} has wrong schema_version")
    compiled_hash = value.get("compiled_model_sha256")
    if not _is_sha256(compiled_hash):
        errors.append(f"{label} compiled_model_sha256 must be lowercase SHA-256")
    elif compiled_hash != measurements.get("compiled_model_sha256"):
        errors.append(f"{label} compiled_model_sha256 does not match measurements")

    inventories: dict[str, dict[int, str | None]] = {}
    for object_type in _MODEL_OBJECT_TYPES:
        plural = _MODEL_LIST_FIELDS[object_type]
        inventory, item_errors = _model_object_inventory(
            value.get(plural),
            object_type=object_type,
            label=f"{label} {plural}",
        )
        errors.extend(item_errors)
        inventories[object_type] = inventory
        expected_count = _integer(measurements.get(f"{object_type}_count"))
        if expected_count is not None and len(inventory) != expected_count:
            errors.append(
                f"{label} {plural} count {len(inventory)} does not match "
                f"measurements {expected_count}"
            )

    errors.extend(
        _role_binding_errors(value.get("role_bindings"), inventories, label)
    )
    return errors


def _simulator_errors(
    value: Mapping[str, Any], scene: SceneSpec, evidence_name: str
) -> list[str]:
    fields = frozenset(
        {
            "simulator_name",
            "simulator_version",
            "simulator_build",
            "model_loaded",
            "compiled_model_format",
            "compiled_model_sha256",
            "model_inventory_sha256",
            "scene_realization_sha256",
            "body_count",
            "joint_count",
            "actuator_count",
            "geom_count",
            "invalid_model_object_count",
        }
    )
    errors = _field_errors(value, fields, f"{evidence_name} measurements")
    if value.get("simulator_name") != "MuJoCo":
        errors.append(f"{evidence_name} measurements require simulator_name='MuJoCo'")
    for field in ("simulator_version", "simulator_build"):
        if not _nonempty_string(value.get(field)):
            errors.append(f"{evidence_name} measurements {field} must be non-empty")
    if value.get("model_loaded") is not True:
        errors.append(f"{evidence_name} measurements model_loaded was not observed")
    if value.get("compiled_model_format") != "mjb":
        errors.append(
            f"{evidence_name} measurements compiled_model_format must equal 'mjb'"
        )
    for field in (
        "compiled_model_sha256",
        "model_inventory_sha256",
        "scene_realization_sha256",
    ):
        if not _is_sha256(value.get(field)):
            errors.append(f"{evidence_name} measurements {field} must be lowercase SHA-256")
    if (
        scene.asset.execution_composite_sha256 is not None
        and value.get("scene_realization_sha256")
        != scene.asset.execution_composite_sha256
    ):
        errors.append(
            f"{evidence_name} measurements scene_realization_sha256 does not match "
            "the registered scene realization"
        )
    for object_type, minimum in _MINIMUM_MODEL_COUNTS.items():
        field = f"{object_type}_count"
        count = _integer(value.get(field))
        if count is None or count < minimum:
            errors.append(
                f"{evidence_name} measurements {field} must be at least {minimum}"
            )
    if _integer(value.get("invalid_model_object_count")) != 0:
        errors.append(
            f"{evidence_name} measurements invalid_model_object_count must be zero"
        )
    return errors


def _model_object_inventory(
    value: Any,
    *,
    object_type: str,
    label: str,
) -> tuple[dict[int, str | None], list[str]]:
    errors: list[str] = []
    if not isinstance(value, list):
        return {}, [f"{label} must be an array"]
    result: dict[int, str | None] = {}
    named: set[str] = set()
    fields = frozenset({"id", "name"})
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_label} must be a JSON object")
            continue
        errors.extend(_field_errors(item, fields, item_label))
        object_id = _integer(item.get("id"))
        name = item.get("name")
        if object_id is None or object_id < 0:
            errors.append(f"{item_label} id must be a non-negative integer")
            continue
        if object_id in result:
            errors.append(f"{item_label} duplicates {object_type} id {object_id}")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            errors.append(f"{item_label} name must be null or a non-empty string")
            name = None
        if isinstance(name, str):
            if name in named:
                errors.append(f"{item_label} duplicates {object_type} name {name}")
            named.add(name)
        result[object_id] = name
    if set(result) != set(range(len(result))):
        errors.append(f"{label} IDs must be the complete contiguous range [0, n)")
    return result, errors


def _role_binding_errors(
    value: Any,
    inventories: Mapping[str, Mapping[int, str | None]],
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list):
        return [f"{label} role_bindings must be an array"]
    expected_roles = {"front_robot", "back_robot", "payload"}
    seen_roles: set[str] = set()
    fields = frozenset({"role", "object_type", "object_id", "object_name"})
    for index, item in enumerate(value):
        item_label = f"{label} role_bindings[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_label} must be a JSON object")
            continue
        errors.extend(_field_errors(item, fields, item_label))
        role = item.get("role")
        object_type = item.get("object_type")
        object_id = _integer(item.get("object_id"))
        object_name = item.get("object_name")
        if role not in expected_roles:
            errors.append(f"{item_label} has unknown role {role!r}")
        elif role in seen_roles:
            errors.append(f"{item_label} duplicates role {role}")
        else:
            seen_roles.add(role)
        if object_type != "body":
            errors.append(f"{item_label} required role must bind a MuJoCo body")
            continue
        if object_id is None or object_id not in inventories.get("body", {}):
            errors.append(f"{item_label} object_id does not identify an inventoried body")
            continue
        inventory_name = inventories["body"][object_id]
        if not _nonempty_string(object_name) or object_name != inventory_name:
            errors.append(f"{item_label} object_name does not match the inventoried body")
    if seen_roles != expected_roles:
        errors.append(
            f"{label} required roles differ: "
            f"missing={sorted(expected_roles - seen_roles)}, "
            f"unknown={sorted(seen_roles - expected_roles)}"
        )
    return errors


def _initial_errors(
    value: Mapping[str, Any], scene: SceneSpec, evidence_name: str
) -> list[str]:
    del scene
    fields = frozenset(
        {
            "sample_count",
            "stable_window_s",
            "nan_sample_count",
            "disallowed_contact_count",
            "support_loss_sample_count",
            "max_initial_penetration_m",
            "max_robot_base_drift_m",
            "max_payload_drift_m",
            "max_linear_speed_mps",
            "max_angular_speed_radps",
        }
    )
    errors = _field_errors(value, fields, f"{evidence_name} measurements")
    sample_count = _integer(value.get("sample_count"))
    if sample_count is None or sample_count < MIN_INITIAL_SAMPLES:
        errors.append(
            f"{evidence_name} measurements sample_count must be at least {MIN_INITIAL_SAMPLES}"
        )
    window = _finite_number(value.get("stable_window_s"))
    if window is None or window < MIN_STABLE_WINDOW_S:
        errors.append(
            f"{evidence_name} measurements stable_window_s must be at least {MIN_STABLE_WINDOW_S}"
        )
    for field in (
        "nan_sample_count",
        "disallowed_contact_count",
        "support_loss_sample_count",
    ):
        if _integer(value.get(field)) != 0:
            errors.append(f"{evidence_name} measurements {field} must be zero")
    for field, maximum in (
        ("max_initial_penetration_m", MAX_INITIAL_PENETRATION_M),
        ("max_robot_base_drift_m", MAX_INITIAL_ROBOT_DRIFT_M),
        ("max_payload_drift_m", MAX_INITIAL_PAYLOAD_DRIFT_M),
        ("max_linear_speed_mps", MAX_INITIAL_LINEAR_SPEED_MPS),
        ("max_angular_speed_radps", MAX_INITIAL_ANGULAR_SPEED_RADPS),
    ):
        measured = _finite_number(value.get(field))
        if measured is None or measured < 0.0 or measured > maximum:
            errors.append(
                f"{evidence_name} measurements {field} must be within [0, {maximum}]"
            )
    return errors


def _scripted_errors(
    value: Mapping[str, Any], scene: SceneSpec, evidence_name: str
) -> list[str]:
    fields = frozenset(
        {
            "script_sha256",
            "controller_id",
            "controller_version",
            "controller_sha256",
            "skill_api_fingerprint",
            "trajectory_sample_count",
            "verifier_sample_count",
            "physical_iteration_count",
            "terminal_reached",
            "terminal_success",
            "invariant_violation_count",
            "nan_sample_count",
            "final_payload_target_distance_m",
        }
    )
    errors = _field_errors(value, fields, f"{evidence_name} measurements")
    for field in ("script_sha256", "controller_sha256", "skill_api_fingerprint"):
        if not _is_sha256(value.get(field)):
            errors.append(f"{evidence_name} measurements {field} must be lowercase SHA-256")
    for field in ("controller_id", "controller_version"):
        if not _nonempty_string(value.get(field)):
            errors.append(f"{evidence_name} measurements {field} must be non-empty")
    for field in ("trajectory_sample_count", "verifier_sample_count"):
        count = _integer(value.get(field))
        if count is None or count < 1:
            errors.append(f"{evidence_name} measurements {field} must be positive")
    iterations = _integer(value.get("physical_iteration_count"))
    if (
        iterations is None
        or iterations < 1
        or iterations > MAX_SCRIPTED_PHYSICAL_ITERATIONS
    ):
        errors.append(
            f"{evidence_name} measurements physical_iteration_count must be within "
            f"[1, {MAX_SCRIPTED_PHYSICAL_ITERATIONS}]"
        )
    for field in ("terminal_reached", "terminal_success"):
        if value.get(field) is not True:
            errors.append(f"{evidence_name} measurements {field} was not observed")
    for field in ("invariant_violation_count", "nan_sample_count"):
        if _integer(value.get(field)) != 0:
            errors.append(f"{evidence_name} measurements {field} must be zero")
    distance = _finite_number(value.get("final_payload_target_distance_m"))
    if distance is None or distance < 0.0 or distance > scene.target.radius_m:
        errors.append(
            f"{evidence_name} measurements final payload distance must be within "
            f"target radius {scene.target.radius_m}"
        )
    return errors


_VALIDATORS: Mapping[
    str, Callable[[Mapping[str, Any], SceneSpec, str], list[str]]
] = {
    "simulator_integrity": _simulator_errors,
    "initial_collision": _initial_errors,
    "scripted_solvability": _scripted_errors,
}


def _field_errors(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> list[str]:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        return [f"{label} fields differ: missing={missing}, unknown={unknown}"]
    return []


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if isfinite(result) else None


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


__all__ = [
    "MUJOCO_MODEL_INVENTORY_KIND",
    "MUJOCO_MODEL_INVENTORY_SCHEMA_VERSION",
    "measurement_errors",
    "mujoco_model_inventory_errors",
]
