"""Machine-readable static carry-skill catalog used by model prompts."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from praxis_coop.sonic.upper_body import (
    DEFAULT_CARRY_READY_POSE_NAME,
    PAYLOAD_RECOVERY_POSE_NAME,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _catalog_path() -> Path:
    """Use the shipped SDK definition in both source checkouts and wheels.

    ``results/`` is private experiment output, not an installation dependency.
    This packaged definition is the explicit source of truth, not a fallback:
    if it is missing or invalid, loading still fails closed. Historical frozen
    runtimes and their original catalog paths are not migrated by this change.
    """
    return Path(__file__).resolve().parents[1] / "data/default_specs.jsonl"


STATIC_SKILL_CATALOG_PATH = _catalog_path()
REQUIRED_SKILL_FAMILIES = frozenset(
    {
        "carry_walk",
        "carry_turn",
        "carry_crouch",
        "carry_lateral",
    }
)

# These are not motion choices. They are exposed only for initialization and
# the one-action transaction protocol around each model-selected skill.
INITIALIZATION_SKILL_METHODS = frozenset({"place_payload_on_arms"})
FINALIZATION_SKILL_METHODS = frozenset(
    {
        "release_payload",
        "exit_forward",
        "exit_backward",
        "exit_lateral_left",
        "exit_lateral_right",
    }
)
PROTOCOL_SKILL_METHODS = frozenset({"hold_position", "stop_all"})


@lru_cache(maxsize=1)
def load_static_skill_catalog() -> tuple[dict[str, Any], ...]:
    """Load and validate the four model-selectable carry-skill families."""

    if not STATIC_SKILL_CATALOG_PATH.is_file():
        raise RuntimeError(f"missing static skill catalog: {STATIC_SKILL_CATALOG_PATH}")
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        STATIC_SKILL_CATALOG_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid static skill catalog JSON at line {line_number}: {exc}"
            ) from exc
        if record.get("kind") != "static_skill_family":
            raise RuntimeError(
                f"static skill catalog line {line_number} has unsupported kind "
                f"{record.get('kind')!r}"
            )
        family = str(record.get("family", "")).strip()
        methods = record.get("methods")
        if not family or not isinstance(methods, list) or not methods:
            raise RuntimeError(
                f"static skill catalog line {line_number} requires family and methods"
            )
        for method in methods:
            if not isinstance(method, dict) or not str(method.get("name", "")).strip():
                raise RuntimeError(
                    f"static skill catalog line {line_number} has an invalid method entry"
                )
            if not isinstance(method.get("parameters", {}), dict):
                raise RuntimeError(
                    f"static skill catalog method {method.get('name')!r} has invalid parameters"
                )
        records.append(record)

    families = {str(record["family"]) for record in records}
    if families != REQUIRED_SKILL_FAMILIES:
        raise RuntimeError(
            "static skill catalog must contain exactly the four carry families; "
            f"expected {sorted(REQUIRED_SKILL_FAMILIES)}, got {sorted(families)}"
        )
    method_names = [
        str(method["name"])
        for record in records
        for method in record["methods"]
    ]
    if len(method_names) != len(set(method_names)):
        raise RuntimeError("static skill catalog contains duplicate API method names")
    return tuple(records)


def model_action_skill_methods() -> frozenset[str]:
    """Return only motion/place actions that the Agent may select."""

    return frozenset(
        str(method["name"])
        for record in load_static_skill_catalog()
        for method in record["methods"]
    )


MODEL_ACTION_SKILL_METHODS = model_action_skill_methods()
MODEL_SCRIPT_API_METHODS = (
    MODEL_ACTION_SKILL_METHODS
    | INITIALIZATION_SKILL_METHODS
    | FINALIZATION_SKILL_METHODS
    | PROTOCOL_SKILL_METHODS
)


def static_skill_method_spec(method_name: str) -> dict[str, Any]:
    """Return one method definition from the catalog."""

    for family in load_static_skill_catalog():
        for method in family["methods"]:
            if method["name"] == method_name:
                return method
    raise ValueError(f"method {method_name!r} is not a model-selectable static skill")


def validate_catalog_parameters(method_name: str, parameters: dict[str, object]) -> None:
    """Validate numeric API parameters against the tracked static catalog."""

    method = static_skill_method_spec(method_name)
    specs = method.get("parameters", {})
    unknown = sorted(set(parameters) - set(specs))
    if unknown:
        raise ValueError(f"api.{method_name} received unsupported parameters: {unknown}")
    for name, value in parameters.items():
        spec = specs[name]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"api.{method_name} {name} must be numeric, got {value!r}") from exc
        if "min" in spec and numeric < float(spec["min"]):
            raise ValueError(
                f"api.{method_name} {name} must be >= {spec['min']}, got {numeric}"
            )
        if "exclusive_min" in spec and numeric <= float(spec["exclusive_min"]):
            raise ValueError(
                f"api.{method_name} {name} must be > {spec['exclusive_min']}, got {numeric}"
            )
        if "max" in spec and numeric > float(spec["max"]):
            raise ValueError(
                f"api.{method_name} {name} must be <= {spec['max']}, got {numeric}"
            )


def static_skill_library_doc() -> str:
    """Render the tracked static catalog as the model's motor SDK documentation."""

    lines = [
        "# Model-selectable G1 carry skill library",
        "",
        "Select exactly one bounded action family for the next physical step using the current "
        "physics state, camera observation, target progress, and rollout evidence.",
        f"Term1 starts by holding {DEFAULT_CARRY_READY_POSE_NAME} with both physical palm "
        "support surfaces tilted to 60 degrees as raised payload backstops. "
        "The only model-selectable arm change is the guarded, one-way "
        "recover_payload_inward action; all other arm locking remains a runtime invariant.",
        "Hand, wrist, and forearm contact all count as carrying support. All carry actions require "
        "the payload to remain supported by both robots. Directional "
        "calls are relative to each robot's current body frame.",
        "distance_m is a MuJoCo payload-projection endpoint target closed by the transport "
        "feedback layer. speed_mps is a nominal SONIC command, not guaranteed measured "
        "velocity. The settled physics endpoint determines the actual result.",
        "payload.support_center_signed_longitudinal_offset_m uses the robot body frame: positive "
        "means the box is outward/body-forward and negative means toward the torso. Select "
        "recover_payload_inward only when that positive offset reaches the reported warning level; "
        f"it is unavailable after Term1 reaches {PAYLOAD_RECOVERY_POSE_NAME}.",
        "",
    ]
    for index, family in enumerate(load_static_skill_catalog(), start=1):
        lines.extend(
            [
                f"## {index}. {family['display_name']} ({family['family']})",
                str(family["description"]),
                f"Choose when: {family['selection_guidance']}",
            ]
        )
        preconditions = family.get("preconditions", [])
        if preconditions:
            lines.append("Preconditions: " + "; ".join(str(item) for item in preconditions))
        lines.append("Calls:")
        for method in family["methods"]:
            lines.append(f"- api.{method['example']}: {method['description']}")
            for name, spec in method.get("parameters", {}).items():
                lines.append(f"  - {name}: {_parameter_doc(spec)}")
        lines.append("")

    lines.extend(
        [
            "# Initialization-only runtime event",
            "- api.place_payload_on_arms(hold_s=4): after both robots are stable and the arm "
            "lock is active, place the deferred payload at the live four-palm support center. "
            "The host requires this only when the payload is absent or unsupported; it is not a "
            "route-planning action.",
            "",
            "# Transaction protocol",
            "- api.hold_position(hold_s=2): keep one selected robot in a loaded support role while "
            "the other robot executes one bounded action in coordination_mode=async.",
            "- api.stop_all(): required final call after every selected action.",
            "Both loaded scripts use the same synchronization_group so commands are published in "
            "one atomic team phase. coordination_mode=sync means one shared phase objective and "
            "identical mover parameters. coordination_mode=async means different temporary roles or "
            "motion objectives inside an arc, pivot, or alignment primitive, or one robot may use "
            "hold_position while the other executes one bounded walk, turn, or lateral action. It "
            "never combines two different mover methods. Identical front/back mover calls are sync; "
            "changing only coordination_mode metadata is not a strategy change. Async support and mover scripts may use "
            "different contact_intent values while preserving payload support.",
            "api.recover_payload_inward is always sync-only: both robot scripts must use identical "
            "parameters in the same synchronization_group. It is never a single-robot or async action.",
            "",
            "# Final placement protocol (not part of the four carry-skill families)",
            "These calls are valid only after the payload reaches the final target region.",
            "- api.release_payload(ramp_s=4, hold_s=2): while both robots remain stationary, "
            "move the locked palms down and back along a joint-controlled trajectory. The call "
            "passes only when the target physically supports the payload and robot upper-limb contact "
            "has ended. The lower-body crouch remains latched after this call.",
            "- api.exit_forward(distance_m=0.5, speed_mps=0.20, ramp_s=1.5)",
            "- api.exit_backward(distance_m=0.5, speed_mps=0.20, ramp_s=1.5)",
            "- api.exit_lateral_left(distance_m=0.5, speed_mps=0.20, ramp_s=1.5)",
            "- api.exit_lateral_right(distance_m=0.5, speed_mps=0.20, ramp_s=1.5)",
            "Exit directions are body-relative unloaded SONIC walking. Select them from current "
            "robot yaw, target-relative displacement, and visible clearance. They must leave the "
            "target-supported payload stationary.",
            "Formal completion uses only api.exit_backward with identical front/back parameters. "
            "Both robots remain in the latched crouch throughout that synchronized retreat.",
            "- api.recover_stand(height=0.80, ramp_s=6, hold_s=2): after the accepted crouched "
            "backward retreat has cleared both robots from the box, recover both robots upright. "
            "The task is complete only after this final standing checkpoint is physically verified.",
        ]
    )
    return "\n".join(lines)


def _parameter_doc(spec: dict[str, Any]) -> str:
    bounds: list[str] = []
    if "exclusive_min" in spec:
        bounds.append(f"> {spec['exclusive_min']}")
    elif "min" in spec:
        bounds.append(f">= {spec['min']}")
    if "max" in spec:
        bounds.append(f"<= {spec['max']}")
    range_text = " and ".join(bounds)
    default = spec.get("default")
    unit = str(spec.get("unit", "")).strip()
    meaning = str(spec.get("description", "")).strip()
    pieces = [range_text, f"default {default}" if default is not None else "", unit, meaning]
    return "; ".join(piece for piece in pieces if piece)
