"""Model-facing coordination primitives built from the static SONIC SDK."""

from __future__ import annotations

from math import radians
from typing import Any

from praxis_coop.sonic.coordination_primitives import (
    COORDINATED_ALIGNMENT,
    COORDINATED_ARC_TRANSPORT,
    COORDINATED_LINEAR_TRANSPORT,
    COORDINATED_PIVOT,
    COORDINATION_PRIMITIVE_METHODS,
    MIN_COORDINATED_ARC_RADIUS_M,
    SONIC_MAX_TRANSLATION_COMMAND_MPS,
    calibrated_arc_translation_speed,
)

COORDINATED_LINEAR_DISTANCE_CHOICES_M = (0.25, 0.5, 0.75, 1.0)
FINAL_APPROACH_LINEAR_DISTANCE_CHOICES_M = (0.25, 0.5)
VERIFIED_LINEAR_SPEED_MPS = 0.50
VERIFIED_LINEAR_RAMP_S = 1.5
VERIFIED_LOADED_LINEAR_MAX_M = 1.0
MODEL_ROUTE_TURN_ANGLE_DEG = 30.0
MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG = (10.0, 15.0, 20.0, 30.0)
CALIBRATED_MODEL_PIVOT_DURATION_S = 8.0
LONGITUDINAL_ALIGNMENT_TRIGGER_M = 0.15
MIN_LONGITUDINAL_ALIGNMENT_DELTA_M = 0.15
MAX_LONGITUDINAL_ALIGNMENT_DELTA_M = 0.15
MODEL_COORDINATION_PRIMITIVE_METHODS = frozenset(
    {
        COORDINATED_LINEAR_TRANSPORT,
        COORDINATED_PIVOT,
        COORDINATED_ALIGNMENT,
    }
)
SYNC_ONLY_STATIC_METHODS = frozenset(
    {
        "place_payload_on_arms",
        "hold_position",
        "carry_backward",
        "turn_clockwise",
        "turn_counterclockwise",
        "place_crouch",
        "recover_stand",
        "recover_payload_inward",
        "release_payload",
        "exit_forward",
        "exit_backward",
        "exit_lateral_left",
        "exit_lateral_right",
        "stop_all",
    }
)
# A synchronized team retreat is valid when both robots retain bilateral
# support and rear clearance. Backward motion remains forbidden inside pivot
# and relative longitudinal-alignment primitives, where one-sided reversal can
# unload the free payload.
LOADED_TRANSLATION_DIRECTIONS = ("forward", "backward", "left", "right")
_LOADED_ARC_TRAVEL_DIRECTIONS = ("forward", "left", "right")

_METHOD_SPECS: dict[str, dict[str, Any]] = {
    COORDINATED_LINEAR_TRANSPORT: {
        "coordination_modes": ("sync",),
        "supports_role_reassignment": False,
        "role_variant_limit": 0,
        "description": (
            "Continuously transport a rigid payload along one body-frame axis. The compiler "
            "atomically aligns both robots' move windows and requires identical parameters."
        ),
        "example": (
            f'coordinated_linear_transport(direction="forward", distance_m={VERIFIED_LOADED_LINEAR_MAX_M:.1f}, '
            "speed_mps=0.50, ramp_s=1.5)"
        ),
        "parameters": {
            "direction": {
                "choices": LOADED_TRANSLATION_DIRECTIONS,
                "default": "forward",
                "description": (
                    "shared robot-body-frame loaded transport direction; backward is "
                    "a synchronized team retreat, never a one-robot correction"
                ),
            },
            "distance_m": {
                "exclusive_min": 0.0,
                "max": VERIFIED_LOADED_LINEAR_MAX_M,
                "default": VERIFIED_LOADED_LINEAR_MAX_M,
                "unit": "m",
                "model_choices": COORDINATED_LINEAR_DISTANCE_CHOICES_M,
                "description": (
                    "nominal bounded segment length; model-generated loaded transport uses "
                    "one of 0.25, 0.5, 0.75, or 1.0m"
                ),
            },
            "speed_mps": {
                "min": 0.2,
                "max": 0.5,
                "default": VERIFIED_LINEAR_SPEED_MPS,
                "model_fixed": VERIFIED_LINEAR_SPEED_MPS,
                "unit": "command m/s",
                "description": (
                    "SONIC command magnitude; model-generated loaded transport keeps the "
                    "manually verified fixed 0.50 command for both robots"
                ),
            },
            "ramp_s": {
                "min": 0.0,
                "max": 4.0,
                "default": VERIFIED_LINEAR_RAMP_S,
                "model_fixed": VERIFIED_LINEAR_RAMP_S,
                "unit": "s",
                "description": (
                    "zero-speed settle before and after continuous motion; fixed to the "
                    "verified 1.5s value for model-generated loaded transport"
                ),
            },
        },
    },
    COORDINATED_ARC_TRANSPORT: {
        "coordination_modes": ("async",),
        "supports_role_reassignment": True,
        "role_variant_limit": 2,
        "description": (
            "Continuously combine one model-selected body-frame travel direction and heading "
            "change around one shared turn. Each robot supplies its own radius; the larger "
            "radius is the outer role and the smaller positive radius is the inner role. Do "
            "not assume body-forward travel: infer forward/left/right from live robot "
            "yaw and the global-camera passage. Use coordinated_pivot when one robot must "
            "rotate in place."
        ),
        "example": (
            'coordinated_arc_transport(turn_direction="counterclockwise", '
            'travel_direction="left", angle_deg=30.0, duration_s=8.0, radius_m=0.8)'
        ),
        "parameters": {
            "turn_direction": {
                "choices": ("clockwise", "counterclockwise"),
                "default": "counterclockwise",
                "description": (
                    "shared world-yaw direction viewed from world +Z/top-down: "
                    "counterclockwise increases yaw and clockwise decreases yaw"
                ),
            },
            "travel_direction": {
                "choices": _LOADED_ARC_TRAVEL_DIRECTIONS,
                "default": "forward",
                "required": False,
                "description": (
                    "shared robot-body-frame translation direction; infer it independently "
                    "from turn_direction using current yaw and visible passage geometry"
                ),
            },
            "angle_deg": {
                "min": 2.0,
                "max": 30.0,
                "default": MODEL_ROUTE_TURN_ANGLE_DEG,
                "model_choices": MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG,
                "unit": "deg",
                "description": (
                    "model-selected route turns use one of 10, 15, 20, or 30 degrees; "
                    "the sign is supplied separately by turn_direction"
                ),
            },
            "duration_s": {
                "min": 2.0,
                "max": 12.0,
                "default": 4.0,
                "unit": "s",
                "description": "shared continuous arc window",
            },
            "radius_m": {
                "min": MIN_COORDINATED_ARC_RADIUS_M,
                "max": 1.5,
                "default": 0.4,
                "unit": "m",
                "description": "per-robot radius about the inferred turn center",
            },
        },
    },
    COORDINATED_PIVOT: {
        "coordination_modes": ("async",),
        "supports_role_reassignment": True,
        "role_variant_limit": 2,
        "description": (
            "Rotate a rigid payload continuously around one temporary inner support. Exactly one "
            "robot is the stationary inner pivot and one is the forward-moving outer swing role. "
            "The inner robot receives zero translation and only changes body heading; the outer "
            "robot follows a forward arc. Neither role is allowed to back up."
        ),
        "example": (
            'coordinated_pivot(turn_direction="counterclockwise", angle_deg=30.0, '
            'travel_direction="forward", duration_s=8.0, role="outer", swing_radius_m=0.8)'
        ),
        "parameters": {
            "turn_direction": {
                "choices": ("clockwise", "counterclockwise"),
                "default": "counterclockwise",
                "description": (
                    "shared world-yaw direction viewed from world +Z/top-down: "
                    "counterclockwise increases yaw and clockwise decreases yaw"
                ),
            },
            "travel_direction": {
                "choices": ("forward",),
                "default": "forward",
                "required": False,
                "description": (
                    "fixed forward body-frame translation for the outer role; ignored by "
                    "the stationary inner role"
                ),
            },
            "angle_deg": {
                "min": 2.0,
                "max": 30.0,
                "default": MODEL_ROUTE_TURN_ANGLE_DEG,
                "model_choices": MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG,
                "unit": "deg",
                "description": (
                    "model-selected route turns use one of 10, 15, 20, or 30 degrees; "
                    "straight-line drift repair may use a verifier-reported correction"
                ),
            },
            "duration_s": {
                "min": 2.0,
                "max": 12.0,
                "default": CALIBRATED_MODEL_PIVOT_DURATION_S,
                "model_fixed": CALIBRATED_MODEL_PIVOT_DURATION_S,
                "unit": "s",
                "description": (
                    "shared continuous pivot window; model-generated route pivots use "
                    "the calibrated fixed 8-second window"
                ),
            },
            "role": {
                "choices": ("inner", "outer"),
                "default": "inner",
                "description": "per-robot pivot or swing role",
            },
            "swing_radius_m": {
                "min": 0.1,
                "max": 1.5,
                "default": 0.8,
                "unit": "m",
                "description": "shared outer-role radius; ignored for inner translation",
            },
        },
    },
    COORDINATED_ALIGNMENT: {
        "coordination_modes": ("async",),
        "supports_role_reassignment": True,
        "role_variant_limit": 2,
        "description": (
            "Apply one small continuous relative correction while both robots remain in one "
            "atomic payload-support phase. Positive delta means forward, body-left, or "
            "counterclockwise; zero makes that robot the temporary reference. Longitudinal "
            "delta must be non-negative because a supported payload must never make either "
            "robot translate body-backward. Lateral and yaw delta remain signed. This repairs "
            "measured relative formation error only: lateral alignment does not turn the shared "
            "team heading or carry the complete team sideways."
        ),
        "example": (
            'coordinated_alignment(axis="lateral", delta=0.10, duration_s=2.0)'
        ),
        "parameters": {
            "axis": {
                "choices": ("longitudinal", "lateral", "yaw"),
                "default": "lateral",
                "description": "shared correction axis",
            },
            "delta": {
                "min": -30.0,
                "max": 30.0,
                "default": 0.0,
                "description": (
                    "signed meters for translation axes (longitudinal is zero or "
                    "abs(delta)>=0.15m; lateral is zero or abs(delta)>=0.05m) or degrees "
                    "for yaw (zero or abs(delta)>=2deg); longitudinal values must be >=0 "
                    "while lateral and yaw values may be positive or negative"
                ),
            },
            "duration_s": {
                "min": 1.0,
                "max": 8.0,
                "default": 2.0,
                "unit": "s",
                "description": "shared correction window",
            },
        },
    },
}


def coordination_primitive_method_spec(method_name: str) -> dict[str, Any]:
    """Return one immutable-by-convention primitive specification."""

    try:
        return _METHOD_SPECS[method_name]
    except KeyError as exc:
        raise ValueError(f"unknown coordination primitive {method_name!r}") from exc


def coordination_modes_for_method(method_name: str) -> frozenset[str]:
    """Return the executable role modes for one model-selectable method."""

    if method_name in SYNC_ONLY_STATIC_METHODS:
        return frozenset({"sync"})
    if method_name not in COORDINATION_PRIMITIVE_METHODS:
        return frozenset({"sync", "async"})
    return frozenset(coordination_primitive_method_spec(method_name)["coordination_modes"])


def coordination_relationship_can_change(
    method_name: str,
    current_mode: str,
) -> bool:
    """Return whether a coordination repair can change mode or temporary roles."""

    modes = coordination_modes_for_method(method_name)
    if any(mode != current_mode for mode in modes):
        return True
    if method_name not in COORDINATION_PRIMITIVE_METHODS:
        return False
    return bool(
        coordination_primitive_method_spec(method_name).get(
            "supports_role_reassignment",
            False,
        )
    )


def coordination_role_variant_limit(method_name: str) -> int | None:
    """Return a finite role-permutation count when the primitive defines one."""

    if method_name not in COORDINATION_PRIMITIVE_METHODS:
        return None
    value = coordination_primitive_method_spec(method_name).get(
        "role_variant_limit"
    )
    return None if value is None else int(value)


def validate_coordination_primitive_parameters(
    method_name: str,
    parameters: dict[str, object],
) -> None:
    """Validate literal model parameters without accepting arbitrary nested programs."""

    spec = coordination_primitive_method_spec(method_name)
    parameter_specs = spec["parameters"]
    unknown = sorted(set(parameters) - set(parameter_specs))
    if unknown:
        raise ValueError(f"api.{method_name} received unsupported parameters: {unknown}")
    missing = sorted(
        name
        for name, parameter_spec in parameter_specs.items()
        if parameter_spec.get("required", True) and name not in parameters
    )
    if missing:
        raise ValueError(f"api.{method_name} requires explicit parameters: {missing}")
    for name, value in parameters.items():
        parameter_spec = parameter_specs[name]
        choices = parameter_spec.get("choices")
        if choices is not None:
            if str(value) not in choices:
                raise ValueError(
                    f"api.{method_name} {name} must be one of {list(choices)}, got {value!r}"
                )
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"api.{method_name} {name} must be numeric, got {value!r}"
            ) from exc
        if "min" in parameter_spec and numeric < float(parameter_spec["min"]):
            raise ValueError(
                f"api.{method_name} {name} must be >= {parameter_spec['min']}, got {numeric}"
            )
        if "exclusive_min" in parameter_spec and numeric <= float(
            parameter_spec["exclusive_min"]
        ):
            raise ValueError(
                f"api.{method_name} {name} must be > {parameter_spec['exclusive_min']}, got {numeric}"
            )
        if "max" in parameter_spec and numeric > float(parameter_spec["max"]):
            raise ValueError(
                f"api.{method_name} {name} must be <= {parameter_spec['max']}, got {numeric}"
            )

    if method_name == COORDINATED_ALIGNMENT:
        axis = str(parameters["axis"])
        signed_delta = float(parameters["delta"])
        if axis == "longitudinal" and signed_delta < 0.0:
            raise ValueError(
                f"api.{method_name} longitudinal delta must be >= 0; loaded "
                "body-backward translation is forbidden"
            )
        delta = abs(signed_delta)
        limit = 0.50 if axis != "yaw" else 30.0
        minimum = (
            MIN_LONGITUDINAL_ALIGNMENT_DELTA_M
            if axis == "longitudinal"
            else 0.05
            if axis == "lateral"
            else 2.0
        )
        unit = "m" if axis != "yaw" else "deg"
        if 0.0 < delta < minimum:
            raise ValueError(
                f"api.{method_name} abs(delta) must be 0 or >= {minimum:g}{unit} "
                f"for axis={axis}; values below this are not observable by SONIC"
            )
        if delta > limit:
            raise ValueError(
                f"api.{method_name} abs(delta) must be <= {limit:g}{unit} for axis={axis}"
            )

    if method_name in {COORDINATED_ARC_TRANSPORT, COORDINATED_PIVOT}:
        radius = float(
            parameters.get("radius_m", parameters.get("swing_radius_m", 0.0))
        )
        angle = radians(abs(float(parameters["angle_deg"])))
        duration = float(parameters["duration_s"])
        command_speed = calibrated_arc_translation_speed(radius, angle, duration)
        if command_speed > SONIC_MAX_TRANSLATION_COMMAND_MPS + 1.0e-9:
            raise ValueError(
                f"api.{method_name} radius/angle/duration imply calibrated "
                f"{command_speed:.3f}m/s, above the "
                f"{SONIC_MAX_TRANSLATION_COMMAND_MPS:.2f}m/s SONIC command limit"
            )


def coordination_primitive_library_doc() -> str:
    """Render the model-facing layer between static skills and Code-as-Policy."""

    lines = [
        "# Coordination primitive library",
        "",
        "For loaded rigid-payload motion, select one coordination primitive per checkpointed "
        "step. Each front/back script still contains exactly one task-changing call followed by "
        "api.stop_all(). The host compiles the pair into one continuous, atomically published "
        "SONIC schedule; never compose primitive walk and turn calls in model code.",
        "The library is route-neutral. Infer the primitive, roles, direction, and bounded geometry "
        "from current physics, camera evidence, and verifier-grounded memory.",
        "",
    ]
    for index, method_name in enumerate(
        (
            COORDINATED_LINEAR_TRANSPORT,
            COORDINATED_PIVOT,
            COORDINATED_ALIGNMENT,
        ),
        start=1,
    ):
        spec = coordination_primitive_method_spec(method_name)
        modes = ", ".join(spec["coordination_modes"])
        lines.extend(
            [
                f"## {index}. api.{method_name}",
                str(spec["description"]),
                f"Allowed coordination_mode: {modes}.",
                f"Example: api.{spec['example']}",
                "Parameters:",
            ]
        )
        for name, parameter in spec["parameters"].items():
            if "choices" in parameter:
                bounds = " | ".join(parameter["choices"])
            else:
                lower = parameter.get("exclusive_min", parameter.get("min", "-inf"))
                upper = parameter.get("max", "inf")
                bounds = f"{lower}..{upper} {parameter.get('unit', '')}".strip()
            lines.append(f"- {name} ({bounds}): {parameter['description']}")
        lines.append("")
    lines.extend(
        [
            "Turn invariants:",
            "- clockwise means decreasing world yaw and counterclockwise means increasing world yaw when viewed from +Z/top-down;",
            "- front/back must use the same direction, angle, and duration;",
            "- coordinated_pivot requires exactly one stationary inner role and one forward-moving outer role;",
            "- coordinated linear transport may retreat both robots backward synchronously; "
            "pivot and longitudinal alignment never command either robot body-backward;",
            "- coordinated_arc_transport remains implemented for manual/ablation testing but is not exposed "
            "to the Agent action space.",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "COORDINATED_ALIGNMENT",
    "COORDINATED_ARC_TRANSPORT",
    "COORDINATED_LINEAR_TRANSPORT",
    "COORDINATED_PIVOT",
    "COORDINATION_PRIMITIVE_METHODS",
    "CALIBRATED_MODEL_PIVOT_DURATION_S",
    "LOADED_TRANSLATION_DIRECTIONS",
    "LONGITUDINAL_ALIGNMENT_TRIGGER_M",
    "MAX_LONGITUDINAL_ALIGNMENT_DELTA_M",
    "MIN_LONGITUDINAL_ALIGNMENT_DELTA_M",
    "MODEL_COORDINATION_PRIMITIVE_METHODS",
    "MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG",
    "SYNC_ONLY_STATIC_METHODS",
    "coordination_modes_for_method",
    "coordination_relationship_can_change",
    "coordination_role_variant_limit",
    "coordination_primitive_library_doc",
    "coordination_primitive_method_spec",
    "validate_coordination_primitive_parameters",
]
