"""Static planning validation and evidence-gated execution preflight."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isclose, pi
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Tuple

from ..contracts import ProtocolLock, SceneId
from .evidence import (
    EVIDENCE_CHECKS,
    EvidenceReceiptBinding,
    execution_evidence_fingerprint as compute_execution_evidence_fingerprint,
    validate_evidence_receipt,
)
from .loader import (
    compute_asset_composite_hash,
    compute_scene_hash,
    directed_route_document,
    load_scene,
)
from .schema import DirectedRouteSegment, EvidencePin, Pose, SceneSpec, Vec2, Vec3


EXACT_V83_BASE_ROUTE_CANONICAL_SHA256 = (
    "c189b4353798b364023f80935e48610bd841db25a688a95f05920ce35c568c50"
)
ROUTE_INITIAL_ALIGNMENT_TOLERANCE_M = 0.02


@dataclass(frozen=True)
class SceneValidationReport:
    scene_id: SceneId
    mode: str
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...] = ()
    execution_evidence_fingerprint: Optional[str] = None
    evidence_receipts: Tuple[EvidenceReceiptBinding, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def execution_ready(self) -> bool:
        """Whether this is a complete, host-derived formal execution report."""

        fingerprint = self.execution_evidence_fingerprint
        return (
            self.mode == "execution"
            and self.ok
            and isinstance(fingerprint, str)
            and len(fingerprint) == 64
            and all(character in "0123456789abcdef" for character in fingerprint)
            and tuple(sorted(item.check for item in self.evidence_receipts))
            == tuple(sorted(EVIDENCE_CHECKS))
            and all(item.raw_artifacts for item in self.evidence_receipts)
        )

    def require_ok(self) -> None:
        if not self.ok:
            raise ScenePreflightError(self)


class ScenePreflightError(RuntimeError):
    def __init__(self, report: SceneValidationReport):
        self.report = report
        joined = "; ".join(report.errors)
        super().__init__(f"{report.scene_id.value} {report.mode} validation failed: {joined}")


def validate_scene(
    scene: SceneSpec,
    mode: str = "planning",
    *,
    protocol_lock: Optional[ProtocolLock] = None,
    repo_root: Optional[Path] = None,
    scene_root: Optional[Path] = None,
) -> SceneValidationReport:
    """Validate a scene without pretending static checks are simulator proof.

    Planning mode validates schema-derived relationships, tracked source bytes,
    route connectivity, and controlled variant geometry.  Execution mode adds
    realized-asset and three simulator-evidence requirements.  It returns an
    explicit failed report when those artifacts have not been collected.
    """

    if mode not in {"planning", "execution"}:
        raise ValueError("mode must be 'planning' or 'execution'")
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    errors = list(_planning_errors(scene, root=root, scene_root=scene_root))
    warnings = []
    evidence_receipts: Tuple[EvidenceReceiptBinding, ...] = ()
    evidence_fingerprint: Optional[str] = None
    if mode == "execution":
        execution_errors, evidence_receipts = _execution_validation(
            scene,
            protocol_lock=protocol_lock,
            root=root,
        )
        errors.extend(execution_errors)
        if protocol_lock is not None and len(evidence_receipts) == len(EVIDENCE_CHECKS):
            evidence_fingerprint = compute_execution_evidence_fingerprint(
                scene,
                protocol_lock,
                evidence_receipts,
            )
    elif scene.evidence.missing_names():
        warnings.append(
            "planning-valid only; simulator evidence is incomplete and execution preflight will fail"
        )
    return SceneValidationReport(
        scene_id=scene.scene_id,
        mode=mode,
        errors=tuple(errors),
        warnings=tuple(warnings),
        execution_evidence_fingerprint=evidence_fingerprint,
        evidence_receipts=evidence_receipts,
    )


def _planning_errors(scene: SceneSpec, *, root: Path, scene_root: Optional[Path]) -> Iterable[str]:
    errors = []
    if compute_scene_hash(scene) != scene.scene_hash:
        errors.append("in-memory scene geometry does not match scene_hash")
    errors.extend(
        _repository_file_hash_errors(
            root,
            scene.asset.source_path,
            scene.asset.source_sha256,
            "source scene asset",
        )
    )

    route_segments = scene.route.segments
    segments = scene.corridor.segments
    if len(segments) != len(route_segments):
        errors.append("corridor and directed-route segment counts must match")
    else:
        for corridor_segment, route_segment in zip(segments, route_segments):
            if corridor_segment.segment_id != route_segment.segment_id:
                errors.append("corridor and directed-route segment IDs must match")
            if _vec2_distance(
                corridor_segment.start,
                Vec2(route_segment.start_x_m, route_segment.start_y_m),
            ) > ROUTE_INITIAL_ALIGNMENT_TOLERANCE_M:
                errors.append(
                    f"corridor segment {corridor_segment.segment_id} start does not "
                    "match directed route"
                )
            if not _vec2_close(
                corridor_segment.end,
                Vec2(route_segment.goal_x_m, route_segment.goal_y_m),
            ):
                errors.append(
                    f"corridor segment {corridor_segment.segment_id} end does not "
                    "match directed route"
                )
    first_route = route_segments[0]
    if _vec2_distance(
        Vec2(first_route.start_x_m, first_route.start_y_m),
        _xy(scene.initial_state.payload.position),
    ) > ROUTE_INITIAL_ALIGNMENT_TOLERANCE_M:
        errors.append("directed route must begin at the payload initial XY")
    final_route = route_segments[-1]
    if not _vec2_close(
        Vec2(final_route.goal_x_m, final_route.goal_y_m),
        _xy(scene.target.center),
    ):
        errors.append("directed route must end at the target center XY")

    if scene.scene_id is SceneId.BASE:
        encoded_route = json.dumps(
            directed_route_document(scene),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if hashlib.sha256(encoded_route).hexdigest() != (
            EXACT_V83_BASE_ROUTE_CANONICAL_SHA256
        ):
            errors.append(
                "base directed route differs from frozen exact V83 route contract"
            )

    if scene.scene_id is not SceneId.BASE:
        try:
            base = load_scene(SceneId.BASE, root=scene_root)
        except (OSError, ValueError) as exc:
            errors.append(f"cannot load base scene for controlled-variant validation: {exc}")
        else:
            if scene.scene_id is SceneId.MIRRORED:
                errors.extend(_mirror_errors(base, scene))
            elif scene.scene_id is SceneId.NARROW:
                errors.extend(_narrow_errors(base, scene))
            elif scene.scene_id is SceneId.ALTERED_TOPOLOGY:
                errors.extend(_altered_topology_errors(base, scene))
    return errors


def _execution_validation(
    scene: SceneSpec,
    *,
    protocol_lock: Optional[ProtocolLock],
    root: Path,
) -> tuple[list[str], Tuple[EvidenceReceiptBinding, ...]]:
    errors = []
    bindings: list[EvidenceReceiptBinding] = []
    if protocol_lock is None:
        errors.append("protocol lock is required for execution preflight")
    if (
        scene.asset.execution_path is None
        or scene.asset.execution_sha256 is None
        or scene.asset.execution_composite_sha256 is None
    ):
        errors.append(
            "realized execution asset is not pinned; generate the candidate asset and register its SHA-256"
        )
    else:
        errors.extend(
            _repository_file_hash_errors(
                root,
                scene.asset.execution_path,
                scene.asset.execution_sha256,
                "execution scene asset",
            )
        )
        computed_composite = compute_asset_composite_hash(scene.asset)
        if computed_composite != scene.asset.execution_composite_sha256:
            errors.append(
                "execution scene asset composite SHA-256 does not bind the registered "
                "source, realization parameters, and generated bytes"
            )
    for evidence_name in EVIDENCE_CHECKS:
        pin = getattr(scene.evidence, evidence_name)
        if pin is None:
            errors.append(f"missing simulator evidence: {evidence_name}")
        else:
            validation = validate_evidence_receipt(
                root,
                pin,
                evidence_name,
                scene,
                protocol_lock,
            )
            errors.extend(validation.errors)
            if validation.binding is not None:
                bindings.append(validation.binding)
    return errors, tuple(sorted(bindings, key=lambda item: item.check))


def _execution_errors(
    scene: SceneSpec,
    *,
    protocol_lock: Optional[ProtocolLock],
    root: Path,
) -> Iterable[str]:
    """Compatibility wrapper for callers that only need error strings."""

    return _execution_validation(scene, protocol_lock=protocol_lock, root=root)[0]


def _evidence_errors(
    root: Path,
    pin: EvidencePin,
    evidence_name: str,
    scene: SceneSpec,
    protocol_lock: Optional[ProtocolLock],
) -> Iterable[str]:
    return validate_evidence_receipt(
        root,
        pin,
        evidence_name,
        scene,
        protocol_lock,
    ).errors


def _mirror_errors(base: SceneSpec, mirrored: SceneSpec) -> Iterable[str]:
    errors = []
    for index, (base_robot, mirror_robot) in enumerate(
        zip(base.initial_state.robots, mirrored.initial_state.robots)
    ):
        if base_robot.robot_id != mirror_robot.robot_id:
            errors.append(f"mirrored robot {index} changes immutable identity")
        if not _pose_mirrored(base_robot.pose, mirror_robot.pose):
            errors.append(f"mirrored robot {base_robot.robot_id} pose is geometrically inconsistent")
    if not _pose_mirrored(base.initial_state.payload, mirrored.initial_state.payload):
        errors.append("mirrored payload pose is geometrically inconsistent")
    if not _vec3_mirrored(base.target.center, mirrored.target.center):
        errors.append("mirrored target center is geometrically inconsistent")
    if not isclose(base.target.radius_m, mirrored.target.radius_m, abs_tol=1e-9):
        errors.append("mirrored target radius changed")
    if not (
        isclose(mirrored.corridor.envelope_min.x, -base.corridor.envelope_max.x, abs_tol=1e-9)
        and isclose(mirrored.corridor.envelope_max.x, -base.corridor.envelope_min.x, abs_tol=1e-9)
        and isclose(mirrored.corridor.envelope_min.y, base.corridor.envelope_min.y, abs_tol=1e-9)
        and isclose(mirrored.corridor.envelope_max.y, base.corridor.envelope_max.y, abs_tol=1e-9)
    ):
        errors.append("mirrored corridor envelope is geometrically inconsistent")
    if len(base.corridor.segments) != len(mirrored.corridor.segments):
        errors.append("mirrored corridor segment count changed")
    else:
        for base_segment, mirror_segment in zip(base.corridor.segments, mirrored.corridor.segments):
            if base_segment.segment_id != mirror_segment.segment_id:
                errors.append("mirrored corridor segment IDs changed")
            if not _vec2_mirrored(base_segment.start, mirror_segment.start):
                errors.append(f"mirrored segment {base_segment.segment_id} start is inconsistent")
            if not _vec2_mirrored(base_segment.end, mirror_segment.end):
                errors.append(f"mirrored segment {base_segment.segment_id} end is inconsistent")
            if not isclose(base_segment.clear_width_m, mirror_segment.clear_width_m, abs_tol=1e-9):
                errors.append(f"mirrored segment {base_segment.segment_id} width changed")
    if len(base.route.segments) != len(mirrored.route.segments):
        errors.append("mirrored directed-route segment count changed")
    else:
        for base_route, mirror_route in zip(
            base.route.segments, mirrored.route.segments
        ):
            errors.extend(_mirrored_directed_segment_errors(base_route, mirror_route))
    if (
        base.route.schema_version != mirrored.route.schema_version
        or base.route.kind != mirrored.route.kind
        or base.route.agent_visible != mirrored.route.agent_visible
        or base.route.host_validated_reference
        != mirrored.route.host_validated_reference
        or base.route.host_selects_primitive
        != mirrored.route.host_selects_primitive
        or base.route.host_selects_coordination_mode
        != mirrored.route.host_selects_coordination_mode
        or base.route.host_selects_robot_roles
        != mirrored.route.host_selects_robot_roles
        or base.route.host_selects_bounded_parameters
        != mirrored.route.host_selects_bounded_parameters
    ):
        errors.append("mirrored route changed the V83 visibility or authority contract")
    if base.route.mission_topology != mirrored.route.mission_topology:
        errors.append("mirrored route changed the abstract mission topology")
    if len(base.route.progress_gates) != len(mirrored.route.progress_gates):
        errors.append("mirrored progress gate count changed")
    else:
        for base_gate, mirror_gate in zip(
            base.route.progress_gates, mirrored.route.progress_gates
        ):
            if base_gate.phase_name != mirror_gate.phase_name:
                errors.append("mirrored progress gate phase names changed")
            if not (
                isclose(mirror_gate.normal_x, -base_gate.normal_x, abs_tol=1e-9)
                and isclose(mirror_gate.normal_y, base_gate.normal_y, abs_tol=1e-9)
                and isclose(mirror_gate.threshold_m, base_gate.threshold_m, abs_tol=1e-9)
            ):
                errors.append(
                    f"mirrored progress gate {base_gate.phase_name} is geometrically inconsistent"
                )
    for name in (
        "global_eye",
        "global_target",
        "global_up",
        "payload_eye_offset",
        "payload_target_offset",
    ):
        if not _vec3_mirrored(getattr(base.camera, name), getattr(mirrored.camera, name)):
            errors.append(f"mirrored camera field {name} is geometrically inconsistent")
    return errors


def _narrow_errors(base: SceneSpec, narrow: SceneSpec) -> Iterable[str]:
    errors = []
    if base.initial_state != narrow.initial_state:
        errors.append("narrow variant must preserve the initial state")
    if base.target != narrow.target:
        errors.append("narrow variant must preserve target geometry")
    if (
        base.route.schema_version != narrow.route.schema_version
        or base.route.kind != narrow.route.kind
        or base.route.agent_visible != narrow.route.agent_visible
        or base.route.host_validated_reference != narrow.route.host_validated_reference
        or base.route.host_selects_primitive != narrow.route.host_selects_primitive
        or base.route.host_selects_coordination_mode
        != narrow.route.host_selects_coordination_mode
        or base.route.host_selects_robot_roles != narrow.route.host_selects_robot_roles
        or base.route.host_selects_bounded_parameters
        != narrow.route.host_selects_bounded_parameters
        or base.route.mission_topology != narrow.route.mission_topology
        or base.route.progress_gates != narrow.route.progress_gates
    ):
        errors.append(
            "narrow variant must preserve route topology, gates, and authority"
        )
    if len(base.corridor.segments) != len(narrow.corridor.segments):
        errors.append("narrow variant must preserve corridor segment count")
    else:
        reductions = []
        for base_segment, narrow_segment in zip(base.corridor.segments, narrow.corridor.segments):
            if (
                base_segment.segment_id != narrow_segment.segment_id
                or base_segment.start != narrow_segment.start
                or base_segment.end != narrow_segment.end
            ):
                errors.append("narrow variant may change widths, not segment topology")
            if narrow_segment.clear_width_m > base_segment.clear_width_m:
                errors.append(f"narrow segment {base_segment.segment_id} became wider")
            reductions.append(narrow_segment.clear_width_m < base_segment.clear_width_m)
        if not any(reductions):
            errors.append("narrow variant does not reduce any corridor width")
    if len(base.route.segments) != len(narrow.route.segments):
        errors.append("narrow variant must preserve directed-route segment count")
    elif len(base.corridor.segments) == len(narrow.corridor.segments):
        for base_route, narrow_route, base_corridor, narrow_corridor in zip(
            base.route.segments,
            narrow.route.segments,
            base.corridor.segments,
            narrow.corridor.segments,
        ):
            expected_half_width = base_route.half_width_m - 0.5 * (
                base_corridor.clear_width_m - narrow_corridor.clear_width_m
            )
            unchanged_fields = (
                "segment_id",
                "direction_x",
                "direction_y",
                "start_x_m",
                "start_y_m",
                "goal_x_m",
                "goal_y_m",
                "center_x_m",
                "center_y_m",
                "detour_margin_m",
            )
            if any(
                getattr(base_route, field) != getattr(narrow_route, field)
                for field in unchanged_fields
            ):
                errors.append(
                    f"narrow directed segment {base_route.segment_id} may change only "
                    "half_width_m"
                )
            if not isclose(
                narrow_route.half_width_m,
                expected_half_width,
                abs_tol=1e-9,
            ):
                errors.append(
                    f"narrow directed segment {base_route.segment_id} half-width must "
                    "track half of the registered clear-width reduction"
                )
    return errors


def _altered_topology_errors(base: SceneSpec, altered: SceneSpec) -> Iterable[str]:
    errors = []
    if base.initial_state != altered.initial_state:
        errors.append("altered-topology variant must preserve the initial state")
    if base.route == altered.route:
        errors.append("altered-topology variant must change the route")
    if base.target == altered.target:
        errors.append("altered-topology variant must change target geometry")
    if len(base.route.segments) == len(altered.route.segments):
        errors.append("altered-topology route must have a different segment count")
    return errors


def _mirrored_directed_segment_errors(
    base: DirectedRouteSegment,
    mirrored: DirectedRouteSegment,
) -> list[str]:
    errors: list[str] = []
    if base.segment_id != mirrored.segment_id:
        errors.append("mirrored directed-route segment IDs changed")
    reflected_fields = {
        "direction_x": -base.direction_x,
        "direction_y": base.direction_y,
        "start_x_m": -base.start_x_m,
        "start_y_m": base.start_y_m,
        "goal_x_m": -base.goal_x_m,
        "goal_y_m": base.goal_y_m,
        "center_x_m": -base.center_x_m,
        "center_y_m": base.center_y_m,
        "half_width_m": base.half_width_m,
        "detour_margin_m": base.detour_margin_m,
    }
    if any(
        not isclose(getattr(mirrored, field), expected, abs_tol=1e-9)
        for field, expected in reflected_fields.items()
    ):
        errors.append(
            f"mirrored directed segment {base.segment_id} is not a strict x reflection"
        )
    return errors


def _repository_file_hash_errors(
    root: Path,
    relative_path: str,
    expected: str,
    label: str,
) -> Iterable[str]:
    """Hash one real repository descendant without following any symlink."""

    root_path = Path(root)
    if root_path.is_symlink():
        return [f"{label} repository root must not be a symlink: {root_path}"]
    try:
        resolved_root = root_path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return [f"{label} repository root is missing: {root_path}"]
    if not resolved_root.is_dir():
        return [f"{label} repository root is not a directory: {root_path}"]

    path = root_path.joinpath(*PurePosixPath(relative_path).parts)
    current = root_path
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            return [f"{label} must not traverse a symlink: {current}"]

    try:
        resolved_path = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return [f"{label} is missing: {path}"]
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return [f"{label} escapes repository root: {relative_path}"]
    if not resolved_path.is_file():
        return [f"{label} is missing: {path}"]
    try:
        content = resolved_path.read_bytes()
    except OSError as exc:
        return [f"{label} is unreadable: {exc}"]
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected:
        return [f"{label} SHA-256 mismatch: expected {expected}, got {digest}"]
    return []


def _pose_mirrored(base: Pose, mirrored: Pose) -> bool:
    expected_yaw = (pi - base.yaw_rad) % (2.0 * pi)
    actual_yaw = mirrored.yaw_rad % (2.0 * pi)
    return _vec3_mirrored(base.position, mirrored.position) and isclose(
        expected_yaw, actual_yaw, abs_tol=1e-9
    )


def _vec2_mirrored(base: Vec2, mirrored: Vec2) -> bool:
    return isclose(mirrored.x, -base.x, abs_tol=1e-9) and isclose(
        mirrored.y, base.y, abs_tol=1e-9
    )


def _vec3_mirrored(base: Vec3, mirrored: Vec3) -> bool:
    return (
        isclose(mirrored.x, -base.x, abs_tol=1e-9)
        and isclose(mirrored.y, base.y, abs_tol=1e-9)
        and isclose(mirrored.z, base.z, abs_tol=1e-9)
    )


def _vec2_close(left: Vec2, right: Vec2) -> bool:
    return isclose(left.x, right.x, abs_tol=1e-9) and isclose(left.y, right.y, abs_tol=1e-9)


def _vec2_distance(left: Vec2, right: Vec2) -> float:
    return ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5


def _xy(value: Vec3) -> Vec2:
    return Vec2(value.x, value.y)


__all__ = [
    "ScenePreflightError",
    "SceneValidationReport",
    "validate_scene",
]
