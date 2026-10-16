"""Qwen-VL safety observation and global-route criticism for box carrying."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from math import degrees, pi, radians
from pathlib import Path
import re
from typing import Any, Protocol

from praxis_coop.configs import DEFAULT_MULTIMODAL_MODEL
from praxis_coop.llm import QwenVisionClient

from .contracts import (
    BoxCarryTask,
    EvaluationReport,
    ExecutionTrace,
    FailureType,
    VisualObservation,
)
from .model_role_pipeline import ModelRolePipeline
from .prompts import (
    VISUAL_EVALUATOR_SYSTEM_PROMPT,
    VISUAL_ROUTE_CRITIC_SYSTEM_PROMPT,
    build_visual_evaluator_user_prompt,
    build_visual_route_critic_user_prompt,
)


_TARGET_HEADING_RELATIONS = {"toward", "away", "crosswise", "mixed", "uncertain"}
_FREE_SPACE_HEADING_RELATIONS = {
    "aligned_with_open_passage",
    "toward_wall_or_blocked_space",
    "crosswise_to_passage",
    "mixed",
    "uncertain",
}
_ROUTE_DIRECTION_RISK_BASES = {
    "none",
    "heading_only",
    "executed_motion_into_blocked_space",
    "unexplained_retreat",
    "uncertain",
}
_PASSAGE_POSTURES = {
    "longitudinal_aligned",
    "lateral_aligned",
    "diagonal_or_misaligned",
    "mixed",
    "uncertain",
}


class VisionGenerationClient(Protocol):
    """Minimal image-and-text model interface used by both visual passes."""

    def analyze_image(
        self,
        *,
        system: str,
        user: str,
        image_path: str | Path,
    ) -> str:
        ...


@dataclass(frozen=True)
class VisualVerifierReport:
    """Structured visual evidence consumed by the physics evaluator and Agent."""

    status: str
    frame_path: str = ""
    model_name: str = ""
    summary: str = ""
    failure_type: FailureType = "none"
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendations: tuple[str, ...] = ()

    @property
    def blocks_success(self) -> bool:
        return self.status != "ok" or (
            self.failure_type != "none" and self.confidence >= 0.55
        )


@dataclass
class QwenVLBoxCarryVisualVerifier:
    """Run visual evaluation and failure analysis through one loaded VLM."""

    client: VisionGenerationClient
    model_name: str = DEFAULT_MULTIMODAL_MODEL
    min_confidence: float = 0.55
    role_pipeline: ModelRolePipeline | None = None

    def preload(self) -> None:
        preload = getattr(self.client, "preload", None)
        if callable(preload):
            preload()

    def evaluate(
        self,
        *,
        task: BoxCarryTask,
        trace: ExecutionTrace,
        visual: VisualObservation,
        runtime_state: dict[str, Any],
        before_state: dict[str, Any] | None = None,
        physics_report: EvaluationReport | dict[str, Any] | None = None,
    ) -> VisualVerifierReport:
        if not visual.frame_path:
            return _missing_frame_report(
                status="missing_frame",
                model_name=self.model_name,
                visual=visual,
            )
        path = Path(visual.frame_path)
        if not path.exists():
            return _missing_frame_report(
                status="missing_frame_file",
                model_name=self.model_name,
                visual=visual,
                frame_path=str(path),
            )

        analyzer_context = {
            "task_id": task.task_id,
            "payload_id": task.payload_id,
            "robot_ids": list(task.robot_ids),
            "final_target": (
                "orange-red cylindrical pedestal; placement is complete only "
                "when the box visibly rests on it"
            ),
            "last_script_id": trace.script_id,
            "last_skill_calls": [
                {
                    "skill": item.skill_name,
                    "status": item.status,
                    "robot_ids": list(item.robot_ids),
                }
                for item in trace.skill_calls
            ],
            "physics_evidence": _compact_physics_evidence(
                runtime_state=runtime_state,
                before_state=before_state,
                physics_report=physics_report,
            ),
            "required_route_fields": [
                "target_visible",
                "target_relation_to_team",
                "team_heading_relative_to_target",
                "team_heading_relative_to_visible_free_space",
                "team_payload_posture_relative_to_visible_passage",
                "visible_free_space_connectivity",
                "passage_geometry_observations",
                "team_footprint_clearance",
                "relative_robot_layout",
                "payload_axis_relative_to_passage",
                "immediate_route_direction_risk",
                "route_direction_risk_basis",
                "route_direction_evidence",
            ],
        }
        if self.role_pipeline is not None:
            safety = self.role_pipeline.analyze_rollout(
                analyzer_context,
                artifact_id=trace.script_id or f"attempt_{trace.attempt:02d}",
                image_path=path,
            )
            safety_raw = json.dumps(safety, ensure_ascii=False)
        else:
            safety_raw = self.client.analyze_image(
                system=VISUAL_EVALUATOR_SYSTEM_PROMPT,
                user=build_visual_evaluator_user_prompt(
                    {
                        "task_id": task.task_id,
                        "payload_id": task.payload_id,
                        "robot_ids": list(task.robot_ids),
                        "final_target": analyzer_context["final_target"],
                        "last_script_id": trace.script_id,
                        "last_skill_calls": analyzer_context["last_skill_calls"],
                    }
                ),
                image_path=path,
            )
            safety = _parse_json(safety_raw)
        confidence = _clamped_confidence(safety.get("confidence"))
        scene_visible = _bool_value(safety.get("scene_visible", True))
        payload_on_floor = _bool_value(
            safety.get("payload_on_floor", safety.get("payload_dropped", False))
        )
        robots_standing_stably = _bool_value(
            safety.get(
                "robots_standing_stably",
                not _bool_value(safety.get("robot_fallen_or_unstable", False)),
            )
        )
        wall_collision_or_stuck = _bool_value(
            safety.get(
                "wall_collision_or_stuck",
                safety.get("payload_wall_contact_or_high_risk", False),
            )
        )
        payload_placed_on_target = _bool_value(
            safety.get("payload_placed_on_target", False)
        )
        model_failure_type = _failure_type(
            str(safety.get("failure_type", "none"))
        )
        failure_type = _visual_failure_type(
            model_failure_type,
            scene_visible=scene_visible,
            payload_on_floor=payload_on_floor,
            robots_standing_stably=robots_standing_stably,
            wall_collision_or_stuck=wall_collision_or_stuck,
        )
        physics_override = _physics_conflict_override(
            failure_type=failure_type,
            runtime_state=runtime_state,
        )
        if physics_override:
            failure_type = "none"
        if (
            failure_type != "manual_review_required"
            and confidence < self.min_confidence
        ):
            failure_type = "none"

        rollout_motion_hint = _rollout_motion_hint(before_state, runtime_state)
        route_raw = ""
        route: dict[str, Any] = {}
        route_error = ""
        if self.role_pipeline is not None:
            route_raw = safety_raw
            route = safety
        elif scene_visible and failure_type == "none":
            try:
                route_raw = self.client.analyze_image(
                    system=VISUAL_ROUTE_CRITIC_SYSTEM_PROMPT,
                    user=build_visual_route_critic_user_prompt(
                        {
                            "task_id": task.task_id,
                            "final_target": "orange-red cylindrical pedestal",
                            "last_script_id": trace.script_id,
                            "last_skill_calls": [
                                {
                                    "skill": item.skill_name,
                                    "status": item.status,
                                }
                                for item in trace.skill_calls
                            ],
                            "safety_observation": {
                                "summary": str(safety.get("summary", "")).strip(),
                                "payload_on_floor": payload_on_floor,
                                "robots_standing_stably": robots_standing_stably,
                                "wall_collision_or_stuck": wall_collision_or_stuck,
                            },
                            "rollout_motion_hint": rollout_motion_hint,
                        }
                    ),
                    image_path=path,
                )
                route = _parse_json(route_raw)
            except Exception as exc:  # A route-pass outage cannot erase safety facts.
                route_error = f"{type(exc).__name__}: {exc}"

        route_data = {**safety, **route}
        original_posture = _passage_posture(
            safety.get(
                "team_payload_posture_relative_to_visible_passage",
                safety.get("posture", "uncertain"),
            )
        )
        posture = _passage_posture(
            route.get(
                "team_payload_posture_relative_to_visible_passage",
                route.get(
                    "posture",
                    route_data.get(
                        "team_payload_posture_relative_to_visible_passage",
                        original_posture,
                    ),
                ),
            )
        )
        heading_consistency = _robot_heading_consistency(runtime_state)
        passage_posture_recheck = {
            "requested": bool(
                heading_consistency.get("headings_agree", False)
                and original_posture in {"mixed", "diagonal_or_misaligned", "uncertain"}
            ),
            "accepted": bool(
                heading_consistency.get("headings_agree", False)
                and original_posture in {"mixed", "diagonal_or_misaligned", "uncertain"}
                and posture in {"longitudinal_aligned", "lateral_aligned"}
            ),
            "original_posture": original_posture,
            "rechecked_posture": posture,
            "robot_heading_consistency": heading_consistency,
        }

        route_confidence = _clamped_confidence(
            route_data.get("confidence", confidence)
        )
        normalized_route, route_normalization = _normalize_route_risk(
            route_data,
            rollout_motion_hint=rollout_motion_hint,
        )
        route_direction_risk_confirmed = _confirmed_route_direction_risk(
            normalized_route,
            confidence=route_confidence,
            min_confidence=self.min_confidence,
            rollout_motion_hint=rollout_motion_hint,
        )
        summary = _agent_visual_summary(
            safety=safety,
            route=normalized_route,
        )

        evidence: dict[str, Any] = {
            "raw_model_output": safety_raw,
            "raw_route_model_output": route_raw,
            "route_critic_error": route_error,
            "model_failure_type": model_failure_type,
            "physics_override": physics_override,
            "scene_visible": scene_visible,
            "payload_on_floor": payload_on_floor,
            "robots_standing_stably": robots_standing_stably,
            "wall_collision_or_stuck": wall_collision_or_stuck,
            "payload_placed_on_target": payload_placed_on_target,
            "visible_evidence": _string_list(
                safety.get("visible_evidence", safety.get("visible_issues", ()))
            ),
            "payload_on_arms": safety.get("payload_on_arms"),
            "both_robots_supporting_payload": safety.get(
                "both_robots_supporting_payload"
            ),
            "payload_dropped": payload_on_floor,
            "payload_wall_contact_or_high_risk": _bool_value(
                safety.get("payload_wall_contact_or_high_risk", False)
            ),
            "robot_fallen_or_unstable": not robots_standing_stably,
            "target_visible": normalized_route.get("target_visible"),
            "target_relation_to_team": normalized_route.get(
                "target_relation_to_team", "uncertain"
            ),
            "team_heading_relative_to_target": _enum_value(
                normalized_route.get("team_heading_relative_to_target"),
                _TARGET_HEADING_RELATIONS,
            ),
            "team_heading_relative_to_visible_free_space": _enum_value(
                normalized_route.get(
                    "team_heading_relative_to_visible_free_space"
                ),
                _FREE_SPACE_HEADING_RELATIONS,
            ),
            "team_payload_posture_relative_to_visible_passage": posture,
            "passage_posture_recheck": passage_posture_recheck,
            "immediate_route_direction_risk": _bool_value(
                normalized_route.get("immediate_route_direction_risk", False)
            ),
            "route_direction_risk_basis": _enum_value(
                normalized_route.get("route_direction_risk_basis"),
                _ROUTE_DIRECTION_RISK_BASES,
            ),
            "route_direction_risk_confirmed": route_direction_risk_confirmed,
            "route_direction_risk_normalization": route_normalization,
            "route_direction_evidence": str(
                normalized_route.get("route_direction_evidence", "")
            ).strip(),
            "rollout_motion_hint": rollout_motion_hint,
            "visible_free_space_connectivity": _string_list(
                normalized_route.get("visible_free_space_connectivity", ())
            ),
            "passage_geometry_observations": _string_list(
                normalized_route.get("passage_geometry_observations", ())
            ),
            "team_footprint_clearance": _enum_value(
                normalized_route.get("team_footprint_clearance"),
                {"clear", "tight", "blocked", "uncertain"},
            ),
            "relative_robot_layout": str(
                normalized_route.get("relative_robot_layout", "uncertain")
            ).strip(),
            "payload_axis_relative_to_passage": str(
                normalized_route.get(
                    "payload_axis_relative_to_passage",
                    "uncertain",
                )
            ).strip(),
            "observations": _string_list(normalized_route.get("observations", ())),
            "risks": _string_list(normalized_route.get("risks", ())),
            "possible_issues": _string_list(
                normalized_route.get("possible_issues", ())
            ),
            "visible_issues": _string_list(safety.get("visible_issues", ())),
            "failure_analysis": _normalized_failure_analysis(
                safety.get("failure_analysis", {})
            ),
        }
        return VisualVerifierReport(
            status="ok",
            frame_path=str(path),
            model_name=self.model_name,
            summary=summary,
            failure_type=failure_type,
            confidence=confidence,
            evidence=evidence,
            recommendations=(),
        )


def make_qwen_vl_visual_verifier(
    *,
    model_name: str,
    device_map: str,
    max_gpu_memory: str | None,
    max_cpu_memory: str | None,
    quantization: str = "none",
    temperature: float,
    max_new_tokens: int,
) -> QwenVLBoxCarryVisualVerifier:
    client = QwenVisionClient(
        model_name=model_name,
        device_map=device_map,
        max_gpu_memory=max_gpu_memory,
        max_cpu_memory=max_cpu_memory,
        quantization=quantization,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return QwenVLBoxCarryVisualVerifier(client=client, model_name=model_name)


def _missing_frame_report(
    *,
    status: str,
    model_name: str,
    visual: VisualObservation,
    frame_path: str = "",
) -> VisualVerifierReport:
    return VisualVerifierReport(
        status=status,
        frame_path=frame_path,
        model_name=model_name,
        summary="The required external-camera frame is unavailable.",
        failure_type="manual_review_required",
        confidence=1.0,
        evidence={"visual_observation": asdict(visual)},
    )


def _failure_type(value: str) -> FailureType:
    aliases: dict[str, FailureType] = {
        "no failure": "none",
        "no_failure": "none",
        "success": "none",
        "ok": "none",
        "wall_collision_or_stuck": "payload_wall_contact",
        "payload_wall_contact_or_high_risk": "payload_wall_contact",
    }
    normalized = aliases.get(str(value or "none").strip().lower(), str(value).strip().lower())
    allowed: set[FailureType] = {
        "none",
        "robot_fall",
        "payload_drop",
        "payload_not_supported",
        "payload_unstable",
        "payload_wall_contact",
        "payload_not_on_target",
        "robots_not_clear",
        "manual_review_required",
    }
    return normalized if normalized in allowed else "manual_review_required"  # type: ignore[return-value]


def _visual_failure_type(
    model_failure_type: FailureType,
    *,
    scene_visible: bool,
    payload_on_floor: bool,
    robots_standing_stably: bool,
    wall_collision_or_stuck: bool,
) -> FailureType:
    if not scene_visible:
        return "manual_review_required"
    if payload_on_floor:
        return "payload_drop"
    if not robots_standing_stably:
        return "robot_fall"
    if wall_collision_or_stuck:
        return "payload_wall_contact"
    return model_failure_type


def _physics_conflict_override(
    *,
    failure_type: FailureType,
    runtime_state: dict[str, Any],
) -> str:
    """Prevent image ambiguity from overriding fresh exact simulator state."""

    payload = runtime_state.get("payload", {})
    if not isinstance(payload, dict):
        return ""
    contacts = payload.get("contacts", {})
    if not isinstance(contacts, dict):
        return ""
    if failure_type == "payload_wall_contact" and not tuple(contacts.get("walls", ())):
        return (
            "fresh MuJoCo contact state reports no payload-wall contact; visual proximity "
            "is retained as route evidence instead of a rollback condition"
        )
    if failure_type == "robot_fall":
        robots = runtime_state.get("robots", {})
        if isinstance(robots, dict) and robots:
            pelvis_heights = [
                _float(robot.get("pelvis_height_m"), 0.0)
                for robot in robots.values()
                if isinstance(robot, dict)
            ]
            control_health = runtime_state.get("control_health", {})
            control_healthy = (
                bool(control_health.get("healthy", False))
                if isinstance(control_health, dict)
                else False
            )
            if pelvis_heights and min(pelvis_heights) >= 0.35 and control_healthy:
                return (
                    "fresh MuJoCo pelvis heights and SONIC LowState health confirm both "
                    "robots are upright; the visual instability flag is advisory"
                )
    if failure_type == "payload_unstable":
        supporters = set(contacts.get("support_robots", ()))
        if (
            bool(payload.get("active", False))
            and {"front", "back"}.issubset(supporters)
            and not bool(contacts.get("floor", False))
        ):
            return (
                "fresh MuJoCo state confirms dual hand/forearm support and no floor "
                "contact; visual tilt alone is advisory"
            )
    target_supported = (
        bool(payload.get("active", False))
        and bool(contacts.get("target", False))
        and not bool(contacts.get("floor", False))
        and not bool(contacts.get("walls", ()))
        and _float(payload.get("tilt_deg"), 180.0) <= 8.0
        and _float(payload.get("linear_speed_mps"), 1.0) <= 0.05
        and _float(payload.get("angular_speed_radps"), 1.0) <= 0.10
    )
    if target_supported and failure_type in {
        "payload_drop",
        "payload_not_supported",
        "payload_not_on_target",
    }:
        return (
            "fresh MuJoCo contact state confirms target support after release; "
            "missing palm contact is intentional"
        )
    if failure_type not in {"payload_drop", "payload_not_supported"}:
        return ""
    supporters = set(contacts.get("support_robots", ()))
    physics_confirms_support = (
        bool(payload.get("active", False))
        and {"front", "back"}.issubset(supporters)
        and not bool(contacts.get("floor", False))
        and _float(payload.get("z"), 0.0) >= 0.80
    )
    if not physics_confirms_support:
        return ""
    return (
        "fresh MuJoCo state confirms active dual hand/forearm support, safe height, "
        "and no floor contact; visual support suspicion is treated as occlusion"
    )


def _normalize_route_risk(
    payload: dict[str, Any],
    *,
    rollout_motion_hint: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    normalized = dict(payload)
    immediate = _bool_value(normalized.get("immediate_route_direction_risk", False))
    basis = _enum_value(
        normalized.get("route_direction_risk_basis"),
        _ROUTE_DIRECTION_RISK_BASES,
    )
    free_space_relation = _enum_value(
        normalized.get("team_heading_relative_to_visible_free_space"),
        _FREE_SPACE_HEADING_RELATIONS,
    )
    clearance = _enum_value(
        normalized.get("team_footprint_clearance"),
        {"clear", "tight", "blocked", "uncertain"},
    )
    direction_evidence = str(
        normalized.get("route_direction_evidence", "")
    ).strip()
    motion_available = bool(rollout_motion_hint.get("available", False))
    motion = str(rollout_motion_hint.get("payload_screen_motion", ""))
    contradiction = bool(
        not immediate
        and basis == "none"
        and free_space_relation == "toward_wall_or_blocked_space"
        and clearance in {"tight", "blocked"}
        and direction_evidence
        and motion_available
        and motion not in {"", "stationary"}
    )
    if contradiction:
        normalized["immediate_route_direction_risk"] = True
        normalized["route_direction_risk_basis"] = (
            "executed_motion_into_blocked_space"
        )
        return (
            normalized,
            (
                "contradictory VLM fields normalized from visible blocked-space "
                "heading, tight clearance, stated evidence, and measured rollout motion"
            ),
        )
    normalized["immediate_route_direction_risk"] = immediate
    normalized["route_direction_risk_basis"] = basis
    return normalized, ""


def _confirmed_route_direction_risk(
    payload: dict[str, Any],
    *,
    confidence: float,
    min_confidence: float,
    rollout_motion_hint: dict[str, Any],
) -> bool:
    risk_basis = str(payload.get("route_direction_risk_basis", "uncertain"))
    motion_direction = str(rollout_motion_hint.get("payload_screen_motion", ""))
    return bool(
        payload.get("immediate_route_direction_risk", False)
        and confidence >= min_confidence
        and risk_basis
        in {"executed_motion_into_blocked_space", "unexplained_retreat"}
        and bool(str(payload.get("route_direction_evidence", "")).strip())
        and bool(rollout_motion_hint.get("available", False))
        and motion_direction not in {"", "stationary"}
    )


def _rollout_motion_hint(
    before_state: dict[str, Any] | None,
    after_state: dict[str, Any],
) -> dict[str, Any]:
    """Project measured payload motion into the fixed offscreen global view."""

    before_payload = (
        before_state.get("payload", {})
        if isinstance(before_state, dict)
        else {}
    )
    after_payload = (
        after_state.get("payload", {})
        if isinstance(after_state, dict)
        else {}
    )
    if not isinstance(before_payload, dict) or not isinstance(after_payload, dict):
        return {"available": False}
    try:
        dx = float(after_payload["x"]) - float(before_payload["x"])
        dy = float(after_payload["y"]) - float(before_payload["y"])
    except (KeyError, TypeError, ValueError):
        return {"available": False}
    displacement = (dx * dx + dy * dy) ** 0.5
    if displacement < 0.01:
        screen_direction = "stationary"
    else:
        screen_dx = -dx
        screen_dy = dy
        horizontal = "right" if screen_dx >= 0.0 else "left"
        vertical = "down" if screen_dy >= 0.0 else "up"
        major = max(abs(screen_dx), abs(screen_dy))
        minor = min(abs(screen_dx), abs(screen_dy))
        if minor >= 0.35 * major:
            screen_direction = f"{vertical}-{horizontal}"
        elif abs(screen_dx) >= abs(screen_dy):
            screen_direction = horizontal
        else:
            screen_direction = vertical
    return {
        "available": True,
        "payload_screen_motion": screen_direction,
        "payload_displacement_m": round(displacement, 4),
        "source": "fresh MuJoCo pre/post displacement projected into the fixed global view",
        "interpretation": "measured rollout result, not a desired route",
    }


def _agent_visual_summary(
    *,
    safety: dict[str, Any],
    route: dict[str, Any],
) -> str:
    scene_visible = _bool_value(safety.get("scene_visible", True))
    if not scene_visible:
        return "The complete global scene is not visible enough for visual evaluation."

    payload_on_floor = _bool_value(safety.get("payload_on_floor", False))
    robots_stable = _bool_value(safety.get("robots_standing_stably", True))
    wall_collision = _bool_value(safety.get("wall_collision_or_stuck", False))
    payload_on_target = _bool_value(safety.get("payload_placed_on_target", False))
    parts = [
        (
            "Global visual safety: "
            f"payload_on_floor={payload_on_floor}; "
            f"robots_standing_stably={robots_stable}; "
            f"wall_collision_or_stuck={wall_collision}; "
            f"payload_placed_on_target={payload_on_target}."
        )
    ]
    target_visible = route.get("target_visible")
    target_relation = str(route.get("target_relation_to_team", "")).strip()
    if target_visible is not None:
        target_fact = f"Global target visible={bool(target_visible)}"
        if target_relation:
            target_fact += f"; image-relative relation: {target_relation}"
        parts.append(target_fact + ".")
    connectivity = _string_list(route.get("visible_free_space_connectivity", ()))
    if connectivity:
        parts.append("Visible free-space connectivity: " + "; ".join(connectivity) + ".")
    return " ".join(part for part in parts if part)


def _passage_posture(value: object) -> str:
    normalized = str(value or "uncertain").strip().lower()
    aliases = {
        "diagonal": "diagonal_or_misaligned",
        "misaligned": "diagonal_or_misaligned",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in _PASSAGE_POSTURES else "uncertain"


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                payload, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            break
        else:
            return {
                "summary": raw[:1000],
                "scene_visible": False,
                "failure_type": "manual_review_required",
                "confidence": 1.0,
                "visible_evidence": ["visual model did not return valid JSON"],
            }
    return payload if isinstance(payload, dict) else {}


def _clamped_confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _enum_value(value: object, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else "uncertain"


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:12]


def _robot_heading_consistency(runtime_state: dict[str, Any]) -> dict[str, Any]:
    robots = runtime_state.get("robots", {})
    robots = robots if isinstance(robots, dict) else {}
    yaw_by_robot: dict[str, float] = {}
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        robot = robot if isinstance(robot, dict) else {}
        if isinstance(robot.get("yaw_rad"), (int, float)):
            yaw_by_robot[robot_id] = float(robot["yaw_rad"])
        elif isinstance(robot.get("yaw_deg"), (int, float)):
            yaw_by_robot[robot_id] = radians(float(robot["yaw_deg"]))
    if set(yaw_by_robot) != {"front", "back"}:
        return {"available": False, "headings_agree": False}
    difference_rad = abs(
        (yaw_by_robot["front"] - yaw_by_robot["back"] + pi) % (2.0 * pi) - pi
    )
    difference_deg = degrees(difference_rad)
    return {
        "available": True,
        "headings_agree": difference_deg <= 15.0,
        "difference_deg": difference_deg,
    }


def _compact_physics_evidence(
    *,
    runtime_state: dict[str, Any],
    before_state: dict[str, Any] | None,
    physics_report: EvaluationReport | dict[str, Any] | None,
) -> dict[str, Any]:
    """Expose decisive physics facts without dumping the simulator state."""

    payload = runtime_state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    robots = runtime_state.get("robots", {})
    robots = robots if isinstance(robots, dict) else {}
    compact_robots = {}
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        robot = robot if isinstance(robot, dict) else {}
        compact_robots[robot_id] = {
            key: robot.get(key)
            for key in (
                "x",
                "y",
                "z",
                "yaw_rad",
                "yaw_deg",
                "pelvis_height_m",
            )
            if key in robot
        }
    before_payload = (
        before_state.get("payload", {})
        if isinstance(before_state, dict)
        and isinstance(before_state.get("payload"), dict)
        else {}
    )
    report_payload: dict[str, Any] = {}
    if isinstance(physics_report, EvaluationReport):
        report_payload = {
            "success": physics_report.success,
            "failure_type": physics_report.failure_type,
            "failure_level": physics_report.failure_level,
            "diagnosis": physics_report.diagnosis,
            "failed_subtask": physics_report.failed_subtask,
            "metrics": dict(physics_report.metrics),
            "recommendations": list(physics_report.recommendations),
        }
    elif isinstance(physics_report, dict):
        report_payload = dict(physics_report)
    return {
        "authority": (
            "MuJoCo contacts and measured dynamics override conflicting image claims"
        ),
        "payload_activation_interpretation": (
            "When payload.active is false, the box is intentionally deferred "
            "outside the task scene; staging-pose floor contact is not a drop."
        ),
        "verifier_report": report_payload,
        "payload": {
            key: payload.get(key)
            for key in (
                "active",
                "x",
                "y",
                "z",
                "yaw_rad",
                "yaw_deg",
                "tilt_deg",
                "linear_speed_mps",
                "angular_speed_radps",
                "support_center_signed_longitudinal_offset_m",
            )
            if key in payload
        },
        "payload_before": {
            key: before_payload.get(key)
            for key in (
                "x",
                "y",
                "z",
                "yaw_rad",
                "yaw_deg",
                "tilt_deg",
                "support_center_signed_longitudinal_offset_m",
            )
            if key in before_payload
        },
        "contacts": {
            "support_robots": list(contacts.get("support_robots", ())),
            "support_geometries": list(contacts.get("support_geometries", ())),
            "floor": bool(contacts.get("floor", False)),
            "walls": list(contacts.get("walls", ())),
        },
        "robots": compact_robots,
    }


def _normalized_failure_analysis(value: object) -> dict[str, Any]:
    analysis = value if isinstance(value, dict) else {}
    return {
        "outcome": str(analysis.get("outcome", "uncertain")).strip(),
        "primary_failure": str(
            analysis.get("primary_failure", "none")
        ).strip(),
        "causal_chain": _string_list(analysis.get("causal_chain", ())),
        "physical_evidence": _string_list(
            analysis.get("physical_evidence", ())
        ),
        "visual_evidence": _string_list(analysis.get("visual_evidence", ())),
        "strategy_defect": _bool_value(analysis.get("strategy_defect", False)),
        "coordination_defect": _bool_value(
            analysis.get("coordination_defect", False)
        ),
        "magnitude_defect": _bool_value(analysis.get("magnitude_defect", False)),
        "reusable_context": _string_list(analysis.get("reusable_context", ())),
        "confidence": _clamped_confidence(analysis.get("confidence", 0.0)),
    }


__all__ = [
    "QwenVLBoxCarryVisualVerifier",
    "VisualVerifierReport",
    "make_qwen_vl_visual_verifier",
]
