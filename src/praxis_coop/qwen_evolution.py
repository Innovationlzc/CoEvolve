"""Qwen-backed critic/coder/spec-memory proposal logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from json import JSONDecodeError
from typing import Any

from praxis_coop.llm import QwenTextClient
from praxis_coop.models import ExecutableSpec, PatchProposal, SceneGraph, SkillGraph, VerifierReport
from praxis_coop.stage1_evolution import (
    Stage1Parameters,
    Stage1SonicParameters,
    stage1_patch_schema,
    stage1_sonic_patch_schema,
)


@dataclass
class QwenEvolutionDecision:
    patch: PatchProposal | None
    spec: ExecutableSpec | None
    raw_response: str


SYSTEM_PROMPT = """You are the Praxis-Coop critic/coder/spec-memory agent.
Your job is to improve a cooperative humanoid SkillGraph after a physics-verifier
rollout and summarize reusable deployment-calling experience.

Return strict JSON only. Do not include Markdown.

Schema:
{
  "patch": null | {
    "patch_id": "short_snake_case_id",
    "patch_type": "parameter" | "graph",
    "target": "SkillGraph field path",
    "change": { ... },
    "rationale": "concise physical reason"
  },
  "spec": null | {
    "spec_id": "learned_short_snake_case_id",
    "task_context": {
      "physical_coupling": "rigid_object",
      "object_type": "long_box",
      "interaction_type": "pure_cooperation"
    },
    "rule_text": "general reusable deployment-calling experience",
    "trigger": ["condition text"],
    "executable_predicate": ["state predicate text"],
    "fallback": ["skill or repair text"],
    "confidence": 0.0,
    "evidence": {
      "failure_cases": ["failure_mode"],
      "success_after_patch": []
    }
  }
}

Important:
- Do not claim success when verifier.success is false.
- The patch is a temporary repair for this rollout, not a long-term patch policy.
- Prefer small typed parameter patches over broad rewrites.
- Use verifier_report.evidence.failed_phase_id and phase_trace to target the
  smallest failing phase.
- A patch must be executable by existing apply_patch logic unless explicitly
  targeting a future graph insertion.
- For transport path failures, patch transport_object_path parameters.
- For contact/tilt/drop failures, patch contact, synchronization, support, speed,
  or path parameters and write a reusable deployment spec.
- The spec should preserve verified success/failure experience about how SONIC
  or registered skills should be called in similar deployment scenes.
- Useful specs include synchronized lateral movement during turns and avoiding
  repeated pose adjustment when payload progress is near zero.
"""


def propose_qwen_phase1_decision(
    report: VerifierReport,
    scene: SceneGraph,
    graph: SkillGraph,
    *,
    best_distance_m: float,
    model_name: str = "qwen3-8b-instruct",
    client: QwenTextClient | None = None,
) -> QwenEvolutionDecision:
    if report.success:
        return QwenEvolutionDecision(patch=None, spec=None, raw_response="")

    client = client or QwenTextClient(model_name=model_name)
    user_prompt = json.dumps(
        {
            "verifier_report": _jsonable(asdict(report)),
            "scene": _jsonable(asdict(scene)),
            "skill_graph": _jsonable(asdict(graph)),
            "runtime_metrics": {"best_distance_m": float(best_distance_m)},
            "allowed_existing_patch_targets": [
                "transport_object_path.object_path",
                "transport_object_path.parameters",
                "transport_object_path.preconditions",
                "transport_object_path.contact_flow",
                "establish_contact.parameters",
                "establish_contact.contact_flow",
                "pre_align_object.parameters",
                "pre_align_object.contact_flow",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    raw = client.generate(system=SYSTEM_PROMPT, user=user_prompt)
    payload = _extract_json_object(raw)
    patch = _patch_from_payload(payload.get("patch"))
    spec = _spec_from_payload(payload.get("spec"))
    if patch is None:
        raise ValueError(f"Qwen returned no executable patch for failed rollout: {raw}")
    if spec is None:
        raise ValueError(f"Qwen returned no reusable spec for failed rollout: {raw}")
    return QwenEvolutionDecision(patch=patch, spec=spec, raw_response=raw)


STAGE1_SYSTEM_PROMPT = """You are the Praxis-Coop Stage1 evolution agent.
Your job is to improve a real IsaacLab two-G1 straight-carry rollout by changing
only bounded Stage1 controller parameters. The rollout is physically judged:
do not claim success when metrics.success is false.

The patch is only the immediate repair action for the next attempt. The lasting
memory is the deployment spec: a reusable experience about how two G1 robots
should call walking/support/contact skills in similar scenes.

Return strict JSON only. Do not include Markdown.

Schema:
{
  "patch": {
    "patch_id": "short_snake_case_id",
    "patch_type": "parameter",
    "target": "stage1_parameters",
    "change": {
      "one_or_more_allowed_parameter_names": "new value"
    },
    "rationale": "concise physical reason"
  },
  "spec": {
    "spec_id": "learned_short_snake_case_id",
    "task_context": {
      "physical_coupling": "rigid_object",
      "object_type": "long_box",
      "interaction_type": "pure_cooperation",
      "stage": "straight_carry"
    },
    "rule_text": "general reusable deployment-calling experience",
    "trigger": ["condition text"],
    "executable_predicate": ["state predicate text"],
    "fallback": ["repair text"],
    "confidence": 0.0,
    "evidence": {
      "failure_cases": ["failure_mode"],
      "success_after_patch": []
    }
  }
}

Important:
- Patch only allowed fields listed in allowed_patch_schema.
- The upper body is locked by hold_initial_payload_hand_pose using the manual
  scene's initial pelvis-relative wrist/hand pose. Do not change upper-body,
  wrist, or hand-pose parameters.
- Prefer small changes to locomotion speed, hip height, path speed, and lower
  body follow mode.
- For box_dropped, reduce speed or adjust lower-body support; do not redesign
  grasp/hand pose.
- For robot_fall, reduce base speed, adjust hip height, or use planned_follow.
- For target_miss/timeout, increase path speed carefully or switch to planned_follow.
- For out_of_bounds, reduce lateral asymmetry or use planned_follow.
- If repeated support/pose adjustment produces little payload progress, write a
  failure spec that says to stop adding more re-approach/align actions and switch
  to synchronized transport with bounded speed.
"""


def propose_qwen_stage1_decision(
    metrics: dict[str, Any],
    history_tail: list[dict[str, Any]],
    params: Stage1Parameters,
    *,
    model_name: str = "qwen3-8b-instruct",
    client: QwenTextClient | None = None,
) -> QwenEvolutionDecision:
    if metrics.get("success") is True:
        return QwenEvolutionDecision(patch=None, spec=None, raw_response="")

    client = client or QwenTextClient(model_name=model_name)
    user_prompt = json.dumps(
        {
            "metrics": _jsonable(metrics),
            "history_tail": _jsonable(history_tail),
            "current_parameters": _jsonable(params.to_json()),
            "allowed_patch_schema": _jsonable(stage1_patch_schema()),
        },
        ensure_ascii=False,
        indent=2,
    )
    raw = client.generate(system=STAGE1_SYSTEM_PROMPT, user=user_prompt)
    payload = _extract_json_object(raw)
    patch = _patch_from_payload(payload.get("patch"))
    spec = _spec_from_payload(payload.get("spec"))
    if patch is None:
        raise ValueError(f"Qwen returned no Stage1 patch for failed rollout: {raw}")
    if spec is None:
        raise ValueError(f"Qwen returned no Stage1 spec for failed rollout: {raw}")
    return QwenEvolutionDecision(patch=patch, spec=spec, raw_response=raw)


STAGE1_SONIC_SYSTEM_PROMPT = """You are the Praxis-Coop Stage1 SONIC evolution agent.
Your job is to improve a real IsaacLab two-G1 straight-carry rollout by changing
only bounded SONIC planner-command parameters. The rollout is physically judged:
do not claim success when metrics.success is false.

The executor is fixed:
- SONIC release planner + encoder + decoder ONNX generates action[29].
- action[29] is mapped to IsaacLab G1 articulation joint targets.
- upper-body and hand joints are locked to the manual-scene initial carrying pose.
- no Isaac task action surface, root teleport, direct box velocity, or forced box
  height is allowed.

The patch is only the immediate repair action for the next attempt. The lasting
memory is the deployment spec: reusable experience about how two G1 robots
should call SONIC walking/carrying skills in similar scenes.

Return strict JSON only. Do not include Markdown.

Schema:
{
  "patch": {
    "patch_id": "short_snake_case_id",
    "patch_type": "parameter",
    "target": "stage1_sonic_parameters",
    "change": {
      "one_or_more_allowed_parameter_names": "new value"
    },
    "rationale": "concise physical reason"
  },
  "spec": {
    "spec_id": "learned_short_snake_case_id",
    "task_context": {
      "physical_coupling": "rigid_object",
      "object_type": "long_box",
      "interaction_type": "pure_cooperation",
      "stage": "straight_carry",
      "motor_layer": "sonic_joint_wbc"
    },
    "rule_text": "general reusable deployment-calling experience",
    "trigger": ["condition text"],
    "executable_predicate": ["state predicate text"],
    "fallback": ["repair text"],
    "confidence": 0.0,
    "evidence": {
      "failure_cases": ["failure_mode"],
      "success_after_patch": []
    }
  }
}

Important:
- Patch only allowed fields listed in allowed_patch_schema.
- Do not change upper-body, wrist, hand-pose locks, or request proxy actions.
- Robot A and Robot B start in the user-authored manual layout, both with the
  same initial heading, at the two payload contact positions around the long
  box. For synchronized world +Y Stage1 transport, both robots should normally
  receive the same +Y movement sign unless verifier evidence proves otherwise.
- Prefer small changes to target_vel, per-robot target_vel, movement vectors,
  facing vectors, default_height, sonic_height, and replan interval.
- For box_dropped, reduce speed, reduce command asymmetry, or lower height
  conservatively.
- For robot_fall or unstable contact, reduce speed, increase warmup, or reduce
  replan aggressiveness.
- For timeout/no payload progress, increase target_vel carefully or correct the
  per-robot movement signs so both robots move the payload in the same world
  direction.
- If repeated movement produces little payload progress, write a failure spec
  that says to stop adding re-approach/align actions and use synchronized
  transport with bounded speed and verified contact.
"""


def propose_qwen_stage1_sonic_decision(
    metrics: dict[str, Any],
    history_tail: list[dict[str, Any]],
    params: Stage1SonicParameters,
    *,
    model_name: str = "qwen3-8b-instruct",
    client: QwenTextClient | None = None,
) -> QwenEvolutionDecision:
    if metrics.get("success") is True:
        return QwenEvolutionDecision(patch=None, spec=None, raw_response="")

    client = client or QwenTextClient(model_name=model_name)
    user_prompt = json.dumps(
        {
            "metrics": _jsonable(metrics),
            "history_tail": _jsonable(history_tail),
            "current_parameters": _jsonable(params.to_json()),
            "allowed_patch_schema": _jsonable(stage1_sonic_patch_schema()),
        },
        ensure_ascii=False,
        indent=2,
    )
    raw = client.generate(system=STAGE1_SONIC_SYSTEM_PROMPT, user=user_prompt)
    payload = _extract_json_object(raw)
    patch = _patch_from_payload(payload.get("patch"))
    spec = _spec_from_payload(payload.get("spec"))
    if patch is None:
        raise ValueError(f"Qwen returned no Stage1 SONIC patch for failed rollout: {raw}")
    if spec is None:
        raise ValueError(f"Qwen returned no Stage1 SONIC spec for failed rollout: {raw}")
    return QwenEvolutionDecision(patch=patch, spec=spec, raw_response=raw)


def _patch_from_payload(payload: Any) -> PatchProposal | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"patch must be object or null, got {type(payload).__name__}")
    return PatchProposal(
        patch_id=str(payload["patch_id"]),
        patch_type=str(payload["patch_type"]),
        target=str(payload["target"]),
        change=dict(payload.get("change", {})),
        rationale=str(payload.get("rationale", "")),
    )


def _spec_from_payload(payload: Any) -> ExecutableSpec | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"spec must be object or null, got {type(payload).__name__}")
    evidence = payload.get("evidence") or {"failure_cases": [], "success_after_patch": []}
    evidence.setdefault("failure_cases", [])
    evidence.setdefault("success_after_patch", [])
    return ExecutableSpec(
        spec_id=str(payload["spec_id"]),
        task_context={str(k): str(v) for k, v in dict(payload.get("task_context", {})).items()},
        rule_text=str(payload["rule_text"]),
        trigger=[str(item) for item in payload.get("trigger", [])],
        executable_predicate=[str(item) for item in payload.get("executable_predicate", [])],
        fallback=[str(item) for item in payload.get("fallback", [])],
        confidence=float(payload.get("confidence", 0.5)),
        evidence={
            "failure_cases": [str(item) for item in evidence.get("failure_cases", [])],
            "success_after_patch": [str(item) for item in evidence.get("success_after_patch", [])],
        },
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Qwen response did not contain a JSON object: {text}")
    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except JSONDecodeError as exc:
        raise ValueError(f"Qwen response JSON parse failed: {exc}. Raw response: {text}") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
