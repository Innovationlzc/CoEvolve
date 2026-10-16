"""Verifier-grounded deployment decision memory for cooperative execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal

from .failure_taxonomy import FailureClassification, classify_failure
from .coordination_specs import COORDINATION_PRIMITIVE_METHODS


CoordinationMode = Literal["sync", "async"]

STRATEGY_MEMORY_SCHEMA_VERSION = 2
ACTION_PARAMETER_KEYS = frozenset(
    {
        "distance_m",
        "speed_mps",
        "front_speed_mps",
        "back_speed_mps",
        "ramp_s",
        "yaw_deg",
        "yaw_rate_degps",
        "turn_segment_deg",
        "place_height",
        "height",
        "hold_s",
        "duration_s",
        "radius_m",
        "swing_radius_m",
        "angle_deg",
        "delta",
        "transport_ramp_seconds",
        "minimum_motion_duration_s",
        "start_pulse_seconds",
        "stand_seconds",
        "place_start_height",
        "place_ramp_seconds",
        "place_hold_seconds",
    }
)


@dataclass(frozen=True)
class StrategyContext:
    """Small phase/failure query used for dynamic-memory retrieval."""

    phase: str
    task_stage: str = ""
    last_failure_type: str = ""
    last_failure_level: str = ""


class StrategyMemoryError(ValueError):
    """Raised when an active strategy file violates the compact schema."""


class StrategyMemoryStore:
    """Persist compact decision rules promoted from physical verifier evidence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self._load_or_default()

    @property
    def specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self.data.get("specs", ()))

    def reset_to_priors(self) -> None:
        """Reset to an empty active memory; priors belong in prompts/static specs."""

        self.data = _default_document()

    def save(self) -> None:
        self._validate_document(self.data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def retrieve(self, context: StrategyContext, *, limit: int = 5) -> list[str]:
        """Return a balanced phase-relevant set of negative and positive rules."""

        phase = normalize_strategy_phase(context.phase)
        aliases = strategy_phase_aliases(phase)
        scored: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        for index, spec in enumerate(self.data.get("specs", ())):
            spec_phase = normalize_strategy_phase(spec.get("phase", "any"))
            if spec_phase not in aliases and spec_phase != "any":
                continue
            failures = {str(item) for item in spec.get("failure_types", ())}
            score = (
                2.0 if spec_phase == phase else 1.0 if spec_phase in aliases else 0.0,
                1.0 if context.last_failure_type in failures else 0.0,
                1.0
                if context.last_failure_level
                and context.last_failure_level == str(spec.get("failure_level", ""))
                else 0.0,
                float(spec.get("confidence", 0.0)),
                float(_evidence_total(spec)),
                float(index),
            )
            scored.append((score, spec))
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        maximum = max(0, int(limit))
        selected: list[dict[str, Any]] = []

        def add_matching(predicate) -> None:
            for _score, spec in ranked:
                if len(selected) >= maximum:
                    return
                if predicate(spec) and spec not in selected:
                    selected.append(spec)

        failures = [
            spec
            for _score, spec in ranked
            if spec.get("kind") == "verified_failure_counterexample"
        ]
        successes = [
            spec
            for _score, spec in ranked
            if spec.get("kind") in {
                "verified_motion_success",
                "verified_support_success",
            }
        ]
        if maximum > 0:
            # At most two counterexamples and, when available, reserve one slot
            # for a verified success so prompts do not become all prohibitions.
            negative_limit = maximum if not successes else max(0, maximum - 1)
            selected.extend(failures[: min(2, negative_limit)])
            if successes and len(selected) < maximum:
                selected.append(successes[0])
        if context.last_failure_type and len(selected) < maximum:
            add_matching(
                lambda spec: context.last_failure_type
                in {str(item) for item in spec.get("failure_types", ())}
            )
        add_matching(lambda spec: spec.get("kind") == "verified_failure_counterexample")
        add_matching(lambda spec: spec.get("kind") == "verified_motion_success")
        add_matching(lambda spec: spec.get("kind") == "verified_support_success")
        add_matching(lambda _spec: True)
        return [render_strategy_spec(spec) for spec in selected]

    def update_from_rollout(
        self,
        record: dict[str, Any],
        *,
        default_phase: str = "any",
        model_proposal: dict[str, Any] | None = None,
    ) -> bool:
        """Promote a model abstraction only when physical evidence supports it."""

        verifier = record.get("verifier", {})
        if record.get("dry_run") is True or not isinstance(verifier, dict) or not verifier:
            return False
        if not bool(record.get("physically_dispatched", False)) or not bool(
            record.get("physical_transaction_completed", False)
        ):
            return False
        if str(record.get("kind", "physical_rollout_finalized")) != (
            "physical_rollout_finalized"
        ):
            return False
        failure_type = str(verifier.get("failure_type", "none"))
        metrics = verifier.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        classification = classify_failure(failure_type, metrics=metrics, evidence=verifier)
        if not classification.promotable:
            return False
        route_quality = verifier.get("accepted_route_quality", {})
        if (
            isinstance(route_quality, dict)
            and route_quality.get("positive_memory_eligible") is False
        ):
            return False
        transition_outcome = str(verifier.get("transition_outcome", "")).strip()
        if transition_outcome == "ACCEPT_SETUP":
            return False

        context = record.get("strategy_context", {})
        context = context if isinstance(context, dict) else {}
        phase = normalize_strategy_phase(context.get("phase") or default_phase)
        mode = coordination_mode_from_record(record)
        if mode is None:
            return False
        accepted = bool(verifier.get("step_accepted", False))
        action_kind = strategy_action_kind(record)
        primitive = coordination_primitive_from_record(record)
        if accepted and action_kind == "unknown":
            return False
        if model_proposal is None:
            embedded = record.get("model_memory_update")
            model_proposal = dict(embedded) if isinstance(embedded, dict) else None
        if model_proposal is not None and accepted:
            if not bool(model_proposal.get("promote", False)):
                return False
            child_strategy = model_proposal.get("child_strategy", {})
            child_strategy = (
                child_strategy if isinstance(child_strategy, dict) else {}
            )
            proposed_primitive = str(
                child_strategy.get("primitive", "")
            ).strip()
            if (
                primitive
                and proposed_primitive
                and proposed_primitive != primitive
            ):
                raise StrategyMemoryError(
                    "model memory child_strategy is inconsistent with the "
                    "physically executed primitive: "
                    f"{proposed_primitive!r} != {primitive!r}"
                )
            candidate = _model_strategy_spec(
                proposal=model_proposal,
                phase=phase,
                mode=mode,
                action_kind=action_kind,
                primitive=primitive,
                accepted=accepted,
                failure_type=failure_type,
                classification=classification,
            )
        else:
            # A failed rollout verifies only the rejected executable behavior.
            # The model's proposed child repair has not run yet and therefore
            # remains raw audit evidence rather than active strategy memory.
            candidate = (
                _success_spec(
                    phase=phase,
                    mode=mode,
                    action_kind=action_kind,
                    primitive=primitive,
                )
                if accepted
                else _failure_spec(
                    phase=phase,
                    mode=mode,
                    failure_type=failure_type,
                    classification=classification,
                    primitive=primitive,
                )
            )
        return self._upsert(
            candidate,
            record=record,
            accepted=accepted,
            failure_type=failure_type,
        )

    def rebuild_from_raw(
        self,
        records: Iterable[dict[str, Any]],
        *,
        default_phase: str,
        revoked_checkpoint_ids: Iterable[str] = (),
    ) -> dict[str, int]:
        """Rebuild active rules from raw evidence without model-authored claims."""

        revoked = {str(item) for item in revoked_checkpoint_ids}
        self.reset_to_priors()
        counts = {"seen": 0, "used": 0, "revoked": 0, "ignored": 0}
        for record in records:
            counts["seen"] += 1
            verifier = record.get("verifier", {})
            verifier = verifier if isinstance(verifier, dict) else {}
            checkpoint_before = str(verifier.get("checkpoint_before", ""))
            checkpoint_after = str(verifier.get("checkpoint_after", ""))
            if checkpoint_before in revoked or checkpoint_after in revoked:
                counts["revoked"] += 1
                continue
            deterministic_record = dict(record)
            deterministic_record.pop("model_memory_update", None)
            if self.update_from_rollout(
                deterministic_record,
                default_phase=default_phase,
            ):
                counts["used"] += 1
            else:
                counts["ignored"] += 1
        self.data["migration"] = {
            "default_phase": normalize_strategy_phase(default_phase),
            "revoked_checkpoint_ids": sorted(revoked),
            "counts": counts,
        }
        self.save()
        return counts

    def _load_or_default(self) -> dict[str, Any]:
        if not self.path.exists():
            return _default_document()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StrategyMemoryError(
                f"invalid strategy memory JSON at {self.path}: {exc}"
            ) from exc
        if isinstance(payload, dict) and int(payload.get("schema_version", -1)) == 1:
            payload = _migrate_v1_document(payload)
        self._validate_document(payload)
        return payload

    def _upsert(
        self,
        candidate: dict[str, Any],
        *,
        record: dict[str, Any],
        accepted: bool,
        failure_type: str,
    ) -> bool:
        self._validate_spec(candidate)
        specs = self.data.setdefault("specs", [])
        existing = next((item for item in specs if item.get("id") == candidate["id"]), None)
        if existing is None:
            existing = candidate
            specs.append(existing)
        evidence = existing.setdefault("evidence", _empty_evidence())
        rollout_key = _rollout_evidence_key(
            record,
            accepted=accepted,
            failure_type=failure_type,
        )
        rollout_keys = evidence.setdefault("rollout_keys", [])
        if rollout_key in rollout_keys:
            return False
        rollout_keys.append(rollout_key)
        count_key = "accepted_count" if accepted else "rejected_count"
        evidence[count_key] = int(evidence.get(count_key, 0)) + 1
        if not accepted:
            failures = evidence.setdefault("failure_counts", {})
            failures[failure_type] = int(failures.get(failure_type, 0)) + 1
        reference = {
            "script_id": str(record.get("script_id", "")),
            "attempt": record.get("attempt"),
            "checkpoint_before": str(record.get("verifier", {}).get("checkpoint_before", "")),
            "checkpoint_after": str(record.get("verifier", {}).get("checkpoint_after", "")),
            "outcome": "accepted" if accepted else failure_type,
        }
        references = evidence.setdefault("rollout_refs", [])
        if reference not in references:
            references.append(reference)
            del references[:-12]
        existing["confidence"] = min(0.95, 0.50 + 0.06 * _evidence_total(existing))
        self._validate_spec(existing)
        return True

    @classmethod
    def _validate_document(cls, payload: object) -> None:
        if not isinstance(payload, dict):
            raise StrategyMemoryError("strategy memory root must be a JSON object")
        if payload.get("kind") != "deployment_decision_memory":
            raise StrategyMemoryError("strategy memory kind must be deployment_decision_memory")
        if int(payload.get("schema_version", -1)) != STRATEGY_MEMORY_SCHEMA_VERSION:
            raise StrategyMemoryError(
                f"unsupported strategy memory schema_version={payload.get('schema_version')!r}"
            )
        specs = payload.get("specs")
        if not isinstance(specs, list):
            raise StrategyMemoryError("strategy memory specs must be a list")
        ids: set[str] = set()
        for spec in specs:
            cls._validate_spec(spec)
            spec_id = str(spec["id"])
            if spec_id in ids:
                raise StrategyMemoryError(f"duplicate strategy spec id: {spec_id}")
            ids.add(spec_id)

    @staticmethod
    def _validate_spec(spec: object) -> None:
        if not isinstance(spec, dict):
            raise StrategyMemoryError("each strategy spec must be an object")
        required = {
            "id",
            "phase",
            "kind",
            "failure_level",
            "condition",
            "diagnosis",
            "decision",
            "forbidden",
            "repair",
            "confidence",
            "evidence",
        }
        missing = sorted(required - set(spec))
        if missing:
            raise StrategyMemoryError(f"strategy spec missing fields: {missing}")
        mode = spec.get("observed_coordination_mode")
        if mode not in {"sync", "async"}:
            raise StrategyMemoryError(
                f"strategy spec {spec.get('id')!r} has invalid observed mode={mode!r}"
            )
        if spec.get("failure_level") not in {
            "none",
            "strategy",
            "coordination",
            "script",
            "execution",
        }:
            raise StrategyMemoryError(
                f"strategy spec {spec.get('id')!r} has invalid failure_level"
            )
        forbidden = spec.get("forbidden")
        if not isinstance(forbidden, list) or not all(
            isinstance(item, str) and item.strip() for item in forbidden
        ):
            raise StrategyMemoryError("strategy spec forbidden must be a non-empty string list")
        confidence = float(spec.get("confidence", -1.0))
        if confidence < 0.0 or confidence > 1.0:
            raise StrategyMemoryError("strategy spec confidence must be in [0, 1]")
        leaked = _find_action_parameter_keys(spec)
        if leaked:
            raise StrategyMemoryError(
                f"active strategy spec {spec.get('id')!r} leaks action parameters: {sorted(leaked)}"
            )


def load_raw_attempts(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def coordination_mode_from_record(record: dict[str, Any]) -> CoordinationMode | None:
    explicit = str(record.get("coordination_mode", "")).strip().lower()
    if explicit in {"sync", "async"}:
        return explicit  # type: ignore[return-value]
    coordination = record.get("coordination", ())
    if isinstance(coordination, list):
        modes = {
            str(item.get("coordination_mode", "")).strip().lower()
            for item in coordination
            if isinstance(item, dict)
        }
        modes.discard("")
        if len(modes) == 1 and next(iter(modes)) in {"sync", "async"}:
            return next(iter(modes))  # type: ignore[return-value]
    calls = record.get("policy_execution_parameters", ())
    if isinstance(calls, list):
        valid_calls = [item for item in calls if isinstance(item, dict)]
        if len(valid_calls) >= 2:
            methods = {str(item.get("skill", "")) for item in valid_calls}
            return "sync" if len(methods) == 1 else "async"
    return None


def strategy_action_kind(record: dict[str, Any]) -> str:
    """Classify evidence without retaining a concrete motion skill."""

    skill = str(record.get("executed_skill", "")).strip()
    if not skill:
        calls = record.get("policy_execution_parameters", ())
        skills = (
            {
                str(item.get("skill", "")).strip()
                for item in calls
                if isinstance(item, dict) and str(item.get("skill", "")).strip()
            }
            if isinstance(calls, list)
            else set()
        )
        if len(skills) == 1:
            skill = next(iter(skills))
    if skill in {"hold_position", "place_crouch", "recover_stand"}:
        return "support"
    if skill:
        return "motion"
    return "unknown"


def coordination_primitive_from_record(record: dict[str, Any]) -> str:
    """Retain the selected coordination abstraction, never its numeric profile."""

    candidates = [str(record.get("executed_skill", "")).strip()]
    calls = record.get("policy_execution_parameters", ())
    if isinstance(calls, list):
        candidates.extend(
            str(item.get("skill", "")).strip()
            for item in calls
            if isinstance(item, dict)
        )
    return next(
        (item for item in candidates if item in COORDINATION_PRIMITIVE_METHODS),
        "",
    )


def normalize_strategy_phase(value: object) -> str:
    text = str(value or "any").strip().lower().replace("-", "_")
    if not text:
        return "any"
    if "target" in text or "place" in text or "alignment" in text:
        return "target_alignment"
    if "corridor" in text or "passage" in text:
        return "corridor_entry" if "entry" in text else "corridor_transport"
    if "straight" in text:
        return "straight_transport"
    if text in {"any", "all", "unknown"}:
        return "any"
    return text


def strategy_phase_aliases(phase: str) -> set[str]:
    aliases = {phase}
    if phase == "corridor_transport":
        aliases.add("straight_transport")
    elif phase == "corridor_entry":
        aliases.add("corridor_transport")
    return aliases


def render_strategy_spec(spec: dict[str, Any]) -> str:
    evidence = spec.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    failures = evidence.get("failure_counts", {})
    failures = failures if isinstance(failures, dict) else {}
    failure_text = ", ".join(f"{key}:{value}" for key, value in sorted(failures.items()))
    forbidden = "; ".join(str(item) for item in spec.get("forbidden", ()))
    return (
        f"[deployment_spec id={spec['id']} phase={spec['phase']} "
        f"observed_mode={spec['observed_coordination_mode']} "
        f"primitive={spec.get('observed_coordination_primitive') or 'legacy'} "
        f"failure_level={spec['failure_level']} "
        f"confidence={float(spec.get('confidence', 0.0)):.2f}]\n"
        f"Condition: {spec['condition']}\n"
        f"Diagnosis: {spec['diagnosis']}\n"
        f"Decision: {spec['decision']}\n"
        f"Forbidden: {forbidden}\n"
        f"Repair: {spec['repair']}\n"
        "Verifier evidence: "
        f"accepted={int(evidence.get('accepted_count', 0))}, "
        f"rejected={int(evidence.get('rejected_count', 0))}"
        + (f", failures={failure_text}" if failure_text else "")
    )


def _default_document() -> dict[str, Any]:
    return {
        "kind": "deployment_decision_memory",
        "schema_version": STRATEGY_MEMORY_SCHEMA_VERSION,
        "description": (
            "Verifier-grounded condition/diagnosis/decision rules. Scene answers and action "
            "magnitudes are excluded; static capability knowledge remains separate."
        ),
        "specs": [],
    }


def _success_spec(
    *,
    phase: str,
    mode: CoordinationMode,
    action_kind: str,
    primitive: str = "",
) -> dict[str, Any]:
    kind = "verified_support_success" if action_kind == "support" else "verified_motion_success"
    primitive_label = primitive or action_kind
    decision = (
        f"Reuse {primitive_label} with {mode} coordination only while phase, support, clearance, "
        "and relative formation remain physically comparable."
    )
    identity = f"{primitive}_{action_kind}" if primitive else action_kind
    return {
        "id": f"verified_{phase}_{mode}_{identity}_success",
        "phase": phase,
        "kind": kind,
        "failure_level": "none",
        "observed_coordination_mode": mode,
        "observed_coordination_primitive": primitive,
        "condition": (
            f"The verifier accepted {primitive_label} with {mode} coordination in {phase}."
        ),
        "diagnosis": "The coordination decision was physically viable under this observed condition.",
        "decision": decision,
        "forbidden": [
            "Do not generalize this result to changed geometry or support state.",
            "Do not preserve the concrete action magnitude as long-term experience.",
        ],
        "repair": (
            "If a later verifier changes the failure level, obey that level's repair scope "
            "instead of replaying this success blindly."
        ),
        "failure_types": [],
        "confidence": 0.50,
        "evidence": _empty_evidence(),
    }


def _failure_spec(
    *,
    phase: str,
    mode: CoordinationMode,
    failure_type: str,
    classification: FailureClassification,
    primitive: str = "",
) -> dict[str, Any]:
    identity = f"{primitive}_" if primitive else ""
    primitive_label = primitive or "legacy motion"
    return {
        "id": f"counterexample_{phase}_{mode}_{identity}{classification.level}_{failure_type}",
        "phase": phase,
        "kind": "verified_failure_counterexample",
        "failure_level": classification.level,
        "observed_coordination_mode": mode,
        "observed_coordination_primitive": primitive,
        "condition": (
            f"The verifier rejected {primitive_label} with {mode} coordination in {phase} "
            f"as {failure_type}."
        ),
        "diagnosis": classification.diagnosis,
        "decision": _decision_text(classification),
        "forbidden": [_symbolic_rule_text(item) for item in classification.forbidden_repairs],
        "repair": _repair_text(classification),
        "failure_types": [failure_type],
        "confidence": 0.50,
        "evidence": _empty_evidence(),
    }


def _model_strategy_spec(
    *,
    proposal: dict[str, Any],
    phase: str,
    mode: CoordinationMode,
    action_kind: str,
    primitive: str,
    accepted: bool,
    failure_type: str,
    classification: FailureClassification,
) -> dict[str, Any]:
    """Map a model-proposed strategy genome into the verifier-owned schema."""

    leaked = _find_action_parameter_keys(proposal)
    if leaked:
        raise StrategyMemoryError(
            "model memory proposal leaks action parameters: "
            + ", ".join(sorted(leaked))
        )
    semantic_text = json.dumps(
        semantic_memory := {
            key: proposal.get(key)
            for key in (
                "strategy_name",
                "memory_kind",
                "applicability",
                "parent_strategy",
                "child_strategy",
                "learned_rule",
                "avoid",
            )
        },
        ensure_ascii=False,
    )
    if _contains_numeric_value(semantic_memory):
        raise StrategyMemoryError(
            "model memory proposal contains a numerical action or scene value"
        )
    if re.search(
        r"(?:\b\d+(?:\.\d+)?\s*(?:m|cm|s|deg|°|m/s)\b|"
        r"\b(?:waypoint|checkpoint pose|world[_ -]?[xy]|target[_ -]?[xy])\b)",
        semantic_text,
        flags=re.IGNORECASE,
    ):
        raise StrategyMemoryError(
            "model memory proposal contains a scene route, pose, or action magnitude"
        )
    strategy_name = str(proposal.get("strategy_name", "")).strip()
    learned_rule = str(proposal.get("learned_rule", "")).strip()
    applicability = [
        str(item).strip()
        for item in proposal.get("applicability", ())
        if str(item).strip()
    ]
    avoid = [
        str(item).strip()
        for item in proposal.get("avoid", ())
        if str(item).strip()
    ]
    if not strategy_name or not learned_rule or not applicability:
        raise StrategyMemoryError(
            "promoted model memory requires strategy_name, learned_rule, and applicability"
        )
    parent = proposal.get("parent_strategy", {})
    child = proposal.get("child_strategy", {})
    parent_text = _behavior_abstraction_text(parent)
    child_text = _behavior_abstraction_text(child)
    if not child_text:
        raise StrategyMemoryError(
            "promoted model memory requires a non-empty child_strategy abstraction"
        )
    identity_payload = {
        "phase": phase,
        "mode": mode,
        "strategy_name": strategy_name,
        "memory_kind": str(proposal.get("memory_kind", "")),
        "parent": parent,
        "child": child,
        "accepted": accepted,
        "failure_type": failure_type,
    }
    identity = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    confidence = min(0.95, max(0.0, float(proposal.get("confidence", 0.5))))
    return {
        "id": f"strategy_genome_{identity}",
        "phase": phase,
        "kind": (
            "verified_motion_success"
            if accepted and action_kind == "motion"
            else "verified_support_success"
            if accepted
            else "verified_failure_counterexample"
        ),
        "failure_level": "none" if accepted else classification.level,
        "observed_coordination_mode": mode,
        "observed_coordination_primitive": primitive,
        "condition": "; ".join(applicability),
        "diagnosis": learned_rule,
        "decision": (
            f"In comparable contexts, inherit the child strategy: {child_text}"
        ),
        "forbidden": avoid
        or ["Do not generalize this strategy outside its verified applicability."],
        "repair": (
            f"Mutate from {parent_text or 'the prior behavior abstraction'} "
            f"to {child_text}; let the next physics verifier select survival."
        ),
        "failure_types": [] if accepted else [failure_type],
        "confidence": confidence,
        "strategy_genome": {
            "strategy_name": strategy_name,
            "memory_kind": str(proposal.get("memory_kind", "")).strip(),
            "mutation_level": str(proposal.get("mutation_level", "")).strip(),
            "parent_strategy": parent,
            "child_strategy": child,
            "learned_rule": learned_rule,
            "inheritance": str(proposal.get("inheritance", "")).strip(),
        },
        "evidence": _empty_evidence(),
    }


def _behavior_abstraction_text(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(
            f"{key}={child}"
            for key, child in sorted(value.items())
            if str(child).strip()
        )
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _decision_text(classification: FailureClassification) -> str:
    if classification.level == "strategy":
        return (
            "Change the strategy signature before tuning magnitude: choose a different motion "
            "objective and/or coordination decision from current geometry and measured world motion."
        )
    if classification.level == "coordination":
        return (
            "Repair the temporary role relationship or coordination mode first while keeping the "
            "validated motion objective fixed for the diagnostic trial."
        )
    if classification.level == "script":
        return (
            "Treat the rejected executable transaction as the defect: identify the bad call, "
            "parameter, or front/back relation and generate a materially different AST."
        )
    return (
        "Keep the current strategy and tune one bounded execution choice from current physics "
        "evidence."
    )


def _repair_text(classification: FailureClassification) -> str:
    if classification.level == "coordination":
        return (
            "Restore valid support when required, then test one coordinated role or timing change "
            "and let the next physics verifier decide."
        )
    if classification.level == "strategy":
        return (
            "Use camera geometry and world-frame progress to select a different bounded strategy; "
            "do not infer a scene route from this card alone."
        )
    if classification.level == "script":
        return (
            "Restore the last accepted checkpoint, explain the executable cause of the physical "
            "failure, then compile one materially different bounded transaction."
        )
    return (
        "Change one bounded execution setting only while support, balance, and clearance remain valid."
    )


def _symbolic_rule_text(value: str) -> str:
    return str(value).replace("_", " ").strip().capitalize() + "."


def _empty_evidence() -> dict[str, Any]:
    return {
        "source": "physics_verifier",
        "accepted_count": 0,
        "rejected_count": 0,
        "failure_counts": {},
        "rollout_refs": [],
        "rollout_keys": [],
    }


def _migrate_v1_document(payload: dict[str, Any]) -> dict[str, Any]:
    """Load old files safely; a formal rebuild remains the preferred migration."""

    migrated = _default_document()
    for old in payload.get("specs", ()):
        if not isinstance(old, dict) or old.get("kind") == "design_prior":
            continue
        phase = normalize_strategy_phase(old.get("phase", "any"))
        mode = str(old.get("strategy", "")).strip().lower()
        if mode not in {"sync", "async"}:
            continue
        if old.get("kind") in {"verified_motion_success", "verified_support_success"}:
            action_kind = "support" if old.get("kind") == "verified_support_success" else "motion"
            spec = _success_spec(
                phase=phase,
                mode=mode,  # type: ignore[arg-type]
                action_kind=action_kind,
            )
        else:
            failures = [str(item) for item in old.get("failure_types", ()) if str(item)]
            if not failures:
                continue
            failure = failures[0]
            classification = classify_failure(failure)
            if not classification.promotable:
                continue
            spec = _failure_spec(
                phase=phase,
                mode=mode,  # type: ignore[arg-type]
                failure_type=failure,
                classification=classification,
            )
        old_evidence = old.get("evidence", {})
        if isinstance(old_evidence, dict):
            spec["evidence"] = {
                **_empty_evidence(),
                **old_evidence,
            }
        spec["confidence"] = float(old.get("confidence", spec["confidence"]))
        existing = next(
            (item for item in migrated["specs"] if item.get("id") == spec["id"]),
            None,
        )
        if existing is None:
            migrated["specs"].append(spec)
        else:
            _merge_evidence(existing, spec)
    migrated["schema_migration"] = {
        "from_version": 1,
        "discarded_design_priors": True,
        "note": "Run scripts/rebuild_strategy_memory.py to regenerate from raw verifier evidence.",
    }
    return migrated


def _merge_evidence(target: dict[str, Any], source: dict[str, Any]) -> None:
    target_evidence = target.setdefault("evidence", _empty_evidence())
    source_evidence = source.get("evidence", {})
    source_evidence = source_evidence if isinstance(source_evidence, dict) else {}
    for count_key in ("accepted_count", "rejected_count"):
        target_evidence[count_key] = int(target_evidence.get(count_key, 0)) + int(
            source_evidence.get(count_key, 0)
        )
    target_failures = target_evidence.setdefault("failure_counts", {})
    source_failures = source_evidence.get("failure_counts", {})
    if isinstance(source_failures, dict):
        for failure, count in source_failures.items():
            target_failures[str(failure)] = int(target_failures.get(str(failure), 0)) + int(count)
    for list_key in ("rollout_refs", "rollout_keys"):
        target_items = target_evidence.setdefault(list_key, [])
        source_items = source_evidence.get(list_key, ())
        if not isinstance(source_items, list):
            continue
        for item in source_items:
            if item not in target_items:
                target_items.append(item)
        if list_key == "rollout_refs":
            del target_items[:-12]
    target["confidence"] = max(
        float(target.get("confidence", 0.0)),
        float(source.get("confidence", 0.0)),
    )


def _evidence_total(spec: dict[str, Any]) -> int:
    evidence = spec.get("evidence", {})
    if not isinstance(evidence, dict):
        return 0
    return int(evidence.get("accepted_count", 0)) + int(evidence.get("rejected_count", 0))


def _rollout_evidence_key(
    record: dict[str, Any],
    *,
    accepted: bool,
    failure_type: str,
) -> str:
    verifier = record.get("verifier", {})
    verifier = verifier if isinstance(verifier, dict) else {}
    fingerprint = {
        "script_id": str(record.get("script_id", "")),
        "attempt": record.get("attempt"),
        "checkpoint_before": str(verifier.get("checkpoint_before", "")),
        "checkpoint_after": str(verifier.get("checkpoint_after", "")),
        "outcome": "accepted" if accepted else failure_type,
        "metrics": verifier.get("metrics", {}),
    }
    return hashlib.sha256(
        json.dumps(
            fingerprint,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _find_action_parameter_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in ACTION_PARAMETER_KEYS:
                found.add(str(key))
            found.update(_find_action_parameter_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_action_parameter_keys(child))
    elif isinstance(value, str):
        for key in ACTION_PARAMETER_KEYS:
            if key in value:
                found.add(key)
    return found


def _contains_numeric_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_contains_numeric_value(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_numeric_value(child) for child in value)
    return False


__all__ = [
    "ACTION_PARAMETER_KEYS",
    "STRATEGY_MEMORY_SCHEMA_VERSION",
    "StrategyContext",
    "StrategyMemoryError",
    "StrategyMemoryStore",
    "coordination_mode_from_record",
    "load_raw_attempts",
    "normalize_strategy_phase",
    "render_strategy_spec",
    "strategy_action_kind",
    "strategy_phase_aliases",
]
