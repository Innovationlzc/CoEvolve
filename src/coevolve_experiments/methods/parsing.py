"""Strict conversion from backend payloads to the common intent contract."""

from __future__ import annotations

from typing import Any, Mapping

from coevolve_experiments.runtime.contracts import (
    PolicyProposal,
    RobotIntent,
    TeamIntent,
)
from coevolve_experiments.runtime.policy import (
    IntentCompiler,
    PolicyValidationError,
    canonical_ast,
)

from .base import AdapterContractError, RoleResponse


def proposal_from_response(
    response: RoleResponse,
    *,
    fallback: PolicyProposal | None = None,
) -> PolicyProposal:
    return proposal_from_content(response.content, fallback=fallback)


def proposal_from_content(
    content: Any,
    *,
    fallback: PolicyProposal | None = None,
) -> PolicyProposal:
    if isinstance(content, PolicyProposal):
        return _compiled_proposal(content)
    if not isinstance(content, Mapping):
        if fallback is not None:
            return fallback
        raise AdapterContractError("policy role did not return a mapping")
    nested = content.get("proposal")
    if nested is not None:
        return proposal_from_content(nested, fallback=fallback)
    if "intent" not in content:
        if fallback is not None:
            return fallback
        raise AdapterContractError(
            "policy response requires structured intent (or proposal)"
        )
    intent = _parse_intent(content["intent"])
    source = content.get("source")
    if source is not None:
        raise AdapterContractError(
            "model/backend responses may provide TeamIntent only; source is compiler-owned"
        )
    rationale = content.get("rationale", "")
    metadata = content.get("metadata", {})
    if not isinstance(rationale, str) or not isinstance(metadata, Mapping):
        raise AdapterContractError("invalid proposal rationale or metadata")
    proposal = PolicyProposal(IntentCompiler().compile(intent), intent, rationale, dict(metadata))
    return _compiled_proposal(proposal)


def _compiled_proposal(proposal: PolicyProposal) -> PolicyProposal:
    """Reject source/intent mismatch, then retain only compiler-owned source."""

    compiler = IntentCompiler()
    compiled = compiler.compile(proposal.intent)
    try:
        matches = canonical_ast(proposal.source) == canonical_ast(compiled)
    except PolicyValidationError as exc:
        raise AdapterContractError(str(exc)) from exc
    if not matches:
        raise AdapterContractError(
            "backend source disagrees with deterministic TeamIntent compilation"
        )
    return PolicyProposal(
        source=compiled,
        intent=proposal.intent,
        rationale=proposal.rationale,
        metadata=dict(proposal.metadata),
    )


def _parse_intent(value: Any) -> TeamIntent:
    if isinstance(value, TeamIntent):
        return value
    if not isinstance(value, Mapping):
        raise AdapterContractError("proposal intent must be a mapping")
    commands_value = value.get("commands")
    if not isinstance(commands_value, (list, tuple)):
        raise AdapterContractError("proposal intent requires commands")
    commands: list[RobotIntent] = []
    for item in commands_value:
        if isinstance(item, RobotIntent):
            commands.append(item)
            continue
        if not isinstance(item, Mapping):
            raise AdapterContractError("intent command must be a mapping")
        try:
            actor = _nonempty_string(item, "actor")
            primitive = _nonempty_string(item, "primitive")
            objective = _nonempty_string(item, "objective")
            mode = item.get("mode", "default")
            if not isinstance(mode, str) or not mode.strip():
                raise AdapterContractError("intent command mode must be a non-empty string")
            arguments = item.get("arguments", {})
            if not isinstance(arguments, Mapping):
                raise AdapterContractError("intent command arguments must be a mapping")
            commands.append(
                RobotIntent(
                    actor=actor,
                    primitive=primitive,
                    objective=objective,
                    mode=mode,
                    arguments=dict(arguments),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterContractError(f"invalid intent command: {exc}") from exc
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise AdapterContractError("intent metadata must be a mapping")
    atomic = value.get("atomic", True)
    if not isinstance(atomic, bool):
        raise AdapterContractError("proposal intent atomic must be a boolean")
    lifecycle_phase = value.get("lifecycle_phase", "unspecified")
    if not isinstance(lifecycle_phase, str) or not lifecycle_phase.strip():
        raise AdapterContractError(
            "proposal intent lifecycle_phase must be a non-empty string"
        )
    return TeamIntent(
        commands=tuple(commands),
        atomic=atomic,
        lifecycle_phase=lifecycle_phase,
        metadata=dict(metadata),
    )


def _nonempty_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise AdapterContractError(f"intent command {key} must be a non-empty string")
    return value
