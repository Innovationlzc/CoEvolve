"""Checkpoint-local recovery for pre-dispatch model-role stalls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

from .contracts import write_json


MAX_CONSECUTIVE_SYSTEM_STALLS_AT_CHECKPOINT = 3
MAX_IDENTICAL_TYPED_ROLE_FAILURES = 2
MAX_INFEASIBLE_SCOPE_BACKTRACKS_AT_CHECKPOINT = 3
ROLE_GENERATION_STALL = "ROLE_GENERATION_STALL"
NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL = "NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL"
MUTATION_SCOPE_INFEASIBLE = "MUTATION_SCOPE_INFEASIBLE"
NO_EXECUTABLE_CHILD_PRODUCED = "NO_EXECUTABLE_CHILD_PRODUCED"
SCOPE_MISMATCH = "SCOPE_MISMATCH"
EXACT_BLOCKED = "EXACT_BLOCKED"
NO_EFFECTIVE_DELTA = "NO_EFFECTIVE_DELTA"
TYPED_ROLE_FAILURES = frozenset(
    {SCOPE_MISMATCH, EXACT_BLOCKED, NO_EFFECTIVE_DELTA}
)
EDITABLE_FIELDS_BY_LEVEL = {
    "coordination_mutation": ("coordination_mode", "temporary_roles"),
    "strategy_mutation": ("primitive", "semantic_objective"),
    "parameter_tuning": ("exactly_one_action_parameter",),
}
_SCOPE_MISMATCH_MARKERS = (
    "ORCHESTRATOR_CHILD_MISMATCH",
    "coordination_mutation task-plan contract preserves",
    "coordination_mutation mutable mask preserves",
    "parameter_tuning strategy contract requires",
    "parameter_tuning mutable mask permits",
    "mutation_level must remain",
    "strategy_mutation cannot rename the same body-frame",
    "strategy_mutation must change the executable primitive",
)
_EXACT_BLOCKED_MARKERS = (
    "exact verifier-rejected executable AST",
    "blocked_fingerprint=",
    "repeats a verifier-rejected executable transaction",
    "exact rejected AST",
)
_KNOWN_CHANGE_FIELDS = (
    "primitive",
    "semantic_objective",
    "coordination_mode",
    "formation_strategy",
    "temporary_roles",
)


def counts_as_model_generation_stall(cycle_outcome: str) -> bool:
    """Return whether a pre-dispatch outcome is attributed to the model.

    A deterministically empty selected mutation operator is a Host-contract
    fact rather than a model failure. It still consumes that complete
    checkpoint-local generation cycle; the total three-cycle cap is tracked
    independently by ``generation_cycle``.
    """

    return str(cycle_outcome).strip() != MUTATION_SCOPE_INFEASIBLE


def classify_typed_generation_failure(error: str) -> str:
    """Return a Host-typed generation failure, or empty for ordinary schema retry."""

    text = str(error)
    if "NO_EFFECTIVE_MUTATION" in text:
        return NO_EFFECTIVE_DELTA
    if any(marker in text for marker in _SCOPE_MISMATCH_MARKERS):
        return SCOPE_MISMATCH
    if any(marker in text for marker in _EXACT_BLOCKED_MARKERS):
        return EXACT_BLOCKED
    return ""


def editable_fields_for_mutation_level(level: str) -> tuple[str, ...]:
    """Name the currently selected mutation operator's legal fields only."""

    return EDITABLE_FIELDS_BY_LEVEL.get(str(level).strip(), ())


def attempted_change_from_error(error: str, failure: str) -> tuple[str, ...]:
    """Extract which executable fields the model tried to change, without naming a fix."""

    text = str(error)
    if failure == NO_EFFECTIVE_DELTA:
        return ("prose_only",) if "prose_only" in text else ()
    if failure != SCOPE_MISMATCH:
        return ()
    blob = ""
    for marker in ("mismatches=", "changed=", "changed strategy fields="):
        if marker in text:
            blob = text.split(marker, 1)[1]
            break
    if not blob:
        return ()
    found: list[str] = []
    for field_name in _KNOWN_CHANGE_FIELDS:
        if re.search(rf"['\"]{field_name}['\"]", blob) and field_name not in found:
            found.append(field_name)
    return tuple(found)


def consecutive_trailing_kind_count(kinds: list[str], kind: str) -> int:
    """Count how many times `kind` closed the current role retry streak."""

    count = 0
    for item in reversed(kinds):
        if item != kind:
            break
        count += 1
    return count


def typed_generation_feedback(
    *,
    selected_mutation_level: str,
    failure: str,
    attempted_change: tuple[str, ...] = (),
    error: str = "",
) -> dict[str, Any]:
    """Structured evidence for Evolution Manager. Does not name a replacement action."""

    attempted = attempted_change or attempted_change_from_error(error, failure)
    return {
        "selected_mutation_level": str(selected_mutation_level).strip(),
        "failure": str(failure).strip(),
        "attempted_change": list(attempted),
        "editable_fields": list(
            editable_fields_for_mutation_level(selected_mutation_level)
        ),
        "result": NO_EXECUTABLE_CHILD_PRODUCED,
    }


class SystemStallError(RuntimeError):
    """Raised after complete generation cycles repeatedly stall at one checkpoint."""


class HostContractError(RuntimeError):
    """Raised when Host-owned hard contracts have no executable intersection."""


@dataclass
class GenerationStallTracker:
    """Track consecutive role-generation stalls without consuming physical budget."""

    output_dir: Path
    maximum_consecutive_stalls: int = MAX_CONSECUTIVE_SYSTEM_STALLS_AT_CHECKPOINT
    checkpoint_id: str = ""
    consecutive_stalls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        checkpoint_id: str,
        generation_cycle: int,
        role: str,
        artifact_id: str,
        error: str,
        response_count: int,
        cycle_outcome: str = ROLE_GENERATION_STALL,
        selected_mutation_level: str = "",
        validation_errors: tuple[str, ...] = (),
        validation_error_kinds: tuple[str, ...] = (),
        typed_failure: str = "",
        attempted_change: tuple[str, ...] = (),
        editable_fields: tuple[str, ...] = (),
        result: str = "",
    ) -> dict[str, Any]:
        checkpoint_id = str(checkpoint_id).strip()
        outcome = str(cycle_outcome).strip() or ROLE_GENERATION_STALL
        if checkpoint_id != self.checkpoint_id:
            self.checkpoint_id = checkpoint_id
            self.consecutive_stalls = 0
        model_stall = counts_as_model_generation_stall(outcome)
        if model_stall:
            self.consecutive_stalls += 1
        event = {
            "event": ROLE_GENERATION_STALL,
            "cycle_outcome": outcome,
            "checkpoint_id": checkpoint_id,
            "generation_cycle": int(generation_cycle),
            "role": str(role),
            "artifact_id": str(artifact_id),
            "selected_mutation_level": str(selected_mutation_level).strip(),
            "response_count": int(response_count),
            "consecutive_stalls_at_checkpoint": self.consecutive_stalls,
            "maximum_consecutive_stalls_at_checkpoint": (
                self.maximum_consecutive_stalls
            ),
            "physical_dispatch": False,
            "memory_promotion": False,
            "checkpoint_change": False,
            "error": str(error),
            "validation_errors": [str(item) for item in validation_errors],
            "validation_error_kinds": [
                str(item) for item in validation_error_kinds
            ],
            "typed_failure": str(typed_failure).strip(),
            "attempted_change": [str(item) for item in attempted_change],
            "editable_fields": [str(item) for item in editable_fields],
            "result": str(result).strip(),
            "counts_as_model_stall": model_stall,
            # Every complete Manager-to-validation cycle consumes the frozen
            # checkpoint-local cycle cap. An empty selected scope remains Host
            # evidence rather than model blame, but it cannot open a fourth
            # generation cycle.
            "system_stall": int(generation_cycle)
            >= self.maximum_consecutive_stalls,
        }
        self.events.append(event)
        write_json(
            Path(self.output_dir) / "generation_stalls.json",
            {
                "tracker": {
                    key: value
                    for key, value in asdict(self).items()
                    if key != "events"
                },
                "events": self.events,
            },
        )
        return event

    def reset_after_physical_transaction(self) -> None:
        """Any accepted or rejected physical transaction starts a fresh audit."""

        self.checkpoint_id = ""
        self.consecutive_stalls = 0

    def reset_after_successful_role_cycle(self) -> None:
        """A recovered role invocation breaks the consecutive-stall sequence."""

        self.checkpoint_id = ""
        self.consecutive_stalls = 0
