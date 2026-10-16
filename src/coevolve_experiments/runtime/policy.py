"""Common static policy validator and exact AST identity."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import PolicyProposal, RobotIntent, TeamIntent, ValidatedPolicy


class PolicyValidationError(ValueError):
    """Raised before physical dispatch; it never consumes physical budget."""


_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Delete,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Global,
    ast.Nonlocal,
)
_FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "vars",
    "__import__",
}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def canonical_ast(source: str) -> str:
    """Return formatting-insensitive but otherwise exact Python AST identity."""

    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError) as exc:
        raise PolicyValidationError(f"invalid Python policy: {exc}") from exc
    if not tree.body:
        raise PolicyValidationError("policy source is empty")
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def canonical_intent(intent: TeamIntent) -> str:
    """Stable full intent encoding for audit (not exact-blocker identity)."""

    payload = {
        "atomic": intent.atomic,
        "lifecycle_phase": intent.lifecycle_phase,
        "commands": [
            {
                "actor": command.actor,
                "primitive": command.primitive,
                "objective": command.objective,
                "mode": command.mode,
                "arguments": command.arguments,
            }
            for command in sorted(intent.commands, key=lambda item: item.actor)
        ],
        "metadata": intent.metadata,
    }
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PolicyValidationError("intent must contain finite JSON values") from exc


def canonical_executable_intent(intent: TeamIntent) -> str:
    """Only effective command fields; annotations cannot evade a blocker."""

    payload = {
        "commands": [
            {
                "actor": command.actor,
                "primitive": command.primitive,
                "objective": command.objective,
                "mode": command.mode,
                "arguments": command.arguments,
            }
            for command in sorted(intent.commands, key=lambda item: item.actor)
        ]
    }
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PolicyValidationError("intent must contain finite JSON values") from exc


def ast_fingerprint(source: str, intent: TeamIntent | None = None) -> str:
    """Exact executable identity after deterministic intent compilation.

    ``intent`` remains an optional compatibility argument; annotations are
    deliberately excluded because effective intent fields are already in AST.
    """

    identity = canonical_ast(source)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IntentCompiler:
    """Compile TeamIntent into the only executable IR accepted by this layer.

    The live integration may lower these two IR calls to the frozen skill API at
    the protocol/base-commit seam.  Model adapters never author arbitrary code.
    """

    transaction_call: str = "atomic_team_transaction"
    command_call: str = "robot_intent"

    def compile(self, intent: TeamIntent) -> str:
        # Canonicalization both validates JSON values and produces deterministic
        # key ordering before Python literals are emitted.
        payload = json.loads(canonical_executable_intent(intent))
        commands = []
        for command in payload["commands"]:
            commands.append(
                f"{self.command_call}("
                f"actor={command['actor']!r}, "
                f"primitive={command['primitive']!r}, "
                f"objective={command['objective']!r}, "
                f"mode={command['mode']!r}, "
                f"arguments={command['arguments']!r})"
            )
        command_source = ", ".join(commands)
        if len(commands) == 1:
            command_source += ","
        return (
            f"{self.transaction_call}(commands=({command_source}))"
        )


@dataclass(frozen=True, slots=True)
class PolicyValidator:
    """Validate executable syntax and the shared two-actor intent contract."""

    allowed_calls: frozenset[str] | None = None
    required_actors: frozenset[str] = frozenset({"front", "back"})
    compiler: IntentCompiler = IntentCompiler()
    require_compiler_match: bool = True

    def __init__(
        self,
        allowed_calls: Iterable[str] | None = None,
        required_actors: Iterable[str] = ("front", "back"),
        compiler: IntentCompiler | None = None,
        require_compiler_match: bool = True,
    ) -> None:
        if not isinstance(require_compiler_match, bool):
            raise TypeError("require_compiler_match must be a bool")
        object.__setattr__(
            self,
            "allowed_calls",
            None if allowed_calls is None else frozenset(allowed_calls),
        )
        object.__setattr__(self, "required_actors", frozenset(required_actors))
        object.__setattr__(self, "compiler", compiler or IntentCompiler())
        object.__setattr__(self, "require_compiler_match", require_compiler_match)

    def validate(self, proposal: PolicyProposal) -> ValidatedPolicy:
        if not isinstance(proposal, PolicyProposal):
            raise PolicyValidationError("adapter must return PolicyProposal")
        try:
            tree = ast.parse(proposal.source, mode="exec")
        except (SyntaxError, ValueError) as exc:
            raise PolicyValidationError(f"invalid Python policy: {exc}") from exc
        if not tree.body:
            raise PolicyValidationError("policy source is empty")
        self._validate_tree(tree)
        self._validate_intent(proposal.intent)
        dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
        if self.require_compiler_match:
            expected = canonical_ast(self.compiler.compile(proposal.intent))
            if dumped != expected:
                raise PolicyValidationError(
                    "policy source does not match deterministic intent compilation"
                )
        # The compiler AST already contains every effective command field.
        # Lifecycle labels, metadata, and rationale must not reopen a rejected
        # executable by changing annotations alone.
        digest = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
        return ValidatedPolicy(proposal, dumped, digest)

    def _validate_tree(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN_NODES):
                raise PolicyValidationError(
                    f"forbidden AST node: {type(node).__name__}"
                )
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise PolicyValidationError("dunder names are forbidden")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise PolicyValidationError("dunder attributes are forbidden")
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            leaf = name.rsplit(".", 1)[-1] if name else None
            if leaf in _FORBIDDEN_CALLS:
                raise PolicyValidationError(f"forbidden call: {name}")
            if self.allowed_calls is not None and name not in self.allowed_calls:
                if leaf not in self.allowed_calls:
                    raise PolicyValidationError(f"call is outside skill API: {name}")

    def _validate_intent(self, intent: TeamIntent) -> None:
        if not isinstance(intent, TeamIntent):
            raise PolicyValidationError("proposal intent must be TeamIntent")
        if not isinstance(intent.atomic, bool):
            raise PolicyValidationError("intent atomic must be a bool")
        if not intent.atomic:
            raise PolicyValidationError("cooperative transaction must be atomic")
        if not isinstance(intent.lifecycle_phase, str) or not intent.lifecycle_phase:
            raise PolicyValidationError("intent lifecycle_phase must be non-empty")
        actors = [command.actor for command in intent.commands]
        if len(actors) != len(set(actors)):
            raise PolicyValidationError("intent contains duplicate actors")
        if set(actors) != set(self.required_actors):
            expected = ", ".join(sorted(self.required_actors))
            raise PolicyValidationError(f"intent must contain exactly: {expected}")
        for command in intent.commands:
            self._validate_command(command)
        try:
            json.dumps(intent.metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise PolicyValidationError(
                "intent metadata must be finite JSON values"
            ) from exc

    @staticmethod
    def _validate_command(command: RobotIntent) -> None:
        if not isinstance(command, RobotIntent):
            raise PolicyValidationError("intent command must be RobotIntent")
        for label, value in (
            ("actor", command.actor),
            ("primitive", command.primitive),
            ("objective", command.objective),
            ("mode", command.mode),
        ):
            if not isinstance(value, str) or not value.strip():
                raise PolicyValidationError(f"command {label} must be non-empty")
        try:
            json.dumps(command.arguments, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise PolicyValidationError(
                "command arguments must be finite JSON values"
            ) from exc
