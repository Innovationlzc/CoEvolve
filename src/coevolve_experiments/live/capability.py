"""Strict capability document for a production Praxis Code-as-Policy runner.

The document is deliberately small and closed-world.  It is not a generic
command description and cannot carry argv, environment, or method overrides.
It records the frozen base-method commit separately from the clean integration
checkout whose HEAD contains the byte-pinned, repository-relative entrypoint.
A per-run launch configuration is bound in ``bridge.py`` and is the runner's
sole command-line input.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping, Tuple

from ..contracts import MethodVariant, SceneId
from ..full_method_compatibility import (
    FullMethodCompatibilityError,
    FullMethodCompatibilityPin,
    full_method_compatibility_pin_from_mapping,
    recompute_full_method_compatibility,
)
from .contracts import ExecutionCapability, PolicyRepresentation


RUNNER_CAPABILITY_KIND = "coevolve_praxis_code_policy_runner_capability"
RUNNER_CAPABILITY_SCHEMA_VERSION = 4
RUNNER_LAUNCH_CONFIG_SCHEMA_VERSION = 4
RUNNER_RESULT_CONTRACT_SCHEMA_VERSION = 4

SUPPORTED_METHOD_VARIANTS: Tuple[str, ...] = tuple(
    variant.value for variant in MethodVariant
)
SUPPORTED_SCENE_IDS: Tuple[str, ...] = tuple(scene.value for scene in SceneId)
REQUIRED_SCENE_INJECTION_CHANNELS: Tuple[str, ...] = (
    "term1_simulator_asset",
    "host_clearance_geometry",
    "vlm_scene_prior",
    "route_and_progress_gates",
    "formal_run_manifest",
)
RUNNER_ACCEPTED_CLI_ARGUMENTS: Tuple[str, ...] = ("--launch-config",)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CAPABILITY_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "runner_repo",
        "base_method_commit",
        "runner_integration_commit",
        "clean_worktree_required",
        "entrypoint_relpath",
        "entrypoint_sha256",
        "policy_representation",
        "execution_capability",
        "supported_method_variants",
        "supported_scene_ids",
        "scene_injection_channels",
        "accepted_cli_arguments",
        "freeform_overrides_allowed",
        "launch_config_schema_version",
        "result_contract_schema_version",
        "full_method_compatibility",
    }
)


class RunnerCapabilityError(ValueError):
    """Raised when a claimed production capability is absent or not exact."""


@dataclass(frozen=True, slots=True)
class RunnerCapability:
    manifest_path: Path
    manifest_sha256: str
    runner_repo: Path
    base_method_commit: str
    runner_integration_commit: str
    entrypoint_path: Path
    entrypoint_sha256: str
    policy_representation: PolicyRepresentation
    execution_capability: ExecutionCapability
    supported_method_variants: Tuple[str, ...]
    supported_scene_ids: Tuple[str, ...]
    scene_injection_channels: Tuple[str, ...]
    accepted_cli_arguments: Tuple[str, ...]
    launch_config_schema_version: int
    result_contract_schema_version: int
    full_method_compatibility: FullMethodCompatibilityPin


def load_runner_capability(
    path: str | Path,
    *,
    expected_base_method_commit: str | None = None,
    expected_full_method_compatibility_fingerprint: str | None = None,
) -> RunnerCapability:
    """Load and independently attest one exact runner capability document."""

    manifest_path = _absolute_regular_path(path, "runner capability manifest")
    raw, payload = _read_exact_json_object(manifest_path, "runner capability manifest")
    _require_exact_fields(payload, _CAPABILITY_FIELDS, "runner capability manifest")

    if payload.get("kind") != RUNNER_CAPABILITY_KIND:
        raise RunnerCapabilityError("runner capability manifest has the wrong kind")
    _require_exact_integer(
        payload.get("schema_version"),
        RUNNER_CAPABILITY_SCHEMA_VERSION,
        "runner capability schema_version",
    )
    if payload.get("clean_worktree_required") is not True:
        raise RunnerCapabilityError(
            "runner capability must require a clean worktree"
        )
    if payload.get("freeform_overrides_allowed") is not False:
        raise RunnerCapabilityError(
            "runner capability must forbid freeform overrides"
        )
    _require_exact_integer(
        payload.get("launch_config_schema_version"),
        RUNNER_LAUNCH_CONFIG_SCHEMA_VERSION,
        "runner launch_config_schema_version",
    )
    _require_exact_integer(
        payload.get("result_contract_schema_version"),
        RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
        "runner result_contract_schema_version",
    )

    policy = _enum_exact(
        PolicyRepresentation,
        payload.get("policy_representation"),
        "runner policy_representation",
    )
    if policy is not PolicyRepresentation.PRAXIS_CODE_AS_POLICY:
        raise RunnerCapabilityError(
            "runner must execute the Praxis Code-as-Policy representation"
        )
    execution = _enum_exact(
        ExecutionCapability,
        payload.get("execution_capability"),
        "runner execution_capability",
    )
    if execution is not ExecutionCapability.PAPER_FORMAL:
        raise RunnerCapabilityError("runner must declare paper_formal capability")

    methods = _require_exact_string_list(
        payload.get("supported_method_variants"),
        SUPPORTED_METHOD_VARIANTS,
        "supported_method_variants",
    )
    scenes = _require_exact_string_list(
        payload.get("supported_scene_ids"),
        SUPPORTED_SCENE_IDS,
        "supported_scene_ids",
    )
    channels = _require_exact_string_list(
        payload.get("scene_injection_channels"),
        REQUIRED_SCENE_INJECTION_CHANNELS,
        "scene_injection_channels",
    )
    cli_arguments = _require_exact_string_list(
        payload.get("accepted_cli_arguments"),
        RUNNER_ACCEPTED_CLI_ARGUMENTS,
        "accepted_cli_arguments",
    )

    repo_text = payload.get("runner_repo")
    if not isinstance(repo_text, str) or not repo_text:
        raise RunnerCapabilityError("runner_repo must be a non-empty string")
    repo = _absolute_directory_path(repo_text, "runner repository")
    base_method_commit = payload.get("base_method_commit")
    if not isinstance(base_method_commit, str) or not _GIT_COMMIT_RE.fullmatch(
        base_method_commit
    ):
        raise RunnerCapabilityError(
            "base_method_commit must be a lowercase 40-hex commit"
        )
    if (
        expected_base_method_commit is not None
        and base_method_commit != expected_base_method_commit
    ):
        raise RunnerCapabilityError(
            "capability base_method_commit differs from RunSpec.base_method_commit"
        )
    integration_commit = payload.get("runner_integration_commit")
    if not isinstance(integration_commit, str) or not _GIT_COMMIT_RE.fullmatch(
        integration_commit
    ):
        raise RunnerCapabilityError(
            "runner_integration_commit must be a lowercase 40-hex commit"
        )
    current_commit = _git_output(repo, ("rev-parse", "--verify", "HEAD^{commit}"))
    if current_commit != integration_commit:
        raise RunnerCapabilityError(
            "runner repository HEAD differs from runner_integration_commit"
        )
    dirty = _git_output(
        repo,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        allow_empty=True,
    )
    if dirty:
        raise RunnerCapabilityError("runner repository worktree is not clean")

    compatibility_payload = payload.get("full_method_compatibility")
    if not isinstance(compatibility_payload, Mapping):
        raise RunnerCapabilityError(
            "runner full_method_compatibility must be an object"
        )
    compatibility_fields = frozenset(
        item.name for item in fields(FullMethodCompatibilityPin)
    )
    _require_exact_fields(
        compatibility_payload,
        compatibility_fields,
        "runner full_method_compatibility",
    )
    try:
        declared_compatibility = full_method_compatibility_pin_from_mapping(
            compatibility_payload
        )
        recomputed_compatibility = recompute_full_method_compatibility(
            repo,
            declared_compatibility,
        )
    except (TypeError, ValueError, FullMethodCompatibilityError) as exc:
        raise RunnerCapabilityError(
            f"runner Full-method compatibility artifacts are invalid: {exc}"
        ) from exc
    if declared_compatibility != recomputed_compatibility:
        raise RunnerCapabilityError(
            "runner Full-method compatibility declaration differs from "
            "recomputed artifact bytes"
        )
    if (
        expected_full_method_compatibility_fingerprint is not None
        and recomputed_compatibility.fingerprint
        != expected_full_method_compatibility_fingerprint
    ):
        raise RunnerCapabilityError(
            "runner Full-method compatibility identity differs from the RunSpec"
        )

    relpath_text = payload.get("entrypoint_relpath")
    relpath = _safe_relative_posix_path(relpath_text, "entrypoint_relpath")
    entrypoint_candidate = repo / relpath.as_posix()
    if any(
        candidate.is_symlink()
        for candidate in (
            entrypoint_candidate,
            *entrypoint_candidate.parents[: len(relpath.parts) - 1],
        )
    ):
        raise RunnerCapabilityError(
            "runner entrypoint path must not contain symlink components"
        )
    entrypoint = entrypoint_candidate.resolve(strict=True)
    try:
        entrypoint.relative_to(repo)
    except ValueError as exc:
        raise RunnerCapabilityError(
            "runner entrypoint escapes the runner repository"
        ) from exc
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise RunnerCapabilityError(
            "runner entrypoint must be a non-symlink regular file"
        )
    mode = entrypoint.stat().st_mode
    if not stat.S_ISREG(mode) or not os.access(entrypoint, os.X_OK):
        raise RunnerCapabilityError("runner entrypoint must be executable")
    entrypoint_sha256 = payload.get("entrypoint_sha256")
    if not isinstance(entrypoint_sha256, str) or not _SHA256_RE.fullmatch(
        entrypoint_sha256
    ):
        raise RunnerCapabilityError(
            "entrypoint_sha256 must be a lowercase SHA-256"
        )
    actual_entrypoint_sha256 = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    if actual_entrypoint_sha256 != entrypoint_sha256:
        raise RunnerCapabilityError("runner entrypoint byte hash differs")

    return RunnerCapability(
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        runner_repo=repo,
        base_method_commit=base_method_commit,
        runner_integration_commit=integration_commit,
        entrypoint_path=entrypoint,
        entrypoint_sha256=entrypoint_sha256,
        policy_representation=policy,
        execution_capability=execution,
        supported_method_variants=methods,
        supported_scene_ids=scenes,
        scene_injection_channels=channels,
        accepted_cli_arguments=cli_arguments,
        launch_config_schema_version=RUNNER_LAUNCH_CONFIG_SCHEMA_VERSION,
        result_contract_schema_version=RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
        full_method_compatibility=recomputed_compatibility,
    )


def _absolute_regular_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RunnerCapabilityError(f"{label} path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise RunnerCapabilityError(
            f"{label} path must be a non-symlink existing regular file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise RunnerCapabilityError(
            f"{label} path must be normalized and contain no symlink components"
        )
    return resolved


def _absolute_directory_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RunnerCapabilityError(f"{label} path must be absolute")
    if path.is_symlink() or not path.is_dir():
        raise RunnerCapabilityError(
            f"{label} path must be a non-symlink existing directory"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise RunnerCapabilityError(
            f"{label} path must be normalized and contain no symlink components"
        )
    return resolved


def _read_exact_json_object(
    path: Path, label: str
) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RunnerCapabilityError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RunnerCapabilityError(f"{label} must be a JSON object")
    return raw, payload


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _require_exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise RunnerCapabilityError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_exact_integer(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise RunnerCapabilityError(f"{label} must equal integer {expected}")


def _require_exact_string_list(
    value: Any, expected: Tuple[str, ...], label: str
) -> Tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RunnerCapabilityError(f"{label} must be an array of strings")
    result = tuple(value)
    if result != expected:
        raise RunnerCapabilityError(
            f"{label} must equal the complete canonical set {list(expected)!r}"
        )
    return result


def _enum_exact(enum_type: type, value: Any, label: str):
    if not isinstance(value, str):
        raise RunnerCapabilityError(f"{label} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise RunnerCapabilityError(f"{label} is unsupported: {value!r}") from exc


def _safe_relative_posix_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RunnerCapabilityError(f"{label} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise RunnerCapabilityError(f"{label} must be a normalized relative POSIX path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise RunnerCapabilityError(f"{label} must not contain dot segments")
    return path


def _git_output(
    repo: Path, arguments: Tuple[str, ...], *, allow_empty: bool = False
) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            shell=False,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise RunnerCapabilityError(f"cannot inspect runner git repository: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git command failed"
        raise RunnerCapabilityError(f"cannot inspect runner git repository: {detail}")
    value = completed.stdout.strip()
    if not value and not allow_empty:
        raise RunnerCapabilityError("runner git command returned no identity")
    return value


__all__ = [
    "REQUIRED_SCENE_INJECTION_CHANNELS",
    "RUNNER_ACCEPTED_CLI_ARGUMENTS",
    "RUNNER_CAPABILITY_KIND",
    "RUNNER_CAPABILITY_SCHEMA_VERSION",
    "RUNNER_LAUNCH_CONFIG_SCHEMA_VERSION",
    "RUNNER_RESULT_CONTRACT_SCHEMA_VERSION",
    "SUPPORTED_METHOD_VARIANTS",
    "SUPPORTED_SCENE_IDS",
    "RunnerCapability",
    "RunnerCapabilityError",
    "load_runner_capability",
]
