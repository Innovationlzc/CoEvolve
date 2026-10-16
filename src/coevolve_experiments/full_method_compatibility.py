"""Byte-recomputable identity for the frozen Full-CoEvolve method contract.

The post-Table-1 runner necessarily adds method adapters and scene injection,
so equality of repository commits is neither possible nor the right fairness
test.  This module instead preserves the exact V83 artifacts that define the
shared Full method.  A production capability is accepted only after these
digests are recomputed from files in its clean checkout.

The recipes intentionally match the V83 formal manifest:

* single files use their raw byte SHA-256;
* source groups hash a canonical ``relative path -> byte SHA-256`` map;
* the static-memory tree hashes a canonical ``tree-relative path -> byte
  SHA-256`` map;
* verifier configuration hashes canonical JSON; and
* the initial empty memory hashes a canonical ``basename -> byte SHA-256``
  map.

Keeping the recipes closed-world prevents a capability from selecting a
convenient unrelated file that happens to preserve an opaque declared digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Tuple

from .serialization import sha256_json


FULL_METHOD_COMPATIBILITY_KIND = "coevolve_full_method_compatibility"
FULL_METHOD_COMPATIBILITY_SCHEMA_VERSION = 1

PROMPT_RELPATH = "src/praxis_coop/cap_policy/prompts.py"
ROUTE_SPEC_RELPATH = "configs/routes/two_g1_box_push_layout.json"
BASE_SCENE_RELPATH = "assets/manual_scenes/two_g1_box_push_layout.usda"
STATIC_MEMORY_ROOT_RELPATH = "results/spec_memory/static"
VERIFIER_CONFIG_RELPATH = (
    "protocol_artifacts/full_method_compatibility/verifier_config.json"
)
INITIAL_ONLINE_MEMORY_ROOT_RELPATH = (
    "protocol_artifacts/full_method_compatibility/initial_online_memory"
)

SKILL_API_RELPATHS: Tuple[str, ...] = (
    "src/praxis_coop/cap_policy/skill_api.py",
    "src/praxis_coop/cap_policy/static_specs.py",
    "src/praxis_coop/cap_policy/coordination_specs.py",
)
VERIFIER_SOURCE_RELPATHS: Tuple[str, ...] = (
    "src/praxis_coop/cap_policy/evaluator.py",
    "src/praxis_coop/cap_policy/simulation_diagnostics.py",
    "src/praxis_coop/cap_policy/route_spec.py",
    "src/praxis_coop/cap_policy/transition_verifier.py",
    "src/praxis_coop/sonic/safety.py",
)
INITIAL_ONLINE_MEMORY_FILENAMES: Tuple[str, ...] = (
    "accepted_episode_manifest.json",
    "action_exemplars.json",
    "raw_rollouts.jsonl",
    "strategy_specs.json",
)
FULL_METHOD_ARTIFACT_FIELDS: Tuple[str, ...] = (
    "initial_online_memory_sha256",
    "prompt_sha256",
    "route_spec_sha256",
    "scene_sha256",
    "skill_api_sha256",
    "static_memory_sha256",
    "verifier_config_sha256",
    "verifier_source_sha256",
)
# Exact source scope captured by every clean V83 run.  New post-Table-1 files
# may be added beside this set, but none of these behavior-bearing files may be
# removed, substituted, or changed while claiming V83 Full compatibility.
V83_FULL_BEHAVIOR_RELPATHS: Tuple[str, ...] = (
    "assets/manual_scenes/two_g1_box_push_layout.usda",
    "assets/manual_scenes/two_g1_box_push_layout_2.usda",
    "assets/manual_scenes/two_g1_box_push_layout_3.usda",
    "assets/manual_scenes/two_g1_box_push_layout_4.usda",
    "assets/manual_scenes/two_g1_box_push_layout_5.usda",
    "assets/manual_scenes/two_g1_box_push_layout_6.usda",
    "assets/manual_scenes/two_g1_box_push_layout_6_plus.usda",
    "configs/routes/two_g1_box_push_layout.json",
    "pyproject.toml",
    "results/spec_memory/static/default_specs.jsonl",
    "scripts/run_cap_box_carry_evolution.py",
    "scripts/run_dual_official_sonic_motor_preflight.py",
    "scripts/run_dual_sonic_manual_layout_sim.py",
    "scripts/run_official_sonic_deploy_zmq_manager.sh",
    "scripts/run_sonic_coordinated_skill.py",
    "scripts/run_sonic_static_skill.py",
    "scripts/validate_static_sonic_skill.py",
    "src/praxis_coop/__init__.py",
    "src/praxis_coop/cap_policy/__init__.py",
    "src/praxis_coop/cap_policy/action_memory.py",
    "src/praxis_coop/cap_policy/action_video.py",
    "src/praxis_coop/cap_policy/backends.py",
    "src/praxis_coop/cap_policy/checkpoints.py",
    "src/praxis_coop/cap_policy/codebase_guard.py",
    "src/praxis_coop/cap_policy/contracts.py",
    "src/praxis_coop/cap_policy/coordination_specs.py",
    "src/praxis_coop/cap_policy/evaluator.py",
    "src/praxis_coop/cap_policy/evolution_agent.py",
    "src/praxis_coop/cap_policy/evolution_controller.py",
    "src/praxis_coop/cap_policy/failure_taxonomy.py",
    "src/praxis_coop/cap_policy/formal_eval.py",
    "src/praxis_coop/cap_policy/generation_recovery.py",
    "src/praxis_coop/cap_policy/local_navigation.py",
    "src/praxis_coop/cap_policy/model_agents.py",
    "src/praxis_coop/cap_policy/model_role_pipeline.py",
    "src/praxis_coop/cap_policy/mutation_contract.py",
    "src/praxis_coop/cap_policy/orchestrator.py",
    "src/praxis_coop/cap_policy/physical_failure_budget.py",
    "src/praxis_coop/cap_policy/policy_executor.py",
    "src/praxis_coop/cap_policy/prompts.py",
    "src/praxis_coop/cap_policy/result_review.py",
    "src/praxis_coop/cap_policy/role_schemas.py",
    "src/praxis_coop/cap_policy/route_spec.py",
    "src/praxis_coop/cap_policy/scene_clearance.py",
    "src/praxis_coop/cap_policy/simulation_diagnostics.py",
    "src/praxis_coop/cap_policy/skill_api.py",
    "src/praxis_coop/cap_policy/static_specs.py",
    "src/praxis_coop/cap_policy/strategy_memory.py",
    "src/praxis_coop/cap_policy/transition_verifier.py",
    "src/praxis_coop/cap_policy/turn_episode.py",
    "src/praxis_coop/cap_policy/visual_observation.py",
    "src/praxis_coop/cap_policy/visual_verifier.py",
    "src/praxis_coop/cap_policy/visualization.py",
    "src/praxis_coop/configs/__init__.py",
    "src/praxis_coop/configs/model_registry.py",
    "src/praxis_coop/llm/__init__.py",
    "src/praxis_coop/llm/errors.py",
    "src/praxis_coop/llm/openai_compatible_client.py",
    "src/praxis_coop/llm/qwen_client.py",
    "src/praxis_coop/sonic/__init__.py",
    "src/praxis_coop/sonic/alignment_feedback.py",
    "src/praxis_coop/sonic/commands.py",
    "src/praxis_coop/sonic/coordination_primitives.py",
    "src/praxis_coop/sonic/linear_distance_feedback.py",
    "src/praxis_coop/sonic/planner_keepalive.py",
    "src/praxis_coop/sonic/runtime_state.py",
    "src/praxis_coop/sonic/safety.py",
    "src/praxis_coop/sonic/skill_presets.py",
    "src/praxis_coop/sonic/team_coordination.py",
    "src/praxis_coop/sonic/team_sync.py",
    "src/praxis_coop/sonic/turn_feedback.py",
    "src/praxis_coop/sonic/upper_body.py",
    "src/praxis_coop/sonic/zmq_publisher.py",
    "src/praxis_coop/sonic_mujoco/__init__.py",
    "src/praxis_coop/sonic_mujoco/checkpoint_store.py",
    "src/praxis_coop/sonic_mujoco/dual_runtime.py",
    "src/praxis_coop/sonic_mujoco/manual_layout_scene.py",
    "src/praxis_coop/sonic_mujoco/offscreen_renderer.py",
    "src/praxis_coop/sonic_mujoco/payload_attitude.py",
    "src/praxis_coop/utils/__init__.py",
    "src/praxis_coop/utils/usda_layout.py",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FullMethodCompatibilityError(ValueError):
    """The frozen Full-method byte identity is missing, malformed, or stale."""


@dataclass(frozen=True, slots=True)
class FullMethodBehaviorArtifact:
    """One V83 behavior-affecting file frozen by relative path and bytes."""

    relpath: str
    sha256: str

    def __post_init__(self) -> None:
        _require_safe_relative_path(self.relpath, "behavior artifact relpath")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise FullMethodCompatibilityError(
                "behavior artifact sha256 must be a lowercase SHA-256"
            )


@dataclass(frozen=True, slots=True)
class FullMethodCompatibilityPin:
    """V83 artifact summaries plus its closed Full behavior-source file set."""

    schema_version: int
    initial_online_memory_sha256: str
    prompt_sha256: str
    route_spec_sha256: str
    scene_sha256: str
    skill_api_sha256: str
    static_memory_sha256: str
    verifier_config_sha256: str
    verifier_source_sha256: str
    behavior_artifacts: Tuple[FullMethodBehaviorArtifact, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != FULL_METHOD_COMPATIBILITY_SCHEMA_VERSION
        ):
            raise FullMethodCompatibilityError(
                "Full-method compatibility schema_version must equal 1"
            )
        hashes = self.artifact_hashes
        for name, digest in hashes.items():
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise FullMethodCompatibilityError(
                    f"Full-method artifact {name} must be a lowercase SHA-256"
                )
        if not isinstance(self.behavior_artifacts, tuple) or not all(
            isinstance(item, FullMethodBehaviorArtifact)
            for item in self.behavior_artifacts
        ):
            raise FullMethodCompatibilityError(
                "behavior_artifacts must be a tuple of FullMethodBehaviorArtifact"
            )
        if not self.behavior_artifacts:
            raise FullMethodCompatibilityError(
                "Full-method compatibility requires behavior-source artifacts"
            )
        paths = tuple(item.relpath for item in self.behavior_artifacts)
        if paths != tuple(sorted(paths)):
            raise FullMethodCompatibilityError(
                "behavior artifacts must be in canonical path order"
            )
        if len(paths) != len(set(paths)):
            raise FullMethodCompatibilityError(
                "behavior artifact paths must be unique"
            )
        if paths != V83_FULL_BEHAVIOR_RELPATHS:
            raise FullMethodCompatibilityError(
                "behavior artifacts must equal the complete frozen V83 source scope"
            )
        if not isinstance(self.fingerprint, str) or not _SHA256_RE.fullmatch(
            self.fingerprint
        ):
            raise FullMethodCompatibilityError(
                "Full-method compatibility fingerprint must be a lowercase SHA-256"
            )
        expected = full_method_compatibility_fingerprint(
            hashes,
            self.behavior_artifacts,
        )
        if self.fingerprint != expected:
            raise FullMethodCompatibilityError(
                "Full-method compatibility fingerprint differs from its artifact hashes"
            )

    @property
    def artifact_hashes(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in FULL_METHOD_ARTIFACT_FIELDS
        }

    @classmethod
    def from_artifact_hashes(
        cls,
        artifact_hashes: Mapping[str, str],
        *,
        behavior_artifacts: Mapping[str, str]
        | Tuple[FullMethodBehaviorArtifact, ...],
    ) -> "FullMethodCompatibilityPin":
        hashes = _exact_artifact_hashes(artifact_hashes)
        behavior = _canonical_behavior_artifacts(behavior_artifacts)
        return cls(
            schema_version=FULL_METHOD_COMPATIBILITY_SCHEMA_VERSION,
            **hashes,
            behavior_artifacts=behavior,
            fingerprint=full_method_compatibility_fingerprint(hashes, behavior),
        )


def full_method_compatibility_fingerprint(
    artifact_hashes: Mapping[str, str],
    behavior_artifacts: Mapping[str, str]
    | Tuple[FullMethodBehaviorArtifact, ...],
) -> str:
    """Domain-separated canonical digest of the complete artifact set."""

    hashes = _exact_artifact_hashes(artifact_hashes)
    behavior = _canonical_behavior_artifacts(behavior_artifacts)
    return sha256_json(
        {
            "kind": FULL_METHOD_COMPATIBILITY_KIND,
            "schema_version": FULL_METHOD_COMPATIBILITY_SCHEMA_VERSION,
            "artifact_hashes": hashes,
            "behavior_artifacts": [
                {"relpath": item.relpath, "sha256": item.sha256}
                for item in behavior
            ],
        }
    )


def recompute_full_method_compatibility(
    runner_repo: str | Path,
    declared: FullMethodCompatibilityPin,
) -> FullMethodCompatibilityPin:
    """Recompute the frozen identity from real files in one runner checkout."""

    repo = Path(runner_repo)
    if not repo.is_absolute() or repo.is_symlink() or not repo.is_dir():
        raise FullMethodCompatibilityError(
            "runner repository must be an absolute non-symlink directory"
        )
    repo = repo.resolve(strict=True)
    prompt = _regular_repo_file(repo, PROMPT_RELPATH)
    route = _regular_repo_file(repo, ROUTE_SPEC_RELPATH)
    scene = _regular_repo_file(repo, BASE_SCENE_RELPATH)
    verifier_config_path = _regular_repo_file(repo, VERIFIER_CONFIG_RELPATH)

    skill_hashes = {
        relpath: _sha256_file(_regular_repo_file(repo, relpath))
        for relpath in SKILL_API_RELPATHS
    }
    verifier_hashes = {
        relpath: _sha256_file(_regular_repo_file(repo, relpath))
        for relpath in VERIFIER_SOURCE_RELPATHS
    }
    verifier_config = _strict_json_object(verifier_config_path)

    static_root = _repo_directory(repo, STATIC_MEMORY_ROOT_RELPATH)
    static_hashes = _tree_hash_map(static_root)
    if not static_hashes:
        raise FullMethodCompatibilityError("frozen static-memory tree is empty")

    initial_root = _repo_directory(repo, INITIAL_ONLINE_MEMORY_ROOT_RELPATH)
    actual_initial_names = tuple(
        sorted(path.name for path in initial_root.iterdir())
    )
    if actual_initial_names != tuple(sorted(INITIAL_ONLINE_MEMORY_FILENAMES)):
        raise FullMethodCompatibilityError(
            "initial online-memory snapshot must contain exactly the four "
            "canonical V83 files"
        )
    initial_hashes = {
        name: _sha256_file(_regular_repo_file(initial_root, name))
        for name in INITIAL_ONLINE_MEMORY_FILENAMES
    }
    behavior_hashes = {
        item.relpath: _sha256_file(
            _regular_repo_file(repo, item.relpath)
        )
        for item in declared.behavior_artifacts
    }

    return FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": sha256_json(initial_hashes),
            "prompt_sha256": _sha256_file(prompt),
            "route_spec_sha256": _sha256_file(route),
            "scene_sha256": _sha256_file(scene),
            "skill_api_sha256": sha256_json(skill_hashes),
            "static_memory_sha256": sha256_json(static_hashes),
            "verifier_config_sha256": sha256_json(verifier_config),
            "verifier_source_sha256": sha256_json(verifier_hashes),
        },
        behavior_artifacts=behavior_hashes,
    )


def derive_full_method_compatibility_from_v83_snapshot(
    source_file_hashes: Mapping[str, Any],
    *,
    verifier_config: Mapping[str, Any],
    initial_online_memory_sha256: str,
) -> FullMethodCompatibilityPin:
    """Derive the same identity from an audited V83 codebase snapshot.

    V83 retained a canonical per-file byte-hash map in
    ``codebase_manifest.json``.  Its run manifest independently binds that
    map's canonical hash.  This function applies exactly the same grouping
    recipes used by the live capability recomputation.
    """

    files: dict[str, str] = {}
    for key, value in source_file_hashes.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise FullMethodCompatibilityError(
                "V83 codebase snapshot files must map paths to SHA-256 strings"
            )
        if not _SHA256_RE.fullmatch(value):
            raise FullMethodCompatibilityError(
                f"V83 codebase snapshot hash is malformed for {key!r}"
            )
        files[key] = value

    required_files = {
        PROMPT_RELPATH,
        ROUTE_SPEC_RELPATH,
        BASE_SCENE_RELPATH,
        *SKILL_API_RELPATHS,
        *VERIFIER_SOURCE_RELPATHS,
    }
    missing = sorted(required_files.difference(files))
    if missing:
        raise FullMethodCompatibilityError(
            f"V83 codebase snapshot lacks Full-method files: {missing}"
        )
    static_prefix = STATIC_MEMORY_ROOT_RELPATH + "/"
    static_hashes = {
        path.removeprefix(static_prefix): digest
        for path, digest in files.items()
        if path.startswith(static_prefix)
    }
    if not static_hashes:
        raise FullMethodCompatibilityError(
            "V83 codebase snapshot lacks the static-memory tree"
        )
    if not isinstance(initial_online_memory_sha256, str) or not _SHA256_RE.fullmatch(
        initial_online_memory_sha256
    ):
        raise FullMethodCompatibilityError(
            "V83 initial online-memory hash must be a lowercase SHA-256"
        )
    if not isinstance(verifier_config, Mapping):
        raise FullMethodCompatibilityError("V83 verifier config must be an object")

    return FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": initial_online_memory_sha256,
            "prompt_sha256": files[PROMPT_RELPATH],
            "route_spec_sha256": files[ROUTE_SPEC_RELPATH],
            "scene_sha256": files[BASE_SCENE_RELPATH],
            "skill_api_sha256": sha256_json(
                {path: files[path] for path in SKILL_API_RELPATHS}
            ),
            "static_memory_sha256": sha256_json(static_hashes),
            "verifier_config_sha256": sha256_json(dict(verifier_config)),
            "verifier_source_sha256": sha256_json(
                {path: files[path] for path in VERIFIER_SOURCE_RELPATHS}
            ),
        },
        behavior_artifacts=files,
    )


def full_method_compatibility_pin_from_mapping(
    value: Mapping[str, Any],
) -> FullMethodCompatibilityPin:
    """Strict JSON-boundary parser for a compatibility pin."""

    if not isinstance(value, Mapping):
        raise FullMethodCompatibilityError(
            "Full-method compatibility pin must be an object"
        )
    expected = {
        "schema_version",
        *FULL_METHOD_ARTIFACT_FIELDS,
        "behavior_artifacts",
        "fingerprint",
    }
    actual = set(value)
    if actual != expected:
        raise FullMethodCompatibilityError(
            "Full-method compatibility pin fields differ: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    raw_behavior = value.get("behavior_artifacts")
    if not isinstance(raw_behavior, list):
        raise FullMethodCompatibilityError(
            "Full-method behavior_artifacts must be a JSON array"
        )
    behavior: list[FullMethodBehaviorArtifact] = []
    for index, item in enumerate(raw_behavior):
        if not isinstance(item, Mapping) or set(item) != {"relpath", "sha256"}:
            raise FullMethodCompatibilityError(
                f"behavior_artifacts[{index}] must contain exactly relpath and sha256"
            )
        try:
            behavior.append(FullMethodBehaviorArtifact(**dict(item)))
        except (TypeError, ValueError) as exc:
            raise FullMethodCompatibilityError(
                f"invalid behavior_artifacts[{index}]: {exc}"
            ) from exc
    kwargs = {
        name: value[name]
        for name in (
            "schema_version",
            *FULL_METHOD_ARTIFACT_FIELDS,
            "fingerprint",
        )
    }
    try:
        return FullMethodCompatibilityPin(
            **kwargs,
            behavior_artifacts=tuple(behavior),
        )
    except (TypeError, ValueError) as exc:
        raise FullMethodCompatibilityError(
            f"invalid Full-method compatibility pin: {exc}"
        ) from exc


def _exact_artifact_hashes(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise FullMethodCompatibilityError("Full-method artifact hashes must be an object")
    actual = set(value)
    expected = set(FULL_METHOD_ARTIFACT_FIELDS)
    if actual != expected:
        raise FullMethodCompatibilityError(
            "Full-method artifact fields differ: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    hashes = {name: value[name] for name in FULL_METHOD_ARTIFACT_FIELDS}
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise FullMethodCompatibilityError(
                f"Full-method artifact {name} must be a lowercase SHA-256"
            )
    return hashes


def _canonical_behavior_artifacts(
    value: Mapping[str, str] | Tuple[FullMethodBehaviorArtifact, ...],
) -> Tuple[FullMethodBehaviorArtifact, ...]:
    if isinstance(value, Mapping):
        artifacts = tuple(
            FullMethodBehaviorArtifact(relpath=path, sha256=digest)
            for path, digest in sorted(value.items())
        )
    elif isinstance(value, tuple):
        artifacts = value
    else:
        raise FullMethodCompatibilityError(
            "behavior artifacts must be a mapping or tuple"
        )
    if not artifacts:
        raise FullMethodCompatibilityError(
            "Full-method compatibility requires behavior-source artifacts"
        )
    if not all(isinstance(item, FullMethodBehaviorArtifact) for item in artifacts):
        raise FullMethodCompatibilityError(
            "behavior artifacts contain an invalid entry"
        )
    paths = tuple(item.relpath for item in artifacts)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise FullMethodCompatibilityError(
            "behavior artifacts must have unique canonically ordered paths"
        )
    return artifacts


def _require_safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FullMethodCompatibilityError(f"{label} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise FullMethodCompatibilityError(
            f"{label} must be a normalized relative POSIX path"
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise FullMethodCompatibilityError(f"{label} must not contain dot segments")
    return value


def _regular_repo_file(repo: Path, relpath: str) -> Path:
    candidate = repo / relpath
    if candidate.is_symlink() or not candidate.is_file():
        raise FullMethodCompatibilityError(
            f"Full-method artifact must be a regular non-symlink file: {relpath}"
        )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise FullMethodCompatibilityError(
            f"Full-method artifact escapes the runner repository: {relpath}"
        ) from exc
    if resolved != candidate:
        raise FullMethodCompatibilityError(
            f"Full-method artifact path contains a symlink: {relpath}"
        )
    return resolved


def _repo_directory(repo: Path, relpath: str) -> Path:
    candidate = repo / relpath
    if candidate.is_symlink() or not candidate.is_dir():
        raise FullMethodCompatibilityError(
            f"Full-method artifact tree must be a non-symlink directory: {relpath}"
        )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise FullMethodCompatibilityError(
            f"Full-method artifact tree escapes the runner repository: {relpath}"
        ) from exc
    if resolved != candidate:
        raise FullMethodCompatibilityError(
            f"Full-method artifact tree path contains a symlink: {relpath}"
        )
    return resolved


def _tree_hash_map(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FullMethodCompatibilityError(
                f"Full-method artifact tree contains a symlink: {path}"
            )
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = _sha256_file(path)
    return hashes


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FullMethodCompatibilityError(
            f"cannot read Full-method artifact {path}: {exc}"
        ) from exc


def _strict_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise FullMethodCompatibilityError(
            f"cannot read Full-method verifier config: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FullMethodCompatibilityError(
            "Full-method verifier config must be a JSON object"
        )
    return payload


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


__all__ = [
    "BASE_SCENE_RELPATH",
    "FULL_METHOD_ARTIFACT_FIELDS",
    "FULL_METHOD_COMPATIBILITY_KIND",
    "FULL_METHOD_COMPATIBILITY_SCHEMA_VERSION",
    "FullMethodCompatibilityError",
    "FullMethodBehaviorArtifact",
    "FullMethodCompatibilityPin",
    "INITIAL_ONLINE_MEMORY_FILENAMES",
    "INITIAL_ONLINE_MEMORY_ROOT_RELPATH",
    "PROMPT_RELPATH",
    "ROUTE_SPEC_RELPATH",
    "SKILL_API_RELPATHS",
    "STATIC_MEMORY_ROOT_RELPATH",
    "VERIFIER_CONFIG_RELPATH",
    "VERIFIER_SOURCE_RELPATHS",
    "V83_FULL_BEHAVIOR_RELPATHS",
    "derive_full_method_compatibility_from_v83_snapshot",
    "full_method_compatibility_fingerprint",
    "full_method_compatibility_pin_from_mapping",
    "recompute_full_method_compatibility",
]
