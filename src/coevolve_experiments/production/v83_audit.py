"""Read-only checkout verification using caller-supplied qualification pins.

Historical commit IDs and source hashes are intentionally not distributed.
Callers must supply independently reviewed pins; never infer them from the
checkout under test or rewrite old evidence to match a public source snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Mapping, Tuple

from .contracts import MigrationRequirement


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FrozenV83AuditError(ValueError):
    """The supplied checkout is not the exact clean audited V83 snapshot."""


@dataclass(frozen=True, slots=True)
class FrozenV83Audit:
    repo: Path
    commit: str
    tag: str
    file_hashes: Tuple[Tuple[str, str], ...]


V83_POST_TABLE_MIGRATION_REQUIREMENTS: Tuple[MigrationRequirement, ...] = (
    MigrationRequirement(
        "V83_LAUNCH_CONFIG_ENTRYPOINT_MISSING",
        "formal_run_manifest",
        "scripts/run_cap_box_carry_evolution.py accepts many independent CLI flags",
        "add a sole --launch-config entrypoint and reject every override surface",
    ),
    MigrationRequirement(
        "V83_SCENE_ASSET_INJECTION_MISSING",
        "term1_simulator_asset",
        "formal_eval.py freezes the authored base scene and target geometry",
        "inject the plan-bound realized USDA and verify its byte/composite hashes",
    ),
    MigrationRequirement(
        "V83_HOST_CLEARANCE_INJECTION_MISSING",
        "host_clearance_geometry",
        "V83 reads only its base-scene clearance/route implementation",
        "bind the selected SceneSpec corridor geometry into every Host verifier",
    ),
    MigrationRequirement(
        "V83_VLM_SCENE_PRIOR_INJECTION_MISSING",
        "vlm_scene_prior",
        "V83 camera and target priors are tied to the base scene",
        "bind the selected camera, target, initial state, and scene label",
    ),
    MigrationRequirement(
        "V83_ROUTE_GATE_INJECTION_MISSING",
        "route_and_progress_gates",
        "formal_eval.py fixes configs/routes/two_g1_box_push_layout.json",
        "load the selected scene's hash-bound directed route and progress gates",
    ),
    MigrationRequirement(
        "V83_METHOD_GATE_INJECTION_MISSING",
        "method_gates",
        "the frozen runner exposes only the Full V83 role/evolution pipeline",
        "wire each comparison/ablation decision structure into the same Praxis loop",
    ),
    MigrationRequirement(
        "V83_MEMORY_GATE_INJECTION_MISSING",
        "memory_gates",
        "formal V83 always starts empty run-local dynamic memory",
        "enforce disabled/from-empty/transfer gates and hash-bound bootstrap paths",
    ),
    MigrationRequirement(
        "V83_PRAXIS_MEMORY_SCHEMA_BRIDGE_MISSING",
        "memory_gates",
        (
            "the contract-smoke lifecycle uses online_strategy_overlay.json, while "
            "V83 executes its StrategyMemoryStore over strategy_specs.json and its "
            "native action-exemplar schema"
        ),
        (
            "implement and audit an explicit bridge that applies the RunSpec gates "
            "to the real Praxis stores and emits the finalized lifecycle audit"
        ),
    ),
    MigrationRequirement(
        "V83_PROVIDER_AND_OUTPUT_BINDING_MISSING",
        "provider_route",
        "formal_eval.py hard-codes the PRIMARY route and V83 result directory layout",
        "consume the registered route/model and exact post-Table output binding",
    ),
    MigrationRequirement(
        "V83_RESULT_EVIDENCE_V3_MISSING",
        "formal_output",
        "V83 does not emit the post-Table terminal raw trace and FM usage ledgers",
        (
            "emit the live result-binding schema v4 plus terminal-evidence schema v3, "
            "stall/drop, scene, and usage evidence"
        ),
    ),
)


def audit_frozen_v83_checkout(
    repo: str | Path,
    *,
    expected_commit: str,
    expected_file_hashes: Mapping[str, str],
    expected_tag: str = "",
) -> FrozenV83Audit:
    """Verify an exact clean checkout without modifying it."""

    if not expected_file_hashes:
        raise FrozenV83AuditError("independently reviewed file hashes are required")
    root = Path(repo)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise FrozenV83AuditError("V83 checkout must be an absolute non-symlink directory")
    root = root.resolve(strict=True)
    if not _COMMIT_RE.fullmatch(expected_commit):
        raise FrozenV83AuditError("expected V83 commit is malformed")
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    if head != expected_commit:
        raise FrozenV83AuditError(
            f"V83 checkout HEAD differs: expected {expected_commit}, got {head}"
        )
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise FrozenV83AuditError("V83 checkout is not clean")
    tags = tuple(filter(None, _git(root, "tag", "--points-at", "HEAD").splitlines()))
    if expected_tag and expected_tag not in tags:
        raise FrozenV83AuditError(f"V83 checkout is missing tag {expected_tag!r}")
    actual: list[tuple[str, str]] = []
    for relpath, expected_hash in sorted(expected_file_hashes.items()):
        if not _SHA256_RE.fullmatch(expected_hash):
            raise FrozenV83AuditError(f"malformed expected hash for {relpath}")
        path = _repo_file(root, relpath)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise FrozenV83AuditError(f"V83 file hash differs for {relpath}")
        actual.append((relpath, digest))
    return FrozenV83Audit(root, head, expected_tag, tuple(actual))


def _repo_file(root: Path, relpath: str) -> Path:
    candidate = root / relpath
    relative_parts = Path(relpath).parts
    chain = (candidate, *candidate.parents[: len(relative_parts) - 1])
    if any(path.is_symlink() for path in chain) or not candidate.is_file():
        raise FrozenV83AuditError(f"V83 file is missing or symlinked: {relpath}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FrozenV83AuditError(f"V83 file escapes checkout: {relpath}") from exc
    return resolved


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FrozenV83AuditError(f"cannot audit V83 git checkout: {exc}") from exc
    return completed.stdout.strip()


__all__ = [
    "FrozenV83Audit",
    "FrozenV83AuditError",
    "V83_POST_TABLE_MIGRATION_REQUIREMENTS",
    "audit_frozen_v83_checkout",
]
