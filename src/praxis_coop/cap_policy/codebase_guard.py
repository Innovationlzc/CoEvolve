"""Freeze the executable codebase used by a formal evolution run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


CODEBASE_MANIFEST_NAME = "codebase_manifest.json"
CODEBASE_DRIFT_REPORT_NAME = "codebase_drift.json"
DEFAULT_EXPERIMENT_SCOPES = (
    "pyproject.toml",
    "src/praxis_coop",
    "scripts/run_cap_box_carry_evolution.py",
    "scripts/run_dual_sonic_manual_layout_sim.py",
    "scripts/run_dual_official_sonic_motor_preflight.py",
    "scripts/run_official_sonic_deploy_zmq_manager.sh",
    "scripts/run_sonic_coordinated_skill.py",
    "scripts/run_sonic_static_skill.py",
    "scripts/validate_static_sonic_skill.py",
    "configs/routes",
    "assets/manual_scenes",
    "results/spec_memory/static",
)
_IGNORED_PARTS = frozenset({".git", ".pytest_cache", "__pycache__"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


class CodebaseFreezeError(RuntimeError):
    """Raised when a formal run cannot preserve one executable codebase."""


@dataclass
class CodebaseGuard:
    """Record one source snapshot and reject changes during the run."""

    repo_root: Path
    output_dir: Path
    snapshot: dict[str, object]
    scopes: tuple[str, ...]

    @classmethod
    def initialize(
        cls,
        *,
        repo_root: Path,
        output_dir: Path,
        require_clean: bool,
        scopes: Iterable[str] = DEFAULT_EXPERIMENT_SCOPES,
    ) -> "CodebaseGuard":
        root = Path(repo_root).resolve()
        destination = Path(output_dir)
        normalized_scopes = tuple(dict.fromkeys(str(scope) for scope in scopes))
        snapshot = capture_codebase_snapshot(root, scopes=normalized_scopes)
        manifest_path = destination / CODEBASE_MANIFEST_NAME
        existing = _read_json_object(manifest_path)

        if existing:
            _require_matching_snapshot(
                expected=existing,
                actual=snapshot,
                context=f"existing experiment manifest {manifest_path}",
            )
        elif _contains_attempt_artifacts(destination):
            raise CodebaseFreezeError(
                f"{destination} already contains attempt artifacts but has no "
                f"{CODEBASE_MANIFEST_NAME}; use a fresh output directory"
            )

        dirty_entries = tuple(str(value) for value in snapshot["git_dirty_entries"])
        if require_clean and dirty_entries:
            preview = "\n".join(dirty_entries[:12])
            suffix = "\n..." if len(dirty_entries) > 12 else ""
            raise CodebaseFreezeError(
                "formal execution requires a clean experiment codebase; commit or "
                f"stash these scoped changes first:\n{preview}{suffix}"
            )

        destination.mkdir(parents=True, exist_ok=True)
        if not existing:
            _write_json(manifest_path, snapshot)
        return cls(
            repo_root=root,
            output_dir=destination,
            snapshot=snapshot,
            scopes=normalized_scopes,
        )

    def verify_unchanged(self, *, checkpoint: str) -> None:
        """Fail before dispatch when executable source changed after startup."""

        current = capture_codebase_snapshot(self.repo_root, scopes=self.scopes)
        try:
            _require_matching_snapshot(
                expected=self.snapshot,
                actual=current,
                context=f"runtime checkpoint {checkpoint}",
            )
        except CodebaseFreezeError:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                self.output_dir / CODEBASE_DRIFT_REPORT_NAME,
                {
                    "kind": "experiment_codebase_drift",
                    "checkpoint": checkpoint,
                    "expected": self.snapshot,
                    "actual": current,
                },
            )
            raise


def capture_codebase_snapshot(
    repo_root: Path,
    *,
    scopes: Iterable[str] = DEFAULT_EXPERIMENT_SCOPES,
) -> dict[str, object]:
    """Return a deterministic snapshot of files that affect physical execution."""

    root = Path(repo_root).resolve()
    normalized_scopes = tuple(dict.fromkeys(str(scope) for scope in scopes))
    files = _scoped_file_hashes(root, normalized_scopes)
    fingerprint_payload = json.dumps(
        files,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "kind": "experiment_codebase_manifest",
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_output(root, "rev-parse", "HEAD"),
        "git_branch": _git_output(root, "branch", "--show-current"),
        "git_dirty_entries": _git_status(root, normalized_scopes),
        "scope": list(normalized_scopes),
        "file_count": len(files),
        "files": files,
        "source_sha256": hashlib.sha256(fingerprint_payload).hexdigest(),
    }


def _scoped_file_hashes(
    repo_root: Path,
    scopes: tuple[str, ...],
) -> dict[str, str]:
    paths: set[Path] = set()
    for scope in scopes:
        candidate = repo_root / scope
        if candidate.is_file() or candidate.is_symlink():
            paths.add(candidate)
        elif candidate.is_dir():
            paths.update(path for path in candidate.rglob("*") if path.is_file())

    hashes: dict[str, str] = {}
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(repo_root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix in _IGNORED_SUFFIXES:
            continue
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _git_status(repo_root: Path, scopes: tuple[str, ...]) -> list[str]:
    command = [
        "git",
        "-C",
        str(repo_root),
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *scopes,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return [f"git_status_unavailable: {completed.stderr.strip()}"]
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _git_output(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _require_matching_snapshot(
    *,
    expected: dict[str, object],
    actual: dict[str, object],
    context: str,
) -> None:
    expected_hash = str(expected.get("source_sha256", ""))
    actual_hash = str(actual.get("source_sha256", ""))
    expected_head = str(expected.get("git_head", ""))
    actual_head = str(actual.get("git_head", ""))
    if expected_hash == actual_hash and expected_head == actual_head:
        return
    raise CodebaseFreezeError(
        f"experiment codebase changed at {context}: "
        f"source_sha256 {expected_hash or '<missing>'} -> "
        f"{actual_hash or '<missing>'}, git_head "
        f"{expected_head or '<missing>'} -> {actual_head or '<missing>'}; "
        "stop this run and restart from the initial state in a new output directory"
    )


def _contains_attempt_artifacts(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    return any(output_dir.glob("attempt_*")) or (
        output_dir / "raw_rollout_evidence.jsonl"
    ).exists()


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
