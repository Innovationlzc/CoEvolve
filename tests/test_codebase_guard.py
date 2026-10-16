import json
from pathlib import Path
import subprocess

import pytest

from praxis_coop.cap_policy.codebase_guard import (
    CODEBASE_DRIFT_REPORT_NAME,
    CODEBASE_MANIFEST_NAME,
    DEFAULT_EXPERIMENT_SCOPES,
    CodebaseFreezeError,
    CodebaseGuard,
    capture_codebase_snapshot,
)


def test_default_scope_covers_every_physical_backend_entrypoint():
    assert {
        "scripts/run_dual_official_sonic_motor_preflight.py",
        "scripts/run_sonic_coordinated_skill.py",
        "scripts/run_sonic_static_skill.py",
        "scripts/validate_static_sonic_skill.py",
    }.issubset(DEFAULT_EXPERIMENT_SCOPES)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "praxis_coop").mkdir(parents=True)
    (repo / "src" / "praxis_coop" / "module.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_codebase_guard_records_clean_snapshot(tmp_path):
    repo = _repo(tmp_path)
    output_dir = tmp_path / "run"

    guard = CodebaseGuard.initialize(
        repo_root=repo,
        output_dir=output_dir,
        require_clean=True,
        scopes=("src/praxis_coop",),
    )

    manifest = json.loads((output_dir / CODEBASE_MANIFEST_NAME).read_text())
    assert manifest["git_dirty_entries"] == []
    assert manifest["source_sha256"] == guard.snapshot["source_sha256"]
    guard.verify_unchanged(checkpoint="before_attempt_01")


def test_codebase_guard_rejects_dirty_formal_run(tmp_path):
    repo = _repo(tmp_path)
    (repo / "src" / "praxis_coop" / "module.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    with pytest.raises(CodebaseFreezeError, match="requires a clean"):
        CodebaseGuard.initialize(
            repo_root=repo,
            output_dir=tmp_path / "run",
            require_clean=True,
            scopes=("src/praxis_coop",),
        )


def test_codebase_guard_detects_runtime_drift(tmp_path):
    repo = _repo(tmp_path)
    output_dir = tmp_path / "run"
    guard = CodebaseGuard.initialize(
        repo_root=repo,
        output_dir=output_dir,
        require_clean=True,
        scopes=("src/praxis_coop",),
    )
    (repo / "src" / "praxis_coop" / "module.py").write_text(
        "VALUE = 3\n", encoding="utf-8"
    )

    with pytest.raises(CodebaseFreezeError, match="codebase changed"):
        guard.verify_unchanged(checkpoint="before_attempt_02")

    drift = json.loads((output_dir / CODEBASE_DRIFT_REPORT_NAME).read_text())
    assert drift["checkpoint"] == "before_attempt_02"
    assert drift["expected"]["source_sha256"] != drift["actual"]["source_sha256"]


def test_codebase_guard_rejects_legacy_attempt_directory(tmp_path):
    repo = _repo(tmp_path)
    output_dir = tmp_path / "run"
    (output_dir / "attempt_01").mkdir(parents=True)

    with pytest.raises(CodebaseFreezeError, match="fresh output directory"):
        CodebaseGuard.initialize(
            repo_root=repo,
            output_dir=output_dir,
            require_clean=True,
            scopes=("src/praxis_coop",),
        )


def test_snapshot_ignores_python_cache_files(tmp_path):
    repo = _repo(tmp_path)
    before = capture_codebase_snapshot(repo, scopes=("src/praxis_coop",))
    cache = repo / "src" / "praxis_coop" / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"not-real-bytecode")
    after = capture_codebase_snapshot(repo, scopes=("src/praxis_coop",))

    assert before["source_sha256"] == after["source_sha256"]
