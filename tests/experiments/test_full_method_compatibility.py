from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from coevolve_experiments.full_method_compatibility import (
    BASE_SCENE_RELPATH,
    FULL_METHOD_ARTIFACT_FIELDS,
    INITIAL_ONLINE_MEMORY_FILENAMES,
    INITIAL_ONLINE_MEMORY_ROOT_RELPATH,
    PROMPT_RELPATH,
    ROUTE_SPEC_RELPATH,
    SKILL_API_RELPATHS,
    STATIC_MEMORY_ROOT_RELPATH,
    VERIFIER_CONFIG_RELPATH,
    VERIFIER_SOURCE_RELPATHS,
    FullMethodCompatibilityError,
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
    full_method_compatibility_pin_from_mapping,
    recompute_full_method_compatibility,
)
from coevolve_experiments.serialization import sha256_json


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _runner_fixture(root: Path) -> FullMethodCompatibilityPin:
    behavior: dict[str, str] = {}
    for relpath in V83_FULL_BEHAVIOR_RELPATHS:
        raw = f"frozen-v83:{relpath}\n".encode("utf-8")
        _write(root / relpath, raw)
        behavior[relpath] = _sha256_bytes(raw)

    verifier_config = {"terminal_stable_s": 3.0, "profile": "task"}
    _write(
        root / VERIFIER_CONFIG_RELPATH,
        (json.dumps(verifier_config, sort_keys=True) + "\n").encode("utf-8"),
    )
    initial_hashes: dict[str, str] = {}
    for name in INITIAL_ONLINE_MEMORY_FILENAMES:
        raw = b"" if name.endswith(".jsonl") else b"{}\n"
        _write(root / INITIAL_ONLINE_MEMORY_ROOT_RELPATH / name, raw)
        initial_hashes[name] = _sha256_bytes(raw)

    static_prefix = STATIC_MEMORY_ROOT_RELPATH + "/"
    static_hashes = {
        path.removeprefix(static_prefix): digest
        for path, digest in behavior.items()
        if path.startswith(static_prefix)
    }
    artifact_hashes = {
        "initial_online_memory_sha256": sha256_json(initial_hashes),
        "prompt_sha256": behavior[PROMPT_RELPATH],
        "route_spec_sha256": behavior[ROUTE_SPEC_RELPATH],
        "scene_sha256": behavior[BASE_SCENE_RELPATH],
        "skill_api_sha256": sha256_json(
            {path: behavior[path] for path in SKILL_API_RELPATHS}
        ),
        "static_memory_sha256": sha256_json(static_hashes),
        "verifier_config_sha256": sha256_json(verifier_config),
        "verifier_source_sha256": sha256_json(
            {path: behavior[path] for path in VERIFIER_SOURCE_RELPATHS}
        ),
    }
    assert set(artifact_hashes) == set(FULL_METHOD_ARTIFACT_FIELDS)
    return FullMethodCompatibilityPin.from_artifact_hashes(
        artifact_hashes,
        behavior_artifacts=behavior,
    )


def test_identity_is_recomputed_from_complete_runner_bytes(tmp_path: Path) -> None:
    declared = _runner_fixture(tmp_path)

    assert recompute_full_method_compatibility(tmp_path, declared) == declared
    assert len(declared.behavior_artifacts) == 81


def test_new_adapter_file_does_not_change_frozen_identity(tmp_path: Path) -> None:
    declared = _runner_fixture(tmp_path)
    _write(tmp_path / "src/new_table2_adapter.py", b"new adapter\n")

    assert recompute_full_method_compatibility(tmp_path, declared) == declared


def test_changed_or_missing_frozen_behavior_source_fails_closed(
    tmp_path: Path,
) -> None:
    declared = _runner_fixture(tmp_path)
    frozen = tmp_path / "src/praxis_coop/cap_policy/model_agents.py"
    frozen.write_bytes(frozen.read_bytes() + b"changed\n")

    assert recompute_full_method_compatibility(tmp_path, declared) != declared

    frozen.unlink()
    with pytest.raises(FullMethodCompatibilityError, match="regular non-symlink"):
        recompute_full_method_compatibility(tmp_path, declared)


@pytest.mark.parametrize("mutation", ("missing", "unknown", "forged"))
def test_json_boundary_rejects_incomplete_or_forged_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = asdict(_runner_fixture(tmp_path))
    payload["behavior_artifacts"] = list(payload["behavior_artifacts"])
    if mutation == "missing":
        payload.pop("prompt_sha256")
    elif mutation == "unknown":
        payload["opaque_claim"] = "accepted"
    else:
        payload["fingerprint"] = "0" * 64

    with pytest.raises(FullMethodCompatibilityError):
        full_method_compatibility_pin_from_mapping(payload)
