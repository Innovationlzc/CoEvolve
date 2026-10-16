from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import coevolve_experiments.scenes.evidence_producer as producer
from coevolve_experiments.cli import main
from coevolve_experiments.contracts import (
    MethodDecisionPromptPin,
    MethodVariant,
    ProtocolLock,
    Table1SourceCohort,
)
from coevolve_experiments.lockfile import protocol_lock_document
from coevolve_experiments.scenes import load_scene, validate_evidence_receipt
from coevolve_experiments.contracts import SceneId
from coevolve_experiments.provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE,
    SECONDARY_GEMINI_PROVIDER_ROUTE,
)
from coevolve_experiments.full_method_compatibility import (
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _lock() -> ProtocolLock:
    prompt = _digest("prompt")
    skill_api = _digest("skill-api")
    full_method = FullMethodCompatibilityPin.from_artifact_hashes(
        {
            "initial_online_memory_sha256": _digest("initial-memory"),
            "prompt_sha256": prompt,
            "route_spec_sha256": _digest("route"),
            "scene_sha256": _digest("scene"),
            "skill_api_sha256": skill_api,
            "static_memory_sha256": _digest("static-memory"),
            "verifier_config_sha256": _digest("verifier-config"),
            "verifier_source_sha256": _digest("verifier-source"),
        },
        behavior_artifacts={
            path: _digest(f"fixture:{path}")
            for path in V83_FULL_BEHAVIOR_RELPATHS
        },
    )
    return ProtocolLock(
        protocol_id="post-table1-evidence-test",
        invariant_fingerprint=_digest("invariant"),
        terminal_fingerprint=_digest("terminal"),
        base_method_commit="a" * 40,
        table1_gpt_source=Table1SourceCohort(
            backbone="gpt",
            protocol_id="table1-gpt",
            method_commit="b" * 40,
            seeds=(200, 201, 202, 203, 204),
            condition_fingerprint=_digest("gpt-condition"),
            result_manifest_hash=_digest("gpt-results"),
            provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
            provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
            shared_prompt_contract_fingerprint=prompt,
            decision_prompt_fingerprint=prompt,
            full_method_compatibility_fingerprint=full_method.fingerprint,
        ),
        table1_gemini_source=Table1SourceCohort(
            backbone="gemini",
            protocol_id="table1-gemini",
            method_commit="c" * 40,
            seeds=(200, 201, 202, 203, 204),
            condition_fingerprint=_digest("gemini-condition"),
            result_manifest_hash=_digest("gemini-results"),
            provider_route_id=SECONDARY_GEMINI_PROVIDER_ROUTE.route_id,
            provider_route_fingerprint=SECONDARY_GEMINI_PROVIDER_ROUTE.fingerprint,
            shared_prompt_contract_fingerprint=prompt,
            decision_prompt_fingerprint=prompt,
            full_method_compatibility_fingerprint=full_method.fingerprint,
        ),
        gpt_strategy_bank_hash=_digest("gpt-bank"),
        gemini_strategy_bank_hash=_digest("gemini-bank"),
        target_provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        target_provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        observation_contract_fingerprint=_digest("observation"),
        task_instruction_fingerprint=_digest("instruction"),
        shared_prompt_contract_fingerprint=prompt,
        method_decision_prompt_pins=tuple(
            MethodDecisionPromptPin(variant, prompt) for variant in MethodVariant
        ),
        skill_api_fingerprint=skill_api,
        full_method_compatibility=full_method,
    )


class _FakeObj:
    mjOBJ_BODY = "body"
    mjOBJ_JOINT = "joint"
    mjOBJ_ACTUATOR = "actuator"
    mjOBJ_GEOM = "geom"


class _FakeModel:
    nbody = 4
    njnt = 2
    nu = 2
    ngeom = 3

    @classmethod
    def from_xml_path(cls, path: str):
        assert Path(path).read_text(encoding="utf-8") == "<mujoco/>\n"
        return cls()

    @classmethod
    def from_binary_path(cls, path: str):
        assert Path(path).read_bytes() == b"real fake compiled MJB\n"
        return cls()


class _FakeMujoco:
    __version__ = "3.test"
    mjtObj = _FakeObj
    MjModel = _FakeModel

    _names = {
        "body": ("world", "front_pelvis", "back_pelvis", "long_box"),
        "joint": ("front_root", "back_root"),
        "actuator": ("front_drive", "back_drive"),
        "geom": ("front_support", "back_support", "payload_geom"),
    }

    @staticmethod
    def mj_versionString() -> str:
        return "3.test-core"

    @staticmethod
    def mj_saveModel(model: _FakeModel, path: str, buffer) -> None:
        assert buffer is None
        Path(path).write_bytes(b"real fake compiled MJB\n")

    @classmethod
    def mj_id2name(cls, model, object_type, object_id: int):
        return cls._names[object_type][object_id]

    @classmethod
    def mj_name2id(cls, model, object_type, name: str) -> int:
        try:
            return cls._names[object_type].index(name)
        except ValueError:
            return -1


def _prepare_repo(repo: Path) -> None:
    scene = load_scene(SceneId.BASE)
    source = Path(__file__).resolve().parents[2] / scene.asset.source_path
    destination = repo / scene.asset.source_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())


def _prepare_gear_root(root: Path) -> None:
    source = root / producer.DEFAULT_G1_SOURCE_XML
    source.parent.mkdir(parents=True)
    source.write_text(
        "<mujoco><compiler meshdir=\"meshes\"/><asset>"
        "<mesh name=\"one\" file=\"one.stl\"/>"
        "<mesh name=\"two\" file=\"nested/two.stl\"/>"
        "</asset></mujoco>\n",
        encoding="utf-8",
    )
    (source.parent / "meshes/nested").mkdir(parents=True)
    (source.parent / "meshes/one.stl").write_bytes(b"mesh one\n")
    (source.parent / "meshes/nested/two.stl").write_bytes(b"mesh two\n")


def _fake_generate(**kwargs):
    scene_path = Path(kwargs["gear_root"]) / kwargs["scene_rel_path"]
    front_path = Path(kwargs["gear_root"]) / kwargs["front_g1_rel_path"]
    back_path = Path(kwargs["gear_root"]) / kwargs["back_g1_rel_path"]
    for path in (scene_path, front_path, back_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<mujoco/>\n", encoding="utf-8")
    return SimpleNamespace(
        scene_path=scene_path,
        robot_xmls=(
            SimpleNamespace(robot_id="front", output_path=front_path),
            SimpleNamespace(robot_id="back", output_path=back_path),
        ),
    )


def test_integrity_producer_emits_a_host_validated_v2_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    gear = tmp_path / "gear"
    repo.mkdir()
    gear.mkdir()
    _prepare_repo(repo)
    _prepare_gear_root(gear)
    scene = load_scene(SceneId.BASE)
    lock = _lock()
    monkeypatch.setattr(
        producer,
        "_verify_frozen_generator_sources",
        lambda root, commit: [
            {"relative_path": "src/frozen_generator.py", "sha256": "1" * 64}
        ],
    )
    monkeypatch.setattr(
        producer,
        "_producer_source_manifest",
        lambda root: {"relative_path": "src/producer.py", "sha256": "2" * 64},
    )
    monkeypatch.setattr(producer, "_load_mujoco", lambda: _FakeMujoco)
    monkeypatch.setattr(producer, "generate_dual_manual_layout_scene", _fake_generate)

    result = producer.produce_simulator_integrity_evidence(
        scene=scene,
        protocol_lock=lock,
        repo_root=repo,
        output_dir="evidence/base/simulator_integrity",
        gear_root=gear,
        invocation=("python", "-m", "coevolve_experiments", "test-invocation"),
    )

    assert result.output_dir.is_dir()
    assert {path.name for path in result.output_dir.iterdir()} == {
        "command.json",
        "config.json",
        "compiled_model.mjb",
        "model_inventory.json",
        "receipt.json",
    }
    assert (result.output_dir / "compiled_model.mjb").read_bytes() == (
        b"real fake compiled MJB\n"
    )
    inventory = json.loads(
        (result.output_dir / "model_inventory.json").read_text(encoding="utf-8")
    )
    assert len(inventory["bodies"]) == 4
    assert len(inventory["joints"]) == 2
    assert len(inventory["actuators"]) == 2
    assert len(inventory["geoms"]) == 3
    assert {item["role"] for item in inventory["role_bindings"]} == {
        "front_robot",
        "back_robot",
        "payload",
    }
    config = json.loads(
        (result.output_dir / "config.json").read_text(encoding="utf-8")
    )
    assert len(config["sonic_source"]["dependencies"]) == 3
    assert config["generation"]["payload_initial_state"] == "deferred"
    assert config["compiled_model"]["sha256"] == result.compiled_model_sha256
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert "passed" not in receipt
    assert receipt["producer_commit"] == lock.base_method_commit
    assert receipt["measurements"]["body_count"] == 4
    validation = validate_evidence_receipt(
        repo,
        result.receipt_pin,
        "simulator_integrity",
        scene,
        lock,
    )
    assert not validation.errors
    assert validation.binding == result.binding

    with pytest.raises(producer.EvidenceProductionError, match="already exists"):
        producer.produce_simulator_integrity_evidence(
            scene=scene,
            protocol_lock=lock,
            repo_root=repo,
            output_dir="evidence/base/simulator_integrity",
            gear_root=gear,
        )


@pytest.mark.parametrize("check", ("initial_collision", "scripted_solvability"))
def test_unimplemented_physical_checks_fail_before_writing(
    tmp_path: Path,
    check: str,
) -> None:
    output = tmp_path / "must-not-exist"
    with pytest.raises(producer.EvidenceProductionBlocked):
        producer.produce_scene_evidence(
            check,
            scene=load_scene(SceneId.BASE),
            protocol_lock=_lock(),
            repo_root=tmp_path,
            output_dir=output,
            gear_root=tmp_path / "missing-gear",
        )
    assert not output.exists()


def test_blocked_evidence_cli_returns_structured_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(protocol_lock_document(_lock())), encoding="utf-8")

    code = main(
        [
            "produce-scene-evidence",
            "--check",
            "scripted_solvability",
            "--scene",
            "base",
            "--lock",
            str(lock_path),
            "--repo-root",
            str(tmp_path),
            "--gear-root",
            str(tmp_path / "missing-gear"),
            "--output-dir",
            "evidence/blocked",
        ]
    )

    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False
    assert "no controller is currently allowlisted" in error["error"]
    assert not (tmp_path / "evidence/blocked").exists()
