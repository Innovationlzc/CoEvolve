from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil

import pytest

from coevolve_experiments.cli import main
from coevolve_experiments.contracts import SceneId
from coevolve_experiments.scenes import (
    MATERIALIZATION_RECEIPT_KIND,
    MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
    SceneMaterializationError,
    compute_asset_composite_hash,
    compute_scene_hash,
    inspect_materialized_transform,
    load_scene,
    materialize_scene,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_RELPATH = Path("assets/manual_scenes/two_g1_box_push_layout.usda")


@pytest.mark.parametrize(
    "scene_id",
    (SceneId.MIRRORED, SceneId.NARROW, SceneId.ALTERED_TOPOLOGY),
)
def test_registered_heldout_asset_and_receipt_are_hash_bound(
    scene_id: SceneId,
) -> None:
    """Keep the checked-in candidate bytes synchronized with the registry.

    This is deliberately not simulator evidence.  It proves only that the
    deterministic materialization already reviewed for the held-out scene is
    present byte-for-byte and that its receipt binds the current SceneSpec.
    """

    scene = load_scene(scene_id)
    assert scene.asset.execution_path is not None
    assert scene.asset.execution_sha256 is not None
    assert scene.asset.execution_composite_sha256 is not None
    execution = _PROJECT_ROOT / scene.asset.execution_path
    receipt_path = execution.with_name(execution.name + ".receipt.json")

    assert execution.is_file() and not execution.is_symlink()
    assert receipt_path.is_file() and not receipt_path.is_symlink()
    assert sha256(execution.read_bytes()).hexdigest() == scene.asset.execution_sha256
    assert compute_asset_composite_hash(scene.asset) == (
        scene.asset.execution_composite_sha256
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["kind"] == MATERIALIZATION_RECEIPT_KIND
    assert receipt["schema_version"] == MATERIALIZATION_RECEIPT_SCHEMA_VERSION
    assert receipt["scene_id"] == scene_id.value
    assert receipt["scene_spec"] == {
        "input_sha256": scene.scene_hash,
        "materialized_sha256": scene.scene_hash,
    }
    assert receipt["execution"] == {
        "relative_path": scene.asset.execution_path,
        "sha256": scene.asset.execution_sha256,
        "composite_sha256": scene.asset.execution_composite_sha256,
        "size_bytes": execution.stat().st_size,
    }


@pytest.fixture
def isolated_repo(tmp_path: Path) -> Path:
    source = _PROJECT_ROOT / _SOURCE_RELPATH
    copy = tmp_path / _SOURCE_RELPATH
    copy.parent.mkdir(parents=True)
    shutil.copyfile(source, copy)
    return tmp_path


@pytest.mark.parametrize("scene_id", tuple(SceneId))
def test_all_four_scenes_materialize_with_hash_bound_receipts(
    isolated_repo: Path,
    scene_id: SceneId,
) -> None:
    scene = load_scene(scene_id)
    output = Path("candidate_scenes") / f"{scene_id.value}.usda"

    result = materialize_scene(scene, isolated_repo, output)

    execution = result.execution_path.read_bytes()
    receipt_bytes = result.receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert result.execution_path == isolated_repo / output
    assert result.receipt_path == isolated_repo / f"{output}.receipt.json"
    assert result.execution_sha256 == sha256(execution).hexdigest()
    assert result.receipt_sha256 == sha256(receipt_bytes).hexdigest()
    assert compute_asset_composite_hash(result.pinned_asset) == (
        result.execution_composite_sha256
    )
    assert compute_scene_hash(replace(scene, asset=result.pinned_asset)) == (
        result.materialized_scene_spec_sha256
    )
    assert receipt == {
        "kind": MATERIALIZATION_RECEIPT_KIND,
        "schema_version": MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        "materializer": {
            "id": "coevolve.scenes.deterministic-usda",
            "version": "1.0.0",
        },
        "scene_id": scene_id.value,
        "scene_spec": {
            "input_sha256": scene.scene_hash,
            "materialized_sha256": result.materialized_scene_spec_sha256,
        },
        "source": {
            "relative_path": _SOURCE_RELPATH.as_posix(),
            "sha256": scene.asset.source_sha256,
            "size_bytes": (isolated_repo / _SOURCE_RELPATH).stat().st_size,
        },
        "realization": {
            "kind": scene.asset.realization,
            "parameters_sha256": receipt["realization"]["parameters_sha256"],
        },
        "execution": {
            "relative_path": output.as_posix(),
            "sha256": result.execution_sha256,
            "composite_sha256": result.execution_composite_sha256,
            "size_bytes": len(execution),
        },
        "receipt_relative_path": f"{output.as_posix()}.receipt.json",
    }
    assert len(receipt["realization"]["parameters_sha256"]) == 64
    assert result.execution_written and result.receipt_written
    assert not tuple(result.execution_path.parent.glob(".*.tmp"))

    if scene_id is SceneId.BASE:
        assert execution == (isolated_repo / _SOURCE_RELPATH).read_bytes()


def test_materialize_scene_cli_emits_candidate_only_contract(
    isolated_repo: Path,
    capsys,
) -> None:
    output = Path("candidate_scenes/mirrored.usda")

    assert main(
        [
            "materialize-scene",
            "--scene",
            "mirrored",
            "--repo-root",
            str(isolated_repo),
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "coevolve_candidate_scene_materialization"
    assert payload["ok"] is True
    assert payload["execution_ready"] is False
    assert payload["paper_evidence"] is False
    assert payload["result"]["execution_path"] == str(isolated_repo / output)
    assert payload["result"]["pinned_asset"]["execution_path"] == output.as_posix()
    assert len(payload["result"]["receipt_sha256"]) == 64


def test_mirror_realization_reflects_robots_payload_target_and_walls(
    isolated_repo: Path,
) -> None:
    base = load_scene(SceneId.BASE)
    mirrored = load_scene(SceneId.MIRRORED)
    result = materialize_scene(mirrored, isolated_repo, "generated/mirrored.usda")
    source = (isolated_repo / _SOURCE_RELPATH).read_bytes()
    output = result.execution_path.read_bytes()

    robot_prims = {"front": "/World/g1", "back": "/World/g1_01"}
    for robot in mirrored.initial_state.robots:
        transform = inspect_materialized_transform(output, robot_prims[robot.robot_id])
        assert transform.translation == pytest.approx(robot.pose.position.as_tuple())
        assert _angle_difference(transform.yaw_rad, robot.pose.yaw_rad) == pytest.approx(0.0)
    payload = inspect_materialized_transform(output, "/World/long_box")
    target = inspect_materialized_transform(output, "/World/target")
    assert payload.translation == pytest.approx(
        mirrored.initial_state.payload.position.as_tuple()
    )
    assert target.translation == pytest.approx(mirrored.target.center.as_tuple())

    for wall_name in ("wall_01", "wall_03", "wall_04", "wall_05", "wall_06", "wall_07", "wall_08"):
        before = inspect_materialized_transform(source, f"/World/{wall_name}")
        after = inspect_materialized_transform(output, f"/World/{wall_name}")
        assert after.translation == pytest.approx(
            (-before.translation[0], before.translation[1], before.translation[2])
        )
        assert after.scale == pytest.approx(before.scale)
        assert _angle_difference(
            after.yaw_rad,
            math.pi - before.yaw_rad,
        ) == pytest.approx(0.0)


def test_narrow_realization_applies_every_explicit_wall_transform(
    isolated_repo: Path,
) -> None:
    scene = load_scene(SceneId.NARROW)
    result = materialize_scene(scene, isolated_repo, "generated/narrow.usda")
    output = result.execution_path.read_bytes()

    for parameter in scene.asset.realization_parameters:
        assert parameter.operation == "set_box_transform_xyz_scale_xyz_yaw"
        transform = inspect_materialized_transform(output, parameter.targets[0])
        assert transform.translation == pytest.approx(parameter.values[:3])
        assert transform.scale == pytest.approx(parameter.values[3:6])
        assert _angle_difference(transform.yaw_rad, parameter.values[6]) == pytest.approx(0.0)


def test_altered_topology_removes_authored_walls_and_adds_collision_boxes(
    isolated_repo: Path,
) -> None:
    scene = load_scene(SceneId.ALTERED_TOPOLOGY)
    result = materialize_scene(
        scene,
        isolated_repo,
        "generated/altered_topology.usda",
    )
    output_bytes = result.execution_path.read_bytes()
    output = output_bytes.decode("utf-8")

    for name in ("wall_01", "wall_03", "wall_04", "wall_05", "wall_06", "wall_07", "wall_08"):
        with pytest.raises(SceneMaterializationError, match="missing required prim"):
            inspect_materialized_transform(output, f"/World/{name}")
    for parameter in scene.asset.realization_parameters[1:]:
        name = parameter.targets[0].removeprefix("/World/")
        assert f'def Cube "{name}"' in output
        transform = inspect_materialized_transform(output, parameter.targets[0])
        assert transform.translation == pytest.approx(parameter.values[:3])
        assert transform.scale == pytest.approx(parameter.values[3:6])
        assert _angle_difference(transform.yaw_rad, parameter.values[6]) == pytest.approx(0.0)
    target = inspect_materialized_transform(output_bytes, "/World/target")
    assert target.translation == pytest.approx(scene.target.center.as_tuple())


def test_materialization_is_byte_deterministic_and_idempotent(
    isolated_repo: Path,
) -> None:
    scene = load_scene(SceneId.MIRRORED)
    first = materialize_scene(scene, isolated_repo, "generated/mirrored.usda")
    first_execution = first.execution_path.read_bytes()
    first_receipt = first.receipt_path.read_bytes()
    execution_stat = first.execution_path.stat()
    receipt_stat = first.receipt_path.stat()

    second = materialize_scene(
        scene,
        isolated_repo,
        isolated_repo / "generated/mirrored.usda",
    )

    assert not second.execution_written
    assert not second.receipt_written
    assert second == replace(
        first,
        execution_written=False,
        receipt_written=False,
    )
    assert second.execution_path.read_bytes() == first_execution
    assert second.receipt_path.read_bytes() == first_receipt
    assert second.execution_path.stat().st_mtime_ns == execution_stat.st_mtime_ns
    assert second.receipt_path.stat().st_mtime_ns == receipt_stat.st_mtime_ns


@pytest.mark.parametrize("tampered_file", ("execution", "receipt"))
def test_existing_tampered_materialization_fails_closed(
    isolated_repo: Path,
    tampered_file: str,
) -> None:
    scene = load_scene(SceneId.NARROW)
    result = materialize_scene(scene, isolated_repo, "generated/narrow.usda")
    path = result.execution_path if tampered_file == "execution" else result.receipt_path
    path.write_bytes(path.read_bytes() + b"# tampered\n")

    with pytest.raises(SceneMaterializationError, match="refusing overwrite"):
        materialize_scene(scene, isolated_repo, "generated/narrow.usda")


def test_source_hash_tampering_fails_before_creating_output(
    isolated_repo: Path,
) -> None:
    source = isolated_repo / _SOURCE_RELPATH
    source.write_bytes(source.read_bytes() + b"# tampered\n")

    with pytest.raises(SceneMaterializationError, match="source scene asset SHA-256 mismatch"):
        materialize_scene(
            load_scene(SceneId.BASE),
            isolated_repo,
            "generated/base.usda",
        )
    assert not (isolated_repo / "generated").exists()


def test_in_memory_scene_spec_tampering_fails_closed(isolated_repo: Path) -> None:
    scene = load_scene(SceneId.BASE)
    changed_target = replace(
        scene.target,
        radius_m=scene.target.radius_m + 0.01,
    )

    with pytest.raises(SceneMaterializationError, match="declared scene_hash"):
        materialize_scene(
            replace(scene, target=changed_target),
            isolated_repo,
            "generated/base.usda",
        )


def test_materializer_rejects_source_and_output_symlinks(
    isolated_repo: Path,
) -> None:
    scene = load_scene(SceneId.BASE)
    source = isolated_repo / _SOURCE_RELPATH
    source_copy = isolated_repo / "source-copy.usda"
    shutil.copyfile(source, source_copy)
    source.unlink()
    source.symlink_to(source_copy)
    with pytest.raises(SceneMaterializationError, match="source path contains a symlink"):
        materialize_scene(scene, isolated_repo, "generated/base.usda")

    source.unlink()
    shutil.copyfile(source_copy, source)
    real_output_dir = isolated_repo / "real-output"
    real_output_dir.mkdir()
    linked_output_dir = isolated_repo / "linked-output"
    linked_output_dir.symlink_to(real_output_dir, target_is_directory=True)
    with pytest.raises(SceneMaterializationError, match="output path contains a symlink"):
        materialize_scene(scene, isolated_repo, "linked-output/base.usda")


def test_materializer_rejects_path_escape_and_source_overwrite(
    isolated_repo: Path,
) -> None:
    scene = load_scene(SceneId.BASE)
    with pytest.raises(SceneMaterializationError, match="dot segments"):
        materialize_scene(scene, isolated_repo, "../escaped.usda")
    with pytest.raises(SceneMaterializationError, match="escapes repository root"):
        materialize_scene(
            scene,
            isolated_repo,
            isolated_repo.parent / "escaped.usda",
        )
    with pytest.raises(SceneMaterializationError, match="must not overwrite"):
        materialize_scene(scene, isolated_repo, _SOURCE_RELPATH)


def _angle_difference(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi
