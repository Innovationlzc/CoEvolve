"""Fail-closed producers for hash-bound MuJoCo scene evidence.

Only ``simulator_integrity`` is produced here.  It runs the repository's
actual USDA -> dual-G1 SONIC MJCF generator, compiles that MJCF with MuJoCo,
saves and reloads an MJB, and inventories every body, joint, actuator, and
geom in the reloaded model.  The generator is staged below the requested
repository output parent, so the external SONIC checkout is read but never
modified.

The other two execution checks intentionally remain blocked.  An initial
contact/support receipt requires the official SONIC controllers to be live for
the complete stable window; stepping an uncontrolled G1 model would be a false
substitute.  Scripted solvability additionally requires a reviewed controller
whose exact bytes are present in ``ALLOWED_SCRIPTED_SOLVABILITY_CONTROLLERS``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from praxis_coop.sonic_mujoco import manual_layout_scene as sonic_scene_generator
from praxis_coop.sonic_mujoco.manual_layout_scene import (
    DEFAULT_BACK_G1_XML,
    DEFAULT_DUAL_SCENE_REL_PATH,
    DEFAULT_FRONT_G1_XML,
    DEFAULT_G1_SOURCE_XML,
    DEFAULT_PAYLOAD_INITIAL_STATE,
    DEFAULT_PAYLOAD_MASS_KG,
    DEFAULT_PAYLOAD_PLACEMENT,
    generate_dual_manual_layout_scene,
)
from praxis_coop.utils import usda_layout as usda_layout_parser

from ..contracts import ProtocolLock
from .evidence import (
    ALLOWED_EVIDENCE_PRODUCERS,
    EVIDENCE_CHECKS,
    EVIDENCE_RECEIPT_KIND,
    EVIDENCE_RECEIPT_SCHEMA_VERSION,
    EvidenceReceiptBinding,
    validate_evidence_receipt,
)
from .integrity import validate_scene
from .loader import compute_asset_composite_hash
from .schema import EvidencePin, SceneSpec


COMMAND_KIND = "coevolve_scene_evidence_command"
COMMAND_SCHEMA_VERSION = 1
CONFIG_KIND = "coevolve_mujoco_integrity_producer_config"
CONFIG_SCHEMA_VERSION = 1
MODEL_INVENTORY_KIND = "coevolve_mujoco_model_inventory"
MODEL_INVENTORY_SCHEMA_VERSION = 1

_GENERATOR_SOURCE_PATHS = (
    "src/praxis_coop/sonic_mujoco/manual_layout_scene.py",
    "src/praxis_coop/utils/usda_layout.py",
)
_ROLE_BODY_NAMES: Mapping[str, str] = {
    "front_robot": "front_pelvis",
    "back_robot": "back_pelvis",
    "payload": "long_box",
}
_ARTIFACT_FILENAMES: Mapping[str, str] = {
    "command": "command.json",
    "config": "config.json",
    "compiled_model": "compiled_model.mjb",
    "model_inventory": "model_inventory.json",
}


class EvidenceProductionError(RuntimeError):
    """Raised before a producer can issue a validator-acceptable receipt."""


class EvidenceProductionBlocked(EvidenceProductionError):
    """Raised when a check has no truthful in-process producer yet."""


@dataclass(frozen=True)
class SimulatorIntegrityResult:
    """Immutable output identity for one simulator-integrity bundle."""

    output_dir: Path
    receipt_path: Path
    receipt_pin: EvidencePin
    binding: EvidenceReceiptBinding
    compiled_model_sha256: str
    model_inventory_sha256: str
    body_count: int
    joint_count: int
    actuator_count: int
    geom_count: int


def produce_scene_evidence(
    check: str,
    *,
    scene: SceneSpec,
    protocol_lock: ProtocolLock,
    repo_root: str | Path,
    output_dir: str | Path,
    gear_root: str | Path,
    invocation: Sequence[str] | None = None,
) -> SimulatorIntegrityResult:
    """Dispatch one evidence check without silently weakening its semantics."""

    if check not in EVIDENCE_CHECKS:
        raise ValueError(f"unsupported scene evidence check: {check}")
    if check == "initial_collision":
        raise EvidenceProductionBlocked(
            "initial_collision requires a >=1 s trace from the live official "
            "dual-SONIC controllers, with classified contacts and bilateral "
            "payload support; uncontrolled MuJoCo stepping is not valid evidence"
        )
    if check == "scripted_solvability":
        raise EvidenceProductionBlocked(
            "scripted_solvability requires a reviewed, byte-pinned controller "
            "approved for the locked skill_api_fingerprint plus raw trajectory "
            "and verifier traces; no controller is currently allowlisted"
        )
    return produce_simulator_integrity_evidence(
        scene=scene,
        protocol_lock=protocol_lock,
        repo_root=repo_root,
        output_dir=output_dir,
        gear_root=gear_root,
        invocation=invocation,
    )


def produce_simulator_integrity_evidence(
    *,
    scene: SceneSpec,
    protocol_lock: ProtocolLock,
    repo_root: str | Path,
    output_dir: str | Path,
    gear_root: str | Path,
    invocation: Sequence[str] | None = None,
) -> SimulatorIntegrityResult:
    """Compile one registered realization and atomically publish its receipt.

    ``output_dir`` must not already exist and must be a repository descendant.
    The large compile workspace and copied mesh inputs are temporary.  Only the
    four raw artifacts required by evidence schema v2 and the receipt survive.
    """

    root = _safe_repository_root(repo_root)
    output, output_relative = _safe_new_output(root, output_dir)
    planning = validate_scene(scene, mode="planning", repo_root=root)
    if not planning.ok:
        raise EvidenceProductionError(
            "scene planning validation failed: " + "; ".join(planning.errors)
        )
    execution_usda = _verified_execution_usda(root, scene)
    generator_sources = _verify_frozen_generator_sources(
        root,
        protocol_lock.base_method_commit,
    )
    external_root = _safe_external_root(gear_root)
    source_g1 = _safe_external_file(
        external_root,
        external_root / DEFAULT_G1_SOURCE_XML,
        "SONIC source G1 MJCF",
    )
    dependency_snapshot = _g1_dependency_snapshot(external_root, source_g1)
    mujoco = _load_mujoco()
    runtime_manifest = _mujoco_runtime_manifest(mujoco)
    command_bytes = _canonical_json_bytes(
        {
            "kind": COMMAND_KIND,
            "schema_version": COMMAND_SCHEMA_VERSION,
            "argv": list(invocation) if invocation is not None else _default_invocation(
                scene,
                root,
                output_relative,
                external_root,
            ),
            "cwd": str(Path.cwd().resolve()),
            "python_executable": str(Path(sys.executable).resolve()),
        }
    )

    _prepare_output_parent(root, output.parent)
    if output.exists() or output.is_symlink():
        raise EvidenceProductionError(f"evidence output already exists: {output}")

    with tempfile.TemporaryDirectory(
        prefix=".coevolve-scene-evidence-",
        dir=str(output.parent),
    ) as temporary_name:
        temporary = Path(temporary_name)
        staged_repo = temporary / "repository"
        staged_output = staged_repo.joinpath(*PurePosixPath(output_relative).parts)
        staged_output.mkdir(parents=True)
        compile_root = temporary / "compile_workspace"
        model_dir = compile_root / DEFAULT_G1_SOURCE_XML.parent
        model_dir.mkdir(parents=True)
        _copy_g1_dependencies(source_g1, dependency_snapshot, model_dir)

        generated = generate_dual_manual_layout_scene(
            source_usda=execution_usda,
            gear_root=compile_root,
            scene_rel_path=DEFAULT_DUAL_SCENE_REL_PATH,
            source_g1_rel_path=source_g1,
            front_g1_rel_path=DEFAULT_FRONT_G1_XML,
            back_g1_rel_path=DEFAULT_BACK_G1_XML,
            use_official_standing_z=True,
            include_payload=True,
            payload_mass_kg=DEFAULT_PAYLOAD_MASS_KG,
            payload_placement=DEFAULT_PAYLOAD_PLACEMENT,
            payload_initial_state=DEFAULT_PAYLOAD_INITIAL_STATE,
        )
        generated_files = _generated_mjcf_manifest(generated)
        if dependency_snapshot != _g1_dependency_snapshot(external_root, source_g1):
            raise EvidenceProductionError(
                "SONIC G1 source or mesh dependencies changed during compilation"
            )

        try:
            model = mujoco.MjModel.from_xml_path(str(generated.scene_path))
        except Exception as exc:  # noqa: BLE001 - preserve native compiler reason
            raise EvidenceProductionError(f"MuJoCo MJCF compilation failed: {exc}") from exc
        compiled_path = staged_output / _ARTIFACT_FILENAMES["compiled_model"]
        try:
            mujoco.mj_saveModel(model, str(compiled_path), None)
        except Exception as exc:  # noqa: BLE001
            raise EvidenceProductionError(f"MuJoCo could not save compiled MJB: {exc}") from exc
        del model
        gc.collect()
        if not compiled_path.is_file() or compiled_path.stat().st_size <= 0:
            raise EvidenceProductionError("MuJoCo emitted an empty compiled MJB")
        try:
            reloaded_model = mujoco.MjModel.from_binary_path(str(compiled_path))
        except Exception as exc:  # noqa: BLE001
            raise EvidenceProductionError(f"saved MuJoCo MJB cannot be reloaded: {exc}") from exc

        compiled_sha256 = _file_sha256(compiled_path)
        inventory = _model_inventory(mujoco, reloaded_model, compiled_sha256)
        inventory_bytes = _canonical_json_bytes(inventory)
        inventory_path = staged_output / _ARTIFACT_FILENAMES["model_inventory"]
        inventory_path.write_bytes(inventory_bytes)
        inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()

        config_bytes = _canonical_json_bytes(
            {
                "kind": CONFIG_KIND,
                "schema_version": CONFIG_SCHEMA_VERSION,
                "scene": {
                    "scene_id": scene.scene_id.value,
                    "scene_hash": scene.scene_hash,
                    "execution_path": scene.asset.execution_path,
                    "execution_sha256": scene.asset.execution_sha256,
                    "execution_composite_sha256": scene.asset.execution_composite_sha256,
                },
                "protocol": {
                    "protocol_id": protocol_lock.protocol_id,
                    "base_method_commit": protocol_lock.base_method_commit,
                    "invariant_fingerprint": protocol_lock.invariant_fingerprint,
                    "terminal_fingerprint": protocol_lock.terminal_fingerprint,
                    "skill_api_fingerprint": protocol_lock.skill_api_fingerprint,
                },
                "producer_source": _producer_source_manifest(root),
                "frozen_generator_sources": generator_sources,
                "sonic_source": {
                    "gear_root": str(external_root),
                    "source_g1_xml": str(source_g1.relative_to(external_root).as_posix()),
                    "dependencies": dependency_snapshot,
                },
                "generation": {
                    "function": (
                        "praxis_coop.sonic_mujoco.manual_layout_scene."
                        "generate_dual_manual_layout_scene"
                    ),
                    "use_official_standing_z": True,
                    "include_payload": True,
                    "payload_mass_kg": DEFAULT_PAYLOAD_MASS_KG,
                    "payload_placement": DEFAULT_PAYLOAD_PLACEMENT,
                    "payload_initial_state": DEFAULT_PAYLOAD_INITIAL_STATE,
                    "generated_mjcf": generated_files,
                },
                "mujoco_runtime": runtime_manifest,
                "compiled_model": {
                    "format": "mjb",
                    "sha256": compiled_sha256,
                    "size_bytes": compiled_path.stat().st_size,
                },
            }
        )
        command_path = staged_output / _ARTIFACT_FILENAMES["command"]
        config_path = staged_output / _ARTIFACT_FILENAMES["config"]
        command_path.write_bytes(command_bytes)
        config_path.write_bytes(config_bytes)

        artifacts = []
        for artifact_type, filename in _ARTIFACT_FILENAMES.items():
            path = staged_output / filename
            artifacts.append(
                {
                    "artifact_type": artifact_type,
                    "relative_path": (
                        PurePosixPath(output_relative) / filename
                    ).as_posix(),
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        producer_id, producer_version = ALLOWED_EVIDENCE_PRODUCERS[
            "simulator_integrity"
        ]
        measurements = {
            "simulator_name": "MuJoCo",
            "simulator_version": runtime_manifest["python_package_version"],
            "simulator_build": runtime_manifest["core_version"],
            "model_loaded": True,
            "compiled_model_format": "mjb",
            "compiled_model_sha256": compiled_sha256,
            "model_inventory_sha256": inventory_sha256,
            "scene_realization_sha256": scene.asset.execution_composite_sha256,
            "body_count": int(reloaded_model.nbody),
            "joint_count": int(reloaded_model.njnt),
            "actuator_count": int(reloaded_model.nu),
            "geom_count": int(reloaded_model.ngeom),
            "invalid_model_object_count": 0,
        }
        receipt = {
            "kind": EVIDENCE_RECEIPT_KIND,
            "schema_version": EVIDENCE_RECEIPT_SCHEMA_VERSION,
            "check": "simulator_integrity",
            "scene_hash": scene.scene_hash,
            "execution_sha256": scene.asset.execution_sha256,
            "execution_composite_sha256": scene.asset.execution_composite_sha256,
            "protocol_id": protocol_lock.protocol_id,
            "invariant_fingerprint": protocol_lock.invariant_fingerprint,
            "terminal_fingerprint": protocol_lock.terminal_fingerprint,
            "skill_api_fingerprint": protocol_lock.skill_api_fingerprint,
            "producer_id": producer_id,
            "producer_version": producer_version,
            # Schema v2 binds this field to the frozen base method commit.  The
            # generator-source comparison above proves that the code actually
            # used here has those exact committed bytes.
            "producer_commit": protocol_lock.base_method_commit,
            "command_sha256": hashlib.sha256(command_bytes).hexdigest(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "measurements": measurements,
            "raw_artifacts": artifacts,
        }
        receipt_bytes = _canonical_json_bytes(receipt)
        receipt_path = staged_output / "receipt.json"
        receipt_path.write_bytes(receipt_bytes)
        receipt_relative = (
            PurePosixPath(output_relative) / "receipt.json"
        ).as_posix()
        receipt_pin = EvidencePin(
            path=receipt_relative,
            sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        )
        validation = validate_evidence_receipt(
            staged_repo,
            receipt_pin,
            "simulator_integrity",
            scene,
            protocol_lock,
        )
        if validation.errors or validation.binding is None:
            raise EvidenceProductionError(
                "producer emitted a receipt rejected by the host validator: "
                + "; ".join(validation.errors)
            )
        if output.exists() or output.is_symlink():
            raise EvidenceProductionError(f"evidence output appeared concurrently: {output}")
        os.rename(staged_output, output)

    return SimulatorIntegrityResult(
        output_dir=output,
        receipt_path=output / "receipt.json",
        receipt_pin=receipt_pin,
        binding=validation.binding,
        compiled_model_sha256=compiled_sha256,
        model_inventory_sha256=inventory_sha256,
        body_count=measurements["body_count"],
        joint_count=measurements["joint_count"],
        actuator_count=measurements["actuator_count"],
        geom_count=measurements["geom_count"],
    )


def _model_inventory(
    mujoco: ModuleType,
    model: Any,
    compiled_model_sha256: str,
) -> dict[str, Any]:
    object_specs = (
        ("bodies", "nbody", mujoco.mjtObj.mjOBJ_BODY),
        ("joints", "njnt", mujoco.mjtObj.mjOBJ_JOINT),
        ("actuators", "nu", mujoco.mjtObj.mjOBJ_ACTUATOR),
        ("geoms", "ngeom", mujoco.mjtObj.mjOBJ_GEOM),
    )
    inventory: dict[str, Any] = {
        "kind": MODEL_INVENTORY_KIND,
        "schema_version": MODEL_INVENTORY_SCHEMA_VERSION,
        "compiled_model_sha256": compiled_model_sha256,
    }
    for plural, count_field, object_type in object_specs:
        count = int(getattr(model, count_field))
        if count < 0:
            raise EvidenceProductionError(f"MuJoCo reported negative {count_field}")
        items = []
        for object_id in range(count):
            name = mujoco.mj_id2name(model, object_type, object_id)
            if name is not None and (not isinstance(name, str) or not name.strip()):
                raise EvidenceProductionError(
                    f"MuJoCo returned malformed {plural} name at ID {object_id}"
                )
            items.append({"id": object_id, "name": name})
        inventory[plural] = items

    role_bindings = []
    for role, body_name in _ROLE_BODY_NAMES.items():
        object_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        )
        if object_id < 0 or object_id >= int(model.nbody):
            raise EvidenceProductionError(
                f"compiled model is missing required role body {body_name!r}"
            )
        observed_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            object_id,
        )
        if observed_name != body_name:
            raise EvidenceProductionError(
                f"MuJoCo role lookup for {body_name!r} is not reversible"
            )
        role_bindings.append(
            {
                "role": role,
                "object_type": "body",
                "object_id": object_id,
                "object_name": body_name,
            }
        )
    inventory["role_bindings"] = role_bindings
    return inventory


def _verified_execution_usda(root: Path, scene: SceneSpec) -> Path:
    asset = scene.asset
    if (
        asset.execution_path is None
        or asset.execution_sha256 is None
        or asset.execution_composite_sha256 is None
    ):
        raise EvidenceProductionError(
            "scene has no registered realized execution asset; materialize and "
            "review/register it before collecting simulator evidence"
        )
    if compute_asset_composite_hash(asset) != asset.execution_composite_sha256:
        raise EvidenceProductionError("scene execution composite SHA-256 is invalid")
    path = _safe_repository_file(root, asset.execution_path, "execution USDA")
    observed = _file_sha256(path)
    if observed != asset.execution_sha256:
        raise EvidenceProductionError(
            f"execution USDA SHA-256 mismatch: expected {asset.execution_sha256}, "
            f"got {observed}"
        )
    return path


def _verify_frozen_generator_sources(
    root: Path,
    base_method_commit: str,
) -> list[dict[str, str]]:
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode("utf-8").strip()
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise EvidenceProductionError(
            "repository Git identity is unavailable for generator pinning"
        ) from exc
    if Path(top).resolve() != root:
        raise EvidenceProductionError("repo_root is not the Git worktree root")

    expected_modules = {
        _GENERATOR_SOURCE_PATHS[0]: Path(sonic_scene_generator.__file__).resolve(),
        _GENERATOR_SOURCE_PATHS[1]: Path(usda_layout_parser.__file__).resolve(),
    }
    pins = []
    for relative_path in _GENERATOR_SOURCE_PATHS:
        path = _safe_repository_file(root, relative_path, "generator source")
        if expected_modules[relative_path] != path:
            raise EvidenceProductionError(
                f"imported generator module is not the repository file {relative_path}"
            )
        try:
            committed = subprocess.run(
                ["git", "-C", str(root), "show", f"{base_method_commit}:{relative_path}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EvidenceProductionError(
                f"cannot resolve {relative_path} at frozen commit {base_method_commit}"
            ) from exc
        current_sha256 = _file_sha256(path)
        committed_sha256 = hashlib.sha256(committed).hexdigest()
        if current_sha256 != committed_sha256:
            raise EvidenceProductionError(
                f"generator source {relative_path} differs from frozen base method commit"
            )
        pins.append({"relative_path": relative_path, "sha256": current_sha256})
    return pins


def _producer_source_manifest(root: Path) -> dict[str, str]:
    source = Path(__file__).resolve()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise EvidenceProductionError(
            "evidence producer was imported from outside repo_root"
        ) from exc
    return {"relative_path": relative, "sha256": _file_sha256(source)}


def _g1_dependency_snapshot(root: Path, source_g1: Path) -> list[dict[str, Any]]:
    try:
        tree = ET.parse(source_g1)
    except (ET.ParseError, OSError) as exc:
        raise EvidenceProductionError(f"cannot parse SONIC source G1 MJCF: {exc}") from exc
    compiler = tree.getroot().find("compiler")
    mesh_dir_value = compiler.get("meshdir", ".") if compiler is not None else "."
    mesh_dir_rel = _safe_dependency_relative_path(mesh_dir_value, "compiler meshdir")
    source_parent = source_g1.parent
    dependencies: list[Path] = [source_g1]
    seen_files: set[str] = set()
    for mesh in tree.getroot().iter("mesh"):
        file_value = mesh.get("file")
        if not file_value:
            raise EvidenceProductionError("SONIC source G1 MJCF has a mesh without file")
        mesh_file_rel = _safe_dependency_relative_path(file_value, "mesh file")
        logical = (mesh_dir_rel / mesh_file_rel).as_posix()
        if logical in seen_files:
            continue
        seen_files.add(logical)
        dependencies.append(source_parent / mesh_dir_rel / mesh_file_rel)
    snapshot = []
    for path in dependencies:
        safe = _safe_external_file(root, path, "SONIC model dependency")
        snapshot.append(
            {
                "relative_path": safe.relative_to(root).as_posix(),
                "sha256": _file_sha256(safe),
                "size_bytes": safe.stat().st_size,
            }
        )
    return sorted(snapshot, key=lambda item: item["relative_path"])


def _copy_g1_dependencies(
    source_g1: Path,
    snapshot: Sequence[Mapping[str, Any]],
    model_dir: Path,
) -> None:
    source_root = source_g1.parents[len(DEFAULT_G1_SOURCE_XML.parts) - 1]
    for item in snapshot:
        relative = PurePosixPath(str(item["relative_path"]))
        if relative.as_posix() == DEFAULT_G1_SOURCE_XML.as_posix():
            continue
        # Snapshot paths are gear-root relative.  The generated namespace MJCF
        # resolves meshes relative to the G1 model directory.
        source = source_root.joinpath(*relative.parts)
        try:
            local_relative = source.relative_to(source_g1.parent)
        except ValueError as exc:
            raise EvidenceProductionError(
                f"SONIC dependency is outside the G1 model directory: {source}"
            ) from exc
        destination = model_dir / local_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _file_sha256(destination) != item["sha256"]:
            raise EvidenceProductionError(
                f"copied SONIC dependency changed bytes: {item['relative_path']}"
            )


def _generated_mjcf_manifest(generated: Any) -> list[dict[str, Any]]:
    files = [("scene", Path(generated.scene_path))]
    files.extend(
        (f"robot_{item.robot_id}", Path(item.output_path))
        for item in generated.robot_xmls
    )
    result = []
    for role, path in files:
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise EvidenceProductionError(f"generator did not emit regular {role} MJCF")
        result.append(
            {
                "role": role,
                "filename": path.name,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return result


def _mujoco_runtime_manifest(mujoco: ModuleType) -> dict[str, Any]:
    package_version = str(getattr(mujoco, "__version__", "")).strip()
    try:
        core_version = str(mujoco.mj_versionString()).strip()
    except Exception as exc:  # noqa: BLE001
        raise EvidenceProductionError("MuJoCo core version is unavailable") from exc
    if not package_version or not core_version:
        raise EvidenceProductionError("MuJoCo runtime version is empty")
    bindings = []
    for name in ("_structs", "_functions", "_enums", "_callbacks"):
        module = getattr(mujoco, name, None)
        module_path = getattr(module, "__file__", None)
        if module_path:
            path = Path(module_path).resolve()
            if path.is_file() and not path.is_symlink():
                bindings.append(
                    {
                        "module": name,
                        "path": str(path),
                        "sha256": _file_sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    return {
        "python_package_version": package_version,
        "core_version": core_version,
        "binding_modules": bindings,
    }


def _load_mujoco() -> ModuleType:
    try:
        import mujoco
    except ImportError as exc:
        raise EvidenceProductionBlocked(
            "MuJoCo Python bindings are unavailable; invoke this command with "
            "the pinned SONIC simulation environment"
        ) from exc
    required = ("MjModel", "mjtObj", "mj_id2name", "mj_name2id", "mj_saveModel")
    missing = [name for name in required if not hasattr(mujoco, name)]
    if missing:
        raise EvidenceProductionBlocked(
            f"MuJoCo bindings are incomplete: missing {', '.join(missing)}"
        )
    return mujoco


def _default_invocation(
    scene: SceneSpec,
    root: Path,
    output_relative: str,
    gear_root: Path,
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-m",
        "coevolve_experiments",
        "produce-scene-evidence",
        "--check",
        "simulator_integrity",
        "--scene",
        scene.scene_id.value,
        "--repo-root",
        str(root),
        "--output-dir",
        output_relative,
        "--gear-root",
        str(gear_root),
    ]


def _safe_repository_root(value: str | Path) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise EvidenceProductionError(f"repo_root must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise EvidenceProductionError(f"repo_root is missing: {path}") from exc
    if not resolved.is_dir():
        raise EvidenceProductionError(f"repo_root is not a directory: {resolved}")
    return resolved


def _safe_new_output(root: Path, value: str | Path) -> tuple[Path, str]:
    raw = Path(value)
    if raw.is_absolute():
        candidate = raw
    else:
        logical = PurePosixPath(str(value))
        if logical.as_posix() in {"", "."} or ".." in logical.parts:
            raise EvidenceProductionError("output_dir must be a non-root relative path")
        candidate = root.joinpath(*logical.parts)
    try:
        relative = candidate.absolute().relative_to(root).as_posix()
    except ValueError as exc:
        raise EvidenceProductionError("output_dir must remain below repo_root") from exc
    if PurePosixPath(relative).as_posix() != relative or relative in {"", "."}:
        raise EvidenceProductionError("output_dir is not a normalized repository path")
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceProductionError(f"output_dir must not traverse a symlink: {current}")
    return root.joinpath(*PurePosixPath(relative).parts), relative


def _prepare_output_parent(root: Path, parent: Path) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    try:
        parent.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidenceProductionError("output parent escaped repo_root") from exc
    current = root
    relative = parent.relative_to(root)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceProductionError(f"output parent traverses a symlink: {current}")


def _safe_repository_file(root: Path, relative_path: str, label: str) -> Path:
    logical = PurePosixPath(relative_path)
    if logical.is_absolute() or ".." in logical.parts or logical.as_posix() != relative_path:
        raise EvidenceProductionError(f"{label} path is not normalized and relative")
    return _safe_external_file(root, root.joinpath(*logical.parts), label)


def _safe_external_root(value: str | Path) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise EvidenceProductionError(f"gear_root must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise EvidenceProductionError(f"gear_root is missing: {path}") from exc
    if not resolved.is_dir():
        raise EvidenceProductionError(f"gear_root is not a directory: {resolved}")
    return resolved


def _safe_external_file(root: Path, path: Path, label: str) -> Path:
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise EvidenceProductionError(f"{label} escapes its allowed root: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceProductionError(f"{label} must not traverse a symlink: {current}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidenceProductionError(f"{label} is missing or escapes its root: {path}") from exc
    if not resolved.is_file():
        raise EvidenceProductionError(f"{label} is not a regular file: {path}")
    return resolved


def _safe_dependency_relative_path(value: str, label: str) -> PurePosixPath:
    logical = PurePosixPath(value)
    if (
        not value
        or logical.is_absolute()
        or ".." in logical.parts
        or logical.as_posix() != value
    ):
        raise EvidenceProductionError(f"{label} is not a safe relative POSIX path")
    return logical


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "EvidenceProductionBlocked",
    "EvidenceProductionError",
    "SimulatorIntegrityResult",
    "produce_scene_evidence",
    "produce_simulator_integrity_evidence",
]
