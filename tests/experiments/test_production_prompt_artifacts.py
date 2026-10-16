from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from coevolve_experiments.contracts import MethodVariant
from coevolve_experiments.decision_prompts import (
    EXTERNAL_PROMPT_VARIANTS,
    PRODUCTION_ARTIFACT_SCOPE,
    PRODUCTION_POLICY_REPRESENTATION,
    ProductionPromptArtifactError,
    build_production_prompt_identity,
    decision_prompt_fingerprint_payload,
    load_production_prompt_registry,
)
from coevolve_experiments.methods import (
    CoherentAdaptedAdapter,
    ExecutionCapability,
    FlatReflectiveRepairAdapter,
    NoDynamicExperienceAdapter,
    PolicyRepresentation,
    RemacAdaptedAdapter,
)
from coevolve_experiments.serialization import sha256_json


FROZEN_V83_PROMPT_SHA256 = (
    "7214a561d9338956f59d5f658d6a4d3e3716d5dc594e475cbbce7998624680ec"
)
SHARED_PROMPT_SHA256 = sha256(b"reviewed shared prompt contract").hexdigest()


def _source_root() -> Path:
    return Path(
        "src/coevolve_experiments/decision_prompts"
    ).resolve()


def _copy_registry(tmp_path: Path) -> Path:
    destination = tmp_path / "decision_prompts"
    shutil.copytree(_source_root(), destination)
    return destination


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _repin_variant(root: Path, variant: MethodVariant) -> None:
    registry_path = root / "registry.json"
    registry = _read_json(registry_path)
    entry = next(
        item
        for item in registry["artifacts"]
        if item["method_variant"] == variant.value
    )
    manifest_path = root / entry["manifest_relpath"]
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    manifest_sha = sha256(manifest_raw).hexdigest()
    hashes = {}
    for role in manifest["roles"]:
        relpath = role["prompt_file"]
        if relpath is not None:
            hashes[relpath] = sha256(
                (manifest_path.parent / relpath).read_bytes()
            ).hexdigest()
    entry["manifest_sha256"] = manifest_sha
    entry["decision_prompt_fingerprint"] = sha256_json(
        decision_prompt_fingerprint_payload(
            method_variant=variant,
            manifest_sha256=manifest_sha,
            prompt_file_hashes=hashes,
        )
    )
    _write_json(registry_path, registry)


def test_registry_loads_four_real_byte_recomputable_artifacts() -> None:
    registry = load_production_prompt_registry()

    assert tuple(item.method_variant for item in registry.artifacts) == (
        EXTERNAL_PROMPT_VARIANTS
    )
    assert len({item.decision_prompt_fingerprint for item in registry.artifacts}) == 4
    assert all(item.manifest_path.is_file() for item in registry.artifacts)
    for artifact in registry.artifacts:
        assert artifact.artifact_scope == PRODUCTION_ARTIFACT_SCOPE
        assert artifact.policy_representation == PRODUCTION_POLICY_REPRESENTATION
        prompt_hashes = {
            role.prompt_relpath: role.prompt_sha256
            for role in artifact.roles
            if role.prompt_relpath is not None
        }
        payload = decision_prompt_fingerprint_payload(
            method_variant=artifact.method_variant,
            manifest_sha256=sha256(artifact.manifest_path.read_bytes()).hexdigest(),
            prompt_file_hashes=prompt_hashes,
        )
        assert sha256_json(payload) == artifact.decision_prompt_fingerprint


def test_full_pin_is_external_and_all_five_lock_pins_are_real() -> None:
    identity = build_production_prompt_identity(
        shared_prompt_contract_fingerprint=SHARED_PROMPT_SHA256,
        full_method_prompt_fingerprint=FROZEN_V83_PROMPT_SHA256,
    )

    assert identity.shared_prompt_contract_fingerprint == SHARED_PROMPT_SHA256
    assert tuple(pin.method_variant for pin in identity.method_decision_prompt_pins) == tuple(
        MethodVariant
    )
    pins = {
        pin.method_variant: pin.decision_prompt_fingerprint
        for pin in identity.method_decision_prompt_pins
    }
    assert pins[MethodVariant.COEVOLE_FULL] == FROZEN_V83_PROMPT_SHA256
    registry = load_production_prompt_registry()
    for variant in EXTERNAL_PROMPT_VARIANTS:
        assert pins[variant] == registry.artifact_for(
            variant
        ).decision_prompt_fingerprint
    with pytest.raises(ProductionPromptArtifactError, match="frozen V83"):
        registry.artifact_for(MethodVariant.COEVOLE_FULL)


def test_no_dynamic_reuses_v83_prompts_and_only_closes_memory() -> None:
    artifact = load_production_prompt_registry().artifact_for(
        MethodVariant.NO_DYNAMIC_EXPERIENCE
    )

    assert artifact.hierarchical_evolution is True
    assert artifact.durable_dynamic_memory is False
    assert artifact.memory_gates.all_disabled
    assert "memory_update" in artifact.forbidden_method_roles
    assert {role.role for role in artifact.roles} == {
        "task_orchestrator",
        "policy_generator",
        "evolution_manager",
        "mutation_operator",
    }
    assert all(role.uses_frozen_v83_base for role in artifact.roles)
    assert all(role.prompt_relpath is None for role in artifact.roles)


def test_method_semantics_and_forbidden_roles_are_frozen() -> None:
    registry = load_production_prompt_registry()
    flat = registry.artifact_for(MethodVariant.FLAT_REFLECTIVE_REPAIR)
    remac = registry.artifact_for(MethodVariant.REMAC_ADAPTED)
    coherent = registry.artifact_for(MethodVariant.COHERENT_ADAPTED)

    assert flat.memory_gates == type(flat.memory_gates)(True, True, True, True, True)
    assert flat.hierarchical_evolution is False
    assert flat.forbidden_method_roles == (
        "evolution_manager",
        "mutation_operator",
    )
    assert flat.role("flat_strategy_selector").output_contract == (
        "v83_flat_strategy_commit_json_v1"
    )
    assert "immutable" in flat.role("flat_strategy_selector").prompt_text
    assert "exact action-exemplar retrieval" in flat.role(
        "flat_strategy_selector"
    ).prompt_text
    assert remac.decision_structure == "task_plan_reflection_and_replanning"
    assert coherent.decision_structure == (
        "central_proposal_execution_feedback_adjustment"
    )
    assert remac.memory_gates.all_disabled
    assert coherent.memory_gates.all_disabled
    assert remac.role("remac_precondition_reflection").output_contract == (
        "v83_team_policy_json_v1"
    )
    assert coherent.role("coherent_front_feasibility").prompt_sha256 == (
        coherent.role("coherent_back_feasibility").prompt_sha256
    )


def test_production_prompt_artifacts_cannot_be_confused_with_smoke_adapters() -> None:
    registry = load_production_prompt_registry()
    adapters = {
        MethodVariant.NO_DYNAMIC_EXPERIENCE: NoDynamicExperienceAdapter,
        MethodVariant.FLAT_REFLECTIVE_REPAIR: FlatReflectiveRepairAdapter,
        MethodVariant.REMAC_ADAPTED: RemacAdaptedAdapter,
        MethodVariant.COHERENT_ADAPTED: CoherentAdaptedAdapter,
    }

    for variant, adapter_type in adapters.items():
        artifact = registry.artifact_for(variant)
        assert artifact.policy_representation == "praxis_code_as_policy"
        assert adapter_type.descriptor.policy_representation is (
            PolicyRepresentation.TEAM_INTENT_IR
        )
        assert adapter_type.descriptor.execution_capability is (
            ExecutionCapability.CONTRACT_SMOKE_ONLY
        )


def test_registry_is_deterministic_and_contains_no_transport_or_secret() -> None:
    first = load_production_prompt_registry()
    second = load_production_prompt_registry()

    assert first.registry_sha256 == second.registry_sha256
    assert tuple(
        item.decision_prompt_fingerprint for item in first.artifacts
    ) == tuple(item.decision_prompt_fingerprint for item in second.artifacts)
    combined = "\n".join(
        role.prompt_text
        for artifact in first.artifacts
        for role in artifact.roles
    ).lower()
    assert "http://" not in combined
    assert "https://" not in combined
    assert "authorization: bearer" not in combined


def test_changed_prompt_bytes_fail_closed(tmp_path: Path) -> None:
    root = _copy_registry(tmp_path)
    prompt = (
        root
        / "artifacts/flat_reflective_repair/flat_policy_generator.txt"
    )
    prompt.write_text(prompt.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")

    with pytest.raises(ProductionPromptArtifactError, match="artifact bytes"):
        load_production_prompt_registry(root)


def test_manifest_semantic_drift_fails_even_when_attacker_repins_bytes(
    tmp_path: Path,
) -> None:
    root = _copy_registry(tmp_path)
    manifest_path = root / "artifacts/remac_adapted/manifest.json"
    manifest = _read_json(manifest_path)
    manifest["hierarchical_evolution"] = True
    _write_json(manifest_path, manifest)
    _repin_variant(root, MethodVariant.REMAC_ADAPTED)

    with pytest.raises(ProductionPromptArtifactError, match="differs in"):
        load_production_prompt_registry(root)


def test_no_dynamic_cannot_add_an_overlay_even_when_repinned(tmp_path: Path) -> None:
    root = _copy_registry(tmp_path)
    bundle = root / "artifacts/no_dynamic_experience"
    overlay = bundle / "task_orchestrator.txt"
    overlay.write_text("Do something different.\n", encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["roles"][0]["prompt_file"] = overlay.name
    _write_json(manifest_path, manifest)
    _repin_variant(root, MethodVariant.NO_DYNAMIC_EXPERIENCE)

    with pytest.raises(ProductionPromptArtifactError, match="byte-for-byte"):
        load_production_prompt_registry(root)


def test_duplicate_json_path_escape_and_symlink_fail_closed(tmp_path: Path) -> None:
    duplicate_root = _copy_registry(tmp_path / "duplicate")
    registry_path = duplicate_root / "registry.json"
    raw = registry_path.read_text(encoding="utf-8")
    registry_path.write_text(
        raw.replace(
            '"kind": "coevolve_production_decision_prompt_registry",',
            '"kind": "coevolve_production_decision_prompt_registry",\n'
            '  "kind": "coevolve_production_decision_prompt_registry",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProductionPromptArtifactError, match="duplicate"):
        load_production_prompt_registry(duplicate_root)

    escape_root = _copy_registry(tmp_path / "escape")
    registry = _read_json(escape_root / "registry.json")
    registry["artifacts"][0]["manifest_relpath"] = "../manifest.json"
    _write_json(escape_root / "registry.json", registry)
    with pytest.raises(ProductionPromptArtifactError, match="normalized relative"):
        load_production_prompt_registry(escape_root)

    symlink_root = _copy_registry(tmp_path / "symlink")
    prompt = (
        symlink_root
        / "artifacts/remac_adapted/remac_postcondition_reflection.txt"
    )
    target = tmp_path / "same-prompt.txt"
    target.write_bytes(prompt.read_bytes())
    prompt.unlink()
    prompt.symlink_to(target)
    with pytest.raises(ProductionPromptArtifactError, match="symlink"):
        load_production_prompt_registry(symlink_root)


@pytest.mark.parametrize("value", ("", "0" * 63, "G" * 64, None))
def test_prompt_identity_rejects_placeholder_or_malformed_hashes(value) -> None:
    with pytest.raises(ProductionPromptArtifactError, match="SHA-256"):
        build_production_prompt_identity(
            shared_prompt_contract_fingerprint=value,
            full_method_prompt_fingerprint=FROZEN_V83_PROMPT_SHA256,
        )
    with pytest.raises(ProductionPromptArtifactError, match="SHA-256"):
        build_production_prompt_identity(
            shared_prompt_contract_fingerprint=SHARED_PROMPT_SHA256,
            full_method_prompt_fingerprint=value,
        )


def test_valid_but_non_v83_full_prompt_hash_is_rejected() -> None:
    with pytest.raises(ProductionPromptArtifactError, match="frozen V83"):
        build_production_prompt_identity(
            shared_prompt_contract_fingerprint=SHARED_PROMPT_SHA256,
            full_method_prompt_fingerprint=sha256(b"different prompt").hexdigest(),
        )
