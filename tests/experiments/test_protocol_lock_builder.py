from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import coevolve_experiments.protocol_lock_builder as builder_module
from coevolve_experiments.cli import main
from coevolve_experiments.contracts import (
    MethodDecisionPromptPin,
    MethodVariant,
    Table1SourceCohort,
)
from coevolve_experiments.full_method_compatibility import (
    FullMethodCompatibilityPin,
    V83_FULL_BEHAVIOR_RELPATHS,
)
from coevolve_experiments.protocol_lock_builder import (
    PROTOCOL_ID_PREFIX,
    ProtocolContractArtifacts,
    ProtocolLockBuildError,
    StrategySourceInputs,
    build_canonical_protocol_lock,
    canonical_protocol_contract_payloads,
    load_protocol_lock_build_inputs,
    protocol_contract_artifact_document,
    protocol_lock_build_input_document,
    write_protocol_lock_build,
)
from coevolve_experiments.provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE,
    SECONDARY_GEMINI_PROVIDER_ROUTE,
)
from coevolve_experiments.serialization import sha256_json


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _contract_artifacts(tmp_path: Path, prompt: bytes) -> ProtocolContractArtifacts:
    paths = {}
    for role, payload in canonical_protocol_contract_payloads().items():
        path = tmp_path / f"{role}.json"
        _write_json(path, protocol_contract_artifact_document(role, payload))
        paths[role] = path
    prompt_path = tmp_path / "v83_prompts.py"
    prompt_path.write_bytes(prompt)
    return ProtocolContractArtifacts(
        invariant=paths["invariant"],
        terminal_predicate=paths["terminal_predicate"],
        observation_contract=paths["observation_contract"],
        task_instruction=paths["task_instruction"],
        shared_prompt_contract=prompt_path,
    )


def _full_method(prompt: bytes) -> FullMethodCompatibilityPin:
    prompt_hash = hashlib.sha256(prompt).hexdigest()
    hashes = {
        "initial_online_memory_sha256": hashlib.sha256(b"initial").hexdigest(),
        "prompt_sha256": prompt_hash,
        "route_spec_sha256": hashlib.sha256(b"route").hexdigest(),
        "scene_sha256": hashlib.sha256(b"scene").hexdigest(),
        "skill_api_sha256": hashlib.sha256(b"skills").hexdigest(),
        "static_memory_sha256": hashlib.sha256(b"static").hexdigest(),
        "verifier_config_sha256": hashlib.sha256(b"verifier-config").hexdigest(),
        "verifier_source_sha256": hashlib.sha256(b"verifier-source").hexdigest(),
    }
    return FullMethodCompatibilityPin.from_artifact_hashes(
        hashes,
        behavior_artifacts={
            relpath: hashlib.sha256(f"fixture:{relpath}".encode()).hexdigest()
            for relpath in V83_FULL_BEHAVIOR_RELPATHS
        },
    )


def _audits(prompt: bytes) -> tuple[SimpleNamespace, SimpleNamespace]:
    full_method = _full_method(prompt)
    shared = full_method.prompt_sha256
    gpt = Table1SourceCohort(
        backbone="gpt-5.4-mini-2026-03-17",
        protocol_id="coevolve-eval-v83-v33-efficient",
        method_commit="1" * 40,
        seeds=(200, 201, 202, 203, 204),
        condition_fingerprint=hashlib.sha256(b"gpt-condition").hexdigest(),
        result_manifest_hash=hashlib.sha256(b"gpt-results").hexdigest(),
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        shared_prompt_contract_fingerprint=shared,
        decision_prompt_fingerprint=shared,
        full_method_compatibility_fingerprint=full_method.fingerprint,
    )
    gemini = Table1SourceCohort(
        backbone="gemini-3.5-flash",
        protocol_id="coevolve-eval-v83-v33-efficient-gemini",
        method_commit="2" * 40,
        seeds=(200, 201, 202, 203, 204),
        condition_fingerprint=hashlib.sha256(b"gemini-condition").hexdigest(),
        result_manifest_hash=hashlib.sha256(b"gemini-results").hexdigest(),
        provider_route_id=SECONDARY_GEMINI_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=SECONDARY_GEMINI_PROVIDER_ROUTE.fingerprint,
        shared_prompt_contract_fingerprint=shared,
        decision_prompt_fingerprint=shared,
        full_method_compatibility_fingerprint=full_method.fingerprint,
    )
    return (
        SimpleNamespace(
            protocol_valid=True,
            source=gpt,
            full_method_compatibility=full_method,
            manifest_sha256=gpt.result_manifest_hash,
        ),
        SimpleNamespace(
            protocol_valid=True,
            source=gemini,
            full_method_compatibility=full_method,
            manifest_sha256=gemini.result_manifest_hash,
        ),
    )


def _sources(tmp_path: Path) -> tuple[StrategySourceInputs, StrategySourceInputs]:
    gpt_manifest = tmp_path / "strict_result_manifest.json"
    gemini_manifest = tmp_path / "gemini_result_manifest.json"
    gpt_manifest.write_text("{}\n", encoding="utf-8")
    gemini_manifest.write_text("{}\n", encoding="utf-8")
    gpt_runs = []
    gemini_runs = []
    for seed in range(200, 205):
        gpt_run = tmp_path / "gpt" / f"seed_{seed:03d}"
        gemini_run = tmp_path / "gemini" / f"seed_{seed:03d}"
        gpt_run.mkdir(parents=True)
        gemini_run.mkdir(parents=True)
        gpt_runs.append(gpt_run)
        gemini_runs.append(gemini_run)

    def source(prefix: str, runs: list[Path], manifest: Path, parent=None):
        evidence = tmp_path / f"{prefix}_evidence"
        evidence.mkdir()
        bank = tmp_path / f"{prefix}_bank.json"
        provenance = tmp_path / f"{prefix}_provenance.json"
        candidate = tmp_path / f"{prefix}_candidate.json"
        for path in (bank, provenance, candidate):
            path.write_text("{}\n", encoding="utf-8")
        return StrategySourceInputs(
            source_manifest=manifest,
            source_runs=tuple(runs),
            evidence_package=evidence,
            candidate_bank=bank,
            candidate_provenance=provenance,
            candidate_attestation=candidate,
            parent_manifest=parent,
        )

    return (
        source("gpt", gpt_runs, gpt_manifest),
        source("gemini", gemini_runs, gemini_manifest, gpt_manifest),
    )


def _patch_complete_evidence(
    monkeypatch: pytest.MonkeyPatch,
    prompt: bytes,
) -> None:
    gpt, gemini = _audits(prompt)

    def fake_audit(path, cohort, **kwargs):
        return gpt if cohort == "gpt" else gemini

    monkeypatch.setattr(builder_module, "audit_frozen_table1_candidate", fake_audit)

    def fake_candidate(cohort, source, audit):
        return {
            "source_cohort": cohort,
            "source": builder_module.to_primitive(audit.source),
            "source_aggregate_manifest_sha256": audit.manifest_sha256,
            "posthoc_evidence_manifest_sha256": hashlib.sha256(
                f"{cohort}-evidence-manifest".encode()
            ).hexdigest(),
            "posthoc_evidence_payload_sha256": hashlib.sha256(
                f"{cohort}-evidence-payload".encode()
            ).hexdigest(),
            "bank_sha256": hashlib.sha256(f"{cohort}-bank".encode()).hexdigest(),
            "provenance_sha256": hashlib.sha256(
                f"{cohort}-provenance".encode()
            ).hexdigest(),
            "candidate_payload_sha256": hashlib.sha256(
                f"{cohort}-candidate-payload".encode()
            ).hexdigest(),
            "candidate_file_sha256": hashlib.sha256(
                f"{cohort}-candidate-file".encode()
            ).hexdigest(),
        }

    monkeypatch.setattr(builder_module, "_audit_strategy_candidate", fake_candidate)
    prompt_hash = hashlib.sha256(prompt).hexdigest()
    monkeypatch.setattr(
        builder_module,
        "_build_method_prompt_identity",
        lambda value: (
            tuple(
                MethodDecisionPromptPin(
                    variant,
                    prompt_hash
                    if variant is MethodVariant.COEVOLE_FULL
                    else hashlib.sha256(variant.value.encode()).hexdigest(),
                )
                for variant in MethodVariant
            ),
            hashlib.sha256(b"prompt-registry").hexdigest(),
        ),
    )


def test_builder_derives_canonical_identity_and_45_bound_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = b"# exact audited V83 prompt bytes\n"
    _patch_complete_evidence(monkeypatch, prompt)
    gpt_source, gemini_source = _sources(tmp_path)
    artifacts = _contract_artifacts(tmp_path, prompt)

    first = build_canonical_protocol_lock(
        gpt_source=gpt_source,
        gemini_source=gemini_source,
        contract_artifacts=artifacts,
    )
    second = build_canonical_protocol_lock(
        gpt_source=gpt_source,
        gemini_source=gemini_source,
        contract_artifacts=artifacts,
    )

    assert first.lock == second.lock
    assert first.lock.protocol_id.startswith(PROTOCOL_ID_PREFIX + "-")
    assert first.receipt["new_run_count"] == 45
    assert first.receipt["reuse_count"] == 3
    assert first.receipt["physics_dispatched"] is False
    assert first.receipt["paper_evidence"] is False
    assert first.receipt["protocol_fingerprint"] == sha256_json(first.identity)
    rendered = json.dumps(first.receipt, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "credential" not in rendered.lower()


def test_checked_in_protocol_artifacts_match_registered_semantics(
    tmp_path: Path,
) -> None:
    root = Path(builder_module.__file__).resolve().parent / "protocol_artifacts"
    prompt_path = tmp_path / "prompts.py"
    prompt_path.write_text("# reviewed prompt artifact\n", encoding="utf-8")
    pins = builder_module._load_contract_artifacts(
        ProtocolContractArtifacts(
            invariant=root / "invariant.json",
            terminal_predicate=root / "terminal_predicate.json",
            observation_contract=root / "observation_contract.json",
            task_instruction=root / "task_instruction.json",
            shared_prompt_contract=prompt_path,
        )
    )
    assert tuple(pin.role for pin in pins) == (
        "invariant",
        "terminal_predicate",
        "observation_contract",
        "task_instruction",
        "shared_prompt_contract",
    )
    assert all(len(pin.sha256) == 64 and pin.size_bytes > 0 for pin in pins)


def test_builder_consumes_byte_recomputed_production_prompt_registry() -> None:
    v83_prompt = "7214a561d9338956f59d5f658d6a4d3e3716d5dc594e475cbbce7998624680ec"
    pins, registry_sha256 = builder_module._build_method_prompt_identity(v83_prompt)
    assert tuple(pin.method_variant for pin in pins) == tuple(MethodVariant)
    assert pins[0].decision_prompt_fingerprint == v83_prompt
    assert len({pin.decision_prompt_fingerprint for pin in pins}) == 5
    assert len(registry_sha256) == 64


def test_builder_rejects_contract_semantic_drift_before_source_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = b"# prompt\n"
    artifacts = _contract_artifacts(tmp_path, prompt)
    terminal = json.loads(artifacts.terminal_predicate.read_text())
    terminal["payload"]["payload_tilt_max_deg"] = 25.0
    terminal["payload_sha256"] = sha256_json(terminal["payload"])
    _write_json(artifacts.terminal_predicate, terminal)
    gpt_source, gemini_source = _sources(tmp_path)
    called = False

    def should_not_audit(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        builder_module, "audit_frozen_table1_candidate", should_not_audit
    )
    with pytest.raises(ProtocolLockBuildError, match="semantics differ"):
        build_canonical_protocol_lock(
            gpt_source=gpt_source,
            gemini_source=gemini_source,
            contract_artifacts=artifacts,
        )
    assert called is False


def test_builder_rejects_shared_prompt_not_equal_to_audited_v83(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audited_prompt = b"# audited prompt\n"
    _patch_complete_evidence(monkeypatch, audited_prompt)
    gpt_source, gemini_source = _sources(tmp_path)
    artifacts = _contract_artifacts(tmp_path, b"# different prompt\n")
    with pytest.raises(ProtocolLockBuildError, match="shared prompt artifact bytes"):
        build_canonical_protocol_lock(
            gpt_source=gpt_source,
            gemini_source=gemini_source,
            contract_artifacts=artifacts,
        )


def test_builder_rejects_gemini_shared_prompt_drift_from_gpt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = b"# audited prompt\n"
    _patch_complete_evidence(monkeypatch, prompt)
    gpt, gemini = _audits(prompt)
    gemini.source = replace(
        gemini.source,
        shared_prompt_contract_fingerprint=hashlib.sha256(b"drift").hexdigest(),
    )
    monkeypatch.setattr(
        builder_module,
        "audit_frozen_table1_candidate",
        lambda path, cohort, **kwargs: gpt if cohort == "gpt" else gemini,
    )
    gpt_source, gemini_source = _sources(tmp_path)
    with pytest.raises(ProtocolLockBuildError, match="Gemini provider companion"):
        build_canonical_protocol_lock(
            gpt_source=gpt_source,
            gemini_source=gemini_source,
            contract_artifacts=_contract_artifacts(tmp_path, prompt),
        )


def test_builder_rejects_incomplete_five_run_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = b"# prompt\n"
    gpt, gemini = _audits(prompt)
    gpt.protocol_valid = False
    monkeypatch.setattr(
        builder_module,
        "audit_frozen_table1_candidate",
        lambda path, cohort, **kwargs: gpt if cohort == "gpt" else gemini,
    )
    gpt_source, gemini_source = _sources(tmp_path)
    with pytest.raises(ProtocolLockBuildError, match="complete audited five-run"):
        build_canonical_protocol_lock(
            gpt_source=gpt_source,
            gemini_source=gemini_source,
            contract_artifacts=_contract_artifacts(tmp_path, prompt),
        )


def test_candidate_chain_requires_existing_bank_provenance_and_attestation(
    tmp_path: Path,
) -> None:
    prompt = b"# prompt\n"
    audit, _ = _audits(prompt)
    gpt_source, _ = _sources(tmp_path)
    gpt_source.candidate_bank.unlink()
    with pytest.raises(ProtocolLockBuildError, match="candidate bank"):
        builder_module._audit_strategy_candidate("gpt", gpt_source, audit)


def test_build_input_manifest_round_trips_only_complete_real_paths(
    tmp_path: Path,
) -> None:
    prompt = b"# prompt\n"
    gpt_source, gemini_source = _sources(tmp_path)
    artifacts = _contract_artifacts(tmp_path, prompt)
    document = protocol_lock_build_input_document(
        gpt_source=gpt_source,
        gemini_source=gemini_source,
        contract_artifacts=artifacts,
    )
    manifest = tmp_path / "build_inputs.json"
    _write_json(manifest, document)
    loaded = load_protocol_lock_build_inputs(manifest)
    assert loaded.gpt_source == gpt_source
    assert loaded.gemini_source == gemini_source
    assert loaded.contract_artifacts == artifacts
    assert "sk-" not in manifest.read_text()


def test_build_input_manifest_self_hash_is_fail_closed(tmp_path: Path) -> None:
    prompt = b"# prompt\n"
    gpt_source, gemini_source = _sources(tmp_path)
    document = protocol_lock_build_input_document(
        gpt_source=gpt_source,
        gemini_source=gemini_source,
        contract_artifacts=_contract_artifacts(tmp_path, prompt),
    )
    document["gpt_source"]["source_runs"] = document["gpt_source"][
        "source_runs"
    ][:-1]
    manifest = tmp_path / "tampered_inputs.json"
    _write_json(manifest, document)
    with pytest.raises(ProtocolLockBuildError, match="self-hash mismatch"):
        load_protocol_lock_build_inputs(manifest)


def test_protocol_lock_write_is_idempotent_and_refuses_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = b"# prompt\n"
    _patch_complete_evidence(monkeypatch, prompt)
    gpt_source, gemini_source = _sources(tmp_path)
    build = build_canonical_protocol_lock(
        gpt_source=gpt_source,
        gemini_source=gemini_source,
        contract_artifacts=_contract_artifacts(tmp_path, prompt),
    )
    lock_path = tmp_path / "out" / "protocol_lock.json"
    first = write_protocol_lock_build(build, lock_path)
    second = write_protocol_lock_build(build, lock_path)
    assert first == second
    assert first.lock_path.is_file()
    assert first.identity_path.is_file()
    lock_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolLockBuildError, match="refusing to overwrite"):
        write_protocol_lock_build(build, lock_path)


def test_contract_artifact_rejects_credentials() -> None:
    with pytest.raises(ProtocolLockBuildError, match="credential"):
        protocol_contract_artifact_document(
            "task_instruction",
            {"instruction": "use sk-abcdefghijklmnop for this run"},
        )


def test_cli_requires_existing_canonical_input_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = str(tmp_path / "missing")
    args = [
        "build-protocol-lock",
        "--input",
        path,
        "--output",
        path,
    ]
    assert main(args) == 2
    error = json.loads(capsys.readouterr().err)
    assert "input manifest" in error["error"]
