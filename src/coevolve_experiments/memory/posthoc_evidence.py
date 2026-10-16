"""Read-only, hash-bound verifier evidence for historical Table-1 memory.

The exact V83 runs predate ``memory/strategy_export_evidence.json``.  This
module reconstructs only the missing *index*: every receipt is derived from an
already-recorded strategy rollout reference and the corresponding immutable
Host ``attempt_*/evaluation.json``.  Nothing is written below a source run.

The resulting directory is an external evidence package.  Its manifest binds
the exact audited five-run cohort, the final online-memory digest already
pinned by each historical run manifest, every strategy document, every raw
Host evaluation, and every generated receipt.  Loading rechecks the entire
chain and fails closed before an exporter can see a strategy record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any

from ..serialization import to_primitive
from .schema import SourceRunValidationError, semantic_sha256
from .source_evidence import (
    LoadedEvidenceReceipt,
    validate_source_evidence_binding,
)


POSTHOC_EVIDENCE_PACKAGE_KIND = "table1_posthoc_strategy_evidence_package"
POSTHOC_EVIDENCE_PACKAGE_SCHEMA_VERSION = 1
POSTHOC_EVIDENCE_MANIFEST = "posthoc_strategy_evidence_manifest.json"

_BINDING_KIND = "strategy_source_evidence_binding"
_BINDING_SCHEMA_VERSION = 2
_RECEIPT_KIND = "strategy_source_verifier_receipt"
_RECEIPT_SCHEMA_VERSION = 2
_MEMORY_FILENAMES = (
    "accepted_episode_manifest.json",
    "action_exemplars.json",
    "raw_rollouts.jsonl",
    "strategy_specs.json",
)
_PACKAGE_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "cohort",
        "aggregate_manifest",
        "runs",
        "package_payload_sha256",
    }
)
_AGGREGATE_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "internal_sha256",
        "kind",
        "provider",
        "provider_only_companion",
        "parent_result_manifest_hash",
    }
)
_RUN_FIELDS = frozenset(
    {
        "seed",
        "run_index",
        "source_run_path",
        "source_run_manifest_sha256",
        "source_metrics_sha256",
        "strategy_specs_sha256",
        "final_online_memory_sha256",
        "memory_artifact_hashes",
        "evidence_binding",
        "evidence_receipts",
    }
)
_ARTIFACT_FIELDS = frozenset({"relpath", "sha256"})
_RECEIPT_INDEX_FIELDS = frozenset(
    {"virtual_relpath", "package_relpath", "sha256", "host_evidence_sha256"}
)
_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class PosthocRunOverlay:
    """External binding/receipt paths for one unchanged source directory."""

    seed: int
    source_directory: Path
    evidence_binding: Path
    evidence_binding_sha256: str
    evidence_receipts: Mapping[str, Path]


@dataclass(frozen=True)
class PosthocEvidencePackage:
    """A fully revalidated external evidence package."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    payload_sha256: str
    runs: tuple[PosthocRunOverlay, ...]


def produce_posthoc_source_evidence(
    source_runs: Sequence[str | Path],
    cohort_audit: object,
    output_dir: str | Path,
) -> PosthocEvidencePackage:
    """Build a new package without mutating any historical source artifact."""

    audit = _require_cohort_audit(cohort_audit)
    audited_paths = _require_exact_source_paths(source_runs, audit)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise SourceRunValidationError(
            f"refusing to overwrite post-hoc evidence output: {destination}"
        )
    destination_parent = destination.parent.resolve()
    for source in audited_paths:
        if destination_parent == source or destination_parent.is_relative_to(source):
            raise SourceRunValidationError(
                "post-hoc evidence output must be outside every historical source run"
            )

    run_documents: list[dict[str, Any]] = []
    files_to_write: dict[str, object] = {}
    for run, source in zip(audit.runs, audited_paths):
        run_document, generated = _derive_run_evidence(source, run)
        run_documents.append(run_document)
        overlap = set(files_to_write).intersection(generated)
        if overlap:
            raise SourceRunValidationError(
                f"post-hoc package generated duplicate paths: {sorted(overlap)}"
            )
        files_to_write.update(generated)

    manifest: dict[str, Any] = {
        "kind": POSTHOC_EVIDENCE_PACKAGE_KIND,
        "schema_version": POSTHOC_EVIDENCE_PACKAGE_SCHEMA_VERSION,
        "cohort": to_primitive(audit.source),
        "aggregate_manifest": _aggregate_identity(audit),
        "runs": run_documents,
    }
    manifest["package_payload_sha256"] = semantic_sha256(manifest)
    files_to_write[POSTHOC_EVIDENCE_MANIFEST] = manifest

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        for relpath, value in sorted(files_to_write.items()):
            _write_json(temporary / relpath, value)
        # Validate the complete package before publishing the destination.
        # This also catches malformed historical rollout refs produced by an
        # older writer without leaving a poisoned, non-overwritable output.
        load_posthoc_source_evidence(temporary, audit)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_posthoc_source_evidence(destination, audit)


def load_posthoc_source_evidence(
    package_dir: str | Path,
    cohort_audit: object,
) -> PosthocEvidencePackage:
    """Revalidate a package against a freshly audited exact Table-1 cohort."""

    audit = _require_cohort_audit(cohort_audit)
    root = Path(package_dir)
    if root.is_symlink() or not root.is_dir():
        raise SourceRunValidationError(
            f"post-hoc evidence package is not a non-symlink directory: {root}"
        )
    root = root.resolve(strict=True)
    manifest_path = root / POSTHOC_EVIDENCE_MANIFEST
    manifest, manifest_sha256 = _read_json_object(
        manifest_path,
        root=root,
        label="post-hoc evidence manifest",
    )
    if set(manifest) != _PACKAGE_FIELDS:
        raise SourceRunValidationError(
            "post-hoc evidence manifest fields differ: "
            f"missing={sorted(_PACKAGE_FIELDS - set(manifest))}, "
            f"unknown={sorted(set(manifest) - _PACKAGE_FIELDS)}"
        )
    if manifest.get("kind") != POSTHOC_EVIDENCE_PACKAGE_KIND:
        raise SourceRunValidationError("post-hoc evidence package kind is invalid")
    if manifest.get("schema_version") != POSTHOC_EVIDENCE_PACKAGE_SCHEMA_VERSION:
        raise SourceRunValidationError("post-hoc evidence package schema is invalid")
    payload_hash = manifest.get("package_payload_sha256")
    unsigned = dict(manifest)
    unsigned.pop("package_payload_sha256", None)
    if not _is_sha256(payload_hash) or payload_hash != semantic_sha256(unsigned):
        raise SourceRunValidationError("post-hoc evidence package self-hash mismatch")
    if manifest.get("cohort") != to_primitive(audit.source):
        raise SourceRunValidationError(
            "post-hoc evidence package differs from the audited Table-1 source identity"
        )
    if manifest.get("aggregate_manifest") != _aggregate_identity(audit):
        raise SourceRunValidationError(
            "post-hoc evidence package differs from the audited aggregate manifest"
        )

    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 5:
        raise SourceRunValidationError(
            "post-hoc evidence package must contain exactly five run entries"
        )
    overlays: list[PosthocRunOverlay] = []
    expected_files = {POSTHOC_EVIDENCE_MANIFEST}
    for index, (raw_run, audited_run) in enumerate(zip(raw_runs, audit.runs)):
        overlay, referenced = _validate_run_entry(
            root,
            raw_run,
            audited_run,
            source_index=index,
        )
        overlays.append(overlay)
        expected_files.update(referenced)

    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SourceRunValidationError(
                "post-hoc evidence package must not contain symlinks"
            )
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise SourceRunValidationError(
            "post-hoc evidence package file inventory differs: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unknown={sorted(actual_files - expected_files)}"
        )
    return PosthocEvidencePackage(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        payload_sha256=str(payload_hash),
        runs=tuple(overlays),
    )


def _derive_run_evidence(source: Path, audited_run: object) -> tuple[dict[str, Any], dict[str, object]]:
    seed = int(getattr(audited_run, "seed"))
    run_index = int(getattr(audited_run, "run_index"))
    strategy_path = source / "memory" / "strategy_specs.json"
    metrics_path = source / "metrics.json"
    manifest_path = source / "run_manifest.json"
    strategy, strategy_hash = _read_json_object(
        strategy_path, root=source, label="historical strategy specs"
    )
    metrics, metrics_hash = _read_json_object(
        metrics_path, root=source, label="historical metrics"
    )
    run_manifest, run_manifest_hash = _read_json_object(
        manifest_path, root=source, label="historical run manifest"
    )
    declared_artifacts = getattr(audited_run, "artifact_hashes")
    if run_manifest_hash != declared_artifacts.get("run_manifest.json"):
        raise SourceRunValidationError(
            f"seed {seed}: historical run manifest differs from aggregate audit"
        )
    if metrics_hash != declared_artifacts.get("metrics.json"):
        raise SourceRunValidationError(
            f"seed {seed}: historical metrics differ from aggregate audit"
        )
    memory_hashes, final_memory_hash = _validate_final_online_memory(
        source, run_manifest, seed=seed
    )

    specs = strategy.get("specs")
    if not isinstance(specs, list) or not all(isinstance(item, Mapping) for item in specs):
        raise SourceRunValidationError(
            f"seed {seed}: historical strategy_specs.specs must be an object array"
        )
    spec_digests = [semantic_sha256(dict(spec)) for spec in specs]
    if len(spec_digests) != len(set(spec_digests)):
        raise SourceRunValidationError(
            f"seed {seed}: historical strategy records are not semantically unique"
        )

    binding_records: list[dict[str, Any]] = []
    receipt_index: list[dict[str, Any]] = []
    generated: dict[str, object] = {}
    seen_attempts: set[int] = set()
    for spec_index, (spec_value, spec_digest) in enumerate(zip(specs, spec_digests)):
        spec = dict(spec_value)
        evidence = spec.get("evidence")
        if not isinstance(evidence, Mapping) or evidence.get("source") != "physics_verifier":
            raise SourceRunValidationError(
                f"seed {seed} strategy[{spec_index}] lacks physics-verifier evidence"
            )
        accepted = _nonnegative_int(
            evidence.get("accepted_count"),
            f"seed {seed} strategy[{spec_index}] accepted_count",
        )
        rejected = _nonnegative_int(
            evidence.get("rejected_count"),
            f"seed {seed} strategy[{spec_index}] rejected_count",
        )
        if (accepted > 0) == (rejected > 0):
            raise SourceRunValidationError(
                f"seed {seed} strategy[{spec_index}] must have exactly one verified outcome"
            )
        outcome = "accepted" if accepted else "rejected"
        count = accepted + rejected
        refs = evidence.get("rollout_refs")
        keys = evidence.get("rollout_keys")
        if not isinstance(refs, list) or not isinstance(keys, list):
            raise SourceRunValidationError(
                f"seed {seed} strategy[{spec_index}] rollout evidence must be arrays"
            )
        if len(refs) != count or len(keys) != count:
            raise SourceRunValidationError(
                f"seed {seed} strategy[{spec_index}] evidence counts disagree"
            )
        receipt_refs: list[dict[str, str]] = []
        for receipt_index_value, (ref_value, key_value) in enumerate(zip(refs, keys)):
            if not isinstance(ref_value, Mapping):
                raise SourceRunValidationError(
                    f"seed {seed} strategy[{spec_index}] rollout_ref is not an object"
                )
            ref = dict(ref_value)
            attempt = _nonnegative_int(
                ref.get("attempt"),
                f"seed {seed} strategy[{spec_index}] rollout attempt",
            )
            if attempt in seen_attempts:
                raise SourceRunValidationError(
                    f"seed {seed}: one physical attempt is reused by multiple strategy receipts"
                )
            seen_attempts.add(attempt)
            evaluation_relpath = f"attempt_{attempt:02d}/evaluation.json"
            evaluation_path = source.joinpath(*PurePosixPath(evaluation_relpath).parts)
            evaluation, evaluation_hash = _read_json_object(
                evaluation_path,
                root=source,
                label=f"seed {seed} Host evaluation attempt {attempt}",
            )
            evaluation_size = evaluation_path.stat().st_size
            raw_outcome = _verified_outcome(evaluation, outcome=outcome, seed=seed, attempt=attempt)
            if ref.get("outcome") != raw_outcome:
                raise SourceRunValidationError(
                    f"seed {seed} attempt {attempt}: rollout outcome contradicts Host evaluation"
                )
            if not isinstance(key_value, str) or key_value != _rollout_key(ref, evaluation):
                raise SourceRunValidationError(
                    f"seed {seed} attempt {attempt}: rollout key contradicts Host metrics"
                )
            receipt = {
                "kind": _RECEIPT_KIND,
                "schema_version": _RECEIPT_SCHEMA_VERSION,
                "source_record_sha256": spec_digest,
                "verified_outcome": outcome,
                "rollout_ref": ref,
                "rollout_key": key_value,
                "host_evidence": {
                    "relative_path": evaluation_relpath,
                    "sha256": evaluation_hash,
                    "size_bytes": evaluation_size,
                },
            }
            virtual_relpath = (
                "memory/strategy_export_receipts/"
                f"{spec_digest}_{receipt_index_value:03d}.json"
            )
            package_relpath = (
                f"runs/seed_{seed:03d}/receipts/"
                f"{spec_digest}_{receipt_index_value:03d}.json"
            )
            receipt_hash = _json_file_hash(receipt)
            generated[package_relpath] = receipt
            receipt_refs.append(
                {
                    "relpath": virtual_relpath,
                    "sha256": receipt_hash,
                    "host_evidence_sha256": evaluation_hash,
                }
            )
            receipt_index.append(
                {
                    "virtual_relpath": virtual_relpath,
                    "package_relpath": package_relpath,
                    "sha256": receipt_hash,
                    "host_evidence_sha256": evaluation_hash,
                }
            )
        binding_records.append(
            {
                "source_record_sha256": spec_digest,
                "verified_outcome": outcome,
                "evidence_count": count,
                "evidence_receipts": receipt_refs,
            }
        )

    binding = {
        "kind": _BINDING_KIND,
        "schema_version": _BINDING_SCHEMA_VERSION,
        "strategy_specs_sha256": strategy_hash,
        "metrics_sha256": metrics_hash,
        "records": binding_records,
    }
    binding_relpath = f"runs/seed_{seed:03d}/strategy_export_evidence.json"
    binding_hash = _json_file_hash(binding)
    generated[binding_relpath] = binding
    run_document = {
        "seed": seed,
        "run_index": run_index,
        "source_run_path": str(source),
        "source_run_manifest_sha256": run_manifest_hash,
        "source_metrics_sha256": metrics_hash,
        "strategy_specs_sha256": strategy_hash,
        "final_online_memory_sha256": final_memory_hash,
        "memory_artifact_hashes": memory_hashes,
        "evidence_binding": {"relpath": binding_relpath, "sha256": binding_hash},
        "evidence_receipts": receipt_index,
    }
    return run_document, generated


def _validate_run_entry(
    root: Path,
    raw_run: object,
    audited_run: object,
    *,
    source_index: int,
) -> tuple[PosthocRunOverlay, set[str]]:
    if not isinstance(raw_run, Mapping) or set(raw_run) != _RUN_FIELDS:
        raise SourceRunValidationError(
            f"post-hoc evidence runs[{source_index}] fields are invalid"
        )
    seed = int(getattr(audited_run, "seed"))
    run_index = int(getattr(audited_run, "run_index"))
    if raw_run.get("seed") != seed or raw_run.get("run_index") != run_index:
        raise SourceRunValidationError(
            f"post-hoc evidence runs[{source_index}] seed/index differs from audit"
        )
    source = Path(str(raw_run.get("source_run_path", "")))
    audited_source = Path(getattr(audited_run, "run_dir")).resolve(strict=True)
    if not source.is_absolute() or source.resolve(strict=True) != audited_source:
        raise SourceRunValidationError(
            f"post-hoc evidence seed {seed} source path differs from aggregate audit"
        )
    source = audited_source
    manifest, manifest_hash = _read_json_object(
        source / "run_manifest.json", root=source, label=f"seed {seed} run manifest"
    )
    metrics, metrics_hash = _read_json_object(
        source / "metrics.json", root=source, label=f"seed {seed} metrics"
    )
    strategy, strategy_hash = _read_json_object(
        source / "memory/strategy_specs.json",
        root=source,
        label=f"seed {seed} strategy specs",
    )
    declared = getattr(audited_run, "artifact_hashes")
    expected_scalars = {
        "source_run_manifest_sha256": manifest_hash,
        "source_metrics_sha256": metrics_hash,
        "strategy_specs_sha256": strategy_hash,
    }
    if manifest_hash != declared.get("run_manifest.json") or metrics_hash != declared.get("metrics.json"):
        raise SourceRunValidationError(
            f"post-hoc evidence seed {seed} source artifacts differ from aggregate audit"
        )
    for field, expected in expected_scalars.items():
        if raw_run.get(field) != expected:
            raise SourceRunValidationError(
                f"post-hoc evidence seed {seed} {field} mismatch"
            )
    memory_hashes, final_memory_hash = _validate_final_online_memory(
        source, manifest, seed=seed
    )
    if raw_run.get("memory_artifact_hashes") != memory_hashes or raw_run.get(
        "final_online_memory_sha256"
    ) != final_memory_hash:
        raise SourceRunValidationError(
            f"post-hoc evidence seed {seed} final online-memory binding mismatch"
        )

    binding_ref = raw_run.get("evidence_binding")
    if not isinstance(binding_ref, Mapping) or set(binding_ref) != _ARTIFACT_FIELDS:
        raise SourceRunValidationError(
            f"post-hoc evidence seed {seed} binding reference is invalid"
        )
    binding_relpath = _safe_package_relpath(binding_ref.get("relpath"))
    expected_binding_relpath = (
        f"runs/seed_{seed:03d}/strategy_export_evidence.json"
    )
    if binding_relpath != expected_binding_relpath:
        raise SourceRunValidationError(
            f"post-hoc evidence seed {seed} binding path is not canonical"
        )
    binding, binding_hash = _read_json_object(
        root.joinpath(*PurePosixPath(binding_relpath).parts),
        root=root,
        label=f"seed {seed} external evidence binding",
    )
    if binding_ref.get("sha256") != binding_hash:
        raise SourceRunValidationError(
            f"post-hoc evidence seed {seed} binding byte hash mismatch"
        )

    raw_receipts = raw_run.get("evidence_receipts")
    if not isinstance(raw_receipts, list):
        raise SourceRunValidationError(
            f"post-hoc evidence seed {seed} receipt index must be an array"
        )
    receipt_paths: dict[str, Path] = {}
    loaded_receipts: dict[str, LoadedEvidenceReceipt] = {}
    referenced = {binding_relpath}
    seen_host_hashes: set[str] = set()
    for receipt_index, raw_receipt in enumerate(raw_receipts):
        if not isinstance(raw_receipt, Mapping) or set(raw_receipt) != _RECEIPT_INDEX_FIELDS:
            raise SourceRunValidationError(
                f"post-hoc evidence seed {seed} receipt[{receipt_index}] is invalid"
            )
        virtual = _safe_virtual_receipt_relpath(raw_receipt.get("virtual_relpath"))
        package_relpath = _safe_package_relpath(raw_receipt.get("package_relpath"))
        expected_package_relpath = (
            f"runs/seed_{seed:03d}/receipts/"
            f"{PurePosixPath(virtual).name}"
        )
        if package_relpath != expected_package_relpath:
            raise SourceRunValidationError(
                f"post-hoc evidence seed {seed} receipt path is not canonical"
            )
        if virtual in receipt_paths:
            raise SourceRunValidationError(
                f"post-hoc evidence seed {seed} reuses a virtual receipt path"
            )
        receipt_path = root.joinpath(*PurePosixPath(package_relpath).parts)
        receipt, receipt_hash = _read_json_object(
            receipt_path,
            root=root,
            label=f"seed {seed} external evidence receipt",
        )
        if raw_receipt.get("sha256") != receipt_hash:
            raise SourceRunValidationError(
                f"post-hoc evidence seed {seed} receipt byte hash mismatch"
            )
        host_ref = receipt.get("host_evidence")
        if not isinstance(host_ref, Mapping):
            raise SourceRunValidationError(
                f"post-hoc evidence seed {seed} receipt lacks Host evidence"
            )
        host_relpath = _safe_host_evidence_relpath(host_ref.get("relative_path"))
        host_path = source.joinpath(*PurePosixPath(host_relpath).parts)
        host_value, host_hash = _read_json_object(
            host_path, root=source, label=f"seed {seed} raw Host evaluation"
        )
        host_size = host_path.stat().st_size
        if (
            raw_receipt.get("host_evidence_sha256") != host_hash
            or host_ref.get("sha256") != host_hash
            or host_ref.get("size_bytes") != host_size
        ):
            raise SourceRunValidationError(
                f"post-hoc evidence seed {seed} Host evaluation binding mismatch"
            )
        if host_hash in seen_host_hashes:
            raise SourceRunValidationError(
                f"post-hoc evidence seed {seed} reuses Host evaluation evidence"
            )
        seen_host_hashes.add(host_hash)
        receipt_paths[virtual] = receipt_path
        loaded_receipts[virtual] = LoadedEvidenceReceipt(
            value=receipt,
            sha256=receipt_hash,
            host_evidence_value=host_value,
            host_evidence_sha256=host_hash,
            host_evidence_size_bytes=host_size,
            host_evidence_relative_path=host_relpath,
        )
        referenced.add(package_relpath)

    validation = validate_source_evidence_binding(
        source_index=source_index,
        binding=binding,
        binding_sha256=binding_hash,
        strategy_specs=strategy,
        strategy_specs_sha256=strategy_hash,
        metrics=metrics,
        metrics_sha256=metrics_hash,
        manifest=manifest,
        audit={"strategy_export_evidence_sha256": binding_hash},
        evidence_receipts=loaded_receipts,
    )
    if validation.errors:
        raise SourceRunValidationError("; ".join(validation.errors))
    return (
        PosthocRunOverlay(
            seed=seed,
            source_directory=source,
            evidence_binding=root.joinpath(*PurePosixPath(binding_relpath).parts),
            evidence_binding_sha256=binding_hash,
            evidence_receipts=receipt_paths,
        ),
        referenced,
    )


def _validate_final_online_memory(
    source: Path,
    run_manifest: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, str], str]:
    memory_hashes: dict[str, str] = {}
    for filename in _MEMORY_FILENAMES:
        path = source / "memory" / filename
        raw = _read_bytes(path, root=source, label=f"seed {seed} memory/{filename}")
        memory_hashes[filename] = hashlib.sha256(raw).hexdigest()
    final_hash = semantic_sha256(memory_hashes)
    artifacts = run_manifest.get("artifact_hashes")
    expected = (
        artifacts.get("final_online_memory_sha256")
        if isinstance(artifacts, Mapping)
        else None
    )
    if not _is_sha256(expected) or expected != final_hash:
        raise SourceRunValidationError(
            f"seed {seed}: final online-memory digest does not match the aggregate-pinned run manifest"
        )
    return memory_hashes, final_hash


def _verified_outcome(
    evaluation: Mapping[str, Any],
    *,
    outcome: str,
    seed: int,
    attempt: int,
) -> str:
    success = evaluation.get("success")
    failure = str(evaluation.get("failure_type") or "").strip()
    if outcome == "accepted":
        if success is not True or failure not in {"", "none"}:
            raise SourceRunValidationError(
                f"seed {seed} attempt {attempt}: accepted strategy contradicts Host evaluation"
            )
        return "accepted"
    if success is not False or not failure or failure == "none":
        raise SourceRunValidationError(
            f"seed {seed} attempt {attempt}: rejected strategy contradicts Host evaluation"
        )
    return failure


def _rollout_key(ref: Mapping[str, Any], evaluation: Mapping[str, Any]) -> str:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SourceRunValidationError("raw Host evaluation lacks verifier metrics")
    failure = str(evaluation.get("failure_type") or "").strip()
    outcome = "accepted" if evaluation.get("success") is True else failure
    fingerprint = {
        "script_id": ref.get("script_id"),
        "attempt": ref.get("attempt"),
        "checkpoint_before": ref.get("checkpoint_before"),
        "checkpoint_after": ref.get("checkpoint_after"),
        "outcome": outcome,
        "metrics": dict(metrics),
    }
    return semantic_sha256(fingerprint)


def _aggregate_identity(audit: object) -> dict[str, Any]:
    return {
        "path": str(Path(getattr(audit, "manifest_path")).resolve(strict=True)),
        "sha256": getattr(audit, "manifest_sha256"),
        "internal_sha256": getattr(audit, "manifest_internal_sha256"),
        "kind": getattr(audit, "manifest_kind"),
        "provider": getattr(audit, "provider"),
        "provider_only_companion": getattr(audit, "provider_only_companion"),
        "parent_result_manifest_hash": getattr(audit, "parent_result_manifest_hash"),
    }


def _require_cohort_audit(value: object) -> object:
    from ..reporting.table1_source import Table1CohortAudit

    if not isinstance(value, Table1CohortAudit) or not value.protocol_valid:
        raise SourceRunValidationError(
            "post-hoc evidence requires a complete Table1CohortAudit"
        )
    if len(value.runs) != 5:
        raise SourceRunValidationError(
            "post-hoc evidence requires exactly five audited source runs"
        )
    return value


def _require_exact_source_paths(
    source_runs: Sequence[str | Path],
    audit: object,
) -> tuple[Path, ...]:
    if len(source_runs) != 5:
        raise SourceRunValidationError(
            "post-hoc evidence requires exactly five source-run paths"
        )
    supplied = tuple(Path(value).resolve(strict=True) for value in source_runs)
    if len(set(supplied)) != 5:
        raise SourceRunValidationError("post-hoc source-run paths must be unique")
    audited = tuple(
        Path(getattr(run, "run_dir")).resolve(strict=True)
        for run in getattr(audit, "runs")
    )
    if set(supplied) != set(audited):
        raise SourceRunValidationError(
            "post-hoc source-run paths differ from the audited Table-1 cohort"
        )
    return audited


def _read_json_object(path: Path, *, root: Path, label: str) -> tuple[dict[str, Any], str]:
    raw = _read_bytes(path, root=root, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceRunValidationError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SourceRunValidationError(f"{label} must contain one JSON object")
    return dict(value), hashlib.sha256(raw).hexdigest()


def _read_bytes(path: Path, *, root: Path, label: str) -> bytes:
    root = root.resolve(strict=True)
    if path.is_symlink():
        raise SourceRunValidationError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SourceRunValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not resolved.is_relative_to(root):
        raise SourceRunValidationError(f"{label} escapes its audited root")
    cursor = root
    for part in resolved.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SourceRunValidationError(f"{label} path contains a symlink")
    if not resolved.is_file():
        raise SourceRunValidationError(f"{label} is not a regular file")
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise SourceRunValidationError(f"cannot read {label} {resolved}: {exc}") from exc


def _safe_package_relpath(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SourceRunValidationError("post-hoc package relative path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix != ".json"
    ):
        raise SourceRunValidationError("post-hoc package relative path is invalid")
    return value


def _safe_virtual_receipt_relpath(value: object) -> str:
    relpath = _safe_package_relpath(value)
    path = PurePosixPath(relpath)
    if path.parent != PurePosixPath("memory/strategy_export_receipts"):
        raise SourceRunValidationError("post-hoc virtual receipt path is invalid")
    return relpath


def _safe_host_evidence_relpath(value: object) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise SourceRunValidationError("post-hoc Host evidence path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or not path.parts[0].startswith("attempt_")
        or not path.parts[0].removeprefix("attempt_").isdigit()
        or path.parts[1] != "evaluation.json"
        or path.as_posix() != value
    ):
        raise SourceRunValidationError("post-hoc Host evidence path is invalid")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceRunValidationError(f"{label} must be a non-negative integer")
    return value


def _json_file_hash(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    path.write_text(rendered, encoding="utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


__all__ = [
    "POSTHOC_EVIDENCE_MANIFEST",
    "POSTHOC_EVIDENCE_PACKAGE_KIND",
    "POSTHOC_EVIDENCE_PACKAGE_SCHEMA_VERSION",
    "PosthocEvidencePackage",
    "PosthocRunOverlay",
    "load_posthoc_source_evidence",
    "produce_posthoc_source_evidence",
]
