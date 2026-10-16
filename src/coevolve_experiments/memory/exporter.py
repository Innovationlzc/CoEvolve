"""Deterministic export of verifier-backed source memories for Table 4."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from ..provider_routes import require_provider_route
from ..serialization import strict_json_loads
from .candidate_extraction import (
    StrategyCandidate,
    candidate_from_spec,
    extract_strategy_specs,
)
from .ranking import deduplicate_and_cap
from .schema import (
    FROZEN_STRATEGY_BANK_POLICY,
    STRATEGY_BANK_SCHEMA_VERSION,
    SourceRunValidationError,
    StrategyBankError,
    semantic_sha256,
)
from .provenance import (
    STRATEGY_TRANSFER_PROVENANCE_SCHEMA_VERSION,
)
from .source_audit import (
    normalize_source_scene,
    require_unique_source_manifests,
    resolve_source_identity,
    stable_manifest_sha256,
)
from .source_evidence import (
    BoundSourceEvidence,
    LoadedEvidenceReceipt,
    validate_source_evidence_binding,
)
from .validation import validate_strategy_bank


JsonSource = object


@dataclass(frozen=True)
class SourceRun:
    """Auditable source documents and host-verifier receipts for one run."""

    strategy_specs: JsonSource
    metrics: JsonSource
    manifest: JsonSource
    audit: Mapping[str, Any] = field(default_factory=dict)
    evidence_binding: JsonSource | None = None
    evidence_receipts: Mapping[str, JsonSource] = field(default_factory=dict)
    source_directory: str | Path | None = field(default=None, repr=False)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        audit: Mapping[str, Any] | None = None,
    ) -> "SourceRun":
        root = Path(directory)
        return cls(
            strategy_specs=root / "memory" / "strategy_specs.json",
            metrics=root / "metrics.json",
            manifest=root / "run_manifest.json",
            audit=audit or {},
            evidence_binding=root / "memory" / "strategy_export_evidence.json",
            source_directory=root,
        )


@dataclass(frozen=True)
class StrategyBankExport(Mapping[str, Any]):
    """Export result which also behaves like the bank mapping for convenience."""

    bank: dict[str, Any]
    provenance: dict[str, Any]
    bank_path: Path | None = None
    provenance_path: Path | None = None

    def __getitem__(self, key: str) -> Any:
        return self.bank[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.bank)

    def __len__(self) -> int:
        return len(self.bank)


@dataclass
class _LoadedJson:
    value: Any
    source_path: str
    sha256: str
    size_bytes: int


@dataclass
class _LoadedRun:
    strategy: _LoadedJson
    metrics: _LoadedJson
    manifest: _LoadedJson
    evidence_binding: _LoadedJson | None
    evidence_receipts: dict[str, LoadedEvidenceReceipt]
    audit: dict[str, Any]
    model: str = ""
    scene: str = ""
    protocol: str = ""
    seed: str = ""
    method_commit: str = ""
    condition_fingerprint: str = ""
    metadata_conflicts: tuple[str, ...] = ()
    evidence_binding_pin_source: str = ""
    bound_evidence: dict[str, BoundSourceEvidence] = field(default_factory=dict)


def export_strategy_bank(
    source_runs: Sequence[object],
    bank_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
    audit_metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    expected_source_model: str | None = None,
    expected_protocol: str | None = None,
    expected_source_seeds: Sequence[int] | None = None,
    expected_source_method_commit: str | None = None,
    expected_source_condition_fingerprint: str | None = None,
    expected_source_result_manifest_hash: str | None = None,
    expected_source_provider_route_id: str | None = None,
    expected_source_provider_route_fingerprint: str | None = None,
    expected_bank_sha256: str | None = None,
    expected_scene: str = "base",
) -> StrategyBankExport:
    """Export one frozen, scene-sanitized bank from exactly five source runs.

    ``source_runs`` accepts :class:`SourceRun`, run directories, three/four-item
    tuples, or mappings containing ``strategy_specs``, ``metrics``,
    ``manifest``, and the required ``evidence_binding``.  ``audit_metadata``
    may fill fields absent from older manifests but cannot override them.  In
    particular, embedded strategy-spec evidence never bypasses the independent
    manifest-pinned per-record verifier binding.
    """

    if output_path is not None:
        if bank_path is not None and Path(output_path) != Path(bank_path):
            raise StrategyBankError("bank_path and output_path disagree")
        bank_path = output_path

    required = FROZEN_STRATEGY_BANK_POLICY.required_source_runs
    if len(source_runs) != required:
        raise SourceRunValidationError(
            f"Table 4 export requires exactly {required} source runs; "
            f"received {len(source_runs)}"
        )

    full_source_identity = _validate_complete_source_identity_arguments(
        expected_source_model=expected_source_model,
        expected_protocol=expected_protocol,
        expected_source_seeds=expected_source_seeds,
        expected_source_method_commit=expected_source_method_commit,
        expected_source_condition_fingerprint=(
            expected_source_condition_fingerprint
        ),
        expected_source_result_manifest_hash=expected_source_result_manifest_hash,
        expected_source_provider_route_id=expected_source_provider_route_id,
        expected_source_provider_route_fingerprint=(
            expected_source_provider_route_fingerprint
        ),
    )
    audits = _normalize_audit_metadata(audit_metadata, len(source_runs))
    loaded = [
        _load_run(descriptor, override_audit=audits[index])
        for index, descriptor in enumerate(source_runs)
    ]
    _validate_source_cohort(
        loaded,
        expected_model=expected_source_model,
        expected_protocol=expected_protocol,
        expected_seeds=expected_source_seeds,
        expected_method_commit=expected_source_method_commit,
        expected_condition_fingerprint=expected_source_condition_fingerprint,
        expected_provider_route_id=expected_source_provider_route_id,
        expected_provider_route_fingerprint=(
            expected_source_provider_route_fingerprint
        ),
        expected_scene=expected_scene,
    )

    per_run_audit: list[dict[str, Any]] = []
    candidates: list[StrategyCandidate] = []
    for run_index, run in enumerate(loaded):
        run_audit = _source_sidecar_entry(run, run_index)
        specs = extract_strategy_specs(run.strategy.value)
        for record_index, spec in enumerate(specs):
            source_id = str(spec.get("id", "")).strip()
            candidate, reason = candidate_from_spec(
                spec,
                source_index=run_index,
                source_record_index=record_index,
                bound_evidence=run.bound_evidence,
            )
            if candidate is None:
                run_audit["dropped"].append(
                    {
                        "source_record_index": record_index,
                        "source_record_id": source_id,
                        "reason": reason,
                    }
                )
                continue
            candidates.append(candidate)
        per_run_audit.append(run_audit)

    selected, dispositions = deduplicate_and_cap(candidates)
    disposition_by_key = {
        (item["source_index"], item["source_record_index"]): item
        for item in dispositions
    }
    for candidate in candidates:
        disposition = disposition_by_key[
            (candidate.source_index, candidate.source_record_index)
        ]
        entry = {
            "source_record_index": candidate.source_record_index,
            "source_record_id": candidate.source_record_id,
            "bank_record_sha256": candidate.digest,
            "reason": disposition["reason"],
        }
        if candidate.sanitization_notes:
            entry["sanitization"] = list(candidate.sanitization_notes)
        bucket = "kept" if disposition["kept"] else "dropped"
        per_run_audit[candidate.source_index][bucket].append(entry)

    records = [candidate.record for candidate in selected]
    records.sort(key=semantic_sha256)
    if not records:
        raise StrategyBankError(
            "no verifier-backed route-abstracted strategy records survived export"
        )
    bank: dict[str, Any] = {
        "schema_version": STRATEGY_BANK_SCHEMA_VERSION,
        "records": records,
    }
    validation = validate_strategy_bank(bank)

    bank_digest = validation.bank_sha256
    if expected_bank_sha256 is not None and bank_digest != expected_bank_sha256:
        raise StrategyBankError(
            "exported strategy bank differs from the selected protocol-lock hash"
        )
    provenance: dict[str, Any] = {
        "kind": "strategy_transfer_provenance",
        "schema_version": (
            STRATEGY_TRANSFER_PROVENANCE_SCHEMA_VERSION
            if full_source_identity
            else 1
        ),
        "bank_sha256": bank_digest,
        "source_model": loaded[0].model,
        "source_scene": loaded[0].scene,
        "source_protocol": loaded[0].protocol,
        "source_seeds": sorted(int(item.seed) for item in loaded),
        "policy": FROZEN_STRATEGY_BANK_POLICY.as_dict(),
        "summary": {
            "source_runs": len(loaded),
            "source_records": sum(
                len(extract_strategy_specs(item.strategy.value)) for item in loaded
            ),
            "kept_records": len(records),
            "dropped_source_records": sum(
                len(item["dropped"]) for item in per_run_audit
            ),
            "rejected_counterexamples_kept": sum(
                record["verified_outcome"] == "rejected" for record in records
            ),
        },
        "sources": per_run_audit,
    }
    if full_source_identity:
        # These values come from the lock-selected source cohort.  Commit and
        # condition are also checked against every source run above; the
        # aggregate result-manifest hash is the immutable cohort anchor.
        provenance.update(
            {
                "source_method_commit": expected_source_method_commit,
                "source_seeds": list(expected_source_seeds or ()),
                "source_condition_fingerprint": (
                    expected_source_condition_fingerprint
                ),
                "source_result_manifest_hash": (
                    expected_source_result_manifest_hash
                ),
                "source_provider_route_id": expected_source_provider_route_id,
                "source_provider_route_fingerprint": (
                    expected_source_provider_route_fingerprint
                ),
            }
        )

    resolved_bank_path: Path | None = None
    resolved_provenance_path: Path | None = None
    if bank_path is not None:
        resolved_bank_path = Path(bank_path)
        if resolved_bank_path.exists() and resolved_bank_path.is_dir():
            resolved_bank_path = resolved_bank_path / "strategy_bank.json"
        if provenance_path is None:
            resolved_provenance_path = resolved_bank_path.with_name(
                f"{resolved_bank_path.stem}.provenance.json"
            )
        else:
            resolved_provenance_path = Path(provenance_path)
        _write_json_idempotent(resolved_bank_path, bank)
        _write_json_idempotent(resolved_provenance_path, provenance)
    elif provenance_path is not None:
        raise StrategyBankError("provenance_path requires bank_path/output_path")

    return StrategyBankExport(
        bank=bank,
        provenance=provenance,
        bank_path=resolved_bank_path,
        provenance_path=resolved_provenance_path,
    )


def export_audited_table1_strategy_bank(
    source_runs: Sequence[str | Path],
    cohort_audit: object,
    *,
    bank_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    expected_bank_sha256: str | None = None,
    posthoc_evidence_package: str | Path | None = None,
) -> StrategyBankExport:
    """Export only from the exact five directories admitted by Table-1 audit.

    This is the paper-facing wrapper.  Unlike :func:`export_strategy_bank`, it
    does not accept caller-authored identity metadata: model, protocol,
    commit, seeds, condition fingerprint, and aggregate-manifest hash are all
    derived from a successfully verified :class:`Table1CohortAudit`.
    """

    # Local import avoids a package-initialization cycle: reporting validates
    # memory artifacts and therefore imports memory submodules itself.
    from ..reporting.table1_source import Table1CohortAudit

    if not isinstance(cohort_audit, Table1CohortAudit):
        raise SourceRunValidationError(
            "formal Table-4 export requires a Table1CohortAudit"
        )
    if not cohort_audit.protocol_valid:
        raise SourceRunValidationError(
            "formal Table-4 export requires a complete five-run Table-1 audit"
        )
    supplied = tuple(Path(value).resolve(strict=True) for value in source_runs)
    if len(supplied) != len(set(supplied)):
        raise SourceRunValidationError("source-run paths must be unique")
    audited = tuple(Path(run.run_dir).resolve(strict=True) for run in cohort_audit.runs)
    if set(supplied) != set(audited):
        missing = sorted(str(path) for path in set(audited) - set(supplied))
        unexpected = sorted(str(path) for path in set(supplied) - set(audited))
        raise SourceRunValidationError(
            "source-run paths differ from the audited Table-1 cohort: "
            f"missing={missing}, unexpected={unexpected}"
        )

    descriptors: Sequence[object] = audited
    package_manifest_sha256 = ""
    package_payload_sha256 = ""
    binding_hashes: dict[int, str] = {}
    if posthoc_evidence_package is not None:
        from .posthoc_evidence import load_posthoc_source_evidence

        package = load_posthoc_source_evidence(
            posthoc_evidence_package,
            cohort_audit,
        )
        package_manifest_sha256 = package.manifest_sha256
        package_payload_sha256 = package.payload_sha256
        overlays = {overlay.seed: overlay for overlay in package.runs}
        descriptors = tuple(
            SourceRun(
                strategy_specs=path / "memory" / "strategy_specs.json",
                metrics=path / "metrics.json",
                manifest=path / "run_manifest.json",
                evidence_binding=overlays[run.seed].evidence_binding,
                evidence_receipts=overlays[run.seed].evidence_receipts,
                source_directory=path,
            )
            for run, path in zip(cohort_audit.runs, audited)
        )
        binding_hashes = {
            overlay.seed: overlay.evidence_binding_sha256
            for overlay in package.runs
        }

    audit_metadata = [
        {
            "kind": "audited_table1_source_run",
            "schema_version": 1,
            "protocol_valid": True,
            "run_valid": True,
            "evidence_valid": True,
            "infrastructure_valid": True,
            "source_model": run.backbone,
            "scene": "base",
            "protocol": run.protocol_id,
            "seed": run.seed,
            "method_commit": run.method_commit,
            "condition_fingerprint": run.condition_fingerprint,
            "physical_budget": 80,
            "aggregate_manifest_path": cohort_audit.manifest_path,
            "aggregate_manifest_sha256": cohort_audit.manifest_sha256,
            "aggregate_manifest_internal_sha256": (
                cohort_audit.manifest_internal_sha256
            ),
            "aggregate_manifest_kind": cohort_audit.manifest_kind,
            "provider": run.provider,
            "provider_route_id": run.provider_route_id,
            "provider_route_fingerprint": run.provider_route_fingerprint,
            "provider_only_companion": cohort_audit.provider_only_companion,
            "audited_result_artifact_hashes": dict(run.artifact_hashes),
            **(
                {
                    "strategy_export_evidence_sha256": binding_hashes[run.seed],
                    "posthoc_evidence_package_manifest_sha256": (
                        package_manifest_sha256
                    ),
                    "posthoc_evidence_package_payload_sha256": (
                        package_payload_sha256
                    ),
                }
                if posthoc_evidence_package is not None
                else {}
            ),
        }
        for run in cohort_audit.runs
    ]
    # Canonical audited seed order, not caller argument order, fixes source
    # indices and makes the provenance sidecar reproducible.
    return export_strategy_bank(
        descriptors,
        bank_path=bank_path,
        provenance_path=provenance_path,
        audit_metadata=audit_metadata,
        expected_source_model=cohort_audit.source.backbone,
        expected_protocol=cohort_audit.source.protocol_id,
        expected_source_seeds=cohort_audit.source.seeds,
        expected_source_method_commit=cohort_audit.source.method_commit,
        expected_source_condition_fingerprint=(
            cohort_audit.source.condition_fingerprint
        ),
        expected_source_result_manifest_hash=cohort_audit.manifest_sha256,
        expected_source_provider_route_id=(
            cohort_audit.source.provider_route_id
        ),
        expected_source_provider_route_fingerprint=(
            cohort_audit.source.provider_route_fingerprint
        ),
        expected_bank_sha256=expected_bank_sha256,
    )


def write_strategy_bank_export(
    result: StrategyBankExport,
    bank_path: str | Path,
    provenance_path: str | Path | None = None,
) -> StrategyBankExport:
    """Persist an already-validated in-memory export without recomputation."""

    if not isinstance(result, StrategyBankExport):
        raise StrategyBankError("result must be a StrategyBankExport")
    resolved_bank = Path(bank_path)
    if resolved_bank.exists() and resolved_bank.is_dir():
        resolved_bank = resolved_bank / "strategy_bank.json"
    resolved_provenance = (
        Path(provenance_path)
        if provenance_path is not None
        else resolved_bank.with_name(f"{resolved_bank.stem}.provenance.json")
    )
    _write_json_idempotent(resolved_bank, result.bank)
    _write_json_idempotent(resolved_provenance, result.provenance)
    return StrategyBankExport(
        bank=result.bank,
        provenance=result.provenance,
        bank_path=resolved_bank,
        provenance_path=resolved_provenance,
    )


def _normalize_audit_metadata(
    value: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    count: int,
) -> list[dict[str, Any]]:
    if value is None:
        return [{} for _ in range(count)]
    if isinstance(value, Mapping):
        if "runs" in value:
            run_values = value["runs"]
            if not isinstance(run_values, Sequence) or isinstance(run_values, str):
                raise SourceRunValidationError("audit_metadata.runs must be a sequence")
            value = run_values  # type: ignore[assignment]
        else:
            # A common explicit audit block is convenient for compact fixtures.
            return [dict(value) for _ in range(count)]
    if len(value) != count:
        raise SourceRunValidationError(
            f"audit metadata must contain {count} entries; received {len(value)}"
        )
    if not all(isinstance(item, Mapping) for item in value):
        raise SourceRunValidationError("each audit metadata entry must be an object")
    return [dict(item) for item in value]


def _coerce_source_run(descriptor: object) -> SourceRun:
    if isinstance(descriptor, SourceRun):
        return descriptor
    if isinstance(descriptor, (str, Path)):
        return SourceRun.from_directory(descriptor)
    if isinstance(descriptor, Mapping):
        def pick(*names: str) -> object:
            for name in names:
                if name in descriptor:
                    return descriptor[name]
            raise SourceRunValidationError(
                f"source run mapping is missing one of {names}"
            )

        return SourceRun(
            strategy_specs=pick("strategy_specs", "strategy_specs_path", "strategy"),
            metrics=pick("metrics", "metrics_path"),
            manifest=pick("manifest", "manifest_path", "run_manifest"),
            audit=dict(
                descriptor.get("audit", descriptor.get("audit_metadata", {})) or {}
            ),
            evidence_binding=descriptor.get(
                "evidence_binding",
                descriptor.get("evidence_binding_path"),
            ),
            evidence_receipts=dict(descriptor.get("evidence_receipts", {}) or {}),
            source_directory=descriptor.get("source_directory"),
        )
    if isinstance(descriptor, Sequence) and not isinstance(descriptor, str):
        if len(descriptor) not in (3, 4):
            raise SourceRunValidationError(
                "source run tuple must be (strategy_specs, metrics, manifest[, "
                "evidence_binding])"
            )
        return SourceRun(
            descriptor[0],
            descriptor[1],
            descriptor[2],
            evidence_binding=descriptor[3] if len(descriptor) == 4 else None,
        )
    raise SourceRunValidationError(
        f"unsupported source run descriptor: {type(descriptor).__name__}"
    )


def _load_run(descriptor: object, *, override_audit: Mapping[str, Any]) -> _LoadedRun:
    source = _coerce_source_run(descriptor)
    audit = dict(source.audit)
    audit.update(override_audit)
    binding = _load_optional_json(
        source.evidence_binding,
        "strategy_export_evidence",
    )
    return _LoadedRun(
        strategy=_load_json(source.strategy_specs, "strategy_specs"),
        metrics=_load_json(source.metrics, "metrics"),
        manifest=_load_json(source.manifest, "manifest"),
        evidence_binding=binding,
        evidence_receipts=_load_evidence_receipts(source, binding),
        audit=audit,
    )


def _load_evidence_receipts(
    source: SourceRun,
    binding: _LoadedJson | None,
) -> dict[str, LoadedEvidenceReceipt]:
    if binding is None or not isinstance(binding.value, Mapping):
        return {}
    records = binding.value.get("records")
    if not isinstance(records, list):
        return {}
    relpaths: set[str] = set()
    for record in records:
        refs = record.get("evidence_receipts") if isinstance(record, Mapping) else None
        if not isinstance(refs, list):
            continue
        for reference in refs:
            relpath = reference.get("relpath") if isinstance(reference, Mapping) else None
            if isinstance(relpath, str) and _safe_receipt_relpath(relpath):
                relpaths.add(relpath)

    loaded: dict[str, LoadedEvidenceReceipt] = {}
    root = Path(source.source_directory) if source.source_directory is not None else None
    for relpath in sorted(relpaths):
        value = source.evidence_receipts.get(relpath)
        if value is not None:
            document = _load_json(value, f"evidence receipt {relpath}")
        elif root is not None:
            candidate = root.joinpath(*PurePosixPath(relpath).parts)
            if not candidate.exists():
                continue
            _require_nonsymlink_descendant(root, candidate)
            document = _load_json(candidate, f"evidence receipt {relpath}")
        else:
            continue
        loaded[relpath] = LoadedEvidenceReceipt(
            value=document.value,
            sha256=document.sha256,
            **_load_receipt_host_evidence(source, document),
        )
    return loaded


def _load_receipt_host_evidence(
    source: SourceRun,
    receipt: _LoadedJson,
) -> dict[str, object]:
    value = receipt.value
    reference = value.get("host_evidence") if isinstance(value, Mapping) else None
    if not isinstance(reference, Mapping):
        return {}
    relpath = reference.get("relative_path")
    if not isinstance(relpath, str) or not _safe_host_evidence_relpath(relpath):
        return {}
    raw_value = source.evidence_receipts.get(relpath)
    root = Path(source.source_directory) if source.source_directory is not None else None
    if raw_value is not None:
        document = _load_json(raw_value, f"raw Host evidence {relpath}")
    elif root is not None:
        candidate = root.joinpath(*PurePosixPath(relpath).parts)
        if not candidate.exists():
            return {}
        _require_nonsymlink_descendant(root, candidate)
        document = _load_json(candidate, f"raw Host evidence {relpath}")
    else:
        return {}
    return {
        "host_evidence_value": document.value,
        "host_evidence_sha256": document.sha256,
        "host_evidence_size_bytes": document.size_bytes,
        "host_evidence_relative_path": relpath,
    }


def _safe_receipt_relpath(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.parent == PurePosixPath("memory/strategy_export_receipts")
        and path.suffix == ".json"
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _safe_host_evidence_relpath(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and len(path.parts) == 2
        and path.parts[0].startswith("attempt_")
        and path.parts[0][len("attempt_") :].isdigit()
        and path.parts[1] == "evaluation.json"
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _require_nonsymlink_descendant(root: Path, candidate: Path) -> None:
    root_resolved = root.resolve(strict=True)
    cursor = root
    if cursor.is_symlink():
        raise SourceRunValidationError(
            "SOURCE_EVIDENCE_RECEIPT_INVALID: source directory is a symlink"
        )
    for part in candidate.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SourceRunValidationError(
                "SOURCE_EVIDENCE_RECEIPT_INVALID: receipt path contains a symlink"
            )
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise SourceRunValidationError(
            "SOURCE_EVIDENCE_RECEIPT_INVALID: receipt escapes source directory"
        )


def _load_optional_json(value: object | None, label: str) -> _LoadedJson | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)) and not Path(value).exists():
        return None
    return _load_json(value, label)


def _load_json(value: object, label: str) -> _LoadedJson:
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise SourceRunValidationError(
                f"cannot read {label} at {path}: {exc}"
            ) from exc
        try:
            payload = strict_json_loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SourceRunValidationError(
                f"invalid JSON in {label} at {path}: {exc}"
            ) from exc
        return _LoadedJson(
            payload,
            str(path.resolve()),
            hashlib.sha256(raw).hexdigest(),
            len(raw),
        )
    if not isinstance(value, (Mapping, list)):
        raise SourceRunValidationError(
            f"{label} must be a JSON path, object, or list; got {type(value).__name__}"
        )
    payload = dict(value) if isinstance(value, Mapping) else list(value)
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _LoadedJson(
        payload,
        f"<in-memory:{label}>",
        hashlib.sha256(raw).hexdigest(),
        len(raw),
    )


def _nested(payload: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            if current not in (None, ""):
                return current
    return None


def _resolve_run_metadata(run: _LoadedRun) -> None:
    manifest = run.manifest.value
    metrics = run.metrics.value
    if not isinstance(manifest, Mapping) or not isinstance(metrics, Mapping):
        raise SourceRunValidationError("source metrics and manifest must be objects")
    identity = resolve_source_identity(manifest, run.audit)
    run.model = identity.model
    run.scene = identity.scene
    run.protocol = identity.protocol
    run.seed = identity.seed
    run.method_commit = identity.method_commit
    run.condition_fingerprint = identity.condition_fingerprint
    run.metadata_conflicts = identity.conflicts


def _validate_source_evidence_binding(
    run: _LoadedRun,
    source_index: int,
) -> list[str]:
    binding = run.evidence_binding
    validation = validate_source_evidence_binding(
        source_index=source_index,
        binding=binding.value if binding is not None else None,
        binding_sha256=binding.sha256 if binding is not None else "",
        strategy_specs=run.strategy.value,
        strategy_specs_sha256=run.strategy.sha256,
        metrics=run.metrics.value,
        metrics_sha256=run.metrics.sha256,
        manifest=run.manifest.value,
        audit=run.audit,
        evidence_receipts=run.evidence_receipts,
    )
    run.evidence_binding_pin_source = validation.pin_source
    run.bound_evidence = dict(validation.records)
    return list(validation.errors)


def _validate_source_cohort(
    runs: list[_LoadedRun],
    *,
    expected_model: str | None,
    expected_protocol: str | None,
    expected_seeds: Sequence[int] | None,
    expected_method_commit: str | None,
    expected_condition_fingerprint: str | None,
    expected_provider_route_id: str | None,
    expected_provider_route_fingerprint: str | None,
    expected_scene: str,
) -> None:
    errors: list[str] = []
    for index, run in enumerate(runs):
        _resolve_run_metadata(run)
        manifest = run.manifest.value
        metrics = run.metrics.value
        audit = run.audit
        prefix = f"source run {index}"
        errors.extend(f"{prefix}: {message}" for message in run.metadata_conflicts)
        errors.extend(_validate_source_evidence_binding(run, index))
        if not run.model:
            errors.append(f"{prefix}: source model is not auditable")
        if normalize_source_scene(run.scene) != normalize_source_scene(
            expected_scene
        ):
            errors.append(
                f"{prefix}: expected base scene {expected_scene!r}, got {run.scene!r}"
            )
        if not run.protocol:
            errors.append(f"{prefix}: evaluation protocol is not auditable")
        if not run.seed:
            errors.append(f"{prefix}: source seed is not auditable")
        elif not _is_nonnegative_integer_text(run.seed):
            errors.append(f"{prefix}: source seed must be a non-negative integer")
        if expected_method_commit is not None:
            if not run.method_commit:
                errors.append(f"{prefix}: source method commit is not auditable")
            elif run.method_commit != expected_method_commit:
                errors.append(
                    f"{prefix}: source method commit differs from the locked "
                    "source cohort"
                )
        if expected_condition_fingerprint is not None:
            if not run.condition_fingerprint:
                errors.append(
                    f"{prefix}: source condition fingerprint is not auditable"
                )
            elif run.condition_fingerprint != expected_condition_fingerprint:
                errors.append(
                    f"{prefix}: source condition fingerprint differs from the "
                    "locked source cohort"
                )
        if expected_provider_route_id is not None:
            provider_route_id = _nested(
                audit,
                ("provider_route_id",),
                ("source_provider_route_id",),
            )
            provider_route_fingerprint = _nested(
                audit,
                ("provider_route_fingerprint",),
                ("source_provider_route_fingerprint",),
            )
            if provider_route_id != expected_provider_route_id:
                errors.append(
                    f"{prefix}: source provider route ID differs from the "
                    "locked source cohort"
                )
            if provider_route_fingerprint != expected_provider_route_fingerprint:
                errors.append(
                    f"{prefix}: source provider route fingerprint differs from "
                    "the locked source cohort"
                )
        if audit.get("protocol_valid") is not True:
            errors.append(
                f"{prefix}: a frozen source audit must explicitly mark "
                "protocol_valid=true"
            )
        if audit.get("run_valid") is False:
            errors.append(f"{prefix}: audit marks the run invalid")
        if audit.get("evidence_valid") is False:
            errors.append(f"{prefix}: audit marks physical evidence invalid")
        if audit.get("infrastructure_valid") is False:
            errors.append(f"{prefix}: audit marks the run infrastructure-invalid")

        identity_verified = _nested(manifest, ("model", "identity_verified"))
        if identity_verified is not True:
            errors.append(f"{prefix}: model identity is not explicitly verified")
        execute = _nested(manifest, ("effective_arguments", "execute"))
        if execute is not True:
            errors.append(f"{prefix}: source is not explicitly a physical execution")
        budget, budget_errors = _resolve_physical_budget(manifest, audit)
        errors.extend(f"{prefix}: {message}" for message in budget_errors)
        if budget is None:
            errors.append(f"{prefix}: physical budget is not auditable")
        else:
            try:
                if int(budget) != 80:
                    errors.append(
                        f"{prefix}: physical budget must be 80, got {budget!r}"
                    )
            except (TypeError, ValueError):
                errors.append(f"{prefix}: invalid physical budget {budget!r}")

        invariant = _nested(metrics, ("physical_accounting", "invariant_satisfied"))
        if invariant is not True:
            errors.append(
                f"{prefix}: physical accounting invariant is not explicitly true"
            )
        pending = _nested(
            metrics,
            ("physical_accounting", "pending_physical_evaluation"),
            ("pending_physical_evaluation_count",),
        )
        if pending not in (None, 0, 0.0):
            errors.append(f"{prefix}: source has pending physical evaluation")

        terminal = _nested(metrics, ("terminal", "status"), ("status",))
        error_type = _nested(metrics, ("terminal", "error_type"), ("error_type",))
        invalid_text = f"{terminal or ''} {error_type or ''}".lower().replace("_", "")
        if any(
            token in invalid_text
            for token in (
                "infrastructureinvalid",
                "providerfatal",
                "backendfatal",
                "hostcontract",
                "manualinterrupt",
            )
        ):
            errors.append(f"{prefix}: infrastructure-invalid terminal state")

    models = {run.model for run in runs}
    scenes = {normalize_source_scene(run.scene) for run in runs}
    protocols = {run.protocol for run in runs}
    seeds = {run.seed for run in runs if run.seed}
    if len(models) != 1:
        errors.append(f"source runs use different models: {sorted(models)}")
    if len(scenes) != 1:
        errors.append(f"source runs use different scenes: {sorted(scenes)}")
    if len(protocols) != 1:
        errors.append(f"source runs use different protocols: {sorted(protocols)}")
    if len(seeds) != FROZEN_STRATEGY_BANK_POLICY.required_source_runs:
        errors.append(
            "source cohort must contain five unique seeds; got "
            f"{sorted(seeds)}"
        )
    normalized_expected_seeds = _normalize_expected_source_seeds(expected_seeds)
    if normalized_expected_seeds is not None and seeds != normalized_expected_seeds:
        errors.append(
            "source cohort seeds differ from the locked source cohort: "
            f"expected {sorted(normalized_expected_seeds)}, "
            f"got {sorted(seeds)}"
        )
    try:
        require_unique_source_manifests(
            [run.manifest.value for run in runs],
            required=FROZEN_STRATEGY_BANK_POLICY.required_source_runs,
        )
    except ValueError as exc:
        errors.append(str(exc))
    if expected_model is not None and models != {expected_model}:
        errors.append(
            f"source model does not match expected_source_model={expected_model!r}: "
            f"{sorted(models)}"
        )
    if expected_protocol is not None and protocols != {expected_protocol}:
        errors.append(
            f"source protocol does not match expected_protocol={expected_protocol!r}: "
            f"{sorted(protocols)}"
        )
    if errors:
        raise SourceRunValidationError("; ".join(errors))


def _validate_complete_source_identity_arguments(
    *,
    expected_source_model: str | None,
    expected_protocol: str | None,
    expected_source_seeds: Sequence[int] | None,
    expected_source_method_commit: str | None,
    expected_source_condition_fingerprint: str | None,
    expected_source_result_manifest_hash: str | None,
    expected_source_provider_route_id: str | None,
    expected_source_provider_route_fingerprint: str | None,
) -> bool:
    """Require all-or-none source identity for provenance schema v3."""

    values = {
        "expected_source_model": expected_source_model,
        "expected_protocol": expected_protocol,
        "expected_source_seeds": expected_source_seeds,
        "expected_source_method_commit": expected_source_method_commit,
        "expected_source_condition_fingerprint": (
            expected_source_condition_fingerprint
        ),
        "expected_source_result_manifest_hash": expected_source_result_manifest_hash,
        "expected_source_provider_route_id": expected_source_provider_route_id,
        "expected_source_provider_route_fingerprint": (
            expected_source_provider_route_fingerprint
        ),
    }
    provided = {name for name, value in values.items() if value is not None}
    if not provided:
        return False
    # Legacy/unit exports may still constrain the original model/protocol/seed
    # subset, but any formal identity field opts into the complete contract.
    new_fields = {
        "expected_source_method_commit",
        "expected_source_condition_fingerprint",
        "expected_source_result_manifest_hash",
        "expected_source_provider_route_id",
        "expected_source_provider_route_fingerprint",
    }
    if not provided.intersection(new_fields):
        return False
    missing = set(values).difference(provided)
    if missing:
        raise SourceRunValidationError(
            "complete Table-4 source identity is required for provenance v3; "
            f"missing={sorted(missing)}"
        )
    method_commit = str(expected_source_method_commit)
    if len(method_commit) != 40 or any(
        character not in "0123456789abcdef" for character in method_commit
    ):
        raise SourceRunValidationError(
            "expected source method commit must be a full lowercase Git commit"
        )
    for name, value in (
        ("condition fingerprint", expected_source_condition_fingerprint),
        ("result-manifest hash", expected_source_result_manifest_hash),
    ):
        digest = str(value)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise SourceRunValidationError(
                f"expected source {name} must be lowercase SHA-256"
            )
    _normalize_expected_source_seeds(expected_source_seeds)
    try:
        require_provider_route(
            str(expected_source_provider_route_id),
            str(expected_source_provider_route_fingerprint),
        )
    except ValueError as exc:
        raise SourceRunValidationError(
            f"expected source provider route is invalid: {exc}"
        ) from exc
    return True


def _normalize_expected_source_seeds(
    values: Sequence[int] | None,
) -> set[str] | None:
    if values is None:
        return None
    seeds = tuple(values)
    required = FROZEN_STRATEGY_BANK_POLICY.required_source_runs
    if len(seeds) != required:
        raise SourceRunValidationError(
            f"expected source cohort must contain exactly {required} seeds"
        )
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise SourceRunValidationError(
            "expected source cohort seeds must be non-negative integers"
        )
    if len(set(seeds)) != required:
        raise SourceRunValidationError(
            f"expected source cohort must contain {required} unique seeds"
        )
    return {str(seed) for seed in seeds}


def _is_nonnegative_integer_text(value: str) -> bool:
    try:
        parsed = int(value)
    except ValueError:
        return False
    return parsed >= 0 and str(parsed) == value


def _resolve_physical_budget(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> tuple[Any | None, tuple[str, ...]]:
    """Resolve the physical budget without allowing an audit to rewrite it.

    The frozen run manifest is authoritative.  A reviewed legacy audit may
    fill a value that the manifest genuinely lacks, but a second declaration
    must agree exactly after integer normalization.  This mirrors the
    manifest-first rule used for source-evidence pins.
    """

    manifest_values = _declared_values(
        manifest,
        ("budgets", "physical_transactions"),
        ("effective_arguments", "iteration_budget"),
    )
    audit_values = _declared_values(
        audit,
        ("physical_budget",),
        ("budget",),
    )
    errors: list[str] = []
    manifest_budget = _one_consistent_budget(
        manifest_values,
        source="manifest",
        errors=errors,
    )
    audit_budget = _one_consistent_budget(
        audit_values,
        source="audit",
        errors=errors,
    )
    if manifest_budget is not None:
        if audit_budget is not None and audit_budget != manifest_budget:
            errors.append(
                "audit physical budget conflicts with manifest "
                f"({audit_budget!r} != {manifest_budget!r})"
            )
        return manifest_budget, tuple(errors)
    return audit_budget, tuple(errors)


def _declared_values(
    payload: Mapping[str, Any],
    *paths: tuple[str, ...],
) -> list[Any]:
    values: list[Any] = []
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            if current not in (None, ""):
                values.append(current)
    return values


def _one_consistent_budget(
    values: Sequence[Any],
    *,
    source: str,
    errors: list[str],
) -> int | None:
    if not values:
        return None
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool):
            errors.append(f"invalid {source} physical budget {value!r}")
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            errors.append(f"invalid {source} physical budget {value!r}")
            continue
        if isinstance(value, float) and value != parsed:
            errors.append(f"invalid {source} physical budget {value!r}")
            continue
        if isinstance(value, str) and value.strip() != str(parsed):
            errors.append(f"invalid {source} physical budget {value!r}")
            continue
        normalized.append(parsed)
    if not normalized:
        return None
    if len(set(normalized)) != 1:
        errors.append(
            f"conflicting {source} physical budget declarations: {normalized!r}"
        )
    return normalized[0]


def _source_sidecar_entry(run: _LoadedRun, index: int) -> dict[str, Any]:
    if run.evidence_binding is None:  # Defensive; cohort preflight rejects this.
        raise SourceRunValidationError(
            "SOURCE_EVIDENCE_BINDING_REQUIRED: validated source lost its binding"
        )
    return {
        "source_index": index,
        "model": run.model,
        "scene": run.scene,
        "protocol": run.protocol,
        "seed": run.seed,
        "frozen_audit": dict(run.audit),
        "frozen_audit_sha256": semantic_sha256(dict(run.audit)),
        "documents": {
            "strategy_specs": {
                "path": run.strategy.source_path,
                "sha256": run.strategy.sha256,
            },
            "metrics": {
                "path": run.metrics.source_path,
                "sha256": run.metrics.sha256,
            },
            "manifest": {
                "path": run.manifest.source_path,
                "sha256": run.manifest.sha256,
                "semantic_sha256": stable_manifest_sha256(run.manifest.value),
            },
            "evidence_binding": {
                "path": run.evidence_binding.source_path,
                "sha256": run.evidence_binding.sha256,
                "pinned_by": run.evidence_binding_pin_source,
            },
        },
        "kept": [],
        "dropped": [],
    }


def _write_json_idempotent(path: Path, value: object) -> None:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StrategyBankError(
                f"cannot inspect existing output {path}: {exc}"
            ) from exc
        if existing == rendered:
            return
        raise StrategyBankError(f"refusing to overwrite non-identical output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)
