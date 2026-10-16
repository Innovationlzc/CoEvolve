"""Cohort data contracts and summaries, without private historical audit pins.

The original campaign-specific qualification implementation is not distributed.
Its entrypoints fail closed before reading input: a public source snapshot must
not certify old experimental evidence using substituted or inferred identities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contracts import Table1SourceCohort
from ..full_method_compatibility import FullMethodCompatibilityPin
from .metrics import mean, median_iqr, sample_std, wilson_interval

class Table1SourceAuditError(RuntimeError):
    """A frozen Table-1 source cannot be admitted as paper evidence."""

@dataclass(frozen=True)
class Table1RunEvidence:
    """One source run after aggregate-manifest and per-run verification."""

    run_dir: str
    seed: int
    run_index: int
    backbone: str
    provider: str
    provider_route_id: str
    provider_route_fingerprint: str
    protocol_id: str
    method_commit: str
    condition_fingerprint: str
    shared_prompt_contract_fingerprint: str
    decision_prompt_fingerprint: str
    full_method_compatibility_fingerprint: str
    terminal_status: str
    task_success: bool
    physical_transactions: int
    accepted_transactions: int
    rejected_transactions: int
    phase_completed: int
    phase_total: int
    phase_completion: float
    fm_calls: int
    input_tokens: int | None
    output_tokens: int | None
    wall_time_s: float | None
    cost_usd: float | None
    artifact_hashes: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifact_hashes"] = dict(self.artifact_hashes)
        return value

@dataclass(frozen=True)
class Table1CohortAudit:
    """Audited identity and five records for one Table-1 backbone row."""

    source: Table1SourceCohort
    manifest_path: str
    manifest_sha256: str
    manifest_internal_sha256: str
    manifest_kind: str
    provider: str
    provider_route_id: str
    provider_route_fingerprint: str
    provider_only_companion: bool
    parent_protocol_id: str
    parent_method_commit: str
    parent_result_manifest_hash: str
    shared_method_artifact_hashes: Mapping[str, str]
    full_method_compatibility: FullMethodCompatibilityPin
    runs: tuple[Table1RunEvidence, ...]

    @property
    def protocol_valid(self) -> bool:
        return len(self.runs) == len(self.source.seeds) == 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": asdict(self.source),
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "manifest_internal_sha256": self.manifest_internal_sha256,
            "manifest_kind": self.manifest_kind,
            "provider": self.provider,
            "provider_route_id": self.provider_route_id,
            "provider_route_fingerprint": self.provider_route_fingerprint,
            "provider_only_companion": self.provider_only_companion,
            "parent_protocol_id": self.parent_protocol_id,
            "parent_method_commit": self.parent_method_commit,
            "parent_result_manifest_hash": self.parent_result_manifest_hash,
            "shared_method_artifact_hashes": dict(
                self.shared_method_artifact_hashes
            ),
            "full_method_compatibility": asdict(
                self.full_method_compatibility
            ),
            "runs": [run.to_dict() for run in self.runs],
        }

def summarize_table1_cohort(
    audit: Table1CohortAudit,
    *,
    physical_budget: int = 80,
) -> dict[str, Any]:
    """Produce one paper-facing backbone row without pooling source identities."""

    if not audit.protocol_valid:
        raise Table1SourceAuditError("Table-1 cohort is not a complete five-run audit")
    runs = audit.runs
    successes = sum(run.task_success for run in runs)
    capped = [
        min(run.physical_transactions, physical_budget)
        if run.task_success
        else physical_budget
        for run in runs
    ]
    phase_values = [run.phase_completion for run in runs]
    rejected_values = [run.rejected_transactions for run in runs]
    fm_call_values = [run.fm_calls for run in runs]
    low, high = wilson_interval(successes, len(runs))
    return {
        "backbone": audit.source.backbone,
        "provider": audit.provider,
        "provider_route_id": audit.provider_route_id,
        "provider_route_fingerprint": audit.provider_route_fingerprint,
        "protocol_id": audit.source.protocol_id,
        "method_commit": audit.source.method_commit,
        "condition_fingerprint": audit.source.condition_fingerprint,
        "full_method_compatibility_fingerprint": (
            audit.full_method_compatibility.fingerprint
        ),
        "result_manifest_sha256": audit.manifest_sha256,
        "provider_only_companion": audit.provider_only_companion,
        "parent_protocol_id": audit.parent_protocol_id,
        "parent_method_commit": audit.parent_method_commit,
        "seeds": list(audit.source.seeds),
        "successful_runs": successes,
        "success_over_expected": f"{successes}/{len(runs)}",
        "success_rate": successes / len(runs),
        "success_rate_wilson_95_low": low,
        "success_rate_wilson_95_high": high,
        "cpt_at_80": mean(capped),
        "cpt_at_80_std": sample_std(capped),
        "phase_completion": mean(phase_values),
        "phase_completion_std": sample_std(phase_values),
        "rejected_transactions": mean(rejected_values),
        "rejected_transactions_std": sample_std(rejected_values),
        "fm_calls": mean(fm_call_values),
        "fm_calls_std": sample_std(fm_call_values),
        "distribution": {
            "cpt_at_80": median_iqr(capped),
            "phase_completion": median_iqr(phase_values),
            "rejected_transactions": median_iqr(rejected_values),
            "fm_calls": median_iqr(fm_call_values),
        },
        "runs": [run.to_dict() for run in runs],
    }

def summarize_table1_backbones(
    audits: Iterable[Table1CohortAudit],
) -> dict[str, Any]:
    """Keep each audited backbone/provider/protocol cohort as a distinct row."""

    selected = tuple(audits)
    identities = [
        (
            audit.source.backbone,
            audit.provider,
            audit.source.protocol_id,
            audit.source.method_commit,
            audit.source.condition_fingerprint,
            audit.manifest_sha256,
        )
        for audit in selected
    ]
    if len(identities) != len(set(identities)):
        raise Table1SourceAuditError("duplicate Table-1 source cohort identity")
    backbones = [audit.source.backbone for audit in selected]
    if len(backbones) != len(set(backbones)):
        raise Table1SourceAuditError(
            "one Table-1 report cannot silently pool multiple cohorts for a backbone"
        )
    return {
        "kind": "coevolve_table1_backbone_summary",
        "schema_version": 1,
        "pooling_policy": "one separately audited source identity per backbone row",
        "rows": [summarize_table1_cohort(audit) for audit in selected],
    }

def audit_table1_source_cohort(
    manifest_path: str | Path,
    source: Table1SourceCohort,
    *,
    parent: Table1CohortAudit | None = None,
) -> Table1CohortAudit:
    """Unavailable: requires the separately retained historical qualification."""
    raise Table1SourceAuditError(
        "Historical cohort qualification is not distributed; "
        "this public checkout cannot certify historical source evidence."
    )


def audit_frozen_table1_candidate(
    manifest_path: str | Path,
    cohort: str,
    *,
    parent_manifest_path: str | Path | None = None,
) -> Table1CohortAudit:
    """Unavailable: never replace private qualification pins with placeholders."""
    raise Table1SourceAuditError(
        "Historical cohort qualification is not distributed; "
        "this public checkout cannot certify historical source evidence."
    )


__all__ = [
    "Table1CohortAudit", "Table1RunEvidence", "Table1SourceAuditError",
    "audit_frozen_table1_candidate", "audit_table1_source_cohort",
    "summarize_table1_backbones", "summarize_table1_cohort",
]
