"""Evidence-linked selection and extraction for the representative trace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .collect import RunRecord, collect_run
from .evidence_artifacts import sha256_file
from .trace_evidence import TraceEvidenceError, validate_trace_evidence


class TraceSelectionError(RuntimeError):
    """Raised when no preregistered, evidence-consistent trace is available."""


def select_representative_run(
    run_dirs: Iterable[str | Path],
    *,
    condition: str,
    target_model: str,
    expected_by_dir: Mapping[str, Mapping[str, Any]] | None = None,
) -> RunRecord:
    """Choose the first successful seed in one preregistered Full/GPT cohort."""

    records: list[RunRecord] = []
    expected_lookup = dict(expected_by_dir or {})
    for run_dir in run_dirs:
        path = Path(run_dir)
        record = collect_run(path, expected=expected_lookup.get(str(path)))
        if record.condition == condition and record.target_model == target_model:
            records.append(record)
    ordered = sorted(
        records,
        key=lambda item: (
            item.seed is None,
            item.seed if item.seed is not None else 10**9,
            item.run_dir,
        ),
    )
    selected = next(
        (row for row in ordered if row.protocol_valid and row.task_success),
        None,
    )
    if selected is None:
        raise TraceSelectionError(
            "no protocol-valid successful run in the preregistered "
            f"condition={condition!r}, model={target_model!r} cohort"
        )
    return selected


def build_figure2_trace(run: RunRecord | str | Path) -> dict[str, Any]:
    """Build one manifest-pinned failure→mutation→accepted-child trace."""

    record = run if isinstance(run, RunRecord) else collect_run(run)
    if not record.protocol_valid or not record.task_success:
        raise TraceSelectionError("Figure 2 requires a protocol-valid successful run")
    root = Path(record.run_dir)
    try:
        evidence = validate_trace_evidence(root, record)
    except TraceEvidenceError as exc:
        raise TraceSelectionError(f"no evidence-complete Figure-2 lineage: {exc}") from exc
    parent = evidence.parent
    child = evidence.child
    return {
        "kind": "representative_evolution_trace",
        "selection": {
            "rule": "first successful seed in the frozen Full GPT condition",
            "protocol_id": record.protocol_id,
            "condition": record.condition,
            "model": record.target_model,
            "seed": record.seed,
            "run_dir": record.run_dir,
            "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
            "trace_evidence_sha256": evidence.receipt_sha256,
            "trace_evidence_size_bytes": evidence.receipt_size_bytes,
        },
        "causal_checkpoint": parent["checkpoint_before"],
        "events": [
            {
                "event": "accepted_checkpoint",
                "checkpoint_id": parent["checkpoint_before"],
            },
            {
                "event": "physical_failure",
                "attempt": parent["attempt"],
                "physical_iteration": parent["physical_iteration"],
                "policy_id": parent["script_id"],
                "policy_sha256": parent["policy_sha256"],
                "failure_type": parent["failure_type"],
                "policy_artifact": parent["policy_ref"],
            },
            {
                "event": "verifier_evidence_and_rollback",
                "transition_outcome": parent["transition_outcome"],
                "rollback_performed": True,
                "evaluation_artifact": parent["evaluation_ref"],
                "rollback_artifact": parent["restore_ref"],
            },
            {
                "event": "failure_classification_and_mutation",
                "failure_level": parent["failure_level"],
                "mutation": dict(evidence.mutation),
            },
            {
                "event": "accepted_direct_child",
                "attempt": child["attempt"],
                "physical_iteration": child["physical_iteration"],
                "policy_id": child["script_id"],
                "policy_sha256": child["policy_sha256"],
                "checkpoint_after": child["checkpoint_after"],
                "policy_artifact": child["policy_ref"],
                "execution_trace_artifact": child["execution_trace_ref"],
                "accepted_manifest_artifact": child["accepted_manifest_ref"],
            },
            {
                "event": "memory_update",
                "memory_update": dict(evidence.memory_update),
            },
        ],
    }
