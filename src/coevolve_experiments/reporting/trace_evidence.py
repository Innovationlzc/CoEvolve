"""Manifest-pinned causal evidence contract for the qualitative Figure 2.

The old extractor inferred ancestry from adjacent ``attempt_*`` directories.
That is not evidence of a mutation lineage.  Formal traces instead require a
host-produced receipt whose own bytes are pinned by the run manifest and whose
artifact references are independently re-hashed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .collect import RunRecord
from .evidence_artifacts import (
    BoundArtifactError,
    BoundJsonArtifact,
    load_bound_json_artifact,
)
from .trace_projection import (
    PolicyProjectionError,
    changed_projection_dimensions,
    executable_policy_projection,
    projection_sha256,
    strategy_matches_projection,
)


TRACE_EVIDENCE_FILENAME = "figure2_trace_evidence.json"
TRACE_EVIDENCE_KIND = "coevolve_figure2_trace_evidence"
TRACE_EVIDENCE_SCHEMA_VERSION = 1
TRACE_PRODUCER_ID = "praxis_coop.host.figure2_trace_evidence"
MUTATION_DELTA_KIND = "host_executable_mutation_delta"
PROMOTION_RECEIPT_KIND = "host_memory_promotion_receipt"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REJECTED_OUTCOMES = frozenset(
    {"REJECT", "REJECT_UNSAFE", "REJECT_REGRESSION", "REJECT_BLOCKED"}
)
_ACCEPTED_OUTCOMES = frozenset(
    {"ACCEPT_SETUP", "ACCEPT_PROGRESS", "TASK_COMPLETE"}
)


class TraceEvidenceError(ValueError):
    """Raised when a purported Figure-2 causal chain is not auditable."""


@dataclass(frozen=True)
class ValidatedTraceEvidence:
    receipt_sha256: str
    receipt_size_bytes: int
    parent: Mapping[str, Any]
    child: Mapping[str, Any]
    mutation: Mapping[str, Any]
    memory_update: Mapping[str, Any]


def validate_trace_evidence(
    run_root: str | Path,
    record: RunRecord,
) -> ValidatedTraceEvidence:
    root = Path(run_root)
    manifest = _read_root_json(root, "run_manifest.json")
    receipt, receipt_sha256, receipt_size = _read_receipt(root)
    result_hashes = _mapping(
        manifest.get("result_artifact_hashes"),
        "run_manifest.result_artifact_hashes",
    )
    expected_receipt_sha256 = result_hashes.get("figure2_trace_evidence_sha256")
    if expected_receipt_sha256 != receipt_sha256:
        raise TraceEvidenceError(
            "run manifest does not pin the exact Figure-2 trace-evidence bytes"
        )

    _require_exact_keys(
        receipt,
        {
            "kind",
            "schema_version",
            "producer",
            "run_binding",
            "parent",
            "child",
            "mutation",
            "promotion",
        },
        "trace evidence",
    )
    if receipt.get("kind") != TRACE_EVIDENCE_KIND:
        raise TraceEvidenceError("invalid Figure-2 trace-evidence kind")
    if receipt.get("schema_version") != TRACE_EVIDENCE_SCHEMA_VERSION:
        raise TraceEvidenceError("unsupported Figure-2 trace-evidence schema")
    _validate_producer(receipt.get("producer"))
    _validate_run_binding(receipt.get("run_binding"), manifest, record)

    parent_meta = _mapping(receipt.get("parent"), "trace evidence parent")
    child_meta = _mapping(receipt.get("child"), "trace evidence child")
    mutation_meta = _mapping(receipt.get("mutation"), "trace evidence mutation")
    promotion_meta = _mapping(receipt.get("promotion"), "trace evidence promotion")
    parent = _load_attempt(root, parent_meta, label="parent", accepted=False)
    child = _load_attempt(root, child_meta, label="child", accepted=True)
    _validate_direct_child(parent, child)
    mutation = _validate_mutation(root, mutation_meta, parent, child)
    memory_update = _validate_promotion(root, promotion_meta, child)
    return ValidatedTraceEvidence(
        receipt_sha256=receipt_sha256,
        receipt_size_bytes=receipt_size,
        parent=parent,
        child=child,
        mutation=mutation,
        memory_update=memory_update,
    )


def _validate_producer(value: object) -> None:
    producer = _mapping(value, "trace evidence producer")
    _require_exact_keys(
        producer,
        {"producer_id", "producer_version"},
        "trace evidence producer",
    )
    if producer.get("producer_id") != TRACE_PRODUCER_ID:
        raise TraceEvidenceError("Figure-2 receipt producer is not allowlisted")
    if producer.get("producer_version") != 1:
        raise TraceEvidenceError("unsupported Figure-2 receipt producer version")


def _validate_run_binding(
    value: object,
    manifest: Mapping[str, Any],
    record: RunRecord,
) -> None:
    binding = _mapping(value, "trace evidence run_binding")
    required = {
        "protocol_id",
        "condition",
        "target_model",
        "seed",
        "base_method_commit",
    }
    _require_exact_keys(binding, required, "trace evidence run_binding")
    commit = _text(binding.get("base_method_commit"), "run_binding.base_method_commit")
    if not _COMMIT_RE.fullmatch(commit):
        raise TraceEvidenceError("run_binding.base_method_commit must be lowercase Git SHA-1")
    seed = _integer(binding.get("seed"), "run_binding.seed", minimum=0)

    expected = {
        "protocol_id": record.protocol_id,
        "condition": record.condition,
        "target_model": record.target_model,
        "seed": record.seed,
        "base_method_commit": record.base_method_commit,
    }
    for key, expected_value in expected.items():
        if expected_value in (None, ""):
            raise TraceEvidenceError(f"RunRecord lacks formal {key} binding")
        actual_value = seed if key == "seed" else binding.get(key)
        if actual_value != expected_value:
            raise TraceEvidenceError(f"trace receipt {key} does not match RunRecord")

    experiment = _mapping(manifest.get("experiment"), "run_manifest.experiment")
    protocol = manifest.get("protocol")
    protocol = protocol if isinstance(protocol, Mapping) else {}
    model = _mapping(manifest.get("model"), "run_manifest.model")
    codebase = _mapping(manifest.get("codebase"), "run_manifest.codebase")
    manifest_values = {
        "protocol_id": experiment.get("protocol_id") or protocol.get("protocol_id"),
        "condition": experiment.get("condition") or experiment.get("condition_id"),
        "target_model": experiment.get("target_model") or model.get("resolved"),
        "seed": experiment.get("seed") if "seed" in experiment else manifest.get("seed"),
        "base_method_commit": experiment.get("base_method_commit")
        or codebase.get("git_head"),
    }
    for key, actual in manifest_values.items():
        if actual in (None, ""):
            raise TraceEvidenceError(f"run manifest lacks Figure-2 {key} binding")
        expected_value = seed if key == "seed" else binding.get(key)
        if actual != expected_value:
            raise TraceEvidenceError(f"trace receipt {key} does not match run manifest")


def _load_attempt(
    root: Path,
    meta: Mapping[str, Any],
    *,
    label: str,
    accepted: bool,
) -> dict[str, Any]:
    required = {
        "attempt",
        "physical_iteration",
        "script_id",
        "checkpoint_before",
        "artifacts",
    }
    if accepted:
        required |= {"parent_script_id", "checkpoint_after"}
    _require_exact_keys(meta, required, f"trace evidence {label}")
    attempt = _integer(meta.get("attempt"), f"{label}.attempt", minimum=1)
    physical_iteration = _integer(
        meta.get("physical_iteration"),
        f"{label}.physical_iteration",
        minimum=1,
    )
    script_id = _text(meta.get("script_id"), f"{label}.script_id")
    checkpoint_before = _text(
        meta.get("checkpoint_before"),
        f"{label}.checkpoint_before",
    )
    parent_script_id = ""
    checkpoint_after = ""
    if accepted:
        parent_script_id = _text(
            meta.get("parent_script_id"),
            f"{label}.parent_script_id",
        )
        checkpoint_after = _text(
            meta.get("checkpoint_after"),
            f"{label}.checkpoint_after",
        )

    artifacts_meta = _mapping(meta.get("artifacts"), f"{label}.artifacts")
    artifact_names = {
        "policy",
        "evaluation",
        "transition_decision",
        "evolution_record",
    }
    artifact_names |= (
        {"pre_dispatch_restore", "execution_trace", "accepted_manifest"}
        if accepted
        else {"rollback_receipt"}
    )
    _require_exact_keys(artifacts_meta, artifact_names, f"{label}.artifacts")
    artifacts = {
        name: _load_ref(root, artifacts_meta[name], f"{label}.artifacts.{name}")
        for name in sorted(artifact_names)
    }
    prefix = f"attempt_{attempt:02d}/"
    for name in ("policy", "evaluation", "transition_decision", "evolution_record"):
        if not artifacts[name].relative_path.startswith(prefix):
            raise TraceEvidenceError(
                f"{label} {name} is not inside the declared attempt directory"
            )
    local_receipt = "pre_dispatch_restore" if accepted else "rollback_receipt"
    if not artifacts[local_receipt].relative_path.startswith(prefix):
        raise TraceEvidenceError(
            f"{label} {local_receipt} is not inside the declared attempt directory"
        )

    policy = artifacts["policy"].payload
    evaluation = artifacts["evaluation"].payload
    transition = artifacts["transition_decision"].payload
    evolution = artifacts["evolution_record"].payload
    if _integer(policy.get("attempt"), f"{label} policy.attempt", minimum=1) != attempt:
        raise TraceEvidenceError(f"{label} policy attempt mismatch")
    if _text(policy.get("script_id"), f"{label} policy.script_id") != script_id:
        raise TraceEvidenceError(f"{label} policy script_id mismatch")
    if accepted and _text(
        policy.get("parent_script_id"),
        f"{label} policy.parent_script_id",
    ) != parent_script_id:
        raise TraceEvidenceError(f"{label} policy parent_script_id mismatch")

    metrics = _mapping(evaluation.get("metrics"), f"{label} evaluation.metrics")
    verifier = _mapping(evaluation.get("evidence"), f"{label} evaluation.evidence")
    if not metrics or not verifier:
        raise TraceEvidenceError(f"{label} evaluation lacks measured verifier evidence")
    evaluation_success = evaluation.get("success")
    if evaluation_success is not accepted:
        raise TraceEvidenceError(f"{label} evaluation success flag contradicts outcome")
    failure_type = _text(
        evaluation.get("failure_type"),
        f"{label} evaluation.failure_type",
    )
    if accepted and failure_type != "none":
        raise TraceEvidenceError("accepted child must have failure_type='none'")
    if not accepted and failure_type == "none":
        raise TraceEvidenceError("rejected parent must name a physical failure")

    outcome = _text(transition.get("outcome"), f"{label} transition.outcome")
    allowed_outcomes = _ACCEPTED_OUTCOMES if accepted else _REJECTED_OUTCOMES
    if outcome not in allowed_outcomes:
        raise TraceEvidenceError(f"{label} transition outcome is not admissible")
    controller = _mapping(
        evolution.get("controller_decision"),
        f"{label} evolution.controller_decision",
    )
    if _integer(
        controller.get("physical_iterations"),
        f"{label} controller physical_iterations",
        minimum=1,
    ) != physical_iteration:
        raise TraceEvidenceError(f"{label} physical iteration mismatch")
    if evolution.get("checkpoint_before") != checkpoint_before:
        raise TraceEvidenceError(f"{label} evolution checkpoint_before mismatch")

    if accepted:
        if evolution.get("checkpoint_after") != checkpoint_after:
            raise TraceEvidenceError("child evolution checkpoint_after mismatch")
        if evolution.get("rollback_performed") is not False:
            raise TraceEvidenceError("accepted child must not report rollback")
        restore = artifacts["pre_dispatch_restore"].payload
        if (
            restore.get("checkpoint_id") != checkpoint_before
            or restore.get("within_tolerance") is not True
        ):
            raise TraceEvidenceError(
                "child pre-dispatch restore does not verify the parent checkpoint"
            )
        execution = artifacts["execution_trace"].payload
        if (
            execution.get("attempt") != attempt
            or execution.get("script_id") != script_id
            or execution.get("status") != "success"
        ):
            raise TraceEvidenceError("child execution trace does not match accepted policy")
        _validate_accepted_manifest(
            artifacts["accepted_manifest"].payload,
            attempt=attempt,
            script_id=script_id,
            checkpoint_before=checkpoint_before,
            checkpoint_after=checkpoint_after,
            outcome=outcome,
        )
    else:
        if evolution.get("checkpoint_after") not in (None, ""):
            raise TraceEvidenceError("rejected parent must not advance checkpoint")
        if evolution.get("rollback_performed") is not True:
            raise TraceEvidenceError("rejected parent must have a host-recorded rollback")
        rollback = artifacts["rollback_receipt"].payload
        if (
            rollback.get("checkpoint_id") != checkpoint_before
            or rollback.get("within_tolerance") is not True
        ):
            raise TraceEvidenceError("rollback receipt does not verify parent checkpoint")

    try:
        projection = executable_policy_projection(policy)
    except PolicyProjectionError as exc:
        raise TraceEvidenceError(f"{label} policy projection invalid: {exc}") from exc
    return {
        "attempt": attempt,
        "physical_iteration": physical_iteration,
        "script_id": script_id,
        "parent_script_id": parent_script_id,
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "transition_outcome": outcome,
        "failure_type": failure_type,
        "failure_level": str(controller.get("repair_level") or ""),
        "policy": policy,
        "policy_ref": artifacts["policy"].public_ref(),
        "policy_sha256": artifacts["policy"].sha256,
        "evaluation_ref": artifacts["evaluation"].public_ref(),
        "evaluation_sha256": artifacts["evaluation"].sha256,
        "evolution_ref": artifacts["evolution_record"].public_ref(),
        "execution_trace_ref": (
            artifacts["execution_trace"].public_ref() if accepted else {}
        ),
        "execution_trace_sha256": (
            artifacts["execution_trace"].sha256 if accepted else ""
        ),
        "restore_ref": artifacts[local_receipt].public_ref(),
        "accepted_manifest_ref": (
            artifacts["accepted_manifest"].public_ref() if accepted else {}
        ),
        "projection": projection,
        "projection_sha256": projection_sha256(projection),
    }


def _validate_direct_child(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
) -> None:
    if child["attempt"] != parent["attempt"] + 1:
        raise TraceEvidenceError("Figure-2 child must be the direct next attempt")
    if child["physical_iteration"] != parent["physical_iteration"] + 1:
        raise TraceEvidenceError("Figure-2 physical iterations must be consecutive")
    if child["parent_script_id"] != parent["script_id"]:
        raise TraceEvidenceError("child policy does not name the rejected parent script")
    if child["checkpoint_before"] != parent["checkpoint_before"]:
        raise TraceEvidenceError("parent and child do not share one trusted checkpoint")
    if child["checkpoint_after"] in ("", child["checkpoint_before"]):
        raise TraceEvidenceError("accepted child checkpoint must be non-empty and advance")
    if child["policy_sha256"] == parent["policy_sha256"]:
        raise TraceEvidenceError("parent and child policy bytes are identical")


def _validate_mutation(
    root: Path,
    meta: Mapping[str, Any],
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(
        meta,
        {"mutation_level", "model_artifact", "effective_delta_artifact"},
        "trace evidence mutation",
    )
    mutation_level = _text(meta.get("mutation_level"), "mutation.mutation_level")
    if mutation_level not in {
        "strategy_mutation",
        "coordination_mutation",
        "parameter_tuning",
    }:
        raise TraceEvidenceError("invalid Figure-2 mutation level")
    model_artifact = _load_ref(root, meta.get("model_artifact"), "mutation.model_artifact")
    delta_artifact = _load_ref(
        root,
        meta.get("effective_delta_artifact"),
        "mutation.effective_delta_artifact",
    )
    model = model_artifact.payload
    if model.get("role") != "mutation_operator" or model.get("valid") is not True:
        raise TraceEvidenceError("mutation artifact is not a valid mutation-operator output")
    if model.get("artifact_id") != f"attempt_{child['attempt']:02d}":
        raise TraceEvidenceError("mutation artifact_id does not match child attempt")
    normalization = _mapping(
        model.get("normalization_audit"),
        "mutation normalization_audit",
    )
    if normalization.get("raw_response_preserved") is not True:
        raise TraceEvidenceError("mutation raw response was not preserved")
    parsed = _mapping(model.get("parsed_response"), "mutation parsed_response")
    if parsed.get("mutation_level") != mutation_level:
        raise TraceEvidenceError("mutation level conflicts with model artifact")

    delta = delta_artifact.payload
    required_delta = {
        "kind",
        "schema_version",
        "source",
        "attempt",
        "mutation_level",
        "parent_policy_sha256",
        "child_policy_sha256",
        "parent_projection_sha256",
        "child_projection_sha256",
        "changed_dimensions",
        "effective",
    }
    _require_exact_keys(delta, required_delta, "host mutation delta")
    if (
        delta.get("kind") != MUTATION_DELTA_KIND
        or delta.get("schema_version") != 1
        or delta.get("source") != "host_executable_projection_diff"
    ):
        raise TraceEvidenceError("invalid host mutation-delta identity")
    if delta.get("attempt") != child["attempt"]:
        raise TraceEvidenceError("host mutation delta attempt mismatch")
    if delta.get("mutation_level") != mutation_level:
        raise TraceEvidenceError("host mutation delta level mismatch")
    if delta.get("parent_policy_sha256") != parent["policy_sha256"]:
        raise TraceEvidenceError("host mutation delta parent-policy hash mismatch")
    if delta.get("child_policy_sha256") != child["policy_sha256"]:
        raise TraceEvidenceError("host mutation delta child-policy hash mismatch")
    if delta.get("parent_projection_sha256") != parent["projection_sha256"]:
        raise TraceEvidenceError("host mutation delta parent projection mismatch")
    if delta.get("child_projection_sha256") != child["projection_sha256"]:
        raise TraceEvidenceError("host mutation delta child projection mismatch")
    derived_dimensions = changed_projection_dimensions(
        parent["projection"],
        child["projection"],
    )
    raw_dimensions = delta.get("changed_dimensions")
    if (
        not isinstance(raw_dimensions, list)
        or raw_dimensions != sorted(set(raw_dimensions))
        or tuple(raw_dimensions) != derived_dimensions
    ):
        raise TraceEvidenceError("host mutation changed_dimensions are not derived")
    if not derived_dimensions or delta.get("effective") is not True:
        raise TraceEvidenceError("mutation is not executable-policy effective")
    if parsed.get("host_effective_delta") != delta:
        raise TraceEvidenceError(
            "embedded and independent host mutation deltas do not match exactly"
        )
    try:
        strategy_matches_projection(
            parsed.get("parent_strategy"),
            parent["projection"],
            label="mutation.parent_strategy",
        )
        strategy_matches_projection(
            parsed.get("child_strategy"),
            child["projection"],
            label="mutation.child_strategy",
        )
    except PolicyProjectionError as exc:
        raise TraceEvidenceError(str(exc)) from exc
    return {
        "artifact": model_artifact.public_ref(),
        "valid": True,
        "mutation_level": mutation_level,
        "changed_dimensions": list(derived_dimensions),
        "effective_delta": dict(delta),
        "effective_delta_artifact": delta_artifact.public_ref(),
        "effective": True,
    }


def _validate_promotion(
    root: Path,
    meta: Mapping[str, Any],
    child: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(
        meta,
        {"proposal_artifact", "receipt", "memory_artifact"},
        "trace evidence promotion",
    )
    proposal_artifact = _load_ref(
        root,
        meta.get("proposal_artifact"),
        "promotion.proposal_artifact",
    )
    receipt_artifact = _load_ref(root, meta.get("receipt"), "promotion.receipt")
    memory_artifact = _load_ref(
        root,
        meta.get("memory_artifact"),
        "promotion.memory_artifact",
    )
    proposal_payload = proposal_artifact.payload
    if proposal_payload.get("role") != "memory_update" or proposal_payload.get(
        "valid"
    ) is not True:
        raise TraceEvidenceError("memory proposal artifact is invalid")
    if proposal_payload.get("artifact_id") != f"attempt_{child['attempt']:02d}":
        raise TraceEvidenceError("memory proposal artifact_id does not match child")
    proposal = _mapping(
        proposal_payload.get("parsed_response"),
        "memory proposal parsed_response",
    )

    receipt = receipt_artifact.payload
    required = {
        "kind",
        "schema_version",
        "producer_id",
        "attempt",
        "script_id",
        "checkpoint_before",
        "checkpoint_after",
        "transition_outcome",
        "memory_kind",
        "record_sha256",
        "policy_sha256",
        "evaluation_sha256",
        "execution_trace_sha256",
        "promoted",
    }
    _require_exact_keys(receipt, required, "host memory promotion receipt")
    if (
        receipt.get("kind") != PROMOTION_RECEIPT_KIND
        or receipt.get("schema_version") != 1
        or receipt.get("producer_id") != "praxis_coop.host.memory_promotion"
        or receipt.get("promoted") is not True
    ):
        raise TraceEvidenceError("invalid host memory-promotion receipt")
    expected_values = {
        "attempt": child["attempt"],
        "script_id": child["script_id"],
        "checkpoint_before": child["checkpoint_before"],
        "checkpoint_after": child["checkpoint_after"],
        "transition_outcome": child["transition_outcome"],
        "policy_sha256": child["policy_sha256"],
        "evaluation_sha256": child["evaluation_sha256"],
        "execution_trace_sha256": child["execution_trace_sha256"],
    }
    for key, expected in expected_values.items():
        if receipt.get(key) != expected:
            raise TraceEvidenceError(f"memory promotion receipt {key} mismatch")
    memory_kind = receipt.get("memory_kind")
    collection = {
        "strategy_spec": "specs",
        "action_exemplar": "exemplars",
    }.get(memory_kind)
    if collection is None:
        raise TraceEvidenceError("unsupported promoted memory kind")
    record_sha256 = receipt.get("record_sha256")
    if not isinstance(record_sha256, str) or not _SHA256_RE.fullmatch(record_sha256):
        raise TraceEvidenceError("promotion record_sha256 is invalid")
    records = memory_artifact.payload.get(collection)
    if not isinstance(records, list):
        raise TraceEvidenceError("promoted memory artifact has wrong collection")
    matches = [
        item
        for item in records
        if isinstance(item, Mapping) and _semantic_sha256(item) == record_sha256
    ]
    if len(matches) != 1:
        raise TraceEvidenceError("promoted record is not uniquely present in memory")
    if not _memory_record_binds_child(matches[0], child):
        raise TraceEvidenceError("promoted record does not bind accepted child evidence")
    return {
        "proposal_artifact": proposal_artifact.public_ref(),
        "proposal": dict(proposal),
        "host_promotion_artifact": receipt_artifact.public_ref(),
        "host_promotion": dict(receipt),
        "memory_artifact": memory_artifact.public_ref(),
        "effective": True,
    }


def _validate_accepted_manifest(
    payload: Mapping[str, Any],
    *,
    attempt: int,
    script_id: str,
    checkpoint_before: str,
    checkpoint_after: str,
    outcome: str,
) -> None:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise TraceEvidenceError("accepted manifest must contain a steps list")
    matches = [
        item
        for item in steps
        if isinstance(item, Mapping)
        and item.get("attempt") == attempt
        and item.get("script_id") == script_id
        and item.get("checkpoint_before") == checkpoint_before
        and item.get("checkpoint_after") == checkpoint_after
        and item.get("transition_outcome") == outcome
    ]
    if len(matches) != 1:
        raise TraceEvidenceError("accepted manifest does not uniquely bind child")


def _memory_record_binds_child(
    record: Mapping[str, Any],
    child: Mapping[str, Any],
) -> bool:
    expected = {
        "attempt": child["attempt"],
        "script_id": child["script_id"],
        "checkpoint_before": child["checkpoint_before"],
        "checkpoint_after": child["checkpoint_after"],
    }
    evidence = record.get("evidence")
    if isinstance(evidence, Mapping):
        refs = evidence.get("rollout_refs")
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, Mapping) and all(
                    ref.get(key) == value for key, value in expected.items()
                ):
                    outcome = str(ref.get("outcome", "")).lower()
                    if outcome in {"accepted", child["transition_outcome"].lower()}:
                        return True
    source = record.get("source")
    if isinstance(source, Mapping) and all(
        source.get(key) == value for key, value in expected.items()
    ):
        return True
    return False


def _load_ref(root: Path, value: object, label: str) -> BoundJsonArtifact:
    try:
        return load_bound_json_artifact(root, value, label=label)
    except BoundArtifactError as exc:
        raise TraceEvidenceError(str(exc)) from exc


def _read_receipt(root: Path) -> tuple[dict[str, Any], str, int]:
    path = root / TRACE_EVIDENCE_FILENAME
    if root.is_symlink() or path.is_symlink():
        raise TraceEvidenceError("Figure-2 run/receipt symlinks are forbidden")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceEvidenceError("manifest-pinned Figure-2 trace receipt is required") from exc
    if not isinstance(payload, dict):
        raise TraceEvidenceError("Figure-2 trace receipt must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest(), len(raw)


def _read_root_json(root: Path, filename: str) -> dict[str, Any]:
    path = root / filename
    if root.is_symlink() or path.is_symlink():
        raise TraceEvidenceError(f"{filename} symlink is forbidden")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceEvidenceError(f"valid {filename} is required") from exc
    if not isinstance(payload, dict):
        raise TraceEvidenceError(f"{filename} must be a JSON object")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceEvidenceError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TraceEvidenceError(
            f"{label} fields mismatch: missing={missing}, extra={extra}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TraceEvidenceError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: object, label: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TraceEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _semantic_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


__all__ = [
    "TRACE_EVIDENCE_FILENAME",
    "TRACE_EVIDENCE_KIND",
    "TRACE_EVIDENCE_SCHEMA_VERSION",
    "TraceEvidenceError",
    "ValidatedTraceEvidence",
    "validate_trace_evidence",
]
