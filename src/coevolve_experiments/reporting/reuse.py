"""Hash-checked aliasing for the three reference-only table rows."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

from ..contracts import ReuseSpec
from ..serialization import sha256_json
from .collect import RunRecord


class ReuseResolutionError(RuntimeError):
    """Raised instead of copying or ambiguously selecting source results."""


def alias_reused_condition(
    reuse: ReuseSpec,
    records: Iterable[RunRecord],
    *,
    observed_hashes: Mapping[str, str] | None = None,
    require_complete: bool = True,
) -> tuple[RunRecord, ...]:
    """Validate one source cohort, then expose it under a target table label.

    Callers must obtain ``RunRecord`` values from the frozen source-version
    auditor.  This function performs the cross-table join only; it never copies
    result directories or reinterprets version-specific physical evidence.
    """

    rows = tuple(records)
    valid = tuple(row for row in rows if row.protocol_valid)
    selector = reuse.source_selector
    valid_seeds = {row.seed for row in valid}
    if len(valid_seeds) != len(valid):
        raise ReuseResolutionError("reuse source contains duplicate valid seeds")
    expected_seeds = set(selector.seeds)
    if not valid_seeds.issubset(expected_seeds):
        raise ReuseResolutionError("reuse source contains a seed outside its selector")
    if require_complete and (
        len(valid) != len(selector.seeds) or valid_seeds != expected_seeds
    ):
        raise ReuseResolutionError(
            "reuse source must contain exactly one protocol-valid row for "
            f"selector seeds {list(selector.seeds)}"
        )

    expected = {item.name: item.sha256 for item in reuse.expected_hashes}
    selector_fingerprint = sha256_json(
        {
            "source_table": selector.source_table,
            "protocol_id": selector.protocol_id,
            "method_commit": selector.method_commit,
            "backbone": selector.backbone,
            "method_variant": selector.method_variant.value,
            "scene_id": selector.scene_id.value,
            "memory_condition": selector.memory_condition.value,
            "seeds": selector.seeds,
            "provider_route_id": selector.provider_route_id,
            "provider_route_fingerprint": selector.provider_route_fingerprint,
            "full_method_compatibility_fingerprint": (
                selector.full_method_compatibility_fingerprint
            ),
            "output_relpath": selector.output_relpath,
            "external": selector.external,
        }
    )
    if expected.get("selector") != selector_fingerprint:
        raise ReuseResolutionError("reuse selector hash differs from the frozen specification")
    supplied = dict(observed_hashes or {})
    for name in ("result_manifest",):
        if name in expected and supplied.get(name) != expected[name]:
            raise ReuseResolutionError(
                f"reuse source {name} hash is missing or differs from the lock"
            )
    for name, value in supplied.items():
        if name in expected and value != expected[name]:
            raise ReuseResolutionError(f"reuse source {name} hash differs from the lock")

    checks = {
        "condition": (row.condition_fingerprint for row in valid),
        "invariant": (row.invariant_fingerprint for row in valid),
        "scene": (row.scene_fingerprint for row in valid),
        "terminal": (row.terminal_fingerprint for row in valid),
        "provider_route": (
            row.provider_route_fingerprint for row in valid
        ),
        "observation_contract": (
            row.observation_contract_fingerprint for row in valid
        ),
        "task_instruction": (
            row.task_instruction_fingerprint for row in valid
        ),
        "shared_prompt_contract": (
            row.shared_prompt_contract_fingerprint for row in valid
        ),
        "decision_prompt": (
            row.decision_prompt_fingerprint for row in valid
        ),
        "skill_api": (row.skill_api_fingerprint for row in valid),
        "full_method_compatibility": (
            row.full_method_compatibility_fingerprint for row in valid
        ),
    }
    for name, values in checks.items():
        if name not in expected:
            continue
        actual = set(values)
        if not valid:
            continue
        if actual != {expected[name]}:
            raise ReuseResolutionError(
                f"reuse source {name} fingerprints differ: {sorted(actual)}"
            )
    for row in valid:
        if row.table != selector.source_table:
            raise ReuseResolutionError("reuse source table differs")
        if row.protocol_id != selector.protocol_id:
            raise ReuseResolutionError("reuse source protocol_id differs")
        if row.base_method_commit != selector.method_commit:
            raise ReuseResolutionError("reuse source method commit differs")
        if row.target_model != selector.backbone:
            raise ReuseResolutionError("reuse source backbone differs")
        if row.provider_route_id != selector.provider_route_id:
            raise ReuseResolutionError("reuse source provider route ID differs")
        if row.provider_route_fingerprint != selector.provider_route_fingerprint:
            raise ReuseResolutionError(
                "reuse source provider route fingerprint differs"
            )
        if (
            row.full_method_compatibility_fingerprint
            != selector.full_method_compatibility_fingerprint
        ):
            raise ReuseResolutionError(
                "reuse source Full-method compatibility identity differs"
            )
        if row.method != selector.method_variant.value:
            raise ReuseResolutionError("reuse source method differs")
        if row.scene != selector.scene_id.value:
            raise ReuseResolutionError("reuse source scene differs")
        if row.memory_condition != selector.memory_condition.value:
            raise ReuseResolutionError("reuse source memory condition differs")

    target_condition = reuse.reuse_id.rsplit("__", 1)[-1]
    return tuple(
        replace(
            row,
            table=reuse.target_table.value,
            condition=target_condition,
            run_dir=row.run_dir,
        )
        for row in sorted(valid, key=lambda item: int(item.seed or 0))
    )


__all__ = ["ReuseResolutionError", "alias_reused_condition"]
