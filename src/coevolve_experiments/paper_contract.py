"""Paper-facing contract for the experiments after the backbone table.

The declarative protocol files are executable inputs, while the manuscript is
the claim surface.  This module keeps a deliberately small, independently
checked map between the two so that adding, removing, or relabelling a YAML row
cannot silently change the registered paper experiment.

It does not inspect or launch physical results.  Formal execution remains the
responsibility of :mod:`coevolve_experiments.live`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

from .contracts import (
    MemoryCondition,
    MethodVariant,
    SceneId,
    SuiteTier,
    TableId,
)
from .protocols import ProtocolCondition, load_protocol_bundle


PAPER_CONTRACT_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class PaperMetric:
    """One reportable metric and the exact paper tables that display it.

    ``main_table`` is retained for compatibility and means "displayed by at
    least one main table".  Callers rendering a particular table must use its
    ``RegisteredTable.displayed_metrics`` projection instead.
    """

    metric_id: str
    direction: str
    main_table: bool = True
    displayed_in: Tuple[TableId, ...] = ()

    def __post_init__(self) -> None:
        if not self.metric_id:
            raise ValueError("paper metric ID cannot be empty")
        if self.direction not in {"up", "down", "report"}:
            raise ValueError(f"unsupported paper metric direction {self.direction!r}")
        if len(self.displayed_in) != len(set(self.displayed_in)):
            raise ValueError(f"duplicate table projection for metric {self.metric_id}")
        if self.main_table != bool(self.displayed_in):
            raise ValueError(
                f"metric {self.metric_id} main_table disagrees with displayed_in"
            )


@dataclass(frozen=True, slots=True)
class RegisteredRow:
    table_id: TableId
    condition_id: str
    label: str
    core: bool
    execution: str
    method_variant: MethodVariant
    scene_id: SceneId
    memory_condition: MemoryCondition
    source: Optional[str]


@dataclass(frozen=True, slots=True)
class RegisteredTable:
    table_id: TableId
    research_question: str
    rows: Tuple[RegisteredRow, ...]
    displayed_metrics: Tuple[str, ...]
    new_run_count: int
    reused_row_count: int


@dataclass(frozen=True, slots=True)
class PaperExperimentContract:
    schema_version: int
    tier: SuiteTier
    seeds: Tuple[int, ...]
    physical_budget: int
    target_backbone: str
    target_provider_route_id: str
    source_backbones: Tuple[str, ...]
    source_provider_route_ids: Tuple[str, ...]
    tables: Tuple[RegisteredTable, ...]
    metrics: Tuple[PaperMetric, ...]
    total_new_runs: int
    total_reused_rows: int

    def displayed_metrics_for(self, table_id: TableId) -> Tuple[PaperMetric, ...]:
        """Return the ordered metric catalogue projection for one paper table."""

        for table in self.tables:
            if table.table_id is table_id:
                wanted = set(table.displayed_metrics)
                return tuple(metric for metric in self.metrics if metric.metric_id in wanted)
        raise KeyError(table_id)


# Values here are claim identities, not a second implementation of the
# protocol.  Every field is compared with the loaded protocol below.
_EXPECTED_ROWS: Mapping[TableId, Tuple[tuple[object, ...], ...]] = {
    TableId.TABLE_2: (
        (
            "remac_adapted",
            "REMAC-Adapted",
            True,
            "new",
            MethodVariant.REMAC_ADAPTED,
            SceneId.BASE,
            MemoryCondition.EMPTY,
            None,
        ),
        (
            "coherent_adapted",
            "COHERENT-Adapted",
            False,
            "new",
            MethodVariant.COHERENT_ADAPTED,
            SceneId.BASE,
            MemoryCondition.EMPTY,
            None,
        ),
        (
            "flat_reflective_repair",
            "Flat Reflective Repair",
            True,
            "new",
            MethodVariant.FLAT_REFLECTIVE_REPAIR,
            SceneId.BASE,
            MemoryCondition.EMPTY,
            None,
        ),
        (
            "no_dynamic_experience",
            "CoEvolve w/o Dynamic Experience",
            True,
            "new",
            MethodVariant.NO_DYNAMIC_EXPERIENCE,
            SceneId.BASE,
            MemoryCondition.NO_DYNAMIC,
            None,
        ),
        (
            "coevolve_full",
            "Full CoEvolve",
            True,
            "reuse",
            MethodVariant.COEVOLE_FULL,
            SceneId.BASE,
            MemoryCondition.EMPTY,
            "table1_gpt_base",
        ),
    ),
    TableId.TABLE_3: (
        (
            "base",
            "Base",
            True,
            "reuse",
            MethodVariant.COEVOLE_FULL,
            SceneId.BASE,
            MemoryCondition.EMPTY,
            "table1_gpt_base",
        ),
        (
            "mirrored",
            "Mirrored",
            True,
            "new",
            MethodVariant.COEVOLE_FULL,
            SceneId.MIRRORED,
            MemoryCondition.EMPTY,
            None,
        ),
        (
            "narrow",
            "Narrow",
            True,
            "new",
            MethodVariant.COEVOLE_FULL,
            SceneId.NARROW,
            MemoryCondition.EMPTY,
            None,
        ),
        (
            "altered_topology",
            "Altered Topology",
            False,
            "new",
            MethodVariant.COEVOLE_FULL,
            SceneId.ALTERED_TOPOLOGY,
            MemoryCondition.EMPTY,
            None,
        ),
    ),
    TableId.TABLE_4: (
        (
            "empty",
            "Empty",
            True,
            "reuse",
            MethodVariant.COEVOLE_FULL,
            SceneId.MIRRORED,
            MemoryCondition.EMPTY,
            "table3_mirrored",
        ),
        (
            "gpt_to_gpt",
            "GPT to GPT",
            True,
            "new",
            MethodVariant.COEVOLE_FULL,
            SceneId.MIRRORED,
            MemoryCondition.GPT_TO_GPT,
            None,
        ),
        (
            "gemini_to_gpt",
            "Gemini to GPT",
            True,
            "new",
            MethodVariant.COEVOLE_FULL,
            SceneId.MIRRORED,
            MemoryCondition.GEMINI_TO_GPT,
            None,
        ),
    ),
}

_RESEARCH_QUESTIONS = {
    TableId.TABLE_2: "method_comparison_and_component_ablation",
    TableId.TABLE_3: "held_out_scene_generalization",
    TableId.TABLE_4: "same_model_reuse_and_cross_model_transfer",
}

_METRICS = (
    PaperMetric(
        "success_at_80",
        "up",
        displayed_in=(TableId.TABLE_2, TableId.TABLE_3, TableId.TABLE_4),
    ),
    PaperMetric(
        "capped_physical_transactions_at_80",
        "down",
        displayed_in=(TableId.TABLE_2, TableId.TABLE_3, TableId.TABLE_4),
    ),
    PaperMetric(
        "verified_phase_completion",
        "up",
        displayed_in=(TableId.TABLE_2, TableId.TABLE_3),
    ),
    PaperMetric(
        "rejected_physical_transactions",
        "down",
        displayed_in=(TableId.TABLE_2, TableId.TABLE_3, TableId.TABLE_4),
    ),
    PaperMetric(
        "foundation_model_calls",
        "down",
        displayed_in=(TableId.TABLE_2, TableId.TABLE_4),
    ),
    PaperMetric("input_tokens", "report", main_table=False),
    PaperMetric("output_tokens", "report", main_table=False),
    PaperMetric("wall_time_s", "report", main_table=False),
    PaperMetric("api_cost_usd", "report", main_table=False),
    PaperMetric("terminal_reason", "report", main_table=False),
    PaperMetric("accepted_setup_transactions", "report", main_table=False),
    PaperMetric("accepted_progress_transactions", "report", main_table=False),
    PaperMetric("infrastructure_invalid_attempts", "report", main_table=False),
)


def build_paper_contract(
    tier: Union[SuiteTier, str] = SuiteTier.FULL,
    *,
    protocol_root: Optional[Path] = None,
) -> PaperExperimentContract:
    """Validate the protocol rows against the manuscript contract.

    The check is intentionally exact, including row order and reuse source.
    This makes the emitted JSON suitable for review before a costly launch.
    """

    normalized_tier = tier if isinstance(tier, SuiteTier) else SuiteTier(tier)
    bundle = load_protocol_bundle(protocol_root)
    loaded_tables = {table.table_id: table for table in bundle.tables}
    tables = []
    total_new_runs = 0
    total_reused_rows = 0

    for table_id in (TableId.TABLE_2, TableId.TABLE_3, TableId.TABLE_4):
        table = loaded_tables[table_id]
        _require_registered_rows(table_id, table.conditions)
        selected = tuple(
            condition
            for condition in table.conditions
            if normalized_tier is SuiteTier.FULL or condition.core
        )
        rows = tuple(_registered_row(table_id, condition) for condition in selected)
        new_count = sum(row.execution == "new" for row in rows) * len(
            bundle.common.seeds
        )
        reuse_count = sum(row.execution == "reuse" for row in rows)
        total_new_runs += new_count
        total_reused_rows += reuse_count
        tables.append(
            RegisteredTable(
                table_id=table_id,
                research_question=_RESEARCH_QUESTIONS[table_id],
                rows=rows,
                displayed_metrics=tuple(
                    metric.metric_id
                    for metric in _METRICS
                    if table_id in metric.displayed_in
                ),
                new_run_count=new_count,
                reused_row_count=reuse_count,
            )
        )

    expected_new = 35 if normalized_tier is SuiteTier.CORE else 45
    if total_new_runs != expected_new:
        raise ValueError(
            f"paper {normalized_tier.value} tier requires {expected_new} new runs; "
            f"protocol declares {total_new_runs}"
        )
    if total_reused_rows != 3:
        raise ValueError("paper contract requires exactly three reference-only rows")

    return PaperExperimentContract(
        schema_version=PAPER_CONTRACT_SCHEMA_VERSION,
        tier=normalized_tier,
        seeds=bundle.common.seeds,
        physical_budget=bundle.common.physical_budget,
        target_backbone=bundle.common.target_backbone,
        target_provider_route_id=bundle.common.target_provider_route_id,
        source_backbones=bundle.common.source_backbones,
        source_provider_route_ids=bundle.common.source_provider_route_ids,
        tables=tuple(tables),
        metrics=_METRICS,
        total_new_runs=total_new_runs,
        total_reused_rows=total_reused_rows,
    )


def _registered_row(
    table_id: TableId, condition: ProtocolCondition
) -> RegisteredRow:
    return RegisteredRow(
        table_id=table_id,
        condition_id=condition.condition_id,
        label=condition.label,
        core=condition.core,
        execution=condition.execution,
        method_variant=condition.method_variant,
        scene_id=condition.scene_id,
        memory_condition=condition.memory_condition,
        source=condition.source,
    )


def _require_registered_rows(
    table_id: TableId, conditions: Tuple[ProtocolCondition, ...]
) -> None:
    actual = tuple(
        (
            row.condition_id,
            row.label,
            row.core,
            row.execution,
            row.method_variant,
            row.scene_id,
            row.memory_condition,
            row.source,
        )
        for row in conditions
    )
    expected = _EXPECTED_ROWS[table_id]
    if actual != expected:
        raise ValueError(
            f"{table_id.value} differs from the registered paper contract"
        )


__all__ = [
    "PAPER_CONTRACT_SCHEMA_VERSION",
    "PaperExperimentContract",
    "PaperMetric",
    "RegisteredRow",
    "RegisteredTable",
    "build_paper_contract",
]
