"""Frozen schema for declarative table protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..contracts import MemoryCondition, MethodVariant, SceneId, TableId
from ..provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE_ID,
    SECONDARY_GEMINI_PROVIDER_ROUTE_ID,
)


@dataclass(frozen=True)
class CommonProtocol:
    schema_version: int
    seeds: Tuple[int, ...]
    physical_budget: int
    target_backbone: str
    target_provider_route_id: str
    source_backbones: Tuple[str, ...]
    source_provider_route_ids: Tuple[str, ...]
    protocol_invariant_ref: str
    terminal_predicate_ref: str

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"unsupported common protocol schema {self.schema_version}")
        if self.seeds != (200, 201, 202, 203, 204):
            raise ValueError("the frozen paired protocol requires seeds 200..204")
        if self.physical_budget != 80:
            raise ValueError("the frozen physical budget must be 80")
        if not self.target_backbone:
            raise ValueError("target_backbone cannot be empty")
        if len(self.source_backbones) != 2 or len(set(self.source_backbones)) != 2:
            raise ValueError("exactly two distinct source backbones are required")
        if self.source_backbones[0] != self.target_backbone:
            raise ValueError("the same-model source must equal target_backbone")
        if self.target_provider_route_id != PRIMARY_GPT_PROVIDER_ROUTE_ID:
            raise ValueError("the target GPT route must be the frozen provider_a route")
        if self.source_provider_route_ids != (
            PRIMARY_GPT_PROVIDER_ROUTE_ID,
            SECONDARY_GEMINI_PROVIDER_ROUTE_ID,
        ):
            raise ValueError(
                "source provider routes must be the frozen provider_a GPT and provider_b Gemini routes"
            )
        if self.protocol_invariant_ref != "lock://invariant_fingerprint":
            raise ValueError("protocol invariant must resolve through the protocol lock")
        if self.terminal_predicate_ref != "lock://terminal_fingerprint":
            raise ValueError("terminal predicate must resolve through the protocol lock")


@dataclass(frozen=True)
class ProtocolCondition:
    condition_id: str
    label: str
    core: bool
    execution: str
    method_variant: MethodVariant
    scene_id: SceneId
    memory_condition: MemoryCondition
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.condition_id or not self.label:
            raise ValueError("protocol condition ID and label cannot be empty")
        if self.execution not in {"new", "reuse"}:
            raise ValueError("condition execution must be 'new' or 'reuse'")
        if self.execution == "reuse" and not self.source:
            raise ValueError("reused conditions require a source selector name")
        if self.execution == "new" and self.source:
            raise ValueError("new conditions cannot declare a reuse source")
        if self.memory_condition is MemoryCondition.NO_DYNAMIC:
            if self.method_variant is not MethodVariant.NO_DYNAMIC_EXPERIENCE:
                raise ValueError("NO_DYNAMIC requires the no-dynamic method variant")
        elif self.method_variant is MethodVariant.NO_DYNAMIC_EXPERIENCE:
            raise ValueError("the no-dynamic method must use the NO_DYNAMIC condition")
        if self.memory_condition in {
            MemoryCondition.GPT_TO_GPT,
            MemoryCondition.GEMINI_TO_GPT,
        } and self.method_variant is not MethodVariant.COEVOLE_FULL:
            raise ValueError("memory transfer is defined only for Full CoEvolve")


@dataclass(frozen=True)
class TableProtocol:
    schema_version: int
    table_id: TableId
    conditions: Tuple[ProtocolCondition, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"unsupported table protocol schema {self.schema_version}")
        if not self.conditions:
            raise ValueError("table protocol must contain at least one condition")
        ids = [condition.condition_id for condition in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate condition IDs in {self.table_id.value}")


@dataclass(frozen=True)
class ProtocolBundle:
    common: CommonProtocol
    tables: Tuple[TableProtocol, ...]

    def __post_init__(self) -> None:
        expected = {TableId.TABLE_2, TableId.TABLE_3, TableId.TABLE_4}
        actual = {table.table_id for table in self.tables}
        if actual != expected or len(self.tables) != len(expected):
            raise ValueError("protocol bundle must define Table 2, Table 3, and Table 4 once")

    def table(self, table_id: TableId) -> TableProtocol:
        for table in self.tables:
            if table.table_id is table_id:
                return table
        raise KeyError(table_id)


__all__ = [
    "CommonProtocol",
    "ProtocolBundle",
    "ProtocolCondition",
    "TableProtocol",
]
