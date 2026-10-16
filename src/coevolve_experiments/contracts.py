"""Immutable contracts for the post-Table-1 experiment matrix.

This package deliberately contains experiment *specifications*, not another
copy of the main method.  The runner consumes these records through an adapter;
the records themselves have no dependency on ``praxis_coop`` or the private
formal-evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
import re
from typing import Iterable, Optional, Tuple

from .full_method_compatibility import FullMethodCompatibilityPin
from .provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE_ID,
    SECONDARY_GEMINI_PROVIDER_ROUTE_ID,
    require_provider_route,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class _StringEnum(str, Enum):
    """``StrEnum``-like base that remains compatible with Python 3.10."""

    def __str__(self) -> str:
        return self.value


class TableId(_StringEnum):
    TABLE_2 = "table2_method_comparison"
    TABLE_3 = "table3_scene_generalization"
    TABLE_4 = "table4_memory_transfer"

    # Readable aliases for callers that prefer unseparated table numbers.
    TABLE2 = TABLE_2
    TABLE3 = TABLE_3
    TABLE4 = TABLE_4


class SuiteTier(_StringEnum):
    """Cost tiers fixed before collection."""

    CORE = "core"
    FULL = "full"


class MethodVariant(_StringEnum):
    COEVOLE_FULL = "coevolve_full"
    NO_DYNAMIC_EXPERIENCE = "no_dynamic_experience"
    FLAT_REFLECTIVE_REPAIR = "flat_reflective_repair"
    REMAC_ADAPTED = "remac_adapted"
    COHERENT_ADAPTED = "coherent_adapted"


class SceneId(_StringEnum):
    BASE = "base"
    MIRRORED = "mirrored"
    NARROW = "narrow"
    ALTERED_TOPOLOGY = "altered_topology"


class MemoryCondition(_StringEnum):
    """Initial/online memory condition of a target run.

    ``EMPTY`` describes the shared *initial* condition.  Full/flat CoEvolve
    conditions may then acquire verifier-gated run-local experience, whereas
    REMAC/COHERENT keep durable cooperative memory disabled.  ``NO_DYNAMIC``
    is the Table-2 ablation and disables only dynamic strategy/action update
    and retrieval; verifier, checkpoints, rollback, manifests, and
    checkpoint-local failure state remain enabled.
    """

    NO_DYNAMIC = "no_dynamic"
    EMPTY = "empty"
    GPT_TO_GPT = "gpt_to_gpt"
    GEMINI_TO_GPT = "gemini_to_gpt"

    # Claim-oriented aliases without creating additional conditions.
    SAME_MODEL_REUSE = GPT_TO_GPT
    CROSS_MODEL_TRANSFER = GEMINI_TO_GPT


class MemoryPolicy(_StringEnum):
    """Durable strategy/action policy only; safety/audit state is never gated."""

    DISABLED = "disabled"
    ONLINE_FROM_EMPTY = "online_from_empty"
    ONLINE_FROM_PRELOADED_STRATEGIES = "online_from_preloaded_strategies"


class ReuseMode(_StringEnum):
    """Reuse is a reference/join operation, never a result copy."""

    REFERENCE_ONLY = "reference_only"


@dataclass(frozen=True)
class MethodDecisionPromptPin:
    """One method's decision prompt, separate from the shared prompt contract."""

    method_variant: MethodVariant
    decision_prompt_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.method_variant, MethodVariant):
            raise ValueError("decision prompt method_variant must be a MethodVariant")
        _require_sha256(
            f"{self.method_variant.value} decision_prompt_fingerprint",
            self.decision_prompt_fingerprint,
        )


@dataclass(frozen=True)
class Table1SourceCohort:
    """Frozen identity of one five-run Table-1 source cohort.

    Table-1 GPT and the provider-only Gemini companion may have different
    protocol IDs and commits.  Keeping their identities in separate records
    prevents a post-Table-1 extension (or one provider companion) from being
    silently relabelled as the other source.
    """

    backbone: str
    protocol_id: str
    method_commit: str
    seeds: Tuple[int, ...]
    condition_fingerprint: str
    result_manifest_hash: str
    provider_route_id: str
    provider_route_fingerprint: str
    shared_prompt_contract_fingerprint: str
    decision_prompt_fingerprint: str
    full_method_compatibility_fingerprint: str

    def __post_init__(self) -> None:
        _require_identifier("Table-1 source backbone", self.backbone)
        _require_identifier("Table-1 source protocol_id", self.protocol_id)
        _require_git_commit("Table-1 source method_commit", self.method_commit)
        _validate_five_seed_cohort(self.seeds, label="Table-1 source cohort")
        _require_sha256(
            "Table-1 source condition_fingerprint",
            self.condition_fingerprint,
        )
        _require_sha256(
            "Table-1 source result_manifest_hash",
            self.result_manifest_hash,
        )
        require_provider_route(
            self.provider_route_id,
            self.provider_route_fingerprint,
        )
        _require_sha256(
            "Table-1 source shared_prompt_contract_fingerprint",
            self.shared_prompt_contract_fingerprint,
        )
        _require_sha256(
            "Table-1 source decision_prompt_fingerprint",
            self.decision_prompt_fingerprint,
        )
        _require_sha256(
            "Table-1 source full_method_compatibility_fingerprint",
            self.full_method_compatibility_fingerprint,
        )


@dataclass(frozen=True)
class ProtocolLock:
    """Hashes that must be frozen before building a runnable matrix.

    The Table-1 result and the two route-abstracted strategy banks are external
    inputs to these tables, so their content hashes are explicit rather than
    silently discovered by directory name.  Observation, instruction, prompt,
    and SONIC skill-API fingerprints are also explicit because live launchers
    may configure them outside the pinned repository commit.  A lock can be
    drafted with real hashes only after those artifacts exist; fabricated
    placeholder hashes are intentionally not supported.
    """

    protocol_id: str
    invariant_fingerprint: str
    terminal_fingerprint: str
    base_method_commit: str
    table1_gpt_source: Table1SourceCohort
    table1_gemini_source: Table1SourceCohort
    gpt_strategy_bank_hash: str
    gemini_strategy_bank_hash: str
    target_provider_route_id: str
    target_provider_route_fingerprint: str
    observation_contract_fingerprint: str
    task_instruction_fingerprint: str
    shared_prompt_contract_fingerprint: str
    method_decision_prompt_pins: Tuple[MethodDecisionPromptPin, ...]
    skill_api_fingerprint: str
    full_method_compatibility: FullMethodCompatibilityPin

    def __post_init__(self) -> None:
        _require_identifier("protocol_id", self.protocol_id)
        _require_git_commit("base_method_commit", self.base_method_commit)
        if not isinstance(self.table1_gpt_source, Table1SourceCohort):
            raise ValueError("table1_gpt_source must be a Table1SourceCohort")
        if not isinstance(self.table1_gemini_source, Table1SourceCohort):
            raise ValueError("table1_gemini_source must be a Table1SourceCohort")
        if not isinstance(
            self.full_method_compatibility, FullMethodCompatibilityPin
        ):
            raise ValueError(
                "full_method_compatibility must be a FullMethodCompatibilityPin"
            )
        if self.table1_gpt_source.backbone == self.table1_gemini_source.backbone:
            raise ValueError(
                "GPT and Gemini Table-1 sources must use different backbones"
            )
        require_provider_route(
            self.target_provider_route_id,
            self.target_provider_route_fingerprint,
            expected_route_id=PRIMARY_GPT_PROVIDER_ROUTE_ID,
        )
        if (
            self.table1_gpt_source.provider_route_id
            != self.target_provider_route_id
            or self.table1_gpt_source.provider_route_fingerprint
            != self.target_provider_route_fingerprint
        ):
            raise ValueError(
                "Table-1 GPT reuse and new GPT runs must use one provider route"
            )
        if (
            self.table1_gemini_source.provider_route_id
            != SECONDARY_GEMINI_PROVIDER_ROUTE_ID
        ):
            raise ValueError("Table-1 Gemini source must use the frozen provider_b route")
        pins = self.method_decision_prompt_pins
        if not isinstance(pins, tuple) or not all(
            isinstance(pin, MethodDecisionPromptPin) for pin in pins
        ):
            raise ValueError(
                "method_decision_prompt_pins must be a tuple of MethodDecisionPromptPin"
            )
        variants = tuple(pin.method_variant for pin in pins)
        if len(variants) != len(set(variants)):
            raise ValueError("method decision prompt variants must be unique")
        if set(variants) != set(MethodVariant):
            raise ValueError("one decision prompt fingerprint is required per method")
        full_prompt = self.decision_prompt_fingerprint_for(
            MethodVariant.COEVOLE_FULL
        )
        for label, source in (
            ("GPT", self.table1_gpt_source),
            ("Gemini", self.table1_gemini_source),
        ):
            if (
                source.shared_prompt_contract_fingerprint
                != self.shared_prompt_contract_fingerprint
            ):
                raise ValueError(
                    f"Table-1 {label} shared prompt contract differs from the lock"
                )
            if source.decision_prompt_fingerprint != full_prompt:
                raise ValueError(
                    f"Table-1 {label} Full decision prompt differs from the lock"
                )
        if (
            self.table1_gpt_source.full_method_compatibility_fingerprint
            != self.full_method_compatibility.fingerprint
        ):
            raise ValueError(
                "Table-1 GPT Full-method compatibility identity differs from the lock"
            )
        if (
            full_prompt
            != self.full_method_compatibility.prompt_sha256
        ):
            raise ValueError(
                "Full decision prompt differs from the Full-method prompt bytes"
            )
        if self.skill_api_fingerprint != self.full_method_compatibility.skill_api_sha256:
            raise ValueError(
                "skill API fingerprint differs from the Full-method skill bytes"
            )
        for name in (
            "invariant_fingerprint",
            "terminal_fingerprint",
            "gpt_strategy_bank_hash",
            "gemini_strategy_bank_hash",
            "observation_contract_fingerprint",
            "task_instruction_fingerprint",
            "shared_prompt_contract_fingerprint",
            "skill_api_fingerprint",
        ):
            _require_sha256(name, getattr(self, name))

    def decision_prompt_fingerprint_for(
        self, method_variant: MethodVariant
    ) -> str:
        selected = tuple(
            pin.decision_prompt_fingerprint
            for pin in self.method_decision_prompt_pins
            if pin.method_variant is method_variant
        )
        if len(selected) != 1:
            raise ValueError(
                f"decision prompt pin does not resolve once for {method_variant.value}"
            )
        return selected[0]


@dataclass(frozen=True)
class HashExpectation:
    """One named digest required when resolving a reused row."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        _require_identifier("hash expectation name", self.name)
        _require_sha256(self.name, self.sha256)


@dataclass(frozen=True)
class SourceSelector:
    """Semantic selector for one already-collected five-seed condition."""

    source_table: str
    protocol_id: str
    method_commit: str
    backbone: str
    method_variant: MethodVariant
    scene_id: SceneId
    memory_condition: MemoryCondition
    seeds: Tuple[int, ...]
    provider_route_id: str
    provider_route_fingerprint: str
    full_method_compatibility_fingerprint: str
    output_relpath: str = ""
    external: bool = False

    def __post_init__(self) -> None:
        _require_identifier("source_table", self.source_table)
        _require_identifier("protocol_id", self.protocol_id)
        _require_git_commit("source method_commit", self.method_commit)
        _require_identifier("backbone", self.backbone)
        _validate_five_seed_cohort(self.seeds, label="source selector")
        require_provider_route(
            self.provider_route_id,
            self.provider_route_fingerprint,
        )
        _require_sha256(
            "source full_method_compatibility_fingerprint",
            self.full_method_compatibility_fingerprint,
        )
        if self.output_relpath:
            _require_relative_path("source output_relpath", self.output_relpath)


@dataclass(frozen=True)
class StrategyBankPin:
    """Pinned, route-abstracted strategy input for a Table-4 target run."""

    source_backbone: str
    source_protocol_id: str
    source_method_commit: str
    source_seeds: Tuple[int, ...]
    source_condition_fingerprint: str
    source_result_manifest_hash: str
    source_provider_route_id: str
    source_provider_route_fingerprint: str
    selector: str
    sha256: str

    def __post_init__(self) -> None:
        _require_identifier("source_backbone", self.source_backbone)
        _require_identifier("source_protocol_id", self.source_protocol_id)
        _require_git_commit("source_method_commit", self.source_method_commit)
        _validate_five_seed_cohort(self.source_seeds, label="strategy-bank source")
        _require_sha256(
            "source_condition_fingerprint", self.source_condition_fingerprint
        )
        _require_sha256(
            "source_result_manifest_hash", self.source_result_manifest_hash
        )
        require_provider_route(
            self.source_provider_route_id,
            self.source_provider_route_fingerprint,
        )
        _require_identifier("strategy bank selector", self.selector)
        _require_sha256("strategy bank", self.sha256)


@dataclass(frozen=True)
class RunSpec:
    """One new physical run.

    The condition fingerprint intentionally excludes ``seed`` and output path;
    all five repetitions of a condition therefore carry the same semantic pin.
    """

    run_id: str
    table_id: TableId
    seed: int
    method_variant: MethodVariant
    scene_id: SceneId
    memory_condition: MemoryCondition
    memory_policy: MemoryPolicy
    target_backbone: str
    provider_route_id: str
    provider_route_fingerprint: str
    physical_budget: int
    condition_fingerprint: str
    invariant_fingerprint: str
    base_method_commit: str
    protocol_id: str
    output_relpath: str
    scene_fingerprint: str
    terminal_fingerprint: str
    observation_contract_fingerprint: str
    task_instruction_fingerprint: str
    shared_prompt_contract_fingerprint: str
    decision_prompt_fingerprint: str
    skill_api_fingerprint: str
    full_method_compatibility_fingerprint: str
    strategy_bank: Optional[StrategyBankPin] = None

    def __post_init__(self) -> None:
        _require_identifier("run_id", self.run_id)
        _require_identifier("target_backbone", self.target_backbone)
        require_provider_route(
            self.provider_route_id,
            self.provider_route_fingerprint,
            expected_route_id=PRIMARY_GPT_PROVIDER_ROUTE_ID,
        )
        _require_git_commit("base_method_commit", self.base_method_commit)
        _require_identifier("protocol_id", self.protocol_id)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.physical_budget, bool)
            or not isinstance(self.physical_budget, int)
            or self.physical_budget <= 0
        ):
            raise ValueError("physical_budget must be a positive integer")
        _require_relative_path("output_relpath", self.output_relpath)
        for name in (
            "condition_fingerprint",
            "invariant_fingerprint",
            "scene_fingerprint",
            "terminal_fingerprint",
            "observation_contract_fingerprint",
            "task_instruction_fingerprint",
            "shared_prompt_contract_fingerprint",
            "decision_prompt_fingerprint",
            "skill_api_fingerprint",
            "full_method_compatibility_fingerprint",
        ):
            _require_sha256(name, getattr(self, name))
        if self.memory_condition is MemoryCondition.NO_DYNAMIC:
            if self.method_variant is not MethodVariant.NO_DYNAMIC_EXPERIENCE:
                raise ValueError("NO_DYNAMIC is reserved for the no-dynamic-experience ablation")
            if self.memory_policy is not MemoryPolicy.DISABLED:
                raise ValueError("NO_DYNAMIC requires the disabled memory policy")
            if self.strategy_bank is not None:
                raise ValueError("NO_DYNAMIC cannot preload a strategy bank")
        elif self.memory_condition is MemoryCondition.EMPTY:
            if self.strategy_bank is not None:
                raise ValueError("EMPTY cannot preload a strategy bank")
            disabled_baselines = {
                MethodVariant.REMAC_ADAPTED,
                MethodVariant.COHERENT_ADAPTED,
            }
            if self.method_variant in disabled_baselines:
                if self.memory_policy is not MemoryPolicy.DISABLED:
                    raise ValueError("REMAC/COHERENT baselines cannot enable durable memory")
            elif self.memory_policy is not MemoryPolicy.ONLINE_FROM_EMPTY:
                raise ValueError("full/flat EMPTY conditions require online-from-empty memory")
        else:
            if self.method_variant is not MethodVariant.COEVOLE_FULL:
                raise ValueError("strategy-bank transfer is defined only for Full CoEvolve")
            if self.memory_policy is not MemoryPolicy.ONLINE_FROM_PRELOADED_STRATEGIES:
                raise ValueError("transfer conditions require preloaded-strategy memory")
            if self.strategy_bank is None:
                raise ValueError("transfer conditions require a pinned strategy bank")


@dataclass(frozen=True)
class ReuseSpec:
    """A table row resolved by joining to an existing condition.

    ``mode`` is fixed to ``REFERENCE_ONLY`` so downstream tools cannot turn
    reuse into copied result directories that later drift from their source.
    """

    reuse_id: str
    target_table: TableId
    target_label: str
    source_selector: SourceSelector
    expected_hashes: Tuple[HashExpectation, ...]
    mode: ReuseMode = field(default=ReuseMode.REFERENCE_ONLY, init=False)

    def __post_init__(self) -> None:
        _require_identifier("reuse_id", self.reuse_id)
        _require_identifier("target_label", self.target_label)
        if not self.expected_hashes:
            raise ValueError("reuse must pin at least one expected hash")
        names = [item.name for item in self.expected_hashes]
        if len(names) != len(set(names)):
            raise ValueError("reuse expected-hash names must be unique")
        required = {
            "condition",
            "invariant",
            "scene",
            "terminal",
            "selector",
            "provider_route",
            "observation_contract",
            "task_instruction",
            "shared_prompt_contract",
            "decision_prompt",
            "skill_api",
            "full_method_compatibility",
        }
        if self.source_selector.external:
            required.add("result_manifest")
        missing = required.difference(names)
        if missing:
            raise ValueError(f"reuse is missing hash pins: {sorted(missing)}")

    @property
    def copies_results(self) -> bool:
        return False


@dataclass(frozen=True)
class RunPlan:
    """New unique runs plus hash-pinned reused rows."""

    tier: SuiteTier
    protocol_id: str
    new_runs: Tuple[RunSpec, ...]
    reuses: Tuple[ReuseSpec, ...]

    def __post_init__(self) -> None:
        _require_identifier("protocol_id", self.protocol_id)
        run_ids = [run.run_id for run in self.new_runs]
        outputs = [run.output_relpath for run in self.new_runs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("new run IDs must be unique")
        if len(outputs) != len(set(outputs)):
            raise ValueError("new run output paths must be unique")
        if any(run.protocol_id != self.protocol_id for run in self.new_runs):
            raise ValueError("all runs must use the plan protocol_id")
        reuse_ids = [reuse.reuse_id for reuse in self.reuses]
        if len(reuse_ids) != len(set(reuse_ids)):
            raise ValueError("reuse IDs must be unique")

    @property
    def unique_run_count(self) -> int:
        return len(self.new_runs)

    @property
    def new_unique_runs(self) -> Tuple[RunSpec, ...]:
        return self.new_runs

    @property
    def reuse_specs(self) -> Tuple[ReuseSpec, ...]:
        return self.reuses

    @property
    def runs(self) -> Tuple[RunSpec, ...]:
        """Compatibility/readability alias; reused rows are not runs."""

        return self.new_runs


@dataclass(frozen=True)
class ExperimentSuite:
    """Top-level return value of :func:`registry.build_suite`."""

    protocol_lock: ProtocolLock
    run_plan: RunPlan

    def __post_init__(self) -> None:
        if self.protocol_lock.protocol_id != self.run_plan.protocol_id:
            raise ValueError("protocol lock and run plan IDs differ")

    @property
    def tier(self) -> SuiteTier:
        return self.run_plan.tier

    @property
    def new_runs(self) -> Tuple[RunSpec, ...]:
        return self.run_plan.new_runs

    @property
    def reuses(self) -> Tuple[ReuseSpec, ...]:
        return self.run_plan.reuses

    @property
    def unique_run_count(self) -> int:
        return self.run_plan.unique_run_count


def hash_expectations(items: Iterable[tuple[str, str]]) -> Tuple[HashExpectation, ...]:
    """Build an immutable expectation tuple without exposing mutable mappings."""

    return tuple(HashExpectation(name=name, sha256=value) for name, value in items)


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_git_commit(name: str, value: str) -> None:
    if not isinstance(value, str) or not _GIT_COMMIT_RE.fullmatch(value):
        raise ValueError(f"{name} must be a full lowercase 40-character Git commit")


def _require_relative_path(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{name} must be a normalized POSIX relative path")


def _validate_five_seed_cohort(seeds: Tuple[int, ...], *, label: str) -> None:
    if not seeds:
        raise ValueError(f"{label} seeds cannot be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError(f"{label} seeds must be non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{label} seeds must be unique")
    if len(seeds) != 5:
        raise ValueError(f"{label} must contain exactly five unique seeds")


__all__ = [
    "ExperimentSuite",
    "HashExpectation",
    "MemoryCondition",
    "MemoryPolicy",
    "MethodDecisionPromptPin",
    "MethodVariant",
    "ProtocolLock",
    "ReuseMode",
    "ReuseSpec",
    "RunPlan",
    "RunSpec",
    "SceneId",
    "SourceSelector",
    "StrategyBankPin",
    "SuiteTier",
    "Table1SourceCohort",
    "TableId",
    "hash_expectations",
]
