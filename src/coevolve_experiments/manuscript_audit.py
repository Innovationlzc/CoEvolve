"""Read-only consistency audit for the executable Tables 2--4 contract.

The protocol YAML and the paper source are independent review surfaces.  This
module deliberately checks both without launching a model, importing the V83
runner, or writing an experiment artifact.  It is narrow by design: only the
claims needed to make Tables 2--4 coding-ready are registered here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Mapping, Optional, Tuple

from .contracts import TableId
from .paper_contract import PaperExperimentContract, build_paper_contract
from .reporting.terminal_evidence import (
    MAX_TERMINAL_PAYLOAD_TILT_DEG,
    ROBOT_EXIT_MIN_DISTANCE_M,
    TARGET_SUPPORT_RADIUS_M,
    TARGET_Z_TOLERANCE_M,
    TERMINAL_STABLE_WINDOW_S,
)


MANUSCRIPT_PROTOCOL_AUDIT_SCHEMA_VERSION = 1
FROZEN_COORDINATION_MODE_TOKENS = ("sync", "async")
FROZEN_SEEDS = (200, 201, 202, 203, 204)
FROZEN_PHYSICAL_BUDGET = 80
FROZEN_NEW_RUN_COUNT = 45
FROZEN_REUSED_ROW_COUNT = 3


@dataclass(frozen=True, slots=True)
class FrozenTerminalClaim:
    target_support_radius_m: float = 0.30
    target_z_tolerance_m: float = 0.08
    max_payload_tilt_deg: float = 20.0
    stable_window_s: float = 3.0
    robot_exit_min_distance_m: float = 1.10


FROZEN_TERMINAL_CLAIM = FrozenTerminalClaim()


@dataclass(frozen=True, slots=True)
class TableMetricProjection:
    table_id: TableId
    latex_label: str
    displayed_metrics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ManuscriptProtocolAudit:
    schema_version: int
    methodology_path: Path
    experiments_path: Path
    methodology_sha256: str
    experiments_sha256: str
    coordination_mode_tokens: Tuple[str, ...]
    seeds: Tuple[int, ...]
    physical_budget: int
    terminal: FrozenTerminalClaim
    total_new_runs: int
    total_reused_rows: int
    table_metric_projections: Tuple[TableMetricProjection, ...]
    errors: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_ok(self) -> None:
        if not self.ok:
            raise ManuscriptProtocolAuditError(self)


class ManuscriptProtocolAuditError(ValueError):
    def __init__(self, report: ManuscriptProtocolAudit):
        self.report = report
        super().__init__(
            "Tables 2--4 manuscript/protocol audit failed: "
            + "; ".join(report.errors)
        )


_TABLE_LABELS: Mapping[TableId, str] = {
    TableId.TABLE_2: "tab:baselines_ablations",
    TableId.TABLE_3: "tab:scene_generalization",
    TableId.TABLE_4: "tab:memory_transfer",
}

_METRIC_HEADER_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "success_at_80": re.compile(
        r"(?:\\textbf\{Success@80\}|"
        r"\\textbf\{Succ\.\}.*?\\textbf\{/\s*5\})",
        re.DOTALL,
    ),
    "capped_physical_transactions_at_80": re.compile(
        r"(?:\\textbf\{CPT@80\}|"
        r"\\textbf\{CPT\}.*?\\textbf\{@80\})",
        re.DOTALL,
    ),
    "verified_phase_completion": re.compile(
        r"\\textbf\{Phase\}.*?\\textbf\{(?:completion|comp\.)\}",
        re.DOTALL,
    ),
    "rejected_physical_transactions": re.compile(r"\\textbf\{Rejected\}"),
    "foundation_model_calls": re.compile(r"\\textbf\{FM calls\}"),
}

_COORDINATION_DECLARATION = re.compile(
    r"m_t\s*\\in\s*\\\{\s*"
    r"\\texttt\{([^{}]+)\}\s*,\s*"
    r"\\texttt\{([^{}]+)\}\s*\\\}",
    re.DOTALL,
)
_SEED_RANGE = re.compile(
    r"\\texttt\{seed\\?_(\d+)\}\s*\$\s*--\s*\$\s*"
    r"\\texttt\{seed\\?_(\d+)\}",
    re.DOTALL,
)
_PHYSICAL_BUDGET_DECLARATION = re.compile(r"B\s*=\s*(\d+)")

# These checks cover method-defining claims that are not recoverable from a
# table header or a numeric terminal threshold.  They intentionally match
# semantic phrases inside the two reviewed sections instead of accepting a
# caller-authored checklist.  Whitespace is normalized before matching so
# harmless TeX line wrapping does not invalidate the audit.
_METHODOLOGY_CLAIMS: Tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "METHODOLOGY_TWO_ROBOT_SCOPE_MISSING",
        re.compile(r"with\s+\$N=2\$\s+in\s+our\s+implementation", re.IGNORECASE),
    ),
    (
        "METHODOLOGY_COORDINATION_PRIMITIVES_MISSING",
        re.compile(
            r"\\\{k_\{\\mathrm\{linear\}\},k_\{\\mathrm\{pivot\}\},"
            r"k_\{\\mathrm\{align\}\}\\\}"
        ),
    ),
    (
        "METHODOLOGY_ATOMIC_TEAM_TRANSACTION_MISSING",
        re.compile(
            r"each\s+robot\s+policy\s+contains\s+one\s+task-changing\s+command\s+"
            r"followed\s+by\s+an\s+explicit\s+stop,\s+and\s+all\s+robot\s+commands\s+"
            r"are\s+compiled\s+into\s+one\s+team\s+transaction",
            re.IGNORECASE,
        ),
    ),
    (
        "METHODOLOGY_CHECKPOINT_ROLLBACK_MISSING",
        re.compile(
            r"rejected\s+transaction\s+is\s+restored\s+to\s+the\s+latest\s+"
            r"accepted\s+checkpoint",
            re.IGNORECASE,
        ),
    ),
    (
        "METHODOLOGY_CHECKPOINT_LOCAL_REPLAY_BLOCK_MISSING",
        re.compile(
            r"exact\s+previously\s+rejected\s+transactions\s+are\s+blocked\s+"
            r"at\s+the\s+same\s+checkpoint",
            re.IGNORECASE,
        ),
    ),
    (
        "METHODOLOGY_NON_EMBEDDING_RETRIEVAL_MISSING",
        re.compile(r"frozen\s+V83\s+non-embedding\s+retrieval\s+rule", re.IGNORECASE),
    ),
    (
        "METHODOLOGY_STRATEGY_TOP3_MISSING",
        re.compile(r"query\s+returns\s+at\s+most\s+three\s+records", re.IGNORECASE),
    ),
    (
        "METHODOLOGY_TRANSFER_BANK_CAP_MISSING",
        re.compile(
            r"at\s+most\s+48\s+records,\s+including\s+at\s+most\s+16\s+"
            r"verifier-backed\s+rejected\s+counterexamples",
            re.IGNORECASE,
        ),
    ),
    (
        "METHODOLOGY_ACTION_TOP2_MISSING",
        re.compile(
            r"at\s+most\s+two\s+complete\s+executable\s+exemplars",
            re.IGNORECASE,
        ),
    ),
    (
        "METHODOLOGY_VERIFIER_GATED_MEMORY_MISSING",
        re.compile(r"memory\s+updates\s+are\s+verifier\s+gated", re.IGNORECASE),
    ),
)

_EXPERIMENT_CLAIMS: Tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EXPERIMENT_FROM_SCRATCH_MEMORY_MISSING",
        re.compile(
            r"every\s+run\s+begins\s+from\s+the\s+initial\s+physical\s+state\s+"
            r"with\s+empty\s+strategy\s+memory,\s+empty\s+action-exemplar\s+memory",
            re.IGNORECASE,
        ),
    ),
    (
        "EXPERIMENT_NO_DYNAMIC_WRITER_GATE_MISSING",
        re.compile(r"makes\s+no\s+Memory\s+Writer\s+model\s+call", re.IGNORECASE),
    ),
    (
        "EXPERIMENT_FIVE_DROP_TERMINAL_MISSING",
        re.compile(
            r"after\s+five\s+consecutive\s+verified\s+payload\s+drops\s+from\s+"
            r"the\s+same\s+trusted\s+checkpoint",
            re.IGNORECASE,
        ),
    ),
    (
        "EXPERIMENT_THREE_CYCLE_STALL_MISSING",
        re.compile(r"frozen\s+three-cycle\s+generation-recovery\s+budget", re.IGNORECASE),
    ),
    (
        "EXPERIMENT_NO_INTERVENTION_MISSING",
        re.compile(
            r"no\s+human\s+selects\s+actions,\s+branches,\s+checkpoints,\s+or\s+"
            r"memories\s+during\s+a\s+formal\s+run",
            re.IGNORECASE,
        ),
    ),
    (
        "EXPERIMENT_TRANSFER_CONTENT_SCOPE_MISSING",
        re.compile(
            r"we\s+do\s+not\s+preload\s+checkpoints,\s+accepted-episode\s+"
            r"manifests,\s+local\s+failure\s+caches,\s+or\s+scene-specific\s+"
            r"action\s+exemplars",
            re.IGNORECASE,
        ),
    ),
)

_BACKBONE_TOKENS = (
    "gpt-5.4-mini-2026-03-17",
    "gemini-3.5-flash",
    "claude-sonnet-4-6",
)

_VALID_TERMINAL_TOKENS = (
    r"\texttt{SUCCESS}",
    r"\texttt{SYSTEM\_STALL}",
    r"\texttt{PHYSICAL\_BUDGET\_EXHAUSTED}",
    r"\texttt{CONSECUTIVE\_PAYLOAD\_DROP\_LIMIT}",
)

_INVALID_TERMINAL_TOKENS = (
    r"\texttt{PROVIDER\_FATAL}",
    r"\texttt{BACKEND\_FATAL}",
    r"\texttt{HOST\_CONTRACT\_ERROR}",
)


def audit_manuscript_protocol(
    *,
    methodology_path: Optional[Path] = None,
    experiments_path: Optional[Path] = None,
    protocol_root: Optional[Path] = None,
) -> ManuscriptProtocolAudit:
    """Audit the paper source against the full registered Tables 2--4 matrix.

    The returned report contains every mismatch.  Call :meth:`require_ok` at a
    preflight boundary.  Keeping collection and failure separate makes this
    useful in CI and review tools without turning an audit into a launcher.
    """

    repo_root = Path(__file__).resolve().parents[2]
    method_path = (
        Path(methodology_path)
        if methodology_path is not None
        else repo_root / "source/CoEvolve/sections/3_methodology.tex"
    )
    experiment_path = (
        Path(experiments_path)
        if experiments_path is not None
        else repo_root / "source/CoEvolve/sections/4_experiments.tex"
    )
    methodology_bytes = method_path.read_bytes()
    experiments_bytes = experiment_path.read_bytes()
    methodology = methodology_bytes.decode("utf-8")
    experiments = experiments_bytes.decode("utf-8")
    contract = build_paper_contract("full", protocol_root=protocol_root)
    errors: list[str] = []

    _audit_required_claims(methodology, _METHODOLOGY_CLAIMS, errors)
    _audit_required_claims(experiments, _EXPERIMENT_CLAIMS, errors)
    _audit_backbones(experiments, contract, errors)
    _audit_terminal_vocabulary(experiments, errors)
    _audit_registered_row_labels(experiments, contract, errors)

    coordination_modes = _extract_coordination_modes(methodology, errors)
    if coordination_modes != FROZEN_COORDINATION_MODE_TOKENS:
        errors.append(
            "COORDINATION_MODE_TOKENS_MISMATCH: expected sync/async, got "
            + "/".join(coordination_modes or ("missing",))
        )
    if re.search(r"\\texttt\{asym\}", methodology):
        errors.append("STALE_ASYM_COORDINATION_TOKEN_PRESENT")

    manuscript_seeds = _extract_seed_range(experiments, errors)
    if manuscript_seeds != FROZEN_SEEDS:
        errors.append(
            "MANUSCRIPT_SEEDS_MISMATCH: expected 200..204, got "
            + _render_integer_tuple(manuscript_seeds)
        )
    if contract.seeds != FROZEN_SEEDS:
        errors.append("PROTOCOL_SEEDS_MISMATCH")

    manuscript_budget = _extract_integer(
        _PHYSICAL_BUDGET_DECLARATION,
        experiments,
        "MANUSCRIPT_PHYSICAL_BUDGET_MISSING",
        errors,
    )
    if manuscript_budget != FROZEN_PHYSICAL_BUDGET:
        errors.append(
            f"MANUSCRIPT_PHYSICAL_BUDGET_MISMATCH: expected 80, got {manuscript_budget}"
        )
    if contract.physical_budget != FROZEN_PHYSICAL_BUDGET:
        errors.append("PROTOCOL_PHYSICAL_BUDGET_MISMATCH")

    terminal = _extract_terminal_claim(experiments, errors)
    if terminal != FROZEN_TERMINAL_CLAIM:
        errors.append("MANUSCRIPT_TERMINAL_CLAIM_MISMATCH")
    if _runtime_terminal_claim() != FROZEN_TERMINAL_CLAIM:
        errors.append("TERMINAL_EVIDENCE_IMPLEMENTATION_MISMATCH")

    if contract.total_new_runs != FROZEN_NEW_RUN_COUNT:
        errors.append("FULL_MATRIX_NEW_RUN_COUNT_MISMATCH")
    if contract.total_reused_rows != FROZEN_REUSED_ROW_COUNT:
        errors.append("FULL_MATRIX_REUSED_ROW_COUNT_MISMATCH")

    projections = _audit_table_metric_projections(contract, experiments, errors)
    return ManuscriptProtocolAudit(
        schema_version=MANUSCRIPT_PROTOCOL_AUDIT_SCHEMA_VERSION,
        methodology_path=method_path,
        experiments_path=experiment_path,
        methodology_sha256=hashlib.sha256(methodology_bytes).hexdigest(),
        experiments_sha256=hashlib.sha256(experiments_bytes).hexdigest(),
        coordination_mode_tokens=coordination_modes,
        seeds=contract.seeds,
        physical_budget=contract.physical_budget,
        terminal=terminal,
        total_new_runs=contract.total_new_runs,
        total_reused_rows=contract.total_reused_rows,
        table_metric_projections=projections,
        errors=tuple(errors),
    )


def _extract_coordination_modes(text: str, errors: list[str]) -> Tuple[str, ...]:
    matches = _COORDINATION_DECLARATION.findall(text)
    if len(matches) != 1:
        errors.append(
            "COORDINATION_MODE_DECLARATION_COUNT_MISMATCH: "
            f"expected 1, got {len(matches)}"
        )
        return tuple(matches[0]) if matches else ()
    return tuple(matches[0])


def _audit_required_claims(
    text: str,
    claims: Tuple[tuple[str, re.Pattern[str]], ...],
    errors: list[str],
) -> None:
    normalized = re.sub(r"\s+", " ", text)
    for error_code, pattern in claims:
        if pattern.search(normalized) is None:
            errors.append(error_code)


def _audit_backbones(
    experiments: str,
    contract: PaperExperimentContract,
    errors: list[str],
) -> None:
    for token in _BACKBONE_TOKENS:
        if rf"\texttt{{{token}}}" not in experiments:
            errors.append(f"EXPERIMENT_BACKBONE_MISSING:{token}")
    if contract.target_backbone != _BACKBONE_TOKENS[0]:
        errors.append("PROTOCOL_TARGET_BACKBONE_MISMATCH")
    if contract.source_backbones != _BACKBONE_TOKENS[:2]:
        errors.append("PROTOCOL_SOURCE_BACKBONES_MISMATCH")


def _audit_terminal_vocabulary(experiments: str, errors: list[str]) -> None:
    for token in _VALID_TERMINAL_TOKENS:
        if token not in experiments:
            errors.append(f"VALID_TERMINAL_TOKEN_MISSING:{token}")
    for token in _INVALID_TERMINAL_TOKENS:
        if token not in experiments:
            errors.append(f"INVALID_TERMINAL_TOKEN_MISSING:{token}")


def _audit_registered_row_labels(
    experiments: str,
    contract: PaperExperimentContract,
    errors: list[str],
) -> None:
    blocks = _table_blocks(experiments)
    latex_labels: Mapping[TableId, Tuple[str, ...]] = {
        TableId.TABLE_2: (
            "REMAC-Adapted",
            "COHERENT-Adapted",
            "Flat Reflective Repair",
            r"\methodname{} w/o Dynamic Experience",
            r"\methodname{} (full)",
        ),
        TableId.TABLE_3: ("Base", "Mirrored", "Narrow", "Altered Topology"),
        TableId.TABLE_4: (
            "Empty",
            r"GPT $\rightarrow$ GPT",
            r"Gemini $\rightarrow$ GPT",
        ),
    }
    tables_by_id = {table.table_id: table for table in contract.tables}
    for table_id, expected_labels in latex_labels.items():
        expected_count = len(tables_by_id[table_id].rows)
        if len(expected_labels) != expected_count:
            errors.append(f"{table_id.value.upper()}_ROW_REGISTRY_COUNT_MISMATCH")
            continue
        label = _TABLE_LABELS[table_id]
        selected = tuple(block for block in blocks if f"\\label{{{label}}}" in block)
        if len(selected) != 1:
            # The table-count error is emitted by the metric projection audit.
            continue
        cursor = -1
        for row_label in expected_labels:
            position = selected[0].find(row_label, cursor + 1)
            if position < 0:
                errors.append(
                    f"{table_id.value.upper()}_ROW_LABEL_MISSING:{row_label}"
                )
                continue
            if position <= cursor:
                errors.append(f"{table_id.value.upper()}_ROW_ORDER_MISMATCH")
                break
            cursor = position


def _extract_seed_range(text: str, errors: list[str]) -> Tuple[int, ...]:
    matches = _SEED_RANGE.findall(text)
    if len(matches) != 1:
        errors.append(
            f"SEED_RANGE_DECLARATION_COUNT_MISMATCH: expected 1, got {len(matches)}"
        )
        return ()
    start, end = (int(value) for value in matches[0])
    if end < start:
        errors.append("SEED_RANGE_IS_DESCENDING")
        return ()
    return tuple(range(start, end + 1))


def _extract_integer(
    pattern: re.Pattern[str],
    text: str,
    missing_error: str,
    errors: list[str],
) -> int:
    matches = pattern.findall(text)
    if len(matches) != 1:
        errors.append(f"{missing_error}: expected 1 declaration, got {len(matches)}")
        return -1
    return int(matches[0])


def _extract_terminal_claim(
    experiments: str, errors: list[str]
) -> FrozenTerminalClaim:
    section = _between(
        experiments,
        r"\\paragraph\{Terminal success\.\}",
        r"\\paragraph\{Valid and invalid terminal states\.\}",
        "TERMINAL_SUCCESS_PARAGRAPH_MISSING",
        errors,
    )
    values = {
        "target_support_radius_m": _terminal_number(
            section,
            r"support radius\s*\n?\s*\(\$([0-9]+(?:\.[0-9]+)?)"
            r"\\,\\mathrm\{m\}\$\)",
            "TARGET_SUPPORT_RADIUS",
            errors,
        ),
        "target_z_tolerance_m": _terminal_number(
            section,
            r"payload-center Z error is at most\s*\$([0-9]+(?:\.[0-9]+)?)"
            r"\\,\\mathrm\{m\}\$",
            "TARGET_Z_TOLERANCE",
            errors,
        ),
        "max_payload_tilt_deg": _terminal_number(
            section,
            r"terminal payload tilt is at most\s*\$([0-9]+(?:\.[0-9]+)?)"
            r"\^\\circ\$",
            "MAX_PAYLOAD_TILT",
            errors,
        ),
        "stable_window_s": _terminal_number(
            section,
            r"stable for\s*\$([0-9]+(?:\.[0-9]+)?)\\,\\mathrm\{s\}\$",
            "TERMINAL_STABLE_WINDOW",
            errors,
        ),
        "robot_exit_min_distance_m": _terminal_number(
            section,
            r"at least\s*\$([0-9]+(?:\.[0-9]+)?)\\,\\mathrm\{m\}\$"
            r"\s*from the target",
            "ROBOT_EXIT_MIN_DISTANCE",
            errors,
        ),
    }
    return FrozenTerminalClaim(**values)


def _between(
    text: str,
    start_pattern: str,
    end_pattern: str,
    error: str,
    errors: list[str],
) -> str:
    match = re.search(
        start_pattern + r"(?P<body>.*?)" + end_pattern,
        text,
        re.DOTALL,
    )
    if match is None:
        errors.append(error)
        return ""
    return match.group("body")


def _terminal_number(
    section: str, pattern: str, label: str, errors: list[str]
) -> float:
    matches = re.findall(pattern, section, re.DOTALL)
    if len(matches) != 1:
        errors.append(
            f"{label}_DECLARATION_COUNT_MISMATCH: expected 1, got {len(matches)}"
        )
        return -1.0
    return float(matches[0])


def _runtime_terminal_claim() -> FrozenTerminalClaim:
    return FrozenTerminalClaim(
        target_support_radius_m=TARGET_SUPPORT_RADIUS_M,
        target_z_tolerance_m=TARGET_Z_TOLERANCE_M,
        max_payload_tilt_deg=MAX_TERMINAL_PAYLOAD_TILT_DEG,
        stable_window_s=TERMINAL_STABLE_WINDOW_S,
        robot_exit_min_distance_m=ROBOT_EXIT_MIN_DISTANCE_M,
    )


def _audit_table_metric_projections(
    contract: PaperExperimentContract,
    experiments: str,
    errors: list[str],
) -> Tuple[TableMetricProjection, ...]:
    blocks = _table_blocks(experiments)
    projections = []
    metric_order = tuple(_METRIC_HEADER_PATTERNS)
    for table in contract.tables:
        label = _TABLE_LABELS[table.table_id]
        candidates = [block for block in blocks if f"\\label{{{label}}}" in block]
        if len(candidates) != 1:
            errors.append(
                f"{table.table_id.value.upper()}_LATEX_TABLE_COUNT_MISMATCH: "
                f"expected 1, got {len(candidates)}"
            )
            displayed = ()
        else:
            header = _table_header(candidates[0], table.table_id, errors)
            displayed = tuple(
                metric_id
                for metric_id in metric_order
                if _METRIC_HEADER_PATTERNS[metric_id].search(header)
            )
        if displayed != table.displayed_metrics:
            errors.append(
                f"{table.table_id.value.upper()}_DISPLAYED_METRICS_MISMATCH: "
                f"expected {','.join(table.displayed_metrics)}, got "
                f"{','.join(displayed)}"
            )
        projections.append(
            TableMetricProjection(
                table_id=table.table_id,
                latex_label=label,
                displayed_metrics=displayed,
            )
        )
    return tuple(projections)


def _table_blocks(text: str) -> Tuple[str, ...]:
    return tuple(
        match.group(0)
        for match in re.finditer(
            r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, re.DOTALL
        )
    )


def _table_header(
    block: str, table_id: TableId, errors: list[str]
) -> str:
    start = block.find(r"\toprule")
    end = block.find(r"\midrule", start + 1)
    if start < 0 or end < 0:
        errors.append(f"{table_id.value.upper()}_LATEX_HEADER_MISSING")
        return ""
    return block[start:end]


def _render_integer_tuple(values: Tuple[int, ...]) -> str:
    if not values:
        return "missing"
    return ",".join(str(value) for value in values)


__all__ = [
    "FROZEN_COORDINATION_MODE_TOKENS",
    "FROZEN_NEW_RUN_COUNT",
    "FROZEN_PHYSICAL_BUDGET",
    "FROZEN_REUSED_ROW_COUNT",
    "FROZEN_SEEDS",
    "FROZEN_TERMINAL_CLAIM",
    "MANUSCRIPT_PROTOCOL_AUDIT_SCHEMA_VERSION",
    "FrozenTerminalClaim",
    "ManuscriptProtocolAudit",
    "ManuscriptProtocolAuditError",
    "TableMetricProjection",
    "audit_manuscript_protocol",
]
