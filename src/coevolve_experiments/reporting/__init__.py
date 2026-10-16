"""Protocol-neutral result collection for Tables 2--4 and the appendix."""

from .collect import INFRASTRUCTURE_INVALID_STATUSES, RunRecord, collect_run
from .diagnostics import collect_replay_diagnostics, collect_skill_calibration
from .from_scratch_memory_audit import (
    FromScratchMemoryAuditValidation,
    validate_from_scratch_memory_audit,
)
from .reuse import ReuseResolutionError, alias_reused_condition
from .replay_evidence import (
    ReplayHistoryMode,
    StrictReplayBinding,
    StrictReplayEvidenceError,
    StrictReplayResult,
    execute_strict_replay,
    load_strict_replay_manifest,
    prepare_strict_replay_manifest,
    validate_and_publish_strict_replay_result,
)
from .paper_summary import (
    IncompletePaperSummaryError,
    PAPER_SUMMARY_KIND,
    PAPER_SUMMARY_SCHEMA_VERSION,
    PaperSummaryError,
    audit_external_reuse_source,
    summarize_paper_tables,
)
from .summarize import (
    IncompleteConditionError,
    MixedConditionError,
    PAPER_RUN_SEEDS,
    build_appendix_tables,
    summarize_condition,
    summarize_suite,
)
from .table1_source import (
    Table1CohortAudit,
    Table1RunEvidence,
    Table1SourceAuditError,
    audit_frozen_table1_candidate,
    audit_table1_source_cohort,
    summarize_table1_backbones,
    summarize_table1_cohort,
)
from .trace import TraceSelectionError, build_figure2_trace, select_representative_run
from .trace_evidence import TraceEvidenceError, validate_trace_evidence
from .trace_projection import build_host_mutation_delta
from .terminal_evidence import (
    TerminalEvidenceValidation,
    validate_terminal_evidence,
)
from .transaction_accounting import (
    ACCEPTED_MANIFEST_PIN,
    ACCEPTED_MANIFEST_RELPATH,
    FROZEN_PHASE_PLAN_KIND,
    FROZEN_PHASE_PLAN_PIN,
    FROZEN_PHASE_PLAN_RELPATH,
    FROZEN_PHASE_PLAN_SCHEMA_VERSION,
    MUTATION_COMMIT_PRODUCER_ID,
    MUTATION_COMMIT_RECEIPT_KIND,
    MUTATION_COMMIT_RECEIPT_SCHEMA_VERSION,
    TRANSACTION_ACCOUNTING_LEDGER_KIND,
    TRANSACTION_ACCOUNTING_LEDGER_PIN,
    TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
    TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION,
    TransactionAccountingDerivation,
    TransactionAccountingEvidenceValidation,
    recompute_transaction_accounting_ledger,
    validate_transaction_accounting_evidence,
)
from .scene_result_binding import (
    SceneResultBinding,
    SceneResultBindingError,
    resolve_scene_result_binding,
)
from .usage_evidence import UsageEvidenceValidation, validate_fm_usage_evidence
from .v83_native_memory_audit import (
    V83NativeMemoryAuditValidation,
    validate_v83_native_memory_audit,
)

__all__ = [
    "ACCEPTED_MANIFEST_PIN",
    "ACCEPTED_MANIFEST_RELPATH",
    "FromScratchMemoryAuditValidation",
    "FROZEN_PHASE_PLAN_KIND",
    "FROZEN_PHASE_PLAN_PIN",
    "FROZEN_PHASE_PLAN_RELPATH",
    "FROZEN_PHASE_PLAN_SCHEMA_VERSION",
    "INFRASTRUCTURE_INVALID_STATUSES",
    "IncompleteConditionError",
    "IncompletePaperSummaryError",
    "MixedConditionError",
    "MUTATION_COMMIT_PRODUCER_ID",
    "MUTATION_COMMIT_RECEIPT_KIND",
    "MUTATION_COMMIT_RECEIPT_SCHEMA_VERSION",
    "PAPER_RUN_SEEDS",
    "PAPER_SUMMARY_KIND",
    "PAPER_SUMMARY_SCHEMA_VERSION",
    "PaperSummaryError",
    "ReplayHistoryMode",
    "RunRecord",
    "SceneResultBinding",
    "SceneResultBindingError",
    "ReuseResolutionError",
    "StrictReplayBinding",
    "StrictReplayEvidenceError",
    "StrictReplayResult",
    "TraceSelectionError",
    "TraceEvidenceError",
    "Table1CohortAudit",
    "Table1RunEvidence",
    "Table1SourceAuditError",
    "TerminalEvidenceValidation",
    "TRANSACTION_ACCOUNTING_LEDGER_KIND",
    "TRANSACTION_ACCOUNTING_LEDGER_PIN",
    "TRANSACTION_ACCOUNTING_LEDGER_RELPATH",
    "TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION",
    "TransactionAccountingDerivation",
    "TransactionAccountingEvidenceValidation",
    "UsageEvidenceValidation",
    "V83NativeMemoryAuditValidation",
    "build_appendix_tables",
    "build_figure2_trace",
    "build_host_mutation_delta",
    "collect_replay_diagnostics",
    "collect_run",
    "collect_skill_calibration",
    "execute_strict_replay",
    "alias_reused_condition",
    "audit_frozen_table1_candidate",
    "audit_external_reuse_source",
    "audit_table1_source_cohort",
    "select_representative_run",
    "summarize_condition",
    "summarize_table1_backbones",
    "summarize_table1_cohort",
    "summarize_paper_tables",
    "summarize_suite",
    "resolve_scene_result_binding",
    "recompute_transaction_accounting_ledger",
    "load_strict_replay_manifest",
    "prepare_strict_replay_manifest",
    "validate_fm_usage_evidence",
    "validate_trace_evidence",
    "validate_from_scratch_memory_audit",
    "validate_terminal_evidence",
    "validate_and_publish_strict_replay_result",
    "validate_transaction_accounting_evidence",
    "validate_v83_native_memory_audit",
]
