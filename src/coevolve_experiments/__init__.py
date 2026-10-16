"""Independent protocol and runner package for paper Tables 2--4."""

from .contracts import (
    ExperimentSuite,
    MemoryCondition,
    MemoryPolicy,
    MethodVariant,
    ProtocolLock,
    SceneId,
    SuiteTier,
    Table1SourceCohort,
    TableId,
)
from .lockfile import load_protocol_lock
from .manuscript_audit import ManuscriptProtocolAudit, audit_manuscript_protocol
from .paper_contract import PaperExperimentContract, build_paper_contract
from .protocol_lock_builder import (
    ProtocolContractArtifacts,
    ProtocolLockBuild,
    ProtocolLockBuildError,
    ProtocolLockBuildInputs,
    StrategySourceInputs,
    build_canonical_protocol_lock,
    load_protocol_lock_build_inputs,
    protocol_lock_build_input_document,
    write_protocol_lock_build,
)
from .registry import build_run_plan, build_suite


__all__ = [
    "ExperimentSuite",
    "MemoryCondition",
    "MemoryPolicy",
    "MethodVariant",
    "ManuscriptProtocolAudit",
    "PaperExperimentContract",
    "ProtocolLock",
    "ProtocolContractArtifacts",
    "ProtocolLockBuild",
    "ProtocolLockBuildError",
    "ProtocolLockBuildInputs",
    "SceneId",
    "SuiteTier",
    "Table1SourceCohort",
    "TableId",
    "StrategySourceInputs",
    "build_canonical_protocol_lock",
    "build_run_plan",
    "audit_manuscript_protocol",
    "build_paper_contract",
    "build_suite",
    "load_protocol_lock",
    "load_protocol_lock_build_inputs",
    "protocol_lock_build_input_document",
    "write_protocol_lock_build",
]
