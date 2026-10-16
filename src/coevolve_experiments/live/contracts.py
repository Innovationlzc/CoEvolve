"""Typed seam to the frozen production Code-as-Policy runner.

The adapter harness in :mod:`coevolve_experiments.runtime` deliberately uses a
small deterministic TeamIntent IR.  The paper method instead executes the
``PolicyScript``/``RobotPolicyScript`` pipeline owned by ``praxis_coop``.  A
formal run may cross this seam only after the final Table-1 runner accepts a
hash-bound post-Table-1 launch configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

class PolicyRepresentation(str, Enum):
    TEAM_INTENT_IR = "team_intent_ir"
    PRAXIS_CODE_AS_POLICY = "praxis_code_as_policy"


class ExecutionCapability(str, Enum):
    CONTRACT_SMOKE_ONLY = "contract_smoke_only"
    PAPER_FORMAL = "paper_formal"


@dataclass(frozen=True, slots=True)
class LiveBridgePreflight:
    available: bool
    policy_representation: PolicyRepresentation
    execution_capability: ExecutionCapability
    reason_code: str
    detail: str


@dataclass(frozen=True, slots=True)
class LiveRunReceipt:
    """Attested artifacts returned only after a protocol-valid subprocess."""

    run_id: str
    output_dir: Path
    run_manifest_path: Path
    metrics_path: Path
    launch_config_path: Path
    bridge_receipt_path: Path
    launch_config_sha256: str
    run_manifest_sha256: str
    metrics_sha256: str
    bridge_receipt_sha256: str
    launch_fingerprint: str
    base_method_commit: str
    runner_integration_commit: str
    terminal_status: str
    task_success: bool
    protocol_valid: bool
    paper_evidence_eligible: bool


@runtime_checkable
class PraxisCodePolicyLiveBridge(Protocol):
    """Production bridge whose only run input is one absolute config path."""

    policy_representation: PolicyRepresentation
    execution_capability: ExecutionCapability

    def launch(
        self,
        launch_config_path: Path,
    ) -> LiveRunReceipt:
        ...


class LiveBridgeUnavailableError(RuntimeError):
    """Raised instead of treating the TeamIntent harness as paper evidence."""
