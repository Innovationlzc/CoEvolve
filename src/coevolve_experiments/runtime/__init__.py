"""Contract-smoke runtime; formal runs use the future Code-as-Policy bridge."""

from .contracts import (
    AcceptedManifestEntry,
    AuditEvent,
    CandidateFeedback,
    CheckpointRef,
    EngineConfig,
    FailureRecord,
    FeedbackKind,
    Observation,
    PhysicalDispatchResult,
    PolicyProposal,
    ProtocolBinding,
    RobotIntent,
    RollbackResult,
    RolloutRecord,
    RunAccounting,
    RunResult,
    RunStatus,
    RuntimeCallbacks,
    TeamIntent,
    ValidatedPolicy,
    VerifierOutcome,
)
from .policy import (
    IntentCompiler,
    PolicyValidationError,
    PolicyValidator,
    ast_fingerprint,
    canonical_ast,
    canonical_executable_intent,
    canonical_intent,
)
from .state import AuditSinkError, AuditTrail, ExactASTBlocker

__all__ = [
    "AcceptedManifestEntry",
    "AuditEvent",
    "AuditSinkError",
    "AuditTrail",
    "CandidateFeedback",
    "CheckpointRef",
    "EngineConfig",
    "ExperimentEngine",
    "ExactASTBlocker",
    "FailureRecord",
    "FeedbackKind",
    "IntentCompiler",
    "Observation",
    "PhysicalDispatchResult",
    "PolicyProposal",
    "ProtocolBinding",
    "PolicyValidationError",
    "PolicyValidator",
    "RobotIntent",
    "RollbackResult",
    "RolloutRecord",
    "RunAccounting",
    "RunResult",
    "RunStatus",
    "RuntimeCallbacks",
    "RuntimeEngineError",
    "ScriptedRuntime",
    "SharedExperimentEngine",
    "TeamIntent",
    "ValidatedPolicy",
    "VerifierOutcome",
    "ast_fingerprint",
    "canonical_ast",
    "canonical_executable_intent",
    "canonical_intent",
    "make_team_proposal",
]


def __getattr__(name: str):
    """Lazy exports avoid a contracts -> methods -> engine import cycle."""

    if name in {"ExperimentEngine", "RuntimeEngineError", "SharedExperimentEngine"}:
        from .engine import (
            ExperimentEngine,
            RuntimeEngineError,
            SharedExperimentEngine,
        )

        return {
            "ExperimentEngine": ExperimentEngine,
            "RuntimeEngineError": RuntimeEngineError,
            "SharedExperimentEngine": SharedExperimentEngine,
        }[name]
    if name in {"ScriptedRuntime", "make_team_proposal"}:
        from .fakes import ScriptedRuntime, make_team_proposal

        return {
            "ScriptedRuntime": ScriptedRuntime,
            "make_team_proposal": make_team_proposal,
        }[name]
    raise AttributeError(name)
