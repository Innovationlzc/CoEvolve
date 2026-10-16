"""Code-as-Policy loop for the two-G1 box-carry task.

This package is the current high-level experiment path. It treats the tuned
SONIC static skills as a safe SDK and evolves executable Python policy scripts
plus verifier-grounded deployment decision memory.
"""

from .contracts import (
    BoxCarryTask,
    CoordinationMode,
    EvaluationReport,
    EvolutionRecord,
    ExecutionTrace,
    FailureLevel,
    ObservationBundle,
    PlannedSkillCall,
    PolicyScript,
    RobotAssignment,
    RobotExecutionTrace,
    RobotPolicyScript,
    RouteProgressGate,
    RouteSegment,
    SkillCallRecord,
    Subtask,
)
from .checkpoints import (
    CheckpointBackendFatalError,
    CheckpointResult,
    SimulatorCheckpointClient,
)
from .action_memory import AcceptedEpisodeManifest, VerifiedActionMemoryStore
from .coordination_specs import (
    COORDINATION_PRIMITIVE_METHODS,
    coordination_primitive_library_doc,
)
from .evaluator import BoxCarryEvaluator
from .evolution_controller import (
    EvolutionAttempt,
    EvolutionController,
    EvolutionDecision,
    EvolutionHistoryView,
)
from .failure_taxonomy import FailureClassification, classify_failure
from .model_agents import ModelCollaborativeEvolutionAgent, ModelTaskOrchestrator
from .model_role_pipeline import (
    MUTATION_LEVELS,
    ModelRoleError,
    ModelRolePipeline,
)
from .orchestrator import TaskOrchestrator
from .policy_executor import PolicyExecutor
from .route_spec import (
    RouteSpecConfigurationError,
    load_task_route_spec,
    validate_task_route_spec,
)
from .skill_api import G1SkillAPI
from .strategy_memory import StrategyContext, StrategyMemoryStore
from .transition_verifier import (
    TransitionDecision,
    TransitionOutcome,
    apply_transition_decision,
    transition_outcome_from_report,
    verify_transition,
)
from .visual_verifier import QwenVLBoxCarryVisualVerifier, VisualVerifierReport

__all__ = [
    "BoxCarryEvaluator",
    "BoxCarryTask",
    "CheckpointResult",
    "CheckpointBackendFatalError",
    "AcceptedEpisodeManifest",
    "CoordinationMode",
    "COORDINATION_PRIMITIVE_METHODS",
    "EvaluationReport",
    "EvolutionRecord",
    "EvolutionAttempt",
    "EvolutionController",
    "EvolutionDecision",
    "EvolutionHistoryView",
    "ExecutionTrace",
    "FailureClassification",
    "FailureLevel",
    "G1SkillAPI",
    "ModelCollaborativeEvolutionAgent",
    "ModelRoleError",
    "ModelRolePipeline",
    "ModelTaskOrchestrator",
    "MUTATION_LEVELS",
    "ObservationBundle",
    "PlannedSkillCall",
    "PolicyExecutor",
    "PolicyScript",
    "QwenVLBoxCarryVisualVerifier",
    "RobotAssignment",
    "RobotExecutionTrace",
    "RobotPolicyScript",
    "RouteProgressGate",
    "RouteSpecConfigurationError",
    "RouteSegment",
    "SimulatorCheckpointClient",
    "SkillCallRecord",
    "Subtask",
    "StrategyContext",
    "StrategyMemoryStore",
    "TaskOrchestrator",
    "TransitionDecision",
    "TransitionOutcome",
    "VisualVerifierReport",
    "VerifiedActionMemoryStore",
    "apply_transition_decision",
    "classify_failure",
    "coordination_primitive_library_doc",
    "load_task_route_spec",
    "transition_outcome_from_report",
    "validate_task_route_spec",
    "verify_transition",
]
