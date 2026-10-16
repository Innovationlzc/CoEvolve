"""Verifier-grounded contract harness for method-structure smoke tests.

This deterministic TeamIntent engine is intentionally not the paper's
Code-as-Policy execution loop.  Formal configurations are rejected before any
runtime callback is invoked.
"""

from __future__ import annotations

from typing import Any, Mapping

from coevolve_experiments.methods.base import (
    AdapterContext,
    MethodAdapter,
    REQUIRED_SAFETY_INVARIANTS,
)

from .contracts import (
    AcceptedManifestEntry,
    CandidateFeedback,
    CheckpointRef,
    EngineConfig,
    FailureRecord,
    FeedbackKind,
    Observation,
    PhysicalDispatchResult,
    RollbackResult,
    RolloutRecord,
    RunAccounting,
    RunResult,
    RunStatus,
    RuntimeCallbacks,
    VerifierOutcome,
)
from .policy import IntentCompiler, PolicyValidationError, PolicyValidator
from .state import AuditSinkError, AuditTrail, ExactASTBlocker


class RuntimeEngineError(RuntimeError):
    pass


class ExperimentEngine:
    """Own all safety, terminal, audit, and physical-budget semantics.

    Adapters can change model roles and memory behavior, but cannot replace the
    validator, verifier, checkpoint lineage, rollback, exact blocker, or
    accounting implemented here.
    """

    def __init__(
        self,
        adapter: MethodAdapter,
        callbacks: RuntimeCallbacks,
        *,
        validator: PolicyValidator | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.callbacks = callbacks
        self.validator = validator if validator is not None else PolicyValidator()
        self.config = config if config is not None else EngineConfig()
        if self.config.require_formal_protocol_pin:
            raise RuntimeEngineError(
                "SIMPLIFIED_TEAM_INTENT_NOT_FORMAL: ExperimentEngine is a "
                "contract-smoke harness; paper runs require the frozen Praxis "
                "Code-as-Policy live bridge"
            )
        if type(self.validator) is not PolicyValidator:
            raise RuntimeEngineError("engine requires the common PolicyValidator")
        if self.validator.require_compiler_match is not True:
            raise RuntimeEngineError(
                "deterministic intent compiler matching cannot be disabled"
            )
        if self.validator.required_actors != frozenset({"front", "back"}):
            raise RuntimeEngineError("engine requires the front/back intent contract")
        if self.validator.compiler != IntentCompiler():
            raise RuntimeEngineError("engine requires the common IntentCompiler")
        if self.validator.allowed_calls != PolicyValidator().allowed_calls:
            raise RuntimeEngineError(
                "engine requires the common PolicyValidator allowed-call contract"
            )
        if not adapter.descriptor.safety_invariants.all_enabled:
            raise RuntimeEngineError(
                "method adapter attempted to disable a common safety invariant"
            )
        if adapter.descriptor.safety_invariants != REQUIRED_SAFETY_INVARIANTS:
            raise RuntimeEngineError("method safety contract does not match runtime")

    def run(self, instruction: str) -> RunResult:
        """Execute one run and seal optional memory even on raised exceptions."""

        try:
            return self._run(instruction)
        except Exception:
            finalize_memory = getattr(self.adapter, "finalize_memory", None)
            if callable(finalize_memory):
                try:
                    finalize_memory()
                except Exception:
                    # Preserve the primary exception.  A file-backed memory
                    # persists its own violated/active lifecycle evidence, and
                    # reporting remains fail-closed if sealing also failed.
                    pass
            raise

    def _run(self, instruction: str) -> RunResult:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")

        accounting = RunAccounting()
        audit = AuditTrail(self.callbacks.audit_sink)
        blockers = ExactASTBlocker()
        manifest: list[AcceptedManifestEntry] = []
        latest_failure: FailureRecord | None = None
        candidate_attempts = 0
        generation_recovery_cycles = 0
        max_candidates = (
            self.config.max_candidate_attempts
            if self.config.max_candidate_attempts is not None
            else self.config.physical_budget * 20
        )

        try:
            observation = self.callbacks.observe()
            checkpoint = _checkpoint_ref(self.callbacks.save_checkpoint(observation))
        except Exception as exc:
            raise RuntimeEngineError(f"initialization callback failed: {exc}") from exc
        accounting.checkpoints_saved = 1
        checkpoint_ids = {checkpoint.checkpoint_id}
        audit.append(
            "run_initialized",
            checkpoint_id=checkpoint.checkpoint_id,
            transaction_index=None,
            payload={
                "protocol_id": self.config.protocol.protocol_id,
                "base_method_commit": self.config.protocol.base_method_commit,
                "invariant_fingerprint": self.config.protocol.invariant_fingerprint,
                "terminal_fingerprint": self.config.protocol.terminal_fingerprint,
                "provider_route_id": self.config.protocol.provider_route_id,
                "provider_route_fingerprint": (
                    self.config.protocol.provider_route_fingerprint
                ),
                "observation_contract_fingerprint": (
                    self.config.protocol.observation_contract_fingerprint
                ),
                "task_instruction_fingerprint": (
                    self.config.protocol.task_instruction_fingerprint
                ),
                "shared_prompt_contract_fingerprint": (
                    self.config.protocol.shared_prompt_contract_fingerprint
                ),
                "decision_prompt_fingerprint": (
                    self.config.protocol.decision_prompt_fingerprint
                ),
                "skill_api_fingerprint": self.config.protocol.skill_api_fingerprint,
                "method_variant": self.adapter.descriptor.variant.value,
                "policy_representation": (
                    self.adapter.descriptor.policy_representation.value
                ),
                "execution_capability": (
                    self.adapter.descriptor.execution_capability.value
                ),
                "paper_evidence": False,
                "physical_budget": self.config.physical_budget,
                "observation_sequence": observation.sequence,
            },
        )

        proposal = self.adapter.propose_initial(
            self._context(
                instruction,
                observation,
                checkpoint,
                manifest,
                latest_failure,
                blockers,
                accounting,
            )
        )

        while True:
            if accounting.physical_transactions >= self.config.physical_budget:
                audit.append(
                    "budget_exhausted",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=accounting.physical_transactions,
                    payload={"physical_budget": self.config.physical_budget},
                )
                return self._result(
                    RunStatus.BUDGET_EXHAUSTED,
                    "physical_budget_exhausted",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                )
            if proposal is None:
                if (
                    generation_recovery_cycles
                    < self.config.max_generation_recovery_cycles
                ):
                    generation_recovery_cycles += 1
                    audit.append(
                        "generation_recovery_cycle",
                        checkpoint_id=checkpoint.checkpoint_id,
                        transaction_index=accounting.physical_transactions,
                        payload={
                            "cycle": generation_recovery_cycles,
                            "maximum_cycles": (
                                self.config.max_generation_recovery_cycles
                            ),
                            "physically_dispatched": False,
                        },
                    )
                    proposal = self.adapter.propose_next(
                        self._context(
                            instruction,
                            observation,
                            checkpoint,
                            manifest,
                            latest_failure,
                            blockers,
                            accounting,
                        ),
                        CandidateFeedback(
                            FeedbackKind.GENERATION_RECOVERY,
                            "method-local proposal generation was exhausted; "
                            "request a fresh executable child without physical "
                            "dispatch",
                        ),
                    )
                    continue
                audit.append(
                    "adapter_exhausted",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=accounting.physical_transactions,
                    payload={
                        "generation_recovery_cycles": generation_recovery_cycles,
                        "maximum_cycles": self.config.max_generation_recovery_cycles,
                    },
                )
                return self._result(
                    RunStatus.ADAPTER_EXHAUSTED,
                    "common_generation_recovery_exhausted",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                )
            candidate_attempts += 1
            if candidate_attempts > max_candidates:
                audit.append(
                    "candidate_retry_limit",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=accounting.physical_transactions,
                    payload={"limit": max_candidates},
                )
                return self._result(
                    RunStatus.ADAPTER_EXHAUSTED,
                    "candidate_retry_limit",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                )

            try:
                policy = self.validator.validate(proposal)
            except PolicyValidationError as exc:
                accounting.invalid_candidates += 1
                audit.append(
                    "invalid_candidate",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=accounting.physical_transactions,
                    payload={"report": str(exc)},
                )
                feedback = CandidateFeedback(
                    FeedbackKind.INVALID_POLICY,
                    str(exc),
                    proposal=proposal,
                )
                proposal = self.adapter.propose_next(
                    self._context(
                        instruction,
                        observation,
                        checkpoint,
                        manifest,
                        latest_failure,
                        blockers,
                        accounting,
                    ),
                    feedback,
                )
                continue

            if blockers.contains(checkpoint.checkpoint_id, policy.ast_fingerprint):
                accounting.exact_blocked_candidates += 1
                evidence = blockers.evidence(
                    checkpoint.checkpoint_id, policy.ast_fingerprint
                )
                report = "exact checkpoint-local rejected AST replay blocked"
                audit.append(
                    "exact_ast_blocked",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=accounting.physical_transactions,
                    payload={
                        "ast_fingerprint": policy.ast_fingerprint,
                        "original_evidence": evidence or {},
                    },
                )
                proposal = self.adapter.propose_next(
                    self._context(
                        instruction,
                        observation,
                        checkpoint,
                        manifest,
                        latest_failure,
                        blockers,
                        accounting,
                    ),
                    CandidateFeedback(
                        FeedbackKind.EXACT_AST_BLOCKED,
                        report,
                        proposal=proposal,
                        ast_fingerprint=policy.ast_fingerprint,
                    ),
                )
                continue

            audit.append(
                "candidate_validated",
                checkpoint_id=checkpoint.checkpoint_id,
                transaction_index=accounting.physical_transactions + 1,
                payload={
                    "ast_fingerprint": policy.ast_fingerprint,
                    "lifecycle_phase": policy.proposal.intent.lifecycle_phase,
                },
            )
            try:
                dispatch = self.callbacks.dispatch(policy)
            except Exception as exc:
                accounting.physical_transactions += 1
                accounting.incomplete_transactions += 1
                transaction_index = accounting.physical_transactions
                self._append_audit_without_raising(
                    audit,
                    "dispatch_callback_failed",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=transaction_index,
                    payload={"error": str(exc)},
                )
                return self._recover_after_physical_change(
                    "dispatch_callback_failed",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                    transaction_index,
                )
            if not isinstance(dispatch, PhysicalDispatchResult):
                accounting.physical_transactions += 1
                accounting.incomplete_transactions += 1
                transaction_index = accounting.physical_transactions
                self._append_audit_without_raising(
                    audit,
                    "invalid_dispatch_result",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=transaction_index,
                    payload={"type": type(dispatch).__name__},
                )
                return self._recover_after_physical_change(
                    "invalid_dispatch_result",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                    transaction_index,
                )

            if not dispatch.physically_dispatched:
                accounting.predispatch_failures += 1
                report = dispatch.error or "host declined dispatch"
                audit.append(
                    "predispatch_failure",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=accounting.physical_transactions,
                    payload={"report": report, "evidence": dict(dispatch.evidence)},
                )
                proposal = self.adapter.propose_next(
                    self._context(
                        instruction,
                        observation,
                        checkpoint,
                        manifest,
                        latest_failure,
                        blockers,
                        accounting,
                    ),
                    CandidateFeedback(
                        FeedbackKind.PREDISPATCH_FAILURE,
                        report,
                        proposal=proposal,
                        ast_fingerprint=policy.ast_fingerprint,
                    ),
                )
                continue

            # The candidate passed static validation, checkpoint-local replay
            # blocking, and the complete Host preflight.  Only now does an
            # executable child actually exist; reset the checkpoint-local
            # generation-stall cycle before accounting for its physical
            # transaction.
            generation_recovery_cycles = 0

            accounting.physical_transactions += 1
            transaction_index = accounting.physical_transactions
            if dispatch.physical_transaction_completed:
                accounting.completed_physical_transactions += 1
            else:
                accounting.incomplete_transactions += 1
                # A partially executed motor transaction has no trustworthy
                # post-state to verify.  Do not let a permissive or stateful
                # verifier turn it into a checkpoint; restore the parent and
                # classify the attempt as infrastructure-invalid immediately.
                self._append_audit_without_raising(
                    audit,
                    "incomplete_physical_dispatch",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=transaction_index,
                    payload={
                        "ast_fingerprint": policy.ast_fingerprint,
                        "dispatch_evidence": dict(dispatch.evidence),
                        "error": dispatch.error or "",
                    },
                )
                return self._recover_after_physical_change(
                    "incomplete_physical_dispatch",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                    transaction_index,
                )
            try:
                observation_after = dispatch.observation
                if observation_after is None:
                    observation_after = self.callbacks.observe()
                    dispatch = PhysicalDispatchResult(
                        physically_dispatched=dispatch.physically_dispatched,
                        physical_transaction_completed=(
                            dispatch.physical_transaction_completed
                        ),
                        observation=observation_after,
                        evidence=dict(dispatch.evidence),
                        error=dispatch.error,
                    )
                verifier = self.callbacks.verify(observation, policy, dispatch)
            except Exception as exc:
                self._append_audit_without_raising(
                    audit,
                    "verification_callback_failed",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=transaction_index,
                    payload={"error": str(exc)},
                )
                return self._recover_after_physical_change(
                    "verification_callback_failed",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                    transaction_index,
                )
            if not isinstance(verifier, VerifierOutcome):
                self._append_audit_without_raising(
                    audit,
                    "invalid_verifier_outcome",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=transaction_index,
                    payload={"type": type(verifier).__name__},
                )
                return self._recover_after_physical_change(
                    "invalid_verifier_outcome",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                    transaction_index,
                )
            try:
                audit.append(
                    "physical_rollout",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=transaction_index,
                    payload={
                        "ast_fingerprint": policy.ast_fingerprint,
                        "dispatch_completed": dispatch.physical_transaction_completed,
                        "dispatch_evidence": dict(dispatch.evidence),
                        "accepted": verifier.accepted,
                        "terminal": verifier.terminal,
                        "failure_code": verifier.failure_code,
                        "report": verifier.report,
                        "phase_completion": verifier.phase_completion,
                        "verifier_evidence": dict(verifier.evidence),
                    },
                )
            except AuditSinkError:
                return self._recover_after_physical_change(
                    "audit_sink_failed_after_physical_rollout",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                    transaction_index,
                )

            if verifier.accepted:
                accounting.accepted_transactions += 1
            else:
                if dispatch.physical_transaction_completed:
                    accounting.rejected_transactions += 1
                latest_failure = FailureRecord(
                    transaction_index=transaction_index,
                    checkpoint_id=checkpoint.checkpoint_id,
                    ast_fingerprint=policy.ast_fingerprint,
                    failure_code=verifier.failure_code or "VERIFIER_REJECTED",
                    report=verifier.report,
                    evidence=dict(verifier.evidence),
                )

            rollout = RolloutRecord(
                transaction_index=transaction_index,
                checkpoint=checkpoint,
                observation_before=observation,
                policy=policy,
                dispatch=dispatch,
                verifier=verifier,
                observation_after=observation_after,
            )
            terminal = False
            if verifier.accepted:
                parent_checkpoint = checkpoint
                try:
                    candidate_checkpoint = _checkpoint_ref(
                        self.callbacks.save_checkpoint(observation_after)
                    )
                except Exception as exc:
                    self._append_audit_without_raising(
                        audit,
                        "checkpoint_callback_failed",
                        checkpoint_id=parent_checkpoint.checkpoint_id,
                        transaction_index=transaction_index,
                        payload={"error": str(exc)},
                    )
                    return self._recover_after_physical_change(
                        "checkpoint_callback_failed",
                        observation,
                        parent_checkpoint,
                        manifest,
                        audit,
                        latest_failure,
                        accounting,
                        transaction_index,
                    )
                accounting.checkpoints_saved += 1
                if candidate_checkpoint.checkpoint_id in checkpoint_ids:
                    self._append_audit_without_raising(
                        audit,
                        "checkpoint_identity_reused",
                        checkpoint_id=parent_checkpoint.checkpoint_id,
                        transaction_index=transaction_index,
                        payload={
                            "candidate_checkpoint_id": candidate_checkpoint.checkpoint_id
                        },
                    )
                    return self._recover_after_physical_change(
                        "checkpoint_identity_reused",
                        observation,
                        parent_checkpoint,
                        manifest,
                        audit,
                        latest_failure,
                        accounting,
                        transaction_index,
                    )

                terminal = verifier.terminal
                if self.callbacks.terminal_predicate is not None:
                    try:
                        terminal_value = self.callbacks.terminal_predicate(
                            observation_after, verifier
                        )
                    except Exception as exc:
                        self._append_audit_without_raising(
                            audit,
                            "terminal_predicate_failed",
                            checkpoint_id=parent_checkpoint.checkpoint_id,
                            transaction_index=transaction_index,
                            payload={"error": str(exc)},
                        )
                        return self._recover_after_physical_change(
                            "terminal_predicate_failed",
                            observation,
                            parent_checkpoint,
                            manifest,
                            audit,
                            latest_failure,
                            accounting,
                            transaction_index,
                        )
                    if not isinstance(terminal_value, bool):
                        self._append_audit_without_raising(
                            audit,
                            "terminal_predicate_non_boolean",
                            checkpoint_id=parent_checkpoint.checkpoint_id,
                            transaction_index=transaction_index,
                            payload={"type": type(terminal_value).__name__},
                        )
                        return self._recover_after_physical_change(
                            "terminal_predicate_non_boolean",
                            observation,
                            parent_checkpoint,
                            manifest,
                            audit,
                            latest_failure,
                            accounting,
                            transaction_index,
                        )
                    # When configured, the frozen host predicate is the sole
                    # authority for task termination.  A verifier-side hint
                    # cannot bypass it.
                    terminal = terminal_value
                    if verifier.terminal is not terminal:
                        verifier = VerifierOutcome(
                            accepted=verifier.accepted,
                            terminal=terminal,
                            failure_code=verifier.failure_code,
                            report=verifier.report,
                            phase_completion=verifier.phase_completion,
                            evidence=dict(verifier.evidence),
                        )
                        # Downstream manifest, feedback, and memory consumers
                        # must see the authoritative terminal value as well as
                        # the run loop itself.  The earlier physical_rollout
                        # audit intentionally retains the raw verifier hint.
                        rollout = RolloutRecord(
                            transaction_index=transaction_index,
                            checkpoint=parent_checkpoint,
                            observation_before=observation,
                            policy=policy,
                            dispatch=dispatch,
                            verifier=verifier,
                            observation_after=observation_after,
                        )

                try:
                    audit.append(
                        "accepted_checkpoint",
                        checkpoint_id=candidate_checkpoint.checkpoint_id,
                        transaction_index=transaction_index,
                        payload={
                            "parent_checkpoint_id": parent_checkpoint.checkpoint_id,
                            "manifest_length": len(manifest) + 1,
                        },
                    )
                except AuditSinkError:
                    return self._recover_after_physical_change(
                        "audit_sink_failed_before_checkpoint_commit",
                        observation,
                        parent_checkpoint,
                        manifest,
                        audit,
                        latest_failure,
                        accounting,
                        transaction_index,
                    )

                checkpoint = candidate_checkpoint
                checkpoint_ids.add(checkpoint.checkpoint_id)
                manifest.append(
                    AcceptedManifestEntry(
                        transaction_index=transaction_index,
                        parent_checkpoint_id=parent_checkpoint.checkpoint_id,
                        checkpoint_id=checkpoint.checkpoint_id,
                        ast_fingerprint=policy.ast_fingerprint,
                        intent=policy.proposal.intent,
                        verifier=verifier,
                    )
                )
                blockers.clear_checkpoint(parent_checkpoint.checkpoint_id)
                observation = observation_after
                feedback = CandidateFeedback(
                    FeedbackKind.VERIFIED_ACCEPTED,
                    verifier.report,
                    proposal=proposal,
                    ast_fingerprint=policy.ast_fingerprint,
                    verifier=verifier,
                )
            else:
                if dispatch.physical_transaction_completed:
                    blockers.add(
                        checkpoint.checkpoint_id,
                        policy.ast_fingerprint,
                        {
                            "failure_code": verifier.failure_code,
                            "report": verifier.report,
                            "transaction_index": transaction_index,
                        },
                    )
                rollback, rollback_audit_failed = self._perform_parent_rollback(
                    checkpoint,
                    observation,
                    accounting,
                    audit,
                    transaction_index,
                    event_kind="rollback",
                )
                if not rollback.restored or not rollback.recovery_verified:
                    return self._result(
                        RunStatus.ROLLBACK_FAILED,
                        "rollback_or_recovery_verification_failed",
                        observation,
                        checkpoint,
                        manifest,
                        audit,
                        latest_failure,
                        accounting,
                    )
                observation = rollback.observation or observation
                if rollback_audit_failed:
                    return self._result(
                        RunStatus.INFRASTRUCTURE_FAILED,
                        "audit_sink_failed_after_rollback",
                        observation,
                        checkpoint,
                        manifest,
                        audit,
                        latest_failure,
                        accounting,
                    )
                feedback = CandidateFeedback(
                    FeedbackKind.VERIFIED_REJECTED,
                    verifier.report,
                    proposal=proposal,
                    ast_fingerprint=policy.ast_fingerprint,
                    verifier=verifier,
                )
            # Memory/reflection updates run only after the new accepted
            # checkpoint is durable, or after a rejected transaction has been
            # restored and recovery-verified.  They can therefore never
            # promote an untrusted candidate state.
            try:
                adapter_audit = self.adapter.on_verified_rollout(
                    rollout,
                    self._context(
                        instruction,
                        observation,
                        checkpoint,
                        manifest,
                        latest_failure,
                        blockers,
                        accounting,
                    ),
                )
            except Exception as exc:
                self._append_audit_without_raising(
                    audit,
                    "adapter_rollout_update_failed",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=transaction_index,
                    payload={"error": str(exc)},
                )
                return self._result(
                    RunStatus.INFRASTRUCTURE_FAILED,
                    "adapter_rollout_update_failed",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                )
            try:
                audit.append(
                    "adapter_rollout_update",
                    checkpoint_id=checkpoint.checkpoint_id,
                    transaction_index=transaction_index,
                    payload={} if adapter_audit is None else dict(adapter_audit),
                )
            except AuditSinkError:
                return self._recover_after_physical_change(
                    "audit_sink_failed_after_adapter_rollout_update",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                    transaction_index,
                )

            if terminal:
                try:
                    audit.append(
                        "terminal_verified",
                        checkpoint_id=checkpoint.checkpoint_id,
                        transaction_index=transaction_index,
                        payload={"report": verifier.report},
                    )
                except AuditSinkError:
                    return self._recover_after_physical_change(
                        "audit_sink_failed_before_terminal_seal",
                        observation,
                        checkpoint,
                        manifest,
                        audit,
                        latest_failure,
                        accounting,
                        transaction_index,
                    )
                return self._result(
                    RunStatus.SUCCEEDED,
                    "verified_terminal",
                    observation,
                    checkpoint,
                    manifest,
                    audit,
                    latest_failure,
                    accounting,
                )

            if accounting.physical_transactions >= self.config.physical_budget:
                proposal = proposal  # budget check at top avoids another model call
                continue
            proposal = self.adapter.propose_next(
                self._context(
                    instruction,
                    observation,
                    checkpoint,
                    manifest,
                    latest_failure,
                    blockers,
                    accounting,
                ),
                feedback,
            )

    def _recover_after_physical_change(
        self,
        reason: str,
        parent_observation: Observation,
        parent_checkpoint: CheckpointRef,
        manifest: list[AcceptedManifestEntry],
        audit: AuditTrail,
        latest_failure: FailureRecord | None,
        accounting: RunAccounting,
        transaction_index: int,
    ) -> RunResult:
        """Fail closed: restore and verify the parent before ending the run."""

        rollback, _ = self._perform_parent_rollback(
            parent_checkpoint,
            parent_observation,
            accounting,
            audit,
            transaction_index,
            event_kind="rollback_after_infrastructure_failure",
        )
        recovered_observation = rollback.observation or parent_observation
        if not rollback.restored or not rollback.recovery_verified:
            return self._result(
                RunStatus.ROLLBACK_FAILED,
                f"rollback_or_recovery_verification_failed_after_{reason}",
                recovered_observation,
                parent_checkpoint,
                manifest,
                audit,
                latest_failure,
                accounting,
            )
        return self._result(
            RunStatus.INFRASTRUCTURE_FAILED,
            reason,
            recovered_observation,
            parent_checkpoint,
            manifest,
            audit,
            latest_failure,
            accounting,
        )

    def _perform_parent_rollback(
        self,
        checkpoint: CheckpointRef,
        expected_observation: Observation,
        accounting: RunAccounting,
        audit: AuditTrail,
        transaction_index: int,
        *,
        event_kind: str,
    ) -> tuple[RollbackResult, bool]:
        try:
            rollback = _rollback_result(
                self.callbacks.rollback(checkpoint, expected_observation),
                expected_observation,
                allow_shorthand=not self.config.require_formal_protocol_pin,
            )
        except Exception as exc:
            rollback = RollbackResult(False, False, report=str(exc))
        accounting.rollbacks += 1
        audit_ok = self._append_audit_without_raising(
            audit,
            event_kind,
            checkpoint_id=checkpoint.checkpoint_id,
            transaction_index=transaction_index,
            payload={
                "restored": rollback.restored,
                "recovery_verified": rollback.recovery_verified,
                "report": rollback.report,
            },
        )
        return rollback, not audit_ok

    @staticmethod
    def _append_audit_without_raising(
        audit: AuditTrail,
        kind: str,
        *,
        checkpoint_id: str | None,
        transaction_index: int | None,
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        try:
            audit.append(
                kind,
                checkpoint_id=checkpoint_id,
                transaction_index=transaction_index,
                payload=payload,
            )
        except AuditSinkError:
            return False
        return True

    def _context(
        self,
        instruction: str,
        observation: Observation,
        checkpoint: CheckpointRef,
        manifest: list[AcceptedManifestEntry],
        latest_failure: FailureRecord | None,
        blockers: ExactASTBlocker,
        accounting: RunAccounting,
    ) -> AdapterContext:
        return AdapterContext(
            instruction=instruction,
            observation=observation,
            checkpoint=checkpoint,
            remaining_physical_budget=max(
                0, self.config.physical_budget - accounting.physical_transactions
            ),
            accepted_manifest=tuple(manifest),
            latest_physical_failure=latest_failure,
            blocked_ast_fingerprints=blockers.fingerprints(
                checkpoint.checkpoint_id
            ),
            accounting=accounting.snapshot(),
        )

    def _result(
        self,
        status: RunStatus,
        reason: str,
        observation: Observation,
        checkpoint: CheckpointRef,
        manifest: list[AcceptedManifestEntry],
        audit: AuditTrail,
        latest_failure: FailureRecord | None,
        accounting: RunAccounting,
    ) -> RunResult:
        final_status = status
        final_reason = reason
        try:
            finalize_memory = getattr(self.adapter, "finalize_memory", None)
            if callable(finalize_memory):
                finalize_memory()
        except Exception as exc:
            self._append_audit_without_raising(
                audit,
                "memory_finalize_failed",
                checkpoint_id=checkpoint.checkpoint_id,
                transaction_index=accounting.physical_transactions,
                payload={"error": str(exc)},
            )
            # An unsealed memory lifecycle invalidates ordinary task outcomes,
            # including budget/adapter exhaustion.  Preserve only an existing
            # infrastructure failure or the more severe rollback failure.
            if status not in {
                RunStatus.INFRASTRUCTURE_FAILED,
                RunStatus.ROLLBACK_FAILED,
            }:
                final_status = RunStatus.INFRASTRUCTURE_FAILED
                final_reason = "memory_finalize_failed"
        usage = self.adapter.usage_summary()
        usage_mapping = (
            usage.as_dict()
            if hasattr(usage, "as_dict")
            else dict(usage)  # type: ignore[arg-type]
        )
        return RunResult(
            status=final_status,
            terminal_reason=final_reason,
            observation=observation,
            checkpoint=checkpoint,
            accepted_manifest=tuple(manifest),
            raw_audit=audit.snapshot(),
            latest_failure=latest_failure,
            accounting=accounting.snapshot(),
            usage=usage_mapping,
        )


SharedExperimentEngine = ExperimentEngine


def _checkpoint_ref(value: CheckpointRef | str) -> CheckpointRef:
    if isinstance(value, CheckpointRef):
        return value
    if isinstance(value, str) and value:
        return CheckpointRef(value)
    raise RuntimeEngineError("checkpoint callback must return CheckpointRef or string")


def _rollback_result(
    value: RollbackResult | Observation | bool,
    expected: Observation,
    *,
    allow_shorthand: bool = True,
) -> RollbackResult:
    if isinstance(value, RollbackResult):
        return value
    if not allow_shorthand:
        raise RuntimeEngineError(
            "formal rollback callback must return an explicit RollbackResult"
        )
    if isinstance(value, Observation):
        return RollbackResult(True, True, value)
    if isinstance(value, bool):
        return RollbackResult(value, value, expected if value else None)
    raise RuntimeEngineError(
        "rollback callback must return RollbackResult, Observation, or bool"
    )
