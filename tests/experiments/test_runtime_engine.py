from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import pytest

from coevolve_experiments.contracts import MethodVariant
from coevolve_experiments.methods import (
    AdapterUsageSummary,
    MemorySwitches,
    MethodDescriptor,
)
from coevolve_experiments.runtime import (
    CandidateFeedback,
    EngineConfig,
    ExperimentEngine,
    FeedbackKind,
    IntentCompiler,
    Observation,
    PhysicalDispatchResult,
    PolicyProposal,
    PolicyValidationError,
    PolicyValidator,
    ProtocolBinding,
    RunStatus,
    RuntimeEngineError,
    RollbackResult,
    ScriptedRuntime,
    TeamIntent,
    VerifierOutcome,
    make_team_proposal,
)
from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE


class QueueAdapter:
    descriptor = MethodDescriptor(
        variant=MethodVariant.FLAT_REFLECTIVE_REPAIR,
        allowed_roles=frozenset(),
        hierarchical_evolution=False,
        durable_dynamic_memory=False,
        memory_switches=MemorySwitches.disabled(),
    )

    def __init__(self, proposals):
        self._proposals = iter(proposals)
        self.feedback: list[CandidateFeedback] = []
        self.rollouts = []

    def propose_initial(self, context):
        return next(self._proposals, None)

    def propose_next(self, context, feedback):
        self.feedback.append(feedback)
        return next(self._proposals, None)

    def on_verified_rollout(self, rollout, context):
        self.rollouts.append(rollout)
        return {"observed": rollout.transaction_index}

    def usage_summary(self):
        return AdapterUsageSummary(0, 0, 0, {})


def _formal_binding() -> ProtocolBinding:
    return ProtocolBinding(
        protocol_id="frozen-v1",
        base_method_commit="a" * 40,
        invariant_fingerprint="b" * 64,
        terminal_fingerprint="c" * 64,
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        observation_contract_fingerprint="d" * 64,
        task_instruction_fingerprint="e" * 64,
        shared_prompt_contract_fingerprint="f" * 64,
        decision_prompt_fingerprint="1" * 64,
        skill_api_fingerprint="0" * 64,
    )


def test_policy_validator_rejects_declared_intent_source_mismatch():
    original = make_team_proposal("forward", arguments={"distance_m": 0.5})
    changed = make_team_proposal("forward", arguments={"distance_m": 1.0})
    mismatch = PolicyProposal(
        source=original.source,
        intent=changed.intent,
        rationale="declare 1.0 while executable says 0.5",
    )

    with pytest.raises(PolicyValidationError, match="does not match"):
        PolicyValidator().validate(mismatch)


def test_metadata_only_change_has_same_exact_executable_fingerprint():
    first = make_team_proposal("same", metadata={"note": "a"})
    second = make_team_proposal("same", metadata={"note": "b", "claim": "new"})

    first_valid = PolicyValidator().validate(first)
    second_valid = PolicyValidator().validate(second)

    assert first_valid.ast_fingerprint == second_valid.ast_fingerprint


def test_exact_blocker_rollback_manifest_and_accounting():
    rejected = make_team_proposal("blocked", metadata={"note": "original"})
    metadata_replay = make_team_proposal(
        "blocked", metadata={"note": "attempted bypass"}
    )
    repair = make_team_proposal("different-effective-command")
    adapter = QueueAdapter([rejected, metadata_replay, repair])
    runtime = ScriptedRuntime(
        [
            VerifierOutcome(
                False,
                failure_code="PAYLOAD_DROP",
                report="payload dropped",
                phase_completion=0.2,
            ),
            VerifierOutcome(True, terminal=True, report="task complete"),
        ]
    )

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry")

    assert result.success
    assert result.accounting.physical_transactions == 2
    assert result.accounting.completed_physical_transactions == 2
    assert result.accounting.rejected_transactions == 1
    assert result.accounting.accepted_transactions == 1
    assert result.accounting.exact_blocked_candidates == 1
    assert result.accounting.rollbacks == 1
    assert result.accounting.checkpoints_saved == 2
    assert len(result.accepted_manifest) == 1
    assert result.latest_failure is not None
    assert result.latest_failure.failure_code == "PAYLOAD_DROP"
    assert runtime.rollback_calls == ["checkpoint_0"]
    assert [item.kind.value for item in adapter.feedback] == [
        "verified_rejected",
        "exact_ast_blocked",
    ]
    assert any(event.kind == "exact_ast_blocked" for event in result.raw_audit)


def test_exact_blocker_is_checkpoint_local_and_clears_after_acceptance():
    first = make_team_proposal("same-command")
    bridge = make_team_proposal("bridge-command")
    same_at_new_checkpoint = make_team_proposal("same-command")
    adapter = QueueAdapter([first, bridge, same_at_new_checkpoint])
    runtime = ScriptedRuntime(
        [
            VerifierOutcome(False, failure_code="NO_PROGRESS", report="failed"),
            VerifierOutcome(True, terminal=False, report="advanced"),
            VerifierOutcome(True, terminal=True, report="done"),
        ]
    )

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry")

    assert result.success
    assert result.accounting.physical_transactions == 3
    assert result.accounting.exact_blocked_candidates == 0
    assert len(result.accepted_manifest) == 2
    assert runtime.dispatched[0].ast_fingerprint == runtime.dispatched[2].ast_fingerprint


def test_invalid_candidate_does_not_dispatch_or_consume_budget():
    valid = make_team_proposal("valid")
    mismatch_intent = make_team_proposal("different").intent
    invalid = PolicyProposal(valid.source, mismatch_intent)
    adapter = QueueAdapter([invalid, valid])
    runtime = ScriptedRuntime([VerifierOutcome(True, terminal=True, report="done")])

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry")

    assert result.success
    assert result.accounting.invalid_candidates == 1
    assert result.accounting.physical_transactions == 1
    assert len(runtime.dispatched) == 1


def test_rejected_last_budget_rolls_back_then_stops_without_extra_proposal():
    adapter = QueueAdapter(
        [make_team_proposal("only"), make_team_proposal("must-not-be-requested")]
    )
    runtime = ScriptedRuntime(
        [VerifierOutcome(False, failure_code="NO_PROGRESS", report="failed")]
    )

    result = ExperimentEngine(
        adapter,
        runtime.callbacks(),
        config=EngineConfig(physical_budget=1),
    ).run("carry")

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.accounting.physical_transactions == 1
    assert result.accounting.rollbacks == 1
    assert adapter.feedback == []


def test_failed_rollback_is_terminal_and_transparently_accounted():
    adapter = QueueAdapter([make_team_proposal("unsafe")])
    runtime = ScriptedRuntime(
        [VerifierOutcome(False, failure_code="SUPPORT_LOSS", report="fell")],
        rollback_succeeds=False,
    )

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry")

    assert result.status is RunStatus.ROLLBACK_FAILED
    assert result.accounting.physical_transactions == 1
    assert result.accounting.rejected_transactions == 1
    assert result.accounting.rollbacks == 1


def test_incomplete_dispatch_cannot_enter_manifest_or_dynamic_experience():
    from coevolve_experiments.runtime import PhysicalDispatchResult

    adapter = QueueAdapter([make_team_proposal("partial")])
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=True, report="incorrect accept")],
        dispatch_results=[PhysicalDispatchResult(True, False, error="motor timeout")],
    )

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.accounting.physical_transactions == 1
    assert result.accounting.completed_physical_transactions == 0
    assert result.accounting.incomplete_transactions == 1
    assert result.accounting.accepted_transactions == 0
    assert result.accepted_manifest == ()
    assert adapter.rollouts == []
    assert runtime.remaining_outcomes == 1
    assert runtime.rollback_calls == ["checkpoint_0"]


def test_engine_defaults_to_80_and_exposes_formal_protocol_commit_seam():
    assert EngineConfig().physical_budget == 80
    with pytest.raises(ValueError, match="pinned protocol"):
        EngineConfig(require_formal_protocol_pin=True)

    commit_only_binding = ProtocolBinding(
        protocol_id="frozen-v1",
        base_method_commit="a" * 40,
        invariant_fingerprint="b" * 64,
        terminal_fingerprint="c" * 64,
    )
    assert not commit_only_binding.formally_pinned
    with pytest.raises(ValueError, match="pinned protocol"):
        EngineConfig(
            protocol=commit_only_binding,
            require_formal_protocol_pin=True,
        )

    binding = _formal_binding()
    formal_config = EngineConfig(
        protocol=binding, require_formal_protocol_pin=True
    )
    assert formal_config.protocol.formally_pinned
    with pytest.raises(ValueError, match="formal physical budget"):
        EngineConfig(
            physical_budget=79,
            protocol=binding,
            require_formal_protocol_pin=True,
        )

    adapter = QueueAdapter([make_team_proposal("formal")])
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=True, report="done")]
    )
    with pytest.raises(
        RuntimeEngineError, match="SIMPLIFIED_TEAM_INTENT_NOT_FORMAL"
    ):
        ExperimentEngine(adapter, runtime.callbacks(), config=formal_config)
    assert runtime.saved_checkpoints == []
    assert runtime.dispatched == []

    result = ExperimentEngine(
        adapter,
        runtime.callbacks(),
        config=EngineConfig(protocol=binding),
    ).run("carry")
    initialized = next(
        event for event in result.raw_audit if event.kind == "run_initialized"
    )
    assert initialized.payload["observation_contract_fingerprint"] == "d" * 64
    assert initialized.payload["task_instruction_fingerprint"] == "e" * 64
    assert initialized.payload["provider_route_id"] == PRIMARY_GPT_PROVIDER_ROUTE.route_id
    assert initialized.payload["provider_route_fingerprint"] == (
        PRIMARY_GPT_PROVIDER_ROUTE.fingerprint
    )
    assert initialized.payload["shared_prompt_contract_fingerprint"] == "f" * 64
    assert initialized.payload["decision_prompt_fingerprint"] == "1" * 64
    assert initialized.payload["skill_api_fingerprint"] == "0" * 64
    assert initialized.payload["policy_representation"] == "team_intent_ir"
    assert initialized.payload["execution_capability"] == "contract_smoke_only"
    assert initialized.payload["paper_evidence"] is False


def test_checkpoint_identity_must_advance_before_memory_promotion():
    proposal = make_team_proposal("accepted")
    adapter = QueueAdapter([proposal])
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=True, report="done")]
    )
    callbacks = runtime.callbacks()

    def save_then_reuse(observation):
        if not runtime.saved_checkpoints:
            return runtime.save_checkpoint(observation)
        return runtime.saved_checkpoints[0]

    callbacks = replace(callbacks, save_checkpoint=save_then_reuse)

    result = ExperimentEngine(adapter, callbacks).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.terminal_reason == "checkpoint_identity_reused"
    assert result.accepted_manifest == ()
    assert adapter.rollouts == []
    assert runtime.rollback_calls == ["checkpoint_0"]


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: TeamIntent(
            make_team_proposal("x").intent.commands, atomic="false"  # type: ignore[arg-type]
        ),
        lambda: PhysicalDispatchResult("false", False),  # type: ignore[arg-type]
        lambda: PhysicalDispatchResult(True, "false"),  # type: ignore[arg-type]
        lambda: VerifierOutcome("false"),  # type: ignore[arg-type]
        lambda: VerifierOutcome(True, terminal="false"),  # type: ignore[arg-type]
        lambda: RollbackResult("false", False),  # type: ignore[arg-type]
        lambda: RollbackResult(True, "false"),  # type: ignore[arg-type]
    ],
)
def test_physical_contract_boolean_fields_are_strict(constructor):
    with pytest.raises(TypeError, match="must be a bool"):
        constructor()


def test_inconsistent_physical_boolean_states_are_rejected():
    with pytest.raises(ValueError, match="must have been dispatched"):
        PhysicalDispatchResult(False, True)
    with pytest.raises(ValueError, match="terminal verifier outcome"):
        VerifierOutcome(False, terminal=True)
    with pytest.raises(ValueError, match="restore failed"):
        RollbackResult(False, True)


def test_engine_rejects_a_validator_that_disables_deterministic_compilation():
    adapter = QueueAdapter([make_team_proposal("x")])
    runtime = ScriptedRuntime([])

    with pytest.raises(RuntimeEngineError, match="cannot be disabled"):
        ExperimentEngine(
            adapter,
            runtime.callbacks(),
            validator=PolicyValidator(require_compiler_match=False),
        )

    with pytest.raises(TypeError, match="must be a bool"):
        PolicyValidator(require_compiler_match="true")  # type: ignore[arg-type]


def test_engine_rejects_drifted_allowed_call_contract():
    adapter = QueueAdapter([make_team_proposal("x")])
    runtime = ScriptedRuntime([])

    with pytest.raises(RuntimeEngineError, match="allowed-call contract"):
        ExperimentEngine(
            adapter,
            runtime.callbacks(),
            validator=PolicyValidator(
                allowed_calls={"atomic_team_transaction", "robot_intent"}
            ),
        )


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_candidate_attempt_limit_requires_a_strict_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        EngineConfig(max_candidate_attempts=value)  # type: ignore[arg-type]


def test_formal_config_rejects_candidate_attempt_override():
    assert EngineConfig(max_candidate_attempts=1).max_candidate_attempts == 1
    with pytest.raises(ValueError, match="fixed by the common runtime"):
        EngineConfig(
            max_candidate_attempts=1,
            protocol=_formal_binding(),
            require_formal_protocol_pin=True,
        )


def test_non_boolean_terminal_predicate_rolls_back_and_cannot_succeed():
    adapter = QueueAdapter([make_team_proposal("accepted")])
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=False, report="accepted, not terminal")]
    )
    callbacks = replace(
        runtime.callbacks(), terminal_predicate=lambda observation, outcome: "false"
    )

    result = ExperimentEngine(adapter, callbacks).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.terminal_reason == "terminal_predicate_non_boolean"
    assert result.accepted_manifest == ()
    assert runtime.rollback_calls == ["checkpoint_0"]


def test_configured_terminal_predicate_is_authoritative_over_verifier_hint():
    adapter = QueueAdapter([make_team_proposal("accepted-not-terminal")])
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=True, report="verifier terminal hint")]
    )
    predicate_calls: list[int] = []

    def not_terminal(observation, outcome):
        predicate_calls.append(observation.sequence)
        return False

    result = ExperimentEngine(
        adapter,
        replace(runtime.callbacks(), terminal_predicate=not_terminal),
    ).run("carry")

    assert result.status is RunStatus.ADAPTER_EXHAUSTED
    assert predicate_calls == [1]
    assert len(result.accepted_manifest) == 1
    assert result.accepted_manifest[0].verifier.terminal is False
    assert adapter.rollouts[0].verifier.terminal is False
    physical_event = next(
        event for event in result.raw_audit if event.kind == "physical_rollout"
    )
    assert physical_event.payload["terminal"] is True


def test_no_proposal_uses_three_common_generation_recovery_cycles():
    adapter = QueueAdapter([None, None, None, None])
    runtime = ScriptedRuntime([])

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry")

    assert result.status is RunStatus.ADAPTER_EXHAUSTED
    assert result.terminal_reason == "common_generation_recovery_exhausted"
    assert result.accounting.physical_transactions == 0
    assert [feedback.kind for feedback in adapter.feedback] == [
        FeedbackKind.GENERATION_RECOVERY,
        FeedbackKind.GENERATION_RECOVERY,
        FeedbackKind.GENERATION_RECOVERY,
    ]
    cycles = [
        event
        for event in result.raw_audit
        if event.kind == "generation_recovery_cycle"
    ]
    assert [event.payload["cycle"] for event in cycles] == [1, 2, 3]
    exhausted = result.raw_audit[-1]
    assert exhausted.kind == "adapter_exhausted"
    assert exhausted.payload["generation_recovery_cycles"] == 3


def test_common_generation_recovery_can_produce_executable_child():
    recovered = make_team_proposal("recovered")
    adapter = QueueAdapter([None, recovered])
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=True, report="done")]
    )

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry")

    assert result.success
    assert result.accounting.physical_transactions == 1
    assert [feedback.kind for feedback in adapter.feedback] == [
        FeedbackKind.GENERATION_RECOVERY
    ]


def test_blocked_recovery_child_does_not_reset_generation_cycle_budget():
    rejected = make_team_proposal("same-blocked-child")
    adapter = QueueAdapter(
        [
            rejected,
            None,
            rejected,
            None,
            rejected,
            None,
            rejected,
            None,
        ]
    )
    runtime = ScriptedRuntime(
        [VerifierOutcome(False, failure_code="NO_PROGRESS", report="blocked")]
    )

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry")

    assert result.status is RunStatus.ADAPTER_EXHAUSTED
    assert result.terminal_reason == "common_generation_recovery_exhausted"
    assert result.accounting.physical_transactions == 1
    assert result.accounting.exact_blocked_candidates == 3
    assert sum(
        event.kind == "generation_recovery_cycle" for event in result.raw_audit
    ) == 3


def test_configured_terminal_predicate_can_authoritatively_confirm_terminal():
    adapter = QueueAdapter([make_team_proposal("accepted-terminal")])
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=False, report="accepted")]
    )
    predicate_calls: list[int] = []

    def is_terminal(observation, outcome):
        predicate_calls.append(observation.sequence)
        return True

    result = ExperimentEngine(
        adapter,
        replace(runtime.callbacks(), terminal_predicate=is_terminal),
    ).run("carry")

    assert result.success
    assert predicate_calls == [1]
    assert result.accepted_manifest[0].verifier.terminal is True
    assert adapter.rollouts[0].verifier.terminal is True
    physical_event = next(
        event for event in result.raw_audit if event.kind == "physical_rollout"
    )
    assert physical_event.payload["terminal"] is False


def test_fallback_post_observation_is_normalized_into_dispatch_for_verifier():
    adapter = QueueAdapter([make_team_proposal("physical")])
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=True, report="done")]
    )
    callbacks = runtime.callbacks()
    verifier_observations: list[Observation | None] = []

    def dispatch_without_observation(policy):
        result = runtime.dispatch(policy)
        return PhysicalDispatchResult(
            result.physically_dispatched,
            result.physical_transaction_completed,
            observation=None,
            evidence=dict(result.evidence),
            error=result.error,
        )

    def verify_with_normalized_dispatch(before, policy, dispatch):
        verifier_observations.append(dispatch.observation)
        return runtime.verify(before, policy, dispatch)

    result = ExperimentEngine(
        adapter,
        replace(
            callbacks,
            dispatch=dispatch_without_observation,
            verify=verify_with_normalized_dispatch,
        ),
    ).run("carry")

    assert result.success
    assert [item.sequence for item in verifier_observations if item is not None] == [1]
    assert adapter.rollouts[0].dispatch.observation == verifier_observations[0]


def test_verifier_exception_after_dispatch_rolls_back_parent_checkpoint():
    adapter = QueueAdapter([make_team_proposal("physical")])
    runtime = ScriptedRuntime([])

    def verifier_raises(*args):
        raise RuntimeError("verifier unavailable")

    result = ExperimentEngine(
        adapter, replace(runtime.callbacks(), verify=verifier_raises)
    ).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.terminal_reason == "verification_callback_failed"
    assert result.accounting.physical_transactions == 1
    assert result.accounting.rollbacks == 1
    assert runtime.rollback_calls == ["checkpoint_0"]


def test_physical_audit_sink_exception_cannot_preempt_rollback():
    adapter = QueueAdapter([make_team_proposal("physical")])
    runtime = ScriptedRuntime(
        [VerifierOutcome(False, failure_code="DROP", report="drop")]
    )

    def sink(event):
        if event.kind == "physical_rollout":
            raise RuntimeError("audit disk full")

    result = ExperimentEngine(
        adapter, replace(runtime.callbacks(), audit_sink=sink)
    ).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.terminal_reason == "audit_sink_failed_after_physical_rollout"
    assert result.accounting.rollbacks == 1
    assert runtime.rollback_calls == ["checkpoint_0"]
    assert any(event.kind == "audit_sink_failed" for event in result.raw_audit)


def test_accepted_checkpoint_audit_failure_rolls_back_before_manifest_commit():
    adapter = QueueAdapter([make_team_proposal("accepted")])
    runtime = ScriptedRuntime([VerifierOutcome(True, terminal=True)])

    def sink(event):
        if event.kind == "accepted_checkpoint":
            raise RuntimeError("audit disk full")

    result = ExperimentEngine(
        adapter, replace(runtime.callbacks(), audit_sink=sink)
    ).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.terminal_reason == "audit_sink_failed_before_checkpoint_commit"
    assert result.accepted_manifest == ()
    assert result.observation.sequence == 0
    assert runtime.rollback_calls == ["checkpoint_0"]


def test_checkpoint_save_exception_after_acceptance_rolls_back_parent():
    adapter = QueueAdapter([make_team_proposal("accepted")])
    runtime = ScriptedRuntime([VerifierOutcome(True, terminal=False)])
    callbacks = runtime.callbacks()
    saves = 0

    def save_then_fail(observation):
        nonlocal saves
        saves += 1
        if saves == 1:
            return runtime.save_checkpoint(observation)
        raise RuntimeError("checkpoint disk full")

    result = ExperimentEngine(
        adapter, replace(callbacks, save_checkpoint=save_then_fail)
    ).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.terminal_reason == "checkpoint_callback_failed"
    assert result.accepted_manifest == ()
    assert result.observation.sequence == 0
    assert runtime.rollback_calls == ["checkpoint_0"]


def test_dispatch_exception_is_conservatively_accounted_and_rolled_back():
    adapter = QueueAdapter([make_team_proposal("physical")])
    runtime = ScriptedRuntime([])

    def dispatch_raises(policy):
        raise RuntimeError("publisher failed after send")

    result = ExperimentEngine(
        adapter, replace(runtime.callbacks(), dispatch=dispatch_raises)
    ).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.accounting.physical_transactions == 1
    assert result.accounting.incomplete_transactions == 1
    assert result.accounting.rollbacks == 1
    assert runtime.rollback_calls == ["checkpoint_0"]


@pytest.mark.parametrize(
    "shorthand",
    [True, Observation(0, {"state": "parent"})],
)
def test_strict_rollback_contract_rejects_shorthand(shorthand):
    from coevolve_experiments.runtime.engine import _rollback_result

    with pytest.raises(RuntimeEngineError, match="explicit RollbackResult"):
        _rollback_result(
            shorthand,
            Observation(0, {"state": "parent"}),
            allow_shorthand=False,
        )


def test_smoke_runtime_retains_boolean_rollback_compatibility():
    adapter = QueueAdapter(
        [make_team_proposal("rejected"), make_team_proposal("repair")]
    )
    runtime = ScriptedRuntime(
        [
            VerifierOutcome(False, failure_code="NO_PROGRESS", report="failed"),
            VerifierOutcome(True, terminal=True, report="done"),
        ]
    )
    callbacks = runtime.callbacks()

    def compatible_rollback(checkpoint, expected):
        runtime.rollback(checkpoint, expected)
        return True

    result = ExperimentEngine(
        adapter,
        replace(callbacks, rollback=compatible_rollback),
    ).run("carry")

    assert result.success
    assert result.accounting.rollbacks == 1


@pytest.mark.parametrize(
    ("failed_event", "expected_reason"),
    [
        (
            "adapter_rollout_update",
            "audit_sink_failed_after_adapter_rollout_update",
        ),
        ("terminal_verified", "audit_sink_failed_before_terminal_seal"),
    ],
)
def test_postcommit_audit_failure_recovers_latest_accepted_checkpoint(
    failed_event,
    expected_reason,
):
    adapter = QueueAdapter([make_team_proposal("accepted")])
    runtime = ScriptedRuntime([VerifierOutcome(True, terminal=True, report="done")])

    def sink(event):
        if event.kind == failed_event:
            raise RuntimeError("audit disk full")

    result = ExperimentEngine(
        adapter,
        replace(runtime.callbacks(), audit_sink=sink),
    ).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.terminal_reason == expected_reason
    assert [entry.checkpoint_id for entry in result.accepted_manifest] == [
        "checkpoint_1"
    ]
    assert runtime.rollback_calls == ["checkpoint_1"]
    assert result.observation.sequence == 1


def test_rollback_audit_failure_stops_after_recovery_without_memory_update():
    adapter = QueueAdapter([make_team_proposal("rejected")])
    runtime = ScriptedRuntime(
        [VerifierOutcome(False, failure_code="DROP", report="drop")]
    )

    def sink(event):
        if event.kind == "rollback":
            raise RuntimeError("audit disk full")

    result = ExperimentEngine(
        adapter,
        replace(runtime.callbacks(), audit_sink=sink),
    ).run("carry")

    assert result.status is RunStatus.INFRASTRUCTURE_FAILED
    assert result.terminal_reason == "audit_sink_failed_after_rollback"
    assert runtime.rollback_calls == ["checkpoint_0"]
    assert result.observation.sequence == 0
    assert adapter.rollouts == []
