from __future__ import annotations

import pytest

from coevolve_experiments.contracts import MethodVariant
from coevolve_experiments.methods import (
    AdapterUsageSummary,
    MemorySwitches,
    MethodDescriptor,
)
from coevolve_experiments.runtime import (
    EngineConfig,
    ExperimentEngine,
    RunStatus,
    ScriptedRuntime,
    VerifierOutcome,
    make_team_proposal,
)


class FinalizeFailingAdapter:
    descriptor = MethodDescriptor(
        variant=MethodVariant.COEVOLE_FULL,
        allowed_roles=frozenset(),
        hierarchical_evolution=True,
        durable_dynamic_memory=True,
        memory_switches=MemorySwitches.enabled(),
    )

    def __init__(self, proposals):
        self._proposals = iter(proposals)

    def propose_initial(self, context):
        return next(self._proposals, None)

    def propose_next(self, context, feedback):
        return next(self._proposals, None)

    def on_verified_rollout(self, rollout, context):
        return {}

    def usage_summary(self):
        return AdapterUsageSummary(0, 0, 0, {})

    def finalize_memory(self):
        raise ValueError("cannot seal memory audit")


def test_finalize_failure_invalidates_success_adapter_exhaustion_and_budget():
    proposal = make_team_proposal("move")
    cases = (
        (
            FinalizeFailingAdapter([proposal]),
            ScriptedRuntime([VerifierOutcome(True, terminal=True)]),
            EngineConfig(),
        ),
        (
            FinalizeFailingAdapter([]),
            ScriptedRuntime([]),
            EngineConfig(),
        ),
        (
            FinalizeFailingAdapter([proposal]),
            ScriptedRuntime([VerifierOutcome(False, failure_code="NO_PROGRESS")]),
            EngineConfig(physical_budget=1),
        ),
    )
    for adapter, runtime, config in cases:
        result = ExperimentEngine(
            adapter,
            runtime.callbacks(),
            config=config,
        ).run("carry")
        assert result.status is RunStatus.INFRASTRUCTURE_FAILED
        assert result.terminal_reason == "memory_finalize_failed"
        assert any(event.kind == "memory_finalize_failed" for event in result.raw_audit)


def test_finalize_failure_does_not_hide_more_severe_rollback_failure():
    proposal = make_team_proposal("move")
    adapter = FinalizeFailingAdapter([proposal])
    runtime = ScriptedRuntime(
        [VerifierOutcome(False, failure_code="NO_PROGRESS")],
        rollback_succeeds=False,
    )

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry")

    assert result.status is RunStatus.ROLLBACK_FAILED
    assert result.terminal_reason == "rollback_or_recovery_verification_failed"
    assert any(event.kind == "memory_finalize_failed" for event in result.raw_audit)


def test_unexpected_adapter_exception_still_invokes_memory_finalize():
    class RaisingAdapter(FinalizeFailingAdapter):
        def __init__(self):
            super().__init__([])
            self.finalized = False

        def propose_initial(self, context):
            raise ValueError("model backend failed")

        def finalize_memory(self):
            self.finalized = True
            return None

    adapter = RaisingAdapter()
    engine = ExperimentEngine(adapter, ScriptedRuntime([]).callbacks())

    with pytest.raises(ValueError, match="model backend failed"):
        engine.run("carry")

    assert adapter.finalized is True
