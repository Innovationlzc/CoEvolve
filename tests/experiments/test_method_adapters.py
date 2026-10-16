from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import pytest

from coevolve_experiments.methods import (
    COHERENT_PROVENANCE,
    REMAC_PROVENANCE,
    AdapterContractError,
    AdapterContext,
    CoherentAdaptedAdapter,
    ExecutionCapability,
    FlatReflectiveRepairAdapter,
    FullCoEvolveAdapter,
    MethodRole,
    NoDynamicExperienceAdapter,
    PolicyRepresentation,
    RemacAdaptedAdapter,
    ScriptedRoleBackend,
    create_method_adapter,
)
from coevolve_experiments.contracts import MethodVariant
from coevolve_experiments.runtime import (
    CandidateFeedback,
    CheckpointRef,
    ExperimentEngine,
    FeedbackKind,
    Observation,
    PolicyValidator,
    RolloutRecord,
    RunAccounting,
    ScriptedRuntime,
    VerifierOutcome,
    make_team_proposal,
)
from coevolve_experiments.runtime.contracts import PhysicalDispatchResult


def _context() -> AdapterContext:
    return AdapterContext(
        instruction="carry the box",
        observation=Observation(0, {"phase": "carry"}),
        checkpoint=CheckpointRef("c0"),
        remaining_physical_budget=80,
        accepted_manifest=(),
        latest_physical_failure=None,
        blocked_ast_fingerprints=frozenset(),
        accounting=RunAccounting(),
    )


def _strategy(label: str = "plan") -> dict[str, Any]:
    return {
        "updated_plan": {"plan": label},
        "committed_strategy": {
            "primitive": "cooperative_move",
            "semantic_objective": "advance",
            "coordination_mode": "synchronized",
        },
    }


def test_all_method_adapters_are_explicitly_contract_smoke_only():
    for adapter_type in (
        FullCoEvolveAdapter,
        NoDynamicExperienceAdapter,
        FlatReflectiveRepairAdapter,
        RemacAdaptedAdapter,
        CoherentAdaptedAdapter,
    ):
        descriptor = adapter_type.descriptor
        assert (
            descriptor.policy_representation
            is PolicyRepresentation.TEAM_INTENT_IR
        )
        assert (
            descriptor.execution_capability
            is ExecutionCapability.CONTRACT_SMOKE_ONLY
        )


def _rollout(proposal, *, accepted: bool = False, index: int = 1) -> RolloutRecord:
    policy = PolicyValidator().validate(proposal)
    outcome = VerifierOutcome(
        accepted=accepted,
        terminal=False,
        failure_code=None if accepted else "NO_PROGRESS",
        report="ok" if accepted else "no progress",
    )
    return RolloutRecord(
        transaction_index=index,
        checkpoint=CheckpointRef("c0"),
        observation_before=Observation(0),
        policy=policy,
        dispatch=PhysicalDispatchResult(True, True, Observation(1)),
        verifier=outcome,
        observation_after=Observation(1),
    )


def _rejected(proposal):
    outcome = VerifierOutcome(
        False, failure_code="NO_PROGRESS", report="no progress"
    )
    return CandidateFeedback(
        FeedbackKind.VERIFIED_REJECTED,
        outcome.report,
        proposal=proposal,
        verifier=outcome,
    )


class SpyMemory:
    def __init__(self) -> None:
        self.strategy_retrievals = 0
        self.exemplar_retrievals = 0
        self.strategy_promotions = 0
        self.exemplar_promotions = 0
        self.exemplar_queries = []

    def retrieve_strategies(
        self, context: AdapterContext
    ) -> Sequence[Mapping[str, Any]]:
        self.strategy_retrievals += 1
        return ({"strategy": "verified"},)

    def retrieve_action_exemplars(
        self, context: AdapterContext, query
    ) -> Sequence[Mapping[str, Any]]:
        self.exemplar_retrievals += 1
        self.exemplar_queries.append(query)
        return ({"action": "verified"},)

    def promote_strategy(self, proposal: Any, rollout: RolloutRecord) -> None:
        self.strategy_promotions += 1

    def promote_action_exemplar(
        self, proposal: Any, rollout: RolloutRecord
    ) -> None:
        self.exemplar_promotions += 1


def test_full_hierarchical_failure_role_sequence_and_memory_update():
    first = make_team_proposal("first")
    repaired = make_team_proposal("repaired")
    backend = ScriptedRoleBackend(
        [
            (MethodRole.TASK_ORCHESTRATOR.value, _strategy("p0")),
            (MethodRole.POLICY_GENERATOR.value, first),
            (MethodRole.MEMORY_UPDATE.value, {"strategy": "avoid no progress"}),
            (MethodRole.EVOLUTION_MANAGER.value, {"level": "coordination"}),
            (MethodRole.MUTATION_OPERATOR.value, _strategy("p1")),
            (MethodRole.POLICY_GENERATOR.value, repaired),
        ]
    )
    memory = SpyMemory()
    adapter = FullCoEvolveAdapter(backend, memory=memory)

    proposal = adapter.propose_initial(_context())
    adapter.on_verified_rollout(_rollout(proposal), _context())
    assert adapter.propose_next(_context(), _rejected(proposal)) == repaired

    assert adapter.persistent_plan == {"plan": "p1"}
    assert adapter.committed_strategy == {
        "primitive": "cooperative_move",
        "semantic_objective": "advance",
        "coordination_mode": "synchronized",
    }

    assert backend.roles == (
        "task_orchestrator",
        "policy_generator",
        "memory_update",
        "evolution_manager",
        "mutation_operator",
        "policy_generator",
    )
    assert memory.strategy_promotions == 1
    assert memory.exemplar_promotions == 0
    assert adapter.descriptor.hierarchical_evolution is True
    assert memory.strategy_retrievals == 2
    assert memory.exemplar_retrievals == 2
    assert all(
        query.primitive == "cooperative_move"
        and query.semantic_objective == "advance"
        and query.coordination_mode == "synchronized"
        for query in memory.exemplar_queries
    )
    for call_index in (0, 3, 4):
        assert "strategy_records" in backend.calls[call_index].payload
        assert "action_exemplars" not in backend.calls[call_index].payload
    for call_index in (1, 5):
        assert "strategy_records" not in backend.calls[call_index].payload
        assert backend.calls[call_index].payload["action_exemplars"] == [
            {"action": "verified"}
        ]
    assert "strategy_records" not in backend.calls[2].payload
    assert "action_exemplars" not in backend.calls[2].payload


def test_no_dynamic_skips_memory_writer_and_closes_four_gates():
    first = make_team_proposal("first")
    repaired = make_team_proposal("repaired")
    backend = ScriptedRoleBackend(
        [
            ("task_orchestrator", _strategy("p0")),
            ("policy_generator", first),
            ("evolution_manager", {"level": "coordination"}),
            ("mutation_operator", _strategy("p1")),
            ("policy_generator", repaired),
        ]
    )
    memory = SpyMemory()
    adapter = NoDynamicExperienceAdapter(backend, memory=memory)

    proposal = adapter.propose_initial(_context())
    update = adapter.on_verified_rollout(_rollout(proposal), _context())
    adapter.propose_next(_context(), _rejected(proposal))

    assert "memory_update" not in backend.roles
    assert adapter.usage_summary().model_calls == 5
    assert update["memory_update_invoked"] is False
    assert update["strategy_promoted"] is False
    assert adapter.descriptor.memory_switches.all_disabled
    assert adapter.descriptor.safety_invariants.all_enabled
    assert (
        memory.strategy_retrievals,
        memory.exemplar_retrievals,
        memory.strategy_promotions,
        memory.exemplar_promotions,
    ) == (0, 0, 0, 0)


def test_flat_keeps_initial_plan_and_rewrites_only_executable():
    first = make_team_proposal("first")
    rewritten = make_team_proposal("rewrite")
    fixed_plan = {
        **_strategy("fixed"),
        "phases": ["carry", "place"],
    }
    backend = ScriptedRoleBackend(
        [
            ("flat_task_planner", fixed_plan),
            ("flat_strategy_selector", fixed_plan["committed_strategy"]),
            ("flat_policy_generator", first),
            ("memory_update", {"strategy": "repair"}),
            ("flat_strategy_selector", fixed_plan["committed_strategy"]),
            ("reflective_rewrite", rewritten),
        ]
    )
    adapter = FlatReflectiveRepairAdapter(backend)

    initial = adapter.propose_initial(_context())
    adapter.on_verified_rollout(_rollout(initial), _context())
    assert adapter.propose_next(_context(), _rejected(initial)) == rewritten

    assert adapter.persistent_plan == fixed_plan["updated_plan"]
    assert backend.roles == (
        "flat_task_planner",
        "flat_strategy_selector",
        "flat_policy_generator",
        "memory_update",
        "flat_strategy_selector",
        "reflective_rewrite",
    )
    assert "evolution_manager" not in backend.roles
    assert "mutation_operator" not in backend.roles


def test_flat_requeries_latest_strategy_memory_without_replanning():
    class UpdatingMemory(SpyMemory):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[Mapping[str, Any]] = [
                {"coordination_pattern": "initial exact-phase record"}
            ]

        def retrieve_strategies(
            self, context: AdapterContext
        ) -> Sequence[Mapping[str, Any]]:
            del context
            self.strategy_retrievals += 1
            return tuple(self.records)

        def promote_strategy(self, proposal: Any, rollout: RolloutRecord) -> None:
            del rollout
            self.strategy_promotions += 1
            self.records.append(dict(proposal))

    first = make_team_proposal("first")
    rewritten = make_team_proposal("rewrite")
    fixed_plan = _strategy("fixed")
    backend = ScriptedRoleBackend(
        [
            ("flat_task_planner", fixed_plan),
            ("flat_strategy_selector", fixed_plan["committed_strategy"]),
            ("flat_policy_generator", first),
            ("memory_update", {"strategy": "newly promoted exact-phase repair"}),
            ("flat_strategy_selector", fixed_plan["committed_strategy"]),
            ("reflective_rewrite", rewritten),
        ]
    )
    memory = UpdatingMemory()
    adapter = FlatReflectiveRepairAdapter(backend, memory=memory)

    initial = adapter.propose_initial(_context())
    adapter.on_verified_rollout(_rollout(initial), _context())
    adapter.propose_next(_context(), _rejected(initial))

    assert adapter.persistent_plan is fixed_plan["updated_plan"]
    assert memory.strategy_retrievals == 2
    assert backend.calls[0].payload["strategy_records"] == [
        {"coordination_pattern": "initial exact-phase record"}
    ]
    assert backend.calls[1].payload["strategy_records"] == [
        {"coordination_pattern": "initial exact-phase record"}
    ]
    assert backend.calls[4].payload["strategy_records"] == [
        {"coordination_pattern": "initial exact-phase record"},
        {"strategy": "newly promoted exact-phase repair"},
    ]
    assert backend.calls[4].payload["fixed_task_plan"] is fixed_plan["updated_plan"]
    assert backend.calls[5].payload["fixed_task_plan"] is fixed_plan["updated_plan"]
    assert backend.calls[5].payload["rewrite_scope"] == "executable_only"
    for call_index in (2, 5):
        assert "strategy_records" not in backend.calls[call_index].payload
        assert backend.calls[call_index].payload["action_exemplars"] == [
            {"action": "verified"}
        ]
    assert "evolution_manager" not in backend.roles
    assert "mutation_operator" not in backend.roles


def test_flat_selector_cannot_smuggle_a_task_plan_update():
    fixed_plan = _strategy("fixed")
    backend = ScriptedRoleBackend(
        [
            ("flat_task_planner", fixed_plan),
            (
                "flat_strategy_selector",
                {
                    "updated_plan": {"plan": "rewritten"},
                    "committed_strategy": fixed_plan["committed_strategy"],
                },
            ),
        ]
    )
    adapter = FlatReflectiveRepairAdapter(backend)

    with pytest.raises(
        AdapterContractError,
        match="cannot return task-plan fields",
    ):
        adapter.propose_initial(_context())

    assert backend.roles == ("flat_task_planner", "flat_strategy_selector")


def test_remac_uses_task_replan_and_bounded_run_local_reflection():
    first = make_team_proposal("first")
    repaired = make_team_proposal("task-replan")
    backend = ScriptedRoleBackend(
        [
            ("remac_planner", {"remaining": ["carry", "place"]}),
            ("remac_precondition_reflection", first),
            ("remac_postcondition_reflection", {"cause": "blocked route"}),
            ("remac_replanner", {"remaining": ["alternate", "place"]}),
            ("remac_precondition_reflection", repaired),
        ]
    )
    memory = SpyMemory()
    adapter = RemacAdaptedAdapter(
        backend, memory=memory, max_postcondition_retries=1
    )

    initial = adapter.propose_initial(_context())
    adapter.on_verified_rollout(_rollout(initial), _context())
    assert adapter.propose_next(_context(), _rejected(initial)) == repaired
    assert adapter.propose_next(_context(), _rejected(repaired)) is None

    assert backend.roles == (
        "remac_planner",
        "remac_precondition_reflection",
        "remac_postcondition_reflection",
        "remac_replanner",
        "remac_precondition_reflection",
    )
    assert adapter.run_local_reflections
    assert adapter.descriptor.durable_dynamic_memory is False
    assert memory.strategy_retrievals == memory.strategy_promotions == 0


def test_remac_candidate_contract_failures_do_not_consume_physical_retry_cap():
    first = make_team_proposal("first")
    contract_repair = make_team_proposal("contract-repair")
    physical_repair = make_team_proposal("physical-repair")
    backend = ScriptedRoleBackend(
        [
            ("remac_planner", {"plan": "initial"}),
            ("remac_precondition_reflection", first),
            ("remac_postcondition_reflection", {"cause": "invalid contract"}),
            ("remac_replanner", {"plan": "contract repair"}),
            ("remac_precondition_reflection", contract_repair),
            ("remac_postcondition_reflection", {"cause": "physical failure"}),
            ("remac_replanner", {"plan": "physical repair"}),
            ("remac_precondition_reflection", physical_repair),
        ]
    )
    adapter = RemacAdaptedAdapter(backend, max_postcondition_retries=1)
    initial = adapter.propose_initial(_context())
    invalid = CandidateFeedback(
        FeedbackKind.INVALID_POLICY,
        "schema mismatch",
        proposal=initial,
    )

    assert adapter.propose_next(_context(), invalid) == contract_repair
    assert adapter.propose_next(_context(), _rejected(contract_repair)) == physical_repair
    assert adapter.propose_next(_context(), _rejected(physical_repair)) is None


def test_remac_local_retry_exhaustion_enters_common_generation_recovery():
    initial = make_team_proposal("initial")
    recovered = make_team_proposal("fresh-full-plan")
    backend = ScriptedRoleBackend(
        [
            ("remac_planner", {"plan": "initial"}),
            ("remac_precondition_reflection", initial),
            ("remac_planner", {"plan": "fresh full task"}),
            ("remac_precondition_reflection", recovered),
        ]
    )
    adapter = RemacAdaptedAdapter(backend, max_postcondition_retries=0)

    proposal = adapter.propose_initial(_context())
    assert adapter.propose_next(_context(), _rejected(proposal)) is None
    recovery = CandidateFeedback(
        FeedbackKind.GENERATION_RECOVERY,
        "common host recovery cycle 1",
    )

    assert adapter.propose_next(_context(), recovery) == recovered
    assert backend.calls[2].payload["replan_scope"] == (
        "full_task_generation_recovery"
    )
    assert adapter._consecutive_retries == 0
    backend.assert_exhausted()


def test_coherent_enforces_paired_feasibility_adjustment_and_history_cap():
    draft = make_team_proposal("draft")
    adjusted = make_team_proposal("adjusted")
    physical_repair = make_team_proposal("physical-repair")
    backend = ScriptedRoleBackend(
        [
            ("coherent_central_proposer", draft),
            ("coherent_front_feasibility", {"feasible": True}),
            ("coherent_back_feasibility", {"feasible": False, "reason": "reach"}),
            ("coherent_feedback_adjuster", adjusted),
            ("coherent_front_feasibility", {"feasible": True}),
            ("coherent_back_feasibility", {"feasible": True}),
            ("coherent_feedback_adjuster", physical_repair),
            ("coherent_front_feasibility", {"feasible": True}),
            ("coherent_back_feasibility", {"feasible": True}),
            ("coherent_central_proposer", draft),
            ("coherent_front_feasibility", {"feasible": True}),
            ("coherent_back_feasibility", {"feasible": True}),
        ]
    )
    adapter = CoherentAdaptedAdapter(backend)
    proposal = adapter.propose_initial(_context())
    assert proposal == adjusted
    assert adapter.propose_next(_context(), _rejected(proposal)) == physical_repair

    for index in range(7):
        adapter.on_verified_rollout(
            _rollout(adjusted, accepted=True, index=index), _context()
        )
    assert len(adapter.history) == 5
    assert adapter.history[0]["transaction_index"] == 2
    assert adapter.descriptor.hierarchical_evolution is False
    assert "evolution_manager" not in backend.roles

    # A reused adapter starts a clean run: PEFA history is ephemeral, not a
    # hidden durable-memory channel between experimental repetitions.
    adapter.propose_initial(_context())
    assert adapter.history == ()
    assert adapter.usage_summary().model_calls == 3


def test_coherent_feasibility_exhaustion_requests_fresh_common_recovery_child():
    blocked = make_team_proposal("blocked")
    recovered = make_team_proposal("fresh-central-child")
    backend = ScriptedRoleBackend(
        [
            ("coherent_central_proposer", blocked),
            ("coherent_front_feasibility", {"feasible": False}),
            ("coherent_back_feasibility", {"feasible": True}),
            ("coherent_central_proposer", recovered),
            ("coherent_front_feasibility", {"feasible": True}),
            ("coherent_back_feasibility", {"feasible": True}),
        ]
    )
    adapter = CoherentAdaptedAdapter(
        backend,
        max_feasibility_adjustments=0,
    )

    assert adapter.propose_initial(_context()) is None
    recovery = CandidateFeedback(
        FeedbackKind.GENERATION_RECOVERY,
        "common host recovery cycle 1",
    )

    assert adapter.propose_next(_context(), recovery) == recovered
    assert backend.calls[3].payload["proposal_mode"] == "generation_recovery"
    backend.assert_exhausted()


@pytest.mark.parametrize(
    ("adapter_factory", "script"),
    [
        (
            FullCoEvolveAdapter,
            [
                ("task_orchestrator", _strategy("p")),
                ("policy_generator", make_team_proposal("terminal")),
                ("memory_update", {}),
            ],
        ),
        (
            NoDynamicExperienceAdapter,
            [
                ("task_orchestrator", _strategy("p")),
                ("policy_generator", make_team_proposal("terminal")),
            ],
        ),
        (
            FlatReflectiveRepairAdapter,
            [
                ("flat_task_planner", _strategy("fixed")),
                (
                    "flat_strategy_selector",
                    _strategy("fixed")["committed_strategy"],
                ),
                ("flat_policy_generator", make_team_proposal("terminal")),
                ("memory_update", {}),
            ],
        ),
        (
            RemacAdaptedAdapter,
            [
                ("remac_planner", {"plan": "full"}),
                ("remac_precondition_reflection", make_team_proposal("terminal")),
            ],
        ),
        (
            CoherentAdaptedAdapter,
            [
                ("coherent_central_proposer", make_team_proposal("terminal")),
                ("coherent_front_feasibility", {"feasible": True}),
                ("coherent_back_feasibility", {"feasible": True}),
            ],
        ),
    ],
)
def test_each_adapter_has_a_scripted_one_transaction_dry_run(
    adapter_factory, script
):
    backend = ScriptedRoleBackend(script)
    adapter = adapter_factory(backend)
    runtime = ScriptedRuntime([VerifierOutcome(True, terminal=True, report="done")])

    result = ExperimentEngine(adapter, runtime.callbacks()).run("carry the box")

    assert result.success
    assert result.accounting.physical_transactions == 1
    assert result.accounting.accepted_transactions == 1
    assert len(result.accepted_manifest) == 1
    backend.assert_exhausted()


def test_clean_room_provenance_is_exact_and_distinguishes_local_remac_pdf():
    assert REMAC_PROVENANCE.repository_commit == (
        "8344151d5e6f5a3e1d6ea36530a274f97542bbbd"
    )
    assert REMAC_PROVENANCE.upstream_paper_revision.endswith("v2")
    assert "v1" in REMAC_PROVENANCE.local_paper_revision
    assert COHERENT_PROVENANCE.repository_commit == (
        "17554a52792e302921bbb5e5fd3b4049e25b50d4"
    )
    assert "clean-room" in COHERENT_PROVENANCE.adaptation_basis


def test_raw_model_payload_cannot_supply_source_or_loose_boolean_types():
    intent = {
        "commands": [
            {
                "actor": actor,
                "primitive": "cooperative_move",
                "objective": "advance",
                "mode": "sync",
                "arguments": {},
            }
            for actor in ("front", "back")
        ],
        "atomic": True,
    }
    with_source = ScriptedRoleBackend(
        [
            ("flat_task_planner", _strategy("fixed")),
            (
                "flat_strategy_selector",
                _strategy("fixed")["committed_strategy"],
            ),
            ("flat_policy_generator", {"intent": intent, "source": "open('x')"}),
        ]
    )
    with pytest.raises(AdapterContractError, match="TeamIntent only"):
        FlatReflectiveRepairAdapter(with_source).propose_initial(_context())

    loose_boolean = ScriptedRoleBackend(
        [
            ("flat_task_planner", _strategy("fixed")),
            (
                "flat_strategy_selector",
                _strategy("fixed")["committed_strategy"],
            ),
            (
                "flat_policy_generator",
                {"intent": {**intent, "atomic": "true"}},
            ),
        ]
    )
    with pytest.raises(AdapterContractError, match="atomic must be a boolean"):
        FlatReflectiveRepairAdapter(loose_boolean).propose_initial(_context())


def test_every_method_payload_receives_the_same_images_and_metadata():
    proposal = make_team_proposal("observe")
    backend = ScriptedRoleBackend(
        [
            ("task_orchestrator", _strategy("p")),
            ("policy_generator", proposal),
        ]
    )
    context = replace(
        _context(),
        observation=Observation(
            0,
            {"phase": "carry"},
            {"global": "global-frame", "payload": "payload-frame"},
            {"serializer": "frozen-v1"},
        ),
    )

    FullCoEvolveAdapter(backend).propose_initial(context)

    for call in backend.calls:
        assert call.payload["images"] == {
            "global": "global-frame",
            "payload": "payload-frame",
        }
        assert call.payload["observation_metadata"] == {"serializer": "frozen-v1"}


def test_hierarchical_orchestration_rejects_legacy_mixed_plan_strategy_shape():
    backend = ScriptedRoleBackend(
        [
            (
                "task_orchestrator",
                {
                    "plan": "legacy opaque plan",
                    "committed_strategy": {
                        "primitive": "cooperative_move",
                        "semantic_objective": "advance",
                        "coordination_mode": "synchronized",
                    },
                },
            )
        ]
    )

    with pytest.raises(AdapterContractError, match="requires updated_plan"):
        FullCoEvolveAdapter(backend).propose_initial(_context())


@pytest.mark.parametrize(
    ("variant", "override"),
    [
        (MethodVariant.REMAC_ADAPTED, {"max_postcondition_retries": 9}),
        (
            MethodVariant.COHERENT_ADAPTED,
            {"max_feasibility_adjustments": 9},
        ),
    ],
)
def test_formal_factory_rejects_unregistered_baseline_tuning(variant, override):
    with pytest.raises(ValueError, match="frozen by the experiment protocol"):
        create_method_adapter(variant, ScriptedRoleBackend([]), **override)
