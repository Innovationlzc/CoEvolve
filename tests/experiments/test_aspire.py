"""No-LLM/no-physics mechanism and accounting qualification for ASPIRE."""

from copy import deepcopy
import json

import pytest

from coevolve_experiments.methods.aspire import (
    AspireAdapted, AspireContractError, AspireSearchExhausted,
    FIX_ATTEMPTS, MAX_GENERATIONS, MODEL, POLICY, POPULATION_SIZE,
)


class FakeModel:
    def __init__(self):
        self.calls = []
        self.programs = 0

    def __call__(self, role, payload):
        self.calls.append((role, deepcopy(payload)))
        if role == "aspire_skill_generalization":
            return {
                "reusable": True, "trigger": "support stable and corridor available",
                "summary": "verified transport recipe; assess current clearance",
                "source": "def unexecuted_rewrite(): pass",
            }
        if role in {"aspire_fix_program", "aspire_search_candidate"}:
            self.programs += 1
            return {
                "source": (
                    "def front_policy(api):\n"
                    f"    api.move({self.programs / 100})\n    api.stop_all()\n"
                    "def back_policy(api):\n"
                    f"    api.move({self.programs / 100})\n    api.stop_all()\n"
                ),
                "coordination_mode": "sync", "objective": "reach remaining goal",
                "rationale": "candidate-specific trace hypothesis",
                "hypothesis": f"hypothesis-{self.programs}",
                "metadata": {"method_decision": "unrestricted", "call": self.programs},
            }
        return {"observed": "box remains supported", "uncertain": "movement response"}


def context(adapter, checkpoint="cp0", **overrides):
    value = {
        "instruction": "Complete transport, placement and exit",
        "checkpoint_id": checkpoint,
        "remaining_physical_budget": 80 - adapter.physical_consumed,
        "physical_consumed": adapter.physical_consumed,
        "observation": {"phase": "transport", "payload_height": 0.72},
        "images": [{"path": "actual_current.png", "kind": "current_observation"}],
        "skill_api": "shared SDK specification; methods may choose any legal actions",
    }
    value.update(overrides)
    return value


def feedback(candidate, transaction, *, accepted=False, dispatched=True,
             completed=True, trace=True, images=True, **overrides):
    before = candidate["checkpoint_id"]
    result = {
        "transaction_id": f"transaction-{transaction}",
        "physically_dispatched": dispatched, "completed": completed,
        "accepted": accepted, "checkpoint_before": before,
        "checkpoint_after": f"{before}-accepted" if accepted else before,
        "restore_verified": not accepted,
        "evidence": {"verifier": "accepted" if accepted else "route_boundary"},
        "trace": [{"function": "team_move", "requested": 0.25, "actual": 0.6}] if trace else [],
        "images": ["before.png", "stop_trigger.png", "after.png"] if images else [],
        "phase_completion": 0.25,
    }
    result.update(overrides)
    return result


def reject_next(adapter, transaction):
    candidate = adapter.next_candidate(context(adapter))
    adapter.record_result(candidate["candidate_id"], feedback(candidate, transaction))
    return candidate


def enter_search(adapter):
    for index in range(FIX_ATTEMPTS):
        reject_next(adapter, index)
    return adapter.next_candidate(context(adapter))


def test_policy_declares_real_online_adaptation_not_original_validation():
    assert POLICY["population_size"] == 8
    assert POLICY["model"] == MODEL
    assert POLICY["held_out_validation"] is False
    assert POLICY["original_protocol_reproduction"] is False
    assert POLICY["source_bank"] is None
    assert POLICY["coevolve_mutation_scope"] is False


def test_initial_scene_is_multimodal_and_candidate_metadata_survives():
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    candidate = adapter.next_candidate(context(
        adapter, mutation_scope="magnitude-only", strategy_memory=["forbidden bank"],
    ))
    assert [role for role, _ in model.calls] == ["aspire_scene_analysis", "aspire_fix_program"]
    assert all(payload["images"] for _, payload in model.calls)
    assert all("mutation_scope" not in payload and "strategy_memory" not in payload
               for _, payload in model.calls)
    assert candidate["metadata"] == {"method_decision": "unrestricted", "call": 1}
    assert candidate["coordination_mode"] == "sync"
    assert candidate["objective"] == "reach remaining goal"
    assert adapter.physical_consumed == 0
    assert adapter.skills == ()


def test_failure_enters_actual_trace_and_image_diagnosis_before_repair():
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    rejected = reject_next(adapter, 0)
    repaired = adapter.next_candidate(context(adapter))
    assert [role for role, _ in model.calls][-2:] == ["aspire_trace_diagnosis", "aspire_fix_program"]
    diagnosis = model.calls[-2][1]
    prior = diagnosis["evaluated_programs_and_traces"][0]
    assert prior["source"] == rejected["source"]
    assert prior["feedback"]["trace"][0]["actual"] == 0.6
    assert prior["feedback"]["images"] == ["before.png", "stop_trigger.png", "after.png"]
    assert repaired["source"] != rejected["source"]
    assert adapter.physical_consumed == 1
    assert not adapter.skills


def test_fix_bound_reaches_eight_real_search_candidates_not_named_fix_loop():
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    search = enter_search(adapter)
    generation_calls = [payload for role, payload in model.calls if role == "aspire_search_candidate"]
    assert len(generation_calls) == POPULATION_SIZE
    assert search["kind"] == "search"
    assert len(adapter.state["pool"]) == 7
    assert generation_calls[-1]["candidate_slot"] == 7
    assert len(generation_calls[-1]["generated_siblings"]) == 7
    assert len(generation_calls[0]["top_same_checkpoint_survivors"]) == 3
    assert adapter.physical_consumed == FIX_ATTEMPTS
    # No phantom robot execution for the seven generated siblings.
    adapter.record_result(search["candidate_id"], feedback(search, 3))
    calls_before = adapter.model_calls
    next_sibling = adapter.next_candidate(context(adapter))
    assert next_sibling["source"] != search["source"]
    assert next_sibling["kind"] == "search"
    assert adapter.model_calls == calls_before
    adapter.record_result(next_sibling["candidate_id"], feedback(next_sibling, 4))
    assert adapter.physical_consumed == 5


def test_acceptance_discards_siblings_and_never_compares_different_origins():
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    selected = enter_search(adapter)
    unexecuted = list(adapter.state["pool"])
    result = feedback(selected, 3, accepted=True)
    adapter.record_result(selected["candidate_id"], result)
    successor = adapter.next_candidate(context(adapter, result["checkpoint_after"]))
    assert successor["kind"] == "fix"
    assert successor["checkpoint_id"] != selected["checkpoint_id"]
    assert all(adapter.state["candidates"][cid]["feedback"] is None for cid in unexecuted)
    discards = [event for event in adapter.state["events"] if event["kind"] == "unevaluated_siblings_discarded"]
    assert discards[-1]["candidate_ids"] == unexecuted
    assert adapter.physical_consumed == 4


def test_accepted_skill_reuses_exact_executed_source_not_unexecuted_generalization():
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    candidate = adapter.next_candidate(context(adapter))
    result = feedback(candidate, 0, accepted=True)
    adapter.record_result(candidate["candidate_id"], result)
    skill = adapter.skills[0]
    assert skill["source"] == candidate["source"]
    assert "unexecuted_rewrite" not in skill["source"]
    assert skill["held_out_validated"] is False
    assert skill["applicability_summary_is_hypothesis"] is True
    adapter.next_candidate(context(adapter, result["checkpoint_after"]))
    assert model.calls[-1][1]["verified_skills"][0] == skill


@pytest.mark.parametrize("missing", ["trace", "images"])
def test_accepted_but_missing_multimodal_evidence_does_not_promote(missing):
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    candidate = adapter.next_candidate(context(adapter))
    result = feedback(candidate, 0, accepted=True, **{missing: False})
    adapter.record_result(candidate["candidate_id"], result)
    assert not adapter.skills
    assert "aspire_skill_generalization" not in [role for role, _ in model.calls]
    assert adapter.physical_consumed == 1


def test_static_rejection_is_visible_but_never_a_free_physical_attempt():
    adapter = AspireAdapted(None, FakeModel())
    candidate = adapter.next_candidate(context(adapter))
    result = feedback(candidate, 0, dispatched=False, completed=False)
    adapter.record_result(candidate["candidate_id"], result)
    assert adapter.physical_consumed == 0
    assert adapter.state["candidates"][candidate["candidate_id"]]["status"] == "static_rejected"
    reject_next(adapter, 0)
    assert adapter.physical_consumed == 1


def test_dispatch_is_charged_before_outcome_and_duplicate_receipts_are_idempotent(tmp_path):
    adapter = AspireAdapted(tmp_path, FakeModel())
    candidate = adapter.next_candidate(context(adapter))
    result = feedback(candidate, 0)
    adapter.record_dispatch(candidate["candidate_id"], result["transaction_id"], "cp0")
    adapter.record_dispatch(candidate["candidate_id"], result["transaction_id"], "cp0")
    assert adapter.physical_consumed == 1
    loaded = AspireAdapted(tmp_path, FakeModel())
    assert loaded.physical_consumed == 1
    with pytest.raises(AspireContractError, match="prior candidate"):
        loaded.next_candidate(context(loaded))
    loaded.record_result(candidate["candidate_id"], result)
    loaded.record_result(candidate["candidate_id"], result)
    assert loaded.physical_consumed == 1
    assert loaded.state["candidates"][candidate["candidate_id"]]["metadata"] == candidate["metadata"]
    changed = {**result, "report": "silently rewritten"}
    with pytest.raises(AspireContractError, match="rewrite"):
        loaded.record_result(candidate["candidate_id"], changed)


def test_failed_or_missing_restore_and_incomplete_physics_block_continuation():
    for overrides in ({"restore_verified": False}, {"completed": False}):
        adapter = AspireAdapted(None, FakeModel())
        candidate = adapter.next_candidate(context(adapter))
        adapter.record_result(candidate["candidate_id"], feedback(candidate, 0, **overrides))
        assert adapter.physical_consumed == 1
        with pytest.raises(AspireSearchExhausted):
            adapter.next_candidate(context(adapter))
        assert not adapter.skills


def test_unexecuted_or_incomplete_code_cannot_be_accepted():
    adapter = AspireAdapted(None, FakeModel())
    candidate = adapter.next_candidate(context(adapter))
    with pytest.raises(AspireContractError, match="unexecuted/incomplete"):
        adapter.record_result(candidate["candidate_id"], feedback(
            candidate, 0, accepted=True, dispatched=False, completed=False,
        ))
    assert adapter.physical_consumed == 0


def test_wrong_checkpoint_or_budget_cannot_create_new_opportunity():
    adapter = AspireAdapted(None, FakeModel())
    reject_next(adapter, 0)
    with pytest.raises(AspireContractError, match="checkpoint differs"):
        adapter.next_candidate(context(adapter, "ancestor0"))
    with pytest.raises(AspireContractError, match="erase charged"):
        adapter.next_candidate(context(adapter, remaining_physical_budget=80))
    with pytest.raises(AspireContractError, match="ledgers disagree"):
        adapter.next_candidate(context(adapter, physical_consumed=0))
    with pytest.raises(AspireSearchExhausted):
        adapter.next_candidate(context(adapter, remaining_physical_budget=0))


def test_resume_preserves_full_model_call_and_physical_accounting(tmp_path):
    model = FakeModel()
    adapter = AspireAdapted(tmp_path, model)
    candidate = enter_search(adapter)
    state = adapter.snapshot()
    saved = json.loads((tmp_path / "aspire_state.json").read_text())
    assert saved == state
    resumed = AspireAdapted(tmp_path, FakeModel())
    assert resumed.snapshot() == state
    assert resumed.model_calls == adapter.model_calls
    resumed.record_result(candidate["candidate_id"], feedback(candidate, 3))
    sibling = resumed.next_candidate(context(resumed))
    assert sibling["candidate_id"] == state["pool"][0]
    assert sibling["metadata"] == state["candidates"][state["pool"][0]]["metadata"]
    with pytest.raises(AspireContractError, match="choose durable"):
        AspireAdapted(tmp_path, FakeModel(), state=state)
    state["policy"]["model"] = "different-model"
    with pytest.raises(AspireContractError, match="configuration changed"):
        AspireAdapted(None, FakeModel(), state=state)


def test_inference_failures_are_counted_and_persisted_without_phantom_physics(tmp_path):
    def fail(role, payload):
        raise TimeoutError("provider did not return")
    adapter = AspireAdapted(tmp_path, fail)
    with pytest.raises(TimeoutError):
        adapter.next_candidate(context(adapter))
    assert adapter.model_calls == 1
    assert adapter.physical_consumed == 0
    assert adapter.state["calls"][0]["status"] == "failed_or_interrupted"
    assert json.loads((tmp_path / "aspire_state.json").read_text())["calls"][0]["error_type"] == "TimeoutError"
    resumed_model = FakeModel()
    resumed = AspireAdapted(tmp_path, resumed_model)
    resumed.next_candidate(context(resumed))
    assert resumed_model.calls[0][0] == "aspire_scene_analysis"
    assert resumed.state["checkpoint_epoch"] == 1
    assert resumed.model_calls == 3


def test_interrupted_population_cannot_silently_shrink_k_on_resume(tmp_path):
    model = FakeModel()
    fail_once = {"failed": False}
    def invoke(role, payload):
        if (role == "aspire_search_candidate" and payload["candidate_slot"] == 2
                and not fail_once["failed"]):
            fail_once["failed"] = True
            raise TimeoutError("interrupted third population member")
        return model(role, payload)
    adapter = AspireAdapted(tmp_path, invoke)
    for transaction in range(FIX_ATTEMPTS):
        reject_next(adapter, transaction)
    with pytest.raises(TimeoutError):
        adapter.next_candidate(context(adapter))
    assert len(adapter.state["pool"]) == 2
    assert adapter.state["population_build_generation"] == 1
    assert adapter.physical_consumed == 3
    resumed = AspireAdapted(tmp_path, invoke)
    candidate = resumed.next_candidate(context(resumed))
    assert candidate["slot"] == 0
    assert len(resumed.state["pool"]) == 7
    assert resumed.state["population_build_generation"] is None
    assert resumed.state["generation"] == 1
    calls = [call for call in resumed.state["calls"] if call["role"] == "aspire_search_candidate"]
    assert len(calls) == 9  # eight generated candidates plus the charged failed call
    assert len([call for call in calls if call["status"] == "completed"]) == 8
    assert resumed.physical_consumed == 3


def test_real_image_absence_is_not_silently_text_only_multimodal():
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    with pytest.raises(AspireContractError, match="images required"):
        adapter.next_candidate(context(adapter, images=[]))
    assert model.calls == []


def test_fixed_search_generation_cap_does_not_reset_at_plateau():
    adapter = AspireAdapted(None, FakeModel())
    total = FIX_ATTEMPTS + POPULATION_SIZE * MAX_GENERATIONS
    for transaction in range(total):
        reject_next(adapter, transaction)
    assert adapter.physical_consumed == total
    assert adapter.state["generation"] == MAX_GENERATIONS
    with pytest.raises(AspireSearchExhausted, match="five fixed"):
        adapter.next_candidate(context(adapter))


def test_second_generation_uses_only_actually_evaluated_same_origin_parents():
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    for transaction in range(FIX_ATTEMPTS + POPULATION_SIZE):
        reject_next(adapter, transaction)
    candidate = adapter.next_candidate(context(adapter))
    assert candidate["generation"] == 2
    payloads = [payload for role, payload in model.calls
                if role == "aspire_search_candidate" and payload["generation"] == 2]
    assert len(payloads) == 8
    assert all(parent["feedback"]["physically_dispatched"] is True
               and parent["checkpoint_id"] == "cp0"
               for parent in payloads[0]["top_same_checkpoint_survivors"])


def test_common_route_and_lifecycle_facts_survive_without_scope_leakage():
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    facts = {"lifecycle": {"release_latched": False},
             "active_route_guidance": {"phase": "route_lower"},
             "route_phase_guard": {"upper_y": 2.2},
             "clearance": {"back_robot": 0.019},
             "route_clearance_conflict": {"source": "authored corridor"}}
    adapter.next_candidate(context(adapter, **facts))
    for _, payload in model.calls:
        assert all(payload[key] == value for key, value in facts.items())


def test_diagnosis_and_promotion_attach_real_evidence_at_transport_image_seam():
    model = FakeModel()
    adapter = AspireAdapted(None, model)
    reject_next(adapter, 0)
    repaired = adapter.next_candidate(context(adapter, images=["current.png", "before.png"]))
    diagnosis = [payload for role, payload in model.calls if role == "aspire_trace_diagnosis"][-1]
    assert diagnosis["images"] == ["current.png", "before.png", "stop_trigger.png", "after.png"]
    result = feedback(repaired, 1, accepted=True)
    adapter.record_result(repaired["candidate_id"], result)
    assert model.calls[-1][0] == "aspire_skill_generalization"
    assert model.calls[-1][1]["images"] == result["images"]


def test_deferred_promotion_cannot_interrupt_host_physical_accounting(tmp_path):
    model = FakeModel()
    def callback(role, payload):
        if role == "aspire_skill_generalization":
            raise TimeoutError("post-commit model failure")
        return model(role, payload)
    adapter = AspireAdapted(tmp_path, callback)
    candidate = adapter.next_candidate(context(adapter))
    calls_before = adapter.model_calls
    result = feedback(candidate, 0, accepted=True)
    adapter.record_result(candidate["candidate_id"], result, defer_promotion=True)
    assert adapter.model_calls == calls_before
    assert adapter.physical_consumed == 1
    assert adapter.state["pending_promotions"] == [candidate["candidate_id"]]
    with pytest.raises(TimeoutError):
        adapter.drain_pending_promotions()
    reloaded = AspireAdapted(tmp_path, FakeModel())
    assert reloaded.physical_consumed == 1
    assert reloaded.state["candidates"][candidate["candidate_id"]]["feedback"] == result
    assert not reloaded.skills
    reloaded.drain_pending_promotions()
    assert len(reloaded.skills) == 1
    assert not reloaded.state["pending_promotions"]


def test_retained_checkpoint_acceptance_does_not_reset_search_budget():
    adapter = AspireAdapted(None, FakeModel())
    selected = enter_search(adapter)
    before = adapter.state["checkpoint_epoch"]
    adapter.record_result(selected["candidate_id"], feedback(
        selected, 3, accepted=True, checkpoint_after="cp0", accepted_checkpoint_retained=True,
    ), defer_promotion=True)
    assert adapter.state["fix_candidates"] == 3
    assert adapter.state["generation"] == 1
    assert adapter.state["checkpoint_epoch"] == before + 1
    assert not adapter.state["pool"]
    following = adapter.next_candidate(context(adapter))
    assert following["generation"] == 2
    assert following["kind"] == "search"
    assert adapter._survivors() == []
