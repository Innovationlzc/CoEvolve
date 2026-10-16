"""Offline mechanism tests: mock all FM calls; never start a physical runtime."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import threading

import pytest


from coevolve_experiments.methods import cap_agent0 as cap

SOURCE = "def front_policy(api):\n    api.stop_all()\ndef back_policy(api):\n    api.stop_all()\n"


class Recorder:
    def __init__(self, *, failures=(), all_fail=False, visual_fail=False):
        self.calls = []
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(9)
        self.failures = set(failures)
        self.all_fail = all_fail
        self.visual_fail = visual_fail

    def __call__(self, role, payload):
        with self.lock:
            self.calls.append((role, payload))
        if role == "cap_visual_difference":
            if self.visual_fail:
                raise RuntimeError("provider vision unavailable")
            return {"content": "Payload remains supported; the current goal is not reached.",
                    "usage": {"input_tokens": 20, "output_tokens": 10}}
        if role == "cap_candidate":
            self.barrier.wait(timeout=5)
            index = int(payload["request_id"].split("_")[-1])
            if self.all_fail or index in self.failures:
                raise RuntimeError("bounded mocked provider error")
            return {"source": SOURCE, "coordination_mode": "sync", "objective": "continue",
                    "candidate_index": index, "usage": {"input_tokens": 30, "output_tokens": 20}}
        assert role == "cap_ensemble"
        return {"content": json.dumps({"source": SOURCE, "coordination_mode": "sync",
                                        "objective": "maintain support", "rationale": "evidence"}),
                "usage": {"input_tokens": 100, "output_tokens": 40}}


def test_nine_concurrent_candidates_then_one_synthesis_only():
    call = Recorder()
    adapter = cap.CaPAgent0Adapter(call)
    output = adapter.propose({"task": "carry"}, images=["certified_01.png"])
    candidates = [request for role, request in call.calls if role == "cap_candidate"]
    assert len(call.calls) == 11
    assert sorted(request["temperature"] for request in candidates) == list(cap.TEMPERATURES)
    assert call.calls[0][0] == "cap_visual_difference"
    assert call.calls[-1][0] == "cap_ensemble"
    assert all(request["model"] == cap.MODEL for _, request in call.calls)
    assert output["source"] == SOURCE
    assert output["coordination_mode"] == "sync"
    assert output["objective"] == "maintain support"
    assert output["candidate_count"] == output["candidate_success_count"] == 9
    assert output["candidate_physical_dispatches"] == 0
    assert all(row["physically_executed"] is False for row in output["candidates"])
    assert adapter.snapshot()["call_attempts"] == 11
    completed = [event for event in adapter.events if event["event"] == "call_completed"]
    assert len(completed) == 11 and all("usage" in event["response"] for event in completed)


def test_real_image_payload_and_previous_current_order():
    call = Recorder()
    adapter = cap.CaPAgent0Adapter(call)
    first = adapter.propose({"task": "carry"}, images=["before_front.png", "before_back.png"])
    assert first["visual_mode"] == "initial_description"
    adapter.observe_execution({"accepted": False, "physically_dispatched": True,
                               "reason": "corridor boundary", "checkpoint_restored": True})
    second = adapter.propose({"task": "carry"}, images=["restored_front.png", "restored_back.png"])
    assert second["visual_mode"] == "difference"
    visuals = [payload for role, payload in call.calls if role == "cap_visual_difference"]
    assert visuals[1]["images"] == ["before_front.png", "before_back.png", "restored_front.png", "restored_back.png"]
    body = json.loads(visuals[1]["user"])
    assert body["image_order"] == ["previous", "previous", "current", "current"]
    assert body["last_execution"][0]["checkpoint_restored"] is True
    candidate = next(payload for role, payload in call.calls if role == "cap_candidate" and payload["round"] == 2)
    assert "corridor boundary" in candidate["user"]
    assert candidate["images"] == []  # VDM text, not a silent image-only replacement.


def test_missing_visual_evidence_is_not_replaced_by_text_only_generation():
    call = Recorder()
    with pytest.raises(ValueError, match="image paths"):
        cap.Adapter(call).propose({"state": "ready"})
    assert call.calls == []


def test_visual_provider_failure_is_logged_and_no_candidates_dispatched():
    call = Recorder(visual_fail=True)
    adapter = cap.Adapter(call)
    with pytest.raises(RuntimeError, match="vision unavailable"):
        adapter.propose({}, images=["current.png"])
    assert len(call.calls) == 1
    assert adapter.snapshot()["call_attempts"] == 1
    assert adapter.events[-1]["event"] == "call_failed"


def test_partial_candidate_failures_do_not_silently_reduce_requested_k():
    call = Recorder(failures=[2, 4, 9])
    output = cap.Adapter(call).propose({}, images=["current.png"])
    assert len(call.calls) == 11
    assert output["candidate_count"] == 9
    assert output["candidate_success_count"] == 6
    assert [row["index"] for row in output["candidates"] if not row["ok"]] == [2, 4, 9]
    synthesis = json.loads(call.calls[-1][1]["user"])
    assert len(synthesis["candidate_solutions"]) == 9


def test_all_candidate_failures_have_no_hidden_retries_or_fallback():
    call = Recorder(all_fail=True)
    adapter = cap.Adapter(call)
    with pytest.raises(RuntimeError, match="all nine"):
        adapter.propose({}, images=["current.png"])
    assert len(call.calls) == 10
    assert not any(role == "cap_ensemble" for role, _ in call.calls)
    assert sum(event["event"] == "call_failed" for event in adapter.events) == 9
    assert adapter.snapshot()["call_attempts"] == 10


def test_any_provider_fatal_joins_and_accounts_k9_without_new_generation_cycle():
    class ProviderFatalError(RuntimeError):
        pass
    recorder = Recorder()
    def invoke(role, payload):
        response = recorder(role, payload)
        if role == 'cap_candidate' and payload['request_id'].endswith('_03'):
            raise ProviderFatalError('provider refused the configured model')
        return response
    adapter = cap.Adapter(invoke)
    with pytest.raises(ProviderFatalError):
        adapter.propose({}, images=['current.png'])
    assert len(recorder.calls) == 10
    assert not any(role == 'cap_ensemble' for role, _ in recorder.calls)
    assert len(adapter.events[-1]['candidates']) == 9


def test_nested_private_scopes_not_visible_but_real_physical_feedback_retained():
    call = Recorder()
    observation = {"state": {"mutation_scope": "ONLY_PARAMETER", "clearance": .1},
                   "evolution_controller_decision": {"scope": "FORBIDDEN_SECRET"},
                   "history": [{"minimum_mutation_scope": "SECRET", "wall_contact": False}]}
    cap.Adapter(call).propose(observation, images=["current.png"])
    for _, payload in call.calls:
        assert "ONLY_PARAMETER" not in payload["user"]
        assert "FORBIDDEN_SECRET" not in payload["user"]
        assert "clearance" in payload["user"]
        assert "wall_contact" in payload["user"]
    assert observation["state"]["mutation_scope"] == "ONLY_PARAMETER"


def test_pending_program_requires_real_admission_feedback_before_next_proposal():
    adapter = cap.Adapter(Recorder())
    adapter.propose({}, images=["current.png"])
    with pytest.raises(RuntimeError, match="record the"):
        adapter.propose({}, images=["next.png"])
    adapter.observe_execution({"admission_rejected": True, "physically_dispatched": False,
                               "error": "shared syntax violation"})
    assert adapter.snapshot()["pending"] is None
    assert adapter.snapshot()["feedback"][0]["proposed_program"] == SOURCE
    assert "executed_program" not in adapter.snapshot()["feedback"][0]
    assert adapter.events[-1]["tool_promotion"] is False
    with pytest.raises(RuntimeError, match="no synthesized"):
        adapter.observe_execution({})


def test_snapshot_resume_preserves_history_without_restarting_costs():
    first = cap.Adapter(Recorder())
    first.propose({}, images=["first.png"])
    state = first.snapshot()
    second = cap.Adapter(Recorder(), state=state)
    second.observe_execution({"accepted": True, "physically_dispatched": True})
    second.propose({}, images=["second.png"])
    assert second.snapshot()["round"] == 2
    assert second.snapshot()["call_attempts"] == 22
    assert state["pending"] is not None
    assert state["round"] == 1


def test_library_is_independent_frozen_and_callable():
    library = cap.tool_library()
    assert library["adapted_tool_count"] == 5
    assert library["upstream_tool_count"] == 9
    assert library["test_bank_imported"] is False
    assert library["online_test_skill_promotion"] is False
    assert len(library["sha256"]) == 64
    assert cap.evaluate_literal_tool("normalize_vector", [[3, 0, 4]]) == [0.6, 0., 0.8]
    with pytest.raises(ValueError, match="unknown"):
        cap.evaluate_literal_tool("os.system", [["touch forbidden"]])
    identity = [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]]
    assert cap.evaluate_literal_tool("transform_points", [[[1, 2, 3]], identity]) == [[3., 5., 7.]]
    assert cap.decompose_transform(identity) == ([2., 3., 4.], [1., 0., 0., 0.])
    assert cap.interpolate_segment([0, 0, 0], [.5, 0, 0], .25) == [[0., 0., 0.], [.25, 0., 0.], [.5, 0., 0.]]
    assert cap.normalize_vector([0, 0, 0]) == [0., 0., 0.]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_quaternion_stability_at_half_turn(axis):
    rotation = [[(1. if i == axis else -1.) if i == j else 0. for j in range(3)] for i in range(3)]
    q = cap.rotation_matrix_to_quaternion(rotation)
    assert q[0] == 0 and q[axis + 1] == 1
    assert math.isclose(sum(x * x for x in q), 1)


@pytest.mark.parametrize("step", [0, -1, float("nan"), float("inf")])
def test_tool_allocation_and_invalid_arguments_are_bounded(step):
    with pytest.raises(ValueError):
        cap.interpolate_segment([0, 0, 0], [1, 0, 0], step)


def test_changed_prepared_library_cannot_resume_silently():
    state = cap.Adapter(Recorder()).snapshot()
    state["tool_library_sha256"] = "changed"
    with pytest.raises(ValueError, match="library changed"):
        cap.Adapter(Recorder(), state=state)


def test_finish_is_only_a_proposal_not_task_success():
    recorder = Recorder()
    def invoke(role, payload):
        response = recorder(role, payload)
        return {"content": "FINISH", "usage": {}} if role == "cap_ensemble" else response
    proposal = cap.Adapter(invoke).propose({}, images=["current.png"])
    assert proposal["decision"] == "finish" and proposal["source"] == ""
    assert "success" not in proposal


def test_invalid_synthesis_is_not_replaced_with_a_candidate():
    recorder = Recorder()
    def invoke(role, payload):
        response = recorder(role, payload)
        return {"content": "def broken(:\n", "usage": {}} if role == "cap_ensemble" else response
    with pytest.raises(SyntaxError):
        cap.Adapter(invoke).propose({}, images=["current.png"])
    assert len(recorder.calls) == 11


def test_callback_logging_can_capture_state_without_a_deadlock():
    snapshots = []
    adapter = cap.Adapter(Recorder())
    adapter.emit = lambda event: snapshots.append(adapter.snapshot())
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(adapter.propose, {}, images=["current.png"]).result(timeout=10)
    assert snapshots
