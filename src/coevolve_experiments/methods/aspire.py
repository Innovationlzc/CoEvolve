"""ASPIRE-Adapted's budgeted online program search, independent of motor code.

This is a clean-room *online adaptation*, not the original LIBERO debug /
held-out evaluation. See docs/BASELINES.md. All model calls
pass through one injected, accounted callback. The Host retains exclusive
ownership of dispatch, rollback, static checks, safety, and terminal decisions.
No method-specific CoEvolve mutation scope or source memory is consumed here.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping


METHOD = "ASPIRE-Adapted"
REVISION = "aspire-online-k8-v1-20260914"
MODEL = "gpt-5.4-mini-2026-03-17"
UPSTREAM_COMMIT = "f4c8939aab0af9b97690c561bd80e282940f7886"
UPSTREAM_URL = f"https://github.com/NVlabs/ASPIRE/tree/{UPSTREAM_COMMIT}"
POPULATION_SIZE = 8
TOP_SURVIVORS = 3
MAX_GENERATIONS = 5
FIX_ATTEMPTS = 3
PHYSICAL_BUDGET = 80

POLICY = {
    "method": METHOD,
    "revision": REVISION,
    "model": MODEL,
    "population_size": POPULATION_SIZE,
    "top_survivors": TOP_SURVIVORS,
    "maximum_generations_per_checkpoint": MAX_GENERATIONS,
    "maximum_fix_candidates_per_checkpoint": FIX_ATTEMPTS,
    "physical_budget": PHYSICAL_BUDGET,
    "evaluation_unit": "one atomic team transaction at a trusted checkpoint",
    "evaluation_protocol": "online, rejection-only common checkpoint recovery",
    "held_out_validation": False,
    "original_protocol_reproduction": False,
    "cross_checkpoint_fitness_comparison": False,
    "unexecuted_candidates_count_as_success": False,
    "skill_validation": "local physical verifier acceptance, not held-out success",
    "source_bank": None,
    "coevolve_mutation_scope": False,
    "original_source_commit": UPSTREAM_COMMIT,
}


class AspireContractError(RuntimeError):
    """An evidence/accounting boundary is missing or inconsistent."""


class AspireSearchExhausted(AspireContractError):
    """The fixed search bound has been reached; the Host decides termination."""


def _clone(value: Any) -> Any:
    """Durable state deliberately admits only JSON-compatible data."""
    return json.loads(json.dumps(value, allow_nan=False))


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _fingerprint(source: str) -> str:
    try:
        value = ast.dump(ast.parse(source), include_attributes=False)
    except (SyntaxError, ValueError):
        value = source
    return hashlib.sha256(value.encode()).hexdigest()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AspireContractError(f"{label} must be a non-empty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise AspireContractError(f"{label} must be a bool")
    return value


class AspireAdapted:
    """A persistent method state machine; this class never executes robot code.

    ``invoke(role, payload) -> dict`` must use the declared mini model and
    durably record its provider/token/cost data (including interrupted calls).
    This class separately journals every attempted model call. ``context``
    carries instruction, observation, images, skill_api, checkpoint_id,
    remaining_physical_budget and optionally physical_consumed.

    A candidate contains ``source`` plus optional ``intent`` / ``strategy``
    fields, unchanged from the coding response. The caller runs its common
    legality/physics checks. Call ``record_dispatch`` before the irreversible
    dispatch seam, then ``record_result`` with the real receipt. Alternatively
    ``record_result`` can import an already charged transaction, provided its
    globally unique transaction_id is preserved.
    """

    def __init__(
        self,
        run_dir: str | Path | None,
        invoke: Callable[[str, dict[str, Any]], Mapping[str, Any]],
        *,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.invoke = invoke
        saved = None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            path = self.run_dir / "aspire_state.json"
            if path.exists():
                saved = json.loads(path.read_text())
        if state is not None and saved is not None:
            raise AspireContractError("choose durable state or explicit state, not both")
        self.state: dict[str, Any] = _clone(state if state is not None else saved) if (
            state is not None or saved is not None
        ) else {
            "policy": _clone(POLICY),
            "checkpoint_id": None,
            "expected_checkpoint_id": None,
            "checkpoint_epoch": 0,
            "fix_candidates": 0,
            "generation": 0,
            "population_build_generation": None,
            "pool": [],
            "outstanding_candidate": None,
            "candidates": {},
            "transactions": {},
            "skills": [],
            "pending_promotions": [],
            "diagnoses": [],
            "events": [],
            "calls": [],
            "scene_analysis": None,
            "stopped": False,
        }
        if self.state.get("policy") != POLICY:
            raise AspireContractError("ASPIRE method configuration changed on resume")
        self._save()

    @property
    def physical_consumed(self) -> int:
        return len(self.state["transactions"])

    @property
    def model_calls(self) -> int:
        return len(self.state["calls"])

    @property
    def skills(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self.state["skills"]))

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def _save(self) -> None:
        if self.run_dir is None:
            return
        path = self.run_dir / "aspire_state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def _event(self, kind: str, **fields: Any) -> None:
        self.state["events"].append(_clone({
            "event_index": len(self.state["events"]), "kind": kind, **fields,
        }))
        self._save()

    def _call(self, role: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        item = {
            "call_index": len(self.state["calls"]), "role": role,
            "model": MODEL, "status": "started", "payload": _clone(payload),
        }
        self.state["calls"].append(item)
        self._save()
        try:
            result = self.invoke(role, _clone(payload))
            if not isinstance(result, Mapping):
                raise AspireContractError(f"{role} did not return a mapping")
            item.update(status="completed", response=_clone(result))
            self._save()
            return _clone(result)
        except BaseException as exc:
            item.update(status="failed_or_interrupted", error_type=type(exc).__name__)
            self._save()
            raise

    def _payload(self, context: Mapping[str, Any]) -> dict[str, Any]:
        # Do not forward a CoEvolve prompt, mutation gate, or memory retrieval
        # object just because it happens to be present in the Host context.
        return _clone({
            key: context[key] for key in (
                "instruction", "observation", "images", "skill_api", "checkpoint_id",
                "remaining_physical_budget", "physical_consumed", "policy_schema",
                "lifecycle", "active_route_guidance", "route_phase_guard",
                "clearance", "route_clearance_conflict",
            ) if key in context
        })

    def _validate_context(self, context: Mapping[str, Any]) -> str:
        checkpoint = _nonempty(context.get("checkpoint_id"), "checkpoint_id")
        remaining = context.get("remaining_physical_budget")
        if isinstance(remaining, bool) or not isinstance(remaining, int) or not 0 <= remaining <= PHYSICAL_BUDGET:
            raise AspireContractError("remaining_physical_budget must be an integer in [0, 80]")
        if remaining == 0 or self.physical_consumed >= PHYSICAL_BUDGET:
            raise AspireSearchExhausted("physical budget exhausted; no extra search/replay")
        if remaining > PHYSICAL_BUDGET - self.physical_consumed:
            raise AspireContractError("remaining budget would erase charged ASPIRE transactions")
        if "physical_consumed" in context and context["physical_consumed"] != self.physical_consumed:
            raise AspireContractError("Host and ASPIRE physical ledgers disagree")
        if not context.get("images"):
            raise AspireContractError("current images required for actual multimodal reasoning")
        expected = self.state["expected_checkpoint_id"]
        if expected is not None and checkpoint != expected:
            raise AspireContractError("checkpoint differs from accepted/restored outcome")
        if self.state["stopped"]:
            raise AspireSearchExhausted("Host terminal or unresolved dispatch is sealed")
        if self.state["outstanding_candidate"] is not None:
            raise AspireContractError("prior candidate needs a recorded result, not a fresh opportunity")
        return checkpoint

    def _start_checkpoint(self, checkpoint: str, context: Mapping[str, Any]) -> None:
        if self.state["checkpoint_id"] == checkpoint and self.state["scene_analysis"] is not None:
            return
        if self.state["checkpoint_id"] != checkpoint:
            if self.state["pool"]:
                self._event("unevaluated_siblings_discarded", candidate_ids=self.state["pool"],
                            reason="accepted transition changed the physical comparison origin")
            self.state.update(
                checkpoint_id=checkpoint, expected_checkpoint_id=checkpoint,
                checkpoint_epoch=self.state["checkpoint_epoch"] + 1,
                fix_candidates=0, generation=0, pool=[], scene_analysis=None,
                population_build_generation=None,
            )
        analysis = self._call("aspire_scene_analysis", {
            **self._payload(context), "verified_skills": self.state["skills"],
            "instruction_for_role": (
                "Inspect the real scene images and allowed state. Analyze object/support, "
                "robot approach, goal geometry, movement/release, and uncertain hypotheses. "
                "Treat visual estimates as hypotheses, not measured guarantees. Return JSON."
            ),
        })
        self.state["scene_analysis"] = analysis
        self._save()

    def _local_candidates(self) -> list[dict[str, Any]]:
        return [item for item in self.state["candidates"].values()
                if item["checkpoint_id"] == self.state["checkpoint_id"]
                and item["checkpoint_epoch"] == self.state["checkpoint_epoch"]]

    @staticmethod
    def _images(*groups: Any) -> list[Any]:
        images = []
        seen = set()
        for group in groups:
            values = list(group.values()) if isinstance(group, Mapping) else group or []
            if isinstance(values, str):
                values = [values]
            for value in values:
                key = _sha(value)
                if key not in seen:
                    images.append(_clone(value))
                    seen.add(key)
        return images

    def _diagnose(self, context: Mapping[str, Any]) -> dict[str, Any]:
        evaluated = [item for item in self._local_candidates() if item.get("feedback") is not None]
        evidence_gaps = [{"candidate_id": item["candidate_id"],
                          "missing_trace": not bool(item["feedback"].get("trace")),
                          "missing_images": not bool(item["feedback"].get("images"))}
                         for item in evaluated]
        result = self._call("aspire_trace_diagnosis", {
            **self._payload(context), "scene_analysis": self.state["scene_analysis"],
            "images": self._images(context.get("images"),
                                   *[item["feedback"].get("images") for item in evaluated]),
            "evidence_gaps": evidence_gaps,
            "evaluated_programs_and_traces": evaluated,
            "prior_diagnoses": [d for d in self.state["diagnoses"]
                                if d["checkpoint_id"] == self.state["checkpoint_id"]],
            "instruction_for_role": (
                "Diagnose actual candidate programs using fine-grained API/physics trace, "
                "stdout/stderr and available before/during/after images. Separate observed "
                "failure from uncertain cause and missing evidence. Update geometry and "
                "eliminated versus blocked-but-untried hypotheses. Revision is unrestricted "
                "across legal actions, task sequences, robot roles, coordination and parameters; "
                "there are no CoEvolve mutation levels. Do not infer held-out success. Return JSON."
            ),
        })
        self.state["diagnoses"].append({
            "checkpoint_id": self.state["checkpoint_id"], "content": result,
            "candidate_ids": [item["candidate_id"] for item in evaluated],
        })
        self._save()
        return result

    def _new_candidate(self, result: Mapping[str, Any], *, kind: str, slot: int,
                       context: Mapping[str, Any]) -> dict[str, Any]:
        source = _nonempty(result.get("source"), "candidate source")
        candidate_id = (
            f"epoch_{self.state['checkpoint_epoch']:03d}_{kind}_"
            f"{self.state['generation']:02d}_{slot:02d}"
        )
        item = _clone(dict(result))
        item.update(
            candidate_id=candidate_id, source=source,
            checkpoint_id=self.state["checkpoint_id"],
            checkpoint_epoch=self.state["checkpoint_epoch"],
            kind=kind, generation=self.state["generation"], slot=slot,
            source_fingerprint=_fingerprint(source),
            generated_from_context_sha256=_sha(self._payload(context)),
            status="unexecuted", feedback=None,
        )
        self.state["candidates"][candidate_id] = item
        if kind == "search":
            # Persist the source and its pool membership together. A restart
            # must not orphan a generated candidate or silently shrink K.
            self.state["pool"].append(candidate_id)
        self._event("candidate_generated", candidate_id=candidate_id,
                    kind_of_candidate=kind, physically_evaluated=False)
        return item

    def _program_payload(self, context: Mapping[str, Any], diagnosis: Any) -> dict[str, Any]:
        return {
            **self._payload(context), "scene_analysis": self.state["scene_analysis"],
            "diagnosis": diagnosis, "verified_skills": self.state["skills"],
            "candidate_schema": {
                "source": "def front_policy(api): ... and def back_policy(api): ...; each side one SDK action plus explicit api.stop_all()",
                "coordination_mode": "sync or async, according to the shared skill API",
                "objective": "free semantic objective for the common team transaction",
                "rationale": "evidence-based reason for the selected program",
                "intent": "optional machine-checkable shared team intent, if required by skill_api",
                "hypothesis": "specific mechanism this candidate tests",
                "expected_failure_if_wrong": "observable trace/image prediction",
                "seeded_from": "prior candidate IDs, or novel",
            },
            "instruction_for_role": (
                "Generate a legal team program from actual observations using the supplied "
                "skill API, not simulator state mutation. Respect the common stop/safety and "
                "atomic transaction contract. You may change action type, order, coordination, "
                "roles or parameters without a CoEvolve mutation-scope restriction. Reuse "
                "verified skills only where current preconditions fit. Return JSON with source "
                "and any required shared intent/strategy. Unexecuted code is not validated."
            ),
        }

    def _survivors(self) -> list[dict[str, Any]]:
        # A diagnostic development ranking, never a pass rate over different
        # checkpoints. All accepted transitions end this checkpoint's search.
        evaluated = [item for item in self._local_candidates()
                     if (item.get("feedback") or {}).get("physically_dispatched") is True]
        def key(item: dict[str, Any]) -> tuple[int, float, int]:
            feedback = item["feedback"]
            score = feedback.get("phase_completion")
            score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else -1.0
            return (int(feedback.get("completed") is True and not feedback.get("code_error")),
                    score, -len(item["source"]))
        return sorted(evaluated, key=key, reverse=True)[:TOP_SURVIVORS]

    def next_candidate(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Return the next program; siblings after acceptance are never replayed."""
        checkpoint = self._validate_context(context)
        self._start_checkpoint(checkpoint, context)
        if self.state["fix_candidates"] < FIX_ATTEMPTS:
            prior = self._local_candidates()
            diagnosis = self._diagnose(context) if prior else None
            self.state["fix_candidates"] += 1
            # Charge a generation slot before calling: interruptions cannot
            # silently provide unlimited retries at the same checkpoint.
            self._save()
            result = self._call("aspire_fix_program", {
                **self._program_payload(context, diagnosis),
                "prior_programs": prior,
                "fix_attempt": self.state["fix_candidates"],
                "maximum_fix_candidates": FIX_ATTEMPTS,
            })
            item = self._new_candidate(result, kind="fix", slot=self.state["fix_candidates"],
                                       context=context)
        else:
            building = self.state["population_build_generation"]
            if not self.state["pool"] or building is not None:
                if building is None and self.state["generation"] >= MAX_GENERATIONS:
                    raise AspireSearchExhausted("five fixed evolutionary generations exhausted")
                if building is None:
                    diagnosis = self._diagnose(context)
                    self.state["generation"] += 1
                    self.state["population_build_generation"] = self.state["generation"]
                    self._save()
                else:
                    diagnosis = self.state["diagnoses"][-1]["content"]
                survivors = self._survivors()
                for slot in range(len(self.state["pool"]), POPULATION_SIZE):
                    # K=8 is not silently reduced with the remaining budget.
                    # All generation calls count; only affordable candidates
                    # can subsequently receive a physical dispatch.
                    result = self._call("aspire_search_candidate", {
                        **self._program_payload(context, diagnosis),
                        "generation": self.state["generation"],
                        "population_size": POPULATION_SIZE, "candidate_slot": slot,
                        "top_same_checkpoint_survivors": survivors,
                        "generated_siblings": [self.state["candidates"][cid]
                                               for cid in self.state["pool"]],
                        "search_instruction": (
                            "Write a distinct hypothesis. Combine or revise useful parts of "
                            "the top-3 actually evaluated survivors; include at least one "
                            "structurally novel candidate, not eight magnitude-only tweaks. "
                            "Do not repeat eliminated code without new evidence. The first "
                            "accepted sibling commits a new state, ending this local search; "
                            "unevaluated siblings will not be labelled failures or successes."
                        ),
                    })
                    self._new_candidate(result, kind="search", slot=slot, context=context)
                self.state["population_build_generation"] = None
                self._event("population_generated", generation=self.state["generation"],
                            candidate_ids=self.state["pool"],
                            unique_programs=len({self.state["candidates"][cid]["source_fingerprint"]
                                                 for cid in self.state["pool"]}))
            item = self.state["candidates"][self.state["pool"].pop(0)]
        self.state["outstanding_candidate"] = item["candidate_id"]
        item["status"] = "proposed"
        self._save()
        return deepcopy(item)

    def record_dispatch(self, candidate_id: str, transaction_id: str,
                        checkpoint_id: str) -> None:
        """Journal a physical commitment before dispatch; never silently uncharge."""
        _nonempty(transaction_id, "transaction_id")
        existing = self.state["transactions"].get(transaction_id)
        if existing is not None:
            if existing["candidate_id"] == candidate_id and existing["checkpoint_id"] == checkpoint_id:
                return
            raise AspireContractError("transaction ID collision")
        if self.state["outstanding_candidate"] != candidate_id:
            raise AspireContractError("dispatch does not belong to the outstanding candidate")
        item = self.state["candidates"][candidate_id]
        if checkpoint_id != item["checkpoint_id"]:
            raise AspireContractError("candidate was generated for a different checkpoint")
        if any(row["candidate_id"] == candidate_id for row in self.state["transactions"].values()):
            raise AspireContractError("one candidate already has a charged physical attempt")
        if self.physical_consumed >= PHYSICAL_BUDGET:
            raise AspireSearchExhausted("all 80 physical transactions are charged")
        self.state["transactions"][transaction_id] = {
            "candidate_id": candidate_id, "checkpoint_id": checkpoint_id,
            "status": "charged_outcome_pending",
        }
        item["status"] = "dispatched_outcome_pending"
        self._event("physical_dispatch_charged", transaction_id=transaction_id,
                    candidate_id=candidate_id, physical_consumed=self.physical_consumed)

    def record_result(self, candidate_id: str, feedback: Mapping[str, Any], *,
                      defer_promotion: bool = False) -> None:
        """Import a Host-owned result, including pre-dispatch rejection or abort.

        Required: physically_dispatched, completed, accepted, checkpoint_before,
        checkpoint_after. A dispatched receipt also needs transaction_id and
        evidence. Rejection needs restore_verified=True before another proposal;
        otherwise it seals this local state. ``terminal=True`` is Host-owned.
        ``trace`` and ``images`` are needed to promote accepted code as a skill.
        """
        if candidate_id not in self.state["candidates"]:
            raise AspireContractError("unknown candidate ID")
        item = self.state["candidates"][candidate_id]
        result = _clone(feedback)
        if item["feedback"] is not None:
            if item["feedback"] == result:
                return
            raise AspireContractError("cannot rewrite an existing physical result")
        if self.state["outstanding_candidate"] != candidate_id:
            raise AspireContractError("result does not belong to the outstanding candidate")
        dispatched = _boolean(result.get("physically_dispatched"), "physically_dispatched")
        completed = _boolean(result.get("completed"), "completed")
        accepted = _boolean(result.get("accepted"), "accepted")
        before = _nonempty(result.get("checkpoint_before"), "checkpoint_before")
        after = _nonempty(result.get("checkpoint_after"), "checkpoint_after")
        if before != item["checkpoint_id"]:
            raise AspireContractError("result origin differs from candidate origin")
        if completed and not dispatched or accepted and not (dispatched and completed):
            raise AspireContractError("unexecuted/incomplete code cannot be physically accepted")
        if accepted:
            if after == before and result.get("accepted_checkpoint_retained") is not True:
                raise AspireContractError("accepted transition must have its new checkpoint identity")
        elif after != before:
            raise AspireContractError("rejected branch cannot become the trusted checkpoint")
        if dispatched:
            _nonempty(result.get("transaction_id"), "transaction_id")
            if not isinstance(result.get("evidence"), dict) or not result["evidence"]:
                raise AspireContractError("physical result requires Host evidence")
            self.record_dispatch(candidate_id, result["transaction_id"], before)
            self.state["transactions"][result["transaction_id"]]["status"] = (
                "accepted" if accepted else "rejected" if completed else "charged_unresolved"
            )
        elif any(row["candidate_id"] == candidate_id for row in self.state["transactions"].values()):
            raise AspireContractError("cannot uncharge an already dispatched candidate")
        item.update(status="accepted" if accepted else "rejected" if completed else
                    "unresolved" if dispatched else "static_rejected", feedback=result)
        self.state["outstanding_candidate"] = None
        self.state["expected_checkpoint_id"] = after
        # Restoration certification is not something the model can waive.
        if result.get("terminal") is True or (dispatched and not completed) or (
            dispatched and not accepted and result.get("restore_verified") is not True
        ):
            self.state["stopped"] = True
        self._event("candidate_result", candidate_id=candidate_id, feedback=result)
        if accepted:
            self.state["pending_promotions"].append(candidate_id)
            if after == before:
                # Host-certified acceptance can retain a checkpoint without
                # claiming progress. Its observation epoch is not the original
                # candidate population's comparison origin. Do not reset the
                # local search budgets, and never rank across those epochs.
                self._event("accepted_checkpoint_retained", candidate_id=candidate_id,
                            discarded_unexecuted_siblings=self.state["pool"],
                            local_fix_and_generation_budgets_reset=False)
                self.state["pool"] = []
                self.state["checkpoint_epoch"] += 1
                self.state["scene_analysis"] = None
                self.state["population_build_generation"] = None
            self._save()
            if not defer_promotion:
                self.drain_pending_promotions()

    def drain_pending_promotions(self) -> None:
        """Only call once the Host controller has committed physical accounting.

        A model failure here cannot erase the already persisted physical
        receipt. Production bridges defer this work out of the raw-record
        callback, and leave terminal-run pending findings unpromoted if there
        is no later proposal. That is not hidden successful knowledge.
        """
        while self.state["pending_promotions"]:
            candidate_id = self.state["pending_promotions"][0]
            item = self.state["candidates"][candidate_id]
            self._promote(item, item["feedback"])
            self.state["pending_promotions"].pop(0)
            self._save()

    def _promote(self, item: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        if not result.get("trace") or not result.get("images"):
            self._event("skill_promotion_withheld", candidate_id=item["candidate_id"],
                        reason="accepted outcome lacks trace or visual evidence")
            return
        finding = self._call("aspire_skill_generalization", {
            "executed_program": item, "physical_verifier_result": result,
            "images": self._images(result.get("images")),
            "existing_verified_skills": self.state["skills"],
            "instruction_for_role": (
                "Extract a reusable support/coordination/programming finding from this "
                "actually accepted transaction and its images/trace. Return reusable:bool, "
                "trigger:str and summary:str. If nothing generalizes, reusable:false. "
                "New code or parameter generalization has not been physically validated; "
                "do not label new code successful. This is local verification only, not "
                "held-out success or proof of task completion."
            ),
        })
        if finding.get("reusable") is not True:
            self._event("skill_promotion_withheld", candidate_id=item["candidate_id"],
                        reason="no reusable finding asserted")
            return
        trigger = _nonempty(finding.get("trigger"), "skill trigger")
        summary = _nonempty(finding.get("summary"), "skill summary")
        skill = {
            "skill_id": f"skill_{len(self.state['skills']) + 1:03d}",
            "trigger": trigger, "summary": summary,
            # Retain only the exact executed program, never a generated rewrite.
            "source": item["source"], "source_fingerprint": item["source_fingerprint"],
            "candidate_id": item["candidate_id"], "transaction_id": result["transaction_id"],
            "checkpoint_before": result["checkpoint_before"],
            "checkpoint_after": result["checkpoint_after"],
            "validation": "locally verifier-accepted executed code",
            "held_out_validated": False,
            "applicability_summary_is_hypothesis": True,
        }
        self.state["skills"].append(skill)
        self._event("verified_local_skill_promoted", **skill)


# Explicit exported alias for simple runtime bridges.
ASPIREAdapter = AspireAdapted
