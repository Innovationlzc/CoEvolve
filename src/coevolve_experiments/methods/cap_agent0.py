"""CaP-Agent0's VDM -> nine candidates -> synthesis, without physical dispatch.

This independent, callback-driven adapter is not a rename of a CoEvolve agent.
The bridge owns model transport/accounting, admissible code, physical safety,
checkpoint restoration and task termination. See the adaptation note for the
upstream revision and the explicit differences from the original evaluation.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import inspect
import json
import math
import re
import threading
from typing import Any, Callable, Mapping, Sequence


REVISION = "cap-agent0-adapted-v1-20260914"
MODEL = "gpt-5.4-mini-2026-03-17"
UPSTREAM_COMMIT = "53e9966d7a8e2fa7494676772bccc35280f5c0ed"
UPSTREAM_URL = "https://github.com/capgym/cap-x"
TEMPERATURES = tuple(i / 10 for i in range(1, 10))
SYNTHESIS_TEMPERATURE = 0.2


def normalize_vector(v):
    """Return a unit vector; preserve a near-zero vector, as upstream does."""
    values = [float(x) for x in v]
    length = math.sqrt(sum(x * x for x in values))
    return values if length < 1e-6 else [x / length for x in values]


def interpolate_segment(p1, p2, step=0.03):
    """Return points on a segment, including both endpoints; no motion occurs."""
    if len(p1) != len(p2) or not p1 or not math.isfinite(step) or step <= 0:
        raise ValueError("matching nonempty points and a positive finite step required")
    start, stop = [float(x) for x in p1], [float(x) for x in p2]
    distance = math.sqrt(sum((b - a) ** 2 for a, b in zip(start, stop)))
    if not math.isfinite(distance):
        raise ValueError("finite points required")
    if distance < 1e-6:
        return [start]
    count = math.ceil(distance / step)
    if count > 10000:
        raise ValueError("tool result exceeds the 10001-point data allocation bound")
    return [[a + (b - a) * i / count for a, b in zip(start, stop)]
            for i in range(count + 1)]


def transform_points(points, transform_matrix):
    """Apply a homogeneous 4x4 transform to a list of 3D points (no actuation)."""
    if len(transform_matrix) != 4 or any(len(row) != 4 for row in transform_matrix):
        raise ValueError("4x4 transform required")
    if any(len(point) != 3 for point in points):
        raise ValueError("3D points required")
    return [[sum(float(transform_matrix[i][j]) * float(point[j]) for j in range(3))
             + float(transform_matrix[i][3]) for i in range(3)] for point in points]


def rotation_matrix_to_quaternion(rotation):
    """Convert a 3x3 rotation matrix to a quaternion in [w, x, y, z] order."""
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("3x3 rotation required")
    r = [[float(x) for x in row] for row in rotation]
    trace = sum(r[i][i] for i in range(3))
    if trace > 0:
        scale = 2 * math.sqrt(trace + 1)
        q = [scale / 4, (r[2][1] - r[1][2]) / scale,
             (r[0][2] - r[2][0]) / scale, (r[1][0] - r[0][1]) / scale]
    else:
        i = max(range(3), key=lambda j: r[j][j])
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = 2 * math.sqrt(max(0., 1 + r[i][i] - r[j][j] - r[k][k]))
        if scale <= 1e-12:
            raise ValueError("degenerate rotation matrix")
        q = [(r[k][j] - r[j][k]) / scale, 0., 0., 0.]
        q[i + 1] = scale / 4
        q[j + 1] = (r[j][i] + r[i][j]) / scale
        q[k + 1] = (r[k][i] + r[i][k]) / scale
    return normalize_vector(q)


def decompose_transform(transform):
    """Return translation and [w,x,y,z] quaternion from a 4x4 transform."""
    if len(transform) != 4 or any(len(row) != 4 for row in transform):
        raise ValueError("4x4 transform required")
    return ([float(transform[i][3]) for i in range(3)],
            rotation_matrix_to_quaternion([row[:3] for row in transform[:3]]))


TOOL_FUNCTIONS = {
    fn.__name__: fn for fn in (normalize_vector, interpolate_segment,
                             transform_points, rotation_matrix_to_quaternion,
                             decompose_transform)
}


def tool_library() -> dict[str, Any]:
    """A frozen, source-declared library; never reads a benchmark output/bank."""
    source = "import math\n\n" + "\n\n".join(inspect.getsource(fn)
                                                  for fn in TOOL_FUNCTIONS.values())
    return {
        "provenance": "interface-compatible pure-math ports of upstream synthesized tools",
        "upstream_repository": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": "capx/integrations/franka/control_reduced_skill_library.py",
        "upstream_tool_count": 9,
        "adapted_tool_count": len(TOOL_FUNCTIONS),
        "omitted_incompatible_tools": ["depth_to_point_cloud", "mask_to_world_points",
                                      "pixel_to_world_point", "select_top_down_grasp"],
        "functions": list(TOOL_FUNCTIONS),
        "source": source,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "test_bank_imported": False,
        "online_test_skill_promotion": False,
    }


def evaluate_literal_tool(name: str, args: Sequence[Any], kwargs: Mapping[str, Any] | None = None):
    """Optional bridge for a literal-only compiler; no eval, globals or robots.

    Availability is not evidence of use. The bridge must log every actual
    expansion and still run its ordinary final-code admission checks.
    """
    if name not in TOOL_FUNCTIONS:
        raise ValueError("unknown prepared tool")
    arguments = json.loads(json.dumps([list(args), dict(kwargs or {})], allow_nan=False))
    result = TOOL_FUNCTIONS[name](*arguments[0], **arguments[1])
    json.dumps(result, allow_nan=False)
    return result


# Reject structured method-specific requests instead of handing a CoEvolve
# strategy/coordination/parameter revision contract to an external baseline.
PRIVATE_SCOPE_KEYS = frozenset({
    "mutation_scope", "mutation_level", "selected_mutation_scope",
    "required_mutation_scope", "minimum_mutation_scope", "revision_scope",
    "evolution_controller_decision", "scope_admission", "failed_parent_scope",
})


def public_context(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): public_context(item) for key, item in value.items()
                if str(key) not in PRIVATE_SCOPE_KEYS}
    if isinstance(value, (tuple, list)):
        return [public_context(item) for item in value]
    return value


def _content(response: Mapping[str, Any]) -> str:
    # Model transports may hand back an already parsed code envelope. Preserve
    # that exact response instead of making another uncounted parsing call.
    if isinstance(response.get("source"), str):
        return str(response["source"])
    for key in ("content", "text", "output_text"):
        if isinstance(response.get(key), str):
            return str(response[key])
    raise ValueError("callback result has no source/content/text/output_text")


def _program(response: Mapping[str, Any]) -> dict[str, Any]:
    raw = _content(response).strip()
    envelope = dict(response) if "source" in response else {}
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            envelope = parsed
            raw = str(parsed.get("source", parsed.get("code", ""))).strip()
    if raw == "FINISH" or envelope.get("decision") == "finish":
        return {"source": "", "decision": "finish", "metadata": envelope}
    if raw.startswith("REGENERATE"):
        raw = raw[len("REGENERATE"):].strip()
    fence = re.fullmatch(r"```(?:python)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    if not raw:
        raise ValueError("empty synthesized program")
    ast.parse(raw)  # Grammar/safety and action admission belong to the bridge.
    return {"source": raw + "\n", "decision": "execute", "metadata": envelope}


class CaPAgent0Adapter:
    """Stateful CaP-Agent0 adaptation; invoke(role, payload) must be thread-safe.

    Every payload has model, temperature, images (paths), system/user text,
    request_id and round. The bridge sends images as real multimodal content,
    not just file names. All nine candidates are attempted exactly once; only
    the final synthesized program is returned for physical admission.

    ``candidate_workers`` limits concurrent provider requests, not candidate
    count. Set it to 1 for serial debugging with a non-thread-safe client.
    The historical setting is 9; changing it is a new runtime configuration
    and must not be presented as a replay with identical timing. Python cannot
    forcibly cancel a blocked request thread: the injected transport must own
    bounded network timeouts. Always join the batch before synthesis/raising.
    """

    def __init__(self, invoke: Callable[[str, dict[str, Any]], Mapping[str, Any]], *,
                 emit: Callable[[dict[str, Any]], None] | None = None,
                 state: Mapping[str, Any] | None = None,
                 candidate_workers: int = 9):
        if type(candidate_workers) is not int or not 1 <= candidate_workers <= len(TEMPERATURES):
            raise ValueError("candidate_workers must be an integer between 1 and 9")
        self.candidate_workers = candidate_workers
        self.invoke = invoke
        self.emit = emit or (lambda event: None)
        self._lock = threading.RLock()
        self.events: list[dict[str, Any]] = []
        self._state = deepcopy(dict(state or {
            "revision": REVISION, "round": 0, "previous_images": [],
            "feedback": [], "pending": None, "call_attempts": 0,
            "tool_library_sha256": tool_library()["sha256"],
        }))
        if self._state.get("revision") != REVISION:
            raise ValueError("incompatible CaP state revision")
        if self._state.get("tool_library_sha256") != tool_library()["sha256"]:
            raise ValueError("prepared tool library changed during run")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def _event(self, event: dict[str, Any]):
        with self._lock:
            self.events.append(deepcopy(event))
            self.emit(deepcopy(event))

    def _call(self, role: str, *, round_id: int, suffix: str, temperature: float,
              system: str, user: str, images: Sequence[str] = ()) -> dict[str, Any]:
        payload = {"model": MODEL, "temperature": temperature,
                   "images": list(images), "system": system, "user": user,
                   "request_id": f"cap_round_{round_id:04d}_{suffix}", "round": round_id}
        with self._lock:
            self._state["call_attempts"] += 1
        self._event({"event": "call_started", "role": role, "payload": payload})
        try:
            response = self.invoke(role, deepcopy(payload))
            if not isinstance(response, Mapping):
                raise ValueError("invoke must return a mapping with full usage metadata")
            result = deepcopy(dict(response))
            _content(result)
        except Exception as error:
            self._event({"event": "call_failed", "role": role,
                         "request_id": payload["request_id"],
                         "error_type": type(error).__name__, "error": str(error)})
            raise
        self._event({"event": "call_completed", "role": role,
                     "request_id": payload["request_id"], "response": result})
        return result

    def propose(self, observation: Mapping[str, Any], *, images: Sequence[str] | None = None,
                api_contract: str = "", output_contract: str = "") -> dict[str, Any]:
        if self._state.get("pending") is not None:
            raise RuntimeError("record the admitted/rejected execution before another proposal")
        current_images = list(images if images is not None else observation.get("images", []))
        if not current_images or any(not isinstance(path, str) or not path for path in current_images):
            raise ValueError("CaP visual differencing requires certified current image paths")
        clean = public_context(observation)
        clean.pop("images", None)
        round_id = int(self._state["round"]) + 1
        self._state["round"] = round_id
        previous = self._state["previous_images"]
        mode = "difference" if previous else "initial_description"
        vdm = self._call(
            "cap_visual_difference", round_id=round_id, suffix="visual", temperature=0.2,
            system=("Describe task-relevant visual evidence objectively. Do not write code. "
                    "Do not infer contact or task success from a picture when it is unobservable. "
                    "World-state facts and the physical verifier remain authoritative."),
            user=json.dumps({
                "mode": mode,
                "instruction": ("Compare previous and current images; identify changes, "
                                "remaining task and observable failure evidence. A disclosed "
                                "checkpoint restore is not a physically executed reverse motion."
                                if previous else "Describe the initial scene in relation to the task."),
                "image_order": (["previous"] * len(previous) + ["current"] * len(current_images)),
                "current_context": clean,
                "last_execution": self._state["feedback"][-1:]}, ensure_ascii=False, default=str),
            images=[*previous, *current_images],
        )
        library = tool_library()
        context = {"observation": clean, "visual_differencing": _content(vdm),
                   "prior_execution_feedback": self._state["feedback"],
                   "shared_robot_api": api_contract,
                   "response_contract": output_contract,
                   "prepared_tool_library": library}
        user = json.dumps(context, ensure_ascii=False, default=str)
        candidate_system = (
            "Generate a Python team program for the current robot task. Use the shared API "
            "and output contract. You may revise action selection, coordination, parameters "
            "and program structure freely within the shared physical and syntax constraints. "
            "Only your eventual synthesized program can be physically executed; you cannot "
            "test a candidate or reset the simulator. Use previous code/errors, current state "
            "and visual differences to repair or continue. The prepared pure-math tools are "
            "available for literal compile-time calls through the bridge; do not claim to have "
            "executed a tool or robot. FINISH is a proposal, never proof of verifier success."
        )

        provider_fatal_errors = []

        def candidate(index: int, temperature: float) -> dict[str, Any]:
            try:
                result = self._call(
                    "cap_candidate", round_id=round_id, suffix=f"candidate_{index:02d}",
                    temperature=temperature, system=candidate_system, user=user)
                return {"index": index, "temperature": temperature, "ok": True,
                        "response": result, "physically_executed": False}
            except Exception as error:
                if any(cls.__name__ == 'ProviderFatalError' for cls in type(error).__mro__):
                    with self._lock:
                        provider_fatal_errors.append(error)
                return {"index": index, "temperature": temperature, "ok": False,
                        "error_type": type(error).__name__, "error": str(error),
                        "physically_executed": False}

        # Keep a stable index in evidence, irrespective of provider completion order.
        with ThreadPoolExecutor(max_workers=self.candidate_workers, thread_name_prefix="cap-candidate") as executor:
            futures = [executor.submit(candidate, i, temperature)
                       for i, temperature in enumerate(TEMPERATURES, start=1)]
            candidates = [future.result() for future in futures]
        self._event({"event": "candidates_complete", "round": round_id,
                     "candidates": candidates, "requested_k": 9})
        if provider_fatal_errors:
            # Join and account the already started K=9 batch, but do not turn
            # a provider terminal into a grammar retry of another whole batch.
            raise provider_fatal_errors[0]
        if not any(row["ok"] for row in candidates):
            raise RuntimeError("all nine CaP candidates failed; no hidden retry or physical fallback")
        synthesis = self._call(
            "cap_ensemble", round_id=round_id, suffix="synthesis", temperature=0.2,
            system=("Synthesize the best final team program from the independent candidates. "
                    "Compare assumptions, error handling, geometry and execution feedback; "
                    "combine or rewrite useful approaches rather than choosing by unsupported "
                    "success claims. Candidate text is unexecuted advice, not observations. "
                    "Return only the response format in the shared output contract. "
                    "Respect the same shared API and safety constraints. You may propose "
                    "FINISH, but only the physical verifier establishes success."),
            user=json.dumps({"context": context, "candidate_solutions": candidates},
                            ensure_ascii=False, default=str),
        )
        proposal = _program(synthesis)
        for key in ("coordination_mode", "objective", "rationale"):
            if key in proposal["metadata"]:
                proposal[key] = deepcopy(proposal["metadata"][key])
        proposal.update({"method": "cap_agent0_adapted", "revision": REVISION,
                         "round": round_id, "candidate_count": 9,
                         "candidate_success_count": sum(row["ok"] for row in candidates),
                         "visual_mode": mode, "visual_feedback": _content(vdm),
                         "candidates": candidates, "synthesis": synthesis,
                         "tool_library_sha256": library["sha256"],
                         "tools_available": library["functions"],
                         "tool_use_claimed": False,
                         "candidate_physical_dispatches": 0})
        self._state["previous_images"] = current_images
        self._state["pending"] = {"round": round_id, "source": proposal["source"],
                                   "decision": proposal["decision"]}
        self._event({"event": "proposal", "proposal": proposal})
        return deepcopy(proposal)

    def observe_execution(self, feedback: Mapping[str, Any], **extra) -> dict[str, Any]:
        """Record the *final* candidate's actual admission/dispatch/verifier report.

        Rejections and restore provenance remain visible in the next turn.
        This method never promotes a rejected or unexecuted program into tools.
        """
        pending = self._state.get("pending")
        if pending is None:
            raise RuntimeError("no synthesized proposal awaiting execution feedback")
        record = public_context({**dict(feedback), **extra})
        record.update({"round": pending["round"], "proposed_program": pending["source"],
                       "proposed_decision": pending["decision"]})
        self._state["feedback"].append(record)
        self._state["pending"] = None
        self._event({"event": "execution_feedback", "record": record,
                     "tool_promotion": False})
        return deepcopy(record)


Adapter = CaPAgent0Adapter
