"""Adversarial pure-data compiler tests; no provider or physical tool execution."""
import ast

import pytest


from coevolve_experiments.methods import cap_agent0 as cap
from coevolve_experiments.methods import team_program as compiler
from praxis_coop.cap_policy.policy_executor import validate_step_policy_code

def pair(body="api.move_forward(distance_m=0.25)\napi.stop_all()", back=None):
    def function(name, value):
        return f"def {name}_policy(api):\n" + "\n".join("    " + line for line in value.splitlines()) + "\n"
    return function("front", body) + function("back", body if back is None else back)


def test_literal_atomic_actions_preserved_and_never_executed():
    result = compiler.compile_program(pair(), {})
    assert result["front_code"] == "def policy(api):\n    api.move_forward(distance_m=0.25)\n    api.stop_all()\n"
    assert result["back_code"] == result["front_code"]
    assert result["physical_calls_executed"] == 0
    assert result["requires_native_admission"] is True
    assert result["tool_calls"] == []
    assert len(result["api_calls"]) == 4


def test_assignment_observation_and_bounded_builtins_reduce_to_literals():
    code = pair('d = min(0.5, max(0.05, abs(obs["goal"][0]) / 2))\nd = round(d, 2)\napi.move_forward(distance_m=d, speed_mps=0.5)\napi.stop_all()')
    obs = {"goal": [-.62, 2]}
    result = compiler.compile_program(code, obs)
    assert "distance_m=0.31" in result["front_code"]
    assert "obs" not in result["front_code"]
    assert "d =" not in result["front_code"]
    assert obs == {"goal": [-.62, 2]}


def test_optional_cap_library_is_really_called_and_fully_logged():
    code = pair('direction = normalize_vector([3, 0, 4])\napi.move_forward(distance_m=direction[0] * 0.5)\napi.stop_all()')
    result = compiler.compile_program(code, {}, cap.TOOL_FUNCTIONS)
    assert "distance_m=0.3" in result["front_code"]
    assert len(result["tool_calls"]) == 2
    record = result["tool_calls"][0]
    assert record["name"] == "normalize_vector"
    assert len(record["sha256"]) == 64
    assert record["args"] == [[3, 0, 4]]
    assert record["result"] == [0.6, 0., 0.8]
    assert record["status"] == "completed"
    assert "normalize_vector" not in result["front_code"]


def test_no_library_is_silently_given_to_another_baseline():
    with pytest.raises(compiler.ProgramCompileError, match="not explicitly admitted"):
        compiler.compile_program(pair("v = normalize_vector([1, 0, 0])\napi.stop_all()"), {})


@pytest.mark.parametrize("body", [
    "import os\napi.stop_all()",
    "from pathlib import Path\napi.stop_all()",
    "__import__('os').system('touch /tmp/should_not_exist')",
    "api.__getattribute__('stop_all')()",
    "getattr(api, 'stop_all')()",
    "fn = api.stop_all\nfn()",
    "api.transport.move()",
    "api = obs\napi.stop_all()",
    "obs = {}\napi.stop_all()",
    "obs['x'] = 1\napi.stop_all()",
    "api.stop_all = 2\napi.stop_all()",
    "global x\napi.stop_all()",
    "nonlocal x\napi.stop_all()",
    "for x in [1, 2]:\n    api.stop_all()",
    "while True:\n    api.stop_all()",
    "if True:\n    api.stop_all()",
    "try:\n    api.stop_all()\nexcept:\n    pass",
    "with open('x') as f:\n    api.stop_all()",
    "return api.stop_all()",
    "yield api.stop_all()",
    "values = [x for x in [1, 2]]\napi.stop_all()",
    "f = lambda: 1\napi.stop_all()",
    "v = (1).__class__\napi.stop_all()",
    "v = obs.get('x', 1)\napi.stop_all()",
    "v = open('/etc/passwd').read()\napi.stop_all()",
    "v = eval('1')\napi.stop_all()",
    "exec('x=1')\napi.stop_all()",
    "api.move_forward(**obs)\napi.stop_all()",
    "api.move_forward(*[1,2])\napi.stop_all()",
    "v = max(*[1,2])\napi.stop_all()",
    "v = {**obs}\napi.stop_all()",
    "x,y = [1,2]\napi.stop_all()",
    "a=b=1\napi.stop_all()",
    "v: int = 1\napi.stop_all()",
    "v = 1\nv += 2\napi.stop_all()",
    "assert True\napi.stop_all()",
    "del obs\napi.stop_all()",
])
def test_forbidden_runtime_python_never_executes(body):
    with pytest.raises(compiler.ProgramCompileError):
        compiler.compile_program(pair(body), {"x": 1})


@pytest.mark.parametrize("source", [
    "import os\n" + pair(),
    pair() + "api.stop_all()\n",
    pair().replace("def front_policy(api):", "@arbitrary\ndef front_policy(api):"),
    pair().replace("def front_policy(api):", "def front_policy(api=fn()):"),
    pair().replace("def front_policy(api):", "def front_policy(api: fn()):"),
    pair().replace("def front_policy(api):", "def front_policy(api) -> fn():"),
    pair().replace("def front_policy(api):", "def front_policy(api, **kwargs):"),
    pair().replace("def front_policy(api):", "def front_policy(api, *args):"),
    pair().replace("def front_policy(api):", "async def front_policy(api):"),
    pair().replace("back_policy", "front_policy"),
    pair() + "def helper(api):\n    api.stop_all()\n",
])
def test_exact_pair_and_signatures(source):
    with pytest.raises(compiler.ProgramCompileError):
        compiler.compile_program(source, {})


@pytest.mark.parametrize("expression", [
    "1e309", "1e300 * 1e300", "1 / 0", "2 ** 1000000000",
    "[0] * 1000000000", "'x' * 1000000000", "1 << 1000000000",
    "round(1.2, 100000000)", "min([])", "abs(True)", "max([1, {}])",
    "999999999999999999999999999999999999999999999999999999",
])
def test_nonfinite_or_unbounded_expressions(expression):
    with pytest.raises(compiler.ProgramCompileError):
        compiler.compile_program(pair(f"v = {expression}\napi.stop_all()"), {})


def test_no_stop_or_action_is_inserted_or_repaired_by_compiler():
    result = compiler.compile_program(pair("api.move_backward(distance_m=0.2)", "api.stop_all()"), {})
    assert "stop_all" not in result["front_code"]
    assert "move_" not in result["back_code"]
    assert result["requires_native_admission"] is True


def test_native_skill_names_not_substituted_by_the_compiler():
    # Unknown but syntactically direct API names remain subject to the *native*
    # allowlist. The compiler never executes an object with this attribute.
    result = compiler.compile_program(pair("api.not_a_real_skill(x=1)\napi.stop_all()"), {})
    assert "api.not_a_real_skill(x=1)" in result["front_code"]


def test_source_ast_depth_and_api_emission_are_bounded(monkeypatch):
    with pytest.raises(compiler.ProgramCompileError, match="source exceeds"):
        compiler.compile_program("x" * (compiler.MAX_SOURCE_BYTES + 1), {})
    monkeypatch.setattr(compiler, "MAX_AST_NODES", 5)
    with pytest.raises(compiler.ProgramCompileError, match="AST node"):
        compiler.compile_program(pair(), {})
    monkeypatch.setattr(compiler, "MAX_AST_NODES", 4096)
    with pytest.raises(compiler.ProgramCompileError, match="API call"):
        compiler.compile_program(pair("\n".join("api.stop_all()" for _ in range(20))), {})


def test_plain_finite_bounded_observation_required():
    with pytest.raises(compiler.ProgramCompileError, match="non-finite"):
        compiler.compile_program(pair(), {"x": float("nan")})
    class Dangerous:
        def __repr__(self):
            raise RuntimeError("must never be reached")
    with pytest.raises(compiler.ProgramCompileError, match="non-literal"):
        compiler.compile_program(pair(), {"x": Dangerous()})
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(compiler.ProgramCompileError, match="depth bound"):
        compiler.compile_program(pair(), cyclic)


def test_local_assignments_are_separate_per_robot():
    with pytest.raises(compiler.ProgramCompileError, match="unknown"):
        compiler.compile_program(pair("d = 0.25\napi.move_forward(distance_m=d)\napi.stop_all()",
                                      "api.move_forward(distance_m=d)\napi.stop_all()"), {})


def test_docstring_and_container_indexing_are_plain_data():
    code = pair('"""This is documentation only."""\nvalues = {"a": (0.25, 0.5)}\nchoice = values["a"][0:1][0]\napi.move_forward(distance_m=choice)\napi.stop_all()')
    result = compiler.compile_program(code, {})
    assert "distance_m=0.25" in result["front_code"]
    assert "documentation" not in result["front_code"]


def test_tool_trace_survives_later_admission_failure():
    code = pair("v = normalize_vector([1,0,0])\napi.move_forward(distance_m=unknown)\napi.stop_all()")
    with pytest.raises(compiler.ProgramCompileError) as error:
        compiler.compile_program(code, {}, cap.TOOL_FUNCTIONS)
    assert len(error.value.tool_calls) == 1
    assert error.value.tool_calls[0]["status"] == "completed"


def test_failed_tool_call_is_logged_not_fabricated_as_success():
    code = pair("v = interpolate_segment([0,0,0], [1,0,0], step=0)\napi.stop_all()")
    with pytest.raises(compiler.ProgramCompileError) as error:
        compiler.compile_program(code, {}, cap.TOOL_FUNCTIONS)
    assert len(error.value.tool_calls) == 1
    assert error.value.tool_calls[0]["status"] == "failed"
    assert "result" not in error.value.tool_calls[0]


def test_tool_call_bound_and_explicit_hash():
    descriptor = {"normalize_vector": {"callable": cap.normalize_vector, "sha256": "a" * 64}}
    code = pair("v = normalize_vector([1,0,0])\napi.stop_all()")
    result = compiler.compile_program(code, {}, descriptor)
    assert result["tool_calls"][0]["sha256"] == "a" * 64
    bad = {"normalize_vector": {"callable": cap.normalize_vector, "sha256": "z" * 64}}
    with pytest.raises(compiler.ProgramCompileError, match="sha256"):
        compiler.compile_program(code, {}, bad)


def test_native_grammar_accepts_actual_compiled_literal_policies():
    # Check the shipped validator with a recording API, never a physical backend.
    result = compiler.compile_program(pair("api.carry_forward(distance_m=0.25, speed_mps=0.5, ramp_s=1.5)\napi.stop_all()"), {})
    for key in ("front_code", "back_code"):
        parsed = ast.parse(result[key])
        assert len(parsed.body) == 1 and parsed.body[0].name == "policy"
        assert all(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   for node in parsed.body[0].body)
        for statement in parsed.body[0].body:
            for argument in statement.value.keywords:
                ast.literal_eval(argument.value)
    for key in ("front_code", "back_code"):
        assert validate_step_policy_code(result[key]) == ("carry_forward", "stop_all")
