"""Bounded, non-executing compiler for an external baseline's two robot functions.

The model's Python is parsed, never eval'd/exec'd. Pure expressions are reduced
against the supplied read-only observation and optional explicitly admitted
host tools. The result contains only literal API calls in two native policy
functions. Native API/team/geometry/physical admission is still mandatory.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import math
import operator
from typing import Any, Callable, Mapping


REVISION = "baseline-team-program-literal-compiler-v1-20260914"
MAX_SOURCE_BYTES = 65536
MAX_AST_NODES = 4096
MAX_VALUE_NODES = 20000
MAX_COLLECTION_SIZE = 10001
MAX_STRING_LENGTH = 32768
MAX_DEPTH = 32
MAX_API_CALLS = 16
MAX_TOOL_CALLS = 32
BUILTINS = frozenset({"min", "max", "abs", "round"})
RESERVED = BUILTINS | {"api", "obs", "True", "False", "None"}


class ProgramCompileError(ValueError):
    """An admission failure retaining any pure-tool trace produced so far."""

    def __init__(self, message: str, tool_calls=()):
        super().__init__(message)
        self.tool_calls = deepcopy(list(tool_calls))


def _number(value: Any):
    if type(value) not in (int, float):
        raise ProgramCompileError("arithmetic requires a finite number, not a bool or container")
    if isinstance(value, int) and value.bit_length() > 64:
        raise ProgramCompileError("integer exceeds the 64-bit literal bound")
    if not math.isfinite(value):
        raise ProgramCompileError("non-finite numeric value")
    return value


def _bounded(value: Any, depth=0, budget=None):
    """Reject custom objects, cycles/deep nesting and unbounded data values."""
    if budget is None:
        budget = [MAX_VALUE_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_DEPTH:
        raise ProgramCompileError("data value exceeds the node/depth bound")
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        return _number(value)
    if type(value) is str:
        if len(value) > MAX_STRING_LENGTH:
            raise ProgramCompileError("string exceeds the length bound")
        return value
    if type(value) in (list, tuple, dict):
        if len(value) > MAX_COLLECTION_SIZE:
            raise ProgramCompileError("collection exceeds the item bound")
        if type(value) is dict:
            for key, item in value.items():
                if type(key) not in (str, int, float, bool) and key is not None:
                    raise ProgramCompileError("only literal scalar dictionary keys are permitted")
                _bounded(key, depth + 1, budget)
                _bounded(item, depth + 1, budget)
        else:
            for item in value:
                _bounded(item, depth + 1, budget)
        return value
    raise ProgramCompileError(f"non-literal value type: {type(value).__name__}")


def _tool_entry(name: str, value: Any) -> tuple[Callable, str]:
    if not name.isidentifier() or name.startswith("_") or name in RESERVED:
        raise ProgramCompileError("tool name conflicts with the expression language")
    if isinstance(value, Mapping):
        function = value.get("callable", value.get("function"))
        explicit_sha256 = value.get("sha256")
    else:
        function, explicit_sha256 = value, None
    if not callable(function):
        raise ProgramCompileError("a tool must be an explicitly injected trusted host callable")
    if explicit_sha256 is not None:
        if not isinstance(explicit_sha256, str) or len(explicit_sha256) != 64:
            raise ProgramCompileError("tool provenance must be a full sha256 digest")
        try:
            int(explicit_sha256, 16)
        except ValueError as error:
            raise ProgramCompileError("invalid tool sha256") from error
        digest = explicit_sha256
    else:
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError) as error:
            raise ProgramCompileError("tool requires inspectable source or an explicit admitted hash") from error
        digest = hashlib.sha256(source.encode()).hexdigest()
    return function, digest


class _Compiler:
    def __init__(self, observation: dict, tools: Mapping[str, Any] | None):
        if type(observation) is not dict:
            raise ProgramCompileError("observation must be a plain data dictionary")
        _bounded(observation)
        self.observation = deepcopy(observation)
        self.tools = {name: _tool_entry(name, function) for name, function in (tools or {}).items()}
        self.tool_calls: list[dict[str, Any]] = []
        self.api_calls: list[dict[str, Any]] = []
        self.robot = ""

    def expr(self, node: ast.AST, environment: dict, depth=0):
        if depth > MAX_DEPTH:
            raise ProgramCompileError("expression exceeds the nesting bound")
        child = lambda value: self.expr(value, environment, depth + 1)
        if isinstance(node, ast.Constant):
            return _bounded(node.value)
        if isinstance(node, ast.Name):
            if node.id not in environment:
                raise ProgramCompileError(f"unknown pure expression name: {node.id}")
            return deepcopy(environment[node.id])
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [child(value) for value in node.elts]
            return _bounded(tuple(values) if isinstance(node, ast.Tuple) else values)
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise ProgramCompileError("dictionary expansion is not allowed")
            result = {}
            for key, value in zip(node.keys, node.values):
                literal_key = child(key)
                if type(literal_key) not in (str, int, float, bool) and literal_key is not None:
                    raise ProgramCompileError("dictionary key must be a scalar literal")
                result[literal_key] = child(value)
            return _bounded(result)
        if isinstance(node, ast.Subscript):
            value = child(node.value)
            if type(value) not in (list, tuple, dict, str):
                raise ProgramCompileError("indexing requires a plain literal container")
            if isinstance(node.slice, ast.Slice):
                if type(value) is dict:
                    raise ProgramCompileError("cannot slice a dictionary")
                components = [None if part is None else child(part)
                              for part in (node.slice.lower, node.slice.upper, node.slice.step)]
                if any(item is not None and type(item) is not int for item in components):
                    raise ProgramCompileError("slice bounds must be integers")
                return _bounded(value[slice(*components)])
            index = child(node.slice)
            if type(value) in (list, tuple, str) and type(index) is not int:
                raise ProgramCompileError("sequence index must be an integer")
            if type(index) not in (str, int, float, bool) and index is not None:
                raise ProgramCompileError("index must be a scalar literal")
            return _bounded(deepcopy(value[index]))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _number(child(node.operand))
            return _bounded(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp):
            operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                          ast.Mult: operator.mul, ast.Div: operator.truediv,
                          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
            if type(node.op) not in operations:
                raise ProgramCompileError("only bounded numeric arithmetic is allowed")
            return _bounded(operations[type(node.op)](_number(child(node.left)),
                                                       _number(child(node.right))))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                raise ProgramCompileError("starred call arguments are not allowed")
            if any(item.arg is None for item in node.keywords):
                raise ProgramCompileError("keyword argument expansion is not allowed")
            args = [child(arg) for arg in node.args]
            kwargs = {item.arg: child(item.value) for item in node.keywords}
            if len(kwargs) != len(node.keywords):
                raise ProgramCompileError("duplicate keyword argument")
            if node.func.id in BUILTINS:
                return self.builtin(node.func.id, args, kwargs)
            if node.func.id not in self.tools:
                raise ProgramCompileError(f"tool is not explicitly admitted: {node.func.id}")
            if len(self.tool_calls) >= MAX_TOOL_CALLS:
                raise ProgramCompileError("pure tool call budget exceeded")
            function, digest = self.tools[node.func.id]
            _bounded([args, kwargs])
            record = {"robot": self.robot, "name": node.func.id, "sha256": digest,
                      "args": deepcopy(args), "kwargs": deepcopy(kwargs),
                      "line": getattr(node, "lineno", None)}
            self.tool_calls.append(record)
            try:
                result = _bounded(function(*deepcopy(args), **deepcopy(kwargs)))
            except Exception as error:
                record.update({"status": "failed", "error_type": type(error).__name__,
                               "error": str(error)})
                raise
            record.update({"status": "completed", "result": deepcopy(result)})
            return deepcopy(result)
        raise ProgramCompileError(f"expression is not in the pure grammar: {type(node).__name__}")

    @staticmethod
    def builtin(name: str, args: list, kwargs: dict):
        if kwargs:
            raise ProgramCompileError("pure builtins accept positional arguments only")
        if name in ("min", "max"):
            values = args[0] if len(args) == 1 and type(args[0]) in (list, tuple) else args
            if not values:
                raise ProgramCompileError("min/max require at least one numeric value")
            values = [_number(value) for value in values]
            return _bounded((min if name == "min" else max)(values))
        if name == "abs":
            if len(args) != 1:
                raise ProgramCompileError("abs expects one numeric argument")
            return _bounded(abs(_number(args[0])))
        if name == "round":
            if len(args) not in (1, 2):
                raise ProgramCompileError("round expects a number and optional digits")
            value = _number(args[0])
            if len(args) == 1:
                return _bounded(round(value))
            if type(args[1]) is not int or not -308 <= args[1] <= 308:
                raise ProgramCompileError("round digits must be an integer within [-308,308]")
            return _bounded(round(value, args[1]))
        raise ProgramCompileError("unreachable builtin")

    def policy(self, function: ast.FunctionDef) -> str:
        arguments = function.args
        if (len(arguments.args) != 1 or arguments.args[0].arg != "api"
                or arguments.args[0].annotation is not None or arguments.posonlyargs
                or arguments.kwonlyargs or arguments.vararg or arguments.kwarg
                or arguments.defaults or arguments.kw_defaults or function.decorator_list
                or function.returns is not None or function.type_comment
                or getattr(function, "type_params", [])):
            raise ProgramCompileError("robot function must have exactly one undecorated api argument")
        self.robot = "front" if function.name == "front_policy" else "back"
        environment = {"obs": self.observation}
        statements = []
        for index, statement in enumerate(function.body):
            if (index == 0 and isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)):
                continue
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise ProgramCompileError("assignments may only bind one local name")
                name = statement.targets[0].id
                if name.startswith("_") or name in RESERVED or name in self.tools:
                    raise ProgramCompileError("assignment shadows a reserved name/tool")
                environment[name] = self.expr(statement.value, environment)
                continue
            if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)):
                raise ProgramCompileError("robot body only allows pure assignments and direct API calls")
            call = statement.value
            if not (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "api" and call.func.attr.isidentifier()
                    and not call.func.attr.startswith("_")):
                raise ProgramCompileError("only direct api.<method>(...) statements are allowed")
            if any(isinstance(arg, ast.Starred) for arg in call.args):
                raise ProgramCompileError("API star arguments are not allowed")
            if any(item.arg is None for item in call.keywords):
                raise ProgramCompileError("API keyword expansion is not allowed")
            args = [self.expr(arg, environment) for arg in call.args]
            kwargs = {item.arg: self.expr(item.value, environment) for item in call.keywords}
            if len(kwargs) != len(call.keywords):
                raise ProgramCompileError("duplicate API keyword argument")
            if len(self.api_calls) >= MAX_API_CALLS:
                raise ProgramCompileError("program exceeds the API call emission bound")
            self.api_calls.append({"robot": self.robot, "method": call.func.attr,
                                   "args": deepcopy(args), "kwargs": deepcopy(kwargs)})
            # repr is applied only to bounded plain data; no custom repr hook can
            # execute. parse/unparse preserve exactly the selected API/arguments.
            literal = lambda value: ast.parse(repr(value), mode="eval").body
            statements.append(ast.Expr(value=ast.Call(func=call.func,
                args=[literal(arg) for arg in args],
                keywords=[ast.keyword(arg=key, value=literal(value)) for key, value in kwargs.items()])))
        if not statements:
            raise ProgramCompileError("each robot function must emit at least one API call")
        output = ast.FunctionDef(name="policy", args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg="api")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=statements, decorator_list=[])
        return ast.unparse(ast.fix_missing_locations(output)) + "\n"


def compile_program(source: str, observation: dict, tools: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compile precisely the model's actions, never insert a skill or a stop.

    With tools=None, no CaP library is imported or gifted to another baseline.
    `tools` is an explicit host-authorized name->pure-callable whitelist (or
    descriptors with callable and admitted sha256). Its callables are trusted
    integration code, never model-supplied Python. The caller must validate
    the returned literal native policies before any physical dispatch.
    """
    compiler = None
    try:
        if not isinstance(source, str) or len(source.encode()) > MAX_SOURCE_BYTES:
            raise ProgramCompileError("program source exceeds the byte bound or is not text")
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise ProgramCompileError("program exceeds the AST node bound")
        if (len(tree.body) != 2 or not all(isinstance(node, ast.FunctionDef) for node in tree.body)
                or {node.name for node in tree.body} != {"front_policy", "back_policy"}):
            raise ProgramCompileError("root must contain exactly front_policy(api) and back_policy(api)")
        compiler = _Compiler(observation, tools)
        functions = {node.name: node for node in tree.body}
        front = compiler.policy(functions["front_policy"])
        back = compiler.policy(functions["back_policy"])
        return {"front_code": front, "back_code": back,
                "tool_calls": deepcopy(compiler.tool_calls),
                "api_calls": deepcopy(compiler.api_calls), "revision": REVISION,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "physical_calls_executed": 0, "requires_native_admission": True}
    except Exception as error:
        records = compiler.tool_calls if compiler is not None else []
        if isinstance(error, ProgramCompileError):
            error.tool_calls = deepcopy(records)
            raise
        raise ProgramCompileError(f"{type(error).__name__}: {error}", records) from error
