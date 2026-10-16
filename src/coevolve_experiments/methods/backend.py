"""Deterministic RoleBackend used by adapter and engine smoke tests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .base import RoleResponse, RoleUsage


@dataclass(frozen=True, slots=True)
class RoleCall:
    index: int
    role: str
    payload: Mapping[str, Any]


class ScriptedRoleBackend:
    """Strict FIFO fake: an unexpected role fails immediately and readably."""

    def __init__(
        self,
        script: Iterable[
            tuple[str, RoleResponse | Any]
        ],
        *,
        default_usage: RoleUsage = RoleUsage(input_tokens=1, output_tokens=1),
    ) -> None:
        self._script = deque(script)
        self._default_usage = default_usage
        self.calls: list[RoleCall] = []

    def invoke(self, role: str, payload: Mapping[str, Any]) -> RoleResponse:
        if not self._script:
            raise AssertionError(f"unexpected role call after script end: {role}")
        expected_role, scripted = self._script.popleft()
        if expected_role != role:
            raise AssertionError(
                f"expected role {expected_role!r}, received {role!r}"
            )
        self.calls.append(RoleCall(len(self.calls), role, dict(payload)))
        if isinstance(scripted, RoleResponse):
            return scripted
        return RoleResponse(content=scripted, usage=self._default_usage)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(call.role for call in self.calls)

    @property
    def remaining(self) -> int:
        return len(self._script)

    def assert_exhausted(self) -> None:
        if self._script:
            pending = [role for role, _ in self._script]
            raise AssertionError(f"unused scripted role calls: {pending}")

