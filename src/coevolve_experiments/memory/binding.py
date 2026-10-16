"""Plan-pinned Full-adapter binding; no physics runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from coevolve_experiments.contracts import RunSpec
from coevolve_experiments.methods.base import MethodAdapter, RoleBackend
from coevolve_experiments.methods.factory import create_method_adapter

from .store import FileBackedAdapterMemory


@dataclass(frozen=True)
class Table4AdapterBinding:
    run_spec: RunSpec
    adapter: MethodAdapter
    memory: FileBackedAdapterMemory

    def finalize(self) -> Mapping[str, Any]:
        return self.memory.finalize()

    def __enter__(self) -> "Table4AdapterBinding":
        self.memory.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return self.memory.__exit__(exc_type, exc, traceback)


def bind_table4_adapter(
    run_spec: RunSpec,
    backend: RoleBackend,
    memory_dir: str | Path,
) -> Table4AdapterBinding:
    """Prevent a formal transfer run from silently using NullAdapterMemory."""

    memory = FileBackedAdapterMemory.open_for_run(run_spec, memory_dir)
    adapter = create_method_adapter(run_spec.method_variant, backend, memory=memory)
    return Table4AdapterBinding(run_spec, adapter, memory)


__all__ = ["Table4AdapterBinding", "bind_table4_adapter"]
