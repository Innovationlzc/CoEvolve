"""Text-first persistence for cooperative specification memory."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from praxis_coop.models import ExecutableSpec, TaskTaxonomy
from praxis_coop.spec_memory import SpecMemory


class TextSpecMemory(SpecMemory):
    """SpecMemory backed by a small JSONL text file.

    This is intentionally simple for phase 1: specs remain human-readable text
    with executable predicates. Vector retrieval can be added later without
    changing the orchestrator contract.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        specs = self._load_specs() if self.path is not None and self.path.exists() else None
        super().__init__(specs=specs or [])
        if not self.specs:
            from praxis_coop.spec_memory import default_specs

            self.specs = default_specs()

    def retrieve(self, taxonomy: TaskTaxonomy, limit: int = 3) -> list[ExecutableSpec]:
        return super().retrieve(taxonomy, limit=limit)

    def save(self) -> None:
        if self.path is None:
            raise ValueError("TextSpecMemory.save requires a path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for spec in self.specs:
                handle.write(json.dumps(asdict(spec), ensure_ascii=False) + "\n")

    def promote(self, spec: ExecutableSpec) -> None:
        super().promote(spec)
        if self.path is not None:
            self.save()

    def _load_specs(self) -> list[ExecutableSpec]:
        assert self.path is not None
        specs: list[ExecutableSpec] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                specs.append(ExecutableSpec(**payload))
        return specs
