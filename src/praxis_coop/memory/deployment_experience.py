"""JSONL memory for verifier-backed deployment experiences."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal


Outcome = Literal["success", "failure", "aborted", "manual_review"]


@dataclass(frozen=True)
class DeploymentExperience:
    """One rollout-level deployment experience.

    This is intentionally separate from executable SpecMemory. Experiences are
    raw, verifier-linked evidence. A spec is promoted only after repeated review
    turns an experience pattern into a reusable rule.
    """

    run_id: str
    skill_name: str
    robot_ids: tuple[str, ...]
    outcome: Outcome
    context: dict[str, Any] = field(default_factory=dict)
    verifier_tags: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_path: str = ""
    notes: str = ""


class DeploymentExperienceMemory:
    """Append-only JSONL store for dynamic experiment evidence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.experiences = self._load() if self.path.exists() else []

    def append(self, experience: DeploymentExperience) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiences.append(experience)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(experience), ensure_ascii=False) + "\n")

    def retrieve(
        self,
        *,
        skill_name: str | None = None,
        outcome: Outcome | None = None,
        limit: int = 5,
    ) -> list[DeploymentExperience]:
        matches = []
        for experience in reversed(self.experiences):
            if skill_name is not None and experience.skill_name != skill_name:
                continue
            if outcome is not None and experience.outcome != outcome:
                continue
            matches.append(experience)
            if len(matches) >= limit:
                break
        return list(reversed(matches))

    def _load(self) -> list[DeploymentExperience]:
        items = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                payload["robot_ids"] = tuple(payload.get("robot_ids", ()))
                payload["verifier_tags"] = tuple(payload.get("verifier_tags", ()))
                items.append(DeploymentExperience(**payload))
        return items
