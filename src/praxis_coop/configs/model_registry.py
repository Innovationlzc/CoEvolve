"""Phase-1 model choices and download targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ROOT = Path("/home/admin1/Project/CoEvolve/models")


@dataclass(frozen=True)
class ModelConfig:
    logical_name: str
    hub_id: str
    local_dir: Path
    role: str
    phase: str = "phase1"


PHASE1_MODELS: dict[str, ModelConfig] = {
    "qwen3-8b-instruct": ModelConfig(
        logical_name="qwen3-8b-instruct",
        # Qwen3-8B is the official post-trained 8B text checkpoint; we keep
        # the logical name aligned with the project's "instruct critic" role.
        hub_id="Qwen/Qwen3-8B",
        local_dir=DEFAULT_MODEL_ROOT / "Qwen3-8B",
        role="text orchestrator, critic, coder, and spec summarizer",
    ),
    "qwen3-vl-8b-instruct": ModelConfig(
        logical_name="qwen3-vl-8b-instruct",
        hub_id="Qwen/Qwen3-VL-8B-Instruct",
        local_dir=DEFAULT_MODEL_ROOT / "Qwen3-VL-8B-Instruct",
        role="optional visual scene explanation and observation grounding",
    ),
}


def model_download_targets() -> list[ModelConfig]:
    return list(PHASE1_MODELS.values())
