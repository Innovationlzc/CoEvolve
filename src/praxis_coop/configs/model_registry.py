"""Phase-1 model choices and download targets."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_ROOT = Path(
    os.environ.get("PRAXIS_MODEL_ROOT", str(REPO_ROOT / "models"))
).expanduser()
DEFAULT_MULTIMODAL_MODEL = "qwen3-vl-4b-instruct"


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
        role="historical full-size multimodal model for prior experiment artifacts",
    ),
    "qwen3-vl-4b-instruct": ModelConfig(
        logical_name="qwen3-vl-4b-instruct",
        hub_id="Qwen/Qwen3-VL-4B-Instruct",
        local_dir=DEFAULT_MODEL_ROOT / "Qwen3-VL-4B-Instruct",
        role=(
            "shared task orchestrator, coder, visual evaluator/failure analyzer, "
            "evolution manager, mutation, code repair, and memory update"
        ),
    ),
}


def model_download_targets() -> list[ModelConfig]:
    return list(PHASE1_MODELS.values())
