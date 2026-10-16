"""Configuration registries for Praxis-Coop."""

from .model_registry import (
    DEFAULT_MULTIMODAL_MODEL,
    ModelConfig,
    PHASE1_MODELS,
    model_download_targets,
)

__all__ = [
    "DEFAULT_MULTIMODAL_MODEL",
    "ModelConfig",
    "PHASE1_MODELS",
    "model_download_targets",
]
