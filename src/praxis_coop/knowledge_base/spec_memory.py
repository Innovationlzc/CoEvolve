"""Compatibility facade for knowledge-base imports.

The implementations live in `praxis_coop.spec_memory` and
`praxis_coop.memory`. Keep this module thin so there is only one real memory
implementation to maintain.
"""

from __future__ import annotations

from praxis_coop.memory import DeploymentExperience, DeploymentExperienceMemory, TextSpecMemory
from praxis_coop.spec_memory import SpecMemory, default_specs

__all__ = ["DeploymentExperience", "DeploymentExperienceMemory", "SpecMemory", "TextSpecMemory", "default_specs"]
