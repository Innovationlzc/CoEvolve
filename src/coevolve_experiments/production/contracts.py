"""Closed-world inputs for the post-Table-1 Praxis production runner.

These records are an internal runner boundary.  They do not replace the
signed launch configuration; each channel is deterministically derived from
that configuration and the hash-pinned scene registry in the runner checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from ..serialization import sha256_json, to_primitive


PRODUCTION_INJECTION_BUNDLE_KIND = "coevolve_production_injection_bundle"
PRODUCTION_INJECTION_BUNDLE_SCHEMA_VERSION = 1
REQUIRED_PRODUCTION_CHANNELS: Tuple[str, ...] = (
    "term1_simulator_asset",
    "host_clearance_geometry",
    "vlm_scene_prior",
    "route_and_progress_gates",
    "formal_run_manifest",
    "method_gates",
    "memory_gates",
    "provider_route",
    "formal_output",
)


@dataclass(frozen=True, slots=True)
class ProductionChannelBinding:
    """One named, independently fingerprinted runtime injection."""

    name: str
    payload: Mapping[str, Any]
    fingerprint: str

    @classmethod
    def create(
        cls,
        name: str,
        payload: Mapping[str, Any],
    ) -> "ProductionChannelBinding":
        if name not in REQUIRED_PRODUCTION_CHANNELS:
            raise ValueError(f"unknown production injection channel: {name!r}")
        primitive = to_primitive(dict(payload))
        return cls(
            name=name,
            payload=primitive,
            fingerprint=sha256_json(
                {
                    "kind": "coevolve_production_channel_binding",
                    "schema_version": 1,
                    "name": name,
                    "payload": primitive,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ProductionInjectionBundle:
    """All information a real runner must consume before touching physics."""

    kind: str
    schema_version: int
    run_id: str
    launch_config_path: str
    launch_config_sha256: str
    launch_fingerprint: str
    channels: Tuple[ProductionChannelBinding, ...]
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        launch_config_path: str,
        launch_config_sha256: str,
        launch_fingerprint: str,
        channels: Tuple[ProductionChannelBinding, ...],
    ) -> "ProductionInjectionBundle":
        names = tuple(item.name for item in channels)
        if names != REQUIRED_PRODUCTION_CHANNELS:
            raise ValueError(
                "production channels must equal the complete canonical injection set"
            )
        payload = {
            "kind": PRODUCTION_INJECTION_BUNDLE_KIND,
            "schema_version": PRODUCTION_INJECTION_BUNDLE_SCHEMA_VERSION,
            "run_id": run_id,
            "launch_config_path": launch_config_path,
            "launch_config_sha256": launch_config_sha256,
            "launch_fingerprint": launch_fingerprint,
            "channels": channels,
        }
        return cls(
            **payload,
            fingerprint=sha256_json(payload),
        )

    def channel(self, name: str) -> ProductionChannelBinding:
        selected = tuple(item for item in self.channels if item.name == name)
        if len(selected) != 1:
            raise KeyError(name)
        return selected[0]


@dataclass(frozen=True, slots=True)
class MigrationRequirement:
    """One audited incompatibility between frozen V83 and the new runner."""

    code: str
    injection_channel: str
    frozen_v83_evidence: str
    required_change: str


@dataclass(frozen=True, slots=True)
class ProductionPreflightReport:
    """A no-side-effect verdict produced before any simulator dispatch."""

    available: bool
    run_id: str
    launch_config_path: str
    launch_config_sha256: str
    injection_bundle_fingerprint: str
    blocker_codes: Tuple[str, ...]
    requirements: Tuple[MigrationRequirement, ...]


__all__ = [
    "MigrationRequirement",
    "PRODUCTION_INJECTION_BUNDLE_KIND",
    "PRODUCTION_INJECTION_BUNDLE_SCHEMA_VERSION",
    "ProductionChannelBinding",
    "ProductionInjectionBundle",
    "ProductionPreflightReport",
    "REQUIRED_PRODUCTION_CHANNELS",
]
