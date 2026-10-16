"""Anonymized provider-route templates for offline protocol validation.

An API key is authorization material, not experiment identity, and must never
enter a lock, plan, result, or provenance artifact.  The records below bind
only template routing facts: provider label, reserved example endpoint,
API family, and a stable route ID. These are NOT the original sealed paper
route fingerprints. Do not relabel historical evidence to match these fixtures.
Live use requires a separately reviewed route and freshly bound provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .serialization import sha256_json


PROVIDER_ROUTE_SCHEMA_VERSION = 1
PRIMARY_GPT_PROVIDER_ROUTE_ID = "provider_a.openai-compatible.chat-completions.v1"
SECONDARY_GEMINI_PROVIDER_ROUTE_ID = "provider_b.openai-compatible.chat-completions.v1"


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    route_id: str
    provider: str
    base_url: str
    api_family: str = "chat_completions"

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            {
                "kind": "coevolve_nonsecret_provider_route",
                "schema_version": PROVIDER_ROUTE_SCHEMA_VERSION,
                "route_id": self.route_id,
                "provider": self.provider,
                "base_url": self.base_url,
                "api_family": self.api_family,
            }
        )


PRIMARY_GPT_PROVIDER_ROUTE = ProviderRoute(
    route_id=PRIMARY_GPT_PROVIDER_ROUTE_ID,
    provider="provider_a",
    base_url="https://provider-a.example.invalid/v1",
)
SECONDARY_GEMINI_PROVIDER_ROUTE = ProviderRoute(
    route_id=SECONDARY_GEMINI_PROVIDER_ROUTE_ID,
    provider="provider_b",
    base_url="https://provider-b.example.invalid/v1",
)

_ROUTES: Mapping[str, ProviderRoute] = {
    route.route_id: route
    for route in (PRIMARY_GPT_PROVIDER_ROUTE, SECONDARY_GEMINI_PROVIDER_ROUTE)
}


def get_provider_route(route_id: str) -> ProviderRoute:
    """Resolve one reviewed route without accepting a caller-authored URL."""

    try:
        return _ROUTES[route_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unregistered provider route: {route_id!r}") from exc


def require_provider_route(
    route_id: str,
    fingerprint: str,
    *,
    expected_route_id: str | None = None,
) -> ProviderRoute:
    """Validate an ID/fingerprint pair against the non-secret route registry."""

    if expected_route_id is not None and route_id != expected_route_id:
        raise ValueError(
            f"provider route must be {expected_route_id!r}, got {route_id!r}"
        )
    route = get_provider_route(route_id)
    if fingerprint != route.fingerprint:
        raise ValueError(
            f"provider route fingerprint differs for registered route {route_id!r}"
        )
    return route


__all__ = [
    "PRIMARY_GPT_PROVIDER_ROUTE",
    "PRIMARY_GPT_PROVIDER_ROUTE_ID",
    "PROVIDER_ROUTE_SCHEMA_VERSION",
    "ProviderRoute",
    "SECONDARY_GEMINI_PROVIDER_ROUTE",
    "SECONDARY_GEMINI_PROVIDER_ROUTE_ID",
    "get_provider_route",
    "require_provider_route",
]
