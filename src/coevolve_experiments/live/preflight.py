"""Current production-bridge readiness.

The existing Table-1 launcher hard-codes its Base-scene protocol, output tree,
and empty-memory rules.  It cannot yet consume a post-Table-1 ``RunSpec`` or a
Table-4 seed bank.  Keep this explicit and fail closed until the main runner
exposes one generic hash-bound launch configuration.
"""

from __future__ import annotations

from .contracts import (
    ExecutionCapability,
    LiveBridgePreflight,
    LiveBridgeUnavailableError,
    PolicyRepresentation,
)


LIVE_BRIDGE_UNAVAILABLE_CODE = "FULL_CODE_POLICY_BRIDGE_UNAVAILABLE"


def current_live_bridge_preflight() -> LiveBridgePreflight:
    return LiveBridgePreflight(
        available=False,
        policy_representation=PolicyRepresentation.PRAXIS_CODE_AS_POLICY,
        execution_capability=ExecutionCapability.PAPER_FORMAL,
        reason_code=LIVE_BRIDGE_UNAVAILABLE_CODE,
        detail=(
            "the frozen Table-1 Code-as-Policy launcher does not yet accept a "
            "RunSpec-bound scene/method/memory configuration"
        ),
    )


def require_current_live_bridge() -> LiveBridgePreflight:
    status = current_live_bridge_preflight()
    if not status.available:
        raise LiveBridgeUnavailableError(
            f"{status.reason_code}: {status.detail}"
        )
    return status


__all__ = [
    "LIVE_BRIDGE_UNAVAILABLE_CODE",
    "current_live_bridge_preflight",
    "require_current_live_bridge",
]
