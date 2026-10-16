"""Fail-closed, single-consumption gate for production injections.

The production stack must prove that every hash-bound injection is wired into
its corresponding adapter before it may dispatch any child process.  This
module only tracks that wiring: it performs no I/O and has no runtime side
effects outside the gate instance.
"""

from __future__ import annotations

from threading import Lock
from types import MappingProxyType
from typing import Any, Mapping

from ..serialization import to_primitive
from .contracts import (
    PRODUCTION_INJECTION_BUNDLE_KIND,
    PRODUCTION_INJECTION_BUNDLE_SCHEMA_VERSION,
    ProductionChannelBinding,
    ProductionInjectionBundle,
    REQUIRED_PRODUCTION_CHANNELS,
)


class ProductionInjectionConsumptionError(RuntimeError):
    """The injection handoff is invalid or can no longer authorize dispatch."""


class ProductionInjectionConsumptionGate:
    """Consume every required production injection exactly once.

    A protocol violation is latched.  Catching the exception cannot make the
    same gate dispatch-ready later; callers must discard it and rebuild the
    handoff from the verified launch configuration.
    """

    __slots__ = ("_consumed", "_failure_reason", "_lock", "_payloads")

    def __init__(self, bundle: ProductionInjectionBundle) -> None:
        self._lock = Lock()
        self._consumed: set[str] = set()
        self._failure_reason: str | None = None
        self._payloads = _validated_payload_snapshot(bundle)

    @property
    def consumed_channels(self) -> tuple[str, ...]:
        """Consumed names in canonical order, never caller-controlled order."""

        with self._lock:
            return tuple(
                name for name in REQUIRED_PRODUCTION_CHANNELS if name in self._consumed
            )

    @property
    def remaining_channels(self) -> tuple[str, ...]:
        """Required names that have not yet been consumed."""

        with self._lock:
            return tuple(
                name
                for name in REQUIRED_PRODUCTION_CHANNELS
                if name not in self._consumed
            )

    @property
    def failed_closed(self) -> bool:
        """Whether a protocol violation permanently invalidated this gate."""

        with self._lock:
            return self._failure_reason is not None

    def consume(self, name: str) -> Mapping[str, Any]:
        """Consume one named channel and return its recursively read-only copy."""

        with self._lock:
            self._raise_if_failed()
            if name not in REQUIRED_PRODUCTION_CHANNELS:
                self._fail(f"unknown production injection channel: {name!r}")
            if name in self._consumed:
                self._fail(
                    f"production injection channel was consumed more than once: {name!r}"
                )
            self._consumed.add(name)
            return self._payloads[name]

    def assert_ready_for_dispatch(self) -> None:
        """Succeed only after all nine channels were consumed exactly once."""

        with self._lock:
            self._raise_if_failed()
            remaining = tuple(
                name
                for name in REQUIRED_PRODUCTION_CHANNELS
                if name not in self._consumed
            )
            if remaining:
                self._fail(
                    "production injections remain unconsumed: "
                    + ", ".join(remaining)
                )

    def _raise_if_failed(self) -> None:
        if self._failure_reason is not None:
            raise ProductionInjectionConsumptionError(
                f"production injection gate is failed closed: {self._failure_reason}"
            )

    def _fail(self, reason: str) -> None:
        self._failure_reason = reason
        raise ProductionInjectionConsumptionError(reason)


def _validated_payload_snapshot(
    bundle: ProductionInjectionBundle,
) -> Mapping[str, Mapping[str, Any]]:
    """Validate fingerprints and detach all payloads from the source bundle."""

    if not isinstance(bundle, ProductionInjectionBundle):
        raise ProductionInjectionConsumptionError(
            "production injection gate requires ProductionInjectionBundle"
        )
    if bundle.kind != PRODUCTION_INJECTION_BUNDLE_KIND:
        raise ProductionInjectionConsumptionError(
            "production injection bundle kind differs from the reviewed contract"
        )
    if bundle.schema_version != PRODUCTION_INJECTION_BUNDLE_SCHEMA_VERSION:
        raise ProductionInjectionConsumptionError(
            "production injection bundle schema differs from the reviewed contract"
        )

    names = tuple(binding.name for binding in bundle.channels)
    if names != REQUIRED_PRODUCTION_CHANNELS:
        raise ProductionInjectionConsumptionError(
            "production injection bundle does not contain the canonical channel set"
        )

    rebuilt_bindings: list[ProductionChannelBinding] = []
    readonly_payloads: dict[str, Mapping[str, Any]] = {}
    for binding in bundle.channels:
        if not isinstance(binding, ProductionChannelBinding):
            raise ProductionInjectionConsumptionError(
                "production injection bundle contains an invalid channel binding"
            )
        try:
            detached = to_primitive(binding.payload)
            rebuilt = ProductionChannelBinding.create(binding.name, detached)
        except (TypeError, ValueError) as exc:
            raise ProductionInjectionConsumptionError(
                f"invalid payload for production injection channel {binding.name!r}"
            ) from exc
        if binding.fingerprint != rebuilt.fingerprint:
            raise ProductionInjectionConsumptionError(
                f"production injection channel fingerprint mismatch: {binding.name!r}"
            )
        frozen = _deep_readonly(detached)
        if not isinstance(frozen, Mapping):  # payload contract requires a mapping
            raise ProductionInjectionConsumptionError(
                f"production injection channel payload is not a mapping: {binding.name!r}"
            )
        rebuilt_bindings.append(rebuilt)
        readonly_payloads[binding.name] = frozen

    rebuilt_bundle = ProductionInjectionBundle.create(
        run_id=bundle.run_id,
        launch_config_path=bundle.launch_config_path,
        launch_config_sha256=bundle.launch_config_sha256,
        launch_fingerprint=bundle.launch_fingerprint,
        channels=tuple(rebuilt_bindings),
    )
    if bundle.fingerprint != rebuilt_bundle.fingerprint:
        raise ProductionInjectionConsumptionError(
            "production injection bundle fingerprint mismatch"
        )
    return MappingProxyType(readonly_payloads)


def _deep_readonly(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_readonly(child) for key, child in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_readonly(child) for child in value)
    return value


__all__ = [
    "ProductionInjectionConsumptionError",
    "ProductionInjectionConsumptionGate",
]
