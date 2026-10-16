from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from coevolve_experiments.production.contracts import (
    ProductionChannelBinding,
    ProductionInjectionBundle,
    REQUIRED_PRODUCTION_CHANNELS,
)
from coevolve_experiments.production.injections import (
    ProductionInjectionConsumptionError,
    ProductionInjectionConsumptionGate,
)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _bundle() -> ProductionInjectionBundle:
    channels = tuple(
        ProductionChannelBinding.create(
            name,
            {
                "name": name,
                "nested": {"sequence": [name, {"enabled": True}]},
            },
        )
        for name in REQUIRED_PRODUCTION_CHANNELS
    )
    return ProductionInjectionBundle.create(
        run_id="fixture-run",
        launch_config_path="/tmp/fixture-launch.json",
        launch_config_sha256=_digest("config"),
        launch_fingerprint=_digest("launch"),
        channels=channels,
    )


def test_all_required_channels_must_be_consumed_once_before_dispatch() -> None:
    gate = ProductionInjectionConsumptionGate(_bundle())

    for name in reversed(REQUIRED_PRODUCTION_CHANNELS):
        payload = gate.consume(name)
        assert payload["name"] == name

    assert gate.consumed_channels == REQUIRED_PRODUCTION_CHANNELS
    assert gate.remaining_channels == ()
    assert gate.failed_closed is False
    assert gate.assert_ready_for_dispatch() is None


def test_readiness_check_with_unconsumed_channels_latches_failure() -> None:
    gate = ProductionInjectionConsumptionGate(_bundle())
    gate.consume(REQUIRED_PRODUCTION_CHANNELS[0])

    with pytest.raises(
        ProductionInjectionConsumptionError,
        match="remain unconsumed",
    ):
        gate.assert_ready_for_dispatch()
    assert gate.failed_closed is True
    with pytest.raises(ProductionInjectionConsumptionError, match="failed closed"):
        gate.consume(REQUIRED_PRODUCTION_CHANNELS[1])


def test_unknown_channel_latches_failure() -> None:
    gate = ProductionInjectionConsumptionGate(_bundle())

    with pytest.raises(ProductionInjectionConsumptionError, match="unknown"):
        gate.consume("unreviewed_override")
    assert gate.failed_closed is True
    with pytest.raises(ProductionInjectionConsumptionError, match="failed closed"):
        gate.assert_ready_for_dispatch()


def test_duplicate_channel_latches_failure() -> None:
    gate = ProductionInjectionConsumptionGate(_bundle())
    name = REQUIRED_PRODUCTION_CHANNELS[0]
    gate.consume(name)

    with pytest.raises(ProductionInjectionConsumptionError, match="more than once"):
        gate.consume(name)
    assert gate.failed_closed is True


def test_consumed_payload_is_detached_and_recursively_read_only() -> None:
    bundle = _bundle()
    gate = ProductionInjectionConsumptionGate(bundle)
    name = REQUIRED_PRODUCTION_CHANNELS[0]
    original_payload = bundle.channel(name).payload
    payload = gate.consume(name)

    assert payload is not original_payload
    assert payload["nested"] is not original_payload["nested"]
    with pytest.raises(TypeError):
        payload["name"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        payload["nested"]["sequence"][1]["enabled"] = False  # type: ignore[index]
    assert original_payload["name"] == name
    assert original_payload["nested"]["sequence"][1]["enabled"] is True


def test_gate_rejects_payload_or_bundle_fingerprint_drift() -> None:
    bundle = _bundle()
    first = bundle.channels[0]
    first.payload["nested"]["sequence"][1]["enabled"] = False  # type: ignore[index]
    with pytest.raises(ProductionInjectionConsumptionError, match="fingerprint mismatch"):
        ProductionInjectionConsumptionGate(bundle)

    clean = _bundle()
    poisoned = replace(clean, fingerprint=_digest("poisoned-bundle"))
    with pytest.raises(
        ProductionInjectionConsumptionError,
        match="bundle fingerprint mismatch",
    ):
        ProductionInjectionConsumptionGate(poisoned)
