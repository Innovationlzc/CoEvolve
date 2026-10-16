from __future__ import annotations

from copy import deepcopy

import pytest

from coevolve_experiments.contracts import StrategyBankPin
from coevolve_experiments.provider_routes import (
    PRIMARY_GPT_PROVIDER_ROUTE,
    SECONDARY_GEMINI_PROVIDER_ROUTE,
)
from coevolve_experiments.memory import (
    FROZEN_STRATEGY_BANK_POLICY,
    StrategyBankProvenanceError,
    validate_strategy_bank_provenance,
)


def _pin(*, gemini: bool = False) -> StrategyBankPin:
    route = SECONDARY_GEMINI_PROVIDER_ROUTE if gemini else PRIMARY_GPT_PROVIDER_ROUTE
    return StrategyBankPin(
        source_backbone=("gemini-3.5-flash" if gemini else "gpt-5.4-mini"),
        source_protocol_id=("v83-gemini" if gemini else "v83-gpt"),
        source_method_commit=("b" if gemini else "a") * 40,
        source_seeds=(200, 201, 202, 203, 204),
        source_condition_fingerprint=("2" if gemini else "1") * 64,
        source_result_manifest_hash=("4" if gemini else "3") * 64,
        source_provider_route_id=route.route_id,
        source_provider_route_fingerprint=route.fingerprint,
        selector=("table1/gemini/bank" if gemini else "table1/gpt/bank"),
        # Deliberately allow identical bank content across cohorts.  Provenance
        # identity must still prevent a cross-cohort substitution.
        sha256="5" * 64,
    )


def _provenance(pin: StrategyBankPin) -> dict[str, object]:
    return {
        "kind": "strategy_transfer_provenance",
        "schema_version": 3,
        "bank_sha256": pin.sha256,
        "source_model": pin.source_backbone,
        "source_provider_route_id": pin.source_provider_route_id,
        "source_provider_route_fingerprint": (
            pin.source_provider_route_fingerprint
        ),
        "source_scene": "base",
        "source_protocol": pin.source_protocol_id,
        "source_method_commit": pin.source_method_commit,
        "source_seeds": list(pin.source_seeds),
        "source_condition_fingerprint": pin.source_condition_fingerprint,
        "source_result_manifest_hash": pin.source_result_manifest_hash,
        "policy": FROZEN_STRATEGY_BANK_POLICY.as_dict(),
        "summary": {},
        "sources": [],
    }


def test_complete_provenance_matches_plan_derived_pin() -> None:
    pin = _pin()
    result = validate_strategy_bank_provenance(
        _provenance(pin),
        strategy_bank_pin=pin,
    )

    assert result.semantic_sha256
    assert result.payload["source_result_manifest_hash"] == (
        pin.source_result_manifest_hash
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_model", "gemini-3.5-flash"),
        ("source_protocol", "different-protocol"),
        ("source_method_commit", "f" * 40),
        ("source_seeds", [200, 201, 202, 203, 999]),
        ("source_condition_fingerprint", "e" * 64),
        ("source_result_manifest_hash", "d" * 64),
        ("source_provider_route_fingerprint", "d" * 64),
        ("bank_sha256", "c" * 64),
    ),
)
def test_each_provenance_identity_field_fails_closed(
    field: str,
    replacement: object,
) -> None:
    pin = _pin()
    provenance = deepcopy(_provenance(pin))
    provenance[field] = replacement

    with pytest.raises(StrategyBankProvenanceError, match=field):
        validate_strategy_bank_provenance(
            provenance,
            strategy_bank_pin=pin,
        )


def test_cross_cohort_substitution_fails_even_for_identical_bank_hash() -> None:
    gpt_pin = _pin()
    gemini_provenance = _provenance(_pin(gemini=True))

    with pytest.raises(StrategyBankProvenanceError, match="source_model"):
        validate_strategy_bank_provenance(
            gemini_provenance,
            strategy_bank_pin=gpt_pin,
        )


def test_legacy_partial_provenance_is_not_formal_evidence() -> None:
    pin = _pin()
    provenance = _provenance(pin)
    provenance["schema_version"] = 1
    provenance.pop("source_result_manifest_hash")

    with pytest.raises(StrategyBankProvenanceError, match="fields differ"):
        validate_strategy_bank_provenance(
            provenance,
            strategy_bank_pin=pin,
        )
