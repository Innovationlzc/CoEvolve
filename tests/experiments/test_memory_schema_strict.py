from __future__ import annotations

from pathlib import Path

import pytest

from coevolve_experiments.memory import (
    MemoryBootstrapError,
    StrategyBankValidationError,
    bootstrap_transfer_memory,
    empty_action_exemplar_memory,
    empty_online_strategy_overlay,
    validate_action_exemplar_memory,
    validate_online_strategy_overlay,
    validate_strategy_bank,
)
from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE
from coevolve_experiments.memory.json_io import MemoryFileError, read_json_object


@pytest.mark.parametrize(
    ("payload", "validator"),
    [
        (
            {**empty_online_strategy_overlay(), "schema_version": True},
            validate_online_strategy_overlay,
        ),
        (
            {**empty_action_exemplar_memory(), "schema_version": True},
            validate_action_exemplar_memory,
        ),
    ],
)
def test_json_boolean_never_satisfies_integer_memory_schema_version(
    payload, validator
):
    with pytest.raises(ValueError, match="schema_version"):
        validator(payload)


def test_transfer_bank_rejects_boolean_schema_version():
    with pytest.raises(StrategyBankValidationError, match="schema_version"):
        validate_strategy_bank({"schema_version": True, "records": []})


def test_formal_binding_rejects_non_string_fairness_digest(tmp_path):
    binding = {
        "run_id": "run",
        "protocol_id": "protocol",
        "condition_fingerprint": "1" * 64,
        "strategy_bank_sha256": "2" * 64,
        "source_model": "gpt-source",
        "target_model": "gpt-target",
        "plan_path": str((tmp_path / "plan.json").resolve()),
        "plan_sha256": "3" * 64,
        "observation_contract_fingerprint": True,
        "task_instruction_fingerprint": "4" * 64,
        "shared_prompt_contract_fingerprint": "5" * 64,
        "decision_prompt_fingerprint": "7" * 64,
        "provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        "source_provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "source_provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        "skill_api_fingerprint": "6" * 64,
        "full_method_compatibility_fingerprint": "8" * 64,
    }

    with pytest.raises(MemoryBootstrapError, match="lowercase SHA-256"):
        bootstrap_transfer_memory(
            tmp_path / "memory",
            "Empty",
            target_model="GPT",
            formal_binding=binding,
        )


@pytest.mark.parametrize(
    "raw",
    (
        '{"kind":"first","kind":"second"}\n',
        '{"value":NaN}\n',
        '{"value":Infinity}\n',
        '{"value":-Infinity}\n',
    ),
)
def test_shared_memory_loader_rejects_ambiguous_or_nonfinite_json(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "memory.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(MemoryFileError, match="cannot read"):
        read_json_object(path, "memory fixture")
