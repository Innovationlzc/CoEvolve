from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from coevolve_experiments.serialization import (
    canonical_json,
    sha256_json,
    strict_json_loads,
)


class Choice(str, Enum):
    A = "a"


@dataclass(frozen=True)
class Record:
    choice: Choice
    path: Path
    values: tuple[int, ...]


def test_canonical_serialization_is_stable_for_experiment_records():
    value = Record(Choice.A, Path("relative/file.json"), (2, 1))

    assert canonical_json(value) == (
        '{"choice":"a","path":"relative/file.json","values":[2,1]}'
    )
    assert sha256_json(value) == sha256_json(value)


@pytest.mark.parametrize(
    "payload",
    (
        '{"run_id":"first","run_id":"second"}',
        '{"physical_budget":NaN}',
        '{"physical_budget":Infinity}',
        '{"physical_budget":-Infinity}',
    ),
)
def test_strict_json_rejects_ambiguous_contract_bytes(payload: str) -> None:
    with pytest.raises(ValueError):
        strict_json_loads(payload)
