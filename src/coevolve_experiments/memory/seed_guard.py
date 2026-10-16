"""Immutable transfer-seed guard, independent of mutable online memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .validation import load_strategy_bank, validate_strategy_bank


class SeedIntegrityError(ValueError):
    """The frozen transfer seed is invalid, writable, or hash-drifted."""


@dataclass(frozen=True)
class SeedBankGuard:
    path: Path
    expected_sha256: str

    def verify(self, stage: str) -> str:
        try:
            bank = load_strategy_bank(self.path)
            validation = validate_strategy_bank(bank)
            mode = self.path.stat().st_mode
        except Exception as exc:
            raise SeedIntegrityError(
                f"seed-bank validation failed at {stage}: {exc}"
            ) from exc
        if mode & 0o222:
            raise SeedIntegrityError(f"seed bank became writable at {stage}")
        if validation.bank_sha256 != self.expected_sha256:
            raise SeedIntegrityError(
                f"seed bank hash drift at {stage}: "
                f"{validation.bank_sha256!r} != {self.expected_sha256!r}"
            )
        return validation.bank_sha256


__all__ = ["SeedBankGuard", "SeedIntegrityError"]
