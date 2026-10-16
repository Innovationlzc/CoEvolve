"""Provider-boundary errors shared by remote model clients."""

from __future__ import annotations


class ProviderFatalError(RuntimeError):
    """Raised when a provider failure invalidates the current evaluation run."""

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status
