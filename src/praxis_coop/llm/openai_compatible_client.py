"""Minimal secret-safe OpenAI-compatible multimodal client."""

from __future__ import annotations

from base64 import b64encode
from contextlib import contextmanager
from dataclasses import dataclass, field
import http.client
import json
import mimetypes
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any
from urllib import error, request
from urllib.parse import quote, urlsplit

from .errors import ProviderFatalError

REGISTERED_MODEL_IDENTITY_ALIASES: dict[str, frozenset[str]] = {
    "claude-sonnet-4-6": frozenset(
        {
            "claude-sonnet-4-6",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-sonnet-4.6",
        }
    ),
    "gpt-5.4-mini-2026-03-17": frozenset(
        {
            "gpt-5.4-mini-2026-03-17",
            "openai/gpt-5.4-mini-2026-03-17",
        }
    ),
    "gemini-3.5-flash": frozenset(
        {
            "gemini-3.5-flash",
            "google/gemini-3.5-flash",
        }
    ),
}


def canonicalize_registered_model_identity(
    model_id: str,
    *,
    required: str,
) -> str:
    """Map a pre-registered provider alias onto the requested formal model id.

    Unregistered names are returned unchanged so identity checks still reject a
    true model swap. This is an exact allowlist, not string similarity.
    """

    required_id = str(required).strip()
    resolved_id = str(model_id).strip()
    aliases = REGISTERED_MODEL_IDENTITY_ALIASES.get(required_id)
    if aliases is None:
        return resolved_id
    if resolved_id in aliases:
        return required_id
    return resolved_id


class _RequestWallClockTimeout(Exception):
    """Internal signal used to bound one complete provider invocation."""


@contextmanager
def _wall_clock_deadline(timeout_s: float):
    """Bound blocking DNS/proxy/TLS work that socket timeouts may not cover."""

    can_arm = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    if not can_arm:
        yield
        return

    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0.0:
        # Do not replace a caller-owned wall-clock deadline.
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline_reached(_signum: int, _frame: Any) -> None:
        raise _RequestWallClockTimeout

    signal.signal(signal.SIGALRM, deadline_reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.1, float(timeout_s)))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _provider_invocation_deadline_s(timeout_s: float, *, retries: int) -> float:
    """Allow every transport attempt its configured timeout plus retry backoff."""

    attempts = max(1, int(retries) + 1)
    backoff_s = sum(min(30.0, 2.0**index) for index in range(attempts - 1))
    return max(0.1, float(timeout_s)) * attempts + backoff_s


def _is_transient_provider_error(exc: ProviderFatalError) -> bool:
    """Return whether repeating the unchanged request can recover safely."""

    status = exc.http_status
    if status is not None:
        return status in {408, 409, 429} or 500 <= status <= 599
    message = str(exc)
    return message.startswith("OpenAI-compatible request failed:") or (
        "provider invocation exceeded" in message
    )


@dataclass
class OpenAICompatibleTextClient:
    """Generate text or inspect one image through a chat-completions endpoint.

    The API key is resolved for every call from ``api_key_env``. It is never
    stored in artifacts, command arguments, usage records, or exceptions.
    """

    model_name: str
    base_url: str = field(default_factory=lambda: os.environ.get("COEVOLVE_BASE_URL", ""))
    api_key_env: str = "COEVOLVE_API_KEY"
    max_new_tokens: int = 1200
    temperature: float = 0.2
    timeout_s: float = 120.0
    network_retries: int = 2
    json_response: bool = True
    seed: int | None = None
    required_resolved_model: str = ""
    keep_alive_on_transient_failure: bool = False
    transient_recovery_backoff_s: float = 60.0
    transient_recovery_wall_time_cap_s: float = 0.0
    # Optional provider-specific accounting adapter; disabled unless configured.
    # Only a same-origin path is accepted, never a separate credential recipient.
    usage_path_template: str = ""
    call_records: list[dict[str, Any]] = field(default_factory=list, init=False)

    def generate(self, *, system: str, user: str) -> str:
        return self._generate(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            kind="remote_text_generation",
        )

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> str:
        """Generate one provider-normalized JSON object."""

        return self._generate(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            kind="remote_text_generation",
            schema_name=schema_name,
            schema=schema,
        )

    def analyze_image(
        self,
        *,
        system: str,
        user: str,
        image_path: str | Path,
    ) -> str:
        path = Path(image_path)
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        data_url = f"data:{media_type};base64," + b64encode(path.read_bytes()).decode(
            "ascii"
        )
        return self._generate(
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            kind="remote_vision_generation",
            image_path=str(path),
        )

    def analyze_image_structured(
        self,
        *,
        system: str,
        user: str,
        image_path: str | Path,
        schema_name: str,
        schema: dict[str, Any],
    ) -> str:
        """Analyze one image with the same logical JSON schema as text roles."""

        path = Path(image_path)
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        data_url = f"data:{media_type};base64," + b64encode(path.read_bytes()).decode(
            "ascii"
        )
        return self._generate(
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            kind="remote_vision_generation",
            image_path=str(path),
            schema_name=schema_name,
            schema=schema,
        )

    def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        kind: str,
        image_path: str = "",
        schema_name: str = "",
        schema: dict[str, Any] | None = None,
    ) -> str:
        recovery_round = 0
        recovery_started = time.monotonic()
        recovery_cap_s = max(0.0, float(self.transient_recovery_wall_time_cap_s))
        while True:
            remaining_recovery_s = (
                recovery_cap_s - (time.monotonic() - recovery_started)
                if recovery_cap_s > 0.0
                else None
            )
            if remaining_recovery_s is not None and remaining_recovery_s <= 0.0:
                self._mark_provider_recovery_exhausted(
                    recovery_round=recovery_round,
                    wall_time_cap_s=recovery_cap_s,
                )
                raise ProviderFatalError(
                    "transient model-provider recovery exceeded the "
                    f"{recovery_cap_s:.1f}s logical-call wall-time cap"
                )
            try:
                result = self._generate_bounded(
                    messages=messages,
                    kind=kind,
                    image_path=image_path,
                    schema_name=schema_name,
                    schema=schema,
                    invocation_wall_time_cap_s=remaining_recovery_s,
                )
            except ProviderFatalError as exc:
                if not (
                    self.keep_alive_on_transient_failure
                    and _is_transient_provider_error(exc)
                ):
                    raise
                recovery_round += 1
                if self.call_records:
                    self.call_records[-1].update(
                        {
                            "provider_fatal": False,
                            "transient_provider_outage": True,
                            "provider_recovery_pending": True,
                            "provider_recovery_round": recovery_round,
                        }
                    )
                delay_s = max(1.0, float(self.transient_recovery_backoff_s))
                if recovery_cap_s > 0.0:
                    remaining_recovery_s = recovery_cap_s - (
                        time.monotonic() - recovery_started
                    )
                    if remaining_recovery_s <= 0.0:
                        self._mark_provider_recovery_exhausted(
                            recovery_round=recovery_round,
                            wall_time_cap_s=recovery_cap_s,
                        )
                        raise ProviderFatalError(
                            "transient model-provider recovery exceeded the "
                            f"{recovery_cap_s:.1f}s logical-call wall-time cap"
                        ) from exc
                    delay_s = min(delay_s, remaining_recovery_s)
                print(
                    "[provider-recovery] transient model-provider failure; "
                    f"keeping Term4 alive and retrying the same call in {delay_s:.1f}s "
                    f"(recovery_round={recovery_round}, http_status={exc.http_status})",
                    flush=True,
                )
                time.sleep(delay_s)
                continue
            if recovery_round and self.call_records:
                self.call_records[-1].update(
                    {
                        "provider_recovered": True,
                        "provider_recovery_pending": False,
                        "provider_recovery_rounds": recovery_round,
                    }
                )
            return result

    def _mark_provider_recovery_exhausted(
        self,
        *,
        recovery_round: int,
        wall_time_cap_s: float,
    ) -> None:
        """Seal an operational outage as provider-fatal, never task failure."""

        if not self.call_records:
            return
        self.call_records[-1].update(
            {
                "provider_fatal": True,
                "transient_provider_outage": True,
                "provider_recovery_pending": False,
                "provider_recovery_exhausted": True,
                "provider_recovery_round": int(recovery_round),
                "provider_recovery_wall_time_cap_s": float(wall_time_cap_s),
            }
        )

    def _generate_bounded(
        self,
        *,
        messages: list[dict[str, Any]],
        kind: str,
        image_path: str = "",
        schema_name: str = "",
        schema: dict[str, Any] | None = None,
        invocation_wall_time_cap_s: float | None = None,
    ) -> str:
        """Run one bounded transport-retry round for a model invocation."""

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise ProviderFatalError(
                f"missing required API credential environment variable {self.api_key_env}"
            )
        _validate_base_url(self.base_url)
        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_new_tokens),
        }
        structured_output_mode = "disabled"
        if schema_name and isinstance(schema, dict) and schema:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _safe_schema_name(schema_name),
                    "strict": True,
                    "schema": schema,
                },
            }
            structured_output_mode = "json_schema"
        elif self.json_response:
            request_payload["response_format"] = {"type": "json_object"}
            structured_output_mode = "json_object"
        if self.seed is not None:
            request_payload["seed"] = int(self.seed)
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        started = time.time()
        timeout_s = max(0.1, float(self.timeout_s))
        last_error: Exception | None = None
        last_http_status: int | None = None
        schema_fallback_used = False
        transport_failures: list[dict[str, Any]] = []
        retry = 0
        invocation_deadline_s = _provider_invocation_deadline_s(
            timeout_s,
            retries=max(0, int(self.network_retries)),
        )
        if invocation_wall_time_cap_s is not None:
            invocation_deadline_s = min(
                invocation_deadline_s,
                max(0.1, float(invocation_wall_time_cap_s)),
            )
        try:
            with _wall_clock_deadline(invocation_deadline_s):
                for retry in range(max(0, int(self.network_retries)) + 1):
                    body = json.dumps(request_payload, ensure_ascii=False).encode(
                        "utf-8"
                    )
                    http_request = request.Request(
                        endpoint,
                        data=body,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "User-Agent": "praxis-coop/0.1",
                        },
                        method="POST",
                    )
                    try:
                        with request.urlopen(
                            http_request, timeout=timeout_s
                        ) as response:
                            response_payload = json.loads(
                                response.read().decode("utf-8")
                            )
                            response_headers = _response_headers(response)
                            http_status = int(
                                getattr(response, "status", 0)
                                or getattr(response, "code", 0)
                                or 200
                            )
                        result = _completion_text(response_payload)
                        usage = response_payload.get("usage", {})
                        usage = usage if isinstance(usage, dict) else {}
                        prompt_details = usage.get("prompt_tokens_details", {})
                        prompt_details = (
                            prompt_details if isinstance(prompt_details, dict) else {}
                        )
                        completion_details = usage.get(
                            "completion_tokens_details", {}
                        )
                        completion_details = (
                            completion_details
                            if isinstance(completion_details, dict)
                            else {}
                        )
                        request_id = (
                            response_headers.get("x-provider-request-id")
                            or response_headers.get("x-request-id")
                            or ""
                        )
                        request_usage = _fetch_openai_compatible_request_usage(
                            base_url=self.base_url,
                            usage_path_template=self.usage_path_template,
                            api_key=api_key,
                            request_id=request_id,
                            timeout_s=min(timeout_s, 30.0),
                            attempts=1,
                        )
                        dimensions = _usage_dimensions(request_usage)
                        billing = request_usage.get("billing", {})
                        billing = billing if isinstance(billing, dict) else {}
                        routing = request_usage.get("routing", {})
                        routing = routing if isinstance(routing, dict) else {}
                        provider_usage = request_usage.get("provider_usage", {})
                        provider_usage = (
                            provider_usage
                            if isinstance(provider_usage, dict)
                            else {}
                        )
                        amount_microusd = _optional_int(
                            billing.get("amount_microusd")
                        )
                        record = {
                            "model_name": self.model_name,
                            "requested_model": self.model_name,
                            "response_model": str(response_payload.get("model", "")),
                            "resolved_model": str(
                                routing.get(
                                    "resolved_model",
                                    response_payload.get("model", ""),
                                )
                            ),
                            "kind": kind,
                            "requested_seed": self.seed,
                            "started_at_unix": started,
                            "latency_s": time.time() - started,
                            "http_status": http_status,
                            "request_id": request_id,
                            "provider_channel_id": (
                                response_headers.get("x-provider-channel-id", "")
                            ),
                            "lane": response_headers.get("x-provider-lane", ""),
                            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                            "output_tokens": int(
                                usage.get("completion_tokens", 0) or 0
                            ),
                            "total_tokens": int(usage.get("total_tokens", 0) or 0),
                            "cached_input_tokens": int(
                                prompt_details.get("cached_tokens", 0) or 0
                            ),
                            "text_input_tokens": int(
                                prompt_details.get(
                                    "text_tokens",
                                    dimensions.get("input", 0),
                                )
                                or 0
                            ),
                            "image_input_tokens": int(
                                prompt_details.get(
                                    "image_tokens",
                                    dimensions.get("image_input", 0),
                                )
                                or 0
                            ),
                            "audio_input_tokens": int(
                                prompt_details.get(
                                    "audio_tokens",
                                    dimensions.get("audio_input", 0),
                                )
                                or 0
                            ),
                            "reasoning_tokens": int(
                                completion_details.get("reasoning_tokens", 0) or 0
                            ),
                            "billing_amount_microusd": amount_microusd,
                            "billing_amount_usd": (
                                amount_microusd / 1_000_000.0
                                if amount_microusd is not None
                                else None
                            ),
                            "billing_currency": str(billing.get("currency", "")),
                            "billing_pricing_version": billing.get(
                                "pricing_version"
                            ),
                            "billing_line_items": _json_list(
                                billing.get("line_items")
                            ),
                            "billing_missing_price_dimensions": _string_list(
                                billing.get("missing_price_dimensions")
                            ),
                            "billing_lookup_status": str(
                                request_usage.get("_lookup_status", "not_attempted")
                            ),
                            "billing_lookup_error": str(
                                request_usage.get("_lookup_error", "")
                            ),
                            "provider_family": str(provider_usage.get("family", "")),
                            "routing": routing,
                            "network_retries": retry,
                            "structured_output_mode": structured_output_mode,
                            "schema_fallback_used": schema_fallback_used,
                            "transport_failures": list(transport_failures),
                            **({"image_path": image_path} if image_path else {}),
                        }
                        self.call_records.append(record)
                        required_model = self.required_resolved_model.strip()
                        resolved_model = str(record["resolved_model"]).strip()
                        canonical_resolved = canonicalize_registered_model_identity(
                            resolved_model,
                            required=required_model,
                        )
                        record["canonical_resolved_model"] = canonical_resolved
                        if required_model and canonical_resolved != required_model:
                            raise ProviderFatalError(
                                "formal model identity mismatch: "
                                f"requested={required_model!r}, resolved={resolved_model!r}"
                            )
                        return result
                    except error.HTTPError as exc:
                        status = int(exc.code)
                        last_http_status = status
                        response_text = _safe_http_error_body(exc, api_key=api_key)
                        last_error = ProviderFatalError(
                            "OpenAI-compatible endpoint returned HTTP "
                            f"{status}: {response_text}",
                            http_status=status,
                        )
                        transport_failures.append(
                            {
                                "attempt_index": retry,
                                "kind": "http_error",
                                "http_status": status,
                                "error": str(last_error),
                            }
                        )
                        if (
                            status in {400, 404, 415, 422}
                            and structured_output_mode == "json_schema"
                            and not schema_fallback_used
                        ):
                            request_payload["response_format"] = {
                                "type": "json_object"
                            }
                            structured_output_mode = "json_object"
                            schema_fallback_used = True
                            continue
                        if status not in {408, 409, 429, 524} and not (
                            500 <= status <= 599
                        ):
                            break
                    except (
                        error.URLError,
                        TimeoutError,
                        OSError,
                        http.client.HTTPException,
                        json.JSONDecodeError,
                        ValueError,
                    ) as exc:
                        last_error = ProviderFatalError(
                            "OpenAI-compatible request failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        transport_failures.append(
                            {
                                "attempt_index": retry,
                                "kind": "transport_error",
                                "http_status": None,
                                "error_type": type(exc).__name__,
                                "error": str(last_error),
                            }
                        )
                    if retry < max(0, int(self.network_retries)):
                        time.sleep(min(30.0, 2.0**retry))
        except _RequestWallClockTimeout:
            last_error = ProviderFatalError(
                "OpenAI-compatible provider invocation exceeded the "
                f"{invocation_deadline_s:.1f}s wall-clock deadline"
            )
            transport_failures.append(
                {
                    "attempt_index": retry,
                    "kind": "wall_clock_timeout",
                    "http_status": None,
                    "error": str(last_error),
                }
            )
        self.call_records.append(
            {
                "model_name": self.model_name,
                "requested_model": self.model_name,
                "response_model": "",
                "resolved_model": "",
                "kind": kind,
                "requested_seed": self.seed,
                "started_at_unix": started,
                "latency_s": time.time() - started,
                "http_status": last_http_status,
                "provider_fatal": True,
                "error": str(last_error or "request failed without a response"),
                "network_retries": retry,
                "structured_output_mode": structured_output_mode,
                "schema_fallback_used": schema_fallback_used,
                "transport_failures": list(transport_failures),
                **({"image_path": image_path} if image_path else {}),
            }
        )
        if isinstance(last_error, ProviderFatalError):
            raise last_error
        raise ProviderFatalError(
            "OpenAI-compatible request failed without a response",
            http_status=last_http_status,
        )

    def usage_summary(self) -> dict[str, Any]:
        completed = [
            item
            for item in self.call_records
            if str(item.get("request_id", "")).strip()
        ]
        billed = [
            item
            for item in completed
            if item.get("billing_amount_microusd") is not None
        ]
        amount_microusd = sum(int(item["billing_amount_microusd"]) for item in billed)
        return {
            "calls": len(self.call_records),
            "input_tokens": sum(
                int(item.get("input_tokens", 0)) for item in self.call_records
            ),
            "output_tokens": sum(
                int(item.get("output_tokens", 0)) for item in self.call_records
            ),
            "cached_input_tokens": sum(
                int(item.get("cached_input_tokens", 0)) for item in self.call_records
            ),
            "image_input_tokens": sum(
                int(item.get("image_input_tokens", 0)) for item in self.call_records
            ),
            "reasoning_tokens": sum(
                int(item.get("reasoning_tokens", 0)) for item in self.call_records
            ),
            "latency_s": sum(
                float(item.get("latency_s", 0.0)) for item in self.call_records
            ),
            "billed_call_count": len(billed),
            "unbilled_call_count": len(completed) - len(billed),
            "failed_transport_round_count": len(self.call_records) - len(completed),
            "billing_amount_microusd": amount_microusd,
            "billing_amount_usd": amount_microusd / 1_000_000.0,
        }

    def reconcile_billing(
        self,
        *,
        max_rounds: int = 1,
        initial_backoff_s: float = 0.0,
        per_request_delay_s: float = 0.0,
        total_timeout_s: float = 8.0,
        lookup_timeout_s: float = 1.5,
    ) -> dict[str, Any]:
        """Best-effort billing lookup bounded by one terminal wall-clock budget."""

        supported = bool(self.usage_path_template)
        candidates = [
            item
            for item in self.call_records
            if isinstance(item, dict)
            and str(item.get("request_id", "")).strip()
            and item.get("billing_amount_microusd") is None
        ]
        initial_pending = len(candidates)
        if not supported or not candidates:
            return {
                "status": "not_required" if not candidates else "not_supported",
                "initial_pending": initial_pending,
                "resolved": 0,
                "remaining": initial_pending,
                "rounds": 0,
            }
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            return {
                "status": "missing_credential",
                "initial_pending": initial_pending,
                "resolved": 0,
                "remaining": initial_pending,
                "rounds": 0,
            }

        rounds = 0
        deadline = time.monotonic() + max(0.1, float(total_timeout_s))
        deadline_exhausted = False
        for round_index in range(max(1, int(max_rounds))):
            rounds = round_index + 1
            pending = [
                item
                for item in candidates
                if item.get("billing_amount_microusd") is None
            ]
            if not pending:
                break
            for index, record in enumerate(pending):
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    deadline_exhausted = True
                    break
                effective_lookup_timeout_s = min(
                    max(0.1, float(lookup_timeout_s)),
                    remaining_s,
                )
                try:
                    with _wall_clock_deadline(effective_lookup_timeout_s):
                        request_usage = _fetch_openai_compatible_request_usage(
                            base_url=self.base_url,
                            usage_path_template=self.usage_path_template,
                            api_key=api_key,
                            request_id=str(record.get("request_id", "")),
                            timeout_s=effective_lookup_timeout_s,
                            attempts=1,
                        )
                except _RequestWallClockTimeout:
                    request_usage = {
                        "_lookup_status": "failed",
                        "_lookup_error": "terminal billing lookup deadline exceeded",
                    }
                    deadline_exhausted = True
                _apply_openai_compatible_request_usage(record, request_usage)
                if index + 1 < len(pending) and per_request_delay_s > 0.0:
                    delay_s = min(
                        float(per_request_delay_s),
                        max(0.0, deadline - time.monotonic()),
                    )
                    if delay_s > 0.0:
                        time.sleep(delay_s)
                if deadline_exhausted:
                    break
            remaining = sum(
                item.get("billing_amount_microusd") is None for item in candidates
            )
            if remaining == 0 or deadline_exhausted:
                break
            if round_index + 1 < max(1, int(max_rounds)):
                delay_s = min(
                    8.0,
                    max(0.0, float(initial_backoff_s)) * (2.0**round_index),
                    max(0.0, deadline - time.monotonic()),
                )
                if delay_s > 0.0:
                    time.sleep(delay_s)

        remaining_records = [
            item
            for item in candidates
            if item.get("billing_amount_microusd") is None
        ]
        return {
            "status": "complete" if not remaining_records else "incomplete",
            "initial_pending": initial_pending,
            "resolved": initial_pending - len(remaining_records),
            "remaining": len(remaining_records),
            "rounds": rounds,
            "total_timeout_s": float(total_timeout_s),
            "deadline_exhausted": deadline_exhausted,
            "unresolved_request_ids": [
                str(item.get("request_id", "")) for item in remaining_records
            ],
        }


def _validate_base_url(value: str) -> None:
    """Reject absent/unsafe configuration without reflecting private values."""
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment
                 and not any(c.isspace() for c in value))
        # HTTP is permitted only for a local development service.
        valid = valid and (parsed.scheme == "https" or parsed.hostname in {"localhost", "127.0.0.1", "::1"})
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ProviderFatalError("configure a valid HTTPS base_url or COEVOLVE_BASE_URL (HTTP allowed only on loopback)")


def _completion_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError("chat completion response is not a JSON object")
    choices = payload.get("choices", ())
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat completion response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("chat completion choice is not an object")
    message = first.get("message", {})
    if not isinstance(message, dict):
        raise ValueError("chat completion choice has no message object")
    content = message.get("content", "")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {None, "text", "output_text"}
        ).strip()
    else:
        text = ""
    if not text:
        raise ValueError("chat completion response contains no text")
    return text


def _safe_schema_name(value: str) -> str:
    name = "".join(character if character.isalnum() else "_" for character in value)
    return (name.strip("_") or "role_output")[:64]


def _safe_http_error_body(exc: error.HTTPError, *, api_key: str) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception:  # noqa: BLE001 - error reporting must not mask the request failure
        body = ""
    if api_key:
        body = body.replace(api_key, "<redacted>")
    return body or exc.reason or "request rejected"


def _response_headers(response: object) -> dict[str, str]:
    headers = getattr(response, "headers", None)
    if headers is None:
        return {}
    try:
        return {str(key).lower(): str(value) for key, value in headers.items()}
    except (AttributeError, TypeError, ValueError):
        return {}


def _fetch_openai_compatible_request_usage(
    *,
    base_url: str,
    api_key: str,
    request_id: str,
    timeout_s: float,
    usage_path_template: str = "",
    attempts: int = 3,
    initial_backoff_s: float = 0.25,
) -> dict[str, Any]:
    if not request_id or not usage_path_template:
        return {"_lookup_status": "not_attempted"}
    _validate_base_url(base_url)
    parsed = urlsplit(base_url)
    # Restrict the template to one absolute path on the configured origin.
    if (not usage_path_template.startswith("/")
            or usage_path_template.startswith("//")
            or any(c in usage_path_template for c in "\\?#\r\n")
            or usage_path_template.count("{request_id}") != 1
            or "{" in usage_path_template.replace("{request_id}", "")
            or "}" in usage_path_template.replace("{request_id}", "")):
        raise ProviderFatalError("usage_path_template must be a same-origin path with one {request_id}")
    endpoint = f"{parsed.scheme}://{parsed.netloc}" + usage_path_template.replace(
        "{request_id}", quote(request_id, safe="")
    )
    last_error = ""
    attempt_count = max(1, int(attempts))
    for lookup_attempt in range(attempt_count):
        usage_request = request.Request(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "praxis-coop/0.1",
            },
            method="GET",
        )
        try:
            with request.urlopen(
                usage_request,
                timeout=max(1.0, float(timeout_s)),
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict):
                return {**payload, "_lookup_status": "ok"}
            last_error = "usage endpoint returned a non-object response"
        except error.HTTPError as exc:
            last_error = (
                f"HTTP {int(exc.code)}: {_safe_http_error_body(exc, api_key=api_key)}"
            )
            if int(exc.code) not in {404, 409, 429, 500, 502, 503, 504}:
                break
        except (
            _RequestWallClockTimeout,
            error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if lookup_attempt + 1 < attempt_count:
            time.sleep(max(0.0, float(initial_backoff_s)) * (lookup_attempt + 1))
    if api_key:
        last_error = last_error.replace(api_key, "<redacted>")
    return {
        "_lookup_status": "failed",
        "_lookup_error": last_error,
    }


def _apply_openai_compatible_request_usage(
    record: dict[str, Any],
    request_usage: dict[str, Any],
) -> None:
    """Merge one deferred usage lookup into its existing secret-free call record."""

    record["billing_lookup_status"] = str(
        request_usage.get("_lookup_status", "not_attempted")
    )
    record["billing_lookup_error"] = str(
        request_usage.get("_lookup_error", "")
    )
    if request_usage.get("_lookup_status") != "ok":
        return
    dimensions = _usage_dimensions(request_usage)
    billing = request_usage.get("billing", {})
    billing = billing if isinstance(billing, dict) else {}
    routing = request_usage.get("routing", {})
    routing = routing if isinstance(routing, dict) else {}
    provider_usage = request_usage.get("provider_usage", {})
    provider_usage = provider_usage if isinstance(provider_usage, dict) else {}
    amount_microusd = _optional_int(billing.get("amount_microusd"))
    record.update(
        {
            "billing_amount_microusd": amount_microusd,
            "billing_amount_usd": (
                amount_microusd / 1_000_000.0
                if amount_microusd is not None
                else None
            ),
            "billing_currency": str(billing.get("currency", "")),
            "billing_pricing_version": billing.get("pricing_version"),
            "billing_line_items": _json_list(billing.get("line_items")),
            "billing_missing_price_dimensions": _string_list(
                billing.get("missing_price_dimensions")
            ),
            "provider_family": str(provider_usage.get("family", "")),
            "routing": routing,
        }
    )
    resolved_model = str(routing.get("resolved_model", "")).strip()
    if resolved_model:
        record["resolved_model"] = resolved_model
    for record_key, dimension_key in (
        ("text_input_tokens", "input"),
        ("image_input_tokens", "image_input"),
        ("audio_input_tokens", "audio_input"),
    ):
        if int(record.get(record_key, 0) or 0) == 0 and dimension_key in dimensions:
            record[record_key] = int(dimensions[dimension_key])


def _usage_dimensions(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    dimensions = usage.get("dimensions", ())
    if not isinstance(dimensions, list):
        return {}
    result: dict[str, int] = {}
    for item in dimensions:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        result[name] = int(item.get("quantity", 0) or 0)
    return result


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
