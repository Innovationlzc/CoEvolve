"""Strict JSON I/O for the final Table-1 protocol lock."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    MethodDecisionPromptPin,
    MethodVariant,
    ProtocolLock,
    Table1SourceCohort,
)
from .full_method_compatibility import (
    FullMethodCompatibilityPin,
    full_method_compatibility_pin_from_mapping,
)
from .serialization import strict_json_loads, to_primitive


LOCK_KIND = "coevolve_post_table1_protocol_lock"
LOCK_SCHEMA_VERSION = 5
_LOCK_FIELDS = tuple(item.name for item in fields(ProtocolLock))
_SOURCE_FIELDS = tuple(item.name for item in fields(Table1SourceCohort))
_FULL_METHOD_FIELDS = tuple(item.name for item in fields(FullMethodCompatibilityPin))


def protocol_lock_from_mapping(value: Mapping[str, Any]) -> ProtocolLock:
    """Parse a lock without accepting misspelled or silently ignored fields."""

    payload = dict(value)
    metadata = {"kind", "schema_version"}.intersection(payload)
    if metadata != {"kind", "schema_version"}:
        raise ValueError("protocol lock requires kind and schema_version")
    if payload.pop("kind") != LOCK_KIND:
        raise ValueError("protocol lock has the wrong document kind")
    schema_version = payload.pop("schema_version")
    if schema_version != LOCK_SCHEMA_VERSION:
        raise ValueError(
            "unsupported protocol lock schema_version; version 5 is required "
            "because the byte-recomputable Full-method compatibility identity "
            "cannot be inferred from an older lock"
        )
    expected = set(_LOCK_FIELDS)
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"protocol lock fields differ: missing={missing}, unknown={unknown}"
        )
    sources = {}
    for name in ("table1_gpt_source", "table1_gemini_source"):
        source = payload.pop(name)
        if not isinstance(source, Mapping):
            raise ValueError(f"protocol lock {name} must be an object")
        source_payload = dict(source)
        missing_source = sorted(set(_SOURCE_FIELDS) - set(source_payload))
        unknown_source = sorted(set(source_payload) - set(_SOURCE_FIELDS))
        if missing_source or unknown_source:
            raise ValueError(
                f"protocol lock {name} fields differ: "
                f"missing={missing_source}, unknown={unknown_source}"
            )
        seeds = source_payload["seeds"]
        if not isinstance(seeds, list):
            raise ValueError(f"protocol lock {name}.seeds must be a JSON array")
        source_payload["seeds"] = tuple(seeds)
        sources[name] = Table1SourceCohort(**source_payload)
    prompt_payload = payload.pop("method_decision_prompt_pins")
    if not isinstance(prompt_payload, list):
        raise ValueError(
            "protocol lock method_decision_prompt_pins must be a JSON array"
        )
    prompt_pins = []
    for index, item in enumerate(prompt_payload):
        if not isinstance(item, Mapping) or set(item) != {
            "method_variant",
            "decision_prompt_fingerprint",
        }:
            raise ValueError(
                "protocol lock method_decision_prompt_pins entries must contain "
                "exactly method_variant and decision_prompt_fingerprint"
            )
        try:
            prompt_pins.append(
                MethodDecisionPromptPin(
                    method_variant=MethodVariant(item["method_variant"]),
                    decision_prompt_fingerprint=item[
                        "decision_prompt_fingerprint"
                    ],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid method_decision_prompt_pins[{index}]: {exc}"
            ) from exc
    full_method_payload = payload.pop("full_method_compatibility")
    if not isinstance(full_method_payload, Mapping):
        raise ValueError(
            "protocol lock full_method_compatibility must be an object"
        )
    missing_full = sorted(set(_FULL_METHOD_FIELDS) - set(full_method_payload))
    unknown_full = sorted(set(full_method_payload) - set(_FULL_METHOD_FIELDS))
    if missing_full or unknown_full:
        raise ValueError(
            "protocol lock full_method_compatibility fields differ: "
            f"missing={missing_full}, unknown={unknown_full}"
        )
    try:
        full_method_compatibility = full_method_compatibility_pin_from_mapping(
            full_method_payload
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid full_method_compatibility: {exc}") from exc
    if not all(isinstance(payload[name], str) for name in payload):
        raise ValueError("non-source protocol lock values must be strings")
    return ProtocolLock(
        **payload,
        **sources,
        method_decision_prompt_pins=tuple(prompt_pins),
        full_method_compatibility=full_method_compatibility,
    )


def load_protocol_lock(path: str | Path) -> ProtocolLock:
    source = Path(path)
    try:
        value = strict_json_loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read protocol lock {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("protocol lock root must be an object")
    return protocol_lock_from_mapping(value)


def protocol_lock_document(lock: ProtocolLock) -> dict[str, Any]:
    return {
        "kind": LOCK_KIND,
        "schema_version": LOCK_SCHEMA_VERSION,
        **{name: to_primitive(getattr(lock, name)) for name in _LOCK_FIELDS},
    }


__all__ = [
    "LOCK_KIND",
    "LOCK_SCHEMA_VERSION",
    "load_protocol_lock",
    "protocol_lock_document",
    "protocol_lock_from_mapping",
]
