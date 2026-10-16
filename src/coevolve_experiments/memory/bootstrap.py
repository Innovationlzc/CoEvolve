"""Fresh target-memory bootstrap for Table 4 transfer conditions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..contracts import StrategyBankPin
from ..serialization import sha256_json
from .online_memory import (
    empty_action_exemplar_memory,
    empty_online_strategy_overlay,
)
from .schema import (
    FROZEN_STRATEGY_BANK_POLICY,
    MemoryBootstrapError,
    empty_strategy_bank,
    semantic_sha256,
)
from .validation import load_strategy_bank, validate_strategy_bank
from .provenance import (
    StrategyBankProvenanceError,
    validate_strategy_bank_provenance,
)


class TransferCondition(str, Enum):
    EMPTY = "Empty"
    GPT_TO_GPT = "GPT→GPT"
    GEMINI_TO_GPT = "Gemini→GPT"


TRANSFER_BOOTSTRAP_SNAPSHOT_KIND = "coevolve_transfer_memory_bootstrap_snapshot"
TRANSFER_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION = 1
TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME = "memory_bootstrap_snapshot.json"
TRANSFER_BOOTSTRAP_INITIAL_FILES = (
    "seed_strategy_bank.json",
    "online_strategy_overlay.json",
    "action_exemplars.json",
    "accepted_episode_manifest.json",
    "failure_cache.json",
    "memory_bootstrap_audit.json",
)
TRANSFER_BOOTSTRAP_MUTABLE_FILES = frozenset(
    {
        "online_strategy_overlay.json",
        "action_exemplars.json",
        "accepted_episode_manifest.json",
        "failure_cache.json",
    }
)
TRANSFER_BOOTSTRAP_IMMUTABLE_FILES = frozenset(
    set(TRANSFER_BOOTSTRAP_INITIAL_FILES) - TRANSFER_BOOTSTRAP_MUTABLE_FILES
)


@dataclass(frozen=True)
class TransferMemoryBootstrap:
    condition: TransferCondition
    target_model: str
    seed_bank_path: Path
    online_overlay_path: Path
    action_exemplars_path: Path
    accepted_manifest_path: Path
    failure_cache_path: Path
    audit_path: Path
    snapshot_path: Path
    seed_bank_sha256: str
    snapshot_sha256: str


@dataclass(frozen=True)
class TransferBootstrapSnapshot:
    """Verified initial bytes retained across one formal target run."""

    path: Path
    file_sha256: str
    snapshot_fingerprint: str
    initial_file_sha256s: tuple[tuple[str, str], ...]
    formal_binding_sha256: str
    seed_bank_sha256: str


def bootstrap_transfer_memory(
    target_dir: str | Path,
    condition: TransferCondition | str,
    *,
    seed_bank: object | None = None,
    seed_provenance: object | None = None,
    provenance: object | None = None,
    source_model: str | None = None,
    target_model: str = "GPT",
    formal_binding: Mapping[str, Any] | None = None,
    expected_strategy_bank_pin: StrategyBankPin | None = None,
) -> TransferMemoryBootstrap:
    """Low-level structural bootstrap; it does not authorize a formal run.

    A target can append only to ``online_strategy_overlay.json``.  The seed is
    normalized, hashed, and made read-only.  Action exemplars, the accepted
    episode lineage, and the local failure cache always start empty.

    Formal launchers must call :func:`prepare_transfer_from_plan`, which adds a
    suite-plan/RunSpec binding.  The file-backed adapter refuses to open a
    bootstrap without that binding.
    """

    parsed_condition = _parse_condition(condition)
    if provenance is not None:
        if seed_provenance is not None and provenance is not seed_provenance:
            raise MemoryBootstrapError("provenance and seed_provenance disagree")
        seed_provenance = provenance
    target_family = _model_family(target_model)
    if target_family != "gpt":
        raise MemoryBootstrapError(
            f"Table 4 target backbone must be GPT, got {target_model!r}"
        )

    provenance_payload, provenance_path = _load_optional_json(seed_provenance)
    provenance_source_model = str(
        provenance_payload.get("source_model", "")
    ).strip()

    if parsed_condition is TransferCondition.EMPTY:
        if seed_bank is None:
            bank = empty_strategy_bank()
        else:
            bank = load_strategy_bank(seed_bank)
            validate_strategy_bank(bank)
            if bank["records"]:
                raise MemoryBootstrapError("Empty condition cannot preload strategy records")
        expected_source_family = None
    else:
        if seed_bank is None:
            raise MemoryBootstrapError(f"{parsed_condition.value} requires a seed bank")
        if not provenance_payload:
            raise MemoryBootstrapError(
                f"{parsed_condition.value} requires the frozen provenance sidecar"
            )
        if not provenance_source_model:
            raise MemoryBootstrapError("seed provenance source model is missing")
        if source_model is not None and source_model.strip() != provenance_source_model:
            raise MemoryBootstrapError(
                "explicit source_model does not match seed provenance: "
                f"{source_model!r} != {provenance_source_model!r}"
            )
        source_model = provenance_source_model
        bank = load_strategy_bank(seed_bank)
        validate_strategy_bank(bank)
        if not bank["records"]:
            raise MemoryBootstrapError(f"{parsed_condition.value} requires a non-empty seed bank")
        expected_source_family = (
            "gpt"
            if parsed_condition is TransferCondition.GPT_TO_GPT
            else "gemini"
        )
        if _model_family(source_model or "") != expected_source_family:
            raise MemoryBootstrapError(
                f"{parsed_condition.value} requires {expected_source_family} source provenance; "
                f"got {source_model!r}"
            )
        if provenance_payload.get("kind") != "strategy_transfer_provenance":
            raise MemoryBootstrapError("seed provenance has the wrong document kind")
        if provenance_payload.get("source_scene") != "base":
            raise MemoryBootstrapError("seed provenance must come from the base scene")
        if not str(provenance_payload.get("source_protocol", "")).strip():
            raise MemoryBootstrapError("seed provenance source protocol is missing")
        if provenance_payload.get("policy") != FROZEN_STRATEGY_BANK_POLICY.as_dict():
            raise MemoryBootstrapError("seed provenance does not use the frozen bank policy")

    seed_digest = semantic_sha256(bank)
    provenance_digest = ""
    if provenance_payload:
        recorded_digest = str(provenance_payload.get("bank_sha256", "")).strip()
        if parsed_condition is not TransferCondition.EMPTY and not recorded_digest:
            raise MemoryBootstrapError("seed provenance bank_sha256 is missing")
        if recorded_digest and recorded_digest != seed_digest:
            raise MemoryBootstrapError(
                "seed bank hash does not match its provenance sidecar"
            )
        provenance_digest = semantic_sha256(provenance_payload)

    if expected_strategy_bank_pin is not None:
        if parsed_condition is TransferCondition.EMPTY:
            raise MemoryBootstrapError(
                "Empty condition cannot carry a strategy-bank source pin"
            )
        if seed_digest != expected_strategy_bank_pin.sha256:
            raise MemoryBootstrapError(
                "seed bank hash differs from the plan-derived StrategyBankPin"
            )
        try:
            validate_strategy_bank_provenance(
                provenance_payload,
                strategy_bank_pin=expected_strategy_bank_pin,
                expected_bank_sha256=seed_digest,
            )
        except StrategyBankProvenanceError as exc:
            raise MemoryBootstrapError(str(exc)) from exc
    elif formal_binding is not None and parsed_condition is not TransferCondition.EMPTY:
        raise MemoryBootstrapError(
            "formal transfer bootstrap requires a plan-derived StrategyBankPin"
        )

    root = Path(target_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_path = root / "seed_strategy_bank.json"
    overlay_path = root / "online_strategy_overlay.json"
    action_path = root / "action_exemplars.json"
    accepted_path = root / "accepted_episode_manifest.json"
    failure_path = root / "failure_cache.json"
    audit_path = root / "memory_bootstrap_audit.json"
    snapshot_path = root / TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME

    _write_fresh_json(seed_path, bank)
    # Read-only permissions are an additional guard.  The audit hash remains
    # the authoritative cross-platform immutability check.
    seed_path.chmod(0o444)
    empty_overlay = empty_online_strategy_overlay()
    empty_actions = empty_action_exemplar_memory()
    _write_fresh_json(overlay_path, empty_overlay)
    _write_fresh_json(action_path, empty_actions)
    _write_fresh_json(
        accepted_path,
        {
            "kind": "accepted_checkpoint_episode",
            "schema_version": 1,
            "initial_checkpoint": "c0",
            "latest_checkpoint": "c0",
            "initial_state": {},
            "steps": [],
        },
    )
    _write_fresh_json(
        failure_path,
        {
            "kind": "run_local_failure_cache",
            "schema_version": 1,
            "failures": [],
        },
    )

    audit = {
        "kind": "table4_memory_bootstrap_audit",
        "schema_version": 1,
        "condition": parsed_condition.value,
        "source_model": source_model or "",
        "source_model_family": expected_source_family or "none",
        "target_model": target_model,
        "target_model_family": target_family,
        "seed": {
            "path": seed_path.name,
            "sha256": seed_digest,
            "record_count": len(bank["records"]),
            "immutable": True,
            "mode": "0444",
            "provenance_path": provenance_path,
            "provenance_sha256": provenance_digest,
        },
        "online_overlay": {
            "path": overlay_path.name,
            "sha256_at_bootstrap": semantic_sha256(empty_overlay),
            "record_count_at_bootstrap": 0,
            "separate_from_seed": True,
            "target_run_local_updates_allowed": True,
        },
        "fresh_target_state": {
            "physical_state": "s0",
            "checkpoint": "c0",
            "action_exemplars_empty": True,
            "accepted_manifest_steps_empty": True,
            "failure_cache_empty": True,
        },
    }
    if formal_binding is not None:
        audit["formal_binding"] = _validate_formal_binding(formal_binding)
    _write_fresh_json(audit_path, audit)

    snapshot = _bootstrap_snapshot_document(
        root,
        condition=parsed_condition,
        target_model=target_model,
        seed_bank_sha256=seed_digest,
        formal_binding=audit.get("formal_binding"),
    )
    _write_fresh_json(snapshot_path, snapshot)
    # The snapshot, audit, and seed form the immutable pre-run identity.  The
    # other four files are intentionally mutable only after physical dispatch.
    audit_path.chmod(0o444)
    snapshot_path.chmod(0o444)
    verified_snapshot = validate_transfer_bootstrap_snapshot(
        root,
        require_initial_state=True,
    )

    return TransferMemoryBootstrap(
        condition=parsed_condition,
        target_model=target_model,
        seed_bank_path=seed_path,
        online_overlay_path=overlay_path,
        action_exemplars_path=action_path,
        accepted_manifest_path=accepted_path,
        failure_cache_path=failure_path,
        audit_path=audit_path,
        snapshot_path=snapshot_path,
        seed_bank_sha256=seed_digest,
        snapshot_sha256=verified_snapshot.file_sha256,
    )


def verify_seed_bank_immutable(bootstrap: TransferMemoryBootstrap) -> bool:
    """Re-hash a bootstrapped seed before/after every target run."""

    try:
        bank = load_strategy_bank(bootstrap.seed_bank_path)
    except Exception:
        return False
    return semantic_sha256(bank) == bootstrap.seed_bank_sha256


def validate_transfer_bootstrap_snapshot(
    memory_dir: str | Path,
    *,
    require_initial_state: bool,
    expected_formal_binding: Mapping[str, Any] | None = None,
    expected_condition: TransferCondition | str | None = None,
    expected_target_model: str | None = None,
    expected_seed_bank_sha256: str | None = None,
) -> TransferBootstrapSnapshot:
    """Validate one immutable bootstrap identity.

    Before launch, all six initialized files must match the recorded bytes.
    After launch, only the four explicitly mutable files may differ; the seed,
    bootstrap audit, and snapshot remain byte-pinned.
    """

    root = _strict_memory_root(memory_dir)
    snapshot_path = root / TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME
    payload, raw = _read_strict_json_object(snapshot_path, "memory bootstrap snapshot")
    expected_fields = {
        "kind",
        "schema_version",
        "condition",
        "target_model",
        "seed_bank_sha256",
        "formal_binding_sha256",
        "initial_files",
        "snapshot_fingerprint",
    }
    if set(payload) != expected_fields:
        raise MemoryBootstrapError(
            "memory bootstrap snapshot fields differ: "
            f"missing={sorted(expected_fields - set(payload))}, "
            f"unknown={sorted(set(payload) - expected_fields)}"
        )
    if payload.get("kind") != TRANSFER_BOOTSTRAP_SNAPSHOT_KIND:
        raise MemoryBootstrapError("memory bootstrap snapshot has the wrong kind")
    if payload.get("schema_version") != TRANSFER_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION:
        raise MemoryBootstrapError(
            "memory bootstrap snapshot has an unsupported schema_version"
        )
    fingerprint_payload = dict(payload)
    actual_fingerprint = fingerprint_payload.pop("snapshot_fingerprint", None)
    if actual_fingerprint != sha256_json(fingerprint_payload):
        raise MemoryBootstrapError("memory bootstrap snapshot fingerprint differs")
    if not _is_sha256(payload.get("seed_bank_sha256")):
        raise MemoryBootstrapError("snapshot seed_bank_sha256 is invalid")
    if not _is_sha256(payload.get("formal_binding_sha256")):
        raise MemoryBootstrapError("snapshot formal_binding_sha256 is invalid")

    if expected_condition is not None:
        parsed_condition = _parse_condition(expected_condition)
        if payload.get("condition") != parsed_condition.value:
            raise MemoryBootstrapError("snapshot condition differs from the selected run")
    if expected_target_model is not None and payload.get("target_model") != expected_target_model:
        raise MemoryBootstrapError("snapshot target_model differs from the selected run")
    if (
        expected_seed_bank_sha256 is not None
        and payload.get("seed_bank_sha256") != expected_seed_bank_sha256
    ):
        raise MemoryBootstrapError(
            "snapshot seed bank differs from the selected RunSpec"
        )

    initial_files = payload.get("initial_files")
    if not isinstance(initial_files, list):
        raise MemoryBootstrapError("snapshot initial_files must be an array")
    expected_names = list(TRANSFER_BOOTSTRAP_INITIAL_FILES)
    names: list[str] = []
    hashes: list[tuple[str, str]] = []
    for index, item in enumerate(initial_files):
        if not isinstance(item, Mapping) or set(item) != {
            "filename",
            "sha256",
            "mutable_after_dispatch",
        }:
            raise MemoryBootstrapError(
                f"snapshot initial_files[{index}] does not use the exact schema"
            )
        name = item.get("filename")
        digest = item.get("sha256")
        if not isinstance(name, str) or not name:
            raise MemoryBootstrapError(
                f"snapshot initial_files[{index}].filename is invalid"
            )
        if not _is_sha256(digest):
            raise MemoryBootstrapError(
                f"snapshot initial_files[{index}].sha256 is invalid"
            )
        mutable = item.get("mutable_after_dispatch")
        if mutable is not (name in TRANSFER_BOOTSTRAP_MUTABLE_FILES):
            raise MemoryBootstrapError(
                f"snapshot mutability policy differs for {name}"
            )
        names.append(name)
        hashes.append((name, str(digest)))
    if names != expected_names:
        raise MemoryBootstrapError(
            "snapshot initial file order/set differs from the frozen bootstrap"
        )

    audit_path = root / "memory_bootstrap_audit.json"
    audit, _audit_raw = _read_strict_json_object(
        audit_path,
        "memory bootstrap audit",
    )
    formal_binding = audit.get("formal_binding")
    binding_payload = dict(formal_binding) if isinstance(formal_binding, Mapping) else {}
    if sha256_json(binding_payload) != payload.get("formal_binding_sha256"):
        raise MemoryBootstrapError(
            "snapshot formal binding differs from the bootstrap audit"
        )
    if expected_formal_binding is not None and binding_payload != dict(
        expected_formal_binding
    ):
        raise MemoryBootstrapError(
            "bootstrap formal binding differs from the selected plan/run"
        )

    seed_path = root / "seed_strategy_bank.json"
    try:
        seed = load_strategy_bank(seed_path)
        validate_strategy_bank(seed)
    except (OSError, TypeError, ValueError) as exc:
        raise MemoryBootstrapError(f"snapshot seed bank is invalid: {exc}") from exc
    if semantic_sha256(seed) != payload.get("seed_bank_sha256"):
        raise MemoryBootstrapError(
            "snapshot semantic seed hash differs from the current seed"
        )

    recorded = dict(hashes)
    files_to_check = (
        TRANSFER_BOOTSTRAP_INITIAL_FILES
        if require_initial_state
        else tuple(
            name
            for name in TRANSFER_BOOTSTRAP_INITIAL_FILES
            if name in TRANSFER_BOOTSTRAP_IMMUTABLE_FILES
        )
    )
    for name in files_to_check:
        path = _strict_memory_file(root, name)
        if _file_sha256(path) != recorded[name]:
            phase = "initial" if require_initial_state else "immutable"
            raise MemoryBootstrapError(
                f"{phase} bootstrap bytes changed for {name}"
            )
    for name in (*TRANSFER_BOOTSTRAP_IMMUTABLE_FILES, TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME):
        path = _strict_memory_file(root, name)
        if path.stat().st_mode & 0o222:
            raise MemoryBootstrapError(f"immutable bootstrap file became writable: {name}")

    return TransferBootstrapSnapshot(
        path=snapshot_path,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        snapshot_fingerprint=str(actual_fingerprint),
        initial_file_sha256s=tuple(hashes),
        formal_binding_sha256=str(payload["formal_binding_sha256"]),
        seed_bank_sha256=str(payload["seed_bank_sha256"]),
    )


def _parse_condition(value: TransferCondition | str) -> TransferCondition:
    if isinstance(value, TransferCondition):
        return value
    normalized = (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("->", "→")
        .replace("_to_", "→")
        .replace("-to-", "→")
    )
    aliases = {
        "empty": TransferCondition.EMPTY,
        "gpt→gpt": TransferCondition.GPT_TO_GPT,
        "gpttogpt": TransferCondition.GPT_TO_GPT,
        "gemini→gpt": TransferCondition.GEMINI_TO_GPT,
        "geminitogpt": TransferCondition.GEMINI_TO_GPT,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise MemoryBootstrapError(f"unsupported Table 4 condition: {value!r}") from exc


def _model_family(value: str) -> str:
    normalized = value.strip().lower()
    if "gemini" in normalized:
        return "gemini"
    if "gpt" in normalized or "openai" in normalized:
        return "gpt"
    return normalized


def _load_optional_json(value: object | None) -> tuple[dict[str, Any], str]:
    if value is None:
        return {}, ""
    if hasattr(value, "provenance"):
        payload = getattr(value, "provenance")
        path_value = getattr(value, "provenance_path", None)
        return dict(payload), str(Path(path_value).resolve()) if path_value else "<in-memory>"
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            payload, _raw = _read_strict_json_object(path, "seed provenance")
        except (OSError, ValueError) as exc:
            raise MemoryBootstrapError(f"cannot read seed provenance {path}: {exc}") from exc
        return dict(payload), str(path.resolve())
    if not isinstance(value, Mapping):
        raise MemoryBootstrapError("seed provenance must be a path or object")
    return dict(value), "<in-memory>"


def _validate_formal_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "run_id",
        "protocol_id",
        "condition_fingerprint",
        "strategy_bank_sha256",
        "source_model",
        "target_model",
        "plan_path",
        "plan_sha256",
    }
    fingerprint_pins = {
        "observation_contract_fingerprint",
        "task_instruction_fingerprint",
        "shared_prompt_contract_fingerprint",
        "decision_prompt_fingerprint",
        "provider_route_fingerprint",
        "source_provider_route_fingerprint",
        "skill_api_fingerprint",
        "full_method_compatibility_fingerprint",
    }
    route_id_pins = {"provider_route_id", "source_provider_route_id"}
    fairness_pins = fingerprint_pins | route_id_pins
    payload = dict(value)
    present_fairness = set(payload).intersection(fairness_pins)
    allowed = required.union(fairness_pins)
    if not required.issubset(payload) or set(payload) - allowed:
        raise MemoryBootstrapError(
            "formal binding fields differ: "
            f"missing={sorted(required - set(payload))}, "
            f"unknown={sorted(set(payload) - allowed)}"
        )
    if present_fairness and present_fairness != fairness_pins:
        raise MemoryBootstrapError(
            "formal binding fairness pins must be provided as one complete set"
        )
    if not all(isinstance(payload[name], str) and payload[name].strip() for name in required):
        raise MemoryBootstrapError("formal binding values must be non-empty strings")
    if any(
        not isinstance(payload[name], str) or not payload[name].strip()
        for name in present_fairness.intersection(route_id_pins)
    ):
        raise MemoryBootstrapError(
            "formal binding provider route IDs must be non-empty strings"
        )
    if not Path(payload["plan_path"]).is_absolute():
        raise MemoryBootstrapError("formal binding plan_path must be absolute")
    for name in (
        "condition_fingerprint",
        "strategy_bank_sha256",
        "plan_sha256",
        *sorted(present_fairness.intersection(fingerprint_pins)),
    ):
        digest = payload[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise MemoryBootstrapError(f"formal binding {name} must be lowercase SHA-256")
    return payload


def _bootstrap_snapshot_document(
    root: Path,
    *,
    condition: TransferCondition,
    target_model: str,
    seed_bank_sha256: str,
    formal_binding: object,
) -> dict[str, Any]:
    binding = dict(formal_binding) if isinstance(formal_binding, Mapping) else {}
    payload = {
        "kind": TRANSFER_BOOTSTRAP_SNAPSHOT_KIND,
        "schema_version": TRANSFER_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION,
        "condition": condition.value,
        "target_model": target_model,
        "seed_bank_sha256": seed_bank_sha256,
        "formal_binding_sha256": sha256_json(binding),
        "initial_files": [
            {
                "filename": name,
                "sha256": _file_sha256(_strict_memory_file(root, name)),
                "mutable_after_dispatch": name in TRANSFER_BOOTSTRAP_MUTABLE_FILES,
            }
            for name in TRANSFER_BOOTSTRAP_INITIAL_FILES
        ],
    }
    return {**payload, "snapshot_fingerprint": sha256_json(payload)}


def _strict_memory_root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = path.absolute()
    if path.is_symlink() or not path.is_dir():
        raise MemoryBootstrapError(
            "transfer memory directory must be a non-symlink existing directory"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise MemoryBootstrapError(
            "transfer memory directory must be normalized and contain no symlink components"
        )
    return resolved


def _strict_memory_file(root: Path, filename: str) -> Path:
    if filename not in {
        *TRANSFER_BOOTSTRAP_INITIAL_FILES,
        TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME,
    }:
        raise MemoryBootstrapError(f"unregistered transfer memory file: {filename}")
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise MemoryBootstrapError(
            f"transfer memory file must be a regular non-symlink: {filename}"
        )
    resolved = path.resolve(strict=True)
    if resolved.parent != root or resolved != path:
        raise MemoryBootstrapError(
            f"transfer memory file escapes its bootstrap directory: {filename}"
        )
    return resolved


def _read_strict_json_object(
    path: Path,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise MemoryBootstrapError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise MemoryBootstrapError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise MemoryBootstrapError(f"{label} root must be an object")
    return value, raw


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = child
    return value


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise MemoryBootstrapError(f"cannot hash bootstrap file {path}: {exc}") from exc


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_fresh_json(path: Path, value: object) -> None:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MemoryBootstrapError(f"cannot inspect target memory {path}: {exc}") from exc
        if existing == rendered:
            return
        raise MemoryBootstrapError(
            f"target memory is not fresh; refusing to overwrite {path}"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


__all__ = [
    "TRANSFER_BOOTSTRAP_IMMUTABLE_FILES",
    "TRANSFER_BOOTSTRAP_INITIAL_FILES",
    "TRANSFER_BOOTSTRAP_MUTABLE_FILES",
    "TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME",
    "TRANSFER_BOOTSTRAP_SNAPSHOT_KIND",
    "TRANSFER_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION",
    "TransferBootstrapSnapshot",
    "TransferCondition",
    "TransferMemoryBootstrap",
    "bootstrap_transfer_memory",
    "validate_transfer_bootstrap_snapshot",
    "verify_seed_bank_immutable",
]
