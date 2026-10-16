"""Recursive leak detection and deterministic text sanitization for Table 4."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, order=True)
class MemoryLeak:
    """One inadmissible key or value found in transferable memory."""

    path: str
    category: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "category": self.category,
            "detail": self.detail,
        }


_KEY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "world_coordinate",
        re.compile(
            r"(?:^|_)(?:world_?)?(?:x|y|xy|xyz)(?:_|$)|"
            r"^(?:position|position_?xyz|location|translation)(?:_|$)"
        ),
    ),
    ("pose_or_yaw", re.compile(r"(?:^|_)(?:pose|yaw|quaternion)(?:_|$)")),
    ("checkpoint_reference", re.compile(r"checkpoint")),
    ("waypoint_reference", re.compile(r"waypoint")),
    ("scene_topology", re.compile(r"(?:^|_)(?:route|gate|segment|scene)(?:_|$)")),
    (
        "specific_motion_value",
        re.compile(
            r"(?:^|_)(?:distance|speed|velocity|angle|duration|elapsed|time|"
            r"timeout|rate)(?:_|$)"
        ),
    ),
    (
        "execution_reference",
        re.compile(r"(?:^|_)(?:script|attempt|run|rollout|artifact)(?:_|$)"),
    ),
    ("manifest_reference", re.compile(r"manifest")),
    ("failure_cache", re.compile(r"failure_?cache|cached_?failure")),
    ("action_exemplar", re.compile(r"action_?exemplar|exemplar_?action")),
    (
        "absolute_lateral_direction",
        re.compile(r"(?:^|_)(?:left|right)(?:_|$)"),
    ),
    (
        "absolute_rotation_direction",
        re.compile(
            r"(?:^|_)(?:clockwise|counter_?clockwise|anti_?clockwise|cw|ccw)"
            r"(?:_|$)"
        ),
    ),
    (
        "absolute_cardinal_direction",
        re.compile(
            r"(?:^|_)(?:north|south|east|west|"
            r"north_?east|north_?west|south_?east|south_?west|"
            r"n|s|e|w|ne|nw|se|sw)(?:_|$)"
        ),
    ),
    (
        "named_scene_landmark",
        re.compile(
            r"(?:^|_)(?:red|orange|yellow|green|blue|purple|violet|pink|"
            r"cyan|magenta|black|white|gr[ae]y|brown)(?:_[a-z]+){0,2}_"
            r"(?:cylinders?|box(?:es)?|crates?|payloads?|walls?|openings?|"
            r"doors?|doorways?|gates?|"
            r"obstacle|barrier|cone|marker|target|bin|table|platform|pillar|"
            r"post|block|cube|sphere)(?:s)?(?:_|$)"
        ),
    ),
)


_ABSOLUTE_LATERAL_PATTERN = re.compile(
    r"\b(?:left|right)(?:ward|wards|most|hand(?:ed)?)?\b",
    re.I,
)

_ABSOLUTE_ROTATION_PATTERN = re.compile(
    r"\b(?:counter[\s_-]*clockwise|anti[\s_-]*clockwise|clockwise|ccw|cw)\b",
    re.I,
)

_ABSOLUTE_CARDINAL_PATTERN = re.compile(
    r"\b(?:(?:north|south)(?:[\s_-]?(?:east|west))?|east|west)"
    r"(?:ern|ward|wards)?\b|"
    r"\b(?-i:N|S|E|W|NE|NW|SE|SW)\b",
    re.I,
)

_NAMED_SCENE_LANDMARK_PATTERN = re.compile(
    r"\b(?:red|orange|yellow|green|blue|purple|violet|pink|cyan|magenta|"
    r"black|white|gr[ae]y|brown)(?:[-\s]+colou?red)?\s+"
    r"(?:(?:small|large|tall|short|inner|outer|concrete|metal|wooden|"
    r"painted|cylindrical|rectangular|round)\s+){0,2}"
    r"(?:cylinders?|box(?:es)?|crates?|payloads?|walls?|openings?|doors?|"
    r"doorways?|gates?|obstacles?|barriers?|cones?|markers?|targets?|bins?|"
    r"tables?|platforms?|pillars?|posts?|blocks?|cubes?|spheres?)\b",
    re.I,
)


_TEXT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "world_coordinate",
        re.compile(
            r"\bworld[\s_-]*(?:x|y|xy|coordinate(?:s)?)\b|"
            r"\b(?:x|y)\s*[:=]\s*[-+]?\d",
            re.IGNORECASE,
        ),
    ),
    ("pose_or_yaw", re.compile(r"\b(?:checkpoint\s+pose|pose|yaw)\b", re.I)),
    ("checkpoint_reference", re.compile(r"\bcheckpoint(?:s|[_-]\w+)?\b", re.I)),
    ("waypoint_reference", re.compile(r"\bwaypoint(?:s|[_-]\w+)?\b", re.I)),
    (
        "scene_topology",
        re.compile(r"\b(?:route|gate|segment|scene)(?:s|[_-]\w+)?\b", re.I),
    ),
    (
        "execution_reference",
        re.compile(
            r"\b(?:script|attempt|run|rollout|artifact)(?:s|[_-]\w+)?\b",
            re.I,
        ),
    ),
    ("manifest_reference", re.compile(r"\bmanifest(?:s|[_-]\w+)?\b", re.I)),
    (
        "failure_cache",
        re.compile(r"\b(?:failure[\s_-]*cache|cached[\s_-]*failure)\b", re.I),
    ),
    (
        "action_exemplar",
        re.compile(r"\baction[\s_-]*exemplar(?:s)?\b", re.I),
    ),
    ("absolute_lateral_direction", _ABSOLUTE_LATERAL_PATTERN),
    ("absolute_rotation_direction", _ABSOLUTE_ROTATION_PATTERN),
    ("absolute_cardinal_direction", _ABSOLUTE_CARDINAL_PATTERN),
    ("named_scene_landmark", _NAMED_SCENE_LANDMARK_PATTERN),
    (
        "numeric_unit",
        re.compile(
            r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*"
            r"(?:cm/s|m/s|cmps|mps|millimeters?|millimetres?|mm|"
            r"centimeters?|centimetres?|cm|kilometers?|kilometres?|km|"
            r"meters?|metres?|m|degrees?|deg|radians?|rad|°|"
            r"milliseconds?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|hz)"
            r"(?=$|[\s,.;:)\]])",
            re.I,
        ),
    ),
    (
        "coordinate_tuple",
        re.compile(
            r"[\[(]\s*[-+]?\d+(?:\.\d+)?\s*,\s*"
            r"[-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?)?\s*[\])]"
        ),
    ),
    (
        "filesystem_or_url_reference",
        re.compile(
            r"(?:https?://\S+|(?:^|\s)(?:/|~[/\\]|[A-Za-z]:[\\/])\S+|"
            r"\b\S+\.(?:jsonl?|log|py|yaml|yml|pt|ckpt|npz|mp4)\b)",
            re.I,
        ),
    ),
    ("hash_reference", re.compile(r"\b[0-9a-f]{32,64}\b", re.I)),
    (
        "source_identifier",
        re.compile(
            r"\b(?=[A-Za-z][A-Za-z0-9_-]*\b)(?=[A-Za-z0-9_-]*\d)"
            r"[A-Za-z][A-Za-z0-9_-]*\b"
        ),
    ),
    # Transfer records have no legitimate numeric identity or magnitude.  This
    # catches unitless coordinates and source indices after the specific rules.
    ("numeric_literal", re.compile(r"(?<![\w])[-+]?\d+(?:\.\d+)?(?![\w])")),
)


def _normalized_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _key_categories(key: object) -> tuple[str, ...]:
    normalized = _normalized_key(key)
    return tuple(
        category for category, pattern in _KEY_RULES if pattern.search(normalized)
    )


def _key_category(key: object) -> str | None:
    categories = _key_categories(key)
    return categories[0] if categories else None


def _text_findings(text: str, path: str) -> list[MemoryLeak]:
    findings: list[MemoryLeak] = []
    for category, pattern in _TEXT_RULES:
        match = pattern.search(text)
        if match:
            excerpt = re.sub(r"\s+", " ", match.group(0)).strip()
            findings.append(MemoryLeak(path, category, excerpt[:120]))
    return findings


def scan_memory_leaks(value: object) -> list[MemoryLeak]:
    """Recursively scan arbitrary JSON-like data for source-run leakage.

    The scanner intentionally reports both forbidden field names and textual
    references.  It never imports or relies on the deployment validator.
    """

    findings: list[MemoryLeak] = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                child_path = f"{path}.{key}"
                for category in _key_categories(key):
                    findings.append(
                        MemoryLeak(child_path, category, f"forbidden key: {key}")
                    )
                visit(child, child_path)
            return
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(item, str):
            findings.extend(_text_findings(item, path))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            # Schema metadata is not a transfer record.  The validator scans
            # records rather than the root, so any number seen here is a leak.
            findings.append(MemoryLeak(path, "numeric_literal", repr(item)))

    visit(value, "$")
    return sorted(set(findings))


_SANITIZE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Named landmarks must be abstracted before generic words such as "gate"
    # are replaced, otherwise the color token could survive by itself.
    (_NAMED_SCENE_LANDMARK_PATTERN, "environment landmark"),
    (re.compile(r"\bworld[\s_-]*(?:x|y|xy|coordinate(?:s)?)\b", re.I), "global state"),
    (re.compile(r"\bcheckpoint\s+pose\b", re.I), "verified configuration"),
    (re.compile(r"\bpose\b", re.I), "configuration"),
    (re.compile(r"\byaw\b", re.I), "orientation"),
    (re.compile(r"\bcheckpoint(?:s|[_-]\w+)?\b", re.I), "verified state"),
    (re.compile(r"\bwaypoint(?:s|[_-]\w+)?\b", re.I), "local objective"),
    (re.compile(r"\broute(?:s|[_-]\w+)?\b", re.I), "task progression"),
    (re.compile(r"\bgate(?:s|[_-]\w+)?\b", re.I), "transition criterion"),
    (re.compile(r"\bsegment(?:s|[_-]\w+)?\b", re.I), "task phase"),
    (re.compile(r"\bscene(?:s|[_-]\w+)?\b", re.I), "environment"),
    (re.compile(r"\bscript(?:s|[_-]\w+)?\b", re.I), "executable policy"),
    (re.compile(r"\battempt(?:s|[_-]\w+)?\b", re.I), "execution"),
    (re.compile(r"\brun(?:s|[_-]\w+)?\b", re.I), "execution"),
    (re.compile(r"\brollout(?:s|[_-]\w+)?\b", re.I), "execution"),
    (re.compile(r"\bartifact(?:s|[_-]\w+)?\b", re.I), "evidence"),
    (re.compile(r"\bmanifest(?:s|[_-]\w+)?\b", re.I), ""),
    (re.compile(r"\b(?:failure[\s_-]*cache|cached[\s_-]*failure)\b", re.I), ""),
    (re.compile(r"\baction[\s_-]*exemplar(?:s)?\b", re.I), ""),
    (_ABSOLUTE_ROTATION_PATTERN, "orientation-compatible"),
    (_ABSOLUTE_CARDINAL_PATTERN, "global-direction-neutral"),
    (_ABSOLUTE_LATERAL_PATTERN, "role-relative"),
)


def sanitize_transfer_text(value: object) -> str:
    """Return a route-abstracted textual value with no local numeric data."""

    if value is None:
        return ""
    if isinstance(value, Mapping):
        parts = [
            sanitize_transfer_text(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if _key_category(key) is None
        ]
        text = "; ".join(part for part in parts if part)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        parts = [sanitize_transfer_text(item) for item in value]
        text = "; ".join(part for part in parts if part)
    elif isinstance(value, bool):
        text = ""
    elif isinstance(value, (int, float)):
        text = ""
    else:
        text = unicodedata.normalize("NFKC", str(value))

    # Remove path/hash/numeric material before replacing semantic reference
    # words.  Applying the public scanner after this function remains the final
    # fail-closed check.
    for category, pattern in _TEXT_RULES:
        if category in {
            "numeric_unit",
            "coordinate_tuple",
            "filesystem_or_url_reference",
            "hash_reference",
            "source_identifier",
            "numeric_literal",
        }:
            text = pattern.sub(" ", text)
    for pattern, replacement in _SANITIZE_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([;,:])(?:\s*[;,:])+", r"\1", text)
    text = re.sub(r"(?:;\s*){2,}", "; ", text)
    return text.strip(" \t\r\n,;:-")


def remove_memory_leaks(value: object) -> object:
    """Recursively remove forbidden keys and sanitize leaf strings.

    This generic helper is useful for inspecting fixtures.  The exporter still
    constructs each six-field record explicitly; it never copies an arbitrary
    sanitized mapping into the transfer bank.
    """

    if isinstance(value, Mapping):
        cleaned_mapping: dict[str, object] = {}
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            if _key_category(key) is not None:
                continue
            cleaned_child = remove_memory_leaks(child)
            if cleaned_child not in (None, "", [], {}):
                cleaned_mapping[str(key)] = cleaned_child
        return cleaned_mapping
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        cleaned = [remove_memory_leaks(item) for item in value]
        return [item for item in cleaned if item not in (None, "", [], {})]
    if isinstance(value, str):
        return sanitize_transfer_text(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None
    return value
