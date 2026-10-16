from __future__ import annotations

import pytest

from coevolve_experiments.memory import (
    StrategyBankValidationError,
    remove_memory_leaks,
    scan_memory_leaks,
    validate_strategy_bank,
)


def _mirrored_scene_record() -> dict[str, str]:
    return {
        "phase_trigger": "at the left corner before the red cylinder",
        "failure_condition": "clockwise motion misses the north opening",
        "coordination_pattern": (
            "front turns CW while back anchors on the inner side"
        ),
        "verified_outcome": "accepted",
        "avoid_rule": (
            "do not turn counterclockwise toward the right blue wall"
        ),
        "repair_direction": "continue CCW toward the south-east red box",
    }


def test_mirrored_absolute_scene_language_is_not_transferable() -> None:
    record = _mirrored_scene_record()
    categories = {finding.category for finding in scan_memory_leaks(record)}

    assert {
        "absolute_lateral_direction",
        "absolute_rotation_direction",
        "absolute_cardinal_direction",
        "named_scene_landmark",
    } <= categories
    with pytest.raises(StrategyBankValidationError, match="leaks"):
        validate_strategy_bank({"schema_version": 1, "records": [record]})


def test_sanitizer_abstracts_mirrored_language_but_preserves_robot_roles() -> None:
    cleaned = remove_memory_leaks(_mirrored_scene_record())

    assert isinstance(cleaned, dict)
    assert not scan_memory_leaks(cleaned)
    serialized = " ".join(str(value) for value in cleaned.values()).lower()
    assert "front" in serialized
    assert "back" in serialized
    assert "role-relative" in serialized
    assert "orientation-compatible" in serialized
    assert "global-direction-neutral" in serialized
    assert "environment landmark" in serialized
    for leaked in (
        "left",
        "right",
        "clockwise",
        "counterclockwise",
        " cw ",
        " ccw ",
        "north",
        "south",
        "red cylinder",
        "blue wall",
        "red box",
    ):
        assert leaked not in f" {serialized} "


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("move leftward", "absolute_lateral_direction"),
        ("rotate anti-clockwise", "absolute_rotation_direction"),
        ("use the NW doorway", "absolute_cardinal_direction"),
        ("approach the western opening", "absolute_cardinal_direction"),
        ("stop beside the orange concrete pillar", "named_scene_landmark"),
        ("wait at the purple gate", "named_scene_landmark"),
    ],
)
def test_direction_and_named_landmark_variants_are_detected(
    text: str,
    category: str,
) -> None:
    assert category in {finding.category for finding in scan_memory_leaks(text)}


def test_role_relative_and_shape_neutral_terms_remain_legal() -> None:
    text = (
        "front maintains upright support while back preserves straight, "
        "role-compatible payload motion"
    )

    assert not scan_memory_leaks(text)
