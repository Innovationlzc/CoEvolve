"""Geometry helpers for object-centric cooperative transport."""

from __future__ import annotations

from math import cos, sin

from praxis_coop.models import Pose2D, wrap_angle


def apply_body_offset(anchor: Pose2D, offset: Pose2D) -> Pose2D:
    """Apply a local SE(2) offset in the anchor pose frame."""

    c = cos(anchor.yaw)
    s = sin(anchor.yaw)
    return Pose2D(
        x=anchor.x + c * offset.x - s * offset.y,
        y=anchor.y + s * offset.x + c * offset.y,
        yaw=anchor.yaw + offset.yaw,
    )


def offset_path(anchor_path: list[Pose2D], offset: Pose2D) -> list[Pose2D]:
    """Project an object-center path into a robot path using a local offset."""

    return [apply_body_offset(pose, offset) for pose in anchor_path]


def relative_body_offset(anchor: Pose2D, pose: Pose2D) -> Pose2D:
    """Return `pose` expressed in the local frame of `anchor`."""

    dx = pose.x - anchor.x
    dy = pose.y - anchor.y
    c = cos(anchor.yaw)
    s = sin(anchor.yaw)
    return Pose2D(
        x=c * dx + s * dy,
        y=-s * dx + c * dy,
        yaw=wrap_angle(pose.yaw - anchor.yaw),
    )
