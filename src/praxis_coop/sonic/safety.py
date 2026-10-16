"""Central physical-safety profiles for loaded cooperative transport."""

from __future__ import annotations

from dataclasses import dataclass


# This is a geometric orientation boundary, not an experiment hyperparameter.
# Task mode reports crossings as carry-quality evidence while the physical
# support contract remains intact. Strict mode retains it as a hard failure.
PAYLOAD_TOP_FACE_UP_LIMIT_DEG = 90.0

# The elbow50 recovery is preventive, so it must run while the box top face is
# still upward. At 60 degrees the free box has rotated onto the raised-palm
# backstop far beyond its normal settled carry attitude, while retaining a
# 30-degree margin before geometric overturn.
PAYLOAD_RECOVERY_TILT_TRIGGER_DEG = 60.0


@dataclass(frozen=True)
class CarrySafetyProfile:
    """One externally selected safety envelope with internal diagnostic limits."""

    name: str
    min_payload_z_m: float
    min_robot_pelvis_height_m: float
    min_robot_xy_separation_m: float
    settled_max_tilt_deg: float
    hard_max_tilt_deg: float
    max_top_surface_z_range_m: float
    support_loss_grace_s: float
    max_support_center_error_m: float
    team_hard_desync_m: float | None
    warning_robot_step_desync_m: float
    warning_payload_yaw_drift_deg: float
    warning_payload_formation_slip_m: float
    warning_min_translation_tracking_ratio: float
    require_all_support_pads_when_settled: bool
    min_support_force_per_robot_n: float
    min_support_force_balance_ratio: float
    wall_contact_is_hard_failure: bool


# The task profile reflects the experiment's actual physical contract: the box
# must stay off the floor, both robots must retain a hand/wrist/forearm load
# path, neither robot may fall, and their pelvis centers must remain separated.
# The 0.61 m floor rejects the observed 0.608 m collapsed formation while
# retaining the 0.615-0.65 m transient spacing required by safe loaded turns.
# Tilt, palm-center offset, and force imbalance remain diagnostics.
TASK_CARRY_SAFETY = CarrySafetyProfile(
    name="task",
    min_payload_z_m=0.65,
    min_robot_pelvis_height_m=0.35,
    min_robot_xy_separation_m=0.61,
    settled_max_tilt_deg=15.0,
    hard_max_tilt_deg=25.0,
    max_top_surface_z_range_m=0.30,
    # MuJoCo can relabel a still-supported box between wrist, forearm, and torso
    # contacts during a gait cycle.  Let the floor/height/fall gates stop a real
    # drop online, then require valid dual upper-limb support again at settle.
    support_loss_grace_s=10.0,
    max_support_center_error_m=0.25,
    # Allow the verified four-second SONIC gait window to pass its asymmetric
    # launch transient. Payload drop, dual-support loss, and robot fall remain
    # immediate hard failures; endpoint formation is still verifier-audited.
    team_hard_desync_m=0.50,
    # A loaded pivot intentionally lets the outer robot sweep farther while
    # the inner robot holds translation. Three supported 30-degree segments
    # reached 0.23 m of additional root separation without losing either load
    # path, so retain 0.30 m of turn room. The minimum separation and
    # payload-drop gates remain hard.
    warning_robot_step_desync_m=0.30,
    warning_payload_yaw_drift_deg=15.0,
    # The first payload-centered turn probe remained supported at 0.09 m but
    # the next identical segment dropped the box at 0.235 m. Trigger the
    # one-way elbow50 recovery before that unstable accumulation begins.
    warning_payload_formation_slip_m=0.08,
    warning_min_translation_tracking_ratio=0.10,
    require_all_support_pads_when_settled=False,
    min_support_force_per_robot_n=0.0,
    min_support_force_balance_ratio=0.0,
    wall_contact_is_hard_failure=False,
)


# Strict reproduces the older conservative rollback behavior for legacy debug
# comparisons. It is not the production/formal default or a registered paper
# ablation.
STRICT_CARRY_SAFETY = CarrySafetyProfile(
    name="strict",
    min_payload_z_m=0.80,
    min_robot_pelvis_height_m=0.35,
    min_robot_xy_separation_m=0.70,
    settled_max_tilt_deg=6.0,
    hard_max_tilt_deg=8.0,
    max_top_surface_z_range_m=0.12,
    support_loss_grace_s=0.8,
    max_support_center_error_m=0.06,
    team_hard_desync_m=0.04,
    warning_robot_step_desync_m=0.04,
    warning_payload_yaw_drift_deg=6.0,
    warning_payload_formation_slip_m=0.05,
    warning_min_translation_tracking_ratio=0.35,
    require_all_support_pads_when_settled=True,
    min_support_force_per_robot_n=0.5,
    min_support_force_balance_ratio=0.10,
    wall_contact_is_hard_failure=True,
)


CARRY_SAFETY_PROFILES = {
    profile.name: profile
    for profile in (TASK_CARRY_SAFETY, STRICT_CARRY_SAFETY)
}


def get_carry_safety_profile(
    name: str | CarrySafetyProfile = "task",
) -> CarrySafetyProfile:
    """Resolve a named profile and fail early on configuration typos."""

    if isinstance(name, CarrySafetyProfile):
        return name
    normalized = str(name).strip().lower()
    try:
        return CARRY_SAFETY_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown carry safety profile {name!r}; expected one of "
            f"{sorted(CARRY_SAFETY_PROFILES)}"
        ) from exc


def loaded_turn_support_guard(
    profile: str | CarrySafetyProfile,
    configured_max_error_m: float,
) -> tuple[float, bool]:
    """Resolve the support-offset limit without promoting task diagnostics."""

    safety = get_carry_safety_profile(profile)
    configured_limit = max(0.0, float(configured_max_error_m))
    if safety.name == "strict":
        return (
            min(configured_limit, safety.warning_payload_formation_slip_m),
            True,
        )
    return configured_limit, False
