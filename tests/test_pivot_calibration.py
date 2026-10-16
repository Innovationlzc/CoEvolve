from argparse import Namespace

from scripts.calibrate_sonic_pivot import _summarize


def _sample(*, angle, geometry_match, stable=True, yaw_gain=1.0):
    return {
        "requested_angle_deg": float(angle),
        "geometry_role_match": geometry_match,
        "stable": stable,
        "semantic_fidelity": 0.5 <= yaw_gain <= 1.5,
        "yaw_gain": yaw_gain,
        "signed_directional_yaw_deg": float(angle) * yaw_gain,
        "bilateral_support_sample_ratio": 1.0,
        "min_front_support_force_n": 2.0,
        "min_back_support_force_n": 3.0,
    }


def test_pivot_allowlist_uses_only_geometry_matched_repetitions():
    args = Namespace(
        angles_deg=(15.0,),
        baseline_checkpoint_id="loaded",
        duration_s=8.0,
        swing_radius_m=0.8,
        turn_directions=("clockwise", "counterclockwise"),
        repetitions=3,
    )
    samples = [
        _sample(angle=15, geometry_match=True, yaw_gain=gain)
        for gain in (0.9, 1.0, 1.1, 0.8, 1.2, 1.0)
    ] + [_sample(angle=15, geometry_match=False, stable=False, yaw_gain=-0.2)]

    summary = _summarize(args, samples)

    assert summary["recommended_model_angle_allowlist_deg"] == [15.0]
    group = summary["groups_by_requested_angle"][0]
    assert group["geometry_matched_sample_count"] == 6
    assert group["all_geometry_matched_samples_stable"] is True


def test_pivot_allowlist_rejects_one_semantically_unfaithful_matched_sample():
    args = Namespace(
        angles_deg=(20.0,),
        baseline_checkpoint_id="loaded",
        duration_s=8.0,
        swing_radius_m=0.8,
        turn_directions=("clockwise", "counterclockwise"),
        repetitions=3,
    )
    samples = [
        _sample(angle=20, geometry_match=True, yaw_gain=gain)
        for gain in (1.0, 1.1, 0.49, 0.9, 1.0, 1.2)
    ]

    summary = _summarize(args, samples)

    assert summary["recommended_model_angle_allowlist_deg"] == []
    assert (
        summary["groups_by_requested_angle"][0][
            "all_geometry_matched_samples_semantically_faithful"
        ]
        is False
    )
