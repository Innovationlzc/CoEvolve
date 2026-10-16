from __future__ import annotations

from pathlib import Path
import time

from PIL import Image

from praxis_coop.cap_policy.action_video import PhysicalActionVideoRecorder


def _write_frame(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (160, 96), color=color).save(path)


def test_action_video_records_only_explicit_segments_and_builds_timeline(tmp_path):
    source = tmp_path / "vlm_global_frame.png"
    _write_frame(source, (20, 40, 80))
    recorder = PhysicalActionVideoRecorder(
        source_frame_path=source,
        output_dir=tmp_path / "action_videos",
        fps=8.0,
    )

    action = recorder.start_segment(
        attempt=3,
        kind="action",
        label="coordinated_pivot",
        metadata={"checkpoint_before": "accepted_step_02"},
    )
    time.sleep(0.18)
    _write_frame(source, (80, 40, 20))
    action_record = recorder.stop_segment(action, status="completed", tail_s=0.18)

    # This gap represents model inference/evaluation and is deliberately absent.
    time.sleep(0.12)
    rollback = recorder.start_segment(
        attempt=3,
        kind="rollback",
        label="to_accepted_step_02",
    )
    _write_frame(source, (20, 40, 80))
    rollback_record = recorder.stop_segment(
        rollback,
        status="checkpoint_restored",
        tail_s=0.18,
    )
    timeline = recorder.finalize(run_status="SYSTEM_STALL")

    assert action_record["frame_count"] >= 2
    assert rollback_record["frame_count"] >= 1
    assert Path(action_record["path"]).is_file()
    assert Path(rollback_record["path"]).is_file()
    assert timeline["segment_count"] == 2
    assert [item["kind"] for item in timeline["segments"]] == [
        "action",
        "rollback",
    ]
    assert Path(timeline["full_video_path"]).is_file()
    assert timeline["static_frame_compression"] == {
        "enabled": True,
        "max_static_hold_s": 0.2,
        "mean_absdiff_threshold": 0.5,
        "comparison_reference": "last_encoded_frame",
    }
    assert (tmp_path / "action_videos" / "attempt_03" / "manifest.json").is_file()
    assert (tmp_path / "action_videos" / "timeline.json").is_file()


def test_action_video_compresses_long_static_runs_without_dropping_changes(tmp_path):
    source = tmp_path / "vlm_global_frame.png"
    _write_frame(source, (20, 40, 80))
    recorder = PhysicalActionVideoRecorder(
        source_frame_path=source,
        output_dir=tmp_path / "action_videos",
        fps=20.0,
        max_static_hold_s=0.2,
    )

    segment = recorder.start_segment(
        attempt=1,
        kind="action",
        label="static_trim_probe",
    )
    time.sleep(0.55)
    _write_frame(source, (80, 40, 20))
    time.sleep(0.20)
    record = recorder.stop_segment(segment, status="completed")
    timeline = recorder.finalize(run_status="SUCCESS")

    assert record["sampled_frame_count"] >= 10
    assert record["suppressed_static_frame_count"] > 0
    assert record["frame_count"] < record["sampled_frame_count"]
    assert record["encoded_duration_s"] < record["duration_s"]
    assert Path(record["path"]).is_file()
    assert Path(timeline["full_video_path"]).is_file()


def test_action_video_accumulates_gradual_motion_against_last_encoded_frame(tmp_path):
    source = tmp_path / "vlm_global_frame.png"
    _write_frame(source, (20, 40, 80))
    recorder = PhysicalActionVideoRecorder(
        source_frame_path=source,
        output_dir=tmp_path / "action_videos",
        fps=20.0,
        max_static_hold_s=0.1,
        static_frame_mean_absdiff=0.5,
    )

    segment = recorder.start_segment(
        attempt=2,
        kind="action",
        label="gradual_motion_probe",
    )
    time.sleep(0.08)
    for red in range(21, 31):
        _write_frame(source, (red, 40, 80))
        time.sleep(0.06)
    record = recorder.stop_segment(segment, status="completed", tail_s=0.08)

    # Individual one-level changes are below the threshold, while cumulative
    # motion from the last encoded frame must still be retained.
    assert record["sampled_frame_count"] >= 8
    assert record["suppressed_static_frame_count"] > 0
    assert record["frame_count"] >= 3
