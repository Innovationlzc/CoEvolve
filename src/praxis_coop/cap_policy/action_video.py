"""Record only physical action and checkpoint-restore intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import threading
import time
from typing import Any

from .contracts import write_json


@dataclass
class VideoSegment:
    sequence: int
    attempt: int
    kind: str
    label: str
    path: Path
    started_at_unix: float
    metadata: dict[str, Any] = field(default_factory=dict)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)
    frame_count: int = 0
    sampled_frame_count: int = 0
    suppressed_static_frame_count: int = 0
    error: str = ""


class PhysicalActionVideoRecorder:
    """Poll the immutable global-camera PNG during physical intervals only."""

    def __init__(
        self,
        *,
        source_frame_path: Path,
        output_dir: Path,
        fps: float = 10.0,
        enabled: bool = True,
        max_static_hold_s: float = 0.2,
        static_frame_mean_absdiff: float = 0.5,
    ) -> None:
        self.source_frame_path = Path(source_frame_path)
        self.output_dir = Path(output_dir)
        self.fps = max(1.0, float(fps))
        self.enabled = bool(enabled)
        self.max_static_hold_s = max(0.0, float(max_static_hold_s))
        self.static_frame_mean_absdiff = max(
            0.0,
            float(static_frame_mean_absdiff),
        )
        self._lock = threading.Lock()
        self._active: VideoSegment | None = None
        self._timeline: list[dict[str, Any]] = []
        self._sequence = 0
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def timeline(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(item) for item in self._timeline)

    def start_segment(
        self,
        *,
        attempt: int,
        kind: str,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> VideoSegment | None:
        if not self.enabled:
            return None
        normalized_kind = _safe_name(kind) or "physical"
        normalized_label = _safe_name(label) or "transaction"
        with self._lock:
            if self._active is not None:
                raise RuntimeError(
                    "physical video segments cannot overlap: "
                    f"{self._active.kind}:{self._active.label} is still active"
                )
            self._sequence += 1
            attempt_index = max(0, int(attempt))
            attempt_dir = self.output_dir / f"attempt_{attempt_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            path = attempt_dir / (
                f"attempt_{attempt_index:02d}_{normalized_kind}_"
                f"{normalized_label}_{self._sequence:03d}.mp4"
            )
            segment = VideoSegment(
                sequence=self._sequence,
                attempt=attempt_index,
                kind=normalized_kind,
                label=normalized_label,
                path=path,
                started_at_unix=time.time(),
                metadata=dict(metadata or {}),
            )
            segment.thread = threading.Thread(
                target=self._capture_segment,
                args=(segment,),
                name=f"physical-video-{self._sequence:03d}",
                daemon=True,
            )
            self._active = segment
            segment.thread.start()
            return segment

    def stop_segment(
        self,
        segment: VideoSegment | None,
        *,
        status: str,
        tail_s: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if segment is None:
            return {}
        if tail_s > 0.0:
            time.sleep(float(tail_s))
        segment.stop_event.set()
        if segment.thread is not None:
            segment.thread.join(timeout=max(5.0, 2.0 + tail_s))
            if segment.thread.is_alive() and not segment.error:
                segment.error = "video capture thread did not stop"
        ended_at = time.time()
        record = {
            "sequence": segment.sequence,
            "attempt": segment.attempt,
            "kind": segment.kind,
            "label": segment.label,
            "path": str(segment.path),
            "started_at_unix": segment.started_at_unix,
            "ended_at_unix": ended_at,
            "duration_s": max(0.0, ended_at - segment.started_at_unix),
            "fps": self.fps,
            "frame_count": segment.frame_count,
            "sampled_frame_count": segment.sampled_frame_count,
            "suppressed_static_frame_count": (
                segment.suppressed_static_frame_count
            ),
            "encoded_duration_s": segment.frame_count / self.fps,
            "max_static_hold_s": self.max_static_hold_s,
            "status": str(status),
            "error": segment.error,
            "metadata": {**segment.metadata, **dict(metadata or {})},
        }
        with self._lock:
            if self._active is segment:
                self._active = None
            self._timeline.append(record)
        self._write_manifests(segment.attempt)
        return record

    def finalize(self, *, run_status: str = "") -> dict[str, Any]:
        active = self._active
        if active is not None:
            self.stop_segment(active, status="interrupted_during_finalization")
        if not self.enabled:
            return {"enabled": False}
        full_video = self.output_dir / "full_run_physical_timeline.mp4"
        concat_error = self._concatenate_timeline(full_video)
        timeline = {
            "schema_version": 1,
            "source_frame_path": str(self.source_frame_path),
            "static_frame_compression": {
                "enabled": True,
                "max_static_hold_s": self.max_static_hold_s,
                "mean_absdiff_threshold": self.static_frame_mean_absdiff,
                "comparison_reference": "last_encoded_frame",
            },
            "recording_scope": (
                "physical action execution and visible checkpoint restore only; "
                "model inference, verifier waiting, and billing are excluded; "
                "unchanged-frame runs are compressed"
            ),
            "run_status": str(run_status),
            "segment_count": len(self._timeline),
            "full_video_path": str(full_video) if full_video.is_file() else "",
            "concatenation_error": concat_error,
            "segments": list(self.timeline),
        }
        write_json(self.output_dir / "timeline.json", timeline)
        return timeline

    def _capture_segment(self, segment: VideoSegment) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency boundary
            segment.error = f"OpenCV unavailable: {type(exc).__name__}: {exc}"
            return
        writer = None
        interval_s = 1.0 / self.fps
        next_frame_at = time.monotonic()
        last_encoded_preview = None
        static_samples_since_change = 0
        max_static_frames = max(0, int(round(self.max_static_hold_s * self.fps)))
        try:
            while not segment.stop_event.is_set() or segment.frame_count == 0:
                now = time.monotonic()
                if now < next_frame_at:
                    segment.stop_event.wait(min(interval_s, next_frame_at - now))
                    continue
                frame = cv2.imread(str(self.source_frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    if segment.stop_event.is_set():
                        break
                    next_frame_at = now + interval_s
                    continue
                segment.sampled_frame_count += 1
                preview = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
                preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                frame_changed = True
                if last_encoded_preview is not None:
                    mean_absdiff = float(
                        cv2.absdiff(preview, last_encoded_preview).mean()
                    )
                    frame_changed = mean_absdiff > self.static_frame_mean_absdiff
                if frame_changed:
                    static_samples_since_change = 0
                else:
                    static_samples_since_change += 1
                    if static_samples_since_change > max_static_frames:
                        segment.suppressed_static_frame_count += 1
                        next_frame_at = max(
                            next_frame_at + interval_s,
                            now + interval_s * 0.25,
                        )
                        if segment.stop_event.is_set():
                            break
                        continue
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(segment.path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (int(width), int(height)),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV could not open MP4 video writer")
                writer.write(frame)
                segment.frame_count += 1
                last_encoded_preview = preview
                next_frame_at = max(next_frame_at + interval_s, now + interval_s * 0.25)
                if segment.stop_event.is_set():
                    break
        except Exception as exc:  # noqa: BLE001 - preserve run and report media failure
            segment.error = f"{type(exc).__name__}: {exc}"
        finally:
            if writer is not None:
                writer.release()
            if segment.frame_count == 0 and not segment.error:
                segment.error = (
                    f"no readable fixed-global frame at {self.source_frame_path}"
                )

    def _write_manifests(self, attempt: int) -> None:
        records = [item for item in self.timeline if item["attempt"] == attempt]
        attempt_dir = self.output_dir / f"attempt_{attempt:02d}"
        write_json(
            attempt_dir / "manifest.json",
            {
                "attempt": attempt,
                "segment_count": len(records),
                "segments": records,
            },
        )
        write_json(
            self.output_dir / "timeline.json",
            {
                "schema_version": 1,
                "source_frame_path": str(self.source_frame_path),
                "static_frame_compression": {
                    "enabled": True,
                    "max_static_hold_s": self.max_static_hold_s,
                    "mean_absdiff_threshold": self.static_frame_mean_absdiff,
                    "comparison_reference": "last_encoded_frame",
                },
                "segment_count": len(self._timeline),
                "segments": list(self.timeline),
            },
        )

    def _concatenate_timeline(self, output_path: Path) -> str:
        clips = [
            Path(str(item["path"]))
            for item in self.timeline
            if not item.get("error") and Path(str(item["path"])).is_file()
        ]
        if not clips:
            return "no readable action or restore clips"
        try:
            import cv2  # type: ignore

            writer = None
            for clip in clips:
                capture = cv2.VideoCapture(str(clip))
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        if writer is None:
                            height, width = frame.shape[:2]
                            writer = cv2.VideoWriter(
                                str(output_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                self.fps,
                                (int(width), int(height)),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(
                                    "OpenCV could not open full-run MP4 writer"
                                )
                        writer.write(frame)
                finally:
                    capture.release()
            if writer is None:
                return "clips contained no readable frames"
            writer.release()
            return ""
        except Exception as exc:  # noqa: BLE001 - media is an audit artifact
            output_path.unlink(missing_ok=True)
            return f"{type(exc).__name__}: {exc}"


def _safe_name(value: object) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    return normalized.strip("._-")[:80]


__all__ = ["PhysicalActionVideoRecorder", "VideoSegment"]
