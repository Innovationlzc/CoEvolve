"""External-camera observation helpers for orchestrator and evolution agent."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin, tan
from pathlib import Path
import shutil
import struct
import subprocess
import time
from typing import Any

from .contracts import VisualObservation


@dataclass
class ExternalCameraObserver:
    """Capture or register a wide-view image for VLM grounding.

    If OpenCV is unavailable or no camera device is configured, this helper
    returns a structured missing-camera observation instead of pretending that
    visual evidence was used.
    """

    output_dir: Path
    camera_device: int | None = None
    frame_path: Path | None = None
    window_name: str = ""
    tracking_inset: bool = True
    tracking_inset_world_span_m: float = 4.6
    max_global_frame_age_s: float = 2.0

    def observe(
        self,
        *,
        label: str,
        runtime_state: dict[str, Any] | None = None,
    ) -> VisualObservation:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.frame_path is not None:
            frame_path = self._prepare_vlm_frame(
                Path(self.frame_path),
                label=label,
                runtime_state=runtime_state,
            )
            return VisualObservation(
                source="provided_frame",
                frame_path=str(frame_path),
                summary=f"provided external camera frame for {label}",
                captured_at_unix=time.time(),
            )
        runtime_global = _runtime_global_capture(runtime_state)
        if runtime_global is not None:
            return self._register_runtime_global_frame(
                runtime_global,
                label=label,
                runtime_state=runtime_state,
            )
        if self.window_name:
            return self._capture_x11_window(label=label, runtime_state=runtime_state)
        if self.camera_device is None:
            return VisualObservation(
                source="missing_external_camera",
                summary="no external camera frame configured for this iteration",
                captured_at_unix=time.time(),
            )
        try:
            import cv2  # type: ignore
        except ImportError:
            return VisualObservation(
                source="camera_unavailable",
                summary="opencv is not installed, so no external camera frame was captured",
                captured_at_unix=time.time(),
            )

        cap = cv2.VideoCapture(int(self.camera_device))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return VisualObservation(
                source="camera_capture_failed",
                summary=f"failed to capture frame from camera device {self.camera_device}",
                captured_at_unix=time.time(),
            )
        path = self.output_dir / f"{label}_external_camera.jpg"
        cv2.imwrite(str(path), frame)
        frame_path = self._prepare_vlm_frame(
            path,
            label=label,
            runtime_state=runtime_state,
        )
        return VisualObservation(
            source=f"camera_device_{self.camera_device}",
            frame_path=str(frame_path),
            summary=f"captured external camera frame for {label}",
            captured_at_unix=time.time(),
        )

    def _register_runtime_global_frame(
        self,
        capture: dict[str, Any],
        *,
        label: str,
        runtime_state: dict[str, Any] | None,
    ) -> VisualObservation:
        available = bool(capture.get("available", False))
        frame_path = Path(str(capture.get("frame_path", "")))
        try:
            age_s = float(capture.get("age_s", float("inf")))
        except (TypeError, ValueError):
            age_s = float("inf")
        if (
            not available
            or not frame_path.is_file()
            or age_s > max(0.1, float(self.max_global_frame_age_s))
        ):
            return VisualObservation(
                source="mujoco_offscreen_global_unavailable",
                summary=(
                    "fixed MuJoCo global capture is unavailable or stale; the interactive "
                    "viewer is deliberately not used as a fallback"
                ),
                captured_at_unix=time.time(),
            )
        preserved = self.output_dir / f"{label}_mujoco_global.png"
        shutil.copyfile(frame_path, preserved)
        prepared = self._prepare_vlm_frame(
            preserved,
            label=label,
            runtime_state=runtime_state,
        )
        return VisualObservation(
            source="mujoco_offscreen_fixed_global",
            frame_path=str(prepared),
            summary=(
                f"captured fixed MuJoCo global camera for {label}; interactive viewer "
                "camera input cannot alter this evidence"
            ),
            captured_at_unix=float(capture.get("captured_at_unix", time.time())),
        )

    def _capture_x11_window(
        self,
        *,
        label: str,
        runtime_state: dict[str, Any] | None,
    ) -> VisualObservation:
        xwd_path = self.output_dir / f"{label}_mujoco_window.xwd"
        png_path = self.output_dir / f"{label}_mujoco_window.png"
        try:
            subprocess.run(
                ["xwd", "-silent", "-name", self.window_name, "-out", str(xwd_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            _xwd_to_png(xwd_path, png_path)
        except Exception as exc:  # noqa: BLE001 - keep observation structured instead of crashing early
            return VisualObservation(
                source="window_capture_failed",
                summary=f"failed to capture X11 window {self.window_name!r}: {type(exc).__name__}: {exc}",
                captured_at_unix=time.time(),
            )
        frame_path = self._prepare_vlm_frame(
            png_path,
            label=label,
            runtime_state=runtime_state,
        )
        return VisualObservation(
            source="x11_window_capture",
            frame_path=str(frame_path),
            summary=(
                f"captured X11 window {self.window_name!r} for {label}; "
                "VLM frame contains the full scene and a payload-following close-up"
            ),
            captured_at_unix=time.time(),
        )

    def _prepare_vlm_frame(
        self,
        path: Path,
        *,
        label: str,
        runtime_state: dict[str, Any] | None,
    ) -> Path:
        if not self.tracking_inset or not runtime_state:
            return path
        try:
            return _write_global_tracking_composite(
                path,
                self.output_dir / f"{label}_vlm_global_and_payload.png",
                runtime_state=runtime_state,
                world_span_m=self.tracking_inset_world_span_m,
            )
        except Exception:  # noqa: BLE001 - retain the raw frame as valid evidence
            return path


def _xwd_to_png(xwd_path: Path, png_path: Path) -> None:
    from PIL import Image

    payload = xwd_path.read_bytes()
    fields = struct.unpack(">25I", payload[:100])
    header_size = fields[0]
    width = fields[4]
    height = fields[5]
    bits_per_pixel = fields[11]
    bytes_per_line = fields[12]
    ncolors = fields[19]
    if bits_per_pixel != 32:
        raise ValueError(f"unsupported XWD bits_per_pixel={bits_per_pixel}; expected 32")
    offset = header_size + ncolors * 12
    rows = []
    bytes_per_pixel = bits_per_pixel // 8
    for y in range(height):
        start = offset + y * bytes_per_line
        rows.append(payload[start : start + width * bytes_per_pixel])
    raw = b"".join(rows)
    image = Image.frombytes("RGBA", (width, height), raw)
    red, green, blue, _alpha = image.split()
    Image.merge("RGB", (red, green, blue)).save(png_path)


def _write_global_tracking_composite(
    source_path: Path,
    output_path: Path,
    *,
    runtime_state: dict[str, Any],
    world_span_m: float,
) -> Path:
    """Keep the global frame intact and append a payload-following detail panel."""

    from PIL import Image, ImageDraw

    payload = runtime_state.get("payload", {})
    if not isinstance(payload, dict):
        return source_path
    try:
        payload_x = float(payload["x"])
        payload_y = float(payload["y"])
    except (KeyError, TypeError, ValueError):
        return source_path

    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    _draw_physics_heading_overlay(
        image,
        runtime_state=runtime_state,
        payload_x=payload_x,
        payload_y=payload_y,
    )
    center_x, center_y, pixels_per_meter = _project_global_xy_to_frame(
        payload_x,
        payload_y,
        width=width,
        height=height,
        runtime_state=runtime_state,
    )
    crop_size = int(
        max(
            220,
            min(width, height, pixels_per_meter * max(2.5, float(world_span_m))),
        )
    )
    left = int(round(center_x - 0.5 * crop_size))
    top = int(round(center_y - 0.5 * crop_size))
    left = max(0, min(width - crop_size, left))
    top = max(0, min(height - crop_size, top))
    closeup = image.crop((left, top, left + crop_size, top + crop_size))

    panel_width = min(720, max(480, height - 40))
    canvas = Image.new("RGB", (width + panel_width, height), (20, 22, 25))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    legend = "FULL GLOBAL SCENE | TARGET: RED CYLINDER | PAYLOAD: CYAN LONG BOX"
    draw.rectangle((0, 0, min(width, 620), 32), fill=(15, 17, 20))
    draw.text((10, 9), legend, fill=(255, 255, 255))

    closeup_size = panel_width - 32
    closeup = closeup.resize((closeup_size, closeup_size), Image.Resampling.LANCZOS)
    panel_x = width + 16
    panel_y = 44
    canvas.paste(closeup, (panel_x, panel_y))
    draw.rectangle(
        (panel_x - 2, panel_y - 2, panel_x + closeup_size + 1, panel_y + closeup_size + 1),
        outline=(245, 195, 66),
        width=3,
    )
    draw.text((panel_x, 14), "PAYLOAD / ROBOT CLOSE-UP", fill=(245, 195, 66))

    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    support = sorted(str(item) for item in contacts.get("support_robots", ()))
    support_geometries = contacts.get("support_geometries", ())
    metrics = [
        f"payload xyz: {payload_x:.3f}, {payload_y:.3f}, {float(payload.get('z', 0.0)):.3f}",
        f"tilt: {float(payload.get('tilt_deg', 0.0)):.2f} deg",
        "heading arrows: WHITE=current MAGENTA=CW30 GREEN=CCW30",
        f"support robots: {', '.join(support) or 'none'}",
        f"hand/forearm contacts: {len(set(support_geometries))}",
        f"floor contact: {bool(contacts.get('floor', False))}",
        f"wall contacts: {len(tuple(contacts.get('walls', ())))}",
    ]
    text_y = panel_y + closeup_size + 14
    for line in metrics:
        if text_y + 18 >= height:
            break
        draw.text((panel_x, text_y), line, fill=(235, 238, 242))
        text_y += 20

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _draw_physics_heading_overlay(
    image: Any,
    *,
    runtime_state: dict[str, Any],
    payload_x: float,
    payload_y: float,
) -> None:
    """Draw exact current/CW/CCW heading candidates in the fixed camera frame."""

    from PIL import ImageDraw

    robots = runtime_state.get("robots", {})
    if not isinstance(robots, dict):
        return
    yaw_samples: list[float] = []
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            continue
        try:
            yaw_samples.append(radians(float(robot["yaw_deg"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not yaw_samples:
        return
    mean_yaw = atan2(
        sum(sin(value) for value in yaw_samples),
        sum(cos(value) for value in yaw_samples),
    )
    width, height = image.size
    origin_x, origin_y, _pixels_per_meter = _project_global_xy_to_frame(
        payload_x,
        payload_y,
        width=width,
        height=height,
        runtime_state=runtime_state,
    )
    draw = ImageDraw.Draw(image)
    arrow_length_m = 0.82
    candidates = (
        ("NOW", mean_yaw, (255, 255, 255)),
        ("CW30", mean_yaw - radians(30.0), (255, 70, 220)),
        ("CCW30", mean_yaw + radians(30.0), (55, 255, 125)),
    )
    for label, yaw_rad, color in candidates:
        endpoint_x, endpoint_y, _ = _project_global_xy_to_frame(
            payload_x + arrow_length_m * cos(yaw_rad),
            payload_y + arrow_length_m * sin(yaw_rad),
            width=width,
            height=height,
            runtime_state=runtime_state,
        )
        _draw_labeled_arrow(
            draw,
            start=(origin_x, origin_y),
            end=(endpoint_x, endpoint_y),
            label=f"{label} {degrees(yaw_rad):.0f}",
            color=color,
        )


def _draw_labeled_arrow(
    draw: Any,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a compact arrow and readable label without hiding the team."""

    start_x, start_y = start
    end_x, end_y = end
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    length = hypot(delta_x, delta_y)
    if length < 1.0:
        return
    unit_x = delta_x / length
    unit_y = delta_y / length
    perp_x = -unit_y
    perp_y = unit_x
    draw.line((start_x, start_y, end_x, end_y), fill=color, width=5)
    arrow_base_x = end_x - 12.0 * unit_x
    arrow_base_y = end_y - 12.0 * unit_y
    draw.polygon(
        (
            (end_x, end_y),
            (arrow_base_x + 5.5 * perp_x, arrow_base_y + 5.5 * perp_y),
            (arrow_base_x - 5.5 * perp_x, arrow_base_y - 5.5 * perp_y),
        ),
        fill=color,
    )
    text_x = end_x + 6.0
    text_y = end_y - 12.0
    text_box = draw.textbbox((text_x, text_y), label)
    draw.rectangle(
        (text_box[0] - 2, text_box[1] - 1, text_box[2] + 2, text_box[3] + 1),
        fill=(12, 14, 17),
    )
    draw.text((text_x, text_y), label, fill=color)


def _project_global_xy_to_frame(
    world_x: float,
    world_y: float,
    *,
    width: int,
    height: int,
    runtime_state: dict[str, Any],
) -> tuple[float, float, float]:
    camera = runtime_state.get("global_observation", {})
    if not isinstance(camera, dict) or not camera:
        camera = runtime_state.get("viewer_camera", {})
    camera = camera if isinstance(camera, dict) else {}
    lookat = camera.get("lookat_xyz", (0.47, 4.93, 0.45))
    if not isinstance(lookat, (list, tuple)) or len(lookat) < 2:
        lookat = (0.47, 4.93, 0.45)
    distance = max(1.0, float(camera.get("distance_m", 12.5)))
    fovy_deg = min(120.0, max(10.0, float(camera.get("fovy_deg", 45.0))))
    pixels_per_meter = (0.5 * height) / (distance * tan(0.5 * radians(fovy_deg)))

    # The formal global preset is near top-down at azimuth=-90. In that view,
    # world -X maps right and world +Y maps down. Keeping this explicit makes
    # the crop deterministic even when the X11 window has a wide aspect ratio.
    screen_x = 0.5 * width - pixels_per_meter * (world_x - float(lookat[0]))
    screen_y = 0.5 * height + pixels_per_meter * (world_y - float(lookat[1]))
    return screen_x, screen_y, pixels_per_meter


def _runtime_global_capture(
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(runtime_state, dict):
        return None
    capture = runtime_state.get("global_observation")
    return capture if isinstance(capture, dict) else None
