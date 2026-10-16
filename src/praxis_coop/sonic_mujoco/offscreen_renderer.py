"""Render fixed MuJoCo observations outside the interactive viewer process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _load_snapshot(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        return {
            "qpos": np.asarray(loaded["qpos"], dtype=np.float64),
            "geom_rgba": np.asarray(loaded["geom_rgba"], dtype=np.float32),
            "site_rgba": np.asarray(loaded["site_rgba"], dtype=np.float32),
            "sim_time_s": float(np.asarray(loaded["sim_time_s"]).reshape(-1)[0]),
            "sequence": int(np.asarray(loaded["sequence"]).reshape(-1)[0]),
        }


def _apply_snapshot(model: Any, data: Any, snapshot: dict[str, Any]) -> None:
    qpos = np.asarray(snapshot["qpos"], dtype=np.float64)
    geom_rgba = np.asarray(snapshot["geom_rgba"], dtype=np.float32)
    site_rgba = np.asarray(snapshot["site_rgba"], dtype=np.float32)
    if qpos.shape != data.qpos.shape:
        raise ValueError(f"qpos shape {qpos.shape} != {data.qpos.shape}")
    if geom_rgba.shape != model.geom_rgba.shape:
        raise ValueError(
            f"geom_rgba shape {geom_rgba.shape} != {model.geom_rgba.shape}"
        )
    if site_rgba.shape != model.site_rgba.shape:
        raise ValueError(
            f"site_rgba shape {site_rgba.shape} != {model.site_rgba.shape}"
        )
    if not np.all(np.isfinite(qpos)):
        raise ValueError("snapshot qpos contains non-finite values")
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    model.geom_rgba[:] = geom_rgba
    model.site_rgba[:] = site_rgba


def run_renderer(
    *,
    model_path: Path,
    snapshot_path: Path,
    output_path: Path,
    status_path: Path,
    camera: str,
    width: int,
    height: int,
    poll_interval_s: float,
) -> int:
    import mujoco
    from PIL import Image

    running = True

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    last_sequence = -1
    try:
        while running:
            try:
                snapshot = _load_snapshot(snapshot_path)
                sequence = int(snapshot["sequence"])
                if sequence == last_sequence:
                    time.sleep(poll_interval_s)
                    continue
                _apply_snapshot(model, data, snapshot)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = output_path.with_name(
                    f".{output_path.stem}.tmp{output_path.suffix}"
                )
                Image.fromarray(frame).save(temporary, format="PNG")
                temporary.replace(output_path)
                captured_at = time.time()
                _atomic_write_json(
                    status_path,
                    {
                        "available": True,
                        "error": "",
                        "captured_at_unix": captured_at,
                        "sim_time_s": float(snapshot["sim_time_s"]),
                        "sequence": sequence,
                        "renderer_backend": "egl_subprocess",
                    },
                )
                last_sequence = sequence
            except FileNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001 - report and keep worker alive
                _atomic_write_json(
                    status_path,
                    {
                        "available": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "captured_at_unix": 0.0,
                        "sim_time_s": 0.0,
                        "sequence": last_sequence,
                        "renderer_backend": "egl_subprocess",
                    },
                )
            time.sleep(poll_interval_s)
    finally:
        renderer.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--poll-interval-s", type=float, default=0.02)
    args = parser.parse_args()
    return run_renderer(
        model_path=args.model,
        snapshot_path=args.snapshot,
        output_path=args.output,
        status_path=args.status,
        camera=args.camera,
        width=max(1, args.width),
        height=max(1, args.height),
        poll_interval_s=max(0.005, args.poll_interval_s),
    )


if __name__ == "__main__":
    raise SystemExit(main())
