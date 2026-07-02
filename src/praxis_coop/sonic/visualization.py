"""HTML visualization helpers for SONIC SkillGraph evolution runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from praxis_coop.models import SceneGraph
from praxis_coop.sonic.commands import SonicCommandStream
from praxis_coop.utils.usda_layout import parse_manual_layout_prims


def write_sonic_evolution_html(
    path: Path,
    *,
    scene: SceneGraph,
    attempts: list[dict[str, Any]],
    streams: list[SonicCommandStream],
    history: list[dict[str, Any]] | None = None,
) -> None:
    """Write a self-contained top-down visualization of attempts and commands."""

    payload = {
        "scene": _scene_payload(scene),
        "attempts": attempts,
        "streams": [stream.to_json() for stream in streams],
        "history": history or [],
    }
    data = json.dumps(payload, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_html_document(data), encoding="utf-8")


def _scene_payload(scene: SceneGraph) -> dict[str, Any]:
    source = str(scene.metadata.get("source", ""))
    layout_path = _resolve_scene_source(source)
    layout_prims = [prim.to_json() for prim in parse_manual_layout_prims(layout_path)] if layout_path else []
    return {
        "scene_id": scene.scene_id,
        "source": source,
        "layout_prims": layout_prims,
        "robots": {
            robot_id: {"x": robot.pose.x, "y": robot.pose.y, "yaw": robot.pose.yaw}
            for robot_id, robot in scene.robots.items()
        },
        "object": {
            "id": scene.movable_object.object_id,
            "x": scene.movable_object.pose.x,
            "y": scene.movable_object.pose.y,
            "yaw": scene.movable_object.pose.yaw,
            "size": scene.movable_object.size,
        },
        "target": {
            "id": scene.target.target_id,
            "x": scene.target.pose.x,
            "y": scene.target.pose.y,
            "yaw": scene.target.pose.yaw,
            "tolerance_xy": scene.target.tolerance_xy,
        },
        "walls": scene.metadata.get("wall_poses", {}),
    }


def _resolve_scene_source(source: str) -> Path | None:
    if not source:
        return None
    path = Path(source)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / path


def _html_document(data_json: str) -> str:
    escaped = html.escape(data_json, quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Praxis-Coop SONIC Evolution</title>
  <style>
    :root {{
      color-scheme: dark;
      --ink: #f4f7fb;
      --muted: #9aa8b8;
      --glass: rgba(14, 20, 30, 0.72);
      --glass-2: rgba(22, 30, 42, 0.82);
      --line: rgba(220, 230, 245, 0.16);
      --blue: #67a2ff;
      --green: #5ee4a7;
      --red: #ff7a59;
      --gold: #ffd166;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #060911;
      overflow: hidden;
    }}
    .demo-root {{
      position: relative;
      min-height: 100vh;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100vh;
      background: radial-gradient(circle at 50% 30%, #162033 0, #080c15 45%, #05070d 100%);
    }}
    .glass {{
      background: var(--glass);
      border: 1px solid var(--line);
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
      backdrop-filter: blur(16px);
    }}
    .side-panel {{
      position: absolute;
      left: 18px;
      top: 18px;
      width: 330px;
      max-height: calc(100vh - 36px);
      overflow: auto;
      display: grid;
      gap: 12px;
      pointer-events: auto;
    }}
    .project-header, .panel-section {{
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .logo {{
      font-size: 21px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .logo span {{ color: var(--green); }}
    .tagline {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .panel-title {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 10px;
    }}
    .status-row, .info-row, .ctrl-row, .metric-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
      padding: 4px 0;
    }}
    .status-dot {{
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--green);
      box-shadow: 0 0 14px rgba(94, 228, 167, 0.8);
    }}
    button, select {{
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.08);
      color: var(--ink);
      height: 32px;
      border-radius: 7px;
      padding: 0 10px;
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    button.active {{ background: rgba(94, 228, 167, 0.22); border-color: rgba(94, 228, 167, 0.46); }}
    .attempt-list {{
      display: grid;
      gap: 8px;
    }}
    .attempt-item {{
      width: 100%;
      height: auto;
      min-height: 48px;
      text-align: left;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px;
      padding: 9px 10px;
      color: var(--ink);
    }}
    .attempt-item small {{
      color: var(--muted);
      grid-column: 1 / -1;
    }}
    .attempt-item.success {{ border-color: rgba(94, 228, 167, 0.55); }}
    .attempt-item.failed {{ border-color: rgba(255, 122, 89, 0.48); }}
    .timeline {{
      height: 8px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.12);
      overflow: hidden;
      margin-top: 10px;
    }}
    .timeline-fill {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--blue), var(--green));
    }}
    .info-panel {{
      position: absolute;
      right: 18px;
      top: 18px;
      min-width: 270px;
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--glass-2);
    }}
    .help-bar {{
      position: absolute;
      left: 50%;
      bottom: 18px;
      transform: translateX(-50%);
      display: flex;
      align-items: center;
      gap: 14px;
      border-radius: 999px;
      padding: 10px 14px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    kbd {{
      display: inline-grid;
      place-items: center;
      min-width: 22px;
      height: 20px;
      border-radius: 5px;
      padding: 0 5px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid var(--line);
      color: var(--ink);
      font-family: inherit;
      margin-right: 4px;
    }}
    .legend {{
      display: grid;
      grid-template-columns: 12px 1fr;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }}
    .swatch {{ width: 12px; height: 3px; border-radius: 4px; }}
    @media (max-width: 860px) {{
      body {{ overflow: auto; }}
      canvas {{ height: 62vh; }}
      .side-panel, .info-panel, .help-bar {{
        position: static;
        transform: none;
        width: auto;
        max-height: none;
        margin: 12px;
      }}
      .help-bar {{ flex-wrap: wrap; border-radius: 8px; justify-content: center; }}
    }}
  </style>
</head>
<body>
<div class="demo-root">
  <canvas id="canvas"></canvas>
  <div class="side-panel">
    <div class="glass project-header">
      <div class="logo"><span>Praxis-Coop</span> Demo</div>
      <div class="tagline">SONIC command evolution replay · Two G1 payload transport</div>
    </div>
    <div class="glass panel-section">
      <div class="status-row">
        <div style="display:flex;align-items:center;gap:8px;"><div class="status-dot"></div><span id="status-text">Ready</span></div>
        <span id="fps" style="color:var(--muted)">0 fps</span>
      </div>
    </div>
    <div class="glass panel-section">
      <div class="panel-title">Attempts</div>
      <div class="attempt-list" id="attempt-list"></div>
    </div>
    <div class="glass panel-section">
      <div class="panel-title">Simulation</div>
      <div class="ctrl-row">
        <span>Replay</span>
        <div>
          <button id="btn-play" class="active" title="Play or pause replay (Space)">Play</button>
          <button id="btn-restart" title="Restart replay (R)">Restart</button>
        </div>
      </div>
      <div class="ctrl-row">
        <span>Speed</span>
        <select id="speed">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
      <div class="timeline"><div class="timeline-fill" id="timeline-fill"></div></div>
    </div>
    <div class="glass panel-section">
      <div class="panel-title">Legend</div>
      <div class="legend"><div class="swatch" style="background:var(--gold)"></div><span>Payload long box</span></div>
      <div class="legend"><div class="swatch" style="background:var(--blue)"></div><span>G1 robots</span></div>
      <div class="legend"><div class="swatch" style="background:var(--green)"></div><span>Successful path</span></div>
      <div class="legend"><div class="swatch" style="background:var(--red)"></div><span>Failed path</span></div>
    </div>
  </div>
  <div class="glass info-panel">
    <div class="info-row"><span>Scene</span><span id="info-scene"></span></div>
    <div class="info-row"><span>Attempt</span><span id="info-attempt"></span></div>
    <div class="info-row"><span>Skill</span><span id="info-skill"></span></div>
    <div class="info-row"><span>Mode</span><span id="info-mode"></span></div>
    <div class="info-row"><span>Distance</span><span id="info-distance"></span></div>
    <div class="info-row"><span>Time</span><span id="info-time"></span></div>
  </div>
  <div class="glass help-bar">
    <span><kbd>Space</kbd>Play</span>
    <span><kbd>R</kbd>Restart</span>
    <span><kbd>1</kbd><kbd>2</kbd>Attempt</span>
    <span>Drag to pan</span>
    <span>Wheel to zoom</span>
  </div>
</div>
<script id="data" type="application/json">{escaped}</script>
<script>
const data = JSON.parse(document.getElementById("data").textContent);
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const world = computeWorldBounds();
let dpr = window.devicePixelRatio || 1;
let playing = true;
let selectedAttempt = data.attempts.length ? data.attempts.length : 1;
let replayT = 0;
let lastTs = 0;
let zoom = 1;
let panX = 0;
let panY = 0;
let dragging = false;
let dragStart = null;
let fpsFrames = 0;
let fpsTime = 0;

const skillNames = {{
  0: "none",
  1: "approach_formation",
  2: "establish_contact",
  3: "pre_align_object",
  4: "transport_object_path",
  5: "release_and_exit"
}};
const modeNames = {{ "-1": "none", 0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 3: "RUN", 21: "OBJECT_CARRYING" }};
const historyByAttempt = new Map();
for (const row of data.history || []) {{
  const key = Number(row.attempt || 1);
  if (!historyByAttempt.has(key)) historyByAttempt.set(key, []);
  historyByAttempt.get(key).push(row);
}}

function computeWorldBounds() {{
  const xs = [], ys = [];
  const addPoint = (x, y) => {{
    if (Number.isFinite(x) && Number.isFinite(y)) {{ xs.push(x); ys.push(y); }}
  }};
  for (const prim of data.scene.layout_prims || []) {{
    const hw = Math.max(0.08, Number(prim.sx || 1) / 2);
    const hh = Math.max(0.08, Number(prim.sy || 1) / 2);
    const c = Math.cos(Number(prim.yaw || 0));
    const s = Math.sin(Number(prim.yaw || 0));
    for (const dx of [-hw, hw]) for (const dy of [-hh, hh]) {{
      addPoint(Number(prim.x) + dx * c - dy * s, Number(prim.y) + dx * s + dy * c);
    }}
  }}
  for (const robot of Object.values(data.scene.robots || {{}})) addPoint(Number(robot.x), Number(robot.y));
  addPoint(Number(data.scene.object.x), Number(data.scene.object.y));
  addPoint(Number(data.scene.target.x), Number(data.scene.target.y));
  for (const attempt of data.attempts || []) {{
    for (const p of attempt.object_path || []) addPoint(Number(p.x), Number(p.y));
    for (const path of Object.values(attempt.robot_paths || {{}})) {{
      for (const p of path || []) addPoint(Number(p.x), Number(p.y));
    }}
  }}
  if (!xs.length) return {{ minX: -3.3, maxX: 2.9, minY: 0.5, maxY: 9.0 }};
  const padX = Math.max(0.6, (Math.max(...xs) - Math.min(...xs)) * 0.08);
  const padY = Math.max(0.6, (Math.max(...ys) - Math.min(...ys)) * 0.08);
  return {{
    minX: Math.min(...xs) - padX,
    maxX: Math.max(...xs) + padX,
    minY: Math.min(...ys) - padY,
    maxY: Math.max(...ys) + padY
  }};
}}

function resize() {{
  dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}
window.addEventListener("resize", resize);
resize();

function sx(x) {{
  const w = canvas.clientWidth;
  return ((x - world.minX) / (world.maxX - world.minX) * w - w / 2) * zoom + w / 2 + panX;
}}
function sy(y) {{
  const h = canvas.clientHeight;
  return (h - (y - world.minY) / (world.maxY - world.minY) * h - h / 2) * zoom + h / 2 + panY;
}}
function dist(a, b) {{ return Math.hypot(a.x - b.x, a.y - b.y); }}
function lerp(a, b, t) {{ return a + (b - a) * t; }}
function currentRows() {{ return historyByAttempt.get(selectedAttempt) || []; }}
function currentAttempt() {{ return data.attempts.find(a => Number(a.attempt_idx) === selectedAttempt) || data.attempts[0] || {{}}; }}
function maxTime(rows) {{ return rows.length ? Number(rows[rows.length - 1].wall_t || rows[rows.length - 1].step || 1) : 1; }}
function sampleRow(rows, t) {{
  if (!rows.length) return null;
  if (t <= Number(rows[0].wall_t || 0)) return rows[0];
  for (let i = 1; i < rows.length; i++) {{
    const a = rows[i - 1], b = rows[i];
    const ta = Number(a.wall_t || a.step || 0);
    const tb = Number(b.wall_t || b.step || ta + 1);
    if (t <= tb) {{
      const f = Math.max(0, Math.min(1, (t - ta) / Math.max(1e-6, tb - ta)));
      const out = {{...b}};
      for (const key of ["mean_box_x","mean_box_y","mean_robot_a_x","mean_robot_a_y","mean_robot_b_x","mean_robot_b_y","mean_distance","stream_t"]) {{
        out[key] = lerp(Number(a[key] || 0), Number(b[key] || 0), f);
      }}
      return out;
    }}
  }}
  return rows[rows.length - 1];
}}

function drawGrid() {{
  ctx.save();
  ctx.lineWidth = 1;
  for (let x = Math.ceil(world.minX); x <= world.maxX; x++) {{
    ctx.strokeStyle = x === 0 ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.055)";
    ctx.beginPath(); ctx.moveTo(sx(x), sy(world.minY)); ctx.lineTo(sx(x), sy(world.maxY)); ctx.stroke();
  }}
  for (let y = Math.ceil(world.minY); y <= world.maxY; y++) {{
    ctx.strokeStyle = "rgba(255,255,255,0.055)";
    ctx.beginPath(); ctx.moveTo(sx(world.minX), sy(y)); ctx.lineTo(sx(world.maxX), sy(y)); ctx.stroke();
  }}
  ctx.restore();
}}
function drawWall(name, p) {{
  const x = sx(p[0]), y = sy(p[1]);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-p[2]);
  ctx.fillStyle = "rgba(156, 169, 184, 0.75)";
  ctx.strokeStyle = "rgba(230, 238, 249, 0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(-44 * zoom, -9 * zoom, 88 * zoom, 18 * zoom, 3);
  ctx.fill(); ctx.stroke();
  ctx.restore();
}}
function drawLayoutPrim(prim) {{
  if (prim.kind === "target") return;
  if (prim.kind === "long_box") return;
  const px = sx(Number(prim.x)), py = sy(Number(prim.y));
  const w = Math.max(4, Math.abs(sx(Number(prim.x) + Number(prim.sx || 1)) - sx(Number(prim.x))));
  const h = Math.max(4, Math.abs(sy(Number(prim.y) + Number(prim.sy || 1)) - sy(Number(prim.y))));
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-Number(prim.yaw || 0));
  if (prim.kind === "wall") {{
    ctx.fillStyle = "rgba(156, 169, 184, 0.78)";
    ctx.strokeStyle = "rgba(230, 238, 249, 0.22)";
  }} else {{
    ctx.fillStyle = "rgba(130, 145, 168, 0.45)";
    ctx.strokeStyle = "rgba(230, 238, 249, 0.16)";
  }}
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(-w / 2, -h / 2, w, h, 3);
  ctx.fill(); ctx.stroke();
  ctx.restore();
}}
function drawSceneBox(x, y, yaw, size) {{
  const px = sx(x), py = sy(y);
  const w = Math.max(16, Math.abs(sx(x + Number(size?.[0] || 2.2)) - sx(x)));
  const h = Math.max(7, Math.abs(sy(y + Number(size?.[1] || 0.1)) - sy(y)));
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-Number(yaw || 0));
  ctx.fillStyle = "#ffd166";
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(-w / 2, -h / 2, w, h, 3);
  ctx.fill(); ctx.stroke();
  ctx.restore();
}}
function drawPath(points, color, width, dash=[]) {{
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(sx(points[0].x), sy(points[0].y));
  for (const p of points.slice(1)) ctx.lineTo(sx(p.x), sy(p.y));
  ctx.stroke();
  ctx.restore();
}}
function drawRobot(x, y, label, color) {{
  const px = sx(x), py = sy(y);
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = "rgba(255,255,255,0.75)";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "rgba(244,247,251,0.95)";
  ctx.font = "12px ui-sans-serif, system-ui";
  ctx.fillText(label, px + 15, py + 4);
  ctx.restore();
}}
function drawBox(x, y) {{
  const prim = (data.scene.layout_prims || []).find(p => p.kind === "long_box");
  const yaw = prim ? Number(prim.yaw || 0) : Number(data.scene.object.yaw || Math.PI / 2);
  const size = prim ? [Number(prim.sx || 2.2), Number(prim.sy || 0.1)] : data.scene.object.size;
  drawSceneBox(x, y, yaw, size);
}}
function drawScene(row) {{
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  drawGrid();
  if ((data.scene.layout_prims || []).length) {{
    for (const prim of data.scene.layout_prims) drawLayoutPrim(prim);
  }} else {{
    Object.entries(data.scene.walls || {{}}).forEach(([name, p]) => drawWall(name, p));
  }}
  const target = data.scene.target;
  const targetPrim = (data.scene.layout_prims || []).find(p => p.kind === "target");
  const tx = targetPrim ? Number(targetPrim.x) : Number(target.x);
  const ty = targetPrim ? Number(targetPrim.y) : Number(target.y);
  const targetRadius = targetPrim ? Math.max(0.15, Number(targetPrim.sx || 0.6) / 2) : Number(target.tolerance_xy || 0.3);
  ctx.save();
  ctx.fillStyle = "rgba(94, 228, 167, 0.16)";
  ctx.strokeStyle = "#5ee4a7";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(sx(tx), sy(ty), Math.max(12, Math.abs(sx(tx + targetRadius) - sx(tx))), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#5ee4a7";
  ctx.font = "13px ui-sans-serif, system-ui";
  ctx.fillText("target", sx(tx) + 14, sy(ty) - 12);
  ctx.restore();

  data.attempts.forEach((a) => {{
    const selected = Number(a.attempt_idx) === selectedAttempt;
    const color = a.success ? "rgba(94,228,167,0.95)" : "rgba(255,122,89,0.75)";
    drawPath(a.object_path || [], color, selected ? 4 : 2, a.success ? [] : [8, 7]);
    for (const path of Object.values(a.robot_paths || {{}})) drawPath(path, selected ? "rgba(103,162,255,0.55)" : "rgba(103,162,255,0.22)", 1.5, [3, 7]);
  }});

  const obj = data.scene.object;
  const bx = row ? Number(row.mean_box_x) : obj.x;
  const by = row ? Number(row.mean_box_y) : obj.y;
  drawBox(bx, by);
  if (row) {{
    drawRobot(Number(row.mean_robot_a_x), Number(row.mean_robot_a_y), "g1", "#67a2ff");
    drawRobot(Number(row.mean_robot_b_x), Number(row.mean_robot_b_y), "g1_01", "#7cdbff");
  }} else {{
    Object.entries(data.scene.robots).forEach(([id, p], idx) => drawRobot(p.x, p.y, id, idx ? "#7cdbff" : "#67a2ff"));
  }}
}}
function render(ts) {{
  if (!lastTs) lastTs = ts;
  const dt = Math.min(0.05, (ts - lastTs) / 1000);
  lastTs = ts;
  fpsFrames++;
  fpsTime += dt;
  if (fpsTime > 0.5) {{
    document.getElementById("fps").textContent = `${{Math.round(fpsFrames / fpsTime)}} fps`;
    fpsFrames = 0; fpsTime = 0;
  }}
  const rows = currentRows();
  const duration = maxTime(rows);
  if (playing && rows.length) replayT = (replayT + dt * Number(document.getElementById("speed").value)) % Math.max(0.1, duration);
  const row = sampleRow(rows, replayT);
  drawScene(row);
  updateInfo(row, duration);
  requestAnimationFrame(render);
}}
function updateInfo(row, duration) {{
  const attempt = currentAttempt();
  document.getElementById("info-scene").textContent = data.scene.scene_id;
  document.getElementById("info-attempt").textContent = `${{selectedAttempt}} / ${{data.attempts.length}} ${{attempt.success ? "success" : "failed"}}`;
  document.getElementById("info-skill").textContent = row ? skillNames[Math.round(Number(row.active_skill || 0))] || "unknown" : "none";
  document.getElementById("info-mode").textContent = row ? modeNames[Math.round(Number(row.active_mode_g1 || -1))] || "unknown" : "none";
  document.getElementById("info-distance").textContent = row ? `${{Number(row.mean_distance).toFixed(3)}} m` : "-";
  document.getElementById("info-time").textContent = `${{replayT.toFixed(1)}} / ${{duration.toFixed(1)}} s`;
  document.getElementById("timeline-fill").style.width = `${{Math.min(100, replayT / Math.max(0.1, duration) * 100)}}%`;
  document.getElementById("status-text").textContent = playing ? "Playing replay" : "Paused";
  document.getElementById("btn-play").textContent = playing ? "Pause" : "Play";
  document.getElementById("btn-play").classList.toggle("active", playing);
}}
function buildAttemptList() {{
  const box = document.getElementById("attempt-list");
  box.innerHTML = "";
  data.attempts.forEach((attempt, idx) => {{
    const btn = document.createElement("button");
    btn.className = `attempt-item ${{attempt.success ? "success" : "failed"}}`;
    btn.innerHTML = `<span>Attempt ${{attempt.attempt_idx}}</span><span>${{attempt.success ? "success" : "failed"}}</span><small>${{attempt.path_strategy}} · best ${{Number(attempt.best_distance_m || 0).toFixed(3)}} m</small>`;
    btn.onclick = () => {{ selectedAttempt = Number(attempt.attempt_idx); replayT = 0; }};
    box.appendChild(btn);
  }});
}}
buildAttemptList();
document.getElementById("btn-play").onclick = () => playing = !playing;
document.getElementById("btn-restart").onclick = () => replayT = 0;
window.addEventListener("keydown", (e) => {{
  if (e.code === "Space") {{ e.preventDefault(); playing = !playing; }}
  if (e.key.toLowerCase() === "r") replayT = 0;
  const n = Number(e.key);
  if (n >= 1 && n <= data.attempts.length) {{ selectedAttempt = n; replayT = 0; }}
}});
canvas.addEventListener("mousedown", (e) => {{ dragging = true; dragStart = {{x: e.clientX - panX, y: e.clientY - panY}}; }});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", (e) => {{ if (dragging) {{ panX = e.clientX - dragStart.x; panY = e.clientY - dragStart.y; }} }});
canvas.addEventListener("wheel", (e) => {{
  e.preventDefault();
  zoom = Math.max(0.65, Math.min(2.4, zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
}}, {{passive:false}});
requestAnimationFrame(render);
</script>
</body>
</html>
"""
