"""HTML visualization for Stage1 Qwen evolution attempts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_stage1_evolution_html(
    path: Path,
    *,
    attempts: list[dict[str, Any]],
    histories: list[list[dict[str, Any]]],
) -> None:
    payload = {"attempts": attempts, "histories": histories}
    data = html.escape(json.dumps(payload, ensure_ascii=False), quote=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_document(data), encoding="utf-8")


def _document(data_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Praxis-Coop Stage1 Evolution</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #07090f;
      --panel: rgba(18, 24, 34, 0.86);
      --line: rgba(230, 238, 250, 0.15);
      --ink: #f4f7fb;
      --muted: #9aa8b8;
      --green: #5ee4a7;
      --red: #ff7a59;
      --blue: #67a2ff;
      --gold: #ffd166;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }}
    canvas {{ display: block; width: 100vw; height: 100vh; background: #0a0f18; }}
    .panel {{
      position: absolute;
      top: 16px;
      left: 16px;
      width: 360px;
      max-height: calc(100vh - 32px);
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 18px 50px rgba(0,0,0,0.35);
      backdrop-filter: blur(14px);
    }}
    h1 {{ font-size: 20px; margin: 0 0 4px; }}
    .sub {{ color: var(--muted); font-size: 12px; line-height: 1.45; margin-bottom: 12px; }}
    .row {{ display: flex; justify-content: space-between; gap: 12px; padding: 4px 0; font-size: 13px; }}
    .attempts {{ display: grid; gap: 8px; margin-top: 12px; }}
    button {{
      width: 100%;
      text-align: left;
      color: var(--ink);
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      cursor: pointer;
      font: inherit;
    }}
    button.active {{ border-color: rgba(94,228,167,0.65); background: rgba(94,228,167,0.12); }}
    button.failed {{ border-color: rgba(255,122,89,0.30); }}
    .patch {{
      margin-top: 12px;
      padding: 10px;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.05);
      font-size: 12px;
      color: var(--muted);
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .legend {{
      position: absolute;
      right: 16px;
      bottom: 16px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 12px;
      color: var(--muted);
    }}
    .key {{ display: flex; align-items: center; gap: 8px; padding: 3px 0; }}
    .swatch {{ width: 22px; height: 3px; border-radius: 999px; background: var(--blue); }}
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <div class="panel">
    <h1>Praxis-Coop Stage1</h1>
    <div class="sub">Qwen evolution replay · real IsaacLab Stage1 rollout histories</div>
    <div class="row"><span>Attempt</span><strong id="attempt-id"></strong></div>
    <div class="row"><span>Status</span><strong id="status"></strong></div>
    <div class="row"><span>Failure</span><strong id="failure"></strong></div>
    <div class="row"><span>Best distance</span><strong id="best"></strong></div>
    <div class="row"><span>Final distance</span><strong id="final"></strong></div>
    <div class="row"><span>Min box z</span><strong id="minz"></strong></div>
    <div class="attempts" id="attempts"></div>
    <div class="patch" id="patch"></div>
  </div>
  <div class="legend">
    <div class="key"><span class="swatch" style="background: var(--blue)"></span> box trajectory</div>
    <div class="key"><span class="swatch" style="background: var(--green)"></span> robot A</div>
    <div class="key"><span class="swatch" style="background: var(--gold)"></span> robot B</div>
    <div class="key"><span class="swatch" style="background: var(--red)"></span> target</div>
  </div>
  <script id="data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById("data").textContent);
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let selected = 0;
    let t = 0;
    let playing = true;
    const target = {{ x: -1.542, y: 3.75 }};
    const bounds = {{ minX: -3.2, maxX: -0.5, minY: 1.5, maxY: 4.4 }};

    function resize() {{
      canvas.width = Math.floor(window.innerWidth * devicePixelRatio);
      canvas.height = Math.floor(window.innerHeight * devicePixelRatio);
      draw();
    }}
    window.addEventListener("resize", resize);

    function sx(x) {{ return (x - bounds.minX) / (bounds.maxX - bounds.minX) * canvas.width; }}
    function sy(y) {{ return canvas.height - (y - bounds.minY) / (bounds.maxY - bounds.minY) * canvas.height; }}

    function num(v, fallback=0) {{
      const n = Number(v);
      return Number.isFinite(n) ? n : fallback;
    }}

    function drawPath(rows, xKey, yKey, color, width=3, upto=null) {{
      const limit = upto == null ? rows.length : Math.min(rows.length, Math.max(1, upto));
      ctx.beginPath();
      let moved = false;
      for (let i = 0; i < limit; i++) {{
        const x = num(rows[i][xKey], NaN);
        const y = num(rows[i][yKey], NaN);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        if (!moved) {{ ctx.moveTo(sx(x), sy(y)); moved = true; }}
        else ctx.lineTo(sx(x), sy(y));
      }}
      ctx.strokeStyle = color;
      ctx.lineWidth = width * devicePixelRatio;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
    }}

    function dot(x, y, r, color) {{
      ctx.beginPath();
      ctx.arc(sx(x), sy(y), r * devicePixelRatio, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#0a0f18";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = devicePixelRatio;
      for (let x = Math.ceil(bounds.minX); x <= bounds.maxX; x++) {{
        ctx.beginPath(); ctx.moveTo(sx(x), 0); ctx.lineTo(sx(x), canvas.height); ctx.stroke();
      }}
      for (let y = Math.ceil(bounds.minY); y <= bounds.maxY; y++) {{
        ctx.beginPath(); ctx.moveTo(0, sy(y)); ctx.lineTo(canvas.width, sy(y)); ctx.stroke();
      }}
      const rows = data.histories[selected] || [];
      const idx = rows.length ? Math.floor(t) % rows.length : 0;
      drawPath(rows, "box_x", "box_y", "rgba(103,162,255,0.92)", 4, idx + 1);
      drawPath(rows, "robot_a_x", "robot_a_y", "rgba(94,228,167,0.80)", 2, idx + 1);
      drawPath(rows, "robot_b_x", "robot_b_y", "rgba(255,209,102,0.80)", 2, idx + 1);
      dot(target.x, target.y, 8, "rgba(255,122,89,0.95)");
      if (rows[idx]) {{
        dot(num(rows[idx].box_x), num(rows[idx].box_y), 7, "rgba(103,162,255,1)");
        dot(num(rows[idx].robot_a_x), num(rows[idx].robot_a_y), 5, "rgba(94,228,167,1)");
        dot(num(rows[idx].robot_b_x), num(rows[idx].robot_b_y), 5, "rgba(255,209,102,1)");
      }}
    }}

    function updatePanel() {{
      const a = data.attempts[selected] || {{}};
      document.getElementById("attempt-id").textContent = String(a.attempt || selected + 1);
      document.getElementById("status").textContent = a.success ? "success" : "failed";
      document.getElementById("failure").textContent = a.failure || "none";
      document.getElementById("best").textContent = Number(a.best_target_distance ?? Infinity).toFixed(3);
      document.getElementById("final").textContent = Number(a.final_target_distance ?? Infinity).toFixed(3);
      document.getElementById("minz").textContent = Number(a.min_physical_box_z ?? NaN).toFixed(3);
      document.getElementById("patch").textContent = a.patch ? JSON.stringify(a.patch, null, 2) : "No patch recorded for this attempt.";
      const box = document.getElementById("attempts");
      box.innerHTML = "";
      data.attempts.forEach((attempt, i) => {{
        const b = document.createElement("button");
        b.className = `${{i === selected ? "active " : ""}}${{attempt.success ? "" : "failed"}}`;
        b.innerHTML = `Attempt ${{attempt.attempt}} · ${{attempt.success ? "success" : attempt.failure || "failed"}}<br><small>best ${{Number(attempt.best_target_distance ?? Infinity).toFixed(3)}} m</small>`;
        b.onclick = () => {{ selected = i; t = 0; updatePanel(); draw(); }};
        box.appendChild(b);
      }});
    }}

    function frame() {{
      if (playing) t += 0.65;
      draw();
      requestAnimationFrame(frame);
    }}
    window.addEventListener("keydown", (event) => {{
      if (event.code === "Space") playing = !playing;
      if (event.key === "r" || event.key === "R") t = 0;
      const n = Number(event.key);
      if (n >= 1 && n <= data.attempts.length) {{ selected = n - 1; t = 0; updatePanel(); }}
    }});
    resize();
    updatePanel();
    frame();
  </script>
</body>
</html>
"""
