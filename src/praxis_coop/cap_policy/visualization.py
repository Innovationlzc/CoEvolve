"""HTML report for Code-as-Policy evolution experiments."""

from __future__ import annotations

from dataclasses import asdict
import html
import json
from pathlib import Path

from .contracts import EvolutionRecord


def write_evolution_html(path: str | Path, records: list[EvolutionRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    data = html.escape(json.dumps(payload, ensure_ascii=False), quote=False)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>CoEvolve Code-as-Policy Run</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f7f8fb; color: #18212f; }}
    header {{ padding: 24px 32px; background: #172033; color: white; }}
    h1 {{ margin: 0; font-size: 24px; }}
    main {{ padding: 24px 32px 48px; max-width: 1280px; margin: 0 auto; }}
    .attempt {{ border: 1px solid #d9deea; background: white; border-radius: 8px; margin: 16px 0; overflow: hidden; }}
    .top {{ display: flex; gap: 16px; align-items: center; padding: 14px 16px; border-bottom: 1px solid #e5e8f0; }}
    .badge {{ padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .success {{ background: #dff7e8; color: #116b38; }}
    .fail {{ background: #ffe4e2; color: #9b231b; }}
    .cols {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 0; }}
    pre {{ margin: 0; padding: 16px; overflow: auto; background: #0f1728; color: #dce7ff; min-height: 220px; }}
    .panel {{ padding: 16px; border-left: 1px solid #e5e8f0; }}
    .small {{ color: #5c677a; font-size: 13px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    img {{ max-width: 100%; border: 1px solid #d9deea; border-radius: 6px; }}
  </style>
</head>
<body>
<header>
  <h1>MultiAgentCoEvolve · Code-as-Policy Evolution Trace</h1>
  <div class="small">Real-time robot visualization is the MuJoCo/Isaac viewer; this report shows scripts, SDK calls, verifier evidence, and visual-frame artifacts.</div>
</header>
<main id="app"></main>
<script type="application/json" id="payload">{data}</script>
<script>
const records = JSON.parse(document.getElementById("payload").textContent);
const app = document.getElementById("app");
for (const record of records) {{
  const status = record.evaluation.success ? "success" : "fail";
  const frame = (((record.observation || {{}}).visual_observations || [])[0] || {{}}).frame_path || "";
  const div = document.createElement("section");
  div.className = "attempt";
  div.innerHTML = `
    <div class="top">
      <strong>${{record.policy.script_id}}</strong>
      <span class="badge ${{status}}">${{record.evaluation.success ? "success" : record.evaluation.failure_type}}</span>
      <span class="small">attempt ${{record.policy.attempt}}</span>
    </div>
    <div class="cols">
      <pre><code>${{escapeHtml(record.policy.code || "")}}</code></pre>
      <div class="panel">
        <h3>Evaluation</h3>
        <pre><code>${{escapeHtml(JSON.stringify(record.evaluation, null, 2))}}</code></pre>
        <h3>SDK Calls</h3>
        <pre><code>${{escapeHtml(JSON.stringify(record.execution.skill_calls, null, 2))}}</code></pre>
        ${{frame ? `<h3>External Camera Frame</h3><img src="${{frame}}"/>` : ""}}
        <h3>Spec Candidate</h3>
        <p>${{escapeHtml(record.evolved_spec || "")}}</p>
      </div>
    </div>`;
  app.appendChild(div);
}}
function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}
</script>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
