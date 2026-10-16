"""Collaborative evolution agent for policy scripts and text specs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import EvaluationReport, ObservationBundle, PolicyScript, write_json


@dataclass
class CollaborativeEvolutionAgent:
    """Repair failed policy scripts and summarize reusable text experience."""

    output_dir: Path

    def evolve(
        self,
        policy: PolicyScript,
        report: EvaluationReport,
        observation: ObservationBundle | None = None,
    ) -> tuple[PolicyScript, str]:
        spec = self._spec_from_report(policy, report, observation)
        if report.success:
            return policy, spec
        repaired = self._repair_policy(policy, report)
        write_json(self.output_dir / "evolved_policy.json", repaired)
        return repaired, spec

    def _repair_policy(self, policy: PolicyScript, report: EvaluationReport) -> PolicyScript:
        code = policy.code
        rationale = f"Repair after {report.failure_type}: {'; '.join(report.recommendations)}"
        if report.failure_type == "payload_drop":
            code = code.replace("payload_z=1.14", "payload_z=1.18")
            code = code.replace("speed_mps=0.15", "speed_mps=0.10")
            code = code.replace("speed_mps=0.35", "speed_mps=0.25")
            code = code.replace("distance_m=1.5", "distance_m=0.8")
        elif report.failure_type == "payload_not_supported":
            code = code.replace("hold_s=8.0", "hold_s=14.0")
            code = code.replace("distance_m=1.5", "distance_m=0.8")
            code = code.replace("speed_mps=0.15", "speed_mps=0.10")
            code = code.replace("speed_mps=0.35", "speed_mps=0.25")
        elif report.failure_type == "payload_wall_contact":
            code = code.replace("distance_m=1.5", "distance_m=0.6")
            code = code.replace("speed_mps=0.15", "speed_mps=0.08")
            code = code.replace("speed_mps=0.35", "speed_mps=0.20")
        elif report.failure_type == "no_payload_progress":
            code = code.replace("distance_m=0.75", "distance_m=1.0")
            code = code.replace("speed_mps=0.15", "speed_mps=0.18")
            code = code.replace("speed_mps=0.35", "speed_mps=0.45")
        elif report.failure_type == "runtime_state_missing":
            rationale = "Runtime state missing; do not change script until Term1 writes live runtime_state.json."
        else:
            code = code.replace("speed_mps=0.15", "speed_mps=0.10")
            code = code.replace("speed_mps=0.35", "speed_mps=0.25")

        attempt = policy.attempt + 1
        return PolicyScript(
            script_id=f"box_carry_attempt_{attempt:02d}",
            attempt=attempt,
            code=code,
            plan=policy.plan,
            subtasks=policy.subtasks,
            parent_script_id=policy.script_id,
            rationale=rationale,
        )

    def _spec_from_report(
        self,
        policy: PolicyScript,
        report: EvaluationReport,
        observation: ObservationBundle | None,
    ) -> str:
        visual = ""
        if observation and observation.visual_observations:
            last = observation.visual_observations[-1]
            visual = f"; visual_source={last.source}; visual_frame={last.frame_path or '<none>'}"
        if report.success:
            return (
                f"success_spec: script={policy.script_id}; keep static SONIC SDK calls "
                "synchronized; require both robots to maintain payload contact before "
                "increasing distance or adding turns."
                + visual
            )
        return (
            f"failure_spec: script={policy.script_id}; failure={report.failure_type}; "
            f"failed_subtask={report.failed_subtask or 'unknown'}; "
            f"recommendation={'; '.join(report.recommendations)}"
            + visual
        )
