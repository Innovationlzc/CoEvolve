# CoEvolve Progress Log

This file records stage-level milestones. Use Git commits for exact diffs and
`HALyPO_REPRO_TODO.md` for the current plan, known gaps, and next decisions.

## 2026-06-02

- Initialized local Git management for the CoEvolve HALyPO reproduction workspace.
- M0 PyTorch two-agent toy game is complete: HALyPO projection unit tests pass,
  rationality gap decreases, and projection constraint checks are tracked.
- M1 2D two-agent transport proxy is complete: independent vs HALyPO comparison
  runs locally and records return/gap/alignment/GCR metrics.
- M2 Isaac Two-G1 Box Push Lite is complete as a plumbing/debug task: registered
  Isaac task, proxy action adapter, base-command adapter smoke, M2 proxy training,
  and Newton evaluation. It is explicitly not physical G1 contact pushing.
- M3 OSP-Alignment-Lite first scaffold is complete: paper-like OSP scene, 20 kg
  object, gate, G1 target markers, human/proxy markers, 11D command interface,
  contact scores, 210D observation shape, temporary root-stabilized tracker, and
  scripted smoke success at step 29.

Current important boundary: M3 is still a paper-bridge scene, not the full paper
controller. The next fidelity jump is replacing the temporary tracker with Isaac
Lab G1 Pink IK or a true WBC command adapter.

