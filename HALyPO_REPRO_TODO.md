# CoEvolve Plan

Source paper: `source/HALyPO.pdf`

Goal: build a two-Unitree-G1 co-evolution framework inspired by HALyPO, then use
a small validation set to test whether Lyapunov-stabilized heterogeneous MARL
improves coordination before attempting full paper-scale reproduction.

## Current Position

- Isaac Lab / Isaac Sim runtime is installed outside this repo.
- Official HALyPO code and datasets are not available.
- The paper's full benchmark is too large for the first milestone: 9 scenarios,
  2e9 action steps, WBC, MoCap, and real-world G1 deployment.
- M0 and M1 are complete as local validation baselines.
- M2 now has a registered Isaac task, proxy independent-vs-HALyPO training, and
  Newton policy evaluation. The current M2 action directly moves the box and
  a first `base_command` adapter that moves the G1 base visuals before pushing
  through a virtual contact zone. The robots are not walking, moving their feet,
  or pushing through true rigid-body contact yet.
- M3 now has a first `OSP-Alignment-Lite` paper-bridge task. It creates an OSP
  directional-alignment scene, a 20 kg rigid object, gate walls, G1 wrist target
  markers, scripted human/proxy partner markers, an 11D command interface, 210D
  policy observation shape, contact scores, and a temporary root-stabilized arm
  tracker. It reaches the scripted smoke goal, but it is still not true G1
  walking, WBC, or physical hand-object manipulation.
- Paper-faithful reproduction requires a real scene/task rebuild, not the current
  proxy demo. Do this programmatically as Isaac Lab tasks/configs; use manual USD
  authoring only for complex object/obstacle geometry if needed.

## Paper Implementation Facts

Extracted from `source/HALyPO.pdf`, especially Section 6 and Appendix B:

- Task families: OSP, SCT, SLH.
- Scenarios: OSP alignment / turnaround / corner entry; SCT narrow gate /
  S-shaped path / U-shaped path; SLH facing mode / lateral shuffle / pivoting.
- Control hierarchy: top-level A* or VLM path planner, 2 Hz tactical MARL
  policy, 50 Hz G1 WBC execution.
- Observation: 210D per agent, including look-ahead waypoints, self
  proprioception, partner state, object geometry, contact feedback, temporal
  snapshots, and 36-ray proximity.
- Command: 11D MARL command with base velocity/yaw, CoM height, torso pitch,
  and wrist XYZ deltas.
- Reward: path progress minus object tilt penalty and drop/contact-loss penalty.
- Simulation constants: 0.5 s MARL step, 10 m synthetic sensing range, 20 kg box
  or 10 kg board, 3000 max simulation steps per episode.

## Research Target

Two G1 agents should co-evolve complementary coordination policies for physically
coupled object transport. The system should measure and reduce the rationality
gap between independent agent updates and team-level optimization:

```text
V(theta) = 0.5 * ||u_ind(theta) - u_team(theta)||^2
```

HALyPO modifies the actor update by projecting `u_ind` onto a Lyapunov-stable
half-space:

```text
h = grad_theta V(theta)
psi = dot(h, detach(u_ind)) + sigma * detach(V)
lambda = max(0, psi / (||h||^2 + eps))
d_star = detach(u_ind) - lambda * h
```

## Minimal Validation Set

Use these milestones in order. Each milestone should produce a short result
table and one plot, not a long report.

| ID | Validation | Purpose | Pass signal |
|---|---|---|---|
| M0 | PyTorch toy two-agent game | prove HALyPO projection is implemented correctly | done: `V` decreases; projection constraint holds |
| M1 | 2D two-agent transport env | test HALyPO loop without Isaac complexity | done: HALyPO lowers gap/conflict vs independent baseline |
| M2 | Isaac Two-G1 Box Push Lite | validate the core idea with two G1 bodies | proxy and base-command adapter smoke pass |
| M3 | One paper-like OSP scenario | bridge toward HALyPO benchmark | success rate and gap metrics are reproducible |

M2 is the first "real" project demo. Keep it intentionally small:

- two G1 agents;
- one rigid box or board;
- one goal region;
- low-dimensional local observations first;
- action can start as base velocity/yaw commands before full 11D wrist command;
- reward = progress to goal - object tilt/drop/contact loss penalties.

## Implementation Plan

### P0. Project Skeleton

- [x] Initialize CoEvolve as the external project workspace.
- [x] Keep Isaac Lab runtime at `/home/admin1/Project/IsaacLab`.
- [x] Add package layout for algorithms, envs, configs, scripts, and tests.
- [ ] Add `morning9393/HAPPO-HATRPO` as a baseline reference or vendored
  third-party module after checking its API.

### P1. Algorithm Kernel

- [x] Implement parameter flatten helpers.
- [x] Implement HALyPO projection kernel with double backprop / HVP.
- [x] Add unit tests for inactive projection, active projection, shape matching,
  and constraint satisfaction.
- [x] Add PCGrad baseline helper for comparison.

### P2. Non-Isaac Minimum

- [x] Build a tiny two-agent differentiable game.
- [x] Build a 2D cooperative transport policy-gradient proxy.
- [x] Run independent baseline and HALyPO on M0.
- [x] Run independent baseline and HALyPO on M1 proxy.
- [ ] Run HAPPO/HATRPO baseline and HALyPO on M1.
- [x] Track return, rationality gap `V`, gradient alignment, and GCR in M0/M1.

### P3. Isaac Two-G1 Box Push Lite

- [x] Create an Isaac Lab external task registered from CoEvolve.
- [ ] Start from a known G1 task such as `Isaac-Velocity-Flat-G1-Play-v0`.
- [x] Add a CoEvolve wrapper for G1 random-agent smoke/eval visualization.
- [x] Spawn two G1 agents and one object.
- [x] Implement reset, observation, proxy action adapter, reward, and success metric.
- [x] First run with scripted/randomized proxy actions.
- [x] Connect M1-style independent vs HALyPO training to M2 proxy action.
- [x] Load trained HALyPO proxy policy into Isaac eval.
- [x] Add first G1 base-command adapter smoke.
- [ ] Replace virtual contact adapter with true G1 contact or WBC command adapter.
- [ ] Stop treating M2 as paper reproduction once the real OSP task starts; keep
  it as a plumbing/debug task only.

### P4. Paper Bridge

- [x] Build `M3-OSP-Alignment-Lite` as the first paper-like scene.
- [x] Add a rigid object and narrow directional entrance matching OSP.
- [x] Add human/proxy partner with scripted or learned hand/wrist trajectory.
- [x] Add G1 end-effector targets and contact checks.
- [x] Add paper-inspired 210D observation.
- [x] Add 11D command interface.
- [x] Implement a temporary IK/PD-style tracker for smoke testing.
- [ ] Replace the temporary tracker with true G1 WBC or Isaac Lab G1 Pink IK.
- [ ] Later expand to SCT/SLH and ablations.

## Metrics

Track these from the first prototype:

- cumulative return;
- success rate;
- rationality gap `V(theta)`;
- gradient alignment `cos(u_ind, u_team)`;
- gradient conflict rate;
- projection active rate;
- projection constraint violation rate.

## Known Gaps

- Full G1 WBC policy and 11D command tracker are not available yet.
- M3 currently uses `G1_MINIMAL_CFG` because the first attempt to use the G1
  29DOF locomanipulation asset blocked on USD asset loading. Switch back to
  `G1_29DOF_CFG` / G1 Pink IK once that asset path is reliable.
- M2 G1 bodies are currently kept upright as visual actors during proxy action
  evaluation; this intentionally avoids the no-controller falling behavior.
- Current M2 base-command motion is still not physical robot pushing: the G1
  base visuals move under command, but feet/joints remain fixed and box motion
  comes from a virtual contact adapter.
- Without a WBC or equivalent IK/PD tracker, G1 feet/hands will not perform the
  paper's physical manipulation behavior. The next scene must expose this as an
  explicit controller dependency rather than hiding it behind proxy motion.
- Paper environment geometry, reset distributions, seeds, and exact training
  parallelism are underspecified.
- Kit GUI (`--viz kit`) currently fails; use headless or `--viz newton`.
- Isaac empty-stage visualizer tests are misleading; test visualizers on real
  tasks such as Cartpole or G1.
- Training should stay headless by default with `--viz none`; open visualization
  through periodic policy evaluation scripts using `--viz newton`.

## Near-Term Decision Log

- Keep Isaac Lab outside CoEvolve.
- Use `morning9393/HAPPO-HATRPO` as the first HAPPO/HATRPO baseline candidate.
- Do not attempt full 2e9-step paper reproduction until M0-M2 pass.
- Keep docs short; put detailed derivations in code comments/tests only when
  needed.
