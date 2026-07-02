# TODO

Current direction: use official SONIC / GEAR-SONIC as the frozen motor layer,
then build verified cooperative SkillGraph evolution above it.

## M0: Workspace Cleanup

- [x] Move old HALyPO/MARL reproduction material under ignored `HaLyPO/`.
- [x] Track the user-authored scene at `assets/manual_scenes/two_g1_box_push_layout.usda`.
- [x] Keep `third_party/GR00T-WholeBodyControl`, `models/`, generated deployment outputs, and local PDFs out of git.
- [x] Remove old proxy/HTML/replay scripts from the root `scripts/` entry points.
- [x] Document the current method, runtime boundary, and reproduction path.

## M1: Official SONIC Motor Baseline

- [x] Build a dual-G1 MuJoCo scene from the tracked manual USD layout.
- [x] Run two official `gear_sonic_deploy` processes with separate robot IDs,
  DDS topic prefixes, and ZMQ ports.
- [x] Add lowcmd readiness plus planner-heartbeat gating before releasing the
  MuJoCo support band.
- [x] Visually pass the known-good no-box dual-G1 synchronized stand/walk run.
- [x] Record the exact four-terminal baseline command in `README.md` and
  `docs/REPRODUCTION.md`.

## M2: Payload-Contact Preparation

- [ ] Add the physical `long_box` contact trial only after the no-box motor
  baseline remains stable.
- [ ] Tune contact geometry, friction, and payload mass in local experiments
  before adding any payload run to public reproduction docs.
- [ ] Keep failed or unstable payload trials out of README and GitHub-facing
  reproduction commands.

## M3: Payload Contact Transport

- [ ] Add `long_box` to the validated dual-G1 MuJoCo runtime path.
- [ ] Tune hand/contact geometry, friction, and payload mass so the task is
  physically credible rather than proxy/root-teleport success.
- [ ] Expose verifier signals for robot fall, object progress, contact loss,
  yaw error, collision, and wall contact.
- [ ] Complete Stage 1: straight synchronized transport toward the wall/target
  region in the user's authored layout.
- [ ] Then add Stage 2 turning and Stage 3 final straight placement.

## M4: Verified SkillGraph Evolution

- [x] Define typed schemas for taxonomy, roles, SkillGraph, verifier reports,
  patches, and deployment specs.
- [x] Add text-first static/dynamic SpecMemory layout.
- [x] Wire Qwen JSON parsing for critic/coder/spec outputs.
- [ ] Promote the official dual SONIC backend into the Stage 1 SkillGraph runner.
- [ ] Save only verifier-backed success/failure deployment specs into dynamic
  SpecMemory.
- [ ] Add regression checks before promoting a learned spec.

## M5: Models And Paper Experiments

- [ ] Finish local Qwen3-8B and Qwen3-VL-8B downloads under ignored `models/`.
- [ ] Run ablations: no spec retrieval, fixed SkillGraph, no role allocation,
  no verifier feedback, and text-only memory.
- [ ] Report only physically valid official-SONIC/contact runs as experiment
  evidence.
- [ ] Keep old Isaac proxy/root-teleport results labeled as invalid historical
  debugging artifacts.
