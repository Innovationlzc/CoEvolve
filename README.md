# CoEvolve / Praxis-Coop

Praxis-Coop studies verified cooperative skill orchestration above a frozen
humanoid motor layer. The current motor backend is official SONIC /
GEAR-SONIC, run through two `gear_sonic_deploy` processes and a dual-G1 MuJoCo
scene generated from the user-authored manual layout.

Old HALyPO/MARL reproduction work is archived locally under ignored `HaLyPO/`
and is not current evidence.

## Current Status

Validated current milestone:

- dual G1 robots in the manual-layout MuJoCo scene;
- official SONIC deployment for both robots;
- synchronized ZMQ planner commands;
- guarded drop after both lowcmd streams and planner heartbeat are ready;
- static skill library for no-box motor validation:
  `parallel_forward`, `carry_forward`, `carry_backward`,
  `carry_lateral_left/right`, `carry_turn_in_place`,
  `carry_place_crouch`, and `carry_recover_stand`.

This is not yet a completed payload transport result. The next milestone is a
5kg `long_box` stability/contact trial using the same official SONIC runtime.

## Layout

| Path | Purpose |
|---|---|
| `assets/manual_scenes/two_g1_box_push_layout.usda` | tracked user-authored scene |
| `src/praxis_coop/` | Praxis-Coop implementation |
| `scripts/` | runnable local entry points |
| `docs/STATIC_SONIC_SKILLS.md` | current successful skill commands |
| `docs/REPRODUCTION.md` | concise reproduction boundary |
| `docs/SYSTEM_DESIGN.md` | method architecture |
| `docs/SONIC_INTEGRATION.md` | SONIC backend boundary |
| `results/spec_memory/static/` | curated static specs for later evolution |

Ignored local runtime/output paths include `third_party/GR00T-WholeBodyControl/`,
`models/`, `results/deployment/`, `HaLyPO/`, and local PDFs under `source/`.

## Setup

The repo assumes:

```text
/home/admin1/Project/CoEvolve
/home/admin1/Project/IsaacLab/env_isaaclab
third_party/GR00T-WholeBodyControl
```

Run tests:

```bash
cd /home/admin1/Project/CoEvolve
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python -m pytest -q
```

List static SONIC skills:

```bash
cd /home/admin1/Project/CoEvolve
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py --list-skills
```

On a new machine, patch the local SONIC runtime before the dual-G1 workflow:

```bash
cd /home/admin1/Project/CoEvolve/third_party/GR00T-WholeBodyControl
git apply /home/admin1/Project/CoEvolve/patches/gear_sonic_multi_robot_dds_prefix.patch
git apply /home/admin1/Project/CoEvolve/patches/gear_sonic_heading_increment.patch
cd gear_sonic_deploy
source scripts/setup_env.sh
just --justfile .justfile build
```

For the current four-terminal workflow, tuned static-skill commands, and the
next 5kg box stability trial, use:

```text
docs/STATIC_SONIC_SKILLS.md
```

## Next Step

Add the physical 5kg `long_box` to the validated dual-G1 runtime and first test
static support stability. Only after the box is physically stable should the
same static skills be used for carry-forward/contact transport.
