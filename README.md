# CoEvolve / Praxis-Coop

Praxis-Coop explores verified cooperative skill orchestration above a frozen
humanoid whole-body controller.

Current direction:

> Official SONIC / GEAR-SONIC motor execution + cooperative SkillGraph
> orchestration + verifier-gated deployment SpecMemory.

The old HALyPO / MARL reproduction work is local archive material only. The
current repo state is organized for the new SONIC-based project.

## Current Reproducible Milestone

The reproducible milestone in this repository is:

- generate a dual-G1 MuJoCo scene from the user-authored manual USD layout;
- run two official `gear_sonic_deploy` processes, one per G1;
- send synchronized ZMQ planner commands;
- use lowcmd readiness plus planner heartbeat before dropping the virtual
  support band;
- hold a high parallel carry arm posture while SONIC controls standing/walking.

Start here:

```text
docs/REPRODUCTION.md
```

## Known-Good Run: Dual-G1 No-Box Synchronized Walk

This is the current stable baseline that has visually passed: two official
GEAR-SONIC deploy processes control two G1 robots in the generated manual-layout
MuJoCo scene, with no box and no upper-body pose lock. Use this before testing
high carry or payload contact.

Use four terminals. Keep the heartbeat path identical in Terminal 1 and
Terminal 4.

Terminal 1, dual-G1 MuJoCo runtime:

```bash
cd /home/admin1/Project/CoEvolve
rm -f results/deployment/dual_sonic_selftest_guarded_walk/planner_heartbeat.json

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_sonic_manual_layout_sim.py \
  --interface auto \
  --upper-body-lock-pose none \
  --auto-drop-when-ready \
  --lowcmd-warmup-s 2 \
  --min-drop-time-s 4 \
  --planner-heartbeat-file results/deployment/dual_sonic_selftest_guarded_walk/planner_heartbeat.json \
  --planner-heartbeat-max-age-s 1
```

Terminal 2, front G1 deploy:

```bash
cd /home/admin1/Project/CoEvolve
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id front --dds-topic-prefix front --zmq-port 5556 --zmq-out-port 5557
```

Terminal 3, back G1 deploy:

```bash
cd /home/admin1/Project/CoEvolve
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id back --dds-topic-prefix back --zmq-port 5558 --zmq-out-port 5559
```

Terminal 4, after Terminal 2 and 3 print `Init Done`:

```bash
cd /home/admin1/Project/CoEvolve
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_official_sonic_motor_preflight.py \
  --front-port 5556 --back-port 5558 \
  --start-pulse-seconds 8 \
  --stand-seconds 10 \
  --walk-meters 1.0 \
  --walk-speed 0.10 \
  --walk-mode slow_walk \
  --movement-x 1 --movement-y 0 \
  --facing-x 1 --facing-y 0 \
  --upper-body-pose none \
  --hand-pose none \
  --final-idle-seconds 3600 \
  --no-stop \
  --heartbeat-file results/deployment/dual_sonic_selftest_guarded_walk/planner_heartbeat.json \
  --output-dir results/deployment/dual_sonic_selftest_guarded_walk
```

Expected behavior:

- Terminal 1 auto-drops only after both deploys are ready and heartbeat is fresh.
- Both robots stand, then walk forward together under official SONIC control.
- This run has no payload contact and should be treated as motor-layer evidence,
  not box-transport success.

## Layout

| Path | Purpose |
|---|---|
| `assets/manual_scenes/` | tracked user-authored manual scene used for current tests |
| `src/praxis_coop/` | current implementation package |
| `scripts/run_dual_sonic_manual_layout_sim.py` | dual-G1 MuJoCo runtime wrapper |
| `scripts/run_official_sonic_deploy_zmq_manager.sh` | official GEAR-SONIC deploy launcher |
| `scripts/run_dual_official_sonic_motor_preflight.py` | synchronized planner command publisher |
| `docs/README.md` | documentation index and ownership map |
| `docs/REPRODUCTION.md` | current end-to-end commands |
| `docs/SYSTEM_DESIGN.md` | method architecture |
| `docs/SONIC_INTEGRATION.md` | SONIC adapter boundary |
| `results/spec_memory/static/` | curated reusable deployment specs |

Local-only paths, ignored by git:

| Path | Reason |
|---|---|
| `third_party/GR00T-WholeBodyControl/` | external SONIC runtime |
| `models/` | Qwen checkpoints |
| `results/deployment/` outputs | generated experiment artifacts |
| `HaLyPO/` | old archived work, not current evidence |
| `source/*.pdf` | local paper PDFs |

## Quick Check

```bash
cd /home/admin1/Project/CoEvolve
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python -m pytest -q
```

Expected current result:

```text
21 passed
```

## Boundary

The current valid motor evidence is official SONIC/GEAR-SONIC in MuJoCo. Older
Isaac proxy/root-teleport runs are not paper evidence and should not be used as
the active experiment path.
