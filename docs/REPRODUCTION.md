# Current Reproduction

This repo tracks the current Praxis-Coop progress:

> Official GEAR-SONIC motor deployment plus a dual-G1 MuJoCo scene generated
> from the user-authored manual layout.

The active reproducible milestone is motor-layer validation, not full box
transport yet:

- two namespaced G1 robots in the manual layout;
- official `gear_sonic_deploy` processes for each robot;
- synchronized planner commands over ZMQ;
- guarded drop using lowcmd readiness plus planner heartbeat;
- motor-side upper-body pose lock for payload-support arm posture;
- high parallel carry pose with forearm approximately level.

## Repository Inputs

Tracked scene:

```text
assets/manual_scenes/two_g1_box_push_layout.usda
```

Local runtime setup is documented in `../ENV_SETUP.md`. This reproduction page
assumes the ignored SONIC runtime already exists at:

```text
third_party/GR00T-WholeBodyControl
```

## Known-Good: Dual-G1 No-Box Synchronized Walk

This is the current validated motor-layer baseline. It uses the manual-layout
MuJoCo scene, two official GEAR-SONIC deploy processes, and synchronized planner
commands. There is no payload and no upper-body pose lock in this run.

Use this command sequence before testing carry-pose or payload-contact runs.
Keep the heartbeat path identical between Terminal 1 and Terminal 4.

Terminal 1:

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

Terminal 2:

```bash
cd /home/admin1/Project/CoEvolve
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id front --dds-topic-prefix front --zmq-port 5556 --zmq-out-port 5557
```

Terminal 3:

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

- Terminal 1 drops the virtual support band only after both deploys are ready
  and planner heartbeat is fresh.
- Both robots stand and walk forward together under official SONIC control.
- Treat this as dual-robot motor-layer evidence, not box-transport evidence.

## Dual-G1 High Carry Hold

Use four terminals. Keep the heartbeat path identical between Terminal 1 and
Terminal 4.

Terminal 1:

```bash
cd /home/admin1/Project/CoEvolve
rm -f results/deployment/dual_sonic_high_parallel_live/planner_heartbeat.json

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_sonic_manual_layout_sim.py \
  --interface auto \
  --upper-body-lock-pose high_parallel_carry \
  --upper-body-lock-kp 3.5 \
  --upper-body-lock-kd 0.8 \
  --upper-body-lock-ramp-s 30 \
  --upper-body-lock-start-after-drop-s 6 \
  --auto-drop-when-ready \
  --lowcmd-warmup-s 2 \
  --min-drop-time-s 4 \
  --planner-heartbeat-file results/deployment/dual_sonic_high_parallel_live/planner_heartbeat.json \
  --planner-heartbeat-max-age-s 1
```

Terminal 2:

```bash
cd /home/admin1/Project/CoEvolve
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id front --dds-topic-prefix front --zmq-port 5556 --zmq-out-port 5557
```

Terminal 3:

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
  --stabilize-seconds 5 \
  --upper-body-start-pose sonic_default \
  --upper-body-pose high_parallel_carry \
  --upper-body-ramp-seconds 30 \
  --hand-pose open \
  --stand-seconds 3600 \
  --walk-meters 0 \
  --final-idle-seconds 0 \
  --no-stop \
  --heartbeat-file results/deployment/dual_sonic_high_parallel_live/planner_heartbeat.json \
  --output-dir results/deployment/dual_sonic_high_parallel_live
```

## Dual-G1 High Carry Walk

Use a separate output directory, but again keep Terminal 1 and Terminal 4
heartbeat paths identical.

Terminal 1:

```bash
cd /home/admin1/Project/CoEvolve
rm -f results/deployment/dual_sonic_high_parallel_walk_live/planner_heartbeat.json

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_sonic_manual_layout_sim.py \
  --interface auto \
  --upper-body-lock-pose high_parallel_carry \
  --upper-body-lock-kp 3.5 \
  --upper-body-lock-kd 0.8 \
  --upper-body-lock-ramp-s 30 \
  --upper-body-lock-start-after-drop-s 6 \
  --auto-drop-when-ready \
  --lowcmd-warmup-s 2 \
  --min-drop-time-s 4 \
  --planner-heartbeat-file results/deployment/dual_sonic_high_parallel_walk_live/planner_heartbeat.json \
  --planner-heartbeat-max-age-s 1
```

Terminal 4:

```bash
cd /home/admin1/Project/CoEvolve
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_official_sonic_motor_preflight.py \
  --front-port 5556 --back-port 5558 \
  --start-pulse-seconds 8 \
  --stabilize-seconds 5 \
  --upper-body-start-pose sonic_default \
  --upper-body-pose high_parallel_carry \
  --upper-body-ramp-seconds 30 \
  --hand-pose open \
  --stand-seconds 20 \
  --walk-meters 1.0 \
  --walk-speed 0.06 \
  --walk-mode slow_walk \
  --movement-x 1 --movement-y 0 \
  --facing-x 1 --facing-y 0 \
  --final-idle-seconds 3600 \
  --no-stop \
  --heartbeat-file results/deployment/dual_sonic_high_parallel_walk_live/planner_heartbeat.json \
  --output-dir results/deployment/dual_sonic_high_parallel_walk_live
```

## Evidence Note

These commands are real official SONIC/GEAR-SONIC low-level motor execution in
MuJoCo. They are not old Isaac proxy/root-teleport results. See
`SONIC_INTEGRATION.md` for the backend boundary and next payload/contact
milestone.
