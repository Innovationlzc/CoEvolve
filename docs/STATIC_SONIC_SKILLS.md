# Static SONIC Skill Commands

This page is the canonical command sheet for the current successful no-payload
static skill library. All commands use official SONIC / GEAR-SONIC in MuJoCo,
not Isaac proxy/root-teleport control.

Before running on a new machine, apply the required SONIC runtime patches from
`docs/SONIC_INTEGRATION.md`. In particular, `--dds-topic-prefix` must appear in
the C++ deploy source, not only in CoEvolve scripts.

## Terminals 1-3

Use these terminals for carry-ready static skills.

Terminal 1:

```bash
cd /home/admin1/Project/CoEvolve
OUT=results/deployment/static_skill_live
mkdir -p "$OUT"
rm -f "$OUT/planner_heartbeat.json" "$OUT/runtime_state.json"

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_sonic_manual_layout_sim.py \
  --interface auto \
  --upper-body-lock-pose elbow110_carry_ready \
  --upper-body-lock-kp 24 \
  --upper-body-lock-kd 1.5 \
  --upper-body-lock-ramp-s 6 \
  --upper-body-lock-start-after-drop-s 1 \
  --planner-heartbeat-drop-phase-keyword static_skill_pre_stand \
  --auto-drop-when-ready \
  --lowcmd-warmup-s 4 \
  --min-drop-time-s 4 \
  --planner-heartbeat-file "$OUT/planner_heartbeat.json" \
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

## Term4 Skills

Default validation distances are in the 1.5-2m range with deployment-safe
speeds. Override `--distance-m`, `--speed-mps`, `--yaw-deg`, or
`--place-height` when tuning.

Forward with carry-ready arms:

```bash
cd /home/admin1/Project/CoEvolve
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_forward \
  --distance-m 1.5 \
  --speed-mps 0.15
```

Backward with carry-ready arms:

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_backward \
  --distance-m 1.5 \
  --speed-mps 0.10
```

Left/right lateral movement:

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_lateral_left \
  --distance-m 1.5 \
  --speed-mps 0.45

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_lateral_right \
  --distance-m 1.5 \
  --speed-mps 0.45
```

Turn in place. Convention: `-yaw` is clockwise, `+yaw` is counterclockwise.

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_turn_in_place \
  --yaw-deg -90 \
  --yaw-rate-degps 5

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_turn_in_place \
  --yaw-deg 90 \
  --yaw-rate-degps 5
```

Equivalent explicit aliases:

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py --skill carry_turn_clockwise_90

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py --skill carry_turn_counterclockwise_90
```

Knee-bend place. The currently useful deep bend uses `--place-height 0.55`.
This skill does not move forward; it only changes the height command and lets
the runtime-side lower-body assist hold both robots in a stationary place pose.

```bash
OUT=results/deployment/static_skill_live

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py \
  --skill carry_place_crouch --robots front back \
  --front-port 5556 --back-port 5558 \
  --place-start-height 0.80 \
  --place-height 0.55 \
  --place-ramp-seconds 2 \
  --place-hold-seconds 10 \
  --start-pulse-seconds 0 \
  --stand-seconds 1 \
  --final-idle-seconds 0 \
  --heartbeat-file "$OUT/planner_heartbeat.json" \
  --output-dir "$OUT/carry_place_crouch_h055" \
  --no-stop
```

The same skill through the convenience wrapper:

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_place_crouch \
  --place-start-height 0.80 \
  --place-height 0.55 \
  --place-ramp-seconds 2 \
  --place-hold-seconds 10
```

Recover to upright after knee bend:

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_recover_stand \
  --height 0.80 \
  --place-ramp-seconds 3 \
  --place-hold-seconds 5
```

Single-robot publishing uses the same Term4 scripts with `--robots front` or
`--robots back`:

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_forward \
  --robots front \
  --distance-m 1.5 \
  --speed-mps 0.15

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/validate_static_sonic_skill.py \
  --skill carry_lateral_right \
  --robots back \
  --distance-m 1.5 \
  --speed-mps 0.45
```

## No-Box Baseline Walk

For a no-arm-lock synchronized walk baseline, start Terminal 1 with
`--upper-body-lock-pose none` and run:

```bash
cd /home/admin1/Project/CoEvolve
OUT=results/deployment/dual_sonic_selftest_guarded_walk
mkdir -p "$OUT"

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_official_sonic_motor_preflight.py \
  --front-port 5556 --back-port 5558 \
  --start-pulse-seconds 8 \
  --stand-seconds 10 \
  --walk-meters 2.0 \
  --walk-speed 0.25 \
  --walk-mode slow_walk \
  --movement-x 1 --movement-y 0 \
  --facing-x 1 --facing-y 0 \
  --upper-body-pose none \
  --hand-pose none \
  --final-idle-seconds 3600 \
  --no-stop \
  --heartbeat-file "$OUT/planner_heartbeat.json" \
  --output-dir "$OUT"
```

## Next Payload Trial

Payload contact is the next milestone, not current evidence. Start by adding
the 5kg box and only test static support stability. Keep Term2 and Term3 the
same as above.

```bash
cd /home/admin1/Project/CoEvolve
OUT=results/deployment/payload_5kg_live
mkdir -p "$OUT"
rm -f "$OUT/planner_heartbeat.json" "$OUT/runtime_state.json"

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_sonic_manual_layout_sim.py \
  --interface auto \
  --with-box \
  --box-mass-kg 5.0 \
  --upper-body-lock-pose elbow110_carry_ready \
  --upper-body-lock-kp 24 \
  --upper-body-lock-kd 1.5 \
  --upper-body-lock-ramp-s 6 \
  --upper-body-lock-start-after-drop-s 1 \
  --planner-heartbeat-drop-phase-keyword static_skill_pre_stand \
  --auto-drop-when-ready \
  --lowcmd-warmup-s 4 \
  --min-drop-time-s 4 \
  --planner-heartbeat-file "$OUT/planner_heartbeat.json" \
  --planner-heartbeat-max-age-s 1
```

Do not count payload trials as paper evidence until both robots remain upright,
the box contact is stable, and verifier signals are recorded.
