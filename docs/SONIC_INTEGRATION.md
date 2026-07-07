# SONIC Integration

Praxis-Coop treats SONIC / GEAR-SONIC as the frozen low-level motor layer.
The project contribution sits above it: cooperative SkillGraph generation,
verification, repair, and SpecMemory.

## Current Valid Backend

The current validated backend is process-level official GEAR-SONIC deployment:

```text
Praxis-Coop command publisher
        |
ZMQ planner messages
        |
two official gear_sonic_deploy processes
        |
Unitree lowcmd topics with front/back DDS namespaces
        |
dual-G1 MuJoCo runtime generated from the tracked manual USD scene
```

The tracked scene source is:

```text
assets/manual_scenes/two_g1_box_push_layout.usda
```

The generated MJCF files are written into the local ignored SONIC runtime:

```text
third_party/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/
```

## Required Local SONIC Patches

The dual-G1 workflow requires two CoEvolve patches inside the ignored SONIC
runtime. Script-level `--dds-topic-prefix` is not enough; the C++ deploy binary
and Python MuJoCo bridge must both understand prefixed DDS topics.

Apply these after cloning or updating `third_party/GR00T-WholeBodyControl`:

```bash
cd /home/admin1/Project/CoEvolve/third_party/GR00T-WholeBodyControl
git apply /home/admin1/Project/CoEvolve/patches/gear_sonic_multi_robot_dds_prefix.patch
git apply /home/admin1/Project/CoEvolve/patches/gear_sonic_heading_increment.patch

cd gear_sonic_deploy
source scripts/setup_env.sh
just --justfile .justfile build
```

Quick checks:

```bash
grep -R "dds-topic-prefix" gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
grep -R "DDS_TOPIC_PREFIX" gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py
```

## Command Boundary

Praxis-Coop sends low-frequency planner intent:

- locomotion mode;
- movement vector;
- facing vector;
- speed;
- height;
- cumulative heading offset for turn-in-place skills;
- start/stop/planner control;
- phase/contact/sync metadata for logging and future verifier use.

Praxis-Coop does not send high-frequency balance actions or raw motor torques
as the main policy output.

Carry-ready arm posture is held by the dual MuJoCo runtime as a motor-side
upper-body pose lock (`elbow110_carry_ready`). The lock is hidden from SONIC's
policy observation by default so lower-body balance stays stable.

Turn skills use SONIC's heading state rather than only changing planner
`facing_direction`. This is required for measured body yaw to change in the
official `zmq_manager` deploy path. Static turn publishing defaults to
`heading_increment`, mirroring the official keyboard Q/E control: each tick
adds a small signed yaw increment to the current heading. This avoids the older
absolute-target failure mode where repeated 90-degree turns accumulated into
large offsets or crossed the +/-180-degree wrap boundary and caused reversal.
The sign convention is explicit in code and docs: `-yaw` is clockwise,
`+yaw` is counterclockwise.
After changing this protocol, rebuild `gear_sonic_deploy` and restart both
deploy terminals.

## Drop Gate

The dual MuJoCo runtime keeps the MuJoCo `ElasticBand` support enabled until:

1. both robot deploy processes have sent body lowcmd;
2. lowcmd has stayed ready for the configured warmup time;
3. the planner heartbeat file is fresh;
4. the minimum drop time has passed.

This avoids the earlier failure mode where the robots were dropped before the
official deploy planner was actually publishing valid commands.

## Current Boundary

Valid current evidence:

- official SONIC-controlled single-G1 stand/walk;
- dual-G1 official deploy stand/walk in the generated manual-layout MuJoCo
  scene without payload contact;
- static dual-G1 carry-ready skills without payload contact: forward, backward,
  lateral left/right, turn-in-place, stationary place-crouch, and recover-stand.

Not valid as current paper evidence:

- old Isaac proxy/root-velocity results;
- kinematic USD replay;
- HTML-only visualizations;
- floating-box or wall-pass-through runs.

## Next Milestone

Add the payload/contact task on top of the validated dual-G1 motor layer:

1. include `long_box` in the generated MuJoCo scene;
2. tune hand/contact geometry and friction;
3. expose contact/fall/progress verifier state;
4. reconnect SkillGraph/Qwen evolution only after the motor-contact layer is
   physically credible.
