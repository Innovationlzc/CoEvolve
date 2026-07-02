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

## Command Boundary

Praxis-Coop sends low-frequency planner intent:

- locomotion mode;
- movement vector;
- facing vector;
- speed;
- height;
- start/stop/planner control;
- phase/contact/sync metadata for logging and future verifier use.

Praxis-Coop does not send high-frequency balance actions or raw motor torques
as the main policy output.

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
  scene without payload contact.

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
