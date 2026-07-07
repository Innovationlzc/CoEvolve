# Reproduction

The current reproducible milestone is official SONIC / GEAR-SONIC motor-layer
validation in a dual-G1 MuJoCo scene generated from the tracked manual layout:

```text
assets/manual_scenes/two_g1_box_push_layout.usda
```

Use the current command sheet:

```text
docs/STATIC_SONIC_SKILLS.md
```

It contains the four-terminal workflow, validated static skills, tuned
parameters, and the next 5kg payload-stability trial.

## Valid Current Evidence

- official SONIC-controlled dual-G1 no-box synchronized walk;
- carry-ready upper-body lock with standing/walking skills;
- `carry_forward`, `carry_backward`, `carry_lateral_left/right`;
- `carry_turn_in_place`;
- `carry_place_crouch` and `carry_recover_stand`.

These are motor-layer/static-skill results. They are not yet full physical
box-transport evidence.

## Tests

```bash
cd /home/admin1/Project/CoEvolve
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python -m pytest -q
```

## Boundary

Do not use old Isaac proxy/root-teleport runs, HTML-only replays, floating-box
runs, or wall-pass-through results as paper evidence. Payload-contact trials
start from the 5kg box stability setup in `docs/STATIC_SONIC_SKILLS.md`.
