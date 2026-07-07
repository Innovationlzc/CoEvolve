# Scripts

Current entry points:

| Script | Purpose |
|---|---|
| `build_sonic_manual_layout_scene.py` | generate SONIC/MuJoCo XML from the tracked manual USD scene |
| `run_dual_sonic_manual_layout_sim.py` | dual-G1 MuJoCo runtime for official SONIC deploy backends |
| `run_official_sonic_deploy_zmq_manager.sh` | launcher for one upstream `gear_sonic_deploy` process |
| `run_dual_official_sonic_motor_preflight.py` | synchronized no-box dual-G1 planner publisher |
| `run_sonic_static_skill.py` | low-level static skill publisher |
| `validate_static_sonic_skill.py` | Term4 wrapper with tuned defaults and dry-run support |
| `download_phase1_models.py` | Qwen model downloader for later evolution experiments |

Current static skills:

```text
parallel_forward
carry_forward
carry_backward
carry_lateral_left
carry_lateral_right
carry_turn_clockwise_90
carry_turn_counterclockwise_90
carry_turn_in_place
carry_place_crouch
carry_recover_stand
```

Use `docs/STATIC_SONIC_SKILLS.md` for the current successful command sheet.
