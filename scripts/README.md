# Scripts

Current GitHub-facing entry points:

- `build_sonic_manual_layout_scene.py`: generate SONIC/MuJoCo XML scenes from
  the tracked manual USD scene.
- `run_sonic_manual_layout_sim.py`: single-G1 official SONIC MuJoCo preflight.
- `run_official_sonic_motor_preflight.py`: single-G1 ZMQ planner command
  publisher.
- `run_dual_sonic_manual_layout_sim.py`: dual-G1 official SONIC MuJoCo runtime.
- `run_official_sonic_deploy_zmq_manager.sh`: wrapper for upstream
  `gear_sonic_deploy`.
- `run_dual_official_sonic_motor_preflight.py`: synchronized dual-G1 planner
  command publisher.
- `download_phase1_models.py`: local Qwen model downloader for later evolution
  experiments.

Old Isaac proxy, HTML replay, and experimental joint-WBC scripts were removed
from the current root so they are not mistaken for valid paper evidence.
