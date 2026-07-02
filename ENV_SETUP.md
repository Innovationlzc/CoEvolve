# Environment Setup

This repository keeps Praxis-Coop code in git and treats heavy runtimes/models
as local external dependencies.

## Local Runtime Paths

| Component | Expected local path | Git status |
|---|---|---|
| CoEvolve repo | `/home/admin1/Project/CoEvolve` | tracked |
| Isaac Lab | `/home/admin1/Project/IsaacLab` | external |
| SONIC / GR00T-WholeBodyControl | `third_party/GR00T-WholeBodyControl` | ignored |
| Qwen models | `models/` | ignored |

## Isaac Lab

Use Isaac Lab's environment for tests that need USD/PXR or Isaac packages:

```bash
cd /home/admin1/Project/IsaacLab
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
```

Run repo tests from CoEvolve:

```bash
cd /home/admin1/Project/CoEvolve
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python -m pytest -q
```

## SONIC Runtime

Clone SONIC locally:

```bash
cd /home/admin1/Project/CoEvolve
mkdir -p third_party
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git third_party/GR00T-WholeBodyControl
```

Install the upstream runtime following NVLabs' instructions. The current local
setup expects:

```text
third_party/GR00T-WholeBodyControl/.venv_sim
third_party/GR00T-WholeBodyControl/gear_sonic_deploy
third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/
third_party/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/
```

The CoEvolve wrappers are:

```text
scripts/run_dual_sonic_manual_layout_sim.py
scripts/run_official_sonic_deploy_zmq_manager.sh
scripts/run_dual_official_sonic_motor_preflight.py
```

See `docs/REPRODUCTION.md` for the four-terminal launch sequence.

## Qwen Models

Qwen is not required for the current motor-layer preflight. For later
SkillGraph/Qwen evolution experiments, download locally:

```bash
cd /home/admin1/Project/CoEvolve
PYTHONPATH=src /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/download_phase1_models.py --model all
```

Expected local paths:

```text
models/Qwen3-8B
models/Qwen3-VL-8B-Instruct
```

These are intentionally ignored by git.

## Spec Memory

Curated static specs are tracked:

```text
results/spec_memory/static/default_specs.jsonl
```

Generated dynamic specs are local outputs:

```text
results/spec_memory/dynamic/
```
