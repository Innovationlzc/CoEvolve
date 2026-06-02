# Environment Notes

This project uses Isaac Lab as an external runtime and keeps CoEvolve focused on
HALyPO / two-G1 co-evolution code, configs, and notes.

## Runtime Location

Do not move the Isaac Lab virtual environment into this repository for now.

| Component | Location | Status |
|---|---|---|
| CoEvolve workspace | `/home/admin1/Project/CoEvolve` | project docs/code |
| Isaac Lab runtime | `/home/admin1/Project/IsaacLab` | external upstream repo |
| Python venv | `/home/admin1/Project/IsaacLab/env_isaaclab` | primary runtime |
| HALyPO paper | `source/HALyPO.pdf` | local source paper |

Reason: `env_isaaclab` is large and tied to Isaac Lab editable packages, Kit
extension caches, and launcher paths. Moving it into CoEvolve risks breaking the
runtime without improving reproducibility. If CoEvolve needs a local entrypoint,
add a small wrapper script later instead of relocating the environment.

## Installed Versions

| Package | Version / state |
|---|---|
| Python | 3.12.13 |
| Isaac Sim | 6.0.0.0 |
| Isaac Lab | `release/3.0.0-beta2`, `isaaclab 6.0.0` |
| PyTorch | 2.10.0+cu128 |
| GPU | RTX 4090, driver 595.58.03 |
| RL extras | `rsl-rl-lib`, `skrl`, `rl-games`, `stable-baselines3` installed |
| HAPPO/HATRPO | not installed yet; candidate baseline is `morning9393/HAPPO-HATRPO` |

## Activate

```bash
cd /home/admin1/Project/IsaacLab
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
```

## Current Validation Commands

Headless Isaac startup:

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

Newton visualizer with a real environment:

```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 8 \
  --viz newton
```

G1 environment smoke test:

```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Velocity-Flat-G1-Play-v0 \
  --num_envs 1 \
  --viz newton
```

CoEvolve M2 Two-G1 Box Push smoke test from this repo:

```bash
cd /home/admin1/Project/CoEvolve
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_two_g1_box_push.py \
  --num_envs 1 \
  --max_steps 600 \
  --log-every 50 \
  --viz none
```

Open the same M2 scene with Newton visualization:

```bash
cd /home/admin1/Project/CoEvolve
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_two_g1_box_push.py \
  --num_envs 1 \
  --max_steps 600 \
  --log-every 50 \
  --viz newton
```

Train the M2 proxy independent-vs-HALyPO comparison:

```bash
cd /home/admin1/Project/CoEvolve
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_proxy_train.py
```

Evaluate the trained HALyPO proxy policy in Isaac with Newton visualization:

```bash
cd /home/admin1/Project/CoEvolve
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_two_g1_box_push.py \
  --num_envs 1 \
  --max_steps 130 \
  --log-every 40 \
  --policy-path results/m2_proxy_training/halypo_policy.json \
  --stop-on-success \
  --viz newton
```

Run the newer M2 base-command adapter smoke:

```bash
cd /home/admin1/Project/CoEvolve
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_two_g1_box_push.py \
  --num_envs 1 \
  --max_steps 120 \
  --log-every 20 \
  --action-mode base_command \
  --stop-on-success \
  --viz newton \
  --output-dir results/m2_base_command_smoke
```

## Known Runtime Notes

- `create_empty.py --viz newton` is not a useful test: the stage has no bodies,
  so Newton has no model to visualize.
- `--viz kit` currently fails because the GUI experience depends on
  `isaacsim.core.experimental.primdata`, which is missing from the installed
  Isaac Sim extension set.
- Training and environment development can proceed headless or with
  `--viz newton`; Kit GUI can be fixed later if needed.
- From CoEvolve, the same G1 visual smoke can be launched with
  `scripts/run_m2_g1_smoke.py`.
- `CoEvolve-TwoG1BoxPush-Direct-v0` registers correctly and runs headless or
  with `--viz newton`. It is currently a proxy action task, not a full G1
  control policy.
- In M2 Lite, the G1 bodies are kept upright as visual actors. Without a G1
  controller, free articulated G1 bodies fall over.
- The current M2 eval does not use robot foot motion, arm motion, or contact to
  push the box. The proxy action moves the box, and the G1 visuals follow
  smoothly with fixed joints. This is training/eval plumbing only.
- `--action-mode base_command` is one step closer: action moves the G1 base
  visuals first, and the box moves through a virtual contact adapter. It still
  is not walking, WBC, or true rigid-body contact pushing.

## Historical Logs

Install and smoke-test logs are archived under `logs/setup/`. Treat them as
diagnostic history, not current truth.
