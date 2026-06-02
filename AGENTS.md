# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Catalogue

| Path | Purpose |
|---|---|
| `ENV_SETUP.md` | concise runtime status and launch commands |
| `HALyPO_REPRO_TODO.md` | current research and implementation plan |
| `source/HALyPO.pdf` | source paper |
| `logs/setup/` | historical install and smoke-test logs |
| `/home/admin1/Project/IsaacLab` | external Isaac Lab runtime; do not edit casually |

Planned project layout:

| Path | Purpose |
|---|---|
| `coevolve/algorithms/` | HALyPO, PCGrad, baseline adapters |
| `coevolve/envs/` | non-Isaac and Isaac Lab task definitions |
| `configs/` | experiment and algorithm configs |
| `scripts/` | train, evaluate, plot, and utility entrypoints |
| `tests/` | projection, env, and smoke tests |
| `third_party/` | optional vendored baseline references |

## Build & Test

Activate the Isaac Lab runtime:

```bash
cd /home/admin1/Project/IsaacLab
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
```

Useful runtime checks:

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 8 --viz newton
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 1 --viz newton
```

Once CoEvolve code exists, prefer:

```bash
pytest -q
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python scripts/run_m0_toy.py
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python scripts/run_m1_transport.py --show
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python scripts/run_m2_g1_smoke.py --timeout 120
```

## Safety

- Keep `/home/admin1/Project/IsaacLab` as an external runtime unless the user
  explicitly asks to modify it.
- Do not move `env_isaaclab` into CoEvolve without a fresh migration plan.
- Do not use `--viz kit` as a required test until the missing Kit extension is
  fixed.
- Do not treat `create_empty.py --viz newton` as a failure signal; empty stages
  have no bodies for Newton to visualize.
- Keep training headless where possible. Use visualization scripts for policy
  evaluation and progress inspection.
- Avoid deleting logs or generated artifacts unless they are clearly obsolete or
  the user asks for cleanup.
- Keep documentation short. Prefer one current plan over many stale TODO lists.

## Working Style

- Make small, verifiable milestones: M0 toy kernel, M1 2D transport, M2 Isaac
  Two-G1 Box Push Lite, M3 one paper-like OSP scenario.
- Use existing Isaac Lab task patterns before inventing new abstractions.
- Keep paper-scale goals separate from minimum validation goals.
- Record decisions in `HALyPO_REPRO_TODO.md`; record runtime facts in
  `ENV_SETUP.md`.
