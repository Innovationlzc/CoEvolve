# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Current Direction

`/home/admin1/Project/CoEvolve` is now the clean workspace for the new
Praxis-Coop / SONIC-based cooperative skill orchestration project.

The old HALyPO / MARL-style reproduction work is archived under `HaLyPO/`.
Treat it as reference material unless the user explicitly asks to revive or
reuse a piece of it.

## Catalogue

| Path | Purpose |
|---|---|
| `README.md` | current workspace overview |
| `TODO.md` | current milestone plan |
| `ENV_SETUP.md` | concise runtime and external dependency notes |
| `docs/README.md` | docs index and ownership rules |
| `docs/REPRODUCTION.md` | current official SONIC/MuJoCo reproduction commands |
| `docs/PROJECT_CHARTER.md` | paper positioning and contribution boundary |
| `docs/SONIC_INTEGRATION.md` | SONIC adapter boundary |
| `docs/SYSTEM_DESIGN.md` | modules, code layout, contracts, verifier, spec memory |
| `docs/PAPER_PLAN_AAAI27.md` | paper experiments, baselines, metrics |
| `assets/manual_scenes/two_g1_box_push_layout.usda` | tracked user-authored manual scene |
| `source/README.md` | local paper/source-note policy |
| `source/praxis_coop_refined_plan_aaai27.md` | refined plan notes |
| `src/praxis_coop/` | new implementation package |
| `HaLyPO/` | local ignored archive of old experiments; not current evidence |
| `/home/admin1/Project/IsaacLab` | external Isaac Lab runtime; do not edit casually |

## Build & Test

Activate the Isaac Lab runtime when Isaac-dependent work is needed:

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

For new CoEvolve work, prefer tests and scripts under the clean root once they
exist:

```bash
pytest -q
```

Current motor-layer reproduction commands live in `docs/REPRODUCTION.md`.
Archived old commands live under local ignored `HaLyPO/scripts/` and should not
be used as current evidence.

## Safety

- Keep `/home/admin1/Project/IsaacLab` as an external runtime unless the user
  explicitly asks to modify it.
- Do not move `env_isaaclab` into CoEvolve without a fresh migration plan.
- Treat `third_party/GR00T-WholeBodyControl` as a git-ignored local runtime.
  Keep CoEvolve integration through a narrow adapter instead of mixing project
  code into the SONIC tree.
- Do not use old `HaLyPO/` experiments as current evidence without clearly
  labeling them as archived.
- Keep new implementation work out of `HaLyPO/` unless it is intentionally
  archived there.
- Keep GitHub uploads lean: do not track `third_party/`, `models/`,
  generated `results/deployment/` outputs, local paper PDFs, or `HaLyPO/`.
- Keep documentation short. Prefer one current plan in root docs over many
  stale TODO lists.

## Working Style

- New method focus: frozen/wrapped SONIC whole-body control plus verified
  self-evolution of cooperative skill graphs.
- Do not put LLMs in the high-frequency control loop.
- Prefer state/physics verifiers over VLM-only judging.
- Record new decisions in `TODO.md` or `docs/`; keep old HALyPO notes in
  `HaLyPO/legacy_docs/`.
