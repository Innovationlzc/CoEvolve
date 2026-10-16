# Architecture

`praxis_coop` owns the method and motor boundary;
`coevolve_experiments` owns experiment contracts, adapters and reporting.

| Layer | Main location | Responsibility |
| --- | --- | --- |
| Model transport | `src/praxis_coop/llm/` | Remote/local inference and usage accounting |
| Planning and repair | `src/praxis_coop/cap_policy/` | Task roles, program generation, Host admission and scoped repair |
| Verification and memory | `cap_policy/` evaluator, checkpoints and memory stores | Physical acceptance, certified restoration and evidence-backed reuse |
| Motor interface | `src/praxis_coop/sonic/`, `sonic_mujoco/` | Frozen SONIC control and MuJoCo state/contact feedback |
| Experiment framework | `src/coevolve_experiments/` | Protocols, scenes, methods, accounting and reports |
| Entry points | `scripts/` | Reviewed setup, live startup, validation and experience maintenance |
| Developer tools | `tools/` | Isolated tests, inventory and recoverable cleanup |

LLMs issue low-frequency cooperative intent, not high-frequency torques.
Generated code passing static checks is not physical success. Only verified
executed behavior can become accepted experience.

## Execution boundaries

`coevolve_experiments.runtime` is a deterministic TeamIntent contract-smoke
engine, not the full physical PolicyScript loop. The separate `production/`
package implements integration/evidence contracts but lacks qualified live
workers and collectors; its generic runner deliberately exits 78.

The live engineering stack uses one MuJoCo process, two official SONIC deploy
processes, and the checkpointed planner. See [startup and concurrency](REPRODUCTION.md).
The generic `PolicyExecutor` executes Python in-process and is not a security
sandbox. Physical verifiers do not replace process/resource isolation.

## Publication and evidence

Dated repair scripts, internal queue managers and paper audit builders remain
local. The public entrypoint allowlist lives in `.gitignore`; adding a script
requires reviewing its dependencies, default behavior and text. Reusable
baseline logic lives in `coevolve_experiments.methods`, not in launch scripts.

Use `tools/run_tests.py` for per-file process isolation. It selects public tests
and reports failures; `--integration` enables checks requiring external assets.

The static Skill SDK is distributed under `src/praxis_coop/data/`, independent
of `results/`. New output remains local. This maintained checkout is not
byte-identical to historical frozen experiment runtimes.
