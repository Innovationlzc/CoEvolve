# Project Charter

## Working Title

Praxis-Coop: Verified Self-Evolution of Cooperative Skill Graphs for Multi-Humanoid Object Transport

The implementation lives in the clean repository root under `src/praxis_coop/`;
the archived old HALyPO work lives under `HaLyPO/`. The paper name can remain
Praxis-Coop if it reads better.

## Core Claim

We should not compete with SONIC on low-level whole-body control. SONIC is the
frozen or lightly wrapped motor prior. The research question is:

> How can LLM/agent systems correctly, efficiently, and safely call whole-body skills for cooperative multi-robot tasks, then turn failures into reusable executable specifications?

## What Changes From the Old Direction

| Old direction | New direction |
|---|---|
| Reproduce HALyPO/HALO-like cooperative transport policy | Use the task setting as inspiration only |
| MARL/decentralized policy is the main algorithm | Skill orchestration, verification, and memory are the main algorithm |
| Policy learns low-level cooperative motion | SONIC handles motor realization; we plan and verify cooperation |
| VLM judge as primary success detector | State/physics verifier first, VLM only for semantic failure explanation |
| Agent edits Python during execution | Episode-level temporary repairs, sandbox tests, regression tests |

## Contribution Boundary

The paper contribution should be stated as:

1. Cooperative skill orchestration over a foundation humanoid WBC.
2. Object-centric planning and role-aware skill graphs for multi-humanoid object transport.
3. Executable specification memory with triggers, predicates, fallbacks, evidence, and confidence.
4. Verifier-gated self-evolution that promotes reusable deployment specs, not a separate patch-selection policy.
5. A reproducible Isaac Lab benchmark for two-humanoid cooperative transport under narrow-passage and partial-observation constraints.

SONIC solves motor realization. Praxis-Coop solves cooperative task
organization, verification, memory, and self-improvement above that motor layer.

## Non-Goals

- Do not retrain SONIC as the main paper result.
- Do not make LLMs part of the high-frequency control loop.
- Do not rely on VLM as the only judge.
- Do not make the second robot a passive judge in the main tasks.
- Do not promise real-robot deployment before the Isaac Lab benchmark and adapter contracts are stable.

## First Target Scenario

The primary test scenario should be the user's self-built Isaac scene, not a new
generic benchmark scene. The first target task remains: two Unitree G1 humanoids
cooperatively move a box or long object from a start region to a target region
through the scene's constrained passage / obstacle layout. Implementation
details for roles, phases, verifier state, and code layout live in
`SYSTEM_DESIGN.md`.
