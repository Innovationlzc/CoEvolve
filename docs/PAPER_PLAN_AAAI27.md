# Paper Plan: AAAI-27

This document only tracks paper experiments, baselines, metrics, and figures.
The contribution boundary lives in `PROJECT_CHARTER.md`; module details live in
`SYSTEM_DESIGN.md`.

## Main Experimental Tasks

All main experiments should use the user's self-built Isaac scene as the primary
scenario. Additional layouts should be controlled variants of that scene rather
than unrelated new worlds.

| Task | Purpose |
|---|---|
| Two-G1 box transport in the self-built scene | main benchmark, object-centric planning, role allocation |
| Long object / narrow-passage variant | yaw-aware planning, pivot behavior, dynamic role switching |
| Scout-assisted variant with scene cameras | asymmetric observation, visual helper role, VLM failure explanation |

## Baselines

| Baseline | What it tests |
|---|---|
| hand-coded fixed graph | whether evolution improves beyond a strong static graph |
| LLM zero-shot Python/script | whether direct script generation is brittle |
| LLM + text Reflexion memory | whether executable memory beats pure text memory |
| object-centric planner without evolution | value of self-evolution |
| independent robot planning | value of object-centric cooperative planning |
| no verifier | value of gated promotion and safety checks |
| no role allocation | value of role specialization |
| no spec retrieval | value of long-term memory |

## Metrics

| Group | Metrics |
|---|---|
| task success | success rate, target pose error, final yaw error, completion time |
| physical safety | collision count, min clearance, robot fall rate, object tilt |
| cooperation | formation error, synchronization error, contact loss count, recovery count |
| evolution | episodes-to-success, improvement after N failures, accepted specs, repeated-failure reduction |
| generalization | held-out layout, held-out object shape, gate width, target position, partial observation |

## Ablation Hypotheses

- Deployment specs should reduce repeated gate-collision, contact-loss, and no-progress pose-adjustment failures compared with text-only memory.
- Verifier-gated spec promotion should improve success without increasing collision or regression failures.
- Object-centric planning should outperform independent robot path planning in narrow passages.
- Role allocation should matter more on long objects and asymmetric observation tasks.
- VLM failure interpretation should help categorize semantic failures, but physics/state verification should carry success judgment.

## Initial Paper Figure Set

1. System architecture: instruction to skill graph to SONIC execution to verified evolution.
2. Skill graph anatomy: node, preconditions, monitors, fallback, verifier report.
3. Benchmark scenes: box passage, long object passage, scout-assisted transport.
4. Self-evolution curve: success over episodes and accepted deployment specs.
5. Spec memory examples: success/failure rollout evidence to reusable skill-calling constraint.
