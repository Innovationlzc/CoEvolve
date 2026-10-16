# Running CoEvolve

The public checkout supports offline software checks and provides engineering
entrypoints for the live two-G1 stack. It does not include historical results,
manuscript exports, frozen sibling runtimes or model weights.

## Offline checks

After [installation](../ENV_SETUP.md):

```bash
python -m coevolve_experiments validate-scenes --mode planning
python -m coevolve_experiments adapter-smoke --method all
python -m pytest -q tests/maintenance tests/experiments/test_cli.py
```

No model or physics is dispatched. The generic
`coevolve_experiments.production` launcher is intentionally unavailable
(exit 78) until qualified live workers exist.

## Live prerequisites

Use Linux, a qualified upstream SONIC/MuJoCo installation, both integration
patches, and authorized weights. First inspect:

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/run_sonic_static_skill.py --list-skills
python scripts/run_cap_box_carry_evolution.py --help
```

Qualify motor behavior with `run_dual_official_sonic_motor_preflight.py` and
`validate_static_sonic_skill.py` before loaded model-driven execution.
These tools can launch physics; read their `--help` and explicitly choose
the test configuration. Offline tests alone do not qualify the live stack.

Keep motor transports on an isolated trusted network. ZMQ can bind all network
interfaces; this is not an authenticated remote-control service. Configure
the same DDS interface in every terminal. Do not connect this simulation
workflow to real robots.

## Four-terminal engineering example

From the repository root in **each terminal**, set the same paths:

```bash
export COEVOLVE_ROOT="$PWD"
export COEVOLVE_RUN="$COEVOLVE_ROOT/results/deployment/my-first-run"
```

Choose a new run name every time; do not reuse a checkpoint or output directory
from another simulator process. This example is a bounded engineering run,
not the exact frozen configuration of any paper cohort.

### Terminal 1 — simulator

Create the fresh output directory once, then start MuJoCo:

```bash
mkdir -p "$COEVOLVE_ROOT/results/deployment"
mkdir "$COEVOLVE_RUN"
```

If the second command reports an existing directory, choose another run name
before continuing.

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_sonic_manual_layout_sim.py \
  --interface auto \
  --with-box --box-mass-kg 0.5 \
  --box-placement authored --box-initial-state deferred \
  --payload-rearward-offset-m 0.10 \
  --upper-body-lock-pose elbow60_hand60_carry_ready \
  --upper-body-lock-kp 24 --upper-body-lock-kd 1.5 \
  --upper-body-lock-ramp-s 6 --upper-body-lock-start-after-drop-s 1 \
  --upper-body-lock-palm-tilt-deg 60 \
  --viewer-camera-preset global \
  --planner-heartbeat-drop-phase-keyword static_skill_pre_stand \
  --auto-drop-when-ready --lowcmd-warmup-s 4 --min-drop-time-s 4 \
  --mujoco-solver newton \
  --planner-heartbeat-file "$COEVOLVE_RUN/planner_heartbeat.json" \
  --planner-heartbeat-max-age-s 1 \
  --runtime-state-file "$COEVOLVE_RUN/runtime_state.json" \
  --continue-after-fall \
  --summary-file "$COEVOLVE_RUN/term1_summary.json"
```

`--continue-after-fall` keeps the checkpoint service alive; it does not make
a fall acceptable. The free 0.5 kg payload is placed only after readiness and
bilateral support checks.

### Terminals 2 and 3 — native SONIC backends

Terminal 2:

```bash
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id front --dds-topic-prefix front \
  --zmq-port 5556 --zmq-out-port 5557
```

Terminal 3:

```bash
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id back --dds-topic-prefix back \
  --zmq-port 5558 --zmq-out-port 5559
```

Wait for both backends to report `CONTROL`. Do not manually drop the robots;
the runtime gates release on fresh heartbeat and low-command readiness.

### Terminal 4 — planner

Activate the planner environment, configure a multimodal provider as described
in [setup](../ENV_SETUP.md), and set `COEVOLVE_MODEL` and
`COEVOLVE_BASE_URL` to its exact model ID and endpoint. Supply
`COEVOLVE_API_KEY` securely; requests can incur charges.

```bash
source .venv/bin/activate
mkdir -p "$COEVOLVE_RUN/memory"
python scripts/run_cap_box_carry_evolution.py \
  --agent-mode model --planner-backend openai_compatible --visual-evaluator model \
  --planner-model "${COEVOLVE_MODEL:?Set the provider model ID}" \
  --planner-base-url "${COEVOLVE_BASE_URL:?Set the provider endpoint}" \
  --planner-api-key-env COEVOLVE_API_KEY \
  --task-stage full_scene --route-discovery \
  --target "Carry and place the box on the red target, release, retreat together while crouched, then stand clear." \
  --require-robots-exit --execute --physical-safety-profile task \
  --bootstrap-start-pulse-seconds 4 --bootstrap-idle-seconds 6 \
  --start-pulse-seconds 0 --stand-seconds 0.5 \
  --max-attempts 20 --iteration-budget 20 \
  --runtime-state-file "$COEVOLVE_RUN/runtime_state.json" \
  --heartbeat-file "$COEVOLVE_RUN/planner_heartbeat.json" \
  --raw-attempt-memory "$COEVOLVE_RUN/memory/raw_rollouts.jsonl" \
  --strategy-memory "$COEVOLVE_RUN/memory/strategy_specs.json" \
  --action-exemplar-memory "$COEVOLVE_RUN/memory/action_exemplars.json" \
  --episode-manifest "$COEVOLVE_RUN/memory/accepted_episode_manifest.json" \
  --output-dir "$COEVOLVE_RUN"
```

Term4 bootstraps standing support before initial payload placement. Do not run
a separate manual placement command. Model-only mode without `--execute` can
still make billable requests; it is not an offline check. Attempt indices and
charged physical transactions are separate counters.

## Parallelism and recovery

- **Model candidates:** CaP-Agent0 retains nine candidates with 1–9 workers.
  Worker count changes concurrency, not K. Use thread-safe or per-request model
  clients, bounded transport timeouts, and one adapter per logical run.
- **Two-robot coordination:** `sync`/`async` describe team roles and phase
  objectives, not an OS-level real-time barrier. Equal nominal speed does not
  guarantee equal displacement; verification uses measured physics.
- **Multiple experiments:** physical multi-stack operation remains experimental
  after observed timing overload; historical queue launchers are not distributed.
  Namespace isolation alone does not establish equivalent simulation pacing.
  Shared-stack queues stay serial. Never background multiple runs sharing
  ports, scene files, output paths or checkpoints.

Before enabling physical concurrency, qualify namespace/port isolation,
CPU/GPU pacing, fresh observations and per-run accounting. No public one-command
multi-stack launcher is claimed here.

Rollback restores a certified checkpoint, not a reverse trajectory. Restarting
Term1 invalidates its in-process checkpoints; only the exact recovery protocol
can establish a valid continuation. Preserve cumulative budgets and lineage;
do not count resumed segments as independent seeds.

## Script catalog

Only these entrypoints are published under `scripts/`. Start with the
four-terminal example; the remaining tools serve specific validation tasks.
Use `--help` before a Python entrypoint. Provider smoke checks make billable
requests; motor checks and replay can dispatch physics.

| Purpose | Scripts |
| --- | --- |
| Core startup | `run_cap_box_carry_evolution.py`, `run_dual_sonic_manual_layout_sim.py`, `run_official_sonic_deploy_zmq_manager.sh` |
| Motor and skill validation | `run_dual_official_sonic_motor_preflight.py`, `run_sonic_static_skill.py`, `run_sonic_coordinated_skill.py`, `validate_static_sonic_skill.py` |
| Scene, calibration, optional local models | `build_sonic_manual_layout_scene.py`, `calibrate_sonic_pivot.py`, `download_phase1_models.py` |
| Evaluation planning and summary | `prepare_formal_evaluation.py`, `summarize_formal_evaluation.py` |
| Provider checks and optional billing reconciliation | `smoke_formal_backbones.py`, `smoke_formal_roles.py`, `reconcile_api_costs.py` |
| Accepted-episode export/replay and experience rebuilding | `manage_accepted_episode.py`, `rebuild_strategy_memory.py`, `rebuild_verified_action_memory.py` |

Evaluation preparation creates plans, not completed experiments. Rebuilding
memory and replay require user-supplied verifier evidence. The billing tool
needs an explicitly configured compatible usage API; it is not a universal
provider billing client. Review options before writing to existing memory.

Historical matrix/repair queues, one-case diagnostics, paper table/figure
builders and original evidence are not distributed. This maintained checkout
is not the paper's byte-identical frozen runtime and cannot regenerate its
tables without those inputs. See [baseline adaptations](BASELINES.md).
