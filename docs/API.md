# Available interfaces

This project provides Python interfaces, a CLI and local motor transport.
It does **not** implement a public HTTP/REST server, authentication service or
remote robot-control API.

## 1. Experiment CLI

```bash
coevolve-experiments --help
coevolve-experiments validate-scenes --mode planning
coevolve-experiments adapter-smoke --method all
coevolve-experiments summarize --help
```

Python equivalent: `coevolve_experiments.cli.main(argv) -> int`.
An installed wheel bundles the static skill catalog, not local scenes/results.
For scene checks outside an editable checkout, use
`validate-scenes --mode planning --repo-root ../CoEvolve` (adjust to your working directory).
`build_parser()` exposes the same arguments. Commands cover protocol plans,
scene checks, strategy-bank export/validation, launch binding, summaries and
replay preparation. Most emit JSON; invalid commands return nonzero.

Planning/smoke does not dispatch physics. Materialization/export writes files;
provider probes can bill requests; live/replay commands are not smoke tests.
Generic live production remains deliberately unavailable (exit 78) until its
reviewed workers and evidence collectors are implemented.

Historical cohort certification is not distributed: `audit_table1_source_cohort`
and `audit_frozen_table1_candidate` raise `Table1SourceAuditError` before reading
inputs. Commands requiring those audits are unavailable in this public checkout.
Generic cohort data contracts, summaries and synthetic tests remain available;
they do not certify historical paper evidence. Checkout verification requires
caller-supplied, independently reviewed commit and file-hash pins.

## 2. Model client (outbound provider API)

```python
from praxis_coop.llm.openai_compatible_client import OpenAICompatibleTextClient

client = OpenAICompatibleTextClient(
    model_name="<exact-provider-model-id>",
    # Omit base_url to read COEVOLVE_BASE_URL from your environment.
    api_key_env="COEVOLVE_API_KEY",  # variable name, never the secret value
    timeout_s=120,
    network_retries=2,
)
# The following methods send real, potentially billable requests:
# client.generate(system="...", user="...")
# client.generate_structured(system="...", user="...", schema_name="...", schema={...})
# client.analyze_image(system="...", user="...", image_path="frame.png")
```

Methods return response text; usage/transport records are exposed through the
client's accounting interfaces. Formal execution additionally pins resolved
model identity. `ModelRolePipeline` supplies role-specific schemas/validation;
the transport alone does not make generated programs executable or safe.
Provider billing lookup is optional and disabled by default. If your provider
supports the adapter's usage schema, set `usage_path_template` to its documented
same-origin path containing `{request_id}`. No automatic account discovery occurs.

The client is stateful: do not share mutable `call_records` or request settings
unsafely across threads. Historical coding-baseline transport clones clients
and merges receipts under a lock. A thread cannot reliably interrupt a blocked
HTTP request; bounded transport timeouts are still necessary.

## 3. Cooperative Skill SDK

```python
from pathlib import Path
from praxis_coop.cap_policy.backends import StaticSonicSkillBackend
from praxis_coop.cap_policy.skill_api import G1SkillAPI

api = G1SkillAPI(
    backend=StaticSonicSkillBackend(execute=False),  # explicit nonphysical mode
    output_dir=Path("results/deployment/sdk-dry-run"),
    heartbeat_file=Path("results/deployment/sdk-dry-run/planner_heartbeat.json"),
)
```

Key methods include `hold_position`, `coordinated_linear_transport`,
`coordinated_pivot`, `coordinated_alignment`, `place_payload_on_arms`, and
`stop_all`. Parameters are allowlisted/range-checked; cooperative calls require
paired robot scope and validated team contracts. This constructor is not a
shortcut around those gates. Inspect method signatures before calling.

The model-selectable catalog is shipped in
`src/praxis_coop/data/default_specs.jsonl`; it does not depend on `results/`.

| Skill family | Methods | Main parameters |
| --- | --- | --- |
| Loaded translation | `carry_forward`, `carry_backward` | `distance_m`, `speed_mps`, `ramp_s` |
| Turning | `turn_clockwise`, `turn_counterclockwise` | `yaw_deg`, `yaw_rate_degps`, `turn_segment_deg` |
| Posture and support | `place_crouch`, `recover_stand`, `recover_payload_inward` | Height or ramp/hold time |
| Lateral translation | `carry_lateral_left`, `carry_lateral_right` | `distance_m`, `speed_mps`, `ramp_s` |

Initialization (`place_payload_on_arms`), transaction control (`hold_position`,
`stop_all`) and final release/exit are separate lifecycle calls. Carry actions
require verified support; `recover_payload_inward` is a guarded, one-way,
synchronized recovery, not an arbitrary arm command. The catalog contains
parameter bounds; nominal speed is not measured displacement. See
[the run guide](REPRODUCTION.md) for motor qualification and startup.

`PolicyExecutor(artifact_dir=...).execute(policy: PolicyScript, api: G1SkillAPI)`
returns an `ExecutionTrace` and writes execution artifacts. The physical
evaluator/transition verifier separately decides acceptance and task success.
Code execution success alone is not task success.

Security boundary: the generic executor uses in-process Python execution and
is not a sandbox for untrusted code. Its direct API-call allowlist does not
provide process isolation or resource limits. Checkpointed transaction checks
are separate contracts, not a general-purpose security guarantee.

Live mode requires the simulator, both native deploy processes, fresh runtime
state and heartbeat, approved initialization and matching output paths. ZMQ
planner defaults use ports 5556/5558; deploy outputs use 5557/5559. Per-robot DDS
prefixes and per-experiment isolation remain required. These are local research
interfaces, not an authenticated network service.

## 4. CaP-Agent0 adaptation

```python
from coevolve_experiments.methods.cap_agent0 import CaPAgent0Adapter

# invoke(role, payload) returns a mapping with content and usage metadata.
# It must be thread-safe when candidate_workers > 1.
adapter = CaPAgent0Adapter(invoke, candidate_workers=3)
# proposal = adapter.propose(observation, images=["certified-current.png"])
# adapter.observe_execution(actual_host_and_verifier_feedback)
```

Nine candidates are still requested exactly once. Worker count changes only
concurrency, not K. Default 9 preserves historical behavior; 1 is serial.
Changing workers changes the runtime configuration and timing interpretation.
Only the synthesized program is returned for Host admission; candidates never
execute physically. Keep one caller per stateful adapter.

The old `scripts/cap_agent0_adapted_v1.py` import path remains supported.

## 5. Evidence/reporting

`coevolve_experiments.reporting` exposes collection/summarization helpers;
the CLI is preferred when protocol/manifest bindings must be checked.
Historical paper builders include `scripts/build_paper_metrics_20260915.py`
and `scripts/build_paper_timing_20260924.py`; both need excluded local evidence.
Do not manufacture zero values for missing calls/time, overwrite raw traces,
or count a continuation as a fresh independent seed.
