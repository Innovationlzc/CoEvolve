# System Design

## Architecture

```text
Human instruction + multi-view observations
        |
Scene Graph Builder / World Model
        |
Task Taxonomy Classifier <---- Executable Spec Memory
        |
Object-Centric Global Planner
        |
Role Allocator + Cooperative Skill Graph Generator
        |
Skill API Adapter
        |
SONIC / GEAR-SONIC process backend
        |
Execution backend
  - current validated path: dual-G1 MuJoCo runtime
  - target benchmark path: Isaac Lab task for payload/contact transport
        |
State Verifier + optional VLM Failure Interpreter
        |
Qwen Evolution Agent -> Temporary Patch -> Verification -> Deployment SpecMemory Update
```

## Code Layout

The current method code lives under `src/praxis_coop`.

```text
src/praxis_coop/
  input/                       # User instruction and authored-scene adapters
  task_orchestrator/           # Taxonomy, spec retrieval, roles, SkillGraph generation
  coder_agent/                 # Applies Qwen-generated typed graph patches
  evaluator/                   # State/physics verifier boundary
  evolution_agent/             # Qwen critic/coder/spec-memory agent
  knowledge_base/              # Static specs and dynamic text SpecMemory
  runtime/                     # Manual USD adapters and future Isaac execution boundary
  sonic/                       # SONIC command envelope, mapper, ZMQ publisher
  sonic_mujoco/                # Official MuJoCo dual-G1 runtime bridge
  skills/                      # Registered legal skill calls
  utils/                       # Geometry and USD parsing helpers

  models.py                    # Typed contracts shared across modules
  taxonomy.py                  # Task type classification
  roles.py                     # Role allocation for two-G1 cooperation
  planner.py                   # Object COM path plus per-robot path planning
  skill_graph.py               # Cooperative phase/skill graph generation
  qwen_evolution.py            # Qwen JSON schema/parser implementation
  experiments.py               # Typed SkillGraph patch application
```

## Models

- Text model logical name: `qwen3-8b-instruct`
- Text model Hub target: `Qwen/Qwen3-8B`
- Vision-language model: `Qwen/Qwen3-VL-8B-Instruct`

Qwen3-8B is the critic/coder/spec generator for the evolution loop.
Qwen3-VL-8B is reserved for optional visual explanation and grounding; it is
not used as the primary success judge.

## Scenario Policy

Use the user's self-built Isaac scene as the primary test scenario. New toy
scenes are allowed only for fast unit testing and debugging. Paper-level
evidence should be collected from the self-built scene and its controlled
variants.

The scene adapter should extract or define:

- robot prims and initial poses;
- movable object prims, sizes, masses, and start poses;
- walls, gates, obstacles, and target zones;
- camera prims and available viewpoints;
- semantic names for verifier state and skill parameters.

Recommended initial role template:

| Robot | Role |
|---|---|
| `G1-front` | navigation leader, object yaw alignment leader, gate/target alignment leader |
| `G1-back` | stabilizer, rear-clearance monitor, contact-maintenance agent, emergency-stop trigger |

## Task Taxonomy

Focus the AAAI-27 version on:

```text
rigid-object coupling + narrow passage + partial/asymmetric observability
```

Required labels:

| Dimension | Initial Values |
|---|---|
| physical coupling | independent, loose, rigid-object, force-coupled |
| role symmetry | symmetric, leader-follower, scout-executor, stabilizer-mover |
| object type | box, long box, fragile object, heavy object |
| environment constraint | open, narrow gate, U-turn, S-turn, clutter, target insertion |
| observability | full state, egocentric, asymmetric vision, occlusion |
| objective | transport, align, insert, handoff |
| interaction type | pure cooperation, mixed-motive, adversarial obstacle |

## Skill Graph Node Contract

```python
SkillNode(
    name="align_object_to_gate",
    agents={"leader": "G1-front", "stabilizer": "G1-back"},
    parameters={
        "object_id": "box_01",
        "gate_id": "gate_01",
        "yaw_tolerance_deg": 7,
        "min_clearance_m": 0.12,
        "max_speed_mps": 0.15,
    },
    preconditions=[
        "both_agents_in_contact",
        "distance(object_com, gate_entry) < 1.0",
    ],
    postconditions=[
        "abs(object_yaw - gate_axis_yaw) < 7deg",
        "min_clearance(object, gate_edges) > 0.12m",
    ],
    monitors=[
        "collision_pairs",
        "contact_loss",
        "object_tilt",
        "robot_fall",
        "progress_stagnation",
    ],
    fallback=[
        "stop_and_realign",
        "reverse_to_pre_gate_pose",
        "switch_pivot_anchor",
    ],
    timeout_s=20,
)
```

## Event-Driven Phases

| Phase | Goal | Exit Condition |
|---|---|---|
| scene_parse | build scene graph | objects, robots, obstacles, target, map available |
| task_classify | assign taxonomy labels | classification has required fields |
| retrieve_specs | load reusable constraints | specs ranked and bound to task context |
| plan_object_path | plan object SE(2) path | path satisfies clearance and yaw constraints |
| assign_roles | bind robot roles | role template and formation offsets chosen |
| approach_formation | move robots to contact poses | base pose error below threshold |
| establish_contact | lock push/grasp contact | contact state and relative pose stable |
| pre_align_object | align object to passage | yaw and clearance constraints satisfied |
| transport_to_gate | move object toward gate | object waypoint reached |
| narrow_passage | slow passage or pivot | no collision, clearance maintained |
| insert_target | final placement | target pose and yaw error below threshold |
| release_exit | release and withdraw | object stable and robots clear |

## Verifier Priority

1. Skill return values and exceptions.
2. Symbolic state: object pose, robot pose, contact, collision, fall state, target error.
3. Geometry: footprint clearance, gate yaw, pivot feasibility.
4. VLM: semantic failure explanation from camera traces.

## Deployment-Experience SpecMemory

Spec memory is centralized under:

```text
results/spec_memory/static/
results/spec_memory/dynamic/
```

SpecMemory is not a patch-selection policy and does not maintain a separate
meta-policy. A patch is only the immediate repair action for one failed rollout.
The long-term memory is the reusable deployment-calling experience extracted
from verified success and failure evidence.

Each deployment spec records:

- trigger conditions;
- executable constraints;
- recommended skill calls or command style;
- failure counterexamples;
- fallback behavior;
- success/failure evidence;
- confidence.

The orchestrator retrieves these specs before generating the initial SkillGraph
so the system can avoid repeating known deployment mistakes. For example, it can
insert `resync_contact`, require synchronized lateral movement, avoid repeated
pose adjustment without payload progress, lower speed, or choose a more direct
object path.

Each spec should contain:

```json
{
  "spec_id": "narrow_gate_pre_alignment_v1",
  "task_context": {
    "physical_coupling": "rigid_object",
    "object_type": "box_or_long_box",
    "environment": "narrow_gate",
    "formation": "front_back"
  },
  "rule_text": "Before entering a narrow passage, align the object's yaw with the gate axis and reduce speed until clearance is verified.",
  "trigger": [
    "distance(object_com, gate_entry) < 1.0m"
  ],
  "executable_predicate": [
    "abs(object_yaw - gate_axis_yaw) < 7deg",
    "min_clearance(object, gate_edges) > 0.12m",
    "object_speed < 0.15m/s"
  ],
  "fallback": [
    "stop_and_realign",
    "reverse_to_pre_gate_pose",
    "switch_pivot_anchor"
  ],
  "evidence": {
    "failure_cases": [],
    "success_after_patch": []
  },
  "confidence": 0.5
}
```

## Self-Evolution Loop

```text
1. Execute current skill graph.
2. Verifier records failure or suboptimal behavior.
3. Qwen3-8B identifies the failure mode from verifier state.
4. Qwen3-8B proposes one temporary typed repair patch for the next attempt.
5. Sandbox test on the failed seed.
6. Regression test on previous success seeds.
7. Randomized validation on new seeds.
8. Keep only verifier-passing behavior.
9. Store reusable deployment spec with evidence and confidence update.
```

Example success spec: during two-G1 long-box lateral motion or turning, both
robots should stabilize contact first and then use synchronized lateral
commands; one robot should not move early while the other compensates later.

Example failure spec: if repeated approach/pre-align phases do not move the
payload toward the target, stop adding more pose adjustment and switch to direct
synchronized transport after one contact resync.
