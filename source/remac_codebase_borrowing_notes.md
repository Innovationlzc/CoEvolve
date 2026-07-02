# REMAC Codebase 借鉴建议

> 主题：`REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration for Long-Horizon Robot Manipulation` 是否值得作为多机器人协同搬箱子项目的 codebase 参考，以及应该借鉴哪些模块。

## 0. 总结结论

REMAC 值得借鉴，而且比 COHERENT / SMART-LLM 更贴近当前的 Plan A++。但是建议只借鉴它的 **agent workflow、reflection loop、memory schema、command-to-skill 映射和 baseline 设计**，不要直接把 REMAC 当作 G1 双机器人刚性搬箱子的主系统底座。

推荐定位：

```text
REMAC = high-level reflective planning scaffold / baseline
CHORUS = decentralized multi-robot role prompting inspiration
SONIC / GEAR-SONIC = frozen whole-body motor layer
Ours = contract-constrained skill-level cooperative co-evolution
```

也就是说，不是“复现 REMAC”，而是“扩展 REMAC 到紧耦合刚性载荷协作”。

---

## 1. REMAC 最值得借鉴的 6 个部分

### 1.1 Self-reflection loop

REMAC 最核心的流程是：

```text
生成初始计划
   ↓
执行 subtask i 之前做 pre-condition check
   ↓
如果 pre-condition 不满足，反思原因并修正计划
   ↓
执行 subtask i
   ↓
执行后做 post-condition check
   ↓
如果 post-condition 不满足，retry 或局部 replan
   ↓
任务结束后，把 reflection 存入 memory
   ↓
下一次任务用 memory 优化 initial plan
```

这个闭环可以迁移到双 G1 搬箱子任务：

```text
执行 carry_forward_with_partner 前：
- both_grasp_contact == True
- box_yaw_error < 5 deg
- min_wall_clearance > 0.12 m
- front_back_distance_error < 0.06 m

执行后：
- box moved forward enough
- no collision
- no contact loss
- box tilt below threshold
```

这部分应该直接借鉴。

---

### 1.2 Reflection memory schema

REMAC 的 memory 里保存了任务、目标、环境信息、计划、执行结果、pre/post check 结果、reflection、final plan、useful plan 和 result。

可以作为第一版 memory schema 的参考。但你的项目不能只存普通 JSON 反思记录，而应该升级成 **Cooperative Spec Memory**。

示例：

```yaml
task_type: multi_robot_transport
physical_coupling: rigid_payload_tight_coupling
environment: narrow_corridor
failed_phase: enter_corridor

pre_check_results:
  - box_yaw_error = 9.8 deg
  - min_clearance = 0.07 m
  - rear_robot_lag = 0.11 m

reflection:
  - The box entered the corridor before yaw alignment was sufficient.
  - Rear robot lag caused clockwise payload rotation.

patch:
  - tighten yaw tolerance from 10 deg to 5 deg
  - reduce entry speed from 0.22 m/s to 0.15 m/s
  - add stop condition if rear lag > 0.06 m

reusable_spec:
  - Before entering a narrow corridor, require payload yaw error < 5 deg and side clearance > 12 cm.
```

区别是：

```text
REMAC memory = task-level reflection memory
你的 memory = physics-grounded cooperative specification memory
```

---

### 1.3 controller_dict / command-to-skill 映射

REMAC 用 `controller_dict.py` 把自然语言 command 映射到 hardcoded planner 或 checkpoint policy。这个设计非常适合改造成你的 `SkillRegistry`。

建议你的 G1 skill registry 长这样：

```python
skill_registry = {
    "walk_to_pose": {
        "type": "sonic_skill",
        "callable": sonic.walk_to_pose,
        "usage": "walk_to_pose(pose, speed)",
        "preconditions": ["robot_is_balanced", "path_is_clear"],
        "postconditions": ["at_target_pose"],
        "safety_guards": ["stop_if_obstacle_too_close"],
    },

    "grasp_box_handle": {
        "type": "manipulation_skill",
        "callable": sonic.grasp_box_handle,
        "usage": "grasp_box_handle(handle_id, force_threshold)",
        "preconditions": ["near_handle", "hand_empty"],
        "postconditions": ["grasp_contact == True"],
        "safety_guards": ["abort_if_force_too_high"],
    },

    "carry_forward_with_partner": {
        "type": "cooperative_skill",
        "callable": sonic.carry_forward_with_partner,
        "usage": "carry_forward_with_partner(role, speed, distance)",
        "preconditions": [
            "both_grasp_contact == True",
            "box_yaw_error < threshold",
            "partner_ready == True"
        ],
        "postconditions": [
            "box_progress > target_distance",
            "no_collision",
            "box_tilt < threshold"
        ],
        "safety_guards": [
            "stop_if_contact_loss",
            "stop_if_wall_clearance_low",
            "stop_if_partner_lag_large"
        ],
    }
}
```

关键价值：Coder Agent 不再自由写危险代码，只能调用合法 skill。

---

### 1.4 Two-agent planning scaffold

REMAC 的 two-agent planner 会生成类似：

```text
agent0: [action1, action2, action3, ...]
agent1: [action1, wait, action3, ...]
```

它要求两个 agent plan length 一样，并用 `wait` 对齐。这个对厨房任务有效，但不适合直接用于刚性箱体搬运。

双 G1 搬箱子应该升级成：

```python
joint_phase_graph = [
    Phase(
        name="approach",
        front_action="walk_to_front_handle",
        back_action="walk_to_back_handle",
        sync="both_at_grasp_pose"
    ),

    Phase(
        name="grasp_and_lift",
        front_action="grasp_and_lift_front",
        back_action="grasp_and_lift_back",
        sync="both_grasp_contact_and_height_aligned"
    ),

    Phase(
        name="pre_align_corridor",
        front_action="align_payload_leader",
        back_action="stabilize_payload_follower",
        sync="box_yaw_error < 5 deg and min_clearance > 0.12"
    ),

    Phase(
        name="enter_corridor",
        front_action="carry_forward_leader",
        back_action="follow_and_yaw_damp",
        sync="continuous_guarded_execution"
    )
]
```

因此 REMAC 的 two-agent planner 适合作为 baseline，而你的主方法应该是 **phase graph + cooperative contract**。

---

### 1.5 Ablation 设计

REMAC 的 ablation 很适合迁移：

```text
BASE: single-robot planning
CC: condition checking
RE: reflective evolution
REMAC: multi-agent collaboration + reflection + evolution
```

对应到你的项目：

| 你的 baseline | 对应 REMAC 思想 | 说明 |
|---|---|---|
| Fixed Script | BASE | 手写固定 phase，不反思、不进化 |
| Script + Condition Check | CC | 加 pre/post condition 和 guard |
| Script + Reflection | RE | 失败后让 Critic 总结并改脚本 |
| REMAC-style Multi-Agent | REMAC | 两机器人 LLM 生成协作计划，有 wait / retry / memory |
| Ours: Contract-CoEvolve | 你的完整方法 | phase graph + cooperative contract + structured judge + spec memory |

实验要证明：

```text
REMAC-style reflection 有用，
但不够处理紧耦合搬运；
你的 contract + structured judge + spec memory 更稳。
```

---

### 1.6 RoboCasa / RoboMimic 仿真接口

REMAC 基于 RoboCasa、RoboSuite、RoboMimic。这部分可以用于：

1. 快速跑通 agent loop：

```text
LLM plan → skill call → pre/post check → reflection → memory → next iteration
```

2. 做 REMAC-style baseline。

但它不是你的最终主环境。你的最终任务是：

```text
humanoid + rigid payload + corridor navigation
```

而 RoboCasa 主要是厨房 pick-place / open-close 等任务，物理耦合结构不同。

---

## 2. 不建议照搬的部分

### 2.1 不要照搬任务类型

REMAC 任务主要是：

```text
导航 → 抓物体 → 放物体 → 开门 / 开水龙头 / 开炉子
```

你的任务是：

```text
双机器人同时抓箱子 → 同步抬起 → 保持刚性箱体姿态 → 穿过窄走廊 → 同步放置
```

REMAC 是 long-horizon symbolic planning；你的任务是 long-horizon + tight physical coordination。

---

### 2.2 不要照搬 “wait 对齐”

REMAC 用 `wait` 对齐两个 agent 的 action list。搬箱子需要连续同步，不能只靠离散 wait。

你需要的是：

```text
phase barrier + continuous runtime guard + sync predicate
```

`wait` 可以保留为一个 skill：

```python
wait_until_partner_ready()
stop_and_hold_payload()
sync_barrier()
```

但不能作为主要协作机制。

---

### 2.3 不要只用 VLM 做 condition checking

REMAC 很多 check 由 VLM 判断，比如门是否打开、物体是否放进微波炉。你的场景关键量是物理和几何量：

```text
box_yaw_error
box_tilt
left/right wall clearance
front_back_distance_error
grasp_contact_force
partner_lag
```

这些应该由 Isaac 物理状态、机器人状态、接触传感器、几何计算来判断。VLM 只作为可选解释模块。

推荐 Judge：

```text
geometry judge + proprioception/contact judge + optional VLM explanation
```

---

### 2.4 不要照搬 task-name JSON retrieval

REMAC 的 memory 更偏 task-specific JSON。你的目标是跨箱子、跨走廊、跨初始姿态复用经验，所以应该做 physics-spec retrieval。

示例 query：

```text
multi_robot_transport
rigid_payload_tight_coupling
narrow_corridor
box_yaw_alignment
doorframe_collision
```

检索到：

```text
- narrow corridor entry alignment spec
- rear robot lag resync spec
- payload clearance guard spec
- synchronized lifting spec
```

---

## 3. 推荐项目代码结构

```text
project/
  agents/
    base_agent.py                  # 参考 REMAC MixAgent
    reflective_agent.py            # 参考 REMAC MixReAgent
    cooperative_agent.py           # 参考 REMAC MixCoAgent，但改成 phase graph
    critic_agent.py                # 你的失败归因模块
    coder_agent.py                 # 你的脚本 patch 模块

  skills/
    skill_registry.py              # 参考 REMAC controller_dict.py
    g1_navigation_skills.py
    g1_payload_skills.py
    cooperative_transport_skills.py

  checkers/
    precondition_checker.py        # REMAC checker 思想
    postcondition_checker.py
    payload_geometry_checker.py    # 你的核心新增
    contact_checker.py
    vlm_judge.py                   # 可选

  memory/
    interaction_memory.py          # 参考 REMAC save_interaction
    spec_memory.py                 # 你的核心新增
    schemas.py

  planner/
    phase_graph.py                 # 你的核心新增
    cooperative_contract.py        # 你的核心新增
    orchestrator.py

  executor/
    robot_executor.py
    two_robot_runtime.py
    trace_logger.py

  experiments/
    fixed_script_baseline.py
    condition_check_baseline.py
    remac_style_baseline.py
    contract_coevolve_ours.py
```

---

## 4. 可以直接参考的 REMAC 文件级模块

### 4.1 `agents/mix_agent.py`

借鉴点：

```text
Base agent class
history_plan
history_execution
history_pre_check_results
history_post_check_results
LLM/VLM client 管理
checker 分类
```

你的版本：

```python
class BaseRobotAgent:
    def __init__(self, robot_id, role, skill_registry, judge, memory):
        self.robot_id = robot_id
        self.role = role
        self.skill_registry = skill_registry
        self.history_phase = []
        self.history_skill_calls = []
        self.history_preconditions = []
        self.history_postconditions = []
        self.history_guard_violations = []
```

---

### 4.2 `agents/mix_reflective_agent.py`

借鉴点：

```text
retrieve_reminders()
retrieve_examples()
reflect_pre_check_result()
reflect_final_result()
get_useful_plan()
save_interaction()
load_interaction()
```

你的版本：

```text
retrieve_specs()
retrieve_failure_cases()
reflect_guard_violation()
extract_reusable_spec()
validate_spec_before_promotion()
save_rollout_trace()
```

---

### 4.3 `agents/mix_cooperative_agent.py`

借鉴点：

```text
two-agent plan generation
single-agent successful plan → two-agent parallel plan
wait / navigate aside 避免冲突
save_twoagent_plan()
```

你的版本：

```text
single_agent_transport_template
    ↓
two_robot_phase_graph
    ↓
front_role_script + back_role_script
    ↓
cooperative_contract
```

---

### 4.4 `robocasa/utils/controller_dict.py`

借鉴点：

```text
language command → planner / policy / checker
```

你的版本：

```python
skill_registry["carry_forward_with_partner"] = {
    "type": "cooperative_skill",
    "api": sonic.carry_forward_with_partner,
    "roles": ["front_leader", "back_follower"],
    "preconditions": [...],
    "postconditions": [...],
    "guards": [...],
    "failure_modes": [...]
}
```

---

## 5. REMAC 对项目的真正价值

### 5.1 支撑“自进化” framing

你的项目可以这样写 related work：

```text
REMAC demonstrates the effectiveness of reflective and self-evolving multi-agent planning for long-horizon manipulation. However, it focuses on loosely coupled kitchen tasks where robots can execute subtasks in parallel. Our work targets tightly coupled rigid-payload transport, where agents must satisfy continuous physical coordination constraints.
```

---

### 5.2 提供一个可运行 baseline

REMAC-style baseline：

```text
LLM 生成 agent0 / agent1 action list
每步执行前后做 condition check
失败后 reflection + retry
下一轮使用 memory
```

你的完整方法：

```text
LLM / rule orchestrator 生成 joint phase graph
每个 phase 有 cooperative contract
每个 skill 有 structured pre/post condition
运行时有 continuous guard
失败后 patch script + promote reusable spec
```

---

### 5.3 提供 prompt 和 API 设计参考

REMAC prompt 强制：

```text
只能使用 available commands
必须 step-by-step
必须遵守 action 格式
two-agent plan length should be same
use wait when one agent should wait
avoid conflict
```

你的 Coder / Orchestrator prompt 应该强制：

```text
只能调用 skill_registry 里的 skill
不能调用未注册 API
每个 phase 必须有 precondition / postcondition / guard
每个 cooperative phase 必须说明 front/back role
输出必须是 JSON / DSL，不允许自由文本
```

---

## 6. 推荐落地步骤

### Step 1：先跑通 REMAC repo

目的是理解它的 agent loop，而不是复现论文指标。

REMAC README 提供了三类入口：

```bash
# CC
python robomimic/robomimic/scripts/run_mix_multitask_agent.py \
  --env multitask_agent_env \
  --renderer "mjviewer"

# RE
python robomimic/robomimic/scripts/run_mix_multitask_agent_reflective.py \
  --env multitask_agent_env \
  --renderer "mjviewer"

# REMAC
python robomimic/robomimic/scripts/run_mix_multitask_twoagent.py \
  --env multitask_twoagent_env \
  --renderer "mjviewer"
```

---

### Step 2：抽象自己的 `SkillRegistry`

参考 REMAC 的 `controller_dict.py`，但换成 G1 / SONIC skills：

```python
skill_registry = {
    "walk_to_pose": ...,
    "grasp_box_handle": ...,
    "lift_payload": ...,
    "align_payload_to_corridor": ...,
    "carry_forward_with_partner": ...,
    "stop_and_resync": ...,
    "place_payload": ...
}
```

---

### Step 3：实现 REMAC-style baseline

先做简单版本：

```text
front: [approach, grasp, lift, align, carry, place]
back:  [approach, grasp, lift, stabilize, follow, place]
```

失败后用 REMAC 式 reflection 修改下一轮计划。

---

### Step 4：实现完整方法

在 REMAC-style baseline 上加入：

```text
phase graph
cooperative contract
runtime guards
structured trace
critic root-cause analysis
coder patch
spec memory promotion
```

---

### Step 5：实验中明确对比

目标证明：

```text
REMAC-style reflection 有用，
但不够处理紧耦合搬运；
Contract-CoEvolve 更稳、更可复用。
```

---

## 7. 最终技术路线表达

推荐论文里把你的方法写成：

```text
REMAC-style reflective evolution
+ CHORUS-inspired decentralized role prompting
+ contract-constrained skill orchestration
+ SONIC / GEAR-SONIC frozen whole-body execution
```

一句话版本：

> REMAC provides the reflective planning scaffold; our contribution is to make it physically grounded for tightly coupled humanoid payload transport through phase graphs, cooperative contracts, continuous runtime guards, and reusable cooperative specifications.
