# 多机器人协同进化项目方案整理

## 0. 总体建议

我建议不要把主线放在完整复现 **HALyPO**，也不要一开始就做完整的 **CHORUS-style 端到端 VLA**。

更稳、更容易落地、也更容易形成清晰贡献的路线是：

# Plan A++：CHORUS-inspired Skill-Level Co-Evolution

核心思想是：

> 底层运动控制固定交给 SONIC / Whole-Body Control / Skill SDK；上层做一个会生成、执行、诊断、修复、记忆协作规范的多机器人 skill orchestration 系统。

也就是说，项目重点不是训练底层连续动作控制器，而是研究：

> LLM / Agent 如何正确、高效、可复用地调用 skill library，让两个机器人完成协同搬箱子任务，并在失败后自动总结、修复和进化。

形式上，底层连续动作策略可以写成：

\[
\pi(a_{t:t+H}^{i}\mid o_t^i, c_i)
\]

但我们建议第一阶段不要直接学习连续动作，而是学习或进化 skill-level action：

\[
\pi(u_t^i \mid z_t^i, c_i, \mathcal{M})
\]

其中：

- \(i\)：机器人编号，例如 `G1-front` / `G1-back`。
- \(z_t^i\)：从视觉、proprioception、箱子位姿、走廊几何中抽象出来的结构化状态。
- \(c_i\)：机器人身份和角色 prompt。
- \(\mathcal{M}\)：长期协作规范记忆库。
- \(u_t^i\)：skill call，例如 `grasp_box_handle`、`align_payload_to_corridor`、`carry_forward_with_yaw_guard`、`wait_until_partner_ready`。

最终你的“自进化”应该定义为：

> 进化的不是底层 motor control，而是两个机器人如何选择 skill、如何给 skill 填参数、如何设置同步条件、如何根据失败修复脚本、如何把成功/失败提炼成可复用的协作 specification。

---

## 1. 为什么不建议主线做 HALyPO / Plan C？

HALyPO 的核心是异构多智能体强化学习。它把人机协作看成 decentralized POMDP，并解决独立策略梯度中可能出现的 rationality gap、非保守向量场和训练振荡问题。它的方法是通过 Lyapunov projection 约束 policy-parameter space 里的更新方向。

但是它不适合作为当前项目主线，原因如下：

### 1.1 研究粒度不一致

HALyPO 是 **policy-parameter learning**，核心是训练策略参数；而当前项目想做的是：

```text
LLM / Agent 如何调用 skill SDK 完成多机器人协同搬运。
```

也就是说，HALyPO 研究的是底层或中层策略训练，你想研究的是上层 skill orchestration 和 code evolution。两者粒度不一样。

### 1.2 Reward 和仿真依赖很重

协同搬箱子穿走廊这类任务，如果用强化学习做，需要设计大量 reward，例如：

- 任务进度 reward。
- 箱体倾斜 penalty。
- 掉落 penalty。
- 碰撞 penalty。
- 接近障碍物 penalty。
- 双机器人同步 penalty。
- 箱体姿态稳定 penalty。

这些 reward 很难调，而且调不好会出现 reward hacking 或策略不稳定。

### 1.3 代码复现风险高

HALyPO 没有明确可直接复现的开源代码，且如果没有作者环境，复现难度会很高。

因此，HALyPO 更适合作为：

- motivation 参考；
- evaluation metric 参考；
- 论文 related work；
- 仿真 MARL baseline 的理论背景；

而不适合作为主线方案。

---

## 2. 为什么不建议一开始完整复现 CHORUS / Plan B？

CHORUS 的思想非常适合启发这个项目：

> 多个机器人共享一个 policy，每个机器人只用自己的本地 observation 和 identity prompt，独立执行协作任务。

但是完整 CHORUS 对当前项目来说风险较大。

### 2.1 需要大量真实多机器人示范数据

CHORUS 是 VLA fine-tuning，不是纯零样本 planning。它需要多机器人协作示范数据。

对于两个 G1 搬箱子穿走廊任务，需要收集高质量数据，包括：

- 两个机器人靠近箱子；
- 同步抓取；
- 同步抬起；
- 箱体对齐走廊；
- 通过走廊；
- 到达目标点；
- 同步放置；
- 异常恢复。

这个数据采集成本很高。

### 2.2 连续动作空间太复杂

两个 G1 humanoid 共同搬箱子，是紧耦合接触任务，涉及：

- whole-body balance；
- 双臂抓取；
- 箱体刚体约束；
- 双机器人相对姿态；
- 走廊窄空间避障；
- contact-rich manipulation；
- 同步步态和载荷稳定。

如果直接让 VLA 输出连续动作，难度会明显高于移动机械臂搬运任务。

### 2.3 局部观察可能不够

CHORUS 的前提之一是：每个机器人本地 observation 里必须包含足够的队友和 workspace 信息。

但两个 humanoid 搬箱子时，箱子可能遮挡：

- 队友；
- 走廊边界；
- 门框；
- 箱体后端；
- 地面障碍。

如果本地观测不完整，直接 decentralized VLA 会很难。

### 2.4 更合理的做法

不要第一阶段完整复现 CHORUS，而是做：

```text
CHORUS-inspired skill-level system
```

也就是保留 CHORUS 的两个思想：

1. 每个机器人根据自己的局部状态和身份 prompt 执行；
2. 多机器人共享一套协作规范和经验表示；

但 action space 从连续动作变成 skill call。

第一阶段：

\[
\pi(u_t^i \mid z_t^i, c_i, \mathcal{M})
\]

后续有数据后，再升级到：

\[
\pi(a_{t:t+H}^i \mid o_t^i, c_i)
\]

---

## 3. 原始 Plan A 的主要问题

原始 Plan A 的方向是对的，但还缺少几个关键模块。

---

## 3.1 问题一：缺少状态抽象层

不能直接把 raw image 全部丢给 LLM，让 LLM 从像素中判断所有任务状态。

更合理的结构是：

```text
Raw Vision / Sensor Data
        ↓
State Estimator / VLM Judge / Geometry Module
        ↓
Structured State
        ↓
Orchestrator / Coder / Critic
```

上层 agent 应该看到结构化状态，例如：

```python
state = {
    "task_phase": "enter_corridor",
    "robots": {
        "G1_front": {
            "pose": [x, y, yaw],
            "role": "navigation_leader",
            "grasp_contact": True,
            "handle_height": 0.72,
            "local_obstacles": [...]
        },
        "G1_back": {
            "pose": [x, y, yaw],
            "role": "stability_follower",
            "grasp_contact": True,
            "handle_height": 0.70,
            "local_obstacles": [...]
        }
    },
    "box": {
        "center": [x, y],
        "yaw": theta_box,
        "tilt_roll": roll,
        "tilt_pitch": pitch,
        "left_wall_clearance": 0.18,
        "right_wall_clearance": 0.22
    },
    "corridor": {
        "centerline_yaw": theta_corridor,
        "width": 1.25,
        "entrance_pose": [x, y, yaw],
        "goal_pose": [x, y, yaw]
    },
    "sync": {
        "front_back_distance_error": 0.04,
        "yaw_error_to_corridor": 3.2,
        "relative_speed_error": 0.01
    }
}
```

LLM 不应该每一步自己从像素里猜“箱子是不是歪了”。
VLM 可以辅助判断，但最终应该转成 symbolic predicates 和几何指标。

---

## 3.2 问题二：缺少协作 contract

如果让两个 agent 各自优化代码，很容易互相冲突。

例如：

- G1-front 觉得应该“快点走”；
- G1-back 觉得应该“慢点稳”；
- 两个脚本各自变好，但组合起来更差。

因此必须引入：

# Joint Cooperative Contract

也就是每个阶段都有共同遵守的协作合同。

例如进入走廊阶段：

```python
contract_enter_corridor = {
    "preconditions": [
        "both_grasp_contact == True",
        "abs(box_yaw - corridor_yaw) < 5_deg",
        "box_left_wall_clearance > 0.12_m",
        "box_right_wall_clearance > 0.12_m"
    ],
    "front_robot_duties": [
        "control_forward_progress",
        "keep_box_center_on_corridor_centerline",
        "stop_if_back_robot_reports_instability"
    ],
    "back_robot_duties": [
        "stabilize_box_yaw",
        "maintain_handle_height",
        "stop_if_contact_loss_or_wall_risk"
    ],
    "global_guards": [
        "if tilt_roll > 8_deg: stop_and_rebalance",
        "if wall_clearance < 0.08_m: stop_and_realign",
        "if front_back_distance_error > 0.10_m: stop_and_resync"
    ]
}
```

Coder Agent 可以修改脚本，但不能违反 contract。

这样系统从：

```text
两个机器人各自写代码
```

变成：

```text
两个机器人在共享协作协议下共同进化
```

---

## 3.3 问题三：step-by-step 不能靠 VLM 临时猜

任务执行过程中的 step boundary 不应该由 VLM 自由猜。

建议使用显式 phase graph。

对于两个 G1 搬箱子穿走廊，可以定义：

| Phase | G1-front | G1-back | 退出条件 | 核心风险 |
|---|---|---|---|---|
| P0 Scene Scan | 识别走廊入口、目标点、箱子前端 | 识别箱子后端、后方障碍 | 箱子/走廊/目标位姿置信度足够 | 感知错误 |
| P1 Approach | 走到箱子前端抓取位 | 走到箱子后端抓取位 | 双方到达抓取位 | 起始姿态不对 |
| P2 Grasp & Lift | 抓住前端并抬到高度 h | 抓住后端并抬到高度 h | 双方 contact=True 且高度差小 | 一方未抓稳 |
| P3 Pre-align | 主导箱体 yaw 对齐走廊 | 稳定箱体姿态 | \(|\theta_{box}-\theta_{corridor}|<\epsilon\) | 箱子斜着进门 |
| P4 Enter Corridor | 慢速前进、控制中心线 | 抑制 yaw 和尾部碰撞 | 箱体中心进入走廊 | 撞门框 |
| P5 Corridor Traverse | 沿中心线前进 | 保持箱体水平和尾部安全 | 箱体通过走廊 | 左右 clearance 不够 |
| P6 Exit & Place | 到目标点后降箱 | 同步降箱 | 箱子放稳且双方松手 | 放置不平稳 |
| P7 Retreat | 后退离开 | 后退离开 | 安全退出 | 踩碰箱子 |

每个 phase 是 step。
每个 skill call 是 sub-step。
每个 skill call 都必须返回结构化结果。

例如：

```python
result = skill.grasp_box_handle(
    robot="G1_front",
    handle="front",
    max_time=8.0,
    force_threshold=12.0,
)

# result
{
    "success": True,
    "failure_reason": None,
    "metrics": {
        "contact_force": 14.2,
        "grasp_confidence": 0.91,
        "handle_pose_error": 0.018
    },
    "start_frame": "...",
    "end_frame": "..."
}
```

step boundary 可以由下面事件确定：

- skill start；
- skill success；
- skill failure；
- phase entry；
- phase exit；
- guard violation；
- sync barrier reached。

VLM judge 只在这些边界和异常发生时介入，而不是每个 control tick 都介入。

---

## 3.4 问题四：VLM as judge 不能单独负责成功/失败判断

可以用 VLM judge，但不能只靠 VLM。

更稳的 judge 应该是：

\[
J = J_{geometry} + J_{proprioception} + J_{contact} + J_{VLM}
\]

也就是：

1. 几何 judge：箱子中心线、yaw、clearance、目标距离。
2. Proprioception judge：机器人姿态、速度、相对距离、手臂高度。
3. Contact judge：是否抓稳、是否掉落、是否力异常。
4. VLM judge：识别“箱子撞到门框”“后端快碰墙”“队友没跟上”等语义事件。

VLM judge 的输出应该是结构化判断，而不是自由文本控制指令。

例如：

```json
{
  "phase": "P4_enter_corridor",
  "progress": 0.42,
  "event": "near_collision",
  "evidence": "box front-right corner is too close to the door frame",
  "suggested_failure_cause": "box_yaw_not_aligned_before_entering",
  "confidence": 0.83,
  "predicates": {
    "box_aligned_with_corridor": false,
    "front_robot_too_fast": false,
    "back_robot_lagging": true,
    "wall_clearance_safe": false
  }
}
```

---

## 3.5 问题五：spec memory 不能只存自然语言

长期记忆库不应该只是自然语言经验，例如“下次注意不要撞门框”。

建议至少存四类信息。

### A. Textual Rule

```text
When carrying a rigid box through a narrow corridor, align the box yaw with the corridor centerline before entering. Do not enter if yaw error is above 5 degrees or side clearance is below 12 cm.
```

### B. Predicate Contract

```yaml
condition:
  task_type: joint_payload_transport
  environment: narrow_corridor
  object_shape: rigid_box
  bottleneck: true

precondition:
  - abs(box_yaw - corridor_yaw) < 5_deg
  - min(left_clearance, right_clearance) > 0.12_m
  - both_grasp_contact == true

recommended_skills:
  - align_payload_to_corridor
  - carry_forward_with_yaw_guard
  - stop_and_resync_if_contact_loss
```

### C. Visual Memory

保存一段失败视频 / 成功视频，并配 caption：

```text
Failure clip: The box entered the corridor with a 15-degree yaw error. The front-right corner hit the door frame. The correct recovery is to stop, back up 20 cm, rotate counterclockwise, then re-enter.
```

### D. Trajectory / Skill Graph Template

```python
successful_template = [
    ("approach_handle", {"side": "front/back"}),
    ("grasp_box_handle", {"force_threshold": 12}),
    ("lift_sync", {"height": 0.72, "height_tolerance": 0.03}),
    ("align_payload_to_corridor", {"yaw_tolerance_deg": 5}),
    ("carry_through_corridor", {"speed": 0.18, "clearance_margin": 0.12}),
    ("place_box", {"target_pose_tolerance": 0.05})
]
```

这样 memory 才能真正被下次任务复用。

---

## 4. 最终系统架构

建议系统命名为：

# Co-SpecEvolve

全称：

```text
Cooperative Specification Evolution for Skill-Based Multi-Robot Transport
```

整体架构如下：

```text
Human Instruction
        |
        v
[Task Classifier + Spec RAG]
        |
        v
[Task Orchestrator]
        |
        v
Joint Phase Graph + Cooperative Contracts
        |
        +--------------------+
        |                    |
        v                    v
[G1-front Skill Script]   [G1-back Skill Script]
        |                    |
        v                    v
 Skill SDK / SONIC / WBC   Skill SDK / SONIC / WBC
        |                    |
        +---------+----------+
                  |
                  v
       Execution Trace Logger
                  |
                  v
      Progress Judge / Failure Localizer
                  |
                  v
      Critic Agent -> Coder Agent
                  |
                  v
        Updated Scripts + Spec Memory
```

关键原则：

- Orchestrator 不要每一步都介入。
- Orchestrator 只在任务开始、phase 切换、异常恢复、任务重规划时介入。
- 执行过程主要靠 skill script + runtime guards。
- 失败之后再由 Critic / Coder / Spec Memory 进入进化闭环。

---

## 5. 任务分类 taxonomy

你提到想按照主客体、合作类别、博弈类别等分类。建议使用下面这套 taxonomy。

```yaml
task_type:
  - single_robot
  - multi_robot_transport
  - multi_robot_handover
  - multi_robot_assembly
  - multi_robot_navigation
  - multi_robot_inspection

physical_coupling:
  - none
  - loose_coupling
  - object_mediated_coupling
  - rigid_payload_tight_coupling
  - direct_body_contact

coordination_topology:
  - independent_parallel
  - leader_follower
  - symmetric_joint_action
  - master_assistant
  - relay
  - scout_executor
  - judge_executor

game_type:
  - fully_cooperative
  - mixed_motive
  - competitive
  - adversarial

temporal_sync:
  - asynchronous
  - phase_barrier
  - continuous_synchronization
  - simultaneous_contact_required

observability:
  - local_only
  - local_plus_peer_message
  - local_plus_global_judge
  - centralized_global_state

environment_constraint:
  - open_space
  - narrow_corridor
  - doorway
  - clutter
  - slope
  - dynamic_obstacles

object_constraint:
  - rigid
  - deformable
  - fragile
  - heavy
  - large_aspect_ratio
```

你的任务可以标记为：

```yaml
task_type: multi_robot_transport
physical_coupling: rigid_payload_tight_coupling
coordination_topology: leader_follower
game_type: fully_cooperative
temporal_sync: continuous_synchronization + phase_barrier
observability: local_plus_global_judge
environment_constraint: narrow_corridor
object_constraint: rigid + large + heavy
```

这套标签的用途不是装饰，而是用于 RAG 检索 spec memory。

例如当前任务会检索：

```text
multi_robot_transport + rigid_payload_tight_coupling + narrow_corridor + leader_follower
```

然后取出历史经验，例如：

- 进入窄通道前必须对齐箱体 yaw。
- 前机器人负责 progress，后机器人负责 yaw damping。
- 不能只看目标点导航，要看箱体四角 clearance。
- 如果后机器人视觉丢失，前机器人必须停。
- 如果箱体 yaw error 增大，先停止平移，再原地微调。

---

## 6. 两个 G1 的角色分配

你原来的设定是：

- G1-front：导航 / 门框对位主导；
- G1-back：姿态稳定 / 尾部避障主导。

这个是合理的。

更具体地，可以定义如下。

---

## 6.1 G1-front：Navigation Leader

主要职责：

1. 估计 corridor entrance 和 goal direction。
2. 控制箱体整体前进速度。
3. 控制箱体中心尽量沿 corridor centerline。
4. 发现前方门框 / 障碍。
5. 如果后方不同步，主动 stop。

允许调用的主要 skill：

```python
walk_to_front_handle()
grasp_front_handle()
lift_payload_front()
align_payload_to_corridor_leader()
carry_forward_leader()
slow_enter_corridor()
stop_and_hold_payload()
place_payload_front()
```

---

## 6.2 G1-back：Stability Follower

主要职责：

1. 保持箱体水平。
2. 稳定箱体 yaw，不让尾部甩动。
3. 观察后端和侧后方 clearance。
4. 如果箱体倾斜、后方碰撞风险、抓取不稳，发出 stop/resync。
5. 在转弯或入门时做 yaw correction。

允许调用的主要 skill：

```python
walk_to_back_handle()
grasp_back_handle()
lift_payload_back()
stabilize_payload_yaw()
follow_front_motion()
guard_rear_clearance()
stop_and_hold_payload()
place_payload_back()
```

注意：G1-back 可以提供辅助视觉信息，但不建议把它设计成纯 judge，因为它也是载荷搬运的一部分。

Judge 最好是软件模块，输入来自：

- G1-front camera；
- G1-back camera；
- 可选外部固定相机 / 手机 / 第三视角相机；
- 机器人 proprioception；
- 箱体位姿估计；
- contact / force 信息。

---

## 7. 视觉输入设计

不要笼统说“输入图像”。建议按模块分别定义输入。

---

## 7.1 Orchestrator 初始输入

任务开始时输入：

```python
initial_input = {
    "instruction": "Two G1 robots carry the box through the corridor and place it at the goal.",
    "front_rgbd": image_or_video_from_G1_front,
    "back_rgbd": image_or_video_from_G1_back,
    "optional_global_view": fixed_camera_or_phone_view,
    "robot_capabilities": skill_library_schema,
    "scene_map": optional_2d_map_or_slam_map
}
```

Orchestrator 输出：

```python
joint_phase_graph
front_robot_script
back_robot_script
cooperative_contracts
runtime_guards
```

---

## 7.2 Executor 运行时输入

每个机器人自己的 executor 只需要本地输入：

```python
local_runtime_obs = {
    "rgbd": onboard_camera,
    "robot_pose": base_pose,
    "joint_state": joint_state,
    "imu": imu,
    "gripper_contact": contact_or_force,
    "estimated_box_relative_pose": box_pose_in_robot_frame,
    "partner_visible": true_or_false
}
```

这部分用于执行 skill，不必每一步都给 LLM。

---

## 7.3 Judge 输入

Judge 在 phase boundary 或异常时使用：

```python
judge_input = {
    "recent_video_clips": {
        "front_view": last_3_seconds,
        "back_view": last_3_seconds,
        "global_view": last_3_seconds
    },
    "state_trace": structured_states,
    "skill_trace": executed_skill_calls,
    "failure_signal": exception_or_guard_violation
}
```

Judge 输出：

```python
judge_output = {
    "phase_success": False,
    "failed_phase": "P4_enter_corridor",
    "failure_type": "collision_with_doorframe",
    "root_cause_candidates": [
        "box_yaw_not_aligned",
        "front_robot_entered_before_back_robot_stabilized"
    ],
    "evidence": [
        "yaw error increased from 4 deg to 13 deg before collision",
        "right clearance dropped below 5 cm"
    ]
}
```

---

## 7.4 Evolver 输入

Critic / Coder 不应该看完整视频，而应该看结构化 trace + 关键帧。

```python
evolution_input = {
    "task_config": task_taxonomy,
    "original_scripts": scripts_v3,
    "contracts": contracts_v3,
    "execution_trace": trace,
    "judge_report": judge_output,
    "retrieved_specs": top_k_specs
}
```

Evolver 输出：

```python
patch = {
    "script_diff": "...",
    "contract_update": "...",
    "new_spec_candidate": "...",
    "validation_tests": [...]
}
```

---

## 8. Skill library 设计

Skill library 建议分四层。

---

## 8.1 Layer 1：基础运动 skill

```python
walk_to_pose(pose, speed, obstacle_avoidance=True)
turn_in_place(yaw)
sidestep(direction, distance)
stop()
hold_position(duration)
```

---

## 8.2 Layer 2：抓取 / 载荷 skill

```python
approach_box_handle(handle_id, approach_pose)
grasp_box_handle(handle_id, force_threshold)
lift_payload(height, tolerance)
lower_payload(height)
release_payload()
maintain_grip(force_range)
```

---

## 8.3 Layer 3：协作 transport skill

```python
sync_barrier(name, required_agents, timeout)
wait_until(predicate, timeout)
align_payload_to_corridor(yaw_target, tolerance)
carry_forward_with_partner(speed, distance, formation)
pivot_payload(angle, pivot_role)
rebalance_payload(height_tolerance, tilt_tolerance)
stop_and_resync()
```

---

## 8.4 Layer 4：judge / recovery skill

```python
estimate_box_pose()
estimate_corridor_centerline()
check_payload_clearance()
diagnose_contact_loss()
recover_from_wall_near_collision()
backtrack_and_realign(distance)
```

每个 skill 都必须有 schema：

```python
class SkillResult:
    success: bool
    failure_reason: str | None
    metrics: dict
    start_time: float
    end_time: float
    start_state: dict
    end_state: dict
    frames: list
```

并且每个 skill 要声明：

```python
skill.preconditions
skill.effects
skill.failure_modes
skill.safety_guards
```

这一点非常重要。
否则 Coder Agent 只是在生成普通 Python 代码，无法知道每个 skill 的能力边界。

---

## 9. Gen-0 脚本形式

不要让 Orchestrator 输出完全自由的 Python。

建议输出受限制的 DSL / Python-like plan。

---

## 9.1 G1-front plan 示例

```python
front_plan = [
    Phase(
        name="approach",
        action=walk_to_pose(target="front_handle_pose", speed=0.25),
        exit_when="at_pose(front_handle_pose, tol=0.05)"
    ),

    Phase(
        name="grasp",
        action=grasp_box_handle(handle_id="front", force_threshold=12.0),
        exit_when="grasp_contact == True"
    ),

    Phase(
        name="lift_sync",
        action=lift_payload(height=0.72, tolerance=0.03),
        sync="both_lifted",
        exit_when="both_grasp_contact and abs(front_height-back_height)<0.03"
    ),

    Phase(
        name="align_corridor",
        action=align_payload_to_corridor(
            role="leader",
            yaw_tolerance_deg=5,
            centerline_tolerance=0.08
        ),
        exit_when="abs(box_yaw-corridor_yaw)<5 and min_clearance>0.12"
    ),

    Phase(
        name="enter_corridor",
        action=carry_forward_with_partner(
            role="leader",
            speed=0.18,
            distance=1.0,
            guards=[
                "min_clearance > 0.08",
                "both_grasp_contact == True",
                "abs(box_roll) < 8_deg",
                "back_robot_ready == True"
            ]
        ),
        exit_when="box_center_passed_corridor_entrance"
    )
]
```

---

## 9.2 G1-back plan 示例

```python
back_plan = [
    Phase(
        name="approach",
        action=walk_to_pose(target="back_handle_pose", speed=0.25),
        exit_when="at_pose(back_handle_pose, tol=0.05)"
    ),

    Phase(
        name="grasp",
        action=grasp_box_handle(handle_id="back", force_threshold=12.0),
        exit_when="grasp_contact == True"
    ),

    Phase(
        name="lift_sync",
        action=lift_payload(height=0.72, tolerance=0.03),
        sync="both_lifted",
        exit_when="both_grasp_contact and abs(front_height-back_height)<0.03"
    ),

    Phase(
        name="align_corridor",
        action=stabilize_payload_yaw(
            role="follower",
            yaw_target="corridor_yaw",
            yaw_tolerance_deg=5
        ),
        exit_when="abs(box_yaw-corridor_yaw)<5"
    ),

    Phase(
        name="enter_corridor",
        action=follow_front_motion(
            role="rear_stabilizer",
            max_speed=0.18,
            yaw_damping=True,
            rear_clearance_guard=0.08
        ),
        exit_when="box_center_passed_corridor_entrance"
    )
]
```

这种形式的好处：

1. Coder Agent 修改的是参数、guard、phase 顺序，不会乱写危险代码。
2. Judge 可以明确知道当前在哪个 phase。
3. Evolution 可以做 diff。
4. Spec memory 可以直接保存 phase graph。

---

## 10. 进化闭环设计

---

## 10.1 一次 rollout 结束后的 trace

系统得到：

```python
trace = {
    "task_id": "box_corridor_001",
    "script_version": "v3",
    "success": False,
    "failed_phase": "enter_corridor",
    "failure_event": "box_hit_right_doorframe",
    "metrics": {
        "max_yaw_error_deg": 14.7,
        "min_wall_clearance_m": 0.03,
        "front_back_speed_error": 0.09,
        "time_to_failure": 42.1
    },
    "skill_trace": [...],
    "video_snippets": [...]
}
```

---

## 10.2 Judge 定位失败

Judge 输出：

```text
Failure happened in P4_enter_corridor.
The box was not sufficiently aligned before entering.
Yaw error exceeded the contract threshold after the front robot started forward.
The rear robot lagged and introduced clockwise rotation.
```

---

## 10.3 Critic 生成 root cause

Critic 不应该只说“撞门框了”。

它要输出可修复原因：

```yaml
root_cause:
  type: contract_missing_or_too_weak
  phase: P4_enter_corridor
  evidence:
    - yaw_error_before_entry was 8.5 deg, but current precondition allowed 10 deg
    - rear robot lagged by 0.12 m
    - speed 0.22 m/s was too high for corridor width 1.15 m
  proposed_fix:
    - tighten yaw tolerance from 10 deg to 5 deg
    - reduce entry speed from 0.22 to 0.15
    - add stop condition when rear lag > 0.06 m
    - add backtrack_and_realign recovery if clearance < 0.08 m
```

---

## 10.4 Coder Agent patch 脚本

```diff
Phase(
    name="align_corridor",
    action=align_payload_to_corridor(
        role="leader",
-       yaw_tolerance_deg=10,
+       yaw_tolerance_deg=5,
        centerline_tolerance=0.08
    ),
-   exit_when="abs(box_yaw-corridor_yaw)<10"
+   exit_when="abs(box_yaw-corridor_yaw)<5 and min_clearance>0.12"
)

Phase(
    name="enter_corridor",
    action=carry_forward_with_partner(
        role="leader",
-       speed=0.22,
+       speed=0.15,
        distance=1.0,
        guards=[
            "min_clearance > 0.08",
            "both_grasp_contact == True",
            "abs(box_roll) < 8_deg",
+           "front_back_distance_error < 0.06",
            "back_robot_ready == True"
        ]
    )
)
```

---

## 10.5 生成新 spec candidate

```yaml
spec_name: narrow_corridor_entry_alignment
condition:
  task_type: multi_robot_transport
  physical_coupling: rigid_payload_tight_coupling
  environment_constraint: narrow_corridor
rule:
  Before entering a corridor, require payload yaw error < 5 deg and side clearance > 12 cm.
  During entry, leader speed should be reduced below 0.15-0.18 m/s.
  If rear robot lag exceeds 6 cm, stop translation and resynchronize before continuing.
evidence:
  source_task: box_corridor_001
  failure_before: collision_with_right_doorframe
  patch_effect: success_after_patch
```

---

## 10.6 进入长期记忆的条件

不要每次失败都直接写入长期记忆。

应该有 promotion rule：

```python
promote_to_memory = (
    patch_improves_success_rate_on_validation
    and no_new_safety_violation
    and spec_generalizes_to_at_least_2_scene_variants
)
```

也就是说，spec memory 不是聊天记录，而是经过验证的经验库。

---

## 11. 技术路线选择结论

| 路线 | 是否主线 | 原因 |
|---|---|---|
| Plan A 原版：LLM orchestrator + skill scripts | 可以做，但要增强 | 方向对，但缺 state abstraction、contract、judge、spec validation |
| Plan A++：contract-constrained skill co-evolution | 主线推荐 | 最容易落地，能体现“自进化”，可以真机跑，可以做 ablation |
| Plan B：完整 CHORUS VLA | 第二阶段 / optional baseline | 思想非常好，但需要多机器人示范和 VLA fine-tuning，难度更高 |
| Plan B-light：CHORUS-inspired skill policy | 建议作为后续亮点 | 用共享 policy / LLM 预测 next skill，而不是连续动作，容易很多 |
| Plan C：HALyPO | 不建议主线 | 需要 MARL、reward、仿真、策略训练；代码不明确 |
| COHERENT / SMART-LLM baseline | 推荐做 planning baseline | 适合比较你的 evolution/spec memory 是否有效 |
| Voyager-style skill memory | 推荐借鉴 | code-as-action + skill library + iterative prompting 很符合 skill evolution 思想 |

---

## 12. 推荐 code baseline

---

## 12.1 高层多机器人 planning baseline

### Baseline 1：COHERENT

COHERENT 是 LLM-based heterogeneous multi-robot task planning 框架，核心是：

```text
Proposal -> Execution -> Feedback -> Adjustment
```

适合作为你的 planning baseline。

你要超过它的地方是：

```text
COHERENT 没有强 spec memory，没有物理协作 contract，没有代码版本进化。
你的方法有。
```

### Baseline 2：SMART-LLM

SMART-LLM 把高层自然语言任务转成 multi-robot task plan，包含：

- task decomposition；
- coalition formation；
- task allocation。

它适合作为更简单的 task decomposition baseline。

对比方式：

```text
SMART-LLM:
Instruction -> task decomposition -> allocation

Your method:
Instruction -> decomposition -> execution trace -> failure diagnosis -> script/spec evolution
```

### Baseline 3：Voyager-style Code-as-Action

Voyager 的关键思想是用 LLM 生成可执行代码，用 skill library 存储复杂行为，并通过环境反馈迭代改进。

你可以把它改成 multi-robot robotics 版本：

```text
Voyager:
Minecraft skill evolution

Your method:
Multi-robot physical skill evolution
```

---

## 12.2 VLA baseline

### Baseline 4：openpi / π0.5

openpi 是最接近 CHORUS 的开源路线，因为 CHORUS 基于 π0.5。

建议不要一开始用它直接输出 G1 连续动作。

更稳的是先做：

```text
openpi / π0.5 as skill selector:
image + role prompt + state summary -> next skill / phase
```

后续数据够了再做：

```text
openpi / π0.5 as action policy:
local images + identity prompt -> action chunk
```

### Baseline 5：OpenVLA

OpenVLA 是 open-source VLA，可以做 baseline。

但它通常需要一定规模 target-domain demos，而且高频动作不一定适合，因此不如 skill-level evolution 稳。

### Baseline 6：RDT-1B

RDT-1B 是 diffusion transformer，可以根据语言和多视角 RGB 图像预测未来动作 chunk。

如果后续想做 learned policy baseline，可以考虑它。

但对主线来说，它仍然比 skill-level evolution 更重。

---

## 12.3 Simulation / MARL baseline

如果要做仿真实验，可以考虑 Isaac Lab。

建议仿真只用于：

1. 评估脚本进化；
2. 产生失败场景；
3. 做 baseline；
4. 不要一开始大规模训练 MARL policy。

---

## 13. 方法贡献应该怎么写

如果只是“LLM 调 skill 搬箱子”，贡献不够强。

建议把贡献定义成下面三点。

---

## 13.1 Contribution 1：Contract-constrained multi-robot skill orchestration

不是普通 LLM planning，而是生成：

```text
joint phase graph + per-robot scripts + cooperative contracts + runtime guards
```

这比普通任务分解方法更贴近真实物理协作。

---

## 13.2 Contribution 2：Execution-grounded cooperative specification evolution

不是普通 memory，而是从真实失败中提炼：

```text
failure trace -> root cause -> code patch -> validation -> reusable spec
```

spec 不是自然语言日记，而是：

```text
text + predicate + visual evidence + trajectory template
```

---

## 13.3 Contribution 3：CHORUS-inspired decentralized execution at skill level

每个机器人运行自己的 script，只看本地 observation 和 role prompt。

共享的是任务前检索到的 spec 和共同 contract，而不是实时 centralized action control。

这借鉴 CHORUS 的 decentralized local observation + identity prompt 思想，但你的 action space 是 skill-level program，不是 continuous VLA action。

---

## 14. 实验设计

---

## 14.1 主任务

两个 G1 抬箱子：

```text
start area -> corridor entrance -> narrow corridor -> destination area -> place box
```

---

## 14.2 场景变化

至少做这些 variation：

| Variation | 目的 |
|---|---|
| corridor width: wide / medium / narrow | 测试 clearance spec |
| box size: small / medium / long | 测试几何泛化 |
| initial box yaw: 0°, 10°, 20° | 测试 pre-align |
| destination left/right | 测试路径泛化 |
| one robot starts delayed | 测试同步恢复 |
| visual distractors / partial occlusion | 测试 judge 鲁棒性 |
| role swap: front/back 对调 | 测试 role prompt 和 spec 迁移 |

---

## 14.3 评价指标

不要只报 success rate。建议报：

```text
success_rate
collision_rate
drop_rate
wall_contact_count
max_box_yaw_error
max_box_tilt
mean_clearance
time_to_goal
number_of_replans
number_of_evolution_iterations_to_success
transfer_success_rate_on_unseen_scenes
human_intervention_count
```

尤其重要的是 learning efficiency：

\[
\text{learning efficiency} = \frac{\text{success after evolution} - \text{success before evolution}}{\text{number of failed trials}}
\]

以及 spec transfer gain：

\[
\text{spec transfer gain} = SR_{\text{with retrieved specs}} - SR_{\text{without retrieved specs}}
\]

这能证明“进化”和“记忆”真的有用。

---

## 14.4 Ablation

必须做 ablation，否则很难说明方法有效。

| 方法 | 说明 |
|---|---|
| Fixed Script | 手写脚本，不进化 |
| LLM Planner Only | 每次失败后只让 LLM 重新规划，不存 memory |
| COHERENT-style PEFA | proposal-execution-feedback-adjustment baseline |
| No Contract | 有 evolution，但没有 cooperative contract |
| No VLM Judge | 只用几何/proprioception judge |
| No Spec Memory | 有 critic/coder，但不跨任务复用 |
| Full Plan A++ | contract + judge + critic + coder + spec memory |

理想结果应该证明：

```text
Full Plan A++ > No Spec Memory > LLM Planner Only > Fixed Script
```

并且证明 spec 能迁移到新箱子 / 新走廊。

---

## 15. 最小可行版本 MVP

第一版不要做太大。

建议 MVP 如下。

---

## 15.1 MVP 任务

两个 G1 抬一个箱子，穿过一个直走廊，不需要转弯。

---

## 15.2 MVP skill set

只需要 10 个 skill：

```python
walk_to_pose()
turn_to_yaw()
approach_box_handle()
grasp_box_handle()
lift_payload()
wait_until_partner_ready()
align_payload_to_corridor()
carry_forward_with_partner()
stop_and_resync()
place_payload()
```

---

## 15.3 MVP judge

第一版先不用复杂 VLM，先用：

```text
AprilTag / marker-based box pose
robot odometry
contact sensor / force threshold
fixed camera optional
```

VLM 只负责语义解释：

```text
why failed?
which phase?
what patch?
```

这样成功概率最高。

等系统跑通后，再逐步替换成 RoboBrain 2.5 / VLM-based judge。

---

## 16. 最终方案一句话

最终方案可以概括为：

> 一个面向多机器人紧耦合搬运任务的 skill-level 自进化框架。系统先根据任务类别和历史规范生成 joint phase graph、per-robot skill scripts 与 cooperative contracts；执行时两个机器人各自基于局部观察调用 skill，并由 hybrid progress judge 判断 phase 进度和失败；失败后 Critic 定位原因，Coder 修改受约束脚本，Spec Memory 将被验证的协作规律沉淀为可复用规范。

主线不是：

```text
train MARL from scratch
```

也不是：

```text
直接复现完整 CHORUS VLA
```

而是：

```text
LLM / Agent evolves how robots call skills.
```

这最符合当前项目的资源、目标和成功概率。

---

## 17. 推荐落地路线

### Stage 1：固定技能 + 固定 phase graph

先让两个 G1 通过手写脚本完成 3–5 次成功搬运。

目的：验证 SONIC / WBC / Skill SDK 足够稳定。

---

### Stage 2：加入 structured judge

把每个 phase 的成功/失败判断做成结构化 predicate：

```python
abs(box_yaw - corridor_yaw) < 5
min_clearance > 0.12
both_grasp_contact == True
box_tilt < 8
front_back_distance_error < 0.06
```

---

### Stage 3：加入 Critic + Coder patch

让系统能从失败 trace 自动改脚本参数和 guard。

---

### Stage 4：加入 Spec Memory

把经过验证的 patch 提炼成可复用 specification。

测试新箱子、新走廊、新初始位置时是否能直接提升成功率。

---

### Stage 5：做 CHORUS-inspired learned skill selector

用前面收集的轨迹训练一个模型：

\[
\pi(u_t^i \mid z_t^i, c_i)
\]

它输出下一步 skill，而不是连续动作。

这就是 Plan A 和 Plan B 的结合点。

---

### Stage 6：可选做 openpi / π0.5 continuous baseline

等有足够数据后，再尝试：

\[
\pi(a_{t:t+H}^i \mid o_t^i, c_i)
\]

这时再做真正 CHORUS-like 的 VLA fine-tuning。

---

## 18. 最终推荐结论

最推荐路线：

```text
Plan A++：Contract-constrained Skill-Level Co-Evolution
```

核心理由：

1. 能复用现有 SONIC / WBC / Skill SDK。
2. 不需要一开始训练端到端 VLA。
3. 不需要手调复杂 MARL reward。
4. 真机落地概率最高。
5. 可以清晰体现“自进化”。
6. 可以做完整 ablation。
7. 可以自然吸收 CHORUS 的 decentralized local observation + identity prompt 思想。
8. 后续可平滑升级到 learned skill selector 或 CHORUS-like continuous VLA。

最终技术路径：

```text
固定 skill library
    ↓
phase graph + cooperative contract
    ↓
per-robot skill script
    ↓
hybrid judge + execution trace
    ↓
critic 定位失败
    ↓
coder 修改脚本和 guard
    ↓
验证 patch
    ↓
沉淀 reusable specification
    ↓
新任务通过 RAG 检索 spec，提高零样本成功率
```

这条路线最适合当前项目。

---

## 19. 参考链接

- CHORUS 项目页：https://chorus-model.github.io/
- CHORUS arXiv：https://arxiv.org/abs/2606.12352
- HALyPO arXiv：https://arxiv.org/abs/2603.03741
- openpi GitHub：https://github.com/Physical-Intelligence/openpi
- OpenVLA GitHub：https://github.com/openvla/openvla
- RDT GitHub：https://github.com/thu-ml/RoboticsDiffusionTransformer
- COHERENT GitHub：https://github.com/mrkeee/coherent
- SMART-LLM GitHub：https://github.com/SMARTlab-Purdue/SMART-LLM
- Voyager 项目页：https://voyager.minedojo.org/
- Isaac Lab：https://developer.nvidia.com/isaac/lab
