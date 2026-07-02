# Praxis-Coop 项目方案 Refine：面向 AAAI-27 的多机器人协同自进化框架

## 0. 项目定位

建议将项目从原先的 **“复现 HALO/HALYP 式协同搬运 policy”** 收敛为：

> **Praxis-Coop: Verified Self-Evolution of Cooperative Skill Graphs for Multi-Humanoid Object Transport**

核心思想不是重新训练一个端到端的多人形协同搬运策略，而是利用已有的 foundation whole-body control 作为底层运动能力，在其上研究：

1. LLM / agent 如何把自然语言协同任务拆解成可执行的协同 skill graph；
2. 如何根据场景、机器人能力和物体约束做角色分配与路径规划；
3. 如何在失败后把经验沉淀成可复用、可验证的 specification；
4. 如何通过 sandbox simulation 和 regression test 让系统安全地自进化。

可以将整篇论文的主线概括成：

> **Frozen foundation WBC + self-evolving cooperative skill orchestration.**

底层 whole-body control 使用 GEAR-SONIC / SONIC，项目贡献主要集中在上层模块：任务理解、路径规划、任务分配、协同 skill orchestration、failure critic、spec memory 和 verified evolution。

---

## 1. 为什么从 HALO/HALYP 切换到 GEAR-SONIC 是合理的

之前参考的人机协同搬运工作可以作为任务设定和 benchmark inspiration，但不适合作为主要 implementation base，原因包括：

- 输入没有视觉模态，和当前想做的视觉-语言-技能调用系统不完全匹配；
- 代码和 policy 没有开源，复现成本和不确定性较高；
- 该工作更偏向 decentralized POMDP / MARL 设定，不适合作为 LLM-agent self-evolution 的直接底座；
- 当前项目真正的创新点不在底层 policy，而在如何组织、调用、验证和进化上层协同技能。

GEAR-SONIC / SONIC 更适合作为底层 whole-body control：

- 它已经面向人形机器人 whole-body control；
- 提供训练、仿真、部署、kinematic planner 等相关工具；
- 更适合被上层 planner / VLA / LLM-agent 调用；
- 可以把它定位成 System-1 motor prior，上层系统只需要输出 motion command、short-horizon reference、formation target 或 skill call。

因此，项目应避免与 HALO 或 SONIC 在底层控制上正面竞争，而应突出：

> **如何把一个 foundation whole-body controller 组织成可泛化、可自进化的多机器人协同执行系统。**

---

## 2. 原方案中应该保留和修改的部分

### 2.1 建议保留

原方案中以下方向是对的：

1. **Orchestrator Agent** 负责高层任务拆解和角色分配；
2. **Coder Agent** 根据任务脚本调用底层 skill SDK；
3. **Judge / Critic Agent** 判断任务执行是否成功，分析失败原因；
4. **Spec Vector Memory** 存储可复用的协同经验；
5. 自进化重点放在 **LLM / agent 如何正确、高效地调用 skill**，而不是重新训练底层控制器。

### 2.2 必须 refine 的地方

| 原设计 | 建议 refine | 原因 |
|---|---|---|
| Orchestrator 直接生成两份 Python 脚本 | 改成生成 **Cooperative Skill Graph / Behavior Tree**，Python 只是执行后端 | 直接生成 Python 太脆弱，不利于验证和复用 |
| VLM as judge | 改成 **physics/state verifier 为主，VLM 为辅** | Isaac Sim 中可以直接拿到 object pose、contact、collision、clearance，物理判断更可复现 |
| 每个 agent 执行过程中优化代码 | 改成 **episode-level evolution** | 在线改代码风险高，容易破坏任务安全性和可复现性 |
| Spec memory 存文本/图像/traj | 改成 **Executable Spec Memory** | 需要能被检索、执行、验证和做 regression test |
| 第二台机器人作为 judge | 不建议常态化 | 第二台机器人应该是 embodied collaborator，而不是旁观 judge |

一句话总结：

> 不要让 LLM 进入高频控制环；不要让 VLM 承担物理 judge 的主责任；不要让 self-evolution 变成随意修改 Python。

---

## 3. 最终系统架构

建议最终系统分为以下模块：

```text
Human instruction + multi-view observations
        ↓
Scene Graph Builder / World Model
        ↓
Task Taxonomy Classifier  ←  Executable Spec Memory
        ↓
Object-Centric Global Planner
        ↓
Role Allocator + Cooperative Skill Graph Generator
        ↓
Skill API Adapter
        ↓
GEAR-SONIC / WBC / Kinematic Planner
        ↓
Isaac Sim / Isaac Lab execution
        ↓
State Verifier + VLM Failure Interpreter
        ↓
Critic → Coder → Sandbox Verification → Spec Memory Update
```

### 3.1 Scene Graph Builder / World Model

输入包括：

- Isaac Sim 中的 object pose；
- 墙体、障碍物、target 区域；
- 机器人初始位姿；
- 多视角相机图像；
- 可选的 depth、segmentation、occupancy map。

输出统一的 scene graph，例如：

```json
{
  "robots": ["G1_front", "G1_back"],
  "objects": [
    {
      "id": "box_01",
      "type": "box",
      "pose": [x, y, yaw],
      "size": [length, width, height]
    }
  ],
  "obstacles": ["wall_01", "wall_02", "gate_01"],
  "targets": ["target_zone"],
  "free_space": "occupancy_or_esdf_map"
}
```

### 3.2 Task Taxonomy Classifier

负责把自然语言任务和场景状态映射到预定义协同任务类型。

例如：

```text
Task: two humanoids move the box to the target through the narrow corridor.

Classification:
- physical_coupling: rigid-object coupling
- role_symmetry: leader-follower
- object_type: box
- environment_constraint: narrow passage
- objective: transport + insert
- observability: multi-view / partial egocentric
```

### 3.3 Object-Centric Global Planner

不要先给两个机器人各自规划路径，而应该先规划 **物体中心的 SE(2) 轨迹**：

```text
object_path = [(x0, y0, yaw0), (x1, y1, yaw1), ..., (xT, yT, yawT)]
```

然后再根据 object path 反推出：

- 两个机器人的 formation offset；
- 接触点或抓取点；
- front/back 或 left/right 角色；
- 每个阶段的同步条件；
- 窄通道处的 yaw alignment 和 clearance constraint。

### 3.4 Role Allocator

针对图中的双 G1 搬运箱子任务，可先使用：

```text
G1-front:
  - navigation leader
  - object yaw alignment leader
  - gate / target alignment leader

G1-back:
  - stabilizer
  - rear clearance monitor
  - contact maintenance
  - emergency stop trigger
```

在 long object 或窄通道任务中，角色可以动态切换：

```text
Phase 1: front robot leads global direction
Phase 2: rear robot leads pivot / rear clearance
Phase 3: front robot leads insertion
```

### 3.5 Cooperative Skill Graph Generator

Orchestrator 不直接输出完整 Python，而是输出 skill graph：

```text
approach_formation
  → establish_contact
  → pre_align_object
  → transport_to_gate
  → slow_pivot_through_gate
  → insert_to_target
  → release_and_exit
```

每个 skill node 都应该有：

- precondition；
- postcondition；
- monitor；
- timeout；
- fallback；
- required agents；
- parameter bindings。

示例：

```python
SkillNode(
    name="align_object_to_gate",
    agents={"leader": "G1-front", "stabilizer": "G1-back"},
    preconditions=[
        "both_agents_in_contact",
        "object_distance_to_gate < 1.0",
    ],
    goal=[
        "abs(object_yaw - gate_axis_yaw) < 7deg",
        "min_clearance(object, walls) > 0.12m",
    ],
    monitor=[
        "collision_pairs",
        "contact_loss",
        "object_tilt",
        "robot_fall",
        "progress_stagnation",
    ],
    fallback=[
        "stop_and_realign",
        "reverse_to_pre_align_pose",
        "switch_pivot_anchor",
    ],
)
```

### 3.6 Skill API Adapter

Skill graph 最终由 Skill API Adapter 转换为底层控制器可接受的命令。

上层只调用：

```python
nav_to_pose(agent, target_pose)
turn_to(agent, yaw)
approach_contact_point(agent, object_id, contact_point)
establish_contact(agent, object_id, mode="push")
maintain_contact(agent, object_id)
transport_object(object_id, object_path, formation)
pivot_object(object_id, anchor_agent, angle, radius)
synchronized_stop(agents)
```

底层由 GEAR-SONIC / WBC / kinematic planner 处理：

- gait；
- balance；
- whole-body motion tracking；
- joint-level action；
- recovery。

---

## 4. Step-by-step 任务执行流程

不建议让 LLM 自由决定 Step 1、Step 2、Step 3。建议采用 **event-driven phase machine**，每一步都有明确的进入条件和终止条件。

以当前 Isaac Sim 场景为例，可以拆成：

| Phase | 目标 | 终止条件 |
|---|---|---|
| Scene Parse | 识别箱子、墙体、通道、target、机器人位姿 | scene graph 建成，collision map 可用 |
| Task Classification | 判断任务属于刚体协同搬运、窄通道、目标对位 | taxonomy label 生成 |
| Role Binding | 分配 G1-front / G1-back 或 left / right | 每个 agent 都有 role、formation offset、skill list |
| Approach Formation | 两个 G1 到达箱子两侧或前后接触点 | base pose error < 阈值 |
| Contact / Grasp / Push Lock | 建立稳定接触或抓取 | contact sensor / relative pose 满足 |
| Pre-align Object | 箱子 yaw 与通道入口对齐 | yaw error < 阈值，clearance > margin |
| Cooperative Transport | 沿 object CoM path 推 / 搬 | object waypoint reached |
| Narrow Passage / Pivot | 低速通过门框或 U / S 型转角 | min clearance 始终大于阈值 |
| Insert / Deliver | 箱子进入 target 区域 | target pose error < 阈值 |
| Release / Exit | 松开箱子并撤离 | object stable，机器人离开 target 区域 |

这样的 step 切分来自 **precondition / effect / monitor**，而不是 LLM 的自然语言猜测。

---

## 5. Skill Library 设计

Skill library 应该定义成可验证 API，而不是只包含 `walk_forward()`、`turn_left()` 这类低级动作。

建议分三层。

### 5.1 Level 0：单机器人运动 skill

```python
nav_to_pose(agent, target_pose, style="walk", max_speed=0.3)
turn_to(agent, yaw)
set_body_height(agent, height_mode)
hold_pose(agent, duration)
recover_balance(agent)
```

作用：连接 GEAR-SONIC / WBC，负责单机器人导航、转向、姿态控制和平衡恢复。

### 5.2 Level 1：接触与物体交互 skill

```python
approach_contact_point(agent, object_id, contact_point)
establish_contact(agent, object_id, mode="push")
maintain_contact(agent, object_id, force_range=None)
release_contact(agent, object_id)
```

作用：建立和维持机器人与箱子 / 长物体之间的物理接触。

### 5.3 Level 2：多机器人协同 skill

```python
set_formation(agents, object_id, formation_type="front_back")
transport_object(object_id, object_path, formation, max_speed, yaw_rate_limit)
pivot_object(object_id, anchor_agent, angle, radius)
synchronized_stop(agents)
recover_lost_contact(agents, object_id)
```

作用：实现真正的多机器人协同。论文的主要 novelty 应该集中在 Level 2 的组合、选择、验证和进化上。

---

## 6. Task Taxonomy 设计

之前提到的“主客体、合作类别、博弈类别”方向是对的，但需要更工程化，能直接影响 planner 和 skill choice。

建议预定义以下维度：

| 维度 | 可选类别 | 对系统的影响 |
|---|---|---|
| Physical coupling | independent / loose coupling / rigid-object coupling / force-coupled | 决定是否采用 object-centric planning |
| Role symmetry | symmetric / leader-follower / scout-executor / stabilizer-mover | 决定 role allocator |
| Object type | box / long board / fragile object / heavy object / nonconvex object | 决定 grasp/contact points 和 clearance margin |
| Environment constraint | open space / narrow gate / U-turn / S-turn / clutter / target insertion | 决定 global planner 和 fallback |
| Observability | full-state / egocentric / asymmetric vision / occlusion | 决定是否需要 scout / visual helper |
| Communication | full / low-bandwidth / delayed / no communication | 决定同步方式 |
| Objective | transport / align / insert / handoff / rescue / inspection | 决定 terminal predicate |
| Interaction type | pure cooperation / mixed-motive / adversarial obstacle | 决定 judge 和 reward 设计 |

AAAI-27 版本不建议覆盖所有类别。建议主线聚焦：

```text
rigid-object coupling + narrow passage + partial observability
```

这是最能体现多机器人协同、路径规划、视觉判断和自进化的组合。

---

## 7. 自进化模块设计

建议把“自进化”拆成四个层级。AAAI-27 主线建议做前三个，第四个作为 future work。

### 7.1 Level A：参数进化

失败后自动调整：

- speed limit；
- yaw threshold；
- clearance margin；
- formation offset；
- pivot radius；
- timeout；
- fallback trigger threshold。

例子：

```text
Failure:
  箱子右前角撞到门框。

Patch:
  - 进入门框前 yaw error threshold 从 12° 收紧到 6°
  - 通过阶段 max speed 从 0.25 m/s 降到 0.12 m/s
  - min clearance margin 从 0.08 m 提高到 0.12 m
```

### 7.2 Level B：Skill graph 结构进化

失败后添加、删除或重排节点。

例子：

```text
Original graph:
  approach → transport → insert

Evolved graph:
  approach → pre_align → slow_pivot → transport_through_gate → insert
```

### 7.3 Level C：Role template 进化

失败后改变角色分配或增加角色责任。

例子：

```text
Original:
  G1-front: navigation leader
  G1-back: passive follower

Evolved:
  G1-front: navigation and target alignment leader
  G1-back: rear-clearance monitor + stabilizer + emergency stop trigger
```

### 7.4 Level D：底层 policy fine-tuning

AAAI-27 主线不建议做。

原因：

- 时间风险高；
- 和 SONIC 底层贡献混淆；
- 评估复杂；
- 不利于突出 LLM-agent / planning / memory 的贡献。

可以在论文里作为 future work 提到：未来可以把 high-level spec evolution 进一步传递到底层 WBC fine-tuning。

---

## 8. Executable Spec Memory

Spec Memory 不应该只存一句“以后要小心门框”。建议每条 spec 都是可检索、可执行、可验证的结构。

示例：

```json
{
  "spec_id": "narrow_gate_pre_alignment_v3",
  "task_context": {
    "physical_coupling": "rigid_object",
    "object_shape": "box_or_long_box",
    "environment": "narrow_gate_or_corridor",
    "formation": "front_back"
  },
  "rule_text": "Before entering a narrow passage, align the object's yaw with the gate axis and slow down until sufficient clearance is verified.",
  "executable_predicate": [
    "abs(object_yaw - gate_axis_yaw) < 7deg",
    "min_clearance(object, gate_edges) > 0.12m",
    "object_speed < 0.15m/s"
  ],
  "trigger": [
    "distance(object_com, gate_entry) < 1.0m"
  ],
  "fallback": [
    "stop_and_realign",
    "reverse_to_pre_gate_pose",
    "switch_pivot_anchor"
  ],
  "evidence": {
    "failure_cases": ["seed_014", "seed_037"],
    "success_after_patch": ["seed_014_patch2", "seed_037_patch1"]
  },
  "confidence": 0.82
}
```

### 8.1 每条 spec 至少包含

1. **适用上下文**：任务类型、物体类型、环境约束、formation；
2. **自然语言规则**：方便 LLM / 人类理解；
3. **可执行 predicate**：系统实际检查的条件；
4. **trigger**：什么时候应用这条规则；
5. **fallback**：违反规则时如何恢复；
6. **evidence**：这条规则来自哪些失败案例，patch 后在哪些场景成功；
7. **confidence**：随着复用成功率更新。

### 8.2 Spec Memory 的检索方式

检索 query 不应该只用语言 embedding，还应该包含结构化标签：

```text
query = {
  task_type: "rigid_object_transport",
  object_type: "box",
  environment: "narrow_gate",
  formation: "front_back",
  failure_mode: optional
}
```

返回结果用于约束 Orchestrator：

```text
Retrieved specs:
- narrow_gate_pre_alignment_v3
- rear_clearance_monitor_v2
- slow_pivot_before_insert_v1
```

这些 spec 作为 system-level constraints 注入 skill graph generation，而不是简单放进 prompt 里让 LLM 自由发挥。

---

## 9. Judge / Verifier 模块

建议采用分层 judge，而不是单独依赖 VLM。

### 9.1 优先级 1：Skill return / exception

判断：

- skill 是否完成；
- 是否 timeout；
- 是否抛出底层控制异常；
- 是否出现 robot fall / unrecoverable state。

### 9.2 优先级 2：Symbolic state verifier

在 Isaac Sim 中可以直接检查：

- object pose；
- robot pose；
- contact state；
- collision pairs；
- object tilt；
- target error；
- robot fall state；
- progress stagnation。

### 9.3 优先级 3：Geometric verifier

用于检查：

- object footprint 和墙体 / 障碍物的 clearance；
- planned path 是否可行；
- narrow gate 处是否满足 yaw constraint；
- long object 是否有 pivot 空间。

### 9.4 优先级 4：VLM failure interpreter

VLM 适合做：

- 多视角图像 failure description；
- 解释失败原因；
- 辅助生成 critic message；
- 判断语义状态，如“箱子卡在货柜边缘”“后方机器人挡住通道”。

但不建议让 VLM 成为唯一 success/failure judge，因为可复现性和物理精度容易被 reviewer 质疑。

---

## 10. 第二台机器人应该如何使用

原方案中提到第二台机器人作为 judge 和辅助视觉信息。建议拆开：

### 10.1 不建议常态化作为 judge

原因：

- 这样会让任务退化成“一个机器人干活，一个机器人看”；
- 不利于突出 embodied multi-robot cooperation；
- judge 更适合由外部 verifier / simulation state / camera system 完成。

### 10.2 推荐作为 embodied collaborator

在主任务中，第二台 G1 应该承担：

```text
- stabilizer
- rear-clearance monitor
- follower / co-carrier
- pivot anchor
- recovery helper
```

### 10.3 可在特定任务中作为 visual scout

在 partial observability 任务中，可以让第二台 G1 扮演 scout-helper：

```text
G1-main:
  - 执行搬运 / 推箱子
  - 局部视野受限

G1-scout:
  - 移动到侧面或后方
  - 提供多视角图像
  - 报告 gate clearance、rear collision risk、target alignment
```

这样第二台机器人仍然是任务参与者，而不是被动 judge。

---

## 11. 推荐实验任务

### 11.1 Task 1：Two-G1 Cooperative Box Transport

对应当前 Isaac Sim 场景。

任务：两个 G1 协同把箱子从起点推 / 搬到 target，途中经过墙体、通道或 U 型结构。

核心考察：

- object-centric path planning；
- front-back role allocation；
- contact maintenance；
- synchronized execution；
- collision recovery；
- spec memory reuse。

### 11.2 Task 2：Long Object Through Narrow Passage

借鉴长物体协同搬运任务。

任务：两个 G1 搬 / 推长箱子或长板，通过窄门、U-turn 或 S-turn。

核心考察：

- yaw-aware planning；
- pivot behavior；
- dynamic role switching；
- rear clearance monitoring；
- slow passage specification。

### 11.3 Task 3：Asymmetric Observation / Scout-Assisted Transport

任务：主执行机器人视野被遮挡，另一台机器人提供辅助视觉或侧后方观察。

核心考察：

- multi-view observation；
- communication；
- scout-executor role specialization；
- VLM failure interpretation；
- partial observability generalization。

---

## 12. Baselines 和 Ablations

AAAI 投稿需要证明系统不是简单 LLM 套壳。建议至少包括以下 baseline。

| Baseline | 目的 |
|---|---|
| Hand-coded fixed skill graph | 证明 evolution 有收益 |
| LLM zero-shot skill script | 证明不是 prompt 一次就能成功 |
| LLM + Reflexion text memory | 证明 executable spec memory 优于纯文本反思 |
| Object-centric planner without evolution | 证明 self-evolution 的增益 |
| Independent robot path planning | 证明 object-centric cooperative planning 的必要性 |
| No VLM critic | 证明视觉 failure explanation 是否有用 |
| No verifier / unsafe code evolution | 证明 verified evolution 的必要性 |
| No role allocation | 证明角色分配的必要性 |
| No spec retrieval | 证明长期经验检索的必要性 |

---

## 13. Evaluation Metrics

建议分成任务成功、物理安全、协同效率、自进化效果和泛化能力五组指标。

### 13.1 任务成功指标

- success rate；
- target pose error；
- object final yaw error；
- completion time；
- path efficiency。

### 13.2 物理安全指标

- collision count；
- min clearance；
- robot fall rate；
- object tilt；
- contact loss count；
- unrecoverable failure rate。

### 13.3 协同质量指标

- formation error；
- inter-agent synchronization error；
- object yaw tracking error；
- force/contact stability；
- recovery count。

### 13.4 自进化指标

- episodes-to-success；
- improvement after N failed episodes；
- spec reuse rate；
- spec precision / activation usefulness；
- regression failure rate；
- number of accepted vs rejected patches。

### 13.5 泛化指标

- held-out layout success rate；
- held-out object shape success rate；
- narrow gate width generalization；
- new target position generalization；
- partial observation generalization。

---

## 14. Verified Self-Evolution Loop

建议将自进化设计成严格的 verified loop。

```text
1. Execute current skill graph in simulation
2. Verifier detects failure or suboptimal behavior
3. Critic summarizes failure mode
4. Coder proposes patch
   - parameter patch
   - skill graph patch
   - role template patch
   - spec memory patch
5. Sandbox test on failed seed
6. Regression test on previous success seeds
7. Randomized validation on new seeds
8. Accept patch only if success improves and regressions are bounded
9. Store updated executable spec with evidence
```

### 14.1 Patch 类型

```json
{
  "patch_type": "skill_graph_patch",
  "target": "transport_through_gate",
  "change": {
    "insert_before": "transport_through_gate",
    "new_node": "pre_align_object_to_gate"
  },
  "reason": "The object collided with the right gate edge because yaw was not aligned before entering the narrow passage.",
  "expected_effect": "Reduce gate collision by enforcing yaw alignment and lower speed."
}
```

### 14.2 Patch 接受条件

可以设置为：

```text
accept patch if:
  success_rate_new >= success_rate_old + delta
  and collision_count_new <= collision_count_old
  and regression_failures <= threshold
  and no new safety violation appears
```

这样 self-evolution 才是可量化、可复现、可发表的。

---

## 15. 面向当前 Isaac Sim 场景的具体建议

根据当前截图，场景中已有：

- 两个 G1；
- U 型 / 墙体结构；
- 一个圆柱或障碍物；
- target；
- GroundPlane；
- 多个 wall mesh。

建议下一步把场景整理成一个标准 benchmark task。

### 15.1 场景对象命名规范

建议在 USD / Isaac Sim 中统一命名：

```text
/World/Robots/G1_front
/World/Robots/G1_back
/World/Objects/box_01
/World/Objects/long_box_01
/World/Obstacles/wall_left
/World/Obstacles/wall_right
/World/Obstacles/gate_left
/World/Obstacles/gate_right
/World/Targets/target_zone
```

### 15.2 必须暴露给 verifier 的状态

```python
state = {
    "robot_poses": {...},
    "object_pose": {...},
    "object_velocity": {...},
    "target_pose": {...},
    "contact_states": {...},
    "collision_pairs": [...],
    "clearance_to_walls": float,
    "object_tilt": float,
    "robot_fall_states": {...},
}
```

### 15.3 第一版任务目标

建议先做：

```text
Task:
  Two G1 humanoids cooperatively move a box from the start region to the target region through a narrow passage without collision.

Success:
  - object center inside target zone
  - object yaw error < 10°
  - no robot fall
  - collision count below threshold
  - contact loss count below threshold
```

### 15.4 第一版 skill graph

```text
1. parse_scene
2. classify_task
3. retrieve_specs
4. plan_object_path
5. assign_roles
6. move_to_formation
7. establish_contact
8. pre_align_object
9. transport_to_gate
10. slow_pivot_or_passage
11. insert_to_target
12. release_and_exit
```

---

## 16. 论文贡献写法

可以把论文贡献写成四点。

### Contribution 1：Cooperative Skill Orchestration

> We introduce a cooperative skill orchestration framework that composes foundation whole-body controllers into multi-humanoid cooperative object transport behaviors through object-centric planning and role-aware skill graphs.

### Contribution 2：Executable Specification Memory

> We propose an executable specification memory that distills physical cooperation failures into reusable and verifiable rules with predicates, triggers, fallback policies, evidence, and confidence scores.

### Contribution 3：Verified Self-Evolution

> We develop a verified self-evolution loop where LLM agents propose parameter, graph, role, and specification updates, while sandbox simulation and regression tests determine whether updates are promoted.

### Contribution 4：Benchmark

> We build a multi-humanoid cooperative transport benchmark in Isaac Sim / Isaac Lab with randomized object shapes, narrow passages, target insertion, partial observability, and multi-view failure traces.

---

## 17. 最大风险和规避方案

### 风险 1：GEAR-SONIC 与当前 Isaac Sim stage 集成成本高

规避：

- 先把当前 USD 场景整理成 Isaac Lab task；
- 用 Isaac Sim 负责世界、感知、碰撞和评估；
- 用 GEAR-SONIC / SONIC 作为底层 motion command tracker；
- 如果 SONIC 接入不顺，先用 kinematic / simple locomotion controller 跑通上层闭环，再替换底层 WBC。

### 风险 2：系统太工程化，AAAI novelty 不够

规避：

- 明确提出 Cooperative Skill Graph；
- 明确提出 Executable Spec Memory；
- 明确提出 Verified Self-Evolution；
- 做充分 ablation，证明每个模块都有贡献。

### 风险 3：VLM judge 被质疑不稳定

规避：

- 物理 state verifier 作为主 judge；
- VLM 只做 failure explanation 和 semantic grounding；
- 所有成功率和碰撞指标都来自 simulator state。

### 风险 4：自进化被认为只是 prompt engineering

规避：

- 每条 spec 都必须有 executable predicate；
- patch 必须经过 sandbox test；
- 需要 regression test；
- 展示 spec 跨场景复用效果；
- 展示 accepted / rejected patches 的统计。

### 风险 5：任务过大，来不及收敛

规避：

AAAI-27 最小可投稿版本只做：

```text
2 个 G1 + box / long box
3 类场景：open transport、narrow gate、U/S turn
3 类 evolution：parameter、skill graph、role template
1 个底层 WBC adapter
完整 ablation
```

不要同时追求：

- 真机部署；
- 底层 policy fine-tuning；
- 大规模多机器人；
- 全开放自然语言任务；
- 多种机器人形态。

---

## 18. 推荐最终 Pipeline

```text
Input:
  "Two humanoids move the box to the target through the narrow corridor."

1. SceneGraphBuilder:
   walls, box, target, robot poses, free space, gate axes, object footprint

2. TaskClassifier:
   rigid-object cooperative transport + narrow passage + front-back formation

3. SpecRetriever:
   retrieve specs for narrow passage, yaw alignment, rear clearance, slow pivot

4. ObjectPlanner:
   plan object CoM + yaw path, not two independent robot paths

5. RoleAllocator:
   G1-front = navigation/alignment leader
   G1-back  = stabilizer/rear-clearance monitor

6. SkillGraphGenerator:
   approach → contact → pre-align → slow transport → pivot → insert → release

7. Execution:
   each skill sends motion references to SONIC adapter

8. Verifier:
   checks predicates, collisions, contact loss, target pose

9. Critic:
   labels failure mode if any

10. Evolution:
   patch graph/spec/parameters → sandbox tests → memory update
```

---

## 19. 推荐论文标题

可以考虑以下标题：

1. **Praxis-Coop: Verified Self-Evolution of Cooperative Skill Graphs for Multi-Humanoid Object Transport**
2. **Self-Evolving Cooperative Skill Orchestration for Multi-Humanoid Object Transport**
3. **Executable Specification Memory for Verified Multi-Humanoid Cooperation**
4. **From Foundation Whole-Body Control to Self-Evolving Multi-Humanoid Cooperation**
5. **Role-Aware Skill Graph Evolution for Cooperative Humanoid Object Transport**

我最推荐第一个：

> **Praxis-Coop: Verified Self-Evolution of Cooperative Skill Graphs for Multi-Humanoid Object Transport**

这个标题能够突出：

- 不是底层控制论文；
- 不是普通 LLM planner；
- 有 verified self-evolution；
- 有 cooperative skill graph；
- 有 multi-humanoid object transport 应用场景。

---

## 20. AAAI-27 收敛建议

面向 AAAI-27，建议不要再铺“多任务、多机器人、多模态、真机、policy training”全套大目标，而是收敛成一个清晰主线：

```text
Frozen foundation WBC
+ object-centric cooperative planning
+ role-aware cooperative skill graph
+ executable spec memory
+ verified self-evolution
+ multi-humanoid transport benchmark
```

最小可投稿版本：

1. 一个稳定的双 G1 + box / long object benchmark；
2. 三类场景：open transport、narrow gate、U/S turn；
3. 一个 object-centric planner；
4. 一个 Cooperative Skill Graph 中间表示；
5. 一个 Executable Spec Memory；
6. 一个 Verified Evolution loop；
7. 完整 ablation：no memory、text-only memory、no verifier、no object-centric planner、no role allocation；
8. 至少一个高质量视频 demo；
9. 如果能做真实 G1 或 sim2sim demo，会显著加分，但不应作为主线依赖。

---

## 21. 参考链接

- HALO/HALYP 参考工作：<https://arxiv.org/pdf/2603.03741>
- GEAR-SONIC paper：<https://nvlabs.github.io/GEAR-SONIC/static/pdf/sonic_paper.pdf>
- GEAR-SONIC / GR00T Whole-Body Control GitHub：<https://github.com/NVlabs/GR00T-WholeBodyControl>
- AAAI-27 Main Technical Track Call：<https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/>
