from pathlib import Path
import json

import pytest

from praxis_coop.adapters import MockSonicAdapter
from praxis_coop.examples import two_g1_box_scene
from praxis_coop.coder_agent import apply_patch
from praxis_coop.isaac_scene_adapter import figure2_passage_scene_graph, manual_two_g1_box_push_scene_graph
from praxis_coop.knowledge_base import TextSpecMemory
from praxis_coop.models import FailureMode, PatchProposal, SkillGraph, SkillNode, SkillStatus
from praxis_coop.task_orchestrator import build_initial_skill_graph
from praxis_coop.qwen_evolution import _patch_from_payload, _spec_from_payload
from praxis_coop.runtime.manual_usd_runtime import DEFAULT_MANUAL_SCENE, ManualUsdSceneRuntime, run_manual_usd_skillgraph
from praxis_coop.sonic import CooperativeCFGen, SonicLocomotionMode, map_skill_graph_to_sonic_stream
from praxis_coop.sonic_mujoco.manual_layout_scene import (
    DEFAULT_G1_SOURCE_XML,
    DEFAULT_USDA_SCENE,
    generate_dual_manual_layout_scene,
)
from praxis_coop.sonic.joint_policy import (
    ACTION_SCALE_MUJOCO_ORDER,
    DEFAULT_ANGLES_MUJOCO_ORDER,
    G1_JOINT_NAMES_MUJOCO_ORDER,
    G1_NUM_MOTOR,
    ISAACLAB_TO_MUJOCO,
    SonicIsaacJointBackend,
    load_onnx_model_info,
    sonic_action_to_motor_command,
    zero_policy_action,
)
from praxis_coop.sonic.zmq_publisher import build_command_message, build_planner_message
from praxis_coop.sonic.upper_body import (
    HORIZONTAL_CARRY_UPPER_BODY,
    OPEN_LEFT_HAND_JOINTS,
    OPEN_RIGHT_HAND_JOINTS,
    UPPER_BODY_JOINT_NAMES,
    resolve_hand_pose,
    resolve_upper_body_pose,
)
from praxis_coop.stage1_evolution import (
    Stage1Parameters,
    Stage1SonicParameters,
    apply_stage1_patch,
    apply_stage1_sonic_patch,
    stage1_sonic_patch_schema,
)
from praxis_coop.stage1_visualization import write_stage1_evolution_html
from praxis_coop.skills import default_g1_skill_registry
from praxis_coop.skill_graph import execute_skill_graph
from praxis_coop.utils.geometry import relative_body_offset
from praxis_coop.verifier import verify_execution


def test_build_initial_skill_graph_retrieves_contract_specs():
    scene = two_g1_box_scene()
    result = build_initial_skill_graph(
        "Two G1 robots transport the box through the narrow gate to the target.",
        scene,
    )

    assert result.graph.taxonomy.physical_coupling == "rigid_object"
    assert result.graph.roles.formation == "front_back"
    assert result.graph.nodes[0].name == "approach_formation"
    assert any(node.name == "lock_initial_hand_pose" for node in result.graph.nodes)
    assert "narrow_gate_pre_alignment_v1" in result.retrieved_specs
    transport = next(node for node in result.graph.nodes if node.name == "transport_object_path")
    assert transport.phase_id == "transport_object_path"
    assert transport.sync_group == "payload_team"
    assert transport.contact_flow is not None
    assert transport.contact_flow.contact is not None
    assert transport.contact_flow.contact.maintain


def test_mock_execution_and_verifier_success():
    scene = two_g1_box_scene(object_at_target=True)
    result = build_initial_skill_graph("transport through gate", scene)
    skill_results = execute_skill_graph(result.graph, scene, MockSonicAdapter())
    report = verify_execution(scene, skill_results)

    assert all(item.status == SkillStatus.SUCCESS for item in skill_results)
    assert report.success
    assert report.metrics["estimated_gate_clearance_m"] > 0.12
    assert report.evidence["phase_trace"]


def test_qwen_payload_parsers_create_patch_and_spec():
    patch = _patch_from_payload(
        {
            "patch_id": "qwen_resync_contact",
            "patch_type": "graph",
            "target": "transport_object_path.preconditions",
            "change": {"insert_before": "transport_object_path", "node": "resync_contact"},
            "rationale": "Contact loss requires synchronized recontact before transport.",
        }
    )
    spec = _spec_from_payload(
        {
            "spec_id": "learned_contact_resync",
            "task_context": {"physical_coupling": "rigid_object", "object_type": "long_box"},
            "rule_text": "Reconfirm dual contact before moving a shared rigid payload.",
            "trigger": ["contact_count < 2"],
            "executable_predicate": ["both_agents_in_contact"],
            "fallback": ["resync_contact"],
            "confidence": 0.7,
            "evidence": {"failure_cases": ["contact_loss"], "success_after_patch": []},
        }
    )

    assert patch is not None
    assert patch.patch_type == "graph"
    assert spec is not None
    assert spec.spec_id == "learned_contact_resync"


def test_apply_patch_inserts_contact_resync_node():
    scene = two_g1_box_scene(object_at_target=True)
    scene.robots["G1-back"].in_contact = False
    result = build_initial_skill_graph("transport through gate", scene)
    skill_results = execute_skill_graph(result.graph, scene, MockSonicAdapter())
    report = verify_execution(scene, skill_results)
    patch = PatchProposal(
        patch_id="qwen_resync_contact",
        patch_type="graph",
        target="transport_object_path.preconditions",
        change={"insert_before": "transport_object_path", "node": "resync_contact"},
        rationale="Qwen critic inserted contact resync.",
    )
    patched = apply_patch(result.graph, patch)

    assert not report.success
    assert report.failure_mode == FailureMode.CONTACT_LOSS
    assert report.evidence["failed_phase_id"]
    assert any(node.name == "resync_contact" for node in patched.nodes)


def test_manual_scene_plans_object_and_each_robot_path():
    scene = manual_two_g1_box_push_scene_graph()
    result = build_initial_skill_graph(
        "Two G1 robots move the long box to the target through the corridor.",
        scene,
    )

    assert result.object_path_len == 4
    assert set(result.robot_path_lens) == {"g1", "g1_01"}
    assert result.robot_path_lens["g1"] == result.object_path_len
    assert result.graph.roles.roles["g1_01"] == "navigation_leader"
    assert result.graph.roles.formation == "authored_payload_contact"

    transport = next(node for node in result.graph.nodes if node.name == "transport_object_path")
    assert transport.parameters["object_path_waypoints"] == 4
    assert transport.parameters["robot_path_waypoints"]["g1"] == 4


def test_skill_registry_and_text_spec_memory(tmp_path):
    registry = default_g1_skill_registry()
    carry = registry.get("carry_payload_path")
    hand_lock = registry.get("hold_initial_payload_hand_pose")
    lateral = registry.get("move_lateral")
    longitudinal = registry.get("move_longitudinal")
    yaw_turn = registry.get("turn_body_yaw_360")
    assert "stop_if_contact_loss" in carry.guards
    assert hand_lock.skill_type == "sonic_upper_body_pose_lock"
    assert lateral.skill_type == "sonic_ground_plane_locomotion"
    assert longitudinal.command_schema["direction"] == "'forward'|'backward'|signed_float"
    assert "[-2*pi, 2*pi]" in yaw_turn.command_schema["delta_yaw_rad"]

    memory_path = tmp_path / "specs.jsonl"
    memory = TextSpecMemory(memory_path)
    memory.save()
    reloaded = TextSpecMemory(memory_path)

    assert reloaded.specs
    assert all(spec.rule_text for spec in reloaded.specs)


def test_manual_usd_runtime_reads_scene_graph():
    runtime = ManualUsdSceneRuntime(DEFAULT_MANUAL_SCENE)
    scene = runtime.read_scene_graph()

    assert set(scene.robots) == {"g1", "g1_01"}
    assert scene.movable_object.object_id == "long_box"
    assert scene.target.target_id == "target"
    assert "wall_01" in scene.obstacles


def test_dual_g1_sonic_mujoco_scene_generator_namespaces_robot_xml(tmp_path):
    source_g1 = Path("third_party/GR00T-WholeBodyControl") / DEFAULT_G1_SOURCE_XML
    gear_root = tmp_path / "gear_root"
    copied_source = gear_root / DEFAULT_G1_SOURCE_XML
    copied_source.parent.mkdir(parents=True)
    copied_source.write_text(source_g1.read_text(encoding="utf-8"), encoding="utf-8")

    result = generate_dual_manual_layout_scene(source_usda=DEFAULT_USDA_SCENE, gear_root=gear_root)

    assert result.scene_path.exists()
    assert {robot.prefix for robot in result.robot_xmls} == {"front_", "back_"}
    front_xml = result.robot_xmls[0].output_path.read_text(encoding="utf-8")
    back_xml = result.robot_xmls[1].output_path.read_text(encoding="utf-8")
    scene_xml = result.scene_path.read_text(encoding="utf-8")
    assert "front_pelvis" in front_xml
    assert "front_left_hip_pitch_joint" in front_xml
    assert "back_pelvis" in back_xml
    assert "back_left_hip_pitch_joint" in back_xml
    assert "praxis_g1_front_29dof_with_hand.xml" in scene_xml
    assert "praxis_g1_back_29dof_with_hand.xml" in scene_xml


def test_manual_usd_runtime_executes_skillgraph_to_output_scene(tmp_path):
    initial_scene = ManualUsdSceneRuntime(DEFAULT_MANUAL_SCENE).read_scene_graph()
    initial_offsets = {
        robot_id: relative_body_offset(initial_scene.movable_object.pose, robot.pose)
        for robot_id, robot in initial_scene.robots.items()
    }
    output_scene = tmp_path / "manual_rollout.usda"
    result, report = run_manual_usd_skillgraph(
        "Two G1 robots move the long box to the target through the corridor.",
        scene_path=DEFAULT_MANUAL_SCENE,
        output_scene=output_scene,
        transport_speed_mps=0.10,
    )

    assert output_scene.exists()
    assert report.success
    assert result.scene.movable_object.pose.distance_to(result.scene.target.pose) == 0.0
    assert all(robot.in_contact for robot in result.scene.robots.values())
    assert result.scene.movable_object.pose.yaw == pytest.approx(initial_scene.movable_object.pose.yaw)

    for robot_id, robot in result.scene.robots.items():
        final_offset = relative_body_offset(result.scene.movable_object.pose, robot.pose)
        assert final_offset.x == pytest.approx(initial_offsets[robot_id].x)
        assert final_offset.y == pytest.approx(initial_offsets[robot_id].y)
        assert final_offset.yaw == pytest.approx(initial_offsets[robot_id].yaw)
        assert abs(final_offset.y) > 0.15


def test_sonic_mapper_generates_object_carrying_commands():
    scene = manual_two_g1_box_push_scene_graph()
    result = build_initial_skill_graph(
        "Two G1 robots move the long box to the target through the corridor.",
        scene,
    )
    stream = map_skill_graph_to_sonic_stream(result.graph, scene)
    by_robot = stream.by_robot()

    assert set(by_robot) == {"g1", "g1_01"}
    assert any(command.mode == SonicLocomotionMode.OBJECT_CARRYING for command in stream.commands)
    assert any(command.skill_node == "lock_initial_hand_pose" for command in stream.commands)
    assert any(
        command.skill_node == "transport_object_path"
        and command.hand_pose_policy == "hold_initial_payload_hand_pose"
        and command.contact_intent == "carry_or_push_path"
        and command.sync_group == "payload_team"
        and command.abort_on_contact_loss
        for command in stream.commands
    )
    assert any(command.phase_id.startswith("transport_object_path.segment_") for command in stream.commands)
    assert all(command.duration_s > 0.0 for command in stream.commands)
    assert stream.duration_s > 0.0


def test_cooperative_cfgen_annotates_legacy_nodes():
    scene = manual_two_g1_box_push_scene_graph()
    initial = build_initial_skill_graph(
        "Two G1 robots move the long box to the target through the corridor.",
        scene,
    ).graph
    legacy = SkillGraph(
        graph_id="legacy_graph",
        taxonomy=initial.taxonomy,
        roles=initial.roles,
        nodes=[
            SkillNode(
                name="establish_contact",
                agents={"leader": "g1_01", "stabilizer": "g1"},
                parameters={"contact_mode": "push"},
                monitors=["contact_loss"],
            ),
            SkillNode(
                name="transport_object_path",
                agents={"leader": "g1_01", "stabilizer": "g1"},
                parameters={"object_path": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}]},
                monitors=["contact_loss", "progress_stagnation"],
            ),
        ],
    )

    flow = CooperativeCFGen().annotate(legacy, scene)
    contact = flow.graph.nodes[0].contact_flow
    transport = flow.graph.nodes[1].contact_flow

    assert len(flow.phases) == 2
    assert contact is not None
    assert contact.contact is not None
    assert contact.contact.mode == "push"
    assert contact.abort_on_contact_loss
    assert transport is not None
    assert transport.object_goal == scene.target.pose


def test_sonic_mapper_generates_lateral_longitudinal_and_360_turn_commands():
    scene = manual_two_g1_box_push_scene_graph()
    initial = build_initial_skill_graph(
        "Two G1 robots move the long box to the target through the corridor.",
        scene,
    ).graph
    graph = SkillGraph(
        graph_id="primitive_motion_smoke",
        taxonomy=initial.taxonomy,
        roles=initial.roles,
        nodes=[
            SkillNode(
                name="sidestep_left",
                agents={"leader": "g1_01", "stabilizer": "g1"},
                parameters={
                    "registry_skill": "move_lateral",
                    "direction": "left",
                    "distance_m": 0.5,
                    "max_speed_mps": 0.12,
                    "reference_frame": "body",
                },
            ),
            SkillNode(
                name="move_forward",
                agents={"leader": "g1_01", "stabilizer": "g1"},
                parameters={
                    "registry_skill": "move_longitudinal",
                    "direction": "forward",
                    "distance_m": 0.75,
                    "max_speed_mps": 0.15,
                    "reference_frame": "body",
                },
            ),
            SkillNode(
                name="turn_full_circle",
                agents={"leader": "g1_01", "stabilizer": "g1"},
                parameters={
                    "registry_skill": "turn_body_yaw_360",
                    "delta_yaw_deg": 360.0,
                    "max_yaw_rate_radps": 0.4,
                },
            ),
        ],
    )

    stream = map_skill_graph_to_sonic_stream(graph, scene)
    lateral = [command for command in stream.commands if command.skill_node == "sidestep_left"]
    longitudinal = [command for command in stream.commands if command.skill_node == "move_forward"]
    turn = [command for command in stream.commands if command.skill_node == "turn_full_circle"]

    assert len(lateral) == 2
    assert all(command.motion_axis == "lateral" for command in lateral)
    assert all(command.reference_frame == "body" for command in lateral)
    assert all(command.mode == SonicLocomotionMode.WALK for command in lateral)
    assert len(longitudinal) == 2
    assert all(command.motion_axis == "longitudinal" for command in longitudinal)
    assert len(turn) >= 16
    assert all(command.motion_axis == "yaw" for command in turn)
    assert all(command.movement == (0.0, 0.0, 0.0) for command in turn)
    assert max(abs(command.delta_yaw_rad or 0.0) for command in turn) == pytest.approx(2.0 * 3.141592653589793)


def test_sonic_zmq_message_builders_match_topic_prefixes():
    command_msg = build_command_message(start=True, stop=False, planner=True)
    planner_msg = build_planner_message(
        mode=int(SonicLocomotionMode.OBJECT_CARRYING),
        movement=(1.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
        speed=0.1,
        left_hand_joints=(0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75),
        right_hand_joints=(0.0, 0.0, -1.75, 1.57, 1.75, 1.57, 1.75),
        upper_body_position=(0.0,) * 17,
    )

    assert command_msg.startswith(b"command")
    assert planner_msg.startswith(b"planner")
    assert len(command_msg) > 1280
    assert len(planner_msg) > 1280

    command_header = json.loads(command_msg[len(b"command") : len(b"command") + 1280].rstrip(b"\x00"))
    planner_header = json.loads(planner_msg[len(b"planner") : len(b"planner") + 1280].rstrip(b"\x00"))
    assert [field["name"] for field in command_header["fields"]] == ["start", "stop", "planner"]
    assert [field["name"] for field in planner_header["fields"]] == [
        "mode",
        "movement",
        "facing",
        "speed",
        "height",
        "upper_body_position",
        "left_hand_joints",
        "right_hand_joints",
    ]
    assert planner_header["fields"][1]["shape"] == [3]
    assert planner_header["fields"][5]["shape"] == [17]
    assert planner_header["fields"][6]["shape"] == [7]


def test_sonic_horizontal_carry_pose_matches_deploy_message_contract():
    upper_body = resolve_upper_body_pose("horizontal_carry")
    left_hand, right_hand = resolve_hand_pose("open")
    planner_msg = build_planner_message(
        mode=int(SonicLocomotionMode.OBJECT_CARRYING),
        movement=(0.0, 1.0, 0.0),
        facing=(1.0, 0.0, 0.0),
        speed=0.08,
        upper_body_position=upper_body,
        upper_body_velocity=(0.0,) * 17,
        left_hand_joints=left_hand,
        right_hand_joints=right_hand,
    )
    planner_header = json.loads(planner_msg[len(b"planner") : len(b"planner") + 1280].rstrip(b"\x00"))

    assert UPPER_BODY_JOINT_NAMES[:5] == (
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
        "left_shoulder_pitch",
        "right_shoulder_pitch",
    )
    assert upper_body == HORIZONTAL_CARRY_UPPER_BODY
    assert left_hand == OPEN_LEFT_HAND_JOINTS
    assert right_hand == OPEN_RIGHT_HAND_JOINTS
    assert [field["name"] for field in planner_header["fields"]] == [
        "mode",
        "movement",
        "facing",
        "speed",
        "height",
        "upper_body_position",
        "upper_body_velocity",
        "left_hand_joints",
        "right_hand_joints",
    ]


def test_sonic_joint_policy_mapping_matches_official_deploy_contract():
    backend = SonicIsaacJointBackend()
    zero_command = sonic_action_to_motor_command(zero_policy_action())
    backend_zero_command = backend.zero_motor_command()
    assert zero_command.joint_names == G1_JOINT_NAMES_MUJOCO_ORDER
    assert zero_command.q_target == DEFAULT_ANGLES_MUJOCO_ORDER
    assert backend_zero_command == zero_command
    assert len(zero_command.kp) == G1_NUM_MOTOR
    assert len(zero_command.kd) == G1_NUM_MOTOR

    action = [0.0] * G1_NUM_MOTOR
    action[ISAACLAB_TO_MUJOCO[0]] = 1.0
    action[ISAACLAB_TO_MUJOCO[4]] = -0.5
    command = sonic_action_to_motor_command(action)

    assert command.q_target[0] == pytest.approx(DEFAULT_ANGLES_MUJOCO_ORDER[0] + ACTION_SCALE_MUJOCO_ORDER[0])
    assert command.q_target[4] == pytest.approx(DEFAULT_ANGLES_MUJOCO_ORDER[4] - 0.5 * ACTION_SCALE_MUJOCO_ORDER[4])


def test_sonic_release_onnx_shapes_are_joint_policy_compatible():
    policy_dir = Path("third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release")
    decoder_info = load_onnx_model_info(policy_dir / "model_decoder.onnx")

    assert decoder_info.inputs[0].name == "obs_dict"
    assert decoder_info.inputs[0].shape == (1, 994)
    assert decoder_info.outputs[0].name == "action"
    assert decoder_info.outputs[0].shape == (1, G1_NUM_MOTOR)


def test_stage1_patch_is_bounded_and_typed():
    patch = PatchProposal(
        patch_id="qwen_reduce_drop_speed",
        patch_type="parameter",
        target="stage1_parameters",
        change={
            "forward_vy": 99.0,
            "path_speed": 0.01,
            "hip_height": 0.90,
        },
        rationale="Reduce box drop risk while improving wrist support.",
    )
    patched = apply_stage1_patch(Stage1Parameters(), patch)

    assert patched.forward_vy == 0.35
    assert patched.path_speed == 0.02
    assert patched.hip_height == 0.86
    assert patched.upper_mode == "hold_initial_scene"


def test_stage1_patch_cannot_change_locked_upper_body_pose():
    patch = PatchProposal(
        patch_id="qwen_bad_upper_patch",
        patch_type="parameter",
        target="stage1_parameters",
        change={"upper_mode": "planned_center_support"},
        rationale="This would violate the locked initial hand-pose contract.",
    )

    with pytest.raises(ValueError, match="unsupported Stage1 parameter"):
        apply_stage1_patch(Stage1Parameters(), patch)


def test_stage1_sonic_patch_is_bounded_and_keeps_real_backend_contract():
    patch = PatchProposal(
        patch_id="qwen_sync_lateral_commands",
        patch_type="parameter",
        target="stage1_sonic_parameters",
        change={
            "target_vel": 99.0,
            "robot_b_movement_y": -99.0,
            "replan_interval_steps": 1,
            "mode": "object_carrying",
        },
        rationale="Keep manual-layout world-frame commands synchronized while bounding speed.",
    )
    patched = apply_stage1_sonic_patch(Stage1SonicParameters(), patch)
    schema = stage1_sonic_patch_schema()

    assert patched.target_vel == 0.22
    assert patched.robot_b_movement_y == -1.0
    assert patched.replan_interval_steps == 2
    assert patched.mode == "object_carrying"
    assert schema["locked_contract"]["env_action_surface_used"] is False
    assert schema["locked_contract"]["direct_root_or_box_write"] is False


def test_stage1_sonic_patch_cannot_change_locked_hand_or_proxy_controls():
    patch = PatchProposal(
        patch_id="qwen_bad_proxy_request",
        patch_type="parameter",
        target="stage1_sonic_parameters",
        change={"proxy_action_used": True},
        rationale="This would bypass the SONIC joint WBC contract.",
    )

    with pytest.raises(ValueError, match="unsupported Stage1 SONIC parameter"):
        apply_stage1_sonic_patch(Stage1SonicParameters(), patch)


def test_stage1_visualization_writes_html(tmp_path):
    output = tmp_path / "stage1.html"
    write_stage1_evolution_html(
        output,
        attempts=[
            {
                "attempt": 1,
                "success": False,
                "failure": "box_dropped",
                "best_target_distance": 1.2,
                "final_target_distance": 1.4,
                "min_physical_box_z": 0.4,
            }
        ],
        histories=[
            [
                {
                    "box_x": -1.5,
                    "box_y": 2.3,
                    "robot_a_x": -1.8,
                    "robot_a_y": 2.0,
                    "robot_b_x": -1.8,
                    "robot_b_y": 2.8,
                }
            ]
        ],
    )

    assert output.exists()
    assert "Praxis-Coop Stage1" in output.read_text(encoding="utf-8")
