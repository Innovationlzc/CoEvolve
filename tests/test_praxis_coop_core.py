from pathlib import Path
import importlib.util
import json
from math import pi
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from praxis_coop.adapters import MockSonicAdapter
from praxis_coop.examples import two_g1_box_scene
from praxis_coop.coder_agent import apply_patch
from praxis_coop.isaac_scene_adapter import figure2_passage_scene_graph, manual_two_g1_box_push_scene_graph
from praxis_coop.knowledge_base import DeploymentExperience, DeploymentExperienceMemory, TextSpecMemory
from praxis_coop.models import FailureMode, PatchProposal, SkillGraph, SkillNode, SkillStatus
from praxis_coop.task_orchestrator import build_initial_skill_graph
from praxis_coop.qwen_evolution import _patch_from_payload, _spec_from_payload
from praxis_coop.runtime.manual_usd_runtime import DEFAULT_MANUAL_SCENE, ManualUsdSceneRuntime, run_manual_usd_skillgraph
from praxis_coop.sonic import (
    CooperativeCFGen,
    ELBOW110_CARRY_READY_UPPER_BODY,
    SonicLocomotionMode,
    SonicStaticSkillBuildConfig,
    build_static_sonic_skill_stream,
    map_skill_graph_to_sonic_stream,
    skill_requires_upper_body_lock,
)
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
    BOX_SUPPORT_PARALLEL_UPPER_BODY,
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
    place = registry.get("bend_knee_place_payload")
    assert "stop_if_contact_loss" in carry.guards
    assert hand_lock.skill_type == "sonic_upper_body_pose_lock"
    assert lateral.skill_type == "sonic_ground_plane_locomotion"
    assert longitudinal.command_schema["direction"] == "'forward'|'backward'|signed_float"
    assert "[-2*pi, 2*pi]" in yaw_turn.command_schema["delta_yaw_rad"]
    assert place.skill_type == "sonic_height_control_with_upper_body_lock"
    assert "upper_body_lock_active" in place.preconditions

    memory_path = tmp_path / "specs.jsonl"
    memory = TextSpecMemory(memory_path)
    memory.save()
    reloaded = TextSpecMemory(memory_path)

    assert reloaded.specs
    assert all(spec.rule_text for spec in reloaded.specs)


def test_dynamic_deployment_experience_memory_is_append_only_and_retrievable(tmp_path):
    memory_path = tmp_path / "experiences.jsonl"
    memory = DeploymentExperienceMemory(memory_path)
    memory.append(
        DeploymentExperience(
            run_id="run_carry_forward_001",
            skill_name="carry_forward",
            robot_ids=("front", "back"),
            outcome="success",
            context={"scene": "manual_layout_no_payload", "upper_body_lock": "elbow110_carry_ready"},
            verifier_tags=("upright", "sync_walk"),
            metrics={"duration_s": 12.0},
            artifact_path="results/deployment/static_sonic_skills/carry_forward_result.json",
        )
    )
    memory.append(
        DeploymentExperience(
            run_id="run_carry_turn_001",
            skill_name="carry_turn_in_place",
            robot_ids=("front", "back"),
            outcome="manual_review",
            verifier_tags=("needs_visual_review",),
        )
    )

    reloaded = DeploymentExperienceMemory(memory_path)
    carry = reloaded.retrieve(skill_name="carry_forward")

    assert len(memory_path.read_text(encoding="utf-8").splitlines()) == 2
    assert carry[0].run_id == "run_carry_forward_001"
    assert carry[0].robot_ids == ("front", "back")
    assert reloaded.retrieve(outcome="manual_review")[0].skill_name == "carry_turn_in_place"


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


def test_static_sonic_skills_cover_dual_and_single_robot_motion_contracts():
    forward = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(skill_name="parallel_forward", robot_ids=("front", "back"), distance_m=1.0)
    )
    carry_forward = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(skill_name="carry_forward", robot_ids=("front", "back"), distance_m=1.0)
    )
    single_back = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(skill_name="carry_backward", robot_ids=("front",), distance_m=0.4)
    )

    assert set(forward.by_robot()) == {"front", "back"}
    assert all(command.mode == SonicLocomotionMode.SLOW_WALK for command in forward.commands)
    assert all(command.hand_pose_policy == "none" for command in forward.commands)
    assert skill_requires_upper_body_lock("carry_forward")
    assert all(command.hand_reference == "elbow110_carry_ready" for command in carry_forward.commands)
    assert set(single_back.by_robot()) == {"front"}
    assert single_back.commands[0].movement[0] < 0.0


def test_static_sonic_lateral_skills_are_body_relative_and_lock_carry_pose():
    left = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(skill_name="carry_lateral_left", robot_ids=("front", "back"), distance_m=0.25)
    )
    right = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(skill_name="carry_lateral_right", robot_ids=("front",), distance_m=0.25)
    )
    rotated_left = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name="parallel_lateral_left",
            robot_ids=("front",),
            facing_xy=(0.0, 1.0),
            distance_m=0.25,
        )
    )

    assert set(left.by_robot()) == {"front", "back"}
    assert all(command.motion_axis == "lateral" for command in left.commands + right.commands)
    assert all(command.reference_frame == "body" for command in left.commands + right.commands)
    assert all(command.hand_reference == "elbow110_carry_ready" for command in left.commands)
    assert left.by_robot()["front"][0].movement == pytest.approx((0.0, 1.0, 0.0))
    assert right.by_robot()["front"][0].movement == pytest.approx((0.0, -1.0, 0.0))
    assert rotated_left.by_robot()["front"][0].movement == pytest.approx((-1.0, 0.0, 0.0))


def test_static_sonic_skills_generate_turn_and_place_without_planner_arm_targets():
    turn = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name="carry_turn_in_place",
            robot_ids=("front", "back"),
            yaw_deg=90.0,
            yaw_rate_degps=30.0,
            turn_segment_deg=30.0,
        )
    )
    place = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name="carry_place_crouch",
            robot_ids=("front", "back"),
            place_start_height=0.80,
            place_height=0.66,
            place_ramp_seconds=3.0,
            place_hold_seconds=2.0,
        )
    )
    recover = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name="carry_recover_stand",
            robot_ids=("front", "back"),
            height=0.80,
            place_ramp_seconds=3.0,
            place_hold_seconds=2.0,
        )
    )

    assert len(turn.commands) == 6
    assert all(command.motion_axis == "yaw" for command in turn.commands)
    assert [command.target_yaw_rad for command in turn.by_robot()["front"]] == pytest.approx(
        [0.5235987755982988, 1.0471975511965976, 1.5707963267948966]
    )
    assert {command.facing for command in turn.commands} == {(1.0, 0.0, 0.0)}
    assert all(command.upper_body_position is None for command in turn.commands + place.commands)
    assert all(command.left_hand_joints is None for command in turn.commands + place.commands)
    assert {command.height for command in place.commands} == {0.66}
    assert {command.mode for command in place.commands} == {SonicLocomotionMode.IDLE}
    assert all(command.movement == (0.0, 0.0, 0.0) for command in place.commands)
    assert all(command.speed == -1.0 for command in place.commands)
    assert all(command.motion_axis == "height" for command in place.commands)
    assert {command.height for command in recover.commands} == {0.80}
    assert {command.mode for command in recover.commands} == {SonicLocomotionMode.IDLE}
    assert all("carry_place_crouch" not in command.phase_id for command in recover.commands)
    assert all(command.motion_axis == "height" for command in recover.commands)


def test_static_turn_command_scale_calibrates_requested_visual_yaw():
    turn = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name="carry_turn_in_place",
            robot_ids=("front",),
            yaw_deg=90.0,
            yaw_rate_degps=5.0,
            turn_segment_deg=15.0,
            turn_command_scale=0.5,
        )
    )

    commands = turn.by_robot()["front"]
    assert sum(command.delta_yaw_rad or 0.0 for command in commands) == pytest.approx(pi / 4.0)
    assert commands[-1].target_yaw_rad == pytest.approx(pi / 4.0)


def test_static_turn_heading_state_stays_bounded_for_repeated_turns(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sonic_static_skill.py"
    spec = importlib.util.spec_from_file_location("run_sonic_static_skill_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    state_file = tmp_path / "turn_heading_state.json"
    state_file.write_text(json.dumps({"cumulative_yaw_rad": 5.0 * pi}), encoding="utf-8")
    stream = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name="carry_turn_in_place",
            robot_ids=("front", "back"),
            yaw_deg=90.0,
            yaw_rate_degps=8.0,
            turn_segment_deg=30.0,
        )
    )

    adjusted, state = module._prepare_turn_heading_stream(
        stream,
        skill_name="carry_turn_in_place",
        state_file=state_file,
        reset_state=False,
        heading_mode="absolute_state",
    )

    front_targets = [command.target_yaw_rad for command in adjusted.by_robot()["front"]]
    assert all(target is not None and -pi <= target <= pi for target in front_targets)
    assert front_targets == pytest.approx([-5.0 * pi / 6.0, -2.0 * pi / 3.0, -pi / 2.0])
    assert state["mode"] == "bounded_relative_state"
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted["cumulative_yaw_rad"] == pytest.approx(-pi / 2.0)
    assert persisted["heading_state_is_wrapped"]


def test_static_skill_wrapper_lateral_does_not_use_turn_heading_state(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_static_sonic_skill.py"
    spec = importlib.util.spec_from_file_location("validate_static_sonic_skill_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    heading_state = tmp_path / "heading_state.json"
    base = SimpleNamespace(
        robots=["front", "back"],
        front_port=5556,
        back_port=5558,
        output_root=tmp_path,
        heading_state_file=heading_state,
        current_yaw_deg=None,
        start_pulse_seconds=0.0,
        stand_seconds=1.0,
        final_idle_seconds=0.0,
        publish_hz=20.0,
        extra=[],
    )

    module._write_heading_state(heading_state, -pi / 2.0, "carry_turn_clockwise_90")
    command = module.build_command(base, "carry_lateral_left", tmp_path / "planner_heartbeat.json")

    assert "--facing-x" in command
    assert "--facing-y" in command
    assert float(command[command.index("--movement-x") + 1]) == pytest.approx(0.0)
    assert float(command[command.index("--movement-y") + 1]) == pytest.approx(1.0)
    assert float(command[command.index("--facing-x") + 1]) == pytest.approx(1.0)
    assert float(command[command.index("--facing-y") + 1]) == pytest.approx(0.0)
    assert command[command.index("--distance-m") + 1] == "1.5"
    assert command[command.index("--speed-mps") + 1] == "0.45"


def test_static_skill_wrapper_prefers_live_runtime_heading_for_lateral(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_static_sonic_skill.py"
    spec = importlib.util.spec_from_file_location("validate_static_sonic_skill_live_heading_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    runtime_state = tmp_path / "runtime_state.json"
    runtime_state.write_text(
        json.dumps(
            {
                "elastic_band_dropped": True,
                "mean_heading_yaw_rad": 0.0,
                "robots": {
                    "front": {"yaw_rad": -pi / 2.0},
                    "back": {"yaw_rad": -pi / 2.0},
                },
            }
        ),
        encoding="utf-8",
    )
    heading_state = tmp_path / "heading_state.json"
    module._write_heading_state(heading_state, pi / 2.0, "stale_local_state")
    base = SimpleNamespace(
        robots=["front", "back"],
        front_port=5556,
        back_port=5558,
        output_root=tmp_path,
        heading_state_file=heading_state,
        runtime_state_file=runtime_state,
        runtime_state_max_age_s=10.0,
        no_live_runtime_heading=False,
        current_yaw_deg=None,
        start_pulse_seconds=0.0,
        stand_seconds=1.0,
        final_idle_seconds=0.0,
        publish_hz=20.0,
        extra=[],
        stand_seconds_explicit=False,
    )

    command = module.build_command(base, "carry_lateral_left", tmp_path / "planner_heartbeat.json")

    assert float(command[command.index("--movement-x") + 1]) == pytest.approx(0.0)
    assert float(command[command.index("--movement-y") + 1]) == pytest.approx(1.0)
    assert float(command[command.index("--facing-x") + 1]) == pytest.approx(1.0)
    assert float(command[command.index("--facing-y") + 1]) == pytest.approx(0.0)
    assert command[command.index("--stand-seconds") + 1] == "0.0"


def test_static_skill_wrapper_has_unambiguous_turn_aliases(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_static_sonic_skill.py"
    spec = importlib.util.spec_from_file_location("validate_static_sonic_skill_alias_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = SimpleNamespace(
        robots=["front", "back"],
        front_port=5556,
        back_port=5558,
        output_root=tmp_path,
        heading_state_file=tmp_path / "heading_state.json",
        current_yaw_deg=None,
        start_pulse_seconds=0.0,
        stand_seconds=1.0,
        final_idle_seconds=0.0,
        publish_hz=20.0,
        extra=[],
    )

    clockwise = module.build_command(base, "carry_turn_clockwise_90", tmp_path / "planner_heartbeat.json")
    counterclockwise = module.build_command(base, "carry_turn_counterclockwise_90", tmp_path / "planner_heartbeat.json")

    assert clockwise[clockwise.index("--skill") + 1] == "carry_turn_in_place"
    assert clockwise[clockwise.index("--yaw-deg") + 1] == "-90"
    assert counterclockwise[counterclockwise.index("--skill") + 1] == "carry_turn_in_place"
    assert counterclockwise[counterclockwise.index("--yaw-deg") + 1] == "90"


def test_static_skill_wrapper_lateral_command_does_not_include_turn_args(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_static_sonic_skill.py"
    spec = importlib.util.spec_from_file_location("validate_static_sonic_skill_lateral_only_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = SimpleNamespace(
        robots=["front", "back"],
        front_port=5556,
        back_port=5558,
        output_root=tmp_path,
        heading_state_file=tmp_path / "heading_state.json",
        runtime_state_file=tmp_path / "runtime_state.json",
        runtime_state_max_age_s=10.0,
        no_live_runtime_heading=True,
        current_yaw_deg=-90.0,
        start_pulse_seconds=0.0,
        stand_seconds=1.0,
        final_idle_seconds=0.0,
        publish_hz=20.0,
        extra=[],
    )

    command = module.build_command(base, "carry_lateral_left", tmp_path / "planner_heartbeat.json")

    assert command[command.index("--skill") + 1] == "carry_lateral_left"
    assert "--movement-x" in command
    assert "--movement-y" in command
    assert "--facing-x" in command
    assert "--facing-y" in command
    assert "--yaw-deg" not in command
    assert "--turn-heading-mode" not in command


def test_static_skill_wrapper_parses_extra_without_separator(monkeypatch: pytest.MonkeyPatch):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_static_sonic_skill.py"
    spec = importlib.util.spec_from_file_location("validate_static_sonic_skill_extra_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "validate_static_sonic_skill.py",
            "--skill",
            "carry_turn_in_place",
            "--extra",
            "--yaw-deg",
            "-90",
        ],
    )

    args = module.parse_args()

    assert args.skill == "carry_turn_in_place"
    assert args.extra == ["--yaw-deg", "-90"]


def test_static_skill_wrapper_accepts_direct_parameter_overrides(monkeypatch: pytest.MonkeyPatch):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_static_sonic_skill.py"
    spec = importlib.util.spec_from_file_location("validate_static_sonic_skill_direct_params_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "validate_static_sonic_skill.py",
            "--skill",
            "carry_forward",
            "--distance-m",
            "0.35",
            "--speed-mps",
            "0.04",
            "--dry-run",
        ],
    )

    args = module.parse_args()
    heartbeat_file = Path("/tmp") / "planner_heartbeat.json"
    command = module.build_command(args, "carry_forward", heartbeat_file)

    assert args.extra == ["--distance-m", "0.35", "--speed-mps", "0.04"]
    assert command[command.index("--distance-m") + 1] == "0.35"
    assert command[command.index("--speed-mps") + 1] == "0.04"


def test_static_skill_wrapper_place_crouch_uses_tuned_live_defaults(monkeypatch: pytest.MonkeyPatch):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_static_sonic_skill.py"
    spec = importlib.util.spec_from_file_location("validate_static_sonic_skill_place_defaults_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "validate_static_sonic_skill.py",
            "--skill",
            "carry_place_crouch",
            "--place-height",
            "0.68",
            "--dry-run",
        ],
    )

    args = module.parse_args()
    command = module.build_command(args, "carry_place_crouch", Path("/tmp") / "planner_heartbeat.json")

    assert command[command.index("--start-pulse-seconds") + 1] == "0.0"
    assert command[command.index("--stand-seconds") + 1] == "1.0"
    assert command[command.index("--place-height") + 1] == "0.68"


def test_sonic_zmq_message_builders_match_topic_prefixes():
    command_msg = build_command_message(start=True, stop=False, planner=True)
    heading_msg = build_command_message(start=False, stop=False, planner=True, delta_heading=1.25)
    heading_increment_msg = build_command_message(
        start=False,
        stop=False,
        planner=True,
        heading_increment=0.05,
    )
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
    assert heading_msg.startswith(b"command")
    assert heading_increment_msg.startswith(b"command")
    assert b"delta_heading" in heading_msg[:1287]
    assert b"heading_increment" in heading_increment_msg[:1287]
    assert planner_msg.startswith(b"planner")
    assert len(command_msg) > 1280
    assert len(heading_msg) > len(command_msg)
    assert len(heading_increment_msg) > len(command_msg)
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


def test_elbow110_carry_ready_pose_is_named_and_matches_tuned_values():
    upper_body = resolve_upper_body_pose("elbow110_carry_ready")

    assert upper_body == ELBOW110_CARRY_READY_UPPER_BODY
    assert upper_body[3] == pytest.approx(-0.75)
    assert upper_body[9] == pytest.approx(0.15)


def test_box_support_parallel_pose_has_level_forearms_in_mujoco_fk():
    mujoco = pytest.importorskip("mujoco")
    result = generate_dual_manual_layout_scene(include_payload=False)
    model = mujoco.MjModel.from_xml_path(str(result.scene_path))
    data = mujoco.MjData(model)

    for joint_name, value in zip(UPPER_BODY_JOINT_NAMES, BOX_SUPPORT_PARALLEL_UPPER_BODY):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"front_{joint_name}_joint")
        assert joint_id >= 0, joint_name
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)

    def body_pos(name: str):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id >= 0, name
        return data.xpos[body_id].copy()

    torso = body_pos("front_torso_link")
    left_elbow = body_pos("front_left_elbow_link")
    left_wrist = body_pos("front_left_wrist_roll_link")
    left_palm = body_pos("front_left_wrist_yaw_link")
    right_elbow = body_pos("front_right_elbow_link")
    right_wrist = body_pos("front_right_wrist_roll_link")
    right_palm = body_pos("front_right_wrist_yaw_link")

    assert abs(left_wrist[2] - left_elbow[2]) < 0.005
    assert abs(right_wrist[2] - right_elbow[2]) < 0.005
    assert abs(left_palm[2] - left_wrist[2]) < 0.015
    assert abs(right_palm[2] - right_wrist[2]) < 0.015
    assert abs(left_wrist[2] - right_wrist[2]) < 0.005
    assert abs(left_palm[2] - right_palm[2]) < 0.005
    assert left_palm[0] - torso[0] > 0.25
    assert right_palm[0] - torso[0] > 0.25


def test_world_level_shoulder_pitch_compensation_preserves_forearm_slope_in_fk():
    mujoco = pytest.importorskip("mujoco")
    import math
    import numpy as np

    result = generate_dual_manual_layout_scene(include_payload=False)
    model = mujoco.MjModel.from_xml_path(str(result.scene_path))
    data = mujoco.MjData(model)
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "front_floating_base_joint")
    root_qpos_adr = int(model.jnt_qposadr[root_joint])
    initial_qpos = data.qpos.copy()

    def set_pose(root_pitch: float, shoulder_compensation: float) -> tuple[float, float]:
        data.qpos[:] = initial_qpos
        data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = np.asarray(
            [math.cos(root_pitch / 2.0), 0.0, math.sin(root_pitch / 2.0), 0.0]
        )
        target = list(ELBOW110_CARRY_READY_UPPER_BODY)
        target[3] += shoulder_compensation
        target[4] += shoulder_compensation
        for joint_name, value in zip(UPPER_BODY_JOINT_NAMES, target):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"front_{joint_name}_joint")
            assert joint_id >= 0, joint_name
            data.qpos[model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(model, data)

        def body_pos(name: str):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert body_id >= 0, name
            return data.xpos[body_id].copy()

        left_dz = body_pos("front_left_wrist_roll_link")[2] - body_pos("front_left_elbow_link")[2]
        right_dz = body_pos("front_right_wrist_roll_link")[2] - body_pos("front_right_elbow_link")[2]
        return float(left_dz), float(right_dz)

    baseline = set_pose(0.0, 0.0)
    pitched_no_comp = set_pose(0.35, 0.0)
    pitched_with_comp = set_pose(0.35, -0.35)

    no_comp_error = abs(pitched_no_comp[0] - baseline[0]) + abs(pitched_no_comp[1] - baseline[1])
    compensated_error = abs(pitched_with_comp[0] - baseline[0]) + abs(pitched_with_comp[1] - baseline[1])
    assert compensated_error < no_comp_error * 0.2


def test_dual_manual_layout_payload_mass_is_configurable():
    result = generate_dual_manual_layout_scene(include_payload=True, payload_mass_kg=5.0)
    root = ET.parse(result.scene_path).getroot()
    long_box = next(
        geom
        for geom in root.iter("geom")
        if geom.get("name") == "long_box_geom"
    )

    assert float(long_box.get("mass", "0")) == pytest.approx(5.0)
    assert "payload_mass_kg=5" in result.robot_start_note


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
