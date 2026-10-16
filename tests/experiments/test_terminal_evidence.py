from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from coevolve_experiments.reporting.terminal_evidence import (
    MUJOCO_TERMINAL_SAMPLE_TRACE_KIND,
    MUJOCO_TERMINAL_STATE_KIND,
    TERMINAL_EVIDENCE_FILENAME,
    TERMINAL_EVIDENCE_KIND,
    validate_terminal_evidence,
)
from coevolve_experiments.reporting.terminal_aux_evidence import (
    FROZEN_PAYLOAD_DROP_SAFETY_PROFILE,
    PAYLOAD_DROP_EVALUATION_KIND,
    PAYLOAD_DROP_EXECUTION_TRACE_KIND,
    PAYLOAD_DROP_POLICY_KIND,
    PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
    PAYLOAD_DROP_TRANSACTION_RECEIPT_KIND,
    RAW_AUX_EVIDENCE_SCHEMA_VERSION,
    SYSTEM_STALL_ACTION_SPACE_KIND,
    SYSTEM_STALL_CHECKPOINT_KIND,
    SYSTEM_STALL_HOST_PREFLIGHT_KIND,
    SYSTEM_STALL_MODEL_OUTPUT_KIND,
    SYSTEM_STALL_STATIC_VALIDATION_KIND,
    capability_bound_auxiliary_producer_allowlist,
    recompute_payload_drop_ledger,
    recompute_system_stall_ledger,
)


RUN_ID = "table2_method_comparison__remac_adapted__seed_200"
CONDITION, TERMINAL, SCENE = "a" * 64, "b" * 64, "c" * 64
COMPILED, REALIZATION, EXECUTION_EVIDENCE = "d" * 64, "e" * 64, "f" * 64
TARGET = (-1.5, 7.6, 0.3)
ENTRYPOINT = "7" * 64
AUXILIARY_PRODUCERS = capability_bound_auxiliary_producer_allowlist(ENTRYPOINT)


def _write(root: Path, relative: str, payload: object) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return {"relative_path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _payload_hash(payload: object) -> str:
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _producer(family: str) -> dict[str, str]:
    return dict(AUXILIARY_PRODUCERS[family])


def _simulator() -> dict[str, object]:
    return {
        "name": "MuJoCo", "version": "3.3.5",
        "compiled_model_sha256": COMPILED,
        "scene_realization_sha256": REALIZATION,
        "scene_fingerprint": SCENE,
        "scene_execution_evidence_fingerprint": EXECUTION_EVIDENCE,
    }


def _quaternion(tilt_deg: float = 0.0) -> list[float]:
    half = math.radians(tilt_deg) / 2
    return [math.cos(half), math.sin(half), 0.0, 0.0]


def _trace(*, success: bool = True, xy: float = 0.12, tilt: float = 12.0) -> dict[str, object]:
    times = (0.0, 1.5, 3.0) if success else (0.0,)
    samples = []
    for index, time_s in enumerate(times):
        contacts = ([{
            "geom1_id": 11, "geom2_id": 22, "geom1_role": "payload",
            "geom2_role": "target", "distance_m": -0.0001,
        }] if success else [])
        samples.append({
            "sample_index": index, "time_s": time_s,
            "payload_released": success,
            "payload_pose": {
                "position_xyz_m": [TARGET[0] + xy, TARGET[1], 0.65],
                "quaternion_wxyz": _quaternion(tilt),
            },
            "robot_poses": {
                "front": {"position_xyz_m": [TARGET[0] + 1.20, TARGET[1], 0.74], "quaternion_wxyz": [1, 0, 0, 0]},
                "back": {"position_xyz_m": [TARGET[0] - 1.22, TARGET[1], 0.75], "quaternion_wxyz": [1, 0, 0, 0]},
            },
            "contacts": contacts,
        })
    return {
        "kind": MUJOCO_TERMINAL_SAMPLE_TRACE_KIND, "schema_version": 2,
        "run_id": RUN_ID, "seed": 200, "checkpoint_id": "checkpoint_6",
        "checkpoint_event_id": "save:checkpoint_6:007",
        "simulator": _simulator(),
        "target": {"center_xyz_m": list(TARGET), "scale_xyz_m": [0.6, 0.6, 0.6]},
        "payload_geometry": {"half_height_m": 0.05},
        "sample_window_s": times[-1] - times[0], "samples": samples,
        "lifecycle": {
            "release_completed": success, "retreat_completed": success,
            "stand_recovered": success,
        },
    }


def _state(*, success: bool = True, xy: float = 0.12, tilt: float = 12.0) -> dict[str, object]:
    return {
        "kind": MUJOCO_TERMINAL_STATE_KIND, "schema_version": 3,
        "run_id": RUN_ID, "seed": 200, "checkpoint_id": "checkpoint_6",
        "checkpoint_event_id": "save:checkpoint_6:007",
        "simulator": _simulator(), "sample_window_s": 3.0 if success else 0.0,
        "sample_count": 3 if success else 1,
        "payload": {
            "released": success, "target_contact_all_samples": success,
            "floor_contact_samples": 0, "wall_contact_samples": 0,
            "robot_contact_samples": 0, "xy_error_m": xy,
            "z_error_m": 0.0, "tilt_deg": tilt,
        },
        "robots": {
            "front": {"upright": True, "pelvis_height_m": 0.74, "distance_to_target_m": 1.20},
            "back": {"upright": True, "pelvis_height_m": 0.75, "distance_to_target_m": 1.22},
        },
        "lifecycle": {
            "release_completed": success, "retreat_completed": success,
            "stand_recovered": success,
        },
    }


def _receipt(
    root: Path, *, status: str = "SUCCESS", state: dict[str, object] | None = None,
    trace: dict[str, object] | None = None, stall: dict[str, object] | None = None,
    drops: dict[str, object] | None = None, physical: int = 7,
) -> dict[str, object]:
    state_ref = _write(root, "evidence/terminal_state.json", state or _state(success=status == "SUCCESS"))
    trace_ref = _write(root, "evidence/terminal_trace.json", trace or _trace(success=status == "SUCCESS"))
    stall_ref = _write(root, "evidence/system_stall.json", stall) if stall else None
    drop_ref = _write(root, "evidence/payload_drop.json", drops) if drops else None
    receipt = {
        "kind": TERMINAL_EVIDENCE_KIND, "schema_version": 3,
        "producer": {"id": "praxis_coop.host.terminal_verifier", "source_sha256": "9" * 64},
        "run_binding": {
            "run_id": RUN_ID, "seed": 200, "condition_fingerprint": CONDITION,
            "terminal_fingerprint": TERMINAL, "scene_fingerprint": SCENE,
            "scene_execution_evidence_fingerprint": EXECUTION_EVIDENCE,
        },
        "terminal_status": status, "physical_transactions": physical,
        "terminal_state": state_ref, "terminal_sample_trace": trace_ref,
        "system_stall": stall_ref, "payload_drop": drop_ref,
    }
    raw = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    (root / TERMINAL_EVIDENCE_FILENAME).write_bytes(raw)
    return {
        "live_launch": {"entrypoint_sha256": ENTRYPOINT},
        "result_artifact_hashes": {
            "terminal_verifier_evidence_sha256": hashlib.sha256(raw).hexdigest()
        },
    }


def _validate(root: Path, manifest: dict[str, object], status: str = "SUCCESS", physical: int = 7):
    return validate_terminal_evidence(
        root, manifest=manifest, expected_run_id=RUN_ID, expected_seed=200,
        expected_condition_fingerprint=CONDITION, expected_terminal_fingerprint=TERMINAL,
        expected_terminal_status=status, expected_scene_fingerprint=SCENE,
        expected_scene_realization_sha256=REALIZATION,
        expected_compiled_model_sha256=COMPILED,
        expected_scene_execution_evidence_fingerprint=EXECUTION_EVIDENCE,
        expected_target_center_xyz_m=TARGET, physical_transactions=physical,
        physical_budget=80,
    )


def test_success_is_recomputed_from_raw_mujoco_pose_and_contacts(tmp_path: Path) -> None:
    manifest = _receipt(tmp_path)
    result = _validate(tmp_path, manifest)
    assert result.valid
    assert "evidence/terminal_trace.json" in (result.artifact_hashes or {})


def test_forged_summary_cannot_override_raw_trace(tmp_path: Path) -> None:
    manifest = _receipt(tmp_path, trace=_trace(xy=0.31))
    result = _validate(tmp_path, manifest)
    assert "SUCCESS_PAYLOAD_OUTSIDE_TARGET_SUPPORT" in result.invalid_reasons
    assert "TERMINAL_STATE_PAYLOAD_XY_ERROR_M_TRACE_MISMATCH" in result.invalid_reasons


def test_raw_quaternion_and_contact_pairs_drive_success(tmp_path: Path) -> None:
    trace = _trace(tilt=30.0)
    trace["samples"][1]["contacts"].append({  # type: ignore[index]
        "geom1_id": 11, "geom2_id": 33, "geom1_role": "payload",
        "geom2_role": "floor", "distance_m": -0.001,
    })
    manifest = _receipt(tmp_path, trace=trace)
    result = _validate(tmp_path, manifest)
    assert "SUCCESS_PAYLOAD_TILT_INVALID" in result.invalid_reasons
    assert "SUCCESS_PAYLOAD_FLOOR_CONTACT" in result.invalid_reasons


def test_compiled_model_and_scene_realization_are_external_expectations(tmp_path: Path) -> None:
    manifest = _receipt(tmp_path)
    result = validate_terminal_evidence(
        tmp_path, manifest=manifest, expected_run_id=RUN_ID, expected_seed=200,
        expected_condition_fingerprint=CONDITION, expected_terminal_fingerprint=TERMINAL,
        expected_terminal_status="SUCCESS", expected_scene_fingerprint=SCENE,
        expected_scene_realization_sha256="0" * 64,
        expected_compiled_model_sha256=COMPILED,
        expected_scene_execution_evidence_fingerprint=EXECUTION_EVIDENCE,
        expected_target_center_xyz_m=TARGET, physical_transactions=7, physical_budget=80,
    )
    assert "TERMINAL_TRACE_SCENE_REALIZATION_SHA256_MISMATCH" in result.invalid_reasons


def _stall(
    root: Path,
    *,
    declared_families: list[str] | None = None,
    model_content: str = "{}",
    declared_parser_passed: bool = True,
    require_valid: bool = True,
):
    trace = _trace(success=False)
    state = _state(success=False)
    checkpoint_id = "checkpoint_6"
    checkpoint_event_id = "save:checkpoint_6:007"
    common = {
        "schema_version": RAW_AUX_EVIDENCE_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "checkpoint_id": checkpoint_id,
        "checkpoint_event_id": checkpoint_event_id,
    }
    checkpoint = _write(root, "evidence/system_stall/checkpoint_validity.json", {
        "kind": SYSTEM_STALL_CHECKPOINT_KIND,
        "producer": _producer("stall.checkpoint"),
        **common,
        "terminal_state_sha256": _payload_hash(state),
        "terminal_sample_trace_sha256": _payload_hash(trace),
        "sample_window_s": 0.0,
        "sample_count": 1,
        "payload": {"floor_contact_samples": 0, "wall_contact_samples": 0},
        "robots": {
            "front": {"upright": True, "pelvis_height_m": 0.74},
            "back": {"upright": True, "pelvis_height_m": 0.75},
        },
    })
    action = _write(root, "evidence/system_stall/host_action_space.json", {
        "kind": SYSTEM_STALL_ACTION_SPACE_KIND,
        "producer": _producer("stall.host_action_space"),
        **common,
        "checkpoint_validity_sha256": checkpoint["sha256"],
        "lifecycle_state": "carry",
        "admissible_action_families": declared_families or ["sync:coordinated_linear_transport"],
        "allowed_primitives": ["coordinated_linear_transport"],
        "allowed_coordination_modes": ["sync"],
    })
    cycles = []
    previous_preflight = None
    for cycle in range(1, 4):
        prefix = f"evidence/system_stall/cycle_{cycle:03d}"
        cycle_common = {**common, "cycle": cycle}
        model = _write(root, f"{prefix}_model_output.json", {
            "kind": SYSTEM_STALL_MODEL_OUTPUT_KIND,
            "producer": _producer("stall.model_output"),
            **cycle_common,
            "checkpoint_validity_sha256": checkpoint["sha256"],
            "host_action_space_sha256": action["sha256"],
            "previous_cycle_preflight_sha256": previous_preflight,
            "request_id": f"request-{cycle}", "usage_call_ordinal": cycle,
            "role": "policy_generator", "model": "gpt", "finish_reason": "stop", "content": model_content,
        })
        static = _write(root, f"{prefix}_static_validation.json", {
            "kind": SYSTEM_STALL_STATIC_VALIDATION_KIND,
            "producer": _producer("stall.static_validation"),
            **cycle_common,
            "model_output_sha256": model["sha256"], "parser_passed": declared_parser_passed,
            "script_present": False, "static_validation_passed": False,
            "failure_codes": ["NO_SCRIPT"],
        })
        preflight = _write(root, f"{prefix}_host_preflight.json", {
            "kind": SYSTEM_STALL_HOST_PREFLIGHT_KIND,
            "producer": _producer("stall.host_preflight"),
            **cycle_common,
            "static_validation_sha256": static["sha256"], "evaluated": False,
            "host_action_space_sha256": action["sha256"],
            "passed": False, "skipped_reason": "STATIC_VALIDATION_FAILED",
            "failure_codes": ["STATIC_VALIDATION_FAILED"],
            "executable_child_produced": False,
            "physical_dispatch": False,
        })
        cycles.append({
            "model_output": model,
            "static_validation": static,
            "host_preflight": preflight,
        })
        previous_preflight = preflight["sha256"]
    result = recompute_system_stall_ledger(
        root,
        source_index={
            "checkpoint_validity": checkpoint,
            "host_action_space": action,
            "generation_cycles": cycles,
        },
        expected_run_id=RUN_ID,
        expected_checkpoint_id=checkpoint_id,
        expected_checkpoint_event_id=checkpoint_event_id,
        expected_terminal_state_sha256=_payload_hash(state),
        expected_terminal_sample_trace_sha256=_payload_hash(trace),
        derived_checkpoint={
            "sample_window_s": 0.0,
            "sample_count": 1,
            "payload": {"floor_contact_samples": 0, "wall_contact_samples": 0},
            "robots": {
                "front": {"upright": True, "pelvis_height_m": 0.74},
                "back": {"upright": True, "pelvis_height_m": 0.75},
            },
        },
        expected_producers=AUXILIARY_PRODUCERS,
    )
    if require_valid:
        assert result.valid
        return dict(result.ledger or {})
    return result


def test_system_stall_binds_every_cycle_stage_artifact(tmp_path: Path) -> None:
    stall = _stall(tmp_path)
    manifest = _receipt(tmp_path, status="SYSTEM_STALL", stall=stall)
    result = _validate(tmp_path, manifest, "SYSTEM_STALL")
    assert result.valid
    assert result.stall_model_request_ids == ("request-1", "request-2", "request-3")
    stall["generation_cycles"][1]["action_space"] = None  # type: ignore[index]
    manifest = _receipt(tmp_path / "missing", status="SYSTEM_STALL", stall=stall)
    result = _validate(tmp_path / "missing", manifest, "SYSTEM_STALL")
    assert "SYSTEM_STALL_CYCLE_ARTIFACT_SET_INCOMPLETE" in result.invalid_reasons


def test_system_stall_recomputes_action_space_and_static_preflight(
    tmp_path: Path,
) -> None:
    family_result = _stall(
        tmp_path / "family",
        declared_families=["sync:hold_position"],
        require_valid=False,
    )
    assert "SYSTEM_STALL_ACTION_SPACE_DERIVATION_MISMATCH" in family_result.invalid_reasons

    parser_result = _stall(
        tmp_path / "parser",
        declared_parser_passed=False,
        require_valid=False,
    )
    assert "SYSTEM_STALL_STATIC_VALIDATION_DERIVATION_MISMATCH" in parser_result.invalid_reasons


def test_system_stall_rejects_a_valid_admissible_team_child(tmp_path: Path) -> None:
    code = (
        "def policy(api):\n"
        '    api.coordinated_linear_transport(direction="forward", distance_m=0.25, '
        "speed_mps=0.5, ramp_s=1.5)\n"
        "    api.stop_all()\n"
    )
    content = json.dumps(
        {
            "robot_scripts": {
                robot_id: {
                    "code": code,
                    "synchronization_group": "team",
                    "coordination_mode": "sync",
                    "contact_intent": "maintain_payload_support",
                }
                for robot_id in ("front", "back")
            }
        }
    )
    result = _stall(
        tmp_path,
        model_content=content,
        require_valid=False,
    )
    assert "SYSTEM_STALL_STATIC_VALIDATION_DERIVATION_MISMATCH" in result.invalid_reasons
    assert "SYSTEM_STALL_HOST_PREFLIGHT_DERIVATION_MISMATCH" in result.invalid_reasons


def _drop_ledger(
    root: Path,
    *,
    physical_z_m: float = 0.59,
    floor_distance_m: float = -0.001,
    forged_physical_producer: bool = False,
    declared_total_support_force_n: float = 0.0,
    require_valid: bool = True,
):
    refs = []
    # Transactions 1--4 are outside this five-drop terminal suffix.  Its first
    # receipt still carries the full-chain predecessor hash; accounting binds
    # that digest to the actual transaction 4 receipt.
    previous_receipt = "0" * 64
    for transaction in range(5, 10):
        offset = transaction - 5
        event_before = f"restore:checkpoint_6:{offset + 2:03d}"
        event_after = (
            "save:checkpoint_6:007"
            if transaction == 9
            else f"restore:checkpoint_6:{offset + 3:03d}"
        )
        common = {
            "schema_version": RAW_AUX_EVIDENCE_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "physical_transaction": transaction,
        }
        script = f"drop_probe({transaction})"
        policy = _write(root, f"evidence/transactions/transaction_{transaction:03d}_policy.json", {
            "kind": PAYLOAD_DROP_POLICY_KIND,
            "producer": _producer("payload_drop.policy"),
            **common,
            "checkpoint_id": "checkpoint_6",
            "checkpoint_event_id": event_before,
            "action_script": script,
            "action_script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        })
        evaluation = _write(root, f"evidence/transactions/transaction_{transaction:03d}_evaluation.json", {
            "kind": PAYLOAD_DROP_EVALUATION_KIND,
            "producer": _producer("payload_drop.evaluation"),
            **common,
            "checkpoint_id": "checkpoint_6",
            "checkpoint_event_id": event_before,
            "accepted": False,
            "failure_type": "payload_drop",
            "transition_outcome": "REJECT",
            "completed_phase_ids_before": ["phase_1"],
            "completed_phase_ids_after": ["phase_1"],
        })
        execution = _write(root, f"evidence/transactions/transaction_{transaction:03d}_execution_trace.json", {
            "kind": PAYLOAD_DROP_EXECUTION_TRACE_KIND,
            "producer": _producer("payload_drop.execution_trace"),
            **common,
            "checkpoint_id_before": "checkpoint_6",
            "checkpoint_event_id_before": event_before,
            "checkpoint_id_after": "checkpoint_6",
            "checkpoint_event_id_after": event_after,
            "physically_dispatched": True,
            "rollback_performed": True,
            "physics_step_count": 50,
            "dispatch_started_s": float(offset),
            "dispatch_finished_s": float(offset) + 0.5,
        })
        physical_producer = _producer("payload_drop.physical_samples")
        if forged_physical_producer:
            physical_producer["source_sha256"] = "8" * 64
        physical_samples = _write(
            root,
            f"evidence/transactions/transaction_{transaction:03d}_physical_samples.json",
            {
                "kind": PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
                "producer": physical_producer,
                **common,
                "checkpoint_id": "checkpoint_6",
                "checkpoint_event_id": event_before,
                "simulator": {
                    "name": "MuJoCo",
                    "version": "3.3.5",
                    "compiled_model_sha256": COMPILED,
                    "scene_realization_sha256": REALIZATION,
                },
                "drop_predicate": {
                    "safety_profile": FROZEN_PAYLOAD_DROP_SAFETY_PROFILE,
                    "minimum_payload_z_m": 0.65,
                    "height_margin_m": 0.05,
                    "drop_z_threshold_m": 0.60,
                    "contact_distance_tolerance_m": 1.0e-6,
                },
                "samples": [
                    {
                        "sample_index": 0,
                        "time_s": float(offset),
                        "payload_active": True,
                        "payload_pose": {
                            "position_xyz_m": [0.0, 0.0, physical_z_m],
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        "contacts": [
                            {
                                "geom1_id": 11,
                                "geom2_id": 12,
                                "geom1_role": "payload",
                                "geom2_role": "floor",
                                "geom1_name": "long_box_collision",
                                "geom2_name": "manual_floor",
                                "body1_name": "long_box",
                                "body2_name": "manual_floor",
                                "distance_m": floor_distance_m,
                                "normal_force_n": 5.0,
                            }
                        ],
                        "support": {
                            "available": True,
                            "supporting_robots": [],
                            "normal_force_n_by_robot": {"front": 0.0, "back": 0.0},
                            "total_normal_force_n": declared_total_support_force_n,
                        },
                    }
                ],
            },
        )
        receipt = _write(root, f"evidence/transactions/transaction_{transaction:03d}.json", {
            "kind": PAYLOAD_DROP_TRANSACTION_RECEIPT_KIND,
            "producer": _producer("payload_drop.transaction_receipt"),
            **common,
            "checkpoint_id_before": "checkpoint_6",
            "checkpoint_event_id_before": event_before,
            "checkpoint_id_after": "checkpoint_6",
            "checkpoint_event_id_after": event_after,
            "physically_dispatched": True,
            "verifier": {"accepted": False, "failure_type": "payload_drop"},
            "rollback_performed": True,
            "drop_counter_before": offset,
            "drop_counter_after": offset + 1,
            "policy": policy,
            "evaluation": evaluation,
            "execution_trace": execution,
            "physical_samples": physical_samples,
            "previous_receipt_sha256": previous_receipt,
        })
        refs.append(receipt)
        previous_receipt = receipt["sha256"]
    result = recompute_payload_drop_ledger(
        root,
        source_index={"source_transaction_receipts": refs},
        expected_run_id=RUN_ID,
        expected_terminal_checkpoint_id="checkpoint_6",
        expected_terminal_checkpoint_event_id="save:checkpoint_6:007",
        expected_compiled_model_sha256=COMPILED,
        expected_scene_realization_sha256=REALIZATION,
        expected_producers=AUXILIARY_PRODUCERS,
        physical_transactions=9,
    )
    if require_valid:
        assert result.valid
        return dict(result.ledger or {})
    return result


def test_five_drop_terminal_requires_contiguous_transaction_coverage(tmp_path: Path) -> None:
    ledger = _drop_ledger(tmp_path)
    manifest = _receipt(
        tmp_path, status="CONSECUTIVE_PAYLOAD_DROP_LIMIT", drops=ledger, physical=9,
    )
    assert _validate(tmp_path, manifest, "CONSECUTIVE_PAYLOAD_DROP_LIMIT", 9).valid
    ledger["transactions"][1]["counter_after"] = 0  # type: ignore[index]
    manifest = _receipt(
        tmp_path / "reset", status="CONSECUTIVE_PAYLOAD_DROP_LIMIT", drops=ledger, physical=9,
    )
    result = _validate(tmp_path / "reset", manifest, "CONSECUTIVE_PAYLOAD_DROP_LIMIT", 9)
    assert "PAYLOAD_DROP_LEDGER_NOT_RECOMPUTED_FROM_RAW_RECEIPTS" in result.invalid_reasons


def test_claimed_drop_is_rejected_when_raw_pose_and_contact_do_not_drop(
    tmp_path: Path,
) -> None:
    result = _drop_ledger(
        tmp_path,
        physical_z_m=0.80,
        floor_distance_m=0.01,
        require_valid=False,
    )
    assert not result.valid
    assert any("FORGED_PAYLOAD_DROP_CLASSIFICATION" in item for item in result.invalid_reasons)
    assert any("RAW_PHYSICS_NOT_PAYLOAD_DROP" in item for item in result.invalid_reasons)


def test_drop_samples_require_capability_bound_producer_and_recomputed_support(
    tmp_path: Path,
) -> None:
    producer_result = _drop_ledger(
        tmp_path / "producer",
        forged_physical_producer=True,
        require_valid=False,
    )
    assert any(
        "PRODUCER_NOT_CAPABILITY_BOUND" in item
        for item in producer_result.invalid_reasons
    )
    support_result = _drop_ledger(
        tmp_path / "support",
        declared_total_support_force_n=1.0,
        require_valid=False,
    )
    assert any(
        "TOTAL_SUPPORT_FORCE_MISMATCH" in item
        for item in support_result.invalid_reasons
    )
