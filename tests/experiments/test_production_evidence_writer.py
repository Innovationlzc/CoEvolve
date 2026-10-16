from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from coevolve_experiments.live.bridge import LIVE_RESULT_BINDING_KIND
from coevolve_experiments.live.capability import RUNNER_RESULT_CONTRACT_SCHEMA_VERSION
from coevolve_experiments.production import (
    EvidenceProducer,
    FoundationModelCall,
    FoundationModelTransportAttempt,
    ManifestArtifactPin,
    PayloadDropEvidence,
    ProductionArtifactRef,
    ProductionChannelBinding,
    ProductionEvidenceWriteError,
    ProductionInjectionBundle,
    ProductionRunAccounting,
    REQUIRED_PRODUCTION_CHANNELS,
    SystemStallCycleEvidence,
    SystemStallEvidence,
    TransactionAccountingEvidence,
    write_immutable_raw_evidence_artifact,
    write_frozen_phase_plan_artifact,
    write_production_result_evidence,
)
from coevolve_experiments.reporting.usage_evidence import (
    FM_USAGE_LEDGER_SCHEMA_VERSION,
)
from coevolve_experiments.production.evidence_writer import (
    METRICS_FILENAME,
    PAYLOAD_DROP_LEDGER_RELPATH,
    RUN_MANIFEST_FILENAME,
    SYSTEM_STALL_LEDGER_RELPATH,
    TERMINAL_SAMPLE_TRACE_RELPATH,
    V83_NATIVE_MEMORY_AUDIT_PIN,
    V83_NATIVE_MEMORY_AUDIT_RELPATH,
)
from coevolve_experiments.reporting.terminal_evidence import (
    MUJOCO_TERMINAL_SAMPLE_TRACE_KIND,
    MUJOCO_TERMINAL_STATE_KIND,
)
from coevolve_experiments.reporting.terminal_aux_evidence import (
    FROZEN_PAYLOAD_DROP_SAFETY_PROFILE,
    PAYLOAD_DROP_EVALUATION_KIND,
    PAYLOAD_DROP_EXECUTION_TRACE_KIND,
    PAYLOAD_DROP_POLICY_KIND,
    PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
    PHYSICAL_TRANSACTION_RECEIPT_KIND,
    RAW_AUX_EVIDENCE_SCHEMA_VERSION,
    SYSTEM_STALL_ACTION_SPACE_KIND,
    SYSTEM_STALL_CHECKPOINT_KIND,
    SYSTEM_STALL_HOST_PREFLIGHT_KIND,
    SYSTEM_STALL_MODEL_OUTPUT_KIND,
    SYSTEM_STALL_STATIC_VALIDATION_KIND,
    capability_bound_auxiliary_producer_allowlist,
)
from coevolve_experiments.reporting.transaction_accounting import (
    ACCEPTED_MANIFEST_PIN,
    ACCEPTED_MANIFEST_RELPATH,
    FROZEN_PHASE_PLAN_PIN,
    FROZEN_PHASE_PLAN_RELPATH,
    MUTATION_COMMIT_PRODUCER_ID,
    MUTATION_COMMIT_RECEIPT_KIND,
    MUTATION_COMMIT_RECEIPT_SCHEMA_VERSION,
    TRANSACTION_ACCOUNTING_LEDGER_PIN,
    TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
    TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION,
)
from coevolve_experiments.serialization import sha256_json


TARGET = (-1.5, 7.6, 0.3)
MODEL = "gpt-5.4-mini-2026-03-17"
ENTRYPOINT_SHA256 = sha256(b"entrypoint").hexdigest()
AUXILIARY_PRODUCERS = capability_bound_auxiliary_producer_allowlist(
    ENTRYPOINT_SHA256
)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _canonical_digest(value: object) -> str:
    raw = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _raw_producer(family: str) -> dict[str, str]:
    return dict(AUXILIARY_PRODUCERS[family])


def _run_root(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "formal"
        / "table2_method_comparison"
        / "coevolve_full"
        / "seed_200"
    ).resolve()


def _bundle(root: Path) -> tuple[ProductionInjectionBundle, dict[str, object]]:
    run = {
        "run_id": "table2_method_comparison__coevolve_full__seed_200",
        "table_id": "table2_method_comparison",
        "seed": 200,
        "method_variant": "coevolve_full",
        "scene_id": "base",
        "memory_condition": "empty",
        "memory_policy": "online_from_empty",
        "target_backbone": MODEL,
        "provider_route_id": "provider_a.openai-compatible.chat-completions.v1",
        "provider_route_fingerprint": _digest("provider-route"),
        "physical_budget": 80,
        "condition_fingerprint": _digest("condition"),
        "invariant_fingerprint": _digest("invariant"),
        "base_method_commit": "a" * 40,
        "protocol_id": "post-table1-test",
        "output_relpath": "table2_method_comparison/coevolve_full/seed_200",
        "scene_fingerprint": _digest("scene"),
        "terminal_fingerprint": _digest("terminal"),
        "observation_contract_fingerprint": _digest("observation"),
        "task_instruction_fingerprint": _digest("instruction"),
        "shared_prompt_contract_fingerprint": _digest("shared-prompt"),
        "decision_prompt_fingerprint": _digest("decision-prompt"),
        "skill_api_fingerprint": _digest("skill-api"),
        "full_method_compatibility_fingerprint": _digest("full-method"),
        "strategy_bank": None,
    }
    launch_config_sha256 = _digest("launch-config")
    launch_fingerprint = _digest("launch")
    binding = {
        "kind": LIVE_RESULT_BINDING_KIND,
        "schema_version": RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
        "run_id": run["run_id"],
        "run_spec_sha256": sha256_json(run),
        "launch_fingerprint": launch_fingerprint,
        "config_fingerprint": _digest("config"),
        "launch_config_sha256": launch_config_sha256,
        "plan_sha256": _digest("plan"),
        "runner_capability_sha256": _digest("capability"),
        "base_method_commit": run["base_method_commit"],
        "runner_integration_commit": "b" * 40,
        "entrypoint_sha256": _digest("entrypoint"),
        "full_method_compatibility_fingerprint": run[
            "full_method_compatibility_fingerprint"
        ],
    }
    payloads = {
        "term1_simulator_asset": {
            "scene_id": "base",
            "execution_composite_sha256": _digest("scene-realization"),
            "execution_evidence_fingerprint": _digest("scene-evidence"),
            "compiled_model_sha256": _digest("compiled-model"),
            "target": {
                "center": {"x": TARGET[0], "y": TARGET[1], "z": TARGET[2]},
                "radius_m": 0.30,
            },
        },
        "host_clearance_geometry": {"scene_id": "base"},
        "vlm_scene_prior": {"scene_id": "base"},
        "route_and_progress_gates": {"scene_id": "base"},
        "formal_run_manifest": {
            "run_spec": run,
            "plan_sha256": binding["plan_sha256"],
            "launch_config_sha256": launch_config_sha256,
            "result_binding": binding,
        },
        "method_gates": {"method_variant": "coevolve_full"},
        "memory_gates": {"memory_condition": "empty"},
        "provider_route": {
            "route_id": run["provider_route_id"],
            "route_fingerprint": run["provider_route_fingerprint"],
            "provider": "provider_a",
            "target_model": MODEL,
        },
        "formal_output": {
            "results_root": str(root.parents[2]),
            "output_dir": str(root),
            "output_relpath": run["output_relpath"],
            "seed": 200,
            "physical_budget": 80,
            "result_contract_schema_version": RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
            "terminal_evidence_schema_version": 3,
            "fm_usage_ledger_schema_version": FM_USAGE_LEDGER_SCHEMA_VERSION,
            "transaction_accounting_ledger_schema_version": (
                TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION
            ),
            "required_result_artifact_pins": {
                "native_memory_audit": V83_NATIVE_MEMORY_AUDIT_PIN,
                "terminal_evidence": "terminal_verifier_evidence_sha256",
                "fm_usage_ledger": "fm_usage_ledger_sha256",
                "transaction_accounting_ledger": TRANSACTION_ACCOUNTING_LEDGER_PIN,
                "accepted_episode_manifest": ACCEPTED_MANIFEST_PIN,
                "frozen_phase_plan": FROZEN_PHASE_PLAN_PIN,
            },
        },
    }
    channels = tuple(
        ProductionChannelBinding.create(name, payloads[name])
        for name in REQUIRED_PRODUCTION_CHANNELS
    )
    return (
        ProductionInjectionBundle.create(
            run_id=str(run["run_id"]),
            launch_config_path=str(root.parent / "launch.json"),
            launch_config_sha256=launch_config_sha256,
            launch_fingerprint=launch_fingerprint,
            channels=channels,
        ),
        run,
    )


def _prepare_pinned_memory(root: Path) -> None:
    fixed = {
        V83_NATIVE_MEMORY_AUDIT_RELPATH: {"kind": "test-native-audit"},
        "memory/from_scratch_memory_bootstrap.json": {"kind": "test-bootstrap"},
        "memory/from_scratch_memory_runtime_audit.json": {"kind": "test-runtime"},
    }
    for relative, payload in fixed.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _simulator(run: dict[str, object]) -> dict[str, object]:
    return {
        "name": "MuJoCo",
        "version": "3.3.5",
        "compiled_model_sha256": _digest("compiled-model"),
        "scene_realization_sha256": _digest("scene-realization"),
        "scene_fingerprint": run["scene_fingerprint"],
        "scene_execution_evidence_fingerprint": _digest("scene-evidence"),
    }


def _trace(run: dict[str, object], *, xy_error: float = 0.10) -> dict[str, object]:
    samples = []
    for index, time_s in enumerate((0.0, 1.5, 3.0)):
        samples.append(
            {
                "sample_index": index,
                "time_s": time_s,
                "payload_released": True,
                "payload_pose": {
                    "position_xyz_m": [TARGET[0] + xy_error, TARGET[1], 0.65],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "robot_poses": {
                    "front": {
                        "position_xyz_m": [TARGET[0] + 1.2, TARGET[1], 0.75],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    "back": {
                        "position_xyz_m": [TARGET[0] - 1.2, TARGET[1], 0.75],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
                "contacts": [
                    {
                        "geom1_id": 1,
                        "geom2_id": 2,
                        "geom1_role": "payload",
                        "geom2_role": "target",
                        "distance_m": -0.001,
                    }
                ],
            }
        )
    return {
        "kind": MUJOCO_TERMINAL_SAMPLE_TRACE_KIND,
        "schema_version": 2,
        "run_id": run["run_id"],
        "seed": run["seed"],
        "checkpoint_id": "checkpoint_6",
        "checkpoint_event_id": "save:checkpoint_6:007",
        "simulator": _simulator(run),
        "target": {"center_xyz_m": list(TARGET), "scale_xyz_m": [0.6, 0.6, 0.6]},
        "payload_geometry": {"half_height_m": 0.05},
        "sample_window_s": 3.0,
        "samples": samples,
        "lifecycle": {
            "release_completed": True,
            "retreat_completed": True,
            "stand_recovered": True,
        },
    }


def _state(run: dict[str, object]) -> dict[str, object]:
    return {
        "kind": MUJOCO_TERMINAL_STATE_KIND,
        "schema_version": 3,
        "run_id": run["run_id"],
        "seed": run["seed"],
        "checkpoint_id": "checkpoint_6",
        "checkpoint_event_id": "save:checkpoint_6:007",
        "simulator": _simulator(run),
        "sample_window_s": 3.0,
        "sample_count": 3,
        "payload": {
            "released": True,
            "target_contact_all_samples": True,
            "floor_contact_samples": 0,
            "wall_contact_samples": 0,
            "robot_contact_samples": 0,
            "xy_error_m": 0.10,
            "z_error_m": 0.0,
            "tilt_deg": 0.0,
        },
        "robots": {
            "front": {
                "upright": True,
                "pelvis_height_m": 0.75,
                "distance_to_target_m": 1.2,
            },
            "back": {
                "upright": True,
                "pelvis_height_m": 0.75,
                "distance_to_target_m": 1.2,
            },
        },
        "lifecycle": {
            "release_completed": True,
            "retreat_completed": True,
            "stand_recovered": True,
        },
    }


def _non_success_terminal(
    run: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    trace = _trace(run)
    sample = trace["samples"][0]  # type: ignore[index]
    sample["payload_released"] = False
    sample["contacts"] = []
    trace["samples"] = [sample]
    trace["sample_window_s"] = 0.0
    trace["lifecycle"] = {
        "release_completed": False,
        "retreat_completed": False,
        "stand_recovered": False,
    }
    state = _state(run)
    state["sample_window_s"] = 0.0
    state["sample_count"] = 1
    state["payload"] = {
        "released": False,
        "target_contact_all_samples": False,
        "floor_contact_samples": 0,
        "wall_contact_samples": 0,
        "robot_contact_samples": 0,
        "xy_error_m": 0.10,
        "z_error_m": 0.0,
        "tilt_deg": 0.0,
    }
    state["lifecycle"] = dict(trace["lifecycle"])
    return trace, state


def _accounting() -> ProductionRunAccounting:
    return ProductionRunAccounting(
        physical_transactions=7,
        accepted_transactions=6,
        rejected_transactions=1,
        accepted_setup_transactions=1,
        accepted_progress_transactions=5,
        elapsed_s=12.5,
        phase_completed=6,
        phase_total=6,
        failure_counts={"STATIC_REJECT": 1},
        mutation_counts={"strategy_mutation": 2},
    )


def _drop_accounting() -> ProductionRunAccounting:
    return ProductionRunAccounting(
        physical_transactions=7,
        accepted_transactions=2,
        rejected_transactions=5,
        accepted_setup_transactions=1,
        accepted_progress_transactions=1,
        elapsed_s=12.5,
        phase_completed=3,
        phase_total=6,
        failure_counts={"payload_drop": 5},
        mutation_counts={"strategy_mutation": 2},
    )


def _calls() -> tuple[FoundationModelCall, ...]:
    return (
        FoundationModelCall(
            request_id="request-1",
            role="policy_generator",
            modality="text",
            requested_model=MODEL,
            resolved_model=MODEL,
            response_model=MODEL,
            http_status=200,
            input_tokens=10,
            output_tokens=2,
            billing_amount_usd=0.01,
        ),
        FoundationModelCall(
            request_id="request-2",
            role="reliable_feedback",
            modality="vision",
            requested_model=MODEL,
            resolved_model=MODEL,
            response_model=MODEL,
            http_status=200,
            input_tokens=20,
            output_tokens=3,
            billing_amount_usd=0.02,
        ),
    )


def _stall_raw_evidence(
    root: Path,
    run: dict[str, object],
    trace: dict[str, object],
    state: dict[str, object],
    *,
    checkpoint_producer_source_sha256: str | None = None,
) -> SystemStallEvidence:
    run_id = str(run["run_id"])
    checkpoint_id = str(trace["checkpoint_id"])
    checkpoint_event_id = str(trace["checkpoint_event_id"])
    common = {
        "schema_version": RAW_AUX_EVIDENCE_SCHEMA_VERSION,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_event_id": checkpoint_event_id,
    }
    checkpoint = write_immutable_raw_evidence_artifact(
        root,
        "evidence/system_stall/checkpoint_validity.json",
        {
            "kind": SYSTEM_STALL_CHECKPOINT_KIND,
            "producer": {
                **_raw_producer("stall.checkpoint"),
                **(
                    {"source_sha256": checkpoint_producer_source_sha256}
                    if checkpoint_producer_source_sha256 is not None
                    else {}
                ),
            },
            **common,
            "terminal_state_sha256": _canonical_digest(state),
            "terminal_sample_trace_sha256": _canonical_digest(trace),
            "sample_window_s": state["sample_window_s"],
            "sample_count": state["sample_count"],
            "payload": {
                "floor_contact_samples": state["payload"]["floor_contact_samples"],  # type: ignore[index]
                "wall_contact_samples": state["payload"]["wall_contact_samples"],  # type: ignore[index]
            },
            "robots": {
                robot_id: {
                    "upright": state["robots"][robot_id]["upright"],  # type: ignore[index]
                    "pelvis_height_m": state["robots"][robot_id]["pelvis_height_m"],  # type: ignore[index]
                }
                for robot_id in ("front", "back")
            },
        },
    )
    action = write_immutable_raw_evidence_artifact(
        root,
        "evidence/system_stall/host_action_space.json",
        {
            "kind": SYSTEM_STALL_ACTION_SPACE_KIND,
            "producer": _raw_producer("stall.host_action_space"),
            **common,
            "checkpoint_validity_sha256": checkpoint.sha256,
            "lifecycle_state": "carry",
            "admissible_action_families": ["sync:coordinated_linear_transport"],
            "allowed_primitives": ["coordinated_linear_transport"],
            "allowed_coordination_modes": ["sync"],
        },
    )
    cycles = []
    previous_preflight_sha256 = None
    for index in range(1, 4):
        cycle_common = {**common, "cycle": index}
        model = write_immutable_raw_evidence_artifact(
            root,
            f"evidence/system_stall/cycle_{index:03d}_model_output.json",
            {
                "kind": SYSTEM_STALL_MODEL_OUTPUT_KIND,
                "producer": _raw_producer("stall.model_output"),
                **cycle_common,
                "checkpoint_validity_sha256": checkpoint.sha256,
                "host_action_space_sha256": action.sha256,
                "previous_cycle_preflight_sha256": previous_preflight_sha256,
                "request_id": f"stall-request-{index}",
                "usage_call_ordinal": index,
                "role": "policy_generator",
                "model": MODEL,
                "finish_reason": "stop",
                "content": "{}",
            },
        )
        static = write_immutable_raw_evidence_artifact(
            root,
            f"evidence/system_stall/cycle_{index:03d}_static_validation.json",
            {
                "kind": SYSTEM_STALL_STATIC_VALIDATION_KIND,
                "producer": _raw_producer("stall.static_validation"),
                **cycle_common,
                "model_output_sha256": model.sha256,
                "parser_passed": True,
                "script_present": False,
                "static_validation_passed": False,
                "failure_codes": ["NO_SCRIPT"],
            },
        )
        preflight = write_immutable_raw_evidence_artifact(
            root,
            f"evidence/system_stall/cycle_{index:03d}_host_preflight.json",
            {
                "kind": SYSTEM_STALL_HOST_PREFLIGHT_KIND,
                "producer": _raw_producer("stall.host_preflight"),
                **cycle_common,
                "static_validation_sha256": static.sha256,
                "host_action_space_sha256": action.sha256,
                "evaluated": False,
                "passed": False,
                "skipped_reason": "STATIC_VALIDATION_FAILED",
                "failure_codes": ["STATIC_VALIDATION_FAILED"],
                "executable_child_produced": False,
                "physical_dispatch": False,
            },
        )
        cycles.append(SystemStallCycleEvidence(model, static, preflight))
        previous_preflight_sha256 = preflight.sha256
    return SystemStallEvidence(checkpoint, action, tuple(cycles))


def _write_memory_artifact(
    root: Path,
    relative_path: str,
    payload: object,
) -> ProductionArtifactRef:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return ProductionArtifactRef(relative_path, sha256(raw).hexdigest(), len(raw))


def _transaction_specs(mode: str) -> list[dict[str, object]]:
    phases = [f"phase_{index}" for index in range(1, 7)]
    if mode == "success":
        return [
            {
                "before": "checkpoint_0",
                "after": "checkpoint_1",
                "outcome": "ACCEPT_SETUP",
                "failure": "none",
                "phases_before": [],
                "phases_after": [],
            },
            {
                "before": "checkpoint_1",
                "after": "checkpoint_2",
                "outcome": "ACCEPT_PROGRESS",
                "failure": "none",
                "phases_before": [],
                "phases_after": phases[:2],
            },
            {
                "before": "checkpoint_2",
                "after": "checkpoint_2",
                "outcome": "REJECT",
                "failure": "STATIC_REJECT",
                "phases_before": phases[:2],
                "phases_after": phases[:2],
            },
            *[
                {
                    "before": f"checkpoint_{transaction - 2}",
                    "after": f"checkpoint_{transaction - 1}",
                    "outcome": "ACCEPT_PROGRESS",
                    "failure": "none",
                    "phases_before": phases[: transaction - 2],
                    "phases_after": phases[: transaction - 1],
                }
                for transaction in range(4, 7)
            ],
            {
                "before": "checkpoint_5",
                "after": "checkpoint_6",
                "outcome": "TASK_COMPLETE",
                "failure": "none",
                "phases_before": phases[:5],
                "phases_after": phases,
            },
        ]
    if mode == "stall":
        return [
            {
                "before": "checkpoint_0",
                "after": "checkpoint_1",
                "outcome": "ACCEPT_SETUP",
                "failure": "none",
                "phases_before": [],
                "phases_after": [],
            },
            {
                "before": "checkpoint_1",
                "after": "checkpoint_2",
                "outcome": "ACCEPT_PROGRESS",
                "failure": "none",
                "phases_before": [],
                "phases_after": phases[:1],
            },
            {
                "before": "checkpoint_2",
                "after": "checkpoint_2",
                "outcome": "REJECT",
                "failure": "STATIC_REJECT",
                "phases_before": phases[:1],
                "phases_after": phases[:1],
            },
            *[
                {
                    "before": f"checkpoint_{transaction - 2}",
                    "after": f"checkpoint_{transaction - 1}",
                    "outcome": "ACCEPT_PROGRESS",
                    "failure": "none",
                    "phases_before": phases[: transaction - 3],
                    "phases_after": phases[: transaction - 2],
                }
                for transaction in range(4, 8)
            ],
        ]
    if mode == "drop":
        carried_phases = phases[:3]
        return [
            {
                "before": "checkpoint_0",
                "after": "checkpoint_1",
                "outcome": "ACCEPT_SETUP",
                "failure": "none",
                "phases_before": [],
                "phases_after": [],
            },
            {
                "before": "checkpoint_1",
                "after": "checkpoint_6",
                "outcome": "ACCEPT_PROGRESS",
                "failure": "none",
                "phases_before": [],
                "phases_after": carried_phases,
            },
            *[
                {
                    "before": "checkpoint_6",
                    "after": "checkpoint_6",
                    "outcome": "REJECT",
                    "failure": "payload_drop",
                    "phases_before": carried_phases,
                    "phases_after": carried_phases,
                    "drop": True,
                }
                for _ in range(5)
            ],
        ]
    raise AssertionError(f"unsupported fixture mode: {mode}")


def _physical_samples(
    run: dict[str, object],
    *,
    transaction: int,
    checkpoint_id: str,
    checkpoint_event_id: str,
    payload_drop: bool,
) -> dict[str, object]:
    contacts: list[dict[str, object]] = []
    payload_z = 0.80
    if payload_drop:
        payload_z = 0.59
        contacts.append(
            {
                "geom1_id": 11,
                "geom2_id": 12,
                "geom1_role": "payload",
                "geom2_role": "floor",
                "geom1_name": "long_box_collision",
                "geom2_name": "manual_floor",
                "body1_name": "long_box",
                "body2_name": "manual_floor",
                "distance_m": -0.001,
                "normal_force_n": 5.0,
            }
        )
    return {
        "kind": PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
        "schema_version": RAW_AUX_EVIDENCE_SCHEMA_VERSION,
        "producer": _raw_producer("payload_drop.physical_samples"),
        "run_id": run["run_id"],
        "physical_transaction": transaction,
        "checkpoint_id": checkpoint_id,
        "checkpoint_event_id": checkpoint_event_id,
        "simulator": {
            "name": "MuJoCo",
            "version": "3.3.5",
            "compiled_model_sha256": _digest("compiled-model"),
            "scene_realization_sha256": _digest("scene-realization"),
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
                "time_s": float(transaction),
                "payload_active": True,
                "payload_pose": {
                    "position_xyz_m": [0.0, 0.0, payload_z],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "contacts": contacts,
                "support": {
                    "available": True,
                    "supporting_robots": [],
                    "normal_force_n_by_robot": {"front": 0.0, "back": 0.0},
                    "total_normal_force_n": 0.0,
                },
            }
        ],
    }


def _transaction_accounting_evidence(
    root: Path,
    run: dict[str, object],
    *,
    mode: str = "success",
    broken_chain_at: int | None = None,
    evaluation_producer_change_at: int | None = None,
) -> TransactionAccountingEvidence:
    phase_ids = tuple(f"phase_{index}" for index in range(1, 7))
    phase_plan = write_frozen_phase_plan_artifact(
        root,
        run_id=str(run["run_id"]),
        phase_ids=phase_ids,
    )
    specifications = _transaction_specs(mode)
    receipts: list[ProductionArtifactRef] = []
    accepted_steps: list[dict[str, object]] = []
    previous_receipt_sha256: str | None = None
    previous_event = "restore:checkpoint_0:000"
    drop_counter = 0
    for transaction, specification in enumerate(specifications, start=1):
        checkpoint_before = str(specification["before"])
        checkpoint_after = str(specification["after"])
        event_before = previous_event
        event_after = (
            "save:checkpoint_6:007"
            if transaction == len(specifications)
            else f"checkpoint-event:{transaction:03d}"
        )
        outcome = str(specification["outcome"])
        accepted = outcome != "REJECT"
        payload_drop = bool(specification.get("drop", False))
        failure = str(specification["failure"])
        common = {
            "schema_version": RAW_AUX_EVIDENCE_SCHEMA_VERSION,
            "run_id": run["run_id"],
            "physical_transaction": transaction,
        }
        script = f"execute_transaction_{transaction}()"
        policy = write_immutable_raw_evidence_artifact(
            root,
            f"evidence/transactions/transaction_{transaction:03d}_policy.json",
            {
                "kind": PAYLOAD_DROP_POLICY_KIND,
                "producer": _raw_producer("payload_drop.policy"),
                **common,
                "checkpoint_id": checkpoint_before,
                "checkpoint_event_id": event_before,
                "action_script": script,
                "action_script_sha256": sha256(script.encode("utf-8")).hexdigest(),
            },
        )
        evaluation = write_immutable_raw_evidence_artifact(
            root,
            f"evidence/transactions/transaction_{transaction:03d}_evaluation.json",
            {
                "kind": PAYLOAD_DROP_EVALUATION_KIND,
                "producer": {
                    **_raw_producer("payload_drop.evaluation"),
                    **(
                        {"source_sha256": _digest("alternate-entrypoint")}
                        if transaction == evaluation_producer_change_at
                        else {}
                    ),
                },
                **common,
                "checkpoint_id": checkpoint_before,
                "checkpoint_event_id": event_before,
                "accepted": accepted,
                "failure_type": failure,
                "transition_outcome": outcome,
                "completed_phase_ids_before": specification["phases_before"],
                "completed_phase_ids_after": specification["phases_after"],
            },
        )
        execution = write_immutable_raw_evidence_artifact(
            root,
            f"evidence/transactions/transaction_{transaction:03d}_execution_trace.json",
            {
                "kind": PAYLOAD_DROP_EXECUTION_TRACE_KIND,
                "producer": _raw_producer("payload_drop.execution_trace"),
                **common,
                "checkpoint_id_before": checkpoint_before,
                "checkpoint_event_id_before": event_before,
                "checkpoint_id_after": checkpoint_after,
                "checkpoint_event_id_after": event_after,
                "physically_dispatched": True,
                "rollback_performed": not accepted,
                "physics_step_count": 50,
                "dispatch_started_s": float(transaction),
                "dispatch_finished_s": float(transaction) + 0.5,
            },
        )
        physical_samples = write_immutable_raw_evidence_artifact(
            root,
            f"evidence/transactions/transaction_{transaction:03d}_physical_samples.json",
            _physical_samples(
                run,
                transaction=transaction,
                checkpoint_id=checkpoint_before,
                checkpoint_event_id=event_before,
                payload_drop=payload_drop,
            ),
        )
        counter_before = drop_counter
        drop_counter = drop_counter + 1 if payload_drop else 0
        receipt = write_immutable_raw_evidence_artifact(
            root,
            f"evidence/transactions/transaction_{transaction:03d}.json",
            {
                "kind": PHYSICAL_TRANSACTION_RECEIPT_KIND,
                "producer": _raw_producer("payload_drop.transaction_receipt"),
                **common,
                "checkpoint_id_before": checkpoint_before,
                "checkpoint_event_id_before": event_before,
                "checkpoint_id_after": checkpoint_after,
                "checkpoint_event_id_after": event_after,
                "physically_dispatched": True,
                "verifier": {"accepted": accepted, "failure_type": failure},
                "rollback_performed": not accepted,
                "drop_counter_before": counter_before,
                "drop_counter_after": drop_counter,
                "policy": policy.as_dict(),
                "evaluation": evaluation.as_dict(),
                "execution_trace": execution.as_dict(),
                "physical_samples": physical_samples.as_dict(),
                "previous_receipt_sha256": (
                    _digest("broken-chain")
                    if transaction == broken_chain_at
                    else previous_receipt_sha256
                ),
            },
        )
        receipts.append(receipt)
        previous_receipt_sha256 = receipt.sha256
        previous_event = event_after
        if accepted:
            accepted_steps.append(
                {
                    "index": len(accepted_steps) + 1,
                    "attempt": transaction,
                    "script_id": f"transaction_{transaction:03d}",
                    "checkpoint_before": checkpoint_before,
                    "checkpoint_after": checkpoint_after,
                    "transition_outcome": outcome,
                    "robot_scripts": {"front": {}, "back": {}},
                }
            )

    accepted_manifest = _write_memory_artifact(
        root,
        ACCEPTED_MANIFEST_RELPATH,
        {
            "kind": "accepted_checkpoint_episode",
            "schema_version": 1,
            "initial_checkpoint": str(specifications[0]["before"]),
            "latest_checkpoint": str(specifications[-1]["after"]),
            "initial_state": {},
            "steps": accepted_steps,
        },
    )
    rejected_transactions = [
        (transaction, specification)
        for transaction, specification in enumerate(specifications, start=1)
        if specification["outcome"] == "REJECT"
    ]
    mutation_refs: list[ProductionArtifactRef] = []
    previous_mutation_sha256: str | None = None
    for ordinal in range(1, 3):
        source_transaction, source_specification = rejected_transactions[
            min(ordinal - 1, len(rejected_transactions) - 1)
        ]
        mutation = write_immutable_raw_evidence_artifact(
            root,
            f"evidence/mutations/mutation_commit_{ordinal:03d}.json",
            {
                "kind": MUTATION_COMMIT_RECEIPT_KIND,
                "schema_version": MUTATION_COMMIT_RECEIPT_SCHEMA_VERSION,
                "producer": {
                    "id": MUTATION_COMMIT_PRODUCER_ID,
                    "source_sha256": ENTRYPOINT_SHA256,
                },
                "run_id": run["run_id"],
                "mutation_ordinal": ordinal,
                "source_rejected_physical_transaction": source_transaction,
                "checkpoint_id": source_specification["after"],
                "mutation_level": "strategy_mutation",
                "committed_mutation_sha256": _digest(f"mutation-{ordinal}"),
                "previous_commit_receipt_sha256": previous_mutation_sha256,
            },
        )
        mutation_refs.append(mutation)
        previous_mutation_sha256 = mutation.sha256
    return TransactionAccountingEvidence(
        transaction_receipts=tuple(receipts),
        frozen_phase_plan=phase_plan,
        accepted_episode_manifest=accepted_manifest,
        mutation_commit_receipts=tuple(mutation_refs),
    )


def _write_success(root: Path, **overrides: object):
    bundle, run = _bundle(root)
    selected_bundle = overrides.pop("bundle", bundle)
    transaction_accounting_evidence = overrides.pop(
        "transaction_accounting_evidence",
        None,
    )
    if transaction_accounting_evidence is None:
        transaction_accounting_evidence = _transaction_accounting_evidence(
            root,
            run,
        )
    arguments = {
        "terminal_status": "SUCCESS",
        "terminal_sample_trace": _trace(run),
        "terminal_state": _state(run),
        "accounting": _accounting(),
        "transaction_accounting_evidence": transaction_accounting_evidence,
        "fm_calls": _calls(),
        "terminal_producer": EvidenceProducer("test.terminal", "c" * 64),
        "fm_producer": EvidenceProducer("test.fm", "d" * 64),
    }
    arguments.update(overrides)
    return write_production_result_evidence(selected_bundle, **arguments)


def test_writer_publishes_validator_checked_bound_result_and_computes_totals(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)

    result = _write_success(root)

    assert result.terminal_validation.valid
    assert result.usage_validation.valid
    assert result.accounting_validation.valid
    assert result.usage_totals == {
        "llm_calls": 1,
        "vlm_calls": 1,
        "successful_responses": 2,
        "failed_requests": 0,
        "transport_attempts": 2,
        "input_tokens": 30,
        "output_tokens": 5,
        "billing_records": 2,
        "billing_amount_usd": pytest.approx(0.03),
        "billing_completeness": "complete",
    }
    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    ledger = json.loads(result.fm_usage_ledger_path.read_text(encoding="utf-8"))
    accounting_ledger = json.loads(
        result.transaction_accounting_ledger_path.read_text(encoding="utf-8")
    )
    assert manifest["live_launch"] == metrics["live_launch"]
    assert manifest["result_artifact_hashes"] == dict(result.result_artifact_hashes)
    assert V83_NATIVE_MEMORY_AUDIT_PIN in manifest["result_artifact_hashes"]
    for pin in (
        TRANSACTION_ACCOUNTING_LEDGER_PIN,
        ACCEPTED_MANIFEST_PIN,
        FROZEN_PHASE_PLAN_PIN,
    ):
        assert pin in manifest["result_artifact_hashes"]
    assert accounting_ledger["physical_transactions"] == 7
    assert accounting_ledger["accepted_transactions"] == 6
    assert accounting_ledger["rejected_transactions"] == 1
    assert accounting_ledger["accepted_setup_transactions"] == 1
    assert accounting_ledger["accepted_progress_transactions"] == 5
    assert accounting_ledger["phase_completion"]["completed"] == 6
    assert accounting_ledger["failure_counts"] == {"STATIC_REJECT": 1}
    assert accounting_ledger["mutation_counts"] == {"strategy_mutation": 2}
    assert [item["source_rejected_physical_transaction"] for item in accounting_ledger["mutations"]] == [3, 3]
    assert manifest["result_artifact_hashes"][ACCEPTED_MANIFEST_PIN] == sha256(
        (root / ACCEPTED_MANIFEST_RELPATH).read_bytes()
    ).hexdigest()
    assert manifest["result_artifact_hashes"][FROZEN_PHASE_PLAN_PIN] == sha256(
        (root / FROZEN_PHASE_PLAN_RELPATH).read_bytes()
    ).hexdigest()
    assert metrics["model"]["input_tokens"] == 30
    assert ledger["totals"] == {
        "llm_calls": 1,
        "vlm_calls": 1,
        "successful_responses": 2,
        "failed_requests": 0,
        "transport_attempts": 2,
        "input_tokens": 30,
        "output_tokens": 5,
        "billing_records": 2,
        "billing_amount_usd": pytest.approx(0.03),
        "billing_completeness": "complete",
    }


def test_writer_separates_logical_calls_retries_and_failed_requests(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)
    successful = replace(
        _calls()[0],
        transport_attempts=(
            FoundationModelTransportAttempt(
                outcome="http_response",
                http_status=429,
            ),
            FoundationModelTransportAttempt(
                outcome="http_response",
                http_status=200,
            ),
        ),
    )
    failed = FoundationModelCall(
        request_id="request-failed",
        role="reliable_feedback",
        modality="text",
        requested_model=MODEL,
        resolved_model=MODEL,
        response_model=None,
        http_status=None,
        input_tokens=None,
        output_tokens=None,
        billing_amount_usd=None,
        outcome="failure",
        transport_attempts=(
            FoundationModelTransportAttempt(
                outcome="transport_error",
                error_type="timeout",
            ),
        ),
    )

    result = _write_success(root, fm_calls=(successful, failed))

    assert result.usage_validation.valid
    assert not result.usage_validation.billing_complete
    assert result.usage_totals == {
        "llm_calls": 2,
        "vlm_calls": 0,
        "successful_responses": 1,
        "failed_requests": 1,
        "transport_attempts": 3,
        "input_tokens": 10,
        "output_tokens": 2,
        "billing_records": 1,
        "billing_amount_usd": None,
        "billing_completeness": "partial",
    }
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["model"]["billing_amount_usd"] is None
    assert metrics["model"]["billing_reconciliation_status"] == "partial"
    ledger = json.loads(result.fm_usage_ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["calls"][0]["transport_attempts"]) == 2
    assert ledger["calls"][1]["outcome"] == "failure"
    assert ledger["calls"][1]["response_model"] is None


def test_writer_publishes_unavailable_billing_as_na_not_zero(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)
    calls = tuple(replace(call, billing_amount_usd=None) for call in _calls())

    result = _write_success(root, fm_calls=calls)

    assert result.usage_validation.valid
    assert result.usage_validation.billing_reasons == ()
    assert result.usage_validation.billing_amount_usd is None
    assert result.usage_totals["billing_amount_usd"] is None
    assert result.usage_totals["billing_completeness"] == "unavailable"


def test_writer_rejects_caller_authored_totals_and_invalid_trace_without_output(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path / "forged-totals")
    _prepare_pinned_memory(root)
    bundle, run = _bundle(root)
    transaction_evidence = _transaction_accounting_evidence(root, run)
    forged = dict(_calls()[0].__dict__ if hasattr(_calls()[0], "__dict__") else {
        "request_id": "request-1",
        "role": "policy_generator",
        "modality": "text",
        "requested_model": MODEL,
        "resolved_model": MODEL,
        "response_model": MODEL,
        "http_status": 200,
        "input_tokens": 10,
        "output_tokens": 2,
        "billing_amount_usd": 0.01,
    })
    forged["totals"] = {"input_tokens": 0}
    with pytest.raises(ProductionEvidenceWriteError, match="aggregate totals"):
        write_production_result_evidence(
            bundle,
            terminal_status="SUCCESS",
            terminal_sample_trace=_trace(run),
            terminal_state=_state(run),
            accounting=_accounting(),
            transaction_accounting_evidence=transaction_evidence,
            fm_calls=(forged,),
            terminal_producer=EvidenceProducer("test.terminal", "c" * 64),
            fm_producer=EvidenceProducer("test.fm", "d" * 64),
        )
    assert not (root / RUN_MANIFEST_FILENAME).exists()

    invalid_root = _run_root(tmp_path / "invalid-trace")
    _prepare_pinned_memory(invalid_root)
    _invalid_bundle, invalid_run = _bundle(invalid_root)
    with pytest.raises(ProductionEvidenceWriteError, match="existing validators"):
        _write_success(
            invalid_root,
            terminal_sample_trace=_trace(invalid_run, xy_error=0.50),
        )
    assert not (invalid_root / RUN_MANIFEST_FILENAME).exists()
    assert not (invalid_root / TERMINAL_SAMPLE_TRACE_RELPATH).exists()


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("physical_transactions", 8),
        ("accepted_transactions", 5),
        ("rejected_transactions", 2),
        ("accepted_setup_transactions", 0),
        ("accepted_progress_transactions", 4),
        ("phase_completed", 5),
        ("phase_total", 7),
        ("failure_counts", {}),
        ("mutation_counts", {}),
    ),
)
def test_writer_rejects_each_forged_accounting_claim_before_manifest(
    tmp_path: Path,
    field: str,
    forged_value: object,
) -> None:
    root = _run_root(tmp_path / field)
    _prepare_pinned_memory(root)
    forged = replace(_accounting(), **{field: forged_value})

    with pytest.raises(
        ProductionEvidenceWriteError,
        match="transaction accounting evidence is invalid",
    ):
        _write_success(root, accounting=forged)

    assert not (root / RUN_MANIFEST_FILENAME).exists()


def test_writer_refuses_overwrite_before_publishing_any_evidence(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)
    sentinel = root / METRICS_FILENAME
    sentinel.write_text("do-not-replace\n", encoding="utf-8")

    with pytest.raises(ProductionEvidenceWriteError, match="refusing to overwrite"):
        _write_success(root)

    assert sentinel.read_text(encoding="utf-8") == "do-not-replace\n"
    assert not (root / RUN_MANIFEST_FILENAME).exists()
    assert not (root / TERMINAL_SAMPLE_TRACE_RELPATH).exists()


def test_writer_rejects_manifest_pin_escape_and_symlink(tmp_path: Path) -> None:
    escape_root = _run_root(tmp_path / "escape")
    _prepare_pinned_memory(escape_root)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ProductionEvidenceWriteError, match="normalized and relative"):
        _write_success(
            escape_root,
            additional_artifact_pins=(ManifestArtifactPin("outside", "../outside.json"),),
        )
    symlink_root = _run_root(tmp_path / "symlink")
    _prepare_pinned_memory(symlink_root)
    link = symlink_root / "memory" / "linked.json"
    link.symlink_to(outside)
    with pytest.raises(ProductionEvidenceWriteError, match="symlink"):
        _write_success(
            symlink_root,
            additional_artifact_pins=(ManifestArtifactPin("linked", "memory/linked.json"),),
        )
    assert not (escape_root / RUN_MANIFEST_FILENAME).exists()
    assert not (symlink_root / RUN_MANIFEST_FILENAME).exists()


def test_writer_rejects_tampered_bundle_fingerprint(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)
    bundle, _run = _bundle(root)
    tampered = replace(bundle, fingerprint="0" * 64)

    with pytest.raises(ProductionEvidenceWriteError, match="bundle fingerprint"):
        _write_success(root, bundle=tampered)


def test_writer_builds_and_validates_complete_system_stall_ledger(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)
    bundle, run = _bundle(root)
    trace, state = _non_success_terminal(run)
    calls = tuple(
        FoundationModelCall(
            request_id=f"stall-request-{index}",
            role="policy_generator",
            modality="text",
            requested_model=MODEL,
            resolved_model=MODEL,
            response_model=MODEL,
            http_status=200,
            input_tokens=10,
            output_tokens=2,
            billing_amount_usd=0.01,
        )
        for index in range(1, 4)
    )
    stall_evidence = _stall_raw_evidence(root, run, trace, state)
    transaction_evidence = _transaction_accounting_evidence(
        root,
        run,
        mode="stall",
    )
    accounting = replace(_accounting(), phase_completed=5)

    result = write_production_result_evidence(
        bundle,
        terminal_status="SYSTEM_STALL",
        terminal_sample_trace=trace,
        terminal_state=state,
        accounting=accounting,
        transaction_accounting_evidence=transaction_evidence,
        fm_calls=calls,
        terminal_producer=EvidenceProducer("test.terminal", "c" * 64),
        fm_producer=EvidenceProducer("test.fm", "d" * 64),
        system_stall=stall_evidence,
    )

    assert result.terminal_validation.valid
    assert result.terminal_validation.stall_model_request_ids == (
        "stall-request-1",
        "stall-request-2",
        "stall-request-3",
    )
    assert result.usage_validation.valid
    ledger = json.loads((root / SYSTEM_STALL_LEDGER_RELPATH).read_text())
    assert ledger["checkpoint_validity"] == stall_evidence.checkpoint_validity.as_dict()
    assert ledger["host_action_space"] == stall_evidence.host_action_space.as_dict()
    assert ledger["schema_version"] == 4


def test_writer_builds_and_validates_five_consecutive_drop_ledger(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)
    bundle, run = _bundle(root)
    trace, state = _non_success_terminal(run)
    accounting = _drop_accounting()
    transaction_evidence = _transaction_accounting_evidence(root, run, mode="drop")
    drop_evidence = PayloadDropEvidence(
        transaction_evidence.transaction_receipts[-5:]
    )

    result = write_production_result_evidence(
        bundle,
        terminal_status="CONSECUTIVE_PAYLOAD_DROP_LIMIT",
        terminal_sample_trace=trace,
        terminal_state=state,
        accounting=accounting,
        transaction_accounting_evidence=transaction_evidence,
        fm_calls=_calls(),
        terminal_producer=EvidenceProducer("test.terminal", "c" * 64),
        fm_producer=EvidenceProducer("test.fm", "d" * 64),
        payload_drop=drop_evidence,
    )

    assert result.terminal_validation.valid
    assert result.usage_validation.valid
    ledger = json.loads((root / PAYLOAD_DROP_LEDGER_RELPATH).read_text())
    assert ledger["schema_version"] == 4
    assert ledger["source_transaction_receipts"] == [
        item.as_dict() for item in drop_evidence.transaction_receipts
    ]
    assert ledger["receipt_chain_tail_sha256"] == ledger[
        "source_transaction_receipts"
    ][-1]["sha256"]
    validation_hashes = result.terminal_validation.artifact_hashes or {}
    for reference in drop_evidence.transaction_receipts:
        assert validation_hashes[reference.relative_path] == reference.sha256
def test_writer_cannot_manufacture_drop_evidence_from_transaction_numbers(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)
    bundle, run = _bundle(root)
    trace, state = _non_success_terminal(run)
    transaction_evidence = _transaction_accounting_evidence(root, run, mode="drop")
    missing = tuple(
        ProductionArtifactRef(
            f"evidence/transactions/transaction_{transaction:03d}.json",
            _digest(f"missing-{transaction}"),
            1,
        )
        for transaction in range(3, 8)
    )
    with pytest.raises(ProductionEvidenceWriteError, match="raw evidence is invalid"):
        write_production_result_evidence(
            bundle,
            terminal_status="CONSECUTIVE_PAYLOAD_DROP_LIMIT",
            terminal_sample_trace=trace,
            terminal_state=state,
            accounting=_drop_accounting(),
            transaction_accounting_evidence=transaction_evidence,
            fm_calls=_calls(),
            terminal_producer=EvidenceProducer("test.terminal", "c" * 64),
            fm_producer=EvidenceProducer("test.fm", "d" * 64),
            payload_drop=PayloadDropEvidence(missing),
        )
    assert not (root / RUN_MANIFEST_FILENAME).exists()


def test_writer_rejects_drop_receipt_tamper_wrong_chain_and_symlink(
    tmp_path: Path,
) -> None:
    # A valid reference becomes invalid if bytes change after exclusive publish.
    tamper_root = _run_root(tmp_path / "tamper")
    _prepare_pinned_memory(tamper_root)
    bundle, run = _bundle(tamper_root)
    trace, state = _non_success_terminal(run)
    transaction_evidence = _transaction_accounting_evidence(
        tamper_root,
        run,
        mode="drop",
    )
    drop = PayloadDropEvidence(transaction_evidence.transaction_receipts[-5:])
    first_path = tamper_root / drop.transaction_receipts[0].relative_path
    first_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        ProductionEvidenceWriteError,
        match="transaction accounting evidence is invalid",
    ):
        write_production_result_evidence(
            bundle,
            terminal_status="CONSECUTIVE_PAYLOAD_DROP_LIMIT",
            terminal_sample_trace=trace,
            terminal_state=state,
            accounting=_drop_accounting(),
            transaction_accounting_evidence=transaction_evidence,
            fm_calls=_calls(),
            terminal_producer=EvidenceProducer("test.terminal", "c" * 64),
            fm_producer=EvidenceProducer("test.fm", "d" * 64),
            payload_drop=drop,
        )

    chain_root = _run_root(tmp_path / "chain")
    _prepare_pinned_memory(chain_root)
    chain_bundle, chain_run = _bundle(chain_root)
    chain_trace, chain_state = _non_success_terminal(chain_run)
    broken_accounting = _transaction_accounting_evidence(
        chain_root,
        chain_run,
        mode="drop",
        broken_chain_at=5,
    )
    broken = PayloadDropEvidence(broken_accounting.transaction_receipts[-5:])
    with pytest.raises(ProductionEvidenceWriteError, match="RECEIPT_HASH_CHAIN_MISMATCH"):
        write_production_result_evidence(
            chain_bundle,
            terminal_status="CONSECUTIVE_PAYLOAD_DROP_LIMIT",
            terminal_sample_trace=chain_trace,
            terminal_state=chain_state,
            accounting=_drop_accounting(),
            transaction_accounting_evidence=broken_accounting,
            fm_calls=_calls(),
            terminal_producer=EvidenceProducer("test.terminal", "c" * 64),
            fm_producer=EvidenceProducer("test.fm", "d" * 64),
            payload_drop=broken,
        )

    symlink_root = _run_root(tmp_path / "symlink")
    _prepare_pinned_memory(symlink_root)
    symlink_bundle, symlink_run = _bundle(symlink_root)
    symlink_trace, symlink_state = _non_success_terminal(symlink_run)
    symlink_accounting = _transaction_accounting_evidence(
        symlink_root,
        symlink_run,
        mode="drop",
    )
    outside = tmp_path / "outside-receipt.json"
    outside.write_text("{}\n", encoding="utf-8")
    linked = symlink_root / "evidence/transactions/transaction_003.json"
    linked.unlink()
    linked.symlink_to(outside)
    with pytest.raises(
        ProductionEvidenceWriteError,
        match="unsafe",
    ):
        write_production_result_evidence(
            symlink_bundle,
            terminal_status="CONSECUTIVE_PAYLOAD_DROP_LIMIT",
            terminal_sample_trace=symlink_trace,
            terminal_state=symlink_state,
            accounting=_drop_accounting(),
            transaction_accounting_evidence=symlink_accounting,
            fm_calls=_calls(),
            terminal_producer=EvidenceProducer("test.terminal", "c" * 64),
            fm_producer=EvidenceProducer("test.fm", "d" * 64),
            payload_drop=PayloadDropEvidence(
                (
                    ProductionArtifactRef(
                        "evidence/transactions/transaction_003.json",
                        sha256(outside.read_bytes()).hexdigest(),
                        outside.stat().st_size,
                    ),
                    *tuple(
                        ProductionArtifactRef(
                            f"evidence/transactions/transaction_{transaction:03d}.json",
                            _digest(str(transaction)),
                            1,
                        )
                        for transaction in range(4, 8)
                    ),
                )
            ),
        )


def test_writer_rejects_stall_self_report_without_raw_producer_identity(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)
    bundle, run = _bundle(root)
    trace, state = _non_success_terminal(run)
    stall = _stall_raw_evidence(
        root,
        run,
        trace,
        state,
        checkpoint_producer_source_sha256="not-a-sha",
    )
    transaction_evidence = _transaction_accounting_evidence(
        root,
        run,
        mode="stall",
    )
    calls = tuple(
        FoundationModelCall(
            request_id=f"stall-request-{index}",
            role="policy_generator",
            modality="text",
            requested_model=MODEL,
            resolved_model=MODEL,
            response_model=MODEL,
            http_status=200,
            input_tokens=1,
            output_tokens=1,
            billing_amount_usd=0.0,
        )
        for index in range(1, 4)
    )
    with pytest.raises(ProductionEvidenceWriteError, match="PRODUCER_INVALID"):
        write_production_result_evidence(
            bundle,
            terminal_status="SYSTEM_STALL",
            terminal_sample_trace=trace,
            terminal_state=state,
            accounting=replace(_accounting(), phase_completed=5),
            transaction_accounting_evidence=transaction_evidence,
            fm_calls=calls,
            terminal_producer=EvidenceProducer("test.terminal", "c" * 64),
            fm_producer=EvidenceProducer("test.fm", "d" * 64),
            system_stall=stall,
        )


def test_writer_rejects_producer_sha_change_inside_drop_receipt_chain(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path)
    _prepare_pinned_memory(root)
    bundle, run = _bundle(root)
    trace, state = _non_success_terminal(run)
    transaction_evidence = _transaction_accounting_evidence(
        root,
        run,
        mode="drop",
        evaluation_producer_change_at=5,
    )
    evidence = PayloadDropEvidence(transaction_evidence.transaction_receipts[-5:])
    with pytest.raises(
        ProductionEvidenceWriteError,
        match="PAYLOAD_DROP_EVALUATION_PRODUCER_NOT_CAPABILITY_BOUND",
    ):
        write_production_result_evidence(
            bundle,
            terminal_status="CONSECUTIVE_PAYLOAD_DROP_LIMIT",
            terminal_sample_trace=trace,
            terminal_state=state,
            accounting=_drop_accounting(),
            transaction_accounting_evidence=transaction_evidence,
            fm_calls=_calls(),
            terminal_producer=EvidenceProducer("test.terminal", "c" * 64),
            fm_producer=EvidenceProducer("test.fm", "d" * 64),
            payload_drop=evidence,
        )


def test_raw_evidence_publisher_is_exclusive_and_rejects_symlink_parent(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path)
    payload = {"kind": "test"}
    ref = write_immutable_raw_evidence_artifact(
        root, "evidence/transactions/test.json", payload
    )
    assert ref.sha256 == _canonical_digest(payload)
    with pytest.raises(ProductionEvidenceWriteError, match="overwrite"):
        write_immutable_raw_evidence_artifact(
            root, "evidence/transactions/test.json", payload
        )

    phase_plan = write_frozen_phase_plan_artifact(
        root,
        run_id="raw-publisher-test",
        phase_ids=("phase_1", "phase_2"),
    )
    mutation = write_immutable_raw_evidence_artifact(
        root,
        "evidence/mutations/mutation_commit_001.json",
        {"kind": "test-mutation"},
    )
    assert phase_plan.relative_path == FROZEN_PHASE_PLAN_RELPATH
    assert mutation.relative_path == "evidence/mutations/mutation_commit_001.json"
    with pytest.raises(ProductionEvidenceWriteError):
        write_immutable_raw_evidence_artifact(
            root,
            "evidence/accounting/../escaped.json",
            payload,
        )

    linked_root = _run_root(tmp_path / "linked")
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    linked_root.mkdir(parents=True)
    (linked_root / "evidence").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProductionEvidenceWriteError, match="symlink"):
        write_immutable_raw_evidence_artifact(
            linked_root, "evidence/transactions/test.json", payload
        )
