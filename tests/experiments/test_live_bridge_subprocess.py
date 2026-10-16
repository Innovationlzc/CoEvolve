from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from coevolve_experiments.contracts import (
    MemoryCondition,
    MemoryPolicy,
    MethodVariant,
    RunSpec,
    SceneId,
    TableId,
)
from coevolve_experiments.live import bridge as bridge_module
from coevolve_experiments.live.bridge import (
    LIVE_BRIDGE_RECEIPT_FILENAME,
    LiveBridgeConfigError,
    LiveResultValidationError,
    LiveSubprocessError,
    launch_live_subprocess,
    load_subprocess_launch_config,
    validate_live_bridge_receipt,
    write_subprocess_launch_config,
)
from coevolve_experiments.live.capability import (
    REQUIRED_SCENE_INJECTION_CHANNELS,
    RUNNER_ACCEPTED_CLI_ARGUMENTS,
    RUNNER_CAPABILITY_KIND,
    RUNNER_CAPABILITY_SCHEMA_VERSION,
    RUNNER_LAUNCH_CONFIG_SCHEMA_VERSION,
    RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
    SUPPORTED_METHOD_VARIANTS,
    SUPPORTED_SCENE_IDS,
    RunnerCapabilityError,
    load_runner_capability,
)
from coevolve_experiments.live.contracts import (
    ExecutionCapability,
    PolicyRepresentation,
)
from coevolve_experiments.live.launch import (
    LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
    CompiledLiveLaunch,
    MemoryExecutionBinding,
    MethodExecutionContract,
    SceneExecutionBinding,
)
from coevolve_experiments.live.preflight import LIVE_BRIDGE_UNAVAILABLE_CODE
from coevolve_experiments.memory import (
    prepare_from_scratch_memory,
    validate_from_scratch_bootstrap,
)
from coevolve_experiments.provider_routes import PRIMARY_GPT_PROVIDER_ROUTE
from coevolve_experiments.full_method_compatibility import (
    FULL_METHOD_ARTIFACT_FIELDS,
    INITIAL_ONLINE_MEMORY_FILENAMES,
    INITIAL_ONLINE_MEMORY_ROOT_RELPATH,
    V83_FULL_BEHAVIOR_RELPATHS,
    VERIFIER_CONFIG_RELPATH,
    FullMethodCompatibilityPin,
    recompute_full_method_compatibility,
)
from coevolve_experiments.serialization import sha256_json


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


_ACCOUNTING_RUNNER_SOURCE = r'''
# A production runner must publish receipt-derived accounting evidence.  The
# package source path is injected because this subprocess runs from its own
# temporary git checkout.
sys.path.insert(0, __COEVOLVE_EXPERIMENTS_SOURCE_ROOT__)
from coevolve_experiments.reporting.terminal_aux_evidence import (
    FROZEN_PAYLOAD_DROP_SAFETY_PROFILE,
    PAYLOAD_DROP_EVALUATION_KIND,
    PAYLOAD_DROP_EXECUTION_TRACE_KIND,
    PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
    PAYLOAD_DROP_POLICY_KIND,
    PHYSICAL_TRANSACTION_RECEIPT_KIND,
    RAW_AUX_EVIDENCE_SCHEMA_VERSION,
    capability_bound_auxiliary_producer_allowlist,
)
from coevolve_experiments.reporting.transaction_accounting import (
    ACCEPTED_MANIFEST_PIN,
    ACCEPTED_MANIFEST_RELPATH,
    FROZEN_PHASE_PLAN_KIND,
    FROZEN_PHASE_PLAN_PIN,
    FROZEN_PHASE_PLAN_RELPATH,
    FROZEN_PHASE_PLAN_SCHEMA_VERSION,
    TRANSACTION_ACCOUNTING_LEDGER_PIN,
    TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
    recompute_transaction_accounting_ledger,
)
from coevolve_experiments.serialization import to_primitive


def write_accounting_document(relative_path, value):
    path = output / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


scene = launch["scene"]
accounting_producers = capability_bound_auxiliary_producer_allowlist(
    capability["entrypoint_sha256"]
)
phase_ids = [f"phase_{index}" for index in range(1, 7)]
phase_plan_ref = write_accounting_document(
    FROZEN_PHASE_PLAN_RELPATH,
    {
        "kind": FROZEN_PHASE_PLAN_KIND,
        "schema_version": FROZEN_PHASE_PLAN_SCHEMA_VERSION,
        "run_id": run["run_id"],
        "phase_ids": phase_ids,
    },
)
accepted_manifest_ref = write_accounting_document(
    ACCEPTED_MANIFEST_RELPATH,
    {
        "kind": "accepted_checkpoint_episode",
        "schema_version": 1,
        "initial_checkpoint": "checkpoint_0",
        "latest_checkpoint": "checkpoint_1",
        "initial_state": {},
        "steps": [{
            "index": 1,
            "attempt": 1,
            "script_id": "fake-complete",
            "checkpoint_before": "checkpoint_0",
            "checkpoint_after": "checkpoint_1",
            "transition_outcome": "TASK_COMPLETE",
            "robot_scripts": {"front": {}, "back": {}},
        }],
    },
)

transaction = 1
checkpoint_before = "checkpoint_0"
checkpoint_after = "checkpoint_1"
event_before = "save:checkpoint_0:000"
event_after = "save:checkpoint_1:001"
common_transaction = {
    "schema_version": RAW_AUX_EVIDENCE_SCHEMA_VERSION,
    "run_id": run["run_id"],
    "physical_transaction": transaction,
}
action_script = "execute_fake_success()"
policy_ref = write_accounting_document(
    "evidence/transactions/transaction_001_policy.json",
    {
        "kind": PAYLOAD_DROP_POLICY_KIND,
        "producer": dict(accounting_producers["payload_drop.policy"]),
        **common_transaction,
        "checkpoint_id": checkpoint_before,
        "checkpoint_event_id": event_before,
        "action_script": action_script,
        "action_script_sha256": hashlib.sha256(
            action_script.encode("utf-8")
        ).hexdigest(),
    },
)
evaluation_ref = write_accounting_document(
    "evidence/transactions/transaction_001_evaluation.json",
    {
        "kind": PAYLOAD_DROP_EVALUATION_KIND,
        "producer": dict(accounting_producers["payload_drop.evaluation"]),
        **common_transaction,
        "checkpoint_id": checkpoint_before,
        "checkpoint_event_id": event_before,
        "accepted": True,
        "failure_type": "none",
        "transition_outcome": "TASK_COMPLETE",
        "completed_phase_ids_before": [],
        "completed_phase_ids_after": phase_ids,
    },
)
execution_ref = write_accounting_document(
    "evidence/transactions/transaction_001_execution_trace.json",
    {
        "kind": PAYLOAD_DROP_EXECUTION_TRACE_KIND,
        "producer": dict(accounting_producers["payload_drop.execution_trace"]),
        **common_transaction,
        "checkpoint_id_before": checkpoint_before,
        "checkpoint_event_id_before": event_before,
        "checkpoint_id_after": checkpoint_after,
        "checkpoint_event_id_after": event_after,
        "physically_dispatched": True,
        "rollback_performed": False,
        "physics_step_count": 50,
        "dispatch_started_s": 0.0,
        "dispatch_finished_s": 0.5,
    },
)
physical_ref = write_accounting_document(
    "evidence/transactions/transaction_001_physical_samples.json",
    {
        "kind": PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
        "producer": dict(accounting_producers["payload_drop.physical_samples"]),
        **common_transaction,
        "checkpoint_id": checkpoint_before,
        "checkpoint_event_id": event_before,
        "simulator": {
            "name": "MuJoCo",
            "version": "test",
            "compiled_model_sha256": scene["compiled_model_sha256"],
            "scene_realization_sha256": scene["execution_composite_sha256"],
        },
        "drop_predicate": {
            "safety_profile": FROZEN_PAYLOAD_DROP_SAFETY_PROFILE,
            "minimum_payload_z_m": 0.65,
            "height_margin_m": 0.05,
            "drop_z_threshold_m": 0.60,
            "contact_distance_tolerance_m": 1.0e-6,
        },
        "samples": [{
            "sample_index": 0,
            "time_s": 0.5,
            "payload_active": True,
            "payload_pose": {
                "position_xyz_m": [0.0, 0.0, 0.80],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "contacts": [],
            "support": {
                "available": True,
                "supporting_robots": [],
                "normal_force_n_by_robot": {"front": 0.0, "back": 0.0},
                "total_normal_force_n": 0.0,
            },
        }],
    },
)
transaction_receipt_ref = write_accounting_document(
    "evidence/transactions/transaction_001.json",
    {
        "kind": PHYSICAL_TRANSACTION_RECEIPT_KIND,
        "producer": dict(
            accounting_producers["payload_drop.transaction_receipt"]
        ),
        **common_transaction,
        "checkpoint_id_before": checkpoint_before,
        "checkpoint_event_id_before": event_before,
        "checkpoint_id_after": checkpoint_after,
        "checkpoint_event_id_after": event_after,
        "physically_dispatched": True,
        "verifier": {"accepted": True, "failure_type": "none"},
        "rollback_performed": False,
        "drop_counter_before": 0,
        "drop_counter_after": 0,
        "policy": policy_ref,
        "evaluation": evaluation_ref,
        "execution_trace": execution_ref,
        "physical_samples": physical_ref,
        "previous_receipt_sha256": None,
    },
)
accounting_derivation = recompute_transaction_accounting_ledger(
    output,
    transaction_receipt_refs=(transaction_receipt_ref,),
    phase_plan_ref=phase_plan_ref,
    accepted_manifest_ref=accepted_manifest_ref,
    mutation_commit_receipt_refs=(),
    expected_run_id=run["run_id"],
    expected_physical_transactions=1,
    physical_budget=run["physical_budget"],
    expected_compiled_model_sha256=scene["compiled_model_sha256"],
    expected_scene_realization_sha256=scene["execution_composite_sha256"],
    expected_producers=accounting_producers,
)
if not accounting_derivation.valid:
    raise RuntimeError(
        "fake accounting evidence is invalid: "
        + ", ".join(accounting_derivation.invalid_reasons)
    )
accounting_ledger_ref = write_accounting_document(
    TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
    to_primitive(accounting_derivation.ledger),
)
accounting_artifact_hashes = {
    TRANSACTION_ACCOUNTING_LEDGER_PIN: accounting_ledger_ref["sha256"],
    ACCEPTED_MANIFEST_PIN: accepted_manifest_ref["sha256"],
    FROZEN_PHASE_PLAN_PIN: phase_plan_ref["sha256"],
}
'''


def _runner_script(behavior: str) -> str:
    accounting_source = _ACCOUNTING_RUNNER_SOURCE.replace(
        "__COEVOLVE_EXPERIMENTS_SOURCE_ROOT__",
        repr(str((Path(__file__).resolve().parents[2] / "src").resolve())),
    )
    return f'''#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

BEHAVIOR = {behavior!r}
if len(sys.argv) != 3 or sys.argv[1] != "--launch-config":
    raise SystemExit(91)
config_path = Path(sys.argv[2])
if not config_path.is_absolute():
    raise SystemExit(92)
if BEHAVIOR == "exit_nonzero":
    print("intentional fake backend failure", file=sys.stderr)
    raise SystemExit(7)
config_raw = config_path.read_bytes()
config = json.loads(config_raw)
launch = config["launch"]
run = launch["run_spec"]
capability = config["runner_capability"]
if BEHAVIOR in {{"timeout_with_child", "exit_nonzero_with_child"}}:
    marker_path = config_path.parent / f"{{BEHAVIOR}}.child-signal"
    ready_path = config_path.parent / f"{{BEHAVIOR}}.child-ready"
    child_code = """
import pathlib
import signal
import sys
import time
marker = pathlib.Path(sys.argv[1])
ready = pathlib.Path(sys.argv[2])
def stop(_signum, _frame):
    marker.write_text("SIGTERM", encoding="utf-8")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
ready.write_text("ready", encoding="utf-8")
while True:
    time.sleep(1)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(marker_path), str(ready_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    (config_path.parent / f"{{BEHAVIOR}}.child-pid").write_text(
        str(child.pid), encoding="utf-8"
    )
    for _ in range(500):
        if ready_path.is_file():
            break
        time.sleep(0.01)
    else:
        raise SystemExit(93)
    if BEHAVIOR == "timeout_with_child":
        time.sleep(60)
    print("intentional fake backend failure with child", file=sys.stderr)
    raise SystemExit(8)
run_spec_sha = hashlib.sha256(json.dumps(
    run,
    ensure_ascii=True,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")).hexdigest()
binding = {{
    "kind": "coevolve_praxis_live_result_binding",
    "schema_version": 4,
    "run_id": run["run_id"],
    "run_spec_sha256": run_spec_sha,
    "launch_fingerprint": launch["launch_fingerprint"],
    "config_fingerprint": config["config_fingerprint"],
    "launch_config_sha256": hashlib.sha256(config_raw).hexdigest(),
    "plan_sha256": launch["plan_sha256"],
    "runner_capability_sha256": capability["manifest_sha256"],
    "base_method_commit": capability["base_method_commit"],
    "runner_integration_commit": capability["runner_integration_commit"],
    "entrypoint_sha256": capability["entrypoint_sha256"],
    "full_method_compatibility_fingerprint": run[
        "full_method_compatibility_fingerprint"
    ],
}}
if BEHAVIOR == "wrong_base_identity":
    binding["base_method_commit"] = "0" * 40
if BEHAVIOR == "wrong_integration_identity":
    binding["runner_integration_commit"] = "0" * 40
if BEHAVIOR == "legacy_result_schema":
    binding["schema_version"] = 3
output = Path(launch["output_dir"])
output.mkdir(parents=True, exist_ok=True)
condition = Path(run["output_relpath"]).parts[1]
memory_dir = output / "memory"
memory_artifact_hashes = {{}}
def semantic_sha(value):
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")).hexdigest()

if BEHAVIOR != "missing_memory_evidence":
    bootstrap_path = memory_dir / "from_scratch_memory_bootstrap.json"
    overlay_path = memory_dir / "online_strategy_overlay.json"
    action_path = memory_dir / "action_exemplars.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    actions = json.loads(action_path.read_text(encoding="utf-8"))
    overlay_sha = semantic_sha(overlay)
    action_sha = semantic_sha(actions)
    runtime_audit = {{
        "kind": "coevolve_from_scratch_memory_runtime_audit",
        "schema_version": 1,
        "lifecycle_state": "finalized",
        "run_binding": bootstrap["run_binding"],
        "memory_condition": run["memory_condition"],
        "memory_policy": run["memory_policy"],
        "cross_run_retrieval": False,
        "operation_gates": bootstrap["operation_gates"],
        "bootstrap_path": bootstrap_path.name,
        "bootstrap_sha256": hashlib.sha256(bootstrap_path.read_bytes()).hexdigest(),
        "initial_state_verified": True,
        "online_overlay": {{
            "path": overlay_path.name,
            "initial_sha256": overlay_sha,
            "current_sha256": overlay_sha,
            "final_sha256": overlay_sha,
            "initial_count": 0,
            "current_count": 0,
            "final_count": 0,
        }},
        "action_exemplars": {{
            "path": action_path.name,
            "initial_sha256": action_sha,
            "current_sha256": action_sha,
            "final_sha256": action_sha,
            "initial_count": 0,
            "current_count": 0,
            "final_count": 0,
        }},
        "operations": {{
            "strategy_retrievals": 0,
            "exemplar_retrievals": 0,
            "strategy_promotions": 0,
            "exemplar_promotions": 0,
            "promotion_noops": 0,
        }},
        "violations": [],
        "finalized": True,
    }}
    runtime_path = memory_dir / "from_scratch_memory_runtime_audit.json"
    runtime_path.write_text(json.dumps(runtime_audit), encoding="utf-8")
    memory_artifact_hashes = {{
        "from_scratch_memory_bootstrap": hashlib.sha256(
            bootstrap_path.read_bytes()
        ).hexdigest(),
        "from_scratch_memory_runtime_audit": hashlib.sha256(
            runtime_path.read_bytes()
        ).hexdigest(),
    }}
if BEHAVIOR != "missing_native_memory_evidence":
    native_root = memory_dir / "v83_native"
    native_root.mkdir(parents=True, exist_ok=True)
    strategy_path = (native_root / "strategy_specs.json").resolve()
    action_exemplar_path = (native_root / "action_exemplars.json").resolve()
    empty_strategy_state = {{
        "path": str(strategy_path),
        "exists": False,
        "sha256": "",
        "record_count": 0,
    }}
    empty_action_state = {{
        "path": str(action_exemplar_path),
        "exists": False,
        "sha256": "",
        "record_count": 0,
    }}
    native_binding = {{
        "run_id": run["run_id"],
        "table_id": run["table_id"],
        "seed": run["seed"],
        "protocol_id": run["protocol_id"],
        "base_method_commit": run["base_method_commit"],
        "condition_fingerprint": run["condition_fingerprint"],
        "invariant_fingerprint": run["invariant_fingerprint"],
        "scene_fingerprint": run["scene_fingerprint"],
        "terminal_fingerprint": run["terminal_fingerprint"],
        "memory_condition": run["memory_condition"],
        "memory_policy": run["memory_policy"],
        "target_backbone": run["target_backbone"],
        "provider_route_id": run["provider_route_id"],
        "provider_route_fingerprint": run["provider_route_fingerprint"],
        "strategy_bank_sha256": "",
        "plan_path": launch["plan_path"],
        "plan_sha256": launch["plan_sha256"],
        "observation_contract_fingerprint": run[
            "observation_contract_fingerprint"
        ],
        "task_instruction_fingerprint": run["task_instruction_fingerprint"],
        "shared_prompt_contract_fingerprint": run[
            "shared_prompt_contract_fingerprint"
        ],
        "decision_prompt_fingerprint": run["decision_prompt_fingerprint"],
        "skill_api_fingerprint": run["skill_api_fingerprint"],
        "full_method_compatibility_fingerprint": run[
            "full_method_compatibility_fingerprint"
        ],
    }}
    zero_operations = {{
        "strategy_retrieval": {{
            "invocations": 0,
            "returned_records": 0,
            "native_records_returned": 0,
            "transfer_seed_records_returned": 0,
        }},
        "strategy_promotion": {{
            "attempts": 0,
            "promotions": 0,
            "noops": 0,
            "errors": 0,
        }},
        "action_exemplar_retrieval": {{
            "invocations": 0,
            "returned_exemplars": 0,
        }},
        "action_exemplar_promotion": {{
            "attempts": 0,
            "promotions": 0,
            "noops": 0,
            "errors": 0,
        }},
        "memory_writer": {{"calls": 0, "failures": 0}},
    }}
    native_audit = {{
        "kind": "coevolve_v83_memory_runtime_audit",
        "schema_version": 1,
        "lifecycle_state": "finalized",
        "finalized": True,
        "mode": "disabled",
        "run_binding": native_binding,
        "run_binding_sha256": semantic_sha(native_binding),
        "cross_run_retrieval": False,
        "operation_gates": {{
            "strategy_retrieval": False,
            "strategy_promotion": False,
            "action_exemplar_retrieval": False,
            "action_exemplar_promotion": False,
        }},
        "initial_state": {{
            "strategy_store": empty_strategy_state,
            "action_exemplar_store": empty_action_state,
            "transfer_seed": None,
        }},
        "operations": zero_operations,
        "suppressed_boundary_calls": {{
            "strategy_retrieval": 0,
            "strategy_promotion": 0,
            "action_exemplar_retrieval": 0,
            "action_exemplar_promotion": 0,
        }},
        "operation_errors": [],
        "violations": [],
        "final_state": {{
            "strategy_store": empty_strategy_state,
            "action_exemplar_store": empty_action_state,
            "transfer_seed": None,
        }},
        "terminal_reason": "SUCCESS",
    }}
    native_audit_path = native_root / "v83_native_memory_runtime_audit.json"
    native_audit_path.write_text(json.dumps(native_audit), encoding="utf-8")
    memory_artifact_hashes[
        "v83_native_memory_runtime_audit_sha256"
    ] = hashlib.sha256(native_audit_path.read_bytes()).hexdigest()
terminal_receipt_hash = None
if BEHAVIOR != "missing_terminal_evidence":
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    scene = launch["scene"]
    target = scene["target_center_xyz_m"]
    simulator = {{
        "name": "MuJoCo",
        "version": "test",
        "compiled_model_sha256": scene["compiled_model_sha256"],
        "scene_realization_sha256": scene["execution_composite_sha256"],
        "scene_fingerprint": scene["scene_fingerprint"],
        "scene_execution_evidence_fingerprint": scene[
            "execution_evidence_fingerprint"
        ],
    }}
    lifecycle = {{
        "release_completed": True,
        "retreat_completed": True,
        "stand_recovered": True,
    }}
    terminal_trace = {{
        "kind": "coevolve_mujoco_terminal_sample_trace",
        "schema_version": 2,
        "run_id": run["run_id"],
        "seed": run["seed"],
        "checkpoint_id": "checkpoint_1",
        "checkpoint_event_id": "save:checkpoint_1:001",
        "simulator": simulator,
        "target": {{
            "center_xyz_m": target,
            "scale_xyz_m": [0.6, 0.6, 0.6],
        }},
        "payload_geometry": {{"half_height_m": 0.05}},
        "sample_window_s": 3.0,
        "samples": [
            {{
                "sample_index": index,
                "time_s": time_s,
                "payload_released": True,
                "payload_pose": {{
                    "position_xyz_m": [target[0] + 0.1, target[1], target[2] + 0.35],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                }},
                "robot_poses": {{
                    "front": {{
                        "position_xyz_m": [target[0] + 1.2, target[1], 0.75],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }},
                    "back": {{
                        "position_xyz_m": [target[0] - 1.2, target[1], 0.75],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }},
                }},
                "contacts": [{{
                    "geom1_id": 11,
                    "geom2_id": 22,
                    "geom1_role": "payload",
                    "geom2_role": "target",
                    "distance_m": -0.0001,
                }}],
            }}
            for index, time_s in enumerate((0.0, 1.5, 3.0))
        ],
        "lifecycle": lifecycle,
    }}
    terminal_trace_raw = json.dumps(
        terminal_trace,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    terminal_trace_path = evidence_dir / "terminal_trace.json"
    terminal_trace_path.write_bytes(terminal_trace_raw)
    terminal_trace_ref = {{
        "relative_path": "evidence/terminal_trace.json",
        "sha256": hashlib.sha256(terminal_trace_raw).hexdigest(),
        "size_bytes": len(terminal_trace_raw),
    }}
    terminal_state = {{
        "kind": "coevolve_mujoco_terminal_state",
        "schema_version": 3,
        "run_id": run["run_id"],
        "seed": run["seed"],
        "checkpoint_id": "checkpoint_1",
        "checkpoint_event_id": "save:checkpoint_1:001",
        "simulator": simulator,
        "sample_window_s": 3.0,
        "sample_count": 3,
        "payload": {{
            "released": True,
            "target_contact_all_samples": True,
            "floor_contact_samples": 0,
            "wall_contact_samples": 0,
            "robot_contact_samples": 0,
            "xy_error_m": 0.1,
            "z_error_m": 0.0,
            "tilt_deg": 0.0,
        }},
        "robots": {{
            "front": {{
                "upright": True,
                "pelvis_height_m": 0.75,
                "distance_to_target_m": 1.2,
            }},
            "back": {{
                "upright": True,
                "pelvis_height_m": 0.75,
                "distance_to_target_m": 1.2,
            }},
        }},
        "lifecycle": lifecycle,
    }}
    terminal_state_raw = json.dumps(
        terminal_state,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    terminal_state_path = evidence_dir / "terminal_state.json"
    terminal_state_path.write_bytes(terminal_state_raw)
    terminal_state_ref = {{
        "relative_path": "evidence/terminal_state.json",
        "sha256": hashlib.sha256(terminal_state_raw).hexdigest(),
        "size_bytes": len(terminal_state_raw),
    }}
    terminal_receipt = {{
        "kind": "coevolve_terminal_verifier_evidence",
        "schema_version": 3,
        "producer": {{
            "id": "test.fake_runner.terminal_verifier",
            "source_sha256": "e" * 64,
        }},
        "run_binding": {{
            "run_id": run["run_id"],
            "seed": run["seed"],
            "condition_fingerprint": run["condition_fingerprint"],
            "terminal_fingerprint": run["terminal_fingerprint"],
            "scene_fingerprint": run["scene_fingerprint"],
            "scene_execution_evidence_fingerprint": scene[
                "execution_evidence_fingerprint"
            ],
        }},
        "terminal_status": "SUCCESS",
        "physical_transactions": 1,
        "terminal_state": terminal_state_ref,
        "terminal_sample_trace": terminal_trace_ref,
        "system_stall": None,
        "payload_drop": None,
    }}
    terminal_receipt_raw = json.dumps(
        terminal_receipt,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    (output / "terminal_verifier_evidence.json").write_bytes(terminal_receipt_raw)
    terminal_receipt_hash = hashlib.sha256(terminal_receipt_raw).hexdigest()
{accounting_source}
usage_ledger = {{
    "kind": "coevolve_fm_usage_ledger",
    "schema_version": 2,
    "producer": {{
        "id": "test.fake_runner.fm_backend",
        "source_sha256": "f" * 64,
    }},
    "run_binding": {{
        "run_id": run["run_id"],
        "seed": run["seed"],
        "condition_fingerprint": run["condition_fingerprint"],
        "provider_route_id": run["provider_route_id"],
        "provider_route_fingerprint": run["provider_route_fingerprint"],
        "target_model": run["target_backbone"],
    }},
    "calls": [{{
        "ordinal": 1,
        "request_id": "fake-request-1",
        "role": "policy_generator",
        "modality": "text",
        "requested_model": run["target_backbone"],
        "resolved_model": run["target_backbone"],
        "outcome": "success",
        "transport_attempts": [{{
            "ordinal": 1,
            "outcome": "http_response",
            "http_status": 200,
            "error_type": None,
        }}],
        "response_model": run["target_backbone"],
        "input_tokens": 1,
        "output_tokens": 1,
        "billing_amount_usd": 0.0,
    }}],
    "totals": {{
        "llm_calls": 1,
        "vlm_calls": 0,
        "successful_responses": 1,
        "failed_requests": 0,
        "transport_attempts": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "billing_records": 1,
        "billing_amount_usd": 0.0,
        "billing_completeness": "complete",
    }},
}}
usage_ledger_raw = json.dumps(
    usage_ledger,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
(output / "fm_usage_ledger.json").write_bytes(usage_ledger_raw)
usage_ledger_hash = hashlib.sha256(usage_ledger_raw).hexdigest()
experiment = {{
    "run_id": run["run_id"],
    "table": run["table_id"],
    "condition": condition,
    "method": run["method_variant"],
    "scene": run["scene_id"],
    "memory_condition": run["memory_condition"],
    "target_model": run["target_backbone"],
    "seed": run["seed"],
    "protocol_id": run["protocol_id"],
    "base_method_commit": run["base_method_commit"],
    "output_relpath": run["output_relpath"],
    "condition_fingerprint": run["condition_fingerprint"],
    "invariant_fingerprint": run["invariant_fingerprint"],
    "scene_fingerprint": run["scene_fingerprint"],
    "terminal_fingerprint": run["terminal_fingerprint"],
    "observation_contract_fingerprint": run["observation_contract_fingerprint"],
    "task_instruction_fingerprint": run["task_instruction_fingerprint"],
    "shared_prompt_contract_fingerprint": run["shared_prompt_contract_fingerprint"],
    "decision_prompt_fingerprint": run["decision_prompt_fingerprint"],
    "skill_api_fingerprint": run["skill_api_fingerprint"],
    "full_method_compatibility_fingerprint": run[
        "full_method_compatibility_fingerprint"
    ],
}}
manifest = {{
    "status": "SUCCESS",
    "experiment": experiment,
    "codebase": {{"git_head": run["base_method_commit"]}},
    "model": {{
        "provider": "provider_a",
        "provider_route_id": run["provider_route_id"],
        "provider_route_fingerprint": run["provider_route_fingerprint"],
        "requested": run["target_backbone"],
        "resolved": run["target_backbone"],
    }},
    "live_launch": binding,
    "result_artifact_hashes": (
        {{
            "terminal_verifier_evidence_sha256": terminal_receipt_hash,
            "fm_usage_ledger_sha256": usage_ledger_hash,
            **accounting_artifact_hashes,
            **memory_artifact_hashes,
        }}
        if terminal_receipt_hash is not None
        else {{
            "fm_usage_ledger_sha256": usage_ledger_hash,
            **accounting_artifact_hashes,
            **memory_artifact_hashes,
        }}
    ),
}}
metrics = {{
    "task_success": True,
    "physical_iterations": 1,
    "accepted_step_count": 1,
    "rejected_step_count": 0,
    "accepted_setup_count": 0,
    "accepted_progress_count": 1,
    "elapsed_s": 0.01,
    "phase_completion": {{"completed": 6, "total": 6, "fraction": 1.0}},
    "failure_counts": {{}},
    "model": {{
        "llm_calls": 1,
        "vlm_calls": 0,
        "input_tokens": 1,
        "output_tokens": 1,
        "billing_amount_usd": 0.0,
        "billing_reconciliation_status": "complete",
    }},
    "terminal": {{"status": "SUCCESS"}},
    "live_launch": binding,
}}
if BEHAVIOR == "invalid_result":
    metrics.pop("live_launch")
(output / "run_plan.json").write_text(json.dumps(run), encoding="utf-8")
(output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
(output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
'''


def _make_runner(
    tmp_path: Path,
    *,
    behavior: str = "ok",
) -> tuple[Path, str, str, Path]:
    repo = tmp_path / f"runner_{behavior}"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "CoEvolve Tests")
    for relpath in V83_FULL_BEHAVIOR_RELPATHS:
        path = repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture bytes for {relpath}\n", encoding="utf-8")
    verifier_config = repo / VERIFIER_CONFIG_RELPATH
    verifier_config.parent.mkdir(parents=True, exist_ok=True)
    verifier_config.write_text('{"terminal":"fixture"}\n', encoding="utf-8")
    initial_memory = repo / INITIAL_ONLINE_MEMORY_ROOT_RELPATH
    initial_memory.mkdir(parents=True, exist_ok=True)
    for filename in INITIAL_ONLINE_MEMORY_FILENAMES:
        (initial_memory / filename).write_text(
            "{}\n" if filename.endswith(".json") else "",
            encoding="utf-8",
        )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "fake frozen base method")
    base_method_commit = _git(repo, "rev-parse", "HEAD")

    entrypoint = repo / "runner.py"
    entrypoint.write_text(_runner_script(behavior), encoding="utf-8")
    entrypoint.chmod(0o755)
    _git(repo, "add", "runner.py")
    _git(repo, "commit", "-q", "-m", "fake integration runner")
    runner_integration_commit = _git(repo, "rev-parse", "HEAD")
    return (
        repo.resolve(),
        base_method_commit,
        runner_integration_commit,
        entrypoint.resolve(),
    )


def _full_method_pin(repo: Path) -> FullMethodCompatibilityPin:
    behavior_hashes = {
        relpath: sha256((repo / relpath).read_bytes()).hexdigest()
        for relpath in V83_FULL_BEHAVIOR_RELPATHS
    }
    declaration = FullMethodCompatibilityPin.from_artifact_hashes(
        {field: _digest(f"placeholder:{field}") for field in FULL_METHOD_ARTIFACT_FIELDS},
        behavior_artifacts=behavior_hashes,
    )
    return recompute_full_method_compatibility(repo, declaration)


def _capability_document(
    repo: Path,
    base_method_commit: str,
    runner_integration_commit: str,
    entrypoint: Path,
) -> dict:
    return {
        "kind": RUNNER_CAPABILITY_KIND,
        "schema_version": RUNNER_CAPABILITY_SCHEMA_VERSION,
        "runner_repo": str(repo),
        "base_method_commit": base_method_commit,
        "runner_integration_commit": runner_integration_commit,
        "clean_worktree_required": True,
        "entrypoint_relpath": entrypoint.relative_to(repo).as_posix(),
        "entrypoint_sha256": sha256(entrypoint.read_bytes()).hexdigest(),
        "policy_representation": "praxis_code_as_policy",
        "execution_capability": "paper_formal",
        "supported_method_variants": list(SUPPORTED_METHOD_VARIANTS),
        "supported_scene_ids": list(SUPPORTED_SCENE_IDS),
        "scene_injection_channels": list(REQUIRED_SCENE_INJECTION_CHANNELS),
        "accepted_cli_arguments": list(RUNNER_ACCEPTED_CLI_ARGUMENTS),
        "freeform_overrides_allowed": False,
        "launch_config_schema_version": RUNNER_LAUNCH_CONFIG_SCHEMA_VERSION,
        "result_contract_schema_version": RUNNER_RESULT_CONTRACT_SCHEMA_VERSION,
        "full_method_compatibility": asdict(_full_method_pin(repo)),
    }


def _write_capability(
    tmp_path: Path,
    repo: Path,
    base_method_commit: str,
    runner_integration_commit: str,
    entrypoint: Path,
    *,
    mutate=None,
) -> Path:
    payload = _capability_document(
        repo,
        base_method_commit,
        runner_integration_commit,
        entrypoint,
    )
    if mutate is not None:
        mutate(payload)
    index = len(tuple(tmp_path.glob(f"capability_{entrypoint.parent.name}_*.json")))
    path = tmp_path / f"capability_{entrypoint.parent.name}_{index}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _compiled_launch(tmp_path: Path, commit: str, repo: Path) -> CompiledLiveLaunch:
    plan_path = (tmp_path / "suite_plan.json").resolve()
    plan_path.write_text('{"kind":"fake-but-byte-pinned-plan"}\n', encoding="utf-8")
    run = RunSpec(
        run_id="table2_method_comparison__remac__seed_000",
        table_id=TableId.TABLE_2,
        seed=0,
        method_variant=MethodVariant.REMAC_ADAPTED,
        scene_id=SceneId.BASE,
        memory_condition=MemoryCondition.EMPTY,
        memory_policy=MemoryPolicy.DISABLED,
        target_backbone="gpt-test",
        provider_route_id=PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        provider_route_fingerprint=PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        physical_budget=80,
        condition_fingerprint=_digest("condition"),
        invariant_fingerprint=_digest("invariant"),
        base_method_commit=commit,
        protocol_id="post-table1-test",
        output_relpath=(
            "table2_method_comparison/remac/seed_000"
        ),
        scene_fingerprint=_digest("scene"),
        terminal_fingerprint=_digest("terminal"),
        observation_contract_fingerprint=_digest("observation"),
        task_instruction_fingerprint=_digest("instruction"),
        shared_prompt_contract_fingerprint=_digest("shared-prompt"),
        decision_prompt_fingerprint=_digest("decision-prompt"),
        skill_api_fingerprint=_digest("skill"),
        full_method_compatibility_fingerprint=_full_method_pin(repo).fingerprint,
    )
    method = MethodExecutionContract(
        method_variant=run.method_variant,
        decision_structure="task_plan_reflection_and_replanning",
        policy_representation=PolicyRepresentation.PRAXIS_CODE_AS_POLICY,
        hierarchical_evolution=False,
        strategy_retrieval=False,
        strategy_promotion=False,
        action_exemplar_retrieval=False,
        action_exemplar_promotion=False,
    )
    scene = SceneExecutionBinding(
        scene_id="base",
        scene_fingerprint=run.scene_fingerprint,
        registry_source_path="assets/base.usda",
        registry_source_sha256=_digest("source"),
        execution_path="generated/base.usda",
        execution_sha256=_digest("execution"),
        execution_composite_sha256=_digest("composite"),
        execution_evidence_fingerprint=_digest("scene-execution-evidence"),
        compiled_model_sha256=_digest("compiled-model"),
        target_center_xyz_m=(-1.5, 7.6, 0.3),
        route_agent_visible=False,
        injection_channels=REQUIRED_SCENE_INJECTION_CHANNELS,
    )
    results_root = (tmp_path / "formal_results").resolve()
    output_dir = (results_root / run.output_relpath).resolve()
    plan_sha = sha256(plan_path.read_bytes()).hexdigest()
    prepare_from_scratch_memory(
        run,
        output_dir / "memory",
        plan_path=plan_path,
        plan_sha256=plan_sha,
    )
    snapshot = validate_from_scratch_bootstrap(
        output_dir / "memory",
        run,
        require_initial_state=True,
        expected_plan_path=plan_path,
        expected_plan_sha256=plan_sha,
    )
    memory = MemoryExecutionBinding(
        memory_condition=run.memory_condition,
        memory_policy=run.memory_policy.value,
        strategy_bank_source_backbone=None,
        strategy_bank_source_protocol_id=None,
        strategy_bank_source_method_commit=None,
        strategy_bank_source_seeds=None,
        strategy_bank_source_condition_fingerprint=None,
        strategy_bank_source_result_manifest_hash=None,
        strategy_bank_expected_sha256=None,
        strategy_bank_path=None,
        strategy_bank_actual_sha256=None,
        strategy_bank_provenance_path=None,
        strategy_bank_provenance_sha256=None,
        bootstrap_snapshot_path=snapshot.path,
        bootstrap_snapshot_sha256=snapshot.file_sha256,
        bootstrap_snapshot_fingerprint=snapshot.snapshot_fingerprint,
        bootstrap_formal_binding_sha256=snapshot.formal_binding_sha256,
        bootstrap_initial_file_sha256s=snapshot.initial_file_sha256s,
        target_online_overlay_starts_empty=True,
        target_action_exemplars_start_empty=True,
        target_manifest_starts_empty=True,
        target_failure_cache_starts_empty=True,
    )
    fingerprint = sha256_json(
        {
            "schema_version": LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
            "run_spec": run,
            "plan_sha256": plan_sha,
            "results_root": results_root,
            "output_dir": output_dir,
            "method": method,
            "scene": scene,
            "memory": memory,
            "execution_capability": ExecutionCapability.PAPER_FORMAL,
        }
    )
    return CompiledLiveLaunch(
        schema_version=LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
        run_spec=run,
        plan_path=plan_path,
        plan_sha256=plan_sha,
        results_root=results_root,
        output_dir=output_dir,
        method=method,
        scene=scene,
        memory=memory,
        execution_capability=ExecutionCapability.PAPER_FORMAL,
        paper_evidence_eligible=False,
        blockers=(LIVE_BRIDGE_UNAVAILABLE_CODE,),
        launch_fingerprint=fingerprint,
    )


def _bind_fake_compiler(monkeypatch, compiled: CompiledLiveLaunch) -> None:
    def fake_compile(
        plan_path,
        run_id,
        *,
        results_root,
        strategy_bank_path=None,
        strategy_bank_provenance_path=None,
        require_launchable=False,
    ):
        assert Path(plan_path) == compiled.plan_path
        assert run_id == compiled.run_spec.run_id
        assert Path(results_root) == compiled.results_root
        assert strategy_bank_path is None
        assert strategy_bank_provenance_path is None
        assert require_launchable is False
        children = (
            tuple(sorted(path.name for path in compiled.output_dir.iterdir()))
            if compiled.output_dir.exists()
            else ()
        )
        if children and children != ("memory",):
            return replace(
                compiled,
                blockers=(
                    LIVE_BRIDGE_UNAVAILABLE_CODE,
                    "PLANNED_OUTPUT_DIRECTORY_NOT_EMPTY",
                ),
            )
        return compiled

    monkeypatch.setattr(bridge_module, "compile_live_launch", fake_compile)


def test_capability_is_closed_world_and_pins_the_checkout(tmp_path: Path) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(tmp_path)
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )

    capability = load_runner_capability(
        capability_path,
        expected_base_method_commit=base_commit,
    )

    assert capability.base_method_commit == base_commit
    assert capability.runner_integration_commit == integration_commit
    assert capability.base_method_commit != capability.runner_integration_commit
    assert capability.entrypoint_path == entrypoint
    assert len(capability.supported_method_variants) == 5
    assert len(capability.supported_scene_ids) == 4
    assert capability.accepted_cli_arguments == ("--launch-config",)

    unknown = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
        mutate=lambda payload: payload.update({"argv": ["--unsafe"]}),
    )
    with pytest.raises(RunnerCapabilityError, match="unknown=.*argv"):
        load_runner_capability(unknown, expected_base_method_commit=base_commit)

    freeform = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
        mutate=lambda payload: payload.update({"freeform_overrides_allowed": True}),
    )
    with pytest.raises(RunnerCapabilityError, match="forbid freeform"):
        load_runner_capability(freeform, expected_base_method_commit=base_commit)

    incomplete_methods = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
        mutate=lambda payload: payload["supported_method_variants"].pop(),
    )
    with pytest.raises(RunnerCapabilityError, match="complete canonical set"):
        load_runner_capability(
            incomplete_methods,
            expected_base_method_commit=base_commit,
        )

    wrong_entrypoint_hash = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
        mutate=lambda payload: payload.update({"entrypoint_sha256": "0" * 64}),
    )
    with pytest.raises(RunnerCapabilityError, match="byte hash differs"):
        load_runner_capability(
            wrong_entrypoint_hash,
            expected_base_method_commit=base_commit,
        )


def test_capability_rejects_legacy_or_incorrect_commit_identity(
    tmp_path: Path,
) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(tmp_path)

    legacy_schema = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
        mutate=lambda payload: payload.update({"schema_version": 3}),
    )
    with pytest.raises(RunnerCapabilityError, match="must equal integer 4"):
        load_runner_capability(
            legacy_schema,
            expected_base_method_commit=base_commit,
        )

    wrong_integration = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
        mutate=lambda payload: payload.update(
            {"runner_integration_commit": "0" * 40}
        ),
    )
    with pytest.raises(RunnerCapabilityError, match="repository HEAD differs"):
        load_runner_capability(
            wrong_integration,
            expected_base_method_commit=base_commit,
        )

    valid = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    with pytest.raises(RunnerCapabilityError, match="differs from RunSpec"):
        load_runner_capability(
            valid,
            expected_base_method_commit="0" * 40,
        )


def test_launch_config_is_rebuilt_and_plan_bytes_are_rechecked(
    tmp_path: Path, monkeypatch
) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(tmp_path)
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    compiled = _compiled_launch(tmp_path, base_commit, repo)
    _bind_fake_compiler(monkeypatch, compiled)
    config_path = (tmp_path / "launches" / "run.json").resolve()

    verified = write_subprocess_launch_config(
        compiled,
        capability_manifest_path=capability_path,
        destination=config_path,
    )
    assert verified.path == config_path
    assert verified.launch.paper_evidence_eligible
    assert verified.launch.blockers == ()

    compiled.plan_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(LiveBridgeConfigError, match="plan bytes"):
        load_subprocess_launch_config(config_path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("base_method_commit", "0" * 40),
        ("runner_integration_commit", "0" * 40),
    ),
)
def test_launch_config_rejects_commit_binding_tamper(
    tmp_path: Path,
    monkeypatch,
    field: str,
    replacement: str,
) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(tmp_path)
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    compiled = _compiled_launch(tmp_path, base_commit, repo)
    _bind_fake_compiler(monkeypatch, compiled)
    config_path = (tmp_path / "launch.json").resolve()
    write_subprocess_launch_config(
        compiled,
        capability_manifest_path=capability_path,
        destination=config_path,
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["runner_capability"][field] = replacement
    fingerprint_payload = {
        key: payload[key]
        for key in ("kind", "schema_version", "launch", "runner_capability")
    }
    payload["config_fingerprint"] = sha256_json(fingerprint_payload)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LiveBridgeConfigError, match="capability binding differs"):
        load_subprocess_launch_config(config_path)


def test_launch_config_rejects_legacy_schema_even_with_new_fingerprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(tmp_path)
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    compiled = _compiled_launch(tmp_path, base_commit, repo)
    _bind_fake_compiler(monkeypatch, compiled)
    config_path = (tmp_path / "launch.json").resolve()
    write_subprocess_launch_config(
        compiled,
        capability_manifest_path=capability_path,
        destination=config_path,
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    fingerprint_payload = {
        key: payload[key]
        for key in ("kind", "schema_version", "launch", "runner_capability")
    }
    payload["config_fingerprint"] = sha256_json(fingerprint_payload)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LiveBridgeConfigError, match="must equal integer 4"):
        load_subprocess_launch_config(config_path)


def test_fake_runner_receives_only_one_absolute_launch_config_and_is_attested(
    tmp_path: Path, monkeypatch
) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(tmp_path)
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    compiled = _compiled_launch(tmp_path, base_commit, repo)
    _bind_fake_compiler(monkeypatch, compiled)
    config = write_subprocess_launch_config(
        compiled,
        capability_manifest_path=capability_path,
        destination=(tmp_path / "launch.json").resolve(),
    )

    original_popen = subprocess.Popen
    runner_calls = []

    def recording_popen(command, *args, **kwargs):
        if command and command[0] == str(entrypoint):
            runner_calls.append((command, kwargs.copy()))
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(bridge_module.subprocess, "Popen", recording_popen)
    receipt = launch_live_subprocess(config.path)

    assert len(runner_calls) == 1
    command, kwargs = runner_calls[0]
    assert command == [str(entrypoint), "--launch-config", str(config.path)]
    assert Path(command[2]).is_absolute()
    assert kwargs["shell"] is False
    assert kwargs["start_new_session"] is True
    assert receipt.protocol_valid
    assert receipt.paper_evidence_eligible
    assert receipt.task_success
    assert receipt.base_method_commit == base_commit
    assert receipt.runner_integration_commit == integration_commit
    assert receipt.bridge_receipt_path.name == LIVE_BRIDGE_RECEIPT_FILENAME
    attestation = json.loads(receipt.bridge_receipt_path.read_text(encoding="utf-8"))
    assert attestation["paper_evidence_eligible"] is True
    assert attestation["base_method_commit"] == base_commit
    assert attestation["runner_integration_commit"] == integration_commit
    assert attestation["run_manifest_sha256"] == receipt.run_manifest_sha256
    assert attestation["metrics_sha256"] == receipt.metrics_sha256
    assert not bridge_module.live_run_reservation_path(receipt.output_dir).exists()

    terminal_state_path = receipt.output_dir / "evidence" / "terminal_state.json"
    terminal_state = json.loads(terminal_state_path.read_text(encoding="utf-8"))
    terminal_state["payload"]["xy_error_m"] = 0.31
    terminal_state_path.write_text(json.dumps(terminal_state), encoding="utf-8")
    with pytest.raises(
        LiveResultValidationError,
        match="no longer resolves to protocol-valid paper evidence",
    ):
        validate_live_bridge_receipt(
            receipt.output_dir,
            expected_run_id=compiled.run_spec.run_id,
        )


@pytest.mark.parametrize(
    "field",
    ("base_method_commit", "runner_integration_commit"),
)
def test_bridge_receipt_rejects_commit_identity_tamper(
    tmp_path: Path,
    monkeypatch,
    field: str,
) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(tmp_path)
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    compiled = _compiled_launch(tmp_path, base_commit, repo)
    _bind_fake_compiler(monkeypatch, compiled)
    config = write_subprocess_launch_config(
        compiled,
        capability_manifest_path=capability_path,
        destination=(tmp_path / "launch.json").resolve(),
    )
    receipt = launch_live_subprocess(config.path)

    document = json.loads(receipt.bridge_receipt_path.read_text(encoding="utf-8"))
    document[field] = "0" * 40
    fingerprint_payload = {
        key: value
        for key, value in document.items()
        if key != "receipt_fingerprint"
    }
    document["receipt_fingerprint"] = sha256_json(fingerprint_payload)
    receipt.bridge_receipt_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(LiveResultValidationError, match=f"{field} differs"):
        validate_live_bridge_receipt(
            receipt.output_dir,
            expected_run_id=compiled.run_spec.run_id,
        )


def test_bridge_receipt_rejects_legacy_schema_even_when_resigned(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(tmp_path)
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    compiled = _compiled_launch(tmp_path, base_commit, repo)
    _bind_fake_compiler(monkeypatch, compiled)
    config = write_subprocess_launch_config(
        compiled,
        capability_manifest_path=capability_path,
        destination=(tmp_path / "launch.json").resolve(),
    )
    receipt = launch_live_subprocess(config.path)

    document = json.loads(receipt.bridge_receipt_path.read_text(encoding="utf-8"))
    document["schema_version"] = 3
    fingerprint_payload = {
        key: value
        for key, value in document.items()
        if key != "receipt_fingerprint"
    }
    document["receipt_fingerprint"] = sha256_json(fingerprint_payload)
    receipt.bridge_receipt_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(LiveResultValidationError, match="must equal integer 4"):
        validate_live_bridge_receipt(
            receipt.output_dir,
            expected_run_id=compiled.run_spec.run_id,
        )


@pytest.mark.parametrize(
    "behavior",
    (
        "exit_nonzero",
        "invalid_result",
        "wrong_base_identity",
        "wrong_integration_identity",
        "legacy_result_schema",
        "missing_terminal_evidence",
        "missing_memory_evidence",
        "missing_native_memory_evidence",
    ),
)
def test_failed_or_unbound_runner_never_receives_a_paper_receipt(
    tmp_path: Path, monkeypatch, behavior: str
) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(
        tmp_path,
        behavior=behavior,
    )
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    compiled = _compiled_launch(tmp_path, base_commit, repo)
    _bind_fake_compiler(monkeypatch, compiled)
    config = write_subprocess_launch_config(
        compiled,
        capability_manifest_path=capability_path,
        destination=(tmp_path / "launch.json").resolve(),
    )

    expected_error = (
        LiveSubprocessError
        if behavior == "exit_nonzero"
        else LiveResultValidationError
    )
    with pytest.raises(expected_error):
        launch_live_subprocess(config.path)
    assert not (compiled.output_dir / LIVE_BRIDGE_RECEIPT_FILENAME).exists()


def test_nonempty_output_blocks_runner_before_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    repo, base_commit, integration_commit, entrypoint = _make_runner(tmp_path)
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    compiled = _compiled_launch(tmp_path, base_commit, repo)
    _bind_fake_compiler(monkeypatch, compiled)
    compiled.output_dir.mkdir(parents=True, exist_ok=True)
    (compiled.output_dir / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(LiveBridgeConfigError, match="unresolved blockers"):
        write_subprocess_launch_config(
            replace(
                compiled,
                blockers=(
                    LIVE_BRIDGE_UNAVAILABLE_CODE,
                    "PLANNED_OUTPUT_DIRECTORY_NOT_EMPTY",
                ),
            ),
            capability_manifest_path=capability_path,
            destination=(tmp_path / "launch.json").resolve(),
        )
    assert not (tmp_path / "launch.json").exists()


def _configured_bridge(
    tmp_path: Path,
    monkeypatch,
    *,
    behavior: str = "ok",
):
    repo, base_commit, integration_commit, entrypoint = _make_runner(
        tmp_path,
        behavior=behavior,
    )
    capability_path = _write_capability(
        tmp_path,
        repo,
        base_commit,
        integration_commit,
        entrypoint,
    )
    compiled = _compiled_launch(tmp_path, base_commit, repo)
    _bind_fake_compiler(monkeypatch, compiled)
    config = write_subprocess_launch_config(
        compiled,
        capability_manifest_path=capability_path,
        destination=(tmp_path / "launch.json").resolve(),
    )
    return compiled, config


def test_bridge_collector_expectation_requires_v83_native_memory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compiled, config = _configured_bridge(tmp_path, monkeypatch)

    expected = bridge_module._collector_expectation(config)

    assert expected["memory_policy"] == compiled.run_spec.memory_policy.value
    assert expected["require_native_memory_evidence"] is True
    assert expected["require_terminal_evidence"] is True
    assert expected["require_usage_evidence"] is True
    assert expected["require_transaction_accounting_evidence"] is True


def test_run_reservation_is_atomic_and_bound_to_run_and_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compiled, config = _configured_bridge(tmp_path, monkeypatch)

    reservation = bridge_module._acquire_run_reservation(config)
    document = json.loads(reservation.path.read_text(encoding="utf-8"))
    assert document["run_id"] == compiled.run_spec.run_id
    assert document["output_dir"] == str(compiled.output_dir)
    assert document["launch_config_sha256"] == config.sha256
    assert document["owner"] == reservation.owner
    with pytest.raises(LiveBridgeConfigError, match="already has a live-run reservation"):
        bridge_module._acquire_run_reservation(config)

    bridge_module._release_run_reservation(reservation)
    assert not reservation.path.exists()


def test_run_reservation_release_never_deletes_replacement_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _compiled, config = _configured_bridge(tmp_path, monkeypatch)
    reservation = bridge_module._acquire_run_reservation(config)
    reservation.path.unlink()
    replacement = b'{"owner":"some-other-launch"}\n'
    reservation.path.write_bytes(replacement)

    with pytest.raises(LiveBridgeConfigError, match="ownership changed"):
        bridge_module._release_run_reservation(reservation)

    assert reservation.path.read_bytes() == replacement


def test_prestart_exec_failure_releases_owned_reservation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compiled, config = _configured_bridge(tmp_path, monkeypatch)

    def fail_popen(*_args, **_kwargs):
        raise OSError("exec unavailable")

    monkeypatch.setattr(bridge_module, "_start_runner_process", fail_popen)
    with pytest.raises(LiveSubprocessError, match="exec unavailable"):
        launch_live_subprocess(config.path)

    assert not bridge_module.live_run_reservation_path(compiled.output_dir).exists()


def test_started_failure_retains_reservation_and_blocks_relaunch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compiled, config = _configured_bridge(
        tmp_path,
        monkeypatch,
        behavior="exit_nonzero",
    )

    with pytest.raises(LiveSubprocessError, match="status 7"):
        launch_live_subprocess(config.path)

    reservation_path = bridge_module.live_run_reservation_path(compiled.output_dir)
    assert reservation_path.is_file()
    with pytest.raises(LiveBridgeConfigError, match="already has a live-run reservation"):
        launch_live_subprocess(config.path)


def _pid_is_running(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except OSError:
        return False
    return len(fields) >= 3 and fields[2] != "Z"


@pytest.mark.parametrize(
    ("behavior", "timeout_seconds", "error_text"),
    (
        ("timeout_with_child", 0.2, "exceeded timeout"),
        ("exit_nonzero_with_child", None, "status 8"),
    ),
)
def test_timeout_or_failed_runner_cleans_entire_process_group(
    tmp_path: Path,
    monkeypatch,
    behavior: str,
    timeout_seconds: float | None,
    error_text: str,
) -> None:
    compiled, config = _configured_bridge(
        tmp_path,
        monkeypatch,
        behavior=behavior,
    )

    with pytest.raises(LiveSubprocessError, match=error_text):
        launch_live_subprocess(config.path, timeout_seconds=timeout_seconds)

    pid = int((tmp_path / f"{behavior}.child-pid").read_text(encoding="utf-8"))
    assert (tmp_path / f"{behavior}.child-signal").read_text(encoding="utf-8") == "SIGTERM"
    deadline = time.monotonic() + 2.0
    while _pid_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _pid_is_running(pid)
    assert bridge_module.live_run_reservation_path(compiled.output_dir).is_file()
