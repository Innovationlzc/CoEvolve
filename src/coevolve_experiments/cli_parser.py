"""Argument definitions, independent of command execution and result mutation.

Keep public entrypoints in ``cli.py``. Injecting handlers preserves downstream
patching of that module without creating circular imports here.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
import sys

from .contracts import MethodVariant, SceneId
from .reporting.replay_evidence import ReplayHistoryMode
from .scenes import EVIDENCE_CHECKS


def build_parser(handlers: Mapping[str, Callable]) -> argparse.ArgumentParser:
    """Register arguments only; handlers are injected by the compatibility facade."""
    parser = argparse.ArgumentParser(
        prog="coevolve-experiments",
        description="Protocol-safe utilities for CoEvolve paper Tables 2--4.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="compile the frozen run matrix")
    plan.add_argument("--lock", type=Path, required=True)
    plan.add_argument("--tier", choices=("core", "full"), default="full")
    plan.add_argument("--output", type=Path)
    plan.add_argument(
        "--require-execution-ready",
        action="store_true",
        help="also require realized scene assets and all simulator evidence",
    )
    plan.set_defaults(handler=handlers["_cmd_plan"])

    paper = subparsers.add_parser(
        "paper-contract",
        help="emit the registered Table 2--4 claim/matrix contract",
    )
    paper.add_argument("--tier", choices=("core", "full"), default="full")
    paper.add_argument("--output", type=Path)
    paper.set_defaults(handler=handlers["_cmd_paper_contract"])

    build_lock = subparsers.add_parser(
        "build-protocol-lock",
        help=(
            "freeze one canonical lock from complete Table-1 source and "
            "Table-4 candidate evidence"
        ),
    )
    build_lock.add_argument(
        "--input",
        type=Path,
        required=True,
        help="absolute canonical protocol-lock build-input manifest",
    )
    build_lock.add_argument("--output", type=Path, required=True)
    build_lock.add_argument("--identity-output", type=Path)
    build_lock.set_defaults(handler=handlers["_cmd_build_protocol_lock"])

    manuscript = subparsers.add_parser(
        "audit-manuscript",
        help="check the Tables 2--4 LaTeX claims against the frozen protocol",
    )
    manuscript.add_argument("--methodology", type=Path)
    manuscript.add_argument("--experiments", type=Path)
    manuscript.add_argument("--protocol-root", type=Path)
    manuscript.add_argument("--output", type=Path)
    manuscript.set_defaults(handler=handlers["_cmd_audit_manuscript"])

    launch = subparsers.add_parser(
        "compile-launch",
        help="compile one plan-bound live launch without starting physics",
    )
    launch.add_argument("--plan", type=Path, required=True)
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--results-root", type=Path, required=True)
    launch.add_argument("--strategy-bank", type=Path)
    launch.add_argument("--strategy-bank-provenance", type=Path)
    launch.add_argument(
        "--require-launchable",
        action="store_true",
        help="fail unless scene evidence, memory, output, and live bridge are ready",
    )
    launch.add_argument("--output", type=Path)
    launch.set_defaults(handler=handlers["_cmd_compile_launch"])

    bind_launch = subparsers.add_parser(
        "bind-live-launch",
        help="bind one compiled run to an audited production runner",
    )
    bind_launch.add_argument("--plan", type=Path, required=True)
    bind_launch.add_argument("--run-id", required=True)
    bind_launch.add_argument("--results-root", type=Path, required=True)
    bind_launch.add_argument("--capability", type=Path, required=True)
    bind_launch.add_argument("--launch-config", type=Path, required=True)
    bind_launch.add_argument("--strategy-bank", type=Path)
    bind_launch.add_argument("--strategy-bank-provenance", type=Path)
    bind_launch.set_defaults(handler=handlers["_cmd_bind_live_launch"])

    run_live = subparsers.add_parser(
        "run-live",
        help="execute one previously bound production launch config",
    )
    run_live.add_argument("--launch-config", type=Path, required=True)
    run_live.add_argument("--timeout-seconds", type=float)
    run_live.set_defaults(handler=handlers["_cmd_run_live"])

    prepare_batch = subparsers.add_parser(
        "prepare-live-batch",
        help="prepare one fixed paper-table partition of an execution-ready suite",
    )
    prepare_batch.add_argument("--plan", type=Path, required=True)
    prepare_batch.add_argument("--results-root", type=Path, required=True)
    prepare_batch.add_argument("--capability", type=Path, required=True)
    prepare_batch.add_argument("--batch-root", type=Path, required=True)
    prepare_batch.add_argument(
        "--partition",
        choices=("full", "table2", "table3", "table4"),
        default="full",
        help="fixed registry partition; arbitrary run/seed selection is not supported",
    )
    prepare_batch.add_argument("--gpt-strategy-bank", type=Path)
    prepare_batch.add_argument(
        "--gpt-strategy-provenance", type=Path
    )
    prepare_batch.add_argument("--gemini-strategy-bank", type=Path)
    prepare_batch.add_argument(
        "--gemini-strategy-provenance", type=Path
    )
    prepare_batch.set_defaults(handler=handlers["_cmd_prepare_live_batch"])

    status_batch = subparsers.add_parser(
        "status-live-batch",
        help="revalidate every config/result in a prepared live batch",
    )
    status_batch.add_argument("--manifest", type=Path, required=True)
    status_batch.add_argument("--output", type=Path)
    status_batch.set_defaults(handler=handlers["_cmd_status_live_batch"])

    run_batch = subparsers.add_parser(
        "run-live-batch",
        help="run a prepared suite serially, stopping at the first invalid cell",
    )
    run_batch.add_argument("--manifest", type=Path, required=True)
    run_batch.add_argument("--timeout-seconds", type=float)
    run_batch.add_argument("--max-runs", type=int)
    run_batch.add_argument("--output", type=Path)
    run_batch.set_defaults(handler=handlers["_cmd_run_live_batch"])

    scenes = subparsers.add_parser(
        "validate-scenes", help="validate every registered scene"
    )
    scenes.add_argument("--mode", choices=("planning", "execution"), default="planning")
    scenes.add_argument("--lock", type=Path)
    scenes.add_argument(
        "--repo-root", type=Path,
        help="checkout containing scene assets (required when validating from an installed wheel)",
    )
    scenes.add_argument("--output", type=Path)
    scenes.set_defaults(handler=handlers["_cmd_validate_scenes"])

    materialize = subparsers.add_parser(
        "materialize-scene",
        help="generate one deterministic candidate USDA plus a hash receipt",
    )
    materialize.add_argument(
        "--scene",
        choices=tuple(scene_id.value for scene_id in SceneId),
        required=True,
    )
    materialize.add_argument("--repo-root", type=Path, required=True)
    materialize.add_argument(
        "--output",
        type=Path,
        required=True,
        help="candidate .usda path below --repo-root",
    )
    materialize.set_defaults(handler=handlers["_cmd_materialize_scene"])

    produce_evidence = subparsers.add_parser(
        "produce-scene-evidence",
        help="compile one real MuJoCo scene-evidence bundle without running an experiment",
    )
    produce_evidence.add_argument("--check", choices=EVIDENCE_CHECKS, required=True)
    produce_evidence.add_argument(
        "--scene",
        choices=tuple(scene_id.value for scene_id in SceneId),
        required=True,
    )
    produce_evidence.add_argument("--lock", type=Path, required=True)
    produce_evidence.add_argument("--repo-root", type=Path, required=True)
    produce_evidence.add_argument(
        "--gear-root",
        type=Path,
        required=True,
        help="read-only root of the pinned local GEAR-SONIC checkout",
    )
    produce_evidence.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new evidence directory below --repo-root; existing paths are refused",
    )
    produce_evidence.set_defaults(handler=handlers["_cmd_produce_scene_evidence"])

    posthoc = subparsers.add_parser(
        "build-posthoc-strategy-evidence",
        help="build an external verifier-evidence package for immutable V83 runs",
    )
    posthoc.add_argument("--source-run", type=Path, action="append", required=True)
    posthoc.add_argument("--source-manifest", type=Path, required=True)
    posthoc.add_argument("--parent-manifest", type=Path)
    posthoc.add_argument("--source-cohort", choices=("gpt", "gemini"), required=True)
    posthoc.add_argument("--output-dir", type=Path, required=True)
    posthoc.set_defaults(handler=handlers["_cmd_build_posthoc_evidence"])

    candidate = subparsers.add_parser(
        "candidate-export-strategy-bank",
        help="audit five Table-1 runs and export a pre-lock candidate bank",
    )
    candidate.add_argument("--source-run", type=Path, action="append", required=True)
    candidate.add_argument("--source-manifest", type=Path, required=True)
    candidate.add_argument("--parent-manifest", type=Path)
    candidate.add_argument("--source-cohort", choices=("gpt", "gemini"), required=True)
    candidate.add_argument("--evidence-package", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--provenance-output", type=Path)
    candidate.add_argument("--candidate-output", type=Path, required=True)
    candidate.set_defaults(handler=handlers["_cmd_candidate_export_bank"])

    export = subparsers.add_parser(
        "export-strategy-bank", help="export one audited six-field Table-4 bank"
    )
    export.add_argument("--source-run", type=Path, action="append", required=True)
    export.add_argument(
        "--source-manifest",
        "--audit",
        dest="source_manifest",
        type=Path,
        required=True,
        help=(
            "exact aggregate Table-1 result manifest; --audit remains only "
            "as a backwards-compatible spelling and is audited identically"
        ),
    )
    export.add_argument(
        "--parent-manifest",
        type=Path,
        help="exact audited-V83 GPT manifest required for the Gemini companion",
    )
    export.add_argument("--lock", type=Path, required=True)
    export.add_argument("--source-cohort", choices=("gpt", "gemini"), required=True)
    export.add_argument("--evidence-package", type=Path, required=True)
    export.add_argument("--candidate-attestation", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--provenance-output", type=Path)
    export.set_defaults(handler=handlers["_cmd_export_bank"])

    validate = subparsers.add_parser(
        "validate-strategy-bank", help="validate bank schema, caps, hash, and leaks"
    )
    validate.add_argument("bank", type=Path)
    validate.add_argument("--expected-sha256")
    validate.add_argument("--output", type=Path)
    validate.set_defaults(handler=handlers["_cmd_validate_bank"])

    transfer = subparsers.add_parser(
        "prepare-transfer",
        help="prepare one plan-pinned Table-4 target memory",
    )
    transfer.add_argument("--plan", type=Path, required=True)
    transfer.add_argument("--run-id", required=True)
    transfer.add_argument("--results-root", type=Path, required=True)
    transfer.add_argument("--bank", type=Path, required=True)
    transfer.add_argument("--provenance", type=Path, required=True)
    transfer.set_defaults(handler=handlers["_cmd_prepare_transfer"])

    summarize = subparsers.add_parser(
        "summarize",
        help="build a formal Tables 2--4 summary or reconcile ad-hoc runs",
    )
    source_group = summarize.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--run-dir", type=Path, action="append")
    source_group.add_argument("--results-root", type=Path)
    summarize.add_argument(
        "--plan",
        type=Path,
        help="external suite plan required with --results-root",
    )
    summarize.add_argument(
        "--table1-source-manifest",
        "--source-aggregate-manifest",
        dest="source_aggregate_manifest",
        type=Path,
        help=(
            "audited Table-1 aggregate manifest required with --results-root "
            "to resolve the Full/Base reference-only rows"
        ),
    )
    summarize.add_argument(
        "--table1-parent-manifest",
        "--source-parent-manifest",
        dest="source_parent_manifest",
        type=Path,
        help=(
            "exact GPT parent aggregate, required only when the selected "
            "external source is a provider companion"
        ),
    )
    summarize.add_argument("--require-complete", action="store_true")
    summarize.add_argument("--output", type=Path)
    summarize.set_defaults(handler=handlers["_cmd_summarize"])

    smoke = subparsers.add_parser(
        "adapter-smoke", help="run model-free contract smoke tests (no physics)"
    )
    smoke.add_argument(
        "--method",
        choices=("all", *(variant.value for variant in MethodVariant)),
        default="all",
    )
    smoke.add_argument("--output", type=Path)
    smoke.set_defaults(handler=handlers["_cmd_adapter_smoke"])

    prepare_replay = subparsers.add_parser(
        "prepare-strict-replay",
        help="freeze a complete hash-bound appendix replay request",
    )
    prepare_replay.add_argument("--run-dir", type=Path, required=True)
    prepare_replay.add_argument("--package-dir", type=Path, required=True)
    prepare_replay.add_argument(
        "--mode",
        choices=tuple(item.value for item in ReplayHistoryMode),
        required=True,
    )
    prepare_replay.add_argument("--manifest-output", type=Path)
    prepare_replay.set_defaults(handler=handlers["_cmd_prepare_strict_replay"])

    run_replay = subparsers.add_parser(
        "run-strict-replay",
        help="execute one prepared full-sequence appendix replay",
    )
    run_replay.add_argument("--manifest", type=Path, required=True)
    run_replay.add_argument("--run-dir", type=Path, required=True)
    run_replay.add_argument("--package-dir", type=Path, required=True)
    run_replay.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
    )
    run_replay.add_argument("--timeout-seconds", type=float)
    run_replay.set_defaults(handler=handlers["_cmd_run_strict_replay"])
    return parser
