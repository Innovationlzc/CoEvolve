#!/usr/bin/env python3
"""Run three consecutive structured outputs for every formal model role."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.cap_policy.contracts import write_json  # noqa: E402
from praxis_coop.cap_policy.formal_eval import (  # noqa: E402
    FORMAL_BACKBONES,
    FORMAL_EVAL_PROTOCOL_ID,
    FORMAL_MAX_NEW_TOKENS,
    FORMAL_PROVIDER_TIMEOUT_S,
    formal_backbone,
)
from praxis_coop.cap_policy.model_role_pipeline import (  # noqa: E402
    ROLE_SYSTEM_PROMPTS,
    _normalize_role_payload,
    _parse_json_object,
    _validate_async_team_review,
    _validate_evolution_manager,
    _validate_memory_update,
    _validate_mutation_operator,
    _validate_policy_payload,
    _validate_task_plan,
    _validate_vlm_failure_analysis,
)
from praxis_coop.cap_policy.role_schemas import role_output_schema  # noqa: E402
from praxis_coop.llm import OpenAICompatibleTextClient  # noqa: E402


ROLE_ORDER = (
    "task_orchestrator",
    "coder",
    "vlm_failure_analyzer",
    "evolution_manager",
    "mutation_operator",
    "code_repair",
    "memory_update",
)
ROLE_VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "task_orchestrator": _validate_task_plan,
    "coder": _validate_policy_payload,
    "vlm_failure_analyzer": _validate_vlm_failure_analysis,
    "evolution_manager": _validate_evolution_manager,
    "mutation_operator": _validate_mutation_operator,
    "code_repair": _validate_async_team_review,
    "memory_update": _validate_memory_update,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Exact frozen model ID. Repeat for a subset; omit for all three.",
    )
    parser.add_argument("--base-url", default=os.environ.get("COEVOLVE_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="COEVOLVE_API_KEY")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--consecutive-valid", type=int, default=3)
    parser.add_argument(
        "--role",
        action="append",
        choices=ROLE_ORDER,
        default=[],
        help="Repeat to debug a subset; omit for the frozen seven-role gate.",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        # v2 observations are diagnostic fixtures only; smoke output is v3.
        default=(
            REPO_ROOT
            / "results"
            / "deployment"
            / "formal"
            / "coevolve-eval-v2"
            / "backbone"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "deployment"
            / "non_paper"
            / "role_smoke"
            / FORMAL_EVAL_PROTOCOL_ID
        ),
    )
    args = parser.parse_args()
    if not os.environ.get(args.api_key_env, "").strip():
        raise SystemExit(
            f"missing API credential environment variable {args.api_key_env}"
        )
    fixtures = _load_role_fixtures(args.fixture_root)
    models = (
        tuple(formal_backbone(value) for value in args.model)
        if args.model
        else FORMAL_BACKBONES
    )
    roles = tuple(args.role) if args.role else ROLE_ORDER
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for backbone in models:
        client = OpenAICompatibleTextClient(
            model_name=backbone.requested_model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            max_new_tokens=FORMAL_MAX_NEW_TOKENS,
            temperature=0.0,
            timeout_s=FORMAL_PROVIDER_TIMEOUT_S,
            seed=args.seed,
            required_resolved_model=backbone.requested_model,
        )
        model_result: dict[str, Any] = {
            "protocol_id": FORMAL_EVAL_PROTOCOL_ID,
            "requested_model": backbone.requested_model,
            "seed": args.seed,
            "required_consecutive_valid": args.consecutive_valid,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "roles": {},
        }
        for role in roles:
            fixture = fixtures[role]
            context = (
                {"generation_scope": "async_team_review"}
                if role == "code_repair"
                else {}
            )
            schema = role_output_schema(role, context)
            role_results: list[dict[str, Any]] = []
            consecutive = 0
            for repetition in range(1, args.consecutive_valid + 1):
                before = len(client.call_records)
                raw = ""
                try:
                    raw = _invoke_fixture(
                        client,
                        role=role,
                        fixture=fixture,
                        schema=schema,
                    )
                    parsed, normalizations = _normalize_role_payload(
                        role,
                        _parse_json_object(raw),
                    )
                    ROLE_VALIDATORS[role](parsed)
                    valid = True
                    error = ""
                    consecutive += 1
                except Exception as exc:  # noqa: BLE001 - preserve smoke evidence
                    parsed = None
                    normalizations = []
                    valid = False
                    error = f"{type(exc).__name__}: {exc}"
                    consecutive = 0
                role_results.append(
                    {
                        "repetition": repetition,
                        "valid": valid,
                        "error": error,
                        "raw_response": raw,
                        "parsed_response": parsed,
                        "host_schema_normalizations": normalizations,
                        "usage": client.call_records[before:],
                    }
                )
                if not valid:
                    break
            passed = consecutive >= args.consecutive_valid
            model_result["roles"][role] = {
                "status": "passed" if passed else "failed",
                "consecutive_valid": consecutive,
                "results": role_results,
                "fixture_artifact_id": fixture.get("artifact_id", ""),
            }
            if not passed:
                failures.append(f"{backbone.requested_model}:{role}")
        model_result["status"] = (
            "passed"
            if all(
                item["status"] == "passed"
                for item in model_result["roles"].values()
            )
            else "failed"
        )
        model_result["usage"] = client.usage_summary()
        write_json(args.output_dir / f"{backbone.key}.json", model_result)
        all_results.append(model_result)
        print(f"[{backbone.requested_model}] {model_result['status']}", flush=True)
    persisted_results = _persisted_model_results(args.output_dir)
    persisted_failures = [
        f"{item['requested_model']}:{role}"
        for item in persisted_results
        for role, role_result in item.get("roles", {}).items()
        if role_result.get("status") != "passed"
    ]
    complete = len(persisted_results) == len(FORMAL_BACKBONES)
    write_json(
        args.output_dir / "summary.json",
        {
            "protocol_id": FORMAL_EVAL_PROTOCOL_ID,
            "status": (
                "failed"
                if persisted_failures
                else "passed"
                if complete
                else "partial_passed"
            ),
            "complete_formal_model_set": complete,
            "models": persisted_results or all_results,
            "failures": persisted_failures,
        },
    )
    if failures:
        raise SystemExit("formal role smoke failed: " + ", ".join(failures))


def _persisted_model_results(output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for backbone in FORMAL_BACKBONES:
        path = Path(output_dir) / f"{backbone.key}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("protocol_id") == FORMAL_EVAL_PROTOCOL_ID
            and payload.get("requested_model") == backbone.requested_model
        ):
            results.append(payload)
    return results


def _invoke_fixture(
    client: OpenAICompatibleTextClient,
    *,
    role: str,
    fixture: dict[str, Any],
    schema: dict[str, Any],
) -> str:
    kwargs = {
        "system": ROLE_SYSTEM_PROMPTS[role],
        "user": str(fixture["user_prompt"]),
        "schema_name": f"{role}_output",
        "schema": schema,
    }
    image_path = str(fixture.get("image_path", "")).strip()
    if image_path:
        path = Path(image_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.is_file():
            return client.analyze_image_structured(image_path=path, **kwargs)
    return client.generate_structured(**kwargs)


def _load_role_fixtures(root: Path) -> dict[str, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(Path(root).glob("*/seed_000/model_roles/role_call_index.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates.extend(
            item
            for item in payload.get("calls", ())
            if isinstance(item, dict) and item.get("valid") is True
        )
    fixtures: dict[str, dict[str, Any]] = {}
    for role in ROLE_ORDER:
        matching = [item for item in candidates if item.get("role") == role]
        if role == "coder":
            matching = [
                item
                for item in matching
                if "robot_scripts" in (item.get("parsed_response") or {})
            ]
        if role == "code_repair":
            matching = [
                item
                for item in matching
                if "decision" in (item.get("parsed_response") or {})
            ]
        if not matching:
            raise SystemExit(f"no valid offline fixture found for role {role}")
        fixtures[role] = matching[-1]
    return fixtures


if __name__ == "__main__":
    main()
