#!/usr/bin/env python3
"""Smoke-test frozen multimodal backbone IDs without starting physics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.cap_policy.contracts import write_json  # noqa: E402
from praxis_coop.cap_policy.formal_eval import (  # noqa: E402
    FORMAL_BACKBONES,
    FORMAL_EVAL_PROTOCOL_ID,
    FORMAL_PROVIDER_TIMEOUT_S,
    formal_backbone,
)
from praxis_coop.cap_policy.model_role_pipeline import (  # noqa: E402
    _parse_json_object,
)
from praxis_coop.llm.openai_compatible_client import (  # noqa: E402
    OpenAICompatibleTextClient,
)


SYSTEM_PROMPT = (
    "You are a multimodal API compatibility probe. Inspect the supplied image "
    "and return exactly one JSON object with status='ok', "
    "image_modality_used=true, and one short observation string."
)
USER_PROMPT = (
    "Confirm that image input and structured JSON output both work. Do not infer "
    "a robot action or provide markdown."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Existing non-sensitive camera frame used only for multimodal smoke testing.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Exact frozen model ID. Repeat to test a subset; omit to test all three.",
    )
    parser.add_argument("--base-url", default=os.environ.get("COEVOLVE_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="COEVOLVE_API_KEY")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "deployment"
            / "non_paper"
            / "api_smoke"
            / FORMAL_EVAL_PROTOCOL_ID
        ),
    )
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"smoke image does not exist: {image_path}")
    if not os.environ.get(args.api_key_env, "").strip():
        raise SystemExit(
            f"missing API credential environment variable {args.api_key_env}"
        )

    models = (
        tuple(formal_backbone(value) for value in args.model)
        if args.model
        else FORMAL_BACKBONES
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    summaries: list[dict[str, object]] = []
    for backbone in models:
        client = OpenAICompatibleTextClient(
            model_name=backbone.requested_model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            max_new_tokens=1024,
            temperature=0.0,
            timeout_s=FORMAL_PROVIDER_TIMEOUT_S,
            seed=args.seed,
            required_resolved_model=backbone.requested_model,
        )
        result: dict[str, object] = {
            "protocol_id": FORMAL_EVAL_PROTOCOL_ID,
            "requested_model": backbone.requested_model,
            "seed": args.seed,
            "image_path": str(image_path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            raw = client.analyze_image(
                system=SYSTEM_PROMPT,
                user=USER_PROMPT,
                image_path=image_path,
            )
            response = _parse_json_object(raw)
            if (
                response.get("status") != "ok"
                or response.get("image_modality_used") is not True
            ):
                raise ValueError(
                    "response did not confirm status=ok and image_modality_used=true"
                )
            record = client.call_records[-1]
            resolved = str(record.get("resolved_model", "")).strip()
            if resolved != backbone.requested_model:
                raise ValueError(
                    "resolved model mismatch: "
                    f"requested={backbone.requested_model!r}, resolved={resolved!r}"
                )
            result.update(
                {
                    "status": "passed",
                    "resolved_model": resolved,
                    "response": response,
                    "usage": client.usage_summary(),
                    "call_record": record,
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-model smoke evidence
            result.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "usage": client.usage_summary(),
                    "call_records": client.call_records,
                }
            )
            failures.append(f"{backbone.requested_model}: {exc}")
        write_json(
            args.output_dir / f"{backbone.key}.json",
            result,
        )
        summaries.append(result)
        print(
            f"[{backbone.requested_model}] {result['status']}",
            flush=True,
        )

    write_json(
        args.output_dir / "summary.json",
        {
            "protocol_id": FORMAL_EVAL_PROTOCOL_ID,
            "status": "failed" if failures else "passed",
            "models": summaries,
        },
    )
    if failures:
        raise SystemExit("formal backbone smoke failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
