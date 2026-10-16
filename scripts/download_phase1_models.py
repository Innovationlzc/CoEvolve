#!/usr/bin/env python3
"""Download phase-1 Qwen models for Praxis-Coop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from praxis_coop.configs import DEFAULT_MULTIMODAL_MODEL, PHASE1_MODELS, model_download_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", *PHASE1_MODELS.keys()],
        default=DEFAULT_MULTIMODAL_MODEL,
        help="Model logical name to download.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--provider",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Download backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = model_download_targets() if args.model == "all" else [PHASE1_MODELS[args.model]]
    for target in targets:
        target.local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {target.logical_name} -> {target.hub_id}", flush=True)
        print(f"[download] local_dir={target.local_dir}", flush=True)
        if args.provider == "modelscope":
            if args.local_files_only:
                raise ValueError("--local-files-only is only supported for HuggingFace")
            from modelscope import snapshot_download as modelscope_snapshot_download

            modelscope_snapshot_download(
                model_id=target.hub_id,
                local_dir=str(target.local_dir),
            )
        else:
            # Keep --help and the alternate backend usable without this extra.
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=target.hub_id,
                local_dir=str(target.local_dir),
                local_files_only=args.local_files_only,
            )
        _verify_snapshot(target.local_dir)
        print(f"[download] done: {target.logical_name}", flush=True)


def _verify_snapshot(model_dir: Path) -> None:
    """Fail if a snapshot is missing config or any indexed weight shard."""

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    json.loads(config_path.read_text(encoding="utf-8"))

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_files = sorted(set(index.get("weight_map", {}).values()))
    else:
        weight_files = [path.name for path in sorted(model_dir.glob("*.safetensors"))]
    if not weight_files:
        raise FileNotFoundError(f"no model weights found in {model_dir}")

    missing = [name for name in weight_files if not (model_dir / name).is_file()]
    empty = [
        name
        for name in weight_files
        if (model_dir / name).is_file() and not (model_dir / name).stat().st_size
    ]
    if missing or empty:
        raise RuntimeError(f"incomplete model snapshot: missing={missing}, empty={empty}")
    total_bytes = sum((model_dir / name).stat().st_size for name in weight_files)
    print(
        f"[download] verified {len(weight_files)} weight file(s), "
        f"{total_bytes / 1_000_000_000:.2f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
