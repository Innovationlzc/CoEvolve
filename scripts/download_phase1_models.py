#!/usr/bin/env python3
"""Download phase-1 Qwen models for Praxis-Coop."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from praxis_coop.configs import PHASE1_MODELS, model_download_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", *PHASE1_MODELS.keys()],
        default="all",
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
            snapshot_download(
                repo_id=target.hub_id,
                local_dir=str(target.local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                local_files_only=args.local_files_only,
            )
        print(f"[download] done: {target.logical_name}", flush=True)


if __name__ == "__main__":
    main()
