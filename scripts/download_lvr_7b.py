#!/usr/bin/env python3
"""
Utility script to download the Hugging Face repository `vincentleebang/LVR-7B`.

Example:
    python scripts/download_lvr_7b.py --target-dir /path/to/store/model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is required. Install it with `pip install huggingface_hub`."
    ) from exc


DEFAULT_REPO_ID = "vincentleebang/LVR-7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model repository via snapshot_download."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repo ID to download (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision (commit SHA, branch, or tag). Defaults to the repo tip.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Falls back to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Directory where the repo snapshot will be stored. Defaults to a folder named after the repo in the current working directory.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent download workers.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional list of glob patterns to restrict which files are downloaded.",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=None,
        help="Optional list of glob patterns to skip while downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_name = args.repo_id.split("/")[-1]
    default_dir = Path.cwd() / repo_name

    target_dir: Path = (args.target_dir or default_dir).expanduser().resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading `{args.repo_id}` to `{target_dir}`...")
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        token=args.token,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=args.max_workers,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    print("Download complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit("Aborted by user.")

