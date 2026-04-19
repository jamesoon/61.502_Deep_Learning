#!/usr/bin/env python3
"""
Upload a local LoRA (or any) checkpoint folder to the Hugging Face Hub.

Security: use HF_TOKEN env or huggingface-cli login (never commit tokens).

Path resolution (for relative --folder):
  1. Current working directory
  2. This script's directory (medmcqa/)
  3. Parent of medmcqa/ (med/) — so checkpoints can live next to medmcqa/

Examples:
  python upload_lora_to_hf.py --folder ../checkpoints/lora-gemma-3-4b-it/final
  python upload_lora_to_hf.py --folder /absolute/path/to/final
  export LORA_CHECKPOINT_DIR=~/runs/gemma-lora/final && python upload_lora_to_hf.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

SCRIPT_DIR = Path(__file__).resolve().parent
MED_ROOT = SCRIPT_DIR.parent  # med/

DEFAULT_RELATIVE = "checkpoints/lora-gemma-3-4b-it/final"
DEFAULT_REPO = "jamezoon/gemma-3-4b-it-medmcqa-lora"


def resolve_checkpoint_folder(user_path: str | None) -> Path:
    """Return first existing directory among standard locations."""
    if user_path:
        raw = Path(user_path).expanduser()
        candidates = []
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            candidates.extend(
                [
                    (Path.cwd() / raw).resolve(),
                    (SCRIPT_DIR / raw).resolve(),
                    (MED_ROOT / raw).resolve(),
                ]
            )
        for c in candidates:
            if c.is_dir():
                return c
        tried = "\n".join(f"  - {c}" for c in candidates)
        print(f"Folder not found for {user_path!r}. Tried:\n{tried}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.get("LORA_CHECKPOINT_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        print(f"LORA_CHECKPOINT_DIR is not a directory: {p}", file=sys.stderr)
        sys.exit(1)

    rel = Path(DEFAULT_RELATIVE)
    candidates = [
        (Path.cwd() / rel).resolve(),
        (SCRIPT_DIR / rel).resolve(),
        (MED_ROOT / rel).resolve(),
    ]
    for c in candidates:
        if c.is_dir():
            return c

    tried = "\n".join(f"  - {c}" for c in candidates)
    print(
        "No checkpoint folder found. Tried default relative path:\n"
        f"{DEFAULT_RELATIVE}\n"
        f"Resolved as:\n{tried}\n\n"
        "Pass the folder explicitly, e.g.:\n"
        f"  python upload_lora_to_hf.py --folder ../checkpoints/lora-gemma-3-4b-it/final\n"
        "or an absolute path:\n"
        "  python upload_lora_to_hf.py --folder /path/to/your/lora/final\n"
        "or set:\n"
        "  export LORA_CHECKPOINT_DIR=/path/to/final",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload LoRA checkpoint folder to Hugging Face Hub")
    ap.add_argument(
        "--folder",
        default=None,
        help=f"Checkpoint directory to upload (default: env LORA_CHECKPOINT_DIR or {DEFAULT_RELATIVE})",
    )
    ap.add_argument(
        "--repo-id",
        default=DEFAULT_REPO,
        help=f"Hub model repo id (default: {DEFAULT_REPO})",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "Set HF_TOKEN in the environment (or run: huggingface-cli login)",
            file=sys.stderr,
        )
        sys.exit(1)

    root = resolve_checkpoint_folder(args.folder)
    print(f"Uploading: {root}")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("Done!")
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
