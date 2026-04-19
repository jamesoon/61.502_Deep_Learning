#!/usr/bin/env python3
"""
Upload a local MedMCQA HF export folder to the Hugging Face Hub (model repo).

Prerequisites:
  pip install huggingface_hub

Auth (pick one):
  huggingface-cli login
  or: export HF_TOKEN=hf_...

Typical local folder (after training runs export_hf_artifacts):
  hf_export/<experiment_name>/
    encoder/          # tokenizer + BertModel weights (pytorch_model.bin or model.safetensors)
    mcqa_head.pt      # linear classifier state_dict
    mcqa_metadata.json

Usage:
  python upload_hf_model.py
  # default repo: jamezoon/medmcqa-pubmedbert-mcqa (override: --repo-id or HF_REPO_ID)

  # explicit folder (same as train.py hf_export/<experiment_name>/)
  python upload_hf_model.py --repo-id ... \\
      --local-dir hf_export/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext@@@data@@@use_contextTrue@@@seqlen192

  python upload_hf_model.py --repo-id ... --private --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "jamezoon/medmcqa-pubmedbert-mcqa"
# Same folder name train.py uses: hf_export/<experiment_name>/ (slashes → _ in export dir only if any)
DEFAULT_HF_EXPORT_DIR = (
    SCRIPT_DIR
    / "hf_export"
    / "microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext@@@data@@@use_contextTrue@@@seqlen192"
)


def resolve_upload_dir(local_dir: Path | None) -> Path:
    """Default: train.py output under hf_export/ (this experiment) or sole subfolder of hf_export/."""
    if local_dir is not None:
        root = local_dir.expanduser().resolve()
        if root.is_dir():
            return root
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(1)
    if DEFAULT_HF_EXPORT_DIR.is_dir():
        return DEFAULT_HF_EXPORT_DIR
    hf_root = SCRIPT_DIR / "hf_export"
    if hf_root.is_dir():
        subs = sorted(p for p in hf_root.iterdir() if p.is_dir())
        if len(subs) == 1:
            return subs[0]
        if subs:
            print(
                "ERROR: multiple hf_export/* folders; pass --local-dir explicitly:\n"
                + "\n".join(f"  - {p}" for p in subs),
                file=sys.stderr,
            )
            sys.exit(1)
    print(
        f"ERROR: no export found. Expected:\n  {DEFAULT_HF_EXPORT_DIR}\n"
        "Run training through export_hf_artifacts or pass --local-dir.",
        file=sys.stderr,
    )
    sys.exit(1)


def _collect_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _warn_if_incomplete(files: list[Path], root: Path) -> None:
    names = {p.relative_to(root).as_posix() for p in files}
    has_weights = any(
        n.endswith(".bin") or n.endswith(".safetensors") for n in names
    )
    has_head = "mcqa_head.pt" in names
    if not has_weights:
        print(
            "WARNING: No pytorch_model.bin / model.safetensors under encoder/. "
            "Upload may be incomplete (tokenizer-only). Check iCloud sync or re-run training export.",
            file=sys.stderr,
        )
    if not has_head:
        print(
            "WARNING: mcqa_head.pt missing — MCQA head will not be on the Hub.",
            file=sys.stderr,
        )


def _default_readme(repo_id: str, meta_path: Path | None) -> str:
    base = "unknown"
    if meta_path and meta_path.is_file():
        try:
            base = json.loads(meta_path.read_text()).get(
                "base_model_name", base
            )
        except json.JSONDecodeError:
            pass
    return f"""---
license: mit
library_name: pytorch
tags:
  - medical
  - question-answering
  - multiple-choice
  - pubmedbert
base_model: {base}
---

# MedMCQA fine-tuned encoder + MCQA head

This repository contains a **Hugging Face–compatible export** from the [medmcqa](https://github.com/medmcqa/medmcqa) training code:

- `encoder/` — `AutoTokenizer` + `AutoModel` (PubMedBERT backbone) saved via `save_pretrained`
- `mcqa_head.pt` — state dict for the linear 4-way classifier used in the paper-style pipeline
- `mcqa_metadata.json` — shapes and training flags

## Loading (conceptual)

1. Load tokenizer and backbone from `encoder/`.
2. `torch.load("mcqa_head.pt", map_location="cpu")` and rebuild `nn.Linear(768, 1)` (see metadata for `in_features`).
3. Run four forward passes per question (or batch as in the original `process_batch`) and softmax over logits.

See the upstream repo for the exact collation and context (`exp`) handling.

**Repo id:** `{repo_id}`
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload MedMCQA HF export to Hugging Face Hub")
    ap.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID),
        help=f"Hub model repo (default: {DEFAULT_REPO_ID}, or set HF_REPO_ID)",
    )
    ap.add_argument(
        "--local-dir",
        default=None,
        type=Path,
        help=(
            "Path to hf_export/<experiment_name> (default: train.py export for PubMedBERT+data+use_context+seqlen192)"
        ),
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Create or keep the repo private",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded; do not call the API",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="HF token (default: HF_TOKEN env or cached login)",
    )
    ap.add_argument(
        "--commit-message",
        default="Upload MedMCQA encoder + MCQA head export",
    )
    ap.add_argument(
        "--no-readme",
        action="store_true",
        help="Do not add README.md if missing",
    )
    args = ap.parse_args()

    if "YOUR_USERNAME" in args.repo_id.upper().replace("-", "_"):
        print(
            "ERROR: do not use the docs placeholder YOUR_USERNAME in --repo-id. "
            f"Default is {DEFAULT_REPO_ID}.",
            file=sys.stderr,
        )
        sys.exit(1)

    root = resolve_upload_dir(args.local_dir)

    files = _collect_files(root)
    if not files:
        print(f"ERROR: no files under {root}", file=sys.stderr)
        sys.exit(1)

    _warn_if_incomplete(files, root)

    meta = root / "mcqa_metadata.json"
    readme_path = root / "README.md"
    if not args.no_readme and not readme_path.is_file():
        readme_path.write_text(_default_readme(args.repo_id, meta if meta.is_file() else None))
        files = _collect_files(root)

    print(f"Repo: {args.repo_id}")
    print(f"Local: {root} ({len(files)} files)")
    for p in files[:30]:
        print(f"  {p.relative_to(root)}")
    if len(files) > 30:
        print(f"  ... and {len(files) - 30} more")

    if args.dry_run:
        print("Dry run — no upload.")
        return

    try:
        from huggingface_hub import HfApi, login
    except ImportError as e:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from e

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
