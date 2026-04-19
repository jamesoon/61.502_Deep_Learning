#!/usr/bin/env python3
"""
Upload quantized GGUF models to HuggingFace Hub under jamezoon.

Repos created:
  jamezoon/qwen3-5-9b-medmcqa-gguf
  jamezoon/qwen3-14b-medmcqa-gguf

Usage:
  HF_TOKEN=hf_xxx python scripts/upload_gguf_hf.py
  HF_TOKEN=hf_xxx python scripts/upload_gguf_hf.py --model lora-9b   # single model
"""
import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_file
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

HF_USER = "jamezoon"

MODELS = {
    "lora-9b": {
        "repo_id":    f"{HF_USER}/qwen3-5-9b-medmcqa-gguf",
        "gguf_name":  "lora-9b-medmcqa-q4_k_m.gguf",
        "base_model": "Qwen/Qwen3.5-9B",
        "quant_type": "Q4_K_M",
        "params":     "9B",
        "adapter_hf": f"{HF_USER}/qwen3-5-9b-medmcqa-lora",
        "model_card": "model_cards/qwen3-5-9b-medmcqa-gguf.md",
    },
    "lora-14b": {
        "repo_id":    f"{HF_USER}/qwen3-14b-medmcqa-gguf",
        "gguf_name":  "lora-14b-medmcqa-q4_k_m.gguf",
        "base_model": "Qwen/Qwen3-14B",
        "quant_type": "Q4_K_M",
        "params":     "14B",
        "adapter_hf": f"{HF_USER}/qwen3-14b-medmcqa-lora",
        "model_card": "model_cards/qwen3-14b-medmcqa-gguf.md",
    },
}

# ── README loader ─────────────────────────────────────────────────────────────

def load_readme(cfg: dict, project_root: Path) -> bytes:
    card_path = project_root / cfg["model_card"]
    if not card_path.exists():
        raise FileNotFoundError(f"Model card not found: {card_path}")
    return card_path.read_bytes()


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_model(name: str, quantized_dir: Path, api: HfApi, project_root: Path) -> None:
    cfg       = MODELS[name]
    repo_id   = cfg["repo_id"]
    gguf_path = quantized_dir / name / cfg["gguf_name"]

    print(f"\n{'═' * 55}")
    print(f"  {name}  →  {repo_id}")
    print(f"{'═' * 55}")

    if not gguf_path.exists():
        print(f"  ✗ GGUF not found: {gguf_path}")
        print(f"    Run fetch_quantized.sh first, then retry.")
        return

    size_gb = gguf_path.stat().st_size / 1e9
    print(f"  GGUF : {gguf_path.name}  ({size_gb:.1f} GB)")

    # Create repo
    create_repo(repo_id, repo_type="model", exist_ok=True, token=api.token)
    print(f"  Repo : https://huggingface.co/{repo_id}")

    # Upload README
    readme_bytes = load_readme(cfg, project_root)
    print("  Uploading README.md…")
    api.upload_file(
        path_or_fileobj=readme_bytes,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
    )

    # Upload tokenizer files
    tok_dir = quantized_dir / name
    for tok_file in tok_dir.glob("tokenizer*.json"):
        print(f"  Uploading {tok_file.name}…")
        api.upload_file(
            path_or_fileobj=str(tok_file),
            path_in_repo=tok_file.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {tok_file.name}",
        )

    # Upload GGUF (large — uses resumable chunked upload automatically)
    print(f"  Uploading {gguf_path.name}  ({size_gb:.1f} GB) — this may take a while…")
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_path.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add {cfg['quant_type']} GGUF",
    )

    print(f"\n  ✓ Done → https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload GGUF models to HuggingFace")
    parser.add_argument("--model", choices=list(MODELS), default=None,
                        help="Upload a single model (default: all)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token
    if not token:
        print("Error: HF_TOKEN not set.")
        print("  Set it via:  export HF_TOKEN=hf_xxx")
        print("  Or pass:     --token hf_xxx")
        sys.exit(1)

    project_root  = Path(__file__).resolve().parent.parent
    quantized_dir = project_root / "quantized"

    if not quantized_dir.exists():
        print(f"Error: quantized/ directory not found at {quantized_dir}")
        print("  Run scripts/fetch_quantized.sh first.")
        sys.exit(1)

    api = HfApi(token=token)

    targets = [args.model] if args.model else list(MODELS)

    print("=" * 55)
    print(f"  HuggingFace Upload — {HF_USER}")
    print("=" * 55)
    for name in targets:
        upload_model(name, quantized_dir, api, project_root)

    print("\n" + "=" * 55)
    print("  All uploads complete.")
    print(f"  Profile: https://huggingface.co/{HF_USER}")
    print("=" * 55)


if __name__ == "__main__":
    main()
