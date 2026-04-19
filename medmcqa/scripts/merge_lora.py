#!/usr/bin/env python3
"""
Merge a LoRA adapter into its base model and save as BF16 safetensors.
Used as the first step before GGUF conversion.

Usage:
    python scripts/merge_lora.py --adapter checkpoints/lora-9b/final --output checkpoints/merged-9b
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter_path: str, output_path: str) -> None:
    adapter_dir = Path(adapter_path)
    out_dir = Path(output_path)

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    base_model_id = cfg["base_model_name_or_path"]

    print(f"  Base model : {base_model_id}")
    print(f"  Adapter    : {adapter_dir}")
    print(f"  Output     : {out_dir}")

    print("\n[1/3] Loading base model (BF16, CPU)…")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
        trust_remote_code=True,
    )

    print("[2/3] Loading LoRA adapter and merging…")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.eval()

    print("[3/3] Saving merged model…")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))

    size_gb = sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / 1e9
    print(f"\n✓ Saved {size_gb:.1f} GB → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory (final/)")
    parser.add_argument("--output",  required=True, help="Output directory for merged model")
    args = parser.parse_args()

    print("=" * 55)
    print("  LoRA Merge")
    print("=" * 55)
    merge(args.adapter, args.output)
