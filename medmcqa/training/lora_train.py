"""
Approach 2: LoRA fine-tuning on MedMCQA — CUDA (DGX/Windows) and MPS (Apple Silicon) compatible.

Supported models:
  google/gemma-3-3b-it  — 6 GB FP16, MPS/CUDA, recommended for local Mac training
  Qwen/Qwen3-14B        — 28 GB BF16 (DGX) or ~8 GB 4-bit QLoRA (RTX 4080)
  Qwen/Qwen3.5-9B       — 18 GB BF16 (DGX) or ~5 GB 4-bit QLoRA (RTX 4080)
  Qwen/Qwen3.5-27B      — 54 GB BF16, DGX only

Device selection is automatic: CUDA → MPS → CPU.
MPS (Apple Silicon): uses FP32 (FP16 overflows → NaN loss on MPS), eager attention, adamw_torch optimizer.
CUDA (DGX): uses BF16, SDPA attention, adamw_torch_fused optimizer.
CUDA + --use_4bit: 4-bit NF4 QLoRA via bitsandbytes (fits 14B in 16 GB VRAM).

Launch via script:
    bash scripts/run_local.sh                               # Gemma 3B on MPS/CUDA
    bash scripts/run_lora.sh                                # Qwen3-14B on DGX (BF16)
    bash scripts/run_qlora.sh                               # Qwen3-14B on RTX 4080 (4-bit)
    MODEL_ID=google/gemma-3-3b-it python training/lora_train.py  # direct
"""

import os
import argparse
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ── Patch Qwen3.5 GatedDeltaNet 5D mask bug ──────────────────────────────────
# Qwen3.5 hybrid layers produce [batch,1,1,seq,seq] masks; transformers
# masking_utils.sdpa_mask expects 4D. Squeeze the extra dim before expand.
# Safe no-op for all other models (guard checks attention_mask.dim() == 5).
import transformers.masking_utils as _mu
_orig_sdpa_mask = _mu.sdpa_mask
def _patched_sdpa_mask(*args, **kwargs):
    attn = kwargs.get("attention_mask")
    if attn is not None:
        if attn.dim() == 5:
            # Qwen3.5 GatedDeltaNet: [b,1,1,s,s] → [b,1,s,s]
            kwargs["attention_mask"] = attn.squeeze(1)
        elif attn.dim() == 3:
            # Gemma3 (transformers 5.x) passes [b,1,s] instead of [b,s].
            # padding_mask_function indexes a 3D tensor → advanced indexing appends
            # the extra dim → 5D mask that fails expand().  Squeeze + cast to bool
            # so padding_mask_function returns bool and bool & bool stays bool
            # (avoids 'where expected bool but got Long' error in eager_mask).
            kwargs["attention_mask"] = attn.squeeze(1).bool()
        elif attn.dtype != torch.bool and attn.dim() == 2:
            # 2D Long attention_mask: also cast to bool to keep mask dtype consistent
            kwargs["attention_mask"] = attn.bool()
    return _orig_sdpa_mask(*args, **kwargs)
_mu.sdpa_mask = _patched_sdpa_mask

# ── Patch Gemma 3 token_type_ids / image-mask requirement ─────────────────────
# transformers 5.x Gemma3 raises if token_type_ids is None during training
# (it's used to separate image vs text tokens in multimodal forward passes).
# For text-only LoRA training there are no images, so we override is_training=False
# in the mask helper — this bypasses the ValueError and skips the MPS-incompatible
# image-block masking code path entirely.  Gradients and causal masking are unaffected.
try:
    import transformers.models.gemma3.modeling_gemma3 as _g3
    _orig_causal_mask = _g3.create_causal_mask_mapping
    def _patched_causal_mask(*args, **kwargs):
        kwargs["is_training"] = False  # suppress token_type_ids requirement
        return _orig_causal_mask(*args, **kwargs)
    _g3.create_causal_mask_mapping = _patched_causal_mask
    print(f"[Patch] Gemma3 create_causal_mask_mapping patched: {_g3.create_causal_mask_mapping.__name__}")
except Exception as e:
    print(f"[Patch] Gemma3 patch FAILED: {e}")


def detect_device() -> str:
    """Return best available device: 'cuda' > 'mps' > 'cpu'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[LoRA] Device: CUDA ({name})")
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "max_split_size_mb:512,expandable_segments:True",
        )
        return "cuda"
    if torch.backends.mps.is_available():
        print("[LoRA] Device: MPS (Apple Silicon)")
        return "mps"
    print("[LoRA] Device: CPU")
    return "cpu"


def tokenize_fn(examples, tokenizer, max_seq_len):
    """
    Chat-format + tokenize to flat input_ids lists (no nested dims).
    Passing pre-tokenized input_ids tells SFTTrainer to skip its own
    tokenization step (which uses AutoProcessor and produces [1,seq]
    nested lists for VLM models like Gemma3, causing [batch,1,seq] batches).
    """
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["messages"]
    ]
    out = tokenizer(texts, truncation=True, max_length=max_seq_len, padding=False)
    return {"input_ids": out["input_ids"]}


def find_target_modules(model) -> list[str]:
    """Auto-detect LoRA target modules by inspecting linear layer names."""
    linear_names = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    attn_patterns = {"q_proj", "k_proj", "v_proj", "o_proj", "q", "k", "v", "out_proj"}
    targets = sorted(linear_names & attn_patterns)
    if not targets:
        print("[LoRA] Warning: no standard attention projection names found. Targeting all linear layers.")
        targets = sorted(linear_names - {"lm_head"})
    print(f"[LoRA] LoRA target modules: {targets}")
    return targets


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning — Approach 2")
    parser.add_argument("--model_id", default="google/gemma-3-4b-it")
    parser.add_argument("--train_file", default="data/processed/train.jsonl")
    parser.add_argument("--dev_file", default="data/processed/dev.jsonl")
    parser.add_argument("--output_dir", default=None,
                        help="Checkpoint dir (default: checkpoints/lora-<model-suffix>)")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--run_name", default=None,
                        help="Run label (default: lora-<model-suffix>)")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Cap training steps (-1 = disabled, use num_epochs)")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Load model in 4-bit NF4 QLoRA mode (requires bitsandbytes; fits 14B in 16 GB VRAM)")
    args = parser.parse_args()

    # Derive run/checkpoint name from model ID
    import re as _re
    model_suffix = args.model_id.split("/")[-1].lower()
    model_suffix = _re.sub(r"[^a-z0-9]+", "-", model_suffix).strip("-")
    run_name = args.run_name or f"lora-{model_suffix}"
    output_dir = args.output_dir or f"checkpoints/lora-{model_suffix}"

    print(f"[LoRA] Model:      {args.model_id}")
    print(f"[LoRA] Run name:   {run_name}")
    print(f"[LoRA] Output dir: {output_dir}")

    # ── Device detection ──────────────────────────────────────────────────────
    device = detect_device()

    # dtype: CUDA supports bfloat16; MPS uses float32 (float16 overflows → NaN loss);
    # CPU uses float32.
    if device == "cuda":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # SDPA works on CUDA; use eager on MPS/CPU to avoid unsupported op errors
    attn_impl = "sdpa" if device == "cuda" else "eager"

    if args.use_4bit and device != "cuda":
        raise RuntimeError("--use_4bit requires a CUDA device (bitsandbytes is CUDA-only)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.use_4bit:
        print(f"[LoRA] Loading model in 4-bit NF4 QLoRA mode: {args.model_id}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 bits/param
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"[LoRA] Loading model ({torch_dtype}, {attn_impl} attn): {args.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            device_map={"": device},
        )
    model.config.use_cache = False
    if not args.use_4bit:
        # gradient checkpointing is handled by prepare_model_for_kbit_training in 4-bit mode
        model.gradient_checkpointing_enable()
    print(f"[LoRA] Gradient checkpointing: ON (use --no_grad_checkpoint to disable)")

    target_modules = find_target_modules(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.dev_file},
    )
    ds = raw_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer, args.max_seq_len),
        batched=True,
        remove_columns=raw_ds["train"].column_names,
    )

    # ── Device-conditional training settings ─────────────────────────────────
    # BF16 mixed-precision only works on CUDA; FP16 AMP is unstable on MPS;
    # use native dtype (float16/float32) without AMP on non-CUDA devices.
    use_bf16 = device == "cuda"
    use_fp16 = False  # MPS AMP not supported; model is already loaded in fp16
    # adamw_torch_fused requires CUDA; fall back to standard adamw_torch
    optimizer = "adamw_torch_fused" if device == "cuda" else "adamw_torch"
    # macOS multiprocessing can deadlock with num_workers > 0
    num_workers = 4 if device == "cuda" else 0

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,  # overrides num_epochs if > 0
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        optim=optimizer,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        run_name=run_name,
        dataloader_num_workers=num_workers,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,  # use text tokenizer, not VLM AutoProcessor
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"[LoRA] Starting training ({args.num_epochs} epochs, {len(ds['train']):,} samples)...")
    trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[LoRA] Adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
