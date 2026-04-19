"""
Approach 1: Full Supervised Fine-Tuning (SFT) of Qwen3.5 dense models.

Memory budget (121 GB unified GB10):
  9B  — weights 18 GB + grad 18 GB + AdamW 36 GB ≈ 72 GB  — fits without ZeRO
  27B — weights 54 GB + grad 54 GB + AdamW 108 GB ≈ 216 GB — requires ZeRO-3

DeepSpeed ZeRO-3 is used for both to keep the setup consistent and safe.
For the 9B model, ZeRO-2 would also work.

Launch (via script — recommended):
    bash scripts/run_sft.sh                              # 9B (default)
    MODEL_ID=Qwen/Qwen3.5-27B bash scripts/run_sft.sh   # 27B

Or directly:
    torchrun --nproc_per_node=1 training/sft_train.py \
        --deepspeed training/ds_config_zero3.json
"""

import os
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer
from datasets import load_dataset

# ── Patch Qwen3.5 GatedDeltaNet 5D mask bug ──────────────────────────────────
import transformers.masking_utils as _mu
_orig_sdpa_mask = _mu.sdpa_mask
def _patched_sdpa_mask(attention_mask, *args, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 5:
        attention_mask = attention_mask.squeeze(1)
    return _orig_sdpa_mask(attention_mask, *args, **kwargs)
_mu.sdpa_mask = _patched_sdpa_mask

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:512,expandable_segments:True",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def apply_chat_template(examples, tokenizer):
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["messages"]
    ]
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Full SFT — Approach 1")
    parser.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--train_file", default="data/processed/train.jsonl")
    parser.add_argument("--dev_file", default="data/processed/dev.jsonl")
    parser.add_argument("--output_dir", default=None,
                        help="Checkpoint dir (default: checkpoints/sft-<model-suffix>)")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--deepspeed", default="training/ds_config_zero3.json")
    args = parser.parse_args()

    model_suffix = args.model_id.split("/")[-1].lower().replace("qwen3.5-", "")
    run_name = args.run_name or f"sft-{model_suffix}"
    output_dir = args.output_dir or f"checkpoints/sft-{model_suffix}"

    print(f"[SFT] Model:      {args.model_id}")
    print(f"[SFT] Run name:   {run_name}")
    print(f"[SFT] Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[SFT] Loading model in BF16 (ZeRO-3 manages memory): {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[SFT] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    raw_ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.dev_file},
    )
    ds = raw_ds.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        batched=True,
        remove_columns=[c for c in raw_ds["train"].column_names if c != "text"],
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        optim="adamw_bnb_8bit",
        bf16=True,
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
        dataloader_num_workers=4,
        deepspeed=args.deepspeed,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"[SFT] Starting training ({args.num_epochs} epochs, {len(ds['train']):,} samples)...")
    trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[SFT] Model saved to {final_dir}")


if __name__ == "__main__":
    main()
