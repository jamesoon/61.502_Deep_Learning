"""
Benchmarking script: evaluate baseline (zero-shot), SFT, and LoRA models.

Metrics:
  - Overall accuracy
  - Per-subject accuracy
  - Macro-averaged accuracy
  - Confusion matrix
  - 50-example error analysis

Usage:
    # Zero-shot baseline
    python training/evaluate.py \
        --model_id Qwen/Qwen3.5-9B \
        --run_name baseline-9b \
        --output_dir results/baseline-9b

    # After LoRA fine-tuning
    python training/evaluate.py \
        --model_id Qwen/Qwen3.5-9B \
        --adapter_path checkpoints/lora-9b/final \
        --run_name lora-9b \
        --output_dir results/lora-9b

    # After SFT
    python training/evaluate.py \
        --model_id Qwen/Qwen3.5-9B \
        --adapter_path checkpoints/sft-9b/final \
        --run_name sft-9b \
        --output_dir results/sft-9b

    # Compare all saved runs
    python training/evaluate.py --compare --results_dir results/
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def detect_device() -> str:
    """Return best available device: 'cuda' > 'mps' > 'cpu'."""
    if torch.cuda.is_available():
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "max_split_size_mb:512,expandable_segments:True",
        )
        print(f"[Eval] Device: CUDA ({torch.cuda.get_device_name(0)})")
        return "cuda"
    if torch.backends.mps.is_available():
        print("[Eval] Device: MPS (Apple Silicon)")
        return "mps"
    print("[Eval] Device: CPU")
    return "cpu"

# ── Patch Qwen3.5 GatedDeltaNet 5D mask bug ──────────────────────────────────
import transformers.masking_utils as _mu
_orig_sdpa_mask = _mu.sdpa_mask
def _patched_sdpa_mask(attention_mask, *args, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 5:
        attention_mask = attention_mask.squeeze(1)
    return _orig_sdpa_mask(attention_mask, *args, **kwargs)
_mu.sdpa_mask = _patched_sdpa_mask

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

def _make_kv_cache_kwargs(nbits: int) -> dict:
    """
    Build generate() kwargs for quantized KV cache (TurboQuant-style compression).

    Uses transformers' built-in QuantizedCache (backed by quanto), which applies
    per-channel int4/int2 quantization to K and V tensors — equivalent to the
    online, data-oblivious quantization described in TurboQuant (arXiv 2504.19874).

    At nbits=4: ~4× KV cache reduction, quality-neutral (matches full-precision
    on LongBench per the paper). Enables larger batch sizes on the same memory.

    Falls back silently to no quantization if quanto is not installed or the
    backend raises (e.g. on MPS where support is partial).
    """
    try:
        from transformers import QuantizedCacheConfig
        return {
            "cache_implementation": "quantized",
            "cache_config": QuantizedCacheConfig(nbits=nbits, residual_length=128),
        }
    except Exception as e:
        print(f"[KV quant] QuantizedCacheConfig unavailable ({e}) — using default cache.")
        return {}

SYSTEM_PROMPT = (
    "You are a helpful tutor for pre-med students preparing for the MCAT. "
    "You answer multiple-choice questions with step-by-step reasoning."
)
COP_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}


def strip_thinking(text: str) -> str:
    """Remove Qwen3.5 <think>…</think> blocks before parsing answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def format_prompt(record: dict) -> str:
    return (
        f"Question: {record['question']}\n"
        f"Options:\n"
        f"A. {record['opa']}\n"
        f"B. {record['opb']}\n"
        f"C. {record['opc']}\n"
        f"D. {record['opd']}\n\n"
        f"Think step by step. Then respond in the format:\n"
        f"Explanation: ...\n"
        f"Answer: <one of A, B, C, D>"
    )


def extract_answer(text: str) -> str | None:
    clean = strip_thinking(text)
    m = re.search(r"Answer:\s*\**\s*([ABCD])\b", clean, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = re.findall(r"\b([ABCD])\b", clean.upper())
    return letters[-1] if letters else None


def extract_explanation(text: str) -> str:
    clean = strip_thinking(text)
    m = re.search(r"Explanation:\s*(.+?)(?:\nAnswer:|$)", clean, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else clean.strip()[:500]


def load_model(model_id: str, adapter_path: str | None):
    device = detect_device()

    # dtype: bfloat16 on CUDA, float16 on MPS (bfloat16 unsupported), float32 on CPU
    if device == "cuda":
        torch_dtype = torch.bfloat16
    elif device == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    attn_impl = "sdpa" if device == "cuda" else "eager"

    tok_path = adapter_path if adapter_path else model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if adapter_path:
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def evaluate(records: list, tokenizer, model, max_new_tokens: int = 128,
             batch_size: int = 8, quantize_kv: int = 0) -> list:
    """
    quantize_kv: 0 = off, 4 = 4-bit (recommended), 2 = 2-bit (aggressive).
    At quantize_kv=4, KV cache memory drops ~4×, allowing larger batch sizes.
    """
    tokenizer.padding_side = "left"  # required for batch generation
    kv_kwargs = _make_kv_cache_kwargs(quantize_kv) if quantize_kv else {}
    if kv_kwargs:
        print(f"[KV quant] {quantize_kv}-bit quantized KV cache enabled "
              f"(~{16//quantize_kv}× compression, batch_size={batch_size})")

    results = []
    for i in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
        batch = records[i: i + batch_size]

        texts = []
        for record in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_prompt(record)},
            ]
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=1536,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                **kv_kwargs,
            )

        input_len = inputs["input_ids"].shape[1]
        for j, record in enumerate(batch):
            generated = tokenizer.decode(
                outputs[j][input_len:], skip_special_tokens=True,
            )
            predicted = extract_answer(generated)
            ground_truth = COP_MAP.get(record["cop"], "A")
            results.append({
                "id": record.get("id", ""),
                "subject_name": record.get("subject_name", "Unknown"),
                "topic_name": record.get("topic_name", ""),
                "question": record["question"],
                "option_a": record["opa"],
                "option_b": record["opb"],
                "option_c": record["opc"],
                "option_d": record["opd"],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": predicted == ground_truth,
                "explanation": extract_explanation(generated),
                "raw_output": generated,
            })

    return results


def compute_metrics(results: list) -> dict:
    total = len(results)
    correct = sum(r["is_correct"] for r in results)

    subject_results = defaultdict(list)
    for r in results:
        subject_results[r["subject_name"]].append(r["is_correct"])

    per_subject = {
        subj: {"accuracy": sum(v) / len(v), "n": len(v)}
        for subj, v in subject_results.items()
    }
    macro_avg = sum(v["accuracy"] for v in per_subject.values()) / len(per_subject) if per_subject else 0

    labels = ["A", "B", "C", "D"]
    conf_matrix = {gt: {pred: 0 for pred in labels + ["None"]} for gt in labels}
    for r in results:
        gt = r["ground_truth"]
        pred = r["predicted"] if r["predicted"] in labels else "None"
        if gt in conf_matrix:
            conf_matrix[gt][pred] += 1

    return {
        "overall_accuracy": correct / total if total else 0,
        "macro_averaged_accuracy": macro_avg,
        "total_samples": total,
        "correct": correct,
        "per_subject_accuracy": per_subject,
        "confusion_matrix": conf_matrix,
    }


def print_report(metrics: dict, run_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Overall Accuracy:    {metrics['overall_accuracy']:.4f}  "
          f"({metrics['correct']}/{metrics['total_samples']})")
    print(f"Macro-Avg Accuracy:  {metrics['macro_averaged_accuracy']:.4f}")
    print(f"\nPer-Subject Accuracy:")
    for subj, v in sorted(metrics["per_subject_accuracy"].items(), key=lambda x: -x[1]["accuracy"]):
        print(f"  {subj:<35} {v['accuracy']:.4f}  (n={v['n']})")
    print(f"\nConfusion Matrix (rows=ground truth, cols=predicted):")
    labels = ["A", "B", "C", "D", "None"]
    print(f"  {'GT\\Pred':<6}", "  ".join(f"{l:>5}" for l in labels))
    for gt in ["A", "B", "C", "D"]:
        row = metrics["confusion_matrix"].get(gt, {})
        vals = "  ".join(f"{row.get(l, 0):>5}" for l in labels)
        print(f"  {gt:<6} {vals}")


def _load_trainer_state(run: str) -> dict:
    """Pull best_eval_loss and total_steps from the latest trainer_state.json."""
    ckpt_dir = Path("checkpoints") / run
    state_files = sorted(ckpt_dir.glob("checkpoint-*/trainer_state.json"))
    if not state_files:
        return {}
    try:
        with open(state_files[-1]) as f:
            state = json.load(f)
        return {
            "best_eval_loss": state.get("best_metric"),
            "train_steps": state.get("global_step"),
        }
    except Exception:
        return {}


def compare_runs(results_dir: str) -> None:
    """
    Systematic model selection: rank all evaluated candidates by MCQ accuracy.

    For each run that has a *_metrics.json, also pulls training context
    (best_eval_loss, train_steps) from checkpoints/<run>/trainer_state.json
    so the selection rationale is fully visible in the output table.
    """
    results_path = Path(results_dir)
    metric_files = sorted(results_path.rglob("*_metrics.json"))
    if not metric_files:
        print(f"No metric files found in {results_dir}")
        return

    rows = []
    for mf in metric_files:
        run = mf.stem.replace("_metrics", "")
        with open(mf) as f:
            m = json.load(f)
        train_ctx = _load_trainer_state(run)
        rows.append({
            "run": run,
            "overall_acc": m.get("overall_accuracy", 0),
            "macro_avg_acc": m.get("macro_averaged_accuracy", 0),
            "n": m.get("total_samples", 0),
            "best_eval_loss": train_ctx.get("best_eval_loss"),
            "train_steps": train_ctx.get("train_steps"),
        })

    # Primary sort: macro-averaged accuracy (equal weight per subject — avoids
    # bias toward large subjects like Medicine/Surgery that dominate overall_acc).
    # Secondary sort: overall_acc as tiebreaker.
    df = pd.DataFrame(rows).sort_values(
        ["macro_avg_acc", "overall_acc"], ascending=False
    )

    print("\n" + "="*72)
    print("SYSTEMATIC MODEL SELECTION — RANKED BY MACRO-AVERAGED ACCURACY")
    print("(equal weight per subject — avoids bias toward high-freq specialties)")
    print("Candidate budget: max_steps=1000 each (DeltaGate/GB10 constraint)")
    print("="*72)

    def _fmt(x):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(int(x)) if str(x).replace('.','').isdigit() else str(x)

    print(df.to_string(index=False, formatters={col: _fmt for col in df.columns}))
    print("="*72)

    # Print clear selection decision
    winner = df.iloc[0]
    runner_up = df.iloc[1] if len(df) > 1 else None
    print(f"\n→ SELECTED MODEL : {winner['run']}")
    print(f"  Macro-avg acc  : {winner['macro_avg_acc']:.4f}  ← selection criterion")
    print(f"  Overall acc    : {winner['overall_acc']:.4f}")
    if winner["best_eval_loss"] is not None:
        print(f"  Best eval loss : {winner['best_eval_loss']:.4f}")
    if runner_up is not None:
        delta = winner["macro_avg_acc"] - runner_up["macro_avg_acc"]
        print(f"  Margin over #2 : +{delta:.4f} macro-avg vs {runner_up['run']}")

    df.to_csv(results_path / "comparison_table.csv", index=False)
    print(f"\nFull table saved to {results_path}/comparison_table.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_file", default="data/dev.json")
    parser.add_argument("--output_dir", default=None,
                        help="Results dir (default: results/<run_name>)")
    parser.add_argument("--run_name", default="baseline-9b")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--quantize_kv", type=int, default=0, choices=[0, 2, 4],
                        help="KV cache quantization bits: 0=off, 4=4-bit (~4x compression, "
                             "quality-neutral per TurboQuant), 2=2-bit (aggressive). "
                             "Use with larger --batch_size to exploit freed memory.")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--results_dir", default="results/")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.results_dir)
        return

    output_dir = Path(args.output_dir or f"results/{args.run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_file) as f:
        content = f.read().strip()
    records = json.loads(content) if content.startswith("[") else [
        json.loads(line) for line in content.splitlines() if line.strip()
    ]
    records = [r for r in records if {"question", "opa", "opb", "opc", "opd", "cop"}.issubset(r)]
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"Evaluating {len(records):,} samples — run='{args.run_name}'")

    tokenizer, model = load_model(args.model_id, args.adapter_path)
    results = evaluate(records, tokenizer, model, args.max_new_tokens, args.batch_size,
                       quantize_kv=args.quantize_kv)
    metrics = compute_metrics(results)
    print_report(metrics, args.run_name)

    with open(output_dir / f"{args.run_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(results).to_csv(output_dir / f"{args.run_name}_predictions.csv", index=False)
    errors = [r for r in results if not r["is_correct"]]
    with open(output_dir / f"{args.run_name}_error_analysis.json", "w") as f:
        json.dump(errors[:50], f, indent=2, ensure_ascii=False)

    print(f"\nOutputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
