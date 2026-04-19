"""
Standalone MCAT evaluation script — Non-CoT evaluation of finetuned LoRA models.

Loads each model via transformers + peft, evaluates on all 7 test sets,
saves results incrementally. Resume-friendly.

Models (local checkpoints from DGX training):
  - Gemma-3-4B-it + LoRA  (bf16, ~8 GB)
  - Qwen3.5-9B + LoRA     (bf16/4-bit, ~18/6 GB)
  - Qwen3-14B + LoRA      (bf16/4-bit, ~28/9 GB)

Usage:
    python run_evaluation_finetuned.py

    # Run only specific models:
    MODELS=gemma3_4b_lora,qwen3_5_9b_lora python run_evaluation_finetuned.py

    # Override checkpoint base dir:
    CHECKPOINT_BASE=/path/to/checkpoints python run_evaluation_finetuned.py
"""

import os
import re
import json
import glob
import time
import sys
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# =============================================================================
# CONFIG
# =============================================================================
DATASET_BASE_DIR = "dataset_json"
RESULT_DIR = "results_finetuned"
PER_MODEL_DIR = os.path.join(RESULT_DIR, "per_model")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")

PREDICTIONS_PATH = os.path.join(RESULT_DIR, "all_predictions.csv")
SUMMARY_BY_SPLIT_PATH = os.path.join(RESULT_DIR, "summary_by_split.csv")
SUMMARY_AGG_PATH = os.path.join(RESULT_DIR, "summary_aggregated.csv")
SUBJECT_SUMMARY_PATH = os.path.join(RESULT_DIR, "summary_by_subject.csv")
PERF_SUMMARY_PATH = os.path.join(RESULT_DIR, "performance_summary.csv")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PER_MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

ALL_SPLIT_NAMES = [f"test_set_{i:02d}" for i in range(1, 8)]

EXPERIMENT_TAG = "mcat_finetuned_v1"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
LABELS = ["A", "B", "C", "D"]

TORCH_DTYPE = torch.bfloat16

# Auto-detect device: CUDA > MPS > CPU
_HAS_CUDA = torch.cuda.is_available()
_HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
# 4-bit NF4 requires CUDA, but crashes on GB10 Blackwell (DGX Spark) — disable via USE_4BIT=0
USE_4BIT = _HAS_CUDA and os.getenv("USE_4BIT", "1") != "0"

# =============================================================================
# CHECKPOINT PATHS — local checkpoints trained on DGX8
# =============================================================================
# Override via env var: CHECKPOINT_BASE=/path python run_evaluation_finetuned.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CHECKPOINT_BASE = os.path.join(_THIS_DIR, "..", "medmcqa", "checkpoints")
CHECKPOINT_BASE = os.getenv("CHECKPOINT_BASE", _DEFAULT_CHECKPOINT_BASE)

# =============================================================================
# MODEL SPECS — ordered smallest -> largest
# =============================================================================
# Filter via env var: MODELS=gemma3_4b_lora,qwen3_5_9b_lora python run_evaluation_finetuned.py
_MODEL_FILTER = set(os.getenv("MODELS", "").split(",")) - {""}

MODEL_SPECS = {
    "gemma3_4b_lora": {
        "display_name": "Gemma-3-4B-it-MedMCQA-LoRA",
        "base_model_id": "google/gemma-3-4b-it",
        "adapter_id": os.path.join(CHECKPOINT_BASE, "lora-gemma-3-4b-it", "final"),
        "param_count": "4B",
        "quantization": "bf16",
        "use_4bit": False,
    },
    "qwen3_5_9b_lora": {
        "display_name": "Qwen3.5-9B-MedMCQA-LoRA",
        "base_model_id": "Qwen/Qwen3.5-9B",
        "adapter_id": os.path.join(CHECKPOINT_BASE, "lora-9b", "final"),
        "param_count": "9B",
        "quantization": "4bit" if USE_4BIT else "bf16",
        "use_4bit": USE_4BIT,
    },
    "qwen3_14b_lora": {
        "display_name": "Qwen3-14B-MedMCQA-LoRA",
        "base_model_id": "Qwen/Qwen3-14B",
        "adapter_id": os.path.join(CHECKPOINT_BASE, "lora-14b", "final"),
        "param_count": "14B",
        "quantization": "4bit" if USE_4BIT else "bf16",
        "use_4bit": USE_4BIT,
    },
}

# Apply model filter if specified
if _MODEL_FILTER:
    MODEL_SPECS = {k: v for k, v in MODEL_SPECS.items() if k in _MODEL_FILTER}

# =============================================================================
# PROMPT — uses model's native training format for best results
# =============================================================================
SYSTEM_PROMPT = (
    "You are a helpful tutor for pre-med students preparing for medical entrance exams. "
    "Answer the following multiple choice question by thinking step by step, then give the answer."
)


# =============================================================================
# DATASET
# =============================================================================
def infer_subject_category(sample: Dict) -> Optional[str]:
    for key in ["subject_category", "subject", "category", "section", "topic"]:
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def load_dataset(dataset_dir: str, eval_split_name: str) -> List[Dict]:
    dataset = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        sample["source_json"] = os.path.basename(path)
        sample["eval_split_name"] = sample.get("test_set") or eval_split_name
        sample["subject_category"] = infer_subject_category(sample)
        dataset.append(sample)
    return dataset


# =============================================================================
# PROMPT BUILDING — non-CoT
# =============================================================================
def build_question_prompt(sample: Dict) -> str:
    """Build prompt matching the finetuning training format."""
    choices = sample.get("choices", {})
    passage = sample.get("passage", "").strip()
    question = sample.get("question", "").strip()

    parts = []
    if passage:
        parts.append(f"Passage: {passage}\n")
    parts.append(f"Question: {question}")
    parts.append(
        f"Options: A. {choices.get('A', '')} B. {choices.get('B', '')} "
        f"C. {choices.get('C', '')} D. {choices.get('D', '')}"
    )
    parts.append(
        "Think step by step. Then respond in the format:\n"
        "Explanation: ...\nAnswer: <one of A, B, C, D>"
    )
    return "\n".join(parts)


def build_messages(sample: Dict) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_question_prompt(sample)},
    ]


# =============================================================================
# MODEL LOADING / UNLOADING
# =============================================================================
def free_gpu_memory():
    gc.collect()
    if _HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif _HAS_MPS:
        torch.mps.empty_cache()


def cleanup_incomplete_downloads():
    """Remove stale .incomplete files from HF cache to prevent stuck resumes."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if not os.path.isdir(cache_dir):
        return
    removed = 0
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".incomplete"):
                try:
                    os.remove(os.path.join(root, f))
                    removed += 1
                except OSError:
                    pass
    if removed:
        print(f"  Cleaned up {removed} stale .incomplete files from HF cache")
        sys.stdout.flush()


def load_model(spec: Dict):
    """Load base model + LoRA adapter."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    # Free GPU memory before loading
    free_gpu_memory()

    base_id = spec["base_model_id"]
    adapter_id = spec["adapter_id"]
    needs_4bit = USE_4BIT and spec.get("use_4bit", False)

    print(f"    Loading tokenizer: {base_id}")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(
        base_id, trust_remote_code=True, local_files_only=True
    )

    # DGX Spark (GB10 unified memory): device_map="auto" behaves unpredictably
    # and causes CPU offload + PEFT unhashable-set errors. Use cuda:0 directly.
    if _HAS_CUDA:
        device_map = "cuda:0"
    elif _HAS_MPS:
        device_map = "mps"
    else:
        device_map = "cpu"

    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "local_files_only": True,
    }

    if needs_4bit:
        print(f"    Loading base model: {base_id} (4-bit NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=TORCH_DTYPE,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        # Force all layers onto GPU — 4-bit doesn't support CPU offload
        if _HAS_CUDA:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            # Reserve 1.5 GB for system/overhead
            usable = int((total_mem - 1.5 * 1e9))
            load_kwargs["max_memory"] = {0: usable, "cpu": "0GiB"}
            print(f"    Max GPU memory for model: {usable / 1e9:.1f} GB")
    else:
        print(f"    Loading base model: {base_id} (dtype={TORCH_DTYPE}, device={device_map})")
        load_kwargs["dtype"] = TORCH_DTYPE  # transformers 5.x uses dtype, not torch_dtype

    sys.stdout.flush()
    model = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)

    print(f"    Loading LoRA adapter: {adapter_id}")
    sys.stdout.flush()
    model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()

    # Fix generation_config: ensure eos_token_id includes the tokenizer's im_end token
    tok_eos = tokenizer.eos_token_id
    gen_eos = model.generation_config.eos_token_id
    eos_set = set()
    if isinstance(gen_eos, list):
        eos_set.update(gen_eos)
    elif gen_eos is not None:
        eos_set.add(gen_eos)
    if tok_eos is not None:
        eos_set.add(tok_eos)
    model.generation_config.eos_token_id = sorted(eos_set)
    print(f"    EOS token IDs set to: {model.generation_config.eos_token_id}")

    print(f"    Model loaded on: {model.device}")
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        print(f"    GPU memory allocated: {alloc_gb:.1f} GB")
    sys.stdout.flush()
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    free_gpu_memory()
    print("    Model unloaded, GPU memory freed.")
    sys.stdout.flush()


# =============================================================================
# INFERENCE
# =============================================================================
# Chat-end token strings (built programmatically to avoid tooling issues)
_CHAT_END_TOKENS = [
    "<" + "|im_end|" + ">",
    "<" + "end_of_turn" + ">",
    "<" + "|endoftext|" + ">",
]


def _get_stop_token_ids(tokenizer) -> list:
    """Collect EOS and chat-end token IDs for early stopping."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    for tok_str in _CHAT_END_TOKENS:
        tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if len(tok_ids) == 1:
            ids.add(tok_ids[0])
    # Also check the vocab directly
    vocab = tokenizer.get_vocab()
    for tok_str in _CHAT_END_TOKENS:
        if tok_str in vocab:
            ids.add(vocab[tok_str])
    return sorted(ids)


def query_model(model, tokenizer, sample: Dict) -> Tuple[Optional[str], str, Dict]:
    """Run inference: generate CoT answer and extract final letter."""
    messages = build_messages(sample)

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    _device = "cuda:0" if _HAS_CUDA else ("mps" if _HAS_MPS else "cpu")
    inputs = tokenizer(text, return_tensors="pt").to(_device)
    input_len = inputs["input_ids"].shape[1]

    eos_ids = _get_stop_token_ids(tokenizer)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=eos_ids if eos_ids else None,
        )

    completion_ids = output_ids[0][input_len:]
    raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    completion_tokens = len(completion_ids)

    pred = extract_letter(raw_text)
    token_info = {
        "prompt_tokens": input_len,
        "completion_tokens": completion_tokens,
        "total_tokens": input_len + completion_tokens,
        "finish_reason": "stop" if completion_tokens < MAX_NEW_TOKENS else "length",
    }
    return pred, raw_text, token_info


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================
def extract_letter(raw_text: str) -> Optional[str]:
    if raw_text is None:
        return None
    text = str(raw_text).strip()

    # Strip <think>...</think> reasoning blocks (Qwen3 may still produce these)
    think_end = text.rfind('</think>')
    if think_end != -1:
        text = text[think_end + len('</think>'):].strip()

    text_upper = text.upper().strip()

    # Direct single letter
    if text_upper in {"A", "B", "C", "D"}:
        return text_upper

    # Find the LAST "Answer: X" pattern (model puts it at the end)
    matches = list(re.finditer(r'[Aa]nswer\s*[:\-]?\s*\**\s*([A-Da-d])\b', text))
    if matches:
        return matches[-1].group(1).upper()

    # "The answer is X" — last occurrence
    matches = list(re.finditer(r'the\s+answer\s+is\s+\**\s*([A-Da-d])\b', text, re.I))
    if matches:
        return matches[-1].group(1).upper()

    # LAST standalone A-D letter (not first — avoids option labels in explanation)
    matches = list(re.finditer(r'\b([A-D])\b', text_upper))
    if matches:
        return matches[-1].group(1)

    return None


# =============================================================================
# METRICS
# =============================================================================
def compute_macro_classification_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["gold_answer"].fillna("INVALID")
    y_pred = df["pred_answer"].fillna("INVALID")
    precisions, recalls, f1s = [], [], []
    for label in LABELS:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


# =============================================================================
# EVALUATION LOOP
# =============================================================================
def evaluate_model(
    model_key: str, model, tokenizer, dataset: List[Dict], eval_split_name: str,
) -> pd.DataFrame:
    rows = []
    n_errors = 0
    spec = MODEL_SPECS[model_key]

    print(f'\n  Evaluating {model_key} on {eval_split_name}')
    sys.stdout.flush()

    # Per-question resume
    partial_path = os.path.join(PER_MODEL_DIR, f'{eval_split_name}__{model_key}.csv')
    done_qids = set()
    if os.path.exists(partial_path):
        partial_df = pd.read_csv(partial_path)
        done_qids = set(partial_df["question_id"].dropna().tolist())
        rows = partial_df.to_dict("records")
        print(f'  Resuming: {len(done_qids)} done, {len(dataset) - len(done_qids)} remaining')
        sys.stdout.flush()

    for idx, sample in enumerate(dataset, start=1):
        qid = sample.get("id")
        if qid in done_qids:
            continue

        t0 = time.perf_counter()
        try:
            pred, raw_text, token_info = query_model(model, tokenizer, sample)
        except Exception as e:
            pred, raw_text = None, f"ERROR: {e}"
            token_info = {
                "prompt_tokens": None, "completion_tokens": None,
                "total_tokens": None, "finish_reason": "error",
            }
            n_errors += 1
            print(f'  [{idx}/{len(dataset)}] {sample.get("id")} | ERROR: {e}')
            sys.stdout.flush()
            if n_errors >= 5:
                print(f'  Too many errors ({n_errors}), stopping {model_key}.')
                sys.stdout.flush()
                break
            continue
        latency_s = time.perf_counter() - t0
        n_errors = 0

        gold = (sample.get("answer") or "").strip().upper() or None
        correct = bool(pred == gold) if (pred is not None and gold is not None) else False

        comp_tokens = token_info.get("completion_tokens")
        tokens_per_sec = round(comp_tokens / latency_s, 2) if (comp_tokens and latency_s > 0) else None

        row = {
            "experiment_tag": EXPERIMENT_TAG,
            "eval_split_name": sample.get("eval_split_name", eval_split_name),
            "model_key": model_key,
            "model_display_name": spec["display_name"],
            "base_model_id": spec["base_model_id"],
            "adapter_id": spec["adapter_id"],
            "param_count": spec.get("param_count", ""),
            "quantization": spec.get("quantization", ""),
            "max_new_tokens": MAX_NEW_TOKENS,
            "question_id": sample.get("id"),
            "source_json": sample.get("source_json"),
            "source_file": sample.get("source_file"),
            "subject_category": sample.get("subject_category"),
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": correct,
            "latency_s": round(latency_s, 4),
            "prompt_tokens": token_info.get("prompt_tokens"),
            "completion_tokens": token_info.get("completion_tokens"),
            "total_tokens": token_info.get("total_tokens"),
            "tokens_per_sec": tokens_per_sec,
            "finish_reason": token_info.get("finish_reason"),
            "raw_output": raw_text,
            "question": sample.get("question", ""),
        }
        rows.append(row)

        marker = "OK" if correct else "WRONG"
        tps_str = f"{tokens_per_sec}tok/s" if tokens_per_sec else "-"
        subj = sample.get("subject_category", "?")
        print(f'  [{idx}/{len(dataset)}] {sample.get("id")} [{subj}] | gold={gold} pred={pred} {marker} | {latency_s:.1f}s | {tps_str}')
        sys.stdout.flush()

        # Incremental save every 5 questions
        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(partial_path, index=False)
        acc = df["correct"].mean()
        print(f'  => {len(df)} rows, accuracy={acc:.4f}, saved to {partial_path}')
    else:
        print(f'  => No results for {model_key}.')
    sys.stdout.flush()
    return df


# =============================================================================
# SUMMARIES
# =============================================================================
def upsert_predictions(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return run_df.copy()
    key_cols = ["experiment_tag", "eval_split_name", "model_key", "question_id"]
    if os.path.exists(PREDICTIONS_PATH):
        existing = pd.read_csv(PREDICTIONS_PATH)
        merged = pd.concat([existing, run_df], ignore_index=True)
    else:
        merged = run_df.copy()
    merged = (
        merged.sort_values(by=key_cols)
              .drop_duplicates(subset=key_cols, keep="last")
              .reset_index(drop=True)
    )
    merged.to_csv(PREDICTIONS_PATH, index=False)
    return merged


def build_summary_by_split(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "experiment_tag", "eval_split_name", "model_key", "model_display_name",
        "base_model_id", "adapter_id", "param_count", "quantization",
    ]
    for keys, grp in pred_df.groupby(group_cols, dropna=False):
        metrics = compute_macro_classification_metrics(grp)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_questions": int(len(grp)),
            "n_correct": int(grp["correct"].sum()),
            "accuracy": float(grp["correct"].mean()),
            "invalid_prediction_rate": float((~grp["pred_answer"].isin(LABELS)).mean()),
            "mean_latency_s": float(grp["latency_s"].mean()),
            "median_latency_s": float(grp["latency_s"].median()),
            "p90_latency_s": float(grp["latency_s"].quantile(0.90)),
        })
        if "prompt_tokens" in grp.columns and grp["prompt_tokens"].notna().any():
            row["mean_prompt_tokens"] = float(grp["prompt_tokens"].mean())
            row["mean_completion_tokens"] = float(grp["completion_tokens"].mean())
            row["mean_total_tokens"] = float(grp["total_tokens"].mean())
        if "tokens_per_sec" in grp.columns and grp["tokens_per_sec"].notna().any():
            row["mean_tokens_per_sec"] = float(grp["tokens_per_sec"].mean())
            row["median_tokens_per_sec"] = float(grp["tokens_per_sec"].median())
        row.update(metrics)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        by=["eval_split_name", "accuracy", "macro_f1", "mean_latency_s"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    for col in ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                 "invalid_prediction_rate", "mean_latency_s", "median_latency_s", "p90_latency_s"]:
        if col in summary.columns:
            summary[col] = summary[col].round(4)
    return summary


def build_summary_aggregated(summary_by_split_df: pd.DataFrame) -> pd.DataFrame:
    if summary_by_split_df.empty:
        return pd.DataFrame()
    agg = (
        summary_by_split_df.groupby(
            ["experiment_tag", "model_key", "model_display_name", "adapter_id"],
            as_index=False,
        ).agg(
            n_test_sets=("eval_split_name", "nunique"),
            total_questions=("n_questions", "sum"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_precision=("macro_precision", "mean"),
            mean_macro_recall=("macro_recall", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_invalid_prediction_rate=("invalid_prediction_rate", "mean"),
            mean_latency_s=("mean_latency_s", "mean"),
            median_latency_s=("median_latency_s", "mean"),
            mean_p90_latency_s=("p90_latency_s", "mean"),
        )
    )
    agg["std_accuracy"] = agg["std_accuracy"].fillna(0.0)
    for col in ["mean_accuracy", "std_accuracy", "mean_macro_precision", "mean_macro_recall",
                 "mean_macro_f1", "mean_invalid_prediction_rate", "mean_latency_s",
                 "median_latency_s", "mean_p90_latency_s"]:
        agg[col] = agg[col].round(4)
    agg = agg.sort_values(
        by=["mean_accuracy", "mean_macro_f1", "mean_latency_s"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return agg


def build_subject_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or "subject_category" not in pred_df.columns:
        return pd.DataFrame()
    subject_df = pred_df.copy()
    subject_df["subject_category"] = subject_df["subject_category"].fillna("Unknown")
    rows = []
    group_cols = ["experiment_tag", "eval_split_name", "model_key", "model_display_name", "subject_category"]
    for keys, grp in subject_df.groupby(group_cols, dropna=False):
        metrics = compute_macro_classification_metrics(grp)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_questions": int(len(grp)),
            "accuracy": float(grp["correct"].mean()),
            "mean_latency_s": float(grp["latency_s"].mean()),
        })
        row.update(metrics)
        rows.append(row)
    subject_summary = pd.DataFrame(rows).sort_values(
        by=["subject_category", "accuracy"], ascending=[True, False],
    ).reset_index(drop=True)
    for col in ["accuracy", "mean_latency_s", "macro_precision", "macro_recall", "macro_f1"]:
        subject_summary[col] = subject_summary[col].round(4)
    return subject_summary


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("MCAT EVALUATION — Finetuned LoRA Models (Non-CoT, Direct Answer)")
    print(f"{len(MODEL_SPECS)} models x {len(ALL_SPLIT_NAMES)} test sets")
    print("=" * 70)
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Experiment:  {EXPERIMENT_TAG}")
    print(f"Results dir: {RESULT_DIR}")
    print(f"Checkpoint base: {CHECKPOINT_BASE}")
    print(f"Torch dtype: {TORCH_DTYPE}")
    print(f"4-bit quant: {USE_4BIT}")
    if _HAS_CUDA:
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {total_mem:.1f} GB")
    elif _HAS_MPS:
        print("Device: MPS (Apple Silicon)")
    else:
        print("Device: CPU")
    print()

    # Clean up stale incomplete downloads
    cleanup_incomplete_downloads()

    print("Models to evaluate:")
    for mk, spec in MODEL_SPECS.items():
        q = spec["quantization"]
        print(f"  {mk}: {spec['display_name']} ({spec['param_count']}, {q})")
    print()

    # Seed all_predictions.csv from existing partial CSVs
    partial_csvs = glob.glob(os.path.join(PER_MODEL_DIR, "*.csv"))
    if partial_csvs and not os.path.exists(PREDICTIONS_PATH):
        print("Seeding all_predictions.csv from partial per-model CSVs...")
        parts = [pd.read_csv(p) for p in partial_csvs]
        seed_df = pd.concat(parts, ignore_index=True)
        seed_df.to_csv(PREDICTIONS_PATH, index=False)
        print(f"  Seeded {len(seed_df)} rows from {len(partial_csvs)} files")

    # Resume check
    already_done = set()
    if os.path.exists(PREDICTIONS_PATH):
        existing_df = pd.read_csv(PREDICTIONS_PATH)
        for (split, model), grp in existing_df.groupby(["eval_split_name", "model_key"]):
            if len(grp) >= 225:
                already_done.add((split, model))

    total_combos = len(ALL_SPLIT_NAMES) * len(MODEL_SPECS)
    remaining = total_combos - len(already_done)
    print(f"Total combos: {total_combos} | Already done: {len(already_done)} | Remaining: {remaining}")
    if already_done:
        print("Completed:")
        for combo in sorted(already_done):
            print(f"  + {combo[0]} / {combo[1]}")
    print()
    sys.stdout.flush()

    # =========================================================================
    # OUTER LOOP: one model at a time, evaluate all splits, then unload
    # =========================================================================
    combo_num = 0
    for model_key, spec in MODEL_SPECS.items():

        all_splits_done = all((s, model_key) in already_done for s in ALL_SPLIT_NAMES)
        if all_splits_done:
            combo_num += len(ALL_SPLIT_NAMES)
            print(f"\n{'='*70}")
            print(f"MODEL: {spec['display_name']} — ALL SPLITS DONE, skipping")
            print(f"{'='*70}")
            sys.stdout.flush()
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {spec['display_name']} ({spec['param_count']}, {spec['quantization']})")
        print(f"{'='*70}")
        sys.stdout.flush()

        t_load = time.perf_counter()
        try:
            model, tokenizer = load_model(spec)
        except Exception as e:
            print(f"  FAILED to load {model_key}: {e}")
            if "google/gemma" in spec["base_model_id"]:
                print(f"  -> Gemma is a gated model. Accept the license at:")
                print(f"     https://huggingface.co/google/gemma-3-4b-it")
            print(f"  Skipping this model.")
            sys.stdout.flush()
            combo_num += len(ALL_SPLIT_NAMES)
            continue
        load_time = time.perf_counter() - t_load
        print(f"    Loaded in {load_time:.1f}s")
        sys.stdout.flush()

        # Inner loop: test sets
        for split_name in ALL_SPLIT_NAMES:
            combo_num += 1
            if (split_name, model_key) in already_done:
                print(f"\n  [{combo_num}/{total_combos}] SKIP {model_key} on {split_name} (already done)")
                sys.stdout.flush()
                continue

            dataset_dir = os.path.join(DATASET_BASE_DIR, split_name)
            if not os.path.isdir(dataset_dir):
                print(f"  WARNING: No dataset directory for {split_name}, skipping.")
                continue

            dataset = load_dataset(dataset_dir, split_name)
            if len(dataset) == 0:
                print(f"  No questions for {split_name}, skipping.")
                continue

            print(f"\n  [{combo_num}/{total_combos}] START {model_key} on {split_name} ({len(dataset)} questions)")
            sys.stdout.flush()
            t_start = time.perf_counter()

            run_df = evaluate_model(model_key, model, tokenizer, dataset, eval_split_name=split_name)

            elapsed = time.perf_counter() - t_start
            print(f"  [{combo_num}/{total_combos}] DONE {model_key} on {split_name} in {elapsed:.0f}s")
            sys.stdout.flush()

            if not run_df.empty:
                upsert_predictions(run_df)

        # Unload before next model
        print(f"\n  Unloading {model_key}...")
        sys.stdout.flush()
        unload_model(model, tokenizer)

    # =========================================================================
    # BUILD SUMMARIES
    # =========================================================================
    print(f"\n{'='*70}")
    print("BUILDING SUMMARIES")
    print(f"{'='*70}")

    all_pred_df = pd.read_csv(PREDICTIONS_PATH) if os.path.exists(PREDICTIONS_PATH) else pd.DataFrame()

    if not all_pred_df.empty:
        summary_by_split_df = build_summary_by_split(all_pred_df)
        summary_agg_df = build_summary_aggregated(summary_by_split_df)
        subject_summary_df = build_subject_summary(all_pred_df)

        summary_by_split_df.to_csv(SUMMARY_BY_SPLIT_PATH, index=False)
        summary_agg_df.to_csv(SUMMARY_AGG_PATH, index=False)
        subject_summary_df.to_csv(SUBJECT_SUMMARY_PATH, index=False)

        print(f"\nTotal predictions: {len(all_pred_df)}")
        print(f"\nAggregated summary:")
        print(summary_agg_df.to_string(index=False))
        print(f"\nSummary by split:")
        cols = ["eval_split_name", "model_key", "accuracy", "macro_f1", "mean_latency_s"]
        avail = [c for c in cols if c in summary_by_split_df.columns]
        print(summary_by_split_df[avail].to_string(index=False))
    else:
        print("No predictions found.")

    print(f"\n{'='*70}")
    print("ALL DONE — Finetuned model evaluation complete.")
    print(f"Results saved to: {RESULT_DIR}/")
    print(f"{'='*70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
