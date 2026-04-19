"""
MCAT evaluation — transformers + peft, NVIDIA GB10 GPU (DGX Spark).

Loads LoRA checkpoints from /home/jamesoon/medmcqa/checkpoints and runs
inference directly on the GPU using BF16 (no quantization — 4-bit NF4
crashes silently on GB10 Blackwell).

NOTE: Qwen3.5-9B is intentionally excluded — its FLA/GatedDeltaNet layers
fall back to CPU on GB10 (no Triton), causing a crash in model.generate().
Use run_evaluation_gguf.py for the 9B model.

Supported models:
  - Gemma-3-4B-it + LoRA  (bf16, ~8 GB)
  - Qwen3-14B  + LoRA     (bf16, ~28 GB)

Usage:
    # Test run — 10 questions per split (default)
    PYTORCH_JIT=0 TORCHDYNAMO_DISABLE=1 python run_eval_gpu_gb10.py

    # Full run
    PYTORCH_JIT=0 TORCHDYNAMO_DISABLE=1 MAX_QUESTIONS=0 python run_eval_gpu_gb10.py

    # Single model
    PYTORCH_JIT=0 TORCHDYNAMO_DISABLE=1 MODELS=qwen3_14b_lora python run_eval_gpu_gb10.py

    # Custom checkpoint base
    PYTORCH_JIT=0 TORCHDYNAMO_DISABLE=1 CHECKPOINT_BASE=/home/jamesoon/medmcqa/checkpoints python run_eval_gpu_gb10.py
"""

import os
import re
import json
import glob
import time
import sys
import gc
from typing import Dict, List, Optional, Tuple

# ── GB10 Blackwell: disable nvrtc JIT / TorchDynamo before importing torch ──
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")

import numpy as np
import pandas as pd
import torch

# =============================================================================
# CONFIG
# =============================================================================
DATASET_BASE_DIR = "dataset_json"
RESULT_DIR       = "results_gpu_eval"
PER_MODEL_DIR    = os.path.join(RESULT_DIR, "per_model")

PREDICTIONS_PATH      = os.path.join(RESULT_DIR, "all_predictions.csv")
SUMMARY_BY_SPLIT_PATH = os.path.join(RESULT_DIR, "summary_by_split.csv")
SUMMARY_AGG_PATH      = os.path.join(RESULT_DIR, "summary_aggregated.csv")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PER_MODEL_DIR, exist_ok=True)

ALL_SPLIT_NAMES = [f"test_set_{i:02d}" for i in range(1, 8)]

EXPERIMENT_TAG  = "mcat_gpu_eval_v1"
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "512"))
# MAX_QUESTIONS: cap per split for quick smoke-test. 0 = unlimited (full run).
MAX_QUESTIONS   = int(os.getenv("MAX_QUESTIONS", "10"))
LABELS          = ["A", "B", "C", "D"]

# GB10: always BF16, never 4-bit
TORCH_DTYPE = torch.bfloat16

if not torch.cuda.is_available():
    print("ERROR: No CUDA device found. This script requires an NVIDIA GPU.", flush=True)
    sys.exit(1)

# =============================================================================
# CHECKPOINT PATHS
# =============================================================================
_DEFAULT_CHECKPOINT_BASE = "/home/jamesoon/medmcqa/checkpoints"
CHECKPOINT_BASE = os.getenv("CHECKPOINT_BASE", _DEFAULT_CHECKPOINT_BASE)

_MODEL_FILTER = set(os.getenv("MODELS", "").split(",")) - {""}

MODEL_SPECS: Dict[str, Dict] = {
    "gemma3_4b_lora": {
        "display_name": "Gemma-3-4B-it-MedMCQA-LoRA-GPU",
        "base_model_id": "google/gemma-3-4b-it",
        "adapter_path":  os.path.join(CHECKPOINT_BASE, "lora-gemma-3-4b-it", "final"),
        "param_count":   "4B",
        "quantization":  "bf16",
    },
    "qwen3_14b_lora": {
        "display_name": "Qwen3-14B-MedMCQA-LoRA-GPU",
        "base_model_id": "Qwen/Qwen3-14B",
        "adapter_path":  os.path.join(CHECKPOINT_BASE, "lora-14b", "final"),
        "param_count":   "14B",
        "quantization":  "bf16",
    },
}

if _MODEL_FILTER:
    MODEL_SPECS = {k: v for k, v in MODEL_SPECS.items() if k in _MODEL_FILTER}

# =============================================================================
# PROMPTS
# =============================================================================
SYSTEM_PROMPT = (
    "You are a helpful tutor for pre-med students preparing for medical entrance exams. "
    "Answer the following multiple choice question by thinking step by step, then give the answer."
)


def build_question_prompt(sample: Dict) -> str:
    choices  = sample.get("choices", {})
    passage  = sample.get("passage", "").strip()
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
        {"role": "user",   "content": build_question_prompt(sample)},
    ]


# =============================================================================
# DATASET
# =============================================================================
def infer_subject_category(sample: Dict) -> Optional[str]:
    for key in ["subject_category", "subject", "category", "section", "topic"]:
        v = sample.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def load_dataset(dataset_dir: str, eval_split_name: str) -> List[Dict]:
    dataset = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            sample = json.load(f)
        sample["source_json"]      = os.path.basename(path)
        sample["eval_split_name"]  = sample.get("test_set") or eval_split_name
        sample["subject_category"] = infer_subject_category(sample)
        dataset.append(sample)
    if MAX_QUESTIONS > 0:
        dataset = dataset[:MAX_QUESTIONS]
    return dataset


# =============================================================================
# MODEL LOADING / UNLOADING
# =============================================================================
def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def load_model(spec: Dict):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    free_gpu()
    base_id      = spec["base_model_id"]
    adapter_path = spec["adapter_path"]

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    print(f"    Loading tokenizer: {base_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base_id, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    print(f"    Loading base model: {base_id}  (dtype=bfloat16, device=cuda:0)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=TORCH_DTYPE,
        device_map="cuda:0",   # GB10: never use "auto" — causes CPU offload + PEFT crash
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",  # Qwen3 GatedDeltaNet not flash_attention_2 compatible
    )

    print(f"    Loading LoRA adapter: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Merge EOS token IDs so generation stops cleanly
    _chat_end = ["<|im_end|>", "<end_of_turn>", "<|endoftext|>"]
    eos_set: set = set()
    existing = model.generation_config.eos_token_id
    if isinstance(existing, list):
        eos_set.update(existing)
    elif existing is not None:
        eos_set.add(existing)
    if tokenizer.eos_token_id is not None:
        eos_set.add(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for tok_str in _chat_end:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if len(ids) == 1:
            eos_set.add(ids[0])
        if tok_str in vocab:
            eos_set.add(vocab[tok_str])
    model.generation_config.eos_token_id = sorted(eos_set)

    alloc_gb = torch.cuda.memory_allocated() / 1e9
    print(f"    Loaded. Device: {model.device}  GPU alloc: {alloc_gb:.1f} GB", flush=True)
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    free_gpu()
    alloc_gb = torch.cuda.memory_allocated() / 1e9
    print(f"    Unloaded. GPU alloc after free: {alloc_gb:.1f} GB", flush=True)


# =============================================================================
# INFERENCE
# =============================================================================
def query_model(model, tokenizer, sample: Dict) -> Tuple[Optional[str], str, Dict]:
    messages = build_messages(sample)
    text     = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs   = tokenizer(text, return_tensors="pt").to("cuda:0")
    input_len = inputs["input_ids"].shape[1]

    eos_ids = model.generation_config.eos_token_id

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
    raw_text       = tokenizer.decode(completion_ids, skip_special_tokens=True)
    comp_tokens    = len(completion_ids)

    pred = extract_letter(raw_text)
    token_info = {
        "prompt_tokens":     input_len,
        "completion_tokens": comp_tokens,
        "total_tokens":      input_len + comp_tokens,
        "finish_reason":     "stop" if comp_tokens < MAX_NEW_TOKENS else "length",
    }
    return pred, raw_text, token_info


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================
def extract_letter(raw_text: str) -> Optional[str]:
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):].strip()

    text_upper = text.upper().strip()
    if text_upper in {"A", "B", "C", "D"}:
        return text_upper

    matches = list(re.finditer(r"[Aa]nswer\s*[:\-]?\s*\**\s*([A-Da-d])\b", text))
    if matches:
        return matches[-1].group(1).upper()

    matches = list(re.finditer(r"the\s+answer\s+is\s+\**\s*([A-Da-d])\b", text, re.I))
    if matches:
        return matches[-1].group(1).upper()

    matches = list(re.finditer(r"\b([A-D])\b", text_upper))
    if matches:
        return matches[-1].group(1)

    return None


# =============================================================================
# METRICS
# =============================================================================
def compute_macro_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true, y_pred = df["gold_answer"].fillna("X"), df["pred_answer"].fillna("X")
    precisions, recalls, f1s = [], [], []
    for label in LABELS:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f1)
    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall":    float(np.mean(recalls)),
        "macro_f1":        float(np.mean(f1s)),
    }


# =============================================================================
# EVALUATION LOOP
# =============================================================================
def evaluate_model(
    model_key: str, model, tokenizer,
    dataset: List[Dict], eval_split_name: str,
) -> pd.DataFrame:
    rows, n_errors = [], 0
    spec = MODEL_SPECS[model_key]

    partial_path = os.path.join(PER_MODEL_DIR, f"{eval_split_name}__{model_key}.csv")
    done_qids: set = set()
    if os.path.exists(partial_path):
        partial_df = pd.read_csv(partial_path)
        done_qids  = set(partial_df["question_id"].dropna().tolist())
        rows       = partial_df.to_dict("records")
        print(f"  Resuming: {len(done_qids)} done, {len(dataset) - len(done_qids)} remaining", flush=True)

    for idx, sample in enumerate(dataset, start=1):
        qid = sample.get("id")
        if qid in done_qids:
            continue

        t0 = time.perf_counter()
        try:
            pred, raw_text, token_info = query_model(model, tokenizer, sample)
        except Exception as e:
            pred, raw_text = None, f"ERROR: {e}"
            token_info = {"prompt_tokens": None, "completion_tokens": None,
                          "total_tokens": None, "finish_reason": "error"}
            n_errors += 1
            print(f"  [{idx}/{len(dataset)}] {qid} | ERROR: {e}", flush=True)
            if n_errors >= 5:
                print(f"  Too many errors, stopping {model_key}.", flush=True)
                break
            continue
        latency_s = time.perf_counter() - t0
        n_errors  = 0

        gold    = (sample.get("answer") or "").strip().upper() or None
        correct = bool(pred == gold) if (pred is not None and gold is not None) else False

        comp    = token_info.get("completion_tokens")
        tps     = round(comp / latency_s, 2) if (comp and latency_s > 0) else None

        row = {
            "experiment_tag":    EXPERIMENT_TAG,
            "eval_split_name":   sample.get("eval_split_name", eval_split_name),
            "model_key":         model_key,
            "model_display_name":spec["display_name"],
            "base_model_id":     spec["base_model_id"],
            "adapter_path":      spec["adapter_path"],
            "param_count":       spec.get("param_count", ""),
            "quantization":      spec.get("quantization", ""),
            "max_new_tokens":    MAX_NEW_TOKENS,
            "question_id":       qid,
            "source_json":       sample.get("source_json"),
            "source_file":       sample.get("source_file"),
            "subject_category":  sample.get("subject_category"),
            "gold_answer":       gold,
            "pred_answer":       pred,
            "correct":           correct,
            "latency_s":         round(latency_s, 4),
            "prompt_tokens":     token_info.get("prompt_tokens"),
            "completion_tokens": token_info.get("completion_tokens"),
            "total_tokens":      token_info.get("total_tokens"),
            "tokens_per_sec":    tps,
            "finish_reason":     token_info.get("finish_reason"),
            "raw_output":        raw_text,
            "question":          sample.get("question", ""),
        }
        rows.append(row)

        marker = "OK" if correct else "WRONG"
        tps_str = f"{tps}tok/s" if tps else "-"
        subj = sample.get("subject_category", "?")
        alloc = torch.cuda.memory_allocated() / 1e9
        print(
            f"  [{idx}/{len(dataset)}] {qid} [{subj}] | gold={gold} pred={pred} {marker} "
            f"| {latency_s:.1f}s | {tps_str} | GPU {alloc:.1f}GB",
            flush=True,
        )

        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(partial_path, index=False)
        acc = df["correct"].mean()
        print(f"  => {len(df)} rows, accuracy={acc:.4f}, saved to {partial_path}", flush=True)
    else:
        print(f"  => No results for {model_key}.", flush=True)
    return df


# =============================================================================
# SUMMARIES
# =============================================================================
def upsert_predictions(run_df: pd.DataFrame) -> None:
    if run_df.empty:
        return
    key_cols = ["experiment_tag", "eval_split_name", "model_key", "question_id"]
    if os.path.exists(PREDICTIONS_PATH):
        merged = pd.concat([pd.read_csv(PREDICTIONS_PATH), run_df], ignore_index=True)
    else:
        merged = run_df.copy()
    merged = (
        merged.sort_values(by=key_cols)
              .drop_duplicates(subset=key_cols, keep="last")
              .reset_index(drop=True)
    )
    merged.to_csv(PREDICTIONS_PATH, index=False)


def build_summary_by_split(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["experiment_tag", "eval_split_name", "model_key",
                  "model_display_name", "param_count", "quantization"]
    for keys, grp in pred_df.groupby(group_cols, dropna=False):
        metrics = compute_macro_metrics(grp)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_questions":            int(len(grp)),
            "n_correct":              int(grp["correct"].sum()),
            "accuracy":               round(float(grp["correct"].mean()), 4),
            "invalid_prediction_rate":round(float((~grp["pred_answer"].isin(LABELS)).mean()), 4),
            "mean_latency_s":         round(float(grp["latency_s"].mean()), 4),
            "mean_tokens_per_sec":    round(float(grp["tokens_per_sec"].mean()), 2)
                                      if grp["tokens_per_sec"].notna().any() else None,
        })
        row.update(metrics)
        rows.append(row)
    return (pd.DataFrame(rows)
              .sort_values(["eval_split_name", "accuracy"], ascending=[True, False])
              .reset_index(drop=True))


def build_summary_aggregated(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    agg = (
        summary_df.groupby(["experiment_tag", "model_key", "model_display_name"], as_index=False)
        .agg(
            n_test_sets=("eval_split_name", "nunique"),
            total_questions=("n_questions", "sum"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_latency_s=("mean_latency_s", "mean"),
        )
    )
    agg["std_accuracy"] = agg["std_accuracy"].fillna(0.0)
    for col in ["mean_accuracy", "std_accuracy", "mean_macro_f1", "mean_latency_s"]:
        agg[col] = agg[col].round(4)
    return agg.sort_values("mean_accuracy", ascending=False).reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70, flush=True)
    print("MCAT EVALUATION — transformers + peft, NVIDIA GB10 GPU", flush=True)
    print("=" * 70, flush=True)
    print(f"Checkpoint base: {CHECKPOINT_BASE}", flush=True)
    print(f"Results dir:     {RESULT_DIR}", flush=True)
    print(f"Max new tokens:  {MAX_NEW_TOKENS}", flush=True)
    print(f"Max questions:   {'ALL' if MAX_QUESTIONS == 0 else MAX_QUESTIONS} per split", flush=True)
    print(f"Experiment tag:  {EXPERIMENT_TAG}", flush=True)
    print(f"CUDA device:     {torch.cuda.get_device_name(0)}", flush=True)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU total mem:   {total_mem_gb:.1f} GB", flush=True)
    print(flush=True)

    print("Models to evaluate:", flush=True)
    for mk, spec in MODEL_SPECS.items():
        adapter = spec["adapter_path"]
        status  = "OK" if os.path.isdir(adapter) else "MISSING"
        print(f"  {mk}: {spec['display_name']} [{status}] -> {adapter}", flush=True)
    print(flush=True)

    # Resume check
    already_done: set = set()
    if os.path.exists(PREDICTIONS_PATH):
        existing_df = pd.read_csv(PREDICTIONS_PATH)
        for (split, model_key), grp in existing_df.groupby(["eval_split_name", "model_key"]):
            expected = MAX_QUESTIONS if MAX_QUESTIONS > 0 else 225
            if len(grp) >= expected:
                already_done.add((split, model_key))

    total_combos = len(ALL_SPLIT_NAMES) * len(MODEL_SPECS)
    print(f"Total combos: {total_combos} | Already done: {len(already_done)}", flush=True)
    print(flush=True)

    for model_key, spec in MODEL_SPECS.items():
        all_done = all((s, model_key) in already_done for s in ALL_SPLIT_NAMES)
        if all_done:
            print(f"\nMODEL: {spec['display_name']} — all splits done, skipping.", flush=True)
            continue

        print(f"\n{'='*70}", flush=True)
        print(f"MODEL: {spec['display_name']}", flush=True)
        print(f"{'='*70}", flush=True)

        t_load = time.perf_counter()
        try:
            model, tokenizer = load_model(spec)
        except Exception as e:
            print(f"  FAILED to load {model_key}: {e}", flush=True)
            continue
        print(f"  Loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

        combo_num = list(MODEL_SPECS.keys()).index(model_key) * len(ALL_SPLIT_NAMES)
        for split_name in ALL_SPLIT_NAMES:
            combo_num += 1
            if (split_name, model_key) in already_done:
                print(f"\n  [{combo_num}/{total_combos}] SKIP {model_key} on {split_name}", flush=True)
                continue

            dataset_dir = os.path.join(DATASET_BASE_DIR, split_name)
            if not os.path.isdir(dataset_dir):
                print(f"  WARNING: No dataset dir for {split_name}, skipping.", flush=True)
                continue

            dataset = load_dataset(dataset_dir, split_name)
            if not dataset:
                continue

            print(f"\n  [{combo_num}/{total_combos}] START {model_key} on {split_name} ({len(dataset)} questions)", flush=True)
            t_start = time.perf_counter()
            run_df  = evaluate_model(model_key, model, tokenizer, dataset, eval_split_name=split_name)
            print(f"  [{combo_num}/{total_combos}] DONE in {time.perf_counter() - t_start:.0f}s", flush=True)

            if not run_df.empty:
                upsert_predictions(run_df)

        print(f"\n  Unloading {model_key}...", flush=True)
        unload_model(model, tokenizer)

    # Build summaries
    print(f"\n{'='*70}", flush=True)
    print("BUILDING SUMMARIES", flush=True)
    all_pred_df = pd.read_csv(PREDICTIONS_PATH) if os.path.exists(PREDICTIONS_PATH) else pd.DataFrame()

    if not all_pred_df.empty:
        summary_split_df = build_summary_by_split(all_pred_df)
        summary_agg_df   = build_summary_aggregated(summary_split_df)
        summary_split_df.to_csv(SUMMARY_BY_SPLIT_PATH, index=False)
        summary_agg_df.to_csv(SUMMARY_AGG_PATH, index=False)
        print(f"\nTotal predictions: {len(all_pred_df)}", flush=True)
        print("\nAggregated summary:", flush=True)
        print(summary_agg_df.to_string(index=False), flush=True)
        print("\nPer-split summary:", flush=True)
        cols = ["eval_split_name", "model_key", "accuracy", "macro_f1", "mean_latency_s", "mean_tokens_per_sec"]
        cols = [c for c in cols if c in summary_split_df.columns]
        print(summary_split_df[cols].to_string(index=False), flush=True)
    else:
        print("No predictions found.", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"ALL DONE — results in {RESULT_DIR}/", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
