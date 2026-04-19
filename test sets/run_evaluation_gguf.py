"""
MCAT evaluation script — GGUF models via llama-cpp-python.

Uses llama-cpp-python for inference (no Triton/FLA required), works on GB10.

Models:
  - Qwen3.5-9B-MedMCQA Q4_K_M  (GGUF)
  - Qwen3-14B-MedMCQA Q4_K_M   (GGUF, optional)

Usage:
    python run_evaluation_gguf.py

    # Run only specific models:
    MODELS=qwen3_5_9b_gguf python run_evaluation_gguf.py

    # Override GGUF base dir:
    GGUF_BASE=/path/to/quantized python run_evaluation_gguf.py

    # Limit GPU layers (default: all):
    N_GPU_LAYERS=40 python run_evaluation_gguf.py
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

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PER_MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

ALL_SPLIT_NAMES = [f"test_set_{i:02d}" for i in range(1, 8)]

EXPERIMENT_TAG = "mcat_finetuned_gguf_v1"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))   # -1 = all layers on GPU
N_CTX = int(os.getenv("N_CTX", "4096"))
LABELS = ["A", "B", "C", "D"]

# =============================================================================
# GGUF MODEL PATHS
# =============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_GGUF_BASE = os.path.join(_THIS_DIR, "..", "medmcqa", "quantized")
GGUF_BASE = os.getenv("GGUF_BASE", _DEFAULT_GGUF_BASE)

_MODEL_FILTER = set(os.getenv("MODELS", "").split(",")) - {""}

MODEL_SPECS = {
    "qwen3_5_9b_gguf": {
        "display_name": "Qwen3.5-9B-MedMCQA-Q4_K_M",
        "gguf_path": os.path.join(GGUF_BASE, "lora-9b", "lora-9b-medmcqa-q4_k_m.gguf"),
        "param_count": "9B",
        "quantization": "Q4_K_M",
        "chat_format": "qwen",
    },
    "qwen3_14b_gguf": {
        "display_name": "Qwen3-14B-MedMCQA-Q4_K_M",
        "gguf_path": os.path.join(GGUF_BASE, "lora-14b", "lora-14b-medmcqa-q4_k_m.gguf"),
        "param_count": "14B",
        "quantization": "Q4_K_M",
        "chat_format": "qwen",
    },
}

if _MODEL_FILTER:
    MODEL_SPECS = {k: v for k, v in MODEL_SPECS.items() if k in _MODEL_FILTER}

# =============================================================================
# PROMPTS
# =============================================================================
SYSTEM_PROMPT = (
    "You are a medical exam tutor. "
    "Do NOT repeat or restate the question or options. "
    "Give a brief explanation (1-2 sentences), then state the answer. "
    "Always end with exactly: Answer: <A, B, C, or D>"
)


def build_question_prompt(sample: Dict) -> str:
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
    parts.append("Respond in the format:\nExplanation: ...\nAnswer: <A, B, C, or D>")
    return "\n".join(parts)


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
# MODEL LOADING
# =============================================================================
def load_model(spec: Dict):
    from llama_cpp import Llama
    gguf_path = spec["gguf_path"]
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")
    print(f"    Loading GGUF: {gguf_path}")
    print(f"    n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}")
    sys.stdout.flush()
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
    )
    return llm


def unload_model(llm):
    del llm
    gc.collect()
    print("    Model unloaded.")
    sys.stdout.flush()


# =============================================================================
# INFERENCE
# =============================================================================
def query_model(llm, sample: Dict) -> Tuple[Optional[str], str, Dict]:
    # Pre-fill the assistant turn with "Explanation:" so the model skips all
    # preamble and continues directly with a short explanation + "Answer: X".
    # This is much more reliable than system prompt instructions for GGUF models.
    user_content = build_question_prompt(sample)
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\nExplanation:"
    )
    response = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stop=["<|im_end|>", "<|endoftext|>", "<end_of_turn>"],
        echo=False,
    )
    completion_text = response["choices"][0]["text"]
    raw_text = "Explanation:" + completion_text
    usage = response.get("usage", {})
    finish_reason = response["choices"][0].get("finish_reason", "unknown")
    pred = extract_letter(raw_text)
    token_info = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": finish_reason,
    }
    return pred, raw_text, token_info


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================
def extract_letter(raw_text: str) -> Optional[str]:
    if raw_text is None:
        return None
    text = str(raw_text).strip()

    # Strip <think>...</think> blocks
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
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


# =============================================================================
# EVALUATION LOOP
# =============================================================================
def evaluate_model(model_key: str, llm, dataset: List[Dict], eval_split_name: str) -> pd.DataFrame:
    rows = []
    n_errors = 0
    spec = MODEL_SPECS[model_key]

    print(f"\n  Evaluating {model_key} on {eval_split_name}")
    sys.stdout.flush()

    partial_path = os.path.join(PER_MODEL_DIR, f"{eval_split_name}__{model_key}.csv")
    done_qids = set()
    if os.path.exists(partial_path):
        partial_df = pd.read_csv(partial_path)
        done_qids = set(partial_df["question_id"].dropna().tolist())
        rows = partial_df.to_dict("records")
        print(f"  Resuming: {len(done_qids)} done, {len(dataset) - len(done_qids)} remaining")
        sys.stdout.flush()

    for idx, sample in enumerate(dataset, start=1):
        qid = sample.get("id")
        if qid in done_qids:
            continue

        t0 = time.perf_counter()
        try:
            pred, raw_text, token_info = query_model(llm, sample)
        except Exception as e:
            pred, raw_text = None, f"ERROR: {e}"
            token_info = {"prompt_tokens": None, "completion_tokens": None,
                          "total_tokens": None, "finish_reason": "error"}
            n_errors += 1
            print(f"  [{idx}/{len(dataset)}] {qid} | ERROR: {e}")
            sys.stdout.flush()
            if n_errors >= 5:
                print(f"  Too many errors ({n_errors}), stopping {model_key}.")
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
            "base_model_id": spec.get("gguf_path", ""),
            "adapter_id": spec.get("gguf_path", ""),
            "param_count": spec.get("param_count", ""),
            "quantization": spec.get("quantization", ""),
            "max_new_tokens": MAX_TOKENS,
            "question_id": qid,
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
        print(f"  [{idx}/{len(dataset)}] {qid} [{subj}] | gold={gold} pred={pred} {marker} | {latency_s:.1f}s | {tps_str}")
        sys.stdout.flush()

        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(partial_path, index=False)
        acc = df["correct"].mean()
        print(f"  => {len(df)} rows, accuracy={acc:.4f}")
    else:
        print(f"  => No results for {model_key}.")
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
    group_cols = ["experiment_tag", "eval_split_name", "model_key", "model_display_name",
                  "param_count", "quantization"]
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
        if "tokens_per_sec" in grp.columns and grp["tokens_per_sec"].notna().any():
            row["mean_tokens_per_sec"] = float(grp["tokens_per_sec"].mean())
        row.update(metrics)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        by=["eval_split_name", "accuracy"], ascending=[True, False]
    ).reset_index(drop=True)
    for col in ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                "invalid_prediction_rate", "mean_latency_s"]:
        if col in summary.columns:
            summary[col] = summary[col].round(4)
    return summary


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
    return pd.DataFrame(rows).sort_values(
        by=["subject_category", "accuracy"], ascending=[True, False]
    ).reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("MCAT EVALUATION — GGUF Models via llama-cpp-python")
    print(f"{len(MODEL_SPECS)} models x {len(ALL_SPLIT_NAMES)} test sets")
    print("=" * 70)
    print(f"GGUF base:   {GGUF_BASE}")
    print(f"Results dir: {RESULT_DIR}")
    print(f"Max tokens:  {MAX_TOKENS}")
    print(f"n_gpu_layers:{N_GPU_LAYERS}  n_ctx:{N_CTX}")
    print(f"Experiment:  {EXPERIMENT_TAG}")
    print()

    print("Models to evaluate:")
    for mk, spec in MODEL_SPECS.items():
        gguf = spec["gguf_path"]
        exists = "OK" if os.path.exists(gguf) else "MISSING"
        print(f"  {mk}: {spec['display_name']} ({spec['param_count']}, {spec['quantization']}) [{exists}]")
    print()

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
    print()
    sys.stdout.flush()

    for model_key, spec in MODEL_SPECS.items():
        all_splits_done = all((s, model_key) in already_done for s in ALL_SPLIT_NAMES)
        if all_splits_done:
            print(f"\n{'='*70}")
            print(f"MODEL: {spec['display_name']} — ALL SPLITS DONE, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {spec['display_name']} ({spec['param_count']}, {spec['quantization']})")
        print(f"{'='*70}")
        sys.stdout.flush()

        t_load = time.perf_counter()
        try:
            llm = load_model(spec)
        except Exception as e:
            print(f"  FAILED to load {model_key}: {e}")
            sys.stdout.flush()
            continue
        print(f"    Loaded in {time.perf_counter() - t_load:.1f}s")
        sys.stdout.flush()

        combo_num = list(MODEL_SPECS.keys()).index(model_key) * len(ALL_SPLIT_NAMES)
        for split_name in ALL_SPLIT_NAMES:
            combo_num += 1
            if (split_name, model_key) in already_done:
                print(f"\n  [{combo_num}/{total_combos}] SKIP {model_key} on {split_name}")
                sys.stdout.flush()
                continue

            dataset_dir = os.path.join(DATASET_BASE_DIR, split_name)
            if not os.path.isdir(dataset_dir):
                print(f"  WARNING: No dataset dir for {split_name}")
                continue

            dataset = load_dataset(dataset_dir, split_name)
            if not dataset:
                continue

            print(f"\n  [{combo_num}/{total_combos}] START {model_key} on {split_name} ({len(dataset)} questions)")
            sys.stdout.flush()
            t_start = time.perf_counter()
            run_df = evaluate_model(model_key, llm, dataset, eval_split_name=split_name)
            elapsed = time.perf_counter() - t_start
            print(f"  [{combo_num}/{total_combos}] DONE in {elapsed:.0f}s")
            sys.stdout.flush()

            if not run_df.empty:
                upsert_predictions(run_df)

        print(f"\n  Unloading {model_key}...")
        sys.stdout.flush()
        unload_model(llm)

    # Summaries
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
    else:
        print("No predictions found.")

    print(f"\n{'='*70}")
    print("ALL DONE — GGUF evaluation complete.")
    print(f"Results saved to: {RESULT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
