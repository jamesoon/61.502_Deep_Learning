"""
Standalone MCAT evaluation script — CoT (Chain-of-Thought) prompting variant.
Uses subject-specific prompts for CARS, BB, CP, PS sections.
Resume-friendly: skips (split, model) combos that already have results.

Usage:
    python run_evaluation_cot.py
"""

import os
import re
import json
import glob
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional

import concurrent.futures
import numpy as np
import pandas as pd
from openai import OpenAI

# =============================================================================
# CONFIG
# =============================================================================
LOCAL_API_BASE_URL = os.getenv("LOCAL_API_BASE_URL", "http://127.0.0.1:1234/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "not-needed")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 min max per request

client = OpenAI(base_url=LOCAL_API_BASE_URL, api_key=LOCAL_API_KEY, timeout=REQUEST_TIMEOUT)

DATASET_BASE_DIR = "dataset_json"
RESULT_DIR = "results_local_llm_cot"
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

EXPERIMENT_TAG = "mcat_local_benchmark_v3_cot"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))  # CoT reasoning + answer (no internal think blocks)
SLEEP_BETWEEN_REQUESTS = float(os.getenv("SLEEP_BETWEEN_REQUESTS", "0.0"))

LABELS = ["A", "B", "C", "D"]

# Ordered fastest → slowest so we get results quickly
MODEL_SPECS = {
    "qwen3_5_9b": {
        "display_name": "Qwen3.5-9B-Q4_K_M",
        "hf_repo": "Qwen/Qwen3.5-9B-GGUF",
        "hf_file": "Qwen3.5-9B-Q4_K_M.gguf",
        "match_tokens": ["qwen3", "5", "9b"],
        "param_count": "9B",
        "quantization": "Q4_K_M",
    },
}

# =============================================================================
# SUBJECT-SPECIFIC CoT PROMPTS
# =============================================================================

SUBJECT_SYSTEM_PROMPTS = {
    "CARS": (
        "You are an expert MCAT tutor specializing in Critical Analysis and Reasoning Skills (CARS). "
        "Answer questions by reasoning carefully through the passage."
    ),
    "BB": (
        "You are an expert MCAT tutor specializing in Biological and Biochemical Foundations of Living Systems. "
        "Answer questions by applying your knowledge of biology and biochemistry."
    ),
    "CP": (
        "You are an expert MCAT tutor specializing in Chemical and Physical Foundations of Biological Systems. "
        "Answer questions by applying your knowledge of chemistry and physics."
    ),
    "PS": (
        "You are an expert MCAT tutor specializing in Psychological, Social, and Biological Foundations of Behavior. "
        "Answer questions by applying your knowledge of psychology and sociology."
    ),
}

SUBJECT_COT_INSTRUCTIONS = {
    "CARS": (
        "Answer the MCAT CARS question step by step.\n\n"
        "Rules:\n"
        "1. Identify the author's main point first.\n"
        "2. Determine what the question is really asking.\n"
        "3. Find the key lines or ideas from the passage that matter most.\n"
        "4. Evaluate each answer choice against the passage, not outside knowledge.\n"
        "5. Eliminate wrong choices briefly.\n"
        "6. Give the final answer as a single option letter.\n\n"
        "Be concise but explicit."
    ),
    "BB": (
        "Answer the MCAT Bio/Biochem question step by step.\n\n"
        "Rules:\n"
        "1. Identify the core biological or biochemical concept first.\n"
        "2. Determine whether the question depends on passage data, background knowledge, or both.\n"
        "3. Extract the key experimental details, variables, or relationships from the passage/question.\n"
        "4. Link the evidence to the biological mechanism.\n"
        "5. Eliminate wrong choices briefly.\n"
        "6. Give the final answer as a single option letter.\n\n"
        "Be concise but explicit."
    ),
    "CP": (
        "You are solving an MCAT Chem/Phys question.\n\n"
        "Return your answer in this format:\n"
        "Concept:\n"
        "Equation/Principle:\n"
        "Given information:\n"
        "Solve:\n"
        "Unit check:\n"
        "Eliminate choices:\n"
        "Final answer:\n\n"
        "Keep each line short and precise.\n"
        "Do not skip unit analysis."
    ),
    "PS": (
        "Answer the MCAT Psych/Soc question step by step.\n\n"
        "Rules:\n"
        "1. Identify the core psychological or sociological concept first.\n"
        "2. Determine whether the question is asking for a definition, application, or distinction between concepts.\n"
        "3. Extract the relevant clues from the passage/question.\n"
        "4. Match the clues to the best-fitting term or principle.\n"
        "5. Eliminate wrong choices briefly, especially close distractors.\n"
        "6. Give the final answer as a single option letter.\n\n"
        "Be concise but explicit."
    ),
}

# Fallback for unknown subject categories
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert MCAT tutor. Answer the question step by step, "
    "then give the final answer as a single option letter (A, B, C, or D)."
)
DEFAULT_COT_INSTRUCTIONS = (
    "Answer the MCAT question step by step.\n\n"
    "Rules:\n"
    "1. Identify the core concept.\n"
    "2. Extract the key details from the passage/question.\n"
    "3. Evaluate each answer choice.\n"
    "4. Eliminate wrong choices briefly.\n"
    "5. Give the final answer as a single option letter.\n\n"
    "Be concise but explicit."
)


# =============================================================================
# FUNCTIONS
# =============================================================================
def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def list_server_models() -> List[str]:
    try:
        response = client.models.list()
        data = getattr(response, "data", []) or []
        return [getattr(item, "id", str(item)) for item in data]
    except Exception as e:
        print(f"Could not list models from local server: {e}")
        return []


def resolve_loaded_model_id(spec: Dict, available_model_ids: List[str]) -> Optional[str]:
    if not available_model_ids:
        return None
    normalized_tokens = [normalize_text(token) for token in spec["match_tokens"]]
    for model_id in available_model_ids:
        normalized_id = normalize_text(model_id)
        if all(token in normalized_id for token in normalized_tokens):
            return model_id
    return None


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


def build_question_prompt(sample: Dict) -> str:
    """Build the user prompt with subject-specific CoT instructions."""
    choices = sample.get("choices", {})
    subject = sample.get("subject_category", "").upper()
    cot_instructions = SUBJECT_COT_INSTRUCTIONS.get(subject, DEFAULT_COT_INSTRUCTIONS)

    return f"""/no_think
{cot_instructions}

Passage:
{sample.get('passage', '').strip()}

Question:
{sample.get('question', '').strip()}

Choices:
A. {choices.get('A', '')}
B. {choices.get('B', '')}
C. {choices.get('C', '')}
D. {choices.get('D', '')}
"""


def get_system_prompt(sample: Dict) -> str:
    """Get subject-specific system prompt."""
    subject = sample.get("subject_category", "").upper()
    return SUBJECT_SYSTEM_PROMPTS.get(subject, DEFAULT_SYSTEM_PROMPT)


def extract_letter(raw_text: str) -> Optional[str]:
    """Extract the predicted answer letter from CoT output."""
    if raw_text is None:
        return None
    text = str(raw_text).strip()

    # Strip Qwen3 <think>...</think> reasoning blocks
    think_end = text.rfind('</think>')
    if think_end != -1:
        text = text[think_end + len('</think>'):].strip()

    # Try "Final answer: X" pattern first (most structured)
    match = re.search(r'[Ff]inal\s+[Aa]nswer\s*[:\-]?\s*\**\s*([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()

    # Try "Answer: X" pattern
    match = re.search(r'[Aa]nswer\s*[:\-]?\s*\**\s*([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()

    # Try "the answer is X"
    match = re.search(r'the\s+answer\s+is\s+\**\s*([A-Da-d])\b', text, re.I)
    if match:
        return match.group(1).upper()

    # Check last line for a standalone letter
    last_lines = text.strip().split('\n')
    for line in reversed(last_lines[-3:]):
        line = line.strip().rstrip('.').strip()
        # Remove bold markers
        line = re.sub(r'\*+', '', line).strip()
        if line.upper() in {"A", "B", "C", "D"}:
            return line.upper()

    # Fallback: last single capital letter A-D in text
    matches = re.findall(r'\b([A-D])\b', text.upper())
    if matches:
        return matches[-1]

    return None


def _raw_query(server_model_id: str, system_prompt: str, user_prompt: str):
    """Inner function that makes the actual API call (runs in thread for timeout)."""
    response = client.chat.completions.create(
        model=server_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response


HARD_TIMEOUT = int(os.getenv("HARD_TIMEOUT", "360"))  # hard kill after this many seconds


def query_local_model(server_model_id: str, sample: Dict):
    system_prompt = get_system_prompt(sample)
    user_prompt = build_question_prompt(sample)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_raw_query, server_model_id, system_prompt, user_prompt)
        try:
            response = future.result(timeout=HARD_TIMEOUT)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Hard timeout after {HARD_TIMEOUT}s")

    raw_text = response.choices[0].message.content
    pred = extract_letter(raw_text)
    finish_reason = response.choices[0].finish_reason
    usage = response.usage
    token_info = {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "finish_reason": finish_reason,
    }
    return pred, raw_text, token_info


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


def evaluate_model(model_key: str, server_model_id: str, dataset: List[Dict], eval_split_name: str) -> pd.DataFrame:
    rows = []
    n_errors = 0
    print(f'\n  Evaluating {model_key} on {eval_split_name} via {server_model_id} [CoT mode]')
    sys.stdout.flush()

    # Per-question resume: load existing partial results for this combo
    partial_path = os.path.join(PER_MODEL_DIR, f'{eval_split_name}__{model_key}.csv')
    done_qids = set()
    if os.path.exists(partial_path):
        partial_df = pd.read_csv(partial_path)
        done_qids = set(partial_df["question_id"].dropna().tolist())
        rows = partial_df.to_dict("records")
        print(f'  Resuming: {len(done_qids)} questions already done, {len(dataset) - len(done_qids)} remaining')
        sys.stdout.flush()

    for idx, sample in enumerate(dataset, start=1):
        qid = sample.get("id")
        if qid in done_qids:
            continue

        t0 = time.perf_counter()
        try:
            pred, raw_text, token_info = query_local_model(server_model_id, sample)
        except Exception as e:
            pred, raw_text = None, f"ERROR: {e}"
            token_info = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "finish_reason": "error"}
            n_errors += 1
            print(f'  [{idx}/{len(dataset)}] {sample.get("id")} | ERROR: {e}')
            sys.stdout.flush()
            if n_errors >= 5:
                print(f'  Too many consecutive errors ({n_errors}), stopping {model_key}.')
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
            "model_display_name": MODEL_SPECS[model_key]["display_name"],
            "server_model_id": server_model_id,
            "hf_repo": MODEL_SPECS[model_key]["hf_repo"],
            "hf_file": MODEL_SPECS[model_key]["hf_file"],
            "param_count": MODEL_SPECS[model_key].get("param_count", ""),
            "quantization": MODEL_SPECS[model_key].get("quantization", ""),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
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
        tps_str = f"{tokens_per_sec}tok/s" if tokens_per_sec else "?tok/s"
        subj = sample.get("subject_category", "?")
        print(f'  [{idx}/{len(dataset)}] {sample.get("id")} [{subj}] | gold={gold} pred={pred} {marker} | {latency_s:.1f}s | {tps_str}')
        sys.stdout.flush()

        # Incremental save every 5 questions
        if len(rows) % 5 == 0:
            partial_df = pd.DataFrame(rows)
            partial_path = os.path.join(PER_MODEL_DIR, f'{eval_split_name}__{model_key}.csv')
            partial_df.to_csv(partial_path, index=False)

        if SLEEP_BETWEEN_REQUESTS > 0:
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    df = pd.DataFrame(rows)
    if not df.empty:
        per_model_path = os.path.join(PER_MODEL_DIR, f'{eval_split_name}__{model_key}.csv')
        df.to_csv(per_model_path, index=False)
        acc = df["correct"].mean()
        print(f'  => {len(df)} rows, accuracy={acc:.4f}, saved to {per_model_path}')
    else:
        print(f'  => No results collected for {model_key}.')
    sys.stdout.flush()
    return df


def upsert_predictions(run_df: pd.DataFrame, predictions_path: str = PREDICTIONS_PATH) -> pd.DataFrame:
    if run_df.empty:
        return run_df.copy()
    key_cols = ["experiment_tag", "eval_split_name", "model_key", "question_id"]
    if os.path.exists(predictions_path):
        existing = pd.read_csv(predictions_path)
        merged = pd.concat([existing, run_df], ignore_index=True)
    else:
        merged = run_df.copy()
    merged = (
        merged.sort_values(by=key_cols)
              .drop_duplicates(subset=key_cols, keep="last")
              .reset_index(drop=True)
    )
    merged.to_csv(predictions_path, index=False)
    return merged


def build_summary_by_split(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "experiment_tag", "eval_split_name", "model_key", "model_display_name",
        "server_model_id", "hf_repo", "hf_file", "temperature", "max_tokens"
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
        ascending=[True, False, False, True]
    ).reset_index(drop=True)
    for col in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "invalid_prediction_rate", "mean_latency_s", "median_latency_s", "p90_latency_s"]:
        if col in summary.columns:
            summary[col] = summary[col].round(4)
    return summary


def build_summary_aggregated(summary_by_split_df: pd.DataFrame) -> pd.DataFrame:
    if summary_by_split_df.empty:
        return pd.DataFrame()
    agg = (
        summary_by_split_df.groupby(["experiment_tag", "model_key", "model_display_name", "hf_repo", "hf_file"], as_index=False)
        .agg(
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
    for col in ["mean_accuracy", "std_accuracy", "mean_macro_precision", "mean_macro_recall", "mean_macro_f1", "mean_invalid_prediction_rate", "mean_latency_s", "median_latency_s", "mean_p90_latency_s"]:
        agg[col] = agg[col].round(4)
    agg = agg.sort_values(by=["mean_accuracy", "mean_macro_f1", "mean_latency_s"], ascending=[False, False, True]).reset_index(drop=True)
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
        by=["subject_category", "accuracy"], ascending=[True, False]
    ).reset_index(drop=True)
    for col in ["accuracy", "mean_latency_s", "macro_precision", "macro_recall", "macro_f1"]:
        subject_summary[col] = subject_summary[col].round(4)
    return subject_summary


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("MCAT EVALUATION — CoT (Chain-of-Thought) Prompting")
    print("3 models x 7 test sets — subject-specific prompts")
    print("=" * 70)
    print(f"Server:      {LOCAL_API_BASE_URL}")
    print(f"Timeout:     {REQUEST_TIMEOUT}s per request")
    print(f"Max tokens:  {MAX_TOKENS}")
    print(f"Experiment:  {EXPERIMENT_TAG}")
    print(f"Results dir: {RESULT_DIR}")
    print()

    # Show prompt mapping
    print("Subject-specific prompts:")
    for subj in ["CARS", "BB", "CP", "PS"]:
        print(f"  {subj}: {SUBJECT_SYSTEM_PROMPTS[subj][:60]}...")
    print()

    # Resolve models
    available_model_ids = list_server_models()
    print("Available models on server:")
    for mid in available_model_ids:
        print(f"  - {mid}")

    resolved_models = {}
    for model_key, spec in MODEL_SPECS.items():
        resolved_id = resolve_loaded_model_id(spec, available_model_ids)
        if resolved_id is not None:
            resolved_models[model_key] = resolved_id

    print(f"\nResolved {len(resolved_models)} target models:")
    for mk, sid in resolved_models.items():
        print(f"  {mk} -> {sid}")

    if not resolved_models:
        print("\nERROR: No target models found on server. Load them in LM Studio first.")
        return

    # Seed all_predictions.csv from any existing partial CSVs
    partial_csvs = glob.glob(os.path.join(PER_MODEL_DIR, "*.csv"))
    if partial_csvs and not os.path.exists(PREDICTIONS_PATH):
        print("Seeding all_predictions.csv from partial per-model CSVs...")
        parts = [pd.read_csv(p) for p in partial_csvs]
        seed_df = pd.concat(parts, ignore_index=True)
        seed_df.to_csv(PREDICTIONS_PATH, index=False)
        print(f"  Seeded {len(seed_df)} rows from {len(partial_csvs)} files")

    # Check existing results for resume (skip fully completed combos)
    already_done = set()
    if os.path.exists(PREDICTIONS_PATH):
        existing_df = pd.read_csv(PREDICTIONS_PATH)
        for (split, model), grp in existing_df.groupby(["eval_split_name", "model_key"]):
            # Only mark as done if we have all questions (dataset has ~230 per split)
            if len(grp) >= 225:  # allow small margin
                already_done.add((split, model))

    total_combos = len(ALL_SPLIT_NAMES) * len(resolved_models)
    remaining = total_combos - len(already_done)
    print(f"\nTotal combos: {total_combos} | Already done: {len(already_done)} | Remaining: {remaining}")
    if already_done:
        print("Completed:")
        for combo in sorted(already_done):
            print(f"  + {combo[0]} / {combo[1]}")
    print()
    sys.stdout.flush()

    # Run evaluation
    combo_num = 0
    for split_name in ALL_SPLIT_NAMES:
        dataset_dir = os.path.join(DATASET_BASE_DIR, split_name)

        if not os.path.isdir(dataset_dir):
            print(f"WARNING: No dataset directory for {split_name}, skipping.")
            continue

        dataset = load_dataset(dataset_dir, split_name)
        print(f"\n{'=' * 70}")
        print(f"SPLIT: {split_name} — {len(dataset)} questions")
        print(f"{'=' * 70}")
        sys.stdout.flush()

        if len(dataset) == 0:
            print(f"No questions found for {split_name}, skipping.")
            continue

        for model_key, server_model_id in resolved_models.items():
            combo_num += 1
            if (split_name, model_key) in already_done:
                print(f"\n  [{combo_num}/{total_combos}] SKIP {model_key} on {split_name} (already done)")
                sys.stdout.flush()
                continue

            print(f"\n  [{combo_num}/{total_combos}] START {model_key} on {split_name}")
            sys.stdout.flush()
            t_start = time.perf_counter()

            run_df = evaluate_model(model_key, server_model_id, dataset, eval_split_name=split_name)

            elapsed = time.perf_counter() - t_start
            print(f"  [{combo_num}/{total_combos}] DONE {model_key} on {split_name} in {elapsed:.0f}s")
            sys.stdout.flush()

            if not run_df.empty:
                upsert_predictions(run_df, PREDICTIONS_PATH)

    # Build final summaries
    print(f"\n{'=' * 70}")
    print("BUILDING SUMMARIES")
    print(f"{'=' * 70}")

    all_predictions_df = pd.read_csv(PREDICTIONS_PATH) if os.path.exists(PREDICTIONS_PATH) else pd.DataFrame()

    if not all_predictions_df.empty:
        summary_by_split_df = build_summary_by_split(all_predictions_df)
        summary_aggregated_df = build_summary_aggregated(summary_by_split_df)
        subject_summary_df = build_subject_summary(all_predictions_df)

        summary_by_split_df.to_csv(SUMMARY_BY_SPLIT_PATH, index=False)
        summary_aggregated_df.to_csv(SUMMARY_AGG_PATH, index=False)
        subject_summary_df.to_csv(SUBJECT_SUMMARY_PATH, index=False)

        print(f"\nTotal predictions: {len(all_predictions_df)}")
        print(f"\nAggregated summary:")
        print(summary_aggregated_df.to_string(index=False))
        print(f"\nSummary by split:")
        print(summary_by_split_df[["eval_split_name", "model_key", "accuracy", "macro_f1", "mean_latency_s"]].to_string(index=False))

        print(f"\nSubject summary:")
        subj_agg = subject_summary_df.groupby(["model_display_name", "subject_category"]).agg(
            accuracy=("accuracy", "mean"),
            n=("n_questions", "sum"),
        ).round(4).reset_index()
        print(subj_agg.to_string(index=False))
    else:
        print("No predictions found.")

    print(f"\n{'=' * 70}")
    print("ALL DONE — CoT evaluation complete.")
    print(f"Results saved to: {RESULT_DIR}/")
    print(f"{'=' * 70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
