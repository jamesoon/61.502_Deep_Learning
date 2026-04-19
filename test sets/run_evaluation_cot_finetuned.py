"""
Standalone MCAT evaluation script — Finetuned models (hybrid backends).

Backends:
  - "discriminative"   : PubMedBERT encoder + linear head via transformers
  - "gguf_api"         : GGUF models served by LM Studio, queried via OpenAI API
  - "generative_lora"  : LoRA adapters loaded via transformers + peft (with optional 4-bit)

Resume-friendly: skips (split, model) combos that already have results.

Usage:
    python run_evaluation_cot_finetuned.py
"""

import os
import re
import json
import glob
import time
import sys
import gc
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# =============================================================================
# CONFIG
# =============================================================================
DATASET_BASE_DIR = "dataset_json"
RESULT_DIR = "results_finetuned_cot"
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

EXPERIMENT_TAG = "mcat_finetuned_cot_v1"
LABELS = ["A", "B", "C", "D"]

# --- API settings (for gguf_api backend) ---
LOCAL_API_BASE_URL = os.getenv("LOCAL_API_BASE_URL", "http://127.0.0.1:1234/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "not-needed")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))
HARD_TIMEOUT = int(os.getenv("HARD_TIMEOUT", "360"))
API_MAX_TOKENS = int(os.getenv("API_MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# --- Local generation settings (for generative_lora backend) ---
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "512"))

# --- PubMedBERT settings ---
PUBMEDBERT_MAX_LENGTH = int(os.getenv("PUBMEDBERT_MAX_LENGTH", "512"))

# --- dtype / quantization ---
TORCH_DTYPE = torch.bfloat16
USE_4BIT = True  # for models with "use_4bit": True

# =============================================================================
# MODEL SPECS — ordered: discriminative first, then GGUF (fast), then LoRA
# =============================================================================
MODEL_SPECS = {
    "pubmedbert_mcqa": {
        "display_name": "PubMedBERT-MedMCQA",
        "backend": "discriminative",
        "repo_id": "jamezoon/medmcqa-pubmedbert-mcqa",
        "param_count": "110M",
        "quantization": "fp32",
    },
    "qwen3_5_9b_gguf": {
        "display_name": "Qwen3.5-9B-MedMCQA-Q4_K_M",
        "backend": "gguf_api",
        "hf_repo": "jamezoon/qwen3-5-9b-medmcqa-gguf",
        "hf_file": "lora-9b-medmcqa-q4_k_m.gguf",
        "match_tokens": ["medmcqa", "9b"],
        "param_count": "9B",
        "quantization": "Q4_K_M",
    },
    "gemma3_4b_lora": {
        "display_name": "Gemma-3-4B-it-MedMCQA-LoRA",
        "backend": "generative_lora",
        "base_model_id": "google/gemma-3-4b-it",
        "adapter_id": "jamezoon/gemma-3-4b-it-medmcqa-lora",
        "param_count": "4B",
        "quantization": "bf16",
    },
    "qwen3_14b_lora": {
        "display_name": "Qwen3-14B-MedMCQA-LoRA",
        "backend": "generative_lora",
        "base_model_id": "Qwen/Qwen3-14B",
        "adapter_id": "jamezoon/qwen3-14b-medmcqa-lora",
        "param_count": "14B",
        "quantization": "4bit",
        "use_4bit": True,
    },
}

# =============================================================================
# PROMPT FORMAT — matches finetuning training format from model cards
# =============================================================================
GENERATIVE_SYSTEM_PROMPT = (
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
# PROMPT BUILDING
# =============================================================================
def build_generative_messages(sample: Dict) -> List[Dict]:
    """Build chat messages for generative models (matching training format)."""
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

    return [
        {"role": "system", "content": GENERATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_discriminative_inputs(sample: Dict) -> Tuple[Optional[str], str, List[str]]:
    """Build inputs for PubMedBERT discriminative model."""
    choices = sample.get("choices", {})
    passage = sample.get("passage", "").strip() or None
    question = sample.get("question", "").strip()
    options = [choices.get(L, "") for L in LABELS]
    return passage, question, options


# =============================================================================
# MODEL LOADING / UNLOADING
# =============================================================================
def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def load_discriminative_model(spec: Dict):
    """Load PubMedBERT encoder + classification head."""
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import hf_hub_download
    import tempfile, shutil

    repo_id = spec["repo_id"]
    print(f"    Downloading model files: {repo_id}")
    sys.stdout.flush()

    # Download only the specific files we need (avoids resuming stale downloads)
    needed_files = [
        "encoder/config.json",
        "encoder/model.safetensors",
        "encoder/tokenizer.json",
        "encoder/tokenizer_config.json",
        "encoder/vocab.txt",
        "encoder/special_tokens_map.json",
        "mcqa_head.pt",
        "mcqa_metadata.json",
    ]

    local_dir = os.path.join(RESULT_DIR, "_pubmedbert_cache")
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(os.path.join(local_dir, "encoder"), exist_ok=True)

    for fname in needed_files:
        dest = os.path.join(local_dir, fname)
        if os.path.exists(dest):
            continue
        print(f"      Fetching {fname}")
        sys.stdout.flush()
        downloaded = hf_hub_download(repo_id=repo_id, filename=fname)
        shutil.copy2(downloaded, dest)

    encoder_dir = os.path.join(local_dir, "encoder")
    print(f"    Loading encoder from: {encoder_dir}")
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    encoder = AutoModel.from_pretrained(encoder_dir)
    encoder.eval()

    head = torch.nn.Linear(768, 1)
    head_path = os.path.join(local_dir, "mcqa_head.pt")
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    head.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    head = head.to(device)

    print(f"    PubMedBERT loaded on: {device}")
    sys.stdout.flush()
    return (encoder, head), tokenizer


def load_generative_lora_model(spec: Dict):
    """Load base model + LoRA adapter for generative inference."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    base_id = spec["base_model_id"]
    adapter_id = spec["adapter_id"]
    needs_4bit = USE_4BIT and spec.get("use_4bit", False)

    print(f"    Loading tokenizer: {base_id}")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}

    if needs_4bit:
        print(f"    Loading base model: {base_id} (4-bit quantized)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=TORCH_DTYPE,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        print(f"    Loading base model: {base_id} (dtype={TORCH_DTYPE})")
        load_kwargs["torch_dtype"] = TORCH_DTYPE

    sys.stdout.flush()
    model = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)

    print(f"    Loading LoRA adapter: {adapter_id}")
    sys.stdout.flush()
    model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()

    print(f"    Model loaded on: {model.device}")
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        print(f"    GPU memory allocated: {alloc_gb:.1f} GB")
    sys.stdout.flush()
    return model, tokenizer


def setup_api_client():
    """Create OpenAI client for LM Studio API."""
    from openai import OpenAI
    return OpenAI(base_url=LOCAL_API_BASE_URL, api_key=LOCAL_API_KEY, timeout=REQUEST_TIMEOUT)


def resolve_gguf_model_id(spec: Dict, client) -> Optional[str]:
    """Find the loaded GGUF model on the LM Studio server."""
    try:
        response = client.models.list()
        available = [getattr(item, "id", str(item)) for item in (getattr(response, "data", []) or [])]
    except Exception as e:
        print(f"    Could not connect to LM Studio at {LOCAL_API_BASE_URL}: {e}")
        return None

    if not available:
        print(f"    No models loaded on LM Studio server.")
        return None

    print(f"    Models on LM Studio: {available}")
    tokens = [normalize_text(t) for t in spec.get("match_tokens", [])]
    for mid in available:
        nid = normalize_text(mid)
        if all(t in nid for t in tokens):
            return mid
    return None


def unload_local_model(model, tokenizer):
    del model
    del tokenizer
    free_gpu_memory()
    print("    Model unloaded, GPU memory freed.")
    sys.stdout.flush()


# =============================================================================
# INFERENCE — three backends
# =============================================================================
def query_discriminative(model_tuple, tokenizer, sample: Dict) -> Tuple[Optional[str], str, Dict]:
    """PubMedBERT discriminative inference."""
    encoder, head = model_tuple
    device = next(encoder.parameters()).device

    passage, question, options = build_discriminative_inputs(sample)

    pairs = []
    for opt in options:
        if passage:
            pairs.append((passage, question + " " + opt))
        else:
            pairs.append(question + " " + opt)

    enc = tokenizer.batch_encode_plus(
        pairs, truncation=True, padding="max_length",
        max_length=PUBMEDBERT_MAX_LENGTH, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        pooled = encoder(**enc).pooler_output
        logits = head(pooled).squeeze(-1)
        probs = torch.softmax(logits, dim=0)
        pred_idx = int(torch.argmax(probs).item())

    pred = LABELS[pred_idx]
    prob_dict = {L: round(float(p), 4) for L, p in zip(LABELS, probs)}
    raw_text = f"Probabilities: {prob_dict} -> Predicted: {pred}"

    token_info = {
        "prompt_tokens": int(enc["input_ids"].numel()),
        "completion_tokens": 0,
        "total_tokens": int(enc["input_ids"].numel()),
        "finish_reason": "discriminative",
    }
    return pred, raw_text, token_info


def query_generative_lora(model, tokenizer, sample: Dict) -> Tuple[Optional[str], str, Dict]:
    """Local transformers generative inference with LoRA model."""
    messages = build_generative_messages(sample)

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=LOCAL_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    completion_ids = output_ids[0][input_len:]
    raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    completion_tokens = len(completion_ids)

    pred = extract_letter(raw_text)
    token_info = {
        "prompt_tokens": input_len,
        "completion_tokens": completion_tokens,
        "total_tokens": input_len + completion_tokens,
        "finish_reason": "stop" if completion_tokens < LOCAL_MAX_NEW_TOKENS else "length",
    }
    return pred, raw_text, token_info


def _raw_api_query(client, server_model_id: str, system_prompt: str, user_prompt: str):
    """Inner function for API call (runs in thread for hard timeout)."""
    response = client.chat.completions.create(
        model=server_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=API_MAX_TOKENS,
    )
    return response


def query_gguf_api(client, server_model_id: str, sample: Dict) -> Tuple[Optional[str], str, Dict]:
    """Query GGUF model via LM Studio OpenAI-compatible API."""
    messages = build_generative_messages(sample)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_raw_api_query, client, server_model_id, system_prompt, user_prompt)
        try:
            response = future.result(timeout=HARD_TIMEOUT)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Hard timeout after {HARD_TIMEOUT}s")

    raw_text = response.choices[0].message.content
    pred = extract_letter(raw_text)
    usage = response.usage
    token_info = {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "finish_reason": response.choices[0].finish_reason,
    }
    return pred, raw_text, token_info


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================
def extract_letter(raw_text: str) -> Optional[str]:
    if raw_text is None:
        return None
    text = str(raw_text).strip()

    # Strip <think>...</think> reasoning blocks
    think_end = text.rfind('</think>')
    if think_end != -1:
        text = text[think_end + len('</think>'):].strip()

    match = re.search(r'[Ff]inal\s+[Aa]nswer\s*[:\-]?\s*\**\s*([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()

    match = re.search(r'[Aa]nswer\s*[:\-]?\s*\**\s*([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()

    match = re.search(r'the\s+answer\s+is\s+\**\s*([A-Da-d])\b', text, re.I)
    if match:
        return match.group(1).upper()

    last_lines = text.strip().split('\n')
    for line in reversed(last_lines[-3:]):
        line = line.strip().rstrip('.').strip()
        line = re.sub(r'\*+', '', line).strip()
        if line.upper() in {"A", "B", "C", "D"}:
            return line.upper()

    matches = re.findall(r'\b([A-D])\b', text.upper())
    if matches:
        return matches[-1]

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
    model_key: str,
    query_fn,
    dataset: List[Dict],
    eval_split_name: str,
) -> pd.DataFrame:
    """Evaluate a single model on a single test set.
    query_fn(sample) -> (pred, raw_text, token_info)
    """
    rows = []
    n_errors = 0
    spec = MODEL_SPECS[model_key]
    backend = spec["backend"]

    print(f'\n  Evaluating {model_key} on {eval_split_name} [{backend}]')
    sys.stdout.flush()

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
            pred, raw_text, token_info = query_fn(sample)
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
            "model_display_name": spec["display_name"],
            "backend": backend,
            "param_count": spec.get("param_count", ""),
            "quantization": spec.get("quantization", ""),
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

        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(partial_path, index=False)
        acc = df["correct"].mean()
        print(f'  => {len(df)} rows, accuracy={acc:.4f}, saved to {partial_path}')
    else:
        print(f'  => No results collected for {model_key}.')
    sys.stdout.flush()
    return df


# =============================================================================
# SUMMARIES
# =============================================================================
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
        "backend", "param_count", "quantization",
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
            ["experiment_tag", "model_key", "model_display_name", "backend"],
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
    print("MCAT EVALUATION — Finetuned Models (Hybrid Backends)")
    print(f"{len(MODEL_SPECS)} models x {len(ALL_SPLIT_NAMES)} test sets")
    print("=" * 70)
    print(f"API server:  {LOCAL_API_BASE_URL} (for gguf_api models)")
    print(f"API timeout: {REQUEST_TIMEOUT}s / hard {HARD_TIMEOUT}s")
    print(f"API max_tokens: {API_MAX_TOKENS}")
    print(f"Local max_new_tokens: {LOCAL_MAX_NEW_TOKENS}")
    print(f"PubMedBERT max_length: {PUBMEDBERT_MAX_LENGTH}")
    print(f"Experiment:  {EXPERIMENT_TAG}")
    print(f"Results dir: {RESULT_DIR}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {total_mem:.1f} GB")
    print()

    print("Models to evaluate:")
    for mk, spec in MODEL_SPECS.items():
        print(f"  {mk}: {spec['display_name']} (backend={spec['backend']}, {spec['param_count']})")
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
    # OUTER LOOP: models (load/connect once, evaluate all splits, then cleanup)
    # =========================================================================
    combo_num = 0
    api_client = None  # lazy init for gguf_api models

    for model_key, spec in MODEL_SPECS.items():
        backend = spec["backend"]

        all_splits_done = all((s, model_key) in already_done for s in ALL_SPLIT_NAMES)
        if all_splits_done:
            combo_num += len(ALL_SPLIT_NAMES)
            print(f"\n{'='*70}")
            print(f"MODEL: {spec['display_name']} — ALL SPLITS DONE, skipping")
            print(f"{'='*70}")
            sys.stdout.flush()
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {spec['display_name']} (backend={backend}, {spec['param_count']})")
        print(f"{'='*70}")
        sys.stdout.flush()

        # --- Setup model / connection ---
        query_fn = None
        local_model = None
        local_tokenizer = None

        if backend == "discriminative":
            t_load = time.perf_counter()
            try:
                local_model, local_tokenizer = load_discriminative_model(spec)
            except Exception as e:
                print(f"  FAILED to load {model_key}: {e}")
                combo_num += len(ALL_SPLIT_NAMES)
                continue
            print(f"    Loaded in {time.perf_counter() - t_load:.1f}s")
            query_fn = lambda sample, m=local_model, t=local_tokenizer: query_discriminative(m, t, sample)

        elif backend == "generative_lora":
            t_load = time.perf_counter()
            try:
                local_model, local_tokenizer = load_generative_lora_model(spec)
            except Exception as e:
                print(f"  FAILED to load {model_key}: {e}")
                print(f"  (If gated model, accept license at https://huggingface.co/<repo>)")
                combo_num += len(ALL_SPLIT_NAMES)
                continue
            print(f"    Loaded in {time.perf_counter() - t_load:.1f}s")
            query_fn = lambda sample, m=local_model, t=local_tokenizer: query_generative_lora(m, t, sample)

        elif backend == "gguf_api":
            if api_client is None:
                api_client = setup_api_client()
            server_model_id = resolve_gguf_model_id(spec, api_client)
            if server_model_id is None:
                print(f"  Model not found on LM Studio server.")
                print(f"  Please load {spec.get('hf_file', '???')} from {spec.get('hf_repo', '???')} in LM Studio and re-run.")
                combo_num += len(ALL_SPLIT_NAMES)
                continue
            print(f"    Resolved to server model: {server_model_id}")
            query_fn = lambda sample, c=api_client, sid=server_model_id: query_gguf_api(c, sid, sample)

        else:
            print(f"  Unknown backend '{backend}', skipping.")
            combo_num += len(ALL_SPLIT_NAMES)
            continue

        sys.stdout.flush()

        # --- Evaluate all splits ---
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
                print(f"  No questions found for {split_name}, skipping.")
                continue

            print(f"\n  [{combo_num}/{total_combos}] START {model_key} on {split_name} ({len(dataset)} questions)")
            sys.stdout.flush()
            t_start = time.perf_counter()

            run_df = evaluate_model(model_key, query_fn, dataset, eval_split_name=split_name)

            elapsed = time.perf_counter() - t_start
            print(f"  [{combo_num}/{total_combos}] DONE {model_key} on {split_name} in {elapsed:.0f}s")
            sys.stdout.flush()

            if not run_df.empty:
                upsert_predictions(run_df, PREDICTIONS_PATH)

        # --- Cleanup local models ---
        if local_model is not None:
            print(f"\n  Unloading {model_key}...")
            sys.stdout.flush()
            unload_local_model(local_model, local_tokenizer)

    # =========================================================================
    # BUILD SUMMARIES
    # =========================================================================
    print(f"\n{'='*70}")
    print("BUILDING SUMMARIES")
    print(f"{'='*70}")

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
        cols = ["eval_split_name", "model_key", "accuracy", "macro_f1", "mean_latency_s"]
        avail_cols = [c for c in cols if c in summary_by_split_df.columns]
        print(summary_by_split_df[avail_cols].to_string(index=False))

        print(f"\nSubject summary:")
        subj_agg = subject_summary_df.groupby(["model_display_name", "subject_category"]).agg(
            accuracy=("accuracy", "mean"),
            n=("n_questions", "sum"),
        ).round(4).reset_index()
        print(subj_agg.to_string(index=False))
    else:
        print("No predictions found.")

    print(f"\n{'='*70}")
    print("ALL DONE — Finetuned model evaluation complete.")
    print(f"Results saved to: {RESULT_DIR}/")
    print(f"{'='*70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
