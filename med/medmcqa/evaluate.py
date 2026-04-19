"""
Batch evaluation for cross-encoder MCQ classifier.

Usage:
    # Evaluate encoder only on dev set
    python evaluate.py --data dev.json --ckpt models/best.pt

    # Evaluate with per-subject breakdown
    python evaluate.py --data dev.json --ckpt models/best.pt --per-subject

    # Compare against generative model results
    python evaluate.py --data dev.json --ckpt models/best.pt \\
        --compare-csv ../medmcqa/results/gemma4b_results.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from dataset import MCQADataset
from infer_select import load_selector, predict_batch, resolve_device

COP_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}
LETTERS = ("A", "B", "C", "D")


def evaluate_encoder(
    ckpt_path: str,
    data_path: str,
    *,
    use_context: bool = True,
    device: str | None = None,
    batch_size: int = 32,
    per_subject: bool = False,
) -> dict:
    """Run encoder on full dataset and compute metrics."""
    model = load_selector(ckpt_path, device)
    dev = model.args.get("device", "cpu")

    # Load raw data for labels and metadata
    path = Path(data_path)
    if path.suffix == ".json":
        with open(path) as f:
            raw = pd.DataFrame(json.load(f))
    elif path.suffix == ".jsonl":
        raw = pd.read_json(path, lines=True)
    else:
        raw = pd.read_csv(path)

    questions = raw["question"].fillna("").astype(str).tolist()
    options_list = [
        [str(raw.iloc[i].get(k, "")) for k in ("opa", "opb", "opc", "opd")]
        for i in range(len(raw))
    ]
    contexts = raw["exp"].fillna("").astype(str).tolist() if use_context else None

    gold_labels = []
    for _, row in raw.iterrows():
        cop = row.get("cop", 0)
        if pd.isna(cop):
            gold_labels.append(-1)
        else:
            cop = int(cop)
            gold_labels.append(cop - 1 if 1 <= cop <= 4 else cop)

    # Run batch prediction
    print(f"[Eval] Running encoder on {len(questions)} samples...")
    results = predict_batch(
        model, questions, options_list,
        contexts=contexts, use_context=use_context,
        batch_size=batch_size,
    )

    # Compute metrics
    correct = 0
    total = 0
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    confidence_buckets = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, (res, gold) in enumerate(zip(results, gold_labels)):
        if gold < 0:
            continue
        total += 1
        pred = res["predicted_index"]
        conf = res["confidence"]
        is_correct = int(pred == gold)
        correct += is_correct

        # Confidence calibration
        bucket = f"{int(conf * 10) * 10}-{int(conf * 10) * 10 + 10}%"
        confidence_buckets[bucket]["total"] += 1
        confidence_buckets[bucket]["correct"] += is_correct

        if per_subject and "subject_name" in raw.columns:
            subj = raw.iloc[i].get("subject_name", "Unknown")
            subject_correct[subj] += is_correct
            subject_total[subj] += 1

    accuracy = correct / total if total else 0
    avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0

    metrics = {
        "model": ckpt_path,
        "data": data_path,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "confidence_calibration": dict(confidence_buckets),
    }

    print(f"\n[Results] Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"[Results] Avg confidence: {avg_confidence:.4f}")

    # Confidence calibration
    print("\n  Confidence calibration:")
    for bucket in sorted(confidence_buckets.keys()):
        b = confidence_buckets[bucket]
        acc = b["correct"] / b["total"] if b["total"] else 0
        print(f"    {bucket}: {b['correct']}/{b['total']} = {acc:.3f}")

    if per_subject and subject_total:
        print("\n  Per-subject accuracy:")
        metrics["per_subject"] = {}
        for subj in sorted(subject_total.keys()):
            acc = subject_correct[subj] / subject_total[subj]
            metrics["per_subject"][subj] = {
                "correct": subject_correct[subj],
                "total": subject_total[subj],
                "accuracy": acc,
            }
            print(f"    {subj}: {subject_correct[subj]}/{subject_total[subj]} = {acc:.3f}")

    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate cross-encoder MCQ classifier")
    ap.add_argument("--data", type=str, required=True, help="Path to dev/test data (csv/json/jsonl)")
    ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: latest)")
    ap.add_argument("--models-dir", type=str, default="./models")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--use-context", action="store_true", default=True)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--per-subject", action="store_true", help="Break down by subject")
    ap.add_argument("--output", type=str, default=None, help="Save metrics JSON to this path")
    ap.add_argument("--compare-csv", type=str, default=None,
                    help="CSV with generative model predictions for comparison")
    args = ap.parse_args()

    use_context = args.use_context and not args.no_context

    if args.ckpt is None:
        from infer_select import find_latest_ckpt
        args.ckpt = find_latest_ckpt(args.models_dir)
        print(f"[Eval] Using latest checkpoint: {args.ckpt}")

    metrics = evaluate_encoder(
        args.ckpt, args.data,
        use_context=use_context,
        device=args.device,
        batch_size=args.batch_size,
        per_subject=args.per_subject,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")

    if args.compare_csv:
        print(f"\n--- Comparison with {args.compare_csv} ---")
        gen_df = pd.read_csv(args.compare_csv)
        if "predictions" in gen_df.columns and "cop" in gen_df.columns:
            gen_correct = (gen_df["predictions"] == gen_df["cop"]).sum()
            gen_total = len(gen_df)
            gen_acc = gen_correct / gen_total
            print(f"Generative model: {gen_correct}/{gen_total} = {gen_acc:.4f}")
            print(f"Cross-encoder:    {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.4f}")
            diff = metrics["accuracy"] - gen_acc
            print(f"Difference:       {diff:+.4f} ({'encoder wins' if diff > 0 else 'generative wins'})")


if __name__ == "__main__":
    main()
