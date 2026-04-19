"""Run all visualization and performance comparison code from the notebook."""
import os, re, json, glob, time, sys
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Paths ----
RESULT_DIR = "results_local_llm"
PER_MODEL_DIR = os.path.join(RESULT_DIR, "per_model")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
PREDICTIONS_PATH = os.path.join(RESULT_DIR, "all_predictions.csv")
SUMMARY_BY_SPLIT_PATH = os.path.join(RESULT_DIR, "summary_by_split.csv")
SUMMARY_AGG_PATH = os.path.join(RESULT_DIR, "summary_aggregated.csv")
SUBJECT_SUMMARY_PATH = os.path.join(RESULT_DIR, "summary_by_subject.csv")
PERF_SUMMARY_PATH = os.path.join(RESULT_DIR, "performance_summary.csv")
os.makedirs(FIGURE_DIR, exist_ok=True)

LABELS = ["A", "B", "C", "D"]

MODEL_SPECS = {
    "qwen3_8b": {
        "display_name": "Qwen3-8B-Q4_K_M",
        "param_count": "8B",
        "quantization": "Q4_K_M",
    },
    "qwen3_5_9b": {
        "display_name": "Qwen3.5-9B-Q4_K_M",
        "param_count": "9B",
        "quantization": "Q4_K_M",
    },
    "gemma3_4b": {
        "display_name": "Gemma-3-4B-it-q4_0",
        "param_count": "4B",
        "quantization": "q4_0",
    },
}

# ---- Helper functions ----

def build_performance_summary(pred_df):
    if pred_df.empty:
        return pd.DataFrame()
    rows = []
    for model_key, grp in pred_df.groupby("model_key"):
        spec = MODEL_SPECS.get(model_key, {})
        row = {
            "Model": spec.get("display_name", model_key),
            "Parameters": spec.get("param_count", "?"),
            "Quantization": spec.get("quantization", "?"),
            "Total Questions": int(len(grp)),
            "Accuracy": round(float(grp["correct"].mean()), 4),
            "Mean Latency (s)": round(float(grp["latency_s"].mean()), 2),
            "Median Latency (s)": round(float(grp["latency_s"].median()), 2),
            "P90 Latency (s)": round(float(grp["latency_s"].quantile(0.90)), 2),
            "Total Time (min)": round(float(grp["latency_s"].sum() / 60.0), 1),
        }
        if "prompt_tokens" in grp.columns and grp["prompt_tokens"].notna().any():
            row["Mean Prompt Tokens"] = round(float(grp["prompt_tokens"].mean()), 0)
            row["Mean Completion Tokens"] = round(float(grp["completion_tokens"].mean()), 0)
            row["Mean Total Tokens"] = round(float(grp["total_tokens"].mean()), 0)
            row["Total Tokens Used"] = int(grp["total_tokens"].sum())
        if "tokens_per_sec" in grp.columns and grp["tokens_per_sec"].notna().any():
            row["Mean Tokens/sec"] = round(float(grp["tokens_per_sec"].mean()), 1)
            row["Median Tokens/sec"] = round(float(grp["tokens_per_sec"].median()), 1)
        if "finish_reason" in grp.columns:
            total = len(grp)
            n_stop = int((grp["finish_reason"] == "stop").sum())
            n_length = int((grp["finish_reason"] == "length").sum())
            row["Stop %"] = round(100.0 * n_stop / total, 1) if total > 0 else 0
            row["Truncated %"] = round(100.0 * n_length / total, 1) if total > 0 else 0
        rows.append(row)
    perf_df = pd.DataFrame(rows)
    perf_df = perf_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return perf_df


def confusion_matrix_counts(grp, labels=LABELS):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for i, gold in enumerate(labels):
        for j, pred in enumerate(labels):
            matrix[i, j] = int(((grp["gold_answer"] == gold) & (grp["pred_answer"] == pred)).sum())
    return matrix


# ---- Load data ----
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

pred_df = pd.read_csv(PREDICTIONS_PATH)
summary_by_split = pd.read_csv(SUMMARY_BY_SPLIT_PATH) if os.path.exists(SUMMARY_BY_SPLIT_PATH) else pd.DataFrame()
summary_agg = pd.read_csv(SUMMARY_AGG_PATH) if os.path.exists(SUMMARY_AGG_PATH) else pd.DataFrame()
subject_summary = pd.read_csv(SUBJECT_SUMMARY_PATH) if os.path.exists(SUBJECT_SUMMARY_PATH) else pd.DataFrame()

print(f"Predictions: {len(pred_df)} rows")
print(f"Models: {pred_df['model_key'].unique().tolist()}")
print(f"Test sets: {sorted(pred_df['eval_split_name'].unique().tolist())}")

# ---- PERFORMANCE COMPARISON TABLE ----
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON — Apples-to-Apples Model Benchmark")
print("=" * 80)

perf_df = build_performance_summary(pred_df)
perf_df.to_csv(PERF_SUMMARY_PATH, index=False)
print(perf_df.to_string(index=False))

# Per-test-set breakdown
print("\n" + "=" * 80)
print("PER-TEST-SET PERFORMANCE BREAKDOWN")
print("=" * 80)
for split_name in sorted(pred_df["eval_split_name"].unique()):
    split_df = pred_df[pred_df["eval_split_name"] == split_name]
    split_perf = build_performance_summary(split_df)
    print(f"\n--- {split_name} ---")
    print(split_perf.to_string(index=False))

# ---- TOKEN VISUALIZATIONS ----
has_tokens = "prompt_tokens" in pred_df.columns and pred_df["prompt_tokens"].notna().any()
has_tps = "tokens_per_sec" in pred_df.columns and pred_df["tokens_per_sec"].notna().any()

model_order = pred_df.groupby("model_display_name")["latency_s"].mean().sort_values().index.tolist()

if has_tps:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data_tps = [pred_df.loc[pred_df["model_display_name"] == m, "tokens_per_sec"].dropna().values for m in model_order]
    axes[0].boxplot(data_tps, tick_labels=model_order, showfliers=False)
    axes[0].set_ylabel("Tokens / second")
    axes[0].set_title("Generation throughput by model")
    axes[0].tick_params(axis='x', rotation=15)
    if has_tokens:
        data_tot = [pred_df.loc[pred_df["model_display_name"] == m, "total_tokens"].dropna().values for m in model_order]
        axes[1].boxplot(data_tot, tick_labels=model_order, showfliers=False)
        axes[1].set_ylabel("Total tokens")
        axes[1].set_title("Total tokens per question by model")
        axes[1].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, "token_performance_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")

if has_tokens:
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = []
    prompt_means = []
    completion_means = []
    for m in model_order:
        grp = pred_df[pred_df["model_display_name"] == m]
        model_names.append(m)
        prompt_means.append(grp["prompt_tokens"].mean())
        completion_means.append(grp["completion_tokens"].mean())
    x = np.arange(len(model_names))
    ax.bar(x, prompt_means, label="Prompt tokens", color="#4C78A8")
    ax.bar(x, completion_means, bottom=prompt_means, label="Completion tokens", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("Mean tokens per question")
    ax.set_title("Token usage breakdown by model")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, "token_usage_breakdown.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# ---- STANDARD VISUALIZATIONS ----
print("\n" + "=" * 80)
print("STANDARD VISUALIZATIONS")
print("=" * 80)

# Aggregated summary
if not summary_agg.empty:
    print("\nAggregated summary:")
    print(summary_agg.to_string(index=False))

# Overall model comparison bar chart
if not summary_agg.empty:
    x_labels = summary_agg["model_display_name"].astype(str).tolist()
    x = np.arange(len(x_labels))
    y_cols = ["mean_accuracy", "mean_macro_f1"]
    width = 0.8 / max(len(y_cols), 1)
    plt.figure(figsize=(10, 5))
    for i, y_col in enumerate(y_cols):
        plt.bar(x + i * width, summary_agg[y_col], width=width, label=y_col)
    plt.xticks(x + width * (len(y_cols) - 1) / 2, x_labels, rotation=15)
    plt.ylabel("Score")
    plt.title("Overall model comparison across completed test sets")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, "overall_model_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# Accuracy by test set
if not summary_by_split.empty:
    pivot = summary_by_split.pivot_table(
        index="eval_split_name", columns="model_display_name",
        values="accuracy", aggfunc="mean"
    ).sort_index()
    if not pivot.empty:
        x_labels = pivot.index.astype(str).tolist()
        model_names = pivot.columns.tolist()
        x = np.arange(len(x_labels))
        width = 0.8 / max(len(model_names), 1)
        plt.figure(figsize=(11, 5))
        for i, model_name in enumerate(model_names):
            plt.bar(x + i * width, pivot[model_name].fillna(0.0).values, width=width, label=model_name)
        plt.xticks(x + width * (len(model_names) - 1) / 2, x_labels, rotation=15)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Accuracy by test set")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(FIGURE_DIR, "accuracy_by_test_set.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

# Latency boxplot
if not pred_df.empty:
    data = [pred_df.loc[pred_df["model_display_name"] == m, "latency_s"].dropna().values for m in model_order]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=model_order, showfliers=False)
    plt.ylabel("Latency (seconds)")
    plt.title("Latency distribution by model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, "latency_distribution.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# Confusion matrices
for model_name, grp in pred_df.groupby("model_display_name"):
    matrix = confusion_matrix_counts(grp, LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted answer")
    ax.set_ylabel("Gold answer")
    ax.set_title(f"Confusion matrix: {model_name}")
    fig.colorbar(im, ax=ax)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10)
    plt.tight_layout()
    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name.lower()) + "_confusion_matrix.png"
    save_path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# Subject heatmap
if not subject_summary.empty and "subject_category" in subject_summary.columns:
    pivot = subject_summary.pivot_table(
        index="subject_category", columns="model_display_name",
        values="accuracy", aggfunc="mean"
    ).sort_index()
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(pivot.index))))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Accuracy by subject/category")
        fig.colorbar(im, ax=ax, label="Accuracy")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
        plt.tight_layout()
        save_path = os.path.join(FIGURE_DIR, "accuracy_by_subject_heatmap.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

# List all figures
print("\n" + "=" * 80)
print("ALL SAVED FIGURES:")
print("=" * 80)
for p in sorted(glob.glob(os.path.join(FIGURE_DIR, "*.png"))):
    print(f"  {p}")

print("\n" + "=" * 80)
print("ALL SAVED CSVs:")
print("=" * 80)
for p in sorted(glob.glob(os.path.join(RESULT_DIR, "*.csv"))):
    print(f"  {p}")

print("\nDONE!")
