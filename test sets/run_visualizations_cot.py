"""
Run all visualization and performance comparison code for CoT evaluation results.
Includes subject-breakdown grouped bar charts inspired by M3Exam-style figures.
"""
import os, re, json, glob, time, sys
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- Paths ----
RESULT_DIR = "results_local_llm_cot"
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

# Full subject names for display
SUBJECT_DISPLAY_NAMES = {
    "BB":   "Biological & Biochemical\nFoundations (BB)",
    "CARS": "Critical Analysis &\nReasoning Skills (CARS)",
    "CP":   "Chemical & Physical\nFoundations (CP)",
    "PS":   "Psychological, Social &\nBiological Foundations (PS)",
}

# Curated color palette — distinct, accessible, vibrant
MODEL_COLORS = {
    "Gemma-3-4B-it-q4_0":  "#4ECDC4",   # teal
    "Qwen3-8B-Q4_K_M":     "#5B7FFF",   # blue
    "Qwen3.5-9B-Q4_K_M":   "#FF6B6B",   # coral
}
FALLBACK_COLORS = ["#4ECDC4", "#5B7FFF", "#FF6B6B", "#FFD93D", "#A78BFA", "#34D399"]


# ---- Style setup ----
def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.facecolor": "#FAFBFC",
        "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#DEE2E6",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.color": "#CED4DA",
        "xtick.color": "#495057",
        "ytick.color": "#495057",
    })

setup_style()


# ---- Helper functions ----
def get_model_color(name):
    return MODEL_COLORS.get(name, FALLBACK_COLORS[hash(name) % len(FALLBACK_COLORS)])


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


# =============================================================================
# SUBJECT BREAKDOWN CHARTS (the main new feature)
# =============================================================================

def plot_subject_accuracy_horizontal_bars(pred_df, save_path):
    """
    Horizontal grouped bar chart — subjects on Y-axis, models as colored bars.
    Inspired by the MMLU / M3Exam benchmark horizontal bar style.
    """
    # Compute per-subject, per-model accuracy
    subj_acc = (
        pred_df.groupby(["subject_category", "model_display_name"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )

    subjects = sorted(subj_acc["subject_category"].unique())
    models = sorted(subj_acc["model_display_name"].unique())
    n_subjects = len(subjects)
    n_models = len(models)

    bar_height = 0.7 / max(n_models, 1)
    fig_height = max(4, n_subjects * 1.2 + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(n_subjects)

    for i, model in enumerate(models):
        model_data = subj_acc[subj_acc["model_display_name"] == model]
        accs = []
        for subj in subjects:
            row = model_data[model_data["subject_category"] == subj]
            accs.append(float(row["accuracy"].iloc[0]) * 100 if not row.empty else 0)

        bar_y = y + (i - (n_models - 1) / 2) * bar_height
        color = get_model_color(model)
        bars = ax.barh(bar_y, accs, height=bar_height * 0.9, label=model,
                       color=color, edgecolor="white", linewidth=0.5, zorder=3)

        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                        f"{acc:.1f}%", ha="left", va="center", fontsize=9, color="#495057")

    # Format
    display_labels = [SUBJECT_DISPLAY_NAMES.get(s, s) for s in subjects]
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("MCAT CoT Accuracy by Subject — Model Comparison", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))

    # 50% reference line
    ax.axvline(x=50, color="#E63946", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(50.5, n_subjects - 0.1, "50% baseline", color="#E63946", fontsize=9, alpha=0.8, va="top")

    ax.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#DEE2E6")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_subject_accuracy_vertical_bars(pred_df, save_path):
    """
    Vertical grouped bar chart — subjects on X-axis, models as colored bars.
    Inspired by the M3Exam Zero-Shot Results style.
    """
    subj_acc = (
        pred_df.groupby(["subject_category", "model_display_name"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )

    subjects = sorted(subj_acc["subject_category"].unique())
    models = sorted(subj_acc["model_display_name"].unique())
    n_subjects = len(subjects)
    n_models = len(models)

    bar_width = 0.7 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=(max(10, n_subjects * 2.5), 6))

    x = np.arange(n_subjects)

    for i, model in enumerate(models):
        model_data = subj_acc[subj_acc["model_display_name"] == model]
        accs = []
        for subj in subjects:
            row = model_data[model_data["subject_category"] == subj]
            accs.append(float(row["accuracy"].iloc[0]) * 100 if not row.empty else 0)

        bar_x = x + (i - (n_models - 1) / 2) * bar_width
        color = get_model_color(model)
        bars = ax.bar(bar_x, accs, width=bar_width * 0.9, label=model,
                      color=color, edgecolor="white", linewidth=0.5, zorder=3)

        # Value labels on top of bars
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{acc:.1f}", ha="center", va="bottom", fontsize=9, color="#495057")

    display_labels = [SUBJECT_DISPLAY_NAMES.get(s, s) for s in subjects]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=10, ha="center")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("MCAT CoT Accuracy by Subject — Model Comparison", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))

    ax.axhline(y=50, color="#E63946", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

    ax.legend(loc="upper right", fontsize=10, framealpha=0.9, edgecolor="#DEE2E6")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_subject_accuracy_per_test_set(pred_df, save_path):
    """
    Faceted plot — one subplot per subject showing accuracy across test sets for each model.
    Shows consistency and variance across splits.
    """
    subjects = sorted(pred_df["subject_category"].dropna().unique())
    models = sorted(pred_df["model_display_name"].unique())
    splits = sorted(pred_df["eval_split_name"].unique())
    n_subjects = len(subjects)

    fig, axes = plt.subplots(1, n_subjects, figsize=(5 * n_subjects, 6), sharey=True)
    if n_subjects == 1:
        axes = [axes]

    bar_width = 0.7 / max(len(models), 1)

    for ax_idx, subj in enumerate(subjects):
        ax = axes[ax_idx]
        subj_df = pred_df[pred_df["subject_category"] == subj]

        x = np.arange(len(splits))

        for mi, model in enumerate(models):
            model_subj_df = subj_df[subj_df["model_display_name"] == model]
            accs = []
            for split in splits:
                split_df = model_subj_df[model_subj_df["eval_split_name"] == split]
                accs.append(float(split_df["correct"].mean()) * 100 if not split_df.empty else 0)

            bar_x = x + (mi - (len(models) - 1) / 2) * bar_width
            color = get_model_color(model)
            ax.bar(bar_x, accs, width=bar_width * 0.9, label=model if ax_idx == 0 else None,
                   color=color, edgecolor="white", linewidth=0.5, zorder=3)

        short_name = SUBJECT_DISPLAY_NAMES.get(subj, subj).split("\n")[0]
        ax.set_title(short_name, fontsize=12, fontweight="bold")
        short_splits = [s.replace("test_set_", "TS") for s in splits]
        ax.set_xticks(x)
        ax.set_xticklabels(short_splits, fontsize=9, rotation=45)
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color="#E63946", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    fig.suptitle("MCAT CoT — Accuracy by Subject and Test Set", fontsize=16, fontweight="bold", y=1.02)
    fig.legend(models, loc="upper center", ncol=len(models), fontsize=10,
               bbox_to_anchor=(0.5, 0.99), framealpha=0.9, edgecolor="#DEE2E6")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_subject_heatmap(pred_df, save_path):
    """
    Enhanced heatmap with subject x model accuracy, annotated with counts.
    """
    subj_acc = (
        pred_df.groupby(["subject_category", "model_display_name"])
        .agg(accuracy=("correct", "mean"), n=("correct", "count"))
        .reset_index()
    )

    pivot_acc = subj_acc.pivot_table(index="subject_category", columns="model_display_name", values="accuracy").sort_index()
    pivot_n = subj_acc.pivot_table(index="subject_category", columns="model_display_name", values="n").sort_index()

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot_acc) * 1.2 + 1)))
    im = ax.imshow(pivot_acc.values * 100, aspect="auto", cmap="RdYlGn", vmin=30, vmax=100)

    ax.set_xticks(np.arange(len(pivot_acc.columns)))
    ax.set_xticklabels(pivot_acc.columns, rotation=20, ha="right", fontsize=11)
    display_labels = [SUBJECT_DISPLAY_NAMES.get(s, s).replace("\n", " ") for s in pivot_acc.index]
    ax.set_yticks(np.arange(len(pivot_acc.index)))
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_title("CoT Accuracy by Subject × Model (%)", fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)

    for i in range(pivot_acc.shape[0]):
        for j in range(pivot_acc.shape[1]):
            acc = pivot_acc.iloc[i, j] * 100
            n = int(pivot_n.iloc[i, j]) if pd.notna(pivot_n.iloc[i, j]) else 0
            text_color = "white" if acc < 55 else "black"
            ax.text(j, i, f"{acc:.1f}%\n(n={n})", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_subject_radar(pred_df, save_path):
    """
    Radar/spider chart showing each model's accuracy profile across subjects.
    """
    subj_acc = (
        pred_df.groupby(["subject_category", "model_display_name"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )

    subjects = sorted(subj_acc["subject_category"].unique())
    models = sorted(subj_acc["model_display_name"].unique())
    n_subjects = len(subjects)

    angles = np.linspace(0, 2 * np.pi, n_subjects, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in models:
        model_data = subj_acc[subj_acc["model_display_name"] == model]
        values = []
        for subj in subjects:
            row = model_data[model_data["subject_category"] == subj]
            values.append(float(row["accuracy"].iloc[0]) * 100 if not row.empty else 0)
        values += values[:1]  # close

        color = get_model_color(model)
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.12, color=color)

    short_labels = [SUBJECT_DISPLAY_NAMES.get(s, s).split("\n")[0] for s in subjects]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9, color="#868E96")
    ax.set_title("MCAT CoT — Model Accuracy Profile by Subject", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), fontsize=10, framealpha=0.9, edgecolor="#DEE2E6")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

# ---- Load data ----
print("=" * 80)
print("LOADING CoT DATA")
print("=" * 80)

pred_df = pd.read_csv(PREDICTIONS_PATH)
summary_by_split = pd.read_csv(SUMMARY_BY_SPLIT_PATH) if os.path.exists(SUMMARY_BY_SPLIT_PATH) else pd.DataFrame()
summary_agg = pd.read_csv(SUMMARY_AGG_PATH) if os.path.exists(SUMMARY_AGG_PATH) else pd.DataFrame()
subject_summary = pd.read_csv(SUBJECT_SUMMARY_PATH) if os.path.exists(SUBJECT_SUMMARY_PATH) else pd.DataFrame()

print(f"Predictions: {len(pred_df)} rows")
print(f"Models: {pred_df['model_key'].unique().tolist()}")
print(f"Subjects: {pred_df['subject_category'].value_counts().to_dict()}")
print(f"Test sets: {sorted(pred_df['eval_split_name'].unique().tolist())}")

# ---- PERFORMANCE COMPARISON TABLE ----
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON — CoT Benchmark")
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
    bp1 = axes[0].boxplot(data_tps, tick_labels=model_order, showfliers=False, patch_artist=True)
    for patch, m in zip(bp1["boxes"], model_order):
        patch.set_facecolor(get_model_color(m))
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Tokens / second")
    axes[0].set_title("Generation throughput by model")
    axes[0].tick_params(axis='x', rotation=15)
    if has_tokens:
        data_tot = [pred_df.loc[pred_df["model_display_name"] == m, "total_tokens"].dropna().values for m in model_order]
        bp2 = axes[1].boxplot(data_tot, tick_labels=model_order, showfliers=False, patch_artist=True)
        for patch, m in zip(bp2["boxes"], model_order):
            patch.set_facecolor(get_model_color(m))
            patch.set_alpha(0.7)
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
    colors_prompt = [get_model_color(m) for m in model_order]
    ax.bar(x, prompt_means, label="Prompt tokens", color=colors_prompt, alpha=0.8)
    ax.bar(x, completion_means, bottom=prompt_means, label="Completion tokens", color=colors_prompt, alpha=0.5)
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
        colors = [get_model_color(m) for m in x_labels]
        alpha = 0.85 if i == 0 else 0.55
        plt.bar(x + i * width, summary_agg[y_col], width=width, label=y_col,
                color=colors, alpha=alpha, edgecolor="white")
    plt.xticks(x + width * (len(y_cols) - 1) / 2, x_labels, rotation=15)
    plt.ylabel("Score")
    plt.title("Overall model comparison across all test sets (CoT)")
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
        width = 0.7 / max(len(model_names), 1)
        plt.figure(figsize=(11, 5))
        for i, model_name in enumerate(model_names):
            color = get_model_color(model_name)
            plt.bar(x + i * width, pivot[model_name].fillna(0.0).values * 100,
                    width=width, label=model_name, color=color, edgecolor="white", linewidth=0.5)
        plt.xticks(x + width * (len(model_names) - 1) / 2, x_labels, rotation=15)
        plt.ylim(0, 105)
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy by test set (CoT)")
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(FIGURE_DIR, "accuracy_by_test_set.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

# Latency boxplot
if not pred_df.empty:
    data = [pred_df.loc[pred_df["model_display_name"] == m, "latency_s"].dropna().values for m in model_order]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=model_order, showfliers=False, patch_artist=True)
    for patch, m in zip(bp["boxes"], model_order):
        patch.set_facecolor(get_model_color(m))
        patch.set_alpha(0.7)
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Latency distribution by model (CoT)")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, "latency_distribution.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# Confusion matrices
for model_name, grp in pred_df.groupby("model_display_name"):
    matrix = confusion_matrix_counts(grp, LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted answer")
    ax.set_ylabel("Gold answer")
    ax.set_title(f"Confusion matrix: {model_name} (CoT)")
    fig.colorbar(im, ax=ax)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    plt.tight_layout()
    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name.lower()) + "_confusion_matrix.png"
    save_path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# SUBJECT BREAKDOWN CHARTS (NEW)
# ============================================================================
print("\n" + "=" * 80)
print("SUBJECT BREAKDOWN VISUALIZATIONS")
print("=" * 80)

if "subject_category" in pred_df.columns and pred_df["subject_category"].notna().any():
    # 1. Horizontal grouped bars (MMLU style)
    plot_subject_accuracy_horizontal_bars(
        pred_df,
        os.path.join(FIGURE_DIR, "subject_accuracy_horizontal.png")
    )

    # 2. Vertical grouped bars (M3Exam style)
    plot_subject_accuracy_vertical_bars(
        pred_df,
        os.path.join(FIGURE_DIR, "subject_accuracy_vertical.png")
    )

    # 3. Faceted per-test-set by subject
    plot_subject_accuracy_per_test_set(
        pred_df,
        os.path.join(FIGURE_DIR, "subject_accuracy_per_test_set.png")
    )

    # 4. Enhanced heatmap with counts
    plot_subject_heatmap(
        pred_df,
        os.path.join(FIGURE_DIR, "subject_accuracy_heatmap.png")
    )

    # 5. Radar chart
    plot_subject_radar(
        pred_df,
        os.path.join(FIGURE_DIR, "subject_accuracy_radar.png")
    )

    # Print subject summary table
    print("\nSubject accuracy summary:")
    subj_table = (
        pred_df.groupby(["model_display_name", "subject_category"])
        .agg(accuracy=("correct", "mean"), n=("correct", "count"))
        .round(4)
        .reset_index()
    )
    print(subj_table.to_string(index=False))
else:
    print("No subject_category data found, skipping subject breakdowns.")


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
