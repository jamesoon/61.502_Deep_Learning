#!/usr/bin/env bash
# ============================================================
# Master pipeline: runs baseline → LoRA → SFT for each model,
# then compares all results.
#
# Usage (on DGX9):
#   bash ~/medmcqa/scripts/run_all.sh               # both 9B and 27B
#
# Run only one model:
#   MODELS="9B" bash ~/medmcqa/scripts/run_all.sh
#   MODELS="27B" bash ~/medmcqa/scripts/run_all.sh
#
# Skip stages:
#   SKIP_BASELINE=1 SKIP_SFT=1 bash ~/medmcqa/scripts/run_all.sh
# ============================================================
set -euo pipefail

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/medmcqa-env/bin/activate"
cd "$HOME/medmcqa"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export HF_TOKEN="${HF_TOKEN:-}"

MODELS="${MODELS:-9B 27B}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_LORA="${SKIP_LORA:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
MAX_TRAIN="${MAX_TRAIN:-}"

wait_for_session() {
    local session="$1"
    echo "  Waiting for '$session' to finish..."
    while tmux has-session -t "$session" 2>/dev/null; do
        sleep 30
    done
    echo "  '$session' complete."
}

# ── Data prep ─────────────────────────────────────────────────────────────────
if [ ! -f data/processed/train.jsonl ]; then
    echo "[DATA] Preparing data..."
    MAX_TRAIN_ARG=""
    [[ -n "$MAX_TRAIN" ]] && MAX_TRAIN_ARG="--max_train $MAX_TRAIN"
    python training/data_prep.py --data_dir data --output_dir data/processed $MAX_TRAIN_ARG
fi

for SIZE in $MODELS; do
    export MODEL_ID="Qwen/Qwen3.5-${SIZE}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Model: $MODEL_ID"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ "$SKIP_BASELINE" -eq 0 ]]; then
        echo ""; echo "── Baseline (zero-shot) ──────────────────────────────"
        bash scripts/run_baseline.sh
        wait_for_session "medmcqa_baseline"
    else
        echo "[SKIP] Baseline for $MODEL_ID"
    fi

    if [[ "$SKIP_LORA" -eq 0 ]]; then
        echo ""; echo "── LoRA fine-tuning ──────────────────────────────────"
        bash scripts/run_lora.sh
        wait_for_session "medmcqa_lora"
    else
        echo "[SKIP] LoRA for $MODEL_ID"
    fi

    if [[ "$SKIP_SFT" -eq 0 ]]; then
        echo ""; echo "── Full SFT (ZeRO-3) ─────────────────────────────────"
        bash scripts/run_sft.sh
        wait_for_session "medmcqa_sft"
    else
        echo "[SKIP] SFT for $MODEL_ID"
    fi
done

# ── Compare all runs ──────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python training/evaluate.py --compare --results_dir results/
echo "✓ Done. Results: results/comparison_table.csv"
