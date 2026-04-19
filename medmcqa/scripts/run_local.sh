#!/usr/bin/env bash
# ============================================================
# Local training: Gemma 3B (or any model) on MPS (Apple Silicon) or CUDA.
# Runs in the current terminal — no tmux, no DGX required.
#
# Usage:
#   bash scripts/run_local.sh                                  # Gemma 3B, full 180K
#   MODEL_ID=google/gemma-3-1b-it bash scripts/run_local.sh   # 1B variant
#   MAX_STEPS=500 bash scripts/run_local.sh                    # quick smoke test
# ============================================================
set -euo pipefail

# Activate project virtualenv so 'python' resolves to the ML environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/../venv/bin/activate"
# shellcheck source=/dev/null
[[ -f "$VENV_ACTIVATE" ]] && source "$VENV_ACTIVATE"

export MODEL_ID="${MODEL_ID:-google/gemma-3-4b-it}"
export MAX_TRAIN="${MAX_TRAIN:-}"          # empty = full dataset (~182K)
export MAX_STEPS="${MAX_STEPS:--1}"        # -1 = run full num_epochs
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export HF_TOKEN="${HF_TOKEN:-}"

# Smaller batch/larger grad_accum for MPS memory efficiency
BATCH="${BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"             # effective batch = 2 × 8 = 16

# Derive checkpoint/result names from model ID
MODEL_SUFFIX=$(echo "$MODEL_ID" | sed 's|.*/||' | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]/-/g' | sed 's/-*$//')
OUTPUT_DIR="checkpoints/lora-${MODEL_SUFFIX}"
RESULTS_DIR="results/lora-${MODEL_SUFFIX}"

echo "============================================================"
echo "  Local LoRA training"
echo "  Model:   $MODEL_ID"
echo "  Output:  $OUTPUT_DIR"
echo "  Epochs:  $NUM_EPOCHS  |  Max steps: $MAX_STEPS"
echo "  Batch:   $BATCH  |  Grad accum: $GRAD_ACCUM  (eff. $(( BATCH * GRAD_ACCUM )))"
echo "============================================================"

# ── Data prep ────────────────────────────────────────────────────────────────
if [ ! -f data/processed/train.jsonl ]; then
    echo "[data_prep] Processed data not found — running data_prep.py ..."
    MAX_TRAIN_ARG=""
    [[ -n "$MAX_TRAIN" ]] && MAX_TRAIN_ARG="--max_train $MAX_TRAIN"
    python training/data_prep.py --data_dir data --output_dir data/processed $MAX_TRAIN_ARG
fi

# ── Training ─────────────────────────────────────────────────────────────────
python training/lora_train.py \
    --model_id       "$MODEL_ID" \
    --train_file     data/processed/train.jsonl \
    --dev_file       data/processed/dev.jsonl \
    --output_dir     "$OUTPUT_DIR" \
    --num_epochs     "$NUM_EPOCHS" \
    --per_device_batch "$BATCH" \
    --grad_accum     "$GRAD_ACCUM" \
    --lr             2e-4 \
    --max_seq_len    512 \
    --lora_rank      16 \
    --lora_alpha     32 \
    --max_steps      "$MAX_STEPS" \
    --eval_steps     200 \
    --save_steps     200

echo ""
echo "Training complete. Adapter saved to $OUTPUT_DIR/final"
echo ""
echo "To evaluate:"
echo "  python training/evaluate.py --model_id $MODEL_ID --adapter_path $OUTPUT_DIR/final --run_name lora-${MODEL_SUFFIX} --output_dir $RESULTS_DIR"
