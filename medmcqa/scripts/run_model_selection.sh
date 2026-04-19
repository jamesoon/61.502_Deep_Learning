#!/usr/bin/env bash
# ============================================================
# Systematic Model Selection — evaluate all trained candidates
# on the dev set and select the best by MCQ accuracy.
#
# Strategy (Whiteboard — Case 1: Large Dataset):
#
#   Candidate set C = { gemma-3-4b-it, Qwen3.5-9B, Qwen3-14B }
#
#   For each C_i:
#     - Already trained with max_steps=1000 (fixed budget — DeltaGate/GB10
#       GB10 performance constraint prevents longer runs on full 182K data)
#     - Evaluate on dev set → MCQ accuracy(C_i)
#
#   Selection: C* = argmax accuracy(C_i)
#
# Run AFTER all training is complete (Gemma + Qwen3.5-9B + Qwen3-14B):
#   bash scripts/run_model_selection.sh
#
# Quick smoke-test (500 dev samples):
#   MAX_SAMPLES=500 bash scripts/run_model_selection.sh
#
# Skip candidates already evaluated:
#   SKIP_GEMMA=1 bash scripts/run_model_selection.sh
# ============================================================
set -euo pipefail

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/medmcqa-env/bin/activate"
cd "$HOME/medmcqa"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export HF_TOKEN="${HF_TOKEN:-}"

MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-24}"       # 4-bit KV cache frees ~3x memory → up from 8
QUANTIZE_KV="${QUANTIZE_KV:-4}"      # 0=off, 4=4-bit (quality-neutral), 2=2-bit
DEV_FILE="data/dev.json"
RESULTS_DIR="results"
LOG_DIR="logs"
SKIP_GEMMA="${SKIP_GEMMA:-0}"
SKIP_9B="${SKIP_9B:-0}"
SKIP_14B="${SKIP_14B:-0}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
LOG="$LOG_DIR/model_selection_$(date +%Y%m%d_%H%M%S).log"

MAX_SAMPLES_ARG=""
[[ -n "$MAX_SAMPLES" ]] && MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"

log() { echo "$@" | tee -a "$LOG"; }

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  SYSTEMATIC MODEL SELECTION — MedMCQA LoRA"
log "  Candidate set C = {gemma-3-4b-it, Qwen3.5-9B, Qwen3-14B}"
log "  Budget  : max_steps=1000 per candidate (DeltaGate/GB10 constraint)"
log "  Data    : stratified 800/subject × 21 subjects ≈ 16,800 samples"
log "            (avoids 10x imbalance: Medicine 9.8% vs Skin 1.0%)"
log "  Metric  : MACRO-AVERAGED accuracy (equal weight per subject)"
log "            (avoids bias toward high-frequency specialties)"
log "  Dev set : 4,183 labeled samples (data/dev.json)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Candidate set ─────────────────────────────────────────────────────────────
# Format: "run_name|model_id|adapter_path|skip_flag"
declare -a CANDIDATES=(
    "lora-gemma-3-4b-it|google/gemma-3-4b-it|checkpoints/lora-gemma-3-4b-it/final|$SKIP_GEMMA"
    "lora-9b|Qwen/Qwen3.5-9B|checkpoints/lora-9b/final|$SKIP_9B"
    "lora-14b|Qwen/Qwen3-14B|checkpoints/lora-14b/final|$SKIP_14B"
)

# ── Step 1–3: Evaluate each candidate ────────────────────────────────────────
C_IDX=1
for ENTRY in "${CANDIDATES[@]}"; do
    IFS='|' read -r RUN_NAME MODEL_ID ADAPTER_PATH SKIP_FLAG <<< "$ENTRY"
    log ""
    log "── C${C_IDX}: $RUN_NAME ─────────────────────────────────────────"
    log "   Model:   $MODEL_ID"
    log "   Adapter: $ADAPTER_PATH"

    if [[ "$SKIP_FLAG" -eq 1 ]]; then
        log "   [SKIP] SKIP flag set."
        C_IDX=$((C_IDX + 1))
        continue
    fi

    if [ ! -d "$ADAPTER_PATH" ]; then
        log "   [SKIP] Checkpoint not found: $ADAPTER_PATH"
        C_IDX=$((C_IDX + 1))
        continue
    fi

    # Skip if metrics already exist (avoid re-running expensive eval)
    METRICS_FILE="$RESULTS_DIR/$RUN_NAME/${RUN_NAME}_metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        log "   [CACHE] Metrics already exist — skipping eval."
        log "           Delete $METRICS_FILE to re-run."
        C_IDX=$((C_IDX + 1))
        continue
    fi

    log "   Evaluating on dev set..."
    python training/evaluate.py \
        --model_id     "$MODEL_ID" \
        --adapter_path "$ADAPTER_PATH" \
        --test_file    "$DEV_FILE" \
        --run_name     "$RUN_NAME" \
        --output_dir   "$RESULTS_DIR/$RUN_NAME" \
        --batch_size   "$BATCH_SIZE" \
        --quantize_kv  "$QUANTIZE_KV" \
        $MAX_SAMPLES_ARG \
        2>&1 | tee -a "$LOG"

    C_IDX=$((C_IDX + 1))
done

# ── Step 4: Compare and select ────────────────────────────────────────────────
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  SELECTION RESULT"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python training/evaluate.py --compare --results_dir "$RESULTS_DIR/" 2>&1 | tee -a "$LOG"

log ""
log "✓ Model selection complete."
log "  Full log:  ~/medmcqa/$LOG"
log "  Table:     ~/medmcqa/results/comparison_table.csv"
