#!/usr/bin/env bash
# ============================================================
# Approach 2: LoRA fine-tuning — run on DGX9.
# Launches training inside a tmux session and tees to a log file.
#
# Usage:
#   bash ~/medmcqa/scripts/run_lora.sh                           # 14B (default)
#   MODEL_ID=Qwen/Qwen3-8B bash ~/medmcqa/scripts/run_lora.sh    # 8B (faster)
#
# Watch from Mac:
#   ssh dgx9 'tmux attach -t medmcqa_lora'
#   ssh dgx9 'tail -f ~/medmcqa/logs/lora_<timestamp>.log'
# ============================================================
set -euo pipefail

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/medmcqa-env/bin/activate"
cd "$HOME/medmcqa"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export HF_TOKEN="${HF_TOKEN:-}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
export MAX_TRAIN="${MAX_TRAIN:-}"
# MAX_STEPS=1000 is fixed for all candidate runs.
# Reason: Qwen3.5 GatedDeltaNet hybrid layers on GB10 (unified memory) run at
# ~45 s/step — 1000 steps ≈ 12 hrs, which is the practical ceiling per candidate.
# Standard transformers (Qwen3-14B, Gemma-3-4B) are ~1–2 s/step but we keep the
# same budget so comparisons across the candidate set are fair.
export MAX_STEPS="${MAX_STEPS:-1000}"
export EVAL_STEPS="${EVAL_STEPS:-200}"
export SAVE_STEPS="${SAVE_STEPS:-200}"

# Derive suffix for directory naming: Qwen/Qwen3-14B → 14b, Qwen/Qwen3.5-9B → 9b
MODEL_SUFFIX=$(echo "$MODEL_ID" | sed 's|.*/Qwen3[^/]*-||' | tr 'A-Z' 'a-z')
RUN_NAME="lora-${MODEL_SUFFIX}"
OUTPUT_DIR="checkpoints/lora-${MODEL_SUFFIX}"
RESULTS_DIR="results/lora-${MODEL_SUFFIX}"

mkdir -p logs logs/archive

# ── Archive old logs for this model ───────────────────────────────────────────
mapfile -t _old < <(ls logs/lora_${MODEL_SUFFIX}_*.log 2>/dev/null | sort)
if [ ${#_old[@]} -gt 0 ]; then
    mv "${_old[@]}" logs/archive/
    echo "Archived ${#_old[@]} old log(s) to logs/archive/"
fi

# ── Data prep ─────────────────────────────────────────────────────────────────
if [ ! -f data/processed/train.jsonl ]; then
    echo "Preparing data..."
    MAX_TRAIN_ARG=""
    [[ -n "$MAX_TRAIN" ]] && MAX_TRAIN_ARG="--max_train $MAX_TRAIN"
    python training/data_prep.py --data_dir data --output_dir data/processed $MAX_TRAIN_ARG
fi

LOG="logs/lora_${MODEL_SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
SESSION="medmcqa_lora"

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting LoRA training in tmux session: $SESSION"
echo "Model:  $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo "Log:    ~/medmcqa/$LOG"

tmux new-session -d -s "$SESSION" -x 220 -y 50 \
    "cd $HOME/medmcqa && source $HOME/medmcqa-env/bin/activate && \
     export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True && \
     export HF_TOKEN=$HF_TOKEN && \
     python training/lora_train.py \
         --model_id \"$MODEL_ID\" \
         --train_file data/processed/train.jsonl \
         --dev_file   data/processed/dev.jsonl \
         --output_dir $OUTPUT_DIR \
         --num_epochs 1 \
         --per_device_batch 4 \
         --grad_accum 4 \
         --lr 2e-4 \
         --max_seq_len 512 \
         --lora_rank 16 \
         --lora_alpha 32 \
         --run_name \"$RUN_NAME\" \
         --max_steps $MAX_STEPS \
         --eval_steps $EVAL_STEPS \
         --save_steps $SAVE_STEPS \
     2>&1 | tee $LOG; echo 'LoRA exited with code:' \$?"

echo ""
echo "To watch live:    ssh dgx9 'tmux attach -t $SESSION'"
echo "To tail log:      ssh dgx9 'tail -f ~/medmcqa/$LOG'"
echo "To check status:  bash scripts/remote_status.sh"
