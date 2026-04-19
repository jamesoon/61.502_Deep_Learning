#!/usr/bin/env bash
# ============================================================
# QLoRA fine-tuning — RTX 4080 (16 GB VRAM) on Windows WSL2
#
# Uses 4-bit NF4 quantization (bitsandbytes) to fit 14B in 16 GB.
# Launches training inside a tmux session and tees to a log file.
#
# Usage (inside WSL2):
#   bash ~/medmcqa/scripts/run_qlora.sh                          # 14B (default)
#   MODEL_ID=Qwen/Qwen3.5-9B bash ~/medmcqa/scripts/run_qlora.sh # 9B (faster)
#   MAX_STEPS=1000 bash ~/medmcqa/scripts/run_qlora.sh           # short test run
#
# Watch live:
#   tmux attach -t medmcqa_qlora
#   tail -f ~/medmcqa/logs/qlora_<timestamp>.log
#
# Memory budget (RTX 4080 16 GB):
#   Qwen3-14B 4-bit: ~7.5 GB weights + ~5 GB activations/optimizer = ~13 GB ✓
#   Qwen3.5-9B 4-bit: ~5 GB weights  + ~4 GB activations/optimizer = ~9 GB  ✓
# ============================================================
set -euo pipefail

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/medmcqa-env/bin/activate"
cd "$HOME/medmcqa"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export HF_TOKEN="${HF_TOKEN:-}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
export MAX_TRAIN="${MAX_TRAIN:-}"
export MAX_STEPS="${MAX_STEPS:--1}"       # -1 = full epoch
export EVAL_STEPS="${EVAL_STEPS:-5713}"   # eval at 50% and 100% of epoch
export SAVE_STEPS="${SAVE_STEPS:-5713}"

# Batch config tuned for 16 GB VRAM with 4-bit model
BATCH="${BATCH:-2}"          # per_device_batch (smaller than DGX due to VRAM)
GRAD_ACCUM="${GRAD_ACCUM:-8}" # effective batch = 2×8 = 16 (same as DGX)

# Derive suffix: Qwen/Qwen3-14B → 14b
MODEL_SUFFIX=$(echo "$MODEL_ID" | sed 's|.*/Qwen3[^/]*-||' | tr 'A-Z' 'a-z')
RUN_NAME="qlora-${MODEL_SUFFIX}"
OUTPUT_DIR="checkpoints/qlora-${MODEL_SUFFIX}"

mkdir -p logs logs/archive

# Archive old logs for this model
mapfile -t _old < <(ls logs/qlora_${MODEL_SUFFIX}_*.log 2>/dev/null | sort)
if [ ${#_old[@]} -gt 0 ]; then
    mv "${_old[@]}" logs/archive/
    echo "Archived ${#_old[@]} old log(s) to logs/archive/"
fi

# Data prep
if [ ! -f data/processed/train.jsonl ]; then
    echo "Preparing data..."
    MAX_TRAIN_ARG=""
    [[ -n "$MAX_TRAIN" ]] && MAX_TRAIN_ARG="--max_train $MAX_TRAIN"
    python training/data_prep.py --data_dir data --output_dir data/processed $MAX_TRAIN_ARG
fi

LOG="logs/qlora_${MODEL_SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
SESSION="medmcqa_qlora"

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting QLoRA training in tmux session: $SESSION"
echo "Model:       $MODEL_ID (4-bit NF4)"
echo "Output:      $OUTPUT_DIR"
echo "Batch:       $BATCH × $GRAD_ACCUM = $((BATCH * GRAD_ACCUM)) effective"
echo "Log:         ~/medmcqa/$LOG"

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
         --per_device_batch $BATCH \
         --grad_accum $GRAD_ACCUM \
         --lr 2e-4 \
         --max_seq_len 512 \
         --lora_rank 16 \
         --lora_alpha 32 \
         --run_name \"$RUN_NAME\" \
         --max_steps $MAX_STEPS \
         --eval_steps $EVAL_STEPS \
         --save_steps $SAVE_STEPS \
         --use_4bit \
     2>&1 | tee $LOG; echo 'QLoRA exited with code:' \$?"

echo ""
echo "To watch live:   tmux attach -t $SESSION"
echo "To tail log:     tail -f ~/medmcqa/$LOG"
