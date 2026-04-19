#!/usr/bin/env bash
# ============================================================
# Approach 1: Full SFT — run on DGX9.
# Requires DeepSpeed ZeRO-3 for 27B; works with both 9B and 27B.
# Launches training inside a tmux session and tees to a log file.
#
# Usage:
#   bash ~/medmcqa/scripts/run_sft.sh                           # 14B (default)
#   MODEL_ID=Qwen/Qwen3-8B bash ~/medmcqa/scripts/run_sft.sh    # 8B (faster)
#
# Watch from Mac:
#   ssh dgx9 'tmux attach -t medmcqa_sft'
#   ssh dgx9 'tail -f ~/medmcqa/logs/sft_<timestamp>.log'
# ============================================================
set -euo pipefail

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/medmcqa-env/bin/activate"
cd "$HOME/medmcqa"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export HF_TOKEN="${HF_TOKEN:-}"
export DS_SKIP_CUDA_CHECK=1
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
export MAX_TRAIN="${MAX_TRAIN:-}"

MODEL_SUFFIX=$(echo "$MODEL_ID" | sed 's|.*/Qwen3[^/]*-||' | tr 'A-Z' 'a-z')
RUN_NAME="sft-${MODEL_SUFFIX}"
OUTPUT_DIR="checkpoints/sft-${MODEL_SUFFIX}"

mkdir -p logs logs/archive

# ── Archive old logs for this model ───────────────────────────────────────────
mapfile -t _old < <(ls logs/sft_${MODEL_SUFFIX}_*.log 2>/dev/null | sort)
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

LOG="logs/sft_${MODEL_SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
SESSION="medmcqa_sft"

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting SFT training in tmux session: $SESSION"
echo "Model:  $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo "Log:    ~/medmcqa/$LOG"

tmux new-session -d -s "$SESSION" -x 220 -y 50 \
    "cd $HOME/medmcqa && source $HOME/medmcqa-env/bin/activate && \
     export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True && \
     export HF_TOKEN=$HF_TOKEN && \
     export DS_SKIP_CUDA_CHECK=1 && \
     torchrun --nproc_per_node=1 --nnodes=1 training/sft_train.py \
         --model_id \"$MODEL_ID\" \
         --train_file data/processed/train.jsonl \
         --dev_file   data/processed/dev.jsonl \
         --output_dir $OUTPUT_DIR \
         --num_epochs 3 \
         --per_device_batch 2 \
         --grad_accum 8 \
         --lr 2e-5 \
         --max_seq_len 1024 \
         --run_name \"$RUN_NAME\" \
         --deepspeed training/ds_config_zero3.json \
     2>&1 | tee $LOG; echo 'SFT exited with code:' \$?"

echo ""
echo "To watch live:    ssh dgx9 'tmux attach -t $SESSION'"
echo "To tail log:      ssh dgx9 'tail -f ~/medmcqa/$LOG'"
echo "To check status:  bash scripts/remote_status.sh"
