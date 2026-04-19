#!/usr/bin/env bash
# ============================================================
# Zero-shot baseline evaluation — run on DGX9.
# Launches inside a tmux session and tees to a log file.
#
# Usage:
#   bash ~/medmcqa/scripts/run_baseline.sh                           # 14B (default)
#   MODEL_ID=Qwen/Qwen3-8B bash ~/medmcqa/scripts/run_baseline.sh    # 8B
#   MAX_SAMPLES=200 bash ~/medmcqa/scripts/run_baseline.sh             # quick smoke test
#
# Watch from Mac:
#   ssh dgx9 'tmux attach -t medmcqa_baseline'
#   ssh dgx9 'tail -f ~/medmcqa/logs/baseline_<timestamp>.log'
# ============================================================
set -euo pipefail

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/medmcqa-env/bin/activate"
cd "$HOME/medmcqa"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export HF_TOKEN="${HF_TOKEN:-}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

MODEL_SUFFIX=$(echo "$MODEL_ID" | sed 's|.*/Qwen3[^/]*-||' | tr 'A-Z' 'a-z')
RUN_NAME="baseline-${MODEL_SUFFIX}"
OUTPUT_DIR="results/baseline-${MODEL_SUFFIX}"

mkdir -p logs logs/archive

# ── Archive old logs for this model ───────────────────────────────────────────
mapfile -t _old < <(ls logs/baseline_${MODEL_SUFFIX}_*.log 2>/dev/null | sort)
if [ ${#_old[@]} -gt 0 ]; then
    mv "${_old[@]}" logs/archive/
    echo "Archived ${#_old[@]} old log(s) to logs/archive/"
fi

LOG="logs/baseline_${MODEL_SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
SESSION="medmcqa_baseline"

MAX_SAMPLES_ARG=""
[[ -n "$MAX_SAMPLES" ]] && MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting baseline eval in tmux session: $SESSION"
echo "Model:  $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo "Log:    ~/medmcqa/$LOG"

tmux new-session -d -s "$SESSION" -x 220 -y 50 \
    "cd $HOME/medmcqa && source $HOME/medmcqa-env/bin/activate && \
     export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True && \
     export HF_TOKEN=$HF_TOKEN && \
     python training/evaluate.py \
         --model_id \"$MODEL_ID\" \
         --test_file data/dev.json \
         --output_dir $OUTPUT_DIR \
         --run_name $RUN_NAME \
         $MAX_SAMPLES_ARG \
     2>&1 | tee $LOG; echo 'Baseline exited with code:' \$?"

echo ""
echo "To watch live:    ssh dgx9 'tmux attach -t $SESSION'"
echo "To tail log:      ssh dgx9 'tail -f ~/medmcqa/$LOG'"
echo "To check status:  bash scripts/remote_status.sh"
