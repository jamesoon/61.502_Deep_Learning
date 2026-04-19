#!/usr/bin/env bash
# ============================================================
# Evaluate cross-encoder on dev set — run on DGX9.
#
# Usage:
#   bash ~/med/medmcqa/scripts/run_eval.sh
#   bash ~/med/medmcqa/scripts/run_eval.sh --per-subject
# ============================================================
set -euo pipefail

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/medmcqa-env/bin/activate"
cd "$HOME/med/medmcqa"

DEV_FILE="${DEV_FILE:-$HOME/medmcqa/data/dev.json}"
CKPT="${CKPT:-}"  # auto-detect if empty
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p results logs

LOG="logs/eval_$(date +%Y%m%d_%H%M%S).log"
SESSION="medmcqa_eval"

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cross-encoder evaluation"
echo "  Data:  $DEV_FILE"
echo "  Log:   ~/med/medmcqa/$LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CKPT_ARG=""
[[ -n "$CKPT" ]] && CKPT_ARG="--ckpt $CKPT"

tmux new-session -d -s "$SESSION" -x 220 -y 50 \
    "cd $HOME/med/medmcqa && source $HOME/medmcqa-env/bin/activate && \
     python evaluate.py \
         --data $DEV_FILE \
         --batch-size $BATCH_SIZE \
         --per-subject \
         --output results/xenc_metrics.json \
         $CKPT_ARG \
         $@ \
     2>&1 | tee $LOG; echo 'Eval exited with code:' \$?"

echo ""
echo "To watch:  ssh dgx9 'tmux attach -t $SESSION'"
echo "To tail:   ssh dgx9 'tail -f ~/med/medmcqa/$LOG'"
