#!/usr/bin/env bash
# ============================================================
# Cross-encoder training (DeBERTa-v3-large) — run on DGX8.
# Launches training inside a tmux session and tees to a log file.
#
# Usage:
#   bash ~/med/medmcqa/scripts/run_cross_encoder.sh                                          # default
#   MODEL_ID=michiyasunaga/BioLinkBERT-large bash ~/med/medmcqa/scripts/run_cross_encoder.sh  # alt model
#
# Watch from Mac:
#   ssh dgx8 'tmux attach -t medmcqa_xenc'
#   ssh dgx8 'tail -f ~/med/medmcqa/logs/xenc_*.log'
# ============================================================
set -euo pipefail

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/medmcqa-env/bin/activate"
cd "$HOME/med/medmcqa"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export TORCHDYNAMO_DISABLE=1        # disable torch.compile
export PYTORCH_JIT=0                # disable all TorchScript/nvfuser JIT
export CUDA_LAUNCH_BLOCKING=0
export HF_TOKEN="${HF_TOKEN:-}"

# ── Model config ─────────────────────────────────────────────────────────────
export MODEL_ID="${MODEL_ID:-microsoft/deberta-v3-large}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-8}"        # micro-batch; effective = 8 * GRAD_ACCUM(2) = 16
export GRAD_ACCUM="${GRAD_ACCUM:-2}"        # gradient accumulation steps
export FREEZE_LAYERS="${FREEZE_LAYERS:-18}" # freeze bottom N/24 encoder layers
export LR="${LR:-2e-5}"
export MAX_LEN="${MAX_LEN:-256}"            # Q+4opts fits in ~200 tokens; dynamic padding handles rest
export MAX_STEPS="${MAX_STEPS:-}"           # empty = no limit (full epochs)

# Data: use the medmcqa JSON data (182K train, 4.1K dev)
TRAIN_FILE="${TRAIN_FILE:-$HOME/medmcqa/data/train.json}"
DEV_FILE="${DEV_FILE:-$HOME/medmcqa/data/dev.json}"

# Derive a clean suffix: microsoft/deberta-v3-large -> deberta-v3-large
MODEL_SUFFIX=$(echo "$MODEL_ID" | sed 's|.*/||' | tr 'A-Z' 'a-z')
RUN_NAME="xenc-${MODEL_SUFFIX}"

mkdir -p logs logs/archive models

# ── Archive old logs ─────────────────────────────────────────────────────────
mapfile -t _old < <(ls logs/xenc_${MODEL_SUFFIX}_*.log 2>/dev/null | sort)
if [ ${#_old[@]} -gt 0 ]; then
    mv "${_old[@]}" logs/archive/
    echo "Archived ${#_old[@]} old log(s) to logs/archive/"
fi

LOG="logs/xenc_${MODEL_SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
SESSION="medmcqa_xenc"

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cross-encoder training"
echo "  Model:  $MODEL_ID"
echo "  Train:  $TRAIN_FILE"
echo "  Dev:    $DEV_FILE"
echo "  Epochs: $NUM_EPOCHS  Batch: ${BATCH_SIZE}x${GRAD_ACCUM}(eff=$((BATCH_SIZE*GRAD_ACCUM)))  LR: $LR  Freeze: ${FREEZE_LAYERS}/24"
echo "  Log:    ~/med/medmcqa/$LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build optional args
EXTRA_ARGS=""
[[ -n "$MAX_STEPS" ]] && EXTRA_ARGS="$EXTRA_ARGS --max_steps $MAX_STEPS"

tmux new-session -d -s "$SESSION" -x 220 -y 50 \
    "cd $HOME/med/medmcqa && source $HOME/medmcqa-env/bin/activate && \
     export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True && \
     export TORCHDYNAMO_DISABLE=1 && \
     export PYTORCH_JIT=0 && \
     export TORCH_COMPILE_DISABLE=1 && \
     export HF_TOKEN=$HF_TOKEN && \
     export PYTHONUNBUFFERED=1 && \
     nohup python3 -u train.py \
         --model \"$MODEL_ID\" \
         --train_file $TRAIN_FILE \
         --dev_file $DEV_FILE \
         --use_context \
         --num_epochs $NUM_EPOCHS \
         --batch_size $BATCH_SIZE \
         --grad_accum $GRAD_ACCUM \
         --freeze_layers $FREEZE_LAYERS \
         --lr $LR \
         --max_len $MAX_LEN \
         $EXTRA_ARGS \
     >> $LOG 2>&1; echo 'Cross-encoder exited with code:' \$? >> $LOG"

echo ""
echo "To watch live:    ssh dgx8 'tmux attach -t $SESSION'"
echo "To tail log:      ssh dgx8 'tail -f ~/med/medmcqa/$LOG'"
