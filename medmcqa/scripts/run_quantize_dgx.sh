#!/usr/bin/env bash
# =============================================================================
# Mac-side launcher: sync scripts to DGX8, run quantization, fetch results.
#
# Usage (from project root on Mac):
#   bash scripts/run_quantize_dgx.sh
#   MODEL=lora-9b bash scripts/run_quantize_dgx.sh   # single model
# =============================================================================
set -euo pipefail

DGX_USER="${DGX_USER:-jamesoon}"
DGX_IP="${DGX_IP:-192.168.0.219}"
REMOTE_DIR="${REMOTE_DIR:-~/medmcqa}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_OUTPUT="$LOCAL_DIR/quantized"

MODEL_ARG="${MODEL:-}"   # empty = both models

echo "══════════════════════════════════════════════════════"
echo "  DGX8 Quantization Launcher"
echo "  Target : $DGX_USER@$DGX_IP"
echo "══════════════════════════════════════════════════════"

# 1. Sync scripts to DGX8
echo "[1/3] Syncing scripts to DGX8…"
rsync -avz --progress \
    "$LOCAL_DIR/scripts/merge_lora.py" \
    "$LOCAL_DIR/scripts/quantize_gguf.sh" \
    "$DGX_USER@$DGX_IP:$REMOTE_DIR/scripts/"

# 2. Run quantization on DGX8
echo "[2/3] Running quantization on DGX8…"
if [ -n "$MODEL_ARG" ]; then
    ssh "$DGX_USER@$DGX_IP" "cd $REMOTE_DIR && MODEL=$MODEL_ARG bash scripts/quantize_gguf.sh"
else
    ssh "$DGX_USER@$DGX_IP" "cd $REMOTE_DIR && bash scripts/quantize_gguf.sh"
fi

# 3. Fetch results back to Mac
echo "[3/3] Fetching quantized files from DGX8…"
bash "$(dirname "$0")/fetch_quantized.sh"

echo ""
echo "✓ Quantized models are in: $LOCAL_OUTPUT"
