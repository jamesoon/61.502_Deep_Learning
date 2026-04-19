#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Sync med/medmcqa code to DGX9.
#
# Usage:
#   bash med/medmcqa/scripts/rsync_to_dgx.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DGX_USER="${DGX_USER:-$(whoami)}"
DGX_IP="${DGX_IP:-192.168.0.219}"
REMOTE_DIR="${REMOTE_DIR:-~/med/medmcqa}"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Source : $LOCAL_DIR"
echo "  Dest   : $DGX_USER@$DGX_IP:$REMOTE_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

rsync -avz --progress \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='*.egg-info/' \
  --exclude='checkpoints/' \
  --exclude='models/' \
  --exclude='hf_export/' \
  --exclude='results/' \
  --exclude='logs/' \
  --exclude='notebooks/' \
  "$LOCAL_DIR/" \
  "$DGX_USER@$DGX_IP:$REMOTE_DIR/"

echo ""
echo "Done. Next on DGX:"
echo "  ssh dgx9"
echo "  bash ~/med/medmcqa/scripts/run_cross_encoder.sh"
