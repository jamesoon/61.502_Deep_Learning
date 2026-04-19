#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Sync local project code to DGX9 (192.168.0.219)
#
# Usage:
#   bash scripts/rsync_to_dgx.sh                    # sync code only
#   DGX_USER=myuser bash scripts/rsync_to_dgx.sh   # override username
#
# Environment vars (all optional):
#   DGX_USER    — remote username (defaults to local $USER)
#   DGX_IP      — remote IP (default 192.168.0.219)
#   DGX_HOST    — ssh host alias if configured in ~/.ssh/config (default dgx9)
#   REMOTE_DIR  — destination path on DGX (default ~/medmcqa)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DGX_USER="${DGX_USER:-$(whoami)}"
DGX_IP="${DGX_IP:-192.168.0.219}"
REMOTE_DIR="${REMOTE_DIR:-~/medmcqa}"

# Resolve the project root (one level up from this script)
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
  --exclude='results/' \
  --exclude='webapp/frontend/node_modules/' \
  --exclude='webapp/frontend/dist/' \
  --exclude='data/*.json' \
  --exclude='data/processed/' \
  "$LOCAL_DIR/" \
  "$DGX_USER@$DGX_IP:$REMOTE_DIR/"

echo "✓ Code sync complete."
echo ""
echo "Next steps on DGX:"
echo "  bash scripts/tmux_session.sh          # SSH + open tmux"
echo "  bash scripts/rsync_data_to_dgx.sh     # sync data files (run separately)"
echo "  conda activate medmcqa && cd ~/medmcqa"
