#!/usr/bin/env bash
# ============================================================
# Sync medmcqa project (code + data) from Mac to Windows WSL2.
#
# Usage (run from Mac):
#   bash scripts/rsync_to_wsl.sh
#
# The Windows PC must be on the same network.
# Set WINDOWS_IP to your PC's local IP (find it with: ipconfig in Windows).
# ============================================================
set -euo pipefail

WINDOWS_IP="${WINDOWS_IP:-192.168.0.XXX}"   # ← replace with your Windows PC IP
WSL_USER="${WSL_USER:-$(whoami)}"            # usually same username
REMOTE="$WSL_USER@$WINDOWS_IP"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # medmcqa project root

if [[ "$WINDOWS_IP" == *"XXX"* ]]; then
    echo "ERROR: Set WINDOWS_IP to your Windows PC's local IP address."
    echo "  Find it in Windows: ipconfig | findstr IPv4"
    echo "  Then run: WINDOWS_IP=192.168.0.X bash scripts/rsync_to_wsl.sh"
    exit 1
fi

echo "Syncing to $REMOTE:~/medmcqa/ ..."

# Sync code (fast — no data files)
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints/' \
    --exclude='results/' \
    --exclude='logs/' \
    --exclude='data/processed/' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    "$LOCAL_DIR/" \
    "$REMOTE:~/medmcqa/"

echo ""
echo "Syncing data files (train/dev/test.json — may be large)..."
rsync -avz --progress \
    "$LOCAL_DIR/data/train.json" \
    "$LOCAL_DIR/data/dev.json" \
    "$LOCAL_DIR/data/test.json" \
    "$REMOTE:~/medmcqa/data/"

echo ""
echo "Done! Now on Windows WSL2:"
echo "  bash ~/medmcqa/scripts/setup_wsl.sh    # first time only"
echo "  bash ~/medmcqa/scripts/run_qlora.sh    # start training"
