#!/usr/bin/env bash
# Sync data files to DGX9 separately (they can be large).
# Run this after rsync_to_dgx.sh.
set -euo pipefail

DGX_USER="${DGX_USER:-$(whoami)}"
DGX_IP="${DGX_IP:-192.168.0.219}"
REMOTE_DIR="${REMOTE_DIR:-~/medmcqa}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing data/ → $DGX_USER@$DGX_IP:$REMOTE_DIR/data/"
rsync -avz --progress \
  "$LOCAL_DIR/data/" \
  "$DGX_USER@$DGX_IP:$REMOTE_DIR/data/"

echo "✓ Data sync complete."
