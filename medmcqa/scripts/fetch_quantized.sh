#!/usr/bin/env bash
# Fetch finished quantized files from DGX8 → local quantized/ directory.
#
# Usage:
#   bash scripts/fetch_quantized.sh
set -euo pipefail

DGX_USER="${DGX_USER:-jamesoon}"
DGX_IP="${DGX_IP:-192.168.0.219}"
REMOTE_DIR="${REMOTE_DIR:-~/medmcqa}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_OUTPUT="$LOCAL_DIR/quantized"

mkdir -p "$LOCAL_OUTPUT"

echo "Fetching quantized models from DGX8…"
rsync -avz --progress \
    --include="*/" \
    --include="*.gguf" \
    --include="*.safetensors" \
    --include="*.json" \
    --include="*.md" \
    --exclude="*" \
    "$DGX_USER@$DGX_IP:$REMOTE_DIR/quantized/" \
    "$LOCAL_OUTPUT/"

echo ""
echo "Files in $LOCAL_OUTPUT:"
find "$LOCAL_OUTPUT" -type f | while read -r f; do
    echo "  $(du -sh "$f" | cut -f1)  $f"
done
