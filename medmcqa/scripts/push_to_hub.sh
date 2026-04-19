#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Push a local LoRA adapter checkpoint to HuggingFace Hub.
#
# Usage:
#   HF_REPO=username/gemma-3-4b-it-medmcqa-lora bash scripts/push_to_hub.sh
#   HF_REPO=username/my-model ADAPTER_PATH=checkpoints/lora-gemma-3-4b-it/final bash scripts/push_to_hub.sh
#
# Requires: huggingface_hub installed, HF_TOKEN set (or already logged in)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HF_REPO="${HF_REPO:-}"
ADAPTER_PATH="${ADAPTER_PATH:-checkpoints/lora-gemma-3-4b-it/final}"

if [[ -z "$HF_REPO" ]]; then
  echo "Error: HF_REPO is not set."
  echo "Usage: HF_REPO=username/model-name bash scripts/push_to_hub.sh"
  exit 1
fi

if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "Error: Adapter directory not found: $ADAPTER_PATH"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Adapter : $ADAPTER_PATH"
echo "  Repo    : https://huggingface.co/$HF_REPO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 - <<PYEOF
import os
from huggingface_hub import HfApi, create_repo

repo_id = "${HF_REPO}"
adapter_path = "${ADAPTER_PATH}"
token = os.environ.get("HF_TOKEN")

api = HfApi(token=token)

# Create repo if it doesn't exist (public by default)
create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
print(f"Repo ready: https://huggingface.co/{repo_id}")

# Upload all files in the adapter directory
print(f"Uploading {adapter_path} ...")
api.upload_folder(
    folder_path=adapter_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Add MedMCQA LoRA adapter (gemma-3-4b-it, 1200 steps)",
)

print(f"\nDone. View at: https://huggingface.co/{repo_id}")
PYEOF
