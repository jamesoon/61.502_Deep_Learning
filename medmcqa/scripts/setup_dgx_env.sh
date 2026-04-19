#!/usr/bin/env bash
# ============================================================
# Environment setup — run on DGX9 before training.
#
# Usage (from your Mac):
#   ssh dgx9 'bash -s' < scripts/setup_dgx_env.sh
#
# Or SSH in and run directly:
#   bash ~/medmcqa/scripts/setup_dgx_env.sh
# ============================================================
set -euo pipefail

VENV_DIR="$HOME/medmcqa-env"
PROJECT_DIR="$HOME/medmcqa"

echo "=== MedMCQA environment setup ==="
echo "Venv:    $VENV_DIR"
echo "Project: $PROJECT_DIR"

# ── Python venv ───────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "[2/5] Upgrading pip..."
pip install --upgrade pip wheel setuptools

# ── PyTorch ───────────────────────────────────────────────────────────────────
# DGX Spark GB200 = CUDA compute capability 12.x / system CUDA 12.8+
# Use nightly cu128 for full Blackwell support; fall back to stable if needed.
echo "[3/5] Installing PyTorch (nightly cu128, Grace Blackwell)..."
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
|| {
    echo "  Nightly failed — installing stable PyTorch cu124..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
}

python3 -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device:   {torch.cuda.get_device_name(0)}')
    print(f'Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Training dependencies ─────────────────────────────────────────────────────
echo "[4/5] Installing training dependencies..."
pip install -r "$PROJECT_DIR/requirements_training.txt"

# ── Flash Attention (optional) ────────────────────────────────────────────────
echo "[5/5] Attempting flash-attn install (optional, may take a while)..."
pip install flash-attn --no-build-isolation 2>/dev/null \
    || echo "     flash-attn skipped — training works without it."

echo ""
echo "=== Setup complete on $(hostname) ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Save your HuggingFace token (needed to download Qwen3.5-35B-A3B):"
echo "       huggingface-cli login"
echo "     Or set it permanently so scripts pick it up automatically:"
echo "       echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc"
echo "       source ~/.bashrc"
echo ""
echo "  2. Log in to WandB:"
echo "       wandb login"
echo ""
echo "  3. Start training:"
echo "       bash ~/medmcqa/scripts/run_lora.sh"
