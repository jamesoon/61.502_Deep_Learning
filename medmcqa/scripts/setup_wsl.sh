#!/usr/bin/env bash
# ============================================================
# WSL2 / Windows RTX 4080 environment setup
#
# Run this inside WSL2 (Ubuntu 22.04+) after cloning the repo:
#   bash ~/medmcqa/scripts/setup_wsl.sh
#
# Requirements (do these in Windows FIRST — see README below):
#   1. Install NVIDIA Game/Studio driver >= 535 from nvidia.com
#   2. Enable WSL2: wsl --install -d Ubuntu-22.04
#   3. Restart Windows, then open Ubuntu terminal
# ============================================================
set -euo pipefail

VENV_DIR="$HOME/medmcqa-env"
PROJECT_DIR="$HOME/medmcqa"

echo "=== MedMCQA WSL2 (RTX 4080) environment setup ==="
echo "Venv:    $VENV_DIR"
echo "Project: $PROJECT_DIR"

# ── Verify CUDA is visible ─────────────────────────────────────────────────────
echo ""
echo "[pre-check] Checking CUDA visibility in WSL2..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found."
    echo "  Make sure you installed the Windows NVIDIA driver (NOT the Linux driver)."
    echo "  WSL2 CUDA passthrough is automatic with a Windows driver >= 535."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── System packages ───────────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv git curl

# ── Python venv ───────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/5] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# ── PyTorch with CUDA 12.4 ────────────────────────────────────────────────────
# RTX 4080 = Ada Lovelace (sm_89) — fully supported by stable cu124.
# Do NOT use the nightly cu128 build (that's for GB10/Blackwell only).
echo "[3/5] Installing PyTorch cu124 (RTX 4080 / Ada Lovelace)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

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

# ── bitsandbytes (QLoRA — works on RTX 4080, unlike DGX GB10) ─────────────────
echo "[5/5] Installing bitsandbytes for QLoRA (4-bit NF4)..."
pip install bitsandbytes>=0.43.0

python3 -c "
import bitsandbytes as bnb
print(f'bitsandbytes: {bnb.__version__} ✓')
"

echo ""
echo "=== Setup complete on $(hostname) ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Set your HuggingFace token (needed to download Qwen models):"
echo "       echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc && source ~/.bashrc"
echo ""
echo "  2. Copy project data from Mac or DGX (run from Mac):"
echo "       bash scripts/rsync_to_wsl.sh"
echo ""
echo "  3. Start QLoRA training:"
echo "       bash ~/medmcqa/scripts/run_qlora.sh"
