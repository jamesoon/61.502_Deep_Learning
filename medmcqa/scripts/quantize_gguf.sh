#!/usr/bin/env bash
# =============================================================================
# Merge LoRA adapters and quantize to GGUF Q4_K_M for HuggingFace upload.
#
# Runs on DGX8 (192.168.0.219) where base models are already cached.
# Output: ~/medmcqa/quantized/{lora-9b,lora-14b}/
#
# Usage (run directly on DGX8, or via run_quantize_dgx.sh from Mac):
#   bash scripts/quantize_gguf.sh                  # both models
#   MODEL=lora-9b bash scripts/quantize_gguf.sh    # single model
# =============================================================================
set -euo pipefail

MEDMCQA_DIR="${MEDMCQA_DIR:-$HOME/medmcqa}"
VENV_DIR="${VENV_DIR:-$HOME/medmcqa-env}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
OUTPUT_DIR="$MEDMCQA_DIR/quantized"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"

# Models to process (space-separated). Override via MODEL= env var.
MODELS="${MODEL:-lora-9b lora-14b}"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

# ── Set up llama.cpp ──────────────────────────────────────────────────────────
setup_llamacpp() {
    if [ ! -d "$LLAMA_DIR" ]; then
        echo "[setup] Cloning llama.cpp…"
        git clone --depth=1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
    fi

    echo "[setup] Installing llama.cpp Python deps…"
    pip install -q gguf sentencepiece

    # Build llama-quantize binary (CPU-only, no CUDA needed)
    if [ ! -f "$LLAMA_DIR/build/bin/llama-quantize" ]; then
        echo "[setup] Building llama-quantize (CPU-only)…"
        cd "$LLAMA_DIR"
        cmake -B build \
            -DLLAMA_NATIVE=OFF \
            -DGGML_CUDA=OFF \
            -DGGML_NATIVE=OFF \
            2>&1 | tail -3
        cmake --build build --target llama-quantize -j"$(nproc)" 2>&1 | tail -5
        echo "[setup] Build complete."
    fi
}

# ── Process one model ─────────────────────────────────────────────────────────
quantize_model() {
    local name="$1"
    local adapter="$MEDMCQA_DIR/checkpoints/$name/final"
    local merged="$MEDMCQA_DIR/checkpoints/${name}-merged"
    local out_dir="$OUTPUT_DIR/$name"
    local f16_gguf="/tmp/${name}-f16.gguf"
    local q4_gguf="$out_dir/${name}-medmcqa-${QUANT_TYPE,,}.gguf"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  $name"
    echo "══════════════════════════════════════════════════════"

    if [ ! -d "$adapter" ]; then
        echo "  ✗ Adapter not found: $adapter — skipping."
        return 1
    fi

    mkdir -p "$out_dir"

    # 1. Merge LoRA
    if [ -f "$merged/config.json" ]; then
        echo "[1/3] Merged model already exists — skipping merge."
    else
        echo "[1/3] Merging LoRA adapter…"
        python3 "$MEDMCQA_DIR/scripts/merge_lora.py" \
            --adapter "$adapter" \
            --output  "$merged"
    fi

    # 2. Convert to F16 GGUF
    if [ -f "$f16_gguf" ]; then
        echo "[2/3] F16 GGUF already exists — skipping conversion."
    else
        echo "[2/3] Converting to GGUF (F16)…"
        python3 "$LLAMA_DIR/convert_hf_to_gguf.py" \
            "$merged" \
            --outtype f16 \
            --outfile "$f16_gguf" \
            2>&1 | grep -v "^$" || true

        if [ ! -f "$f16_gguf" ]; then
            echo "  ✗ GGUF conversion failed (architecture may not be supported)."
            echo "  → Falling back to safetensors output (BF16 merged model)."
            # Copy merged safetensors as the upload artifact instead
            cp -r "$merged"/. "$out_dir/"
            echo "  ✓ BF16 merged model copied to $out_dir"
            return 0
        fi
    fi

    # 3. Quantize to Q4_K_M
    echo "[3/3] Quantizing to $QUANT_TYPE…"
    "$LLAMA_DIR/build/bin/llama-quantize" \
        "$f16_gguf" \
        "$q4_gguf" \
        "$QUANT_TYPE" \
        2>&1 | tail -5

    # Clean up large intermediate
    rm -f "$f16_gguf"

    # Copy tokenizer files alongside GGUF for reference
    cp "$merged"/tokenizer*.json "$out_dir/" 2>/dev/null || true
    cp "$merged"/tokenizer_config.json "$out_dir/" 2>/dev/null || true

    local size
    size=$(du -sh "$q4_gguf" | cut -f1)
    echo "  ✓ $q4_gguf ($size)"
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  GGUF Quantization Pipeline"
echo "  Quant type : $QUANT_TYPE"
echo "  Output dir : $OUTPUT_DIR"
echo "======================================================"

setup_llamacpp

mkdir -p "$OUTPUT_DIR"

for model in $MODELS; do
    quantize_model "$model" || echo "  Warning: $model failed, continuing…"
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Done. Quantized files:"
find "$OUTPUT_DIR" -name "*.gguf" -o -name "*.safetensors" 2>/dev/null | \
    while read -r f; do echo "  $(du -sh "$f" | cut -f1)  $f"; done
echo "══════════════════════════════════════════════════════"
echo ""
echo "Next: run 'bash scripts/fetch_quantized.sh' from your Mac."
