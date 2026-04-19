"""
Central model and training configuration.
Change MODEL_ID here to switch models across all scripts.

Supported models:
  google/gemma-3-4b-it  — 6 GB FP16, instruction-tuned, MPS/CUDA-compatible [LOCAL DEFAULT]
  Qwen/Qwen3-14B        — 28 GB BF16, standard transformer (no GatedDeltaNet), fast [DGX DEFAULT]
  Qwen/Qwen3-8B         — 16 GB BF16, standard transformer, fastest DGX iteration
  Qwen/Qwen3.5-9B       — 18 GB BF16, GatedDeltaNet hybrid (~45s/step — avoid for training)
  Qwen/Qwen3.5-27B      — 54 GB BF16, GatedDeltaNet hybrid (avoid for training)

Usage — override at CLI:
    python training/lora_train.py --model_id google/gemma-3-4b-it
Or set env var:
    MODEL_ID=google/gemma-3-4b-it bash scripts/run_local.sh
"""

import os

# ── Model selection ──────────────────────────────────────────────────────────
# Change this single line to swap the base model for all scripts.
MODEL_ID: str = os.getenv("MODEL_ID", "Qwen/Qwen3-14B")

# ── LoRA hyperparameters ─────────────────────────────────────────────────────
LORA_RANK: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05

LORA_BATCH: int = 4
LORA_GRAD_ACCUM: int = 4         # Effective batch = LORA_BATCH × LORA_GRAD_ACCUM = 16
LORA_LR: float = 2e-4
LORA_EPOCHS: int = 3
LORA_MAX_SEQ_LEN: int = 1024

# ── SFT hyperparameters ──────────────────────────────────────────────────────
SFT_BATCH: int = 2
SFT_GRAD_ACCUM: int = 8          # Effective batch = 16
SFT_LR: float = 2e-5
SFT_EPOCHS: int = 3
SFT_MAX_SEQ_LEN: int = 1024

# ── Shared ───────────────────────────────────────────────────────────────────
WARMUP_RATIO: float = 0.03
MAX_GRAD_NORM: float = 1.0
SAVE_STEPS: int = 200
EVAL_STEPS: int = 200
EARLY_STOPPING_PATIENCE: int = 3

# ── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are a helpful tutor for pre-med students preparing for the MCAT. "
    "You answer multiple-choice questions with step-by-step reasoning."
)
