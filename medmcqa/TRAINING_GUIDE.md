# Training Guide

## Overview

**Recommended order:** QLoRA first (fast, ~25 GB GPU) → review results → SFT (thorough, slow).

```
Local machine  ──rsync──▶  DGX9 (192.168.0.219)
                               │
                          tmux session
                          ├── window: main    (training)
                          └── window: monitor (live stats)
```

---

## Step 1 — One-time setup (local machine)

Open a terminal in the `medmcqa/` directory.

**1a. Sync code to DGX9:**
```bash
bash scripts/rsync_to_dgx.sh
```

**1b. Sync data files (separate step — can be large):**
```bash
bash scripts/rsync_data_to_dgx.sh
```

**1c. SSH in and set up the conda environment (first time only):**
```bash
bash scripts/tmux_session.sh --new    # opens tmux with main + monitor windows
```

Inside DGX9 tmux (window: `main`):
```bash
cd ~/medmcqa
bash scripts/setup_dgx_env.sh        # creates conda env, installs deps (~10 min)
wandb login                           # paste your WandB API key when prompted
```

---

## Step 2 — Start training

### Option A: Recommended — run everything automatically

```bash
# Inside DGX9 tmux (window: main)
cd ~/medmcqa && conda activate medmcqa
bash scripts/run_all.sh
```

This runs: **baseline → QLoRA → SFT → comparison table**.
QLoRA results appear first. SFT starts automatically after QLoRA finishes.

### Option B: QLoRA only (fastest path to results)

```bash
bash scripts/run_lora.sh
```

### Option C: Run stages manually in order

```bash
# Step 1: Zero-shot baseline (no fine-tuning, ~1-2 hrs)
bash scripts/run_baseline.sh

# Step 2: QLoRA fine-tuning (~3-6 hrs depending on dataset size)
bash scripts/run_lora.sh

# Step 3: Full SFT — after you have QLoRA results (requires ZeRO-3, slower)
bash scripts/run_sft.sh
```

### Quick smoke test (5 min — checks everything works before long run)

```bash
MAX_TRAIN=500 MAX_SAMPLES=50 bash scripts/run_lora.sh
```

---

## Step 3 — Monitor progress

You have three ways to watch training progress:

### 3a. Live monitor on DGX9 (best)

Switch to the `monitor` tmux window (press `Ctrl+B` then `2`, or `Ctrl+B` then window name):

```bash
# In DGX9 tmux window: monitor
cd ~/medmcqa && conda activate medmcqa
bash scripts/monitor.sh
```

Displays every 10 seconds:
- GPU memory usage + utilization bar
- System RAM
- Active training process
- Current step / epoch / train loss / eval loss
- Disk usage

### 3b. WandB dashboard (richest — loss curves, charts)

Open in browser: **https://wandb.ai**

- Project `medmcqa-lora` → QLoRA run
- Project `medmcqa-sft` → SFT run

Tracks: training loss, eval loss, learning rate, GPU utilization, step time.

### 3c. Quick status from local machine (no SSH attachment)

```bash
# Run this on your LOCAL machine
bash scripts/remote_status.sh

# Auto-refresh every 60 seconds
watch -n 60 bash scripts/remote_status.sh
```

---

## Navigating tmux

| Action | Keys |
|--------|------|
| Switch to monitor window | `Ctrl+B` then `2` |
| Switch to main window | `Ctrl+B` then `1` |
| Detach (leave training running) | `Ctrl+B` then `D` |
| Re-attach from local machine | `bash scripts/tmux_session.sh` |
| Scroll up to see past output | `Ctrl+B` then `[`, then arrow keys, `Q` to exit |
| Create a new window | `Ctrl+B` then `C` |

---

## What to expect

| Stage | GPU Memory | Estimated Time | Output |
|-------|-----------|----------------|--------|
| Data prep | CPU only | 2–5 min | `data/processed/*.jsonl` |
| Baseline eval | ~18 GB | 1–3 hrs (full test set) | `results/baseline/` |
| QLoRA training | ~25 GB | 4–12 hrs (full dataset) | `checkpoints/lora/` |
| QLoRA eval | ~18 GB | 1–3 hrs | `results/lora/` |
| SFT training | ~15 GB GPU (offloaded) | 12–48 hrs | `checkpoints/sft/` |
| SFT eval | ~18 GB | 1–3 hrs | `results/sft/` |

> Estimated times assume full 182K training set on DGX Spark.
> Use `MAX_TRAIN=5000` to cap training samples for faster iteration.

---

## Checking results

```bash
# After any run completes, compare all runs so far:
python training/evaluate.py --compare --results_dir results/

# View per-subject breakdown:
cat results/lora/lora_metrics.json | python3 -m json.tool

# View 50 failure cases:
cat results/lora/lora_error_analysis.json | python3 -m json.tool | head -100
```

---

## Changing the model

Edit **one line** in `training/config.py`:

```python
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"   # ← change this
```

Or override per-run without editing:
```bash
MODEL_ID=Qwen/Qwen2-7B-Instruct bash scripts/run_lora.sh
```

---

## Troubleshooting

**Training hangs / GPU stuck at 0%:**
```bash
# Check if process is alive
pgrep -af "python\|torchrun"

# Check GPU
nvidia-smi

# Kill a stuck job
pkill -f "torchrun\|sft_train\|lora_train"
```

**Out of memory error:**
- QLoRA: already capped at 105 GiB — should not happen. If it does, reduce `--per_device_batch` to 2.
- SFT: ensure you used `run_sft.sh` (which calls `torchrun --deepspeed`). Plain `python sft_train.py` will OOM.

**WandB not logging:**
```bash
wandb login    # re-authenticate
# Or disable and use CSV logs only:
WANDB_DISABLED=true bash scripts/run_lora.sh
```

**Sync updated code mid-training:**
```bash
# From local machine — safe to run while training continues
bash scripts/rsync_to_dgx.sh
```
