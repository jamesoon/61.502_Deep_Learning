#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Training progress monitor — run this ON DGX9 in a second tmux window.
# Refreshes every 10 seconds showing GPU stats + latest training metrics.
#
# Usage (on DGX9):
#   bash scripts/monitor.sh                    # watch all active jobs
#   bash scripts/monitor.sh --run lora         # watch specific run
#   bash scripts/monitor.sh --interval 5       # refresh every 5s
# ─────────────────────────────────────────────────────────────────────────────

INTERVAL=10
RUN_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --run)      RUN_FILTER="$2"; shift 2 ;;
    *) shift ;;
  esac
done

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/medmcqa-env/bin/activate"
cd ~/medmcqa

print_separator() { printf '%.0s─' {1..60}; echo; }

while true; do
  clear
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "  MedMCQA Training Monitor  —  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""

  # ── GPU stats ─────────────────────────────────────────────────────────────
  echo "▶ GPU"
  print_separator
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
  awk -F', ' '{
    pct = $2/$3*100
    bar_len = int(pct/5)
    bar = ""
    for(i=0;i<bar_len;i++) bar = bar "█"
    for(i=bar_len;i<20;i++) bar = bar "░"
    printf "  %-20s  Mem: %6d/%6d MB [%s] %3d%%  GPU: %3d%%  Temp: %d°C\n",
      $1, $2, $3, bar, pct, $4, $5
  }'
  echo ""

  # ── CPU / RAM ─────────────────────────────────────────────────────────────
  echo "▶ System Memory"
  print_separator
  free -h | awk 'NR==2{printf "  RAM: used %s / total %s (free %s)\n", $3, $2, $4}'
  echo ""

  # ── Active training processes ─────────────────────────────────────────────
  echo "▶ Training Processes"
  print_separator
  PROCS=$(pgrep -af "python.*train\|torchrun" 2>/dev/null || true)
  if [[ -z "$PROCS" ]]; then
    echo "  No training process running."
  else
    echo "$PROCS" | while read -r line; do
      echo "  $line" | cut -c1-80
    done
  fi
  echo ""

  # ── Latest checkpoint metrics ─────────────────────────────────────────────
  echo "▶ Latest Metrics (from trainer_state.json)"
  print_separator

  for ckpt_dir in checkpoints/lora checkpoints/sft; do
    # Look for trainer_state.json in checkpoint subdirs
    STATE=$(find "$ckpt_dir" -name "trainer_state.json" 2>/dev/null | sort | tail -1)
    if [[ -n "$STATE" ]]; then
      label=$(echo "$ckpt_dir" | sed 's|checkpoints/||')
      echo "  [$label]"
      python3 - "$STATE" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    s = json.load(f)
logs = s.get("log_history", [])
if not logs:
    print("    No log entries yet.")
    sys.exit()

# Separate train and eval entries
train_logs = [l for l in logs if "loss" in l and "eval_loss" not in l]
eval_logs  = [l for l in logs if "eval_loss" in l]

total_steps = s.get("max_steps", "?")
epoch_total = s.get("num_train_epochs", "?")

if train_logs:
    last = train_logs[-1]
    step = last.get("step", "?")
    loss = last.get("loss", "?")
    lr   = last.get("learning_rate", "?")
    epoch = last.get("epoch", "?")
    print(f"    Step: {step}/{total_steps}  Epoch: {epoch}/{epoch_total}  "
          f"Train Loss: {loss:.4f}  LR: {lr:.2e}" if isinstance(loss, float) else
          f"    Step: {step}  Loss: {loss}")

if eval_logs:
    last_eval = eval_logs[-1]
    print(f"    Best Eval Loss: {last_eval.get('eval_loss', '?'):.4f}  "
          f"@ step {last_eval.get('step', '?')}")
PYEOF
    fi
  done

  # ── WandB log tail (if wandb/latest-run exists) ───────────────────────────
  WANDB_LOG=$(find . -path "*/wandb/latest-run/logs/debug.log" 2>/dev/null | head -1)
  if [[ -n "$WANDB_LOG" ]]; then
    echo ""
    echo "▶ WandB Log (last 3 lines)"
    print_separator
    tail -3 "$WANDB_LOG" 2>/dev/null | sed 's/^/  /'
  fi

  # ── Disk usage ───────────────────────────────────────────────────────────
  echo ""
  echo "▶ Disk Usage"
  print_separator
  du -sh checkpoints/ results/ data/processed/ 2>/dev/null | sed 's/^/  /' || true

  echo ""
  echo "  Refreshing every ${INTERVAL}s — Ctrl+C to stop"
  sleep "$INTERVAL"
done
